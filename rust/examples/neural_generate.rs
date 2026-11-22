//! Bounded teacher generation. JSON requests, compact binary responses (see generate.py).
use schotten_native::bots::Bot;
use schotten_native::engine::{Game, Options, Play, PlayerView, TurnRequest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};

const ROW_SIZE: usize = 200;
const MAX_ROWS: usize = 65536;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    op: String,
    seed: u64,
    split: String,
    start: u64,
    count: usize,
    teacher: String,
    opponents: Vec<String>,
    #[serde(default)]
    trace: bool,
}

fn bit(bytes: &mut [u8], index: usize) {
    bytes[index / 8] |= 1 << (index % 8);
}

fn side(value: Option<bool>) -> usize {
    match value {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    }
}

fn encode(
    view: &PlayerView,
    deck: u8,
    opponent_hand: u8,
    play: Option<&Play>,
) -> Result<[u8; ROW_SIZE], String> {
    let mut row = [0; ROW_SIZE];
    for card in 0..54 {
        if view.hand & (1u64 << card) != 0 {
            bit(&mut row[..135], card);
        }
    }
    for (index, stone) in view.stones.iter().enumerate() {
        for (which, cards) in [stone.yours, stone.theirs].iter().enumerate() {
            for &card in cards.as_slice() {
                bit(
                    &mut row[..135],
                    54 + index * 108 + which * 54 + card as usize,
                );
            }
        }
        bit(&mut row[..135], 1026 + index * 3 + side(stone.claimed_by));
        bit(
            &mut row[..135],
            1053 + index * 3 + side(stone.first_to_complete),
        );
        if stone.claimed_by.is_none() && stone.yours.len() < 3 {
            for card in 0..54 {
                if view.hand & (1u64 << card) != 0 {
                    bit(&mut row[137..198], card * 9 + index);
                }
            }
        }
    }
    if row[137..198].iter().all(|&byte| byte == 0) {
        bit(&mut row[137..198], 486);
    }
    row[135] = deck;
    row[136] = opponent_hand;
    let label = play.map_or(486, |play| play.card.id() as usize * 9 + play.stone);
    if label >= 487 || row[137 + label / 8] & (1 << (label % 8)) == 0 {
        return Err("teacher selected an illegal action".into());
    }
    row[198..].copy_from_slice(&(label as u16).to_le_bytes());
    Ok(row)
}

fn deck_seed(request: &Request, counter: u64) -> Result<i64, String> {
    let split = u64::from(request.split == "validation");
    counter
        .checked_mul(2)
        .and_then(|n| n.checked_add(request.seed))
        .and_then(|n| n.checked_add(split))
        .and_then(|n| i64::try_from(n).ok())
        .ok_or_else(|| "deck seed counter exhausted".into())
}

struct Generated {
    rows: Vec<[u8; ROW_SIZE]>,
    trace: Vec<Value>,
}

fn generate_game(request: &Request, counter: u64) -> Result<Generated, String> {
    let seed = deck_seed(request, counter)?;
    let opponent = &request.opponents[counter as usize % request.opponents.len()];
    let teacher_seat = (counter / request.opponents.len() as u64 % 2) as usize;
    let names = if teacher_seat == 0 {
        [&request.teacher, opponent]
    } else {
        [opponent, &request.teacher]
    };
    let agent_seed = |seat: u64| {
        ((seed as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(seat + 1442695040888963407)
            & i64::MAX as u64) as i64
    };
    let mut bots = [
        Bot::new(names[0], Some(agent_seed(0)))?,
        Bot::new(names[1], Some(agent_seed(1)))?,
    ];
    let mut game = Game::new(
        format!("generate-{seed}"),
        Options {
            seed: Some(seed),
            ..Default::default()
        },
        0,
        false,
    );
    game.join(0).map_err(|e| e.to_string())?;
    let mut generated = Generated {
        rows: Vec::new(),
        trace: Vec::new(),
    };
    let mut deck = 42;
    let mut hands = [6, 6];
    while !game.over() {
        if game.version > 128 {
            return Err("game exceeded turn bound".into());
        }
        let seat = game.turn.ok_or("missing game turn")?;
        let view = game.player_view(seat);
        let (play, claims) = bots[seat].choose(&view);
        if names[seat] == &request.teacher {
            let row = encode(&view, deck, hands[1 - seat], play.as_ref())?;
            if request.trace {
                generated
                    .trace
                    .push(json!({"deck_seed":seed,"counter":counter,
                    "teacher_seat":teacher_seat,"opponent":opponent,"view":game.view(seat),
                    "play":play,"claims":claims,"row":row.as_slice()}));
            }
            generated.rows.push(row);
        }
        let placed = play.is_some();
        game.play_turn(
            seat,
            TurnRequest {
                version: game.version as i64,
                turn_id: None,
                play,
                claims,
            },
            0,
        )
        .map_err(|e| e.to_string())?;
        if placed {
            hands[seat] -= 1;
            if deck > 0 && !game.over() {
                deck -= 1;
                hands[seat] += 1;
            }
        }
    }
    Ok(generated)
}

fn validate(request: &Request) -> Result<(), String> {
    if request.op != "generate" || !matches!(request.split.as_str(), "train" | "validation") {
        return Err("expected generate operation and train/validation split".into());
    }
    if request.count == 0 || request.count > MAX_ROWS || request.opponents.is_empty() {
        return Err("count must be 1..65536 and opponents must be nonempty".into());
    }
    if request.teacher != "mcb" && !request.teacher.starts_with("mcb:") {
        return Err("teacher must be an MCB strategy".into());
    }
    Bot::new(&request.teacher, Some(0))?;
    for opponent in &request.opponents {
        Bot::new(opponent, Some(0))?;
    }
    deck_seed(request, request.start)?;
    Ok(())
}

type Job = dyn Fn(&Request, u64) -> Result<Generated, String> + Send + Sync;

// Permits cover running jobs, jobs still queued in Rayon, channel results and
// reorder-buffer entries, including cancelled generations. A rapid series of
// resets therefore cannot accumulate detached jobs or retained requests.
struct Budget {
    limit: usize,
    used: Mutex<usize>,
    released: Condvar,
}

impl Budget {
    fn acquire(self: &Arc<Self>, wait: bool) -> Option<Permit> {
        let mut used = self.used.lock().unwrap();
        while *used == self.limit {
            if !wait {
                return None;
            }
            used = self.released.wait(used).unwrap();
        }
        *used += 1;
        Some(Permit(self.clone()))
    }
}

struct Permit(Arc<Budget>);

impl Drop for Permit {
    fn drop(&mut self) {
        let mut used = self.0.used.lock().unwrap();
        *used -= 1;
        self.0.released.notify_all();
    }
}

struct Completed {
    result: Result<Generated, String>,
    _permit: Permit,
}

struct Queue {
    request: Arc<Request>,
    cancel: Arc<AtomicBool>,
    sender: mpsc::Sender<(u64, Completed)>,
    receiver: mpsc::Receiver<(u64, Completed)>,
    ready: BTreeMap<u64, Completed>,
    next_emit: u64,
    next_schedule: Option<u64>,
    outstanding: usize,
}

impl Queue {
    fn new(request: Request) -> Self {
        let (sender, receiver) = mpsc::channel();
        let next_emit = request.start;
        Self {
            request: Arc::new(request),
            cancel: Arc::new(AtomicBool::new(false)),
            sender,
            receiver,
            ready: BTreeMap::new(),
            next_emit,
            next_schedule: Some(next_emit),
            outstanding: 0,
        }
    }

    fn matches(&self, request: &Request) -> bool {
        let old = &self.request;
        self.next_emit == request.start
            && old.seed == request.seed
            && old.split == request.split
            && old.teacher == request.teacher
            && old.opponents == request.opponents
            && old.trace == request.trace
    }

    fn replenish(&mut self, budget: &Arc<Budget>, job: &Arc<Job>, wait: bool) {
        while let Some(counter) = self.next_schedule {
            // An unused future seed beyond the valid range is not an error.
            if deck_seed(&self.request, counter).is_err() {
                self.next_schedule = None;
                break;
            }
            let Some(permit) = budget.acquire(wait && self.outstanding == 0) else {
                break;
            };
            self.next_schedule = counter.checked_add(1);
            self.outstanding += 1;
            let request = self.request.clone();
            let cancel = self.cancel.clone();
            let sender = self.sender.clone();
            let job = job.clone();
            // Use the persistent global pool, with no scope, borrowed jobs or
            // nested parallel iteration. Cancelled queued jobs do no game work.
            rayon::spawn(move || {
                let result = if cancel.load(Ordering::Acquire) {
                    None
                } else {
                    Some(job(&request, counter))
                };
                // Release the request before its permit, including cancellation.
                drop(request);
                if let Some(result) = result.filter(|_| !cancel.load(Ordering::Acquire)) {
                    let _ = sender.send((
                        counter,
                        Completed {
                            result,
                            _permit: permit,
                        },
                    ));
                }
            });
        }
    }

    fn pop(&mut self, budget: &Arc<Budget>, job: &Arc<Job>) -> Result<Generated, String> {
        // After a reset, old running jobs can temporarily hold all permits.
        // Wait for capacity only when the consumer has no work of its own.
        self.replenish(budget, job, true);
        if self.outstanding == 0 {
            return Err("deck seed counter exhausted".into());
        }
        let completed = loop {
            if let Some(completed) = self.ready.remove(&self.next_emit) {
                break completed;
            }
            let (counter, completed) = self.receiver.recv().map_err(|e| e.to_string())?;
            self.ready.insert(counter, completed);
        };
        self.outstanding -= 1;
        self.next_emit = self
            .next_emit
            .checked_add(1)
            .ok_or("game counter exhausted")?;
        let Completed { result, _permit } = completed;
        drop(_permit);
        self.replenish(budget, job, false);
        result
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // Never join speculative games at EOF. Dropping the receiver and ready
        // buffer releases results; detached jobs own all their data and permits.
        self.cancel.store(true, Ordering::Release);
    }
}

struct Worker {
    state: Option<Queue>,
    budget: Arc<Budget>,
    job: Arc<Job>,
}

impl Worker {
    fn new() -> Self {
        Self::with_job(
            rayon::current_num_threads().saturating_mul(4),
            generate_game,
        )
    }

    fn with_job(
        lookahead: usize,
        job: impl Fn(&Request, u64) -> Result<Generated, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            state: None,
            budget: Arc::new(Budget {
                limit: lookahead.max(1),
                used: Mutex::new(0),
                released: Condvar::new(),
            }),
            job: Arc::new(job),
        }
    }

    fn generate(&mut self, request: Request, writer: &mut impl Write) -> Result<(), String> {
        validate(&request)?;
        let count = request.count;
        let trace_mode = request.trace;
        if !self
            .state
            .as_ref()
            .is_some_and(|state| state.matches(&request))
        {
            // Drop before allocating replacement jobs: buffered old results free
            // permits immediately, running/queued jobs free theirs asynchronously.
            self.state = None;
            self.state = Some(Queue::new(request));
        }
        let mut rows = Vec::with_capacity(count);
        let mut trace = Vec::new();
        while rows.len() < count {
            let generated = match self.state.as_mut().unwrap().pop(&self.budget, &self.job) {
                Ok(generated) => generated,
                Err(error) => {
                    self.state = None;
                    return Err(error);
                }
            };
            let take = (count - rows.len()).min(generated.rows.len());
            rows.extend(generated.rows.into_iter().take(take));
            trace.extend(generated.trace.into_iter().take(take));
            // The last game's unused rows are deliberately discarded. Only
            // subsequent whole games survive across requests.
        }
        let next = self.state.as_ref().unwrap().next_emit;
        if trace_mode {
            serde_json::to_writer(&mut *writer, &json!({"next_game":next,"examples":trace}))
                .map_err(|e| e.to_string())?;
            writer.write_all(b"\n").map_err(|e| e.to_string())?;
        } else {
            writer.write_all(b"SPGEN001").map_err(|e| e.to_string())?;
            writer
                .write_all(&next.to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&(rows.len() as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for row in rows {
                writer.write_all(&row).map_err(|e| e.to_string())?;
            }
        }
        writer.flush().map_err(|e| e.to_string())
    }
}

fn main() {
    let result = (|| -> Result<(), String> {
        let mut writer = io::BufWriter::new(io::stdout().lock());
        let mut worker = Worker::new();
        for line in io::stdin().lock().lines() {
            let line = line.map_err(|e| e.to_string())?;
            let request: Request = serde_json::from_str(&line).map_err(|e| e.to_string())?;
            worker.generate(request, &mut writer)?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("neural generate: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schotten_native::engine::{Card, Line, ViewStone};

    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Duration;

    fn test_worker(
        lookahead: usize,
        job: impl Fn(&Request, u64) -> Result<Generated, String> + Send + Sync + 'static,
    ) -> Worker {
        // Blocking mock jobs require multiple workers, independently of a
        // caller's RAYON_NUM_THREADS. The production worker uses the normal pool.
        static POOL: std::sync::Once = std::sync::Once::new();
        POOL.call_once(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build_global()
                .unwrap();
        });
        Worker::with_job(lookahead, job)
    }

    fn request(start: u64, count: usize) -> Request {
        serde_json::from_value(json!({"op":"generate", "seed":11, "split":"train",
            "start":start, "count":count, "teacher":"mcb:0", "opponents":["mcb:0"]}))
        .unwrap()
    }

    fn mock_game(counter: u64) -> Generated {
        Generated {
            rows: vec![
                [counter as u8 * 10; ROW_SIZE],
                [counter as u8 * 10 + 1; ROW_SIZE],
            ],
            trace: vec![
                json!({"counter":counter,"turn":0}),
                json!({"counter":counter,"turn":1}),
            ],
        }
    }

    fn binary_output(bytes: &[u8]) -> (u64, Vec<u8>) {
        assert_eq!(&bytes[..8], b"SPGEN001");
        let next = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), 24 + count * ROW_SIZE);
        (
            next,
            bytes[24..]
                .chunks_exact(ROW_SIZE)
                .map(|row| row[0])
                .collect(),
        )
    }

    // Catches replay of prefetched counters and reuse of a truncated game's tail.
    #[test]
    fn queue_reuses_whole_future_games_and_discards_partial_tail() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let seen = calls.clone();
        let (completed_tx, completed_rx) = mpsc::channel();
        let mut worker = test_worker(4, move |_, counter| {
            seen.lock().unwrap().push(counter);
            completed_tx.send(counter).unwrap();
            Ok(mock_game(counter))
        });
        let mut first = Vec::new();
        worker.generate(request(0, 3), &mut first).unwrap();
        assert_eq!(binary_output(&first), (2, vec![0, 1, 10]));
        let mut second = Vec::new();
        worker.generate(request(2, 3), &mut second).unwrap();
        assert_eq!(binary_output(&second), (4, vec![20, 21, 30]));
        for _ in 0..8 {
            completed_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        }
        drop(worker); // cancellation never waits for speculative games
        let mut calls = calls.lock().unwrap().clone();
        calls.sort_unstable();
        assert_eq!(calls, (0..8).collect::<Vec<_>>());
    }

    // Catches a batch barrier, unordered output, or scheduling beyond the bound.
    #[test]
    fn queue_runs_later_jobs_while_first_waits_and_emits_in_order() {
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let (done_tx, done_rx) = mpsc::channel();
        let worker = test_worker(4, move |_, counter| {
            if counter == 0 {
                release_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();
            }
            let _ = done_tx.send(counter);
            Ok(mock_game(counter))
        });
        let consumer = std::thread::spawn(move || {
            let mut worker = worker;
            let mut bytes = Vec::new();
            worker.generate(request(0, 5), &mut bytes).unwrap();
            (worker, bytes)
        });
        let mut completed = Vec::new();
        for _ in 0..3 {
            completed.push(done_rx.recv_timeout(Duration::from_secs(3)));
        }
        let extra = done_rx.recv_timeout(Duration::from_millis(30));
        release_tx.send(()).unwrap();
        let (worker, bytes) = consumer.join().unwrap();
        let mut completed: Vec<_> = completed.into_iter().map(Result::unwrap).collect();
        completed.sort_unstable();
        assert_eq!(completed, [1, 2, 3]);
        assert!(matches!(extra, Err(mpsc::RecvTimeoutError::Timeout)));
        assert_eq!(binary_output(&bytes), (3, vec![0, 1, 10, 11, 20]));
        let state = worker.state.as_ref().unwrap();
        assert_eq!(state.outstanding, 4); // includes running, channel and reorder buffer
        assert!(state.ready.len() <= state.outstanding);
    }

    // Catches eager propagation of an unused speculative failure.
    #[test]
    fn queue_defers_future_errors_until_consumed() {
        let mut worker = test_worker(4, |_, counter| {
            if counter == 1 {
                Err("future failure".into())
            } else {
                Ok(mock_game(counter))
            }
        });
        let mut bytes = Vec::new();
        worker.generate(request(0, 1), &mut bytes).unwrap();
        assert_eq!(binary_output(&bytes), (1, vec![0]));
        // Reset past the still-unused error, then reproduce it for consumption.
        worker.generate(request(3, 1), &mut Vec::new()).unwrap();
        worker.generate(request(0, 1), &mut Vec::new()).unwrap();
        let mut failed = Vec::new();
        assert_eq!(
            worker.generate(request(1, 1), &mut failed).unwrap_err(),
            "future failure"
        );
        assert!(failed.is_empty());
    }

    // Catches a missing cache-key member or an ignored resume cursor.
    #[test]
    fn queue_resets_for_every_key_field_and_unexpected_cursor() {
        for field in ["seed", "split", "teacher", "opponents", "trace", "start"] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let seen = calls.clone();
            let mut worker = test_worker(2, move |req, counter| {
                seen.lock().unwrap().push((req.trace, req.seed, counter));
                Ok(mock_game(counter))
            });
            worker.generate(request(0, 1), &mut Vec::new()).unwrap();
            let mut changed = request(1, 1);
            match field {
                "seed" => changed.seed = 12,
                "split" => changed.split = "validation".into(),
                "teacher" => changed.teacher = "mcb:1".into(),
                "opponents" => changed.opponents.push("builder".into()),
                "trace" => changed.trace = true,
                "start" => changed.start = 0,
                _ => unreachable!(),
            }
            let old_cancel = worker.state.as_ref().unwrap().cancel.clone();
            worker.generate(changed, &mut Vec::new()).unwrap();
            assert!(
                old_cancel.load(std::sync::atomic::Ordering::Acquire),
                "{field}"
            );
            assert!(
                !Arc::ptr_eq(&old_cancel, &worker.state.as_ref().unwrap().cancel),
                "{field}"
            );
        }
    }

    // Catches joining speculative games on reset/EOF or exceeding the budget
    // across cancelled generations. One old job remains blocked throughout.
    #[test]
    fn queue_reset_and_drop_do_not_join_running_games() {
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let (started_tx, started_rx) = mpsc::channel();
        let mut worker = test_worker(2, move |_, counter| {
            if counter == 1 {
                started_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap();
            }
            Ok(mock_game(counter))
        });
        worker.generate(request(0, 1), &mut Vec::new()).unwrap();
        started_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        let old_request = Arc::downgrade(&worker.state.as_ref().unwrap().request);
        let old_cancel = worker.state.as_ref().unwrap().cancel.clone();
        let budget = worker.budget.clone();
        let (finished_tx, finished_rx) = mpsc::channel();
        let consumer = std::thread::spawn(move || {
            // No new jobs may exceed the total budget, even over many resets.
            for counter in 10..20 {
                let mut bytes = Vec::new();
                worker.generate(request(counter, 1), &mut bytes).unwrap();
                assert_eq!(
                    binary_output(&bytes),
                    (counter + 1, vec![counter as u8 * 10])
                );
                assert!(*worker.budget.used.lock().unwrap() <= 2);
            }
            drop(worker);
            finished_tx.send(()).unwrap();
        });
        let finished = finished_rx.recv_timeout(Duration::from_secs(3));
        let cancelled = old_cancel.load(Ordering::Acquire);
        release_tx.send(()).unwrap(); // always unblock before asserting
        consumer.join().unwrap();
        assert!(finished.is_ok(), "reset/drop waited for a speculative game");
        assert!(cancelled);
        let used = budget.used.lock().unwrap();
        let (used, timeout) = budget
            .released
            .wait_timeout_while(used, Duration::from_secs(3), |used| *used != 0)
            .unwrap();
        assert!(!timeout.timed_out());
        assert_eq!(*used, 0);
        assert!(
            old_request.upgrade().is_none(),
            "cancelled request retained after completion"
        );
    }

    // Catches running cancelled queued jobs: the same cancellation path is
    // exercised without races by cancelling before any job is submitted.
    #[test]
    fn queue_cancelled_jobs_skip_work_and_release_owned_requests() {
        let mut worker = test_worker(4, |_, _| panic!("cancelled job ran"));
        let state = Queue::new(request(0, 1));
        state.cancel.store(true, Ordering::Release);
        let old_request = Arc::downgrade(&state.request);
        worker.state = Some(state);
        worker
            .state
            .as_mut()
            .unwrap()
            .replenish(&worker.budget, &worker.job, false);
        worker.state = None;
        let used = worker.budget.used.lock().unwrap();
        let (used, timeout) = worker
            .budget
            .released
            .wait_timeout_while(used, Duration::from_secs(3), |used| *used != 0)
            .unwrap();
        assert!(!timeout.timed_out());
        assert_eq!(*used, 0);
        assert!(old_request.upgrade().is_none());
    }

    // Catches incorrectly reusing binary-mode jobs when trace is requested.
    #[test]
    fn queue_trace_preserves_order_and_truncation() {
        let mut worker = test_worker(3, |_, counter| Ok(mock_game(counter)));
        let mut req = request(0, 3);
        req.trace = true;
        let mut bytes = Vec::new();
        worker.generate(req, &mut bytes).unwrap();
        assert_eq!(bytes, b"{\"examples\":[{\"counter\":0,\"turn\":0},{\"counter\":0,\"turn\":1},{\"counter\":1,\"turn\":0}],\"next_game\":2}\n");
    }

    // Catches speculative seed exhaustion leaking into a valid final game.
    #[test]
    fn queue_stops_prefetch_at_seed_limit() {
        let mut worker = test_worker(4, |_, _| Ok(mock_game(0)));
        let mut req = request(0, 1);
        req.seed = i64::MAX as u64;
        let mut bytes = Vec::new();
        worker.generate(req, &mut bytes).unwrap();
        assert_eq!(binary_output(&bytes), (1, vec![0]));
        assert_eq!(worker.state.as_ref().unwrap().outstanding, 0);
        let mut req = request(0, 3);
        req.seed = i64::MAX as u64;
        assert_eq!(
            worker.generate(req, &mut Vec::new()).unwrap_err(),
            "deck seed counter exhausted"
        );
    }

    #[test]
    fn encoder_handles_pass_and_ignores_hidden_pool() {
        let mut view = PlayerView {
            version: 91,
            hand: 0,
            unseen: 0,
            stones: [ViewStone::default(); 9],
        };
        let row = encode(&view, 0, 2, None).unwrap();
        assert_eq!(row[135..137], [0, 2]);
        assert_eq!(row[197], 64); // pass bit, with high padding bit clear
        assert_eq!(u16::from_le_bytes([row[198], row[199]]), 486);
        assert!(row[137..197].iter().all(|&byte| byte == 0));
        view.unseen = (1u64 << 54) - 1;
        view.version = 123;
        assert_eq!(encode(&view, 0, 2, None).unwrap(), row);
    }

    #[test]
    fn claimed_and_full_stones_are_illegal_and_side_bits_are_relative() {
        let mut view = PlayerView {
            version: 1,
            hand: 1,
            unseen: 0,
            stones: [ViewStone::default(); 9],
        };
        view.stones[0].claimed_by = Some(false);
        view.stones[0].first_to_complete = Some(true);
        view.stones[1].yours = Line::from_slice(&[1, 2, 3]);
        let play = Play {
            card: Card::from_id(0),
            stone: 2,
        };
        let row = encode(&view, 42, 6, Some(&play)).unwrap();
        assert_eq!(row[137], 0b11111100);
        assert_eq!(row[138], 1);
        assert_ne!(row[1028 / 8] & (1 << (1028 % 8)), 0);
        assert_ne!(row[1054 / 8] & (1 << (1054 % 8)), 0);
        assert!(encode(&view, 42, 6, None).is_err());
        assert!(encode(&view, 42, 6, Some(&Play { stone: 0, ..play })).is_err());
    }
}
