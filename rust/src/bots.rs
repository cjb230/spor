use crate::deterministic::{self, Mode};
use crate::engine::{Card, Game, Options, Play, PlayerView, TurnRequest, ViewStone};
use crate::montecarlo::{
    self, Rollout,
    obs::{Observation, Side, StoneObs},
};
use crate::rng::PythonRandom;
use crate::rules;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Strategy {
    Greedy,
    Random,
    Deterministic(Mode),
    /// Monte Carlo search; `samples` possible worlds per decision.
    MonteCarlo {
        samples: usize,
        rollout: Rollout,
    },
}

pub const STRATEGIES: [&str; 5] = ["greedy", "random", "builder", "tactician", "planner"];

/// Samples used by a bare Monte Carlo strategy name. Simulated games cost very different
/// amounts depending on the policy that plays them, so each kind has its own default.
pub const DEFAULT_MCG_SAMPLES: usize = 100;
pub const DEFAULT_MCB_SAMPLES: usize = 20;
pub const DEFAULT_MCP_SAMPLES: usize = 5;

/// Monte Carlo strategies: `mcg` (greedy rollouts), `mcb` (builder), `mcp` (planner).
const MC_KINDS: [(&str, Rollout, usize); 3] = [
    ("mcg", Rollout::Greedy, DEFAULT_MCG_SAMPLES),
    ("mcb", Rollout::Builder, DEFAULT_MCB_SAMPLES),
    ("mcp", Rollout::Planner, DEFAULT_MCP_SAMPLES),
];

/// Accepted strategy names: the five in [`STRATEGIES`], plus `mcg`, `mcb` and `mcp`, each
/// optionally followed by `:N` for N samples per decision (bare means the kind's default:
/// 100, 20 and 5; `:0` skips the search and plays the greedy heuristic's best move).
pub fn strategy_help() -> String {
    format!(
        "{}, mcg, mcb, mcp (each also as NAME:N)",
        STRATEGIES.join(", ")
    )
}

pub fn is_strategy(name: &str) -> bool {
    strategy(name).is_ok()
}

/// Clap value parser for strategy names.
pub fn parse_strategy_arg(name: &str) -> Result<String, String> {
    strategy(name).map(|_| name.to_string())
}

fn strategy(name: &str) -> Result<Strategy, String> {
    let (base, count) = match name.split_once(':') {
        Some((base, count)) => (base, Some(count)),
        None => (name, None),
    };
    if let Some(&(_, rollout, default)) = MC_KINDS.iter().find(|(n, _, _)| *n == base) {
        let samples = match count {
            None => default,
            Some(n) => match n.parse::<usize>() {
                Ok(samples) => samples,
                _ => {
                    return Err(format!(
                        "invalid sample count in {name:?}; use {base}:N with a nonnegative integer that fits usize"
                    ));
                }
            },
        };
        return Ok(Strategy::MonteCarlo { samples, rollout });
    }
    match name {
        "greedy" => Ok(Strategy::Greedy),
        "random" => Ok(Strategy::Random),
        "builder" => Ok(Strategy::Deterministic(Mode::Builder)),
        "tactician" => Ok(Strategy::Deterministic(Mode::Tactician)),
        "planner" => Ok(Strategy::Deterministic(Mode::Planner)),
        _ => Err(format!(
            "unknown strategy {name:?}; choose {}",
            strategy_help()
        )),
    }
}

pub struct Bot {
    strategy: Strategy,
    rng: PythonRandom,
}

pub fn partial_score(cards: &[u8]) -> f64 {
    partial_score_quarters(cards) as f64 / 4.0
}

fn partial_score_quarters(cards: &[u8]) -> i32 {
    if cards.len() == 3 {
        let s = rules::evaluate(cards) as i32;
        return (s / 32 * 100 + s % 32) * 4;
    }
    if cards.len() == 1 {
        return (cards[0] % 9 + 1) as i32 * 2;
    }
    assert_eq!(cards.len(), 2);
    let (a, b) = ((cards[0] % 9 + 1) as i32, (cards[1] % 9 + 1) as i32);
    let gap = (a - b).abs();
    let same = cards[0] / 9 == cards[1] / 9;
    if same && gap == 1 {
        240 + a + b
    } else if gap == 0 {
        200 + a * 4
    } else if same && gap == 2 {
        180
    } else if same {
        140 + a + b
    } else if gap == 1 {
        120 + a + b
    } else {
        (a + b) * 2
    }
}

fn can_claim(s: &ViewStone, play: Option<u8>, pool: u64) -> bool {
    if s.claimed_by.is_some() {
        return false;
    }
    let mut mine = s.yours;
    let mut first = s.first_to_complete;
    if let Some(c) = play {
        mine.push(c);
        if mine.len() == 3 && first.is_none() {
            first = Some(true);
        }
    }
    mine.len() == 3
        && rules::claim_is_provable(
            mine.as_slice(),
            s.theirs.as_slice(),
            pool,
            first == Some(true),
        )
}

impl Bot {
    pub fn new(name: &str, seed: Option<i64>) -> Result<Self, String> {
        let strategy = strategy(name)?;
        Ok(Self {
            strategy,
            rng: if matches!(strategy, Strategy::Deterministic(_)) {
                PythonRandom::seeded(0)
            } else {
                PythonRandom::new(seed)
            },
        })
    }

    pub fn provable_stones(view: &PlayerView, play: Option<&Play>) -> Vec<usize> {
        (0..9)
            .filter(|&i| {
                can_claim(
                    &view.stones[i],
                    play.filter(|p| p.stone == i).map(|p| p.card.id()),
                    view.unseen,
                )
            })
            .collect()
    }

    pub fn score(view: &PlayerView, play: &Play) -> f64 {
        Self::score_quarters(view, play) as f64 / 4.0
    }

    fn score_quarters(view: &PlayerView, play: &Play) -> i32 {
        let s = &view.stones[play.stone];
        let mut mine = s.yours;
        mine.push(play.card.id());
        let mut score = partial_score_quarters(mine.as_slice());
        if mine.len() == 3 && can_claim(s, Some(play.card.id()), view.unseen) {
            score += 4000;
        } else if s.theirs.len() == 3 && mine.len() == 3 {
            score += if rules::evaluate(mine.as_slice()) > rules::evaluate(s.theirs.as_slice()) {
                20
            } else {
                -80
            };
        }
        score
    }

    /// Monte Carlo decision. The search seed is drawn from this bot's own seeded RNG, so a
    /// given agent seed and game always replay identically, whatever the worker count.
    fn choose_monte_carlo(
        &mut self,
        view: &PlayerView,
        samples: usize,
        rollout: Rollout,
    ) -> (Option<Play>, Vec<usize>) {
        let seed = self.rng.bits(64);
        let cfg = montecarlo::McConfig {
            samples,
            // Tournaments already run one game per core; nested threads would only contend.
            threads: 1,
            seed,
            rollout,
            ..Default::default()
        };
        let obs = observation(view);
        let decision = montecarlo::decide(&obs, &cfg)
            .unwrap_or_else(|e| panic!("monte carlo agent rejected a player view: {e}"));
        let play = decision.play.map(|(card, stone)| Play {
            card: Card::from_id(card),
            stone: stone as usize,
        });
        let claims = decision.claims.into_iter().map(usize::from).collect();
        (play, claims)
    }

    pub fn choose(&mut self, view: &PlayerView) -> (Option<Play>, Vec<usize>) {
        if let Strategy::MonteCarlo { samples, rollout } = self.strategy {
            return self.choose_monte_carlo(view, samples, rollout);
        }
        if let Strategy::Deterministic(mode) = self.strategy {
            let play = deterministic::choose(view, mode);
            let claims = Self::provable_stones(view, play.as_ref());
            return (play, claims);
        }
        let mut choices = Vec::with_capacity(54);
        let mut best = i32::MIN;
        for card in rules::cards(view.hand) {
            for stone in 0..9 {
                if view.stones[stone].claimed_by.is_some() || view.stones[stone].yours.len() >= 3 {
                    continue;
                }
                let play = Play {
                    card: Card::from_id(card),
                    stone,
                };
                match self.strategy {
                    Strategy::Random => choices.push(play),
                    Strategy::Greedy => {
                        let score = Self::score_quarters(view, &play);
                        if score > best {
                            best = score;
                            choices.clear();
                        }
                        if score == best {
                            choices.push(play);
                        }
                    }
                    Strategy::Deterministic(_) | Strategy::MonteCarlo { .. } => unreachable!(),
                }
            }
        }
        let play = if choices.is_empty() {
            None
        } else {
            Some(choices.swap_remove(self.rng.below(choices.len())))
        };
        let claims = Self::provable_stones(view, play.as_ref());
        (play, claims)
    }
}

/// The Monte Carlo agent's input, built only from what `view` shows. Seats are not part of a
/// view, so the agent is always "seat 0"; hand size and deck size follow from the unseen pool:
/// the opponent keeps a full hand of 6 while the deck lasts, then holds whatever remains.
fn observation(view: &PlayerView) -> Observation {
    const HAND_SIZE: usize = 6;
    let side = |flag: Option<bool>| flag.map(|mine| if mine { Side::You } else { Side::Them });
    let unseen = view.unseen.count_ones() as usize;
    let opp_hand_count = unseen.min(HAND_SIZE);
    Observation {
        you: 0,
        hand: rules::cards(view.hand).collect(),
        stones: view
            .stones
            .iter()
            .map(|s| StoneObs {
                yours: s.yours.as_slice().to_vec(),
                theirs: s.theirs.as_slice().to_vec(),
                claimed: side(s.claimed_by),
                first: side(s.first_to_complete),
            })
            .collect(),
        deck_count: unseen - opp_hand_count,
        opp_hand_count,
        unseen: view.unseen,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Outcome {
    winner: Option<usize>,
    reason: String,
    turns: usize,
}

pub(crate) struct GameResult {
    pub winner: Option<usize>,
    pub reason: String,
    pub turns: usize,
    pub log: Option<serde_json::Value>,
}

fn native_game(
    index: usize,
    names: [&str; 2],
    seed: Option<i64>,
    bot_seed: Option<i64>,
) -> Result<Outcome, String> {
    let order = if index.is_multiple_of(2) {
        [0, 1]
    } else {
        [1, 0]
    };
    let deck_seed = seed.map(|base| base as i128 + index as i128);
    let agent_seeds = order
        .map(|entrant| bot_seed.map(|base| base as i128 + index as i128 * 2 + entrant as i128));
    let result = play_game(
        String::new(),
        [names[order[0]], names[order[1]]],
        deck_seed,
        agent_seeds,
        false,
    )
    .map_err(|error| format!("game {}: {error}", index + 1))?;
    Ok(Outcome {
        winner: result.winner.map(|seat| order[seat]),
        reason: result.reason,
        turns: result.turns,
    })
}

pub(crate) fn play_game(
    game_id: String,
    names: [&str; 2],
    deck_seed: Option<i128>,
    agent_seeds: [Option<i128>; 2],
    record: bool,
) -> Result<GameResult, String> {
    let mut game = Game::new(
        game_id,
        Options {
            seed: deck_seed.map(|s| i64::try_from(s).unwrap_or(0)),
            ..Default::default()
        },
        0,
        record,
    );
    if let Some(seed) = deck_seed
        && i64::try_from(seed).is_err()
    {
        let mut n = seed.unsigned_abs();
        let mut words = Vec::new();
        loop {
            words.push(n as u32);
            n >>= 32;
            if n == 0 {
                break;
            }
        }
        game.reseed_words(&words).map_err(|e| e.to_string())?;
    }
    game.join(0).map_err(|e| e.to_string())?;
    let make_bot = |seat: usize| -> Result<Bot, String> {
        let strategy = strategy(names[seat])?;
        Ok(Bot {
            strategy,
            rng: if matches!(strategy, Strategy::Deterministic(_)) {
                PythonRandom::seeded(0)
            } else {
                match agent_seeds[seat] {
                    Some(seed) => PythonRandom::seeded(seed),
                    None => PythonRandom::new(None),
                }
            },
        })
    };
    let mut bots = [make_bot(0)?, make_bot(1)?];
    let mut turns = 0;
    while !game.over() {
        if turns >= 128 {
            return Err("game did not terminate".into());
        }
        let seat = game.turn.unwrap();
        let view = game.player_view(seat);
        let (play, claims) = bots[seat].choose(&view);
        game.play_turn(
            seat,
            TurnRequest {
                version: view.version as i64,
                turn_id: None,
                play,
                claims,
            },
            0,
        )
        .map_err(|e| e.to_string())?;
        turns += 1;
    }
    let (winner, reason) = game.outcome().unwrap();
    Ok(GameResult {
        winner,
        reason: reason.into(),
        turns,
        log: record.then(|| game.log_view()),
    })
}

fn run_series(
    games: usize,
    name: &str,
    opponent: &str,
    seed: Option<i64>,
    bot_seed: Option<i64>,
    threads: usize,
) -> Result<Vec<Outcome>, String> {
    if games == 0 {
        return Err("games must be at least 1".into());
    }
    strategy(name)?;
    strategy(opponent)?;
    rules::warmup();
    let mut builder = rayon::ThreadPoolBuilder::new();
    if threads > 0 {
        builder = builder.num_threads(threads);
    }
    let pool = builder.build().map_err(|e| e.to_string())?;
    pool.install(|| {
        (0..games)
            .into_par_iter()
            .map(|index| native_game(index, [name, opponent], seed, bot_seed))
            .collect()
    })
}

pub fn series(
    games: usize,
    strategy: &str,
    opponent: &str,
    seed: Option<i64>,
    bot_seed: Option<i64>,
    threads: usize,
    quiet: bool,
) -> Result<(), String> {
    let started = Instant::now();
    let results = run_series(games, strategy, opponent, seed, bot_seed, threads)?;
    let elapsed = started.elapsed().as_secs_f64();
    let labels = if strategy == opponent {
        [format!("{strategy} (A)"), format!("{opponent} (B)")]
    } else {
        [strategy.into(), opponent.into()]
    };
    let mut wins = [0usize; 2];
    let mut draws = 0;
    for (i, r) in results.iter().enumerate() {
        let winner = if let Some(w) = r.winner {
            wins[w] += 1;
            labels[w].as_str()
        } else {
            draws += 1;
            "draw"
        };
        if !quiet {
            println!("Game {}/{}: winner={} ({})", i + 1, games, winner, r.reason);
        }
    }
    if !quiet {
        println!("\nCompleted: {games}/{games}");
    }
    for (i, label) in labels.iter().enumerate() {
        println!(
            "{label}: {} wins ({:.1}%)",
            wins[i],
            wins[i] as f64 / games as f64 * 100.0
        );
    }
    println!("Draws: {draws}");
    println!("Elapsed: {elapsed:.2}s");
    println!(
        "Throughput: {:.2} games/s",
        games as f64 / elapsed.max(f64::MIN_POSITIVE)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_match_recording_preserves_decisions_and_replays() {
        let plain = play_game(
            "match".into(),
            ["greedy", "random"],
            Some(7),
            [Some(3), Some(4)],
            false,
        )
        .unwrap();
        let recorded = play_game(
            "match".into(),
            ["greedy", "random"],
            Some(7),
            [Some(3), Some(4)],
            true,
        )
        .unwrap();
        assert_eq!(
            (plain.winner, plain.reason, plain.turns),
            (recorded.winner, recorded.reason, recorded.turns)
        );
        assert!(plain.log.is_none());
        let log = recorded.log.unwrap();
        let mut replay = Game::new(
            "match".into(),
            Options {
                seed: Some(7),
                ..Default::default()
            },
            0,
            true,
        );
        replay.join(0).unwrap();
        for entry in log["entries"].as_array().unwrap() {
            let request = serde_json::from_value(serde_json::json!({"version":replay.version,"play":entry["play"],"claims":entry["claims"]})).unwrap();
            replay
                .play_turn(entry["seat"].as_u64().unwrap() as usize, request, 0)
                .unwrap();
        }
        assert_eq!(replay.log_view(), log);
        assert!(replay.over());
    }
    #[test]
    fn new_strategy_series_ignore_bot_seeds_and_worker_count() {
        for name in ["builder", "tactician", "planner"] {
            let single = run_series(2, name, "tactician", Some(7), Some(1), 1).unwrap();
            let parallel = run_series(2, name, "tactician", Some(7), Some(999), 4).unwrap();
            assert_eq!(single, parallel, "{name}");
        }
    }
    #[test]
    fn python_series_schedule_and_thread_count_have_identical_results() {
        let a = run_series(12, "greedy", "random", Some(1), Some(3), 1).unwrap();
        let b = run_series(12, "greedy", "random", Some(1), Some(3), 4).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 12);
        assert_eq!(
            a.iter()
                .map(|r| (r.winner, r.reason.as_str(), r.turns))
                .collect::<Vec<_>>(),
            vec![
                (Some(0), "three_adjacent", 43),
                (Some(0), "five_stones", 36),
                (Some(0), "three_adjacent", 35),
                (Some(0), "three_adjacent", 42),
                (Some(0), "three_adjacent", 35),
                (Some(0), "three_adjacent", 38),
                (Some(0), "three_adjacent", 43),
                (Some(0), "five_stones", 42),
                (Some(0), "five_stones", 37),
                (Some(0), "five_stones", 40),
                (Some(0), "five_stones", 41),
                (Some(0), "five_stones", 36)
            ]
        );
    }
    #[test]
    fn full_transitions_match_python_views_legal_moves_events_and_logs() {
        for (seed, bot_seed, names, auto_claim, expected) in [
            (1, 1, ["greedy", "random"], false, 15699770788672255273u64),
            (7, 7, ["random", "random"], false, 8088702801527338241),
            (0, 3, ["greedy", "greedy"], true, 14083120144195975732),
            (-1, 3, ["greedy", "random"], true, 11013017854945848014),
        ] {
            let mut g = Game::new(
                "oracle".into(),
                Options {
                    seed: Some(seed),
                    auto_claim,
                    ..Default::default()
                },
                0,
                true,
            );
            let mut bots = [
                Bot::new(names[0], Some(bot_seed)).unwrap(),
                Bot::new(names[1], Some(bot_seed + 1)).unwrap(),
            ];
            let mut hash = 14695981039346656037u64;
            let update = |g: &Game, hash: &mut u64| {
                let v = serde_json::json!({"a":g.view(0),"b":g.view(1),"events":g.events,
                    "legal":g.turn.map(|s| g.legal_moves(s).unwrap()),"log":g.log_view()});
                for byte in serde_json::to_vec(&v).unwrap() {
                    *hash = (*hash ^ byte as u64).wrapping_mul(1099511628211);
                }
            };
            update(&g, &mut hash);
            g.join(0).unwrap();
            update(&g, &mut hash);
            while !g.over() {
                let seat = g.turn.unwrap();
                let view = g.player_view(seat);
                let (play, claims) = bots[seat].choose(&view);
                g.play_turn(
                    seat,
                    TurnRequest {
                        version: g.version as i64,
                        turn_id: None,
                        play,
                        claims,
                    },
                    0,
                )
                .unwrap();
                update(&g, &mut hash);
            }
            assert_eq!(
                hash, expected,
                "seed={seed}, names={names:?}, auto_claim={auto_claim}"
            );
        }
    }
    #[test]
    fn python_hundred_game_reference_totals() {
        let outcomes = run_series(100, "greedy", "random", Some(1), Some(1), 4).unwrap();
        assert_eq!(outcomes.iter().filter(|r| r.winner == Some(0)).count(), 95);
        assert_eq!(outcomes.iter().filter(|r| r.winner == Some(1)).count(), 5);
        assert_eq!(outcomes.iter().filter(|r| r.winner.is_none()).count(), 0);
    }
    #[test]
    fn bots_use_only_player_view_and_conserve_all_cards() {
        for strategy in STRATEGIES {
            for auto_claim in [false, true] {
                let mut g = Game::new(
                    "test".into(),
                    Options {
                        seed: Some(7),
                        auto_claim,
                        ..Default::default()
                    },
                    0,
                    true,
                );
                g.join(0).unwrap();
                let mut bots = [
                    Bot::new(strategy, Some(7)).unwrap(),
                    Bot::new(strategy, Some(8)).unwrap(),
                ];
                for _ in 0..120 {
                    if g.over() {
                        break;
                    }
                    let seat = g.turn.unwrap();
                    let v = g.player_view(seat);
                    let (play, claims) = bots[seat].choose(&v);
                    g.play_turn(
                        seat,
                        TurnRequest {
                            version: g.version as i64,
                            turn_id: None,
                            play,
                            claims,
                        },
                        0,
                    )
                    .unwrap();
                    let a = g.view(0);
                    let b = g.view(1);
                    let mut cards = Vec::new();
                    for hand in [&a["hand"], &b["hand"]] {
                        for c in hand.as_array().unwrap() {
                            cards.push(serde_json::from_value::<Card>(c.clone()).unwrap().id());
                        }
                    }
                    for s in a["stones"].as_array().unwrap() {
                        for side in ["yours", "theirs"] {
                            for c in s[side].as_array().unwrap() {
                                cards.push(serde_json::from_value::<Card>(c.clone()).unwrap().id());
                            }
                        }
                    }
                    let public = rules::mask(cards.iter().copied());
                    assert_eq!(public.count_ones() as usize, cards.len());
                    assert_eq!(cards.len() + a["deck_count"].as_u64().unwrap() as usize, 54);
                }
                assert!(g.over());
            }
        }
    }
    #[test]
    fn monte_carlo_strategy_names_parse() {
        for (base, kind, default) in [
            ("mcg", Rollout::Greedy, 100),
            ("mcb", Rollout::Builder, 20),
            ("mcp", Rollout::Planner, 5),
        ] {
            assert!(matches!(
                strategy(base),
                Ok(Strategy::MonteCarlo { samples, rollout }) if rollout == kind && samples == default
            ));
            assert!(matches!(
                strategy(&format!("{base}:0")),
                Ok(Strategy::MonteCarlo { samples: 0, rollout }) if rollout == kind
            ));
            assert!(matches!(
                strategy(&format!("{base}:250")),
                Ok(Strategy::MonteCarlo { samples: 250, rollout }) if rollout == kind
            ));
            for samples in [1_000_001, usize::MAX] {
                assert!(is_strategy(&format!("{base}:{samples}")));
            }
            for bad in [
                ":",
                ":-1",
                ":x",
                ":1.5",
                ":18446744073709551616",
                ":5:5",
                ":",
            ] {
                let name = format!("{base}{bad}");
                assert!(strategy(&name).is_err(), "{name}");
            }
        }
        // the old name is gone: it was renamed mcg
        for bad in ["mc", "mc:20", "mcx", "mcg200", "MCG", "mcg:", ""] {
            assert!(!is_strategy(bad), "{bad}");
        }
        assert!(is_strategy("planner"));
    }
    #[test]
    fn monte_carlo_plays_complete_legal_games_from_the_player_view() {
        // Every decision is validated inside the agent (all 54 cards accounted for), and the
        // engine rejects illegal plays or claims, so finishing is the check.
        for name in ["mcg:6", "mcb:3", "mcp:1"] {
            for auto_claim in [false, true] {
                for opponent in ["mcg:2", "greedy", "random"] {
                    let mut g = Game::new(
                        "mc".into(),
                        Options {
                            seed: Some(11),
                            auto_claim,
                            ..Default::default()
                        },
                        0,
                        false,
                    );
                    g.join(0).unwrap();
                    let mut bots = [
                        Bot::new(name, Some(1)).unwrap(),
                        Bot::new(opponent, Some(2)).unwrap(),
                    ];
                    while !g.over() {
                        let seat = g.turn.unwrap();
                        let v = g.player_view(seat);
                        let (play, claims) = bots[seat].choose(&v);
                        g.play_turn(
                            seat,
                            TurnRequest {
                                version: g.version as i64,
                                turn_id: None,
                                play,
                                claims,
                            },
                            0,
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
    #[test]
    fn monte_carlo_series_are_seeded_and_independent_of_worker_count() {
        for name in ["mcg:6", "mcb:3", "mcp:1"] {
            let single = run_series(2, name, "greedy", Some(3), Some(9), 1).unwrap();
            let parallel = run_series(2, name, "greedy", Some(3), Some(9), 4).unwrap();
            assert_eq!(single, parallel, "{name}");
            let again = run_series(2, name, "greedy", Some(3), Some(9), 2).unwrap();
            assert_eq!(single, again, "{name}");
        }
    }
    #[test]
    fn rollout_policies_choose_what_the_native_policy_chooses() {
        // The simulator's view of a seat must be the same view the engine would give, or the
        // builder/planner rollouts would not be playing those policies.
        use crate::montecarlo::{obs::Observation, sim::Sim};
        let _ = std::marker::PhantomData::<(Observation, Sim)>;
        let mut g = Game::new(
            "r".into(),
            Options {
                seed: Some(9),
                ..Default::default()
            },
            0,
            false,
        );
        g.join(0).unwrap();
        let mut bots = [
            Bot::new("builder", None).unwrap(),
            Bot::new("planner", None).unwrap(),
        ];
        for _ in 0..30 {
            if g.over() {
                break;
            }
            let seat = g.turn.unwrap();
            let view = g.player_view(seat);
            for (mode, name) in [(Mode::Builder, "builder"), (Mode::Planner, "planner")] {
                let expected = deterministic::choose(&view, mode);
                let observation = observation(&view);
                // rebuild the simulator from the observation; with a full-information
                // "world" the seat's own view is unchanged by the hidden split
                let mut rng = montecarlo::rng::Rng::new(1);
                let sim = montecarlo::sim::Sim::from_observation(&observation, &mut rng);
                let kind = if name == "builder" {
                    Rollout::Builder
                } else {
                    Rollout::Planner
                };
                let got = montecarlo::policy::rollout_move_with(kind, &sim, 0, &mut rng, 0.0);
                if let Some(play) = expected {
                    assert_eq!(got, Some((play.card.id(), play.stone as u8)), "{name}");
                } else {
                    assert_eq!(got, None);
                }
            }
            let (play, claims) = bots[seat].choose(&view);
            g.play_turn(
                seat,
                TurnRequest {
                    version: g.version as i64,
                    turn_id: None,
                    play,
                    claims,
                },
                0,
            )
            .unwrap();
        }
    }
    #[test]
    fn monte_carlo_with_zero_samples_matches_the_greedy_heuristic_choice() {
        // mcg:0 plays the heuristic's best move; greedy picks randomly among ties, so compare
        // that mcg:0's move is one of greedy's top-scoring moves.
        let mut g = Game::new(
            "z".into(),
            Options {
                seed: Some(5),
                ..Default::default()
            },
            0,
            false,
        );
        g.join(0).unwrap();
        let seat = g.turn.unwrap();
        let v = g.player_view(seat);
        let (play, _) = Bot::new("mcg:0", Some(1)).unwrap().choose(&v);
        let play = play.unwrap();
        let best = rules::cards(v.hand)
            .flat_map(|c| {
                (0..9).map(move |s| Play {
                    card: Card::from_id(c),
                    stone: s,
                })
            })
            .map(|p| Bot::score(&v, &p))
            .fold(f64::MIN, f64::max);
        assert_eq!(Bot::score(&v, &play), best);
    }
}
