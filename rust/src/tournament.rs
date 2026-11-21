use crate::bots;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seed {
    Fixed(i64),
    Random,
}

impl Seed {
    fn resolve(self) -> Result<Self, String> {
        match self {
            Self::Fixed(_) => Ok(self),
            Self::Random => {
                let mut bytes = [0u8; 8];
                getrandom::fill(&mut bytes)
                    .map_err(|error| format!("cannot obtain seed from OS randomness: {error}"))?;
                Ok(Self::Fixed(i64::from_le_bytes(bytes) & i64::MAX))
            }
        }
    }

    pub(crate) fn value(self) -> i64 {
        match self {
            Self::Fixed(value) => value,
            Self::Random => panic!("random seeds must be resolved before scheduling games"),
        }
    }
}

impl From<i64> for Seed {
    fn from(value: i64) -> Self {
        Self::Fixed(value)
    }
}

impl FromStr for Seed {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        if value == "random" {
            Ok(Self::Random)
        } else {
            value
                .parse::<i64>()
                .map(Self::Fixed)
                .map_err(|_| format!("seed must be an integer or random, got {value:?}"))
        }
    }
}

impl std::fmt::Display for Seed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(value) => value.fmt(formatter),
            Self::Random => formatter.write_str("random"),
        }
    }
}

impl Serialize for Seed {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Fixed(value) => serializer.serialize_i64(*value),
            Self::Random => serializer.serialize_str("random"),
        }
    }
}

impl<'de> Deserialize<'de> for Seed {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Input {
            Integer(i64),
            Text(String),
        }
        match Input::deserialize(deserializer)? {
            Input::Integer(value) => Ok(Self::Fixed(value)),
            Input::Text(value) if value == "random" => Ok(Self::Random),
            Input::Text(value) => Err(serde::de::Error::custom(format!(
                "seed must be an integer or random, got {value:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentSpec {
    Strategy(String),
    Detailed(AgentConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    pub name: String,
    pub strategy: String,
    #[serde(default)]
    pub seed: Option<Seed>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Recording {
    Count(usize),
    All(String),
}

impl Default for Recording {
    fn default() -> Self {
        Self::Count(0)
    }
}

impl FromStr for Recording {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "all" {
            Ok(Self::All(value.into()))
        } else {
            value.parse::<usize>().map(Self::Count).map_err(|_| {
                format!("record_games must be a nonnegative integer or \"all\", got {value:?}")
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub agents: Vec<AgentSpec>,
    pub games_per_pair: usize,
    pub deck_seed: Seed,
    pub agent_seed: Seed,
    pub record_games: Recording,
    pub record_dir: PathBuf,
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            agents: bots::STRATEGIES
                .iter()
                .map(|name| AgentSpec::Strategy((*name).into()))
                .collect(),
            games_per_pair: 100,
            deck_seed: Seed::Fixed(1),
            agent_seed: Seed::Fixed(1),
            record_games: Recording::default(),
            record_dir: "tournament-recordings".into(),
            threads: 0,
        }
    }
}

impl Config {
    pub fn resolve_seeds(&self) -> Result<Self, String> {
        let mut resolved = self.clone();
        resolved.deck_seed = resolved.deck_seed.resolve()?;
        resolved.agent_seed = resolved.agent_seed.resolve()?;
        for spec in &mut resolved.agents {
            if let AgentSpec::Detailed(agent) = spec {
                agent.seed = agent.seed.map(Seed::resolve).transpose()?;
            }
        }
        Ok(resolved)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.agents.len() < 2 {
            return Err("agents must contain at least two entrants".into());
        }
        let mut names = HashSet::new();
        for spec in &self.agents {
            let (name, strategy) = match spec {
                AgentSpec::Strategy(strategy) => (strategy.as_str(), strategy.as_str()),
                AgentSpec::Detailed(agent) => (agent.name.as_str(), agent.strategy.as_str()),
            };
            if name.trim().is_empty() || name.chars().any(char::is_control) {
                return Err(format!(
                    "entrant name must be nonempty and contain no control characters: {:?}",
                    name
                ));
            }
            if !names.insert(name) {
                return Err(format!("duplicate entrant name {:?}", name));
            }
            if !bots::is_strategy(strategy) {
                return Err(format!(
                    "unknown strategy {:?}; choose {}",
                    strategy,
                    bots::strategy_help()
                ));
            }
        }
        if self.games_per_pair == 0 {
            return Err("games_per_pair must be at least 1".into());
        }
        let total = self.total_games()?;
        self.recording_count(total)?;
        Ok(())
    }

    fn total_games(&self) -> Result<usize, String> {
        let n = self.agents.len();
        let (a, b) = if n.is_multiple_of(2) {
            (n / 2, n.saturating_sub(1))
        } else {
            (n, n.saturating_sub(1) / 2)
        };
        a.checked_mul(b)
            .and_then(|pairs| pairs.checked_mul(self.games_per_pair))
            .ok_or_else(|| "total tournament game count overflows usize".into())
    }

    fn recording_count(&self, total: usize) -> Result<usize, String> {
        match &self.record_games {
            Recording::Count(count) if *count <= total => Ok(*count),
            Recording::Count(count) => {
                Err(format!("record_games {count} exceeds total games {total}"))
            }
            Recording::All(value) if value == "all" => Ok(total),
            Recording::All(value) => Err(format!(
                "record_games must be a nonnegative integer or \"all\", got {value:?}"
            )),
        }
    }

    fn entrants(&self) -> Vec<Entrant<'_>> {
        self.agents
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let (name, strategy, seed) = match spec {
                    AgentSpec::Strategy(strategy) => (strategy.as_str(), strategy.as_str(), None),
                    AgentSpec::Detailed(agent) => {
                        (agent.name.as_str(), agent.strategy.as_str(), agent.seed)
                    }
                };
                Entrant {
                    name,
                    strategy,
                    seed: seed
                        .map(|seed| i128::from(seed.value()))
                        .unwrap_or(self.agent_seed.value() as i128 + index as i128),
                }
            })
            .collect()
    }
}

struct Entrant<'a> {
    name: &'a str,
    strategy: &'a str,
    seed: i128,
}

#[derive(Debug, Serialize)]
pub struct Standing {
    pub name: String,
    pub strategy: String,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub played: usize,
}

/// One entrant's record against one opponent.
#[derive(Debug, Default, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct Record {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
}

impl Record {
    pub fn played(&self) -> usize {
        self.wins + self.losses + self.draws
    }
}

#[derive(Debug, Serialize)]
pub struct Results {
    pub standings: Vec<Standing>,
    /// `head_to_head[i][j]` is standing `i`'s record against standing `j` (same order as
    /// `standings`). The diagonal is empty.
    pub head_to_head: Vec<Vec<Record>>,
    pub games_played: usize,
    pub total_games: usize,
    pub draws: usize,
    pub elapsed_seconds: f64,
    pub recording_dir: Option<PathBuf>,
}

struct Aggregate {
    standings: Vec<Standing>,
    /// Indexed by entrant input order: `[row][column]` is row's record against column.
    head_to_head: Vec<Vec<Record>>,
    draws: usize,
}

impl Aggregate {
    fn new(entrants: &[Entrant<'_>]) -> Self {
        Self {
            standings: entrants
                .iter()
                .map(|agent| Standing {
                    name: agent.name.into(),
                    strategy: agent.strategy.into(),
                    wins: 0,
                    losses: 0,
                    draws: 0,
                    played: 0,
                })
                .collect(),
            head_to_head: vec![vec![Record::default(); entrants.len()]; entrants.len()],
            draws: 0,
        }
    }

    /// Sort standings by victories (stable, so ties keep input order) and reorder the
    /// head-to-head matrix to match.
    fn into_sorted(self) -> (Vec<Standing>, Vec<Vec<Record>>, usize) {
        let mut order: Vec<usize> = (0..self.standings.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.standings[i].wins));
        let matrix = order
            .iter()
            .map(|&row| {
                order
                    .iter()
                    .map(|&col| self.head_to_head[row][col])
                    .collect()
            })
            .collect();
        let mut slots: Vec<Option<Standing>> = self.standings.into_iter().map(Some).collect();
        let standings = order.iter().map(|&i| slots[i].take().unwrap()).collect();
        (standings, matrix, self.draws)
    }

    fn add(&mut self, seats: [usize; 2], winner: Option<usize>) {
        for seat in 0..2 {
            let standing = &mut self.standings[seats[seat]];
            let record = &mut self.head_to_head[seats[seat]][seats[1 - seat]];
            standing.played += 1;
            match winner {
                Some(winner) if winner == seat => {
                    standing.wins += 1;
                    record.wins += 1;
                }
                Some(_) => {
                    standing.losses += 1;
                    record.losses += 1;
                }
                None => {
                    standing.draws += 1;
                    record.draws += 1;
                }
            }
        }
        if winner.is_none() {
            self.draws += 1;
        }
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn prepare_recordings(
    config: &Config,
    entrants: &[Entrant<'_>],
    total: usize,
    count: usize,
) -> Result<PathBuf, String> {
    fs::create_dir_all(&config.record_dir)
        .map_err(|error| format!("{}: {error}", config.record_dir.display()))?;
    let directory = loop {
        let path = config.record_dir.join(uuid::Uuid::new_v4().to_string());
        match fs::create_dir(&path) {
            Ok(()) => break path,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("{}: {error}", path.display())),
        }
    };
    let agents: Vec<_> = entrants.iter().enumerate().map(|(index, agent)| serde_json::json!({
        "entrant_index":index, "name":agent.name, "strategy":agent.strategy, "seed":agent.seed.to_string()
    })).collect();
    write_json(
        &directory.join("manifest.json"),
        &serde_json::json!({
            "schema_version":1, "config":config, "agents":agents,
            "total_games":total, "recorded_games":count,
            "schedule":{
                "pair_order":"unordered pairs in entrant input order",
                "round_order":"ascending from zero within each pair",
                "starting_seat":"first entrant on even rounds, second entrant on odd rounds",
                "deck_seed":"deck_seed + round",
                "agent_seed":"resolved entrant seed + round"
            }
        }),
    )?;
    Ok(directory)
}

fn play_scheduled(
    config: &Config,
    entrants: &[Entrant<'_>],
    pairs: &[[usize; 2]],
    index: usize,
    recording_dir: Option<&Path>,
) -> Result<([usize; 2], Option<usize>), String> {
    let round = index % config.games_per_pair;
    let mut seats = pairs[index / config.games_per_pair];
    if !round.is_multiple_of(2) {
        seats.swap(0, 1);
    }
    let deck_seed = config.deck_seed.value() as i128 + round as i128;
    let agent_seeds = seats.map(|entrant| entrants[entrant].seed + round as i128);
    let game_id = format!("tournament-{}", index + 1);
    let outcome = bots::play_game(
        game_id.clone(),
        seats.map(|entrant| entrants[entrant].strategy),
        Some(deck_seed),
        agent_seeds.map(Some),
        recording_dir.is_some(),
    )
    .map_err(|error| format!("game {}: {error}", index + 1))?;
    if outcome.winner.is_some_and(|seat| seat >= 2) {
        return Err(format!(
            "game {}: invalid winner seat {:?}",
            index + 1,
            outcome.winner
        ));
    }
    if let Some(directory) = recording_dir {
        let log = outcome
            .log
            .ok_or_else(|| format!("game {}: missing replay log", index + 1))?;
        let metadata = seats
            .iter()
            .enumerate()
            .map(|(seat, &entrant)| {
                serde_json::json!({
                    "seat":seat, "entrant_index":entrant, "name":entrants[entrant].name,
                    "strategy":entrants[entrant].strategy, "seed":agent_seeds[seat].to_string()
                })
            })
            .collect::<Vec<_>>();
        write_json(
            &directory.join(format!("game-{:06}.json", index + 1)),
            &serde_json::json!({
                "schema_version":1, "game_id":game_id, "schedule_index":index, "round":round,
                "seats":metadata, "deck_seed":deck_seed.to_string(),
                "winner_seat":outcome.winner,
                "winner_entrant_index":outcome.winner.map(|seat| seats[seat]),
                "winner":outcome.winner.map(|seat| entrants[seats[seat]].name),
                "reason":outcome.reason, "turns":outcome.turns, "log":log
            }),
        )?;
    }
    Ok((seats, outcome.winner))
}

pub fn run<F>(config: &Config, mut progress: F) -> Result<Results, String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    config.validate()?;
    let resolved = config.resolve_seeds()?;
    let config = &resolved;
    let total = config.total_games()?;
    let recorded = config.recording_count(total)?;
    let started = Instant::now();
    progress(0, total)?;
    let completed = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let mut recording_dir = None;
    let result = (|| {
        let entrants = config.entrants();
        if recorded > 0 {
            recording_dir = Some(prepare_recordings(config, &entrants, total, recorded)?);
        }
        let mut builder = rayon::ThreadPoolBuilder::new();
        if config.threads > 0 {
            builder = builder.num_threads(config.threads);
        }
        let pool = builder
            .build()
            .map_err(|error| format!("tournament worker pool: {error}"))?;
        let pairs = (0..entrants.len())
            .flat_map(|a| (a + 1..entrants.len()).map(move |b| [a, b]))
            .collect::<Vec<_>>();
        let stats = Mutex::new(Aggregate::new(&entrants));
        std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::sync_channel(1);
            let (pool, entrants, pairs, stats, recording_dir, cancelled, completed) = (
                &pool,
                &entrants,
                &pairs,
                &stats,
                &recording_dir,
                &cancelled,
                &completed,
            );
            let worker = scope.spawn(move || {
                crate::rules::warmup();
                let result = pool.install(|| {
                    (0..total).into_par_iter().try_for_each(|index| {
                        if cancelled.load(Ordering::Acquire) {
                            return Err("tournament cancelled".to_string());
                        }
                        let directory = if index < recorded {
                            recording_dir.as_deref()
                        } else {
                            None
                        };
                        let (seats, winner) =
                            play_scheduled(config, entrants, pairs, index, directory)?;
                        let mut stats = stats
                            .lock()
                            .map_err(|_| "tournament standings lock poisoned".to_string())?;
                        stats.add(seats, winner);
                        completed.fetch_add(1, Ordering::Release);
                        Ok(())
                    })
                });
                let _ = sender.send(result);
            });
            let mut callback_error = None;
            let mut last_completed = 0;
            let worker_result = loop {
                match receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(result) => break result,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        break Err("tournament worker disconnected".into());
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let done = completed.load(Ordering::Acquire);
                        if callback_error.is_none() && done != last_completed && done < total {
                            if let Err(error) = progress(done, total) {
                                cancelled.store(true, Ordering::Release);
                                callback_error = Some(error);
                            }
                            last_completed = done;
                        }
                    }
                }
            };
            let joined = worker
                .join()
                .map_err(|_| "tournament worker panicked".to_string());
            if let Some(error) = callback_error {
                return Err(error);
            }
            joined?;
            worker_result
        })?;
        let aggregate = stats
            .into_inner()
            .map_err(|_| "tournament standings lock poisoned".to_string())?;
        let (standings, head_to_head, draws) = aggregate.into_sorted();
        let results = Results {
            standings,
            head_to_head,
            games_played: completed.load(Ordering::Acquire),
            total_games: total,
            draws,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            recording_dir: recording_dir.clone(),
        };
        if let Some(directory) = &recording_dir {
            write_json(&directory.join("results.json"), &results)?;
        }
        Ok(results)
    })();
    let final_progress = progress(completed.load(Ordering::Acquire), total);
    result
        .and_then(|results| final_progress.map(|_| results))
        .map_err(|error| match &recording_dir {
            Some(directory) => format!("{error} (recordings: {})", directory.display()),
            None => error,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Game, Options, TurnRequest};
    use serde_json::{Value, json};

    #[test]
    fn random_seed_requests_resolve_once_and_serialize_as_reproducible_integers() {
        let config: Config=serde_json::from_value(json!({"agents":[{"name":"a","strategy":"random","seed":"random"},"greedy"],"deck_seed":"random","agent_seed":"random","games_per_pair":2})).unwrap();
        config.validate().unwrap();
        let resolved = config.resolve_seeds().unwrap();
        let value = serde_json::to_value(&resolved).unwrap();
        assert!(value["deck_seed"].as_i64().is_some());
        assert!(value["agent_seed"].as_i64().is_some());
        assert!(value["agents"][0]["seed"].as_i64().is_some());
        assert_eq!(
            serde_json::to_value(resolved.resolve_seeds().unwrap()).unwrap(),
            value
        );
        for invalid in ["Random", "anything", ""] {
            assert!(serde_json::from_value::<Config>(json!({"deck_seed":invalid})).is_err());
        }
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("spor-tournament-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn config() -> Config {
        serde_json::from_value(json!({
            "agents": ["random", {"name":"other", "strategy":"random", "seed":-7}],
            "games_per_pair":3, "threads":2
        }))
        .unwrap()
    }

    fn read(path: impl AsRef<std::path::Path>) -> Value {
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap()
    }

    #[test]
    fn config_defaults_and_strict_validation() {
        let defaults: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(defaults.agents.len(), 5);
        assert_eq!(defaults.games_per_pair, 100);
        assert_eq!(
            (defaults.deck_seed, defaults.agent_seed, defaults.threads),
            (Seed::Fixed(1), Seed::Fixed(1), 0)
        );
        assert_eq!(defaults.record_dir, PathBuf::from("tournament-recordings"));
        assert!(matches!(defaults.record_games, Recording::Count(0)));
        defaults.validate().unwrap();
        for value in [
            json!({"agents":[]}),
            json!({"agents":["random"]}),
            json!({"agents":["random","random"]}),
            json!({"agents":["random",{"name":"", "strategy":"greedy"}]}),
            json!({"agents":["random",{"name":"  ", "strategy":"greedy"}]}),
            json!({"agents":["random",{"name":"line\nbreak", "strategy":"greedy"}]}),
            json!({"agents":["missing","random"]}),
            json!({"games_per_pair":0}),
            json!({"games_per_pair":usize::MAX}),
            json!({"record_games":1001}),
            json!({"record_games":"ALL"}),
        ] {
            let config: Config = serde_json::from_value(value.clone()).unwrap();
            assert!(config.validate().is_err(), "accepted {value}");
        }
        for value in [
            json!({"unknown":1}),
            json!({"agents":[{"name":"a","strategy":"random","typo":1}]}),
            json!({"record_games":-1}),
            json!({"record_games":true}),
        ] {
            assert!(serde_json::from_value::<Config>(value).is_err());
        }
        assert!("all".parse::<Recording>().is_ok());
        assert!("3".parse::<Recording>().is_ok());
        assert!("many".parse::<Recording>().is_err());
        let all: Config = serde_json::from_value(json!({"record_games":"all"})).unwrap();
        all.validate().unwrap();
        let detailed: AgentConfig =
            serde_json::from_value(json!({"name":"a","strategy":"random"})).unwrap();
        assert_eq!(detailed.seed, None);
    }

    #[test]
    fn workers_produce_identical_standings_and_consistent_totals() {
        let mut config = config();
        config.agents.push(AgentSpec::Detailed(AgentConfig {
            name: "third".into(),
            strategy: "greedy".into(),
            seed: None,
        }));
        config.threads = 1;
        let one = run(&config, |_, _| Ok(())).unwrap();
        config.threads = 4;
        let four = run(&config, |_, _| Ok(())).unwrap();
        assert_eq!(
            serde_json::to_value(&one.standings).unwrap(),
            serde_json::to_value(&four.standings).unwrap()
        );
        assert_eq!((one.games_played, one.total_games), (9, 9));
        assert_eq!(one.draws, four.draws);
        assert_eq!(one.standings.iter().map(|s| s.played).sum::<usize>(), 18);
        assert_eq!(
            one.standings.iter().map(|s| s.wins).sum::<usize>() + one.draws,
            9
        );
        assert_eq!(
            one.standings.iter().map(|s| s.losses).sum::<usize>() + one.draws,
            9
        );
        assert_eq!(
            one.standings.iter().map(|s| s.draws).sum::<usize>(),
            one.draws * 2
        );
        assert!(
            one.standings
                .iter()
                .all(|s| s.played == 6 && s.played == s.wins + s.losses + s.draws)
        );
        assert!(one.standings.windows(2).all(|s| s[0].wins >= s[1].wins));
    }

    #[test]
    fn progress_stays_on_caller_and_reports_completed_games() {
        let caller = std::thread::current().id();
        let mut updates = Vec::new();
        let result = run(&config(), |done, total| {
            assert_eq!(std::thread::current().id(), caller);
            updates.push((done, total, std::time::Instant::now()));
            Ok(())
        })
        .unwrap();
        assert_eq!(updates.first().map(|v| (v.0, v.1)), Some((0, 3)));
        assert_eq!(updates.last().map(|v| (v.0, v.1)), Some((3, 3)));
        assert!(updates.windows(2).all(|v| v[0].0 <= v[1].0));
        for pair in updates[..updates.len() - 1].windows(2) {
            assert!(pair[1].2.duration_since(pair[0].2) >= Duration::from_millis(100));
        }
        assert_eq!(result.games_played, 3);
    }

    #[test]
    fn callback_error_cancels_and_is_returned() {
        let mut config = config();
        config.games_per_pair = 100_000;
        let mut counts = Vec::new();
        let error = run(&config, |done, _| {
            counts.push(done);
            if counts.len() == 2 {
                Err("progress failed".into())
            } else {
                Ok(())
            }
        })
        .err()
        .unwrap();
        assert!(error.contains("progress failed"));
        assert_eq!(counts[0], 0);
        assert!(*counts.last().unwrap() < 100_000);
        assert!(counts.windows(2).all(|v| v[0] <= v[1]));
    }

    #[test]
    fn initial_callback_error_prevents_all_work_and_files() {
        let temp = TempDir::new();
        let mut config = config();
        config.record_games = Recording::All("all".into());
        config.record_dir = temp.0.join("absent");
        let error = run(&config, |_, _| Err("cannot report progress".into())).unwrap_err();
        assert!(error.contains("cannot report progress"));
        assert!(!config.record_dir.exists());
    }

    #[test]
    fn cancellation_short_circuits_even_the_largest_valid_schedule() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let mut config = config();
            config.games_per_pair = usize::MAX;
            config.threads = 1;
            let result = run(&config, |done, _| {
                if done > 0 {
                    Err("cancel large schedule".into())
                } else {
                    Ok(())
                }
            });
            sender.send(result).unwrap();
        });
        let result = receiver
            .recv_timeout(Duration::from_secs(3))
            .expect("cancellation scanned the remaining schedule");
        assert!(result.unwrap_err().contains("cancel large schedule"));
        worker.join().unwrap();
    }

    #[test]
    fn failed_recording_stops_without_counting_unwritten_games() {
        let temp = TempDir::new();
        let mut config = config();
        config.games_per_pair = 100_000;
        config.threads = 1;
        config.record_games = Recording::All("all".into());
        config.record_dir = temp.0.join("runs");
        let preserved = temp.0.join("preserved");
        let mut blocked = None;
        let mut counts = Vec::new();
        let error = run(&config, |done, _| {
            counts.push(done);
            if done > 0 && blocked.is_none() {
                let directory = fs::read_dir(&config.record_dir)
                    .unwrap()
                    .next()
                    .unwrap()
                    .unwrap()
                    .path();
                fs::rename(&directory, &preserved).unwrap();
                fs::write(&directory, b"blocked").unwrap();
                blocked = Some(directory);
            }
            Ok(())
        })
        .unwrap_err();
        let directory = blocked.unwrap();
        assert!(error.contains(directory.to_str().unwrap()));
        let count = fs::read_dir(&preserved)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_str()
                    .unwrap()
                    .starts_with("game-")
            })
            .count();
        assert_eq!(counts.first(), Some(&0));
        assert_eq!(counts.last(), Some(&count));
        assert!(count > 0 && count < config.games_per_pair);
        assert!(!preserved.join("results.json").exists());
    }

    #[test]
    fn head_to_head_matrix_follows_standings_order_and_matches_totals() {
        let config = config();
        let agents = config.entrants();
        let mut aggregate = Aggregate::new(&agents);
        aggregate.add([0, 1], Some(1)); // entrant 1 beats 0
        aggregate.add([1, 0], Some(0)); // entrant 1 (seat 0) beats 0 again
        aggregate.add([0, 1], None);
        let (standings, matrix, draws) = aggregate.into_sorted();
        assert_eq!(draws, 1);
        // entrant 1 has 2 wins, so it is now first; the matrix is reordered the same way
        assert_eq!(standings[0].wins, 2);
        let rec = |wins, losses, draws| Record {
            wins,
            losses,
            draws,
        };
        assert_eq!(matrix[0][1], rec(2, 0, 1));
        assert_eq!(matrix[1][0], rec(0, 2, 1));
        assert_eq!(matrix[0][0], Record::default());
        for (i, standing) in standings.iter().enumerate() {
            let row = &matrix[i];
            assert_eq!(
                row.iter().map(Record::played).sum::<usize>(),
                standing.played
            );
            assert_eq!(row.iter().map(|r| r.wins).sum::<usize>(), standing.wins);
        }
    }

    #[test]
    fn draws_and_seat_winners_aggregate_into_input_order_standings() {
        let config = config();
        let agents = config.entrants();
        let mut aggregate = Aggregate::new(&agents);
        aggregate.add([0, 1], None);
        aggregate.add([1, 0], Some(0));
        aggregate.add([0, 1], Some(0));
        aggregate
            .standings
            .sort_by_key(|standing| std::cmp::Reverse(standing.wins));
        assert_eq!(aggregate.draws, 1);
        assert_eq!(
            aggregate
                .standings
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["random", "other"]
        );
        for standing in &aggregate.standings {
            assert_eq!(
                (
                    standing.wins,
                    standing.losses,
                    standing.draws,
                    standing.played
                ),
                (1, 1, 1, 3)
            );
        }
    }

    #[test]
    fn recordings_are_identical_across_worker_counts() {
        let temp = TempDir::new();
        let mut config = config();
        config.record_games = Recording::All("all".into());
        config.record_dir = temp.0.clone();
        config.threads = 1;
        let one = run(&config, |_, _| Ok(())).unwrap().recording_dir.unwrap();
        config.threads = 4;
        let four = run(&config, |_, _| Ok(())).unwrap().recording_dir.unwrap();
        for index in 1..=3 {
            let filename = format!("game-{index:06}.json");
            assert_eq!(read(one.join(&filename)), read(four.join(&filename)));
        }
    }

    #[test]
    fn recording_zero_never_creates_or_opens_paths() {
        let temp = TempDir::new();
        let mut config = config();
        config.record_dir = temp.0.join("absent").join("nested");
        assert!(run(&config, |_, _| Ok(())).unwrap().recording_dir.is_none());
        assert!(!temp.0.join("absent").exists());
        std::fs::write(temp.0.join("file"), b"unchanged").unwrap();
        config.record_dir = temp.0.join("file");
        run(&config, |_, _| Ok(())).unwrap();
        assert_eq!(std::fs::read(&config.record_dir).unwrap(), b"unchanged");
    }

    #[test]
    fn recording_invalid_path_fails_and_runs_never_overwrite() {
        let temp = TempDir::new();
        let mut config = config();
        config.record_games = Recording::Count(1);
        config.record_dir = temp.0.join("file");
        std::fs::write(&config.record_dir, b"sentinel").unwrap();
        assert!(run(&config, |_, _| Ok(())).err().unwrap().contains("file"));
        assert_eq!(std::fs::read(&config.record_dir).unwrap(), b"sentinel");
        config.record_dir = temp.0.join("runs");
        let first = run(&config, |_, _| Ok(())).unwrap().recording_dir.unwrap();
        let original = std::fs::read(first.join("game-000001.json")).unwrap();
        let second = run(&config, |_, _| Ok(())).unwrap().recording_dir.unwrap();
        assert_ne!(first, second);
        assert_eq!(
            std::fs::read(first.join("game-000001.json")).unwrap(),
            original
        );
        assert!(second.join("manifest.json").is_file());
        assert_eq!(read(second.join("results.json"))["games_played"], 3);
    }

    #[test]
    fn recording_selects_first_scheduled_games_and_exact_seeds() {
        let temp = TempDir::new();
        let mut config = config();
        config.agent_seed = i64::MAX.into();
        config.deck_seed = i64::MAX.into();
        config.agents.push(AgentSpec::Detailed(AgentConfig {
            name: "third".into(),
            strategy: "random".into(),
            seed: None,
        }));
        config.record_games = Recording::Count(4);
        config.record_dir = temp.0.clone();
        config.threads = 4;
        let directory = run(&config, |_, _| Ok(())).unwrap().recording_dir.unwrap();
        let expected = [
            (
                0,
                [0, 1],
                "9223372036854775807",
                ["9223372036854775807", "-7"],
            ),
            (
                1,
                [1, 0],
                "9223372036854775808",
                ["-6", "9223372036854775808"],
            ),
            (
                2,
                [0, 1],
                "9223372036854775809",
                ["9223372036854775809", "-5"],
            ),
            (
                0,
                [0, 2],
                "9223372036854775807",
                ["9223372036854775807", "9223372036854775809"],
            ),
        ];
        for (index, (round, entrants, deck, seeds)) in expected.into_iter().enumerate() {
            let game = read(directory.join(format!("game-{:06}.json", index + 1)));
            assert_eq!(game["schedule_index"], index);
            assert_eq!(game["round"], round);
            assert_eq!(game["deck_seed"], deck);
            for seat in 0..2 {
                assert_eq!(game["seats"][seat]["entrant_index"], entrants[seat]);
                assert_eq!(game["seats"][seat]["seed"], seeds[seat]);
            }
        }
        assert!(!directory.join("game-000005.json").exists());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 6);
        let manifest = read(directory.join("manifest.json"));
        assert_eq!(manifest["total_games"], 9);
        assert_eq!(manifest["agents"][2]["seed"], "9223372036854775809");
    }

    #[test]
    fn all_recordings_replay_real_engine_intents_and_alternate_seats() {
        let temp = TempDir::new();
        let mut config = config();
        config.agents.push(AgentSpec::Detailed(AgentConfig {
            name: "third".into(),
            strategy: "greedy".into(),
            seed: None,
        }));
        config.record_games = Recording::All("all".into());
        config.record_dir = temp.0.clone();
        config.deck_seed = i64::MAX.into();
        let directory = run(&config, |_, _| Ok(())).unwrap().recording_dir.unwrap();
        let mut seats = [[0; 2]; 3];
        for index in 0..9 {
            let record = read(directory.join(format!("game-{:06}.json", index + 1)));
            for seat in 0..2 {
                seats[record["seats"][seat]["entrant_index"].as_u64().unwrap() as usize][seat] += 1;
            }
            replay(&record);
        }
        assert_eq!(seats, [[4, 2], [3, 3], [2, 4]]);
    }

    fn replay(record: &Value) {
        let seed: i128 = record["deck_seed"].as_str().unwrap().parse().unwrap();
        let mut game = Game::new(
            record["game_id"].as_str().unwrap().into(),
            Options {
                seed: Some(i64::try_from(seed).unwrap_or(0)),
                ..Default::default()
            },
            0,
            true,
        );
        if i64::try_from(seed).is_err() {
            let value = seed.unsigned_abs();
            let mut words = vec![
                value as u32,
                (value >> 32) as u32,
                (value >> 64) as u32,
                (value >> 96) as u32,
            ];
            while words.len() > 1 && words.last() == Some(&0) {
                words.pop();
            }
            game.reseed_words(&words).unwrap();
        }
        game.join(0).unwrap();
        let log = &record["log"];
        for entry in log["entries"].as_array().unwrap() {
            let seat = entry["seat"].as_u64().unwrap() as usize;
            assert_eq!(game.turn, Some(seat));
            game.play_turn(
                seat,
                TurnRequest {
                    version: game.version as i64,
                    turn_id: None,
                    play: serde_json::from_value(entry["play"].clone()).unwrap(),
                    claims: serde_json::from_value(entry["claims"].clone()).unwrap(),
                },
                0,
            )
            .unwrap();
        }
        assert_eq!(game.log_view(), *log);
        assert_eq!(
            game.log_view()["revealed"]["initial_hands"],
            log["revealed"]["initial_hands"]
        );
        let (winner, reason) = game.outcome().unwrap();
        assert_eq!(serde_json::to_value(winner).unwrap(), record["winner_seat"]);
        assert_eq!(reason, record["reason"].as_str().unwrap());
        assert_eq!(
            log["entries"].as_array().unwrap().len(),
            record["turns"].as_u64().unwrap() as usize
        );
    }
}
