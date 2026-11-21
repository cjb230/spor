use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use crate::tournament::{self, AgentConfig, AgentSpec, Config, Recording, Results, Seed};

#[derive(clap::Args, Default)]
pub struct Args {
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long, num_args=2.., help="Policies or name=policy entrants; defaults to all native policies")]
    pub agents: Option<Vec<String>>,
    #[arg(long, alias = "games")]
    pub games_per_pair: Option<usize>,
    #[arg(
        long,
        alias = "seed",
        allow_negative_numbers = true,
        help = "Starting deck seed: integer or random"
    )]
    pub deck_seed: Option<Seed>,
    #[arg(
        long,
        allow_negative_numbers = true,
        help = "Starting agent seed base: integer or random"
    )]
    pub agent_seed: Option<Seed>,
    #[arg(long, num_args=1.., allow_negative_numbers=true, help="Individual starting seeds in entrant order: integers or random")]
    pub agent_seeds: Option<Vec<Seed>>,
    #[arg(long, num_args=0..=1, default_missing_value="all", help="Record 0, N, or all games; bare flag records all")]
    pub record_games: Option<String>,
    #[arg(long)]
    pub record_dir: Option<PathBuf>,
    #[arg(long)]
    pub threads: Option<usize>,
}

impl Args {
    pub fn resolve(self) -> Result<Config, String> {
        let mut config = if let Some(path) = self.config {
            let text = fs::read_to_string(&path)
                .map_err(|error| format!("cannot read config {}: {error}", path.display()))?;
            let value: serde_json::Value = serde_json::from_str(&text)
                .map_err(|error| format!("invalid config {}: {error}", path.display()))?;
            if !value.is_object() {
                return Err("tournament config must be a JSON object".into());
            }
            if let Some(agents) = value.get("agents").and_then(serde_json::Value::as_array)
                && agents
                    .iter()
                    .any(|agent| !agent.is_string() && !agent.is_object())
            {
                return Err("each config agent must be a policy name or an object".into());
            }
            serde_json::from_str::<Config>(&text)
                .map_err(|error| format!("invalid config {}: {error}", path.display()))?
        } else {
            Config::default()
        };
        if let Some(agents) = self.agents {
            config.agents = agents
                .into_iter()
                .map(|agent| {
                    if let Some((name, strategy)) = agent.split_once('=') {
                        AgentSpec::Detailed(AgentConfig {
                            name: name.into(),
                            strategy: strategy.into(),
                            seed: None,
                        })
                    } else {
                        AgentSpec::Strategy(agent)
                    }
                })
                .collect();
        }
        if let Some(value) = self.games_per_pair {
            config.games_per_pair = value;
        }
        if let Some(value) = self.deck_seed {
            config.deck_seed = value;
        }
        if let Some(value) = self.agent_seed {
            config.agent_seed = value;
        }
        if let Some(value) = self.threads {
            config.threads = value;
        }
        if let Some(value) = self.record_dir {
            config.record_dir = value;
        }
        if let Some(value) = self.record_games {
            config.record_games =
                if value == "all" {
                    Recording::All(value)
                } else {
                    Recording::Count(value.parse::<usize>().map_err(|_| {
                        "record-games must be 0, a positive count, or all".to_string()
                    })?)
                };
        }
        if let Some(seeds) = self.agent_seeds {
            if seeds.len() != config.agents.len() {
                return Err(format!(
                    "expected {} agent seeds, got {}",
                    config.agents.len(),
                    seeds.len()
                ));
            }
            config.agents = std::mem::take(&mut config.agents)
                .into_iter()
                .zip(seeds)
                .map(|(agent, seed)| {
                    let mut agent = match agent {
                        AgentSpec::Strategy(strategy) => AgentConfig {
                            name: strategy.clone(),
                            strategy,
                            seed: None,
                        },
                        AgentSpec::Detailed(agent) => agent,
                    };
                    agent.seed = Some(seed);
                    AgentSpec::Detailed(agent)
                })
                .collect();
        }
        config.validate()?;
        config.resolve_seeds()
    }
}

struct Progress<W: Write> {
    writer: W,
    started: bool,
}

impl<W: Write> Progress<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            started: false,
        }
    }

    fn update(&mut self, completed: usize, total: usize) -> io::Result<()> {
        self.started = true;
        write!(self.writer, "\r{completed} / {total} games played")?;
        self.writer.flush()
    }

    fn finish(&mut self) -> io::Result<()> {
        if self.started {
            self.started = false;
            writeln!(self.writer)?;
            self.writer.flush()?;
        }
        Ok(())
    }
}

/// Win % of each row entrant against each column entrant (draws count as not won).
fn write_matrix(text: &mut String, results: &Results) {
    use std::fmt::Write;
    let names: Vec<&str> = results.standings.iter().map(|s| s.name.as_str()).collect();
    let label_width = names
        .iter()
        .map(|n| n.chars().count())
        .max()
        .unwrap_or(0)
        .max(5);
    let cell_width = names
        .iter()
        .map(|n| n.chars().count())
        .max()
        .unwrap_or(0)
        .max(13);
    writeln!(
        text,
        "\nHead to head (row's win % against column; games in brackets)"
    )
    .unwrap();
    write!(text, "{:<label_width$}", "").unwrap();
    for name in &names {
        write!(text, "  {name:>cell_width$}").unwrap();
    }
    writeln!(text).unwrap();
    for (i, name) in names.iter().enumerate() {
        write!(text, "{name:<label_width$}").unwrap();
        for (j, record) in results.head_to_head[i].iter().enumerate() {
            let cell = if i == j || record.played() == 0 {
                "-".to_string()
            } else {
                format!(
                    "{:.1}% ({})",
                    100.0 * record.wins as f64 / record.played() as f64,
                    record.played()
                )
            };
            write!(text, "  {cell:>cell_width$}").unwrap();
        }
        writeln!(text).unwrap();
    }
}

fn render_results(results: &Results) -> String {
    use std::fmt::Write;
    let name_width = results
        .standings
        .iter()
        .map(|row| row.name.chars().count())
        .max()
        .unwrap_or(0)
        .max(5);
    let strategy_width = results
        .standings
        .iter()
        .map(|row| row.strategy.chars().count())
        .max()
        .unwrap_or(0)
        .max(8);
    let mut text = String::new();
    writeln!(
        text,
        "{:<name_width$}  {:<strategy_width$}  {:>9}  {:>8}  {:>8}  {:>8}  {:>7}",
        "Agent", "Strategy", "Victories", "Played", "Losses", "Draws", "Win %"
    )
    .unwrap();
    writeln!(text, "{}", "-".repeat(name_width + strategy_width + 52)).unwrap();
    for row in &results.standings {
        let percent = if row.played == 0 {
            0.0
        } else {
            100.0 * row.wins as f64 / row.played as f64
        };
        writeln!(
            text,
            "{:<name_width$}  {:<strategy_width$}  {:>9}  {:>8}  {:>8}  {:>8}  {:>6.1}%",
            row.name, row.strategy, row.wins, row.played, row.losses, row.draws, percent
        )
        .unwrap();
    }
    if results.standings.len() >= 2 && results.head_to_head.len() == results.standings.len() {
        write_matrix(&mut text, results);
    }
    writeln!(
        text,
        "\nGames played: {} / {} | Draws: {} | Elapsed: {:.2}s",
        results.games_played, results.total_games, results.draws, results.elapsed_seconds
    )
    .unwrap();
    if let Some(path) = &results.recording_dir {
        writeln!(text, "Recordings: {}", path.display()).unwrap();
    }
    text
}

pub fn execute(args: Args) -> Result<(), String> {
    let config = args.resolve()?;
    let seeds = config
        .agents
        .iter()
        .enumerate()
        .map(|(index, agent)| {
            let (name, seed) = match agent {
                AgentSpec::Strategy(name) => (
                    name.as_str(),
                    config.agent_seed.value() as i128 + index as i128,
                ),
                AgentSpec::Detailed(agent) => (
                    agent.name.as_str(),
                    agent
                        .seed
                        .map_or(config.agent_seed.value() as i128 + index as i128, |seed| {
                            i128::from(seed.value())
                        }),
                ),
            };
            format!("{name}={seed}")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let stderr = io::stderr();
    let mut writer = stderr.lock();
    writeln!(
        writer,
        "Deck seed: {} | Agent starting seeds: {seeds}",
        config.deck_seed
    )
    .map_err(|error| error.to_string())?;
    let mut progress = Progress::new(writer);
    let outcome = tournament::run(&config, |completed, total| {
        progress
            .update(completed, total)
            .map_err(|error| error.to_string())
    });
    let finished = progress.finish();
    let results = outcome?;
    finished.map_err(|error| error.to_string())?;
    io::stdout()
        .lock()
        .write_all(render_results(&results).as_bytes())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_reuses_one_line_and_finishes_once() {
        let mut output = Vec::new();
        {
            let mut progress = Progress::new(&mut output);
            progress.update(0, 100).unwrap();
            progress.update(42, 100).unwrap();
            progress.update(100, 100).unwrap();
            progress.finish().unwrap();
            progress.finish().unwrap();
        }
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "\r0 / 100 games played\r42 / 100 games played\r100 / 100 games played\n"
        );
    }

    #[test]
    fn cli_values_override_json_and_support_individual_seeds() {
        let path =
            std::env::temp_dir().join(format!("schotten-config-{}.json", uuid::Uuid::new_v4()));
        std::fs::write(&path, r#"{"agents":["greedy","random"],"games_per_pair":20,"deck_seed":7,"agent_seed":10,"record_games":"all"}"#).unwrap();
        let config = Args {
            config: Some(path.clone()),
            agents: Some(vec!["left=greedy".into(), "right=greedy".into()]),
            games_per_pair: Some(2),
            deck_seed: Some(11.into()),
            agent_seeds: Some(vec![12.into(), 13.into()]),
            record_games: Some("0".into()),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(config.games_per_pair, 2);
        assert_eq!(config.deck_seed, Seed::Fixed(11));
        assert_eq!(
            serde_json::to_value(&config.record_games).unwrap(),
            serde_json::json!(0)
        );
        assert_eq!(
            serde_json::to_value(&config.agents).unwrap(),
            serde_json::json!([
                {"name":"left","strategy":"greedy","seed":12},
                {"name":"right","strategy":"greedy","seed":13}
            ])
        );
    }

    #[test]
    fn invalid_override_and_bad_config_fail_before_execution() {
        assert!(
            Args {
                agents: Some(vec!["random".into(), "greedy".into()]),
                agent_seeds: Some(vec![1.into()]),
                ..Default::default()
            }
            .resolve()
            .unwrap_err()
            .contains("seed")
        );
        assert!(
            Args {
                record_games: Some("garbage".into()),
                ..Default::default()
            }
            .resolve()
            .is_err()
        );
        assert!(
            Args {
                games_per_pair: Some(0),
                ..Default::default()
            }
            .resolve()
            .is_err()
        );
        let path =
            std::env::temp_dir().join(format!("schotten-config-{}.json", uuid::Uuid::new_v4()));
        std::fs::write(&path, r#"{"games_per_pair":5,"typo":true}"#).unwrap();
        let result = Args {
            config: Some(path.clone()),
            ..Default::default()
        }
        .resolve();
        std::fs::remove_file(path).unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn standings_table_includes_victories_and_totals() {
        let results = crate::tournament::Results {
            standings: vec![crate::tournament::Standing {
                name: "a".into(),
                strategy: "greedy".into(),
                wins: 3,
                losses: 1,
                draws: 0,
                played: 4,
            }],
            head_to_head: vec![vec![Default::default()]],
            games_played: 4,
            total_games: 4,
            draws: 0,
            elapsed_seconds: 1.25,
            recording_dir: None,
        };
        let table = render_results(&results);
        assert!(table.contains("Victories") && table.contains("Played"));
        assert!(table.contains("75.0%") && table.contains("1.25s"));
        assert!(table.contains("4 / 4"));
    }

    #[test]
    fn config_rejects_positional_arrays_instead_of_running_defaults() {
        let path =
            std::env::temp_dir().join(format!("schotten-config-{}.json", uuid::Uuid::new_v4()));
        for text in ["[]", r#"{"agents":[["a","random",1],"greedy"]}"#] {
            std::fs::write(&path, text).unwrap();
            assert!(
                Args {
                    config: Some(path.clone()),
                    ..Default::default()
                }
                .resolve()
                .is_err()
            );
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn duplicate_config_fields_cannot_replace_recording_or_seed_settings() {
        let path =
            std::env::temp_dir().join(format!("schotten-config-{}.json", uuid::Uuid::new_v4()));
        for text in [
            r#"{"agents":["random","greedy"],"games_per_pair":1,"record_games":"all","record_games":0}"#,
            r#"{"deck_seed":1,"deck_seed":2}"#,
            r#"{"agents":[{"name":"a","strategy":"random","seed":1,"seed":2},"greedy"]}"#,
        ] {
            std::fs::write(&path, text).unwrap();
            assert!(
                Args {
                    config: Some(path.clone()),
                    ..Default::default()
                }
                .resolve()
                .is_err(),
                "accepted duplicate fields: {text}"
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}
