use std::process::Command;

#[test]
fn command_line_tournament_has_single_progress_line_and_a_final_table() {
    let output = Command::new(env!("CARGO_BIN_EXE_schotten-native"))
        .args([
            "tournament",
            "--agents",
            "random",
            "greedy",
            "builder",
            "--games-per-pair",
            "2",
            "--deck-seed",
            "7",
            "--agent-seeds",
            "10",
            "20",
            "30",
            "--threads",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let progress = String::from_utf8(output.stderr).unwrap();
    assert!(progress.contains("\r0 / 6 games played"));
    assert!(progress.ends_with("6 / 6 games played\n"));
    assert_eq!(
        progress
            .lines()
            .filter(|line| line.contains("games played"))
            .count(),
        1
    );
    let table = String::from_utf8(output.stdout).unwrap();
    assert!(table.contains("Victories"));
    for strategy in ["random", "greedy", "builder"] {
        assert!(table.contains(strategy));
    }
    assert!(!table.contains("Game 1/"));
}

#[test]
fn json_file_starts_a_tournament_and_cli_overrides_game_count() {
    let path =
        std::env::temp_dir().join(format!("schotten-tournament-{}.json", uuid::Uuid::new_v4()));
    std::fs::write(
        &path,
        r#"{"agents":["greedy","random"],"games_per_pair":100,"deck_seed":3,"agent_seed":4}"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_schotten-native"))
        .args([
            "tournament",
            "--config",
            path.to_str().unwrap(),
            "--games-per-pair",
            "2",
        ])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("2 / 2 games played")
    );
}

#[test]
fn named_entrants_and_negative_individual_seeds_are_accepted() {
    let output = Command::new(env!("CARGO_BIN_EXE_schotten-native"))
        .args([
            "tournament",
            "--agents",
            "left=greedy",
            "right=greedy",
            "--games-per-pair",
            "2",
            "--deck-seed",
            "-7",
            "--agent-seeds",
            "-10",
            "-20",
            "--record-games",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let table = String::from_utf8(output.stdout).unwrap();
    assert!(table.contains("left") && table.contains("right"));
    let progress = String::from_utf8(output.stderr).unwrap();
    assert!(progress.contains("Deck seed: -7") && progress.contains("left=-10, right=-20"));
}

#[test]
fn random_cli_seeds_are_recorded_and_resolved_config_reproduces_standings() {
    let root = std::env::temp_dir().join(format!("schotten-random-seeds-{}", uuid::Uuid::new_v4()));
    let output = Command::new(env!("CARGO_BIN_EXE_schotten-native"))
        .args([
            "tournament",
            "--agents",
            "random",
            "greedy",
            "--games-per-pair",
            "4",
            "--deck-seed",
            "random",
            "--agent-seeds",
            "random",
            "random",
            "--record-games",
            "all",
            "--record-dir",
            root.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let run = std::fs::read_dir(&root)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let manifest: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(run.join("manifest.json")).unwrap()).unwrap();
    let mut config = manifest["config"].clone();
    assert!(config["deck_seed"].is_i64());
    assert!(
        config["agents"]
            .as_array()
            .unwrap()
            .iter()
            .all(|agent| agent["seed"].is_i64())
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains(&format!("Deck seed: {}", config["deck_seed"])));
    for agent in config["agents"].as_array().unwrap() {
        assert!(stderr.contains(&format!(
            "{}={}",
            agent["name"].as_str().unwrap(),
            agent["seed"]
        )));
    }
    config["record_games"] = serde_json::json!(0);
    let path = root.join("resolved.json");
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let replay = Command::new(env!("CARGO_BIN_EXE_schotten-native"))
        .args(["tournament", "--config", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        replay.status.success(),
        "{}",
        String::from_utf8_lossy(&replay.stderr)
    );
    let first = String::from_utf8(output.stdout).unwrap();
    let second = String::from_utf8(replay.stdout).unwrap();
    assert_eq!(
        first.split("\nGames played:").next(),
        second.split("\nGames played:").next()
    );
    std::fs::remove_dir_all(root).unwrap();
}
