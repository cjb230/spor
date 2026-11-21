//! Read-only tournament recording library and authoritative replay validation.
use crate::engine::{Game, Options, TurnRequest};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::get,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    io::Read,
    path::{Path as FsPath, PathBuf},
};

const MAX_FILE_BYTES: u64 = 1024 * 1024;
const MAX_TURNS: usize = 128;
const MAX_FILES: usize = 10_000;
type ApiError = (StatusCode, Json<Value>);

pub fn router(root: PathBuf) -> Router {
    Router::new()
        .route("/web/recordings", get(list))
        .route("/web/recordings/{run}/{file}", get(replay))
        .with_state(root)
}

fn error(status: StatusCode, message: &str) -> ApiError {
    (status, Json(json!({"error":message})))
}

fn component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 200
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
fn game_file(value: &str) -> bool {
    value
        .strip_suffix(".json")
        .is_some_and(|stem| stem.starts_with("game-") && component(stem))
}

fn safe_path(root: &FsPath, run: &str, file: &str) -> Result<PathBuf, ApiError> {
    if !component(run) || !game_file(file) {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Invalid recording identifier",
        ));
    }
    let root = root
        .canonicalize()
        .map_err(|_| error(StatusCode::NOT_FOUND, "Recording not found"))?;
    let path = root
        .join(run)
        .join(file)
        .canonicalize()
        .map_err(|_| error(StatusCode::NOT_FOUND, "Recording not found"))?;
    if !path.starts_with(&root) || !path.is_file() {
        return Err(error(StatusCode::NOT_FOUND, "Recording not found"));
    }
    Ok(path)
}

#[derive(Deserialize)]
struct Document {
    schema_version: u64,
    game_id: String,
    deck_seed: String,
    seats: [Seat; 2],
    winner_seat: Option<usize>,
    winner_entrant_index: Option<usize>,
    winner: Option<String>,
    reason: String,
    turns: usize,
    log: Value,
}
#[derive(Deserialize, serde::Serialize)]
struct Seat {
    seat: usize,
    entrant_index: usize,
    name: String,
    strategy: String,
    seed: String,
}

fn load(path: &FsPath, frames: bool) -> Result<Value, &'static str> {
    let file = fs::File::open(path).map_err(|_| "cannot read recording")?;
    if file.metadata().map_err(|_| "cannot read recording")?.len() > MAX_FILE_BYTES {
        return Err("recording exceeds size limit");
    }
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "cannot read recording")?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err("recording exceeds size limit");
    }
    let doc: Document =
        serde_json::from_slice(&bytes).map_err(|_| "invalid or incomplete recording JSON")?;
    if doc.schema_version != 1 {
        return Err("unsupported recording schema");
    }
    if doc.turns > MAX_TURNS || doc.game_id.is_empty() || doc.game_id.len() > 200 {
        return Err("invalid recording metadata");
    }
    for (i, seat) in doc.seats.iter().enumerate() {
        if seat.seat != i
            || seat.name.is_empty()
            || seat.name.len() > 200
            || seat.strategy.len() > 200
            || seat.seed.parse::<i128>().is_err()
        {
            return Err("invalid seat metadata");
        }
    }
    if doc.seats[0].entrant_index == doc.seats[1].entrant_index {
        return Err("invalid seat metadata");
    }
    if doc.winner_seat.is_some_and(|s| s >= 2) {
        return Err("invalid winner metadata");
    }
    if doc.winner.as_deref() != doc.winner_seat.map(|s| doc.seats[s].name.as_str())
        || doc.winner_entrant_index != doc.winner_seat.map(|s| doc.seats[s].entrant_index)
    {
        return Err("winner metadata disagrees with seats");
    }
    let seed = doc
        .deck_seed
        .parse::<i128>()
        .map_err(|_| "invalid deck seed")?;
    // Python integer seeding uses the absolute integer split into little-endian u32s.
    // This also preserves i128::MIN, whose absolute value cannot fit in i128.
    let mut n = seed.unsigned_abs();
    let mut words = Vec::new();
    loop {
        words.push(n as u32);
        n >>= 32;
        if n == 0 {
            break;
        }
    }
    let mut game = Game::new(
        doc.game_id.clone(),
        Options {
            seed: Some(0),
            ..Default::default()
        },
        0,
        true,
    );
    game.reseed_words(&words)
        .map_err(|_| "cannot reconstruct deal")?;
    game.join(0).map_err(|_| "cannot reconstruct game")?;
    let entries = doc.log["entries"].as_array().ok_or("missing turn log")?;
    if entries.len() != doc.turns {
        return Err("turn count disagrees with log");
    }
    let frame = |game: &Game, last: Value| {
        let mut view = game.view(0);
        view["opponent_hand"] = game.view(1)["hand"].clone();
        view["last_turn"] = last;
        view
    };
    let mut snapshots = Vec::new();
    if frames {
        snapshots.push(frame(&game, Value::Null));
    }
    for entry in entries {
        let seat = entry["seat"]
            .as_u64()
            .filter(|s| *s < 2)
            .ok_or("invalid turn seat")? as usize;
        let version = entry["version"].as_u64().ok_or("invalid turn version")?;
        if version != game.version + 1 {
            return Err("turn versions are not sequential");
        }
        let req: TurnRequest = serde_json::from_value(json!({
            "version":game.version,"play":entry["play"],"claims":entry["claims"]
        }))
        .map_err(|_| "invalid recorded turn")?;
        game.play_turn(seat, req, 0)
            .map_err(|_| "recorded turn is illegal")?;
        if frames {
            snapshots.push(frame(&game, entry.clone()));
        }
    }
    if !game.over() || game.outcome() != Some((doc.winner_seat, doc.reason.as_str())) {
        return Err("recorded outcome disagrees with replay");
    }
    // Exact comparison validates the complete original deal, draw order, accepted
    // explicit claims, turn versions, game identity and terminal status.
    if game.log_view() != doc.log {
        return Err("recorded deal or log disagrees with replay");
    }
    let metadata = json!({"game_id":doc.game_id,"seats":doc.seats,"winner":doc.winner,
                          "reason":doc.reason,"turns":doc.turns});
    if frames {
        Ok(json!({"metadata":metadata,"frames":snapshots}))
    } else {
        Ok(metadata)
    }
}

async fn replay(
    State(root): State<PathBuf>,
    Path((run, file)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    tokio::task::spawn_blocking(move || {
        let path = safe_path(&root, &run, &file)?;
        load(&path, true)
            .map(Json)
            .map_err(|why| error(StatusCode::UNPROCESSABLE_ENTITY, why))
    })
    .await
    .map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Recording worker unavailable",
        )
    })?
}

fn library(root: &FsPath) -> Value {
    let mut games = Vec::new();
    let mut warnings = Vec::new();
    let root = match root.canonicalize() {
        Ok(root) => root,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return json!({"games":[],"warnings":[]});
        }
        Err(_) => return json!({"games":[],"warnings":["Cannot read recordings directory"]}),
    };
    let runs = match fs::read_dir(&root) {
        Ok(runs) => runs,
        Err(_) => return json!({"games":[],"warnings":["Cannot read recordings directory"]}),
    };
    let mut count = 0;
    'runs: for run in runs {
        let Ok(run) = run else {
            warnings.push("Cannot read a recordings entry".to_string());
            continue;
        };
        let Some(run_name) = run
            .file_name()
            .to_str()
            .filter(|n| component(n))
            .map(str::to_owned)
        else {
            continue;
        };
        let Ok(run_path) = run.path().canonicalize() else {
            continue;
        };
        if !run_path.starts_with(&root) {
            warnings.push(format!("{run_name}: path escapes recordings directory"));
            continue;
        }
        if !run_path.is_dir() {
            continue;
        }
        let files = match fs::read_dir(run_path) {
            Ok(files) => files,
            Err(_) => {
                warnings.push(format!("{run_name}: cannot read run directory"));
                continue;
            }
        };
        for file in files {
            let Ok(file) = file else {
                warnings.push(format!("{run_name}: cannot read a run entry"));
                continue;
            };
            let Some(name) = file
                .file_name()
                .to_str()
                .filter(|n| game_file(n))
                .map(str::to_owned)
            else {
                continue;
            };
            count += 1;
            if count > MAX_FILES {
                warnings.push("Recording library limit reached".to_string());
                break 'runs;
            }
            let id = format!("{run_name}/{name}");
            let result = safe_path(&root, &run_name, &name)
                .map_err(|_| "unavailable or unsafe recording path")
                .and_then(|p| load(&p, false));
            match result {
                Ok(mut metadata) => {
                    metadata["id"] = json!(id);
                    games.push(metadata);
                }
                Err(why) => warnings.push(format!("{id}: {why}")),
            }
        }
    }
    games.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    warnings.sort();
    json!({"games":games,"warnings":warnings})
}
async fn list(State(root): State<PathBuf>) -> Result<Json<Value>, ApiError> {
    tokio::task::spawn_blocking(move || Json(library(&root)))
        .await
        .map_err(|_| {
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Recording worker unavailable",
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use std::fs;
    use tower::ServiceExt;

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("spor-recordings-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(path.join("run")).unwrap();
            Self(path)
        }
        fn write(&self, file: &str, value: &Value) {
            fs::write(
                self.0.join("run").join(file),
                serde_json::to_vec(value).unwrap(),
            )
            .unwrap();
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn recording(seed: i128) -> Value {
        let result = crate::bots::play_game(
            "recorded-test".into(),
            ["random", "random"],
            Some(seed),
            [Some(1), Some(2)],
            true,
        )
        .unwrap();
        json!({"schema_version":1,"game_id":"recorded-test","deck_seed":seed.to_string(),
            "seats":[{"seat":0,"entrant_index":0,"name":"Alice","strategy":"random","seed":"1"},
                     {"seat":1,"entrant_index":1,"name":"Bob","strategy":"random","seed":"2"}],
            "winner_seat":result.winner,"winner_entrant_index":result.winner,"winner":result.winner.map(|s| ["Alice","Bob"][s]),
            "reason":result.reason,"turns":result.turns,"log":result.log.unwrap()})
    }
    async fn request(root: &Directory, path: &str) -> (u16, Value) {
        let response = router(root.0.clone())
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status().as_u16();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }
    #[tokio::test]
    async fn real_recordings_replay_every_turn_and_preserve_wide_signed_seeds() {
        let root = Directory::new();
        for (i, seed) in [1, i64::MAX as i128 + 1, i128::MIN].into_iter().enumerate() {
            let value = recording(seed);
            let name = format!("game-{i}.json");
            root.write(&name, &value);
            let (status, replay) = request(&root, &format!("/web/recordings/run/{name}")).await;
            assert_eq!(status, 200);
            let frames = replay["frames"].as_array().unwrap();
            assert_eq!(frames.len(), value["turns"].as_u64().unwrap() as usize + 1);
            assert_eq!(frames[0]["version"], 1);
            assert_eq!(frames[0]["opponent_hand"].as_array().unwrap().len(), 6);
            assert!(frames[0]["last_turn"].is_null());
            assert_eq!(
                frames.last().unwrap()["last_turn"],
                value["log"]["entries"]
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap()
                    .clone()
            );
            assert_eq!(frames.last().unwrap()["status"], "finished");
            assert_eq!(frames.last().unwrap()["win_reason"], value["reason"]);
        }
        let (status, library) = request(&root, "/web/recordings").await;
        assert_eq!(status, 200);
        assert_eq!(library["games"].as_array().unwrap().len(), 3);
        assert_eq!(library["games"][0]["id"], "run/game-0.json");
        assert_eq!(library["warnings"], json!([]));
    }
    #[tokio::test]
    async fn reads_actual_tournament_writer_output() {
        let root = Directory::new();
        let config = crate::tournament::Config {
            agents: vec![
                crate::tournament::AgentSpec::Strategy("random".into()),
                crate::tournament::AgentSpec::Detailed(crate::tournament::AgentConfig {
                    name: "Other".into(),
                    strategy: "random".into(),
                    seed: None,
                }),
            ],
            games_per_pair: 2,
            deck_seed: crate::tournament::Seed::Fixed(i64::MAX),
            record_games: crate::tournament::Recording::All("all".into()),
            record_dir: root.0.clone(),
            threads: 1,
            ..Default::default()
        };
        let result = crate::tournament::run(&config, |_, _| Ok(())).unwrap();
        assert_eq!(result.games_played, 2);
        let (status, library) = request(&root, "/web/recordings").await;
        assert_eq!(status, 200);
        assert_eq!(library["warnings"], json!([]));
        let games = library["games"].as_array().unwrap();
        assert_eq!(games.len(), 2);
        for game in games {
            let (status, replay) = request(
                &root,
                &format!("/web/recordings/{}", game["id"].as_str().unwrap()),
            )
            .await;
            assert_eq!(status, 200);
            assert_eq!(replay["metadata"]["winner"], game["winner"]);
            let frames = replay["frames"].as_array().unwrap();
            assert_eq!(frames.len() as u64, game["turns"].as_u64().unwrap() + 1);
        }
    }
    #[tokio::test]
    async fn oversized_files_are_bounded_and_reported() {
        let root = Directory::new();
        let file = fs::File::create(root.0.join("run/game-large.json")).unwrap();
        file.set_len(MAX_FILE_BYTES + 1).unwrap();
        let (status, body) = request(&root, "/web/recordings/run/game-large.json").await;
        assert_eq!(status, 422);
        assert!(body["error"].as_str().unwrap().contains("size limit"));
        let (_, body) = request(&root, "/web/recordings").await;
        assert_eq!(body["games"], json!([]));
        assert_eq!(body["warnings"].as_array().unwrap().len(), 1);
    }
    #[tokio::test]
    async fn corrupt_recordings_are_skipped_with_diagnostics() {
        let root = Directory::new();
        let valid = recording(7);
        root.write("game-good.json", &valid);
        for (index, change) in ["hands", "draws", "outcome", "log", "turns", "schema"]
            .into_iter()
            .enumerate()
        {
            let mut bad = valid.clone();
            match change {
                "hands" => bad["log"]["revealed"]["initial_hands"][0][0]["value"] = json!(0),
                "draws" => bad["log"]["revealed"]["draw_order"][0]["value"] = json!(0),
                "outcome" => bad["winner"] = json!("wrong"),
                "log" => bad["log"]["entries"][0]["seat"] = json!(9),
                "turns" => bad["turns"] = json!(0),
                _ => bad["schema_version"] = json!(2),
            }
            root.write(&format!("game-bad-{index}.json"), &bad);
        }
        fs::write(root.0.join("run/game-partial.json"), b"{").unwrap();
        let (status, library) = request(&root, "/web/recordings").await;
        assert_eq!(status, 200);
        assert_eq!(library["games"].as_array().unwrap().len(), 1);
        assert_eq!(library["warnings"].as_array().unwrap().len(), 7);
        assert_eq!(
            request(&root, "/web/recordings/run/game-bad-0.json")
                .await
                .0,
            422
        );
    }
    #[tokio::test]
    async fn missing_root_is_empty_and_unsafe_ids_are_rejected() {
        let root = Directory::new();
        let absent = Directory(root.0.join("absent"));
        assert_eq!(
            request(&absent, "/web/recordings").await,
            (200, json!({"games":[],"warnings":[]}))
        );
        for path in [
            "/web/recordings/%2e%2e/Cargo.toml",
            "/web/recordings/run/%2e%2e",
            "/web/recordings/run/a%2fb.json",
            "/web/recordings/run/manifest.json",
        ] {
            assert_ne!(request(&root, path).await.0, 200);
        }
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_escapes_are_not_listed_or_replayed() {
        use std::os::unix::fs::symlink;
        let root = Directory::new();
        let outside = Directory::new();
        outside.write("game-outside.json", &recording(9));
        symlink(outside.0.join("run"), root.0.join("escape")).unwrap();
        symlink(
            outside.0.join("run/game-outside.json"),
            root.0.join("run/game-link.json"),
        )
        .unwrap();
        assert_ne!(
            request(&root, "/web/recordings/escape/game-outside.json")
                .await
                .0,
            200
        );
        assert_ne!(
            request(&root, "/web/recordings/run/game-link.json").await.0,
            200
        );
        let (_, library) = request(&root, "/web/recordings").await;
        assert_eq!(library["games"], json!([]));
    }
}
