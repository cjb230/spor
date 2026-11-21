//! Browser-only sessions. The shared Python-compatible REST API stays unchanged.
use std::{collections::HashMap, path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use subtle::ConstantTimeEq;
use tokio::sync::{Mutex, RwLock, Semaphore};

use crate::{
    bots::Bot,
    engine::{Card, Game, GameError, Options, TurnRequest},
};

type Registry = Arc<WebState>;
struct WebState {
    sessions: RwLock<HashMap<String, Arc<Mutex<Session>>>>,
    workers: Arc<Semaphore>,
    recordings: PathBuf,
}
struct Session {
    game: Game,
    bot: Option<Bot>,
    thinking: bool,
    error: Option<String>,
    deck_seed: i64,
    bot_seed: i64,
    opponent: Opponent,
    recordings: PathBuf,
    recording_saved: bool,
    recording_error: Option<String>,
    last_drawn: Option<u8>,
}

impl Session {
    fn authorize(&self, headers: &HeaderMap) -> Result<(), GameError> {
        let supplied = headers
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| {
                h.get(..7)
                    .filter(|p| p.eq_ignore_ascii_case("bearer "))
                    .map(|_| h[7..].trim())
            })
            .unwrap_or("");
        let expected = self.game.tokens[0].as_deref().unwrap_or("");
        if supplied.is_empty() || !bool::from(supplied.as_bytes().ct_eq(expected.as_bytes())) {
            return Err(GameError::new("UNAUTHORIZED", "invalid seat token"));
        }
        Ok(())
    }
    fn view(&self) -> Value {
        let mut view = self.game.view(0);
        view["bot_thinking"] = json!(self.thinking);
        view["bot_error"] = json!(self.error);
        view["opponent"] = json!(self.opponent);
        let opponent_play = self
            .game
            .events
            .iter()
            .rev()
            .find(|event| event["type"] == "card_played" && event["seat"] == 1)
            .map(|event| json!({"card":event["card"],"stone":event["stone"]}));
        let hand = self.game.player_view(0).hand;
        let drawn = self
            .last_drawn
            .filter(|card| hand & (1u64 << card) != 0)
            .map(Card::from_id);
        view["recent_cards"] = json!({"opponentPlay":opponent_play,"drawn":drawn});
        view["recording_saved"] = json!(self.recording_saved);
        view["recording_error"] = json!(self.recording_error);
        view["recording_id"] = if self.recording_saved {
            json!(format!("browser-games/game-{}.json", self.game.id))
        } else {
            Value::Null
        };
        view["legal_moves"] = if self.game.turn == Some(0) && !self.game.over() {
            self.game.legal_moves(0).unwrap_or(Value::Null)
        } else {
            Value::Null
        };
        view
    }
}

pub fn router(recordings: PathBuf) -> Router {
    let state = Arc::new(WebState {
        sessions: RwLock::new(HashMap::new()),
        recordings: recordings.clone(),
        workers: Arc::new(Semaphore::new(
            std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(4),
        )),
    });
    Router::new()
        .route("/web/games", post(create))
        .route("/web/games/{id}", get(read))
        .route("/web/games/{id}/turn", post(turn))
        .with_state(state)
        .merge(crate::recordings::router(recordings))
        .route(
            "/",
            get(|| async { Html(include_str!("../web/index.html")) }),
        )
        .route(
            "/web/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("../web/app.js"),
                )
            }),
        )
        .route(
            "/web/style.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    include_str!("../web/style.css"),
                )
            }),
        )
        .layer(axum::middleware::map_response(
            |mut response: axum::response::Response| async move {
                response.headers_mut().insert(
                    header::CACHE_CONTROL,
                    axum::http::HeaderValue::from_static("no-store"),
                );
                response
            },
        ))
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Create {
    opponent: String,
    samples: Option<usize>,
    seed: Option<i64>,
    bot_seed: Option<i64>,
}
impl Default for Create {
    fn default() -> Self {
        Self {
            opponent: "mcb".into(),
            samples: None,
            seed: None,
            bot_seed: None,
        }
    }
}

#[derive(serde::Serialize)]
struct Opponent {
    name: &'static str,
    strategy: String,
}

impl Opponent {
    fn resolve(request: &Create) -> Result<Self, GameError> {
        use crate::bots::{DEFAULT_MCB_SAMPLES, DEFAULT_MCG_SAMPLES, DEFAULT_MCP_SAMPLES};
        let (name, default_samples) = match request.opponent.as_str() {
            "random" => ("Random", None),
            "greedy" => ("Greedy", None),
            "builder" => ("Builder", None),
            "tactician" => ("Tactician", None),
            "planner" => ("Planner", None),
            "mcg" => ("MCG", Some(DEFAULT_MCG_SAMPLES)),
            "mcb" => ("MCB", Some(DEFAULT_MCB_SAMPLES)),
            "mcp" => ("MCP", Some(DEFAULT_MCP_SAMPLES)),
            _ => {
                return Err(GameError::new(
                    "INVALID_REQUEST",
                    "unknown browser opponent",
                ));
            }
        };
        let strategy = if let Some(default) = default_samples {
            let samples = request.samples.unwrap_or(default);
            format!("{}:{samples}", request.opponent)
        } else {
            if request.samples.is_some() {
                return Err(GameError::new(
                    "INVALID_REQUEST",
                    "samples apply only to Monte Carlo opponents",
                ));
            }
            request.opponent.clone()
        };
        Ok(Self { name, strategy })
    }
}

async fn create(
    State(state): State<Registry>,
    body: Bytes,
) -> Result<impl IntoResponse, GameError> {
    let request: Create = if body.is_empty() {
        Create::default()
    } else {
        decode_object(&body)?
    };
    let opponent = Opponent::resolve(&request)?;
    let id = format!("web_{}", uuid::Uuid::new_v4().simple());
    let deck_seed = resolve_seed(request.seed)?;
    let bot_seed = resolve_seed(request.bot_seed)?;
    let mut game = Game::new(
        id.clone(),
        Options {
            seed: Some(deck_seed),
            ..Default::default()
        },
        0,
        true,
    );
    game.join(0)?;
    let grant = json!({"game_id":id,"seat":0,"token":game.tokens[0]});
    let bot = Bot::new(&opponent.strategy, Some(bot_seed))
        .map_err(|e| GameError::new("INVALID_REQUEST", e))?;
    state.sessions.write().await.insert(
        id,
        Arc::new(Mutex::new(Session {
            game,
            bot: Some(bot),
            thinking: false,
            error: None,
            deck_seed,
            bot_seed,
            opponent,
            recordings: state.recordings.clone(),
            recording_saved: false,
            recording_error: None,
            last_drawn: None,
        })),
    );
    Ok((StatusCode::CREATED, Json(grant)))
}

fn resolve_seed(seed: Option<i64>) -> Result<i64, GameError> {
    if let Some(seed) = seed {
        return Ok(seed);
    }
    let mut bytes = [0; 8];
    getrandom::fill(&mut bytes)
        .map_err(|_| GameError::new("INVALID_REQUEST", "cannot obtain a random game seed"))?;
    Ok(i64::from_le_bytes(bytes))
}

async fn save_finished(session: &mut Session) {
    if !session.game.over() || session.recording_saved {
        return;
    }
    let (winner, reason) = session
        .game
        .outcome()
        .expect("finished game has an outcome");
    let names = ["You", session.opponent.name];
    let log = session.game.log_view();
    let record = json!({
        "schema_version": 1,
        "game_id": session.game.id,
        "deck_seed": session.deck_seed.to_string(),
        "seats": [
            {"seat":0, "entrant_index":0, "name":"You", "strategy":"human", "seed":"0"},
            {"seat":1, "entrant_index":1, "name":session.opponent.name, "strategy":session.opponent.strategy, "seed":session.bot_seed.to_string()}
        ],
        "winner_seat": winner,
        "winner_entrant_index": winner,
        "winner": winner.map(|seat| names[seat]),
        "reason": reason,
        "turns": log["entries"].as_array().expect("game log entries").len(),
        "log": log
    });
    let root = session.recordings.clone();
    let name = format!("game-{}.json", session.game.id);
    let result = tokio::task::spawn_blocking(move || write_recording(&root, &name, &record)).await;
    match result {
        Ok(Ok(())) => {
            session.recording_saved = true;
            session.recording_error = None;
        }
        _ => {
            session.recording_error = Some(
                "Could not save the replay. Check the recordings directory and reconnect to retry."
                    .into(),
            )
        }
    }
}

fn write_recording(root: &std::path::Path, name: &str, record: &Value) -> std::io::Result<()> {
    use std::{fs, io::Write};
    fs::create_dir_all(root)?;
    let root = root.canonicalize()?;
    let directory = root.join("browser-games");
    fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    if !directory.starts_with(&root) {
        return Err(std::io::Error::other(
            "browser recordings directory escapes its root",
        ));
    }
    let bytes = serde_json::to_vec_pretty(record)?;
    let target = directory.join(name);
    let temporary = directory.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        // Linking publishes a complete file atomically and cannot overwrite an
        // existing recording. Retries accept only the exact same saved content.
        match fs::hard_link(&temporary, &target) {
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if fs::symlink_metadata(&target)?.file_type().is_symlink()
                    || fs::read(&target)? != bytes
                {
                    return Err(std::io::Error::other(
                        "recording already exists with different content",
                    ));
                }
                Ok(())
            }
            result => result,
        }
    })();
    let _ = fs::remove_file(&temporary);
    result
}

async fn find(state: &WebState, id: &str) -> Result<Arc<Mutex<Session>>, GameError> {
    state
        .sessions
        .read()
        .await
        .get(id)
        .cloned()
        .ok_or_else(|| GameError::new("NOT_FOUND", "no such browser game"))
}

fn decode_object<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, GameError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| GameError::new("INVALID_REQUEST", e.to_string()))?;
    if !value.is_object() {
        return Err(GameError::new(
            "INVALID_REQUEST",
            "request must be a JSON object",
        ));
    }
    serde_json::from_value(value).map_err(|e| GameError::new("INVALID_REQUEST", e.to_string()))
}

async fn read(
    State(state): State<Registry>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Value>, GameError> {
    let session = find(&state, &id).await?;
    let mut session = session.lock().await;
    session.authorize(&headers)?;
    save_finished(&mut session).await;
    Ok(Json(session.view()))
}

async fn turn(
    State(state): State<Registry>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<Value>, GameError> {
    let mut request: TurnRequest = decode_object(&body)?;
    request.validate()?;
    let hosted = find(&state, &id).await?;
    let mut session = hosted.lock().await;
    session.authorize(&headers)?;
    // Invalid placements must be rejected by the engine before claim helpers
    // touch its fixed-size formation storage. Retries still reach engine UUID dedup.
    let valid_play = request.play.as_ref().is_none_or(|play| {
        let view = session.game.player_view(0);
        view.hand & (1u64 << play.card.id()) != 0
            && view.stones[play.stone].claimed_by.is_none()
            && view.stones[play.stone].yours.len() < 3
    });
    if valid_play && session.game.turn == Some(0) && request.version == session.game.version as i64
    {
        request.claims = Bot::provable_stones(&session.game.player_view(0), request.play.as_ref());
    }
    let before_hand = session.game.player_view(0).hand;
    session.game.play_turn(0, request, 0)?;
    let drawn = session.game.player_view(0).hand & !before_hand;
    if drawn != 0 {
        session.last_drawn = Some(drawn.trailing_zeros() as u8);
    }
    save_finished(&mut session).await;
    if session.game.turn == Some(1)
        && !session.game.over()
        && !session.thinking
        && session.error.is_none()
    {
        session.thinking = true;
        let view = session.game.player_view(1);
        let bot = session.bot.take().expect("idle session owns its bot");
        let hosted = hosted.clone();
        let workers = state.workers.clone();
        // Own the work independently of the HTTP connection, including queued searches.
        tokio::spawn(async move {
            let permit = workers.acquire_owned().await;
            let result = match permit {
                Ok(permit) => tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    let mut bot = bot;
                    let choice = bot.choose(&view);
                    (bot, choice, view.version)
                })
                .await
                .map_err(|e| format!("Opponent search failed: {e}")),
                Err(_) => Err("Opponent worker is unavailable".into()),
            };
            let mut session = hosted.lock().await;
            match result {
                Ok((bot, (play, claims), version)) => {
                    session.bot = Some(bot);
                    if session.game.version == version && session.game.turn == Some(1) {
                        if let Err(error) = session.game.play_turn(
                            1,
                            TurnRequest {
                                version: version as i64,
                                turn_id: None,
                                play,
                                claims,
                            },
                            0,
                        ) {
                            session.error = Some(format!("Opponent could not move: {error}"));
                        }
                    } else {
                        session.error =
                            Some("The game changed during the opponent's search".into());
                    }
                }
                Err(error) => session.error = Some(error),
            }
            session.thinking = false;
            save_finished(&mut session).await;
        });
    }
    Ok(Json(session.view()))
}

#[cfg(test)]
mod recording_tests {
    use super::*;

    #[test]
    fn publishing_a_replay_is_idempotent_and_never_overwrites_other_content() {
        let root = std::env::temp_dir().join(format!("spor-web-atomic-{}", uuid::Uuid::new_v4()));
        let first = json!({"game_id":"first", "log":{"entries":[]}});
        write_recording(&root, "game-first.json", &first).unwrap();
        write_recording(&root, "game-first.json", &first).unwrap();
        assert!(
            write_recording(&root, "game-first.json", &json!({"game_id":"different"})).is_err()
        );
        let path = root.join("browser-games/game-first.json");
        assert_eq!(
            serde_json::from_slice::<Value>(&std::fs::read(path).unwrap()).unwrap(),
            first
        );
        assert_eq!(
            std::fs::read_dir(root.join("browser-games"))
                .unwrap()
                .count(),
            1
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn publishing_cannot_follow_a_browser_directory_outside_the_recordings_root() {
        let root = std::env::temp_dir().join(format!("spor-web-path-{}", uuid::Uuid::new_v4()));
        let outside =
            std::env::temp_dir().join(format!("spor-web-outside-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("browser-games")).unwrap();
        assert!(write_recording(&root, "game-first.json", &json!({})).is_err());
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_dir_all(outside).unwrap();
    }
}
