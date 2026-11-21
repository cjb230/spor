use std::{collections::HashMap, convert::Infallible, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, Query, State, rejection::QueryRejection},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use chrono::Utc;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
use subtle::ConstantTimeEq;
use tokio::{
    sync::{Mutex, RwLock, watch},
    time::Instant,
};

use crate::engine::{Game, GameError, Options, TurnRequest};

type Registry = Arc<ServerState>;

struct ServerState {
    games: RwLock<HashMap<String, Arc<HostedGame>>>,
    shutdown: watch::Sender<bool>,
}

struct HostedGame {
    game: Mutex<Game>,
    changes: watch::Sender<u64>,
}

impl HostedGame {
    fn tick(&self, game: &mut Game) {
        if game.check_timeout(now()) {
            self.changes.send_replace(game.version);
        }
    }
}

impl IntoResponse for GameError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (
            status,
            Json(json!({"error": self.code, "detail": self.detail})),
        )
            .into_response()
    }
}

fn now() -> i64 {
    Utc::now().timestamp_millis()
}

fn invalid(detail: impl ToString) -> GameError {
    GameError::new("INVALID_REQUEST", detail.to_string())
}

fn decode<T: DeserializeOwned>(body: &[u8]) -> Result<T, GameError> {
    serde_json::from_slice(body).map_err(invalid)
}

fn coerce_integer(value: &mut Value) {
    let text = match value {
        Value::Bool(boolean) => {
            *value = json!(u8::from(*boolean));
            return;
        }
        Value::String(text) => text.trim().to_owned(),
        Value::Number(number) if number.is_f64() => {
            let Some(number) = number
                .as_f64()
                .filter(|number| number.is_finite() && number.fract() == 0.0)
            else {
                return;
            };
            format!("{number:.0}")
        }
        _ => return,
    };
    let magnitude = text.strip_prefix(['+', '-']).unwrap_or(&text);
    let digits = if let Some((integer, fraction)) = magnitude.split_once('.') {
        if fraction.is_empty() || !fraction.bytes().all(|digit| digit == b'0') {
            return;
        }
        integer
    } else {
        magnitude
    };
    if digits.is_empty()
        || !digits.bytes().enumerate().all(|(index, digit)| {
            digit.is_ascii_digit()
                || (digit == b'_'
                    && index > 0
                    && index + 1 < digits.len()
                    && digits.as_bytes()[index - 1].is_ascii_digit()
                    && digits.as_bytes()[index + 1].is_ascii_digit())
        })
    {
        return;
    }
    let digits = digits.replace('_', "");
    let magnitude = digits.trim_start_matches('0');
    let sign = if text.starts_with('-') && !magnitude.is_empty() {
        "-"
    } else {
        ""
    };
    let magnitude = if magnitude.is_empty() { "0" } else { magnitude };
    if let Ok(integer) = serde_json::from_str(&format!("{sign}{magnitude}")) {
        *value = integer;
    }
}

fn require_object(value: &Value) -> Result<(), GameError> {
    if value.is_object() {
        Ok(())
    } else {
        Err(invalid("request fields must be JSON objects"))
    }
}

fn normalize_options(value: &mut Value) {
    for name in ["seed", "turn_timeout_seconds"] {
        if let Some(field) = value.get_mut(name) {
            coerce_integer(field);
        }
    }
    if let Some(field) = value.get_mut("auto_claim") {
        let boolean = match &*field {
            Value::String(text) => match text.to_ascii_lowercase().as_str() {
                "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
                "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
                _ => None,
            },
            Value::Number(number) if number.as_f64() == Some(0.0) => Some(false),
            Value::Number(number) if number.as_f64() == Some(1.0) => Some(true),
            _ => None,
        };
        if let Some(boolean) = boolean {
            *field = json!(boolean);
        }
    }
}

fn normalize_turn(value: &mut Value) {
    if let Some(field) = value.get_mut("version") {
        coerce_integer(field);
    }
    if let Some(fields) = value.get_mut("claims").and_then(Value::as_array_mut) {
        for field in fields {
            coerce_integer(field);
        }
    }
    if let Some(play) = value.get_mut("play") {
        if let Some(stone) = play.get_mut("stone") {
            coerce_integer(stone);
        }
        if let Some(number) = play.get_mut("card").and_then(|card| card.get_mut("value")) {
            coerce_integer(number);
        }
    }
}

fn wide_seed(value: &mut Value) -> Result<Option<Vec<u32>>, GameError> {
    let Some(Value::Number(number)) = value.get("seed") else {
        return Ok(None);
    };
    if number.as_i64().is_some() {
        return Ok(None);
    }
    let text = number.to_string();
    let digits = text.strip_prefix('-').unwrap_or(&text);
    if digits.is_empty() || !digits.bytes().all(|digit| digit.is_ascii_digit()) {
        return Ok(None);
    }
    if digits.len() > 4300 {
        return Err(invalid("integer seed is too long"));
    }
    let mut words = vec![0u32];
    for digit in digits.bytes() {
        let mut carry = u64::from(digit - b'0');
        for word in &mut words {
            let next = u64::from(*word) * 10 + carry;
            *word = next as u32;
            carry = next >> 32;
        }
        if carry != 0 {
            words.push(carry as u32);
        }
    }
    value.as_object_mut().unwrap().remove("seed");
    Ok(Some(words))
}

async fn find(registry: &Registry, id: &str) -> Result<Arc<HostedGame>, GameError> {
    registry
        .games
        .read()
        .await
        .get(id)
        .cloned()
        .ok_or_else(|| GameError::new("NOT_FOUND", format!("no such game: {id}")))
}

fn seat_of(game: &Game, headers: &HeaderMap) -> Result<usize, GameError> {
    let authorization = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    if let Some(value) = authorization
        && value
            .get(..7)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("bearer "))
    {
        let token = value[7..].trim();
        for (seat, stored) in game.tokens.iter().enumerate() {
            if stored
                .as_ref()
                .is_some_and(|stored| bool::from(stored.as_bytes().ct_eq(token.as_bytes())))
            {
                return Ok(seat);
            }
        }
        return Err(GameError::new(
            "UNAUTHORIZED",
            "invalid token for this game",
        ));
    }
    Err(GameError::new("UNAUTHORIZED", "missing bearer token"))
}

fn clock_wait(game: &Game, maximum: Duration) -> Duration {
    game.deadline.map_or(maximum, |deadline| {
        maximum.min(Duration::from_millis(
            deadline.saturating_sub(now()).max(0) as u64
        ))
    })
}

pub fn router() -> Router {
    let (shutdown, _) = watch::channel(false);
    router_with_shutdown(shutdown)
}

pub fn router_with_shutdown(shutdown: watch::Sender<bool>) -> Router {
    router_with_options(shutdown, "tournament-recordings".into())
}

pub fn router_with_options(
    shutdown: watch::Sender<bool>,
    recordings: std::path::PathBuf,
) -> Router {
    crate::rules::warmup();
    let registry: Registry = Arc::new(ServerState {
        games: RwLock::new(HashMap::new()),
        shutdown,
    });
    Router::new()
        .route("/games", post(create_game))
        .route("/games/{id}/join", post(join_game))
        .route("/games/{id}", get(read_game))
        .route("/games/{id}/legal-moves", get(legal_moves))
        .route("/games/{id}/log", get(read_log))
        .route("/games/{id}/events", get(events))
        .route("/games/{id}/turn", post(submit_turn))
        .route(
            "/openapi.yaml",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "application/yaml")],
                    include_str!("../../openapi.yaml"),
                )
            }),
        )
        .with_state(registry)
        .merge(crate::web::router(recordings))
}

async fn create_game(State(registry): State<Registry>, body: Bytes) -> Result<Response, GameError> {
    let mut value: Value = if body.iter().all(u8::is_ascii_whitespace) {
        Value::Null
    } else {
        decode(&body)?
    };
    if value.is_null() {
        value = json!({});
    }
    require_object(&value)?;
    normalize_options(&mut value);
    let seed_words = wide_seed(&mut value)?;
    let options: Options = serde_json::from_value(value).map_err(invalid)?;
    options.validate()?;
    let id = format!("g_{}", uuid::Uuid::new_v4().simple());
    let mut game = Game::new(id.clone(), options, now(), true);
    if let Some(words) = seed_words {
        game.reseed_words(&words)?;
    }
    let grant = json!({"game_id":id,"seat":0,"token":game.tokens[0]});
    let (changes, _) = watch::channel(game.version);
    registry.games.write().await.insert(
        id,
        Arc::new(HostedGame {
            game: Mutex::new(game),
            changes,
        }),
    );
    Ok((StatusCode::CREATED, Json(grant)).into_response())
}

async fn join_game(
    State(registry): State<Registry>,
    Path(id): Path<String>,
) -> Result<Response, GameError> {
    let hosted = find(&registry, &id).await?;
    let mut game = hosted.game.lock().await;
    let token = game.join(now())?;
    hosted.changes.send_replace(game.version);
    Ok((
        StatusCode::CREATED,
        Json(json!({"game_id":id,"seat":1,"token":token})),
    )
        .into_response())
}

#[derive(Deserialize)]
struct Poll {
    since: Option<String>,
    timeout: Option<String>,
}

fn query_integer(value: String) -> Result<u64, GameError> {
    let mut value = Value::String(value);
    coerce_integer(&mut value);
    let Value::Number(number) = value else {
        return Err(invalid("query parameter must be an integer"));
    };
    if number.to_string().starts_with('-') {
        return Err(invalid("query parameter must not be negative"));
    }
    Ok(number.as_u64().unwrap_or(u64::MAX))
}

async fn read_game(
    State(registry): State<Registry>,
    Path(id): Path<String>,
    headers: HeaderMap,
    query: Result<Query<Poll>, QueryRejection>,
) -> Result<Response, GameError> {
    let Query(query) = query.map_err(invalid)?;
    let timeout = query.timeout.map(query_integer).transpose()?.unwrap_or(30);
    let since = query.since.map(query_integer).transpose()?;
    if !(1..=60).contains(&timeout) {
        return Err(invalid("timeout must be between 1 and 60 seconds"));
    }
    let hosted = find(&registry, &id).await?;
    let mut changes = hosted.changes.subscribe();
    let mut shutdown = registry.shutdown.subscribe();
    let end = Instant::now() + Duration::from_secs(timeout);
    loop {
        let delay = {
            let mut game = hosted.game.lock().await;
            let seat = seat_of(&game, &headers)?;
            hosted.tick(&mut game);
            if since != Some(game.version) || game.over() {
                return Ok(Json(game.view(seat)).into_response());
            }
            let remaining = end.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(StatusCode::NO_CONTENT.into_response());
            }
            clock_wait(&game, remaining)
        };
        tokio::select! {
            _ = changes.changed() => {},
            _ = tokio::time::sleep(delay) => {},
            _ = shutdown.changed() => return Ok(StatusCode::NO_CONTENT.into_response()),
        }
    }
}

async fn legal_moves(
    State(registry): State<Registry>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Value>, GameError> {
    let hosted = find(&registry, &id).await?;
    let mut game = hosted.game.lock().await;
    let seat = seat_of(&game, &headers)?;
    hosted.tick(&mut game);
    Ok(Json(game.legal_moves(seat)?))
}

async fn read_log(
    State(registry): State<Registry>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Value>, GameError> {
    let hosted = find(&registry, &id).await?;
    let mut game = hosted.game.lock().await;
    seat_of(&game, &headers)?;
    hosted.tick(&mut game);
    Ok(Json(game.log_view()))
}

async fn submit_turn(
    State(registry): State<Registry>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<Value>, GameError> {
    let mut value: Value = decode(&body)?;
    require_object(&value)?;
    if let Some(play) = value.get("play").filter(|play| !play.is_null()) {
        require_object(play)?;
        if let Some(card) = play.get("card") {
            require_object(card)?;
        }
    }
    normalize_turn(&mut value);
    let request: TurnRequest = serde_json::from_value(value).map_err(invalid)?;
    request.validate()?;
    let hosted = find(&registry, &id).await?;
    let mut game = hosted.game.lock().await;
    let seat = seat_of(&game, &headers)?;
    let before = game.version;
    let result = game.play_turn(seat, request, now());
    if game.version != before {
        hosted.changes.send_replace(game.version);
    }
    result?;
    Ok(Json(game.view(seat)))
}

async fn events(
    State(registry): State<Registry>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Response, GameError> {
    let hosted = find(&registry, &id).await?;
    {
        let game = hosted.game.lock().await;
        seat_of(&game, &headers)?;
    }
    let mut last = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0);
    let mut changes = hosted.changes.subscribe();
    let mut shutdown = registry.shutdown.subscribe();
    let stream = async_stream::stream! {
        loop {
            if *shutdown.borrow() { break; }
            let (fresh, over, delay) = {
                let mut game = hosted.game.lock().await;
                hosted.tick(&mut game);
                let fresh: Vec<Value> = game.events.iter().filter(|event| {
                    event["version"].as_i64().is_some_and(|version| version > last)
                }).cloned().collect();
                (fresh, game.over(), clock_wait(&game, Duration::from_secs(15)))
            };
            let empty = fresh.is_empty();
            for event in fresh {
                last = event["version"].as_i64().unwrap_or(last);
                yield Ok::<_, Infallible>(Event::default().id(last.to_string()).data(event.to_string()));
            }
            if over {
                break;
            }
            if empty {
                yield Ok(Event::default().comment("keepalive"));
            }
            tokio::select! {
                _ = changes.changed() => {},
                _ = tokio::time::sleep(delay) => {},
                _ = shutdown.changed() => break,
            }
        }
    };
    Ok(Sse::new(stream).into_response())
}
