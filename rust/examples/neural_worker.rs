use rayon::prelude::*;
use schotten_native::bots::Bot;
use schotten_native::engine::{Game, Options, Play, TurnRequest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "lowercase", deny_unknown_fields)]
enum Command {
    New { games: Vec<GameSpec> },
    Turn { actions: Vec<Action> },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GameSpec {
    seed: i64,
    bot_seed: i64,
    mlp_seat: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    id: usize,
    version: u64,
    play: Option<Play>,
}

struct Match {
    game: Game,
    bot: Bot,
    mlp_seat: usize,
}

impl Match {
    fn bot_turn(&mut self) -> Result<(), String> {
        if self.game.over() {
            return Ok(());
        }
        let seat = 1 - self.mlp_seat;
        let (play, claims) = self.bot.choose(&self.game.player_view(seat));
        self.game
            .play_turn(
                seat,
                TurnRequest {
                    version: i64::try_from(self.game.version).map_err(|_| "version exceeds i64")?,
                    turn_id: None,
                    play,
                    claims,
                },
                0,
            )
            .map_err(|error| error.to_string())
    }

    fn validate_action(&self, action: &Action) -> Result<TurnRequest, String> {
        let request = TurnRequest {
            version: i64::try_from(action.version).map_err(|_| "version exceeds i64")?,
            turn_id: None,
            play: action.play.clone(),
            claims: Vec::new(),
        };
        request.validate().map_err(|error| error.to_string())?;
        let legal = self
            .game
            .legal_moves(self.mlp_seat)
            .map_err(|error| error.to_string())?;
        if action.version != self.game.version {
            return Err(format!(
                "STALE_VERSION: expected {}, got {}",
                self.game.version, action.version
            ));
        }
        let plays = legal["plays"]
            .as_array()
            .ok_or("invalid engine legal moves")?;
        let valid = match &action.play {
            Some(play) => plays.iter().any(|entry| {
                entry["card"] == json!(play.card)
                    && entry["stones"]
                        .as_array()
                        .is_some_and(|stones| stones.contains(&json!(play.stone)))
            }),
            None => plays.is_empty(),
        };
        if !valid {
            return Err("illegal MLP placement or pass".into());
        }
        Ok(request)
    }

    fn mlp_turn(&mut self, mut request: TurnRequest) -> Result<(), String> {
        request.claims =
            Bot::provable_stones(&self.game.player_view(self.mlp_seat), request.play.as_ref());
        self.game
            .play_turn(self.mlp_seat, request, 0)
            .map_err(|error| error.to_string())?;
        self.bot_turn()
    }
}

#[derive(Default)]
struct Worker {
    games: Vec<Match>,
    generation: u64,
}

impl Worker {
    fn handle_line(&mut self, line: &str) -> Result<Value, String> {
        let command: Command = serde_json::from_str(line).map_err(|error| error.to_string())?;
        match command {
            Command::New { games } => {
                if games.iter().any(|spec| spec.mlp_seat > 1) {
                    return Err("mlp_seat must be 0 or 1".into());
                }
                let generation = self
                    .generation
                    .checked_add(1)
                    .ok_or("game id counter exhausted")?;
                let batch: Result<Vec<Match>, String> = games
                    .into_par_iter()
                    .enumerate()
                    .map(|(id, spec)| {
                        let mut game = Game::new(
                            format!("neural-{generation}-{id}"),
                            Options {
                                seed: Some(spec.seed),
                                ..Default::default()
                            },
                            0,
                            false,
                        );
                        game.join(0).map_err(|error| error.to_string())?;
                        let mut entry = Match {
                            game,
                            bot: Bot::new("mcb:16", Some(spec.bot_seed))?,
                            mlp_seat: spec.mlp_seat,
                        };
                        if spec.mlp_seat == 1 {
                            entry.bot_turn()?;
                        }
                        Ok(entry)
                    })
                    .collect();
                self.games = batch?;
                self.generation = generation;
            }
            Command::Turn { actions } => {
                let mut requests: Vec<Option<TurnRequest>> = vec![None; self.games.len()];
                for action in actions {
                    let game = self
                        .games
                        .get(action.id)
                        .ok_or_else(|| format!("game id {} out of range", action.id))?;
                    if requests[action.id].is_some() {
                        return Err(format!("duplicate game id {}", action.id));
                    }
                    requests[action.id] = Some(
                        game.validate_action(&action)
                            .map_err(|error| format!("game {}: {error}", action.id))?,
                    );
                }
                self.games
                    .par_iter_mut()
                    .zip(requests.into_par_iter())
                    .enumerate()
                    .try_for_each(|(id, (game, request))| {
                        if let Some(request) = request {
                            game.mlp_turn(request)
                                .map_err(|error| format!("game {id}: {error}"))?;
                        }
                        Ok::<(), String>(())
                    })?;
            }
        }
        let games: Vec<Value> = self
            .games
            .par_iter()
            .enumerate()
            .map(|(id, entry)| json!({"id":id,"view":entry.game.view(entry.mlp_seat)}))
            .collect();
        Ok(json!({"games":games}))
    }
}

fn serve(reader: impl BufRead, mut writer: impl Write) -> Result<(), String> {
    let mut worker = Worker::default();
    for line in reader.lines() {
        let result = line
            .map_err(|error| error.to_string())
            .and_then(|line| worker.handle_line(&line));
        let response = match &result {
            Ok(response) => response.clone(),
            Err(error) => json!({"error":error}),
        };
        serde_json::to_writer(&mut writer, &response).map_err(|error| error.to_string())?;
        writeln!(writer).map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        result?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = serve(io::stdin().lock(), io::stdout().lock()) {
        eprintln!("neural worker: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(worker: &mut Worker, command: Value) -> Result<Value, String> {
        worker.handle_line(&command.to_string())
    }

    fn new_batch(worker: &mut Worker) -> Value {
        request(
            worker,
            json!({"op":"new","games":[
                {"seed":1,"bot_seed":7,"mlp_seat":0},
                {"seed":1,"bot_seed":7,"mlp_seat":1}
            ]}),
        )
        .unwrap()
    }

    fn action(snapshot: &Value, play: Option<Play>) -> Value {
        json!({"id":snapshot["id"],"version":snapshot["view"]["version"],"play":play})
    }

    fn first_play(snapshot: &Value) -> Option<Play> {
        let view = &snapshot["view"];
        let card = view["hand"].as_array().unwrap().first()?;
        let stone = view["stones"].as_array().unwrap().iter().find(|stone| {
            stone["claimed_by"].is_null() && stone["yours"].as_array().unwrap().len() < 3
        })?;
        Some(Play {
            card: serde_json::from_value(card.clone()).unwrap(),
            stone: stone["index"].as_u64().unwrap() as usize,
        })
    }

    #[test]
    fn opening_is_seat_correct_and_hides_opponent_cards() {
        let response = new_batch(&mut Worker::default());
        let games = response["games"].as_array().unwrap();
        assert_eq!(games.len(), 2);
        for (seat, snapshot) in games.iter().enumerate() {
            let view = &snapshot["view"];
            assert_eq!(snapshot["id"], seat);
            assert_eq!(view["you"], seat);
            assert_eq!(view["turn"], seat);
            assert_eq!(view["phase"], "your_turn");
            assert_eq!(view["version"], 1 + seat);
            assert_eq!(view["deck_count"], 42 - seat);
            assert_eq!(view["hand"].as_array().unwrap().len(), 6);
            assert_eq!(view["opponent_hand_count"], 6);
            assert!(view.get("opponent_hand").is_none());
            assert!(view.get("draw_order").is_none());
            assert_eq!(
                view["stones"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|stone| { stone["theirs"].as_array().unwrap().len() })
                    .sum::<usize>(),
                seat
            );
        }
        assert_eq!(
            games[0]["view"]["hand"],
            json!([
                {"color":"red","value":3},{"color":"blue","value":1},
                {"color":"blue","value":3},{"color":"green","value":2},
                {"color":"green","value":8},{"color":"purple","value":3}
            ])
        );
    }

    #[test]
    fn subset_turn_returns_all_games_and_only_advances_target() {
        let mut worker = Worker::default();
        let before = new_batch(&mut worker);
        let submitted = action(&before["games"][0], first_play(&before["games"][0]));
        let after = request(&mut worker, json!({"op":"turn","actions":[submitted]})).unwrap();
        assert_eq!(after["games"][0]["view"]["version"], 3);
        assert_eq!(after["games"][0]["view"]["phase"], "your_turn");
        assert_eq!(after["games"][0]["view"]["deck_count"], 40);
        assert_eq!(after["games"][1], before["games"][1]);
    }

    #[test]
    fn rejects_duplicate_out_of_range_and_stale_actions_before_mutation() {
        let mut worker = Worker::default();
        let before = new_batch(&mut worker);
        let valid = action(&before["games"][0], first_play(&before["games"][0]));
        let mut bad_id = valid.clone();
        bad_id["id"] = json!(2);
        let mut stale = valid.clone();
        stale["version"] = json!(0);
        let mut overflow = valid.clone();
        overflow["version"] = json!(u64::MAX);
        for actions in [
            json!([valid.clone(), valid]),
            json!([bad_id]),
            json!([stale]),
            json!([overflow]),
        ] {
            assert!(request(&mut worker, json!({"op":"turn","actions":actions})).is_err());
            assert_eq!(
                request(&mut worker, json!({"op":"turn","actions":[]})).unwrap(),
                before
            );
        }
    }

    #[test]
    fn malformed_and_illegal_moves_never_reach_claim_helper() {
        let mut worker = Worker::default();
        let before = request(
            &mut worker,
            json!({"op":"new","games":[
                {"seed":1,"bot_seed":7,"mlp_seat":0}
            ]}),
        )
        .unwrap();
        for play in [
            Value::Null,
            json!({"card":{"color":"red","value":0},"stone":0}),
            json!({"card":{"color":"red","value":10},"stone":0}),
            json!({"card":{"color":"red","value":3},"stone":9}),
            json!({"card":{"color":"red","value":1},"stone":0}),
        ] {
            assert!(
                request(
                    &mut worker,
                    json!({"op":"turn","actions":[
                        {"id":0,"version":1,"play":play}
                    ]})
                )
                .is_err()
            );
            assert_eq!(
                request(&mut worker, json!({"op":"turn","actions":[]})).unwrap(),
                before
            );
        }
        for _ in 0..3 {
            let snapshot = request(&mut worker, json!({"op":"turn","actions":[]})).unwrap();
            let mut play = first_play(&snapshot["games"][0]).unwrap();
            play.stone = 0;
            request(
                &mut worker,
                json!({"op":"turn","actions":[action(&snapshot["games"][0],Some(play))]}),
            )
            .unwrap();
        }
        let snapshot = request(&mut worker, json!({"op":"turn","actions":[]})).unwrap();
        let mut play = first_play(&snapshot["games"][0]).unwrap();
        play.stone = 0;
        assert!(
            request(
                &mut worker,
                json!({"op":"turn","actions":[action(&snapshot["games"][0],Some(play))]})
            )
            .is_err()
        );
    }

    #[test]
    fn reset_replaces_batch_and_unknown_operations_are_errors() {
        let mut worker = Worker::default();
        let before = new_batch(&mut worker);
        for line in [
            "not json",
            r#"{"op":"unknown"}"#,
            r#"{"op":"new","games":[{"seed":1,"bot_seed":1,"mlp_seat":2}]}"#,
        ] {
            assert!(worker.handle_line(line).is_err());
        }
        let after = new_batch(&mut worker);
        assert_ne!(
            before["games"][0]["view"]["game_id"],
            after["games"][0]["view"]["game_id"]
        );
        assert_eq!(
            request(&mut worker, json!({"op":"new","games":[]})).unwrap(),
            json!({"games":[]})
        );
        assert!(
            request(
                &mut worker,
                json!({"op":"turn","actions":[{"id":0,"version":1,"play":null}]})
            )
            .is_err()
        );
    }

    fn native_turn(game: &mut Game, seat: usize, play: Option<Play>) {
        let claims = Bot::provable_stones(&game.player_view(seat), play.as_ref());
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
        .unwrap();
    }

    #[test]
    fn both_seats_follow_seeded_native_trace_through_terminal() {
        let mut worker = Worker::default();
        let mut response = new_batch(&mut worker);
        let mut reference: Vec<_> = (0..2)
            .map(|seat| {
                let mut game = Game::new(
                    response["games"][seat]["view"]["game_id"]
                        .as_str()
                        .unwrap()
                        .into(),
                    Options {
                        seed: Some(1),
                        ..Default::default()
                    },
                    0,
                    false,
                );
                game.join(0).unwrap();
                let mut bot = Bot::new("mcb:16", Some(7)).unwrap();
                if seat == 1 {
                    let (play, claims) = bot.choose(&game.player_view(0));
                    game.play_turn(
                        0,
                        TurnRequest {
                            version: game.version as i64,
                            turn_id: None,
                            play,
                            claims,
                        },
                        0,
                    )
                    .unwrap();
                }
                (game, bot)
            })
            .collect();
        for _ in 0..60 {
            let mut actions = Vec::new();
            for (seat, (game, bot)) in reference.iter_mut().enumerate() {
                assert_eq!(response["games"][seat]["view"], game.view(seat));
                if game.over() {
                    continue;
                }
                let play = first_play(&response["games"][seat]);
                actions.push(action(&response["games"][seat], play.clone()));
                native_turn(game, seat, play);
                if !game.over() {
                    let (play, claims) = bot.choose(&game.player_view(1 - seat));
                    game.play_turn(
                        1 - seat,
                        TurnRequest {
                            version: game.version as i64,
                            turn_id: None,
                            play,
                            claims,
                        },
                        0,
                    )
                    .unwrap();
                }
            }
            if actions.is_empty() {
                break;
            }
            response = request(&mut worker, json!({"op":"turn","actions":actions})).unwrap();
        }
        for (seat, (game, _)) in reference.iter().enumerate() {
            assert!(game.over());
            assert_eq!(response["games"][seat]["view"], game.view(seat));
            assert_eq!(response["games"][seat]["view"]["phase"], "game_over");
            assert!(
                matches!(
                    response["games"][seat]["view"]["winner"].as_str(),
                    Some("you" | "them")
                ) || game.outcome().unwrap().0.is_none()
            );
            assert!(
                request(
                    &mut worker,
                    json!({"op":"turn","actions":[action(&response["games"][seat],None)]})
                )
                .is_err()
            );
        }
    }
}
