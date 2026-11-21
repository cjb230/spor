use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use schotten_native::server::router;
use serde_json::{Value, json};
use tower::ServiceExt;

async fn call(
    app: axum::Router,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Value,
) -> (u16, Value) {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    if let Some(token) = token {
        request = request.header("authorization", format!("Bearer {token}"));
    }
    let response = app
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn human_game_has_private_bot_and_requires_its_own_token() {
    let app = router();
    let (status, grant) = call(
        app.clone(),
        "POST",
        "/web/games",
        None,
        json!({"samples":1,"seed":7,"bot_seed":1}),
    )
    .await;
    assert_eq!(status, 201);
    assert_eq!(grant["seat"], 0);
    assert!(grant.get("bot_token").is_none());
    let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
    assert_eq!(
        call(app.clone(), "GET", &path, None, Value::Null).await.0,
        401
    );
    assert_eq!(
        call(app.clone(), "GET", &path, Some("wrong"), Value::Null)
            .await
            .0,
        401
    );
    let (status, view) = call(app, "GET", &path, grant["token"].as_str(), Value::Null).await;
    assert_eq!(status, 200);
    assert_eq!(view["phase"], "your_turn");
    assert_eq!(view["hand"].as_array().unwrap().len(), 6);
    assert_eq!(view["opponent_hand_count"], 6);
    assert!(view.get("opponent_hand").is_none());
    assert!(!view["legal_moves"]["plays"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn web_game_rejects_malformed_input() {
    let app = router();
    for body in [
        json!({"samples":1.5}),
        json!({"samples":-1}),
        json!({"seed":"bad"}),
        json!([]),
    ] {
        assert_eq!(
            call(app.clone(), "POST", "/web/games", None, body).await.0,
            422
        );
    }
}

#[tokio::test]
async fn accepted_human_move_schedules_one_real_mcb_reply_even_on_retry() {
    let app = router();
    let (_, grant) = call(
        app.clone(),
        "POST",
        "/web/games",
        None,
        json!({"samples":1,"seed":7,"bot_seed":1}),
    )
    .await;
    assert!(
        grant["game_id"].is_string(),
        "bot-game creation must succeed"
    );
    let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
    let token = grant["token"].as_str();
    let (_, view) = call(app.clone(), "GET", &path, token, Value::Null).await;
    let turn = json!({"version":view["version"],"turn_id":"b300d918-2c46-4b3c-b052-81aa305f4567","play":{"card":view["hand"][0],"stone":0}});
    assert_eq!(
        call(
            app.clone(),
            "POST",
            &format!("{path}/turn"),
            token,
            turn.clone()
        )
        .await
        .0,
        200
    );
    assert_eq!(
        call(app.clone(), "POST", &format!("{path}/turn"), token, turn)
            .await
            .0,
        200
    );
    let completed = tokio::time::timeout(std::time::Duration::from_secs(20), async {
        loop {
            let (_, view) = call(app.clone(), "GET", &path, token, Value::Null).await;
            assert!(view["bot_error"].is_null(), "{view}");
            if view["phase"] == "your_turn" {
                break view;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("MCB must finish its reply");
    assert_eq!(completed["version"], 3);
    assert_eq!(completed["deck_count"], 40);
    let stones = completed["stones"].as_array().unwrap();
    assert_eq!(
        stones
            .iter()
            .map(|s| s["theirs"].as_array().unwrap().len())
            .sum::<usize>(),
        1
    );
    assert_eq!(completed["bot_thinking"], false);
    let recent = &completed["recent_cards"];
    let drawn = completed["hand"]
        .as_array()
        .unwrap()
        .iter()
        .find(|card| !view["hand"].as_array().unwrap().contains(card))
        .unwrap();
    assert_eq!(recent["drawn"], *drawn);
    let played = &recent["opponentPlay"];
    assert!(
        completed["stones"][played["stone"].as_u64().unwrap() as usize]["theirs"]
            .as_array()
            .unwrap()
            .contains(&played["card"])
    );
    let (_, restored) = call(app, "GET", &path, token, Value::Null).await;
    assert_eq!(restored["recent_cards"], *recent);
}

#[tokio::test]
async fn browser_assets_are_not_cached_between_rebuilds() {
    for path in ["/", "/web/app.js", "/web/style.css"] {
        let response = router()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            response
                .headers()
                .get("cache-control")
                .and_then(|value| value.to_str().ok()),
            Some("no-store")
        );
    }
}

#[tokio::test]
async fn rejected_human_turn_does_not_start_bot_or_change_board() {
    let app = router();
    let (_, grant) = call(
        app.clone(),
        "POST",
        "/web/games",
        None,
        json!({"samples":1,"seed":7}),
    )
    .await;
    assert!(
        grant["game_id"].is_string(),
        "bot-game creation must succeed"
    );
    let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
    let token = grant["token"].as_str();
    let (_, before) = call(app.clone(), "GET", &path, token, Value::Null).await;
    assert_eq!(
        call(
            app.clone(),
            "POST",
            &format!("{path}/turn"),
            token,
            json!({"version":1})
        )
        .await
        .0,
        422
    );
    let (_, after) = call(app, "GET", &path, token, Value::Null).await;
    assert_eq!(before, after);
}

#[tokio::test]
async fn invalid_cards_indices_and_stale_versions_leave_session_intact() {
    let app = router();
    let (_, grant) = call(
        app.clone(),
        "POST",
        "/web/games",
        None,
        json!({"samples":1,"seed":7}),
    )
    .await;
    assert!(grant["game_id"].is_string());
    let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
    let token = grant["token"].as_str();
    let (_, before) = call(app.clone(), "GET", &path, token, Value::Null).await;
    for body in [
        json!({"version":1,"play":{"card":{"color":"red","value":0},"stone":0}}),
        json!({"version":1,"play":{"card":before["hand"][0],"stone":9}}),
        json!({"version":0,"play":{"card":before["hand"][0],"stone":0}}),
        json!({"version":1,"turn_id":"invalid","play":{"card":before["hand"][0],"stone":0}}),
    ] {
        let (status, _) = call(app.clone(), "POST", &format!("{path}/turn"), token, body).await;
        assert!(status == 422 || status == 409);
        assert_eq!(
            call(app.clone(), "GET", &path, token, Value::Null).await.1,
            before
        );
    }
}

#[tokio::test]
async fn browser_seat_tokens_are_independent_secrets() {
    let app = router();
    let (_, first) = call(
        app.clone(),
        "POST",
        "/web/games",
        None,
        json!({"samples":1}),
    )
    .await;
    let (_, second) = call(
        app.clone(),
        "POST",
        "/web/games",
        None,
        json!({"samples":1}),
    )
    .await;
    assert_ne!(first["token"], second["token"]);
    let path = format!("/web/games/{}", second["game_id"].as_str().unwrap());
    assert_eq!(
        call(
            app.clone(),
            "GET",
            &path,
            first["token"].as_str(),
            Value::Null
        )
        .await
        .0,
        401
    );
    assert_eq!(
        call(app, "GET", &path, Some("native-seat-0"), Value::Null)
            .await
            .0,
        401
    );
}

#[tokio::test]
async fn completed_browser_games_are_saved_and_replayable_after_server_restart() {
    let root = std::env::temp_dir().join(format!("spor-web-recording-{}", uuid::Uuid::new_v4()));
    let app = schotten_native::web::router(root.clone());
    let (grant, finished) = finish_browser_game(&app, json!({"samples":1})).await;
    let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
    let token = grant["token"].as_str();
    assert_eq!(finished["recording_saved"], true, "{finished}");
    assert!(finished["recording_error"].is_null());
    let id = finished["recording_id"].as_str().expect("saved replay ID");
    let saved: Value = serde_json::from_slice(&std::fs::read(root.join(id)).unwrap()).unwrap();
    assert_eq!(saved["seats"][0]["name"], "You");
    assert_eq!(saved["seats"][1]["strategy"], "mcb:1");
    assert!(saved.get("token").is_none());
    assert!(saved["deck_seed"].as_str().unwrap().parse::<i64>().is_ok());
    assert_eq!(
        saved["log"]["revealed"]["initial_hands"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    // A fresh server must discover and play the file with no in-memory session.
    let restarted = schotten_native::web::router(root.clone());
    let (status, library) = call(
        restarted.clone(),
        "GET",
        "/web/recordings",
        None,
        Value::Null,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(library["games"].as_array().unwrap().len(), 1);
    assert!(
        library["warnings"].as_array().unwrap().is_empty(),
        "{library}"
    );
    let (status, replay) = call(
        restarted,
        "GET",
        &format!("/web/recordings/{id}"),
        None,
        Value::Null,
    )
    .await;
    assert_eq!(status, 200, "{replay}");
    let last = replay["frames"].as_array().unwrap().last().unwrap();
    assert_eq!(last["stones"], finished["stones"]);
    assert_eq!(last["winner"], finished["winner"]);
    assert_eq!(last["win_reason"], finished["win_reason"]);
    // Re-reading/retrying finished sessions does not generate duplicate recordings.
    call(app.clone(), "GET", &path, token, Value::Null).await;
    let (_, library) = call(app, "GET", "/web/recordings", None, Value::Null).await;
    assert_eq!(library["games"].as_array().unwrap().len(), 1);
    std::fs::remove_dir_all(root).unwrap();
}

async fn finish_browser_game(app: &axum::Router, options: Value) -> (Value, Value) {
    let (_, grant) = call(app.clone(), "POST", "/web/games", None, options).await;
    let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
    let token = grant["token"].as_str();
    let finished = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            let (_, view) = call(app.clone(), "GET", &path, token, Value::Null).await;
            assert!(view["bot_error"].is_null(), "{view}");
            if view["phase"] == "game_over" {
                break view;
            }
            if view["phase"] == "your_turn" {
                let plays = view["legal_moves"]["plays"].as_array().unwrap();
                let play = plays
                    .first()
                    .map(|p| json!({"card":p["card"], "stone":p["stones"][0]}));
                let (status, response) = call(
                    app.clone(),
                    "POST",
                    &format!("{path}/turn"),
                    token,
                    json!({"version":view["version"],"turn_id":uuid::Uuid::new_v4(),"play":play}),
                )
                .await;
                assert_eq!(status, 200, "{response}");
            } else {
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            }
        }
    })
    .await
    .expect("browser game terminates");
    (grant, finished)
}

#[tokio::test]
async fn completed_games_report_save_failures_and_retry_without_changing_the_outcome() {
    let root = std::env::temp_dir().join(format!("spor-web-unwritable-{}", uuid::Uuid::new_v4()));
    std::fs::write(&root, b"a file cannot contain recordings").unwrap();
    let app = schotten_native::web::router(root.clone());
    let (grant, failed) = finish_browser_game(
        &app,
        json!({"samples":1,"seed":-9223372036854775808i64,"bot_seed":9223372036854775807i64}),
    )
    .await;
    assert_eq!(failed["recording_saved"], false);
    assert!(failed["recording_error"].is_string());
    assert!(failed["recording_id"].is_null());
    std::fs::remove_file(&root).unwrap();
    let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
    let (_, saved) = call(
        app.clone(),
        "GET",
        &path,
        grant["token"].as_str(),
        Value::Null,
    )
    .await;
    assert_eq!(saved["recording_saved"], true);
    assert!(saved["recording_error"].is_null());
    assert_eq!(saved["version"], failed["version"]);
    assert_eq!(saved["stones"], failed["stones"]);
    assert_eq!(saved["winner"], failed["winner"]);
    let id = saved["recording_id"].as_str().unwrap();
    let file: Value = serde_json::from_slice(&std::fs::read(root.join(id)).unwrap()).unwrap();
    assert_eq!(file["deck_seed"], "-9223372036854775808");
    assert_eq!(file["seats"][1]["seed"], "9223372036854775807");
    let (status, replay) = call(
        app,
        "GET",
        &format!("/web/recordings/{id}"),
        None,
        Value::Null,
    )
    .await;
    assert_eq!(status, 200, "{replay}");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn browser_opponents_use_selected_policies_and_restore_their_metadata() {
    let app = router();
    for (policy, name) in [
        ("random", "Random"),
        ("greedy", "Greedy"),
        ("builder", "Builder"),
        ("tactician", "Tactician"),
        ("planner", "Planner"),
        ("mcg", "MCG"),
        ("mcb", "MCB"),
        ("mcp", "MCP"),
    ] {
        let mut options = json!({"opponent":policy,"seed":7,"bot_seed":1});
        if policy.starts_with("mc") {
            options["samples"] = json!(1);
        }
        let (status, grant) = call(app.clone(), "POST", "/web/games", None, options).await;
        assert_eq!(status, 201, "{policy}: {grant}");
        let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
        let (_, view) = call(
            app.clone(),
            "GET",
            &path,
            grant["token"].as_str(),
            Value::Null,
        )
        .await;
        assert_eq!(view["opponent"]["name"], name);
        let strategy = if policy.starts_with("mc") {
            format!("{policy}:1")
        } else {
            policy.into()
        };
        assert_eq!(view["opponent"]["strategy"], strategy);
        let (status, _) = call(
            app.clone(),
            "POST",
            &format!("{path}/turn"),
            grant["token"].as_str(),
            json!({"version":1,"play":{"card":view["hand"][0],"stone":0}}),
        )
        .await;
        assert_eq!(status, 200);
        let completed = tokio::time::timeout(std::time::Duration::from_secs(30), async {
            loop {
                let (_, next) = call(
                    app.clone(),
                    "GET",
                    &path,
                    grant["token"].as_str(),
                    Value::Null,
                )
                .await;
                assert!(next["bot_error"].is_null(), "{policy}: {next}");
                if next["phase"] == "your_turn" {
                    break next;
                }
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("native opponent replies");
        assert_eq!(completed["version"], 3);
        assert_eq!(completed["opponent"], view["opponent"]);
        // Compare transport-selected play against the existing native policy,
        // using the known seeded deal rather than trusting only a display name.
        use schotten_native::{
            bots::Bot,
            engine::{Game, Options, TurnRequest},
        };
        let mut reference = Game::new(
            "reference".into(),
            Options {
                seed: Some(7),
                ..Default::default()
            },
            0,
            true,
        );
        reference.join(0).unwrap();
        reference
            .play_turn(
                0,
                TurnRequest {
                    version: 1,
                    turn_id: None,
                    play: serde_json::from_value(json!({"card":view["hand"][0],"stone":0}))
                        .unwrap(),
                    claims: vec![],
                },
                0,
            )
            .unwrap();
        let mut bot = Bot::new(&strategy, Some(1)).unwrap();
        let (play, claims) = bot.choose(&reference.player_view(1));
        reference
            .play_turn(
                1,
                TurnRequest {
                    version: 2,
                    turn_id: None,
                    play,
                    claims,
                },
                0,
            )
            .unwrap();
        assert_eq!(completed["stones"], reference.view(0)["stones"], "{policy}");
    }
}

#[tokio::test]
async fn browser_opponent_defaults_and_validation_match_native_policies() {
    let app = router();
    for (options, strategy) in [
        (json!({}), "mcb:20"),
        (json!({"opponent":"mcg"}), "mcg:100"),
        (json!({"opponent":"mcp"}), "mcp:5"),
        (json!({"opponent":"mcb", "samples":0}), "mcb:0"),
        (json!({"opponent":"mcg", "samples":101}), "mcg:101"),
        (json!({"opponent":"mcp", "samples":1000001}), "mcp:1000001"),
        (
            json!({"opponent":"mcb", "samples":usize::MAX}),
            "mcb:18446744073709551615",
        ),
    ] {
        let (status, grant) = call(app.clone(), "POST", "/web/games", None, options).await;
        assert_eq!(status, 201, "{grant}");
        let path = format!("/web/games/{}", grant["game_id"].as_str().unwrap());
        let (_, view) = call(
            app.clone(),
            "GET",
            &path,
            grant["token"].as_str(),
            Value::Null,
        )
        .await;
        assert_eq!(view["opponent"]["strategy"], strategy);
    }
    for options in [
        json!({"opponent":"unknown"}),
        json!({"opponent":"mcg:1000000"}),
        json!({"opponent":"random", "samples":10}),
        json!({"opponent":null}),
    ] {
        assert_eq!(
            call(app.clone(), "POST", "/web/games", None, options)
                .await
                .0,
            422
        );
    }
}

#[tokio::test]
async fn saved_browser_replays_identify_the_chosen_opponent() {
    let root = std::env::temp_dir().join(format!("spor-web-opponent-{}", uuid::Uuid::new_v4()));
    let app = schotten_native::web::router(root.clone());
    let (_, finished) =
        finish_browser_game(&app, json!({"opponent":"random","seed":7,"bot_seed":1})).await;
    assert_eq!(finished["recording_saved"], true);
    let id = finished["recording_id"].as_str().unwrap();
    let saved: Value = serde_json::from_slice(&std::fs::read(root.join(id)).unwrap()).unwrap();
    assert_eq!(saved["seats"][1]["name"], "Random");
    assert_eq!(saved["seats"][1]["strategy"], "random");
    assert!(saved["winner"].is_null() || saved["winner"] == "You" || saved["winner"] == "Random");
    let (status, replay) = call(
        app,
        "GET",
        &format!("/web/recordings/{id}"),
        None,
        Value::Null,
    )
    .await;
    assert_eq!(status, 200, "{replay}");
    assert_eq!(replay["metadata"]["seats"][1]["name"], "Random");
    std::fs::remove_dir_all(root).unwrap();
}
