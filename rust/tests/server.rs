use axum::{
    body::Body,
    http::{Request, StatusCode},
};
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
        if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        },
    )
}

#[tokio::test]
async fn lobby_turn_auth_and_idempotence() {
    let app = router();
    let (status, host) = call(app.clone(), "POST", "/games", None, json!({"seed":7})).await;
    assert_eq!(status, 201);
    let path = format!("/games/{}", host["game_id"].as_str().unwrap());
    let token = host["token"].as_str().unwrap();
    assert_eq!(
        call(app.clone(), "GET", &path, None, Value::Null).await.0,
        401
    );
    let (status, _) = call(
        app.clone(),
        "POST",
        &format!("{path}/join"),
        None,
        Value::Null,
    )
    .await;
    assert_eq!(status, 201);
    let (_, view) = call(app.clone(), "GET", &path, Some(token), Value::Null).await;
    assert_eq!(view["phase"], "your_turn");
    assert_eq!(view["hand"].as_array().unwrap().len(), 6);
    let (_, legal) = call(
        app.clone(),
        "GET",
        &format!("{path}/legal-moves"),
        Some(token),
        Value::Null,
    )
    .await;
    assert_eq!(legal["plays"].as_array().unwrap().len(), 6);
    let turn = json!({"version":view["version"],"turn_id":"ae2d46c4-ff54-4128-b0d5-d0d9c38e1814","play":{"card":view["hand"][0],"stone":0}});
    let (status, after) = call(
        app.clone(),
        "POST",
        &format!("{path}/turn"),
        Some(token),
        turn.clone(),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(after["phase"], "opponent_turn");
    let (status, retry) = call(app, "POST", &format!("{path}/turn"), Some(token), turn).await;
    assert_eq!(status, 200);
    assert_eq!(retry, after);
}

#[tokio::test]
async fn invalid_requests_use_contract_error() {
    let app = router();
    let (status, body) = call(
        app.clone(),
        "POST",
        "/games",
        None,
        json!({"turn_timeout_seconds":1}),
    )
    .await;
    assert_eq!(status, 422);
    assert_eq!(body["error"], "INVALID_REQUEST");
    let (_, host) = call(app.clone(), "POST", "/games", None, Value::Null).await;
    let path = format!("/games/{}", host["game_id"].as_str().unwrap());
    let token = host["token"].as_str().unwrap();
    for body in [
        json!({"version":0,"claims":[9]}),
        json!({"version":0,"play":{"card":{"color":"blue","value":10},"stone":0}}),
        json!({"version":0,"turn_id":"bad"}),
    ] {
        let (status, error) = call(
            app.clone(),
            "POST",
            &format!("{path}/turn"),
            Some(token),
            body,
        )
        .await;
        assert_eq!(
            (status, error["error"].as_str().unwrap()),
            (422, "INVALID_REQUEST")
        );
    }
    assert_eq!(
        call(
            app,
            "GET",
            &format!("{path}?timeout=0"),
            Some(token),
            Value::Null
        )
        .await
        .0,
        422
    );
}

#[tokio::test]
async fn long_poll_wakes_on_join_without_holding_game_lock() {
    let app = router();
    let (_, host) = call(app.clone(), "POST", "/games", None, Value::Null).await;
    let path = format!("/games/{}", host["game_id"].as_str().unwrap());
    let token = host["token"].as_str().unwrap().to_owned();
    let waiting_app = app.clone();
    let waiting_path = format!("{path}?since=0&timeout=5");
    let waiting = tokio::spawn(async move {
        call(waiting_app, "GET", &waiting_path, Some(&token), Value::Null).await
    });
    tokio::task::yield_now().await;
    assert_eq!(
        call(app, "POST", &format!("{path}/join"), None, Value::Null)
            .await
            .0,
        201
    );
    let (status, view) = tokio::time::timeout(std::time::Duration::from_secs(2), waiting)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status, 200);
    assert_eq!(view["version"], 1);
}

#[tokio::test]
async fn nullable_and_coerced_create_options_match_python() {
    let app = router();
    let (status, _) = call(
        app.clone(),
        "POST",
        "/games",
        None,
        json!({"seed":"7", "auto_claim":"false", "turn_timeout_seconds":"60"}),
    )
    .await;
    assert_eq!(status, 201);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/games")
                .header("content-type", "application/json")
                .body(Body::from(" null \n"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
}

#[tokio::test]
async fn positional_arrays_and_fractional_strings_are_rejected() {
    let app = router();
    for body in [
        json!([]),
        json!([false, null, 7]),
        json!({"seed":"1.00000000000000001"}),
        json!({"seed":"+-7"}),
    ] {
        let (status, body) = call(app.clone(), "POST", "/games", None, body).await;
        assert_eq!(
            (status, body["error"].as_str().unwrap()),
            (422, "INVALID_REQUEST")
        );
    }
    let (_, host) = call(app.clone(), "POST", "/games", None, json!({"seed":7})).await;
    let path = format!("/games/{}", host["game_id"].as_str().unwrap());
    let token = host["token"].as_str().unwrap();
    call(
        app.clone(),
        "POST",
        &format!("{path}/join"),
        None,
        Value::Null,
    )
    .await;
    let (_, view) = call(app.clone(), "GET", &path, Some(token), Value::Null).await;
    let card = &view["hand"][0];
    for body in [
        json!([1, null, null, []]),
        json!({"version":1,"play":[card,0]}),
        json!({"version":1,"play":{"card":[card["color"],card["value"]],"stone":0}}),
    ] {
        let (status, error) = call(
            app.clone(),
            "POST",
            &format!("{path}/turn"),
            Some(token),
            body,
        )
        .await;
        assert_eq!(
            (status, error["error"].as_str().unwrap()),
            (422, "INVALID_REQUEST")
        );
    }
}

#[tokio::test]
async fn decimal_seed_strings_are_exact_and_query_numbers_match_python() {
    let app = router();
    let mut hands = Vec::new();
    for seed in [json!("9007199254740993.0"), json!(9007199254740993i64)] {
        let (_, host) = call(app.clone(), "POST", "/games", None, json!({"seed":seed})).await;
        let path = format!("/games/{}", host["game_id"].as_str().unwrap());
        let token = host["token"].as_str().unwrap();
        let (_, view) = call(app.clone(), "GET", &path, Some(token), Value::Null).await;
        hands.push(view["hand"].clone());
        assert_eq!(
            call(
                app.clone(),
                "GET",
                &format!("{path}?since=1.0&timeout=1.0"),
                Some(token),
                Value::Null
            )
            .await
            .0,
            200
        );
        assert_eq!(
            call(
                app.clone(),
                "GET",
                &format!("{path}?since=999999999999999999999999999999&timeout=1"),
                Some(token),
                Value::Null
            )
            .await
            .0,
            200
        );
    }
    assert_eq!(hands[0], hands[1]);
}

#[tokio::test]
async fn shutdown_terminates_waiting_sse_streams() {
    let (shutdown, _) = tokio::sync::watch::channel(false);
    let app = schotten_native::server::router_with_shutdown(shutdown.clone());
    let (_, host) = call(app.clone(), "POST", "/games", None, Value::Null).await;
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/games/{}/events",
                    host["game_id"].as_str().unwrap()
                ))
                .header(
                    "authorization",
                    format!("Bearer {}", host["token"].as_str().unwrap()),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let mut body = response.into_body();
    assert!(body.frame().await.is_some());
    shutdown.send_replace(true);
    let next = tokio::time::timeout(std::time::Duration::from_secs(1), body.frame())
        .await
        .unwrap();
    assert!(next.is_none());
}
