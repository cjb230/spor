"""External Axum compatibility tests (no subprocesses or socket fixtures).

Start the Rust server yourself, then run::

    SCHOTTEN_RUST_URL=http://127.0.0.1:8001 pytest tests/test_rust_server.py

With no URL these tests skip, keeping the normal Python suite self-contained.
Every network request/stream has a finite timeout. Validation compares the exact
422/INVALID_REQUEST contract, not Pydantic-specific diagnostic wording.
"""

import copy
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from schotten.app import create_app
from schotten.bot import Bot
from schotten.client import SchottenClient
from schotten.schemas import GameView

from rust_reference import (
    DifferentialGame, Endpoint, assert_response, normalized, sse_frames, terminal_events,
)

RUST_URL = os.environ.get("SCHOTTEN_RUST_URL")
pytestmark = pytest.mark.skipif(not RUST_URL, reason="set SCHOTTEN_RUST_URL to a running Rust server")


@pytest.fixture
def python():
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def rust():
    with httpx.Client(base_url=RUST_URL.rstrip("/"), timeout=10, trust_env=False) as client:
        yield client


@pytest.fixture
def pair(python, rust):
    return DifferentialGame(python, rust)


@pytest.fixture(params=["python", "rust"])
def endpoint(request, python, rust):
    return Endpoint(python if request.param == "python" else rust)


def turn_body(view, stone=0, **extra):
    return {
        "version": view["version"],
        "play": {"card": view["hand"][0], "stone": stone},
        **extra,
    }


def bot_body(bot, view):
    play, claims = bot.choose(GameView.model_validate(view))
    return {
        "version": view["version"], "turn_id": str(uuid4()),
        "play": None if play is None else play.model_dump(mode="json"),
        "claims": claims,
    }


def finish_game(pair, seed):
    bots = [
        Bot(SchottenClient(http=pair.python.http), strategy="greedy" if seed % 4 == 0 else "random",
            rng=random.Random(seed * 2 + seat))
        for seat in (0, 1)
    ]
    for _ in range(110):
        views = pair.snapshot()
        if views[0]["phase"] == "game_over":
            return views
        seat = views[0]["turn"]
        body = bot_body(bots[seat], views[seat])
        response, _ = pair.check("POST", "/turn", seat, json=body)
        assert response.json()["version"] == body["version"] + 1
    pytest.fail(f"seed {seed} did not terminate within 110 turns")


@pytest.mark.parametrize("auto_claim", [False, True])
@pytest.mark.parametrize("seed", [*range(8), -7, 2**32 + 1, 2**63 - 1, 2**80 + 7])
def test_seeded_initial_views_and_entire_bot_games_match_python(pair, seed, auto_claim):
    pair.create(seed=seed, auto_claim=auto_claim).join()
    views = pair.snapshot()
    assert views[0]["version"] == 1
    assert views[0]["deck_count"] == 42
    assert views[0]["opponent_hand_count"] == 6
    assert len(views[0]["unseen_cards"]) == 48
    assert not any(card in views[0]["hand"] for card in views[1]["hand"])
    assert all(card in views[0]["unseen_cards"] for card in views[1]["hand"])
    assert pair.check("GET", "/log")[0].json()["revealed"] is None

    final = finish_game(pair, seed)
    assert final[0]["status"] == "finished"
    assert final[0]["win_reason"] in {"five_stones", "three_adjacent", "stalemate"}
    for seat in (0, 1):
        log = pair.check("GET", "/log", seat)[0].json()
        assert log["revealed"] is not None
        assert len(log["revealed"]["initial_hands"]) == 2
        assert len(log["revealed"]["draw_order"]) == 42
        cards = sum(log["revealed"]["initial_hands"], []) + log["revealed"]["draw_order"]
        assert len({(card["color"], card["value"]) for card in cards}) == 54
        assert len(log["entries"]) == final[0]["version"] - 1
        response, _ = pair.check("POST", "/turn", seat, status=403,
                                 json={"version": final[0]["version"]})
        assert response.json()["error"] == "GAME_OVER"
        pair.check("GET", seat=seat, params={"since": final[0]["version"], "timeout": 1})


@pytest.mark.parametrize("options", [
    {}, {"seed": None, "turn_timeout_seconds": None},
    {"seed": "7", "auto_claim": "false", "ignored": "extra fields are ignored"},
    {"seed": -1, "auto_claim": 1, "turn_timeout_seconds": "60"},
    {"seed": 7.0, "auto_claim": 0, "turn_timeout_seconds": 5.0},
    {"seed": True, "auto_claim": "yes"},
])
def test_default_nullable_and_coerced_options_preserve_python_behavior(pair, options):
    if options.get("seed") is None:
        for endpoint in (pair.python, pair.rust):
            endpoint.create(**options)
        left, right = pair.python.view(), pair.rust.view()
        for view in (left, right):
            assert len(view["hand"]) == 6 and len(view["unseen_cards"]) == 48
        assert set(left) == set(right)
        for key in set(left) - {"game_id", "hand", "unseen_cards"}:
            assert left[key] == right[key]
    else:
        pair.create(**options).join()
        pair.snapshot()


@pytest.mark.parametrize("body", [None, b"null"])
def test_create_accepts_absent_or_json_null_body(python, rust, body):
    endpoints = [Endpoint(client) for client in (python, rust)]
    for endpoint in endpoints:
        response = endpoint.http.post("/games", content=body,
                                      headers={"Content-Type": "application/json"})
        endpoint.adopt(response, 0)
        view = endpoint.view()
        assert view["status"] == "waiting_for_opponent" and view["version"] == 0
        assert len(view["hand"]) == 6 and len(view["unseen_cards"]) == 48
    assert set(endpoints[0].view()) == set(endpoints[1].view())


@pytest.mark.parametrize("coerce", [str, float, bool], ids=["numeric-strings", "integral-floats", "booleans"])
def test_turn_scalars_are_coerced_before_validation_and_extras_ignored(pair, coerce):
    pair.create(seed=7).join()
    view = pair.view()
    assert view["hand"][0]["value"] == 1
    body = copy.deepcopy(turn_body(view))
    body["version"] = coerce(body["version"])
    body["play"]["stone"] = coerce(0)
    body["play"]["card"]["value"] = coerce(1)
    body["ignored"] = "top-level extra"
    body["play"]["ignored"] = "play extra"
    body["play"]["card"]["ignored"] = "card extra"
    response, _ = pair.check("POST", "/turn", json={**body, "claims": [coerce(1)]}, status=422)
    assert response.json()["error"] == "CLAIM_INCOMPLETE"
    assert pair.view() == view
    pair.check("POST", "/turn", json=body)
    pair.snapshot()


INVALID_OPTIONS = [
    [], "body", 3, {"auto_claim": None}, {"auto_claim": "maybe"},
    {"turn_timeout_seconds": 4}, {"turn_timeout_seconds": -1},
    {"turn_timeout_seconds": "bad"}, {"turn_timeout_seconds": 5.5},
    {"seed": "bad"}, {"seed": 1.5}, {"seed": []},
]


@pytest.mark.parametrize("body", INVALID_OPTIONS)
def test_invalid_create_json_is_422_invalid_request(python, rust, body):
    assert_response(python.post("/games", json=body), rust.post("/games", json=body),
                    422, validation=True)


INVALID_TURNS = [
    {}, [], None, "body", 1,
    {"version": None}, {"version": "bad"}, {"version": 1.5},
    {"version": []}, {"version": 1, "play": {}},
    {"version": 1, "play": []}, {"version": 1, "play": {"stone": 0}},
    *[{"version": 1, "turn_id": value} for value in ["", "no-uuid", 123, []]],
    *[{"version": 1, "claims": value} for value in [None, {}, [-1], [9], [None], ["x"]]],
    *[{"version": 1, "play": {"card": card, "stone": 0}} for card in [
        None, [], "red1", {}, {"value": 1}, {"color": "red"},
        {"color": "RED", "value": 1}, {"color": "orange", "value": 1},
        *[{"color": "red", "value": value} for value in [0, 10, None, "x", 1.5]],
    ]],
    *[{"version": 1, "play": {"card": {"color": "red", "value": 1}, "stone": value}}
      for value in [-1, 9, None, "x", 0.5]],
]


@pytest.mark.parametrize("body", INVALID_TURNS)
def test_invalid_turn_json_is_422_and_does_not_mutate(pair, body):
    pair.create(seed=7).join()
    before = pair.snapshot()
    pair.check("POST", "/turn", json=body, status=422, validation=True)
    assert pair.snapshot() == before


@pytest.mark.parametrize("path", ["/games", "turn"])
@pytest.mark.parametrize("raw", [b'{"broken":', b"{", b"not json"])
def test_malformed_json_uses_the_same_error_envelope(pair, path, raw):
    pair.create(seed=7).join()
    kwargs = {"content": raw, "headers": {"Content-Type": "application/json"}}
    if path == "/games":
        assert_response(pair.python.http.post(path, **kwargs), pair.rust.http.post(path, **kwargs),
                        422, validation=True)
    else:
        kwargs["headers"].update(pair.python.headers())
        reference = pair.python.request("POST", "/turn", **kwargs)
        kwargs["headers"].update(pair.rust.headers())
        actual = pair.rust.request("POST", "/turn", **kwargs)
        assert_response(reference, actual, 422, validation=True)


@pytest.mark.parametrize("params", [
    {"since": "-1"}, {"since": "nope"}, {"since": "1.5"},
    {"timeout": "0"}, {"timeout": "61"}, {"timeout": "nope"}, {"timeout": "1.5"},
])
def test_invalid_observe_query_is_422_invalid_request(pair, params):
    pair.create(seed=7).join()
    pair.check("GET", params=params, status=422, validation=True)


def test_lobby_authentication_and_error_precedence_match_python(pair):
    pair.create(seed=7)
    for suffix in ("", "/legal-moves", "/log", "/events"):
        for headers in ({}, {"Authorization": "Basic bad"}, {"Authorization": "Bearer wrong"}):
            response, _ = pair.check("GET", suffix, headers=headers, status=401)
            assert response.json()["error"] == "UNAUTHORIZED"
    for suffix in ("", "/legal-moves", "/log", "/events"):
        assert_response(pair.python.http.get(f"/games/no_such_game{suffix}"),
                        pair.rust.http.get(f"/games/no_such_game{suffix}"), 404)
    response, _ = pair.check("POST", "/turn", status=403, json={"version": 0})
    assert response.json()["error"] == "GAME_NOT_STARTED"
    pair.check("GET", "/legal-moves", status=403)
    pair.join()
    response, _ = pair.check("POST", "/join", seat=None, status=409)
    assert response.json()["error"] == "GAME_FULL"
    for endpoint in (pair.python, pair.rust):
        other = Endpoint(endpoint.http).create(seed=8)
        for suffix in ("", "/legal-moves", "/log", "/events"):
            response = endpoint.request("GET", suffix, headers=other.headers())
            assert response.status_code == 401 and response.json()["error"] == "UNAUTHORIZED"
        assert endpoint.request("GET", headers={"Authorization": f"bEaReR  {endpoint.tokens[0]}"}).status_code == 200
    view = pair.view(1)
    response, _ = pair.check("POST", "/turn", 1, status=403,
                             json=turn_body(view, version=-100))
    assert response.json()["error"] == "NOT_YOUR_TURN"


def test_rejected_turns_are_atomic_and_failed_uuid_can_be_reused(pair):
    pair.create(seed=7).join()
    view = pair.view()
    tid = str(uuid4())
    cases = [
        (turn_body(view, version=0), 409, "STALE_VERSION"),
        ({"version": view["version"]}, 422, "PLAY_REQUIRED"),
        (turn_body(view, play={"card": view["unseen_cards"][0], "stone": 0}), 422, "CARD_NOT_IN_HAND"),
        (turn_body(view, claims=[1]), 422, "CLAIM_INCOMPLETE"),
    ]
    before = pair.snapshot()
    for body, status, code in cases:
        response, _ = pair.check("POST", "/turn", status=status, json={**body, "turn_id": tid})
        assert response.json()["error"] == code
        assert pair.snapshot() == before
    pair.check("POST", "/turn", json=turn_body(view, turn_id=tid))
    v1 = pair.view(1)
    response, _ = pair.check("POST", "/turn", 1, status=409, json=turn_body(v1, version=1))
    assert response.json()["error"] == "STALE_VERSION"


def test_uuid_retry_returns_current_view_and_is_scoped_to_the_seat(pair):
    pair.create(seed=7).join()
    tid = str(uuid4())
    body = turn_body(pair.view(), turn_id=tid)
    pair.check("POST", "/turn", json=body)
    pair.check("POST", "/turn", 1, json=turn_body(pair.view(1), turn_id=tid))
    current = pair.snapshot()
    retry = {**body, "turn_id": tid.upper(), "version": -123, "claims": [8]}
    response, _ = pair.check("POST", "/turn", json=retry)
    assert response.json() == current[0]
    assert pair.snapshot() == current


def test_full_stone_and_unprovable_claim_rejections_are_atomic(pair):
    pair.create(seed=7).join()
    for _ in range(3):
        pair.check("POST", "/turn", json=turn_body(pair.view(), stone=0))
        pair.check("POST", "/turn", 1, json=turn_body(pair.view(1), stone=1))
    before = pair.snapshot()
    response, _ = pair.check("POST", "/turn", status=422, json=turn_body(before[0], stone=0))
    assert response.json()["error"] == "STONE_FULL"
    assert pair.snapshot() == before
    assert 0 not in Bot.provable_stones(GameView.model_validate(before[0]), None), "choose an unprovable fixture"
    response, _ = pair.check("POST", "/turn", status=422,
                             json=turn_body(before[0], stone=2, claims=[0]))
    assert response.json()["error"] == "CLAIM_NOT_PROVABLE"
    assert pair.snapshot() == before


def test_duplicate_claims_commit_once_and_claimed_stones_reject_atomically(pair):
    pair.create(seed=4, auto_claim=False).join()
    bots = [Bot(SchottenClient(http=pair.python.http), rng=random.Random(seat)) for seat in (0, 1)]
    for _ in range(100):
        views = pair.snapshot()
        assert views[0]["phase"] != "game_over", "fixture ended without an explicit claim"
        seat = views[0]["turn"]
        body = bot_body(bots[seat], views[seat])
        claims = body["claims"]
        if claims:
            body["claims"] = claims + claims
        pair.check("POST", "/turn", seat, json=body)
        if claims:
            log = pair.check("GET", "/log")[0].json()
            assert log["entries"][-1]["claims"] == claims
            break
    else:
        pytest.fail("fixture did not produce a claim")
    before = pair.snapshot()
    assert before[0]["status"] == "in_progress"
    seat = before[0]["turn"]
    view = before[seat]
    response, _ = pair.check("POST", "/turn", seat, status=422,
                             json=turn_body(view, stone=claims[0]))
    assert response.json()["error"] == "STONE_ALREADY_CLAIMED"
    options = Bot.options(GameView.model_validate(view))
    assert options
    response, _ = pair.check("POST", "/turn", seat, status=422, json={
        "version": view["version"], "play": options[0].model_dump(mode="json"), "claims": claims,
    })
    assert response.json()["error"] == "STONE_ALREADY_CLAIMED"
    assert pair.snapshot() == before


@pytest.mark.parametrize("same_uuid", [True, False])
def test_concurrent_duplicate_turns_commit_exactly_once(endpoint, same_uuid):
    endpoint.create(seed=7).join()
    for round_number in range(4):
        seat = round_number % 2
        before = endpoint.view(seat)
        tid = str(uuid4())
        barrier = threading.Barrier(8)

        def send(index):
            body = turn_body(before, stone=round_number // 2,
                             turn_id=tid if same_uuid else str(uuid4()))
            barrier.wait(timeout=4)
            return endpoint.request("POST", "/turn", seat, json=body)

        with ThreadPoolExecutor(8) as pool:
            futures = [pool.submit(send, index) for index in range(8)]
            replies = [future.result(timeout=8) for future in futures]
        successful = [reply for reply in replies if reply.status_code == 200]
        assert len(successful) == (8 if same_uuid else 1), [r.text for r in replies]
        for reply in replies:
            if reply.status_code == 200:
                assert reply.json()["version"] == before["version"] + 1
            else:
                assert reply.status_code == 403 and reply.json()["error"] == "NOT_YOUR_TURN"
        assert endpoint.view()["version"] == before["version"] + 1
        log = endpoint.request("GET", "/log").json()
        assert len(log["entries"]) == round_number + 1


def test_longpoll_204_has_empty_body_and_stale_since_returns_view(endpoint):
    endpoint.create(seed=7).join()
    version = endpoint.view()["version"]
    started = time.monotonic()
    response = endpoint.request("GET", params={"since": version, "timeout": 1})
    assert response.status_code == 204 and response.content == b""
    assert 0.8 <= time.monotonic() - started < 4
    for since in (0, version + 10):
        started = time.monotonic()
        response = endpoint.request("GET", params={"since": since, "timeout": 3})
        assert response.status_code == 200 and response.json()["version"] == version
        assert time.monotonic() - started < 2


@pytest.mark.parametrize("change", ["join", "move"])
def test_longpoll_wakes_on_join_or_move_without_blocking_other_games(endpoint, change):
    endpoint.create(seed=7)
    if change == "move":
        endpoint.join()
    before = endpoint.view()
    entered = threading.Event()

    def poll():
        entered.set()
        return endpoint.request("GET", params={"since": before["version"], "timeout": 4})

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(poll)
        assert entered.wait(2)
        time.sleep(0.15)
        assert not future.done(), "long-poll returned before a change"
        unrelated = Endpoint(endpoint.http).create(seed=123)
        assert unrelated.view()["version"] == 0
        if change == "join":
            endpoint.join()
        else:
            assert endpoint.request("POST", "/turn", json=turn_body(before)).status_code == 200
        response = future.result(timeout=2)
    assert response.status_code == 200
    assert response.json()["version"] == before["version"] + 1
    assert response.json()["status"] == "in_progress"


def test_sse_replay_preserves_shared_ids_filters_versions_and_closes(pair):
    pair.create(seed=4, auto_claim=True).join()
    finish_game(pair, 4)
    expected = terminal_events(pair.python)
    assert expected[:2] == [
        {"version": 1, "type": "player_joined", "seat": 1},
        {"version": 1, "type": "game_started"},
    ]
    assert expected[-1]["type"] == "game_finished"
    assert any(event["type"] == "stone_claimed" for event in expected)
    assert terminal_events(pair.rust) == expected
    latest = expected[-1]["version"]
    for last in (1, latest // 2, latest, latest + 1, -1, "invalid"):
        reference = terminal_events(pair.python, last)
        actual = terminal_events(pair.rust, last)
        assert actual == reference
        cursor = int(last) if last != "invalid" else 0
        assert reference == [event for event in expected if event["version"] > cursor]


def test_live_sse_wakes_on_join_and_move(rust):
    endpoint = Endpoint(rust).create(seed=7)
    opened = threading.Event()

    def receive():
        with rust.stream("GET", f"/games/{endpoint.game_id}/events", headers=endpoint.headers(),
                         timeout=httpx.Timeout(8, read=3)) as response:
            opened.set()
            events = []
            for event in sse_frames(response):
                events.append(event)
                if len(events) == 4:
                    return events
        pytest.fail("live SSE closed before the move")

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(receive)
        assert opened.wait(3), "SSE headers were not flushed"
        endpoint.join()
        body = turn_body(endpoint.view())
        assert endpoint.request("POST", "/turn", json=body).status_code == 200
        events = future.result(timeout=5)
    assert events == [
        {"version": 1, "type": "player_joined", "seat": 1},
        {"version": 1, "type": "game_started"},
        {"version": 2, "type": "card_played", "seat": 0, **body["play"]},
        {"version": 2, "type": "turn_passed", "seat": 0},
    ]


@pytest.mark.parametrize("trigger", ["view", "turn", "legal-moves", "log", "events", "longpoll"])
def test_deadline_reset_lazy_forfeit_reveal_and_retry_after_timeout(pair, trigger):
    pair.create(seed=7, turn_timeout_seconds=5)
    assert pair.view()["turn_deadline"] is None
    before_join = datetime.now(timezone.utc)
    pair.join()
    after_join = datetime.now(timezone.utc)
    for endpoint in (pair.python, pair.rust):
        deadline = datetime.fromisoformat(endpoint.view()["turn_deadline"].replace("Z", "+00:00"))
        assert before_join + timedelta(seconds=5, milliseconds=-1) <= deadline <= after_join + timedelta(seconds=5)
    view = pair.view()
    body = turn_body(view, turn_id=str(uuid4()))
    before_move = datetime.now(timezone.utc)
    pair.check("POST", "/turn", json=body)
    after_move = datetime.now(timezone.utc)
    deadlines = []
    for endpoint in (pair.python, pair.rust):
        deadline = datetime.fromisoformat(endpoint.view()["turn_deadline"].replace("Z", "+00:00"))
        assert before_move + timedelta(seconds=5, milliseconds=-1) <= deadline <= after_move + timedelta(seconds=5)
        deadlines.append(deadline)
    version = pair.view()["version"]
    if trigger == "longpoll":
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(endpoint.request, "GET", params={"since": version, "timeout": 8})
                       for endpoint in (pair.python, pair.rust)]
            responses = [future.result(timeout=9) for future in futures]
        assert_response(*responses, 200)
        assert responses[0].json()["status"] == "forfeit"
    else:
        time.sleep(max(0, (max(deadlines) - datetime.now(timezone.utc)).total_seconds()) + 0.1)
        if trigger == "events":
            assert terminal_events(pair.rust) == terminal_events(pair.python)
        elif trigger == "turn":
            response, _ = pair.check("POST", "/turn", 1, status=403,
                                     json={"version": version, "turn_id": str(uuid4())})
            assert response.json()["error"] == "GAME_OVER"
        else:
            suffix = "" if trigger == "view" else f"/{trigger}"
            pair.check("GET", suffix, status=403 if trigger == "legal-moves" else 200)
    final = pair.snapshot()
    assert final[0]["status"] == "forfeit" and final[0]["winner"] == "you"
    assert final[1]["winner"] == "them" and final[0]["win_reason"] == "forfeit"
    assert final[0]["version"] == version + 1 and final[0]["turn_deadline"] is None
    assert final[0]["turn"] is None and final[0]["phase"] == "game_over"
    log = pair.check("GET", "/log")[0].json()
    assert len(log["entries"]) == 1 and log["revealed"] is not None
    response, _ = pair.check("POST", "/turn", json=body)
    assert normalized(response.json()) == normalized(final[0])
    events = terminal_events(pair.python)
    assert terminal_events(pair.rust) == events
    assert events[-1] == {"version": version + 1, "type": "game_finished",
                          "winner_seat": 0, "win_reason": "forfeit"}
