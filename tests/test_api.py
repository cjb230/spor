import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from schotten.app import create_app


@pytest.fixture
def client():
    # The `with` block keeps one event loop alive, which long-polling relies on.
    with TestClient(create_app()) as c:
        yield c


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def start(client, **options):
    a = client.post("/games", json={"seed": 7, **options}).json()
    b = client.post(f"/games/{a['game_id']}/join").json()
    return a["game_id"], a["token"], b["token"]


def view(client, gid, token):
    return client.get(f"/games/{gid}", headers=auth(token)).json()


def move(client, gid, token, v, stone=0, **extra):
    card = v["hand"][0]
    body = {"version": v["version"], "play": {"card": card, "stone": stone}, **extra}
    return client.post(f"/games/{gid}/turn", headers=auth(token), json=body)


def test_full_turn_cycle(client):
    gid, t0, t1 = start(client)
    v0, v1 = view(client, gid, t0), view(client, gid, t1)
    assert (v0["you"], v1["you"]) == (0, 1)
    assert v0["phase"] == "your_turn" and v1["phase"] == "opponent_turn"

    r = move(client, gid, t0, v0)
    assert r.status_code == 200
    after = r.json()
    assert len(after["hand"]) == 6 and after["phase"] == "opponent_turn"
    assert len(after["stones"][0]["yours"]) == 1

    theirs = view(client, gid, t1)
    assert theirs["phase"] == "your_turn"
    assert len(theirs["stones"][0]["theirs"]) == 1  # the opponent's play is public


def test_error_shapes(client):
    gid, t0, _ = start(client)
    r = client.get(f"/games/{gid}")
    assert r.status_code == 401 and r.json()["error"] == "UNAUTHORIZED"
    r = client.get(f"/games/{gid}", headers=auth("wrong"))
    assert r.status_code == 401
    r = client.get("/games/nope", headers=auth(t0))
    assert r.status_code == 404 and r.json()["error"] == "NOT_FOUND"

    v = view(client, gid, t0)
    r = move(client, gid, t0, v, stone=12)
    assert r.status_code == 422 and r.json()["error"] == "INVALID_REQUEST"

    stranger = v["unseen_cards"][0]
    r = client.post(
        f"/games/{gid}/turn",
        headers=auth(t0),
        json={"version": v["version"], "play": {"card": stranger, "stone": 0}},
    )
    assert r.status_code == 422 and r.json()["error"] == "CARD_NOT_IN_HAND"


def test_turn_order_and_stale_versions(client):
    gid, t0, t1 = start(client)
    v0, v1 = view(client, gid, t0), view(client, gid, t1)
    r = move(client, gid, t1, v1)
    assert r.status_code == 403 and r.json()["error"] == "NOT_YOUR_TURN"

    move(client, gid, t0, v0)
    v1 = view(client, gid, t1)
    stale = dict(v1, version=v1["version"] - 1)
    r = move(client, gid, t1, stale)
    assert r.status_code == 409 and r.json()["error"] == "STALE_VERSION"
    assert move(client, gid, t1, v1).status_code == 200


def test_idempotent_retry(client):
    gid, t0, _ = start(client)
    v0 = view(client, gid, t0)
    tid = str(uuid4())
    first = move(client, gid, t0, v0, turn_id=tid)
    retry = move(client, gid, t0, v0, turn_id=tid)
    assert first.status_code == 200 and retry.status_code == 200
    assert retry.json()["version"] == first.json()["version"]


def test_third_player_cannot_join(client):
    gid, _, _ = start(client)
    r = client.post(f"/games/{gid}/join")
    assert r.status_code == 409 and r.json()["error"] == "GAME_FULL"


def test_legal_moves(client):
    gid, t0, t1 = start(client)
    r = client.get(f"/games/{gid}/legal-moves", headers=auth(t0))
    body = r.json()
    assert r.status_code == 200
    assert len(body["plays"]) == 6 and all(len(p["stones"]) == 9 for p in body["plays"])
    assert body["claimable"] == []
    r = client.get(f"/games/{gid}/legal-moves", headers=auth(t1))
    assert r.status_code == 403 and r.json()["error"] == "NOT_YOUR_TURN"


def test_log_hides_hands_until_the_game_ends(client):
    gid, t0, _ = start(client)
    move(client, gid, t0, view(client, gid, t0))
    log = client.get(f"/games/{gid}/log", headers=auth(t0)).json()
    assert len(log["entries"]) == 1 and log["revealed"] is None


def test_long_poll_times_out_with_204(client):
    gid, t0, _ = start(client)
    v = view(client, gid, t0)["version"]
    r = client.get(f"/games/{gid}?since={v}&timeout=1", headers=auth(t0))
    assert r.status_code == 204


def test_long_poll_returns_immediately_when_already_stale(client):
    gid, t0, _ = start(client)
    r = client.get(f"/games/{gid}?since=0&timeout=1", headers=auth(t0))
    assert r.status_code == 200


def test_long_poll_wakes_on_opponent_move(client):
    gid, t0, t1 = start(client)
    v0 = view(client, gid, t0)
    since = view(client, gid, t1)["version"]
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(
            lambda: client.get(f"/games/{gid}?since={since}&timeout=10", headers=auth(t1))
        )
        time.sleep(0.5)
        assert move(client, gid, t0, v0).status_code == 200
        resp = waiting.result(timeout=5)
    assert resp.status_code == 200
    assert resp.json()["version"] > since and resp.json()["phase"] == "your_turn"
