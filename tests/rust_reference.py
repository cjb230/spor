"""Utilities for opt-in, real-HTTP Rust/Python REST differential tests.

The Python oracle has its own app and event loop. No server is started here.
Only generated game IDs and non-null turn deadlines are normalized; missing
keys, nulls, ordering, cards, versions, and every other response field survive.
"""

import json
import re
import time

import httpx


def normalized(value):
    if isinstance(value, dict):
        return {
            key: (
                "<game_id>" if key == "game_id"
                else "<turn_deadline>" if key == "turn_deadline" and item is not None
                else normalized(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def assert_response(reference, actual, status=None, validation=False):
    if status is not None:
        assert reference.status_code == status, reference.text
    assert actual.status_code == reference.status_code, actual.text
    if reference.status_code == 204:
        assert reference.content == actual.content == b""
        return
    assert actual.headers["content-type"].startswith("application/json")
    expected, received = reference.json(), actual.json()
    if validation:
        for body in (expected, received):
            assert set(body) == {"error", "detail"}
            assert body["error"] == "INVALID_REQUEST"
            assert isinstance(body["detail"], str) and body["detail"]
    else:
        assert normalized(received) == normalized(expected)


class Endpoint:
    def __init__(self, http):
        self.http = http
        self.game_id = None
        self.tokens = []

    def adopt(self, response, seat):
        assert response.status_code == 201, response.text
        grant = response.json()
        assert set(grant) == {"game_id", "seat", "token"}
        assert grant["seat"] == seat
        assert re.fullmatch(r"g_[0-9a-f]+", grant["game_id"])
        assert isinstance(grant["token"], str) and grant["token"]
        if seat == 0:
            self.game_id = grant["game_id"]
        else:
            assert grant["game_id"] == self.game_id
            assert grant["token"] != self.tokens[0]
        self.tokens.append(grant["token"])

    def create(self, **options):
        self.adopt(self.http.post("/games", json=options), 0)
        return self

    def join(self):
        self.adopt(self.request("POST", "/join", seat=None), 1)

    def headers(self, seat=0):
        return {"Authorization": f"Bearer {self.tokens[seat]}"}

    def request(self, method, suffix="", seat=0, **kwargs):
        kwargs.setdefault("headers", {} if seat is None else self.headers(seat))
        return self.http.request(method, f"/games/{self.game_id}{suffix}", **kwargs)

    def view(self, seat=0):
        response = self.request("GET", seat=seat)
        assert response.status_code == 200, response.text
        return response.json()


class DifferentialGame:
    def __init__(self, python, rust):
        self.python = Endpoint(python)
        self.rust = Endpoint(rust)

    def create(self, **options):
        self.python.create(**options)
        self.rust.create(**options)
        self.view()
        return self

    def join(self):
        self.python.join()
        self.rust.join()
        return self

    def check(self, method, suffix="", seat=0, status=200, validation=False, **kwargs):
        reference = self.python.request(method, suffix, seat, **kwargs)
        actual = self.rust.request(method, suffix, seat, **kwargs)
        assert_response(reference, actual, status, validation)
        return reference, actual

    def view(self, seat=0):
        return self.check("GET", seat=seat)[0].json()

    def snapshot(self):
        views = [self.view(seat) for seat in (0, 1)]
        for seat, view in enumerate(views):
            status = 200 if view["phase"] == "your_turn" else 403
            self.check("GET", "/legal-moves", seat, status=status)
        self.check("GET", "/log", seat=views[0]["version"] % 2)
        return views


def sse_frames(response):
    """Parse complete SSE frames, retaining every event with a shared version.

    Call only with a finite/terminal oracle stream. Real HTTP streams also have
    a read timeout, a total budget, and a frame cap, including keepalives.
    """
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    fields = {}
    started = time.monotonic()
    count = 0
    for line in response.iter_lines():
        assert time.monotonic() - started < 12, "SSE did not close/advance promptly"
        if line == "":
            count += 1
            assert count < 2000, "unbounded SSE frames"
            if fields:
                assert set(fields) == {"id", "data"}, fields
                event = json.loads(fields["data"])
                assert int(fields["id"]) == event["version"]
                yield event
                fields = {}
        elif not line.startswith(":"):
            name, separator, value = line.partition(":")
            assert separator and name not in fields, line
            fields[name] = value.lstrip(" ")
    assert not fields, "SSE ended with an incomplete frame"


def terminal_events(endpoint, last="0"):
    headers = {**endpoint.headers(), "Last-Event-ID": str(last)}
    with endpoint.http.stream(
        "GET", f"/games/{endpoint.game_id}/events", headers=headers,
        timeout=httpx.Timeout(8, read=3),
    ) as response:
        return list(sse_frames(response))
