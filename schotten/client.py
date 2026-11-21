"""Thin typed wrapper over the REST API.

Works against a real server (`SchottenClient("http://host:8000")`) or any
httpx-compatible client, e.g. FastAPI's TestClient (`SchottenClient(http=tc)`).
"""
from typing import Optional, Sequence
from uuid import UUID

import httpx

from .schemas import GameLog, GameView, LegalMoves, Play, SeatGrant


class ApiError(Exception):
    """A non-2xx response. `code` is the machine-readable error from the server."""

    def __init__(self, status: int, code: str, detail: str = ""):
        self.status, self.code, self.detail = status, code, detail
        super().__init__(f"{status} {code}: {detail}")


class SchottenClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        http: Optional[httpx.Client] = None,
        token: Optional[str] = None,
        game_id: Optional[str] = None,
    ):
        # Long-polls hold the connection for up to 60 s, so the read timeout must exceed that.
        self.http = http or httpx.Client(
            base_url=base_url, timeout=httpx.Timeout(10.0, read=75.0)
        )
        self.token = token
        self.game_id = game_id
        self.seat: Optional[int] = None

    # ---- plumbing ------------------------------------------------------

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        r = self.http.request(method, path, headers=headers, **kw)
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = {}
            raise ApiError(r.status_code, body.get("error", "UNKNOWN"), body.get("detail", ""))
        return r

    def _adopt(self, grant: SeatGrant) -> SeatGrant:
        self.game_id, self.seat, self.token = grant.game_id, grant.seat, grant.token
        return grant

    # ---- lobby ---------------------------------------------------------

    def create_game(
        self,
        auto_claim: bool = False,
        turn_timeout_seconds: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> SeatGrant:
        body = {
            "auto_claim": auto_claim,
            "turn_timeout_seconds": turn_timeout_seconds,
            "seed": seed,
        }
        return self._adopt(SeatGrant(**self._call("POST", "/games", json=body).json()))

    def join(self, game_id: str) -> SeatGrant:
        return self._adopt(SeatGrant(**self._call("POST", f"/games/{game_id}/join").json()))

    # ---- observe -------------------------------------------------------

    def view(self, since: Optional[int] = None, timeout: int = 30) -> Optional[GameView]:
        """Current view. With `since`, long-polls; returns None if nothing changed in time."""
        params = {} if since is None else {"since": since, "timeout": timeout}
        r = self._call("GET", f"/games/{self.game_id}", params=params)
        return None if r.status_code == 204 else GameView(**r.json())

    def legal_moves(self) -> LegalMoves:
        return LegalMoves(**self._call("GET", f"/games/{self.game_id}/legal-moves").json())

    def log(self) -> GameLog:
        return GameLog(**self._call("GET", f"/games/{self.game_id}/log").json())

    # ---- play ----------------------------------------------------------

    def turn(
        self,
        version: int,
        play: Optional[Play] = None,
        claims: Sequence[int] = (),
        turn_id: Optional[UUID] = None,
    ) -> GameView:
        body = {"version": version, "claims": list(claims)}
        if play is not None:
            body["play"] = play.model_dump(mode="json")
        if turn_id is not None:
            body["turn_id"] = str(turn_id)
        return GameView(**self._call("POST", f"/games/{self.game_id}/turn", json=body).json())
