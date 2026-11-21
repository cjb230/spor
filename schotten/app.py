"""FastAPI layer. All game logic lives in game.py; this only does transport.

State is held in memory. Swap `games` for a store (and persist the event log)
to survive restarts or run more than one worker.
"""
import asyncio
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .errors import GameError
from .game import Game
from .schemas import (
    CreateGameRequest,
    GameLog,
    GameView,
    LegalMoves,
    SeatGrant,
    TurnRequest,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_app() -> FastAPI:
    app = FastAPI(title="Schotten Totten Game Server", version="0.1.0")
    games: dict = {}
    conds: dict = {}

    # ---- helpers -------------------------------------------------------

    def find_game(game_id: str) -> Game:
        game = games.get(game_id)
        if game is None:
            raise GameError("NOT_FOUND", f"no such game: {game_id}")
        return game

    def seat_of(game: Game, authorization: Optional[str]) -> int:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise GameError("UNAUTHORIZED", "missing bearer token")
        token = authorization[7:].strip()
        for seat, seat_token in enumerate(game.tokens):
            if seat_token and secrets.compare_digest(seat_token, token):
                return seat
        raise GameError("UNAUTHORIZED", "invalid token for this game")

    def cond_for(game: Game) -> asyncio.Condition:
        if game.id not in conds:
            conds[game.id] = asyncio.Condition()
        return conds[game.id]

    async def notify(game: Game) -> None:
        cond = cond_for(game)
        async with cond:
            cond.notify_all()

    async def tick(game: Game) -> None:
        """Apply a lapsed turn clock (lazily) and wake watchers if it fired."""
        if game.check_timeout(utcnow()):
            await notify(game)

    async def wait_change(game: Game, max_wait: float) -> None:
        """Sleep until the game is notified, its turn clock lapses, or max_wait passes."""
        if game.deadline is not None:
            max_wait = min(max_wait, max((game.deadline - utcnow()).total_seconds(), 0.0))
        cond = cond_for(game)
        async with cond:
            try:
                await asyncio.wait_for(cond.wait(), max_wait)
            except asyncio.TimeoutError:
                pass

    # ---- error shape ---------------------------------------------------

    @app.exception_handler(GameError)
    async def on_game_error(request: Request, exc: GameError):
        return JSONResponse(
            status_code=exc.status, content={"error": exc.code, "detail": exc.detail}
        )

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return JSONResponse(
            status_code=422, content={"error": "INVALID_REQUEST", "detail": detail}
        )

    # ---- lobby ---------------------------------------------------------

    @app.post("/games", status_code=201, response_model=SeatGrant, tags=["lobby"])
    async def create_game(body: Optional[CreateGameRequest] = None):
        options = body or CreateGameRequest()
        game_id = "g_" + secrets.token_hex(4)
        game = Game(game_id, options, utcnow())
        games[game_id] = game
        return SeatGrant(game_id=game_id, seat=0, token=game.tokens[0])

    @app.post("/games/{game_id}/join", status_code=201, response_model=SeatGrant, tags=["lobby"])
    async def join_game(game_id: str):
        game = find_game(game_id)
        token = game.join(utcnow())
        await notify(game)
        return SeatGrant(game_id=game.id, seat=1, token=token)

    # ---- observe -------------------------------------------------------

    @app.get("/games/{game_id}", response_model=GameView, tags=["observe"])
    async def read_game(
        game_id: str,
        since: Optional[int] = Query(None, ge=0),
        timeout: int = Query(30, ge=1, le=60),
        authorization: Optional[str] = Header(None),
    ):
        game = find_game(game_id)
        seat = seat_of(game, authorization)
        await tick(game)
        if since is not None and since == game.version and not game.over:
            loop = asyncio.get_running_loop()
            end = loop.time() + timeout
            while True:
                await tick(game)
                if game.version != since or game.over:
                    break
                remaining = end - loop.time()
                if remaining <= 0:
                    return Response(status_code=204)
                await wait_change(game, remaining)
        return game.view(seat)

    @app.get("/games/{game_id}/legal-moves", response_model=LegalMoves, tags=["observe"])
    async def read_legal_moves(game_id: str, authorization: Optional[str] = Header(None)):
        game = find_game(game_id)
        seat = seat_of(game, authorization)
        await tick(game)
        return game.legal_moves(seat)

    @app.get("/games/{game_id}/log", response_model=GameLog, tags=["observe"])
    async def read_log(game_id: str, authorization: Optional[str] = Header(None)):
        game = find_game(game_id)
        seat_of(game, authorization)
        await tick(game)
        return game.log_view()

    @app.get("/games/{game_id}/events", tags=["observe"])
    async def stream_events(
        game_id: str, request: Request, authorization: Optional[str] = Header(None)
    ):
        game = find_game(game_id)
        seat_of(game, authorization)
        try:
            last = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            last = 0

        async def frames():
            nonlocal last
            while True:
                await tick(game)
                fresh = game.events_after(last)
                for event in fresh:
                    last = event.version
                    yield f"id: {event.version}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"
                if game.over:
                    return
                if not fresh:
                    yield ": keepalive\n\n"
                if await request.is_disconnected():
                    return
                await wait_change(game, 15)

        return StreamingResponse(
            frames(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    # ---- play ----------------------------------------------------------

    @app.post("/games/{game_id}/turn", response_model=GameView, tags=["play"])
    async def submit_turn(
        game_id: str, req: TurnRequest, authorization: Optional[str] = Header(None)
    ):
        game = find_game(game_id)
        seat = seat_of(game, authorization)
        try:
            game.play_turn(seat, req, utcnow())
        finally:
            await notify(game)  # a rejected turn may still have lapsed the clock
        return game.view(seat)

    return app


app = create_app()
