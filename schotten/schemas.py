"""Pydantic models for requests, responses and events (mirrors openapi.yaml)."""
from datetime import datetime
from typing import Annotated, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .cards import Card

StoneIndex = Annotated[int, Field(ge=0, le=8)]
Seat = Literal[0, 1]
Side = Literal["you", "them"]
Status = Literal["waiting_for_opponent", "in_progress", "finished", "forfeit"]
Phase = Literal["waiting", "your_turn", "opponent_turn", "game_over"]
WinReason = Literal["five_stones", "three_adjacent", "forfeit", "stalemate"]
FormationName = Literal["color_run", "three_of_a_kind", "color", "run", "sum"]
EventType = Literal[
    "player_joined", "game_started", "card_played", "stone_claimed", "turn_passed", "game_finished"
]


class CreateGameRequest(BaseModel):
    auto_claim: bool = False
    turn_timeout_seconds: Optional[int] = Field(default=None, ge=5)
    seed: Optional[int] = None


class SeatGrant(BaseModel):
    game_id: str
    seat: Seat
    token: str


class Stone(BaseModel):
    index: StoneIndex
    yours: list[Card]
    theirs: list[Card]
    claimed_by: Optional[Side] = None
    first_to_complete: Optional[Side] = None


class GameView(BaseModel):
    game_id: str
    version: int
    status: Status
    you: Seat
    turn: Optional[Seat] = None
    phase: Phase
    hand: list[Card]
    opponent_hand_count: int
    deck_count: int
    stones: list[Stone]
    unseen_cards: list[Card]
    turn_deadline: Optional[datetime] = None
    winner: Optional[Side] = None
    win_reason: Optional[WinReason] = None


class Play(BaseModel):
    card: Card
    stone: StoneIndex


class TurnRequest(BaseModel):
    version: int
    turn_id: Optional[UUID] = None
    play: Optional[Play] = None
    claims: list[StoneIndex] = Field(default_factory=list)


class PlayOptions(BaseModel):
    card: Card
    stones: list[StoneIndex]


class LegalMoves(BaseModel):
    version: int
    plays: list[PlayOptions]
    claimable: list[StoneIndex]


class Event(BaseModel):
    """Public event. `seat` is the acting player; for turn_passed, the player who just finished."""

    version: int
    type: EventType
    seat: Optional[Seat] = None
    card: Optional[Card] = None
    stone: Optional[StoneIndex] = None
    formation: Optional[FormationName] = None
    winner_seat: Optional[Seat] = None
    win_reason: Optional[WinReason] = None


class LogEntry(BaseModel):
    version: int
    seat: Seat
    play: Optional[Play] = None
    claims: list[StoneIndex]


class Revealed(BaseModel):
    initial_hands: list[list[Card]]
    draw_order: list[Card]


class GameLog(BaseModel):
    game_id: str
    status: Status
    entries: list[LogEntry]
    revealed: Optional[Revealed] = None
