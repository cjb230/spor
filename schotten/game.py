"""Authoritative game state machine.

Pure and synchronous: no I/O and no clock of its own (callers pass `now`).
Every transition validates against copies first and commits only if the whole
request is legal, so a rejected turn changes nothing.
"""
import random
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from .cards import FULL_DECK, Card, sort_key
from .errors import GameError
from .rules import Formation, check_win, claim_is_provable, evaluate, unseen_pool
from .schemas import (
    CreateGameRequest,
    Event,
    GameLog,
    GameView,
    LegalMoves,
    LogEntry,
    Play,
    PlayOptions,
    Revealed,
    Stone,
    TurnRequest,
)

HAND_SIZE = 6
STONES = 9


@dataclass
class StoneState:
    cards: list = field(default_factory=lambda: [[], []])  # cards[seat] -> list[Card]
    claimed_by: Optional[int] = None
    first_to_complete: Optional[int] = None

    def copy(self) -> "StoneState":
        return StoneState(
            [list(self.cards[0]), list(self.cards[1])], self.claimed_by, self.first_to_complete
        )


def table_cards(stones) -> list:
    return [c for s in stones for side in s.cards for c in side]


def open_stones(stones, seat: int) -> list:
    """Stones this seat may still play on."""
    return [i for i, s in enumerate(stones) if s.claimed_by is None and len(s.cards[seat]) < 3]


def has_legal_play(stones, hand, seat: int) -> bool:
    return bool(hand) and bool(open_stones(stones, seat))


def provable(stones, idx: int, who: int, pool) -> bool:
    s = stones[idx]
    return (
        s.claimed_by is None
        and len(s.cards[who]) == 3
        and claim_is_provable(s.cards[who], s.cards[1 - who], pool, s.first_to_complete == who)
    )


class Game:
    def __init__(self, game_id: str, options: CreateGameRequest, now: datetime):
        self.id = game_id
        self.options = options
        self.created_at = now

        rng = random.Random(options.seed)
        deck = list(FULL_DECK)
        rng.shuffle(deck)
        self.hands: list = [deck[:HAND_SIZE], deck[HAND_SIZE : 2 * HAND_SIZE]]
        self.initial_hands = [list(h) for h in self.hands]
        self.deck: list = deck[2 * HAND_SIZE :]  # drawn with pop() from the end
        self.draw_order = list(reversed(self.deck))

        self.stones = [StoneState() for _ in range(STONES)]
        self.tokens: list = [secrets.token_urlsafe(24), None]
        self.version = 0
        self.status = "waiting_for_opponent"
        self.turn: Optional[int] = None
        self.deadline: Optional[datetime] = None
        self.winner: Optional[int] = None
        self.win_reason: Optional[str] = None
        self.events: list = []
        self.log: list = []
        self._seen_turns: set = set()

    # ---- small helpers -------------------------------------------------

    @property
    def over(self) -> bool:
        return self.status in ("finished", "forfeit")

    def _restart_clock(self, now: datetime) -> None:
        secs = self.options.turn_timeout_seconds
        self.deadline = now + timedelta(seconds=secs) if secs else None

    def _commit(self, pending: list) -> None:
        """Bump the version once and publish this transition's events under it."""
        self.version += 1
        for type_, fields in pending:
            self.events.append(Event(version=self.version, type=type_, **fields))

    def _finish(self, winner: Optional[int], reason: str, status: str = "finished") -> tuple:
        self.status = status
        self.winner = winner
        self.win_reason = reason
        self.turn = None
        self.deadline = None
        fields = {"win_reason": reason}
        if winner is not None:
            fields["winner_seat"] = winner
        return ("game_finished", fields)

    def _claim_event(self, who: int, idx: int) -> tuple:
        formation = Formation(evaluate(self.stones[idx].cards[who])[0]).label
        return ("stone_claimed", {"seat": who, "stone": idx, "formation": formation})

    def _require_turn(self, seat: int) -> None:
        if self.status == "waiting_for_opponent":
            raise GameError("GAME_NOT_STARTED", "waiting for a second player")
        if self.over:
            raise GameError("GAME_OVER", "the game has finished")
        if self.turn != seat:
            raise GameError("NOT_YOUR_TURN", "it is the other player's turn")

    # ---- lobby / clock -------------------------------------------------

    def join(self, now: datetime) -> str:
        if self.tokens[1] is not None:
            raise GameError("GAME_FULL", "both seats are taken")
        self.tokens[1] = secrets.token_urlsafe(24)
        self.status = "in_progress"
        self.turn = 0
        self._restart_clock(now)
        self._commit([("player_joined", {"seat": 1}), ("game_started", {})])
        return self.tokens[1]

    def check_timeout(self, now: datetime) -> bool:
        """Forfeit the player on turn if their clock ran out. Returns True if it fired."""
        if self.status != "in_progress" or self.deadline is None or now < self.deadline:
            return False
        self._commit([self._finish(1 - self.turn, "forfeit", status="forfeit")])
        return True

    # ---- the turn ------------------------------------------------------

    def play_turn(self, seat: int, req: TurnRequest, now: datetime) -> None:
        self.check_timeout(now)
        if req.turn_id is not None and (seat, req.turn_id) in self._seen_turns:
            return  # retry of a turn that already succeeded
        self._require_turn(seat)
        if req.version != self.version:
            raise GameError(
                "STALE_VERSION", f"game is at version {self.version}, request used {req.version}"
            )

        hand = list(self.hands[seat])
        stones = [s.copy() for s in self.stones]
        play = req.play

        # 1. validate and place the card (on copies)
        if play is None:
            if has_legal_play(stones, hand, seat):
                raise GameError("PLAY_REQUIRED", "you have a legal play, so you must make one")
        else:
            if play.card not in hand:
                raise GameError("CARD_NOT_IN_HAND", "that card is not in your hand")
            target = stones[play.stone]
            if target.claimed_by is not None:
                raise GameError("STONE_ALREADY_CLAIMED", f"stone {play.stone} is already claimed")
            if len(target.cards[seat]) >= 3:
                raise GameError("STONE_FULL", f"you already have three cards on stone {play.stone}")
            hand.remove(play.card)
            target.cards[seat].append(play.card)
            if len(target.cards[seat]) == 3 and target.first_to_complete is None:
                target.first_to_complete = seat

        # 2. explicit claims. Claiming never changes the pool (cards stay on the table).
        pool = unseen_pool(table_cards(stones), hand)
        claimed_now: list = []  # (who, stone index)
        for idx in dict.fromkeys(req.claims):
            stone = stones[idx]
            if stone.claimed_by is not None:
                raise GameError("STONE_ALREADY_CLAIMED", f"stone {idx} is already claimed")
            if len(stone.cards[seat]) < 3:
                raise GameError("CLAIM_INCOMPLETE", f"you do not have three cards on stone {idx}")
            if not provable(stones, idx, seat, pool):
                raise GameError("CLAIM_NOT_PROVABLE", f"stone {idx} could still be lost")
            stone.claimed_by = seat
            claimed_now.append((seat, idx))
        explicit = [idx for _, idx in claimed_now]

        # 3. optional server-side auto-claiming, for both players
        if self.options.auto_claim:
            for who in (seat, 1 - seat):
                who_pool = pool if who == seat else unseen_pool(table_cards(stones), self.hands[who])
                for idx in range(STONES):
                    if provable(stones, idx, who, who_pool):
                        stones[idx].claimed_by = who
                        claimed_now.append((who, idx))

        # ---- everything is legal: commit ----
        self.hands[seat] = hand
        self.stones = stones
        pending: list = []
        if play is not None:
            pending.append(("card_played", {"seat": seat, "card": play.card, "stone": play.stone}))
        for who, idx in claimed_now:
            pending.append(self._claim_event(who, idx))

        result = check_win([s.claimed_by for s in stones], order=(seat, 1 - seat))
        if result is not None:
            pending.append(self._finish(*result))
        else:
            if play is not None and self.deck:
                hand.append(self.deck.pop())
            self.turn = 1 - seat
            self._restart_clock(now)
            pending.append(("turn_passed", {"seat": seat}))
            if not (
                has_legal_play(self.stones, self.hands[0], 0)
                or has_legal_play(self.stones, self.hands[1], 1)
            ):
                self._resolve_stuck(pending, seat)

        self._commit(pending)
        self.log.append(LogEntry(version=self.version, seat=seat, play=play, claims=explicit))
        if req.turn_id is not None:
            self._seen_turns.add((seat, req.turn_id))

    def _resolve_stuck(self, pending: list, mover: int) -> None:
        """Neither player can play: settle every remaining stone and end the game."""
        for who in (mover, 1 - mover):
            pool = unseen_pool(table_cards(self.stones), self.hands[who])
            for idx in range(STONES):
                if provable(self.stones, idx, who, pool):
                    self.stones[idx].claimed_by = who
                    pending.append(self._claim_event(who, idx))
        claimed = [s.claimed_by for s in self.stones]
        result = check_win(claimed, order=(mover, 1 - mover))
        if result is not None:
            winner, reason = result
        else:
            winner = 0 if claimed.count(0) > claimed.count(1) else 1 if claimed.count(1) > claimed.count(0) else None
            reason = "stalemate"
        pending.append(self._finish(winner, reason))

    # ---- read side -----------------------------------------------------

    def view(self, seat: int) -> GameView:
        def side(owner: Optional[int]):
            return None if owner is None else ("you" if owner == seat else "them")

        if self.status == "waiting_for_opponent":
            phase = "waiting"
        elif self.over:
            phase = "game_over"
        else:
            phase = "your_turn" if self.turn == seat else "opponent_turn"

        hand = self.hands[seat]
        return GameView(
            game_id=self.id,
            version=self.version,
            status=self.status,
            you=seat,
            turn=self.turn,
            phase=phase,
            hand=sorted(hand, key=sort_key),
            opponent_hand_count=len(self.hands[1 - seat]),
            deck_count=len(self.deck),
            stones=[
                Stone(
                    index=i,
                    yours=list(s.cards[seat]),
                    theirs=list(s.cards[1 - seat]),
                    claimed_by=side(s.claimed_by),
                    first_to_complete=side(s.first_to_complete),
                )
                for i, s in enumerate(self.stones)
            ],
            unseen_cards=unseen_pool(table_cards(self.stones), hand),
            turn_deadline=self.deadline,
            winner=side(self.winner),
            win_reason=self.win_reason,
        )

    def legal_moves(self, seat: int) -> LegalMoves:
        self._require_turn(seat)
        hand = self.hands[seat]
        targets = open_stones(self.stones, seat)
        plays = (
            [PlayOptions(card=c, stones=targets) for c in sorted(set(hand), key=sort_key)]
            if targets
            else []
        )
        pool = unseen_pool(table_cards(self.stones), hand)
        claimable = [i for i in range(STONES) if provable(self.stones, i, seat, pool)]
        return LegalMoves(version=self.version, plays=plays, claimable=claimable)

    def events_after(self, version: int) -> list:
        return [e for e in self.events if e.version > version]

    def log_view(self) -> GameLog:
        revealed = (
            Revealed(initial_hands=self.initial_hands, draw_order=self.draw_order)
            if self.over
            else None
        )
        return GameLog(
            game_id=self.id, status=self.status, entries=list(self.log), revealed=revealed
        )
