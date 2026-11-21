"""Pure rules: formation strength, claim proofs and win conditions."""
from enum import IntEnum
from itertools import combinations
from typing import Iterable, Optional, Sequence

from .cards import FULL_DECK, Card

Strength = tuple  # (formation, total), compared lexicographically


class Formation(IntEnum):
    SUM = 0
    RUN = 1
    COLOR = 2
    THREE_OF_A_KIND = 3
    COLOR_RUN = 4

    @property
    def label(self) -> str:
        return self.name.lower()


def evaluate(cards: Sequence[Card]) -> Strength:
    """Strength of a completed three-card formation. Higher tuple wins."""
    if len(cards) != 3:
        raise ValueError("a formation has exactly three cards")
    values = sorted(c.value for c in cards)
    same_color = len({c.color for c in cards}) == 1
    is_run = values[1] == values[0] + 1 and values[2] == values[1] + 1
    same_value = values[0] == values[2]
    if same_color and is_run:
        formation = Formation.COLOR_RUN
    elif same_value:
        formation = Formation.THREE_OF_A_KIND
    elif same_color:
        formation = Formation.COLOR
    elif is_run:
        formation = Formation.RUN
    else:
        formation = Formation.SUM
    return (int(formation), sum(values))


def unseen_pool(table: Iterable[Card], hand: Iterable[Card]) -> list:
    """Cards a player cannot see: the full deck minus the table minus their own hand.

    These are exactly the cards the opponent might hold or the deck might yield.
    """
    seen = set(table) | set(hand)
    return [c for c in FULL_DECK if c not in seen]


def claim_is_provable(
    mine: Sequence[Card],
    theirs: Sequence[Card],
    pool: Sequence[Card],
    tiebreak_mine: bool,
) -> bool:
    """Can the owner of `mine` (three cards) prove the opponent cannot win this stone?

    `theirs` holds 0-3 cards. `pool` is every card the opponent could still use.
    An exact tie goes to whoever completed first (`tiebreak_mine`). When the
    opponent is incomplete, the claimer completed first, so only a strictly
    stronger formation can block the claim.
    """
    my_strength = evaluate(mine)
    needed = 3 - len(theirs)
    if needed == 0:
        their_strength = evaluate(theirs)
        return my_strength > their_strength or (my_strength == their_strength and tiebreak_mine)
    if len(pool) < needed:
        return True  # the opponent can never complete this stone
    for extra in combinations(pool, needed):
        their_strength = evaluate((*theirs, *extra))
        if their_strength > my_strength or (their_strength == my_strength and not tiebreak_mine):
            return False
    return True


def check_win(claimed: Sequence[Optional[int]], order=(0, 1)) -> Optional[tuple]:
    """Return (seat, reason) if someone has won, else None. `order` breaks simultaneous wins."""
    for seat in order:
        owned = [owner == seat for owner in claimed]
        if sum(owned) >= 5:
            return seat, "five_stones"
        if any(owned[i] and owned[i + 1] and owned[i + 2] for i in range(len(owned) - 2)):
            return seat, "three_adjacent"
    return None
