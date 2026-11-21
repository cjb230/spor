"""Card model shared by the engine and the API."""
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from pydantic import Field


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    PURPLE = "purple"
    BROWN = "brown"


@dataclass(frozen=True)
class Card:
    color: Color
    value: Annotated[int, Field(ge=1, le=9)]


FULL_DECK: tuple = tuple(Card(color, value) for color in Color for value in range(1, 10))
_ORDER = {card: i for i, card in enumerate(FULL_DECK)}


def sort_key(card: Card) -> int:
    """Stable display order: by colour, then value."""
    return _ORDER[card]
