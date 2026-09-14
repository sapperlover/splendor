from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterator


class Color(str, Enum):
    WHITE = "white"
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    BLACK = "black"
    GOLD = "gold"


GEM_COLORS: tuple[Color, ...] = (
    Color.WHITE,
    Color.BLUE,
    Color.GREEN,
    Color.RED,
    Color.BLACK,
)
TOKEN_COLORS: tuple[Color, ...] = GEM_COLORS + (Color.GOLD,)


class Tier(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3


TIERS: tuple[Tier, ...] = (Tier.ONE, Tier.TWO, Tier.THREE)


class DecisionPhase(str, Enum):
    MAIN_ACTION = "main_action"
    PAYMENT = "payment"
    RETURN_TOKENS = "return_tokens"
    CHOOSE_NOBLE = "choose_noble"
    TERMINAL = "terminal"


class ReservationSource(str, Enum):
    MARKET = "market"
    DECK = "deck"


@dataclass(frozen=True, slots=True)
class GemVector:
    white: int = 0
    blue: int = 0
    green: int = 0
    red: int = 0
    black: int = 0

    def __iter__(self) -> Iterator[int]:
        yield self.white
        yield self.blue
        yield self.green
        yield self.red
        yield self.black

    def for_color(self, color: Color) -> int:
        if color is Color.GOLD:
            raise ValueError("gold is not a gem/bonus color")
        return getattr(self, color.value)

    @property
    def total(self) -> int:
        return sum(self)


@dataclass(frozen=True, slots=True)
class TokenVector:
    white: int = 0
    blue: int = 0
    green: int = 0
    red: int = 0
    black: int = 0
    gold: int = 0

    def __iter__(self) -> Iterator[int]:
        yield self.white
        yield self.blue
        yield self.green
        yield self.red
        yield self.black
        yield self.gold

    def for_color(self, color: Color) -> int:
        return getattr(self, color.value)

    @property
    def total(self) -> int:
        return sum(self)

    def plus(self, other: TokenVector) -> TokenVector:
        return TokenVector(*(left + right for left, right in zip(self, other, strict=True)))

    def minus(self, other: TokenVector) -> TokenVector:
        return TokenVector(*(left - right for left, right in zip(self, other, strict=True)))
