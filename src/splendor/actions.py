from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .model import Color, GEM_COLORS, Tier, TokenVector


@dataclass(frozen=True, slots=True)
class TakeDistinct:
    """Take every listed non-gold color once; order has no meaning."""

    colors: tuple[Color, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.colors) <= 3:
            raise ValueError("a distinct-color take must contain one to three colors")
        if len(set(self.colors)) != len(self.colors) or any(color not in GEM_COLORS for color in self.colors):
            raise ValueError("a distinct-color take requires unique non-gold colors")
        canonical = tuple(color for color in GEM_COLORS if color in self.colors)
        object.__setattr__(self, "colors", canonical)


@dataclass(frozen=True, slots=True)
class TakePair:
    color: Color


@dataclass(frozen=True, slots=True)
class ReserveMarket:
    card_id: str


@dataclass(frozen=True, slots=True)
class ReserveDeck:
    """Blind reservation exposes only the tier, never the future card ID."""

    tier: Tier


@dataclass(frozen=True, slots=True)
class BuyMarket:
    card_id: str


@dataclass(frozen=True, slots=True)
class BuyReserved:
    card_id: str


MainAction: TypeAlias = (
    TakeDistinct | TakePair | ReserveMarket | ReserveDeck | BuyMarket | BuyReserved
)
MAIN_ACTION_TYPES = (TakeDistinct, TakePair, ReserveMarket, ReserveDeck, BuyMarket, BuyReserved)


@dataclass(frozen=True, slots=True)
class Pay:
    """Actual tokens spent; gold allocation with the same result is deduplicated."""

    tokens: TokenVector


@dataclass(frozen=True, slots=True)
class ReturnTokens:
    tokens: TokenVector


@dataclass(frozen=True, slots=True)
class ClaimNoble:
    noble_id: str


Action: TypeAlias = MainAction | Pay | ReturnTokens | ClaimNoble


def expected_action_types(phase: object) -> tuple[type[object], ...]:
    """Structural action/phase mapping; semantic legality remains engine-owned."""

    from .model import DecisionPhase

    mapping: dict[DecisionPhase, tuple[type[object], ...]] = {
        DecisionPhase.MAIN_ACTION: MAIN_ACTION_TYPES,
        DecisionPhase.PAYMENT: (Pay,),
        DecisionPhase.RETURN_TOKENS: (ReturnTokens,),
        DecisionPhase.CHOOSE_NOBLE: (ClaimNoble,),
        DecisionPhase.TERMINAL: (),
    }
    if not isinstance(phase, DecisionPhase):
        return ()
    return mapping[phase]
