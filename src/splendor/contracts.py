"""Information-safe contracts passed across the engine/policy boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .actions import (
    Action,
    MainAction,
    expected_action_types,
)
from .model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from .result import TerminalResult


@dataclass(frozen=True, slots=True)
class CardView:
    card_id: str
    tier: Tier
    bonus: Color
    points: int
    cost: GemVector


@dataclass(frozen=True, slots=True)
class NobleView:
    noble_id: str
    points: int
    requirement: GemVector


@dataclass(frozen=True, slots=True)
class PublicReservedCard:
    """Opponent-safe reservation information.

    ``card`` is present for a market reservation and absent for a blind deck
    reservation.  A blind reservation still exposes tier and source.
    """

    tier: Tier
    source: ReservationSource
    card: CardView | None


@dataclass(frozen=True, slots=True)
class PlayerPublicState:
    tokens: TokenVector
    bonuses: GemVector
    score: int
    purchased_card_ids: tuple[str, ...]
    noble_ids: tuple[str, ...]
    reserved_cards: tuple[PublicReservedCard, ...]


@dataclass(frozen=True, slots=True)
class PlayerPrivateState:
    """Only the owning player's blindly reserved cards."""

    blind_reserved_cards: tuple[CardView, ...]


@dataclass(frozen=True, slots=True)
class PendingObservation:
    main_action: MainAction
    payment_target: CardView | None = None
    tokens_to_return: int = 0
    eligible_nobles: tuple[NobleView, ...] = ()


@dataclass(frozen=True, slots=True)
class MarketView:
    tier_one: tuple[CardView, ...]
    tier_two: tuple[CardView, ...]
    tier_three: tuple[CardView, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything a policy may know, expressed relative to its own seat."""

    phase: DecisionPhase
    self_is_first_player: bool
    supply: TokenVector
    deck_counts: tuple[int, int, int]
    market: MarketView
    nobles_in_play: tuple[NobleView, ...]
    self_public: PlayerPublicState
    self_private: PlayerPrivateState
    opponent_public: PlayerPublicState
    public_purchased: tuple[bool, ...]
    own_completed_turns: int
    opponent_completed_turns: int
    consecutive_forced_skips: int
    final_round_triggered: bool
    final_round_triggered_by_self: bool
    pending: PendingObservation | None
    result: TerminalResult | None = None


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """The sole policy input. Legal candidates are produced by the rule engine."""

    observation: Observation
    legal_actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        if self.observation.phase is DecisionPhase.TERMINAL:
            raise ValueError("a terminal state cannot be sent to a policy")
        if not self.legal_actions:
            raise ValueError("a non-terminal policy request must have legal actions")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("legal actions must be deduplicated")
        if not all(isinstance(action, expected_action_types(self.observation.phase)) for action in self.legal_actions):
            raise ValueError("legal action type does not match the decision phase")


class Policy(Protocol):
    def choose(self, request: DecisionRequest) -> Action:
        """Return one exact member of ``request.legal_actions``."""
