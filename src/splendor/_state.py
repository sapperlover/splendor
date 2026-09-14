"""Engine-only authoritative state.

Nothing in this module belongs in a policy call.  The future engine owns these
objects and derives public contracts in ``contracts.py`` from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from .actions import MainAction
from .model import DecisionPhase, ReservationSource, Tier, TokenVector
from .result import TerminalResult
from .rng import RngState


@dataclass(frozen=True, slots=True)
class ReservedCard:
    card_id: str
    source: ReservationSource


@dataclass(frozen=True, slots=True)
class PlayerState:
    tokens: TokenVector
    purchased_card_ids: tuple[str, ...] = ()
    reserved_cards: tuple[ReservedCard, ...] = ()
    noble_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MarketState:
    tier_one: tuple[str, ...]
    tier_two: tuple[str, ...]
    tier_three: tuple[str, ...]

    def for_tier(self, tier: Tier) -> tuple[str, ...]:
        return getattr(self, f"tier_{('one', 'two', 'three')[tier - 1]}")

    def all_cards(self) -> tuple[str, ...]:
        return self.tier_one + self.tier_two + self.tier_three

    def replacing_tier(self, tier: Tier, cards: tuple[str, ...]) -> MarketState:
        from dataclasses import replace

        field = ("tier_one", "tier_two", "tier_three")[tier - 1]
        return replace(self, **{field: cards})


@dataclass(frozen=True, slots=True)
class DeckState:
    """Ordered card IDs per tier.  The top card is the final tuple element."""

    tier_one: tuple[str, ...]
    tier_two: tuple[str, ...]
    tier_three: tuple[str, ...]

    def for_tier(self, tier: Tier) -> tuple[str, ...]:
        return getattr(self, f"tier_{('one', 'two', 'three')[tier - 1]}")

    def all_cards(self) -> tuple[str, ...]:
        return self.tier_one + self.tier_two + self.tier_three

    def replacing_tier(self, tier: Tier, cards: tuple[str, ...]) -> DeckState:
        from dataclasses import replace

        field = ("tier_one", "tier_two", "tier_three")[tier - 1]
        return replace(self, **{field: cards})


@dataclass(frozen=True, slots=True)
class PendingTurn:
    """Context retained after a main action until the same turn settles."""

    main_action: MainAction
    tokens_to_return: int = 0
    eligible_noble_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete truth, including secrets. Accessible only to engine/debug replay."""

    supply: TokenVector
    decks: DeckState
    market: MarketState
    players: tuple[PlayerState, PlayerState]
    nobles_in_play: tuple[str, ...]
    nobles_out_of_game: tuple[str, ...]
    first_player: int
    current_player: int
    completed_turns: tuple[int, int]
    consecutive_forced_skips: int
    phase: DecisionPhase
    pending: PendingTurn | None
    final_round_triggered_by: int | None
    result: TerminalResult | None
    rng: RngState
