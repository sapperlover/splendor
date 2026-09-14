from __future__ import annotations

import pytest

from splendor.model import DecisionPhase
from splendor.result import TerminalResult
from splendor._state import PlayerState
from splendor.catalog import Catalog
from splendor.model import GEM_COLORS, TokenVector
from splendor.rules import determine_winners, eligible_noble_ids
from splendor.state_machine import allowed_next_phases, assert_phase_transition


def test_all_terminal_tiebreak_branches() -> None:
    assert determine_winners((16, 15), (8, 7)) == (0,)
    assert determine_winners((15, 15), (8, 7)) == (1,)
    assert determine_winners((15, 15), (7, 7)) == (0, 1)
    assert TerminalResult((15, 15), (7, 7), (0, 1), (20, 20)).payoff(0) == 0


def test_noble_eligibility_uses_purchased_bonuses_only(catalog: Catalog) -> None:
    target = catalog.noble("N01")
    required_cards: list[str] = []
    for color in GEM_COLORS:
        candidates = [card.card_id for card in catalog.cards if card.bonus is color]
        required_cards.extend(candidates[: target.requirement.for_color(color)])
    rich_in_tokens = PlayerState(
        tokens=TokenVector(white=4, blue=4, green=4, red=4, black=4, gold=5),
        purchased_card_ids=tuple(required_cards),
    )
    assert eligible_noble_ids(rich_in_tokens, (target.noble_id,), catalog) == (target.noble_id,)


def test_visible_phase_topology_allows_automatic_skips() -> None:
    assert DecisionPhase.PAYMENT in allowed_next_phases(DecisionPhase.MAIN_ACTION)
    assert DecisionPhase.MAIN_ACTION in allowed_next_phases(DecisionPhase.RETURN_TOKENS)
    assert DecisionPhase.TERMINAL in allowed_next_phases(DecisionPhase.CHOOSE_NOBLE)
    assert allowed_next_phases(DecisionPhase.TERMINAL) == frozenset()


def test_terminal_phase_cannot_transition() -> None:
    with pytest.raises(ValueError, match="illegal phase transition"):
        assert_phase_transition(DecisionPhase.TERMINAL, DecisionPhase.MAIN_ACTION)
