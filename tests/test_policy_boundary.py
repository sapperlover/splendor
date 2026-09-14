from __future__ import annotations

import inspect
from typing import get_type_hints

import splendor
import pytest
import splendor.actions as action_contracts

from splendor.actions import ReserveDeck, TakeDistinct
from splendor.contracts import DecisionRequest, Policy
from splendor.model import Color, Tier


def test_public_package_does_not_export_complete_state() -> None:
    assert "GameState" not in splendor.__all__
    assert not hasattr(splendor, "GameState")


def test_policy_signature_only_accepts_policy_safe_request() -> None:
    hints = get_type_hints(Policy.choose)
    assert hints["request"] is DecisionRequest
    assert "GameState" not in inspect.getsource(Policy)


def test_forced_skip_is_not_a_policy_action() -> None:
    assert not hasattr(action_contracts, "PassAction")
    assert not hasattr(action_contracts, "ForcedSkip")
    assert "skip" not in inspect.getsource(action_contracts.expected_action_types).lower()


def test_blind_reservation_candidate_cannot_contain_hidden_card_id() -> None:
    action = ReserveDeck(Tier.THREE)
    assert not hasattr(action, "card_id")


def test_distinct_take_has_one_order_independent_action_identity() -> None:
    first = TakeDistinct((Color.RED, Color.WHITE, Color.BLUE))
    second = TakeDistinct((Color.BLUE, Color.RED, Color.WHITE))
    assert first == second
    assert first.colors == (Color.WHITE, Color.BLUE, Color.RED)
    with pytest.raises(ValueError, match="unique non-gold"):
        TakeDistinct((Color.WHITE, Color.GOLD))
