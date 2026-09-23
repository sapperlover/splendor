from __future__ import annotations

import inspect

import pytest
import torch

from splendor.actions import ReserveDeck, TakeDistinct
from splendor.actor_critic import ActorCritic
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.model import Tier
from splendor.random_policy import RandomPolicy
from splendor.result import TerminalResult
from splendor.search_policy import (
    PolicyValueSearchPolicy,
    SearchConfig,
    determinize_request,
)


def _zero_model() -> ActorCritic:
    model = ActorCritic().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("top_k", 0),
        ("depth", 0),
        ("determinizations", 0),
        ("leaf_value_clip", -1.0),
    ),
)
def test_search_config_rejects_invalid_values(field: str, value: int | float) -> None:
    values = {
        "top_k": 5,
        "depth": 4,
        "determinizations": 1,
        "leaf_value_clip": 1.0,
    }
    values[field] = value
    with pytest.raises(ValueError):
        SearchConfig(**values)


def test_search_policy_retains_request_only_policy_signature() -> None:
    signature = inspect.signature(PolicyValueSearchPolicy.choose)
    assert tuple(signature.parameters) == ("self", "request")
    assert signature.parameters["request"].annotation in (
        "DecisionRequest",
        DecisionRequest,
    )


def test_determinization_exactly_rebuilds_public_request(catalog) -> None:
    engine = GameEngine.new(44, first_player=0, catalog=catalog)
    request = engine.decision()
    assert isinstance(request, DecisionRequest)

    rebuilt_a = determinize_request(request, seed=101, catalog=catalog)
    rebuilt_b = determinize_request(request, seed=202, catalog=catalog)

    assert rebuilt_a.decision() == request
    assert rebuilt_b.decision() == request
    assert rebuilt_a._debug_state().decks != rebuilt_b._debug_state().decks


def test_determinization_hides_opponent_blind_reserve_and_preserves_own(
    catalog,
) -> None:
    engine = GameEngine.new(91, first_player=0, catalog=catalog)
    opening = engine.decision()
    assert isinstance(opening, DecisionRequest)
    reserve = next(
        action
        for action in opening.legal_actions
        if isinstance(action, ReserveDeck) and action.tier is Tier.ONE
    )
    opponent_request = engine.apply(reserve)
    assert isinstance(opponent_request, DecisionRequest)
    assert opponent_request.observation.opponent_public.reserved_cards[0].card is None
    assert not opponent_request.observation.self_private.blind_reserved_cards
    assert determinize_request(
        opponent_request, seed=303, catalog=catalog
    ).decision() == opponent_request

    reply = next(
        action
        for action in opponent_request.legal_actions
        if isinstance(action, TakeDistinct)
    )
    owner_request = engine.apply(reply)
    assert isinstance(owner_request, DecisionRequest)
    assert owner_request.observation.self_private.blind_reserved_cards
    assert determinize_request(
        owner_request, seed=404, catalog=catalog
    ).decision() == owner_request


def test_determinization_rebuilds_random_intermediate_decisions(catalog) -> None:
    engine = GameEngine.new(1234, first_player=1, catalog=catalog)
    policies = (RandomPolicy(10), RandomPolicy(11))
    checked = 0
    for index in range(120):
        response = engine.decision()
        if isinstance(response, TerminalResult):
            break
        rebuilt = determinize_request(
            response,
            seed=5_000 + index,
            catalog=catalog,
        )
        assert rebuilt.decision() == response
        checked += 1
        engine.apply(policies[engine.current_player].choose(response))
    assert checked >= 40


def test_top_k_depth_and_per_node_leaf_evaluation_are_deterministic(catalog) -> None:
    engine = GameEngine.new(777, first_player=0, catalog=catalog)
    request = engine.decision()
    assert isinstance(request, DecisionRequest)
    config = SearchConfig(top_k=2, depth=2, determinizations=1, seed=99)

    first = PolicyValueSearchPolicy(
        _zero_model(),
        config=config,
        catalog=catalog,
    )
    second = PolicyValueSearchPolicy(
        _zero_model(),
        config=config,
        catalog=catalog,
    )
    first_action = first.choose(request)
    second_action = second.choose(request)

    assert first_action == second_action == request.legal_actions[0]
    assert first_action in request.legal_actions
    assert first.stats.expanded_nodes == 6
    assert first.stats.leaf_nodes == 4
    # Alpha-beta evaluates each visited node on its own, so batches now count
    # model calls rather than levels; the number of positions is unchanged.
    assert first.stats.model_batches == 7
    assert first.stats.model_positions == 7
    assert first.stats.pruned_branches == 0


def test_default_four_ply_search_returns_a_legal_action(catalog) -> None:
    engine = GameEngine.new(888, first_player=0, catalog=catalog)
    request = engine.decision()
    assert isinstance(request, DecisionRequest)
    policy = PolicyValueSearchPolicy(
        _zero_model(),
        config=SearchConfig(top_k=5, depth=4, seed=12),
        catalog=catalog,
    )

    action = policy.choose(request)

    assert action in request.legal_actions
    assert policy.stats.expanded_nodes == 780
    assert policy.stats.leaf_nodes == 625
