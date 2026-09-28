from __future__ import annotations

from dataclasses import replace
from random import Random

import torch

from splendor._state import DeckState, ReservedCard
from splendor.actor_critic import ActorCritic, ActorCriticConfig
from splendor.contracts import DecisionRequest
from splendor.invariants import assert_game_state
from splendor.model import ReservationSource, Tier
from splendor.observation import build_observation
from splendor.rules import legal_actions
from splendor.search_policy import (
    PolicyValueSearchPolicy,
    SearchConfig,
    _request_seed,
    determinize_request,
)


def _hidden_state_variants(initial_state, catalog, count: int = 24):
    """Keep the acting player's information fixed while permuting hidden cards."""

    tier_one = initial_state.decks.for_tier(Tier.ONE)
    variants = []
    requests = []
    signatures = set()

    for index in range(count):
        rng = Random(0x51A7 + index)
        blind_ids = tuple(rng.sample(tier_one, 2))
        decks_by_tier = {}
        for tier in Tier:
            cards = list(initial_state.decks.for_tier(tier))
            if tier is Tier.ONE:
                cards = [card for card in cards if card not in blind_ids]
            rng.shuffle(cards)
            decks_by_tier[tier] = tuple(cards)

        reservations = tuple(
            ReservedCard(card_id, ReservationSource.DECK) for card_id in blind_ids
        )
        if index % 2:
            reservations = tuple(reversed(reservations))
        opponent = replace(
            initial_state.players[1], reserved_cards=reservations
        )
        state = replace(
            initial_state,
            decks=DeckState(
                tier_one=decks_by_tier[Tier.ONE],
                tier_two=decks_by_tier[Tier.TWO],
                tier_three=decks_by_tier[Tier.THREE],
            ),
            players=(initial_state.players[0], opponent),
        )
        assert_game_state(state, catalog)

        request = DecisionRequest(
            build_observation(state, 0, catalog),
            legal_actions(state, catalog),
        )
        variants.append(state)
        requests.append(request)
        signatures.add(
            (
                state.decks,
                tuple(
                    reserved.card_id
                    for reserved in state.players[1].reserved_cards
                ),
            )
        )

    assert len(signatures) == count
    assert all(request == requests[0] for request in requests[1:])
    return variants, requests


def test_search_determinization_is_invariant_to_actual_hidden_world(
    initial_state, catalog
) -> None:
    variants, requests = _hidden_state_variants(initial_state, catalog)
    assert len(variants) == len(requests) == 24

    policy_seed = 9_100_123
    configured_search_seed = 9_200_456
    effective_search_seed = configured_search_seed ^ policy_seed
    determinization_seed = _request_seed(
        requests[0], effective_search_seed, 0
    )
    worlds = [
        determinize_request(
            request,
            seed=determinization_seed,
            catalog=catalog,
        )._debug_state()
        for request in requests
    ]
    assert all(world == worlds[0] for world in worlds[1:])

    # A compact model and depth-2 search keep this regression test inexpensive;
    # the seed combination mirrors evaluation._create_serializable_policy.
    model_config = ActorCriticConfig(
        card_hidden=8,
        noble_hidden=4,
        state_hidden=16,
        actor_hidden=16,
        critic_hidden=16,
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5_500_123)
            model = ActorCritic(model_config).eval()

        search_config = SearchConfig(
            top_k=2,
            depth=2,
            determinizations=1,
            seed=effective_search_seed,
        )
        results = []
        for request in requests:
            policy = PolicyValueSearchPolicy(
                model,
                config=search_config,
                device="cpu",
                catalog=catalog,
            )
            action = policy.choose(request)
            assert action in request.legal_actions
            results.append(
                (action, policy.last_root_values, policy.last_root_bound_values)
            )
    finally:
        torch.set_num_threads(previous_threads)

    assert all(result == results[0] for result in results[1:])


def test_mcts_root_is_invariant_to_real_hidden_world(initial_state, catalog) -> None:
    from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy

    _, requests = _hidden_state_variants(initial_state, catalog, count=3)
    model = ActorCritic(ActorCriticConfig(
        card_hidden=8, noble_hidden=4, state_hidden=16,
        actor_hidden=16, critic_hidden=16,
    )).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    policy = PUCTSearchPolicy(model, config=MCTSConfig(simulations=5, determinizations=2, seed=901), catalog=catalog)
    first = policy.search(requests[0])
    assert all(policy.search(request) == first for request in requests[1:])
