from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from splendor._state import ReservedCard
from splendor.actions import BuyReserved, ClaimNoble, ReserveMarket, TakeDistinct
from splendor.actor_critic import ActorCritic, tensorize_batch
from splendor.contracts import DecisionRequest, PendingObservation, PublicReservedCard
from splendor.features import (
    FEATURE_SCHEMA,
    collate_encoded_decisions,
    encode_decision,
)
from splendor.model import Color, DecisionPhase, ReservationSource, Tier
from splendor.observation import build_observation
from splendor.rules import legal_actions


def _forward_by_action(model: ActorCritic, encoded):
    with torch.no_grad():
        logits, value = model(collate_encoded_decisions((encoded,)))
    return {
        action: logits[0, index]
        for index, action in enumerate(encoded.candidate_actions)
    }, value[0]


def _assert_same_semantic_outputs(model: ActorCritic, left, right) -> None:
    left_logits, left_value = _forward_by_action(model, left)
    right_logits, right_value = _forward_by_action(model, right)
    assert left_logits.keys() == right_logits.keys()
    torch.testing.assert_close(left_value, right_value, rtol=1e-6, atol=1e-7)
    for action in left_logits:
        torch.testing.assert_close(
            left_logits[action], right_logits[action], rtol=1e-6, atol=1e-7
        )


def test_market_permutation_preserves_value_and_semantic_action_logits(
    initial_state, catalog
) -> None:
    torch.manual_seed(40)
    model = ActorCritic().eval()
    observation = build_observation(initial_state, 0, catalog)
    actions = legal_actions(initial_state, catalog)
    permuted = replace(
        observation,
        market=replace(
            observation.market,
            tier_one=tuple(reversed(observation.market.tier_one)),
            tier_two=tuple(reversed(observation.market.tier_two)),
            tier_three=tuple(reversed(observation.market.tier_three)),
        ),
    )
    left = encode_decision(DecisionRequest(observation, actions))
    right = encode_decision(DecisionRequest(permuted, tuple(reversed(actions))))
    _assert_same_semantic_outputs(model, left, right)


def test_self_and_opponent_reserve_permutation_is_invariant(initial_state, catalog) -> None:
    torch.manual_seed(41)
    model = ActorCritic().eval()
    base = build_observation(initial_state, 0, catalog)
    own_cards = (base.market.tier_one[0], base.market.tier_two[0])
    opponent_cards = (base.market.tier_one[1], base.market.tier_three[0])
    own_reserved = tuple(
        PublicReservedCard(card.tier, ReservationSource.MARKET, card)
        for card in own_cards
    )
    opponent_reserved = tuple(
        PublicReservedCard(card.tier, ReservationSource.MARKET, card)
        for card in opponent_cards
    )
    observation = replace(
        base,
        self_public=replace(base.self_public, reserved_cards=own_reserved),
        opponent_public=replace(base.opponent_public, reserved_cards=opponent_reserved),
    )
    permuted = replace(
        observation,
        self_public=replace(
            observation.self_public,
            reserved_cards=tuple(reversed(observation.self_public.reserved_cards)),
        ),
        opponent_public=replace(
            observation.opponent_public,
            reserved_cards=tuple(reversed(observation.opponent_public.reserved_cards)),
        ),
    )
    actions = tuple(BuyReserved(card.card_id) for card in own_cards)
    left = encode_decision(DecisionRequest(observation, actions))
    right = encode_decision(DecisionRequest(permuted, tuple(reversed(actions))))
    _assert_same_semantic_outputs(model, left, right)


def test_noble_permutation_preserves_value_and_claim_logits(initial_state, catalog) -> None:
    torch.manual_seed(42)
    model = ActorCritic().eval()
    base = build_observation(initial_state, 0, catalog)
    main = TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN))
    observation = replace(
        base,
        phase=DecisionPhase.CHOOSE_NOBLE,
        pending=PendingObservation(main, eligible_nobles=base.nobles_in_play),
    )
    permuted_nobles = tuple(reversed(base.nobles_in_play))
    permuted = replace(
        observation,
        nobles_in_play=permuted_nobles,
        pending=PendingObservation(main, eligible_nobles=permuted_nobles),
    )
    actions = tuple(ClaimNoble(noble.noble_id) for noble in base.nobles_in_play)
    left = encode_decision(DecisionRequest(observation, actions))
    right = encode_decision(DecisionRequest(permuted, tuple(reversed(actions))))
    _assert_same_semantic_outputs(model, left, right)


def test_hidden_blind_reserve_identity_cannot_change_model_output(
    initial_state, catalog
) -> None:
    hidden_x, hidden_y = initial_state.decks.tier_one[:2]

    def with_hidden(card_id: str):
        opponent = replace(
            initial_state.players[1],
            reserved_cards=(ReservedCard(card_id, ReservationSource.DECK),),
        )
        return replace(
            initial_state,
            decks=initial_state.decks.replacing_tier(
                Tier.ONE,
                tuple(card for card in initial_state.decks.tier_one if card != card_id),
            ),
            players=(initial_state.players[0], opponent),
        )

    state_x = with_hidden(hidden_x)
    state_y = with_hidden(hidden_y)
    request_x = DecisionRequest(
        build_observation(state_x, 0, catalog), legal_actions(state_x, catalog)
    )
    request_y = DecisionRequest(
        build_observation(state_y, 0, catalog), legal_actions(state_y, catalog)
    )
    encoded_x = encode_decision(request_x)
    encoded_y = encode_decision(request_y)
    for name in encoded_x.__dataclass_fields__:
        if name != "candidate_actions":
            left = getattr(encoded_x, name)
            right = getattr(encoded_y, name)
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                assert left == right
    assert encoded_x.opponent_reserved_mask[0]
    assert encoded_x.opponent_reserved[0, 0] == 1.0
    assert encoded_x.opponent_reserved[0, 1] == 0.0

    torch.manual_seed(43)
    _assert_same_semantic_outputs(ActorCritic().eval(), encoded_x, encoded_y)


def test_unknown_future_deck_order_cannot_change_model_output(initial_state, catalog) -> None:
    cards = initial_state.decks.tier_two
    permuted_state = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(
            Tier.TWO, (cards[-1], cards[1], cards[0], *cards[2:-1])
        ),
    )
    left_request = DecisionRequest(
        build_observation(initial_state, 0, catalog),
        legal_actions(initial_state, catalog),
    )
    right_request = DecisionRequest(
        build_observation(permuted_state, 0, catalog),
        legal_actions(permuted_state, catalog),
    )
    assert left_request == right_request
    left = encode_decision(left_request)
    right = encode_decision(right_request)
    torch.manual_seed(44)
    _assert_same_semantic_outputs(ActorCritic().eval(), left, right)


def test_state_and_action_target_use_identical_card_representation(
    initial_state, catalog
) -> None:
    base = build_observation(initial_state, 0, catalog)
    card = base.market.tier_one[0]
    action = ReserveMarket(card.card_id)
    original = encode_decision(DecisionRequest(base, (action,)))
    changed_card = replace(card, points=card.points + 1)
    changed_observation = replace(
        base,
        market=replace(
            base.market,
            tier_one=(changed_card, *base.market.tier_one[1:]),
        ),
    )
    changed = encode_decision(DecisionRequest(changed_observation, (action,)))
    original_batch = tensorize_batch(collate_encoded_decisions((original,)))
    changed_batch = tensorize_batch(collate_encoded_decisions((changed,)))

    model = ActorCritic().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(0.01)
                module.bias.fill_(0.01)

        card_start = model.action_meta_dim
        card_stop = card_start + FEATURE_SCHEMA.card_dim
        state_card = model.card_encoder(original_batch.market_t1[:, 0])
        action_card = model.card_encoder(
            original_batch.candidate_features[:, 0, card_start:card_stop]
        )
        changed_state_card = model.card_encoder(changed_batch.market_t1[:, 0])
        changed_action_card = model.card_encoder(
            changed_batch.candidate_features[:, 0, card_start:card_stop]
        )
        torch.testing.assert_close(state_card, action_card, rtol=0, atol=0)
        torch.testing.assert_close(
            changed_state_card, changed_action_card, rtol=0, atol=0
        )
        assert not torch.equal(changed_state_card, state_card)

        original_state = model.encode_state(original_batch)
        changed_state = model.encode_state(changed_batch)
        original_action = model.encode_actions(original_batch.candidate_features)
        changed_action = model.encode_actions(changed_batch.candidate_features)
        assert not torch.equal(original_state, changed_state)
        assert not torch.equal(original_action, changed_action)
