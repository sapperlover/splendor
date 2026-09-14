from __future__ import annotations

from dataclasses import replace

import numpy as np

from splendor._state import ReservedCard
from splendor.actions import BuyReserved, ClaimNoble, ReserveDeck, ReserveMarket, TakeDistinct
from splendor.catalog import Catalog
from splendor.contracts import (
    CardView,
    DecisionRequest,
    MarketView,
    NobleView,
    PendingObservation,
    PublicReservedCard,
)
from splendor.features import (
    CARD_FEATURE_NAMES,
    FEATURE_SCHEMA,
    GLOBAL_FEATURE_NAMES,
    NOBLE_FEATURE_NAMES,
    SCALES,
    EncodedDecision,
    encode_decision,
)
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from splendor.observation import build_observation
from splendor.rules import legal_actions


def _request(state, observer: int, catalog: Catalog) -> DecisionRequest:
    return DecisionRequest(
        build_observation(state, observer, catalog),
        legal_actions(state, catalog),
    )


def _numeric_arrays(encoded: EncodedDecision) -> tuple[np.ndarray, ...]:
    return (
        encoded.global_features,
        encoded.market_t1,
        encoded.market_t1_mask,
        encoded.market_t2,
        encoded.market_t2_mask,
        encoded.market_t3,
        encoded.market_t3_mask,
        encoded.self_reserved,
        encoded.self_reserved_mask,
        encoded.opponent_reserved,
        encoded.opponent_reserved_mask,
        encoded.nobles,
        encoded.nobles_mask,
        encoded.pending_card,
        encoded.pending_card_mask,
        encoded.candidate_features,
        encoded.candidate_mask,
    )


def _assert_encoded_equal(left: EncodedDecision, right: EncodedDecision) -> None:
    assert left.feature_schema_version == right.feature_schema_version
    assert left.candidate_actions == right.candidate_actions
    for left_array, right_array in zip(_numeric_arrays(left), _numeric_arrays(right), strict=True):
        np.testing.assert_array_equal(left_array, right_array)


def _row_multiset(features: np.ndarray, mask: np.ndarray) -> list[tuple[float, ...]]:
    return sorted(tuple(float(value) for value in row) for row in features[mask])


def _candidate_row(encoded: EncodedDecision, action) -> np.ndarray:
    return encoded.candidate_features[encoded.candidate_actions.index(action)]


def test_opponent_blind_identity_swap_is_fully_encoding_invariant(
    initial_state, catalog: Catalog
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
    assert_game_state(state_x, catalog)
    assert_game_state(state_y, catalog)
    request_x = _request(state_x, 0, catalog)
    request_y = _request(state_y, 0, catalog)
    assert request_x == request_y

    encoded_x = encode_decision(request_x)
    encoded_y = encode_decision(request_y)
    _assert_encoded_equal(encoded_x, encoded_y)
    reserve_deck = ReserveDeck(Tier.ONE)
    np.testing.assert_array_equal(
        _candidate_row(encoded_x, reserve_deck),
        _candidate_row(encoded_y, reserve_deck),
    )

    owner_x = build_observation(state_x, 1, catalog)
    owner_y = build_observation(state_y, 1, catalog)
    owner_action = TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN))
    owner_encoded_x = encode_decision(DecisionRequest(owner_x, (owner_action,)))
    owner_encoded_y = encode_decision(DecisionRequest(owner_y, (owner_action,)))
    assert not np.array_equal(owner_encoded_x.self_reserved, owner_encoded_y.self_reserved)


def test_unknown_future_deck_order_and_reserve_deck_candidate_do_not_leak(
    initial_state, catalog: Catalog
) -> None:
    cards = initial_state.decks.tier_two
    permuted = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(
            Tier.TWO,
            (cards[-1], cards[1], cards[0], *cards[2:-1]),
        ),
    )
    assert_game_state(permuted, catalog)
    left = _request(initial_state, 0, catalog)
    right = _request(permuted, 0, catalog)
    assert left == right
    encoded_left = encode_decision(left)
    encoded_right = encode_decision(right)
    _assert_encoded_equal(encoded_left, encoded_right)
    action = ReserveDeck(Tier.TWO)
    np.testing.assert_array_equal(
        _candidate_row(encoded_left, action),
        _candidate_row(encoded_right, action),
    )


def test_public_purchased_encoding_only_marks_actual_purchases(
    initial_state, catalog: Catalog
) -> None:
    public_reserved_id = initial_state.market.tier_one[0]
    blind_reserved_id = initial_state.decks.tier_one[0]
    purchased_id = initial_state.decks.tier_one[1]

    market_without = tuple(
        card for card in initial_state.market.tier_one if card != public_reserved_id
    )
    refill = initial_state.decks.tier_one[-1]
    public_reserved_state = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(Tier.ONE, initial_state.decks.tier_one[:-1]),
        market=initial_state.market.replacing_tier(Tier.ONE, market_without + (refill,)),
        players=(
            initial_state.players[0],
            replace(
                initial_state.players[1],
                reserved_cards=(ReservedCard(public_reserved_id, ReservationSource.MARKET),),
            ),
        ),
    )
    blind_reserved_state = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(
            Tier.ONE,
            tuple(card for card in initial_state.decks.tier_one if card != blind_reserved_id),
        ),
        players=(
            initial_state.players[0],
            replace(
                initial_state.players[1],
                reserved_cards=(ReservedCard(blind_reserved_id, ReservationSource.DECK),),
            ),
        ),
    )
    purchased_state = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(
            Tier.ONE,
            tuple(card for card in initial_state.decks.tier_one if card != purchased_id),
        ),
        players=(
            initial_state.players[0],
            replace(initial_state.players[1], purchased_card_ids=(purchased_id,)),
        ),
    )

    public_start = GLOBAL_FEATURE_NAMES.index("public_purchased_000")
    public_stop = public_start + FEATURE_SCHEMA.public_purchased_dim
    card_index = {card.card_id: index for index, card in enumerate(catalog.cards)}
    for state in (public_reserved_state, blind_reserved_state, purchased_state):
        assert_game_state(state, catalog)
    public_encoded = encode_decision(_request(public_reserved_state, 0, catalog))
    blind_encoded = encode_decision(_request(blind_reserved_state, 0, catalog))
    bought_encoded = encode_decision(_request(purchased_state, 0, catalog))

    assert not public_encoded.global_features[public_start:public_stop].any()
    assert not blind_encoded.global_features[public_start:public_stop].any()
    bought_bits = bought_encoded.global_features[public_start:public_stop]
    assert bought_bits.sum() == 1.0
    assert bought_bits[card_index[purchased_id]] == 1.0

    known = CARD_FEATURE_NAMES.index("known")
    points = CARD_FEATURE_NAMES.index("points_scaled")
    assert public_encoded.opponent_reserved[0, known] == 1.0
    assert np.isclose(
        public_encoded.opponent_reserved[0, points],
        catalog.card(public_reserved_id).points / SCALES.card_points,
    )
    assert blind_encoded.opponent_reserved[0, known] == 0.0


def test_market_noble_and_reserved_permutations_only_permute_object_rows(
    initial_state, catalog: Catalog
) -> None:
    base = build_observation(initial_state, 0, catalog)
    own_cards = base.market.tier_one[:2]
    opponent_cards = base.market.tier_two[:2]
    own_reserved = tuple(
        PublicReservedCard(card.tier, ReservationSource.MARKET, card)
        for card in own_cards
    )
    opponent_reserved = (
        PublicReservedCard(opponent_cards[0].tier, ReservationSource.MARKET, opponent_cards[0]),
        PublicReservedCard(Tier.THREE, ReservationSource.DECK, None),
    )
    observation = replace(
        base,
        self_public=replace(base.self_public, reserved_cards=own_reserved),
        opponent_public=replace(base.opponent_public, reserved_cards=opponent_reserved),
    )
    permuted = replace(
        observation,
        market=replace(
            observation.market,
            tier_one=tuple(reversed(observation.market.tier_one)),
        ),
        nobles_in_play=tuple(reversed(observation.nobles_in_play)),
        self_public=replace(
            observation.self_public,
            reserved_cards=tuple(reversed(observation.self_public.reserved_cards)),
        ),
        opponent_public=replace(
            observation.opponent_public,
            reserved_cards=tuple(reversed(observation.opponent_public.reserved_cards)),
        ),
    )
    target_market = ReserveMarket(observation.market.tier_one[0].card_id)
    target_reserved = BuyReserved(own_cards[0].card_id)
    actions = (target_market, target_reserved)
    encoded = encode_decision(DecisionRequest(observation, actions))
    encoded_permuted = encode_decision(DecisionRequest(permuted, actions))

    for left_features, left_mask, right_features, right_mask in (
        (encoded.market_t1, encoded.market_t1_mask, encoded_permuted.market_t1, encoded_permuted.market_t1_mask),
        (encoded.nobles, encoded.nobles_mask, encoded_permuted.nobles, encoded_permuted.nobles_mask),
        (encoded.self_reserved, encoded.self_reserved_mask, encoded_permuted.self_reserved, encoded_permuted.self_reserved_mask),
        (encoded.opponent_reserved, encoded.opponent_reserved_mask, encoded_permuted.opponent_reserved, encoded_permuted.opponent_reserved_mask),
    ):
        assert _row_multiset(left_features, left_mask) == _row_multiset(right_features, right_mask)

    np.testing.assert_array_equal(
        _candidate_row(encoded, target_market),
        _candidate_row(encoded_permuted, target_market),
    )
    np.testing.assert_array_equal(
        _candidate_row(encoded, target_reserved),
        _candidate_row(encoded_permuted, target_reserved),
    )

    noble = observation.nobles_in_play[0]
    claim_observation = replace(
        observation,
        phase=DecisionPhase.CHOOSE_NOBLE,
        pending=PendingObservation(
            TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
            eligible_nobles=observation.nobles_in_play,
        ),
    )
    claim_permuted = replace(
        claim_observation,
        nobles_in_play=tuple(reversed(claim_observation.nobles_in_play)),
        pending=replace(
            claim_observation.pending,
            eligible_nobles=tuple(reversed(claim_observation.pending.eligible_nobles)),
        ),
    )
    claim = ClaimNoble(noble.noble_id)
    left_claim = encode_decision(DecisionRequest(claim_observation, (claim,)))
    right_claim = encode_decision(DecisionRequest(claim_permuted, (claim,)))
    np.testing.assert_array_equal(left_claim.candidate_features, right_claim.candidate_features)


def test_card_and_noble_execution_ids_are_mapping_only(initial_state, catalog: Catalog) -> None:
    base = build_observation(initial_state, 0, catalog)
    template = base.market.tier_one[0]
    left_card = replace(template, card_id="execution-handle-left")
    right_card = replace(template, card_id="execution-handle-right")
    left_observation = replace(base, market=MarketView((left_card,), (), ()))
    right_observation = replace(base, market=MarketView((right_card,), (), ()))
    left_action = ReserveMarket(left_card.card_id)
    right_action = ReserveMarket(right_card.card_id)
    left = encode_decision(DecisionRequest(left_observation, (left_action,)))
    right = encode_decision(DecisionRequest(right_observation, (right_action,)))

    np.testing.assert_array_equal(left.market_t1, right.market_t1)
    np.testing.assert_array_equal(left.candidate_features, right.candidate_features)
    assert left.candidate_actions == (left_action,)
    assert right.candidate_actions == (right_action,)
    assert left.candidate_actions != right.candidate_actions

    noble = base.nobles_in_play[0]
    left_noble = replace(noble, noble_id="noble-handle-left")
    right_noble = replace(noble, noble_id="noble-handle-right")
    main = TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN))
    left_noble_observation = replace(
        base,
        phase=DecisionPhase.CHOOSE_NOBLE,
        nobles_in_play=(left_noble,),
        pending=PendingObservation(main, eligible_nobles=(left_noble,)),
    )
    right_noble_observation = replace(
        base,
        phase=DecisionPhase.CHOOSE_NOBLE,
        nobles_in_play=(right_noble,),
        pending=PendingObservation(main, eligible_nobles=(right_noble,)),
    )
    left_noble_encoded = encode_decision(
        DecisionRequest(left_noble_observation, (ClaimNoble(left_noble.noble_id),))
    )
    right_noble_encoded = encode_decision(
        DecisionRequest(right_noble_observation, (ClaimNoble(right_noble.noble_id),))
    )
    np.testing.assert_array_equal(
        left_noble_encoded.candidate_features,
        right_noble_encoded.candidate_features,
    )


def test_relative_player_view_swaps_self_and_opponent_without_seat_feature(
    initial_state, catalog: Catalog
) -> None:
    base = build_observation(initial_state, 0, catalog)
    player_zero = replace(
        base.self_public,
        tokens=TokenVector(white=1, gold=2),
        bonuses=GemVector(white=3),
        score=7,
    )
    player_one = replace(
        base.opponent_public,
        tokens=TokenVector(white=3, gold=1),
        bonuses=GemVector(white=5),
        score=11,
    )
    observation_zero = replace(
        base,
        self_public=player_zero,
        opponent_public=player_one,
        self_is_first_player=True,
        own_completed_turns=6,
        opponent_completed_turns=5,
    )
    observation_one = replace(
        base,
        self_public=player_one,
        opponent_public=player_zero,
        self_is_first_player=False,
        own_completed_turns=5,
        opponent_completed_turns=6,
    )
    action = TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN))
    encoded_zero = encode_decision(DecisionRequest(observation_zero, (action,)))
    encoded_one = encode_decision(DecisionRequest(observation_one, (action,)))

    self_white = GLOBAL_FEATURE_NAMES.index("self_token_white_scaled")
    opponent_white = GLOBAL_FEATURE_NAMES.index("opponent_token_white_scaled")
    self_score = GLOBAL_FEATURE_NAMES.index("self_score_scaled")
    opponent_score = GLOBAL_FEATURE_NAMES.index("opponent_score_scaled")
    first = GLOBAL_FEATURE_NAMES.index("self_is_first_player")
    assert encoded_zero.global_features[self_white] == 1 / SCALES.token_by_color[0]
    assert encoded_zero.global_features[opponent_white] == 3 / SCALES.token_by_color[0]
    assert encoded_one.global_features[self_white] == 3 / SCALES.token_by_color[0]
    assert encoded_one.global_features[opponent_white] == 1 / SCALES.token_by_color[0]
    assert np.isclose(encoded_zero.global_features[self_score], 7 / SCALES.player_score)
    assert np.isclose(encoded_zero.global_features[opponent_score], 11 / SCALES.player_score)
    assert np.isclose(encoded_one.global_features[self_score], 11 / SCALES.player_score)
    assert np.isclose(encoded_one.global_features[opponent_score], 7 / SCALES.player_score)
    assert encoded_zero.global_features[first] == 1.0
    assert encoded_one.global_features[first] == 0.0
    assert not any("player_0" in name or "player_1" in name for name in GLOBAL_FEATURE_NAMES)


def test_card_and_noble_auxiliary_features_are_observation_only(initial_state, catalog: Catalog) -> None:
    base = build_observation(initial_state, 0, catalog)
    card = CardView(
        "visible-card",
        Tier.ONE,
        Color.WHITE,
        2,
        GemVector(white=3, blue=2, green=1),
    )
    noble = NobleView(
        "visible-noble",
        3,
        GemVector(white=4, blue=3, green=2),
    )
    observation = replace(
        base,
        market=MarketView((card,), (), ()),
        nobles_in_play=(noble,),
        self_public=replace(
            base.self_public,
            tokens=TokenVector(white=1, gold=1),
            bonuses=GemVector(white=1, blue=3),
        ),
        opponent_public=replace(
            base.opponent_public,
            bonuses=GemVector(white=2, green=1),
        ),
    )
    encoded = encode_decision(DecisionRequest(observation, (ReserveMarket(card.card_id),)))
    card_row = encoded.market_t1[0]
    expected_need = (2, 0, 1, 0, 0)
    expected_deficit = (1, 0, 1, 0, 0)
    for color, need, deficit in zip(
        ("white", "blue", "green", "red", "black"),
        expected_need,
        expected_deficit,
        strict=True,
    ):
        assert np.isclose(
            card_row[CARD_FEATURE_NAMES.index(f"discounted_need_{color}_scaled")],
            need / SCALES.card_cost,
        )
        assert np.isclose(
            card_row[CARD_FEATURE_NAMES.index(f"ordinary_deficit_{color}_scaled")],
            deficit / SCALES.card_cost,
        )
    assert np.isclose(
        card_row[CARD_FEATURE_NAMES.index("gold_uncovered_deficit_scaled")],
        1 / SCALES.purchase_gap_total,
    )

    noble_row = encoded.nobles[0]
    expected_self_gap = (3, 0, 2, 0, 0)
    expected_opponent_gap = (2, 3, 1, 0, 0)
    for color, self_gap, opponent_gap in zip(
        ("white", "blue", "green", "red", "black"),
        expected_self_gap,
        expected_opponent_gap,
        strict=True,
    ):
        assert np.isclose(
            noble_row[NOBLE_FEATURE_NAMES.index(f"self_gap_{color}_scaled")],
            self_gap / SCALES.noble_requirement,
        )
        assert np.isclose(
            noble_row[NOBLE_FEATURE_NAMES.index(f"opponent_gap_{color}_scaled")],
            opponent_gap / SCALES.noble_requirement,
        )
