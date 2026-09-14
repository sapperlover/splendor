from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import splendor.features as feature_module

from splendor.actions import (
    BuyMarket,
    BuyReserved,
    ClaimNoble,
    Pay,
    ReserveDeck,
    ReserveMarket,
    ReturnTokens,
    TakeDistinct,
    TakePair,
)
from splendor.contracts import (
    DecisionRequest,
    MarketView,
    PendingObservation,
    PlayerPrivateState,
    PublicReservedCard,
)
from splendor.features import (
    ACTION_TYPE_NAMES,
    CARD_FEATURE_NAMES,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_VERSION,
    GLOBAL_FEATURE_NAMES,
    PHASE_ORDER,
    SCALES,
    collate_encoded_decisions,
    encode_decision,
)
from splendor.heuristic_policy import DevelopmentPolicy, ScorePolicy
from splendor.model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from splendor.observation import build_observation
from splendor.random_policy import RandomPolicy
from splendor.result import TerminalResult
from splendor.rules import legal_actions
from splendor.engine import GameEngine


def _index(names: tuple[str, ...], name: str) -> int:
    return names.index(name)


def _target_card_slice() -> slice:
    start = FEATURE_SCHEMA.action_feature_names.index("target_card.valid")
    return slice(start, start + FEATURE_SCHEMA.card_dim)


def _target_noble_slice() -> slice:
    start = FEATURE_SCHEMA.action_feature_names.index("target_noble.valid")
    return slice(start, start + FEATURE_SCHEMA.noble_dim)


def test_frozen_schema_shapes_dtypes_masks_and_padding(initial_state, catalog) -> None:
    request = DecisionRequest(
        build_observation(initial_state, 0, catalog),
        legal_actions(initial_state, catalog),
    )
    encoded = encode_decision(request)

    assert FEATURE_SCHEMA_VERSION == "splendor-observation-action-v1"
    assert FEATURE_SCHEMA.card_dim == 29
    assert FEATURE_SCHEMA.noble_dim == 17
    assert FEATURE_SCHEMA.global_dim == 148
    assert FEATURE_SCHEMA.action_dim == 82
    assert FEATURE_SCHEMA.gem_color_order == ("white", "blue", "green", "red", "black")
    assert FEATURE_SCHEMA.token_color_order == (
        "white",
        "blue",
        "green",
        "red",
        "black",
        "gold",
    )
    assert FEATURE_SCHEMA.phase_order == (
        "main_action",
        "payment",
        "return_tokens",
        "choose_noble",
    )
    assert len(set(FEATURE_SCHEMA.card_feature_names)) == FEATURE_SCHEMA.card_dim
    assert len(set(FEATURE_SCHEMA.noble_feature_names)) == FEATURE_SCHEMA.noble_dim
    assert len(set(FEATURE_SCHEMA.global_feature_names)) == FEATURE_SCHEMA.global_dim
    assert len(set(FEATURE_SCHEMA.action_feature_names)) == FEATURE_SCHEMA.action_dim
    assert encoded.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert encoded.global_features.shape == (148,)
    assert encoded.market_t1.shape == encoded.market_t2.shape == encoded.market_t3.shape == (4, 29)
    assert encoded.self_reserved.shape == encoded.opponent_reserved.shape == (3, 29)
    assert encoded.nobles.shape == (3, 17)
    assert encoded.pending_card.shape == (29,)
    assert encoded.pending_card_mask.shape == ()
    assert encoded.candidate_features.shape == (len(request.legal_actions), 82)
    assert encoded.candidate_mask.shape == (len(request.legal_actions),)

    arrays = (
        encoded.global_features,
        encoded.market_t1,
        encoded.market_t2,
        encoded.market_t3,
        encoded.self_reserved,
        encoded.opponent_reserved,
        encoded.nobles,
        encoded.pending_card,
        encoded.candidate_features,
    )
    assert all(array.dtype == np.float32 for array in arrays)
    assert all(np.isfinite(array).all() for array in arrays)
    masks = (
        encoded.market_t1_mask,
        encoded.market_t2_mask,
        encoded.market_t3_mask,
        encoded.self_reserved_mask,
        encoded.opponent_reserved_mask,
        encoded.nobles_mask,
        encoded.pending_card_mask,
        encoded.candidate_mask,
    )
    assert all(mask.dtype == np.bool_ for mask in masks)
    assert encoded.market_t1_mask.tolist() == [True] * 4
    assert encoded.nobles_mask.tolist() == [True] * 3
    assert not encoded.self_reserved_mask.any()
    assert not encoded.opponent_reserved_mask.any()
    assert not encoded.pending_card_mask
    assert np.array_equal(encoded.self_reserved, np.zeros((3, 29), dtype=np.float32))
    assert np.array_equal(encoded.pending_card, np.zeros(29, dtype=np.float32))
    assert encoded.candidate_actions == request.legal_actions


def test_empty_sets_are_zero_padded_and_masked(initial_state, catalog) -> None:
    observation = build_observation(initial_state, 0, catalog)
    observation = replace(
        observation,
        market=MarketView((), (), ()),
        nobles_in_play=(),
    )
    request = DecisionRequest(observation, (TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),))
    encoded = encode_decision(request)

    for features, mask in (
        (encoded.market_t1, encoded.market_t1_mask),
        (encoded.market_t2, encoded.market_t2_mask),
        (encoded.market_t3, encoded.market_t3_mask),
        (encoded.nobles, encoded.nobles_mask),
    ):
        assert not mask.any()
        assert not features.any()


def test_unknown_opponent_reserve_is_valid_and_not_padding(initial_state, catalog) -> None:
    observation = build_observation(initial_state, 0, catalog)
    card = observation.market.tier_one[0]
    opponent_blind = PublicReservedCard(Tier.ONE, ReservationSource.DECK, None)
    own_blind = PublicReservedCard(Tier.ONE, ReservationSource.DECK, None)
    observation = replace(
        observation,
        self_public=replace(observation.self_public, reserved_cards=(own_blind,)),
        self_private=PlayerPrivateState((card,)),
        opponent_public=replace(
            observation.opponent_public,
            reserved_cards=(opponent_blind,),
        ),
    )
    encoded = encode_decision(
        DecisionRequest(observation, (TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),))
    )

    valid = _index(CARD_FEATURE_NAMES, "valid")
    known = _index(CARD_FEATURE_NAMES, "known")
    deck_source = _index(CARD_FEATURE_NAMES, "source_deck")
    point = _index(CARD_FEATURE_NAMES, "points_scaled")
    assert encoded.self_reserved_mask.tolist() == [True, False, False]
    assert encoded.self_reserved[0, [valid, known, deck_source]].tolist() == [1.0, 1.0, 1.0]
    assert encoded.opponent_reserved_mask.tolist() == [True, False, False]
    assert encoded.opponent_reserved[0, [valid, known, deck_source]].tolist() == [1.0, 0.0, 1.0]
    assert encoded.opponent_reserved[0, point] == 0.0
    assert encoded.opponent_reserved[0].any()
    assert not encoded.opponent_reserved[1].any()


def test_all_action_types_and_phase_context_have_stable_blocks(initial_state, catalog) -> None:
    base = build_observation(initial_state, 0, catalog)
    reserved_card = base.market.tier_one[0]
    known_reservation = PublicReservedCard(
        reserved_card.tier,
        ReservationSource.MARKET,
        reserved_card,
    )
    main_observation = replace(
        base,
        self_public=replace(base.self_public, reserved_cards=(known_reservation,)),
    )
    main_actions = (
        TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
        TakePair(Color.RED),
        ReserveMarket(base.market.tier_two[0].card_id),
        ReserveDeck(Tier.THREE),
        BuyMarket(base.market.tier_three[0].card_id),
        BuyReserved(reserved_card.card_id),
    )
    phase_cases = [
        (main_observation, action) for action in main_actions
    ]
    payment_target = base.market.tier_two[0]
    phase_cases.append(
        (
            replace(
                base,
                phase=DecisionPhase.PAYMENT,
                pending=PendingObservation(
                    BuyMarket(payment_target.card_id),
                    payment_target=payment_target,
                ),
            ),
            Pay(TokenVector(white=1, gold=1)),
        )
    )
    phase_cases.append(
        (
            replace(
                base,
                phase=DecisionPhase.RETURN_TOKENS,
                pending=PendingObservation(
                    TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
                    tokens_to_return=2,
                ),
            ),
            ReturnTokens(TokenVector(blue=1, gold=1)),
        )
    )
    noble = base.nobles_in_play[0]
    phase_cases.append(
        (
            replace(
                base,
                phase=DecisionPhase.CHOOSE_NOBLE,
                pending=PendingObservation(
                    TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
                    eligible_nobles=(noble,),
                ),
            ),
            ClaimNoble(noble.noble_id),
        )
    )

    action_types = (
        TakeDistinct,
        TakePair,
        ReserveMarket,
        ReserveDeck,
        BuyMarket,
        BuyReserved,
        Pay,
        ReturnTokens,
        ClaimNoble,
    )
    for observation, action in phase_cases:
        encoded = encode_decision(DecisionRequest(observation, (action,)))
        row = encoded.candidate_features[0]
        phase_values = [row[_index(FEATURE_SCHEMA.action_feature_names, f"phase_{phase.value}")] for phase in PHASE_ORDER]
        type_values = [row[_index(FEATURE_SCHEMA.action_feature_names, f"type_{name}")] for name in ACTION_TYPE_NAMES]
        assert sum(phase_values) == 1.0
        assert sum(type_values) == 1.0
        assert type_values[action_types.index(type(action))] == 1.0

    action_names = FEATURE_SCHEMA.action_feature_names
    take_distinct = encode_decision(DecisionRequest(phase_cases[0][0], (phase_cases[0][1],)))
    assert take_distinct.candidate_features[0, _index(action_names, "take_white_scaled")] == 0.25
    assert take_distinct.candidate_features[0, _index(action_names, "take_blue_scaled")] == 0.25
    assert take_distinct.candidate_features[0, _index(action_names, "take_green_scaled")] == 0.25
    take_pair = encode_decision(DecisionRequest(phase_cases[1][0], (phase_cases[1][1],)))
    assert take_pair.candidate_features[0, _index(action_names, "take_red_scaled")] == 0.5
    reserve_deck = encode_decision(DecisionRequest(phase_cases[3][0], (phase_cases[3][1],)))
    assert reserve_deck.candidate_features[0, _index(action_names, "source_deck")] == 1.0
    assert reserve_deck.candidate_features[0, _index(action_names, "tier_3")] == 1.0

    payment_encoded = encode_decision(DecisionRequest(phase_cases[6][0], (phase_cases[6][1],)))
    assert payment_encoded.pending_card_mask
    assert payment_encoded.candidate_features[0, _index(action_names, "payment_white_scaled")] == 0.25
    assert payment_encoded.candidate_features[0, _index(action_names, "payment_gold_scaled")] == 0.2
    assert np.array_equal(
        payment_encoded.pending_card,
        payment_encoded.candidate_features[0, _target_card_slice()],
    )
    return_encoded = encode_decision(DecisionRequest(phase_cases[7][0], (phase_cases[7][1],)))
    assert return_encoded.candidate_features[0, _index(action_names, "return_blue_scaled")] == 0.25
    assert return_encoded.candidate_features[0, _index(action_names, "return_gold_scaled")] == 0.2
    assert np.isclose(
        return_encoded.global_features[_index(GLOBAL_FEATURE_NAMES, "pending_tokens_to_return_scaled")],
        2 / SCALES.tokens_to_return,
    )
    noble_encoded = encode_decision(DecisionRequest(phase_cases[8][0], (phase_cases[8][1],)))
    assert noble_encoded.candidate_features[0, _target_noble_slice()][0] == 1.0
    assert np.isclose(
        noble_encoded.global_features[_index(GLOBAL_FEATURE_NAMES, "pending_eligible_nobles_scaled")],
        1 / SCALES.eligible_nobles,
    )


def test_target_card_feature_is_shared_with_market_object(initial_state, catalog) -> None:
    observation = build_observation(initial_state, 0, catalog)
    card = observation.market.tier_one[2]
    action = ReserveMarket(card.card_id)
    encoded = encode_decision(DecisionRequest(observation, (action,)))
    market_row = next(
        row
        for row in encoded.market_t1[encoded.market_t1_mask]
        if np.array_equal(row, encoded.candidate_features[0, _target_card_slice()])
    )
    assert np.array_equal(market_row, encoded.candidate_features[0, _target_card_slice()])


def test_batch_collator_uses_only_batch_max_candidates(initial_state, catalog) -> None:
    observation = build_observation(initial_state, 0, catalog)
    many = encode_decision(
        DecisionRequest(observation, legal_actions(initial_state, catalog))
    )
    one = encode_decision(
        DecisionRequest(observation, (TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),))
    )
    batch = collate_encoded_decisions((one, many))

    assert batch.global_features.shape == (2, FEATURE_SCHEMA.global_dim)
    assert batch.market_t1.shape == (2, 4, FEATURE_SCHEMA.card_dim)
    assert batch.nobles.shape == (2, 3, FEATURE_SCHEMA.noble_dim)
    assert batch.pending_card.shape == (2, FEATURE_SCHEMA.card_dim)
    assert batch.pending_card_mask.shape == (2,)
    assert batch.candidate_features.shape == (
        2,
        len(many.candidate_actions),
        FEATURE_SCHEMA.action_dim,
    )
    assert batch.candidate_mask.shape == (2, len(many.candidate_actions))
    assert batch.candidate_mask[0].tolist() == [True] + [False] * (len(many.candidate_actions) - 1)
    assert not batch.candidate_features[0, 1:].any()
    assert batch.candidate_mask[1].all()
    assert not hasattr(batch, "candidate_actions")


def test_reachable_large_candidate_state_is_encoded_without_a_fixed_dense_limit(
    tmp_path,
) -> None:
    engine = GameEngine.new(24, failure_dir=tmp_path)
    policies = (RandomPolicy(49), RandomPolicy(50))
    largest = None
    while True:
        decision = engine.decision()
        if isinstance(decision, TerminalResult):
            break
        if largest is None or len(decision.legal_actions) > len(largest.legal_actions):
            largest = decision
        engine.apply(policies[engine.current_player].choose(decision))

    assert largest is not None
    assert largest.observation.phase is DecisionPhase.RETURN_TOKENS
    assert len(largest.legal_actions) == 51
    encoded = encode_decision(largest)
    assert encoded.candidate_features.shape == (51, FEATURE_SCHEMA.action_dim)
    assert encoded.candidate_mask.all()


def test_scaling_does_not_clip_score_or_bonus(initial_state, catalog) -> None:
    observation = build_observation(initial_state, 0, catalog)
    observation = replace(
        observation,
        self_public=replace(
            observation.self_public,
            score=30,
            bonuses=GemVector(white=36, blue=36, green=36, red=36, black=36),
        ),
    )
    encoded = encode_decision(
        DecisionRequest(observation, (TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),))
    )
    assert encoded.global_features[_index(GLOBAL_FEATURE_NAMES, "self_score_scaled")] == 2.0
    assert encoded.global_features[_index(GLOBAL_FEATURE_NAMES, "self_bonus_white_scaled")] == 2.0
    assert SCALES.player_score == 15.0
    assert SCALES.player_bonus == 18.0


def test_public_final_round_and_completed_turn_context_are_encoded(initial_state, catalog) -> None:
    observation = replace(
        build_observation(initial_state, 0, catalog),
        final_round_triggered=True,
        final_round_triggered_by_self=False,
        own_completed_turns=9,
        opponent_completed_turns=8,
        consecutive_forced_skips=1,
    )
    encoded = encode_decision(
        DecisionRequest(observation, (TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),))
    )
    values = encoded.global_features
    assert values[_index(GLOBAL_FEATURE_NAMES, "final_round_triggered")] == 1.0
    assert values[_index(GLOBAL_FEATURE_NAMES, "final_round_triggered_by_self")] == 0.0
    assert np.isclose(
        values[_index(GLOBAL_FEATURE_NAMES, "own_completed_turns_scaled")],
        9 / SCALES.completed_turns,
    )
    assert np.isclose(
        values[_index(GLOBAL_FEATURE_NAMES, "opponent_completed_turns_scaled")],
        8 / SCALES.completed_turns,
    )
    assert values[_index(GLOBAL_FEATURE_NAMES, "completed_turn_delta_scaled")] == 1.0
    assert values[_index(GLOBAL_FEATURE_NAMES, "completed_turns_equal")] == 0.0
    assert values[_index(GLOBAL_FEATURE_NAMES, "consecutive_forced_skips_scaled")] == 0.5


def test_heuristic_policies_do_not_depend_on_feature_encoding() -> None:
    for policy_type in (ScorePolicy, DevelopmentPolicy):
        source = inspect.getsource(inspect.getmodule(policy_type))
        assert "splendor.features" not in source
        assert "from .features" not in source


def test_encoder_only_imports_policy_safe_contracts() -> None:
    source = inspect.getsource(feature_module)
    for forbidden in (
        "from ._state",
        "from .engine",
        "from .rules",
        "from .rng",
        "from .replay",
        "GameState",
    ):
        assert forbidden not in source
    for name in (
        *FEATURE_SCHEMA.card_feature_names,
        *FEATURE_SCHEMA.noble_feature_names,
        *FEATURE_SCHEMA.action_feature_names,
    ):
        assert "card_id" not in name
        assert "noble_id" not in name
        assert "slot" not in name


def test_real_engine_requests_in_every_decision_phase_encode_cleanly(tmp_path) -> None:
    seen: set[DecisionPhase] = set()
    for seed in (0, 51):
        engine = GameEngine.new(seed, failure_dir=tmp_path / str(seed))
        policies = (RandomPolicy(seed * 2 + 1), RandomPolicy(seed * 2 + 2))
        while True:
            decision = engine.decision()
            if isinstance(decision, TerminalResult):
                break
            seen.add(decision.observation.phase)
            encoded = encode_decision(decision)
            assert encoded.candidate_actions == decision.legal_actions
            assert encoded.candidate_mask.all()
            assert np.isfinite(encoded.global_features).all()
            assert np.isfinite(encoded.candidate_features).all()
            engine.apply(policies[engine.current_player].choose(decision))

    assert seen == set(PHASE_ORDER)
