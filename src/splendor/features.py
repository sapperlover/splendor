"""Information-safe NumPy features for observations and legal candidates.

This module is deliberately downstream of the public policy contract.  It
does not import the engine, authoritative state, rules, RNG, or replay code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .actions import (
    Action,
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
from .contracts import CardView, DecisionRequest, NobleView, Observation
from .model import (
    DecisionPhase,
    GEM_COLORS,
    ReservationSource,
    Tier,
    TOKEN_COLORS,
)


FEATURE_SCHEMA_VERSION = "splendor-observation-action-v1"
MARKET_CAPACITY = 4
RESERVED_CAPACITY = 3
NOBLE_CAPACITY = 3
PUBLIC_PURCHASED_DIM = 90

PHASE_ORDER: tuple[DecisionPhase, ...] = (
    DecisionPhase.MAIN_ACTION,
    DecisionPhase.PAYMENT,
    DecisionPhase.RETURN_TOKENS,
    DecisionPhase.CHOOSE_NOBLE,
)
ACTION_TYPE_ORDER: tuple[type[object], ...] = (
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
ACTION_TYPE_NAMES: tuple[str, ...] = (
    "take_distinct",
    "take_pair",
    "reserve_market",
    "reserve_deck",
    "buy_market",
    "buy_reserved",
    "pay",
    "return_tokens",
    "claim_noble",
)
MAIN_ACTION_TYPE_ORDER: tuple[type[object], ...] = ACTION_TYPE_ORDER[:6]
MAIN_ACTION_TYPE_NAMES: tuple[str, ...] = ACTION_TYPE_NAMES[:6]
SOURCE_ORDER: tuple[ReservationSource, ...] = (
    ReservationSource.MARKET,
    ReservationSource.DECK,
)
TIER_ORDER: tuple[Tier, ...] = (Tier.ONE, Tier.TWO, Tier.THREE)


@dataclass(frozen=True, slots=True)
class FeatureScales:
    """Frozen divisors.  Encoding never clips values to these scales."""

    token_by_color: tuple[float, ...] = (4.0, 4.0, 4.0, 4.0, 4.0, 5.0)
    deck_by_tier: tuple[float, ...] = (40.0, 30.0, 20.0)
    card_cost: float = 7.0
    card_points: float = 5.0
    noble_points: float = 3.0
    noble_requirement: float = 4.0
    purchase_gap_total: float = 14.0
    player_score: float = 15.0
    player_bonus: float = 18.0
    purchased_cards: float = 90.0
    noble_count: float = 3.0
    completed_turns: float = 40.0
    completed_turn_delta: float = 1.0
    consecutive_forced_skips: float = 2.0
    tokens_to_return: float = 3.0
    eligible_nobles: float = 3.0


SCALES = FeatureScales()

CARD_FEATURE_NAMES: tuple[str, ...] = (
    "valid",
    "known",
    *(f"tier_{tier.value}" for tier in TIER_ORDER),
    *(f"bonus_{color.value}" for color in GEM_COLORS),
    *(f"source_{source.value}" for source in SOURCE_ORDER),
    "points_scaled",
    *(f"cost_{color.value}_scaled" for color in GEM_COLORS),
    *(f"discounted_need_{color.value}_scaled" for color in GEM_COLORS),
    *(f"ordinary_deficit_{color.value}_scaled" for color in GEM_COLORS),
    "gold_uncovered_deficit_scaled",
)

NOBLE_FEATURE_NAMES: tuple[str, ...] = (
    "valid",
    "points_scaled",
    *(f"requirement_{color.value}_scaled" for color in GEM_COLORS),
    *(f"self_gap_{color.value}_scaled" for color in GEM_COLORS),
    *(f"opponent_gap_{color.value}_scaled" for color in GEM_COLORS),
)

GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    *(f"bank_{color.value}_scaled" for color in TOKEN_COLORS),
    *(f"deck_tier_{tier.value}_scaled" for tier in TIER_ORDER),
    *(f"self_token_{color.value}_scaled" for color in TOKEN_COLORS),
    *(f"self_bonus_{color.value}_scaled" for color in GEM_COLORS),
    "self_score_scaled",
    "self_purchased_cards_scaled",
    "self_nobles_scaled",
    *(f"opponent_token_{color.value}_scaled" for color in TOKEN_COLORS),
    *(f"opponent_bonus_{color.value}_scaled" for color in GEM_COLORS),
    "opponent_score_scaled",
    "opponent_purchased_cards_scaled",
    "opponent_nobles_scaled",
    *(f"public_purchased_{index:03d}" for index in range(PUBLIC_PURCHASED_DIM)),
    "self_is_first_player",
    *(f"phase_{phase.value}" for phase in PHASE_ORDER),
    "final_round_triggered",
    "final_round_triggered_by_self",
    "own_completed_turns_scaled",
    "opponent_completed_turns_scaled",
    "completed_turn_delta_scaled",
    "completed_turns_equal",
    "consecutive_forced_skips_scaled",
    "pending_present",
    *(f"pending_main_{name}" for name in MAIN_ACTION_TYPE_NAMES),
    "pending_tokens_to_return_scaled",
    "pending_eligible_nobles_scaled",
)

ACTION_FEATURE_NAMES: tuple[str, ...] = (
    *(f"phase_{phase.value}" for phase in PHASE_ORDER),
    *(f"type_{name}" for name in ACTION_TYPE_NAMES),
    *(f"source_{source.value}" for source in SOURCE_ORDER),
    *(f"tier_{tier.value}" for tier in TIER_ORDER),
    *(f"take_{color.value}_scaled" for color in TOKEN_COLORS),
    *(f"payment_{color.value}_scaled" for color in TOKEN_COLORS),
    *(f"return_{color.value}_scaled" for color in TOKEN_COLORS),
    *(f"target_card.{name}" for name in CARD_FEATURE_NAMES),
    *(f"target_noble.{name}" for name in NOBLE_FEATURE_NAMES),
)


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    version: str = FEATURE_SCHEMA_VERSION
    card_feature_names: tuple[str, ...] = CARD_FEATURE_NAMES
    noble_feature_names: tuple[str, ...] = NOBLE_FEATURE_NAMES
    global_feature_names: tuple[str, ...] = GLOBAL_FEATURE_NAMES
    action_feature_names: tuple[str, ...] = ACTION_FEATURE_NAMES
    gem_color_order: tuple[str, ...] = tuple(color.value for color in GEM_COLORS)
    token_color_order: tuple[str, ...] = tuple(color.value for color in TOKEN_COLORS)
    tier_order: tuple[int, ...] = tuple(tier.value for tier in TIER_ORDER)
    phase_order: tuple[str, ...] = tuple(phase.value for phase in PHASE_ORDER)
    action_type_order: tuple[str, ...] = ACTION_TYPE_NAMES
    source_order: tuple[str, ...] = tuple(source.value for source in SOURCE_ORDER)
    scales: FeatureScales = SCALES
    market_capacity: int = MARKET_CAPACITY
    reserved_capacity: int = RESERVED_CAPACITY
    noble_capacity: int = NOBLE_CAPACITY
    public_purchased_dim: int = PUBLIC_PURCHASED_DIM

    @property
    def card_dim(self) -> int:
        return len(self.card_feature_names)

    @property
    def noble_dim(self) -> int:
        return len(self.noble_feature_names)

    @property
    def global_dim(self) -> int:
        return len(self.global_feature_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_feature_names)


FEATURE_SCHEMA = FeatureSchema()


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    feature_schema_version: str
    global_features: np.ndarray
    market_t1: np.ndarray
    market_t1_mask: np.ndarray
    market_t2: np.ndarray
    market_t2_mask: np.ndarray
    market_t3: np.ndarray
    market_t3_mask: np.ndarray
    self_reserved: np.ndarray
    self_reserved_mask: np.ndarray
    opponent_reserved: np.ndarray
    opponent_reserved_mask: np.ndarray
    nobles: np.ndarray
    nobles_mask: np.ndarray
    pending_card: np.ndarray
    pending_card_mask: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    candidate_actions: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class EncodedDecisionBatch:
    feature_schema_version: str
    global_features: np.ndarray
    market_t1: np.ndarray
    market_t1_mask: np.ndarray
    market_t2: np.ndarray
    market_t2_mask: np.ndarray
    market_t3: np.ndarray
    market_t3_mask: np.ndarray
    self_reserved: np.ndarray
    self_reserved_mask: np.ndarray
    opponent_reserved: np.ndarray
    opponent_reserved_mask: np.ndarray
    nobles: np.ndarray
    nobles_mask: np.ndarray
    pending_card: np.ndarray
    pending_card_mask: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class EncodedState:
    """State-only encoding: everything a critic needs, with no candidate list.

    The pending-turn context (payment target, tokens to return, eligible
    nobles) is part of the state encoding, so a critic evaluated through this
    path sees exactly the same information as one evaluated through
    :class:`EncodedDecision`.
    """

    feature_schema_version: str
    global_features: np.ndarray
    market_t1: np.ndarray
    market_t1_mask: np.ndarray
    market_t2: np.ndarray
    market_t2_mask: np.ndarray
    market_t3: np.ndarray
    market_t3_mask: np.ndarray
    self_reserved: np.ndarray
    self_reserved_mask: np.ndarray
    opponent_reserved: np.ndarray
    opponent_reserved_mask: np.ndarray
    nobles: np.ndarray
    nobles_mask: np.ndarray
    pending_card: np.ndarray
    pending_card_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class EncodedStateBatch:
    feature_schema_version: str
    global_features: np.ndarray
    market_t1: np.ndarray
    market_t1_mask: np.ndarray
    market_t2: np.ndarray
    market_t2_mask: np.ndarray
    market_t3: np.ndarray
    market_t3_mask: np.ndarray
    self_reserved: np.ndarray
    self_reserved_mask: np.ndarray
    opponent_reserved: np.ndarray
    opponent_reserved_mask: np.ndarray
    nobles: np.ndarray
    nobles_mask: np.ndarray
    pending_card: np.ndarray
    pending_card_mask: np.ndarray


# ---------------------------------------------------------------------------
# Vectorized encoding internals
#
# These helpers sit on the search hot path (feature encoding was ~55% of a
# single search node before this rewrite), so card / noble / action features
# are built with NumPy over whole sets and candidate lists instead of one
# Python list per row.  Arithmetic stays bit-identical to the previous scalar
# implementation: integer quantities remain integral, every division happens
# in float64, and the single float32 cast happens at the end.
# ``tests/fixtures/feature_encoding_golden.json`` pins that equality.
# ---------------------------------------------------------------------------

_CARD_VALID = 0
_CARD_KNOWN = 1
_CARD_TIER = _CARD_KNOWN + 1
_CARD_BONUS = _CARD_TIER + len(TIER_ORDER)
_CARD_SOURCE = _CARD_BONUS + len(GEM_COLORS)
_CARD_POINTS = _CARD_SOURCE + len(SOURCE_ORDER)
_CARD_COST = _CARD_POINTS + 1
_CARD_NEED = _CARD_COST + len(GEM_COLORS)
_CARD_DEFICIT = _CARD_NEED + len(GEM_COLORS)
_CARD_GOLD = _CARD_DEFICIT + len(GEM_COLORS)

_NOBLE_VALID = 0
_NOBLE_POINTS = 1
_NOBLE_REQUIREMENT = _NOBLE_POINTS + 1
_NOBLE_SELF_GAP = _NOBLE_REQUIREMENT + len(GEM_COLORS)
_NOBLE_OPPONENT_GAP = _NOBLE_SELF_GAP + len(GEM_COLORS)

_ACTION_PHASE = 0
_ACTION_TYPE = _ACTION_PHASE + len(PHASE_ORDER)
_ACTION_SOURCE = _ACTION_TYPE + len(ACTION_TYPE_ORDER)
_ACTION_TIER = _ACTION_SOURCE + len(SOURCE_ORDER)
_ACTION_TAKE = _ACTION_TIER + len(TIER_ORDER)
_ACTION_PAYMENT = _ACTION_TAKE + len(TOKEN_COLORS)
_ACTION_RETURN = _ACTION_PAYMENT + len(TOKEN_COLORS)
_ACTION_CARD = _ACTION_RETURN + len(TOKEN_COLORS)
_ACTION_NOBLE = _ACTION_CARD + len(CARD_FEATURE_NAMES)

_TIER_INDEX = {tier: index for index, tier in enumerate(TIER_ORDER)}
_SOURCE_INDEX = {source: index for index, source in enumerate(SOURCE_ORDER)}
_PHASE_INDEX = {phase: index for index, phase in enumerate(PHASE_ORDER)}
_ACTION_TYPE_INDEX = {
    action_type: index for index, action_type in enumerate(ACTION_TYPE_ORDER)
}
_MAIN_ACTION_TYPE_INDEX = {
    action_type: index for index, action_type in enumerate(MAIN_ACTION_TYPE_ORDER)
}
_GEM_INDEX = {color: index for index, color in enumerate(GEM_COLORS)}
_TOKEN_INDEX = {color: index for index, color in enumerate(TOKEN_COLORS)}
_TOKEN_SCALES = np.asarray(SCALES.token_by_color, dtype=np.float64)
_DECK_SCALES = np.asarray(SCALES.deck_by_tier, dtype=np.float64)


def _assert_feature_offsets() -> None:
    """Fail loudly at import time if the schema layout drifts from the code."""

    checks = (
        (CARD_FEATURE_NAMES, _CARD_VALID, "valid"),
        (CARD_FEATURE_NAMES, _CARD_TIER, "tier_1"),
        (CARD_FEATURE_NAMES, _CARD_BONUS, "bonus_white"),
        (CARD_FEATURE_NAMES, _CARD_SOURCE, "source_market"),
        (CARD_FEATURE_NAMES, _CARD_POINTS, "points_scaled"),
        (CARD_FEATURE_NAMES, _CARD_COST, "cost_white_scaled"),
        (CARD_FEATURE_NAMES, _CARD_NEED, "discounted_need_white_scaled"),
        (CARD_FEATURE_NAMES, _CARD_DEFICIT, "ordinary_deficit_white_scaled"),
        (CARD_FEATURE_NAMES, _CARD_GOLD, "gold_uncovered_deficit_scaled"),
        (NOBLE_FEATURE_NAMES, _NOBLE_VALID, "valid"),
        (NOBLE_FEATURE_NAMES, _NOBLE_POINTS, "points_scaled"),
        (NOBLE_FEATURE_NAMES, _NOBLE_REQUIREMENT, "requirement_white_scaled"),
        (NOBLE_FEATURE_NAMES, _NOBLE_SELF_GAP, "self_gap_white_scaled"),
        (NOBLE_FEATURE_NAMES, _NOBLE_OPPONENT_GAP, "opponent_gap_white_scaled"),
        (ACTION_FEATURE_NAMES, _ACTION_PHASE, "phase_main_action"),
        (ACTION_FEATURE_NAMES, _ACTION_TYPE, "type_take_distinct"),
        (ACTION_FEATURE_NAMES, _ACTION_SOURCE, "source_market"),
        (ACTION_FEATURE_NAMES, _ACTION_TIER, "tier_1"),
        (ACTION_FEATURE_NAMES, _ACTION_TAKE, "take_white_scaled"),
        (ACTION_FEATURE_NAMES, _ACTION_PAYMENT, "payment_white_scaled"),
        (ACTION_FEATURE_NAMES, _ACTION_RETURN, "return_white_scaled"),
        (ACTION_FEATURE_NAMES, _ACTION_CARD, "target_card.valid"),
        (ACTION_FEATURE_NAMES, _ACTION_NOBLE, "target_noble.valid"),
    )
    for names, offset, expected in checks:
        if names[offset] != expected:
            raise AssertionError(
                f"feature offset {offset} is {names[offset]!r}, expected {expected!r}"
            )


_assert_feature_offsets()


def _one_hot_row(value: object, order: Sequence[object]) -> np.ndarray:
    row = np.zeros(len(order), dtype=np.float64)
    for index, candidate in enumerate(order):
        if value == candidate:
            row[index] = 1.0
            break
    return row


def _pad_rows(
    matrix: np.ndarray,
    *,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a whole-set feature matrix into the frozen capacity shape."""

    if matrix.shape[0] > capacity:
        raise ValueError(f"set has {matrix.shape[0]} objects but capacity is {capacity}")
    features = np.zeros((capacity, matrix.shape[1]), dtype=np.float32)
    mask = np.zeros(capacity, dtype=np.bool_)
    if matrix.shape[0]:
        features[: matrix.shape[0]] = matrix
        mask[: matrix.shape[0]] = True
    return features, mask


def _card_feature_matrix(
    observation: Observation,
    rows: Sequence[tuple[CardView | None, Tier | None, ReservationSource | None]],
) -> np.ndarray:
    """Encode a whole set of card slots; unknown slots keep tier and source."""

    count = len(rows)
    width = FEATURE_SCHEMA.card_dim
    features = np.zeros((count, width), dtype=np.float64)
    if count == 0:
        return features.astype(np.float32)

    known = np.zeros(count, dtype=np.bool_)
    tier_index = np.full(count, -1, dtype=np.intp)
    source_index = np.full(count, -1, dtype=np.intp)
    cards: list[CardView | None] = []
    for index, (card, tier, source) in enumerate(rows):
        if card is not None:
            if tier is not None and card.tier is not tier:
                raise ValueError("card tier disagrees with its public container")
            known[index] = True
        if tier is not None:
            tier_index[index] = _TIER_INDEX[tier]
        if source is not None:
            source_index[index] = _SOURCE_INDEX[source]
        cards.append(card)

    row_index = np.arange(count)
    # ``valid`` marks a card slot that exists at all; a fully empty placeholder
    # row (no card, no tier, no source) stays entirely zero.
    present = known | (tier_index >= 0) | (source_index >= 0)
    features[row_index[present], _CARD_VALID] = 1.0
    features[known, _CARD_KNOWN] = 1.0
    present = tier_index >= 0
    features[row_index[present], _CARD_TIER + tier_index[present]] = 1.0
    present = source_index >= 0
    features[row_index[present], _CARD_SOURCE + source_index[present]] = 1.0

    known_index = np.flatnonzero(known)
    if known_index.size:
        known_cards = [cards[index] for index in known_index]
        features[
            known_index,
            _CARD_BONUS
            + np.asarray(
                [_GEM_INDEX[card.bonus] for card in known_cards], dtype=np.intp
            ),
        ] = 1.0
        points = np.fromiter(
            (card.points for card in known_cards), dtype=np.int64, count=known_index.size
        )
        costs = np.asarray([tuple(card.cost) for card in known_cards], dtype=np.int64)
        bonuses = np.asarray(tuple(observation.self_public.bonuses), dtype=np.int64)
        held = np.asarray(tuple(observation.self_public.tokens), dtype=np.int64)[
            : len(GEM_COLORS)
        ]
        need = np.maximum(0, costs - bonuses)
        deficit = np.maximum(0, need - held)
        gold_uncovered = np.maximum(
            0, deficit.sum(axis=1) - observation.self_public.tokens.gold
        )
        features[known_index, _CARD_POINTS] = points / SCALES.card_points
        features[
            known_index, _CARD_COST : _CARD_COST + len(GEM_COLORS)
        ] = costs / SCALES.card_cost
        features[
            known_index, _CARD_NEED : _CARD_NEED + len(GEM_COLORS)
        ] = need / SCALES.card_cost
        features[
            known_index, _CARD_DEFICIT : _CARD_DEFICIT + len(GEM_COLORS)
        ] = deficit / SCALES.card_cost
        features[known_index, _CARD_GOLD] = gold_uncovered / SCALES.purchase_gap_total
    return features.astype(np.float32)


def _noble_feature_matrix(
    observation: Observation,
    nobles: Sequence[NobleView],
) -> np.ndarray:
    count = len(nobles)
    width = FEATURE_SCHEMA.noble_dim
    features = np.zeros((count, width), dtype=np.float64)
    if count == 0:
        return features.astype(np.float32)

    points = np.fromiter(
        (noble.points for noble in nobles), dtype=np.int64, count=count
    )
    requirements = np.asarray(
        [tuple(noble.requirement) for noble in nobles], dtype=np.int64
    )
    self_bonuses = np.asarray(tuple(observation.self_public.bonuses), dtype=np.int64)
    opponent_bonuses = np.asarray(
        tuple(observation.opponent_public.bonuses), dtype=np.int64
    )
    features[:, _NOBLE_VALID] = 1.0
    features[:, _NOBLE_POINTS] = points / SCALES.noble_points
    features[
        :, _NOBLE_REQUIREMENT : _NOBLE_REQUIREMENT + len(GEM_COLORS)
    ] = requirements / SCALES.noble_requirement
    features[
        :, _NOBLE_SELF_GAP : _NOBLE_SELF_GAP + len(GEM_COLORS)
    ] = np.maximum(0, requirements - self_bonuses) / SCALES.noble_requirement
    features[
        :, _NOBLE_OPPONENT_GAP : _NOBLE_OPPONENT_GAP + len(GEM_COLORS)
    ] = np.maximum(0, requirements - opponent_bonuses) / SCALES.noble_requirement
    return features.astype(np.float32)


def _market_rows(
    cards: Sequence[CardView],
) -> list[tuple[CardView | None, Tier | None, ReservationSource | None]]:
    return [(card, card.tier, ReservationSource.MARKET) for card in cards]


def _self_reserved_rows(
    observation: Observation,
) -> list[tuple[CardView | None, Tier | None, ReservationSource | None]]:
    rows: list[tuple[CardView | None, Tier | None, ReservationSource | None]] = [
        (reserved.card, reserved.tier, reserved.source)
        for reserved in observation.self_public.reserved_cards
        if reserved.card is not None
    ]
    rows.extend(
        (card, card.tier, ReservationSource.DECK)
        for card in observation.self_private.blind_reserved_cards
    )
    if len(rows) != len(observation.self_public.reserved_cards):
        raise ValueError(
            "private blind reservations disagree with public reservation count"
        )
    return rows


def _opponent_reserved_rows(
    observation: Observation,
) -> list[tuple[CardView | None, Tier | None, ReservationSource | None]]:
    return [
        (reserved.card, reserved.tier, reserved.source)
        for reserved in observation.opponent_public.reserved_cards
    ]


def _global_feature(observation: Observation) -> np.ndarray:
    if len(observation.public_purchased) != FEATURE_SCHEMA.public_purchased_dim:
        raise ValueError(
            f"public_purchased must have {FEATURE_SCHEMA.public_purchased_dim} entries"
        )

    values = np.zeros(FEATURE_SCHEMA.global_dim, dtype=np.float64)
    cursor = 0

    def take(size: int) -> np.ndarray:
        nonlocal cursor
        block = values[cursor : cursor + size]
        cursor += size
        return block

    take(len(TOKEN_COLORS))[:] = (
        np.asarray(tuple(observation.supply), dtype=np.int64) / _TOKEN_SCALES
    )
    take(len(TIER_ORDER))[:] = (
        np.asarray(tuple(observation.deck_counts), dtype=np.int64) / _DECK_SCALES
    )
    for player in (observation.self_public, observation.opponent_public):
        take(len(TOKEN_COLORS))[:] = (
            np.asarray(tuple(player.tokens), dtype=np.int64) / _TOKEN_SCALES
        )
        take(len(GEM_COLORS))[:] = (
            np.asarray(tuple(player.bonuses), dtype=np.int64) / SCALES.player_bonus
        )
        summary = take(3)
        summary[0] = player.score / SCALES.player_score
        summary[1] = len(player.purchased_card_ids) / SCALES.purchased_cards
        summary[2] = len(player.noble_ids) / SCALES.noble_count
    take(FEATURE_SCHEMA.public_purchased_dim)[:] = np.asarray(
        observation.public_purchased, dtype=np.float64
    )
    take(1)[0] = float(observation.self_is_first_player)
    take(len(PHASE_ORDER))[:] = _one_hot_row(observation.phase, PHASE_ORDER)
    take(1)[0] = float(observation.final_round_triggered)
    take(1)[0] = float(observation.final_round_triggered_by_self)
    take(1)[0] = observation.own_completed_turns / SCALES.completed_turns
    take(1)[0] = observation.opponent_completed_turns / SCALES.completed_turns
    take(1)[0] = (
        observation.own_completed_turns - observation.opponent_completed_turns
    ) / SCALES.completed_turn_delta
    take(1)[0] = float(
        observation.own_completed_turns == observation.opponent_completed_turns
    )
    take(1)[0] = (
        observation.consecutive_forced_skips / SCALES.consecutive_forced_skips
    )
    pending = observation.pending
    take(1)[0] = float(pending is not None)
    main_action = take(len(MAIN_ACTION_TYPE_ORDER))
    if pending is not None:
        main_action[_MAIN_ACTION_TYPE_INDEX[type(pending.main_action)]] = 1.0
    tail = take(2)
    if pending is not None:
        tail[0] = pending.tokens_to_return / SCALES.tokens_to_return
        tail[1] = len(pending.eligible_nobles) / SCALES.eligible_nobles
    if cursor != FEATURE_SCHEMA.global_dim:
        raise AssertionError("global feature implementation disagrees with schema")

    result = values.astype(np.float32)
    if result.shape != (FEATURE_SCHEMA.global_dim,):
        raise AssertionError("global feature implementation disagrees with schema")
    return result


def _known_targets(
    observation: Observation,
) -> tuple[dict[str, tuple[CardView, ReservationSource]], dict[str, tuple[CardView, ReservationSource]]]:
    market = {
        card.card_id: (card, ReservationSource.MARKET)
        for cards in (
            observation.market.tier_one,
            observation.market.tier_two,
            observation.market.tier_three,
        )
        for card in cards
    }
    reserved: dict[str, tuple[CardView, ReservationSource]] = {}
    for item in observation.self_public.reserved_cards:
        if item.card is not None:
            reserved[item.card.card_id] = (item.card, item.source)
    for card in observation.self_private.blind_reserved_cards:
        reserved[card.card_id] = (card, ReservationSource.DECK)
    return market, reserved


def _pending_card(
    observation: Observation,
    reserved: dict[str, tuple[CardView, ReservationSource]],
) -> tuple[CardView, ReservationSource] | None:
    pending = observation.pending
    if pending is None or pending.payment_target is None:
        return None
    if isinstance(pending.main_action, BuyMarket):
        return pending.payment_target, ReservationSource.MARKET
    if isinstance(pending.main_action, BuyReserved):
        try:
            _, source = reserved[pending.main_action.card_id]
        except KeyError as error:
            raise ValueError("pending reserved purchase target is not visible to its owner") from error
        return pending.payment_target, source
    raise ValueError("payment target exists outside a purchase pending context")


def _state_arrays(
    observation: Observation,
    *,
    reserved: dict[str, tuple[CardView, ReservationSource]] | None = None,
) -> dict[str, np.ndarray]:
    """Build every state-only field of an encoded decision."""

    if reserved is None:
        _, reserved = _known_targets(observation)
    pending_target = _pending_card(observation, reserved)
    # One vectorized call for every card slot in the state; the per-set NumPy
    # call overhead would otherwise dominate these tiny blocks.
    market_counts = (
        len(observation.market.tier_one),
        len(observation.market.tier_two),
        len(observation.market.tier_three),
    )
    self_rows = _self_reserved_rows(observation)
    opponent_rows = _opponent_reserved_rows(observation)
    pending_rows = (
        []
        if pending_target is None
        else [(pending_target[0], pending_target[0].tier, pending_target[1])]
    )
    card_rows = (
        _market_rows(observation.market.tier_one)
        + _market_rows(observation.market.tier_two)
        + _market_rows(observation.market.tier_three)
        + self_rows
        + opponent_rows
        + pending_rows
    )
    card_matrix = _card_feature_matrix(observation, card_rows)
    cursor = 0

    def take_cards(count: int) -> np.ndarray:
        nonlocal cursor
        block = card_matrix[cursor : cursor + count]
        cursor += count
        return block

    market_t1, market_t1_mask = _pad_rows(
        take_cards(market_counts[0]), capacity=FEATURE_SCHEMA.market_capacity
    )
    market_t2, market_t2_mask = _pad_rows(
        take_cards(market_counts[1]), capacity=FEATURE_SCHEMA.market_capacity
    )
    market_t3, market_t3_mask = _pad_rows(
        take_cards(market_counts[2]), capacity=FEATURE_SCHEMA.market_capacity
    )
    self_reserved, self_reserved_mask = _pad_rows(
        take_cards(len(self_rows)),
        capacity=FEATURE_SCHEMA.reserved_capacity,
    )
    opponent_reserved, opponent_reserved_mask = _pad_rows(
        take_cards(len(opponent_rows)),
        capacity=FEATURE_SCHEMA.reserved_capacity,
    )
    nobles, nobles_mask = _pad_rows(
        _noble_feature_matrix(observation, observation.nobles_in_play),
        capacity=FEATURE_SCHEMA.noble_capacity,
    )
    pending_card = (
        take_cards(1)[0]
        if pending_target is not None
        else np.zeros(FEATURE_SCHEMA.card_dim, dtype=np.float32)
    )
    return {
        "global_features": _global_feature(observation),
        "market_t1": market_t1,
        "market_t1_mask": market_t1_mask,
        "market_t2": market_t2,
        "market_t2_mask": market_t2_mask,
        "market_t3": market_t3,
        "market_t3_mask": market_t3_mask,
        "self_reserved": self_reserved,
        "self_reserved_mask": self_reserved_mask,
        "opponent_reserved": opponent_reserved,
        "opponent_reserved_mask": opponent_reserved_mask,
        "nobles": nobles,
        "nobles_mask": nobles_mask,
        "pending_card": pending_card,
        "pending_card_mask": np.asarray(pending_target is not None, dtype=np.bool_),
    }


def _candidate_feature_matrix(
    observation: Observation,
    actions: Sequence[Action],
    market: dict[str, tuple[CardView, ReservationSource]],
    reserved: dict[str, tuple[CardView, ReservationSource]],
    pending_card: tuple[CardView, ReservationSource] | None,
) -> np.ndarray:
    count = len(actions)
    width = FEATURE_SCHEMA.action_dim
    features = np.zeros((count, width), dtype=np.float64)
    if count == 0:
        raise ValueError("cannot encode a decision without legal candidates")

    row_index = np.arange(count)
    features[:, _ACTION_PHASE + _PHASE_INDEX[observation.phase]] = 1.0
    features[
        row_index,
        _ACTION_TYPE
        + np.asarray(
            [_ACTION_TYPE_INDEX[type(action)] for action in actions], dtype=np.intp
        ),
    ] = 1.0

    source_index = np.full(count, -1, dtype=np.intp)
    tier_index = np.full(count, -1, dtype=np.intp)
    take = np.zeros((count, len(TOKEN_COLORS)), dtype=np.int64)
    payment = np.zeros((count, len(TOKEN_COLORS)), dtype=np.int64)
    returned = np.zeros((count, len(TOKEN_COLORS)), dtype=np.int64)
    card_rows: list[tuple[CardView | None, Tier | None, ReservationSource | None]] = []
    noble_rows: list[NobleView | None] = []

    for index, action in enumerate(actions):
        action_type = type(action)
        target_card: tuple[CardView, ReservationSource] | None = None
        target_noble: NobleView | None = None
        source: ReservationSource | None = None
        tier: Tier | None = None

        if isinstance(action, TakeDistinct):
            for color in action.colors:
                take[index, _TOKEN_INDEX[color]] = 1
        elif isinstance(action, TakePair):
            take[index, _TOKEN_INDEX[action.color]] = 2
        elif isinstance(action, (ReserveMarket, BuyMarket)):
            try:
                target_card = market[action.card_id]
            except KeyError as error:
                raise ValueError("market action target is absent from the public market") from error
            source = ReservationSource.MARKET
            tier = target_card[0].tier
        elif isinstance(action, ReserveDeck):
            source = ReservationSource.DECK
            tier = action.tier
        elif isinstance(action, BuyReserved):
            try:
                target_card = reserved[action.card_id]
            except KeyError as error:
                raise ValueError("reserved purchase target is not visible to its owner") from error
            source = target_card[1]
            tier = target_card[0].tier
        elif isinstance(action, Pay):
            payment[index, :] = tuple(action.tokens)
            target_card = pending_card
            if target_card is None:
                raise ValueError("payment candidate has no visible pending target")
            source = target_card[1]
            tier = target_card[0].tier
        elif isinstance(action, ReturnTokens):
            returned[index, :] = tuple(action.tokens)
        elif isinstance(action, ClaimNoble):
            nobles = (
                observation.pending.eligible_nobles
                if observation.pending is not None
                else observation.nobles_in_play
            )
            try:
                target_noble = next(
                    noble for noble in nobles if noble.noble_id == action.noble_id
                )
            except StopIteration as error:
                raise ValueError("noble action target is absent from public pending context") from error
        else:  # pragma: no cover - Action is a closed union
            raise TypeError(f"unknown action type: {action_type.__name__}")

        if source is not None:
            source_index[index] = _SOURCE_INDEX[source]
        if tier is not None:
            tier_index[index] = _TIER_INDEX[tier]
        card_rows.append(
            (None, None, None)
            if target_card is None
            else (target_card[0], target_card[0].tier, target_card[1])
        )
        noble_rows.append(target_noble)

    present = source_index >= 0
    features[row_index[present], _ACTION_SOURCE + source_index[present]] = 1.0
    present = tier_index >= 0
    features[row_index[present], _ACTION_TIER + tier_index[present]] = 1.0
    features[:, _ACTION_TAKE : _ACTION_TAKE + len(TOKEN_COLORS)] = take / _TOKEN_SCALES
    features[
        :, _ACTION_PAYMENT : _ACTION_PAYMENT + len(TOKEN_COLORS)
    ] = payment / _TOKEN_SCALES
    features[
        :, _ACTION_RETURN : _ACTION_RETURN + len(TOKEN_COLORS)
    ] = returned / _TOKEN_SCALES
    features[
        :, _ACTION_CARD : _ACTION_CARD + len(CARD_FEATURE_NAMES)
    ] = _card_feature_matrix(observation, card_rows)
    noble_present = [index for index, noble in enumerate(noble_rows) if noble is not None]
    if noble_present:
        features[
            np.asarray(noble_present), _ACTION_NOBLE : _ACTION_NOBLE + len(NOBLE_FEATURE_NAMES)
        ] = _noble_feature_matrix(
            observation, [noble for noble in noble_rows if noble is not None]
        )

    result = features.astype(np.float32)
    if result.shape != (count, FEATURE_SCHEMA.action_dim):
        raise AssertionError("action feature implementation disagrees with schema")
    return result


def encode_decision_state(request: DecisionRequest) -> EncodedState:
    """Encode only the state part of one public decision.

    The candidate list is never read, so this is the information-safe path for
    critic-only evaluation of search leaves.  Pending turn context (payment
    target, tokens to return, eligible nobles) is part of the state encoding and
    is therefore preserved exactly as :func:`encode_decision` encodes it.
    """

    return EncodedState(
        feature_schema_version=FEATURE_SCHEMA.version,
        **_state_arrays(request.observation),
    )


def encode_decision(request: DecisionRequest) -> EncodedDecision:
    """Encode one public decision without checking or changing legality."""

    observation = request.observation
    market, reserved = _known_targets(observation)
    pending_target = _pending_card(observation, reserved)
    candidate_features = _candidate_feature_matrix(
        observation,
        request.legal_actions,
        market,
        reserved,
        pending_target,
    )
    return EncodedDecision(
        feature_schema_version=FEATURE_SCHEMA.version,
        **_state_arrays(observation, reserved=reserved),
        candidate_features=candidate_features,
        candidate_mask=np.ones(len(request.legal_actions), dtype=np.bool_),
        candidate_actions=request.legal_actions,
    )


def collate_encoded_decisions(
    decisions: Sequence[EncodedDecision],
) -> EncodedDecisionBatch:
    """Pad candidates only to this batch's maximum ragged length."""

    if not decisions:
        raise ValueError("cannot collate an empty decision batch")
    if any(item.feature_schema_version != FEATURE_SCHEMA.version for item in decisions):
        raise ValueError("feature schema version mismatch in batch")

    batch_size = len(decisions)
    max_candidates = max(len(item.candidate_actions) for item in decisions)
    candidates = np.zeros(
        (batch_size, max_candidates, FEATURE_SCHEMA.action_dim),
        dtype=np.float32,
    )
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    for index, item in enumerate(decisions):
        count = item.candidate_features.shape[0]
        candidates[index, :count] = item.candidate_features
        candidate_mask[index, :count] = item.candidate_mask

    def stack(name: str) -> np.ndarray:
        return np.stack([getattr(item, name) for item in decisions])

    return EncodedDecisionBatch(
        feature_schema_version=FEATURE_SCHEMA.version,
        global_features=stack("global_features"),
        market_t1=stack("market_t1"),
        market_t1_mask=stack("market_t1_mask"),
        market_t2=stack("market_t2"),
        market_t2_mask=stack("market_t2_mask"),
        market_t3=stack("market_t3"),
        market_t3_mask=stack("market_t3_mask"),
        self_reserved=stack("self_reserved"),
        self_reserved_mask=stack("self_reserved_mask"),
        opponent_reserved=stack("opponent_reserved"),
        opponent_reserved_mask=stack("opponent_reserved_mask"),
        nobles=stack("nobles"),
        nobles_mask=stack("nobles_mask"),
        pending_card=stack("pending_card"),
        pending_card_mask=stack("pending_card_mask"),
        candidate_features=candidates,
        candidate_mask=candidate_mask,
    )


def collate_encoded_states(
    states: Sequence[EncodedState],
) -> EncodedStateBatch:
    """Stack state-only encodings without touching any candidate list."""

    if not states:
        raise ValueError("cannot collate an empty state batch")
    if any(item.feature_schema_version != FEATURE_SCHEMA.version for item in states):
        raise ValueError("feature schema version mismatch in batch")

    def stack(name: str) -> np.ndarray:
        return np.stack([getattr(item, name) for item in states])

    return EncodedStateBatch(
        feature_schema_version=FEATURE_SCHEMA.version,
        global_features=stack("global_features"),
        market_t1=stack("market_t1"),
        market_t1_mask=stack("market_t1_mask"),
        market_t2=stack("market_t2"),
        market_t2_mask=stack("market_t2_mask"),
        market_t3=stack("market_t3"),
        market_t3_mask=stack("market_t3_mask"),
        self_reserved=stack("self_reserved"),
        self_reserved_mask=stack("self_reserved_mask"),
        opponent_reserved=stack("opponent_reserved"),
        opponent_reserved_mask=stack("opponent_reserved_mask"),
        nobles=stack("nobles"),
        nobles_mask=stack("nobles_mask"),
        pending_card=stack("pending_card"),
        pending_card_mask=stack("pending_card_mask"),
    )
