"""Information-safe NumPy features for observations and legal candidates.

This module is deliberately downstream of the public policy contract.  It
does not import the engine, authoritative state, rules, RNG, or replay code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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
    TokenVector,
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


def _one_hot(value: object, order: Sequence[object]) -> list[float]:
    return [1.0 if value == candidate else 0.0 for candidate in order]


def _scaled(values: Iterable[int], scales: Sequence[float]) -> list[float]:
    return [value / scale for value, scale in zip(values, scales, strict=True)]


def _card_feature(
    observation: Observation,
    card: CardView | None,
    *,
    tier: Tier,
    source: ReservationSource,
) -> np.ndarray:
    known = card is not None
    if known and card.tier is not tier:
        raise ValueError("card tier disagrees with its public container")

    values: list[float] = [1.0, float(known)]
    values.extend(_one_hot(tier, TIER_ORDER))
    values.extend(
        _one_hot(card.bonus, GEM_COLORS) if card is not None else [0.0] * len(GEM_COLORS)
    )
    values.extend(_one_hot(source, SOURCE_ORDER))
    if card is None:
        values.extend([0.0] * (FEATURE_SCHEMA.card_dim - len(values)))
        return np.asarray(values, dtype=np.float32)

    costs = tuple(card.cost)
    bonuses = tuple(observation.self_public.bonuses)
    held = tuple(observation.self_public.tokens)[: len(GEM_COLORS)]
    need = tuple(max(0, cost - bonus) for cost, bonus in zip(costs, bonuses, strict=True))
    deficit = tuple(max(0, required - token) for required, token in zip(need, held, strict=True))
    gold_uncovered = max(0, sum(deficit) - observation.self_public.tokens.gold)
    values.append(card.points / SCALES.card_points)
    values.extend(cost / SCALES.card_cost for cost in costs)
    values.extend(required / SCALES.card_cost for required in need)
    values.extend(shortage / SCALES.card_cost for shortage in deficit)
    values.append(gold_uncovered / SCALES.purchase_gap_total)
    return np.asarray(values, dtype=np.float32)


def _noble_feature(observation: Observation, noble: NobleView) -> np.ndarray:
    requirements = tuple(noble.requirement)
    self_gap = tuple(
        max(0, need - have)
        for need, have in zip(requirements, observation.self_public.bonuses, strict=True)
    )
    opponent_gap = tuple(
        max(0, need - have)
        for need, have in zip(requirements, observation.opponent_public.bonuses, strict=True)
    )
    values = [1.0, noble.points / SCALES.noble_points]
    values.extend(value / SCALES.noble_requirement for value in requirements)
    values.extend(value / SCALES.noble_requirement for value in self_gap)
    values.extend(value / SCALES.noble_requirement for value in opponent_gap)
    return np.asarray(values, dtype=np.float32)


def _encode_set(
    rows: Sequence[np.ndarray],
    *,
    capacity: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) > capacity:
        raise ValueError(f"set has {len(rows)} objects but capacity is {capacity}")
    features = np.zeros((capacity, width), dtype=np.float32)
    mask = np.zeros(capacity, dtype=np.bool_)
    if rows:
        features[: len(rows)] = np.stack(rows)
        mask[: len(rows)] = True
    return features, mask


def _market_rows(observation: Observation, cards: Sequence[CardView]) -> list[np.ndarray]:
    return [
        _card_feature(
            observation,
            card,
            tier=card.tier,
            source=ReservationSource.MARKET,
        )
        for card in cards
    ]


def _self_reserved_rows(observation: Observation) -> list[np.ndarray]:
    rows = [
        _card_feature(
            observation,
            reserved.card,
            tier=reserved.tier,
            source=reserved.source,
        )
        for reserved in observation.self_public.reserved_cards
        if reserved.card is not None
    ]
    rows.extend(
        _card_feature(
            observation,
            card,
            tier=card.tier,
            source=ReservationSource.DECK,
        )
        for card in observation.self_private.blind_reserved_cards
    )
    if len(rows) != len(observation.self_public.reserved_cards):
        raise ValueError("private blind reservations disagree with public reservation count")
    return rows


def _opponent_reserved_rows(observation: Observation) -> list[np.ndarray]:
    return [
        _card_feature(
            observation,
            reserved.card,
            tier=reserved.tier,
            source=reserved.source,
        )
        for reserved in observation.opponent_public.reserved_cards
    ]


def _global_feature(observation: Observation) -> np.ndarray:
    if len(observation.public_purchased) != FEATURE_SCHEMA.public_purchased_dim:
        raise ValueError(
            f"public_purchased must have {FEATURE_SCHEMA.public_purchased_dim} entries"
        )

    values: list[float] = []
    values.extend(_scaled(observation.supply, SCALES.token_by_color))
    values.extend(_scaled(observation.deck_counts, SCALES.deck_by_tier))
    for player in (observation.self_public, observation.opponent_public):
        values.extend(_scaled(player.tokens, SCALES.token_by_color))
        values.extend(value / SCALES.player_bonus for value in player.bonuses)
        values.extend(
            (
                player.score / SCALES.player_score,
                len(player.purchased_card_ids) / SCALES.purchased_cards,
                len(player.noble_ids) / SCALES.noble_count,
            )
        )
    values.extend(float(value) for value in observation.public_purchased)
    values.append(float(observation.self_is_first_player))
    values.extend(_one_hot(observation.phase, PHASE_ORDER))
    values.extend(
        (
            float(observation.final_round_triggered),
            float(observation.final_round_triggered_by_self),
            observation.own_completed_turns / SCALES.completed_turns,
            observation.opponent_completed_turns / SCALES.completed_turns,
            (observation.own_completed_turns - observation.opponent_completed_turns)
            / SCALES.completed_turn_delta,
            float(observation.own_completed_turns == observation.opponent_completed_turns),
            observation.consecutive_forced_skips / SCALES.consecutive_forced_skips,
        )
    )
    pending = observation.pending
    values.append(float(pending is not None))
    values.extend(
        _one_hot(type(pending.main_action), MAIN_ACTION_TYPE_ORDER)
        if pending is not None
        else [0.0] * len(MAIN_ACTION_TYPE_ORDER)
    )
    values.extend(
        (
            0.0 if pending is None else pending.tokens_to_return / SCALES.tokens_to_return,
            0.0 if pending is None else len(pending.eligible_nobles) / SCALES.eligible_nobles,
        )
    )
    result = np.asarray(values, dtype=np.float32)
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


def _action_feature(
    observation: Observation,
    action: Action,
    market: dict[str, tuple[CardView, ReservationSource]],
    reserved: dict[str, tuple[CardView, ReservationSource]],
    pending_card: tuple[CardView, ReservationSource] | None,
) -> np.ndarray:
    action_type = type(action)
    values: list[float] = []
    values.extend(_one_hot(observation.phase, PHASE_ORDER))
    values.extend(_one_hot(action_type, ACTION_TYPE_ORDER))

    target_card: tuple[CardView, ReservationSource] | None = None
    target_noble: NobleView | None = None
    source: ReservationSource | None = None
    tier: Tier | None = None
    take = TokenVector()
    payment = TokenVector()
    returned = TokenVector()

    if isinstance(action, TakeDistinct):
        take = TokenVector(**{color.value: 1 for color in action.colors})
    elif isinstance(action, TakePair):
        take = TokenVector(**{action.color.value: 2})
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
        payment = action.tokens
        target_card = pending_card
        if target_card is None:
            raise ValueError("payment candidate has no visible pending target")
        source = target_card[1]
        tier = target_card[0].tier
    elif isinstance(action, ReturnTokens):
        returned = action.tokens
    elif isinstance(action, ClaimNoble):
        nobles = (
            observation.pending.eligible_nobles
            if observation.pending is not None
            else observation.nobles_in_play
        )
        try:
            target_noble = next(noble for noble in nobles if noble.noble_id == action.noble_id)
        except StopIteration as error:
            raise ValueError("noble action target is absent from public pending context") from error
    else:  # pragma: no cover - Action is a closed union
        raise TypeError(f"unknown action type: {action_type.__name__}")

    values.extend(_one_hot(source, SOURCE_ORDER))
    values.extend(_one_hot(tier, TIER_ORDER))
    values.extend(_scaled(take, SCALES.token_by_color))
    values.extend(_scaled(payment, SCALES.token_by_color))
    values.extend(_scaled(returned, SCALES.token_by_color))
    values.extend(
        _card_feature(
            observation,
            target_card[0],
            tier=target_card[0].tier,
            source=target_card[1],
        )
        if target_card is not None
        else np.zeros(FEATURE_SCHEMA.card_dim, dtype=np.float32)
    )
    values.extend(
        _noble_feature(observation, target_noble)
        if target_noble is not None
        else np.zeros(FEATURE_SCHEMA.noble_dim, dtype=np.float32)
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_SCHEMA.action_dim,):
        raise AssertionError("action feature implementation disagrees with schema")
    return result


def encode_decision(request: DecisionRequest) -> EncodedDecision:
    """Encode one public decision without checking or changing legality."""

    observation = request.observation
    market, reserved = _known_targets(observation)
    pending_target = _pending_card(observation, reserved)

    market_t1, market_t1_mask = _encode_set(
        _market_rows(observation, observation.market.tier_one),
        capacity=FEATURE_SCHEMA.market_capacity,
        width=FEATURE_SCHEMA.card_dim,
    )
    market_t2, market_t2_mask = _encode_set(
        _market_rows(observation, observation.market.tier_two),
        capacity=FEATURE_SCHEMA.market_capacity,
        width=FEATURE_SCHEMA.card_dim,
    )
    market_t3, market_t3_mask = _encode_set(
        _market_rows(observation, observation.market.tier_three),
        capacity=FEATURE_SCHEMA.market_capacity,
        width=FEATURE_SCHEMA.card_dim,
    )
    self_reserved, self_reserved_mask = _encode_set(
        _self_reserved_rows(observation),
        capacity=FEATURE_SCHEMA.reserved_capacity,
        width=FEATURE_SCHEMA.card_dim,
    )
    opponent_reserved, opponent_reserved_mask = _encode_set(
        _opponent_reserved_rows(observation),
        capacity=FEATURE_SCHEMA.reserved_capacity,
        width=FEATURE_SCHEMA.card_dim,
    )
    nobles, nobles_mask = _encode_set(
        [_noble_feature(observation, noble) for noble in observation.nobles_in_play],
        capacity=FEATURE_SCHEMA.noble_capacity,
        width=FEATURE_SCHEMA.noble_dim,
    )
    pending_card_feature = (
        _card_feature(
            observation,
            pending_target[0],
            tier=pending_target[0].tier,
            source=pending_target[1],
        )
        if pending_target is not None
        else np.zeros(FEATURE_SCHEMA.card_dim, dtype=np.float32)
    )
    candidate_features = np.stack(
        [
            _action_feature(observation, action, market, reserved, pending_target)
            for action in request.legal_actions
        ]
    ).astype(np.float32, copy=False)

    return EncodedDecision(
        feature_schema_version=FEATURE_SCHEMA.version,
        global_features=_global_feature(observation),
        market_t1=market_t1,
        market_t1_mask=market_t1_mask,
        market_t2=market_t2,
        market_t2_mask=market_t2_mask,
        market_t3=market_t3,
        market_t3_mask=market_t3_mask,
        self_reserved=self_reserved,
        self_reserved_mask=self_reserved_mask,
        opponent_reserved=opponent_reserved,
        opponent_reserved_mask=opponent_reserved_mask,
        nobles=nobles,
        nobles_mask=nobles_mask,
        pending_card=pending_card_feature,
        pending_card_mask=np.asarray(pending_target is not None, dtype=np.bool_),
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
