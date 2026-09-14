from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
from .model import Color, GEM_COLORS, TokenVector


_COLOR_INDEX = {color: index for index, color in enumerate(GEM_COLORS)}
_ACTION_ORDER = {
    BuyMarket: 0,
    BuyReserved: 1,
    TakeDistinct: 2,
    TakePair: 3,
    ReserveMarket: 4,
    ReserveDeck: 5,
    Pay: 6,
    ReturnTokens: 7,
    ClaimNoble: 8,
}


@dataclass(frozen=True, slots=True)
class ScorePolicyWeights:
    version: str = "score-v1"
    buy_base: int = 100_000
    buy_point: int = 10_000
    buy_printed_cost: int = -25
    reserve_market_base: int = 22_000
    reserve_point: int = 4_500
    reserve_shortage: int = -100
    reserve_threat_point: int = 250
    reserve_deck_base: int = 12_000
    reserve_deck_tier: int = 1_000
    take_base: int = 30_000
    take_color_value: int = 100
    payment_gold_cost: int = 500
    payment_color_cost: int = 10
    return_gold_cost: int = 1_000
    return_color_cost: int = 20
    noble_point: int = 1_000


@dataclass(frozen=True, slots=True)
class DevelopmentPolicyWeights:
    version: str = "development-v1"
    buy_base: int = 60_000
    point_early: int = 250
    point_per_current_score: int = 280
    noble_bonus: int = 2_200
    balance_bonus: int = 500
    cheap_card: int = 350
    early_low_tier: int = 120
    reserve_market_base: int = 28_000
    reserve_value_percent: int = 55
    reserve_opponent_threat: int = 350
    reserve_deck_base: int = 15_000
    reserve_deck_stage: int = 800
    take_base: int = 25_000
    take_color_value: int = 120
    payment_gold_cost: int = 700
    payment_color_cost: int = 12
    return_gold_cost: int = 1_200
    return_color_cost: int = 25
    noble_point: int = 1_000
    noble_denial: int = 400


SCORE_POLICY_WEIGHTS = ScorePolicyWeights()
DEVELOPMENT_POLICY_WEIGHTS = DevelopmentPolicyWeights()


def _market_cards(observation: Observation) -> tuple[CardView, ...]:
    market = observation.market
    return market.tier_one + market.tier_two + market.tier_three


def _own_reserved_cards(observation: Observation) -> tuple[CardView, ...]:
    public = tuple(
        reserved.card
        for reserved in observation.self_public.reserved_cards
        if reserved.card is not None
    )
    return public + observation.self_private.blind_reserved_cards


def _visible_targets(observation: Observation) -> tuple[CardView, ...]:
    return _market_cards(observation) + _own_reserved_cards(observation)


def _card_for_action(observation: Observation, action: Action) -> CardView:
    if not isinstance(action, (BuyMarket, BuyReserved, ReserveMarket)):
        raise TypeError(f"action has no visible card target: {type(action).__name__}")
    for card in _visible_targets(observation):
        if card.card_id == action.card_id:
            return card
    raise ValueError(f"legal action target is absent from Observation: {action.card_id}")


def _noble_for_action(observation: Observation, action: ClaimNoble) -> NobleView:
    candidates = observation.nobles_in_play
    if observation.pending is not None:
        candidates += observation.pending.eligible_nobles
    for noble in candidates:
        if noble.noble_id == action.noble_id:
            return noble
    raise ValueError(f"legal noble target is absent from Observation: {action.noble_id}")


def _discounted_cost(card: CardView, observation: Observation) -> tuple[int, ...]:
    return tuple(
        max(0, cost - bonus)
        for cost, bonus in zip(card.cost, observation.self_public.bonuses, strict=True)
    )


def _ordinary_shortage(card: CardView, observation: Observation) -> tuple[int, ...]:
    return tuple(
        max(0, need - held)
        for need, held in zip(
            _discounted_cost(card, observation),
            tuple(observation.self_public.tokens)[:-1],
            strict=True,
        )
    )


def _card_semantic_key(card: CardView) -> tuple[object, ...]:
    return (
        card.tier.value,
        _COLOR_INDEX[card.bonus],
        card.points,
        tuple(card.cost),
        card.card_id,
    )


def _noble_semantic_key(noble: NobleView) -> tuple[object, ...]:
    return (noble.points, tuple(noble.requirement), noble.noble_id)


def _action_semantic_key(observation: Observation, action: Action) -> tuple[object, ...]:
    order = _ACTION_ORDER[type(action)]
    if isinstance(action, (BuyMarket, BuyReserved, ReserveMarket)):
        detail: tuple[object, ...] = _card_semantic_key(_card_for_action(observation, action))
    elif isinstance(action, ReserveDeck):
        detail = (action.tier.value,)
    elif isinstance(action, TakeDistinct):
        detail = tuple(_COLOR_INDEX[color] for color in action.colors)
    elif isinstance(action, TakePair):
        detail = (_COLOR_INDEX[action.color],)
    elif isinstance(action, (Pay, ReturnTokens)):
        detail = tuple(action.tokens)
    elif isinstance(action, ClaimNoble):
        detail = _noble_semantic_key(_noble_for_action(observation, action))
    else:  # pragma: no cover - Action is a closed union
        raise TypeError(f"unknown action type: {type(action).__name__}")
    return (order, *detail)


def _choose_highest(
    request: DecisionRequest,
    score_action: Callable[[Action], int],
) -> Action:
    ordered = sorted(
        request.legal_actions,
        key=lambda action: _action_semantic_key(request.observation, action),
    )
    best = ordered[0]
    best_score = score_action(best)
    for action in ordered[1:]:
        score = score_action(action)
        if score > best_score:
            best = action
            best_score = score
    return best


def _score_color_values(observation: Observation) -> tuple[int, ...]:
    cards = _visible_targets(observation)
    values = [1] * len(GEM_COLORS)
    if not cards:
        return tuple(values)
    target = min(
        cards,
        key=lambda card: (
            -card.points,
            sum(_ordinary_shortage(card, observation)),
            sum(card.cost),
            card.tier.value,
            _card_semantic_key(card),
        ),
    )
    for index, shortage in enumerate(_ordinary_shortage(target, observation)):
        values[index] += 12 * shortage
    for card in cards:
        if card.points <= 0:
            continue
        for index, shortage in enumerate(_ordinary_shortage(card, observation)):
            values[index] += (card.points + 1) * shortage
    return tuple(values)


def _development_card_value(
    card: CardView,
    observation: Observation,
    weights: DevelopmentPolicyWeights,
) -> int:
    bonuses = tuple(observation.self_public.bonuses)
    bonus_index = _COLOR_INDEX[card.bonus]
    noble_relevance = sum(
        1
        for noble in observation.nobles_in_play
        if noble.requirement.for_color(card.bonus) > bonuses[bonus_index]
    )
    balance_gain = max(bonuses) - bonuses[bonus_index]
    score = observation.self_public.score
    point_weight = weights.point_early + weights.point_per_current_score * min(score, 15)
    cheapness = max(0, 15 - sum(card.cost))
    early = max(0, 8 - score)
    return (
        card.points * point_weight
        + noble_relevance * weights.noble_bonus
        + balance_gain * weights.balance_bonus
        + cheapness * weights.cheap_card
        + (4 - card.tier.value) * early * weights.early_low_tier
    )


def _development_color_values(
    observation: Observation,
    weights: DevelopmentPolicyWeights,
) -> tuple[int, ...]:
    values = [1] * len(GEM_COLORS)
    ranked = sorted(
        _visible_targets(observation),
        key=lambda card: (-_development_card_value(card, observation, weights), _card_semantic_key(card)),
    )
    for rank, card in enumerate(ranked[:5]):
        importance = 5 - rank
        for index, shortage in enumerate(_ordinary_shortage(card, observation)):
            values[index] += importance * shortage
    return tuple(values)


def _token_vector_cost(
    tokens: TokenVector,
    color_values: tuple[int, ...],
    *,
    gold_cost: int,
    color_cost: int,
) -> int:
    values = tuple(tokens)
    return values[-1] * gold_cost + sum(
        amount * usefulness * color_cost
        for amount, usefulness in zip(values[:-1], color_values, strict=True)
    )


class ScorePolicy:
    """Deterministic short-horizon baseline biased toward immediate prestige."""

    __slots__ = ("weights",)

    def __init__(self, weights: ScorePolicyWeights = SCORE_POLICY_WEIGHTS) -> None:
        self.weights = weights

    def choose(self, request: DecisionRequest) -> Action:
        color_values = _score_color_values(request.observation)
        return _choose_highest(
            request,
            lambda action: self._score(request.observation, action, color_values),
        )

    def _score(
        self,
        observation: Observation,
        action: Action,
        color_values: tuple[int, ...],
    ) -> int:
        weights = self.weights
        if isinstance(action, (BuyMarket, BuyReserved)):
            card = _card_for_action(observation, action)
            return (
                weights.buy_base
                + card.points * weights.buy_point
                + sum(card.cost) * weights.buy_printed_cost
            )
        if isinstance(action, ReserveMarket):
            card = _card_for_action(observation, action)
            return (
                weights.reserve_market_base
                + card.points * weights.reserve_point
                + sum(_ordinary_shortage(card, observation)) * weights.reserve_shortage
                + observation.opponent_public.score * card.points * weights.reserve_threat_point
            )
        if isinstance(action, ReserveDeck):
            return weights.reserve_deck_base + action.tier.value * weights.reserve_deck_tier
        if isinstance(action, TakeDistinct):
            return weights.take_base + weights.take_color_value * sum(
                color_values[_COLOR_INDEX[color]] for color in action.colors
            )
        if isinstance(action, TakePair):
            return weights.take_base + 2 * weights.take_color_value * color_values[_COLOR_INDEX[action.color]]
        if isinstance(action, Pay):
            return -_token_vector_cost(
                action.tokens,
                color_values,
                gold_cost=weights.payment_gold_cost,
                color_cost=weights.payment_color_cost,
            )
        if isinstance(action, ReturnTokens):
            return -_token_vector_cost(
                action.tokens,
                color_values,
                gold_cost=weights.return_gold_cost,
                color_cost=weights.return_color_cost,
            )
        if isinstance(action, ClaimNoble):
            return _noble_for_action(observation, action).points * weights.noble_point
        raise TypeError(f"unknown action type: {type(action).__name__}")


class DevelopmentPolicy:
    """Deterministic engine-building baseline biased toward bonuses and nobles."""

    __slots__ = ("weights",)

    def __init__(
        self,
        weights: DevelopmentPolicyWeights = DEVELOPMENT_POLICY_WEIGHTS,
    ) -> None:
        self.weights = weights

    def choose(self, request: DecisionRequest) -> Action:
        color_values = _development_color_values(request.observation, self.weights)
        return _choose_highest(
            request,
            lambda action: self._score(request.observation, action, color_values),
        )

    def _score(
        self,
        observation: Observation,
        action: Action,
        color_values: tuple[int, ...],
    ) -> int:
        weights = self.weights
        if isinstance(action, (BuyMarket, BuyReserved)):
            card = _card_for_action(observation, action)
            return weights.buy_base + _development_card_value(card, observation, weights)
        if isinstance(action, ReserveMarket):
            card = _card_for_action(observation, action)
            development = _development_card_value(card, observation, weights)
            return (
                weights.reserve_market_base
                + development * weights.reserve_value_percent // 100
                + observation.opponent_public.score * card.points * weights.reserve_opponent_threat
            )
        if isinstance(action, ReserveDeck):
            if observation.self_public.score < 8:
                stage_value = 4 - action.tier.value
            else:
                stage_value = action.tier.value
            return weights.reserve_deck_base + stage_value * weights.reserve_deck_stage
        if isinstance(action, TakeDistinct):
            return weights.take_base + weights.take_color_value * sum(
                color_values[_COLOR_INDEX[color]] for color in action.colors
            )
        if isinstance(action, TakePair):
            return weights.take_base + 2 * weights.take_color_value * color_values[_COLOR_INDEX[action.color]]
        if isinstance(action, Pay):
            return -_token_vector_cost(
                action.tokens,
                color_values,
                gold_cost=weights.payment_gold_cost,
                color_cost=weights.payment_color_cost,
            )
        if isinstance(action, ReturnTokens):
            return -_token_vector_cost(
                action.tokens,
                color_values,
                gold_cost=weights.return_gold_cost,
                color_cost=weights.return_color_cost,
            )
        if isinstance(action, ClaimNoble):
            noble = _noble_for_action(observation, action)
            opponent_gap = sum(
                max(0, need - have)
                for need, have in zip(
                    noble.requirement,
                    observation.opponent_public.bonuses,
                    strict=True,
                )
            )
            return noble.points * weights.noble_point - opponent_gap * weights.noble_denial
        raise TypeError(f"unknown action type: {type(action).__name__}")
