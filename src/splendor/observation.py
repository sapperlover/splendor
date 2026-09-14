from __future__ import annotations

from ._state import GameState, PlayerState, ReservedCard
from .actions import BuyMarket, BuyReserved
from .catalog import CardDefinition, Catalog, NobleDefinition
from .contracts import (
    CardView,
    MarketView,
    NobleView,
    Observation,
    PendingObservation,
    PlayerPrivateState,
    PlayerPublicState,
    PublicReservedCard,
)
from .model import ReservationSource, Tier
from .rules import player_bonuses, player_score


def _card_view(card: CardDefinition) -> CardView:
    return CardView(card.card_id, card.tier, card.bonus, card.points, card.cost)


def _noble_view(noble: NobleDefinition) -> NobleView:
    return NobleView(noble.noble_id, noble.points, noble.requirement)


def _public_reserved(reserved: ReservedCard, catalog: Catalog) -> PublicReservedCard:
    card = catalog.card(reserved.card_id)
    visible = _card_view(card) if reserved.source is ReservationSource.MARKET else None
    return PublicReservedCard(card.tier, reserved.source, visible)


def _public_player(player: PlayerState, catalog: Catalog) -> PlayerPublicState:
    reservations = tuple(_public_reserved(card, catalog) for card in player.reserved_cards)
    reservations = tuple(
        sorted(
            reservations,
            key=lambda item: (
                item.source.value,
                item.tier.value,
                item.card.card_id if item.card is not None else "",
            ),
        )
    )
    return PlayerPublicState(
        tokens=player.tokens,
        bonuses=player_bonuses(player, catalog),
        score=player_score(player, catalog),
        purchased_card_ids=tuple(sorted(player.purchased_card_ids)),
        noble_ids=tuple(sorted(player.noble_ids)),
        reserved_cards=reservations,
    )


def _private_player(player: PlayerState, catalog: Catalog) -> PlayerPrivateState:
    blind = (
        _card_view(catalog.card(reserved.card_id))
        for reserved in player.reserved_cards
        if reserved.source is ReservationSource.DECK
    )
    return PlayerPrivateState(tuple(sorted(blind, key=lambda card: card.card_id)))


def build_observation(state: GameState, observer: int, catalog: Catalog) -> Observation:
    """Derive an information-safe view; never infer identities from missing cards."""

    if observer not in (0, 1):
        raise ValueError("observer must be player 0 or 1")
    own = state.players[observer]
    opponent = state.players[1 - observer]

    def market_cards(tier: Tier) -> tuple[CardView, ...]:
        return tuple(
            _card_view(catalog.card(card_id))
            for card_id in sorted(state.market.for_tier(tier))
        )

    pending = None
    if state.pending is not None:
        payment_target = None
        if isinstance(state.pending.main_action, (BuyMarket, BuyReserved)):
            payment_target = _card_view(catalog.card(state.pending.main_action.card_id))
        pending = PendingObservation(
            main_action=state.pending.main_action,
            payment_target=payment_target,
            tokens_to_return=state.pending.tokens_to_return,
            eligible_nobles=tuple(
                _noble_view(catalog.noble(noble_id))
                for noble_id in sorted(state.pending.eligible_noble_ids)
            ),
        )

    purchased = {
        card_id
        for player in state.players
        for card_id in player.purchased_card_ids
    }
    trigger = state.final_round_triggered_by
    return Observation(
        phase=state.phase,
        self_is_first_player=observer == state.first_player,
        supply=state.supply,
        deck_counts=tuple(len(state.decks.for_tier(tier)) for tier in Tier),
        market=MarketView(
            tier_one=market_cards(Tier.ONE),
            tier_two=market_cards(Tier.TWO),
            tier_three=market_cards(Tier.THREE),
        ),
        nobles_in_play=tuple(
            _noble_view(catalog.noble(noble_id))
            for noble_id in sorted(state.nobles_in_play)
        ),
        self_public=_public_player(own, catalog),
        self_private=_private_player(own, catalog),
        opponent_public=_public_player(opponent, catalog),
        public_purchased=tuple(card.card_id in purchased for card in catalog.cards),
        own_completed_turns=state.completed_turns[observer],
        opponent_completed_turns=state.completed_turns[1 - observer],
        consecutive_forced_skips=state.consecutive_forced_skips,
        final_round_triggered=trigger is not None,
        final_round_triggered_by_self=trigger == observer,
        pending=pending,
        result=state.result,
    )
