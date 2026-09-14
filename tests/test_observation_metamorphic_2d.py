from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from splendor._state import GameState, PendingTurn, PlayerState, ReservedCard
from splendor.actions import BuyReserved, ClaimNoble, Pay, ReserveMarket, TakeDistinct
from splendor.catalog import Catalog
from splendor.engine import GameEngine
from splendor.invariants import assert_game_state
from splendor.model import (
    Color,
    DecisionPhase,
    GEM_COLORS,
    ReservationSource,
    Tier,
    TokenVector,
)
from splendor.observation import build_observation
from splendor.replay import new_replay
from splendor.rules import eligible_noble_ids, legal_actions, legal_payments_for_card


def _engine(state: GameState, catalog: Catalog, failure_dir: Path) -> GameEngine:
    return GameEngine(state, catalog, new_replay(catalog, 123, state.first_player), failure_dir)


def test_unknown_deck_order_does_not_change_observation_or_candidates(
    initial_state: GameState, catalog: Catalog
) -> None:
    cards = initial_state.decks.tier_one
    permuted = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(
            Tier.ONE,
            (cards[1], cards[0]) + cards[2:],
        ),
    )

    assert_game_state(permuted, catalog)
    assert build_observation(initial_state, 0, catalog) == build_observation(permuted, 0, catalog)
    assert legal_actions(initial_state, catalog) == legal_actions(permuted, catalog)


def test_market_and_noble_reordering_is_semantic_and_card_handle_stays_exact(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    reordered = replace(
        initial_state,
        market=replace(
            initial_state.market,
            tier_one=tuple(reversed(initial_state.market.tier_one)),
            tier_two=tuple(reversed(initial_state.market.tier_two)),
            tier_three=tuple(reversed(initial_state.market.tier_three)),
        ),
        nobles_in_play=tuple(reversed(initial_state.nobles_in_play)),
    )
    target = initial_state.market.tier_two[1]

    assert_game_state(reordered, catalog)
    assert build_observation(initial_state, 0, catalog) == build_observation(reordered, 0, catalog)
    assert legal_actions(initial_state, catalog) == legal_actions(reordered, catalog)
    assert ReserveMarket(target) in legal_actions(initial_state, catalog)

    left = _engine(initial_state, catalog, tmp_path / "left")
    right = _engine(reordered, catalog, tmp_path / "right")
    left.apply(ReserveMarket(target))
    right.apply(ReserveMarket(target))
    for engine in (left, right):
        state = engine._debug_state()
        assert state.players[0].reserved_cards[-1].card_id == target
        assert target not in state.market.all_cards()
    assert left.decision() == right.decision()


def test_reserved_card_reordering_does_not_change_buy_handle(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    card_ids = ("005", "006")
    deck = tuple(card for card in initial_state.decks.tier_one if card not in card_ids)
    tokens = TokenVector(white=2, blue=2, green=2, red=2, black=2)
    supply = initial_state.supply.minus(tokens)
    reservations = (
        ReservedCard(card_ids[0], ReservationSource.MARKET),
        ReservedCard(card_ids[1], ReservationSource.DECK),
    )
    player = PlayerState(tokens=tokens, reserved_cards=reservations)
    state = replace(
        initial_state,
        supply=supply,
        decks=initial_state.decks.replacing_tier(Tier.ONE, deck),
        players=(player, initial_state.players[1]),
    )
    reordered = replace(
        state,
        players=(replace(player, reserved_cards=tuple(reversed(reservations))), state.players[1]),
    )

    assert_game_state(state, catalog)
    assert_game_state(reordered, catalog)
    assert build_observation(state, 0, catalog) == build_observation(reordered, 0, catalog)
    assert legal_actions(state, catalog) == legal_actions(reordered, catalog)

    target = card_ids[0]
    left = _engine(state, catalog, tmp_path / "left")
    right = _engine(reordered, catalog, tmp_path / "right")
    left.apply(BuyReserved(target))
    right.apply(BuyReserved(target))
    assert left.decision() == right.decision()
    payment = legal_payments_for_card(player, catalog.card(target), catalog)[0]
    assert isinstance(payment, Pay)
    left.apply(payment)
    right.apply(payment)
    for engine in (left, right):
        bought = engine._debug_state().players[0]
        assert target in bought.purchased_card_ids
        assert target not in {reserved.card_id for reserved in bought.reserved_cards}
    assert left.decision() == right.decision()


def test_opponent_blind_reservation_order_and_identity_do_not_leak(
    initial_state: GameState, catalog: Catalog
) -> None:
    hidden_ids = initial_state.decks.tier_one[:3]
    remaining = tuple(card for card in initial_state.decks.tier_one if card not in hidden_ids)
    opponent = replace(
        initial_state.players[1],
        reserved_cards=tuple(
            ReservedCard(card_id, ReservationSource.DECK) for card_id in hidden_ids
        ),
    )
    state = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(Tier.ONE, remaining),
        players=(initial_state.players[0], opponent),
    )
    permuted_opponent = replace(opponent, reserved_cards=tuple(reversed(opponent.reserved_cards)))
    permuted = replace(state, players=(state.players[0], permuted_opponent))

    assert_game_state(state, catalog)
    assert_game_state(permuted, catalog)
    assert build_observation(state, 0, catalog) == build_observation(permuted, 0, catalog)
    assert legal_actions(state, catalog) == legal_actions(permuted, catalog)


def _multi_noble_state(initial_state: GameState, catalog: Catalog) -> GameState:
    noble_ids = initial_state.nobles_in_play[:2]
    required = {
        color: max(catalog.noble(noble_id).requirement.for_color(color) for noble_id in noble_ids)
        for color in GEM_COLORS
    }
    purchased: list[str] = []
    remaining_by_tier = {tier: list(initial_state.decks.for_tier(tier)) for tier in Tier}
    for color, count in required.items():
        candidates = [
            card.card_id
            for card in catalog.cards
            if card.bonus is color
            and card.card_id in remaining_by_tier[card.tier]
        ]
        chosen = candidates[:count]
        purchased.extend(chosen)
        for card_id in chosen:
            remaining_by_tier[catalog.card(card_id).tier].remove(card_id)
    player = replace(initial_state.players[0], purchased_card_ids=tuple(purchased))
    eligible = tuple(sorted(eligible_noble_ids(player, initial_state.nobles_in_play, catalog)))
    return replace(
        initial_state,
        decks=replace(
            initial_state.decks,
            tier_one=tuple(remaining_by_tier[Tier.ONE]),
            tier_two=tuple(remaining_by_tier[Tier.TWO]),
            tier_three=tuple(remaining_by_tier[Tier.THREE]),
        ),
        players=(player, initial_state.players[1]),
        phase=DecisionPhase.CHOOSE_NOBLE,
        pending=PendingTurn(
            TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
            eligible_noble_ids=eligible,
        ),
    )


def test_noble_reordering_does_not_change_choice_or_handle(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state = _multi_noble_state(initial_state, catalog)
    assert state.pending is not None and len(state.pending.eligible_noble_ids) >= 2
    reordered = replace(
        state,
        nobles_in_play=tuple(reversed(state.nobles_in_play)),
        pending=replace(
            state.pending,
            eligible_noble_ids=tuple(reversed(state.pending.eligible_noble_ids)),
        ),
    )

    assert_game_state(state, catalog)
    assert_game_state(reordered, catalog)
    assert build_observation(state, 0, catalog) == build_observation(reordered, 0, catalog)
    assert legal_actions(state, catalog) == legal_actions(reordered, catalog)
    target = state.pending.eligible_noble_ids[0]
    assert ClaimNoble(target) in legal_actions(state, catalog)

    left = _engine(state, catalog, tmp_path / "left")
    right = _engine(reordered, catalog, tmp_path / "right")
    left.apply(ClaimNoble(target))
    right.apply(ClaimNoble(target))
    for engine in (left, right):
        settled = engine._debug_state()
        assert target in settled.players[0].noble_ids
        assert target not in settled.nobles_in_play
    assert left.decision() == right.decision()


def test_public_purchased_is_exactly_the_bought_union(
    initial_state: GameState, catalog: Catalog
) -> None:
    public_reserved = initial_state.market.tier_one[0]
    blind_reserved = initial_state.decks.tier_one[0]
    purchased = initial_state.decks.tier_one[1]
    deck = tuple(
        card
        for card in initial_state.decks.tier_one
        if card not in (blind_reserved, purchased)
    )
    market = tuple(card for card in initial_state.market.tier_one if card != public_reserved)
    refill = deck[-1]
    deck = deck[:-1]
    market = market + (refill,)
    players = (
        replace(
            initial_state.players[0],
            purchased_card_ids=(purchased,),
            reserved_cards=(ReservedCard(public_reserved, ReservationSource.MARKET),),
        ),
        replace(
            initial_state.players[1],
            reserved_cards=(ReservedCard(blind_reserved, ReservationSource.DECK),),
        ),
    )
    state = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(Tier.ONE, deck),
        market=initial_state.market.replacing_tier(Tier.ONE, market),
        players=players,
    )

    assert_game_state(state, catalog)
    observation = build_observation(state, 0, catalog)
    marked = {
        card.card_id
        for card, is_purchased in zip(catalog.cards, observation.public_purchased, strict=True)
        if is_purchased
    }
    assert marked == {purchased}
    assert public_reserved not in marked
    assert blind_reserved not in marked
