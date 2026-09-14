from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

from splendor._state import GameState, PlayerState, ReservedCard
from splendor.actions import (
    BuyMarket,
    BuyReserved,
    ClaimNoble,
    Pay,
    ReserveDeck,
    TakeDistinct,
    TakePair,
)
from splendor.catalog import CardDefinition, Catalog
from splendor.engine import GameEngine, IllegalActionError, replay_game
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from splendor.replay import new_replay, state_digest
from splendor.rules import legal_actions, legal_payments_for_card, player_bonuses, player_score


def _engine(state: GameState, catalog: Catalog, failure_dir: Path) -> GameEngine:
    return GameEngine(state, catalog, new_replay(catalog, 123, state.first_player), failure_dir)


def _fund_current(state: GameState, tokens: TokenVector) -> GameState:
    players = list(state.players)
    players[state.current_player] = replace(players[state.current_player], tokens=tokens)
    return replace(state, supply=state.supply.minus(tokens), players=tuple(players))


def _remove_from_deck(state: GameState, card_ids: tuple[str, ...]) -> GameState:
    removed = set(card_ids)
    decks = state.decks
    for tier in Tier:
        decks = decks.replacing_tier(
            tier,
            tuple(card_id for card_id in decks.for_tier(tier) if card_id not in removed),
        )
    return replace(state, decks=decks)


def _oracle_payments(need: tuple[int, ...], held: TokenVector) -> set[TokenVector]:
    expected: set[TokenVector] = set()
    for gold_by_color in product(*(range(amount + 1) for amount in need)):
        ordinary = tuple(amount - gold for amount, gold in zip(need, gold_by_color, strict=True))
        if all(amount <= available for amount, available in zip(ordinary, tuple(held)[:5], strict=True)):
            gold = sum(gold_by_color)
            if gold <= held.gold:
                expected.add(TokenVector(*ordinary, gold))
    return expected


def test_payment_enumerator_matches_independent_small_exhaustive_oracle(catalog: Catalog) -> None:
    for need in product(range(2), repeat=5):
        card = CardDefinition("oracle", Tier.ONE, Color.WHITE, 0, GemVector(*need))
        for ordinary_held in product(range(2), repeat=5):
            for gold in range(3):
                held = TokenVector(*ordinary_held, gold)
                player = PlayerState(held)
                actual = {action.tokens for action in legal_payments_for_card(player, card, catalog)}
                assert actual == _oracle_payments(need, held)


def test_payment_enumerator_keeps_optional_gold_and_all_distinct_actual_payments(
    catalog: Catalog,
) -> None:
    card = CardDefinition("choice", Tier.ONE, Color.BLACK, 0, GemVector(white=2, blue=1))
    player = PlayerState(TokenVector(white=2, blue=1, gold=3))
    payments = {action.tokens for action in legal_payments_for_card(player, card, catalog)}

    assert payments == {
        TokenVector(white=2, blue=1),
        TokenVector(white=2, gold=1),
        TokenVector(white=1, blue=1, gold=1),
        TokenVector(white=1, gold=2),
        TokenVector(blue=1, gold=2),
        TokenVector(gold=3),
    }
    assert TokenVector(white=1, blue=1, gold=1) in payments  # voluntary gold
    assert TokenVector(gold=3) in payments  # all gold


def test_payment_enumerator_covers_zero_partial_exact_and_unaffordable(catalog: Catalog) -> None:
    card = CardDefinition("cases", Tier.ONE, Color.BLACK, 0, GemVector(white=2, blue=1))

    exact = legal_payments_for_card(PlayerState(TokenVector(white=2, blue=1)), card, catalog)
    assert exact == (Pay(TokenVector(white=2, blue=1)),)

    partial = legal_payments_for_card(
        PlayerState(TokenVector(white=1, blue=1, gold=1)), card, catalog
    )
    assert partial == (Pay(TokenVector(white=1, blue=1, gold=1)),)

    assert not legal_payments_for_card(PlayerState(TokenVector(white=1)), card, catalog)


def test_main_purchase_candidates_require_a_legal_payment(
    initial_state: GameState, catalog: Catalog
) -> None:
    target = initial_state.market.tier_one[0]
    expensive = initial_state.market.tier_three[0]
    funded = _fund_current(initial_state, TokenVector(*catalog.card(target).cost))
    actions = legal_actions(funded, catalog)

    assert BuyMarket(target) in actions
    assert BuyMarket(expensive) not in actions


def test_market_target_selection_defers_all_mutation_until_payment(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    target = initial_state.market.tier_one[0]
    payment = Pay(TokenVector(*catalog.card(target).cost))
    funded = _fund_current(initial_state, payment.tokens)
    before_bonus = player_bonuses(funded.players[0], catalog)
    before_score = player_score(funded.players[0], catalog)
    refill = funded.decks.tier_one[-1]
    engine = _engine(funded, catalog, tmp_path)

    request = engine.apply(BuyMarket(target))
    pending = engine._debug_state()
    assert pending.phase is DecisionPhase.PAYMENT
    assert pending.players == funded.players
    assert pending.supply == funded.supply
    assert pending.market == funded.market
    assert pending.decks == funded.decks
    assert pending.current_player == funded.current_player
    assert pending.completed_turns == funded.completed_turns
    assert request.observation.pending is not None
    assert request.observation.pending.payment_target is not None
    assert request.observation.pending.payment_target.card_id == target
    assert payment in request.legal_actions

    engine.apply(payment)
    state = engine._debug_state()
    bought = state.players[0]
    assert target in bought.purchased_card_ids
    assert target not in state.market.tier_one
    assert refill in state.market.tier_one
    assert len(state.decks.tier_one) == len(funded.decks.tier_one) - 1
    assert bought.tokens == TokenVector()
    assert state.supply == catalog.constants.initial_tokens
    expected_bonus = replace(before_bonus, black=before_bonus.black + 1)
    assert player_bonuses(bought, catalog) == expected_bonus
    assert player_score(bought, catalog) == before_score + catalog.card(target).points
    assert state.current_player == 1
    assert state.completed_turns == (1, 0)
    assert state.phase is DecisionPhase.MAIN_ACTION
    assert state.pending is None
    for color in (Color.WHITE, Color.BLUE, Color.GREEN, Color.RED, Color.BLACK, Color.GOLD):
        assert state.supply.for_color(color) + sum(
            candidate.tokens.for_color(color) for candidate in state.players
        ) == catalog.constants.initial_tokens.for_color(color)
    assert_game_state(state, catalog)


@pytest.mark.parametrize("source", [ReservationSource.MARKET, ReservationSource.DECK])
def test_buy_own_reserved_card_removes_reservation_without_touching_market(
    initial_state: GameState,
    catalog: Catalog,
    tmp_path: Path,
    source: ReservationSource,
) -> None:
    target = initial_state.decks.tier_one[-1]
    without_target = _remove_from_deck(initial_state, (target,))
    player = replace(
        without_target.players[0],
        reserved_cards=(ReservedCard(target, source),),
    )
    reserved = replace(without_target, players=(player, without_target.players[1]))
    payment = Pay(TokenVector(*catalog.card(target).cost))
    funded = _fund_current(reserved, payment.tokens)
    market_before = funded.market
    decks_before = funded.decks
    engine = _engine(funded, catalog, tmp_path / source.value)

    request = engine.apply(BuyReserved(target))
    assert request.observation.pending is not None
    assert request.observation.pending.payment_target is not None
    assert request.observation.pending.payment_target.card_id == target
    next_request = engine.apply(payment)

    state = engine._debug_state()
    assert not state.players[0].reserved_cards
    assert target in state.players[0].purchased_card_ids
    assert state.market == market_before
    assert state.decks == decks_before
    assert target in next_request.observation.opponent_public.purchased_card_ids
    card_index = next(
        index for index, card in enumerate(catalog.cards) if card.card_id == target
    )
    assert next_request.observation.public_purchased[card_index]
    assert_game_state(state, catalog)


def test_cannot_buy_opponents_reserved_card(initial_state: GameState, catalog: Catalog) -> None:
    target = initial_state.decks.tier_one[-1]
    state = _remove_from_deck(initial_state, (target,))
    opponent = replace(
        state.players[1],
        reserved_cards=(ReservedCard(target, ReservationSource.DECK),),
    )
    state = replace(state, players=(state.players[0], opponent))
    state = _fund_current(state, TokenVector(*catalog.card(target).cost))

    assert BuyReserved(target) not in legal_actions(state, catalog)


def test_market_purchase_with_empty_deck_does_not_refill(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    target = initial_state.market.tier_one[0]
    exhausted_ids = initial_state.decks.tier_one
    opponent = replace(initial_state.players[1], purchased_card_ids=exhausted_ids)
    exhausted = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(Tier.ONE, ()),
        players=(initial_state.players[0], opponent),
    )
    payment = Pay(TokenVector(*catalog.card(target).cost))
    exhausted = _fund_current(exhausted, payment.tokens)
    engine = _engine(exhausted, catalog, tmp_path)

    engine.apply(BuyMarket(target))
    engine.apply(payment)
    state = engine._debug_state()
    assert not state.decks.tier_one
    assert len(state.market.tier_one) == 3
    assert target in state.players[0].purchased_card_ids
    assert_game_state(state, catalog)


def test_bonuses_discount_payment_and_score_and_bonus_recompute_from_cards(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    target = initial_state.market.tier_three[0]
    bonus_ids = tuple(
        next(
            card.card_id
            for card in catalog.cards
            if card.card_id in initial_state.decks.all_cards()
            and card.bonus is color
            and card.points == 0
        )
        for color in (Color.WHITE, Color.BLUE, Color.GREEN, Color.RED)
    )
    state = _remove_from_deck(initial_state, bonus_ids)
    player = replace(state.players[0], purchased_card_ids=bonus_ids)
    state = replace(state, players=(player, state.players[1]))
    need = GemVector(
        *(max(0, amount - 1) for amount in catalog.card(target).cost)
    )
    payment = Pay(TokenVector(*need))
    state = _fund_current(state, payment.tokens)
    assert payment.tokens.total <= catalog.constants.max_tokens_at_turn_end
    engine = _engine(state, catalog, tmp_path)

    request = engine.apply(BuyMarket(target))
    assert payment in request.legal_actions
    engine.apply(payment)
    bought = engine._debug_state().players[0]

    assert player_bonuses(bought, catalog) == GemVector(white=1, blue=1, green=1, red=1, black=1)
    assert player_score(bought, catalog) == catalog.card(target).points
    assert_game_state(engine._debug_state(), catalog)


def test_payment_phase_save_restore_replay_and_continue_identically(
    catalog: Catalog, tmp_path: Path
) -> None:
    engine = GameEngine.new(991, first_player=0, catalog=catalog, failure_dir=tmp_path / "source")
    target = min(
        engine._debug_state().market.tier_one,
        key=lambda card_id: catalog.card(card_id).cost.total,
    )

    while True:
        request = engine.decision()
        if BuyMarket(target) in request.legal_actions:
            engine.apply(BuyMarket(target))
            break
        if engine._debug_state().current_player == 0:
            held = engine._debug_state().players[0].tokens
            cost = catalog.card(target).cost

            def usefulness(action: object) -> int:
                if isinstance(action, TakePair):
                    return min(2, max(0, cost.for_color(action.color) - held.for_color(action.color)))
                if isinstance(action, TakeDistinct):
                    return sum(
                        min(1, max(0, cost.for_color(color) - held.for_color(color)))
                        for color in action.colors
                    )
                return -1

            takes = tuple(
                action
                for action in request.legal_actions
                if isinstance(action, (TakePair, TakeDistinct))
            )
            engine.apply(max(takes, key=usefulness))
        else:
            reserve = next(
                action for action in request.legal_actions if isinstance(action, ReserveDeck)
            )
            engine.apply(reserve)

    assert engine._debug_state().phase is DecisionPhase.PAYMENT
    payment_request = engine.decision()
    save_path = tmp_path / "payment-state.json"
    engine.save(save_path)
    restored = GameEngine.restore(
        save_path,
        catalog=catalog,
        failure_dir=tmp_path / "restored",
    )
    assert restored.decision() == payment_request
    assert state_digest(restored._debug_state()) == state_digest(engine._debug_state())

    payment = payment_request.legal_actions[-1]
    assert isinstance(payment, Pay)
    engine.apply(payment)
    restored.apply(payment)
    assert state_digest(restored._debug_state()) == state_digest(engine._debug_state())
    assert restored.replay_record == engine.replay_record
    assert engine.replay_record.events[-2].actor == engine.replay_record.events[-1].actor == 0
    assert engine.replay_record.events[-2].phase is DecisionPhase.MAIN_ACTION
    assert engine.replay_record.events[-1].phase is DecisionPhase.PAYMENT

    replayed = replay_game(
        engine.replay_record,
        catalog=catalog,
        failure_dir=tmp_path / "replayed",
    )
    assert replayed.replay_record == engine.replay_record
    assert state_digest(replayed._debug_state()) == state_digest(engine._debug_state())


def test_illegal_payment_fails_without_mutation_and_saves_scene(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    target = initial_state.market.tier_one[0]
    exact = TokenVector(*catalog.card(target).cost)
    engine = _engine(_fund_current(initial_state, exact), catalog, tmp_path)
    engine.apply(BuyMarket(target))
    before = state_digest(engine._debug_state())

    with pytest.raises(IllegalActionError):
        engine.apply(Pay(exact.plus(TokenVector(gold=1))))

    assert state_digest(engine._debug_state()) == before
    assert len(list(tmp_path.glob("failure-*.json"))) == 1


def test_purchase_that_creates_one_noble_claims_it_automatically(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    target = initial_state.market.tier_one[0]
    target_bonus = catalog.card(target).bonus
    noble = catalog.noble("N02")
    needed_bonuses = {
        color: noble.requirement.for_color(color) - (1 if color is target_bonus else 0)
        for color in (Color.WHITE, Color.BLUE, Color.GREEN, Color.RED, Color.BLACK)
    }
    deck_ids = set(initial_state.decks.all_cards())
    purchased = []
    for color, count in needed_bonuses.items():
        candidates = [
            card.card_id
            for card in catalog.cards
            if card.card_id in deck_ids and card.bonus is color and card.points == 0
        ]
        assert len(candidates) >= count
        purchased.extend(candidates[:count])

    state = _remove_from_deck(initial_state, tuple(purchased))
    player = replace(state.players[0], purchased_card_ids=tuple(purchased))
    state = replace(state, players=(player, state.players[1]))
    bonuses = player_bonuses(player, catalog)
    need = GemVector(
        *(max(0, cost - bonus) for cost, bonus in zip(catalog.card(target).cost, bonuses, strict=True))
    )
    state = _fund_current(state, TokenVector(*need))
    engine = _engine(state, catalog, tmp_path)
    engine.apply(BuyMarket(target))
    payment = engine.decision().legal_actions[0]
    assert isinstance(payment, Pay)
    engine.apply(payment)

    settled = engine._debug_state()
    assert settled.players[0].noble_ids == ("N02",)
    assert "N02" not in settled.nobles_in_play
    assert settled.completed_turns == (1, 0)
    assert settled.current_player == 1
    assert engine.replay_record.events[-1].action == ClaimNoble("N02")
    assert engine.replay_record.events[-1].automatic
    assert_game_state(settled, catalog)
