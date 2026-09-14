from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from splendor._state import DeckState, GameState, MarketState, PendingTurn, PlayerState, ReservedCard
from splendor.actions import BuyMarket, TakeDistinct
from splendor.catalog import Catalog
from splendor.invariants import InvariantViolation, assert_game_state
from splendor.model import Color, DecisionPhase, ReservationSource, TokenVector
from splendor.replay import InvariantGuard, new_replay
from splendor.result import TerminalResult
from splendor.rules import player_score


def test_initial_contract_state_is_valid(initial_state: GameState, catalog: Catalog) -> None:
    assert_game_state(initial_state, catalog)


def test_token_conservation_fails_immediately(initial_state: GameState, catalog: Catalog) -> None:
    corrupt = replace(initial_state, supply=replace(initial_state.supply, white=3))
    with pytest.raises(InvariantViolation, match="token_conservation") as caught:
        assert_game_state(corrupt, catalog)
    assert caught.value.code == "token_conservation"


def test_duplicate_card_location_is_rejected(initial_state: GameState, catalog: Catalog) -> None:
    market_card = initial_state.market.tier_one[0]
    corrupt_deck = replace(
        initial_state.decks,
        tier_one=(market_card,) + initial_state.decks.tier_one[1:],
    )
    with pytest.raises(InvariantViolation, match="card_uniqueness"):
        assert_game_state(replace(initial_state, decks=corrupt_deck), catalog)


def test_nonempty_deck_requires_a_full_market(initial_state: GameState, catalog: Catalog) -> None:
    removed = initial_state.market.tier_one[0]
    player = replace(
        initial_state.players[0],
        reserved_cards=(ReservedCard(removed, ReservationSource.MARKET),),
    )
    corrupt = replace(
        initial_state,
        market=replace(initial_state.market, tier_one=initial_state.market.tier_one[1:]),
        players=(player, initial_state.players[1]),
    )
    with pytest.raises(InvariantViolation, match="market_refill"):
        assert_game_state(corrupt, catalog)


def test_only_return_phase_may_temporarily_exceed_ten_tokens(
    initial_state: GameState, catalog: Catalog
) -> None:
    player = PlayerState(TokenVector(white=4, blue=4, green=3))
    supply = TokenVector(white=0, blue=0, green=1, red=4, black=4, gold=5)
    over_limit = replace(initial_state, players=(player, initial_state.players[1]), supply=supply)
    with pytest.raises(InvariantViolation, match="turn_end_token_limit"):
        assert_game_state(over_limit, catalog)

    returning = replace(
        over_limit,
        phase=DecisionPhase.RETURN_TOKENS,
        pending=PendingTurn(
            main_action=TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
            tokens_to_return=1,
        ),
    )
    assert_game_state(returning, catalog)


def test_non_acting_player_cannot_exceed_turn_end_token_limit(
    initial_state: GameState, catalog: Catalog
) -> None:
    opponent = PlayerState(TokenVector(white=4, blue=4, green=3))
    supply = TokenVector(white=0, blue=0, green=1, red=4, black=4, gold=5)
    corrupt = replace(initial_state, players=(initial_state.players[0], opponent), supply=supply)
    with pytest.raises(InvariantViolation, match="turn_end_token_limit"):
        assert_game_state(corrupt, catalog)


def test_pending_context_must_match_phase(initial_state: GameState, catalog: Catalog) -> None:
    corrupt = replace(initial_state, phase=DecisionPhase.PAYMENT)
    with pytest.raises(InvariantViolation, match="phase_context"):
        assert_game_state(corrupt, catalog)


def test_payment_context_identifies_an_existing_target(initial_state: GameState, catalog: Catalog) -> None:
    target = initial_state.market.tier_one[0]
    cost = catalog.card(target).cost
    tokens = TokenVector(*cost)
    paying = replace(
        initial_state,
        supply=initial_state.supply.minus(tokens),
        players=(replace(initial_state.players[0], tokens=tokens), initial_state.players[1]),
        phase=DecisionPhase.PAYMENT,
        pending=PendingTurn(BuyMarket(target)),
    )
    assert_game_state(paying, catalog)

    corrupt = replace(paying, pending=PendingTurn(BuyMarket("not-a-card")))
    with pytest.raises(InvariantViolation, match="payment_target"):
        assert_game_state(corrupt, catalog)


def test_guard_saves_full_failure_scene_before_reraising(
    tmp_path: Path, initial_state: GameState, catalog: Catalog
) -> None:
    corrupt = replace(initial_state, supply=replace(initial_state.supply, gold=4))
    guard = InvariantGuard(catalog, new_replay(catalog, game_seed=123, first_player=0), tmp_path)
    with pytest.raises(InvariantViolation):
        guard.check(corrupt)

    reports = list(tmp_path.glob("failure-*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["error"]["code"] == "token_conservation"
    assert report["replay"]["header"]["game_seed"] == 123
    assert report["state"]["decks"]["tier_one"]
    assert len(report["state_digest"]) == 64


def test_terminal_result_is_recomputed_from_authoritative_holdings(
    initial_state: GameState, catalog: Catalog
) -> None:
    chosen: list[str] = []
    points = 0
    deck_ids = set(initial_state.decks.all_cards())
    for card in sorted(
        (card for card in catalog.cards if card.card_id in deck_ids),
        key=lambda item: item.points,
        reverse=True,
    ):
        chosen.append(card.card_id)
        points += card.points
        if points >= catalog.constants.end_game_prestige:
            break
    chosen_set = set(chosen)

    def remove_chosen(cards: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(card_id for card_id in cards if card_id not in chosen_set)

    player_zero = replace(initial_state.players[0], purchased_card_ids=tuple(chosen))
    players = (player_zero, initial_state.players[1])
    scores = (player_score(player_zero, catalog), 0)
    shared_fields = {
        "decks": DeckState(
            remove_chosen(initial_state.decks.tier_one),
            remove_chosen(initial_state.decks.tier_two),
            remove_chosen(initial_state.decks.tier_three),
        ),
        "market": MarketState(
            remove_chosen(initial_state.market.tier_one),
            remove_chosen(initial_state.market.tier_two),
            remove_chosen(initial_state.market.tier_three),
        ),
        "players": players,
        "current_player": 1,
        "final_round_triggered_by": 0,
    }
    awaiting_reply = replace(
        initial_state,
        **shared_fields,
        completed_turns=(7, 6),
    )
    assert_game_state(awaiting_reply, catalog)

    result = TerminalResult(scores, (len(chosen), 0), (0,), (7, 7))
    terminal = replace(
        initial_state,
        **shared_fields,
        completed_turns=(7, 7),
        phase=DecisionPhase.TERMINAL,
        result=result,
    )
    assert_game_state(terminal, catalog)

    corrupt_result = replace(result, winner_ids=(1,))
    with pytest.raises(InvariantViolation, match="result_winners"):
        assert_game_state(replace(terminal, result=corrupt_result), catalog)

    with pytest.raises(InvariantViolation, match="terminal_trigger_score"):
        assert_game_state(replace(terminal, final_round_triggered_by=1), catalog)
