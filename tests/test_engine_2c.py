from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from splendor._state import GameState, PlayerState, ReservedCard
from splendor.actions import (
    BuyReserved,
    ClaimNoble,
    Pay,
    ReserveDeck,
    ReturnTokens,
    TakeDistinct,
    TakePair,
)
from splendor.catalog import Catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine, IllegalActionError, replay_game
from splendor.invariants import InvariantViolation, assert_game_state
from splendor.model import Color, DecisionPhase, ReservationSource, Tier, TokenVector
from splendor.replay import new_replay, state_digest
from splendor.result import TerminalResult
from splendor.rng import DeterministicRng
from splendor.rules import player_bonuses, player_score


def _engine(state: GameState, catalog: Catalog, failure_dir: Path) -> GameEngine:
    return GameEngine(state, catalog, new_replay(catalog, 123, state.first_player), failure_dir)


def _with_assets(
    state: GameState,
    catalog: Catalog,
    *,
    purchased: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
    reserved: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
    nobles: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
    tokens: tuple[TokenVector, TokenVector] = (TokenVector(), TokenVector()),
    current_player: int = 0,
    completed_turns: tuple[int, int] = (0, 0),
    final_round_triggered_by: int | None = None,
) -> GameState:
    moved_cards = set(purchased[0] + purchased[1] + reserved[0] + reserved[1])
    assert moved_cards <= set(state.decks.all_cards())
    decks = state.decks
    for tier in Tier:
        decks = decks.replacing_tier(
            tier,
            tuple(card_id for card_id in decks.for_tier(tier) if card_id not in moved_cards),
        )

    players = tuple(
        PlayerState(
            tokens=tokens[player_id],
            purchased_card_ids=purchased[player_id],
            reserved_cards=tuple(
                ReservedCard(card_id, ReservationSource.DECK)
                for card_id in reserved[player_id]
            ),
            noble_ids=nobles[player_id],
        )
        for player_id in (0, 1)
    )
    claimed_nobles = set(nobles[0] + nobles[1])
    result = replace(
        state,
        supply=state.supply.minus(tokens[0]).minus(tokens[1]),
        decks=decks,
        players=players,
        nobles_in_play=tuple(
            noble_id for noble_id in state.nobles_in_play if noble_id not in claimed_nobles
        ),
        current_player=current_player,
        completed_turns=completed_turns,
        final_round_triggered_by=final_round_triggered_by,
    )
    assert_game_state(result, catalog)
    return result


def _eligible_cards(state: GameState, catalog: Catalog, noble_ids: tuple[str, ...]) -> tuple[str, ...]:
    required = {color: 0 for color in (Color.WHITE, Color.BLUE, Color.GREEN, Color.RED, Color.BLACK)}
    for noble_id in noble_ids:
        requirement = catalog.noble(noble_id).requirement
        for color in required:
            required[color] = max(required[color], requirement.for_color(color))

    deck_ids = set(state.decks.all_cards())
    chosen: list[str] = []
    for color, count in required.items():
        candidates = [
            card.card_id
            for card in catalog.cards
            if card.card_id in deck_ids and card.bonus is color and card.points == 0
        ]
        assert len(candidates) >= count
        chosen.extend(candidates[:count])
    return tuple(chosen)


def _one_noble_state(initial_state: GameState, catalog: Catalog) -> GameState:
    cards = _eligible_cards(initial_state, catalog, ("N01",))
    return _with_assets(initial_state, catalog, purchased=(cards, ()))


@pytest.mark.parametrize('noble_id', ['N11', 'N12'])
def test_added_noble_claim_agrees_between_checked_and_native(initial_state, catalog, tmp_path, noble_id):
    import numpy as np
    from splendor.features import encode_decision
    from splendor.native_policy import _native_catalog, _extension

    selected = (noble_id, 'N03', 'N04')
    state = replace(initial_state, nobles_in_play=selected,
                    nobles_out_of_game=tuple(n.noble_id for n in catalog.nobles if n.noble_id not in selected))
    purchased = _eligible_cards(state, catalog, (noble_id,))
    state = _with_assets(state, catalog, purchased=(purchased, ()))
    engine = _engine(state, catalog, tmp_path)
    game = _extension().NativeGame(_native_catalog(catalog), state)
    request = engine.decision()
    encoded = game.encoded()
    expected = encode_decision(request)
    np.testing.assert_allclose(encoded['nobles'], expected.nobles, rtol=0, atol=1e-6)
    action = TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN))
    game.apply_index(request.legal_actions.index(action))
    engine.apply(action)
    actual = game.snapshot()
    assert engine._debug_state().players[0].noble_ids == (noble_id,)
    noble_index = next(i for i, n in enumerate(catalog.nobles) if n.noble_id == noble_id)
    assert actual['players'][0]['nobles'] == [noble_index]
    assert player_score(engine._debug_state().players[0], catalog) == 3
    assert actual['actor'] == engine.current_player == 1


def _pay_for_reserved(engine: GameEngine, card_id: str) -> DecisionRequest | TerminalResult:
    payment_request = engine.apply(BuyReserved(card_id))
    assert isinstance(payment_request, DecisionRequest)
    assert payment_request.observation.phase is DecisionPhase.PAYMENT
    payment = payment_request.legal_actions[-1]
    assert isinstance(payment, Pay)
    return engine.apply(payment)


def test_no_eligible_noble_is_skipped(initial_state: GameState, catalog: Catalog, tmp_path: Path) -> None:
    engine = _engine(initial_state, catalog, tmp_path)
    engine.apply(TakePair(Color.WHITE))

    state = engine._debug_state()
    assert state.phase is DecisionPhase.MAIN_ACTION
    assert state.current_player == 1
    assert state.completed_turns == (1, 0)
    assert not state.players[0].noble_ids
    assert len(engine.replay_record.events) == 1


@pytest.mark.parametrize(
    "action",
    [TakePair(Color.WHITE), ReserveDeck(Tier.TWO)],
    ids=["take_tokens", "reserve"],
)
def test_one_eligible_noble_is_auto_claimed_after_every_nonpurchase_path(
    initial_state: GameState,
    catalog: Catalog,
    tmp_path: Path,
    action: TakePair | ReserveDeck,
) -> None:
    state = _one_noble_state(initial_state, catalog)
    engine = _engine(state, catalog, tmp_path)
    engine.apply(action)

    settled = engine._debug_state()
    assert settled.players[0].noble_ids == ("N01",)
    assert "N01" not in settled.nobles_in_play
    assert player_score(settled.players[0], catalog) == 3
    assert settled.completed_turns == (1, 0)
    assert engine.replay_record.events[-1].action == ClaimNoble("N01")
    assert engine.replay_record.events[-1].automatic
    assert_game_state(settled, catalog)


def test_return_tokens_then_uses_the_same_noble_settlement(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    cards = _eligible_cards(initial_state, catalog, ("N01",))
    state = _with_assets(
        initial_state,
        catalog,
        purchased=(cards, ()),
        tokens=(TokenVector(blue=2, green=2, red=2, black=2, gold=1), TokenVector()),
    )
    engine = _engine(state, catalog, tmp_path)
    returning = engine.apply(TakePair(Color.WHITE))
    assert isinstance(returning, DecisionRequest)
    assert returning.observation.phase is DecisionPhase.RETURN_TOKENS

    engine.apply(ReturnTokens(TokenVector(white=1)))
    settled = engine._debug_state()
    assert settled.players[0].tokens.total == 10
    assert settled.players[0].noble_ids == ("N01",)
    assert settled.completed_turns == (1, 0)
    assert engine.replay_record.events[-1].action == ClaimNoble("N01")
    assert engine.replay_record.events[-1].automatic
    assert_game_state(settled, catalog)


def test_multiple_eligible_nobles_are_complete_choice_and_only_one_is_claimed(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    cards = _eligible_cards(initial_state, catalog, ("N01", "N02"))
    state = _with_assets(initial_state, catalog, purchased=(cards, ()))
    engine = _engine(state, catalog, tmp_path)

    request = engine.apply(TakePair(Color.BLUE))
    assert isinstance(request, DecisionRequest)
    assert request.observation.phase is DecisionPhase.CHOOSE_NOBLE
    assert set(request.legal_actions) == {ClaimNoble("N01"), ClaimNoble("N02")}
    choosing = engine._debug_state()
    assert choosing.current_player == 0
    assert choosing.completed_turns == (0, 0)
    assert choosing.pending is not None
    assert set(choosing.pending.eligible_noble_ids) == {"N01", "N02"}

    engine.apply(ClaimNoble("N02"))
    settled = engine._debug_state()
    assert settled.players[0].noble_ids == ("N02",)
    assert "N01" in settled.nobles_in_play
    assert "N02" not in settled.nobles_in_play
    assert settled.completed_turns == (1, 0)
    assert not engine.replay_record.events[-1].automatic
    assert_game_state(settled, catalog)

    corrupt = replace(settled, nobles_in_play=settled.nobles_in_play + ("N02",))
    with pytest.raises(InvariantViolation, match="noble_uniqueness"):
        assert_game_state(corrupt, catalog)


def test_choose_noble_save_restore_continues_identically(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    cards = _eligible_cards(initial_state, catalog, ("N01", "N02"))
    state = _with_assets(initial_state, catalog, purchased=(cards, ()))
    original = _engine(state, catalog, tmp_path / "original")
    original.apply(TakePair(Color.BLUE))
    save_path = tmp_path / "choose-noble.json"
    original.save(save_path)
    restored = GameEngine.restore(save_path, catalog=catalog, failure_dir=tmp_path / "restored")

    assert restored.decision() == original.decision()
    assert state_digest(restored._debug_state()) == state_digest(original._debug_state())
    original.apply(ClaimNoble("N01"))
    restored.apply(ClaimNoble("N01"))
    assert state_digest(restored._debug_state()) == state_digest(original._debug_state())
    assert restored.replay_record == original.replay_record


def _first_player_purchase_state(
    initial_state: GameState,
    catalog: Catalog,
    *,
    opponent_purchased: tuple[str, ...] = (),
    opponent_reserved: tuple[str, ...] = (),
    opponent_tokens: TokenVector = TokenVector(),
) -> GameState:
    return _with_assets(
        initial_state,
        catalog,
        purchased=(("078", "082"), opponent_purchased),
        reserved=(("086",), opponent_reserved),
        tokens=(TokenVector(blue=4, green=3, gold=2), opponent_tokens),
    )


def test_first_player_trigger_waits_for_second_and_final_round_restores(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state = _first_player_purchase_state(initial_state, catalog)
    engine = _engine(state, catalog, tmp_path / "source")
    outcome = _pay_for_reserved(engine, "086")
    assert isinstance(outcome, DecisionRequest)
    assert outcome.observation.final_round_triggered
    assert not outcome.observation.final_round_triggered_by_self

    triggered = engine._debug_state()
    assert player_score(triggered.players[0], catalog) == 15
    assert triggered.final_round_triggered_by == 0
    assert triggered.completed_turns == (1, 0)
    assert triggered.current_player == 1
    assert triggered.phase is DecisionPhase.MAIN_ACTION
    assert triggered.result is None

    save_path = tmp_path / "final-round.json"
    engine.save(save_path)
    restored = GameEngine.restore(save_path, catalog=catalog, failure_dir=tmp_path / "restored")
    assert restored.decision() == engine.decision()
    assert state_digest(restored._debug_state()) == state_digest(triggered)

    result = engine.apply(TakePair(Color.WHITE))
    restored_result = restored.apply(TakePair(Color.WHITE))
    assert isinstance(result, TerminalResult)
    assert result == restored_result
    assert result.completed_turns == (1, 1)
    assert result.winner_ids == (0,)
    assert engine._debug_state().phase is DecisionPhase.TERMINAL
    assert state_digest(restored._debug_state()) == state_digest(engine._debug_state())


def test_second_player_first_trigger_at_equal_turns_is_immediately_terminal(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state = _with_assets(
        initial_state,
        catalog,
        purchased=((), ("078", "082")),
        reserved=((), ("086",)),
        tokens=(TokenVector(), TokenVector(blue=4, green=3, gold=2)),
        current_player=1,
        completed_turns=(1, 0),
    )
    engine = _engine(state, catalog, tmp_path)
    result = _pay_for_reserved(engine, "086")

    assert isinstance(result, TerminalResult)
    terminal = engine._debug_state()
    assert terminal.final_round_triggered_by == 1
    assert terminal.completed_turns == (1, 1)
    assert terminal.phase is DecisionPhase.TERMINAL
    assert result.scores == (0, 15)
    assert result.winner_ids == (1,)


def test_score_above_fifteen_triggers_normally(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state = _with_assets(
        initial_state,
        catalog,
        purchased=((), ("090", "076", "080", "008")),
        reserved=((), ("045",)),
        tokens=(TokenVector(), TokenVector(white=4)),
        current_player=1,
        completed_turns=(1, 0),
    )
    engine = _engine(state, catalog, tmp_path)
    result = _pay_for_reserved(engine, "045")

    assert isinstance(result, TerminalResult)
    assert result.scores == (0, 16)
    assert engine._debug_state().final_round_triggered_by == 1


def test_final_reply_can_overtake_without_retriggering_or_extending_round(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state = _first_player_purchase_state(
        initial_state,
        catalog,
        opponent_purchased=("090", "076", "080", "008"),
        opponent_reserved=("045",),
        opponent_tokens=TokenVector(white=4),
    )
    engine = _engine(state, catalog, tmp_path)
    _pay_for_reserved(engine, "086")
    assert engine._debug_state().final_round_triggered_by == 0

    result = _pay_for_reserved(engine, "045")
    assert isinstance(result, TerminalResult)
    assert result.scores == (15, 16)
    assert result.winner_ids == (1,)
    assert result.completed_turns == (1, 1)
    assert engine._debug_state().final_round_triggered_by == 0


def test_auto_claimed_noble_counts_before_end_trigger(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    bonus_cards = _eligible_cards(initial_state, catalog, ("N01",))
    scoring = ("078", "080", "046")
    assert not set(bonus_cards) & set(scoring)
    state = _with_assets(
        initial_state,
        catalog,
        purchased=((), scoring + bonus_cards),
        current_player=1,
        completed_turns=(1, 0),
    )
    assert player_score(state.players[1], catalog) == 12
    engine = _engine(state, catalog, tmp_path)
    result = engine.apply(TakePair(Color.WHITE))

    assert isinstance(result, TerminalResult)
    terminal = engine._debug_state()
    assert terminal.players[1].noble_ids == ("N01",)
    assert player_score(terminal.players[1], catalog) == 15
    assert terminal.final_round_triggered_by == 1
    assert engine.replay_record.events[-1].action == ClaimNoble("N01")
    assert engine.replay_record.events[-1].automatic


def test_tied_score_uses_fewer_purchased_development_cards(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state = _with_assets(
        initial_state,
        catalog,
        purchased=(("078", "082", "086"), ("090", "076", "080", "045")),
        current_player=1,
        completed_turns=(1, 0),
        final_round_triggered_by=0,
    )
    engine = _engine(state, catalog, tmp_path)
    result = engine.apply(TakePair(Color.WHITE))

    assert isinstance(result, TerminalResult)
    assert result.scores == (15, 15)
    assert result.purchased_card_counts == (3, 4)
    assert result.winner_ids == (0,)


def test_equal_score_and_card_count_is_draw_and_noble_is_not_a_card(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    player_zero = ("025", "026", "027", "028", "033", "070", "088", "090")
    player_one = ("078", "082", "086", "005", "006", "009", "010", "017")
    state = _with_assets(
        initial_state,
        catalog,
        purchased=(player_zero, player_one),
        nobles=(("N01",), ()),
        current_player=1,
        completed_turns=(1, 0),
        final_round_triggered_by=0,
    )
    assert player_score(state.players[0], catalog) == 15
    assert player_score(state.players[1], catalog) == 15
    engine = _engine(state, catalog, tmp_path)
    result = engine.apply(TakePair(Color.WHITE))

    assert isinstance(result, TerminalResult)
    assert result.purchased_card_counts == (8, 8)
    assert result.winner_ids == (0, 1)
    assert result.payoff(0) == result.payoff(1) == 0


def test_terminal_rejects_further_actions_and_saves_scene(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state = _with_assets(
        initial_state,
        catalog,
        purchased=(("078", "082", "086"), ()),
        current_player=1,
        completed_turns=(1, 0),
        final_round_triggered_by=0,
    )
    engine = _engine(state, catalog, tmp_path)
    result = engine.apply(TakePair(Color.WHITE))
    assert isinstance(result, TerminalResult)
    before = state_digest(engine._debug_state())

    with pytest.raises(IllegalActionError):
        engine.apply(TakeDistinct((Color.BLUE, Color.GREEN, Color.RED)))

    assert state_digest(engine._debug_state()) == before
    assert len(list(tmp_path.glob("failure-*.json"))) == 1


def test_seeded_full_game_replay_reproduces_automatic_noble_and_terminal(
    catalog: Catalog, tmp_path: Path
) -> None:
    engine = GameEngine.new(0, first_player=0, catalog=catalog, failure_dir=tmp_path / "source")
    chooser = DeterministicRng(10_000)
    result: TerminalResult | None = None

    for _ in range(500):
        request = engine.decision()
        outcome = engine.apply(
            request.legal_actions[chooser.randbelow(len(request.legal_actions))]
        )
        if isinstance(outcome, TerminalResult):
            result = outcome
            break

    assert result is not None
    assert any(
        event.automatic and isinstance(event.action, ClaimNoble)
        for event in engine.replay_record.events
    )
    assert engine.replay_record.result == result
    replayed = replay_game(
        engine.replay_record,
        catalog=catalog,
        failure_dir=tmp_path / "replayed",
    )
    assert replayed.replay_record == engine.replay_record
    assert state_digest(replayed._debug_state()) == state_digest(engine._debug_state())
    assert replayed._debug_state().phase is DecisionPhase.TERMINAL


def test_mcts_accepts_all_decision_phases(initial_state, catalog, tmp_path) -> None:
    import torch
    from splendor.actor_critic import ActorCritic
    from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy

    model = ActorCritic().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    policy = PUCTSearchPolicy(model, config=MCTSConfig(simulations=2, determinizations=1), catalog=catalog)
    main = _engine(initial_state, catalog, tmp_path / 'main').decision()
    reserved_id = initial_state.decks.tier_one[0]
    reserved_state = _with_assets(initial_state, catalog, reserved=((reserved_id,), ()), tokens=(TokenVector(green=2, red=1), TokenVector()))
    payment = _engine(reserved_state, catalog, tmp_path / 'payment').apply(BuyReserved(reserved_id))
    return_state = _with_assets(initial_state, catalog, tokens=(TokenVector(blue=2, green=2, red=2, black=2, gold=1), TokenVector()))
    returning = _engine(return_state, catalog, tmp_path / 'return').apply(TakePair(Color.WHITE))
    noble_cards = _eligible_cards(initial_state, catalog, ('N01', 'N02'))
    noble_state = _with_assets(initial_state, catalog, purchased=(noble_cards, ()))
    noble = _engine(noble_state, catalog, tmp_path / 'noble').apply(TakePair(Color.BLUE))
    for phase, request in zip((DecisionPhase.MAIN_ACTION, DecisionPhase.PAYMENT, DecisionPhase.RETURN_TOKENS, DecisionPhase.CHOOSE_NOBLE), (main, payment, returning, noble)):
        assert isinstance(request, DecisionRequest)
        assert request.observation.phase is phase
        result = policy.search(request)
        assert request.legal_actions[result.chosen_candidate_index] in request.legal_actions
        assert sum(result.policy_target) == pytest.approx(1)
