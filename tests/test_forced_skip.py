from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from splendor.actions import (
    Action,
    BuyMarket,
    Pay,
    ReserveDeck,
    ReserveMarket,
    ReturnTokens,
    TakeDistinct,
    TakePair,
)
from splendor.catalog import Catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine, replay_game
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, Tier, TokenVector
from splendor.replay import ActionEvent, ForcedSkipEvent, new_replay, state_digest
from splendor.result import TerminalReason, TerminalResult
from splendor.rules import apply_legal_action, legal_actions


_A_ACTIONS: tuple[Action, ...] = (
    ReserveMarket("071"),
    ReserveMarket("081"),
    TakeDistinct((Color.BLUE, Color.RED, Color.BLACK)),
    ReserveDeck(Tier.THREE),
    TakeDistinct((Color.BLUE, Color.GREEN, Color.BLACK)),
    TakeDistinct((Color.WHITE, Color.GREEN, Color.RED)),
    ReserveMarket("087"),
    ReserveMarket("089"),
    ReserveDeck(Tier.TWO),
    TakeDistinct((Color.BLUE, Color.GREEN, Color.BLACK)),
    TakeDistinct((Color.BLUE, Color.GREEN, Color.RED)),
    ReturnTokens(TokenVector(red=1)),
    BuyMarket("032"),
    Pay(TokenVector(black=1, gold=3)),
    BuyMarket("015"),
    Pay(TokenVector(black=1, gold=2)),
    TakeDistinct((Color.WHITE, Color.RED, Color.BLACK)),
    TakeDistinct((Color.WHITE, Color.RED, Color.BLACK)),
    TakeDistinct((Color.WHITE, Color.BLACK)),
)

_B_ACTIONS: tuple[Action, ...] = (
    TakePair(Color.BLUE),
    ReserveDeck(Tier.THREE),
    ReserveDeck(Tier.ONE),
    TakePair(Color.GREEN),
    ReserveMarket("075"),
    ReserveMarket("076"),
    TakePair(Color.BLACK),
    TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
    ReserveDeck(Tier.ONE),
    ReserveDeck(Tier.ONE),
    TakeDistinct((Color.WHITE, Color.BLUE, Color.BLACK)),
    TakeDistinct((Color.WHITE, Color.GREEN, Color.RED)),
    BuyMarket("001"),
    Pay(TokenVector(blue=1, gold=3)),
    TakeDistinct((Color.WHITE, Color.BLUE, Color.RED)),
    ReturnTokens(TokenVector(blue=1, gold=2)),
    TakeDistinct((Color.BLUE, Color.RED, Color.BLACK)),
    TakeDistinct((Color.RED,)),
    ReturnTokens(TokenVector(red=1)),
    TakeDistinct((Color.RED,)),
)


def _engine_at_raw_deadlock(
    seed: int,
    first_player: int,
    actions: tuple[Action, ...],
    catalog: Catalog,
    failure_dir: Path,
) -> GameEngine:
    initial = GameEngine.new(
        seed,
        first_player=first_player,
        catalog=catalog,
        failure_dir=failure_dir,
    )
    state = initial._debug_state()
    replay = new_replay(catalog, seed, first_player)
    events: list[ActionEvent] = []
    for action in actions:
        candidates = legal_actions(state, catalog)
        assert action in candidates
        before = state
        state = apply_legal_action(before, action, catalog)
        assert_game_state(state, catalog)
        events.append(
            ActionEvent(
                sequence=len(events),
                actor=before.current_player,
                phase=before.phase,
                action=action,
                automatic=False,
                state_digest_before=state_digest(before),
                state_digest_after=state_digest(state),
            )
        )
    assert state.phase is DecisionPhase.MAIN_ACTION
    assert not legal_actions(state, catalog)
    return GameEngine(
        state,
        catalog,
        replace(replay, events=tuple(events)),
        failure_dir,
    )


def _a_engine(catalog: Catalog, path: Path) -> GameEngine:
    return _engine_at_raw_deadlock(7155, 0, _A_ACTIONS, catalog, path)


def _b_engine(catalog: Catalog, path: Path) -> GameEngine:
    return _engine_at_raw_deadlock(2639, 1, _B_ACTIONS, catalog, path)


def test_one_sided_no_action_is_forced_skip_and_opponent_continues(
    catalog: Catalog, tmp_path: Path
) -> None:
    engine = _a_engine(catalog, tmp_path)
    before = engine._debug_state()

    outcome = engine.decision()

    assert isinstance(outcome, DecisionRequest)
    after = engine._debug_state()
    assert after.current_player == 1
    assert after.completed_turns == (9, 8)
    assert after.consecutive_forced_skips == 1
    assert after.supply == before.supply
    assert after.decks == before.decks
    assert after.market == before.market
    assert after.players == before.players
    assert after.nobles_in_play == before.nobles_in_play
    assert after.rng == before.rng
    assert isinstance(engine.replay_record.events[-1], ForcedSkipEvent)
    assert BuyMarket("020") in outcome.legal_actions

    payment = engine.apply(BuyMarket("020"))
    assert isinstance(payment, DecisionRequest)
    assert payment.observation.phase is DecisionPhase.PAYMENT
    assert engine._debug_state().consecutive_forced_skips == 0


def test_two_sided_no_action_is_immediate_forced_skip_draw(
    catalog: Catalog, tmp_path: Path
) -> None:
    engine = _b_engine(catalog, tmp_path)
    before = engine._debug_state()

    result = engine.decision()

    assert isinstance(result, TerminalResult)
    assert result.reason is TerminalReason.CONSECUTIVE_FORCED_SKIPS
    assert result.winner_ids == (0, 1)
    assert engine._debug_state().phase is DecisionPhase.TERMINAL
    assert engine._debug_state().consecutive_forced_skips == 2
    assert engine._debug_state().supply == before.supply
    assert engine._debug_state().players == before.players
    assert all(isinstance(event, ForcedSkipEvent) for event in engine.replay_record.events[-2:])


def test_legal_main_action_never_forces_skip(tmp_path: Path) -> None:
    engine = GameEngine.new(1, first_player=0, failure_dir=tmp_path)

    request = engine.decision()

    assert isinstance(request, DecisionRequest)
    assert request.legal_actions
    assert engine.replay_record.events == ()
    assert engine._debug_state().consecutive_forced_skips == 0


def test_normal_main_action_clears_forced_skip_streak(
    catalog: Catalog, tmp_path: Path
) -> None:
    engine = _a_engine(catalog, tmp_path)
    request = engine.decision()
    assert isinstance(request, DecisionRequest)
    assert engine._debug_state().consecutive_forced_skips == 1

    engine.apply(BuyMarket("020"))

    assert engine._debug_state().phase is DecisionPhase.PAYMENT
    assert engine._debug_state().consecutive_forced_skips == 0


def test_save_restore_before_and_after_forced_skip_are_identical(
    catalog: Catalog, tmp_path: Path
) -> None:
    before = _a_engine(catalog, tmp_path / "before")
    before_path = tmp_path / "before.json"
    before.save(before_path)
    before_restored = GameEngine.restore(before_path, catalog=catalog, failure_dir=tmp_path / "before-r")

    left = before.decision()
    right = before_restored.decision()
    assert left == right
    assert state_digest(before._debug_state()) == state_digest(before_restored._debug_state())
    assert before.replay_record == before_restored.replay_record

    after_path = tmp_path / "after.json"
    before.save(after_path)
    after_restored = GameEngine.restore(after_path, catalog=catalog, failure_dir=tmp_path / "after-r")
    assert before.decision() == after_restored.decision()
    assert state_digest(before._debug_state()) == state_digest(after_restored._debug_state())
    assert before.replay_record == after_restored.replay_record


def test_forced_skip_replay_is_exact_for_continue_and_draw(
    catalog: Catalog, tmp_path: Path
) -> None:
    continuing = _a_engine(catalog, tmp_path / "a")
    assert isinstance(continuing.decision(), DecisionRequest)
    replayed_a = replay_game(continuing.replay_record, catalog=catalog, failure_dir=tmp_path / "ar")
    assert replayed_a.replay_record == continuing.replay_record
    assert state_digest(replayed_a._debug_state()) == state_digest(continuing._debug_state())

    drawing = _b_engine(catalog, tmp_path / "b")
    assert isinstance(drawing.decision(), TerminalResult)
    replayed_b = replay_game(drawing.replay_record, catalog=catalog, failure_dir=tmp_path / "br")
    assert replayed_b.replay_record == drawing.replay_record
    assert state_digest(replayed_b._debug_state()) == state_digest(drawing._debug_state())
