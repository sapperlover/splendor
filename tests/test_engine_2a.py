from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import comb
from pathlib import Path

import pytest
import splendor.engine as engine_module
import splendor.replay as replay_module

from splendor._state import GameState, PlayerState, ReservedCard
from splendor.actions import (
    BuyMarket,
    ClaimNoble,
    ReserveDeck,
    ReserveMarket,
    ReturnTokens,
    TakeDistinct,
    TakePair,
)
from splendor.catalog import Catalog
from splendor.engine import GameEngine, IllegalActionError, replay_game
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, ReservationSource, Tier, TokenVector
from splendor.observation import build_observation
from splendor.replay import new_replay, state_digest
from splendor.rng import DeterministicRng
from splendor.rules import apply_legal_action, legal_actions


def _engine(state: GameState, catalog: Catalog, failure_dir: Path) -> GameEngine:
    return GameEngine(state, catalog, new_replay(catalog, 123, state.first_player), failure_dir)


def _distinct(actions: tuple[object, ...]) -> tuple[TakeDistinct, ...]:
    return tuple(action for action in actions if isinstance(action, TakeDistinct))


def test_seeded_setup_is_complete_and_reproducible(catalog: Catalog, tmp_path: Path) -> None:
    first = GameEngine.new(2026, catalog=catalog, failure_dir=tmp_path / "a")
    second = GameEngine.new(2026, catalog=catalog, failure_dir=tmp_path / "b")
    explicit = GameEngine.new(2026, first_player=1, catalog=catalog, failure_dir=tmp_path / "c")

    assert state_digest(first._debug_state()) == state_digest(second._debug_state())
    state = first._debug_state()
    assert tuple(len(state.market.for_tier(tier)) for tier in Tier) == (4, 4, 4)
    assert tuple(len(state.decks.for_tier(tier)) for tier in Tier) == (36, 26, 16)
    assert len(state.nobles_in_play) == 3
    assert state.supply == TokenVector(white=4, blue=4, green=4, red=4, black=4, gold=5)
    tier_one_csv_order = tuple(card.card_id for card in catalog.cards if card.tier is Tier.ONE)
    assert state.decks.tier_one + state.market.tier_one != tier_one_csv_order
    assert explicit._debug_state().first_player == 1
    assert_game_state(state, catalog)


def test_decision_candidates_are_reused_for_the_following_apply(
    catalog: Catalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    authoritative = engine_module.legal_actions

    def counted(state, current_catalog):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return authoritative(state, current_catalog)

    monkeypatch.setattr(engine_module, "legal_actions", counted)
    engine = GameEngine.new(2027, first_player=0, catalog=catalog, failure_dir=tmp_path)
    request = engine.decision()
    assert calls == 1

    engine.apply(request.legal_actions[0])

    assert calls == 2


def test_transition_digest_is_chained_and_fork_skips_only_parent_validation(
    catalog: Catalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = GameEngine.new(2028, first_player=0, catalog=catalog, failure_dir=tmp_path)
    request = engine.decision()
    initial_digest = engine.current_state_digest
    assert initial_digest == state_digest(engine._debug_state())

    checks = 0
    authoritative_check = replay_module.InvariantGuard.check

    def counted_check(self, state, attempted_action=None):  # type: ignore[no-untyped-def]
        nonlocal checks
        checks += 1
        return authoritative_check(self, state, attempted_action)

    monkeypatch.setattr(replay_module.InvariantGuard, "check", counted_check)
    forked = engine.fork(failure_dir=tmp_path / "fork")
    assert checks == 0
    assert forked.current_state_digest == initial_digest
    assert forked.replay_record == engine.replay_record

    forked.apply(request.legal_actions[0])

    assert checks == 1
    event = forked.replay_record.events[-1]
    assert event.state_digest_before == initial_digest
    assert event.state_digest_after == forked.current_state_digest
    assert forked.current_state_digest == state_digest(forked._debug_state())
    assert engine.current_state_digest == initial_digest
    assert not engine.replay_record.events


@pytest.mark.parametrize(
    ("supply", "player_zero", "player_one", "expected"),
    [
        (
            TokenVector(white=1, blue=2, gold=5),
            TokenVector(white=3, blue=2, green=4, red=1),
            TokenVector(red=3, black=4),
            ((Color.WHITE, Color.BLUE),),
        ),
        (
            TokenVector(white=1, gold=5),
            TokenVector(white=3, blue=4, green=3),
            TokenVector(green=1, red=4, black=4),
            ((Color.WHITE,),),
        ),
        (
            TokenVector(gold=5),
            TokenVector(white=4, blue=4, green=2),
            TokenVector(green=2, red=4, black=4),
            (),
        ),
    ],
)
def test_take_distinct_with_two_one_or_zero_nonempty_colors(
    initial_state: GameState,
    catalog: Catalog,
    supply: TokenVector,
    player_zero: TokenVector,
    player_one: TokenVector,
    expected: tuple[tuple[Color, ...], ...],
) -> None:
    state = replace(
        initial_state,
        supply=supply,
        players=(PlayerState(player_zero), PlayerState(player_one)),
    )
    assert_game_state(state, catalog)
    assert tuple(action.colors for action in _distinct(legal_actions(state, catalog))) == expected


def test_take_distinct_with_at_least_three_colors_enumerates_only_triples(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    actions = _distinct(legal_actions(initial_state, catalog))
    assert len(actions) == comb(5, 3)
    assert all(len(action.colors) == 3 for action in actions)

    chosen = TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN))
    engine = _engine(initial_state, catalog, tmp_path)
    engine.apply(chosen)
    state = engine._debug_state()
    assert state.players[0].tokens == TokenVector(white=1, blue=1, green=1)
    assert state.supply == TokenVector(white=3, blue=3, green=3, red=4, black=4, gold=5)
    assert state.current_player == 1


def test_take_pair_uses_bank_count_before_the_action(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    initial_pairs = {action.color for action in legal_actions(initial_state, catalog) if isinstance(action, TakePair)}
    assert initial_pairs == {Color.WHITE, Color.BLUE, Color.GREEN, Color.RED, Color.BLACK}

    bank_three = replace(
        initial_state,
        supply=replace(initial_state.supply, white=3),
        players=(PlayerState(TokenVector(white=1)), initial_state.players[1]),
    )
    actions = legal_actions(bank_three, catalog)
    assert TakePair(Color.WHITE) not in actions

    engine = _engine(initial_state, catalog, tmp_path)
    engine.apply(TakePair(Color.WHITE))
    state = engine._debug_state()
    assert state.supply.white == 2
    assert state.players[0].tokens.white == 2


def test_public_reserve_refills_market_and_takes_gold(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    target = initial_state.market.tier_one[0]
    refill = initial_state.decks.tier_one[-1]
    engine = _engine(initial_state, catalog, tmp_path)
    engine.apply(ReserveMarket(target))
    state = engine._debug_state()

    assert ReservedCard(target, ReservationSource.MARKET) in state.players[0].reserved_cards
    assert target not in state.market.tier_one
    assert refill in state.market.tier_one
    assert len(state.decks.tier_one) == 35
    assert state.players[0].tokens.gold == 1
    assert state.supply.gold == 4


def test_reserve_without_gold_still_succeeds(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    no_gold = replace(
        initial_state,
        supply=replace(initial_state.supply, gold=0),
        players=(PlayerState(TokenVector(gold=5)), initial_state.players[1]),
    )
    target = no_gold.market.tier_two[0]
    engine = _engine(no_gold, catalog, tmp_path)
    engine.apply(ReserveMarket(target))
    assert engine._debug_state().players[0].tokens.gold == 5

    blind_engine = _engine(no_gold, catalog, tmp_path / "blind")
    blind_engine.apply(ReserveDeck(Tier.THREE))
    assert blind_engine._debug_state().players[0].tokens.gold == 5


def test_reserve_gold_can_trigger_return_and_be_returned(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    player = PlayerState(TokenVector(white=2, blue=2, green=2, red=2, black=2))
    state = replace(
        initial_state,
        supply=TokenVector(white=2, blue=2, green=2, red=2, black=2, gold=5),
        players=(player, initial_state.players[1]),
    )
    engine = _engine(state, catalog, tmp_path)
    engine.apply(ReserveMarket(state.market.tier_one[0]))
    assert engine._debug_state().phase is DecisionPhase.RETURN_TOKENS
    assert ReturnTokens(TokenVector(gold=1)) in engine.decision().legal_actions
    engine.apply(ReturnTokens(TokenVector(gold=1)))
    assert engine._debug_state().players[0].tokens.total == 10


def test_empty_deck_disables_blind_reserve_and_public_market_is_not_refilled(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    removed = initial_state.decks.tier_one
    claimed = initial_state.nobles_in_play
    opponent = replace(
        initial_state.players[1],
        purchased_card_ids=removed,
        noble_ids=claimed,
    )
    exhausted = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(Tier.ONE, ()),
        players=(initial_state.players[0], opponent),
        nobles_in_play=(),
    )
    assert_game_state(exhausted, catalog)
    assert ReserveDeck(Tier.ONE) not in legal_actions(exhausted, catalog)

    target = exhausted.market.tier_one[0]
    engine = _engine(exhausted, catalog, tmp_path)
    engine.apply(ReserveMarket(target))
    state = engine._debug_state()
    assert len(state.market.tier_one) == 3
    assert not state.decks.tier_one


def test_reserve_limit_removes_all_reserve_actions(
    initial_state: GameState, catalog: Catalog
) -> None:
    reserved_ids = initial_state.decks.tier_one[-3:]
    player = replace(
        initial_state.players[0],
        reserved_cards=tuple(ReservedCard(card_id, ReservationSource.DECK) for card_id in reserved_ids),
    )
    full = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(Tier.ONE, initial_state.decks.tier_one[:-3]),
        players=(player, initial_state.players[1]),
    )
    assert_game_state(full, catalog)
    actions = legal_actions(full, catalog)
    assert not any(isinstance(action, (ReserveMarket, ReserveDeck)) for action in actions)


def _returning_engine(initial_state: GameState, catalog: Catalog, failure_dir: Path) -> GameEngine:
    player = PlayerState(TokenVector(blue=2, green=2, red=2, black=2, gold=1))
    state = replace(
        initial_state,
        supply=TokenVector(white=4, blue=2, green=2, red=2, black=2, gold=4),
        players=(player, initial_state.players[1]),
    )
    engine = _engine(state, catalog, failure_dir)
    engine.apply(TakePair(Color.WHITE))
    return engine


def test_return_phase_enumerates_all_six_colors_and_accepts_new_token_or_gold(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    engine = _returning_engine(initial_state, catalog, tmp_path)
    state = engine._debug_state()
    assert state.phase is DecisionPhase.RETURN_TOKENS
    assert state.pending is not None and state.pending.tokens_to_return == 1

    actions = engine.decision().legal_actions
    assert len(actions) == 6
    assert ReturnTokens(TokenVector(white=1)) in actions  # just taken
    assert ReturnTokens(TokenVector(gold=1)) in actions
    assert all(action.tokens.total == 1 for action in actions if isinstance(action, ReturnTokens))

    engine.apply(ReturnTokens(TokenVector(gold=1)))
    finished = engine._debug_state()
    assert finished.phase is DecisionPhase.MAIN_ACTION
    assert finished.current_player == 1
    assert finished.completed_turns == (1, 0)
    assert finished.players[0].tokens.total == 10
    assert finished.supply.gold == 5


def test_return_enumeration_matches_every_bounded_vector_for_excess_three(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    player = PlayerState(TokenVector(white=2, blue=1, green=1, red=2, black=2, gold=2))
    state = replace(
        initial_state,
        supply=TokenVector(white=2, blue=3, green=3, red=2, black=2, gold=3),
        players=(player, initial_state.players[1]),
    )
    engine = _engine(state, catalog, tmp_path)
    engine.apply(TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)))
    held = engine._debug_state().players[0].tokens
    actual = {
        action.tokens
        for action in engine.decision().legal_actions
        if isinstance(action, ReturnTokens)
    }
    expected = {
        TokenVector(*values)
        for values in product(*(range(amount + 1) for amount in held))
        if sum(values) == 3
    }
    assert actual == expected


def test_blind_identity_is_private_but_market_reservation_remains_known(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    blind_engine = _engine(initial_state, catalog, tmp_path / "blind")
    opponent_request = blind_engine.apply(ReserveDeck(Tier.ONE))
    state = blind_engine._debug_state()
    hidden = state.players[0].reserved_cards[0]
    assert hidden.source is ReservationSource.DECK
    assert hidden.card_id == initial_state.decks.tier_one[-1]
    assert len(state.decks.tier_one) == len(initial_state.decks.tier_one) - 1
    assert opponent_request.observation.opponent_public.reserved_cards[0].card is None
    assert not any(opponent_request.observation.public_purchased)
    assert build_observation(state, 0, catalog).self_private.blind_reserved_cards[0].card_id == hidden.card_id

    other = state.decks.tier_one[-1]
    swapped_player = replace(
        state.players[0],
        reserved_cards=(ReservedCard(other, ReservationSource.DECK),),
    )
    swapped = replace(
        state,
        decks=state.decks.replacing_tier(
            Tier.ONE,
            state.decks.tier_one[:-1] + (hidden.card_id,),
        ),
        players=(swapped_player, state.players[1]),
    )
    assert_game_state(swapped, catalog)
    assert build_observation(state, 1, catalog) == build_observation(swapped, 1, catalog)
    assert legal_actions(state, catalog) == legal_actions(swapped, catalog)
    assert build_observation(state, 0, catalog) != build_observation(swapped, 0, catalog)

    public_engine = _engine(initial_state, catalog, tmp_path / "public")
    target = initial_state.market.tier_three[0]
    public_view = public_engine.apply(ReserveMarket(target)).observation
    known = public_view.opponent_public.reserved_cards[0]
    assert known.card is not None and known.card.card_id == target


def test_save_restore_at_return_phase_continues_identically(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    original = _returning_engine(initial_state, catalog, tmp_path / "original_failures")
    save_path = tmp_path / "return-state.json"
    original.save(save_path)
    restored = GameEngine.restore(save_path, catalog=catalog, failure_dir=tmp_path / "restored_failures")

    assert state_digest(original._debug_state()) == state_digest(restored._debug_state())
    assert original.decision() == restored.decision()
    action = ReturnTokens(TokenVector(white=1))
    original.apply(action)
    restored.apply(action)
    assert state_digest(original._debug_state()) == state_digest(restored._debug_state())
    assert original.replay_record == restored.replay_record


def test_same_seed_and_action_sequence_replays_exactly(catalog: Catalog, tmp_path: Path) -> None:
    engine = GameEngine.new(991, catalog=catalog, failure_dir=tmp_path / "source")
    engine.apply(ReserveDeck(Tier.ONE))
    engine.apply(TakePair(Color.WHITE))
    take = next(action for action in engine.decision().legal_actions if isinstance(action, TakeDistinct))
    engine.apply(take)

    replayed = replay_game(engine.replay_record, catalog=catalog, failure_dir=tmp_path / "replay")
    assert replayed.replay_record == engine.replay_record
    assert state_digest(replayed._debug_state()) == state_digest(engine._debug_state())


def test_actions_from_the_wrong_phase_fail_and_save_scene(
    catalog: Catalog, tmp_path: Path
) -> None:
    engine = GameEngine.new(77, first_player=0, catalog=catalog, failure_dir=tmp_path)
    before = state_digest(engine._debug_state())
    with pytest.raises(IllegalActionError):
        engine.apply(ReturnTokens(TokenVector(white=1)))
    with pytest.raises(IllegalActionError):
        engine.apply(ClaimNoble(engine._debug_state().nobles_in_play[0]))
    assert state_digest(engine._debug_state()) == before
    assert len(list(tmp_path.glob("failure-*.json"))) == 2


def test_enumerated_actions_preserve_invariants_across_many_transitions(
    catalog: Catalog, tmp_path: Path
) -> None:
    chooser = DeterministicRng(4567)
    for seed in range(12):
        state = GameEngine.new(seed, catalog=catalog, failure_dir=tmp_path / str(seed))._debug_state()
        for _ in range(40):
            actions = legal_actions(state, catalog)
            if not actions:
                break
            state = apply_legal_action(state, actions[chooser.randbelow(len(actions))], catalog)
            assert_game_state(state, catalog)
