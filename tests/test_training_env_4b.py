from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor._state import DeckState, GameState, MarketState, PlayerState, ReservedCard
from splendor.actions import (
    Action,
    BuyReserved,
    ClaimNoble,
    Pay,
    TakeDistinct,
    TakePair,
)
from splendor.actor_critic import ActorCritic
from splendor.catalog import Catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine, replay_game
from splendor.features import collate_encoded_decisions, encode_decision
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, ReservationSource, Tier, TokenVector
from splendor.random_policy import RandomPolicy
from splendor.replay import ForcedSkipEvent, new_replay, state_digest
from splendor.result import TerminalReason, TerminalResult
from splendor.rules import legal_actions, legal_payments_for_card, player_score
from splendor.training_env import (
    InvalidCandidateIndexError,
    LearningEnvironment,
    LearningEnvironmentError,
    StaleLearningDecisionError,
    run_learning_episode,
)


class _FirstPolicy:
    def __init__(self) -> None:
        self.requests: list[DecisionRequest] = []

    def choose(self, request: DecisionRequest) -> Action:
        self.requests.append(request)
        return request.legal_actions[0]


class _ScriptedPolicy:
    def __init__(self, actions: tuple[Action, ...]) -> None:
        self.actions = list(actions)
        self.requests: list[DecisionRequest] = []

    def choose(self, request: DecisionRequest) -> Action:
        self.requests.append(request)
        if not self.actions:
            raise AssertionError("scripted opponent ran out of actions")
        action = self.actions.pop(0)
        assert action in request.legal_actions
        return action


def _engine(state: GameState, catalog: Catalog, failure_dir: Path) -> GameEngine:
    replay = new_replay(catalog, 123, state.first_player)
    return GameEngine(state, catalog, replay, failure_dir)


def _with_assets(
    state: GameState,
    catalog: Catalog,
    *,
    purchased: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
    reserved: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
    nobles: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
    tokens: tuple[TokenVector, TokenVector] = (TokenVector(), TokenVector()),
    first_player: int = 0,
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
            tokens=tokens[player],
            purchased_card_ids=purchased[player],
            reserved_cards=tuple(
                ReservedCard(card_id, ReservationSource.DECK)
                for card_id in reserved[player]
            ),
            noble_ids=nobles[player],
        )
        for player in (0, 1)
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
        first_player=first_player,
        current_player=current_player,
        completed_turns=completed_turns,
        final_round_triggered_by=final_round_triggered_by,
    )
    assert_game_state(result, catalog)
    return result


def _eligible_cards(
    state: GameState, catalog: Catalog, noble_ids: tuple[str, ...]
) -> tuple[str, ...]:
    required = {color: 0 for color in tuple(Color)[:-1]}
    for noble_id in noble_ids:
        requirement = catalog.noble(noble_id).requirement
        for color in required:
            required[color] = max(required[color], requirement.for_color(color))
    chosen: list[str] = []
    deck_ids = set(state.decks.all_cards())
    for color, count in required.items():
        candidates = [
            card.card_id
            for card in catalog.cards
            if card.card_id in deck_ids and card.bonus is color and card.points == 0
        ]
        chosen.extend(candidates[:count])
    return tuple(chosen)


def _multiple_noble_purchase_state(
    initial_state: GameState, catalog: Catalog
) -> tuple[GameState, str]:
    bonus_cards = _eligible_cards(initial_state, catalog, ("N01", "N02"))
    target = next(
        card
        for card in catalog.cards
        if card.card_id in initial_state.decks.all_cards()
        and card.card_id not in bonus_cards
        and card.tier is Tier.ONE
    )
    tokens = TokenVector(*target.cost)
    state = _with_assets(
        initial_state,
        catalog,
        purchased=(bonus_cards, ()),
        reserved=((target.card_id,), ()),
        tokens=(tokens, TokenVector()),
    )
    return state, target.card_id


def test_main_payment_noble_are_separate_same_player_transitions(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state, target = _multiple_noble_purchase_state(initial_state, catalog)
    opponent = _FirstPolicy()
    environment = LearningEnvironment(
        _engine(state, catalog, tmp_path),
        learning_player=0,
        opponent_policy=opponent,
        episode_id="multi-stage",
    )

    main = environment.current_decision()
    buy_index = main.legal_actions.index(BuyReserved(target))
    buy = environment.step_learning_action(buy_index, decision_id=main.decision_id)
    assert buy.decision.observation.phase is DecisionPhase.MAIN_ACTION
    assert buy.chosen_action == BuyReserved(target)
    assert buy.reward == 0
    assert not buy.terminated and not buy.truncated
    assert buy.next_decision is not None
    assert buy.next_decision.observation.phase is DecisionPhase.PAYMENT
    assert buy.next_decision.observation.self_is_first_player
    assert buy.next_decision.encoded.candidate_actions == buy.next_decision.legal_actions

    payment = environment.current_decision()
    pay = environment.step_learning_action(0, decision_id=payment.decision_id)
    assert pay.decision is buy.next_decision
    assert pay.decision.observation.phase is DecisionPhase.PAYMENT
    assert pay.reward == 0
    assert pay.next_decision is not None
    assert pay.next_decision.observation.phase is DecisionPhase.CHOOSE_NOBLE
    assert pay.next_decision.observation.pending is not None
    assert pay.next_decision.observation.pending.payment_target is None
    assert not pay.next_decision.encoded.pending_card_mask
    assert len(pay.next_decision.legal_actions) >= 2

    noble = environment.current_decision()
    claim = environment.step_learning_action(0, decision_id=noble.decision_id)
    assert claim.decision is pay.next_decision
    assert claim.decision.observation.phase is DecisionPhase.CHOOSE_NOBLE
    assert isinstance(claim.chosen_action, ClaimNoble)
    assert claim.reward == 0
    assert not claim.terminated and not claim.truncated
    assert claim.next_decision is not None
    assert claim.next_decision.observation.phase is DecisionPhase.MAIN_ACTION
    assert environment.current_decision() is claim.next_decision
    assert all(transition.learning_player == 0 for transition in environment.transitions)
    assert environment.opponent_decision_count == len(opponent.requests) >= 1
    assert all(isinstance(request, DecisionRequest) for request in opponent.requests)
    assert claim.completed_turns_after == (1, 1)


def test_candidate_index_mapping_and_stale_decision_fail_explicitly(
    tmp_path: Path,
) -> None:
    environment = LearningEnvironment.new(
        8,
        first_player=0,
        learning_player=0,
        opponent_policy=_FirstPolicy(),
        episode_id="candidate-boundary",
        failure_dir=tmp_path,
    )
    decision = environment.current_decision()
    before_events = environment.replay_record.events
    with pytest.raises(InvalidCandidateIndexError):
        environment.step_learning_action(-1, decision_id=decision.decision_id)
    assert environment.replay_record.events == before_events
    assert len(list(tmp_path.glob("failure-*-invalid_candidate_index.json"))) == 1

    action = next(
        candidate for candidate in decision.legal_actions if isinstance(candidate, TakePair)
    )
    index = decision.legal_actions.index(action)
    transition = environment.step_learning_action(index, decision_id=decision.decision_id)
    assert transition.chosen_candidate_index == index
    assert transition.chosen_action == action
    assert environment.replay_record.events[transition.replay_event_start].action == action
    with pytest.raises(StaleLearningDecisionError):
        environment.step_learning_action(0, decision_id=decision.decision_id)
    assert len(list(tmp_path.glob("failure-*-stale_learning_decision.json"))) == 1


def test_engineering_truncation_waits_for_real_learning_successor(
    catalog: Catalog, tmp_path: Path
) -> None:
    environment = LearningEnvironment.new(
        5,
        first_player=0,
        learning_player=0,
        opponent_policy=_FirstPolicy(),
        episode_id="safe-truncation",
        max_completed_turns=1,
        catalog=catalog,
        failure_dir=tmp_path / "source",
    )
    decision = environment.current_decision()
    take_pair = next(
        action for action in decision.legal_actions if isinstance(action, TakePair)
    )
    transition = environment.step_learning_action(
        decision.legal_actions.index(take_pair), decision_id=decision.decision_id
    )

    assert transition.truncated and not transition.terminated
    assert transition.reward == 0
    assert transition.terminal_result is None
    assert transition.next_decision is not None
    assert transition.next_decision.observation.phase is DecisionPhase.MAIN_ACTION
    assert transition.next_decision.observation.own_completed_turns == 1
    assert transition.next_decision.observation.opponent_completed_turns == 1
    assert environment.bootstrap_decision is transition.next_decision
    assert environment.replay_record.result is None
    with pytest.raises(LearningEnvironmentError, match="truncated"):
        environment.current_decision()

    episode = environment.episode_run(learning_policy_seed=10, opponent_seed=20)
    assert episode.truncated and not episode.terminated
    assert episode.terminal_result is None and episode.winner_ids is None
    assert episode.bootstrap_decision is transition.next_decision
    assert episode.completed_turns == (1, 1)
    replayed = replay_game(
        episode.replay, catalog=catalog, failure_dir=tmp_path / "replayed"
    )
    assert state_digest(replayed._debug_state()) == episode.final_state_digest
    assert replayed.decision() == transition.next_decision.request


@pytest.mark.parametrize("learning_player", (0, 1))
def test_complete_episode_is_single_player_and_seed_reproducible(
    learning_player: int, tmp_path: Path
) -> None:
    arguments = dict(
        game_seed=13,
        learning_player=learning_player,
        learning_policy_seed=101,
        opponent_seed=202,
        first_player=0,
        episode_id=f"deterministic-{learning_player}",
    )
    torch.manual_seed(1)
    left = run_learning_episode(
        RandomPolicy(101),
        RandomPolicy(202),
        failure_dir=tmp_path / "left",
        **arguments,
    )
    torch.manual_seed(999)
    right = run_learning_episode(
        RandomPolicy(101),
        RandomPolicy(202),
        failure_dir=tmp_path / "right",
        **arguments,
    )

    assert left.terminated and not left.truncated
    assert left.replay == right.replay
    assert left.terminal_result == right.terminal_result
    assert left.final_state_digest == right.final_state_digest
    assert left.learning_decision_count == len(left.transitions)
    assert left.opponent_decision_count > 0
    assert all(item.learning_player == learning_player for item in left.transitions)
    assert [
        (
            item.decision.decision_id,
            item.chosen_action,
            item.reward,
            item.terminated,
            item.truncated,
            item.replay_event_start,
            item.replay_event_end,
        )
        for item in left.transitions
    ] == [
        (
            item.decision.decision_id,
            item.chosen_action,
            item.reward,
            item.terminated,
            item.truncated,
            item.replay_event_start,
            item.replay_event_end,
        )
        for item in right.transitions
    ]
    for left_item, right_item in zip(left.transitions, right.transitions, strict=True):
        np.testing.assert_array_equal(
            left_item.decision.encoded.candidate_features,
            right_item.decision.encoded.candidate_features,
        )


def test_actor_critic_sampled_candidate_executes_through_engine(
    initial_state: GameState, catalog: Catalog, tmp_path: Path,
) -> None:
    environment = LearningEnvironment.new(
        21,
        first_player=0,
        learning_player=0,
        opponent_policy=RandomPolicy(303),
        episode_id="model-sanity",
        failure_dir=tmp_path,
    )
    decision = environment.current_decision()
    payment_state, target = _multiple_noble_purchase_state(initial_state, catalog)
    payment_environment = LearningEnvironment(
        _engine(payment_state, catalog, tmp_path / "payment"),
        learning_player=0,
        opponent_policy=_FirstPolicy(),
        episode_id="model-payment-sanity",
    )
    purchase = payment_environment.current_decision()
    payment_environment.step_learning_action(
        purchase.legal_actions.index(BuyReserved(target)),
        decision_id=purchase.decision_id,
    )
    payment_decision = payment_environment.current_decision()
    assert payment_decision.observation.phase is DecisionPhase.PAYMENT

    batch = collate_encoded_decisions(
        (decision.encoded, payment_decision.encoded)
    )
    torch.manual_seed(404)
    model = ActorCritic().eval()
    distribution, values = model.distribution(batch)
    sampled_indices = distribution.sample()

    assert distribution.logits.shape == batch.candidate_mask.shape
    assert values.shape == (2,)
    assert not batch.candidate_mask[1, len(payment_decision.legal_actions) :].any()
    assert not distribution.probs[~torch.as_tensor(batch.candidate_mask)].any()
    sampled_index = int(sampled_indices[0].item())
    sampled_payment_index = int(sampled_indices[1].item())
    assert batch.candidate_mask[0, sampled_index]
    assert batch.candidate_mask[1, sampled_payment_index]
    transition = environment.step_learning_action(
        sampled_index, decision_id=decision.decision_id
    )
    payment_transition = payment_environment.step_learning_action(
        sampled_payment_index,
        decision_id=payment_decision.decision_id,
    )
    assert transition.chosen_action == decision.encoded.candidate_actions[sampled_index]
    assert payment_transition.chosen_action == (
        payment_decision.encoded.candidate_actions[sampled_payment_index]
    )
    assert environment.replay_record.events[transition.replay_event_start].action == (
        transition.chosen_action
    )
    if transition.next_decision is not None:
        direct = encode_decision(transition.next_decision.request)
        np.testing.assert_array_equal(
            transition.next_decision.encoded.global_features,
            direct.global_features,
        )
        np.testing.assert_array_equal(
            transition.next_decision.encoded.candidate_features,
            direct.candidate_features,
        )
        assert transition.next_decision.encoded.candidate_actions == direct.candidate_actions


def test_episode_runner_passes_only_decision_requests_to_both_policies(
    tmp_path: Path,
) -> None:
    learning = _FirstPolicy()
    opponent = _FirstPolicy()
    episode = run_learning_episode(
        learning,
        opponent,
        game_seed=2,
        learning_player=0,
        learning_policy_seed=11,
        opponent_seed=22,
        first_player=0,
        max_completed_turns=1,
        failure_dir=tmp_path,
    )

    assert episode.truncated and not episode.terminated
    assert learning.requests and opponent.requests
    assert all(
        isinstance(request, DecisionRequest)
        for request in (*learning.requests, *opponent.requests)
    )


def _opponent_overtake_state(
    initial_state: GameState,
    catalog: Catalog,
    learning_player: int,
) -> GameState:
    learner_cards = ("078", "082", "086")
    opponent_cards = ("090", "076", "080", "008")
    purchased = [(), ()]
    purchased[learning_player] = learner_cards
    purchased[1 - learning_player] = opponent_cards
    reserved = [(), ()]
    reserved[1 - learning_player] = ("045",)
    tokens = [TokenVector(), TokenVector()]
    tokens[1 - learning_player] = TokenVector(white=4)
    state = _with_assets(
        initial_state,
        catalog,
        purchased=(purchased[0], purchased[1]),
        reserved=(reserved[0], reserved[1]),
        tokens=(tokens[0], tokens[1]),
        first_player=learning_player,
        current_player=learning_player,
    )
    assert player_score(state.players[learning_player], catalog) == 15
    assert player_score(state.players[1 - learning_player], catalog) == 14
    return state


@pytest.mark.parametrize("learning_player", (0, 1))
def test_opponent_final_turn_loss_is_assigned_to_previous_learning_transition(
    learning_player: int,
    initial_state: GameState,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    state = _opponent_overtake_state(initial_state, catalog, learning_player)
    opponent = _ScriptedPolicy(
        (BuyReserved("045"), Pay(TokenVector(white=4)))
    )
    environment = LearningEnvironment(
        _engine(state, catalog, tmp_path / str(learning_player)),
        learning_player=learning_player,
        opponent_policy=opponent,
        episode_id=f"opponent-overtake-{learning_player}",
    )
    decision = environment.current_decision()
    take = TakePair(Color.BLUE)
    transition = environment.step_learning_action(
        decision.legal_actions.index(take), decision_id=decision.decision_id
    )

    assert transition.terminated and not transition.truncated
    assert transition.reward == -1
    assert transition.next_decision is None
    assert transition.terminal_result is not None
    assert transition.terminal_result.winner_ids == (1 - learning_player,)
    assert transition.terminal_result.scores[1 - learning_player] == 16
    assert environment.opponent_decision_count == 2
    assert [request.observation.phase for request in opponent.requests] == [
        DecisionPhase.MAIN_ACTION,
        DecisionPhase.PAYMENT,
    ]
    assert len(environment.transitions) == 1


@pytest.mark.parametrize("learning_player", (0, 1))
def test_terminal_win_reward_is_relative_to_either_learning_seat(
    learning_player: int,
    initial_state: GameState,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    purchased = [(), ()]
    purchased[learning_player] = ("078", "082", "086")
    state = _with_assets(
        initial_state,
        catalog,
        purchased=(purchased[0], purchased[1]),
        first_player=learning_player,
        current_player=learning_player,
    )
    environment = LearningEnvironment(
        _engine(state, catalog, tmp_path / str(learning_player)),
        learning_player=learning_player,
        opponent_policy=_FirstPolicy(),
        episode_id=f"relative-win-{learning_player}",
    )
    decision = environment.current_decision()
    take = TakePair(Color.BLUE)
    transition = environment.step_learning_action(
        decision.legal_actions.index(take), decision_id=decision.decision_id
    )

    assert transition.terminated and not transition.truncated
    assert transition.reward == 1
    assert transition.terminal_result is not None
    assert transition.terminal_result.winner_ids == (learning_player,)


def test_purchase_target_has_zero_reward_and_payment_gets_formal_win(
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
    environment = LearningEnvironment(
        _engine(state, catalog, tmp_path),
        learning_player=1,
        opponent_policy=_FirstPolicy(),
        episode_id="purchase-win",
    )
    main = environment.current_decision()
    selected = environment.step_learning_action(
        main.legal_actions.index(BuyReserved("086")), decision_id=main.decision_id
    )
    assert selected.reward == 0
    assert not selected.terminated
    assert selected.next_decision is not None
    assert selected.next_decision.observation.phase is DecisionPhase.PAYMENT

    payment = environment.current_decision()
    paid = environment.step_learning_action(
        len(payment.legal_actions) - 1, decision_id=payment.decision_id
    )
    assert paid.terminated and not paid.truncated
    assert paid.reward == 1
    assert paid.terminal_result is not None
    assert paid.terminal_result.scores[1] == 15
    assert paid.terminal_result.winner_ids == (1,)


def test_formal_score_and_card_count_draw_is_terminated_not_truncated(
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
    environment = LearningEnvironment(
        _engine(state, catalog, tmp_path),
        learning_player=1,
        opponent_policy=_FirstPolicy(),
        episode_id="formal-draw",
    )
    decision = environment.current_decision()
    action = TakePair(Color.WHITE)
    transition = environment.step_learning_action(
        decision.legal_actions.index(action), decision_id=decision.decision_id
    )

    assert transition.terminated and not transition.truncated
    assert transition.reward == 0
    assert transition.terminal_result is not None
    assert transition.terminal_result.reason is TerminalReason.FINAL_ROUND
    assert transition.terminal_result.winner_ids == (0, 1)


def _forced_skip_draw_state(initial_state: GameState, catalog: Catalog) -> GameState:
    before_learning = TokenVector(white=4, blue=4, green=1)
    after_learning = before_learning.plus(TokenVector(black=1))
    opponent_tokens = TokenVector(green=3, red=4, black=3)
    player_after = PlayerState(after_learning)
    opponent_after = PlayerState(opponent_tokens)
    unavailable_by_tier = {
        tier: [
            card.card_id
            for card in catalog.cards
            if card.tier is tier
            and not legal_payments_for_card(player_after, card, catalog)
            and not legal_payments_for_card(opponent_after, card, catalog)
        ]
        for tier in Tier
    }
    market_by_tier = {
        tier: tuple(unavailable_by_tier[tier][:4]) for tier in Tier
    }
    market_ids = set(card for cards in market_by_tier.values() for card in cards)
    reserve_candidates = [
        card_id
        for tier in Tier
        for card_id in unavailable_by_tier[tier]
        if card_id not in market_ids
    ]
    player_reserve = tuple(reserve_candidates[:3])
    opponent_reserve = tuple(reserve_candidates[3:6])
    reserved_ids = set(player_reserve + opponent_reserve)
    deck_by_tier = {
        tier: tuple(
            card.card_id
            for card in catalog.cards
            if card.tier is tier
            and card.card_id not in market_ids
            and card.card_id not in reserved_ids
        )
        for tier in Tier
    }
    state = replace(
        initial_state,
        supply=TokenVector(black=1, gold=5),
        decks=DeckState(
            deck_by_tier[Tier.ONE],
            deck_by_tier[Tier.TWO],
            deck_by_tier[Tier.THREE],
        ),
        market=MarketState(
            market_by_tier[Tier.ONE],
            market_by_tier[Tier.TWO],
            market_by_tier[Tier.THREE],
        ),
        players=(
            PlayerState(
                before_learning,
                reserved_cards=tuple(
                    ReservedCard(card_id, ReservationSource.DECK)
                    for card_id in player_reserve
                ),
            ),
            PlayerState(
                opponent_tokens,
                reserved_cards=tuple(
                    ReservedCard(card_id, ReservationSource.DECK)
                    for card_id in opponent_reserve
                ),
            ),
        ),
        first_player=0,
        current_player=0,
        completed_turns=(0, 0),
        consecutive_forced_skips=0,
        phase=DecisionPhase.MAIN_ACTION,
        pending=None,
        final_round_triggered_by=None,
        result=None,
    )
    assert_game_state(state, catalog)
    assert legal_actions(state, catalog) == (TakeDistinct((Color.BLACK,)),)
    return state


def test_consecutive_forced_skip_draw_is_formal_zero_reward(
    initial_state: GameState, catalog: Catalog, tmp_path: Path
) -> None:
    state = _forced_skip_draw_state(initial_state, catalog)
    environment = LearningEnvironment(
        _engine(state, catalog, tmp_path),
        learning_player=0,
        opponent_policy=_FirstPolicy(),
        episode_id="forced-skip-draw",
    )
    decision = environment.current_decision()
    transition = environment.step_learning_action(0, decision_id=decision.decision_id)

    assert transition.chosen_action == TakeDistinct((Color.BLACK,))
    assert transition.terminated and not transition.truncated
    assert transition.reward == 0
    assert transition.terminal_result is not None
    assert transition.terminal_result.reason is TerminalReason.CONSECUTIVE_FORCED_SKIPS
    assert transition.terminal_result.winner_ids == (0, 1)
    events = environment.replay_record.events[transition.replay_event_start :]
    assert len(events) == 3
    assert all(isinstance(event, ForcedSkipEvent) for event in events[-2:])
