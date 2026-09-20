from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from splendor._state import ReservedCard
from splendor.actor_critic import ActorCritic
from splendor.contracts import DecisionRequest
from splendor.evaluation import PolicySpec, baseline_policy_spec
from splendor.features import encode_decision
from splendor.invariants import assert_game_state
from splendor.model import ReservationSource, Tier
from splendor.neural_policy import (
    GreedyActorCriticPolicy,
    SamplingActorCriticPolicy,
    frozen_actor_critic_copy,
)
from splendor.observation import build_observation
from splendor.opponent_pool import OpponentPool
from splendor.ppo import BatchedRolloutCollector, PPOTrainer, PolicySampler
from splendor.replay import RULESET_ID
from splendor.rules import legal_actions
from splendor.train_5b import evaluate_fixed_panel
from splendor.train_5b_stage4 import (
    STAGE4_LEARNING_RATE,
    STAGE4_MIX,
    Stage4Config,
    Stage4OpponentSelector,
    _configure_stage4_learning_rate,
    evaluate_stage4_panel,
    run_stage4_training,
)
from splendor.training_env import LearningEnvironment


def _checkpoint(path: Path, *, step: int) -> tuple[ActorCritic, Path]:
    torch.manual_seed(step)
    model = ActorCritic()
    trainer = PPOTrainer(model)
    trainer.global_learning_decision_step = step
    sampler = PolicySampler(model, seed=step)
    checkpoint = path / f"step-{step}.pt"
    trainer.save_checkpoint(
        checkpoint,
        sampler=sampler,
        metadata={"rules_version": RULESET_ID},
    )
    return model, checkpoint


def test_stage4_config_freezes_reward_mix_and_learning_rate(tmp_path: Path) -> None:
    _, checkpoint = _checkpoint(tmp_path, step=100)
    valid = {
        "historical_checkpoints": (str(checkpoint),),
        "historical_eval_steps": (100,),
    }
    config = Stage4Config(**valid)
    assert config.opponent_mix == STAGE4_MIX
    assert config.learning_rate == STAGE4_LEARNING_RATE
    with pytest.raises(ValueError, match="learning rate"):
        Stage4Config(**valid, learning_rate=3e-4)
    with pytest.raises(ValueError, match="opponent mix"):
        Stage4Config(**valid, opponent_mix={"latest": 1.0})
    with pytest.raises(ValueError, match="unshaped"):
        Stage4Config(**valid, reward_shaping_alpha=0.1)


def test_stage4_reuses_exact_stage1_fixed_dev_seed_protocol(tmp_path: Path) -> None:
    torch.manual_seed(39)
    model = ActorCritic()
    expected, _ = evaluate_fixed_panel(
        model,
        training_seed=2,
        paired_seed_count=2,
        device="cpu",
        max_completed_turns=400,
        failure_dir=tmp_path / "stage1",
    )
    frozen = frozen_actor_critic_copy(model)
    opponents = {
        name: baseline_policy_spec(name)
        for name in ("random", "score", "development")
    }
    opponents["source_best"] = PolicySpec(
        "source",
        lambda seed: GreedyActorCriticPolicy(frozen),
    )
    actual, _ = evaluate_stage4_panel(
        model,
        opponents=opponents,
        dev_panel_seed=2,
        paired_seed_count=2,
        device="cpu",
        max_completed_turns=400,
        failure_dir=tmp_path / "stage4",
    )

    for name in ("random", "score", "development"):
        assert actual[name].a_wins == expected[name].a_wins
        assert actual[name].draws == expected[name].draws
        assert actual[name].a_losses == expected[name].a_losses
        assert actual[name].win_rate_score == expected[name].win_rate_score


def test_frozen_latest_is_not_shared_or_updated_by_learner_optimizer() -> None:
    torch.manual_seed(40)
    trainer = PPOTrainer(ActorCritic())
    previous_rate = _configure_stage4_learning_rate(trainer)
    pool = OpponentPool()
    entry = pool.set_latest(trainer.model, training_step=0)
    opponent = pool.choose_for_training(
        "latest", policy_seed=41
    ).policy.model
    before = tuple(parameter.detach().clone() for parameter in opponent.parameters())

    assert previous_rate == 3e-4
    assert all(group["lr"] == STAGE4_LEARNING_RATE for group in trainer.optimizer.param_groups)
    assert all(not parameter.requires_grad for parameter in opponent.parameters())
    assert entry.opponent_id.startswith("latest-step-")
    assert all(
        learner.data_ptr() != frozen.data_ptr()
        for learner, frozen in zip(
            trainer.model.parameters(), opponent.parameters(), strict=True
        )
    )

    trainer.optimizer.zero_grad(set_to_none=True)
    for parameter in trainer.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer.optimizer.step()

    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, opponent.parameters(), strict=True)
    )
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) not in optimizer_ids for parameter in opponent.parameters())


def test_mixture_and_historical_sampling_rng_restore_exactly(tmp_path: Path) -> None:
    _, checkpoint_a = _checkpoint(tmp_path, step=100)
    _, checkpoint_b = _checkpoint(tmp_path, step=200)
    selector = Stage4OpponentSelector(50)
    first_counts = Counter(selector.choose_kind() for _ in range(10_000))
    assert 0.28 < first_counts["latest"] / 10_000 < 0.32
    assert 0.18 < first_counts["historical"] / 10_000 < 0.22
    assert 0.23 < first_counts["score"] / 10_000 < 0.27
    assert 0.23 < first_counts["development"] / 10_000 < 0.27
    snapshot = selector.snapshot
    expected_kinds = tuple(selector.choose_kind() for _ in range(50))
    restored_selector = Stage4OpponentSelector.restore(snapshot)
    assert tuple(restored_selector.choose_kind() for _ in range(50)) == expected_kinds

    pool = OpponentPool(seed=51)
    pool.add_historical(checkpoint_a, reason="early", added_at_step=300)
    pool.add_historical(checkpoint_b, reason="mid", added_at_step=300)
    for index in range(7):
        pool.choose_for_training("historical", policy_seed=index)
    manifest = tmp_path / "pool.json"
    pool.save_manifest(manifest)
    expected_ids = tuple(
        pool.choose_for_training("historical", policy_seed=100 + index).entry.opponent_id
        for index in range(30)
    )
    restored_pool = OpponentPool.load_manifest(manifest)
    assert tuple(
        restored_pool.choose_for_training(
            "historical", policy_seed=100 + index
        ).entry.opponent_id
        for index in range(30)
    ) == expected_ids


def test_sampled_neural_opponent_hidden_identity_boundary(
    initial_state, catalog
) -> None:
    hidden_x, hidden_y = initial_state.decks.tier_one[:2]

    def request_with_hidden(card_id: str) -> DecisionRequest:
        opponent = replace(
            initial_state.players[1],
            reserved_cards=(ReservedCard(card_id, ReservationSource.DECK),),
        )
        state = replace(
            initial_state,
            decks=initial_state.decks.replacing_tier(
                Tier.ONE,
                tuple(
                    current
                    for current in initial_state.decks.tier_one
                    if current != card_id
                ),
            ),
            players=(initial_state.players[0], opponent),
        )
        assert_game_state(state, catalog)
        return DecisionRequest(
            build_observation(state, 0, catalog),
            legal_actions(state, catalog),
        )

    left_request = request_with_hidden(hidden_x)
    right_request = request_with_hidden(hidden_y)
    assert left_request == right_request
    assert encode_decision(left_request).candidate_actions == encode_decision(
        right_request
    ).candidate_actions
    model = ActorCritic().eval().requires_grad_(False)
    left = SamplingActorCriticPolicy(model, seed=52)
    right = SamplingActorCriticPolicy(model, seed=52)
    assert left.choose(left_request) == right.choose(right_request)


def test_batched_sampled_neural_opponent_finishes_both_learning_seats(
    tmp_path: Path,
) -> None:
    torch.manual_seed(53)
    learner = ActorCritic()
    sampler = PolicySampler(learner, seed=53)
    pool = OpponentPool()
    latest = pool.set_latest(learner, training_step=0)
    frozen = pool.choose_for_training("latest", policy_seed=1).policy.model
    before = tuple(parameter.detach().clone() for parameter in frozen.parameters())

    def factory(index: int):
        choice = pool.choose_for_training("latest", policy_seed=1000 + index)
        environment = LearningEnvironment.new(
            5_300 + index,
            learning_player=index % 2,
            first_player=index % 2,
            opponent_policy=choice.policy,
            episode_id=f"neural-{index}",
            max_completed_turns=400,
            failure_dir=tmp_path / f"game-{index}",
        )
        return environment, 2_000 + index

    collector = BatchedRolloutCollector(sampler, factory, active_envs=2)
    collector.collect(1)
    collector.drain_active_environments()
    completed = collector.take_completed_environments()

    assert latest.opponent_id.startswith("latest-step-")
    assert {environment.learning_player for environment in completed} == {0, 1}
    assert all(environment.terminated and not environment.truncated for environment in completed)
    assert all(environment.opponent_decision_count > 0 for environment in completed)
    assert all(
        all(
            transition.learning_player == environment.learning_player
            for transition in environment.transitions
        )
        for environment in completed
    )
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, frozen.parameters(), strict=True)
    )


def test_episode_keeps_original_latest_after_pool_refresh(tmp_path: Path) -> None:
    torch.manual_seed(54)
    learner = ActorCritic()
    pool = OpponentPool()
    first_entry = pool.set_latest(learner, training_step=10)
    first_choice = pool.choose_for_training("latest", policy_seed=55)
    environment = LearningEnvironment.new(
        5_400,
        learning_player=0,
        first_player=0,
        opponent_policy=first_choice.policy,
        episode_id="fixed-opponent-version",
        failure_dir=tmp_path,
    )
    original_model = environment.opponent_policy.model

    with torch.no_grad():
        next(learner.parameters()).add_(1.0)
    second_entry = pool.set_latest(learner, training_step=11)

    assert first_entry.opponent_id != second_entry.opponent_id
    assert environment.opponent_policy is first_choice.policy
    assert environment.opponent_policy.model is original_model
    assert environment.opponent_policy.model is not pool.choose_for_training(
        "latest", policy_seed=56
    ).policy.model


def test_stage4_runner_real_save_exit_reload_and_continue(tmp_path: Path) -> None:
    _, historical = _checkpoint(tmp_path / "historical", step=100)
    _, source = _checkpoint(tmp_path / "source", step=300)
    output = tmp_path / "run"
    common = {
        "training_seed": 57,
        "device": "cpu",
        "active_envs": 2,
        "rollout_steps": 4,
        "eval_interval": 128,
        "checkpoint_interval": 128,
        "eval_pairs": 1,
        "historical_checkpoints": (str(historical),),
    }
    first = run_stage4_training(
        Stage4Config(stage_learning_decisions=128, **common),
        output_dir=output,
        source_checkpoint=source,
    )
    first_checkpoint = Path(first["latest_checkpoint"])
    assert first_checkpoint.exists()
    assert first["stage_learning_decisions"] >= 128
    assert first["ppo_config"]["learning_rate"] == STAGE4_LEARNING_RATE

    resumed = run_stage4_training(
        Stage4Config(stage_learning_decisions=256, **common),
        output_dir=output,
        resume_checkpoint=first_checkpoint,
    )
    assert resumed["stage_learning_decisions"] >= 256
    assert resumed["progress"]["resume_count"] == 1
    assert resumed["global_learning_decisions"] > first["global_learning_decisions"]
    assert Path(resumed["latest_checkpoint"]).exists()
