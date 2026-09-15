from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.ppo import PPOTrainer, PolicySampler, RolloutCollector
from splendor.random_policy import RandomPolicy
from splendor.train_5a import (
    FiveATrainingConfig,
    _finish_active_episode,
    dev_eval_game_seeds,
    episode_plan,
    evaluate_actor_critic_vs_random,
    main,
    summarize_rollout,
    training_game_seed,
)
from splendor.training_env import LearningEnvironment


def test_train_and_dev_seed_domains_are_disjoint_and_seat_is_randomized() -> None:
    training_seeds = {training_game_seed(7, index) for index in range(1000)}
    evaluation_seeds = set(dev_eval_game_seeds(7, 200))
    plans = tuple(episode_plan(7, index) for index in range(100))

    assert training_seeds.isdisjoint(evaluation_seeds)
    assert all(seed >= 0 for seed in training_seeds)
    assert all(seed < 0 for seed in evaluation_seeds)
    assert {plan.learning_player for plan in plans} == {0, 1}
    assert {plan.first_player for plan in plans} == {0, 1}
    assert len({plan.game_seed for plan in plans}) == len(plans)
    assert all(
        len({plan.game_seed, plan.learning_policy_seed, plan.opponent_seed}) == 3
        for plan in plans
    )


def test_5a_configuration_cannot_enable_future_training_features() -> None:
    with pytest.raises(ValueError, match="RandomPolicy"):
        FiveATrainingConfig(opponent="score")
    with pytest.raises(ValueError, match="unshaped terminal"):
        FiveATrainingConfig(reward_shaping_alpha=0.1)
    with pytest.raises(ValueError, match="unshaped terminal"):
        FiveATrainingConfig(reward="shaped")


def test_dev_evaluation_is_greedy_frozen_and_does_not_change_model(
    tmp_path: Path,
) -> None:
    torch.manual_seed(93)
    model = ActorCritic().train()
    before = tuple(parameter.detach().clone() for parameter in model.parameters())

    result, elapsed = evaluate_actor_critic_vs_random(
        model,
        training_seed=93,
        paired_seed_count=1,
        device="cpu",
        max_completed_turns=400,
        failure_dir=tmp_path,
    )

    assert result.total_pairs == 1 and result.total_games == 2
    assert result.exceptions == result.illegal_actions == result.invariant_failures == 0
    assert elapsed > 0.0
    assert model.training
    assert all(
        torch.equal(previous, current)
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_random_opponent_and_learning_seat_are_fixed_for_whole_episode(
    tmp_path: Path,
) -> None:
    plan = episode_plan(4, 0)
    opponent = RandomPolicy(plan.opponent_seed)
    environment = LearningEnvironment.new(
        plan.game_seed,
        first_player=plan.first_player,
        learning_player=plan.learning_player,
        opponent_policy=opponent,
        episode_id="fixed-episode",
        max_completed_turns=400,
        failure_dir=tmp_path,
    )
    original_opponent = environment.opponent_policy
    decision = environment.current_decision()
    environment.step_learning_action(0, decision_id=decision.decision_id)

    assert environment.learning_player == plan.learning_player
    assert environment.opponent_policy is original_opponent is opponent
    assert all(
        transition.learning_player == plan.learning_player
        for transition in environment.transitions
    )


def test_rollout_metrics_distinguish_engineering_truncation(
    tmp_path: Path,
) -> None:
    torch.manual_seed(10)
    model = ActorCritic()
    sampler = PolicySampler(model, seed=11)

    def factory(index: int) -> LearningEnvironment:
        return LearningEnvironment.new(
            100 + index,
            first_player=0,
            learning_player=0,
            opponent_policy=RandomPolicy(200 + index),
            episode_id=f"truncated-{index}",
            max_completed_turns=1,
            failure_dir=tmp_path / str(index),
        )

    collector = RolloutCollector(sampler, factory)
    rollout = _finish_active_episode(collector, collector.collect(1))
    completed = collector.take_completed_environments()
    metrics = summarize_rollout(rollout, completed)

    assert len(completed) == 1 and completed[0].truncated
    assert metrics["games_finished"] == 1
    assert metrics["games_completed"] == 0
    assert metrics["engineering_truncations"] == 1
    assert (
        metrics["train_win_rate"]
        == metrics["train_draw_rate"]
        == metrics["train_loss_rate"]
        == 0.0
    )


def test_short_cli_records_initial_eval_metrics_and_real_resume(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    common = [
        "--training-seed",
        "93",
        "--device",
        "cpu",
        "--rollout-size",
        "4",
        "--eval-interval",
        "4",
        "--checkpoint-interval",
        "4",
        "--eval-pairs",
        "1",
        "--final-eval-pairs",
        "1",
        "--output-dir",
        str(output),
    ]
    assert main(["--total-learning-steps", "4", *common]) == 0
    first_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    first_step = first_summary["global_learning_decisions"]
    first_updates = first_summary["ppo_updates"]
    initial_checkpoint = output / "checkpoints" / "step_000000000.pt"
    final_checkpoint = Path(first_summary["final_checkpoint"])

    initial_trainer, _, initial_metadata = PPOTrainer.load_checkpoint(initial_checkpoint)
    assert initial_trainer.global_learning_decision_step == 0
    assert initial_metadata["training_config"]["opponent"] == "random"
    assert initial_metadata["training_config"]["reward"] == "terminal"
    assert initial_metadata["training_config"]["reward_shaping_alpha"] == 0.0
    assert initial_metadata["collector_at_episode_boundary"]
    assert (output / "eval" / "step_000000000_dev.json").exists()
    assert any((output / "tensorboard").iterdir())

    assert main(
        [
            "--total-learning-steps",
            str(first_step + 1),
            "--resume-checkpoint",
            str(final_checkpoint),
            *common,
        ]
    ) == 0
    resumed_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert resumed_summary["global_learning_decisions"] > first_step
    assert resumed_summary["ppo_updates"] == first_updates + 1
    assert resumed_summary["progress"]["games_completed"] >= 2
    assert resumed_summary["progress"]["engineering_truncations"] == 0
    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {record["record_type"] for record in records} == {"evaluation", "update"}
    update = next(record for record in records if record["record_type"] == "update")
    assert update["games_finished"] == (
        update["games_completed"] + update["engineering_truncations"]
    )
    assert sum(update[f"{name}_count"] for name in (
        "TakeDistinct",
        "TakePair",
        "ReserveMarket",
        "ReserveDeck",
        "BuyMarket",
        "BuyReserved",
    )) > 0
