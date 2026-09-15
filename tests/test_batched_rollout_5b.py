from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.features import collate_encoded_decisions
from splendor.ppo import (
    BatchedRolloutCollector,
    PolicySampler,
    RolloutCollector,
    compute_gae,
    group_rollout_by_episode,
)
from splendor.random_policy import RandomPolicy
from splendor.train_5a import episode_plan
from splendor.training_env import LearningEnvironment


def _environment(plan, output: Path, episode_id: str) -> LearningEnvironment:
    return LearningEnvironment.new(
        plan.game_seed,
        first_player=plan.first_player,
        learning_player=plan.learning_player,
        opponent_policy=RandomPolicy(plan.opponent_seed),
        episode_id=episode_id,
        max_completed_turns=400,
        failure_dir=output / episode_id,
    )


def test_batched_single_environment_matches_serial_sampler(tmp_path: Path) -> None:
    torch.manual_seed(101)
    serial_model = ActorCritic()
    batched_model = copy.deepcopy(serial_model)
    serial_sampler = PolicySampler(serial_model, seed=0)
    batched_sampler = PolicySampler(batched_model, seed=0)

    def serial_factory(index: int) -> LearningEnvironment:
        plan = episode_plan(31, index)
        serial_sampler.manual_seed(plan.learning_policy_seed)
        return _environment(plan, tmp_path / "serial", f"episode-{index}")

    def batched_factory(index: int) -> tuple[LearningEnvironment, int]:
        plan = episode_plan(31, index)
        return (
            _environment(plan, tmp_path / "batched", f"episode-{index}"),
            plan.learning_policy_seed,
        )

    serial = RolloutCollector(serial_sampler, serial_factory)
    batched = BatchedRolloutCollector(
        batched_sampler, batched_factory, active_envs=1
    )
    serial_rollout = serial.collect(100)
    batched_rollout = batched.collect(100)

    assert len(serial_rollout) == len(batched_rollout) == 100
    for expected, actual in zip(
        serial_rollout.samples, batched_rollout.samples, strict=True
    ):
        assert actual.episode_id == expected.episode_id
        assert actual.learning_player == expected.learning_player
        assert actual.chosen_candidate_index == expected.chosen_candidate_index
        assert actual.encoded.candidate_actions == expected.encoded.candidate_actions
        assert np.array_equal(
            actual.encoded.candidate_mask, expected.encoded.candidate_mask
        )
        assert np.allclose(
            actual.encoded.candidate_features,
            expected.encoded.candidate_features,
            atol=0.0,
        )
        assert actual.reward == expected.reward
        assert actual.terminated == expected.terminated
        assert actual.truncated == expected.truncated
        assert actual.old_logprob == pytest.approx(expected.old_logprob, abs=1e-7)
        assert actual.old_value == pytest.approx(expected.old_value, abs=1e-7)
        assert actual.bootstrap_value == pytest.approx(expected.bootstrap_value, abs=1e-7)


def test_multi_environment_rollout_is_grouped_for_unchanged_gae(
    tmp_path: Path,
) -> None:
    torch.manual_seed(102)
    model = ActorCritic()
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    sampler = PolicySampler(model, seed=0)

    def factory(index: int) -> tuple[LearningEnvironment, int]:
        plan = episode_plan(32, index)
        return (
            _environment(plan, tmp_path, f"episode-{index}"),
            plan.learning_policy_seed,
        )

    collector = BatchedRolloutCollector(sampler, factory, active_envs=4)
    first = collector.collect(100)
    active_rng_states = collector.active_policy_rng_states
    drained = collector.drain_active_environments()
    rollout = group_rollout_by_episode((*first.samples, *drained.samples))
    completed = collector.take_completed_environments()

    seen: set[str] = set()
    previous: str | None = None
    for sample in rollout.samples:
        if sample.episode_id != previous:
            assert sample.episode_id not in seen
            seen.add(sample.episode_id)
            previous = sample.episode_id
    assert collector.active_environment_count == 0
    assert len({episode_id for episode_id, _ in active_rng_states}) == len(
        active_rng_states
    )
    assert all(state.device.type == "cpu" for _, state in active_rng_states)
    assert len(completed) >= 4
    assert all(environment.done and not environment.truncated for environment in completed)
    prepared = compute_gae(first, gamma=1.0, gae_lambda=0.95)
    assert len(prepared) == 100
    assert sampler.mean_inference_batch_size > 1.0
    assert all(
        torch.equal(previous_parameter, current)
        for previous_parameter, current in zip(before, model.parameters(), strict=True)
    )


def test_four_batched_games_are_repeatable_and_logits_match_serial_forward(
    tmp_path: Path,
) -> None:
    torch.manual_seed(103)
    model = ActorCritic()

    initial_decisions = tuple(
        _environment(
            episode_plan(34, index),
            tmp_path / "logits",
            f"logits-{index}",
        ).current_decision().encoded
        for index in range(4)
    )
    handles_before = tuple(
        decision.candidate_actions for decision in initial_decisions
    )
    model.eval()
    with torch.inference_mode():
        batched_logits, batched_values = model(collate_encoded_decisions(initial_decisions))
        serial_outputs = tuple(
            model(collate_encoded_decisions((decision,)))
            for decision in initial_decisions
        )
    for index, (serial_logits, serial_value) in enumerate(serial_outputs):
        valid = initial_decisions[index].candidate_mask.shape[0]
        assert torch.allclose(
            batched_logits[index, :valid], serial_logits[0, :valid], atol=1e-6
        )
        assert torch.allclose(batched_values[index], serial_value[0], atol=1e-6)
        assert initial_decisions[index].candidate_actions == handles_before[index]

    def run_batched(label: str):
        sampler = PolicySampler(copy.deepcopy(model), seed=0)

        def factory(index: int) -> tuple[LearningEnvironment, int]:
            plan = episode_plan(34, index)
            return (
                _environment(plan, tmp_path / label, f"episode-{index}"),
                plan.learning_policy_seed,
            )

        collector = BatchedRolloutCollector(sampler, factory, active_envs=4)
        collector.collect(1)
        collector.drain_active_environments()
        return {
            environment.episode_id: (
                tuple(
                    transition.chosen_action
                    for transition in environment.transitions
                ),
                environment.transitions[-1].state_digest_after,
            )
            for environment in collector.take_completed_environments()
        }

    assert run_batched("first") == run_batched("second")


def test_batched_collector_rejects_invalid_sizes(tmp_path: Path) -> None:
    sampler = PolicySampler(ActorCritic(), seed=0)

    def factory(index: int) -> tuple[LearningEnvironment, int]:
        plan = episode_plan(33, index)
        return _environment(plan, tmp_path, f"episode-{index}"), plan.learning_policy_seed

    with pytest.raises(ValueError, match="active_envs"):
        BatchedRolloutCollector(sampler, factory, active_envs=0)
    collector = BatchedRolloutCollector(sampler, factory, active_envs=2)
    with pytest.raises(ValueError, match="positive"):
        collector.collect(0)
