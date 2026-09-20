from __future__ import annotations

import copy
from pathlib import Path

import torch

from splendor.actor_critic import ActorCritic
from splendor.multiprocess_collector import (
    MultiprocessEpisodeLaunch,
    MultiprocessEpisodeSpec,
    MultiprocessRolloutCollector,
    restored_worker_rng_states,
)
from splendor.ppo import BatchedRolloutCollector, PolicySampler
from splendor.random_policy import RandomPolicy
from splendor.train_5a import episode_plan
from splendor.training_env import LearningEnvironment


def _launch(
    root: Path,
    training_seed: int,
    index: int,
    *,
    neural_model: ActorCritic | None = None,
) -> MultiprocessEpisodeLaunch:
    plan = episode_plan(training_seed, index)
    neural = neural_model is not None
    return MultiprocessEpisodeLaunch(
        MultiprocessEpisodeSpec(
            episode_index=index,
            episode_id=f"mp-{training_seed}-{index}",
            game_seed=plan.game_seed,
            first_player=plan.first_player,
            learning_player=plan.learning_player,
            learning_policy_seed=plan.learning_policy_seed,
            opponent_seed=plan.opponent_seed,
            opponent_kind="historical" if neural else "random",
            opponent_id="test-neural" if neural else "random",
            max_completed_turns=400,
            failure_dir=str(root / f"episode-{index}"),
        ),
        neural_opponent_model=neural_model,
    )


def _completed_signature(completed):
    return {
        episode.episode_id: (
            episode.learning_player,
            episode.terminated,
            episode.truncated,
            tuple(
                transition.chosen_action for transition in episode.transitions
            ),
            episode.transitions[-1].state_digest_after,
        )
        for episode in completed
    }


def test_spawn_collector_matches_single_process_fixed_opponent_rules(
    tmp_path: Path,
) -> None:
    torch.manual_seed(501)
    base_model = ActorCritic()
    serial_sampler = PolicySampler(copy.deepcopy(base_model), seed=0)
    process_sampler = PolicySampler(copy.deepcopy(base_model), seed=0)

    def serial_factory(index: int):
        plan = episode_plan(61, index)
        return (
            LearningEnvironment.new(
                plan.game_seed,
                first_player=plan.first_player,
                learning_player=plan.learning_player,
                opponent_policy=RandomPolicy(plan.opponent_seed),
                episode_id=f"mp-61-{index}",
                failure_dir=tmp_path / "serial" / f"episode-{index}",
            ),
            plan.learning_policy_seed,
        )

    serial = BatchedRolloutCollector(
        serial_sampler, serial_factory, active_envs=2
    )
    serial.collect(1)
    serial.drain_active_environments()
    serial_signature = _completed_signature(serial.take_completed_environments())

    process = MultiprocessRolloutCollector(
        process_sampler,
        lambda index: _launch(tmp_path / "process", 61, index),
        num_workers=2,
        envs_per_worker=1,
        master_seed=61,
    )
    try:
        assert process.start_method == "spawn"
        process.collect(1)
        process.drain_active_environments()
        process_signature = _completed_signature(
            process.take_completed_environments()
        )
        snapshot = process.snapshot()
    finally:
        process.close()

    assert process_signature == serial_signature
    assert {episode_id for episode_id in process_signature} == {"mp-61-0", "mp-61-1"}
    assert {values[0] for values in process_signature.values()} == {0, 1}
    assert snapshot.active_episode_count == 0
    assert len({item.rng.state for item in snapshot.worker_statistics}) == 2
    assert all(item.learner_steps > 0 for item in snapshot.worker_statistics)


def test_spawn_collector_centralizes_neural_opponent_and_restores_worker_rng(
    tmp_path: Path,
) -> None:
    torch.manual_seed(502)
    learner = ActorCritic()
    opponent = copy.deepcopy(learner).eval().requires_grad_(False)
    opponent_before = tuple(parameter.detach().clone() for parameter in opponent.parameters())
    sampler = PolicySampler(learner, seed=0)
    collector = MultiprocessRolloutCollector(
        sampler,
        lambda index: _launch(
            tmp_path / "neural", 62, index, neural_model=opponent
        ),
        num_workers=2,
        envs_per_worker=1,
        master_seed=62,
    )
    try:
        rollout = collector.collect(2)
        collector.drain_active_environments()
        completed = collector.take_completed_environments()
        snapshot = collector.snapshot()
        collector.collect(1)
        collector.drain_active_environments()
        uninterrupted_after_snapshot = _completed_signature(
            collector.take_completed_environments()
        )
    finally:
        collector.close()

    assert len(rollout) == 2
    assert len(completed) == 2
    assert all(episode.terminated and not episode.truncated for episode in completed)
    assert collector.opponent_inference_items > 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(opponent_before, opponent.parameters(), strict=True)
    )

    serialized = {
        "format_version": snapshot.format_version,
        "num_workers": snapshot.num_workers,
        "envs_per_worker": snapshot.envs_per_worker,
        "next_episode_index": snapshot.next_episode_index,
        "active_episode_count": snapshot.active_episode_count,
        "worker_statistics": [
            {
                "rng": {
                    "algorithm": item.rng.algorithm,
                    "state": item.rng.state,
                    "draws": item.rng.draws,
                }
            }
            for item in snapshot.worker_statistics
        ],
    }
    restored = restored_worker_rng_states(
        serialized, num_workers=2, envs_per_worker=1
    )
    assert restored == tuple(item.rng for item in snapshot.worker_statistics)

    resumed = MultiprocessRolloutCollector(
        PolicySampler(copy.deepcopy(learner), seed=0),
        lambda index: _launch(
            tmp_path / "resumed", 62, index, neural_model=opponent
        ),
        num_workers=2,
        envs_per_worker=1,
        master_seed=62,
        start_episode_index=snapshot.next_episode_index,
        restored_worker_rng_states=restored,
    )
    try:
        resumed.collect(1)
        resumed.drain_active_environments()
        restored_after_snapshot = _completed_signature(
            resumed.take_completed_environments()
        )
    finally:
        resumed.close()
    assert restored_after_snapshot == uninterrupted_after_snapshot
