from __future__ import annotations

from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.evaluation import PolicySpec, evaluate_policies
from splendor.neural_policy import GreedyActorCriticPolicy
from splendor.opponent_pool import OpponentPool
from splendor.ppo import PPOTrainer, PolicySampler
from splendor.random_policy import RandomPolicy
from splendor.replay import RULESET_ID
from splendor.training_env import run_learning_episode


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


def test_latest_is_a_parameter_independent_frozen_snapshot(tmp_path: Path) -> None:
    model = ActorCritic()
    pool = OpponentPool()
    entry = pool.set_latest(model, training_step=10)
    choice = pool.choose("latest", seed=1)
    assert choice.entry == entry
    assert isinstance(choice.policy, GreedyActorCriticPolicy)
    snapshot = tuple(parameter.detach().clone() for parameter in choice.policy.model.parameters())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    assert all(not parameter.requires_grad for parameter in choice.policy.model.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(snapshot, choice.policy.model.parameters(), strict=True)
    )

    manifest = tmp_path / "latest-pool.json"
    pool.save_manifest(manifest)
    restored = OpponentPool.load_manifest(manifest, latest_model=model)
    restored_choice = restored.choose("latest")
    assert restored_choice.entry == entry
    assert all(
        not parameter.requires_grad
        for parameter in restored_choice.policy.model.parameters()
    )


def test_historical_pool_manifest_round_trip_and_deterministic_sampling(
    tmp_path: Path,
) -> None:
    _, checkpoint_a = _checkpoint(tmp_path, step=100)
    _, checkpoint_b = _checkpoint(tmp_path, step=200)
    pool = OpponentPool(max_historical=2)
    first = pool.add_historical(
        checkpoint_a, reason="time-spaced snapshot", added_at_step=300
    )
    second = pool.add_historical(
        checkpoint_b, reason="development improvement", added_at_step=300
    )
    manifest = tmp_path / "pool.json"
    expected_sequence = tuple(
        pool.choose("historical").entry.opponent_id for _ in range(8)
    )
    pool.save_manifest(manifest)
    restored = OpponentPool.load_manifest(manifest)

    assert restored.historical_entries == (first, second)
    assert tuple(
        restored.choose("historical").entry.opponent_id for _ in range(8)
    ) == tuple(
        pool.choose("historical").entry.opponent_id for _ in range(8)
    )
    assert len(expected_sequence) == 8
    assert restored.choose("historical", seed=44).entry == restored.choose(
        "historical", seed=44
    ).entry
    assert restored.choose("historical", seed=45).entry in (first, second)


def test_historical_removal_is_explicit_and_manifest_persistent(tmp_path: Path) -> None:
    _, checkpoint_a = _checkpoint(tmp_path, step=100)
    _, checkpoint_b = _checkpoint(tmp_path, step=200)
    pool = OpponentPool(max_historical=2)
    first = pool.add_historical(checkpoint_a, reason="protected", added_at_step=0)
    second = pool.add_historical(checkpoint_b, reason="dynamic", added_at_step=200)

    assert pool.remove_historical(second.opponent_id) == second
    assert pool.historical_entries == (first,)
    with pytest.raises(ValueError, match="unknown historical"):
        pool.remove_historical(second.opponent_id)

    manifest = tmp_path / "after-removal.json"
    pool.save_manifest(manifest)
    assert OpponentPool.load_manifest(manifest).historical_entries == (first,)


def test_historical_neural_opponent_completes_game_without_entering_trajectory(
    tmp_path: Path,
) -> None:
    _, checkpoint = _checkpoint(tmp_path, step=123)
    pool = OpponentPool()
    pool.add_historical(checkpoint, reason="integration test", added_at_step=123)
    choice = pool.choose("historical", seed=50)
    before = tuple(
        parameter.detach().clone() for parameter in choice.policy.model.parameters()
    )

    run = run_learning_episode(
        RandomPolicy(60),
        choice.policy,
        game_seed=70,
        first_player=0,
        learning_player=0,
        learning_policy_seed=60,
        opponent_seed=50,
        max_completed_turns=400,
        failure_dir=tmp_path / "game",
    )

    assert run.terminated and not run.truncated
    assert all(transition.learning_player == 0 for transition in run.transitions)
    assert run.opponent_decision_count > 0
    assert all(
        torch.equal(previous, current)
        for previous, current in zip(before, choice.policy.model.parameters(), strict=True)
    )


def test_two_historical_checkpoints_play_swapped_complete_games(
    tmp_path: Path,
) -> None:
    _, checkpoint_a = _checkpoint(tmp_path, step=100_000)
    _, checkpoint_b = _checkpoint(tmp_path, step=157_351)
    pool = OpponentPool()
    entry_a = pool.add_historical(
        checkpoint_a, reason="early", added_at_step=200_000
    )
    entry_b = pool.add_historical(
        checkpoint_b, reason="best", added_at_step=200_000
    )
    policy_a = PolicySpec(
        entry_a.opponent_id,
        lambda seed: pool.policy_for(entry_a.opponent_id, seed=seed).policy,
    )
    policy_b = PolicySpec(
        entry_b.opponent_id,
        lambda seed: pool.policy_for(entry_b.opponent_id, seed=seed).policy,
    )

    result = evaluate_policies(
        policy_a,
        policy_b,
        (-91, -92),
        max_completed_turns=400,
        failure_dir=tmp_path / "evaluation",
    )

    assert result.total_games == result.valid_games == 4
    assert result.passed
    assert result.illegal_actions == result.invariant_failures == 0


def test_incompatible_neural_checkpoint_fails_explicitly(tmp_path: Path) -> None:
    _, checkpoint = _checkpoint(tmp_path, step=100)
    payload = torch.load(checkpoint, weights_only=True)
    payload["feature_schema_version"] = "incompatible-schema"
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)

    with pytest.raises(ValueError, match="feature schema mismatch"):
        OpponentPool().add_historical(
            incompatible, reason="must fail", added_at_step=100
        )
