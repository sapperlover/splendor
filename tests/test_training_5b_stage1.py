from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.heuristic_policy import DevelopmentPolicy, ScorePolicy
from splendor.random_policy import RandomPolicy
from splendor.train_5b import (
    FIXED_MIX,
    FiveBFixedMixConfig,
    evaluate_fixed_panel,
    fixed_mix_episode_plan,
    fixed_opponent,
)


def test_fixed_mix_episode_draw_is_deterministic_and_close_to_declared_weights() -> None:
    first = [fixed_mix_episode_plan(2, index) for index in range(10_000)]
    second = [fixed_mix_episode_plan(2, index) for index in range(10_000)]
    counts = Counter(plan.opponent_kind for plan in first)

    assert first == second
    assert set(counts) == set(FIXED_MIX)
    assert 0.18 < counts["random"] / len(first) < 0.22
    assert 0.38 < counts["score"] / len(first) < 0.42
    assert 0.38 < counts["development"] / len(first) < 0.42
    assert all(plan.game_seed >= 0 for plan in first)


def test_fixed_opponent_factory_returns_one_policy_for_whole_episode() -> None:
    assert isinstance(fixed_opponent("random", 10), RandomPolicy)
    assert isinstance(fixed_opponent("score", 10), ScorePolicy)
    assert isinstance(fixed_opponent("development", 10), DevelopmentPolicy)
    with pytest.raises(ValueError, match="unsupported"):
        fixed_opponent("latest", 10)


def test_stage1_configuration_cannot_enable_shaping_or_change_mix() -> None:
    with pytest.raises(ValueError, match="unshaped terminal"):
        FiveBFixedMixConfig(reward_shaping_alpha=0.1)
    with pytest.raises(ValueError, match="20/40/40"):
        FiveBFixedMixConfig(opponent_mix={"random": 1.0})


def test_fixed_panel_is_paired_frozen_and_complete(tmp_path: Path) -> None:
    torch.manual_seed(15)
    model = ActorCritic().train()
    before = tuple(parameter.detach().clone() for parameter in model.parameters())

    results, elapsed = evaluate_fixed_panel(
        model,
        training_seed=15,
        paired_seed_count=1,
        device="cpu",
        max_completed_turns=400,
        failure_dir=tmp_path,
    )

    assert set(results) == {"random", "score", "development"}
    assert all(result.total_pairs == 1 and result.total_games == 2 for result in results.values())
    assert all(result.passed for result in results.values())
    assert elapsed > 0.0
    assert model.training
    assert all(
        torch.equal(previous, current)
        for previous, current in zip(before, model.parameters(), strict=True)
    )
