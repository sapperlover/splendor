from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.ppo import PPOTrainer, PolicySampler
from splendor.replay import RULESET_ID
from splendor.train_5a import dev_eval_game_seeds, episode_plan
from splendor.train_5c import (
    FINAL_ENTROPY_COEF,
    FINAL_LEARNING_RATE,
    INITIAL_ENTROPY_COEF,
    INITIAL_LEARNING_RATE,
    PHASE_A_MIX,
    PHASE_B_MIX,
    PHASE_B_START,
    RUN_LOCAL_TARGET,
    FiveCConfig,
    FiveCOpponentSelector,
    _apply_schedules,
    _next_multiple_after,
    final_eval_game_seeds,
    load_five_c_config,
    prepare_five_c_run,
    scheduled_entropy_coef,
    scheduled_learning_rate,
)


def _checkpoint(path: Path, *, step: int, optimizer_state: bool = False) -> Path:
    torch.manual_seed(step)
    model = ActorCritic()
    trainer = PPOTrainer(model)
    if optimizer_state:
        trainer.optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        trainer.optimizer.step()
    trainer.global_learning_decision_step = step
    sampler = PolicySampler(model, seed=step)
    checkpoint = path / f"step_{step:09d}.pt"
    trainer.save_checkpoint(
        checkpoint,
        sampler=sampler,
        metadata={"rules_version": RULESET_ID},
    )
    return checkpoint


def _config(tmp_path: Path) -> FiveCConfig:
    steps = (100_005, 157_351, 206_542, 255_701, 304_858, 354_015, 403_183)
    histories = tuple(_checkpoint(tmp_path, step=step) for step in steps)
    source = _checkpoint(tmp_path, step=512_112, optimizer_state=True)
    return FiveCConfig(
        run_id="test-5c",
        source_checkpoint=str(source),
        historical_checkpoints=tuple(str(path) for path in (*histories, source)),
        anchor_403183_checkpoint=str(histories[-1]),
        anchor_512112_checkpoint=str(source),
    )


def test_five_c_linear_schedules_are_run_local_and_clamped(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert scheduled_learning_rate(config, 0) == INITIAL_LEARNING_RATE
    assert scheduled_entropy_coef(config, 0) == INITIAL_ENTROPY_COEF
    assert scheduled_learning_rate(config, RUN_LOCAL_TARGET // 2) == pytest.approx(6.5e-5)
    assert scheduled_entropy_coef(config, RUN_LOCAL_TARGET // 2) == pytest.approx(0.006)
    assert scheduled_learning_rate(config, RUN_LOCAL_TARGET) == FINAL_LEARNING_RATE
    assert scheduled_entropy_coef(config, RUN_LOCAL_TARGET) == FINAL_ENTROPY_COEF
    assert scheduled_learning_rate(config, RUN_LOCAL_TARGET + 10) == FINAL_LEARNING_RATE

    trainer, _, _ = PPOTrainer.load_checkpoint(config.source_checkpoint)
    learning_rate, entropy = _apply_schedules(trainer, config, 1_250_000)
    assert trainer.config.learning_rate == learning_rate
    assert trainer.config.entropy_coef == entropy
    assert all(group["lr"] == learning_rate for group in trainer.optimizer.param_groups)


def test_five_c_mixture_switches_only_by_run_local_phase() -> None:
    phase_a = FiveCOpponentSelector(70)
    phase_b = FiveCOpponentSelector(70)
    counts_a = Counter(phase_a.choose_kind(PHASE_B_START - 1) for _ in range(20_000))
    counts_b = Counter(phase_b.choose_kind(PHASE_B_START) for _ in range(20_000))

    for kind, expected in PHASE_A_MIX.items():
        assert counts_a[kind] / 20_000 == pytest.approx(expected, abs=0.015)
    for kind, expected in PHASE_B_MIX.items():
        assert counts_b[kind] / 20_000 == pytest.approx(expected, abs=0.015)

    selector = FiveCOpponentSelector(71)
    snapshot = selector.snapshot
    expected = tuple(selector.choose_kind(PHASE_B_START) for _ in range(100))
    restored = FiveCOpponentSelector.restore(snapshot)
    assert tuple(restored.choose_kind(PHASE_B_START) for _ in range(100)) == expected


def test_five_c_train_dev_final_domains_and_event_cadence_are_disjoint() -> None:
    train = set(episode_plan(1, index).game_seed for index in range(100))
    dev = set(dev_eval_game_seeds(2, 100))
    final = set(final_eval_game_seeds(1, 100))
    assert not train & dev
    assert not train & final
    assert not dev & final
    assert _next_multiple_after(0, 100_000) == 100_000
    assert _next_multiple_after(100_001, 100_000) == 200_000
    assert _next_multiple_after(249_999, 250_000) == 250_000


def test_five_c_config_is_frozen_and_prepare_is_non_overwriting(tmp_path: Path) -> None:
    config = _config(tmp_path / "checkpoints")
    run_root = tmp_path / "run"
    metadata = prepare_five_c_run(config, run_root=run_root)

    assert metadata["status"] == "prepared"
    assert load_five_c_config(run_root / "config.json") == config
    assert (run_root / "historical_pool" / "initial.json").is_file()
    assert (run_root / "logs" / "train.log").is_file()
    assert not (run_root / "checkpoints" / "latest.json").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_five_c_run(config, run_root=run_root)
    with pytest.raises(ValueError, match="5,000,000"):
        replace(config, run_local_learning_decisions=1_000)
