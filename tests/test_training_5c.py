from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.evaluation import EvaluationResult, OutcomeBreakdown, baseline_policy_spec
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
    FiveCProgress,
    _apply_schedules,
    _initial_formal_dev_required,
    _next_multiple_after,
    evaluate_five_c_panel,
    final_eval_game_seeds,
    load_five_c_config,
    prepare_five_c_run,
    run_five_c_training,
    scheduled_entropy_coef,
    scheduled_learning_rate,
)
import splendor.train_5c as train_5c_module


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


def _evaluation_result(name: str, seeds: tuple[int, ...]) -> EvaluationResult:
    games = len(seeds) * 2
    breakdown = OutcomeBreakdown(
        games=len(seeds),
        valid_games=len(seeds),
        wins=0,
        draws=len(seeds),
        losses=0,
        win_rate_score=0.5,
    )
    return EvaluationResult(
        policy_a="candidate",
        policy_b=name,
        game_seeds=seeds,
        total_pairs=len(seeds),
        total_games=games,
        valid_games=games,
        a_wins=0,
        draws=games,
        a_losses=0,
        win_rate_score=0.5,
        a_first=breakdown,
        a_second=breakdown,
        mean_completed_player_turns=20.0,
        median_completed_player_turns=20.0,
        mean_decisions=20.0,
        forced_skips=0,
        engineering_truncations=0,
        exceptions=0,
        illegal_actions=0,
        invariant_failures=0,
        matches=(),
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


def test_initial_formal_dev_marker_distinguishes_bootstrap_checkpoint() -> None:
    trainer = PPOTrainer(ActorCritic())
    trainer.global_learning_decision_step = 512_112
    progress = FiveCProgress(
        source_checkpoint="source.pt",
        source_lifetime_step=512_112,
        source_update_count=0,
    )
    assert _initial_formal_dev_required(trainer, progress)

    progress.evaluation_steps.append(0)
    progress.best_heuristic_composite = 0.5
    progress.best_checkpoint = "step-zero.pt"
    assert not _initial_formal_dev_required(trainer, progress)


def test_five_c_panel_reports_each_opponent_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = (-1, -2)
    names = (
        "random",
        "score",
        "development",
        "anchor_403183",
        "anchor_512112",
    )
    opponents = {
        name: baseline_policy_spec("random")
        for name in names
    }

    def fake_evaluate(
        candidate,
        opponent,
        game_seeds,
        *,
        progress_callback=None,
        **kwargs,
    ):
        del candidate, opponent, kwargs
        if progress_callback is not None:
            progress_callback(1, len(game_seeds))
            progress_callback(len(game_seeds), len(game_seeds))
        return _evaluation_result("fixed", tuple(game_seeds))

    monkeypatch.setattr(train_5c_module, "evaluate_policies", fake_evaluate)
    events: list[tuple[str, int, int, str]] = []
    pair_events: list[tuple[str, int, int, int, int]] = []
    results, _ = evaluate_five_c_panel(
        ActorCritic(),
        opponents=opponents,
        seeds=seeds,
        policy_seed_base=0,
        device="cpu",
        max_completed_turns=400,
        failure_dir=tmp_path,
        progress_callback=lambda name, index, total, event, result: events.append(
            (name, index, total, event)
        ),
        pair_progress_callback=lambda name, index, opponent_total, completed, total: (
            pair_events.append((name, index, opponent_total, completed, total))
        ),
    )

    assert tuple(results) == names
    assert events == [
        item
        for index, name in enumerate(names, start=1)
        for item in (
            (name, index, len(names), "started"),
            (name, index, len(names), "completed"),
        )
    ]
    assert pair_events == [
        (name, index, len(names), 1, len(seeds))
        for index, name in enumerate(names, start=1)
    ]


def test_resume_retries_interrupted_step_zero_formal_dev(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The formal 5C contract is CUDA-only.  This test exercises orchestration
    # and checkpoint semantics on CPU, so disable that constructor guard only
    # inside this test while retaining every other frozen config value.
    monkeypatch.setattr(FiveCConfig, "__post_init__", lambda self: None)
    original = _config(tmp_path / "checkpoints")
    monkeypatch.setattr(train_5c_module, "RUN_LOCAL_TARGET", 0)
    monkeypatch.setattr(
        train_5c_module,
        "scheduled_learning_rate",
        lambda config, step: INITIAL_LEARNING_RATE,
    )
    monkeypatch.setattr(
        train_5c_module,
        "scheduled_entropy_coef",
        lambda config, step: INITIAL_ENTROPY_COEF,
    )
    monkeypatch.setattr(
        train_5c_module,
        "_apply_schedules",
        lambda trainer, config, step: (
            trainer.config.learning_rate,
            trainer.config.entropy_coef,
        ),
    )
    config = replace(
        original,
        run_local_learning_decisions=0,
        device="cpu",
    )
    run_root = tmp_path / "run"
    prepare_five_c_run(config, run_root=run_root)
    names = (
        "random",
        "score",
        "development",
        "anchor_403183",
        "anchor_512112",
    )
    monkeypatch.setattr(
        train_5c_module,
        "build_five_c_eval_opponents",
        lambda config, device, include_historical: {
            name: baseline_policy_spec("random") for name in names
        },
    )

    def interrupted_panel(*args, progress_callback=None, **kwargs):
        del args, kwargs
        if progress_callback is not None:
            progress_callback("random", 1, len(names), "started", None)
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        train_5c_module, "evaluate_five_c_panel", interrupted_panel
    )
    with pytest.raises(KeyboardInterrupt):
        run_five_c_training(config, run_root=run_root)

    failed = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["run_local_step"] == 0
    assert failed["error_type"] == "KeyboardInterrupt"
    latest = json.loads(
        (run_root / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )

    calls = 0

    def successful_panel(*args, seeds, progress_callback=None, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        results = {}
        for index, name in enumerate(names, start=1):
            if progress_callback is not None:
                progress_callback(name, index, len(names), "started", None)
            result = _evaluation_result(name, tuple(seeds))
            results[name] = result
            if progress_callback is not None:
                progress_callback(name, index, len(names), "completed", result)
        return results, 0.01

    monkeypatch.setattr(
        train_5c_module, "evaluate_five_c_panel", successful_panel
    )
    summary = run_five_c_training(
        config,
        run_root=run_root,
        resume_checkpoint=Path(latest["checkpoint"]),
    )

    assert calls == 1
    assert summary["actual_run_local_decisions"] == 0
    assert summary["progress"]["evaluation_steps"] == [0]
    completed = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["phase"] == "complete"
    output = capsys.readouterr().out
    assert '"event": "training_process_started"' in output
    assert '"event": "initial_formal_dev_required"' in output
    assert '"event": "initial_formal_dev_complete"' in output
