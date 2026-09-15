"""Work-package 5B staged training and validation runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter

from .evaluation import EvaluationResult, PolicySpec, baseline_policy_spec, evaluate_policies
from .features import FEATURE_SCHEMA_VERSION
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .neural_policy import GreedyActorCriticPolicy
from .ppo import PPOTrainer, PolicySampler, RolloutCollector, compute_gae
from .random_policy import RandomPolicy
from .replay import RULESET_ID
from .rng import DeterministicRng
from .train_5a import (
    TrainingProgress,
    _METRIC_FIELDS,
    _finish_active_episode,
    _save_truncation,
    _tensorboard_rollout,
    _update_progress,
    _write_json,
    dev_eval_game_seeds,
    episode_plan,
    runtime_metadata,
    summarize_rollout,
)
from .training_env import LearningEnvironment


FIVE_B_RUN_FORMAT_VERSION = 1
FIXED_MIX = {"random": 0.20, "score": 0.40, "development": 0.40}
COMPOSITE_WEIGHTS = FIXED_MIX
_OPPONENT_KIND_STREAM_OFFSET = 5_000_000_000_000_000
_PANEL_POLICY_SEED_OFFSET = 6_000_000_000_000_000
_PANEL_NAMES = ("random", "score", "development")


@dataclass(frozen=True, slots=True)
class FiveBFixedMixConfig:
    total_global_learning_decisions: int = 400_000
    training_seed: int = 2
    device: str = "cuda"
    rollout_steps: int = 8192
    eval_interval: int = 50_000
    checkpoint_interval: int = 50_000
    eval_pairs: int = 200
    max_completed_turns: int = 400
    reward: str = "terminal"
    reward_shaping_alpha: float = 0.0
    opponent_mix: dict[str, float] = field(default_factory=lambda: dict(FIXED_MIX))

    def __post_init__(self) -> None:
        if self.total_global_learning_decisions <= 0 or self.training_seed < 0:
            raise ValueError("training target must be positive and seed non-negative")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if min(
            self.rollout_steps,
            self.eval_interval,
            self.checkpoint_interval,
            self.eval_pairs,
            self.max_completed_turns,
        ) <= 0:
            raise ValueError("training sizes, intervals, and watchdog must be positive")
        if self.reward != "terminal" or self.reward_shaping_alpha != 0.0:
            raise ValueError("5B stage 1 only permits unshaped terminal reward")
        if self.opponent_mix != FIXED_MIX:
            raise ValueError("5B stage 1 opponent mix is frozen at 20/40/40")


@dataclass(frozen=True, slots=True)
class FixedMixEpisodePlan:
    episode_index: int
    game_seed: int
    learning_policy_seed: int
    opponent_seed: int
    learning_player: int
    first_player: int
    opponent_kind: str


def fixed_mix_episode_plan(
    training_seed: int, episode_index: int
) -> FixedMixEpisodePlan:
    base = episode_plan(training_seed, episode_index)
    draw = DeterministicRng(
        _OPPONENT_KIND_STREAM_OFFSET + base.game_seed
    ).randbelow(100)
    opponent_kind = "random" if draw < 20 else "score" if draw < 60 else "development"
    return FixedMixEpisodePlan(
        episode_index=episode_index,
        game_seed=base.game_seed,
        learning_policy_seed=base.learning_policy_seed,
        opponent_seed=base.opponent_seed,
        learning_player=base.learning_player,
        first_player=base.first_player,
        opponent_kind=opponent_kind,
    )


def fixed_opponent(kind: str, seed: int):
    if kind == "random":
        return RandomPolicy(seed)
    if kind == "score":
        return ScorePolicy()
    if kind == "development":
        return DevelopmentPolicy()
    raise ValueError(f"unsupported fixed opponent: {kind}")


@dataclass(slots=True)
class FiveBProgress:
    training: TrainingProgress = field(default_factory=TrainingProgress)
    opponent_games: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _PANEL_NAMES}
    )
    evaluation_steps: list[int] = field(default_factory=list)
    best_composite_score: float | None = None
    best_checkpoint: str | None = None
    source_checkpoint: str | None = None
    source_global_step: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FiveBProgress:
        return cls(
            training=TrainingProgress.from_dict(values["training"]),
            opponent_games=dict(values["opponent_games"]),
            evaluation_steps=list(values["evaluation_steps"]),
            best_composite_score=values.get("best_composite_score"),
            best_checkpoint=values.get("best_checkpoint"),
            source_checkpoint=values.get("source_checkpoint"),
            source_global_step=int(values.get("source_global_step", 0)),
            elapsed_seconds=float(values.get("elapsed_seconds", 0.0)),
        )


_FIVE_B_EXTRA_FIELDS = (
    "opponent_random_count",
    "opponent_score_count",
    "opponent_development_count",
    "opponent_random_fraction",
    "opponent_score_fraction",
    "opponent_development_fraction",
    "active_envs",
    "batch_inference_size",
    "panel_pairs",
    "random_wins",
    "random_draws",
    "random_losses",
    "random_score_rate",
    "random_first_player_score",
    "random_second_player_score",
    "random_mean_turns",
    "score_wins",
    "score_draws",
    "score_losses",
    "score_policy_score_rate",
    "score_first_player_score",
    "score_second_player_score",
    "score_mean_turns",
    "development_wins",
    "development_draws",
    "development_losses",
    "development_policy_score_rate",
    "development_first_player_score",
    "development_second_player_score",
    "development_mean_turns",
    "composite_score",
)
_FIVE_B_METRIC_FIELDS = tuple(dict.fromkeys((*_METRIC_FIELDS, *_FIVE_B_EXTRA_FIELDS)))


class FiveBMetricWriter:
    def __init__(self, output_dir: Path) -> None:
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.csv_path = output_dir / "metrics.csv"

    def append(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_FIVE_B_METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({name: record.get(name, "") for name in _FIVE_B_METRIC_FIELDS})


def evaluate_fixed_panel(
    model: torch.nn.Module,
    *,
    training_seed: int,
    paired_seed_count: int,
    device: str,
    max_completed_turns: int,
    failure_dir: Path,
) -> tuple[dict[str, EvaluationResult], float]:
    parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())
    was_training = model.training
    model.eval()
    learning_policy = PolicySpec(
        "ppo-greedy",
        lambda seed: GreedyActorCriticPolicy(model, device=device),
    )
    seeds = dev_eval_game_seeds(training_seed, paired_seed_count)
    started = time.perf_counter()
    results = {
        name: evaluate_policies(
            learning_policy,
            baseline_policy_spec(name),
            seeds,
            policy_seed_base=-(_PANEL_POLICY_SEED_OFFSET + training_seed * 10_000_000),
            max_completed_turns=max_completed_turns,
            failure_dir=failure_dir / name,
        )
        for name in _PANEL_NAMES
    }
    elapsed = time.perf_counter() - started
    model.train(was_training)
    if not all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    ):
        raise AssertionError("fixed-panel evaluation modified model parameters")
    return results, elapsed


def _panel_record(
    results: dict[str, EvaluationResult],
    *,
    step: int,
    update_count: int,
    checkpoint_id: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if any(result.win_rate_score is None for result in results.values()):
        raise RuntimeError("fixed evaluation panel has no valid score")
    record: dict[str, Any] = {
        "record_type": "evaluation",
        "eval_kind": "fixed_panel",
        "step": step,
        "update_count": update_count,
        "checkpoint_id": checkpoint_id,
        "panel_pairs": next(iter(results.values())).total_pairs,
        "evaluation_seconds": elapsed_seconds,
        "engineering_truncations": sum(
            result.engineering_truncations for result in results.values()
        ),
        "illegal_actions": sum(result.illegal_actions for result in results.values()),
        "invariant_failures": sum(
            result.invariant_failures for result in results.values()
        ),
        "exceptions": sum(result.exceptions for result in results.values()),
    }
    for name, result in results.items():
        prefix = "score" if name == "score" else name
        score_key = f"{prefix}_policy_score_rate" if name != "random" else "random_score_rate"
        record.update(
            {
                f"{prefix}_wins": result.a_wins,
                f"{prefix}_draws": result.draws,
                f"{prefix}_losses": result.a_losses,
                score_key: result.win_rate_score,
                f"{prefix}_first_player_score": result.a_first.win_rate_score,
                f"{prefix}_second_player_score": result.a_second.win_rate_score,
                f"{prefix}_mean_turns": result.mean_completed_player_turns,
            }
        )
    record["composite_score"] = sum(
        COMPOSITE_WEIGHTS[name] * float(results[name].win_rate_score)
        for name in _PANEL_NAMES
    )
    return record


def _write_panel(
    path: Path,
    results: dict[str, EvaluationResult],
    record: dict[str, Any],
    config: FiveBFixedMixConfig,
    runtime: dict[str, Any],
) -> None:
    _write_json(
        path,
        {
            "global_learning_decisions": record["step"],
            "checkpoint_id": record["checkpoint_id"],
            "training_config": asdict(config),
            "runtime": runtime,
            "rules_version": RULESET_ID,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "panel_summary": record,
            "evaluations": {
                name: result.to_dict(include_matches=True)
                for name, result in results.items()
            },
        },
    )


def _tensorboard_panel(
    writer: SummaryWriter, record: dict[str, Any], step: int
) -> None:
    for tag, key in (
        ("random_score_rate", "random_score_rate"),
        ("score_policy_score_rate", "score_policy_score_rate"),
        ("development_policy_score_rate", "development_policy_score_rate"),
        ("composite_score", "composite_score"),
    ):
        writer.add_scalar(f"eval/{tag}", record[key], step)


def _checkpoint_metadata(
    config: FiveBFixedMixConfig,
    progress: FiveBProgress,
    collector: RolloutCollector,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if collector.active_environment is not None:
        raise ValueError("5B stage-1 checkpoints require an episode boundary")
    return {
        "run_format_version": FIVE_B_RUN_FORMAT_VERSION,
        "stage": "fixed_mix",
        "training_config": asdict(config),
        "progress": progress.to_dict(),
        "next_episode_index": collector.next_episode_index,
        "collector_at_episode_boundary": True,
        "runtime": runtime,
        "rules_version": RULESET_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def _save_checkpoint(
    trainer: PPOTrainer,
    sampler: PolicySampler,
    collector: RolloutCollector,
    config: FiveBFixedMixConfig,
    progress: FiveBProgress,
    runtime: dict[str, Any],
    path: Path,
) -> None:
    trainer.save_checkpoint(
        path,
        sampler=sampler,
        metadata=_checkpoint_metadata(config, progress, collector, runtime),
    )


def _immutable_resume_config(config: FiveBFixedMixConfig) -> dict[str, Any]:
    values = asdict(config)
    for name in ("total_global_learning_decisions", "eval_interval"):
        values.pop(name)
    return values


def run_fixed_mix_training(
    config: FiveBFixedMixConfig,
    *,
    output_dir: Path,
    source_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if (source_checkpoint is None) == (resume_checkpoint is None):
        raise ValueError("provide exactly one of source_checkpoint or resume_checkpoint")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir = output_dir.resolve()
    if resume_checkpoint is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {output_dir}")
    for name in ("checkpoints", "eval", "logs", "tensorboard"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    runtime = runtime_metadata(config.device)
    checkpoint_to_load = source_checkpoint if source_checkpoint is not None else resume_checkpoint
    trainer, sampler, metadata = PPOTrainer.load_checkpoint(
        checkpoint_to_load, device=config.device
    )
    if metadata.get("rules_version") != RULESET_ID:
        raise ValueError("source checkpoint rules version mismatch")
    if trainer.config.gamma != 1.0 or trainer.config.gae_lambda != 0.95:
        raise ValueError("source checkpoint does not use the frozen 5B GAE configuration")
    if resume_checkpoint is None:
        progress = FiveBProgress(
            source_checkpoint=str(source_checkpoint),
            source_global_step=trainer.global_learning_decision_step,
        )
        start_episode_index = 0
        _write_json(
            output_dir / "config.json",
            {
                "run_format_version": FIVE_B_RUN_FORMAT_VERSION,
                "stage": "fixed_mix",
                "training": asdict(config),
                "ppo": asdict(trainer.config),
                "model": asdict(trainer.model.config),
                "runtime": runtime,
                "source_checkpoint": str(source_checkpoint),
                "train_seed_domain": "non-negative",
                "dev_eval_seed_domain": "negative",
            },
        )
    else:
        if metadata.get("run_format_version") != FIVE_B_RUN_FORMAT_VERSION:
            raise ValueError("resume checkpoint is not a 5B run")
        saved = FiveBFixedMixConfig(**metadata["training_config"])
        if _immutable_resume_config(saved) != _immutable_resume_config(config):
            raise ValueError("resume configuration changes an immutable 5B setting")
        if metadata.get("stage") != "fixed_mix":
            raise ValueError("resume checkpoint is not from fixed-mix stage")
        progress = FiveBProgress.from_dict(metadata["progress"])
        start_episode_index = int(metadata["next_episode_index"])
        if not metadata.get("collector_at_episode_boundary"):
            raise ValueError("resume checkpoint is not at an episode boundary")
        _write_json(
            output_dir
            / f"resume_config_step_{trainer.global_learning_decision_step:09d}.json",
            {
                "run_format_version": FIVE_B_RUN_FORMAT_VERSION,
                "stage": "fixed_mix",
                "training": asdict(config),
                "ppo": asdict(trainer.config),
                "model": asdict(trainer.model.config),
                "runtime": runtime,
                "resume_checkpoint": str(resume_checkpoint),
            },
        )

    episode_opponents: dict[str, str] = {}

    def environment_factory(index: int) -> LearningEnvironment:
        plan = fixed_mix_episode_plan(config.training_seed, index)
        sampler.manual_seed(plan.learning_policy_seed)
        episode_id = f"5b-fixed-{config.training_seed}-{index:08d}"
        episode_opponents[episode_id] = plan.opponent_kind
        return LearningEnvironment.new(
            plan.game_seed,
            first_player=plan.first_player,
            learning_player=plan.learning_player,
            opponent_policy=fixed_opponent(plan.opponent_kind, plan.opponent_seed),
            episode_id=episode_id,
            max_completed_turns=config.max_completed_turns,
            failure_dir=output_dir / "logs" / f"failure-{index:08d}",
        )

    collector = RolloutCollector(
        sampler,
        environment_factory,
        trainer.config,
        start_episode_index=start_episode_index,
    )
    metrics = FiveBMetricWriter(output_dir)
    tensorboard = SummaryWriter(
        log_dir=str(output_dir / "tensorboard"),
        purge_step=(trainer.global_learning_decision_step + 1 if resume_checkpoint else None),
    )
    previous_elapsed = progress.elapsed_seconds
    started = time.perf_counter()

    def checkpoint_path(step: int) -> Path:
        return output_dir / "checkpoints" / f"step_{step:09d}.pt"

    def perform_panel() -> dict[str, Any]:
        step = trainer.global_learning_decision_step
        results, elapsed = evaluate_fixed_panel(
            trainer.model,
            training_seed=config.training_seed,
            paired_seed_count=config.eval_pairs,
            device=config.device,
            max_completed_turns=config.max_completed_turns,
            failure_dir=output_dir / "eval" / f"failures-step-{step:09d}",
        )
        record = _panel_record(
            results,
            step=step,
            update_count=trainer.update_count,
            checkpoint_id=checkpoint_path(step).name,
            elapsed_seconds=elapsed,
        )
        metrics.append(record)
        _tensorboard_panel(tensorboard, record, step)
        _write_panel(
            output_dir / "eval" / f"step_{step:09d}_fixed_panel.json",
            results,
            record,
            config,
            runtime,
        )
        if any(not result.passed for result in results.values()):
            raise RuntimeError("fixed evaluation panel contained an error or truncation")
        composite = float(record["composite_score"])
        if progress.best_composite_score is None or composite > progress.best_composite_score:
            progress.best_composite_score = composite
            progress.best_checkpoint = str(checkpoint_path(step))
        if step not in progress.evaluation_steps:
            progress.evaluation_steps.append(step)
        tensorboard.flush()
        return record

    try:
        if resume_checkpoint is None:
            initial_path = checkpoint_path(trainer.global_learning_decision_step)
            _save_checkpoint(
                trainer, sampler, collector, config, progress, runtime, initial_path
            )
            perform_panel()
            progress.elapsed_seconds = previous_elapsed + (time.perf_counter() - started)
            _save_checkpoint(
                trainer, sampler, collector, config, progress, runtime, initial_path
            )

        next_eval = (
            trainer.global_learning_decision_step // config.eval_interval + 1
        ) * config.eval_interval
        next_checkpoint = (
            trainer.global_learning_decision_step // config.checkpoint_interval + 1
        ) * config.checkpoint_interval

        while trainer.global_learning_decision_step < config.total_global_learning_decisions:
            current = trainer.global_learning_decision_step
            target = min(
                config.rollout_steps,
                config.total_global_learning_decisions - current,
            )
            boundary_due = current + target >= min(
                next_eval,
                next_checkpoint,
                config.total_global_learning_decisions,
            )
            sampler.reset_timing()
            rollout_started = time.perf_counter()
            rollout = collector.collect(target)
            if boundary_due:
                rollout = _finish_active_episode(collector, rollout)
            rollout_seconds = time.perf_counter() - rollout_started
            completed = collector.take_completed_environments()
            rollout_metrics = summarize_rollout(rollout, completed)
            opponent_counts = Counter(
                episode_opponents.pop(environment.episode_id) for environment in completed
            )
            opponent_total = sum(opponent_counts.values())
            for name in _PANEL_NAMES:
                progress.opponent_games[name] += opponent_counts[name]
                rollout_metrics[f"opponent_{name}_count"] = opponent_counts[name]
                rollout_metrics[f"opponent_{name}_fraction"] = (
                    opponent_counts[name] / opponent_total if opponent_total else 0.0
                )

            gae_started = time.perf_counter()
            prepared = compute_gae(
                rollout,
                gamma=trainer.config.gamma,
                gae_lambda=trainer.config.gae_lambda,
            )
            gae_seconds = time.perf_counter() - gae_started
            if config.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            update_started = time.perf_counter()
            update_stats = trainer.update(prepared)
            if config.device == "cuda":
                torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            _update_progress(progress.training, rollout_metrics)
            step = trainer.global_learning_decision_step
            environment_seconds = max(0.0, rollout_seconds - sampler.inference_seconds)
            record = {
                "record_type": "update",
                "step": step,
                "update_count": trainer.update_count,
                "wall_seconds": previous_elapsed + (time.perf_counter() - started),
                "policy_loss": update_stats.policy_loss,
                "value_loss": update_stats.value_loss,
                "total_loss": update_stats.total_loss,
                "entropy": update_stats.entropy,
                "explained_variance": update_stats.explained_variance,
                "approx_kl": update_stats.approx_kl,
                "clip_fraction": update_stats.clip_fraction,
                "grad_norm": update_stats.grad_norm,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "epochs_completed": update_stats.epochs_completed,
                "minibatches_processed": update_stats.minibatches_processed,
                "gradient_clipped_minibatches": update_stats.gradient_clipped_minibatches,
                "kl_early_stopped": update_stats.early_stopped,
                "learning_decisions_per_second": len(rollout) / rollout_seconds,
                "games_per_second": len(completed) / rollout_seconds,
                "rollout_seconds": rollout_seconds,
                "environment_encoding_seconds": environment_seconds,
                "model_inference_seconds": sampler.inference_seconds,
                "gae_seconds": gae_seconds,
                "ppo_update_seconds": update_seconds,
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                    if config.device == "cuda"
                    else 0.0
                ),
                "cpu_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
                "active_envs": 1,
                "batch_inference_size": 1,
                **{
                    key: value
                    for key, value in rollout_metrics.items()
                    if not key.startswith("_")
                },
            }
            for value in (
                update_stats.policy_loss,
                update_stats.value_loss,
                update_stats.total_loss,
                update_stats.entropy,
                update_stats.approx_kl,
                update_stats.grad_norm,
            ):
                if not math.isfinite(value):
                    raise FloatingPointError("non-finite 5B training metric")
            metrics.append(record)
            for key in (
                "policy_loss",
                "value_loss",
                "total_loss",
                "entropy",
                "explained_variance",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
                "learning_rate",
            ):
                tensorboard.add_scalar(f"ppo/{key}", record[key], step)
            _tensorboard_rollout(tensorboard, record, step)
            for name in _PANEL_NAMES:
                tensorboard.add_scalar(
                    f"opponent/{name}_fraction",
                    record[f"opponent_{name}_fraction"],
                    step,
                )
            for key in (
                "learning_decisions_per_second",
                "games_per_second",
                "rollout_seconds",
                "ppo_update_seconds",
                "cuda_peak_allocated_mib",
                "active_envs",
                "batch_inference_size",
            ):
                tensorboard.add_scalar(f"system/{key}", record[key], step)
            tensorboard.flush()

            truncated = [environment for environment in completed if environment.truncated]
            for environment in truncated:
                _save_truncation(output_dir, environment)
            if truncated:
                raise RuntimeError(
                    "engineering truncation encountered; replay/state saved for investigation"
                )

            evaluation_due = step >= next_eval
            checkpoint_due = step >= next_checkpoint
            final_due = step >= config.total_global_learning_decisions
            if evaluation_due:
                perform_panel()
                while next_eval <= step:
                    next_eval += config.eval_interval
            if evaluation_due or checkpoint_due or final_due:
                progress.elapsed_seconds = previous_elapsed + (
                    time.perf_counter() - started
                )
                _save_checkpoint(
                    trainer,
                    sampler,
                    collector,
                    config,
                    progress,
                    runtime,
                    checkpoint_path(step),
                )
                while next_checkpoint <= step:
                    next_checkpoint += config.checkpoint_interval

        final_step = trainer.global_learning_decision_step
        if final_step not in progress.evaluation_steps:
            perform_panel()
        progress.elapsed_seconds = previous_elapsed + (time.perf_counter() - started)
        final_path = checkpoint_path(final_step)
        _save_checkpoint(
            trainer, sampler, collector, config, progress, runtime, final_path
        )
        summary = {
            "stage": "fixed_mix",
            "training_config": asdict(config),
            "ppo_config": asdict(trainer.config),
            "runtime": runtime,
            "global_learning_decisions": final_step,
            "stage_learning_decisions": final_step - progress.source_global_step,
            "ppo_updates": trainer.update_count,
            "progress": progress.to_dict(),
            "final_checkpoint": str(final_path),
        }
        _write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        tensorboard.flush()
        tensorboard.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5B stage 1 fixed mixed-opponent training")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-checkpoint", type=Path)
    source.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--total-global-learning-steps", type=int, default=400_000)
    parser.add_argument("--training-seed", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rollout-size", type=int, default=8192)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--eval-pairs", type=int, default=200)
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = FiveBFixedMixConfig(
        total_global_learning_decisions=args.total_global_learning_steps,
        training_seed=args.training_seed,
        device=args.device,
        rollout_steps=args.rollout_size,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        eval_pairs=args.eval_pairs,
        max_completed_turns=args.max_completed_turns,
    )
    summary = run_fixed_mix_training(
        config,
        output_dir=args.output_dir,
        source_checkpoint=args.source_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
