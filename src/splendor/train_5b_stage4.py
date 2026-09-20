"""Work-package 5B Stage 4 short historical-pool self-play validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter

from .evaluation import EvaluationResult, PolicySpec, baseline_policy_spec, evaluate_policies
from .features import FEATURE_SCHEMA_VERSION
from .neural_policy import (
    GreedyActorCriticPolicy,
    NeuralInferenceTiming,
    SamplingActorCriticPolicy,
)
from .opponent_pool import OpponentChoice, OpponentPool, load_frozen_actor_critic_checkpoint
from .ppo import (
    BatchedRolloutCollector,
    PPOTrainer,
    PolicySampler,
    RolloutBatch,
    compute_gae,
    group_rollout_by_episode,
)
from .replay import RULESET_ID
from .rng import DeterministicRng, RngState
from .train_5a import (
    TrainingProgress,
    _METRIC_FIELDS,
    _save_truncation,
    _tensorboard_rollout,
    _update_progress,
    _write_json,
    dev_eval_game_seeds,
    episode_plan,
    runtime_metadata,
    summarize_rollout,
)
from .train_5b import COMPOSITE_WEIGHTS
from .training_env import LearningEnvironment


STAGE4_RUN_FORMAT_VERSION = 1
STAGE4_MIX = {
    "latest": 0.30,
    "historical": 0.20,
    "score": 0.25,
    "development": 0.25,
}
STAGE4_LEARNING_RATE = 1e-4
_MIXTURE_RNG_OFFSET = 7_000_000_000_000_000
_POOL_RNG_OFFSET = 8_000_000_000_000_000
_STAGE1_PANEL_POLICY_SEED_OFFSET = 6_000_000_000_000_000
_OPPONENT_KINDS = tuple(STAGE4_MIX)


@dataclass(frozen=True, slots=True)
class Stage4Config:
    stage_learning_decisions: int = 150_000
    training_seed: int = 4
    device: str = "cuda"
    active_envs: int = 32
    rollout_steps: int = 8192
    eval_interval: int = 50_000
    checkpoint_interval: int = 50_000
    eval_pairs: int = 200
    dev_panel_seed: int = 2
    max_completed_turns: int = 400
    learning_rate: float = STAGE4_LEARNING_RATE
    reward: str = "terminal"
    reward_shaping_alpha: float = 0.0
    opponent_mix: dict[str, float] = field(default_factory=lambda: dict(STAGE4_MIX))
    historical_checkpoints: tuple[str, ...] = ()
    historical_eval_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.stage_learning_decisions <= 0 or self.training_seed < 0:
            raise ValueError("Stage 4 budget must be positive and seed non-negative")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if min(
            self.active_envs,
            self.rollout_steps,
            self.eval_interval,
            self.checkpoint_interval,
            self.eval_pairs,
            self.max_completed_turns,
        ) <= 0:
            raise ValueError("Stage 4 sizes, intervals, and watchdog must be positive")
        if self.learning_rate != STAGE4_LEARNING_RATE:
            raise ValueError("Stage 4 learning rate is frozen at 1e-4")
        if self.reward != "terminal" or self.reward_shaping_alpha != 0.0:
            raise ValueError("Stage 4 only permits unshaped terminal reward")
        if self.opponent_mix != STAGE4_MIX:
            raise ValueError("Stage 4 opponent mix is frozen at 30/20/25/25")
        if not 1 <= len(self.historical_checkpoints) <= 6:
            raise ValueError("Stage 4 requires 1 to 6 real historical checkpoints")
        if len(set(self.historical_checkpoints)) != len(self.historical_checkpoints):
            raise ValueError("historical checkpoints must be unique")
        if not set(self.historical_eval_steps).issubset(
            {
                _checkpoint_step(Path(checkpoint))
                for checkpoint in self.historical_checkpoints
            }
        ):
            raise ValueError("historical DEV steps must exist in the historical pool")


@dataclass(slots=True)
class Stage4Progress:
    training: TrainingProgress = field(default_factory=TrainingProgress)
    source_checkpoint: str = ""
    source_global_step: int = 0
    source_learning_rate: float = 3e-4
    stage_learning_rate: float = STAGE4_LEARNING_RATE
    opponent_games: dict[str, int] = field(
        default_factory=lambda: {kind: 0 for kind in _OPPONENT_KINDS}
    )
    historical_games: dict[str, int] = field(default_factory=dict)
    evaluation_steps: list[int] = field(default_factory=list)
    best_heuristic_composite: float | None = None
    best_checkpoint: str | None = None
    elapsed_seconds: float = 0.0
    resume_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Stage4Progress:
        return cls(
            training=TrainingProgress.from_dict(values["training"]),
            source_checkpoint=str(values["source_checkpoint"]),
            source_global_step=int(values["source_global_step"]),
            source_learning_rate=float(values["source_learning_rate"]),
            stage_learning_rate=float(values["stage_learning_rate"]),
            opponent_games=dict(values["opponent_games"]),
            historical_games=dict(values["historical_games"]),
            evaluation_steps=list(values["evaluation_steps"]),
            best_heuristic_composite=values.get("best_heuristic_composite"),
            best_checkpoint=values.get("best_checkpoint"),
            elapsed_seconds=float(values.get("elapsed_seconds", 0.0)),
            resume_count=int(values.get("resume_count", 0)),
        )


class Stage4OpponentSelector:
    """Persistent mixture RNG; historical checkpoint RNG remains pool-owned."""

    def __init__(self, seed: int) -> None:
        self._rng = DeterministicRng(seed)

    @property
    def snapshot(self) -> RngState:
        return self._rng.snapshot()

    @classmethod
    def restore(cls, snapshot: RngState) -> Stage4OpponentSelector:
        selector = cls(0)
        selector._rng = DeterministicRng.restore(snapshot)
        return selector

    def choose_kind(self) -> str:
        draw = self._rng.randbelow(100)
        if draw < 30:
            return "latest"
        if draw < 50:
            return "historical"
        if draw < 75:
            return "score"
        return "development"


@dataclass(frozen=True, slots=True)
class _EpisodeOpponent:
    kind: str
    opponent_id: str


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return int(payload["global_learning_decision_step"])


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


class Stage4MetricWriter:
    def __init__(self, output_dir: Path, historical_ids: Sequence[str]) -> None:
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.csv_path = output_dir / "metrics.csv"
        opponent_fields = tuple(
            value
            for kind in _OPPONENT_KINDS
            for value in (f"opponent_{kind}_count", f"opponent_{kind}_fraction")
        )
        historical_fields = tuple(
            value
            for opponent_id in historical_ids
            for value in (
                f"historical_{_safe_id(opponent_id)}_count",
                f"historical_{_safe_id(opponent_id)}_fraction",
            )
        )
        eval_fields = (
            "panel_pairs",
            "random_score_rate",
            "score_policy_score_rate",
            "development_policy_score_rate",
            "heuristic_composite",
            "vs_source_score_rate",
        )
        system_fields = (
            "active_envs",
            "mean_inference_batch_size",
            "neural_opponent_inference_seconds",
            "neural_opponent_decisions",
            "latest_snapshot_seconds",
        )
        self.fields = tuple(
            dict.fromkeys(
                (*_METRIC_FIELDS, *opponent_fields, *historical_fields, *eval_fields, *system_fields)
            )
        )

    def append(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if write_header:
                writer.writeheader()
            writer.writerow({name: record.get(name, "") for name in self.fields})


def evaluate_stage4_panel(
    model: torch.nn.Module,
    *,
    opponents: dict[str, PolicySpec],
    dev_panel_seed: int,
    paired_seed_count: int,
    device: str,
    max_completed_turns: int,
    failure_dir: Path,
) -> tuple[dict[str, EvaluationResult], float]:
    required = {"random", "score", "development", "source_best"}
    if not required.issubset(opponents):
        raise ValueError("Stage 4 DEV panel is missing a required opponent")
    parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())
    was_training = model.training
    model.eval()
    learner = PolicySpec(
        "ppo-greedy",
        lambda seed: GreedyActorCriticPolicy(model, device=device),
    )
    seeds = dev_eval_game_seeds(dev_panel_seed, paired_seed_count)
    started = time.perf_counter()
    results = {
        name: evaluate_policies(
            learner,
            opponent,
            seeds,
            policy_seed_base=-(
                _STAGE1_PANEL_POLICY_SEED_OFFSET
                + dev_panel_seed * 10_000_000
            ),
            max_completed_turns=max_completed_turns,
            failure_dir=failure_dir / name,
        )
        for name, opponent in opponents.items()
    }
    elapsed = time.perf_counter() - started
    model.train(was_training)
    if not all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    ):
        raise AssertionError("Stage 4 DEV evaluation modified learner parameters")
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
        raise RuntimeError("Stage 4 DEV panel has no valid score")
    record: dict[str, Any] = {
        "record_type": "evaluation",
        "eval_kind": "stage4_fixed_panel",
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
        prefix = _safe_id(name)
        record.update(
            {
                f"{prefix}_wins": result.a_wins,
                f"{prefix}_draws": result.draws,
                f"{prefix}_losses": result.a_losses,
                f"{prefix}_score_rate": result.win_rate_score,
                f"{prefix}_first_player_score": result.a_first.win_rate_score,
                f"{prefix}_second_player_score": result.a_second.win_rate_score,
                f"{prefix}_mean_turns": result.mean_completed_player_turns,
            }
        )
    record["random_score_rate"] = results["random"].win_rate_score
    record["score_policy_score_rate"] = results["score"].win_rate_score
    record["development_policy_score_rate"] = results["development"].win_rate_score
    record["vs_source_score_rate"] = results["source_best"].win_rate_score
    record["heuristic_composite"] = sum(
        COMPOSITE_WEIGHTS[name] * float(results[name].win_rate_score)
        for name in ("random", "score", "development")
    )
    return record


def _write_panel(
    path: Path,
    results: dict[str, EvaluationResult],
    record: dict[str, Any],
    config: Stage4Config,
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
        ("heuristic_composite", "heuristic_composite"),
        ("vs_403183_score_rate", "vs_source_score_rate"),
    ):
        writer.add_scalar(f"eval/{tag}", record[key], step)


def _configure_stage4_learning_rate(trainer: PPOTrainer) -> float:
    source_rate = trainer.config.learning_rate
    trainer.config = replace(trainer.config, learning_rate=STAGE4_LEARNING_RATE)
    for group in trainer.optimizer.param_groups:
        group["lr"] = STAGE4_LEARNING_RATE
    return source_rate


def _immutable_resume_config(config: Stage4Config) -> dict[str, Any]:
    values = asdict(config)
    values.pop("stage_learning_decisions")
    return values


def _checkpoint_metadata(
    config: Stage4Config,
    progress: Stage4Progress,
    collector: BatchedRolloutCollector,
    selector: Stage4OpponentSelector,
    pool_manifest: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if collector.active_environment_count:
        raise ValueError("Stage 4 checkpoints require all active episodes to finish")
    return {
        "run_format_version": STAGE4_RUN_FORMAT_VERSION,
        "stage": "short_historical_self_play",
        "training_config": asdict(config),
        "progress": progress.to_dict(),
        "next_episode_index": collector.next_episode_index,
        "collector_at_episode_boundary": True,
        "mixture_rng": asdict(selector.snapshot),
        "pool_manifest": str(pool_manifest.resolve()),
        "learning_rate_transition": {
            "from": progress.source_learning_rate,
            "to": progress.stage_learning_rate,
        },
        "neural_opponent_action_mode": "sampled_policy",
        "runtime": runtime,
        "rules_version": RULESET_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def _save_checkpoint(
    trainer: PPOTrainer,
    sampler: PolicySampler,
    collector: BatchedRolloutCollector,
    selector: Stage4OpponentSelector,
    pool: OpponentPool,
    config: Stage4Config,
    progress: Stage4Progress,
    runtime: dict[str, Any],
    path: Path,
    manifest_dir: Path,
) -> None:
    manifest = manifest_dir / f"step_{trainer.global_learning_decision_step:09d}.json"
    pool.save_manifest(manifest)
    trainer.save_checkpoint(
        path,
        sampler=sampler,
        metadata=_checkpoint_metadata(
            config, progress, collector, selector, manifest, runtime
        ),
    )


def _drain(
    collector: BatchedRolloutCollector,
    rollout: RolloutBatch,
) -> RolloutBatch:
    drained = collector.drain_active_environments()
    return group_rollout_by_episode((*rollout.samples, *drained.samples))


def _opponent_training_breakdown(
    rollout_metrics: dict[str, Any],
    contexts: dict[str, _EpisodeOpponent],
) -> dict[str, Any]:
    rewards: dict[str, list[float]] = defaultdict(list)
    for episode in rollout_metrics["_episodes"]:
        if not episode["truncated"]:
            rewards[contexts[episode["episode_id"]].kind].append(float(episode["reward"]))
    record: dict[str, Any] = {}
    for kind in _OPPONENT_KINDS:
        values = rewards[kind]
        record[f"opponent_{kind}_train_score"] = (
            sum(1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in values)
            / len(values)
            if values
            else 0.0
        )
    return record


def run_stage4_training(
    config: Stage4Config,
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
    for name in ("checkpoints", "eval", "logs", "tensorboard", "opponent_pool"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    runtime = runtime_metadata(config.device)
    loaded_path = source_checkpoint if source_checkpoint is not None else resume_checkpoint
    trainer, sampler, metadata = PPOTrainer.load_checkpoint(
        loaded_path, device=config.device
    )
    if metadata.get("rules_version") != RULESET_ID:
        raise ValueError("Stage 4 source checkpoint rules version mismatch")
    if resume_checkpoint is None:
        if trainer.config.gamma != 1.0 or trainer.config.gae_lambda != 0.95:
            raise ValueError("Stage 4 source checkpoint has incompatible GAE settings")
        source_rate = _configure_stage4_learning_rate(trainer)
        progress = Stage4Progress(
            source_checkpoint=str(source_checkpoint.resolve()),
            source_global_step=trainer.global_learning_decision_step,
            source_learning_rate=source_rate,
        )
        selector = Stage4OpponentSelector(
            _MIXTURE_RNG_OFFSET + config.training_seed
        )
        pool = OpponentPool(
            device=config.device,
            max_historical=12,
            seed=_POOL_RNG_OFFSET + config.training_seed,
        )
        for checkpoint in config.historical_checkpoints:
            entry = pool.add_historical(
                Path(checkpoint),
                reason="Stage 4 initial time-spaced historical snapshot",
                added_at_step=progress.source_global_step,
            )
            progress.historical_games[entry.opponent_id] = 0
        start_episode_index = 0
        _write_json(
            output_dir / "config.json",
            {
                "run_format_version": STAGE4_RUN_FORMAT_VERSION,
                "stage": "short_historical_self_play",
                "training": asdict(config),
                "ppo": asdict(trainer.config),
                "model": asdict(trainer.model.config),
                "runtime": runtime,
                "source_checkpoint": str(source_checkpoint.resolve()),
                "learning_rate_transition": {
                    "from": source_rate,
                    "to": STAGE4_LEARNING_RATE,
                },
                "neural_opponent_action_mode": "sampled_policy",
            },
        )
    else:
        if metadata.get("run_format_version") != STAGE4_RUN_FORMAT_VERSION:
            raise ValueError("resume checkpoint is not a Stage 4 run")
        if metadata.get("stage") != "short_historical_self_play":
            raise ValueError("resume checkpoint has the wrong training stage")
        saved_config = Stage4Config(**metadata["training_config"])
        if _immutable_resume_config(saved_config) != _immutable_resume_config(config):
            raise ValueError("resume configuration changes an immutable Stage 4 setting")
        if trainer.config.learning_rate != STAGE4_LEARNING_RATE or any(
            group["lr"] != STAGE4_LEARNING_RATE
            for group in trainer.optimizer.param_groups
        ):
            raise ValueError("Stage 4 checkpoint did not restore learning rate 1e-4")
        progress = Stage4Progress.from_dict(metadata["progress"])
        progress.resume_count += 1
        start_episode_index = int(metadata["next_episode_index"])
        if not metadata.get("collector_at_episode_boundary"):
            raise ValueError("resume checkpoint is not at an episode boundary")
        selector = Stage4OpponentSelector.restore(RngState(**metadata["mixture_rng"]))
        pool = OpponentPool.load_manifest(
            Path(metadata["pool_manifest"]), device=config.device
        )
        _write_json(
            output_dir
            / f"resume_config_step_{trainer.global_learning_decision_step:09d}.json",
            {
                "run_format_version": STAGE4_RUN_FORMAT_VERSION,
                "stage": "short_historical_self_play",
                "training": asdict(config),
                "ppo": asdict(trainer.config),
                "runtime": runtime,
                "resume_checkpoint": str(resume_checkpoint.resolve()),
                "mixture_rng": metadata["mixture_rng"],
                "pool_manifest": metadata["pool_manifest"],
            },
        )

    historical_by_step = {
        int(entry.training_step): entry
        for entry in pool.historical_entries
        if entry.training_step is not None
    }
    source_model = load_frozen_actor_critic_checkpoint(
        Path(progress.source_checkpoint), device=config.device
    )
    eval_opponents: dict[str, PolicySpec] = {
        "random": baseline_policy_spec("random"),
        "score": baseline_policy_spec("score"),
        "development": baseline_policy_spec("development"),
        "source_best": PolicySpec(
            f"ppo-source-{progress.source_global_step}",
            lambda seed: GreedyActorCriticPolicy(source_model, device=config.device),
        ),
    }
    for step in config.historical_eval_steps:
        entry = historical_by_step[step]
        eval_opponents[f"historical_{step}"] = PolicySpec(
            entry.opponent_id,
            lambda seed, opponent_id=entry.opponent_id: pool.policy_for(
                opponent_id, seed=seed
            ).policy,
        )

    opponent_timing = NeuralInferenceTiming()
    episode_opponents: dict[str, _EpisodeOpponent] = {}

    def environment_factory(index: int) -> tuple[LearningEnvironment, int]:
        plan = episode_plan(config.training_seed, index)
        kind = selector.choose_kind()
        choice: OpponentChoice = pool.choose_for_training(
            kind,
            policy_seed=plan.opponent_seed,
            timing=opponent_timing,
        )
        episode_id = f"5b-stage4-{config.training_seed}-{index:08d}"
        episode_opponents[episode_id] = _EpisodeOpponent(
            kind=kind,
            opponent_id=choice.entry.opponent_id,
        )
        return (
            LearningEnvironment.new(
                plan.game_seed,
                first_player=plan.first_player,
                learning_player=plan.learning_player,
                opponent_policy=choice.policy,
                episode_id=episode_id,
                max_completed_turns=config.max_completed_turns,
                failure_dir=output_dir / "logs" / f"failure-{index:08d}",
            ),
            plan.learning_policy_seed,
        )

    collector = BatchedRolloutCollector(
        sampler,
        environment_factory,
        active_envs=config.active_envs,
        start_episode_index=start_episode_index,
    )
    historical_ids = tuple(entry.opponent_id for entry in pool.historical_entries)
    metrics = Stage4MetricWriter(output_dir, historical_ids)
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
        results, elapsed = evaluate_stage4_panel(
            trainer.model,
            opponents=eval_opponents,
            dev_panel_seed=config.dev_panel_seed,
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
        for name in results:
            if name.startswith("historical_"):
                tensorboard.add_scalar(
                    f"eval/{name}_score_rate", record[f"{_safe_id(name)}_score_rate"], step
                )
        _write_panel(
            output_dir / "eval" / f"step_{step:09d}_stage4_panel.json",
            results,
            record,
            config,
            runtime,
        )
        if any(not result.passed for result in results.values()):
            raise RuntimeError("Stage 4 DEV panel contained an error or truncation")
        composite = float(record["heuristic_composite"])
        if (
            progress.best_heuristic_composite is None
            or composite > progress.best_heuristic_composite
        ):
            progress.best_heuristic_composite = composite
            progress.best_checkpoint = str(checkpoint_path(step))
        if step not in progress.evaluation_steps:
            progress.evaluation_steps.append(step)
        tensorboard.flush()
        return record

    target_global_step = progress.source_global_step + config.stage_learning_decisions
    try:
        if resume_checkpoint is None:
            pool.set_latest(
                trainer.model, training_step=trainer.global_learning_decision_step
            )
            initial_path = checkpoint_path(trainer.global_learning_decision_step)
            _save_checkpoint(
                trainer,
                sampler,
                collector,
                selector,
                pool,
                config,
                progress,
                runtime,
                initial_path,
                output_dir / "opponent_pool",
            )
            perform_panel()
            progress.elapsed_seconds = previous_elapsed + (time.perf_counter() - started)
            _save_checkpoint(
                trainer,
                sampler,
                collector,
                selector,
                pool,
                config,
                progress,
                runtime,
                initial_path,
                output_dir / "opponent_pool",
            )

        relative_step = trainer.global_learning_decision_step - progress.source_global_step
        next_eval = progress.source_global_step + (
            relative_step // config.eval_interval + 1
        ) * config.eval_interval
        next_checkpoint = progress.source_global_step + (
            relative_step // config.checkpoint_interval + 1
        ) * config.checkpoint_interval

        while trainer.global_learning_decision_step < target_global_step:
            cycle_snapshot_started = time.perf_counter()
            latest_entry = pool.set_latest(
                trainer.model, training_step=trainer.global_learning_decision_step
            )
            latest_snapshot_seconds = time.perf_counter() - cycle_snapshot_started
            latest_policy = pool.policy_for(latest_entry.opponent_id).policy
            latest_model = latest_policy.model
            if any(
                learner.data_ptr() == opponent.data_ptr()
                for learner, opponent in zip(
                    trainer.model.parameters(), latest_model.parameters(), strict=True
                )
            ):
                raise AssertionError("latest opponent shares learner parameter storage")
            if any(parameter.requires_grad for parameter in latest_model.parameters()):
                raise AssertionError("latest opponent unexpectedly requires gradients")
            optimizer_parameter_ids = {
                id(parameter)
                for group in trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            if any(id(parameter) in optimizer_parameter_ids for parameter in latest_model.parameters()):
                raise AssertionError("latest opponent appears in learner optimizer")
            latest_before = tuple(
                parameter.detach().clone() for parameter in latest_model.parameters()
            )

            current = trainer.global_learning_decision_step
            cycle_boundary = min(next_eval, next_checkpoint, target_global_step)
            target = min(config.rollout_steps, target_global_step - current)
            boundary_due = current + target >= cycle_boundary
            sampler.reset_timing()
            opponent_timing.reset()
            if config.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rollout_started = time.perf_counter()
            rollout = collector.collect(target)
            if boundary_due:
                rollout = _drain(collector, rollout)
            if config.device == "cuda":
                torch.cuda.synchronize()
            rollout_seconds = time.perf_counter() - rollout_started
            completed = collector.take_completed_environments()
            completed_contexts = {
                environment.episode_id: episode_opponents[environment.episode_id]
                for environment in completed
            }
            rollout_metrics = summarize_rollout(rollout, completed)
            kind_counts = Counter(
                completed_contexts[environment.episode_id].kind
                for environment in completed
            )
            historical_counts = Counter(
                completed_contexts[environment.episode_id].opponent_id
                for environment in completed
                if completed_contexts[environment.episode_id].kind == "historical"
            )
            completed_total = len(completed)
            for kind in _OPPONENT_KINDS:
                progress.opponent_games[kind] += kind_counts[kind]
                rollout_metrics[f"opponent_{kind}_count"] = kind_counts[kind]
                rollout_metrics[f"opponent_{kind}_fraction"] = (
                    kind_counts[kind] / completed_total if completed_total else 0.0
                )
            historical_total = sum(historical_counts.values())
            for opponent_id in historical_ids:
                progress.historical_games[opponent_id] += historical_counts[opponent_id]
                prefix = f"historical_{_safe_id(opponent_id)}"
                rollout_metrics[f"{prefix}_count"] = historical_counts[opponent_id]
                rollout_metrics[f"{prefix}_fraction"] = (
                    historical_counts[opponent_id] / historical_total
                    if historical_total
                    else 0.0
                )
            rollout_metrics.update(
                _opponent_training_breakdown(rollout_metrics, completed_contexts)
            )

            gae_started = time.perf_counter()
            prepared = compute_gae(
                rollout,
                gamma=trainer.config.gamma,
                gae_lambda=trainer.config.gae_lambda,
            )
            gae_seconds = time.perf_counter() - gae_started
            update_started = time.perf_counter()
            update_stats = trainer.update(prepared)
            if config.device == "cuda":
                torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            if not all(
                torch.equal(before, after)
                for before, after in zip(
                    latest_before, latest_model.parameters(), strict=True
                )
            ):
                raise AssertionError("learner optimizer changed frozen latest opponent")
            _update_progress(progress.training, rollout_metrics)
            for environment in completed:
                episode_opponents.pop(environment.episode_id)
            step = trainer.global_learning_decision_step
            environment_seconds = max(
                0.0,
                rollout_seconds
                - sampler.inference_seconds
                - opponent_timing.seconds,
            )
            record = {
                "record_type": "update",
                "step": step,
                "stage_learning_decisions": step - progress.source_global_step,
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
                "neural_opponent_inference_seconds": opponent_timing.seconds,
                "neural_opponent_decisions": opponent_timing.decisions,
                "mean_inference_batch_size": sampler.mean_inference_batch_size,
                "latest_snapshot_seconds": latest_snapshot_seconds,
                "gae_seconds": gae_seconds,
                "ppo_update_seconds": update_seconds,
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                    if config.device == "cuda"
                    else 0.0
                ),
                "cpu_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
                "active_envs": config.active_envs,
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
                    raise FloatingPointError("non-finite Stage 4 training metric")
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
            for kind in _OPPONENT_KINDS:
                tensorboard.add_scalar(
                    f"opponent/{kind}_episode_fraction",
                    record[f"opponent_{kind}_fraction"],
                    step,
                )
            for opponent_id in historical_ids:
                prefix = f"historical_{_safe_id(opponent_id)}"
                tensorboard.add_scalar(
                    f"historical/{_safe_id(opponent_id)}_episode_fraction",
                    record[f"{prefix}_fraction"],
                    step,
                )
            for key in (
                "learning_decisions_per_second",
                "games_per_second",
                "neural_opponent_inference_seconds",
                "mean_inference_batch_size",
                "environment_encoding_seconds",
                "ppo_update_seconds",
                "cuda_peak_allocated_mib",
                "cpu_max_rss_mib",
            ):
                tensorboard.add_scalar(f"system/{key}", record[key], step)
            tensorboard.add_scalar(
                "system/learner_decisions_per_sec",
                record["learning_decisions_per_second"],
                step,
            )
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
            final_due = step >= target_global_step
            if evaluation_due:
                perform_panel()
                while next_eval <= step:
                    next_eval += config.eval_interval
            if evaluation_due or checkpoint_due or final_due:
                if collector.active_environment_count:
                    raise AssertionError("checkpoint boundary retained active environments")
                progress.elapsed_seconds = previous_elapsed + (
                    time.perf_counter() - started
                )
                _save_checkpoint(
                    trainer,
                    sampler,
                    collector,
                    selector,
                    pool,
                    config,
                    progress,
                    runtime,
                    checkpoint_path(step),
                    output_dir / "opponent_pool",
                )
                while next_checkpoint <= step:
                    next_checkpoint += config.checkpoint_interval

        final_step = trainer.global_learning_decision_step
        if final_step not in progress.evaluation_steps:
            perform_panel()
        progress.elapsed_seconds = previous_elapsed + (time.perf_counter() - started)
        final_path = checkpoint_path(final_step)
        _save_checkpoint(
            trainer,
            sampler,
            collector,
            selector,
            pool,
            config,
            progress,
            runtime,
            final_path,
            output_dir / "opponent_pool",
        )
        summary = {
            "stage": "short_historical_self_play",
            "training_config": asdict(config),
            "ppo_config": asdict(trainer.config),
            "runtime": runtime,
            "global_learning_decisions": final_step,
            "stage_learning_decisions": final_step - progress.source_global_step,
            "ppo_updates": trainer.update_count,
            "progress": progress.to_dict(),
            "neural_opponent_action_mode": "sampled_policy",
            "latest_checkpoint": str(final_path),
            "best_dev_checkpoint": progress.best_checkpoint,
            "starting_checkpoint": progress.source_checkpoint,
        }
        _write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        tensorboard.flush()
        tensorboard.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5B Stage 4 short self-play validation")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-checkpoint", type=Path)
    source.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--stage-learning-decisions", type=int, default=150_000)
    parser.add_argument("--training-seed", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--active-envs", type=int, choices=(16, 32), default=32)
    parser.add_argument("--rollout-size", type=int, default=8192)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--eval-pairs", type=int, default=200)
    parser.add_argument("--dev-panel-seed", type=int, default=2)
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument(
        "--historical-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--historical-eval-step", type=int, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = Stage4Config(
        stage_learning_decisions=args.stage_learning_decisions,
        training_seed=args.training_seed,
        device=args.device,
        active_envs=args.active_envs,
        rollout_steps=args.rollout_size,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        eval_pairs=args.eval_pairs,
        dev_panel_seed=args.dev_panel_seed,
        max_completed_turns=args.max_completed_turns,
        historical_checkpoints=tuple(
            str(path.resolve()) for path in args.historical_checkpoint
        ),
        historical_eval_steps=tuple(args.historical_eval_step),
    )
    summary = run_stage4_training(
        config,
        output_dir=args.output_dir,
        source_checkpoint=args.source_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
