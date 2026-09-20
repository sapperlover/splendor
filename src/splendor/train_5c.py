"""Work-package 5C frozen 5M self-play training entrypoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .catalog import load_default_catalog
from .evaluation import EvaluationResult, PolicySpec, baseline_policy_spec, evaluate_policies
from .features import FEATURE_SCHEMA_VERSION
from .neural_policy import GreedyActorCriticPolicy, NeuralInferenceTiming
from .multiprocess_collector import (
    CompletedLearningEpisode,
    MultiprocessEpisodeLaunch,
    MultiprocessEpisodeSpec,
    MultiprocessRolloutCollector,
    restored_worker_rng_states,
)
from .opponent_pool import OpponentChoice, OpponentPool, load_frozen_actor_critic_checkpoint
from .paired_statistics import pair_scores, paired_bootstrap_difference
from .ppo import (
    BatchedRolloutCollector,
    PPOConfig,
    PPOTrainer,
    PolicySampler,
    RolloutBatch,
    compute_gae,
    group_rollout_by_episode,
)
from .replay import RULESET_ID, to_primitive
from .rng import DeterministicRng, RNG_ALGORITHM, RngState
from .train_5a import (
    TrainingProgress,
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


FIVE_C_RUN_FORMAT_VERSION = 1
FIVE_C_STAGE = "5c_main_5m_self_play"
RUN_LOCAL_TARGET = 5_000_000
PHASE_B_START = 1_000_000
INITIAL_LEARNING_RATE = 1e-4
FINAL_LEARNING_RATE = 3e-5
INITIAL_ENTROPY_COEF = 0.01
FINAL_ENTROPY_COEF = 0.002
PHASE_A_MIX = {
    "latest": 0.30,
    "historical": 0.20,
    "score": 0.25,
    "development": 0.25,
}
PHASE_B_MIX = {
    "latest": 0.50,
    "historical": 0.30,
    "score": 0.10,
    "development": 0.10,
}
_OPPONENT_KINDS = tuple(PHASE_A_MIX)
_MIXTURE_RNG_OFFSET = 7_100_000_000_000_000
_POOL_RNG_OFFSET = 8_100_000_000_000_000
_DEV_POLICY_RNG_OFFSET = 6_000_000_000_000_000
_FINAL_GAME_RNG_OFFSET = 9_000_000_000_000_000
_FINAL_POLICY_RNG_OFFSET = 9_500_000_000_000_000
_PREFLIGHT_GAME_RNG_OFFSET = 8_500_000_000_000_000
_PREFLIGHT_POLICY_RNG_OFFSET = 8_600_000_000_000_000
_BEST_DIRECT_THRESHOLD = 0.01
_BEST_BOOTSTRAP_RESAMPLES = 100_000


@dataclass(frozen=True, slots=True)
class FiveCConfig:
    run_id: str
    source_checkpoint: str
    historical_checkpoints: tuple[str, ...]
    anchor_403183_checkpoint: str
    anchor_512112_checkpoint: str
    historical_eval_steps: tuple[int, ...] = (157_351, 304_858)
    run_local_learning_decisions: int = RUN_LOCAL_TARGET
    training_seed: int = 1
    device: str = "cuda"
    active_envs: int = 32
    collector_backend: str = "single_process"
    num_workers: int = 8
    envs_per_worker: int = 4
    rollout_steps: int = 8_192
    checkpoint_interval: int = 100_000
    quick_eval_interval: int = 100_000
    quick_eval_pairs: int = 50
    formal_eval_interval: int = 250_000
    formal_eval_pairs: int = 200
    dev_panel_seed: int = 2
    final_panel_seed: int = 1
    final_eval_min_pairs: int = 500
    max_completed_turns: int = 400
    phase_b_start: int = PHASE_B_START
    initial_learning_rate: float = INITIAL_LEARNING_RATE
    final_learning_rate: float = FINAL_LEARNING_RATE
    initial_entropy_coef: float = INITIAL_ENTROPY_COEF
    final_entropy_coef: float = FINAL_ENTROPY_COEF
    phase_a_mix: dict[str, float] = field(default_factory=lambda: dict(PHASE_A_MIX))
    phase_b_mix: dict[str, float] = field(default_factory=lambda: dict(PHASE_B_MIX))
    historical_capacity: int = 12
    dynamic_history_min_interval: int = 250_000
    reward: str = "terminal"
    reward_shaping_alpha: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    ppo_epochs: int = 4
    minibatch_size: int = 512
    advantage_normalization: bool = True
    value_clipping: bool = False
    precision: str = "float32"
    optimizer: str = "Adam"
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("5C run id must not be empty")
        if self.run_local_learning_decisions != RUN_LOCAL_TARGET:
            raise ValueError("the first 5C main run is frozen at 5,000,000 decisions")
        if self.training_seed != 1:
            raise ValueError("the first 5C main run training seed is frozen at 1")
        if self.device != "cuda" or self.active_envs != 32:
            raise ValueError("the first 5C main run requires CUDA and 32 active envs")
        if self.collector_backend not in ("single_process", "multiprocessing"):
            raise ValueError("unsupported 5C collector backend")
        if self.num_workers not in (1, 2, 4, 8):
            raise ValueError("5C multiprocessing workers must be one of 1, 2, 4, or 8")
        if self.num_workers * self.envs_per_worker != self.active_envs:
            raise ValueError("num_workers * envs_per_worker must equal active_envs")
        if (
            self.rollout_steps != 8_192
            or self.checkpoint_interval != 100_000
            or self.quick_eval_interval != 100_000
            or self.quick_eval_pairs != 50
            or self.formal_eval_interval != 250_000
            or self.formal_eval_pairs != 200
            or self.final_eval_min_pairs != 500
        ):
            raise ValueError("5C rollout/checkpoint/evaluation cadence is frozen")
        if self.phase_b_start != PHASE_B_START:
            raise ValueError("5C Phase B must start at run-local decision 1,000,000")
        if self.phase_a_mix != PHASE_A_MIX or self.phase_b_mix != PHASE_B_MIX:
            raise ValueError("5C opponent mixture schedule is frozen")
        if (
            self.initial_learning_rate != INITIAL_LEARNING_RATE
            or self.final_learning_rate != FINAL_LEARNING_RATE
            or self.initial_entropy_coef != INITIAL_ENTROPY_COEF
            or self.final_entropy_coef != FINAL_ENTROPY_COEF
        ):
            raise ValueError("5C LR and entropy schedules are frozen")
        if (
            self.reward != "terminal"
            or self.reward_shaping_alpha != 0.0
            or self.gamma != 1.0
            or self.gae_lambda != 0.95
            or self.clip_coef != 0.2
            or self.value_coef != 0.5
            or self.max_grad_norm != 0.5
            or self.target_kl != 0.02
            or self.ppo_epochs != 4
            or self.minibatch_size != 512
            or not self.advantage_normalization
            or self.value_clipping
            or self.precision != "float32"
            or self.optimizer != "Adam"
            or self.weight_decay != 0.0
        ):
            raise ValueError("5C PPO/reward/precision configuration is frozen")
        if not 8 <= len(self.historical_checkpoints) <= self.historical_capacity <= 12:
            raise ValueError("5C requires 8-12 historical checkpoints and capacity <= 12")
        if len(set(self.historical_checkpoints)) != len(self.historical_checkpoints):
            raise ValueError("5C historical checkpoints must be unique")
        if len(set(self.historical_eval_steps)) != len(self.historical_eval_steps):
            raise ValueError("5C historical evaluation anchors must be unique")
        if self.max_completed_turns <= 0 or self.dynamic_history_min_interval <= 0:
            raise ValueError("5C watchdog and historical interval must be positive")


@dataclass(slots=True)
class FiveCProgress:
    training: TrainingProgress = field(default_factory=TrainingProgress)
    source_checkpoint: str = ""
    source_lifetime_step: int = 0
    source_update_count: int = 0
    evaluation_steps: list[int] = field(default_factory=list)
    quick_evaluation_steps: list[int] = field(default_factory=list)
    best_heuristic_composite: float | None = None
    best_checkpoint: str | None = None
    best_formal_eval_path: str | None = None
    opponent_games: dict[str, int] = field(
        default_factory=lambda: {kind: 0 for kind in _OPPONENT_KINDS}
    )
    historical_games: dict[str, int] = field(default_factory=dict)
    dynamic_historical_ids: list[str] = field(default_factory=list)
    last_historical_add_local_step: int = 0
    elapsed_seconds: float = 0.0
    resume_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FiveCProgress:
        return cls(
            training=TrainingProgress.from_dict(values["training"]),
            source_checkpoint=str(values["source_checkpoint"]),
            source_lifetime_step=int(values["source_lifetime_step"]),
            source_update_count=int(values["source_update_count"]),
            evaluation_steps=list(values["evaluation_steps"]),
            quick_evaluation_steps=list(values["quick_evaluation_steps"]),
            best_heuristic_composite=values.get("best_heuristic_composite"),
            best_checkpoint=values.get("best_checkpoint"),
            best_formal_eval_path=values.get("best_formal_eval_path"),
            opponent_games=dict(values["opponent_games"]),
            historical_games=dict(values["historical_games"]),
            dynamic_historical_ids=list(values.get("dynamic_historical_ids", [])),
            last_historical_add_local_step=int(
                values.get("last_historical_add_local_step", 0)
            ),
            elapsed_seconds=float(values.get("elapsed_seconds", 0.0)),
            resume_count=int(values.get("resume_count", 0)),
        )


class FiveCOpponentSelector:
    def __init__(self, seed: int) -> None:
        self._rng = DeterministicRng(seed)

    @property
    def snapshot(self) -> RngState:
        return self._rng.snapshot()

    @classmethod
    def restore(cls, snapshot: RngState) -> FiveCOpponentSelector:
        selector = cls(0)
        selector._rng = DeterministicRng.restore(snapshot)
        return selector

    def choose_kind(self, run_local_step: int) -> str:
        if run_local_step < 0:
            raise ValueError("run-local step must be non-negative")
        draw = self._rng.randbelow(100)
        if run_local_step < PHASE_B_START:
            if draw < 30:
                return "latest"
            if draw < 50:
                return "historical"
            if draw < 75:
                return "score"
            return "development"
        if draw < 50:
            return "latest"
        if draw < 80:
            return "historical"
        if draw < 90:
            return "score"
        return "development"


@dataclass(frozen=True, slots=True)
class _EpisodeOpponent:
    kind: str
    opponent_id: str


def final_eval_game_seeds(final_panel_seed: int, count: int) -> tuple[int, ...]:
    if final_panel_seed < 0 or count <= 0:
        raise ValueError("FINAL panel seed must be non-negative and count positive")
    return tuple(
        -(_FINAL_GAME_RNG_OFFSET + final_panel_seed * 10_000_000 + index + 1)
        for index in range(count)
    )


def linear_schedule(
    start: float, end: float, run_local_step: int, total_steps: int
) -> float:
    if total_steps <= 0 or run_local_step < 0:
        raise ValueError("schedule total must be positive and step non-negative")
    if run_local_step == 0:
        return start
    if run_local_step >= total_steps:
        return end
    fraction = run_local_step / total_steps
    return start + (end - start) * fraction


def scheduled_learning_rate(config: FiveCConfig, run_local_step: int) -> float:
    return linear_schedule(
        config.initial_learning_rate,
        config.final_learning_rate,
        run_local_step,
        config.run_local_learning_decisions,
    )


def scheduled_entropy_coef(config: FiveCConfig, run_local_step: int) -> float:
    return linear_schedule(
        config.initial_entropy_coef,
        config.final_entropy_coef,
        run_local_step,
        config.run_local_learning_decisions,
    )


def _apply_schedules(
    trainer: PPOTrainer, config: FiveCConfig, run_local_step: int
) -> tuple[float, float]:
    learning_rate = scheduled_learning_rate(config, run_local_step)
    entropy_coef = scheduled_entropy_coef(config, run_local_step)
    trainer.config = replace(
        trainer.config,
        learning_rate=learning_rate,
        entropy_coef=entropy_coef,
    )
    for group in trainer.optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate, entropy_coef


def _configure_trainer(trainer: PPOTrainer, config: FiveCConfig) -> None:
    trainer.config = PPOConfig(
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        learning_rate=config.initial_learning_rate,
        clip_coef=config.clip_coef,
        value_coef=config.value_coef,
        entropy_coef=config.initial_entropy_coef,
        max_grad_norm=config.max_grad_norm,
        ppo_epochs=config.ppo_epochs,
        minibatch_size=config.minibatch_size,
        target_kl=config.target_kl,
        rollout_steps=config.rollout_steps,
        advantage_epsilon=trainer.config.advantage_epsilon,
    )
    for group in trainer.optimizer.param_groups:
        group["lr"] = config.initial_learning_rate
        if group["weight_decay"] != config.weight_decay:
            raise ValueError("source checkpoint Adam weight_decay is incompatible")
        if group["betas"] != (0.9, 0.999) or group["eps"] != 1e-8:
            raise ValueError("source checkpoint Adam betas/eps are incompatible")
    if not isinstance(trainer.optimizer, torch.optim.Adam):
        raise ValueError("5C requires the existing Adam optimizer")
    if any(parameter.dtype is not torch.float32 for parameter in trainer.model.parameters()):
        raise ValueError("5C requires FP32 model parameters")


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return int(payload["global_learning_decision_step"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def save_five_c_config(path: Path, config: FiveCConfig) -> None:
    _write_json(
        path,
        {
            "run_format_version": FIVE_C_RUN_FORMAT_VERSION,
            "stage": FIVE_C_STAGE,
            "training": asdict(config),
        },
    )


def load_five_c_config(path: Path) -> FiveCConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_format_version") != FIVE_C_RUN_FORMAT_VERSION:
        raise ValueError("unsupported 5C config format")
    if payload.get("stage") != FIVE_C_STAGE:
        raise ValueError("config is not a 5C main run")
    values = dict(payload["training"])
    values["historical_checkpoints"] = tuple(values["historical_checkpoints"])
    values["historical_eval_steps"] = tuple(values["historical_eval_steps"])
    return FiveCConfig(**values)


def _validate_checkpoint_set(config: FiveCConfig) -> dict[str, Any]:
    source_path = Path(config.source_checkpoint)
    if not source_path.is_file():
        raise FileNotFoundError(f"missing 5C source checkpoint: {source_path}")
    trainer, _, metadata = PPOTrainer.load_checkpoint(source_path, device="cpu")
    if trainer.global_learning_decision_step != 512_112:
        raise ValueError("5C source checkpoint is not step_000512112.pt")
    if metadata.get("rules_version") != RULESET_ID:
        raise ValueError("5C source rules version mismatch")
    model_config = asdict(trainer.model.config)
    optimizer_state_entries = len(trainer.optimizer.state)
    if optimizer_state_entries == 0:
        raise ValueError("5C source checkpoint has no recoverable Adam state")
    checkpoints = (
        *config.historical_checkpoints,
        config.anchor_403183_checkpoint,
        config.anchor_512112_checkpoint,
    )
    seen_steps: dict[int, str] = {}
    for checkpoint in checkpoints:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"missing 5C checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"feature schema mismatch: {path}")
        if payload.get("metadata", {}).get("rules_version") != RULESET_ID:
            raise ValueError(f"rules version mismatch: {path}")
        if dict(payload["model_config"]) != model_config:
            raise ValueError(f"model config mismatch: {path}")
        seen_steps[int(payload["global_learning_decision_step"])] = str(path)
    if 403_183 not in seen_steps or 512_112 not in seen_steps:
        raise ValueError("5C fixed anchors are not the required 403183 and 512112 models")
    historical_steps = {_checkpoint_step(Path(path)) for path in config.historical_checkpoints}
    if not set(config.historical_eval_steps).issubset(historical_steps):
        raise ValueError("historical DEV anchors are absent from the initial pool")
    return {
        "source_lifetime_step": trainer.global_learning_decision_step,
        "source_update_count": trainer.update_count,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_config": model_config,
        "ppo_config": asdict(trainer.config),
        "optimizer": type(trainer.optimizer).__name__,
        "optimizer_state_entries": optimizer_state_entries,
        "source_sha256": _file_sha256(source_path),
    }


def prepare_five_c_run(config: FiveCConfig, *, run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing 5C run directory: {run_root}")
    validation = _validate_checkpoint_set(config)
    directories = (
        "checkpoints",
        "eval/quick",
        "eval/formal",
        "eval/failures",
        "final",
        "failures",
        "historical_pool",
        "logs",
        "tensorboard",
    )
    for name in directories:
        (run_root / name).mkdir(parents=True, exist_ok=False)
    save_five_c_config(run_root / "config.json", config)

    pool = OpponentPool(
        device="cpu",
        max_historical=config.historical_capacity,
        seed=_POOL_RNG_OFFSET + config.training_seed,
    )
    for checkpoint in config.historical_checkpoints:
        pool.add_historical(
            Path(checkpoint),
            reason="5C initial protected time-spaced snapshot",
            added_at_step=0,
        )
    initial_manifest = run_root / "historical_pool" / "initial.json"
    pool.save_manifest(initial_manifest)
    metadata = {
        "run_format_version": FIVE_C_RUN_FORMAT_VERSION,
        "stage": FIVE_C_STAGE,
        "status": "prepared",
        "prepared_at_utc": _utc_now(),
        "run_root": str(run_root),
        "config_path": str((run_root / "config.json").resolve()),
        "source": validation,
        "source_checkpoint": str(Path(config.source_checkpoint).resolve()),
        "initial_pool_manifest": str(initial_manifest.resolve()),
        "runtime": runtime_metadata("cpu"),
        "data_fingerprint": load_default_catalog().data_fingerprint,
        "rules_version": RULESET_ID,
        "rng_algorithm": RNG_ALGORITHM,
        "seed_domains": {
            "train": "episode_plan(seed=1): non-negative 10,000,000 + episode_index",
            "dev": "fixed negative Stage-1-compatible domain -20,000,001...",
            "final": f"negative offset {_FINAL_GAME_RNG_OFFSET} + final panel seed",
            "preflight": f"positive offset {_PREFLIGHT_GAME_RNG_OFFSET}; never used by TRAIN/DEV/FINAL",
        },
        "budget_boundary": (
            "exactly 5,000,000 PPO learning-decision samples; active games are "
            "finished at event boundaries and their untrained drain tails are "
            "reported separately"
        ),
    }
    _write_json(run_root / "metadata.json", metadata)
    _write_json(
        run_root / "status.json",
        {
            "status": "prepared",
            "run_local_step": 0,
            "target_run_local_step": RUN_LOCAL_TARGET,
            "lifetime_step": validation["source_lifetime_step"],
            "last_quick_eval": None,
            "last_formal_eval": None,
            "best_composite": None,
            "nan_or_inf": 0,
            "engineering_truncations": 0,
        },
    )
    (run_root / "logs" / "train.log").write_text("", encoding="utf-8")
    return metadata


class FiveCMetricWriter:
    _FIELDS = (
        "record_type",
        "eval_kind",
        "run_local_step",
        "lifetime_step",
        "run_local_update_count",
        "lifetime_update_count",
        "wall_seconds",
        "policy_loss",
        "value_loss",
        "total_loss",
        "entropy",
        "entropy_coef",
        "explained_variance",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "learning_rate",
        "epochs_completed",
        "minibatches_processed",
        "gradient_clipped_minibatches",
        "kl_early_stopped",
        "train_win_rate",
        "train_draw_rate",
        "mean_reward",
        "mean_valid_candidates",
        "single_candidate_fraction",
        "games_completed",
        "engineering_truncations",
        "forced_skips",
        "mean_completed_player_turns",
        "TakeDistinct_fraction",
        "TakePair_fraction",
        "ReserveMarket_fraction",
        "ReserveDeck_fraction",
        "BuyMarket_fraction",
        "BuyReserved_fraction",
        "mean_buys_per_game",
        "mean_reserves_per_game",
        "mean_token_taking_per_game",
        "opponent_phase",
        "opponent_latest_fraction",
        "opponent_historical_fraction",
        "opponent_score_fraction",
        "opponent_development_fraction",
        "learning_decisions_per_second",
        "games_per_second",
        "mean_inference_batch_size",
        "mean_opponent_inference_batch_size",
        "num_workers",
        "envs_per_worker",
        "environment_encoding_seconds",
        "model_inference_seconds",
        "neural_opponent_inference_seconds",
        "ipc_wait_seconds",
        "worker_cpu_seconds",
        "ppo_update_seconds",
        "cuda_peak_allocated_mib",
        "cpu_max_rss_mib",
        "boundary_drain_decisions_not_trained",
        "panel_pairs",
        "random_score_rate",
        "score_policy_score_rate",
        "development_policy_score_rate",
        "heuristic_composite",
        "vs_403183_score_rate",
        "vs_512112_score_rate",
        "evaluation_seconds",
        "illegal_actions",
        "invariant_failures",
        "exceptions",
    )

    def __init__(self, run_root: Path) -> None:
        self.jsonl_path = run_root / "metrics.jsonl"
        self.csv_path = run_root / "metrics.csv"

    def append(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({name: record.get(name, "") for name in self._FIELDS})


def build_five_c_eval_opponents(
    config: FiveCConfig,
    *,
    device: str,
    include_historical: bool,
) -> dict[str, PolicySpec]:
    opponents: dict[str, PolicySpec] = {
        name: baseline_policy_spec(name)
        for name in ("random", "score", "development")
    }
    anchors = {
        "anchor_403183": Path(config.anchor_403183_checkpoint),
        "anchor_512112": Path(config.anchor_512112_checkpoint),
    }
    historical_paths = {
        _checkpoint_step(Path(path)): Path(path)
        for path in config.historical_checkpoints
    }
    if include_historical:
        anchors.update(
            {
                f"historical_{step}": historical_paths[step]
                for step in config.historical_eval_steps
            }
        )
    for name, path in anchors.items():
        model = load_frozen_actor_critic_checkpoint(path, device=device)
        opponents[name] = PolicySpec(
            name,
            lambda seed, frozen=model: GreedyActorCriticPolicy(
                frozen, device=device
            ),
        )
    return opponents


def evaluate_five_c_panel(
    model: torch.nn.Module,
    *,
    opponents: dict[str, PolicySpec],
    seeds: Sequence[int],
    policy_seed_base: int,
    device: str,
    max_completed_turns: int,
    failure_dir: Path,
    progress_callback: Callable[
        [str, int, int, str, EvaluationResult | None], None
    ]
    | None = None,
    pair_progress_callback: Callable[[str, int, int, int, int], None]
    | None = None,
) -> tuple[dict[str, EvaluationResult], float]:
    required = {
        "random",
        "score",
        "development",
        "anchor_403183",
        "anchor_512112",
    }
    if not required.issubset(opponents):
        raise ValueError("5C evaluation panel is missing a required fixed opponent")
    parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())
    was_training = model.training
    model.eval()
    candidate = PolicySpec(
        "ppo-greedy",
        lambda seed: GreedyActorCriticPolicy(model, device=device),
    )
    started = time.perf_counter()
    results: dict[str, EvaluationResult] = {}
    total_opponents = len(opponents)
    for index, (name, opponent) in enumerate(opponents.items(), start=1):
        if progress_callback is not None:
            progress_callback(name, index, total_opponents, "started", None)

        def pair_progress(completed_pairs: int, total_pairs: int) -> None:
            if pair_progress_callback is None:
                return
            report_interval = max(1, total_pairs // 10)
            if completed_pairs < total_pairs and completed_pairs % report_interval == 0:
                pair_progress_callback(
                    name,
                    index,
                    total_opponents,
                    completed_pairs,
                    total_pairs,
                )

        result = evaluate_policies(
            candidate,
            opponent,
            seeds,
            policy_seed_base=policy_seed_base,
            max_completed_turns=max_completed_turns,
            failure_dir=failure_dir / name,
            progress_callback=pair_progress,
        )
        results[name] = result
        if progress_callback is not None:
            progress_callback(name, index, total_opponents, "completed", result)
    elapsed = time.perf_counter() - started
    model.train(was_training)
    if not all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    ):
        raise AssertionError("5C evaluation modified candidate model parameters")
    if any(not result.passed for result in results.values()):
        raise RuntimeError("5C evaluation contained an error or engineering truncation")
    return results, elapsed


def _panel_record(
    results: dict[str, EvaluationResult],
    *,
    kind: str,
    run_local_step: int,
    lifetime_step: int,
    run_local_update_count: int,
    lifetime_update_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": "evaluation",
        "eval_kind": kind,
        "run_local_step": run_local_step,
        "lifetime_step": lifetime_step,
        "run_local_update_count": run_local_update_count,
        "lifetime_update_count": lifetime_update_count,
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
    record["vs_403183_score_rate"] = results["anchor_403183"].win_rate_score
    record["vs_512112_score_rate"] = results["anchor_512112"].win_rate_score
    record["heuristic_composite"] = sum(
        COMPOSITE_WEIGHTS[name] * float(results[name].win_rate_score)
        for name in ("random", "score", "development")
    )
    return record


def _write_panel(
    path: Path,
    *,
    results: dict[str, EvaluationResult],
    record: dict[str, Any],
    config: FiveCConfig,
    seed_domain: str,
    checkpoint_path: Path,
) -> None:
    _write_json(
        path,
        {
            "run_format_version": FIVE_C_RUN_FORMAT_VERSION,
            "stage": FIVE_C_STAGE,
            "seed_domain": seed_domain,
            "checkpoint": str(checkpoint_path.resolve()),
            "training_config": asdict(config),
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
    writer: SummaryWriter, record: dict[str, Any], run_local_step: int
) -> None:
    for tag, key in (
        ("random_score_rate", "random_score_rate"),
        ("score_policy_score_rate", "score_policy_score_rate"),
        ("development_policy_score_rate", "development_policy_score_rate"),
        ("heuristic_composite", "heuristic_composite"),
        ("vs_403183_score_rate", "vs_403183_score_rate"),
        ("vs_512112_score_rate", "vs_512112_score_rate"),
    ):
        writer.add_scalar(f"eval/{tag}", record[key], run_local_step)
    writer.add_scalar("system/eval_seconds", record["evaluation_seconds"], run_local_step)


def _weighted_pair_scores(
    results: dict[str, EvaluationResult],
) -> tuple[tuple[int, ...], np.ndarray]:
    shared_seeds: tuple[int, ...] | None = None
    weighted: np.ndarray | None = None
    for name in ("random", "score", "development"):
        seeds, scores = pair_scores(results[name])
        if shared_seeds is None:
            shared_seeds = seeds
            weighted = np.zeros_like(scores)
        elif seeds != shared_seeds:
            raise ValueError("formal DEV heuristic panels do not share ordered seeds")
        weighted += COMPOSITE_WEIGHTS[name] * scores
    if shared_seeds is None or weighted is None:
        raise AssertionError("formal DEV panel has no heuristic results")
    return shared_seeds, weighted


def _serialized_weighted_pair_scores(
    path: Path,
) -> tuple[tuple[int, ...], np.ndarray]:
    from .paired_statistics import serialized_pair_scores

    payload = json.loads(path.read_text(encoding="utf-8"))["evaluations"]
    shared_seeds: tuple[int, ...] | None = None
    weighted: np.ndarray | None = None
    for name in ("random", "score", "development"):
        seeds, scores = serialized_pair_scores(payload[name])
        if shared_seeds is None:
            shared_seeds = seeds
            weighted = np.zeros_like(scores)
        elif seeds != shared_seeds:
            raise ValueError("stored formal DEV heuristic panels do not share seeds")
        weighted += COMPOSITE_WEIGHTS[name] * scores
    if shared_seeds is None or weighted is None:
        raise AssertionError("stored formal DEV panel has no heuristic results")
    return shared_seeds, weighted


def _best_dev_decision(
    progress: FiveCProgress,
    results: dict[str, EvaluationResult],
    composite: float,
    *,
    run_local_step: int,
) -> tuple[bool, dict[str, Any]]:
    if progress.best_heuristic_composite is None:
        return True, {"reason": "initial formal DEV baseline", "delta": None, "ci": None}
    delta = composite - progress.best_heuristic_composite
    if delta >= _BEST_DIRECT_THRESHOLD:
        return True, {"reason": "composite improved by at least 0.01", "delta": delta, "ci": None}
    if delta <= 0.0:
        return False, {"reason": "composite did not improve", "delta": delta, "ci": None}
    if progress.best_formal_eval_path is None:
        raise AssertionError("best DEV score has no stored formal panel")
    candidate_seeds, candidate_scores = _weighted_pair_scores(results)
    best_seeds, best_scores = _serialized_weighted_pair_scores(
        Path(progress.best_formal_eval_path)
    )
    interval = paired_bootstrap_difference(
        candidate_seeds,
        candidate_scores,
        best_seeds,
        best_scores,
        resamples=_BEST_BOOTSTRAP_RESAMPLES,
        seed=5_000_000_000 + run_local_step,
    )
    selected = interval.lower > 0.0
    return selected, {
        "reason": (
            "close composite improvement confirmed by paired bootstrap"
            if selected
            else "close composite improvement not confirmed by paired bootstrap"
        ),
        "delta": delta,
        "ci": interval.to_dict(),
    }


def _run_local_step(trainer: PPOTrainer, progress: FiveCProgress) -> int:
    return trainer.global_learning_decision_step - progress.source_lifetime_step


def _checkpoint_name(run_local_step: int, lifetime_step: int) -> str:
    return f"local_{run_local_step:09d}_lifetime_{lifetime_step:09d}.pt"


def _checkpoint_metadata(
    config: FiveCConfig,
    progress: FiveCProgress,
    collector: BatchedRolloutCollector | MultiprocessRolloutCollector,
    selector: FiveCOpponentSelector,
    pool_manifest: Path,
    runtime: dict[str, Any],
    trainer: PPOTrainer,
) -> dict[str, Any]:
    if collector.active_environment_count:
        raise ValueError("5C checkpoints require a complete-episode collector boundary")
    run_local_step = _run_local_step(trainer, progress)
    expected_lr = scheduled_learning_rate(config, run_local_step)
    expected_entropy = scheduled_entropy_coef(config, run_local_step)
    if not math.isclose(trainer.config.learning_rate, expected_lr, abs_tol=1e-15):
        raise ValueError("5C checkpoint learning-rate schedule is not current")
    if not math.isclose(trainer.config.entropy_coef, expected_entropy, abs_tol=1e-15):
        raise ValueError("5C checkpoint entropy schedule is not current")
    collector_snapshot: dict[str, Any]
    if isinstance(collector, MultiprocessRolloutCollector):
        collector_snapshot = asdict(collector.snapshot())
    else:
        collector_snapshot = {
            "kind": "single-process-counter-derived-episode-streams",
            "next_episode_index": collector.next_episode_index,
            "active_policy_rng_states": [],
        }
    return {
        "run_format_version": FIVE_C_RUN_FORMAT_VERSION,
        "stage": FIVE_C_STAGE,
        "training_config": asdict(config),
        "progress": progress.to_dict(),
        "source_lifetime_step": progress.source_lifetime_step,
        "run_local_step": run_local_step,
        "lifetime_step": trainer.global_learning_decision_step,
        "run_local_update_count": trainer.update_count - progress.source_update_count,
        "lifetime_update_count": trainer.update_count,
        "schedule": {
            "position_run_local_step": run_local_step,
            "learning_rate": expected_lr,
            "entropy_coef": expected_entropy,
        },
        "next_episode_index": collector.next_episode_index,
        "collector_at_episode_boundary": True,
        "collector_rng": {
            "kind": (
                "spawn-multiprocessing-counter-derived-episode-streams"
                if isinstance(collector, MultiprocessRolloutCollector)
                else "counter-derived episode streams"
            ),
            "next_episode_index": collector.next_episode_index,
            "active_policy_rng_states": [],
            "multiprocess_snapshot": (
                collector_snapshot
                if isinstance(collector, MultiprocessRolloutCollector)
                else None
            ),
        },
        "mixture_rng": asdict(selector.snapshot),
        "pool_manifest": str(pool_manifest.resolve()),
        "neural_opponent_action_mode": "sampled_policy",
        "runtime": runtime,
        "rules_version": RULESET_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "data_fingerprint": load_default_catalog().data_fingerprint,
    }


def _save_checkpoint(
    *,
    trainer: PPOTrainer,
    sampler: PolicySampler,
    collector: BatchedRolloutCollector | MultiprocessRolloutCollector,
    selector: FiveCOpponentSelector,
    pool: OpponentPool,
    config: FiveCConfig,
    progress: FiveCProgress,
    runtime: dict[str, Any],
    run_root: Path,
    append_manifest: bool = True,
) -> Path:
    run_local_step = _run_local_step(trainer, progress)
    lifetime_step = trainer.global_learning_decision_step
    pool_manifest = (
        run_root
        / "historical_pool"
        / f"local_{run_local_step:09d}_lifetime_{lifetime_step:09d}.json"
    )
    pool.save_manifest(pool_manifest)
    path = run_root / "checkpoints" / _checkpoint_name(run_local_step, lifetime_step)
    trainer.save_checkpoint(
        path,
        sampler=sampler,
        metadata=_checkpoint_metadata(
            config,
            progress,
            collector,
            selector,
            pool_manifest,
            runtime,
            trainer,
        ),
    )
    latest = {
        "run_local_step": run_local_step,
        "lifetime_step": lifetime_step,
        "checkpoint": str(path.resolve()),
        "pool_manifest": str(pool_manifest.resolve()),
        "saved_at_utc": _utc_now(),
    }
    _write_json(run_root / "checkpoints" / "latest.json", latest)
    if append_manifest:
        with (run_root / "checkpoints" / "manifest.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(latest, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _drain(
    collector: BatchedRolloutCollector | MultiprocessRolloutCollector,
    rollout: RolloutBatch,
) -> RolloutBatch:
    drained = collector.drain_active_environments()
    return group_rollout_by_episode((*rollout.samples, *drained.samples))


def _save_5c_truncation(
    run_root: Path,
    environment: LearningEnvironment | CompletedLearningEpisode,
) -> None:
    if isinstance(environment, CompletedLearningEpisode):
        _write_json(
            run_root / "logs" / f"{environment.episode_id}-truncation.json",
            {
                "episode_id": environment.episode_id,
                "learning_player": environment.learning_player,
                "state": environment.final_state,
                "replay": to_primitive(environment.replay_record),
            },
        )
    else:
        _save_truncation(run_root, environment)


def _opponent_training_breakdown(
    rollout_metrics: dict[str, Any], contexts: dict[str, _EpisodeOpponent]
) -> dict[str, float]:
    rewards: dict[str, list[float]] = defaultdict(list)
    for episode in rollout_metrics["_episodes"]:
        if not episode["truncated"]:
            rewards[contexts[episode["episode_id"]].kind].append(
                float(episode["reward"])
            )
    output: dict[str, float] = {}
    for kind in _OPPONENT_KINDS:
        values = rewards[kind]
        output[f"opponent_{kind}_train_score"] = (
            sum(1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in values)
            / len(values)
            if values
            else 0.0
        )
    return output


def _write_status(
    run_root: Path,
    *,
    status: str,
    trainer: PPOTrainer,
    progress: FiveCProgress,
    error: BaseException | None = None,
    phase: str | None = None,
    phase_detail: dict[str, Any] | None = None,
) -> None:
    _write_json(
        run_root / "status.json",
        {
            "status": status,
            "run_local_step": _run_local_step(trainer, progress),
            "target_run_local_step": RUN_LOCAL_TARGET,
            "lifetime_step": trainer.global_learning_decision_step,
            "last_quick_eval": (
                progress.quick_evaluation_steps[-1]
                if progress.quick_evaluation_steps
                else None
            ),
            "last_formal_eval": (
                progress.evaluation_steps[-1] if progress.evaluation_steps else None
            ),
            "best_composite": progress.best_heuristic_composite,
            "best_checkpoint": progress.best_checkpoint,
            "nan_or_inf": int(isinstance(error, FloatingPointError)),
            "engineering_truncations": progress.training.engineering_truncations,
            "phase": phase,
            "phase_detail": phase_detail,
            "error_type": None if error is None else type(error).__name__,
            "error_detail": None if error is None else str(error),
            "updated_at_utc": _utc_now(),
        },
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _maybe_add_historical_best(
    *,
    selected: bool,
    checkpoint_path: Path,
    run_local_step: int,
    lifetime_step: int,
    config: FiveCConfig,
    progress: FiveCProgress,
    pool: OpponentPool,
    run_root: Path,
) -> bool:
    event: dict[str, Any] = {
        "considered_at_run_local_step": run_local_step,
        "lifetime_step": lifetime_step,
        "checkpoint": str(checkpoint_path.resolve()),
        "selected_as_best_dev": selected,
        "decision": "not_added",
        "reason": "candidate was not selected as best-dev",
    }
    if selected and run_local_step > 0:
        gap = run_local_step - progress.last_historical_add_local_step
        if gap < config.dynamic_history_min_interval:
            event["reason"] = "minimum 250k run-local spacing not reached"
        else:
            if len(pool.historical_entries) >= pool.max_historical:
                if not progress.dynamic_historical_ids:
                    event["reason"] = "pool full and all entries are protected initial snapshots"
                    _append_jsonl(run_root / "historical_pool" / "events.jsonl", event)
                    return False
                removed_id = progress.dynamic_historical_ids.pop(0)
                removed = pool.remove_historical(removed_id)
                event["removed"] = asdict(removed)
            entry = pool.add_historical(
                checkpoint_path,
                reason="5C formal best-dev snapshot",
                added_at_step=lifetime_step,
            )
            progress.dynamic_historical_ids.append(entry.opponent_id)
            progress.historical_games.setdefault(entry.opponent_id, 0)
            progress.last_historical_add_local_step = run_local_step
            event["decision"] = "added"
            event["reason"] = "new formal best-dev with required spacing"
            event["entry"] = asdict(entry)
            _append_jsonl(run_root / "historical_pool" / "events.jsonl", event)
            return True
    _append_jsonl(run_root / "historical_pool" / "events.jsonl", event)
    return False


def _next_multiple_after(current: int, interval: int) -> int:
    return (current // interval + 1) * interval


def _initial_formal_dev_required(
    trainer: PPOTrainer,
    progress: FiveCProgress,
) -> bool:
    """A step-0 checkpoint is not initialized until its formal baseline is saved."""

    return (
        _run_local_step(trainer, progress) == 0
        and 0 not in progress.evaluation_steps
        and progress.best_heuristic_composite is None
        and progress.best_checkpoint is None
    )


def run_five_c_training(
    config: FiveCConfig,
    *,
    run_root: Path,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    run_root = run_root.resolve()
    stored = load_five_c_config(run_root / "config.json")
    if stored != config:
        raise ValueError("in-memory 5C config disagrees with frozen config.json")
    runtime = runtime_metadata(config.device)

    if resume_checkpoint is None:
        if (run_root / "checkpoints" / "latest.json").exists():
            raise FileExistsError("5C run already started; use the resume command")
        trainer, sampler, source_metadata = PPOTrainer.load_checkpoint(
            Path(config.source_checkpoint), device=config.device
        )
        if source_metadata.get("rules_version") != RULESET_ID:
            raise ValueError("5C source checkpoint rules version mismatch")
        _configure_trainer(trainer, config)
        progress = FiveCProgress(
            source_checkpoint=str(Path(config.source_checkpoint).resolve()),
            source_lifetime_step=trainer.global_learning_decision_step,
            source_update_count=trainer.update_count,
        )
        selector = FiveCOpponentSelector(
            _MIXTURE_RNG_OFFSET + config.training_seed
        )
        initial_manifest = run_root / "historical_pool" / "initial.json"
        pool = OpponentPool.load_manifest(initial_manifest, device=config.device)
        progress.historical_games = {
            entry.opponent_id: 0 for entry in pool.historical_entries
        }
        start_episode_index = 0
        restored_worker_states = None
        previous_elapsed = 0.0
    else:
        trainer, sampler, metadata = PPOTrainer.load_checkpoint(
            resume_checkpoint, device=config.device
        )
        if metadata.get("run_format_version") != FIVE_C_RUN_FORMAT_VERSION:
            raise ValueError("resume checkpoint is not a 5C run")
        if metadata.get("stage") != FIVE_C_STAGE:
            raise ValueError("resume checkpoint has the wrong 5C stage")
        saved_values = dict(metadata["training_config"])
        saved_values["historical_checkpoints"] = tuple(
            saved_values["historical_checkpoints"]
        )
        saved_values["historical_eval_steps"] = tuple(
            saved_values["historical_eval_steps"]
        )
        if FiveCConfig(**saved_values) != config:
            raise ValueError("resume configuration changes a frozen 5C setting")
        if not metadata.get("collector_at_episode_boundary"):
            raise ValueError("5C resume checkpoint is not at an episode boundary")
        progress = FiveCProgress.from_dict(metadata["progress"])
        if trainer.global_learning_decision_step != int(metadata["lifetime_step"]):
            raise ValueError("5C checkpoint lifetime step metadata mismatch")
        local_step = _run_local_step(trainer, progress)
        if local_step != int(metadata["run_local_step"]):
            raise ValueError("5C checkpoint run-local step metadata mismatch")
        expected_lr = scheduled_learning_rate(config, local_step)
        expected_entropy = scheduled_entropy_coef(config, local_step)
        if not math.isclose(trainer.config.learning_rate, expected_lr, abs_tol=1e-15):
            raise ValueError("5C resume learning-rate schedule discontinuity")
        if not math.isclose(trainer.config.entropy_coef, expected_entropy, abs_tol=1e-15):
            raise ValueError("5C resume entropy schedule discontinuity")
        if any(
            not math.isclose(group["lr"], expected_lr, abs_tol=1e-15)
            for group in trainer.optimizer.param_groups
        ):
            raise ValueError("5C resume Adam learning rate disagrees with schedule")
        selector = FiveCOpponentSelector.restore(RngState(**metadata["mixture_rng"]))
        pool = OpponentPool.load_manifest(
            Path(metadata["pool_manifest"]), device=config.device
        )
        start_episode_index = int(metadata["next_episode_index"])
        restored_worker_states = restored_worker_rng_states(
            metadata.get("collector_rng", {}).get("multiprocess_snapshot"),
            num_workers=config.num_workers,
            envs_per_worker=config.envs_per_worker,
        )
        progress.resume_count += 1
        previous_elapsed = progress.elapsed_seconds

    evaluation_opponents_quick = build_five_c_eval_opponents(
        config, device=config.device, include_historical=False
    )
    evaluation_opponents_formal = build_five_c_eval_opponents(
        config, device=config.device, include_historical=True
    )
    opponent_timing = NeuralInferenceTiming()
    episode_opponents: dict[str, _EpisodeOpponent] = {}
    selection_local_step = _run_local_step(trainer, progress)

    def episode_choice(index: int) -> tuple[Any, OpponentChoice, str]:
        plan = episode_plan(config.training_seed, index)
        kind = selector.choose_kind(selection_local_step)
        choice: OpponentChoice = pool.choose_for_training(
            kind,
            policy_seed=plan.opponent_seed,
            timing=opponent_timing,
        )
        episode_id = f"5c-{config.training_seed}-{index:09d}"
        episode_opponents[episode_id] = _EpisodeOpponent(
            kind=kind,
            opponent_id=choice.entry.opponent_id,
        )
        return plan, choice, episode_id

    def environment_factory(index: int) -> tuple[LearningEnvironment, int]:
        plan, choice, episode_id = episode_choice(index)
        return (
            LearningEnvironment.new(
                plan.game_seed,
                first_player=plan.first_player,
                learning_player=plan.learning_player,
                opponent_policy=choice.policy,
                episode_id=episode_id,
                max_completed_turns=config.max_completed_turns,
                failure_dir=run_root / "failures" / f"episode-{index:09d}",
            ),
            plan.learning_policy_seed,
        )

    def multiprocess_environment_factory(index: int) -> MultiprocessEpisodeLaunch:
        plan, choice, episode_id = episode_choice(index)
        model = getattr(choice.policy, "model", None)
        return MultiprocessEpisodeLaunch(
            spec=MultiprocessEpisodeSpec(
                episode_index=index,
                episode_id=episode_id,
                game_seed=plan.game_seed,
                first_player=plan.first_player,
                learning_player=plan.learning_player,
                learning_policy_seed=plan.learning_policy_seed,
                opponent_seed=plan.opponent_seed,
                opponent_kind=choice.entry.kind,
                opponent_id=choice.entry.opponent_id,
                max_completed_turns=config.max_completed_turns,
                failure_dir=str(run_root / "failures" / f"episode-{index:09d}"),
            ),
            neural_opponent_model=model,
        )

    collector: BatchedRolloutCollector | MultiprocessRolloutCollector
    if config.collector_backend == "multiprocessing":
        collector = MultiprocessRolloutCollector(
            sampler,
            multiprocess_environment_factory,
            num_workers=config.num_workers,
            envs_per_worker=config.envs_per_worker,
            master_seed=config.training_seed,
            start_episode_index=start_episode_index,
            restored_worker_rng_states=restored_worker_states,
        )
    else:
        collector = BatchedRolloutCollector(
            sampler,
            environment_factory,
            active_envs=config.active_envs,
            start_episode_index=start_episode_index,
        )
    metrics = FiveCMetricWriter(run_root)
    tensorboard = SummaryWriter(
        log_dir=str(run_root / "tensorboard"),
        purge_step=(
            _run_local_step(trainer, progress) + 1
            if resume_checkpoint is not None
            else None
        ),
    )
    started = time.perf_counter()

    def checkpoint_now(*, append_manifest: bool = True) -> Path:
        nonlocal selection_local_step
        selection_local_step = _run_local_step(trainer, progress)
        _apply_schedules(trainer, config, selection_local_step)
        progress.elapsed_seconds = previous_elapsed + (time.perf_counter() - started)
        return _save_checkpoint(
            trainer=trainer,
            sampler=sampler,
            collector=collector,
            selector=selector,
            pool=pool,
            config=config,
            progress=progress,
            runtime=runtime,
            run_root=run_root,
            append_manifest=append_manifest,
        )

    def perform_panel(kind: str, checkpoint_path: Path) -> tuple[dict[str, Any], bool]:
        local = _run_local_step(trainer, progress)
        if kind == "quick":
            pairs = config.quick_eval_pairs
            opponents = evaluation_opponents_quick
        elif kind == "formal":
            pairs = config.formal_eval_pairs
            opponents = evaluation_opponents_formal
        else:
            raise ValueError(f"unknown in-training 5C evaluation kind: {kind}")
        seeds = dev_eval_game_seeds(config.dev_panel_seed, pairs)
        panel_started = time.perf_counter()
        print(
            json.dumps(
                {
                    "event": "evaluation_panel_started",
                    "kind": kind,
                    "run_local_step": local,
                    "opponents": len(opponents),
                    "paired_seeds_per_opponent": pairs,
                    "total_games": len(opponents) * pairs * 2,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _write_status(
            run_root,
            status="running",
            trainer=trainer,
            progress=progress,
            phase=f"{kind}_evaluation",
            phase_detail={
                "completed_opponents": 0,
                "total_opponents": len(opponents),
                "paired_seeds_per_opponent": pairs,
            },
        )

        def evaluation_progress(
            opponent_name: str,
            opponent_index: int,
            opponent_total: int,
            event: str,
            result: EvaluationResult | None,
        ) -> None:
            payload: dict[str, Any] = {
                "event": f"evaluation_opponent_{event}",
                "kind": kind,
                "run_local_step": local,
                "opponent": opponent_name,
                "opponent_index": opponent_index,
                "opponent_total": opponent_total,
                "elapsed_seconds": time.perf_counter() - panel_started,
            }
            if result is not None:
                payload.update(
                    {
                        "score_rate": result.win_rate_score,
                        "valid_games": result.valid_games,
                        "total_games": result.total_games,
                    }
                )
            print(json.dumps(payload, sort_keys=True), flush=True)
            completed = opponent_index if event == "completed" else opponent_index - 1
            _write_status(
                run_root,
                status="running",
                trainer=trainer,
                progress=progress,
                phase=f"{kind}_evaluation",
                phase_detail={
                    "opponent": opponent_name,
                    "opponent_index": opponent_index,
                    "completed_opponents": completed,
                    "total_opponents": opponent_total,
                    "paired_seeds_per_opponent": pairs,
                    "event": event,
                    "elapsed_seconds": time.perf_counter() - panel_started,
                },
            )
            tensorboard.add_scalar(
                f"system/{kind}_evaluation_progress",
                completed / opponent_total,
                local,
            )
            tensorboard.flush()

        def evaluation_pair_progress(
            opponent_name: str,
            opponent_index: int,
            opponent_total: int,
            completed_pairs: int,
            total_pairs: int,
        ) -> None:
            elapsed = time.perf_counter() - panel_started
            print(
                json.dumps(
                    {
                        "event": "evaluation_opponent_progress",
                        "kind": kind,
                        "run_local_step": local,
                        "opponent": opponent_name,
                        "opponent_index": opponent_index,
                        "opponent_total": opponent_total,
                        "completed_pairs": completed_pairs,
                        "total_pairs": total_pairs,
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            _write_status(
                run_root,
                status="running",
                trainer=trainer,
                progress=progress,
                phase=f"{kind}_evaluation",
                phase_detail={
                    "opponent": opponent_name,
                    "opponent_index": opponent_index,
                    "completed_opponents": opponent_index - 1,
                    "total_opponents": opponent_total,
                    "completed_pairs": completed_pairs,
                    "total_pairs": total_pairs,
                    "event": "progress",
                    "elapsed_seconds": elapsed,
                },
            )
            overall_progress = (
                opponent_index - 1 + completed_pairs / total_pairs
            ) / opponent_total
            tensorboard.add_scalar(
                f"system/{kind}_evaluation_progress",
                overall_progress,
                local,
            )
            tensorboard.flush()

        results, elapsed = evaluate_five_c_panel(
            trainer.model,
            opponents=opponents,
            seeds=seeds,
            policy_seed_base=-(
                _DEV_POLICY_RNG_OFFSET + config.dev_panel_seed * 10_000_000
            ),
            device=config.device,
            max_completed_turns=config.max_completed_turns,
            failure_dir=(
                run_root
                / "eval"
                / "failures"
                / f"{kind}-local-{local:09d}"
            ),
            progress_callback=evaluation_progress,
            pair_progress_callback=evaluation_pair_progress,
        )
        record = _panel_record(
            results,
            kind=kind,
            run_local_step=local,
            lifetime_step=trainer.global_learning_decision_step,
            run_local_update_count=trainer.update_count - progress.source_update_count,
            lifetime_update_count=trainer.update_count,
            elapsed_seconds=elapsed,
        )
        path = run_root / "eval" / kind / f"local_{local:09d}_{kind}.json"
        _write_panel(
            path,
            results=results,
            record=record,
            config=config,
            seed_domain="DEV",
            checkpoint_path=checkpoint_path,
        )
        metrics.append(record)
        _tensorboard_panel(tensorboard, record, local)
        for name in results:
            if name.startswith("historical_"):
                tensorboard.add_scalar(
                    f"eval/{name}_score_rate",
                    record[f"{_safe_id(name)}_score_rate"],
                    local,
                )
        selected = False
        if kind == "quick":
            if local not in progress.quick_evaluation_steps:
                progress.quick_evaluation_steps.append(local)
        else:
            composite = float(record["heuristic_composite"])
            old_best = progress.best_checkpoint
            old_score = progress.best_heuristic_composite
            selected, rationale = _best_dev_decision(
                progress,
                results,
                composite,
                run_local_step=local,
            )
            if selected:
                progress.best_heuristic_composite = composite
                progress.best_checkpoint = str(checkpoint_path.resolve())
                progress.best_formal_eval_path = str(path.resolve())
                _write_json(
                    run_root / "checkpoints" / "best_dev.json",
                    {
                        "old_best_checkpoint": old_best,
                        "old_best_composite": old_score,
                        "new_best_checkpoint": str(checkpoint_path.resolve()),
                        "new_best_composite": composite,
                        "run_local_step": local,
                        "lifetime_step": trainer.global_learning_decision_step,
                        "formal_eval_path": str(path.resolve()),
                        "eval_seed_set": {
                            "version": "stage1-compatible-dev-v1",
                            "panel_seed": config.dev_panel_seed,
                            "paired_seeds": config.formal_eval_pairs,
                        },
                        "panel_summary": record,
                        "selection": rationale,
                    },
                )
                _append_jsonl(
                    run_root / "checkpoints" / "best_dev_history.jsonl",
                    json.loads(
                        (run_root / "checkpoints" / "best_dev.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )
            else:
                _append_jsonl(
                    run_root / "checkpoints" / "best_dev_history.jsonl",
                    {
                        "old_best_checkpoint": old_best,
                        "old_best_composite": old_score,
                        "candidate_checkpoint": str(checkpoint_path.resolve()),
                        "candidate_composite": composite,
                        "run_local_step": local,
                        "lifetime_step": trainer.global_learning_decision_step,
                        "formal_eval_path": str(path.resolve()),
                        "selection": rationale,
                    },
                )
            if local not in progress.evaluation_steps:
                progress.evaluation_steps.append(local)
        tensorboard.flush()
        print(
            json.dumps(
                {
                    "event": f"{kind}_dev",
                    "run_local_step": local,
                    "lifetime_step": trainer.global_learning_decision_step,
                    "heuristic_composite": record["heuristic_composite"],
                    "selected_best_dev": selected,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return record, selected

    try:
        _write_status(
            run_root,
            status="running",
            trainer=trainer,
            progress=progress,
            phase="startup",
            phase_detail={"resume": resume_checkpoint is not None},
        )
        print(
            json.dumps(
                {
                    "event": "training_process_started",
                    "resume": resume_checkpoint is not None,
                    "run_local_step": _run_local_step(trainer, progress),
                    "lifetime_step": trainer.global_learning_decision_step,
                    "device": config.device,
                    "collector_backend": config.collector_backend,
                    "active_envs": config.active_envs,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        tensorboard.add_scalar(
            "system/process_alive",
            1.0,
            _run_local_step(trainer, progress),
        )
        tensorboard.add_text(
            "run/status",
            "startup" if resume_checkpoint is None else "resume startup",
            _run_local_step(trainer, progress),
        )
        tensorboard.flush()
        if _initial_formal_dev_required(trainer, progress):
            pool.set_latest(
                trainer.model, training_step=trainer.global_learning_decision_step
            )
            initial_path = checkpoint_now(
                append_manifest=resume_checkpoint is None
            )
            print(
                json.dumps(
                    {
                        "event": "initial_formal_dev_required",
                        "resume": resume_checkpoint is not None,
                        "checkpoint": str(initial_path),
                        "run_local_step": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            _, selected = perform_panel("formal", initial_path)
            _maybe_add_historical_best(
                selected=selected,
                checkpoint_path=initial_path,
                run_local_step=0,
                lifetime_step=trainer.global_learning_decision_step,
                config=config,
                progress=progress,
                pool=pool,
                run_root=run_root,
            )
            checkpoint_now(append_manifest=False)
            print(
                json.dumps(
                    {
                        "event": "initial_formal_dev_complete",
                        "run_local_step": 0,
                        "best_composite": progress.best_heuristic_composite,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        local = _run_local_step(trainer, progress)
        next_checkpoint = _next_multiple_after(local, config.checkpoint_interval)
        next_quick = _next_multiple_after(local, config.quick_eval_interval)
        next_formal = _next_multiple_after(local, config.formal_eval_interval)
        phase_boundary = PHASE_B_START if local < PHASE_B_START else RUN_LOCAL_TARGET

        while local < config.run_local_learning_decisions:
            selection_local_step = local
            snapshot_started = time.perf_counter()
            latest_entry = pool.set_latest(
                trainer.model, training_step=trainer.global_learning_decision_step
            )
            latest_snapshot_seconds = time.perf_counter() - snapshot_started
            latest_model = pool.policy_for(latest_entry.opponent_id).policy.model
            if any(
                learner.data_ptr() == opponent.data_ptr()
                for learner, opponent in zip(
                    trainer.model.parameters(), latest_model.parameters(), strict=True
                )
            ):
                raise AssertionError("5C latest opponent shares learner parameter storage")
            if any(parameter.requires_grad for parameter in latest_model.parameters()):
                raise AssertionError("5C latest opponent unexpectedly requires gradients")
            optimizer_ids = {
                id(parameter)
                for group in trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            if any(id(parameter) in optimizer_ids for parameter in latest_model.parameters()):
                raise AssertionError("5C latest opponent appears in learner optimizer")
            latest_before = tuple(
                parameter.detach().clone() for parameter in latest_model.parameters()
            )

            next_event = min(
                next_checkpoint,
                next_quick,
                next_formal,
                phase_boundary,
                config.run_local_learning_decisions,
            )
            target = min(config.rollout_steps, next_event - local)
            if target <= 0:
                raise AssertionError("5C scheduler produced a non-positive rollout")
            boundary_due = local + target >= next_event
            update_lr, update_entropy = _apply_schedules(trainer, config, local)
            sampler.reset_timing()
            opponent_timing.reset()
            worker_statistics_before = (
                collector.worker_statistics()
                if isinstance(collector, MultiprocessRolloutCollector)
                else ()
            )
            if isinstance(collector, MultiprocessRolloutCollector):
                collector.reset_timing()
            if config.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rollout_started = time.perf_counter()
            rollout = collector.collect(target)
            boundary_drain_decisions = 0
            if boundary_due:
                drained = collector.drain_active_environments()
                boundary_drain_decisions = len(drained)
            if config.device == "cuda":
                torch.cuda.synchronize()
            rollout_seconds = time.perf_counter() - rollout_started
            worker_statistics_after = (
                collector.worker_statistics()
                if isinstance(collector, MultiprocessRolloutCollector)
                else ()
            )
            completed = collector.take_completed_environments()
            completed_contexts = {
                environment.episode_id: episode_opponents[environment.episode_id]
                for environment in completed
            }
            rollout_metrics = summarize_rollout(rollout, completed)
            rollout_metrics.update(
                _opponent_training_breakdown(rollout_metrics, completed_contexts)
            )
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
            for opponent_id, count in historical_counts.items():
                progress.historical_games[opponent_id] = (
                    progress.historical_games.get(opponent_id, 0) + count
                )

            gae_started = time.perf_counter()
            prepared = compute_gae(
                rollout,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
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
                raise AssertionError("5C optimizer changed frozen latest opponent")
            _update_progress(progress.training, rollout_metrics)
            for environment in completed:
                episode_opponents.pop(environment.episode_id)

            new_local = _run_local_step(trainer, progress)
            next_lr, next_entropy = _apply_schedules(trainer, config, new_local)
            if isinstance(collector, MultiprocessRolloutCollector):
                environment_seconds = sum(
                    after.environment_encoding_seconds
                    - before.environment_encoding_seconds
                    for before, after in zip(
                        worker_statistics_before,
                        worker_statistics_after,
                        strict=True,
                    )
                )
                neural_opponent_seconds = collector.opponent_inference_seconds
                neural_opponent_decisions = collector.opponent_inference_items
                ipc_wait_seconds = collector.ipc_wait_seconds
                worker_cpu_seconds = sum(
                    after.cpu_seconds - before.cpu_seconds
                    for before, after in zip(
                        worker_statistics_before,
                        worker_statistics_after,
                        strict=True,
                    )
                )
            else:
                environment_seconds = max(
                    0.0,
                    rollout_seconds
                    - sampler.inference_seconds
                    - opponent_timing.seconds,
                )
                neural_opponent_seconds = opponent_timing.seconds
                neural_opponent_decisions = opponent_timing.decisions
                ipc_wait_seconds = 0.0
                worker_cpu_seconds = 0.0
            record = {
                "record_type": "update",
                "run_local_step": new_local,
                "lifetime_step": trainer.global_learning_decision_step,
                "run_local_update_count": trainer.update_count - progress.source_update_count,
                "lifetime_update_count": trainer.update_count,
                "wall_seconds": previous_elapsed + (time.perf_counter() - started),
                "policy_loss": update_stats.policy_loss,
                "value_loss": update_stats.value_loss,
                "total_loss": update_stats.total_loss,
                "entropy": update_stats.entropy,
                "entropy_coef": update_entropy,
                "next_entropy_coef": next_entropy,
                "explained_variance": update_stats.explained_variance,
                "approx_kl": update_stats.approx_kl,
                "clip_fraction": update_stats.clip_fraction,
                "grad_norm": update_stats.grad_norm,
                "learning_rate": update_lr,
                "next_learning_rate": next_lr,
                "epochs_completed": update_stats.epochs_completed,
                "minibatches_processed": update_stats.minibatches_processed,
                "gradient_clipped_minibatches": update_stats.gradient_clipped_minibatches,
                "kl_early_stopped": update_stats.early_stopped,
                "opponent_phase": "A" if local < PHASE_B_START else "B",
                "learning_decisions_per_second": len(rollout) / rollout_seconds,
                "games_per_second": len(completed) / rollout_seconds,
                "rollout_seconds": rollout_seconds,
                "environment_encoding_seconds": environment_seconds,
                "model_inference_seconds": sampler.inference_seconds,
                "neural_opponent_inference_seconds": neural_opponent_seconds,
                "neural_opponent_decisions": neural_opponent_decisions,
                "mean_inference_batch_size": sampler.mean_inference_batch_size,
                "mean_opponent_inference_batch_size": (
                    collector.mean_opponent_inference_batch_size
                    if isinstance(collector, MultiprocessRolloutCollector)
                    else 1.0 if opponent_timing.decisions else 0.0
                ),
                "ipc_wait_seconds": ipc_wait_seconds,
                "worker_cpu_seconds": worker_cpu_seconds,
                "num_workers": (
                    config.num_workers
                    if isinstance(collector, MultiprocessRolloutCollector)
                    else 0
                ),
                "envs_per_worker": (
                    config.envs_per_worker
                    if isinstance(collector, MultiprocessRolloutCollector)
                    else config.active_envs
                ),
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
                "boundary_drain_decisions_not_trained": boundary_drain_decisions,
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
                update_lr,
                update_entropy,
            ):
                if not math.isfinite(value):
                    raise FloatingPointError("non-finite 5C training metric")
            metrics.append(record)
            for key in (
                "policy_loss",
                "value_loss",
                "total_loss",
                "entropy",
                "entropy_coef",
                "explained_variance",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
                "learning_rate",
            ):
                tensorboard.add_scalar(f"ppo/{key}", record[key], new_local)
            _tensorboard_rollout(tensorboard, record, new_local)
            tensorboard.add_scalar(
                "rollout/mean_candidates",
                record["mean_valid_candidates"],
                new_local,
            )
            tensorboard.add_scalar(
                "behavior/mean_turns",
                record["mean_completed_player_turns"],
                new_local,
            )
            for kind in _OPPONENT_KINDS:
                tensorboard.add_scalar(
                    f"opponent/{kind}_fraction",
                    record[f"opponent_{kind}_fraction"],
                    new_local,
                )
            for tag, key in (
                ("decisions_per_sec", "learning_decisions_per_second"),
                ("games_per_sec", "games_per_second"),
                ("mean_inference_batch_size", "mean_inference_batch_size"),
                (
                    "mean_opponent_inference_batch_size",
                    "mean_opponent_inference_batch_size",
                ),
                ("env_encoding_seconds", "environment_encoding_seconds"),
                ("learner_inference_seconds", "model_inference_seconds"),
                ("neural_opponent_inference_seconds", "neural_opponent_inference_seconds"),
                ("ipc_wait_seconds", "ipc_wait_seconds"),
                ("worker_cpu_seconds", "worker_cpu_seconds"),
                ("ppo_update_seconds", "ppo_update_seconds"),
                ("cuda_peak_memory_mb", "cuda_peak_allocated_mib"),
                (
                    "boundary_drain_decisions_not_trained",
                    "boundary_drain_decisions_not_trained",
                ),
            ):
                tensorboard.add_scalar(f"system/{tag}", record[key], new_local)
            tensorboard.flush()

            truncated = [environment for environment in completed if environment.truncated]
            for environment in truncated:
                _save_5c_truncation(run_root, environment)
            if truncated:
                raise RuntimeError(
                    "engineering truncation encountered; replay/state saved for investigation"
                )

            local = new_local
            checkpoint_due = local >= next_checkpoint
            formal_due = local >= next_formal
            quick_due = local >= next_quick and not formal_due
            phase_due = local >= phase_boundary
            final_due = local >= config.run_local_learning_decisions
            if boundary_due and collector.active_environment_count:
                raise AssertionError("5C event boundary retained active environments")
            checkpoint_path: Path | None = None
            if checkpoint_due or formal_due or quick_due or phase_due or final_due:
                checkpoint_path = checkpoint_now()
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "run_local_step": local,
                            "lifetime_step": trainer.global_learning_decision_step,
                            "checkpoint": str(checkpoint_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            selected = False
            if formal_due:
                if checkpoint_path is None:
                    raise AssertionError("formal DEV has no candidate checkpoint")
                _, selected = perform_panel("formal", checkpoint_path)
                _maybe_add_historical_best(
                    selected=selected,
                    checkpoint_path=checkpoint_path,
                    run_local_step=local,
                    lifetime_step=trainer.global_learning_decision_step,
                    config=config,
                    progress=progress,
                    pool=pool,
                    run_root=run_root,
                )
                checkpoint_now(append_manifest=False)
                while next_formal <= local:
                    next_formal += config.formal_eval_interval
            elif quick_due:
                if checkpoint_path is None:
                    raise AssertionError("quick DEV has no candidate checkpoint")
                perform_panel("quick", checkpoint_path)
                checkpoint_now(append_manifest=False)
            while next_quick <= local:
                next_quick += config.quick_eval_interval
            while next_checkpoint <= local:
                next_checkpoint += config.checkpoint_interval
            if phase_due and phase_boundary == PHASE_B_START:
                phase_boundary = RUN_LOCAL_TARGET
                print(
                    json.dumps(
                        {
                            "event": "opponent_phase_change",
                            "phase": "B",
                            "run_local_step": local,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _write_status(
                run_root,
                status="running",
                trainer=trainer,
                progress=progress,
                phase="training",
                phase_detail={
                    "run_local_update_count": (
                        trainer.update_count - progress.source_update_count
                    )
                },
            )

        latest_manifest = json.loads(
            (run_root / "checkpoints" / "latest.json").read_text(encoding="utf-8")
        )
        if int(latest_manifest["lifetime_step"]) == trainer.global_learning_decision_step:
            final_path = Path(latest_manifest["checkpoint"])
        else:
            final_path = checkpoint_now()
        progress.elapsed_seconds = previous_elapsed + (time.perf_counter() - started)
        summary = {
            "run_format_version": FIVE_C_RUN_FORMAT_VERSION,
            "stage": FIVE_C_STAGE,
            "training_config": asdict(config),
            "ppo_config": asdict(trainer.config),
            "runtime": runtime,
            "requested_run_local_decisions": config.run_local_learning_decisions,
            "actual_run_local_decisions": _run_local_step(trainer, progress),
            "lifetime_learning_decisions": trainer.global_learning_decision_step,
            "run_local_ppo_updates": trainer.update_count - progress.source_update_count,
            "lifetime_ppo_updates": trainer.update_count,
            "latest_checkpoint": str(final_path.resolve()),
            "best_dev_checkpoint": progress.best_checkpoint,
            "progress": progress.to_dict(),
        }
        _write_json(run_root / "summary.json", summary)
        _write_status(
            run_root,
            status="complete",
            trainer=trainer,
            progress=progress,
            phase="complete",
        )
        print(json.dumps({"event": "complete", **summary}, sort_keys=True), flush=True)
        return summary
    except BaseException as caught:
        _write_status(
            run_root,
            status="failed",
            trainer=trainer,
            progress=progress,
            error=caught,
            phase="failed",
        )
        raise
    finally:
        collector.close() if isinstance(collector, MultiprocessRolloutCollector) else None
        tensorboard.flush()
        tensorboard.close()


def run_five_c_preflight(
    config: FiveCConfig,
    *,
    run_root: Path,
    device: str,
) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    run_root = run_root.resolve()
    if load_five_c_config(run_root / "config.json") != config:
        raise ValueError("preflight config disagrees with frozen config.json")
    preflight_runtime = runtime_metadata(device)
    source_validation = _validate_checkpoint_set(config)
    trainer, sampler, metadata = PPOTrainer.load_checkpoint(
        Path(config.source_checkpoint), device=device
    )
    if metadata.get("rules_version") != RULESET_ID:
        raise ValueError("preflight source rules version mismatch")
    _configure_trainer(trainer, config)
    _apply_schedules(trainer, config, 0)
    pool = OpponentPool.load_manifest(
        run_root / "historical_pool" / "initial.json", device=device
    )
    latest = pool.set_latest(trainer.model, training_step=trainer.global_learning_decision_step)
    latest_model = pool.policy_for(latest.opponent_id).policy.model
    if any(
        learner.data_ptr() == opponent.data_ptr()
        for learner, opponent in zip(
            trainer.model.parameters(), latest_model.parameters(), strict=True
        )
    ):
        raise AssertionError("preflight latest snapshot shares learner storage")
    before = tuple(parameter.detach().clone() for parameter in latest_model.parameters())
    timing = NeuralInferenceTiming()

    def factory(index: int) -> tuple[LearningEnvironment, int]:
        kind = "latest" if index % 2 == 0 else "historical"
        opponent_seed = _PREFLIGHT_POLICY_RNG_OFFSET + index * 2 + 1
        choice = pool.choose_for_training(
            kind, policy_seed=opponent_seed, timing=timing
        )
        return (
            LearningEnvironment.new(
                _PREFLIGHT_GAME_RNG_OFFSET + index,
                first_player=index % 2,
                learning_player=(index // 2) % 2,
                opponent_policy=choice.policy,
                episode_id=f"5c-preflight-{index:04d}",
                max_completed_turns=config.max_completed_turns,
                failure_dir=run_root / "failures" / f"preflight-{index:04d}",
            ),
            _PREFLIGHT_POLICY_RNG_OFFSET + index * 2,
        )

    collector = BatchedRolloutCollector(
        sampler,
        factory,
        active_envs=config.active_envs,
    )
    started = time.perf_counter()
    rollout = collector.collect(64)
    rollout = _drain(collector, rollout)
    completed = collector.take_completed_environments()
    if collector.active_environment_count:
        raise AssertionError("preflight collector did not drain")
    if len(completed) < config.active_envs:
        raise AssertionError("preflight did not complete all initial logical environments")
    if any(environment.truncated or not environment.terminated for environment in completed):
        raise RuntimeError("preflight game failed to terminate naturally")
    prepared = compute_gae(
        rollout, gamma=config.gamma, gae_lambda=config.gae_lambda
    )
    trainer.update(prepared)
    if not all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, latest_model.parameters(), strict=True)
    ):
        raise AssertionError("preflight optimizer changed frozen latest opponent")

    schedule_probe_step = 1_234_567
    probe_lr = scheduled_learning_rate(config, schedule_probe_step)
    probe_entropy = scheduled_entropy_coef(config, schedule_probe_step)
    preflight_progress = FiveCProgress(
        source_checkpoint=str(Path(config.source_checkpoint).resolve()),
        source_lifetime_step=source_validation["source_lifetime_step"],
        source_update_count=source_validation["source_update_count"],
        historical_games={entry.opponent_id: 0 for entry in pool.historical_entries},
    )
    preflight_selector = FiveCOpponentSelector(
        _MIXTURE_RNG_OFFSET + config.training_seed
    )
    actual_local_step = _run_local_step(trainer, preflight_progress)
    expected_lr, expected_entropy = _apply_schedules(
        trainer, config, actual_local_step
    )
    with tempfile.TemporaryDirectory(prefix="5c-preflight-", dir=run_root) as temporary:
        temporary_dir = Path(temporary)
        pool_manifest = temporary_dir / "pool.json"
        pool.save_manifest(pool_manifest)
        checkpoint = temporary_dir / "schedule_resume.pt"
        sampler_state = sampler.rng_state()
        trainer.save_checkpoint(
            checkpoint,
            sampler=sampler,
            metadata=_checkpoint_metadata(
                config,
                preflight_progress,
                collector,
                preflight_selector,
                pool_manifest,
                preflight_runtime,
                trainer,
            ),
        )
        restored, restored_sampler, restored_metadata = PPOTrainer.load_checkpoint(
            checkpoint, device=device
        )
        if restored_metadata["run_local_step"] != actual_local_step:
            raise AssertionError("preflight checkpoint lost schedule position")
        if not math.isclose(restored.config.learning_rate, expected_lr, abs_tol=1e-15):
            raise AssertionError("preflight resume reset learning rate")
        if not math.isclose(restored.config.entropy_coef, expected_entropy, abs_tol=1e-15):
            raise AssertionError("preflight resume reset entropy coefficient")
        if not restored_sampler.rng_state().equal(sampler_state):
            raise AssertionError("preflight checkpoint lost sampler RNG state")
        restored_selector = FiveCOpponentSelector.restore(
            RngState(**restored_metadata["mixture_rng"])
        )
        if restored_selector.snapshot != preflight_selector.snapshot:
            raise AssertionError("preflight checkpoint lost mixture RNG state")
        restored_pool = OpponentPool.load_manifest(
            Path(restored_metadata["pool_manifest"]), device=device
        )
        if restored_pool.rng_snapshot != pool.rng_snapshot:
            raise AssertionError("preflight checkpoint lost historical pool RNG")

        tensorboard_dir = temporary_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        writer.add_scalar("preflight/ok", 1.0, 0)
        writer.flush()
        writer.close()
        if not tuple(tensorboard_dir.glob("events.out.tfevents.*")):
            raise AssertionError("preflight TensorBoard writer produced no event file")

    train_seeds = {episode_plan(config.training_seed, index).game_seed for index in range(8)}
    dev_seeds = set(dev_eval_game_seeds(config.dev_panel_seed, 8))
    final_seeds = set(final_eval_game_seeds(config.final_panel_seed, 8))
    preflight_seeds = {_PREFLIGHT_GAME_RNG_OFFSET + index for index in range(8)}
    domains = (train_seeds, dev_seeds, final_seeds, preflight_seeds)
    if any(left & right for i, left in enumerate(domains) for right in domains[i + 1 :]):
        raise AssertionError("TRAIN/DEV/FINAL/preflight seed domains overlap")
    if FiveCOpponentSelector(1).choose_kind(PHASE_B_START - 1) not in PHASE_A_MIX:
        raise AssertionError("preflight Phase A selector failed")
    if FiveCOpponentSelector(1).choose_kind(PHASE_B_START) not in PHASE_B_MIX:
        raise AssertionError("preflight Phase B selector failed")

    result = {
        "status": "PASS",
        "completed_at_utc": _utc_now(),
        "device": device,
        "runtime": preflight_runtime,
        "source_validation": source_validation,
        "active_envs": config.active_envs,
        "rollout_samples": len(rollout),
        "completed_games": len(completed),
        "latest_and_historical_opponents": True,
        "frozen_latest_unchanged": True,
        "schedule_probe": {
            "run_local_step": schedule_probe_step,
            "learning_rate": probe_lr,
            "entropy_coef": probe_entropy,
            "resume_continuous": True,
        },
        "checkpoint_resume": {
            "run_local_step": actual_local_step,
            "learning_rate": expected_lr,
            "entropy_coef": expected_entropy,
            "collector_next_episode_index": collector.next_episode_index,
            "mixture_rng_restored": True,
            "pool_rng_restored": True,
        },
        "event_schedule": {
            "checkpoint": config.checkpoint_interval,
            "quick_dev": config.quick_eval_interval,
            "formal_dev": config.formal_eval_interval,
            "phase_b": config.phase_b_start,
        },
        "seed_domains_disjoint": True,
        "tensorboard_writer": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(run_root / "preflight.json", result)
    return result


def _default_config(repo_root: Path) -> FiveCConfig:
    def absolute(path: str) -> str:
        return str((repo_root / path).resolve())

    return FiveCConfig(
        run_id="5c_main_seed1",
        source_checkpoint=absolute(
            "runs/5b_stage4_selfplay_seed_004/checkpoints/step_000512112.pt"
        ),
        historical_checkpoints=(
            absolute("runs/5a_random_seed_001/checkpoints/step_000100005.pt"),
            absolute("runs/5b_fixed_mix_seed_002/checkpoints/step_000157351.pt"),
            absolute("runs/5b_fixed_mix_seed_002/checkpoints/step_000206542.pt"),
            absolute(
                "runs/5b_fixed_mix_seed_002_resume_206542/checkpoints/step_000255701.pt"
            ),
            absolute(
                "runs/5b_fixed_mix_seed_002_resume_206542/checkpoints/step_000304858.pt"
            ),
            absolute(
                "runs/5b_fixed_mix_seed_002_resume_206542/checkpoints/step_000354015.pt"
            ),
            absolute(
                "runs/5b_fixed_mix_seed_002_resume_206542/checkpoints/step_000403183.pt"
            ),
            absolute(
                "runs/5b_stage4_selfplay_seed_004/checkpoints/step_000512112.pt"
            ),
        ),
        anchor_403183_checkpoint=absolute(
            "runs/5b_fixed_mix_seed_002_resume_206542/checkpoints/step_000403183.pt"
        ),
        anchor_512112_checkpoint=absolute(
            "runs/5b_stage4_selfplay_seed_004/checkpoints/step_000512112.pt"
        ),
    )


def _resolve_latest_checkpoint(run_root: Path) -> Path:
    manifest = run_root / "checkpoints" / "latest.json"
    if not manifest.is_file():
        raise FileNotFoundError("5C latest checkpoint manifest does not exist")
    return Path(json.loads(manifest.read_text(encoding="utf-8"))["checkpoint"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze config and create run directory")
    prepare.add_argument("--run-root", type=Path, default=Path("runs/5c_main_seed1"))
    train = commands.add_parser("train", help="start the frozen 5M run")
    train.add_argument("--config", type=Path, required=True)
    resume = commands.add_parser("resume", help="resume from checkpoints/latest.json")
    resume.add_argument("--config", type=Path, required=True)
    preflight = commands.add_parser("preflight", help="run short non-DEV smoke checks")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        repo_root = Path.cwd().resolve()
        config = _default_config(repo_root)
        result = prepare_five_c_run(config, run_root=args.run_root)
    else:
        config_path = args.config.resolve()
        config = load_five_c_config(config_path)
        run_root = config_path.parent
        if args.command == "preflight":
            result = run_five_c_preflight(
                config, run_root=run_root, device=args.device
            )
        elif args.command == "train":
            result = run_five_c_training(config, run_root=run_root)
        elif args.command == "resume":
            result = run_five_c_training(
                config,
                run_root=run_root,
                resume_checkpoint=_resolve_latest_checkpoint(run_root),
            )
        else:
            raise AssertionError(f"unhandled 5C command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
