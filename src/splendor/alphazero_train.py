"""AlphaZero supervised policy/value learner over frozen MCTS self-play data.

The learner is deliberately separate from the MCTS search, the rules engine
and the historical PPO trainer.  It consumes archived ``policy_target`` and
terminal ``value_target`` labels through the existing ragged collator, trains
every original ActorCritic parameter with plain AdamW, and keeps one
continuous optimizer/RNG state across rounds via the AlphaZero checkpoint
schema.

Loss semantics (per minibatch, ``i`` indexes positions):

``L_pi = sum_i(policy_weight_i * CE(target_i, masked_softmax(logits_i))) / sum_i(policy_weight_i)``
``L_v  = sum_i(value_weight_i * (value_i - value_target_i)^2) / sum_i(value_weight_i)``
``L    = policy_loss_weight * L_pi + value_loss_weight * L_v``

Padding rows never enter the softmax or either denominator, and padding
log-probabilities are replaced by zero before multiplying the target so
``0 * -inf`` cannot produce NaN.  Single-candidate samples carry
``policy_weight=0``; a batch made only of them yields a differentiable zero
policy loss while still training the critic.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .actor_critic import ActorCritic
from .alphazero_checkpoint import (
    load_alphazero_checkpoint, move_optimizer_state, save_alphazero_checkpoint,
)
from .alphazero_replay import PositionPool, ReplayReference
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .rng import DeterministicRng
from .selfplay_data import SelfPlayBatch, SelfPlaySample, collate_selfplay_samples

DEFAULT_DEVICE = "cuda"
_EPS = torch.finfo(torch.float32).tiny


class NonFiniteTrainingError(RuntimeError):
    """Raised instead of silently skipping a numerically broken update."""

    def __init__(self, message: str, *, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    device: str = DEFAULT_DEVICE
    batch_size: int = 256
    history_batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    steps_per_new_position: float = 4.0
    checkpoint_every_steps: int = 100
    validation_batch_size: int = 256
    max_steps_per_round: int | None = None

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("learner device must be cpu or cuda")
        for name in ("batch_size", "history_batch_size", "checkpoint_every_steps", "validation_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.history_batch_size > self.batch_size:
            raise ValueError("history_batch_size cannot exceed batch_size")
        for name in ("learning_rate", "weight_decay", "policy_loss_weight", "value_loss_weight", "max_grad_norm", "steps_per_new_position"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_steps_per_round is not None and (
            type(self.max_steps_per_round) is not int or self.max_steps_per_round <= 0
        ):
            raise ValueError("max_steps_per_round must be a positive integer or null")

    def steps_for_round(self, new_train_positions: int) -> int:
        if type(new_train_positions) is not int or new_train_positions < 0:
            raise ValueError("new_train_positions must be a nonnegative integer")
        steps = math.ceil(self.steps_per_new_position * new_train_positions / self.batch_size)
        if self.max_steps_per_round is not None:
            steps = min(steps, self.max_steps_per_round)
        return steps


@dataclass(slots=True)
class AlphazeroLossSums:
    """Unnormalized loss sums so batches of different weights can be combined."""

    policy_ce_sum: Tensor
    policy_weight_sum: Tensor
    value_sq_sum: Tensor
    value_weight_sum: Tensor
    entropy_sum: Tensor

    @property
    def policy_loss(self) -> Tensor:
        return self.policy_ce_sum / self.policy_weight_sum.clamp_min(_EPS)

    @property
    def value_loss(self) -> Tensor:
        return self.value_sq_sum / self.value_weight_sum.clamp_min(_EPS)

    @property
    def entropy(self) -> Tensor:
        return self.entropy_sum / self.policy_weight_sum.clamp_min(_EPS)

    def total(self, config: LearnerConfig) -> Tensor:
        return (
            config.policy_loss_weight * self.policy_loss
            + config.value_loss_weight * self.value_loss
        )


def alphazero_loss_sums(
    logits: Tensor,
    values: Tensor,
    *,
    candidate_mask: Tensor,
    policy_targets: Tensor,
    value_targets: Tensor,
    policy_weights: Tensor,
    value_weights: Tensor,
) -> AlphazeroLossSums:
    """Masked policy/value losses with padding-safe log-probabilities."""

    if logits.ndim != 2 or candidate_mask.shape != logits.shape:
        raise ValueError("logits and candidate mask must be [B, M]")
    if values.shape != value_targets.shape or values.shape != value_weights.shape:
        raise ValueError("value predictions and targets must share [B]")
    if policy_targets.shape != logits.shape or policy_weights.shape != values.shape:
        raise ValueError("policy targets and weights must match their predictions")
    masked_logits = logits.masked_fill(~candidate_mask, -torch.inf)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    safe_log_probs = torch.where(
        candidate_mask, log_probs, torch.zeros_like(log_probs)
    )
    policy_ce = -(policy_targets * safe_log_probs).sum(dim=-1)
    probabilities = torch.where(
        candidate_mask, log_probs.exp(), torch.zeros_like(log_probs)
    )
    entropy = -(probabilities * safe_log_probs).sum(dim=-1)
    squared_error = (values - value_targets) ** 2
    return AlphazeroLossSums(
        policy_ce_sum=(policy_ce * policy_weights).sum(),
        policy_weight_sum=policy_weights.sum(),
        value_sq_sum=(squared_error * value_weights).sum(),
        value_weight_sum=value_weights.sum(),
        entropy_sum=(entropy * policy_weights).sum(),
    )


@dataclass(frozen=True, slots=True)
class DrawnBatch:
    samples: tuple[SelfPlaySample, ...]
    references: tuple[ReplayReference, ...]
    current_positions: int
    history_positions: int


def draw_training_batch(
    current_pool: PositionPool,
    history_pool: PositionPool | None,
    rng: DeterministicRng,
    config: LearnerConfig,
) -> DrawnBatch:
    """Draw one minibatch: 128 current + 128 history when history exists."""

    if len(current_pool) == 0:
        raise ValueError("current position pool is empty")
    if history_pool is None or len(history_pool) == 0:
        history_count = 0
        history_pool = None
    else:
        history_count = config.history_batch_size
    current_count = config.batch_size - history_count
    samples: list[SelfPlaySample] = []
    references: list[ReplayReference] = []
    for _ in range(current_count):
        sample, reference = current_pool.sample(rng)
        samples.append(sample)
        references.append(reference)
    if history_pool is not None:
        for _ in range(history_count):
            sample, reference = history_pool.sample(rng)
            samples.append(sample)
            references.append(reference)
    return DrawnBatch(tuple(samples), tuple(references), current_count, history_count)


def collate_training_samples(samples: Sequence[SelfPlaySample]) -> SelfPlayBatch:
    return collate_selfplay_samples(samples)


def _torch_batch(batch: SelfPlayBatch, device: torch.device) -> dict[str, Tensor]:
    return {
        "candidate_mask": torch.as_tensor(batch.decisions.candidate_mask, device=device),
        "policy_targets": torch.as_tensor(batch.policy_targets, device=device),
        "policy_weights": torch.as_tensor(batch.policy_weights, device=device),
        "value_targets": torch.as_tensor(batch.value_targets, device=device),
        "value_weights": torch.as_tensor(batch.value_weights, device=device),
    }


def resolve_learner_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "learner device 'cuda' was requested but CUDA is unavailable; "
            "set learner.device to 'cpu' explicitly"
        )
    return device


class AlphazeroLearner:
    """Trainable ActorCritic plus one continuous AdamW/sampling-RNG state."""

    def __init__(
        self,
        model: ActorCritic,
        config: LearnerConfig,
        *,
        sampling_rng: DeterministicRng,
    ) -> None:
        self.config = config
        self.device = resolve_learner_device(config.device)
        self.model = model.to(self.device)
        self.model.train()
        self.model.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.sampling_rng = sampling_rng
        self.round_index = 0
        self.round_train_step = 0
        self.total_train_step = 0
        self.round_steps_target = 0
        self.replay_identity: dict[str, Any] = {}
        self.parent_checkpoint: dict[str, Any] = {}

    @classmethod
    def warm_start(
        cls,
        checkpoint: str | Path,
        config: LearnerConfig,
        *,
        sampling_seed: int,
        replay_identity: Mapping[str, Any],
        parent_checkpoint: Mapping[str, Any],
        round_index: int,
        round_steps_target: int,
    ) -> AlphazeroLearner:
        """Load old weights once, then create a fresh optimizer for this loop."""

        model = load_frozen_actor_critic_checkpoint(Path(checkpoint), device="cpu")
        learner = cls(model, config, sampling_rng=DeterministicRng(sampling_seed))
        learner.round_index = int(round_index)
        learner.round_steps_target = int(round_steps_target)
        learner.replay_identity = dict(replay_identity)
        learner.parent_checkpoint = dict(parent_checkpoint)
        return learner

    @classmethod
    def resume(
        cls,
        checkpoint: str | Path,
        config: LearnerConfig,
        *,
        replay_identity: Mapping[str, Any],
        round_index: int,
        round_steps_target: int,
    ) -> AlphazeroLearner:
        """Restore parameters, optimizer, RNG and progress from one checkpoint."""

        resolve_learner_device(config.device)
        payload = load_alphazero_checkpoint(checkpoint, device=config.device)
        if payload.learner_config != asdict(config):
            raise ValueError("checkpoint learner configuration does not match this loop")
        if payload.replay_identity.get("digest") != replay_identity.get("digest"):
            raise ValueError("checkpoint was trained on a different frozen replay index")
        if payload.round_index != round_index:
            raise ValueError("checkpoint round index does not match the current round")
        if payload.round_steps_target != round_steps_target:
            raise ValueError("checkpoint step target does not match this round")
        model = payload.build_model(device=config.device)
        learner = cls(model, config, sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
        learner.optimizer.load_state_dict(payload.optimizer_state_dict)
        move_optimizer_state(learner.optimizer, learner.device)
        learner.round_index = payload.round_index
        learner.round_train_step = payload.round_train_step
        learner.total_train_step = payload.total_train_step
        learner.round_steps_target = payload.round_steps_target
        learner.replay_identity = dict(payload.replay_identity)
        learner.parent_checkpoint = dict(payload.parent_checkpoint)
        torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
        if payload.torch_cuda_rng_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload.torch_cuda_rng_states]
            )
        return learner

    @classmethod
    def continue_from(
        cls,
        checkpoint: str | Path,
        config: LearnerConfig,
        *,
        sampling_seed: int,
        replay_identity: Mapping[str, Any],
        parent_checkpoint: Mapping[str, Any],
        round_index: int,
        round_steps_target: int,
    ) -> AlphazeroLearner:
        """Start the round after ``checkpoint`` without resetting the optimizer.

        Model weights, AdamW moments and ``total_train_step`` stay continuous
        across the round boundary; only the round-local counters restart.  The
        sampling stream is re-seeded from this round's own seed, so a round's
        batch order follows that seed instead of how many steps the previous
        round happened to run.
        """

        resolve_learner_device(config.device)
        payload = load_alphazero_checkpoint(checkpoint, device=config.device)
        if payload.learner_config != asdict(config):
            raise ValueError("checkpoint learner configuration does not match this loop")
        if payload.round_index + 1 != round_index:
            raise ValueError("continuation must start the round after the checkpoint")
        model = payload.build_model(device=config.device)
        learner = cls(model, config, sampling_rng=DeterministicRng(sampling_seed))
        learner.optimizer.load_state_dict(payload.optimizer_state_dict)
        move_optimizer_state(learner.optimizer, learner.device)
        learner.round_index = int(round_index)
        learner.round_train_step = 0
        learner.total_train_step = payload.total_train_step
        learner.round_steps_target = int(round_steps_target)
        learner.replay_identity = dict(replay_identity)
        learner.parent_checkpoint = dict(parent_checkpoint)
        return learner

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return save_alphazero_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            learner_config=asdict(self.config),
            round_index=self.round_index,
            round_train_step=self.round_train_step,
            total_train_step=self.total_train_step,
            round_steps_target=self.round_steps_target,
            parent_checkpoint=self.parent_checkpoint,
            replay_identity=self.replay_identity,
            sampling_rng_state=self.sampling_rng.snapshot(),
            metadata=metadata,
        )

    def train_step(self, batch: SelfPlayBatch, *, references: Sequence[ReplayReference]) -> dict[str, Any]:
        """One optimizer update; raises instead of skipping a bad batch."""

        tensors = _torch_batch(batch, self.device)
        self.model.train()
        logits, values = self.model(batch.decisions)
        terms = alphazero_loss_sums(
            logits,
            values,
            candidate_mask=tensors["candidate_mask"],
            policy_targets=tensors["policy_targets"],
            value_targets=tensors["value_targets"],
            policy_weights=tensors["policy_weights"],
            value_weights=tensors["value_weights"],
        )
        total = terms.total(self.config)
        self.optimizer.zero_grad(set_to_none=True)
        finite = (
            torch.isfinite(logits).all()
            and torch.isfinite(values).all()
            and torch.isfinite(total)
            and torch.isfinite(terms.policy_loss)
            and torch.isfinite(terms.value_loss)
        )
        if not bool(finite):
            raise NonFiniteTrainingError(
                f"nonfinite learner output at round {self.round_index} step {self.round_train_step}",
                details={
                    "round_index": self.round_index,
                    "round_train_step": self.round_train_step,
                    "total_train_step": self.total_train_step,
                    "policy_loss": float(terms.policy_loss),
                    "value_loss": float(terms.value_loss),
                    "references": [reference.to_dict() for reference in references],
                },
            )
        total.backward()
        grad_norm = float(clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm))
        if not math.isfinite(grad_norm):
            raise NonFiniteTrainingError(
                f"nonfinite gradient norm at round {self.round_index} step {self.round_train_step}",
                details={
                    "round_index": self.round_index,
                    "round_train_step": self.round_train_step,
                    "total_train_step": self.total_train_step,
                    "grad_norm": grad_norm,
                    "references": [reference.to_dict() for reference in references],
                },
            )
        self.optimizer.step()
        self.round_train_step += 1
        self.total_train_step += 1
        return {
            "event": "train_step",
            "round_index": self.round_index,
            "round_train_step": self.round_train_step,
            "total_train_step": self.total_train_step,
            "policy_loss": float(terms.policy_loss.detach()),
            "value_loss": float(terms.value_loss.detach()),
            "total_loss": float(total.detach()),
            "policy_entropy": float(terms.entropy.detach()),
            "grad_norm": grad_norm,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "policy_weight_sum": float(terms.policy_weight_sum.detach()),
            "value_weight_sum": float(terms.value_weight_sum.detach()),
            "batch_size": int(tensors["value_targets"].shape[0]),
        }

    def evaluate_samples(self, samples: Sequence[SelfPlaySample]) -> dict[str, Any]:
        """Validation loss over whole games, never mixed into training pools."""

        if not samples:
            return {"available": False, "positions": 0, "policy_loss": None, "value_loss": None}
        was_training = self.model.training
        self.model.eval()
        totals = {
            "policy_ce_sum": 0.0,
            "policy_weight_sum": 0.0,
            "value_sq_sum": 0.0,
            "value_weight_sum": 0.0,
            "entropy_sum": 0.0,
            "positions": 0,
        }
        with torch.inference_mode():
            for start in range(0, len(samples), self.config.validation_batch_size):
                chunk = samples[start : start + self.config.validation_batch_size]
                batch = collate_selfplay_samples(chunk)
                tensors = _torch_batch(batch, self.device)
                logits, values = self.model(batch.decisions)
                terms = alphazero_loss_sums(
                    logits,
                    values,
                    candidate_mask=tensors["candidate_mask"],
                    policy_targets=tensors["policy_targets"],
                    value_targets=tensors["value_targets"],
                    policy_weights=tensors["policy_weights"],
                    value_weights=tensors["value_weights"],
                )
                totals["policy_ce_sum"] += float(terms.policy_ce_sum)
                totals["policy_weight_sum"] += float(terms.policy_weight_sum)
                totals["value_sq_sum"] += float(terms.value_sq_sum)
                totals["value_weight_sum"] += float(terms.value_weight_sum)
                totals["entropy_sum"] += float(terms.entropy_sum)
                totals["positions"] += len(chunk)
        self.model.train(was_training)
        policy_denom = max(totals["policy_weight_sum"], _EPS)
        value_denom = max(totals["value_weight_sum"], _EPS)
        return {
            "available": True,
            "positions": totals["positions"],
            "policy_loss": totals["policy_ce_sum"] / policy_denom,
            "value_loss": totals["value_sq_sum"] / value_denom,
            "policy_entropy": totals["entropy_sum"] / policy_denom,
        }


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _reconcile_metrics_file(metrics_path: Path, *, round_index: int, committed_step: int) -> None:
    """Drop metrics for work the interrupted session never checkpointed.

    A session appends its metrics line before the next periodic checkpoint, so
    a crash can leave lines for steps that the resume runs again; the
    validation line is always rewritten when the round finishes.
    """

    if not metrics_path.is_file():
        return
    kept: list[str] = []
    dropped = False
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            # A crash can tear the final append; only complete lines count.
            dropped = True
            continue
        if (
            isinstance(record, dict)
            and record.get("event") == "train_step"
            and record.get("round_index") == round_index
            and type(record.get("round_train_step")) is int
            and record["round_train_step"] <= committed_step
        ):
            kept.append(stripped)
        else:
            dropped = True
    if not dropped:
        return
    temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for line in kept:
            handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(metrics_path)


def run_training_round(
    learner: AlphazeroLearner,
    *,
    steps_target: int,
    current_pool: PositionPool,
    history_pool: PositionPool | None,
    validation_samples: Sequence[SelfPlaySample],
    validation_games: int,
    checkpoint_dir: Path,
    metrics_path: Path,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Drive one round of optimizer steps and checkpoint every N updates."""

    if steps_target < 0:
        raise ValueError("steps_target cannot be negative")
    if learner.round_steps_target != steps_target:
        raise ValueError("learner step target disagrees with this training round")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # Metrics are appended ahead of the periodic checkpoint, so a resume drops
    # the lines that its checkpoint cannot account for before appending again.
    _reconcile_metrics_file(
        metrics_path, round_index=learner.round_index, committed_step=learner.round_train_step
    )
    started = time.perf_counter()
    history_length = 0 if history_pool is None else len(history_pool)
    step_records: list[dict[str, Any]] = []
    while learner.round_train_step < steps_target:
        drawn = draw_training_batch(current_pool, history_pool, learner.sampling_rng, learner.config)
        batch = collate_training_samples(drawn.samples)
        record = learner.train_step(batch, references=drawn.references)
        record["current_positions"] = drawn.current_positions
        record["history_positions"] = drawn.history_positions
        _append_jsonl(metrics_path, record)
        if learner.round_train_step % learner.config.checkpoint_every_steps == 0 or learner.round_train_step == steps_target:
            learner.save(checkpoint_dir / f"step-{learner.round_train_step:09d}.pt")
        if progress_callback is not None:
            progress_callback(record)
        step_records.append(record)
    validation = learner.evaluate_samples(validation_samples)
    validation["games"] = validation_games
    validation["event"] = "validation"
    validation["round_index"] = learner.round_index
    _append_jsonl(metrics_path, validation)
    losses = [record["total_loss"] for record in step_records]
    return {
        "round_index": learner.round_index,
        "steps_target": steps_target,
        # Cumulative round progress, so a resumed session never double-counts
        # optimizer steps that were already committed by an earlier session.
        "steps_completed": learner.round_train_step,
        "steps_this_session": len(step_records),
        "history_pool_positions": history_length,
        "checkpoint_every_steps": learner.config.checkpoint_every_steps,
        "metrics_path": str(metrics_path.resolve()),
        "mean_total_loss": sum(losses) / len(losses) if losses else None,
        "first_total_loss": losses[0] if losses else None,
        "last_total_loss": losses[-1] if losses else None,
        "validation": validation,
        "training_seconds": time.perf_counter() - started,
    }
