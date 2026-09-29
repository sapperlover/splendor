"""AlphaZero policy/value checkpoint format and frozen-loader compatibility.

The AlphaZero loop owns its own checkpoint schema.  It deliberately does not
reuse the PPO trainer payload: a PPO checkpoint may only be *read* for model
weights, and an AlphaZero checkpoint must never be mistaken for a resumable
PPO state.  Both formats expose the same read-only subset that frozen search
and evaluation already consume (``model_config`` / ``model_state_dict`` /
``feature_schema_version``), so :func:`normalize_actor_critic_payload` is the
single place where the frozen loader learns about the new schema.

Everything stored here is a basic Python value or a tensor so the existing
``torch.load(..., weights_only=True)`` path keeps working.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .actor_critic import ActorCritic, ActorCriticConfig
from .features import FEATURE_SCHEMA_VERSION
from .replay import RULESET_ID
from .rng import RngState

ALGORITHM = "alphazero-policy-value"
CHECKPOINT_SCHEMA = "splendor-alphazero-checkpoint-v1"
CHECKPOINT_FORMAT_VERSION = 1
OPTIMIZER_NAME = "AdamW"


@dataclass(frozen=True, slots=True)
class AlphazeroCheckpoint:
    """Validated contents of one AlphaZero checkpoint."""

    model_config: dict[str, Any]
    model_state_dict: dict[str, Any]
    model_sha256: str
    optimizer_config: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    round_index: int
    round_train_step: int
    total_train_step: int
    round_steps_target: int
    parent_checkpoint: dict[str, Any]
    replay_identity: dict[str, Any]
    learner_config: dict[str, Any]
    sampling_rng_state: RngState
    torch_cpu_rng_state: Any
    torch_cuda_rng_states: tuple[Any, ...]
    metadata: dict[str, Any]
    path: Path | None = None

    def build_model(self, *, device: torch.device | str = "cpu") -> ActorCritic:
        model = ActorCritic(ActorCriticConfig(**self.model_config))
        model.load_state_dict(self.model_state_dict, strict=True)
        return model.to(device)


def is_alphazero_payload(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("algorithm") == ALGORITHM
        and payload.get("checkpoint_schema") == CHECKPOINT_SCHEMA
    )


def model_state_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash parameter names, shapes and raw bytes; used only at save/load edges."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        contiguous = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_alphazero_payload(label: str, payload: Mapping[str, Any]) -> None:
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported AlphaZero checkpoint format: {label}")
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"AlphaZero checkpoint feature schema mismatch: {label}")
    if payload.get("rules_version") != RULESET_ID:
        raise ValueError(f"AlphaZero checkpoint rules version mismatch: {label}")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or optimizer.get("name") != OPTIMIZER_NAME:
        raise ValueError(f"AlphaZero checkpoint optimizer mismatch: {label}")
    if not isinstance(payload.get("model_config"), Mapping) or not isinstance(
        payload.get("model_state_dict"), Mapping
    ):
        raise ValueError(f"AlphaZero checkpoint model payload mismatch: {label}")


def load_alphazero_checkpoint(
    path: str | PathLike[str], *, device: torch.device | str = "cpu"
) -> AlphazeroCheckpoint:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not is_alphazero_payload(payload):
        raise ValueError(f"not an AlphaZero checkpoint: {checkpoint_path}")
    _validate_alphazero_payload(str(checkpoint_path), payload)
    return AlphazeroCheckpoint(
        model_config=dict(payload["model_config"]),
        model_state_dict=dict(payload["model_state_dict"]),
        model_sha256=str(payload["model_sha256"]),
        optimizer_config=dict(payload["optimizer"]["config"]),
        optimizer_state_dict=dict(payload["optimizer_state_dict"]),
        round_index=int(payload["round_index"]),
        round_train_step=int(payload["round_train_step"]),
        total_train_step=int(payload["total_train_step"]),
        round_steps_target=int(payload["round_steps_target"]),
        parent_checkpoint=dict(payload["parent_checkpoint"]),
        replay_identity=dict(payload["replay_identity"]),
        learner_config=dict(payload["learner_config"]),
        sampling_rng_state=RngState(**payload["sampling_rng_state"]),
        torch_cpu_rng_state=payload["torch_cpu_rng_state"],
        torch_cuda_rng_states=tuple(payload["torch_cuda_rng_states"]),
        metadata=dict(payload["metadata"]),
        path=checkpoint_path,
    )


def save_alphazero_checkpoint(
    path: str | PathLike[str],
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    learner_config: Mapping[str, Any],
    round_index: int,
    round_train_step: int,
    total_train_step: int,
    round_steps_target: int,
    parent_checkpoint: Mapping[str, Any],
    replay_identity: Mapping[str, Any],
    sampling_rng_state: RngState,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one checkpoint atomically and return its lightweight identity record."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    payload = {
        "algorithm": ALGORITHM,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "rules_version": RULESET_ID,
        "model_config": asdict(model.config),
        "model_state_dict": state_dict,
        "model_sha256": model_state_sha256(state_dict),
        "optimizer": {
            "name": OPTIMIZER_NAME,
            "config": {
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            },
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "round_index": int(round_index),
        "round_train_step": int(round_train_step),
        "total_train_step": int(total_train_step),
        "round_steps_target": int(round_steps_target),
        "parent_checkpoint": dict(parent_checkpoint),
        "replay_identity": dict(replay_identity),
        "learner_config": dict(learner_config),
        "sampling_rng_state": asdict(sampling_rng_state),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else []
        ),
        "metadata": dict(metadata or {}),
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    return {
        "algorithm": ALGORITHM,
        "schema": CHECKPOINT_SCHEMA,
        "path": str(checkpoint_path.resolve()),
        "model_sha256": payload["model_sha256"],
        "round_index": int(round_index),
        "round_train_step": int(round_train_step),
        "total_train_step": int(total_train_step),
        "round_steps_target": int(round_steps_target),
        "parent_checkpoint": dict(parent_checkpoint),
        "replay_identity": dict(replay_identity),
    }


def normalize_actor_critic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map either checkpoint format to the frozen loader's read-only subset.

    PPO payloads pass through unchanged.  AlphaZero payloads are reduced to the
    same keys ``opponent_pool`` already reads, so collection and evaluation can
    consume new checkpoints without learning the AlphaZero schema.
    """

    if not is_alphazero_payload(payload):
        return dict(payload)
    _validate_alphazero_payload("frozen-loader payload", payload)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "feature_schema_version": payload["feature_schema_version"],
        "model_config": dict(payload["model_config"]),
        "model_state_dict": dict(payload["model_state_dict"]),
        # Frozen-opponent identity uses a training step; the AlphaZero learner
        # step is the closest equivalent and is unique per checkpoint.
        "global_learning_decision_step": int(payload["total_train_step"]),
        "metadata": {
            "rules_version": payload["rules_version"],
            "algorithm": ALGORITHM,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "round_index": int(payload["round_index"]),
            "round_train_step": int(payload["round_train_step"]),
            "model_sha256": payload["model_sha256"],
            "parent_checkpoint": dict(payload["parent_checkpoint"]),
        },
    }


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Place every optimizer state tensor on the learner device after a restore."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def model_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
