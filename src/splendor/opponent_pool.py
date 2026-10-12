"""Small frozen-opponent registry for work package 5B."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch

from .actor_critic import ActorCritic, ActorCriticConfig, MODEL_WEIGHTS_FORMAT_VERSION, build_actor_critic
from .alphazero_checkpoint import is_alphazero_payload, normalize_actor_critic_payload
from .contracts import Policy
from .features import FEATURE_SCHEMA_VERSION
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .neural_policy import (
    GreedyActorCriticPolicy,
    NeuralInferenceTiming,
    SamplingActorCriticPolicy,
    frozen_actor_critic_copy,
)
from .ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from .random_policy import RandomPolicy
from .replay import RULESET_ID
from .rng import DeterministicRng, RngState


OpponentKind = Literal["latest", "historical", "score", "development", "random"]
HistoricalBucket = Literal["fixed", "dynamic"]


@dataclass(frozen=True, slots=True)
class OpponentEntry:
    opponent_id: str
    kind: OpponentKind
    checkpoint_path: str | None
    checkpoint_id: str | None
    training_step: int | None
    feature_schema_version: str
    model_config: dict[str, object] | None
    reason: str
    added_at_step: int
    added_at_utc: str
    historical_bucket: HistoricalBucket | None = None


@dataclass(frozen=True, slots=True)
class OpponentChoice:
    entry: OpponentEntry
    policy: Policy


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_payload(path: Path, device: torch.device | str) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("checkpoint_schema") == "splendor-frozen-actor-critic-v1":
        if payload.get("format_version") != MODEL_WEIGHTS_FORMAT_VERSION:
            raise ValueError("unsupported frozen model weights format")
        if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("neural opponent feature schema mismatch")
        if payload.get("metadata", {}).get("rules_version") != RULESET_ID:
            raise ValueError("neural opponent rules version mismatch")
        return {"model_config": payload["config"], "model_state_dict": payload["state_dict"],
                "feature_schema_version": payload["feature_schema_version"], "metadata": payload["metadata"]}
    if is_alphazero_payload(payload):
        # AlphaZero checkpoints are read-only model sources here; their
        # optimizer state stays with the AlphaZero learner.
        normalized = normalize_actor_critic_payload(payload)
        if normalized.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("neural opponent feature schema mismatch")
        return normalized
    if payload.get("format_version") != TRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported neural opponent checkpoint format")
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("neural opponent feature schema mismatch")
    metadata = payload.get("metadata", {})
    if metadata.get("rules_version") != RULESET_ID:
        raise ValueError("neural opponent rules version mismatch")
    return payload


def _model_from_payload(
    payload: dict[str, object], device: torch.device | str
) -> ActorCritic:
    model = build_actor_critic(payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


def load_frozen_actor_critic_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> ActorCritic:
    """Load checkpoint weights without restoring any training RNG state."""

    return _model_from_payload(_checkpoint_payload(path, device), device)


class OpponentPool:
    """Manifest-backed historical models plus one per-rollout latest snapshot."""

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        max_historical: int = 12,
        seed: int = 0,
    ) -> None:
        if not 1 <= max_historical <= 12:
            raise ValueError("max_historical must be between 1 and 12")
        self.device = torch.device(device)
        self.max_historical = max_historical
        self._historical: list[OpponentEntry] = []
        self._models: dict[str, ActorCritic] = {}
        self._latest: OpponentEntry | None = None
        self._rng = DeterministicRng(seed)

    @property
    def historical_entries(self) -> tuple[OpponentEntry, ...]:
        return tuple(self._historical)

    @property
    def fixed_historical_entries(self) -> tuple[OpponentEntry, ...]:
        return tuple(
            entry
            for entry in self._historical
            if entry.historical_bucket == "fixed"
        )

    @property
    def dynamic_historical_entries(self) -> tuple[OpponentEntry, ...]:
        return tuple(
            entry
            for entry in self._historical
            if entry.historical_bucket == "dynamic"
        )

    @property
    def latest_entry(self) -> OpponentEntry | None:
        return self._latest

    @property
    def rng_snapshot(self) -> RngState:
        return self._rng.snapshot()

    def set_latest(
        self,
        model: ActorCritic,
        *,
        training_step: int,
        reason: str = "rollout-start frozen latest snapshot",
    ) -> OpponentEntry:
        opponent_id = f"latest-step-{training_step:09d}"
        entry = OpponentEntry(
            opponent_id=opponent_id,
            kind="latest",
            checkpoint_path=None,
            checkpoint_id=None,
            training_step=training_step,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            model_config=asdict(model.config),
            reason=reason,
            added_at_step=training_step,
            added_at_utc=_now_utc(),
            historical_bucket=None,
        )
        if self._latest is not None:
            self._models.pop(self._latest.opponent_id, None)
        self._latest = entry
        self._models[opponent_id] = frozen_actor_critic_copy(
            model, device=self.device
        )
        return entry

    def add_historical(
        self,
        checkpoint_path: Path,
        *,
        reason: str,
        added_at_step: int,
        bucket: HistoricalBucket = "dynamic",
    ) -> OpponentEntry:
        if bucket not in ("fixed", "dynamic"):
            raise ValueError("historical bucket must be fixed or dynamic")
        path = checkpoint_path.resolve()
        payload = _checkpoint_payload(path, self.device)
        training_step = int(payload["global_learning_decision_step"])
        opponent_id = f"historical-step-{training_step:09d}"
        if any(entry.opponent_id == opponent_id for entry in self._historical):
            raise ValueError(f"historical opponent already exists: {opponent_id}")
        if len(self._historical) >= self.max_historical:
            raise ValueError("historical opponent pool is full")
        entry = OpponentEntry(
            opponent_id=opponent_id,
            kind="historical",
            checkpoint_path=str(path),
            checkpoint_id=path.name,
            training_step=training_step,
            feature_schema_version=str(payload["feature_schema_version"]),
            model_config=dict(payload["model_config"]),
            reason=reason,
            added_at_step=added_at_step,
            added_at_utc=_now_utc(),
            historical_bucket=bucket,
        )
        self._historical.append(entry)
        self._models[opponent_id] = _model_from_payload(payload, self.device)
        return entry

    def remove_historical(self, opponent_id: str) -> OpponentEntry:
        """Remove one explicitly selected historical entry without implicit eviction."""

        for index, entry in enumerate(self._historical):
            if entry.opponent_id == opponent_id:
                self._historical.pop(index)
                self._models.pop(opponent_id)
                return entry
        raise ValueError(f"unknown historical opponent id: {opponent_id}")

    def choose(
        self, kind: OpponentKind, *, seed: int | None = None
    ) -> OpponentChoice:
        if kind == "random":
            if seed is None:
                raise ValueError("RandomPolicy opponent requires its independent seed")
            return OpponentChoice(self._fixed_entry("random"), RandomPolicy(seed))
        if kind == "score":
            return OpponentChoice(self._fixed_entry("score"), ScorePolicy())
        if kind == "development":
            return OpponentChoice(
                self._fixed_entry("development"), DevelopmentPolicy()
            )
        if kind == "latest":
            if self._latest is None:
                raise ValueError("latest opponent snapshot has not been set")
            entry = self._latest
        elif kind == "historical":
            if not self._historical:
                raise ValueError("historical opponent pool is empty")
            rng = self._rng if seed is None else DeterministicRng(seed)
            entry = self._historical[
                rng.randbelow(len(self._historical))
            ]
        else:
            raise ValueError(f"unsupported opponent kind: {kind}")
        return OpponentChoice(
            entry,
            GreedyActorCriticPolicy(self._models[entry.opponent_id], device=self.device),
        )

    def choose_for_training(
        self,
        kind: OpponentKind,
        *,
        policy_seed: int,
        timing: NeuralInferenceTiming | None = None,
        dynamic_historical_probability: float | None = None,
    ) -> OpponentChoice:
        """Choose one whole-game opponent; neural policies sample, not greedily act."""

        if kind in ("random", "score", "development"):
            return self.choose(kind, seed=policy_seed)
        if kind == "latest":
            if self._latest is None:
                raise ValueError("latest opponent snapshot has not been set")
            entry = self._latest
        elif kind == "historical":
            if not self._historical:
                raise ValueError("historical opponent pool is empty")
            if dynamic_historical_probability is None:
                candidates = self._historical
            else:
                if not 0.0 <= dynamic_historical_probability <= 1.0:
                    raise ValueError(
                        "dynamic historical probability must be between zero and one"
                    )
                fixed = [
                    entry
                    for entry in self._historical
                    if entry.historical_bucket == "fixed"
                ]
                dynamic = [
                    entry
                    for entry in self._historical
                    if entry.historical_bucket == "dynamic"
                ]
                if not fixed:
                    raise ValueError("bucketed historical sampling requires fixed anchors")
                if not dynamic:
                    candidates = fixed
                else:
                    scale = 1_000_000
                    choose_dynamic = self._rng.randbelow(scale) < int(
                        dynamic_historical_probability * scale
                    )
                    candidates = dynamic if choose_dynamic else fixed
            entry = candidates[self._rng.randbelow(len(candidates))]
        else:
            raise ValueError(f"unsupported opponent kind: {kind}")
        return OpponentChoice(
            entry,
            SamplingActorCriticPolicy(
                self._models[entry.opponent_id],
                seed=policy_seed,
                device=self.device,
                timing=timing,
            ),
        )

    def policy_for(self, opponent_id: str, *, seed: int = 0) -> OpponentChoice:
        if opponent_id in ("random", "score", "development"):
            return self.choose(opponent_id, seed=seed)
        entries = tuple(self._historical) + (
            () if self._latest is None else (self._latest,)
        )
        for entry in entries:
            if entry.opponent_id == opponent_id:
                return OpponentChoice(
                    entry,
                    GreedyActorCriticPolicy(
                        self._models[entry.opponent_id], device=self.device
                    ),
                )
        raise ValueError(f"unknown opponent id: {opponent_id}")

    def save_manifest(self, path: Path) -> None:
        payload = {
            "format_version": 2,
            "max_historical": self.max_historical,
            "rules_version": RULESET_ID,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "rng": asdict(self._rng.snapshot()),
            "latest": None if self._latest is None else asdict(self._latest),
            "historical": [asdict(entry) for entry in self._historical],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load_manifest(
        cls,
        path: Path,
        *,
        device: torch.device | str = "cpu",
        latest_model: ActorCritic | None = None,
    ) -> OpponentPool:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") not in (1, 2):
            raise ValueError("unsupported opponent-pool manifest format")
        if payload.get("rules_version") != RULESET_ID:
            raise ValueError("opponent-pool rules version mismatch")
        if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("opponent-pool feature schema mismatch")
        pool = cls(device=device, max_historical=int(payload["max_historical"]))
        pool._rng = DeterministicRng.restore(RngState(**payload["rng"]))
        for values in payload["historical"]:
            values = dict(values)
            values.setdefault("historical_bucket", "dynamic")
            entry = OpponentEntry(**values)
            if entry.kind != "historical" or entry.checkpoint_path is None:
                raise ValueError("invalid historical opponent entry")
            checkpoint_path = Path(entry.checkpoint_path)
            checkpoint_payload = _checkpoint_payload(checkpoint_path, device)
            if int(checkpoint_payload["global_learning_decision_step"]) != entry.training_step:
                raise ValueError("historical checkpoint step disagrees with manifest")
            if (
                entry.checkpoint_id != checkpoint_path.name
                or entry.feature_schema_version
                != checkpoint_payload["feature_schema_version"]
                or entry.model_config != dict(checkpoint_payload["model_config"])
            ):
                raise ValueError("historical checkpoint metadata disagrees with manifest")
            pool._historical.append(entry)
            pool._models[entry.opponent_id] = _model_from_payload(
                checkpoint_payload, device
            )
        latest_values = payload.get("latest")
        if latest_values is not None and latest_model is not None:
            latest_values = dict(latest_values)
            latest_values.setdefault("historical_bucket", None)
            latest = OpponentEntry(**latest_values)
            if latest.kind != "latest" or latest.model_config != asdict(
                latest_model.config
            ):
                raise ValueError("latest model disagrees with opponent-pool manifest")
            pool._latest = latest
            pool._models[latest.opponent_id] = frozen_actor_critic_copy(
                latest_model, device=device
            )
        return pool

    def _fixed_entry(self, kind: OpponentKind) -> OpponentEntry:
        return OpponentEntry(
            opponent_id=kind,
            kind=kind,
            checkpoint_path=None,
            checkpoint_id=None,
            training_step=None,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            model_config=None,
            reason="fixed policy",
            added_at_step=0,
            added_at_utc="",
            historical_bucket=None,
        )
