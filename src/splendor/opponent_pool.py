"""Small frozen-opponent registry for work package 5B."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch

from .actor_critic import ActorCritic, ActorCriticConfig
from .contracts import Policy
from .features import FEATURE_SCHEMA_VERSION
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .neural_policy import GreedyActorCriticPolicy, frozen_actor_critic_copy
from .ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from .random_policy import RandomPolicy
from .replay import RULESET_ID
from .rng import DeterministicRng, RngState


OpponentKind = Literal["latest", "historical", "score", "development", "random"]


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


@dataclass(frozen=True, slots=True)
class OpponentChoice:
    entry: OpponentEntry
    policy: Policy


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_payload(path: Path, device: torch.device | str) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=True)
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
    model = ActorCritic(ActorCriticConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


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
    ) -> OpponentEntry:
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
        )
        self._historical.append(entry)
        self._models[opponent_id] = _model_from_payload(payload, self.device)
        return entry

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
            "format_version": 1,
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
        if payload.get("format_version") != 1:
            raise ValueError("unsupported opponent-pool manifest format")
        if payload.get("rules_version") != RULESET_ID:
            raise ValueError("opponent-pool rules version mismatch")
        if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("opponent-pool feature schema mismatch")
        pool = cls(device=device, max_historical=int(payload["max_historical"]))
        pool._rng = DeterministicRng.restore(RngState(**payload["rng"]))
        for values in payload["historical"]:
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
        )
