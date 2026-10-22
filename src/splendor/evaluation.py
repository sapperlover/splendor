from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from .contracts import Policy
from .catalog import load_default_catalog
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .random_policy import RandomPolicy
from .runner import MatchOutcome, MatchResult, RunStatus, run_match


PolicyFactory = Callable[[int], Policy]


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    factory: PolicyFactory
    parallel_spec: SerializablePolicySpec | None = None

    def create(self, seed: int) -> Policy:
        return self.factory(seed)


@dataclass(frozen=True, slots=True)
class SerializablePolicySpec:
    """Spawn-safe policy description for pair-parallel CPU evaluation."""

    name: str
    kind: str
    baseline_name: str | None = None
    checkpoint: str | None = None
    search_config: dict[str, Any] | None = None
    search_backend: str = "python"
    inference: dict[str, Any] | None = None
    chance_widening: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.search_backend not in ("python", "native", "native_chance"):
            raise ValueError("search_backend must be python, native or native_chance")
        if self.search_backend != "python" and self.kind != "mcts_checkpoint":
            raise ValueError("native search backend requires mcts_checkpoint")
        if self.chance_widening is not None:
            if self.search_backend != "native_chance":
                raise ValueError("chance_widening requires native_chance")
            if not isinstance(self.chance_widening, Mapping):
                raise ValueError("chance_widening must be a mapping")
            from .native_policy import ChanceWideningConfig
            ChanceWideningConfig(**self.chance_widening)
        if self.kind == "baseline":
            if self.baseline_name not in _BASELINES:
                raise ValueError("parallel baseline policy is unknown")
            if self.checkpoint is not None or self.search_config is not None:
                raise ValueError("baseline policy cannot carry neural configuration")
            return
        if self.kind not in (
            "greedy_checkpoint", "search_checkpoint", "mcts_checkpoint"
        ):
            raise ValueError(f"unsupported parallel policy kind: {self.kind}")
        if not self.checkpoint:
            raise ValueError("checkpoint-backed policy requires a checkpoint")
        if self.kind == "greedy_checkpoint" and self.search_config is not None:
            raise ValueError("greedy policy cannot carry search configuration")
        if (
            self.kind in ("search_checkpoint", "mcts_checkpoint")
            and self.search_config is None
        ):
            raise ValueError("search policy requires search configuration")
        if self.kind == "mcts_checkpoint":
            from .mcts_policy import MCTSConfig

            search = MCTSConfig(**self.search_config)
            if self.search_backend == "native_chance" and search.determinizations != 1:
                raise ValueError("native_chance requires determinizations=1")

    @classmethod
    def baseline(cls, name: str) -> SerializablePolicySpec:
        return cls(name=name, kind="baseline", baseline_name=name)

    @classmethod
    def greedy_checkpoint(
        cls, name: str, checkpoint: Path | str
    ) -> SerializablePolicySpec:
        return cls(
            name=name,
            kind="greedy_checkpoint",
            checkpoint=str(Path(checkpoint).resolve()),
        )

    @classmethod
    def search_checkpoint(
        cls,
        name: str,
        checkpoint: Path | str,
        *,
        search_config: dict[str, Any],
    ) -> SerializablePolicySpec:
        return cls(
            name=name,
            kind="search_checkpoint",
            checkpoint=str(Path(checkpoint).resolve()),
            search_config=dict(search_config),
        )

    @classmethod
    def mcts_checkpoint(
        cls, name: str, checkpoint: Path | str, *, search_config: dict[str, Any],
        search_backend: str = "python",
        chance_widening: dict[str, Any] | None = None,
    ) -> SerializablePolicySpec:
        return cls(
            name=name,
            kind="mcts_checkpoint",
            checkpoint=str(Path(checkpoint).resolve()),
            search_config=dict(search_config), search_backend=search_backend, chance_widening=chance_widening,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SerializablePolicySpec:
        """Build a policy description from evaluator JSON configuration."""

        kind = str(value.get("kind", ""))
        backend = str(value.get("search_backend", "python"))
        if backend not in ("python", "native", "native_chance") or (backend != "python" and kind != "mcts_checkpoint"):
            raise ValueError("native search backend requires mcts_checkpoint")
        if value.get("chance_widening") is not None and backend != "native_chance":
            raise ValueError("chance_widening requires native_chance")
        name = str(value.get("name", kind))
        if kind == "baseline":
            return cls.baseline(str(value.get("baseline_name", name)))
        checkpoint = value.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"policy {name!r} requires a checkpoint path")
        inference = value.get("inference")
        if inference is not None and not isinstance(inference, Mapping):
            raise ValueError(f"policy {name!r} inference settings must be a mapping")
        if kind == "greedy_checkpoint":
            spec = cls.greedy_checkpoint(name, checkpoint)
            return replace(spec, inference=None if inference is None else dict(inference))
        if kind == "search_checkpoint":
            search_config = value.get("search_config")
            if not isinstance(search_config, Mapping):
                raise ValueError(f"search policy {name!r} requires search_config")
            spec = cls.search_checkpoint(name, checkpoint, search_config=dict(search_config))
            return replace(spec, inference=None if inference is None else dict(inference))
        if kind == "mcts_checkpoint":
            search_config = value.get("search_config")
            if not isinstance(search_config, Mapping):
                raise ValueError(f"MCTS policy {name!r} requires search_config")
            spec = cls.mcts_checkpoint(
                name, checkpoint, search_config=dict(search_config),
                search_backend=backend,
                chance_widening=value.get("chance_widening"),
            )
            return replace(spec, inference=None if inference is None else dict(inference))
        raise ValueError(f"unsupported policy kind: {kind!r}")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "kind": self.kind}
        if self.inference is not None:
            result["inference"] = dict(self.inference)
        if self.baseline_name is not None:
            result["baseline_name"] = self.baseline_name
        if self.checkpoint is not None:
            result["checkpoint"] = str(Path(self.checkpoint).resolve())
        if self.search_config is not None:
            result["search_config"] = dict(self.search_config)
        if self.search_backend != "python":
            result["search_backend"] = self.search_backend
        if self.chance_widening is not None:
            result["chance_widening"] = dict(self.chance_widening)
        return result


@dataclass(frozen=True, slots=True)
class OutcomeBreakdown:
    games: int
    valid_games: int
    wins: int
    draws: int
    losses: int
    win_rate_score: float | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    policy_a: str
    policy_b: str
    game_seeds: tuple[int, ...]
    total_pairs: int
    total_games: int
    valid_games: int
    a_wins: int
    draws: int
    a_losses: int
    win_rate_score: float | None
    a_first: OutcomeBreakdown
    a_second: OutcomeBreakdown
    mean_completed_player_turns: float | None
    median_completed_player_turns: float | None
    mean_decisions: float | None
    forced_skips: int
    engineering_truncations: int
    exceptions: int
    illegal_actions: int
    invariant_failures: int
    matches: tuple[MatchResult, ...]
    policy_cycle_draws: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.valid_games == self.total_games
            and self.engineering_truncations == 0
            and self.exceptions == 0
            and self.illegal_actions == 0
            and self.invariant_failures == 0
        )

    def to_dict(self, *, include_matches: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "game_seeds": list(self.game_seeds),
            "total_pairs": self.total_pairs,
            "total_games": self.total_games,
            "valid_games": self.valid_games,
            "a_wins": self.a_wins,
            "draws": self.draws,
            "a_losses": self.a_losses,
            "win_rate_score": self.win_rate_score,
            "a_first": _breakdown_dict(self.a_first),
            "a_second": _breakdown_dict(self.a_second),
            "mean_completed_player_turns": self.mean_completed_player_turns,
            "median_completed_player_turns": self.median_completed_player_turns,
            "mean_decisions": self.mean_decisions,
            "forced_skips": self.forced_skips,
            "engineering_truncations": self.engineering_truncations,
            "policy_cycle_draws": self.policy_cycle_draws,
            "exceptions": self.exceptions,
            "illegal_actions": self.illegal_actions,
            "invariant_failures": self.invariant_failures,
            "passed": self.passed,
        }
        if include_matches:
            pairs: list[list[dict[str, object]]] = [[] for _ in self.game_seeds]
            for match in self.matches:
                if match.pair_id is None:
                    raise ValueError("evaluation match is missing pair_id")
                pairs[match.pair_id].append(_match_reference(match))
            result["pairs"] = [
                {
                    "pair_id": pair_id,
                    "game_seed": game_seed,
                    "games": pairs[pair_id],
                }
                for pair_id, game_seed in enumerate(self.game_seeds)
            ]
        return result


@dataclass(frozen=True, slots=True)
class FixedEvaluationConfig:
    """One fixed matchup and its reproducible paired-game schedule."""

    policy_a: SerializablePolicySpec
    policy_b: SerializablePolicySpec
    game_seeds: tuple[int, ...]
    policy_seed_base: int = 0
    workers: int = 1
    device: str = "cpu"
    max_completed_turns: int = 400
    adjudicate_deterministic_policy_cycles: bool = True
    bootstrap_resamples: int = 0
    bootstrap_seed: int = 0
    metadata: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.game_seeds:
            raise ValueError("at least one paired game seed is required")
        if len(set(self.game_seeds)) != len(self.game_seeds) or not all(
            type(seed) is int for seed in self.game_seeds
        ):
            raise ValueError("game seeds must be unique integers")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.workers > 1 and self.device != "cpu":
            raise ValueError("parallel evaluation requires CPU inference")
        if self.max_completed_turns <= 0:
            raise ValueError("max_completed_turns must be positive")
        if self.bootstrap_resamples < 0:
            raise ValueError("bootstrap_resamples cannot be negative")
        if self.execution is not None:
            pooled = self.execution.get("backend") == "native_cuda_pool"
            if pooled:
                from .evaluation_pool import validate_pool_evaluation
                validate_pool_evaluation(self)
            else:
                from .evaluation_batch import BatchEvaluationConfig
                BatchEvaluationConfig(**self.execution)
            if self.workers != 1:
                raise ValueError("batched evaluation has one inference owner; workers must be 1")
            for spec in (self.policy_a, self.policy_b):
                if spec.kind == "greedy_checkpoint":
                    continue
                if spec.kind != "mcts_checkpoint" or spec.search_backend not in ("native", "native_chance"):
                    raise ValueError("batched evaluation requires native mcts_checkpoint or greedy_checkpoint policies")

    def to_dict(self) -> dict[str, object]:
        value = {
            "policy_a": self.policy_a.to_dict(),
            "policy_b": self.policy_b.to_dict(),
            "game_seeds": list(self.game_seeds),
            "policy_seed_base": self.policy_seed_base,
            "workers": self.workers,
            "device": self.device,
            "max_completed_turns": self.max_completed_turns,
            "adjudicate_deterministic_policy_cycles": (
                self.adjudicate_deterministic_policy_cycles
            ),
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "metadata": dict(self.metadata or {}),
        }
        if self.execution is not None:
            value["execution"] = dict(self.execution)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FixedEvaluationConfig:
        raw_a = value.get("policy_a")
        raw_b = value.get("policy_b")
        if not isinstance(raw_a, Mapping) or not isinstance(raw_b, Mapping):
            raise ValueError("config must define policy_a and policy_b objects")
        raw_seeds = value.get("game_seeds")
        if raw_seeds is None:
            pairs = value.get("pairs")
            seed_start = value.get("seed_start")
            if not isinstance(pairs, int) or not isinstance(seed_start, int):
                raise ValueError("config requires game_seeds or pairs and seed_start")
            if pairs <= 0:
                raise ValueError("pairs must be positive")
            seeds = tuple(range(seed_start, seed_start + pairs))
        else:
            if not isinstance(raw_seeds, list):
                raise ValueError("game_seeds must be a JSON array")
            seeds = tuple(raw_seeds)
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            policy_a=SerializablePolicySpec.from_dict(raw_a),
            policy_b=SerializablePolicySpec.from_dict(raw_b),
            game_seeds=seeds,
            policy_seed_base=int(value.get("policy_seed_base", 0)),
            workers=int(value.get("workers", 1)),
            device=str(value.get("device", "cpu")),
            max_completed_turns=int(value.get("max_completed_turns", 400)),
            adjudicate_deterministic_policy_cycles=bool(
                value.get("adjudicate_deterministic_policy_cycles", True)
            ),
            bootstrap_resamples=int(value.get("bootstrap_resamples", 0)),
            bootstrap_seed=int(value.get("bootstrap_seed", 0)),
            metadata=dict(metadata),
            execution=None if value.get("execution") is None else dict(value["execution"]),
        )


@dataclass(frozen=True, slots=True)
class FixedEvaluationRun:
    """A resumable fixed matchup result plus its durable per-pair records."""

    summary: dict[str, Any]
    pair_records: dict[int, list[dict[str, Any]]]
    output_path: Path
    game_log_root: Path


def _random_factory(seed: int) -> Policy:
    return RandomPolicy(seed)


def _score_factory(seed: int) -> Policy:
    del seed
    return ScorePolicy()


def _development_factory(seed: int) -> Policy:
    del seed
    return DevelopmentPolicy()


_BASELINES: dict[str, PolicyFactory] = {
    "random": _random_factory,
    "score": _score_factory,
    "development": _development_factory,
}


def baseline_policy_spec(name: str) -> PolicySpec:
    try:
        factory = _BASELINES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_BASELINES))
        raise ValueError(f"unknown baseline policy {name!r}; choose one of: {choices}") from error
    return PolicySpec(name, factory, SerializablePolicySpec.baseline(name))


def baseline_policy_names() -> tuple[str, ...]:
    return tuple(sorted(_BASELINES))


def evaluate_policies(
    policy_a: PolicySpec,
    policy_b: PolicySpec,
    game_seeds: Sequence[int],
    *,
    policy_seed_base: int = 0,
    max_completed_turns: int = 400,
    failure_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    adjudicate_deterministic_policy_cycles: bool = False,
) -> EvaluationResult:
    """Evaluate paired games: A starts once and B starts once per game seed."""

    seeds = tuple(game_seeds)
    if not seeds:
        raise ValueError("at least one game seed is required")
    if len(set(seeds)) != len(seeds) or not all(type(seed) is int for seed in seeds):
        raise ValueError("game seeds must be unique integers")
    if max_completed_turns <= 0:
        raise ValueError("max_completed_turns must be positive")
    failure_dir = failure_dir or Path.cwd() / "evaluation_failures"

    matches: list[MatchResult] = []
    for pair_id, game_seed in enumerate(seeds):
        policy_seeds = (
            policy_seed_base + pair_id * 2,
            policy_seed_base + pair_id * 2 + 1,
        )
        for game_in_pair, first_player in enumerate((0, 1)):
            matches.append(
                run_match(
                    policy_a.create(policy_seeds[0]),
                    policy_b.create(policy_seeds[1]),
                    game_seed=game_seed,
                    policy_seeds=policy_seeds,
                    first_player=first_player,
                    policy_names=(policy_a.name, policy_b.name),
                    pair_id=pair_id,
                    game_in_pair=game_in_pair,
                    max_completed_turns=max_completed_turns,
                    failure_dir=failure_dir / f"pair-{pair_id:06d}-game-{game_in_pair}",
                    adjudicate_deterministic_policy_cycles=(
                        adjudicate_deterministic_policy_cycles
                    ),
                    raise_on_error=False,
                    lean=True,
                )
            )
        if progress_callback is not None:
            progress_callback(pair_id + 1, len(seeds))

    return _evaluation_from_matches(policy_a.name, policy_b.name, seeds, tuple(matches))


@dataclass(frozen=True, slots=True)
class ParallelEvaluation:
    result: EvaluationResult
    policy_a_stats: dict[str, int | float]
    policy_b_stats: dict[str, int | float]


class ParallelEvaluationPool:
    """Reusable spawn pool so a multi-opponent panel loads checkpoints once."""

    def __init__(self, workers: int) -> None:
        if workers <= 1:
            raise ValueError("a reusable parallel pool requires at least two workers")
        self.workers = workers
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_parallel_worker_init,
        )

    def map(
        self, tasks: tuple[_ParallelPairTask, ...]
    ) -> Sequence[ParallelPairResult]:
        return self._executor.map(_run_parallel_pair, tasks, chunksize=1)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> ParallelEvaluationPool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _ParallelPairTask:
    policy_a: SerializablePolicySpec
    policy_b: SerializablePolicySpec
    pair_id: int
    game_seed: int
    policy_seed_base: int
    max_completed_turns: int
    failure_dir: str
    adjudicate_deterministic_policy_cycles: bool
    game_log_dir: str | None = None


@dataclass(frozen=True, slots=True)
class ParallelPairResult:
    """One finished paired seed: both seat-swapped games and per-policy stats."""

    matches: tuple[MatchResult, MatchResult]
    policy_a_stats: dict[str, int | float]
    policy_b_stats: dict[str, int | float]


class EvaluationGameStore:
    """Durable JSONL storage shared by fixed evaluations and experiment drivers.

    A flat store keeps pair-*.jsonl directly under root, which is convenient
    for one fixed matchup. A hierarchical store adds stage/block directories
    for orchestrators such as the depth ladder; the game-record format and
    resume checks remain the same in both layouts.
    """

    def __init__(
        self,
        root: Path,
        header: dict[str, Any] | None = None,
        *,
        flat: bool = False,
    ) -> None:
        self.root = Path(root)
        self.header = dict(header or {})
        self.flat = flat
        self._cache: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = {}

    @property
    def header_path(self) -> Path:
        return self.root / "header.json"

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.header_path, self.header)

    def load(self) -> None:
        self.header = json.loads(self.header_path.read_text(encoding="utf-8"))

    def prepare(
        self,
        expected_header: dict[str, Any],
        *,
        resume: bool,
        allow_legacy_resume: bool = False,
    ) -> None:
        """Create a manifest, or validate an existing one before resuming."""

        self.root.mkdir(parents=True, exist_ok=True)
        has_pair_logs = any(self.root.glob("pair-*.jsonl"))
        has_data = any(self.root.iterdir())
        if self.header_path.exists():
            if not resume:
                raise ValueError(
                    f"{self.root} already contains an evaluation; pass resume=True"
                )
            self.load()
            if self.header != expected_header:
                raise ValueError(
                    "existing evaluation was created with a different configuration"
                )
            return
        if has_data and not resume:
            raise ValueError(
                f"{self.root} already contains data; pass resume=True or choose a new path"
            )
        if has_pair_logs and resume and not allow_legacy_resume:
            raise ValueError(
                f"{self.root} has game logs without an evaluation manifest"
            )
        if has_pair_logs and resume and allow_legacy_resume and "catalog_fingerprint" in expected_header:
            raise ValueError("legacy evaluation logs have no frozen catalog fingerprint")
        if has_data and resume and not (has_pair_logs and allow_legacy_resume):
            raise ValueError(f"{self.root} cannot be resumed without its manifest")
        self.header = expected_header
        self.start()

    def stage_dir(self, index: int) -> Path:
        return self.root / f"stage-{index:03d}"

    def block_dir(self, index: int, block: int) -> Path:
        return self.stage_dir(index) / f"b{block:02d}"

    def _record_dir(self, index: int, block: int) -> Path:
        return self.root if self.flat else self.block_dir(index, block)

    def invalidate(self, index: int | None = None, block: int | None = None) -> None:
        if self.flat or index is None or block is None:
            self._cache.clear()
        else:
            self._cache.pop((index, block), None)

    def clear_cache(self) -> None:
        self._cache.clear()

    def games(
        self, index: int = 0, block: int = 0
    ) -> dict[int, list[dict[str, Any]]]:
        key = (-1, -1) if self.flat else (index, block)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        games: dict[int, list[dict[str, Any]]] = {}
        directory = self._record_dir(index, block)
        if directory.is_dir():
            for path in sorted(directory.glob("pair-*.jsonl")):
                try:
                    pair_id = int(path.stem.split("-")[1])
                except (IndexError, ValueError):  # pragma: no cover - defensive
                    continue
                records: list[dict[str, Any]] = []
                for raw_line in path.read_bytes().splitlines():
                    if not raw_line.strip():
                        continue
                    try:
                        line = raw_line.decode("utf-8")
                        records.append(json.loads(line))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # A hard stop can leave only the final JSONL line partial.
                        break
                games[pair_id] = records
        self._cache[key] = games
        return games

    @staticmethod
    def _is_scored_pair(records: Sequence[dict[str, Any]]) -> bool:
        ordered = sorted(records, key=lambda record: record.get("game_in_pair", -1))
        return (
            len(ordered) == 2
            and [record.get("game_in_pair") for record in ordered] == [0, 1]
            and ordered[0].get("game_seed") == ordered[1].get("game_seed")
            and all(
                record.get("status")
                in (RunStatus.NATURAL_TERMINAL.value, RunStatus.POLICY_CYCLE_DRAW.value)
                and record.get("outcome_for_a")
                in (MatchOutcome.A_WIN.value, MatchOutcome.DRAW.value, MatchOutcome.A_LOSS.value)
                for record in ordered
            )
        )

    def complete_pairs(
        self,
        index: int = 0,
        block: int = 0,
        *,
        require_scored: bool = False,
    ) -> dict[int, list[dict[str, Any]]]:
        complete: dict[int, list[dict[str, Any]]] = {}
        for pair_id, records in self.games(index, block).items():
            ordered = sorted(records, key=lambda record: record.get("game_in_pair", -1))
            is_complete = (
                len(ordered) == 2
                and [record.get("game_in_pair") for record in ordered] == [0, 1]
                and ordered[0].get("game_seed") == ordered[1].get("game_seed")
            )
            if is_complete and (not require_scored or self._is_scored_pair(ordered)):
                complete[pair_id] = ordered
        return complete

    def scored_pairs(
        self, index: int = 0, block: int = 0
    ) -> dict[int, list[dict[str, Any]]]:
        """Pairs with both seat-swapped games finished and scored."""

        return self.complete_pairs(index, block, require_scored=True)

    def discard_pair(
        self, index: int, block: int, pair_id: int, *, archive_dir: Path | None = None
    ) -> None:
        path = self._record_dir(index, block) / f"pair-{pair_id:07d}.jsonl"
        if path.exists():
            if archive_dir is None:
                path.unlink()
            else:
                archive_dir.mkdir(parents=True, exist_ok=True)
                path.replace(archive_dir / path.name)
        self.invalidate(index, block)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


_PARALLEL_MODEL_CACHE: dict[tuple[str, str], object] = {}
_PARALLEL_MODEL_HASH_CACHE: dict[str, str] = {}
_PARALLEL_POLICY_DEVICE = "cpu"


def _parallel_worker_init() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work.
        pass


def _create_serializable_policy(
    spec: SerializablePolicySpec,
    *,
    seed: int,
    device: str = "cpu",
) -> Policy:
    if spec.kind == "baseline":
        if spec.baseline_name is None:
            raise AssertionError("baseline policy lost its name")
        return baseline_policy_spec(spec.baseline_name).create(seed)

    if spec.checkpoint is None:
        raise AssertionError("checkpoint-backed policy lost its checkpoint")
    cache_key = (spec.checkpoint, device)
    model = _PARALLEL_MODEL_CACHE.get(cache_key)
    if model is None:
        from .opponent_pool import load_frozen_actor_critic_checkpoint

        model = load_frozen_actor_critic_checkpoint(
            Path(spec.checkpoint), device=device
        )
        _PARALLEL_MODEL_CACHE[cache_key] = model
    if spec.kind == "greedy_checkpoint":
        from .neural_policy import GreedyActorCriticPolicy

        return GreedyActorCriticPolicy(model, device=device)  # type: ignore[arg-type]

    if spec.search_config is None:
        raise AssertionError("search policy lost its configuration")
    values = dict(spec.search_config)
    values["seed"] = int(values.get("seed", 0)) ^ seed
    evaluator = None
    if spec.inference is not None:
        from .search_inference import FrozenSearchInference, InferenceConfig

        evaluator = FrozenSearchInference(
            model, device=device, config=InferenceConfig(**spec.inference)  # type: ignore[arg-type]
        )
    if spec.kind == "mcts_checkpoint":
        from .mcts_policy import MCTSConfig, PUCTSearchPolicy
        if spec.search_backend in ("native", "native_chance"):
            from .native_policy import NativePUCTSearchPolicy, NativeChancePUCTSearchPolicy
            policy_cls = NativeChancePUCTSearchPolicy if spec.search_backend == "native_chance" else NativePUCTSearchPolicy
        else:
            policy_cls = PUCTSearchPolicy

        kwargs = {}
        if spec.chance_widening is not None:
            from .native_policy import ChanceWideningConfig
            kwargs["chance_widening"] = ChanceWideningConfig(**spec.chance_widening)
        if spec.search_backend in ("native", "native_chance"):
            version = _PARALLEL_MODEL_HASH_CACHE.get(spec.checkpoint)
            if version is None:
                version = hashlib.sha256(Path(spec.checkpoint).read_bytes()).hexdigest()
                _PARALLEL_MODEL_HASH_CACHE[spec.checkpoint] = version
            kwargs["model_version"] = version
        return policy_cls(
            model,  # type: ignore[arg-type]
            config=MCTSConfig(**values),
            device=device, evaluator=evaluator, **kwargs,
        )
    from .search_policy import PolicyValueSearchPolicy, SearchConfig

    return PolicyValueSearchPolicy(
        model,  # type: ignore[arg-type]
        config=SearchConfig(**values),
        device=device, evaluator=evaluator,
    )


def _numeric_policy_stats(policy: Policy) -> dict[str, int | float]:
    stats = getattr(policy, "stats", None)
    if stats is None:
        return {}
    values = stats.to_dict()
    return {
        str(key): value
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def _sum_numeric_stats(
    target: dict[str, int | float], values: dict[str, int | float]
) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def _game_log_record(
    match: MatchResult,
    a_stats: dict[str, int | float],
    b_stats: dict[str, int | float],
) -> dict[str, object]:
    return {
        "pair_id": match.pair_id,
        "game_in_pair": match.game_in_pair,
        "game_seed": match.game_seed,
        "first_player": match.first_player,
        "status": match.status.value,
        "normal_end": match.normal_end,
        "outcome_for_a": (
            None if match.outcome_for_a is None else match.outcome_for_a.value
        ),
        "winner_id": match.winner_id,
        "draw": match.draw,
        "scores": None if match.scores is None else list(match.scores),
        "purchased_card_counts": (
            None
            if match.purchased_card_counts is None
            else list(match.purchased_card_counts)
        ),
        "terminal_reason": (
            None
            if match.terminal_result is None
            else match.terminal_result.reason.value
        ),
        "error_type": match.error_type,
        "error_detail": match.error_detail,
        "policy_names": list(match.policy_names),
        "policy_seeds": list(match.policy_seeds),
        "completed_turns": list(match.completed_turns),
        "completed_player_turns": match.completed_player_turns,
        "decision_count": match.decision_count,
        "forced_skips": match.forced_skips,
        "final_state_digest": match.final_state_digest,
        "failure_scene": (
            None if match.failure_scene is None else str(match.failure_scene)
        ),
        "policy_a_stats": dict(a_stats),
        "policy_b_stats": dict(b_stats),
    }


def _append_game_log(task: _ParallelPairTask, record: dict[str, object]) -> None:
    """Durably record one finished game so a stop never loses played games."""

    if task.game_log_dir is None:
        return
    directory = Path(task.game_log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"pair-{task.pair_id:07d}.jsonl"
    line = json.dumps(record, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        # Pool workers can finish the two seats of one pair concurrently.
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_parallel_pair(task: _ParallelPairTask) -> ParallelPairResult:
    policy_seeds = (
        task.policy_seed_base + task.pair_id * 2,
        task.policy_seed_base + task.pair_id * 2 + 1,
    )
    matches: list[MatchResult] = []
    a_stats: dict[str, int | float] = {}
    b_stats: dict[str, int | float] = {}
    for game_in_pair, first_player in enumerate((0, 1)):
        policy_a = _create_serializable_policy(
            task.policy_a, seed=policy_seeds[0], device=_PARALLEL_POLICY_DEVICE
        )
        policy_b = _create_serializable_policy(
            task.policy_b, seed=policy_seeds[1], device=_PARALLEL_POLICY_DEVICE
        )
        match = run_match(
            policy_a,
            policy_b,
            game_seed=task.game_seed,
            policy_seeds=policy_seeds,
            first_player=first_player,
            policy_names=(task.policy_a.name, task.policy_b.name),
            pair_id=task.pair_id,
            game_in_pair=game_in_pair,
            max_completed_turns=task.max_completed_turns,
            failure_dir=(
                Path(task.failure_dir)
                / f"pair-{task.pair_id:06d}-game-{game_in_pair}"
            ),
            adjudicate_deterministic_policy_cycles=(
                task.adjudicate_deterministic_policy_cycles
            ),
            raise_on_error=False,
            lean=True,
        )
        matches.append(match)
        game_a_stats = _numeric_policy_stats(policy_a)
        game_b_stats = _numeric_policy_stats(policy_b)
        _sum_numeric_stats(a_stats, game_a_stats)
        _sum_numeric_stats(b_stats, game_b_stats)
        _append_game_log(
            task, _game_log_record(match, game_a_stats, game_b_stats)
        )
    return ParallelPairResult(
        matches=(matches[0], matches[1]),
        policy_a_stats=a_stats,
        policy_b_stats=b_stats,
    )


def evaluate_policies_parallel(
    policy_a: SerializablePolicySpec,
    policy_b: SerializablePolicySpec,
    game_seeds: Sequence[int],
    *,
    workers: int,
    device: str = "cpu",
    policy_seed_base: int = 0,
    max_completed_turns: int = 400,
    failure_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    adjudicate_deterministic_policy_cycles: bool = False,
    pool: ParallelEvaluationPool | None = None,
    pair_ids: Sequence[int] | None = None,
    pair_callback: Callable[[ParallelPairResult], None] | None = None,
    game_log_dir: Path | str | None = None,
) -> ParallelEvaluation:
    """Evaluate independent paired seeds in long-lived spawn workers.

    Neural checkpoints are loaded once per worker. Multi-worker inference is CPU-only;
    a single-worker evaluation can select either CPU or CUDA.
    A pair remains the scheduling unit, so its two seat-swapped games cannot be
    split across workers and pair-level statistics retain their original order.

    ``pair_callback`` fires in the parent process as soon as one pair finishes,
    which lets long runs persist completed pairs instead of waiting for the
    whole panel.  ``pair_ids`` keeps the original pair identity when only a
    subset of seeds is replayed (resume), so policy seeds stay reproducible.
    ``game_log_dir`` additionally makes every *single finished game* durable
    from inside the worker (one fsynced line per game), so stopping the run
    never loses played games.
    """

    global _PARALLEL_POLICY_DEVICE
    seeds = tuple(game_seeds)
    if workers <= 0:
        raise ValueError("parallel evaluation workers must be positive")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if workers > 1 and device != "cpu":
        raise ValueError("parallel evaluation requires CPU inference")
    if not seeds:
        raise ValueError("at least one game seed is required")
    if len(set(seeds)) != len(seeds) or not all(type(seed) is int for seed in seeds):
        raise ValueError("game seeds must be unique integers")
    if max_completed_turns <= 0:
        raise ValueError("max_completed_turns must be positive")
    if pair_ids is None:
        resolved_pair_ids = tuple(range(len(seeds)))
    else:
        resolved_pair_ids = tuple(pair_ids)
        if len(resolved_pair_ids) != len(seeds):
            raise ValueError("pair_ids must match the number of game seeds")
        if len(set(resolved_pair_ids)) != len(resolved_pair_ids):
            raise ValueError("pair_ids must be unique")
    _PARALLEL_POLICY_DEVICE = device
    root = failure_dir or Path.cwd() / "evaluation_failures"
    tasks = tuple(
        _ParallelPairTask(
            policy_a=policy_a,
            policy_b=policy_b,
            pair_id=pair_id,
            game_seed=game_seed,
            policy_seed_base=policy_seed_base,
            max_completed_turns=max_completed_turns,
            failure_dir=str(root),
            adjudicate_deterministic_policy_cycles=(
                adjudicate_deterministic_policy_cycles
            ),
            game_log_dir=None if game_log_dir is None else str(game_log_dir),
        )
        for pair_id, game_seed in zip(resolved_pair_ids, seeds, strict=True)
    )
    pair_results: list[ParallelPairResult] = []

    def record(pair_result: ParallelPairResult) -> None:
        pair_results.append(pair_result)
        if pair_callback is not None:
            pair_callback(pair_result)
        if progress_callback is not None:
            progress_callback(len(pair_results), len(tasks))

    if workers == 1:
        _parallel_worker_init()
        for task in tasks:
            record(_run_parallel_pair(task))
    elif pool is not None:
        if pool.workers != workers:
            raise ValueError("parallel pool worker count does not match request")
        for pair_result in pool.map(tasks):
            record(pair_result)
    else:
        context = mp.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_parallel_worker_init,
        )
        try:
            for pair_result in executor.map(_run_parallel_pair, tasks, chunksize=1):
                record(pair_result)
        except BaseException:
            # ProcessPoolExecutor's context manager waits for every running
            # task on exit. On Ctrl-C that can leave a long evaluation running
            # for hours, so explicitly stop its owned workers first.
            # Keep process handles before shutdown clears executor state.
            worker_processes = tuple((executor._processes or {}).values())
            executor.shutdown(wait=False, cancel_futures=True)
            for process in worker_processes:
                try:
                    if process.is_alive():
                        process.terminate()
                except OSError:
                    pass
            for process in worker_processes:
                process.join(timeout=2)
                try:
                    if process.is_alive():
                        process.kill()
                except OSError:
                    pass
                process.join()
            raise
        else:
            executor.shutdown(wait=True)

    matches = tuple(
        match for pair_result in pair_results for match in pair_result.matches
    )
    a_stats: dict[str, int | float] = {}
    b_stats: dict[str, int | float] = {}
    for pair_result in pair_results:
        _sum_numeric_stats(a_stats, pair_result.policy_a_stats)
        _sum_numeric_stats(b_stats, pair_result.policy_b_stats)
    return ParallelEvaluation(
        result=_evaluation_from_matches(policy_a.name, policy_b.name, seeds, matches),
        policy_a_stats=a_stats,
        policy_b_stats=b_stats,
    )


def _evaluation_from_matches(
    policy_a: str,
    policy_b: str,
    seeds: tuple[int, ...],
    frozen_matches: tuple[MatchResult, ...],
) -> EvaluationResult:
    valid = tuple(
        match
        for match in frozen_matches
        if (
            match.status is RunStatus.NATURAL_TERMINAL
            and match.terminal_result is not None
        )
        or (
            match.status is RunStatus.POLICY_CYCLE_DRAW
            and match.outcome_for_a is MatchOutcome.DRAW
        )
    )
    a_wins = sum(match.outcome_for_a is MatchOutcome.A_WIN for match in valid)
    draws = sum(match.outcome_for_a is MatchOutcome.DRAW for match in valid)
    a_losses = sum(match.outcome_for_a is MatchOutcome.A_LOSS for match in valid)
    lengths = tuple(match.completed_player_turns for match in valid)
    decisions = tuple(match.decision_count for match in valid)
    return EvaluationResult(
        policy_a=policy_a,
        policy_b=policy_b,
        game_seeds=seeds,
        total_pairs=len(seeds),
        total_games=len(frozen_matches),
        valid_games=len(valid),
        a_wins=a_wins,
        draws=draws,
        a_losses=a_losses,
        win_rate_score=_win_rate(a_wins, draws, len(valid)),
        a_first=_outcome_breakdown(
            tuple(match for match in frozen_matches if match.first_player == 0)
        ),
        a_second=_outcome_breakdown(
            tuple(match for match in frozen_matches if match.first_player == 1)
        ),
        mean_completed_player_turns=(sum(lengths) / len(lengths) if lengths else None),
        median_completed_player_turns=(float(median(lengths)) if lengths else None),
        mean_decisions=(sum(decisions) / len(decisions) if decisions else None),
        forced_skips=sum(match.forced_skips for match in frozen_matches),
        engineering_truncations=sum(
            match.status is RunStatus.ENGINEERING_TRUNCATION for match in frozen_matches
        ),
        exceptions=sum(match.status is RunStatus.ERROR for match in frozen_matches),
        illegal_actions=sum(match.error_type == "IllegalActionError" for match in frozen_matches),
        invariant_failures=sum(match.error_type == "InvariantViolation" for match in frozen_matches),
        matches=frozen_matches,
        policy_cycle_draws=sum(
            match.status is RunStatus.POLICY_CYCLE_DRAW for match in frozen_matches
        ),
    )


def save_evaluation(path: Path, result: EvaluationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_identity(spec: SerializablePolicySpec) -> dict[str, object] | None:
    if spec.checkpoint is None:
        return None
    checkpoint = Path(spec.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(checkpoint), "sha256": digest.hexdigest()}


def _scored_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("status")
        in (RunStatus.NATURAL_TERMINAL.value, RunStatus.POLICY_CYCLE_DRAW.value)
        and record.get("outcome_for_a")
        in (MatchOutcome.A_WIN.value, MatchOutcome.DRAW.value, MatchOutcome.A_LOSS.value)
    )


def _stats_from_records(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, int | float]:
    total: dict[str, int | float] = {}
    for record in records:
        values = record.get(key, {})
        if not isinstance(values, Mapping):
            continue
        for name, value in values.items():
            if isinstance(value, (int, float)):
                total[str(name)] = total.get(str(name), 0) + value
    return total


def _evaluation_summary_from_records(
    config: FixedEvaluationConfig,
    records_by_pair: Mapping[int, Sequence[dict[str, Any]]],
    *,
    output_path: Path,
    game_log_root: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    all_records = [
        record
        for pair_id in sorted(records_by_pair)
        for record in sorted(
            records_by_pair[pair_id], key=lambda value: value.get("game_in_pair", -1)
        )
    ]
    valid = [record for record in all_records if _scored_record(record)]
    wins = sum(record.get("outcome_for_a") == MatchOutcome.A_WIN.value for record in valid)
    draws = sum(record.get("outcome_for_a") == MatchOutcome.DRAW.value for record in valid)
    losses = sum(record.get("outcome_for_a") == MatchOutcome.A_LOSS.value for record in valid)
    expected_pairs = len(config.game_seeds)
    complete_pairs = sum(
        len(records_by_pair.get(pair_id, ())) == 2
        and [record.get("game_in_pair") for record in sorted(
            records_by_pair[pair_id], key=lambda value: value.get("game_in_pair", -1)
        )] == [0, 1]
        for pair_id in range(expected_pairs)
    )
    expected_games = expected_pairs * 2
    outcomes_by_seat: dict[str, dict[str, int]] = {}
    for key, first_player in (("a_first", 0), ("a_second", 1)):
        seat_records = [record for record in all_records if record.get("first_player") == first_player]
        seat_valid = [record for record in seat_records if _scored_record(record)]
        seat_wins = sum(record.get("outcome_for_a") == MatchOutcome.A_WIN.value for record in seat_valid)
        seat_draws = sum(record.get("outcome_for_a") == MatchOutcome.DRAW.value for record in seat_valid)
        seat_losses = sum(record.get("outcome_for_a") == MatchOutcome.A_LOSS.value for record in seat_valid)
        outcomes_by_seat[key] = {
            "games": len(seat_records),
            "valid_games": len(seat_valid),
            "wins": seat_wins,
            "draws": seat_draws,
            "losses": seat_losses,
            "win_rate_score": (
                (seat_wins + 0.5 * seat_draws) / len(seat_valid) if seat_valid else None
            ),
        }
    pair_scores: list[float] = []
    for pair_id in range(expected_pairs):
        games = sorted(
            records_by_pair.get(pair_id, ()),
            key=lambda value: value.get("game_in_pair", -1),
        )
        if len(games) == 2 and all(_scored_record(game) for game in games):
            pair_scores.append(
                sum(
                    1.0 if game["outcome_for_a"] == MatchOutcome.A_WIN.value
                    else 0.5 if game["outcome_for_a"] == MatchOutcome.DRAW.value
                    else 0.0
                    for game in games
                ) / 2.0
            )
    bootstrap = None
    if (
        config.bootstrap_resamples
        and pair_scores
        and complete_pairs == expected_pairs
    ):
        from .paired_statistics import paired_bootstrap_mean

        bootstrap = paired_bootstrap_mean(
            pair_scores,
            resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed,
        ).to_dict()
    turns = [record.get("completed_player_turns", 0) for record in valid]
    decisions = [record.get("decision_count", 0) for record in valid]
    truncations = sum(
        record.get("status") == RunStatus.ENGINEERING_TRUNCATION.value
        for record in all_records
    )
    exceptions = sum(record.get("status") == RunStatus.ERROR.value for record in all_records)
    illegal_actions = sum(record.get("error_type") == "IllegalActionError" for record in all_records)
    invariant_failures = sum(record.get("error_type") == "InvariantViolation" for record in all_records)
    pair_list = [
        {
            "pair_id": pair_id,
            "game_seed": seed,
            "games": sorted(
                records_by_pair.get(pair_id, ()),
                key=lambda value: value.get("game_in_pair", -1),
            ),
        }
        for pair_id, seed in enumerate(config.game_seeds)
    ]
    passed = (
        complete_pairs == expected_pairs
        and len(valid) == expected_games
        and truncations == 0
        and exceptions == 0
        and illegal_actions == 0
        and invariant_failures == 0
    )
    return {
        "protocol": "fixed-paired-evaluation-v1",
        "catalog_fingerprint": load_default_catalog().data_fingerprint,
        "config": config.to_dict(),
        "runtime": {"workers": config.workers, "device": config.device},
        "output": str(output_path.resolve()),
        "game_log_root": str(game_log_root.resolve()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "total_pairs": expected_pairs,
        "completed_pairs": complete_pairs,
        "total_games": expected_games,
        "completed_games": len(all_records),
        "valid_games": len(valid),
        "a_wins": wins,
        "draws": draws,
        "a_losses": losses,
        "win_rate_score": (wins + 0.5 * draws) / len(valid) if valid else None,
        **outcomes_by_seat,
        "mean_completed_player_turns": (
            sum(turns) / len(turns) if turns else None
        ),
        "median_completed_player_turns": (
            float(median(turns)) if turns else None
        ),
        "mean_decisions": sum(decisions) / len(decisions) if decisions else None,
        "forced_skips": sum(record.get("forced_skips", 0) for record in all_records),
        "engineering_truncations": truncations,
        "exceptions": exceptions,
        "illegal_actions": illegal_actions,
        "invariant_failures": invariant_failures,
        "policy_cycle_draws": sum(
            record.get("status") == RunStatus.POLICY_CYCLE_DRAW.value
            for record in all_records
        ),
        "policy_a_stats": _stats_from_records(all_records, "policy_a_stats"),
        "policy_b_stats": _stats_from_records(all_records, "policy_b_stats"),
        "paired_bootstrap": bootstrap,
        "paired_bootstrap_unit": "one_paired_seed",
        "passed": passed,
        "pairs": pair_list,
    }


def _lock_evaluation_directory(root: Path) -> int:
    """Prevent two resumptions from scheduling the same missing pairs."""

    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(f"evaluation is already active in {root}") from error
    return descriptor


def _unlock_evaluation_directory(descriptor: int) -> None:
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def run_fixed_evaluation(
    config: FixedEvaluationConfig,
    output_path: Path,
    *,
    store_dir: Path | None = None,
    failure_dir: Path | None = None,
    resume: bool = False,
    allow_legacy_resume: bool = False,
    require_scored_pairs: bool = False,
    progress_every: int = 1,
    summary_every: int = 25,
    progress_callback: Callable[[int, int], None] | None = None,
    pair_callback: Callable[[ParallelPairResult], None] | None = None,
) -> FixedEvaluationRun:
    """Run, persist, summarize, and optionally resume one fixed matchup.

    Search and greedy opponents are declared as SerializablePolicySpec values;
    this runner only constructs and calls their policy interface. A completed
    paired seed is stored only after both seat-swapped games have durable JSONL
    records. Incomplete pairs are replayed with their original pair and policy
    seeds, while complete pairs are skipped on resume.
    """

    if progress_every <= 0 or summary_every <= 0:
        raise ValueError("progress_every and summary_every must be positive")
    output_path = Path(output_path)
    root = Path(store_dir) if store_dir is not None else Path(
        str(output_path) + ".games"
    )
    root_was_present = root.exists()
    lock_descriptor = _lock_evaluation_directory(root)
    try:
        if output_path.exists() and not resume:
            raise ValueError(f"{output_path} already exists; pass resume=True or choose a new path")
        if resume and output_path.exists() and not root_was_present:
            raise ValueError("cannot resume a summary without its game-log directory")
        if config.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")

        identities = {
            "policy_a": _checkpoint_identity(config.policy_a),
            "policy_b": _checkpoint_identity(config.policy_b),
        }
        manifest = {
            "protocol": "fixed-paired-evaluation-v1",
            "catalog_fingerprint": load_default_catalog().data_fingerprint,
            "config": config.to_dict(),
            "checkpoint_identity": identities,
        }
        store = EvaluationGameStore(root, manifest, flat=True)
        store.prepare(manifest, resume=resume, allow_legacy_resume=allow_legacy_resume)
        store.invalidate()

        selected_ids = tuple(range(len(config.game_seeds)))
        existing = store.games()
        expected_names = (config.policy_a.name, config.policy_b.name)
        for pair_id, game_records in existing.items():
            if not 0 <= pair_id < len(config.game_seeds):
                raise ValueError(f"stored pair id {pair_id} is outside this evaluation")
            expected_policy_seeds = (
                config.policy_seed_base + pair_id * 2,
                config.policy_seed_base + pair_id * 2 + 1,
            )
            for record in game_records:
                game_index = record.get("game_in_pair")
                if game_index not in (0, 1):
                    raise ValueError(f"stored pair {pair_id} has an invalid game slot")
                if record.get("pair_id") != pair_id:
                    raise ValueError(f"stored game does not match pair file {pair_id}")
                if record.get("game_seed") != config.game_seeds[pair_id]:
                    raise ValueError(f"stored pair {pair_id} has a different game seed")
                if record.get("first_player") != game_index:
                    raise ValueError(f"stored pair {pair_id} has an invalid seat swap")
                if record.get("policy_names") != list(expected_names):
                    raise ValueError(f"stored pair {pair_id} has different policy names")
                if record.get("policy_seeds") != list(expected_policy_seeds):
                    raise ValueError(f"stored pair {pair_id} has different policy seeds")
        complete = store.complete_pairs(require_scored=require_scored_pairs)
        retry_archive = None
        for pair_id in selected_ids:
            if pair_id not in complete and pair_id in existing:
                # Discard a partial pair before replay to keep one authoritative
                # record per game slot and avoid duplicate lines after a restart.
                # Scored-only recovery also retries engineering failures; keep
                # their original pair logs and summary for diagnosis.
                if require_scored_pairs and retry_archive is None:
                    retry_archive = root / "retries" / str(time.time_ns())
                    retry_archive.mkdir(parents=True)
                    if output_path.exists():
                        shutil.copyfile(output_path, retry_archive / "summary.json")
                store.discard_pair(0, 0, pair_id, archive_dir=retry_archive)
        store.invalidate()
        complete = store.complete_pairs(require_scored=require_scored_pairs)
        missing_ids = tuple(pair_id for pair_id in selected_ids if pair_id not in complete)
        existing_count = len(complete)
        previous_summary = (json.loads(output_path.read_text()) if resume and output_path.exists() else {})
        started = time.perf_counter()
        pair_callbacks = 0
        execution_stats = {}

        def save_summary() -> dict[str, Any]:
            store.invalidate()
            records = store.games()
            summary = _evaluation_summary_from_records(
                config,
                records,
                output_path=output_path,
                game_log_root=root,
                elapsed_seconds=time.perf_counter() - started,
            )
            summary["checkpoint_identity"] = identities
            if execution_stats:
                summary["execution_stats"] = execution_stats
            atomic_write_json(output_path, summary)
            return summary

        summary = save_summary()
        if progress_callback is not None and (
            existing_count % progress_every == 0 or existing_count == len(selected_ids)
        ):
            progress_callback(existing_count, len(selected_ids))

        if missing_ids:
            missing_seeds = tuple(config.game_seeds[pair_id] for pair_id in missing_ids)

            def on_pair(pair_result: ParallelPairResult) -> None:
                nonlocal pair_callbacks, summary
                if pair_callback is not None:
                    pair_callback(pair_result)
                pair_callbacks += 1
                completed_now = existing_count + pair_callbacks
                if progress_callback is not None and (
                    completed_now % progress_every == 0
                    or completed_now == len(selected_ids)
                ):
                    progress_callback(completed_now, len(selected_ids))
                if pair_callbacks % summary_every == 0:
                    summary = save_summary()

            try:
                if config.execution is not None:
                    if config.execution.get("backend") == "native_cuda_pool":
                        from .evaluation_pool import evaluate_pooled as evaluate_execution
                    else:
                        from .evaluation_batch import evaluate_batched as evaluate_execution
                    execution_stats.update(evaluate_execution(
                        config, missing_ids, root,
                        failure_dir or output_path.parent / "failures" / output_path.stem,
                        on_pair))
                else:
                    evaluate_policies_parallel(
                        config.policy_a,
                        config.policy_b,
                        missing_seeds,
                        workers=config.workers,
                        device=config.device,
                        policy_seed_base=config.policy_seed_base,
                        max_completed_turns=config.max_completed_turns,
                        failure_dir=failure_dir or output_path.parent / "failures" / output_path.stem,
                        progress_callback=None,
                        adjudicate_deterministic_policy_cycles=(
                            config.adjudicate_deterministic_policy_cycles
                        ),
                        pair_ids=missing_ids,
                        pair_callback=on_pair,
                        game_log_dir=root,
                    )
            except KeyboardInterrupt:
                # Per-game JSONL is authoritative; refresh the summary from
                # every complete line before returning control to the CLI.
                save_summary()
                raise

        store.invalidate()
        records = store.games()
        summary = _evaluation_summary_from_records(
            config,
            records,
            output_path=output_path,
            game_log_root=root,
            elapsed_seconds=time.perf_counter() - started,
        )
        summary["checkpoint_identity"] = identities
        if execution_stats:
            summary["execution_stats"] = execution_stats
        elif not missing_ids and "execution_stats" in previous_summary:
            summary["execution_stats"] = previous_summary["execution_stats"]
        atomic_write_json(output_path, summary)
        return FixedEvaluationRun(summary, records, output_path, root)
    finally:
        _unlock_evaluation_directory(lock_descriptor)


def _evaluation_config_from_file(path: Path) -> tuple[FixedEvaluationConfig, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation config must be a JSON object")
    config = FixedEvaluationConfig.from_dict(raw)
    return config, dict(raw)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paired Splendor evaluation from a JSON matchup config"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int)
    parser.add_argument("--summary-every", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    config, raw = _evaluation_config_from_file(args.config)
    progress_every = int(
        args.progress_every
        if args.progress_every is not None
        else raw.get("progress_every", 1)
    )
    summary_every = int(
        args.summary_every
        if args.summary_every is not None
        else raw.get("summary_every", 25)
    )

    def progress(completed: int, total: int) -> None:
        if completed % progress_every == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "event": "evaluation_progress",
                        "completed_pairs": completed,
                        "total_pairs": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    try:
        run = run_fixed_evaluation(
            config,
            args.output,
            resume=args.resume,
            progress_every=progress_every,
            summary_every=summary_every,
            progress_callback=progress,
        )
    except KeyboardInterrupt:
        print(
            json.dumps(
                {"event": "evaluation_interrupted", "summary_written": True},
                sort_keys=True,
            ),
            flush=True,
        )
        return 130
    print(
        json.dumps(
            {
                "event": "evaluation_completed",
                "passed": run.summary["passed"],
                "completed_pairs": run.summary["completed_pairs"],
                "total_pairs": run.summary["total_pairs"],
                "score_rate": run.summary["win_rate_score"],
                "output": str(args.output.resolve()),
                "game_log_root": str(run.game_log_root.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if run.summary["passed"] else 1


def _win_rate(wins: int, draws: int, valid_games: int) -> float | None:
    if valid_games == 0:
        return None
    return (wins + 0.5 * draws) / valid_games


def _outcome_breakdown(matches: tuple[MatchResult, ...]) -> OutcomeBreakdown:
    valid = tuple(
        match
        for match in matches
        if match.status in (RunStatus.NATURAL_TERMINAL, RunStatus.POLICY_CYCLE_DRAW)
        and match.outcome_for_a is not None
    )
    wins = sum(match.outcome_for_a is MatchOutcome.A_WIN for match in valid)
    draws = sum(match.outcome_for_a is MatchOutcome.DRAW for match in valid)
    losses = sum(match.outcome_for_a is MatchOutcome.A_LOSS for match in valid)
    return OutcomeBreakdown(
        games=len(matches),
        valid_games=len(valid),
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate_score=_win_rate(wins, draws, len(valid)),
    )


def _breakdown_dict(breakdown: OutcomeBreakdown) -> dict[str, object]:
    return {
        "games": breakdown.games,
        "valid_games": breakdown.valid_games,
        "wins": breakdown.wins,
        "draws": breakdown.draws,
        "losses": breakdown.losses,
        "win_rate_score": breakdown.win_rate_score,
    }


def _match_reference(match: MatchResult) -> dict[str, object]:
    terminal = match.terminal_result
    return {
        "pair_id": match.pair_id,
        "game_in_pair": match.game_in_pair,
        "game_seed": match.game_seed,
        "first_player": match.first_player,
        "policy_seeds": list(match.policy_seeds),
        "status": match.status.value,
        "outcome_for_a": None if match.outcome_for_a is None else match.outcome_for_a.value,
        "scores": None if match.scores is None else list(match.scores),
        "purchased_card_counts": (
            None if match.purchased_card_counts is None else list(match.purchased_card_counts)
        ),
        "completed_turns": list(match.completed_turns),
        "decision_count": match.decision_count,
        "forced_skips": match.forced_skips,
        "terminal_reason": None if terminal is None else terminal.reason.value,
        "replay": {
            "ruleset_id": match.replay.header.ruleset_id,
            "schema_version": match.replay.header.schema_version,
            "event_count": len(match.replay.events),
            "final_state_digest": match.final_state_digest,
        },
        "failure_scene": None if match.failure_scene is None else str(match.failure_scene),
        "error_type": match.error_type,
        "error_detail": match.error_detail,
    }


if __name__ == "__main__":
    raise SystemExit(main())
