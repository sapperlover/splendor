"""Fast and batch-cooperative MCTS self-play with v2 compact traces."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .compact_trace import replay_trace
from .features import FEATURE_SCHEMA_VERSION, PHASE_ORDER, encode_decision
from .match_stepper import MatchStepper
from .mcts_policy import MCTSConfig, MCTSSearchJob, PUCTSearchPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .replay import REPLAY_SCHEMA_VERSION, RULESET_ID, to_primitive
from .rng import RNG_ALGORITHM
from .search_inference import FrozenSearchInference
from .selfplay import TemperatureSchedule, derive_seed, label_terminal_values
from .selfplay_data import (
    DATA_SCHEMA_V2, STATE_FIELDS, commit_game, game_path, load_game,
    pack_game, validate_game_arrays,
)

PROTOCOL = "mcts-selfplay-v2"
SEARCH_PROTOCOL = "determinized-puct-v2-batched"
NUMERIC_VERSION = "frozen-fp32-card-merged-v1"
PROGRESS_POLL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    backend: str = "batched_cpu"
    engine_mode: str = "fast"
    workers: int = 1
    active_games_per_worker: int = 8
    max_batch_size: int = 16
    max_batch_wait_ms: float = 2.0
    torch_threads: int = 1
    precision: str = "float32"

    def __post_init__(self) -> None:
        if self.backend not in ("serial_cpu", "batched_cpu", "batched_cuda"):
            raise ValueError("backend must be serial_cpu, batched_cpu or batched_cuda")
        if self.engine_mode not in ("checked", "fast"):
            raise ValueError("engine_mode must be checked or fast")
        if any(type(getattr(self, name)) is not int or getattr(self, name) <= 0 for name in
               ("workers", "active_games_per_worker", "max_batch_size", "torch_threads")):
            raise ValueError("workers, active games, batch size and torch threads must be positive")
        if not math.isfinite(self.max_batch_wait_ms) or self.max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be finite and nonnegative")
        if self.precision != "float32":
            raise ValueError("only float32 inference is supported")
        if self.backend == "batched_cuda":
            if self.workers != 1:
                raise ValueError("batched_cuda uses one GPU owner; set execution.workers=1")
        if self.backend == "serial_cpu" and (self.active_games_per_worker != 1 or self.max_batch_size != 1):
            raise ValueError("serial_cpu requires active_games_per_worker=1 and max_batch_size=1")


@dataclass(frozen=True, slots=True)
class FastSelfPlayConfig:
    protocol: str
    run_id: str
    round_index: int
    checkpoint: str
    num_games: int
    seed: int
    search_config: MCTSConfig
    execution: ExecutionConfig
    add_root_noise: bool = True
    temperature_schedule: TemperatureSchedule = TemperatureSchedule()
    validation_fraction: float = 0.1
    split_seed: int = 0
    max_completed_turns: int = 400

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL or not self.run_id:
            raise ValueError("invalid v2 self-play protocol or run_id")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in
               ("round_index", "num_games", "max_completed_turns")):
            raise ValueError("round_index, num_games and max_completed_turns must be nonnegative integers")
        if type(self.seed) is not int or type(self.split_seed) is not int:
            raise ValueError("seed and split_seed must be integers")
        if self.num_games == 0 or self.max_completed_turns == 0:
            raise ValueError("num_games and max_completed_turns must be positive")
        if type(self.add_root_noise) is not bool:
            raise ValueError("add_root_noise must be bool")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(self.checkpoint)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FastSelfPlayConfig:
        return cls(
            protocol=str(raw["protocol"]), run_id=str(raw["run_id"]),
            round_index=raw["round_index"], checkpoint=str(Path(raw["checkpoint"]).resolve()),
            num_games=raw["num_games"], seed=raw["seed"],
            search_config=MCTSConfig(**raw["search_config"]),
            execution=ExecutionConfig(**raw["execution"]),
            add_root_noise=raw.get("add_root_noise", True),
            temperature_schedule=TemperatureSchedule(**raw.get("temperature_schedule", {})),
            validation_fraction=float(raw.get("validation_fraction", 0.1)),
            split_seed=raw.get("split_seed", 0),
            max_completed_turns=raw.get("max_completed_turns", 400),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _split(config: FastSelfPlayConfig, game_id: int) -> str:
    word = derive_seed(config.split_seed, config.round_index, game_id, "split")
    return "validation" if word / (1 << 64) < config.validation_fraction else "train"


def _search_seeds(config: FastSelfPlayConfig, game_id: int) -> tuple[int, int]:
    return tuple(config.search_config.seed ^ derive_seed(config.seed, config.round_index, game_id, "search_world", p) for p in (0, 1))  # type: ignore[return-value]


def build_manifest(config: FastSelfPlayConfig) -> dict[str, Any]:
    checkpoint = Path(config.checkpoint)
    checkpoint_identity = {"path": str(checkpoint), "sha256": _sha256(checkpoint)}
    versions = {"data": DATA_SCHEMA_V2, "search": SEARCH_PROTOCOL, "numeric": NUMERIC_VERSION,
                "rules": RULESET_ID, "replay": REPLAY_SCHEMA_VERSION,
                "rng": RNG_ALGORITHM, "feature": FEATURE_SCHEMA_VERSION,
                "seed": "splendor-selfplay-seed-v1"}
    frozen = {"config": config.to_dict(), "checkpoint_identity": checkpoint_identity, "versions": versions}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repository,
                                capture_output=True, text=True, check=True).stdout)
    return {"protocol": PROTOCOL, "semantic_fingerprint": fingerprint, **frozen,
            "source": {"git_revision": revision, "git_dirty": dirty}}


def _sample_record(session: _GameSession, result: object) -> dict[str, Any]:
    request = session.stepper.request
    encoded = session.encoded
    if encoded is None:
        raise AssertionError("missing root encoding")
    completed = request.observation.own_completed_turns + request.observation.opponent_completed_turns
    record = {name: getattr(encoded, name).copy() for name in STATE_FIELDS}
    record.update({
        "candidate_features": encoded.candidate_features.copy(),
        "player_id": session.stepper.engine.current_player,
        "decision_index": session.stepper.decision_count,
        "phase": PHASE_ORDER.index(request.observation.phase),
        "completed_player_turns": completed,
        "chosen_candidate_index": result.chosen_candidate_index,
        "temperature": session.temperature,
        "sampling_seed": session.sampling_seed,
        "searchable": result.searchable,
        "policy_weight": float(result.searchable), "value_weight": 1.0,
        "policy_target": result.policy_target,
        "behavior_policy": result.behavior_policy,
        "root_network_prior": result.root_network_prior,
        "root_used_prior": result.root_used_prior,
        "root_visit_counts": result.root_visit_counts,
        "simulation_budgets": result.simulation_budgets,
        "per_world_root_visit_counts": result.per_world_root_visit_counts,
        "valid_world_count": len(result.simulation_budgets),
        "simulations": result.simulations,
        "max_tree_depth": result.max_tree_depth,
        "transition_cap_hits": result.transition_cap_hits,
    })
    return record


@dataclass(slots=True)
class _GameSession:
    config: FastSelfPlayConfig
    game_id: int
    evaluator: FrozenSearchInference
    started: float = field(default_factory=time.perf_counter)
    stepper: MatchStepper = field(init=False)
    policies: tuple[PUCTSearchPolicy, PUCTSearchPolicy] = field(init=False)
    records: list[dict[str, Any]] = field(default_factory=list)
    job: MCTSSearchJob | None = None
    encoded: Any = None
    temperature: float = 0.0
    sampling_seed: int = 0
    job_started: float = 0.0
    decision_latencies_ms: list[float] = field(default_factory=list)
    root_encoding_seconds: float = 0.0
    real_rule_seconds: float = 0.0

    def __post_init__(self) -> None:
        game_seed = derive_seed(self.config.seed, self.config.round_index, self.game_id, "game")
        self.stepper = MatchStepper(self.game_id, game_seed, self.game_id % 2,
                                    mode=self.config.execution.engine_mode)
        seeds = _search_seeds(self.config, self.game_id)
        self.policies = tuple(
            PUCTSearchPolicy(self.evaluator.model,
                             config=MCTSConfig(**(asdict(self.config.search_config) | {"seed": seeds[p]})),
                             device=self.evaluator.device,
                             engine_mode=self.config.execution.engine_mode,
                             evaluator=self.evaluator)
            for p in (0, 1)
        )  # type: ignore[assignment]

    def ensure_job(self) -> None:
        while self.stepper.terminal is None and self.job is None:
            if self.stepper.completed_player_turns >= self.config.max_completed_turns:
                raise RuntimeError("engineering turn limit reached without formal terminal")
            request = self.stepper.request
            player_id = self.stepper.engine.current_player
            encoding_started = time.perf_counter()
            self.encoded = encode_decision(request)
            self.root_encoding_seconds += time.perf_counter() - encoding_started
            self.temperature = self.config.temperature_schedule.for_request(request)
            self.sampling_seed = derive_seed(
                self.config.seed, self.config.round_index, self.game_id,
                "search_sampling", player_id, self.stepper.decision_count,
            )
            self.job_started = time.perf_counter()
            self.job = self.policies[player_id].start_job(
                request, add_root_noise=self.config.add_root_noise,
                temperature=self.temperature, sampling_seed=self.sampling_seed,
                root_encoded=self.encoded,
            )
            if self.job.done:
                self.take_result()

    def take_result(self) -> None:
        if self.job is None or not self.job.done:
            raise RuntimeError("search result is not ready")
        result = self.job.result
        self.decision_latencies_ms.append((time.perf_counter() - self.job_started) * 1000)
        record = _sample_record(self, result)
        rule_started = time.perf_counter()
        self.stepper.apply(result.chosen_candidate_index)
        self.real_rule_seconds += time.perf_counter() - rule_started
        self.records.append(record)
        self.job = None
        self.encoded = None

    def commit(self, root: Path, fingerprint: str, checkpoint_sha: str) -> dict[str, Any]:
        terminal = self.stepper.terminal
        if terminal is None:
            raise RuntimeError("cannot commit an incomplete game")
        label_terminal_values(self.records, terminal)
        metadata = {
            "data_schema": DATA_SCHEMA_V2, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "semantic_fingerprint": fingerprint, "checkpoint_sha256": checkpoint_sha,
            "run_id": self.config.run_id, "round_index": self.config.round_index,
            "game_id": self.game_id, "game_seed": self.stepper.trace["game_seed"],
            "search_seeds": list(_search_seeds(self.config, self.game_id)),
            "first_player": self.stepper.trace["first_player"],
            "split": _split(self.config, self.game_id), "status": "natural_terminal",
            "terminal_result": to_primitive(terminal),
            "completed_player_turns": self.stepper.completed_player_turns,
            "decision_count": len(self.records),
            "player_decisions": [sum(r["player_id"] == p for r in self.records) for p in (0, 1)],
            "policy_samples": sum(r["searchable"] for r in self.records),
            "mcts_stats": [policy.stats.to_dict() | {"rule_seconds": policy.stats.rule_seconds}
                           for policy in self.policies],
            "forced_skips": self.stepper.engine.forced_skip_count,
            "trace": self.stepper.trace,
            "collection_seconds": time.perf_counter() - self.started,
        }
        write_started = time.perf_counter()
        arrays = pack_game(self.records, metadata, self.config.search_config.determinizations)
        validate_game_arrays({name: value for name, value in arrays.items() if name != "metadata_json"},
                             metadata, expected_game_id=self.game_id, fingerprint=fingerprint,
                             simulations=self.config.search_config.simulations,
                             worlds=self.config.search_config.determinizations,
                             split=metadata["split"])
        path = commit_game(root, self.game_id, arrays)
        return {"game_id": self.game_id, "decisions": len(self.records),
                "policy_samples": metadata["policy_samples"], "bytes": path.stat().st_size,
                "simulations": sum(r["simulations"] for r in self.records),
                "write_seconds": time.perf_counter() - write_started,
                "root_encoding_seconds": self.root_encoding_seconds,
                "real_rule_seconds": self.real_rule_seconds,
                "search_rule_seconds": sum(p.stats.rule_seconds for p in self.policies)}


def _committed_game_count(root: Path) -> int:
    """Count archived games without validating them; progress reporting only."""

    games_root = root / "games"
    return len(list(games_root.glob("game-*.npz"))) if games_root.is_dir() else 0


def _collect_group(
    config: FastSelfPlayConfig, root: Path, game_ids: tuple[int, ...],
    fingerprint: str, checkpoint_sha: str, stop_event: Any = None,
    completed_before: int = 0,
    on_game_complete: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    initialization_started = time.perf_counter()
    torch.set_num_threads(config.execution.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = "cuda" if config.execution.backend == "batched_cuda" else "cpu"
    model = load_frozen_actor_critic_checkpoint(Path(config.checkpoint), device=device)
    evaluator = FrozenSearchInference(model, device=device)
    initialization_seconds = time.perf_counter() - initialization_started
    started = time.perf_counter()
    todo = iter(game_ids)
    active: dict[int, _GameSession] = {}
    completed: list[dict[str, Any]] = []
    peak_active = 0
    peak_pending = 0
    failed: _GameSession | None = None
    batch_context: list[tuple[_GameSession, Any]] = []
    batch_outputs: Any = None
    active_request_id: str | None = None
    decision_latencies_ms: list[float] = []
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            while len(active) < config.execution.active_games_per_worker:
                try:
                    game_id = next(todo)
                except StopIteration:
                    break
                active[game_id] = _GameSession(config, game_id, evaluator)
            if not active:
                break
            peak_active = max(peak_active, len(active))
            for game_id, session in tuple(active.items()):
                failed = session
                if session.job is not None and session.job.done:
                    session.take_result()
                session.ensure_job()
                if session.stepper.terminal is not None:
                    completed.append(session.commit(root, fingerprint, checkpoint_sha))
                    decision_latencies_ms.extend(session.decision_latencies_ms)
                    del active[game_id]
                    if on_game_complete is not None:
                        on_game_complete(completed_before + len(completed), config.num_games)
            pending = [(session, item) for session in active.values() if session.job is not None
                       for item in session.job.pending_requests()]
            peak_pending = max(peak_pending, len(pending))
            if not pending:
                if active:
                    raise RuntimeError("active games have no pending search work")
                continue
            for kind in ("policy_value", "value_only"):
                work = [(session, item) for session, item in pending if item.kind == kind]
                for start in range(0, len(work), config.execution.max_batch_size):
                    group = work[start:start + config.execution.max_batch_size]
                    failed = group[0][0]
                    active_request_id = group[0][1].request_id
                    batch_context = group
                    batch_outputs = None
                    requests = tuple(item.request for _, item in group)
                    if kind == "policy_value":
                        encoded = tuple(item.encoded if item.encoded is not None else encode_decision(item.request)
                                        for _, item in group)
                        results = evaluator.policy_value(requests, encoded=encoded)
                        batch_outputs = results
                        for (session, item), (logits, value) in zip(group, results):
                            failed = session
                            active_request_id = item.request_id
                            assert session.job is not None
                            session.job.submit(item.request_id, logits, value)
                    else:
                        results = evaluator.value_only(requests)
                        batch_outputs = results
                        for (session, item), value in zip(group, results):
                            failed = session
                            active_request_id = item.request_id
                            assert session.job is not None
                            session.job.submit(item.request_id, None, value)
                    batch_context = []
                    batch_outputs = None
                    active_request_id = None
    except Exception as error:
        if failed is not None:
            import traceback
            scene = {
                "protocol": "search-accel-failure-v1", "game_id": failed.game_id,
                "game_seed": failed.stepper.trace["game_seed"],
                "first_player": failed.stepper.trace["first_player"],
                "trace": failed.stepper.trace,
                "root_decision_index": failed.stepper.decision_count,
                "root_request": to_primitive(failed.stepper.request) if failed.stepper.terminal is None else None,
                "search_config": asdict(config.search_config),
                "checkpoint_sha256": checkpoint_sha,
                "batch_requests": [
                    {"request_id": item.request_id, "kind": item.kind,
                     "game_id": session.game_id,
                     "root_decision_index": session.stepper.decision_count,
                     "request": to_primitive(item.request),
                     "output": None if batch_outputs is None else to_primitive(batch_outputs[index])}
                    for index, (session, item) in enumerate(batch_context)
                ],
                "search_context": (None if failed.job is None else to_primitive(failed.job.failure_context(active_request_id))),
                "error_type": type(error).__name__, "error_detail": str(error),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(root / "failures" / f"game-{failed.game_id:06d}.json", scene)
        raise
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "completed": completed, "wall_seconds": time.perf_counter() - started,
        "initialization_seconds": initialization_seconds,
        "decision_latencies_ms": decision_latencies_ms,
        "inference": evaluator.stats.to_dict(),
        "peak_active_games": peak_active, "peak_pending_requests": peak_pending,
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
    }


def _scan_committed(root: Path, config: FastSelfPlayConfig, manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    complete: dict[int, dict[str, Any]] = {}
    for path in sorted((root / "games").glob("game-*.npz")):
        game_id = int(path.stem.removeprefix("game-"))
        if not 0 <= game_id < config.num_games or game_id in complete:
            raise ValueError(f"unexpected committed game {path}")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
        if (metadata.get("data_schema") != DATA_SCHEMA_V2 or
                metadata.get("semantic_fingerprint") != manifest["semantic_fingerprint"] or
                metadata.get("checkpoint_sha256") != manifest["checkpoint_identity"]["sha256"] or
                metadata.get("game_id") != game_id or
                metadata.get("game_seed") != derive_seed(config.seed, config.round_index, game_id, "game") or
                metadata.get("search_seeds") != list(_search_seeds(config, game_id))):
            raise ValueError(f"committed game identity mismatch: {path}")
        complete[game_id] = metadata
    return complete


def validate_run(root: Path, *, replay: bool = False) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("v2 run manifest protocol mismatch")
    config = FastSelfPlayConfig.from_dict(manifest["config"])
    committed = _scan_committed(root, config, manifest)
    summary: dict[str, Any] = {
        "protocol": PROTOCOL, "expected_games": config.num_games,
        "completed_games": len(committed), "missing_games": config.num_games - len(committed),
        "p0_wins": 0, "p1_wins": 0, "draws": 0,
        "samples": 0, "policy_samples": 0, "simulations": 0,
        "data_bytes": 0, "search_seconds_sum": 0.0, "logical_model_positions": 0,
        "failed_games": 0, "exceptions": 0, "engineering_truncations": 0,
        "semantic_fingerprint": manifest["semantic_fingerprint"],
        "statistics_scope": "unavailable", "statistics_complete": False,
        "statistics_missing_game_ids": sorted(committed),
        "games_per_wall_second": None, "policy_samples_per_wall_second": None,
        "simulations_per_wall_second": None,
    }
    for game_id, metadata in committed.items():
        path = game_path(root, game_id)
        arrays, metadata = load_game(path)
        counts = validate_game_arrays(arrays, metadata, expected_game_id=game_id,
                                      fingerprint=manifest["semantic_fingerprint"],
                                      simulations=config.search_config.simulations,
                                      worlds=config.search_config.determinizations,
                                      split=_split(config, game_id))
        if replay:
            replay_trace(metadata["trace"])
        summary["samples"] += counts["decisions"]
        summary["policy_samples"] += counts["policy_samples"]
        summary["simulations"] += counts["simulations"]
        summary["data_bytes"] += path.stat().st_size
        summary["search_seconds_sum"] += sum(s["total_seconds"] for s in metadata["mcts_stats"])
        summary["logical_model_positions"] += sum(s["model_positions"] for s in metadata["mcts_stats"])
        winners = metadata["terminal_result"]["winner_ids"]
        summary["draws" if len(winners) == 2 else ("p0_wins" if winners[0] == 0 else "p1_wins")] += 1
    for scene_path in (root / "failures").glob("game-*.json"):
        game_id = int(scene_path.stem.removeprefix("game-"))
        if game_id in committed:
            continue
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        summary["failed_games"] += 1
        if scene.get("error_detail") == "engineering turn limit reached without formal terminal":
            summary["engineering_truncations"] += 1
        else:
            summary["exceptions"] += 1
    status_path = root / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary.update({key: status[key] for key in ("wall_seconds", "inference", "cpu_peak_rss_kib",
                                                    "gpu_peak_bytes", "peak_active_games", "peak_pending_requests",
                                                    "decision_latency", "root_encoding_seconds",
                                                    "real_rule_seconds", "search_rule_seconds", "write_seconds",
                                                    "queue_wait_seconds", "worker_initialization_seconds_sum",
                                                    "worker_collection_wall_seconds_max",
                                                    "statistics_scope", "statistics_game_ids")
                        if key in status})
        # Wall time and shared physical batches cannot be recovered by summing
        # archived game durations: games/workers overlap. Only use the games
        # covered by this collection session as throughput numerators.
        measured_ids = status.get("statistics_game_ids")
        if status.get("statistics_scope") == "collection_session" and isinstance(measured_ids, list):
            measured = set(measured_ids)
            if len(measured) != len(measured_ids) or not measured.issubset(committed):
                raise ValueError("collection statistics refer to invalid game ids")
            summary["statistics_complete"] = measured == set(committed)
            summary["statistics_missing_game_ids"] = sorted(set(committed) - measured)
            wall = summary.get("wall_seconds", 0.0)
            summary["games_per_wall_second"] = len(measured) / wall if wall else 0.0
            policy_samples = sum(committed[i]["policy_samples"] for i in measured)
            simulations = sum(s["simulations"] for i in measured for s in committed[i]["mcts_stats"])
            summary["policy_samples_per_wall_second"] = policy_samples / wall if wall else 0.0
            summary["simulations_per_wall_second"] = simulations / wall if wall else 0.0
        else:
            # Legacy status files do not identify their coverage; resume may
            # already have lost historical costs. Do not invent whole-run rates.
            summary["statistics_scope"] = "unknown"
            summary["statistics_complete"] = False
            for metric in ("games", "policy_samples", "simulations"):
                summary[f"{metric}_per_wall_second"] = None
    return summary


def generate(
    config: FastSelfPlayConfig,
    root: Path,
    *,
    resume: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval_seconds: float = PROGRESS_POLL_SECONDS,
) -> dict[str, Any]:
    """Collect the missing games of one run.

    ``progress_callback`` runs in the parent process with
    ``(completed_games, total_games)``: single-worker runs report every commit,
    multi-worker runs poll the archive directory every
    ``progress_interval_seconds`` so partial progress stays visible while the
    pool is still working.
    """

    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("output run exists; use --resume")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            current = build_manifest(config)
            if manifest.get("semantic_fingerprint") != current["semantic_fingerprint"]:
                raise ValueError("resume semantic fingerprint mismatch")
        else:
            if any(root.iterdir()):
                raise ValueError("nonempty output directory has no v2 manifest")
            if resume:
                raise ValueError("cannot resume an empty run")
            manifest = build_manifest(config)
            _atomic_json(manifest_path, manifest)
        complete = _scan_committed(root, config, manifest)
        missing = tuple(game_id for game_id in range(config.num_games) if game_id not in complete)
        if not missing:
            return validate_run(root)
        if config.execution.backend == "batched_cuda" and not torch.cuda.is_available():
            raise RuntimeError("batched_cuda requested but CUDA is unavailable; use batched_cpu")
        started = time.perf_counter()
        groups = tuple(tuple(missing[i::config.execution.workers]) for i in range(config.execution.workers))
        groups = tuple(group for group in groups if group)
        checkpoint_sha = manifest["checkpoint_identity"]["sha256"]
        if len(groups) == 1:
            results = [_collect_group(
                config, root, groups[0], manifest["semantic_fingerprint"], checkpoint_sha,
                completed_before=len(complete), on_game_complete=progress_callback,
            )]
        else:
            with mp.Manager() as manager:
                stop_event = manager.Event()
                with ProcessPoolExecutor(max_workers=len(groups), mp_context=mp.get_context("spawn")) as pool:
                    futures = [pool.submit(_collect_group, config, root, group,
                                           manifest["semantic_fingerprint"], checkpoint_sha,
                                           stop_event) for group in groups]
                    try:
                        if progress_callback is None:
                            results = [future.result() for future in as_completed(futures)]
                        else:
                            results = []
                            pending = set(futures)
                            while pending:
                                finished, pending = wait(
                                    pending,
                                    timeout=progress_interval_seconds,
                                    return_when=FIRST_COMPLETED,
                                )
                                for future in finished:
                                    results.append(future.result())
                                progress_callback(_committed_game_count(root), config.num_games)
                    except Exception:
                        stop_event.set()
                        for future in futures:
                            future.cancel()
                        raise
        # Each successful invocation reports its own cost and explicit coverage.
        # Never merge an old session with possibly unmeasured committed games.
        status_path = root / "status.json"
        physical = {
            "physical_model_batches": sum(item["inference"]["physical_model_batches"] for item in results),
            "logical_model_positions": sum(item["inference"]["logical_model_positions"] for item in results),
            "feature_seconds": sum(item["inference"]["feature_seconds"] for item in results),
            "forward_seconds": sum(item["inference"]["forward_seconds"] for item in results),
            "batch_size_histogram": {},
        }
        hist = physical["batch_size_histogram"]
        for item in results:
            for size, count in item["inference"]["batch_size_histogram"].items():
                hist[size] = hist.get(size, 0) + count
        batches = physical["physical_model_batches"]
        physical["mean_batch_size"] = physical["logical_model_positions"] / batches if batches else 0.0
        import statistics
        latencies = [value for result in results for value in result["decision_latencies_ms"]]
        ordered_latencies = sorted(latencies)
        status = {
            "statistics_scope": "collection_session",
            "statistics_game_ids": sorted(game["game_id"] for result in results for game in result["completed"]),
            "wall_seconds": time.perf_counter() - started,
            "decision_latencies_ms": latencies,
            "decision_latency": {
                "count": len(latencies),
                "p50_ms": statistics.median(latencies) if latencies else 0.0,
                "p95_ms": ordered_latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0,
                "max_ms": max(latencies) if latencies else 0.0,
            },
            "root_encoding_seconds": sum(
                game["root_encoding_seconds"] for result in results for game in result["completed"]),
            "real_rule_seconds": sum(
                game["real_rule_seconds"] for result in results for game in result["completed"]),
            "search_rule_seconds": sum(
                game["search_rule_seconds"] for result in results for game in result["completed"]),
            "write_seconds": sum(
                game["write_seconds"] for result in results for game in result["completed"]),
            "queue_wait_seconds": 0.0,
            "worker_initialization_seconds_sum": sum(
                result["initialization_seconds"] for result in results),
            "worker_collection_wall_seconds_max": max(result["wall_seconds"] for result in results),
            "inference": physical,
            "peak_active_games": max(r["peak_active_games"] for r in results),
            "peak_pending_requests": max(r["peak_pending_requests"] for r in results),
            "cpu_peak_rss_kib": max(r["cpu_peak_rss_kib"] for r in results),
            "gpu_peak_bytes": max(r["gpu_peak_bytes"] for r in results),
        }
        _atomic_json(status_path, status)
        summary = validate_run(root)
        _atomic_json(root / "summary.json", summary)
        return summary
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("generate")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--resume", action="store_true")
    for name in ("validate", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        config = FastSelfPlayConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        summary = generate(config, args.output, resume=args.resume)
    else:
        summary = validate_run(args.input, replay=args.replay)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
