"""Reproducible, resumable frozen-network MCTS self-play collection."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import uuid
from datetime import datetime, timezone
import subprocess
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import DecisionRequest
from .features import FEATURE_SCHEMA_VERSION, PHASE_ORDER, encode_decision
from .mcts_policy import MCTSConfig, PUCTSearchPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .replay import REPLAY_SCHEMA_VERSION, RULESET_ID, to_primitive
from .rng import RNG_ALGORITHM
from .runner import RunStatus, run_match
from .result import TerminalResult
from .selfplay_data import (
    DATA_SCHEMA, STATE_FIELDS, commit_game, game_path, load_game, pack_game,
    validate_game_arrays,
)

PROTOCOL = "mcts-selfplay-v1"
SEED_PROTOCOL = "splendor-selfplay-seed-v1"
SPLIT_PROTOCOL = "splendor-selfplay-game-hash-v1"
SEARCH_PROTOCOL = "determinized-puct-v1"
IMPLEMENTATION_FILES = (
    "src/splendor/selfplay.py", "src/splendor/selfplay_data.py",
    "src/splendor/mcts_policy.py", "src/splendor/search_policy.py",
    "src/splendor/runner.py", "src/splendor/engine.py",
    "src/splendor/rules.py", "src/splendor/features.py",
)


def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class TemperatureSchedule:
    opening_completed_turns: int = 10
    opening_temperature: float = 1.0
    late_temperature: float = 0.0

    def __post_init__(self) -> None:
        _positive_int("opening_completed_turns", self.opening_completed_turns, minimum=0)
        for name in ("opening_temperature", "late_temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def for_request(self, request: DecisionRequest) -> float:
        observation = request.observation
        turns = observation.own_completed_turns + observation.opponent_completed_turns
        return self.opening_temperature if turns < self.opening_completed_turns else self.late_temperature


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    protocol: str
    run_id: str
    round_index: int
    checkpoint: str
    num_games: int
    seed: int
    search_config: MCTSConfig
    add_root_noise: bool = True
    temperature_schedule: TemperatureSchedule = TemperatureSchedule()
    validation_fraction: float = 0.1
    split_seed: int = 0
    workers: int = 1
    device: str = "cpu"
    max_completed_turns: int = 400

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL or not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("invalid self-play protocol or run_id")
        _positive_int("round_index", self.round_index, minimum=0)
        _positive_int("num_games", self.num_games)
        _positive_int("max_completed_turns", self.max_completed_turns)
        _positive_int("workers", self.workers)
        if type(self.seed) is not int or type(self.split_seed) is not int:
            raise ValueError("seed and split_seed must be integers")
        if type(self.add_root_noise) is not bool:
            raise ValueError("add_root_noise must be bool")
        if isinstance(self.validation_fraction, bool) or not isinstance(self.validation_fraction, (int, float)) or not math.isfinite(self.validation_fraction) or not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be finite and in [0, 1)")
        if self.device != "cpu":
            raise ValueError("self-play v1 supports CPU only")
        if not Path(self.checkpoint).is_file():
            raise ValueError(f"checkpoint does not exist: {self.checkpoint}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SelfPlayConfig:
        expected = set(cls.__dataclass_fields__)
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown self-play fields: {sorted(unknown)}")
        search = raw.get("search_config")
        schedule = raw.get("temperature_schedule", {})
        if not isinstance(search, Mapping) or not isinstance(schedule, Mapping):
            raise ValueError("search_config and temperature_schedule must be objects")
        checkpoint = raw.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("checkpoint must be a path string")
        return cls(
            protocol=raw.get("protocol"), run_id=raw.get("run_id"),
            round_index=raw.get("round_index"), checkpoint=str(Path(checkpoint).resolve()),
            num_games=raw.get("num_games"), seed=raw.get("seed"),
            search_config=MCTSConfig(**search),
            add_root_noise=raw.get("add_root_noise", True),
            temperature_schedule=TemperatureSchedule(**schedule),
            validation_fraction=raw.get("validation_fraction", 0.1),
            split_seed=raw.get("split_seed", 0), workers=raw.get("workers", 1),
            device=raw.get("device", "cpu"),
            max_completed_turns=raw.get("max_completed_turns", 400),
        )

    def semantic_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("workers")
        return value


def derive_seed(base_seed: int, round_index: int, game_id: int, domain: str, *parts: int) -> int:
    payload = json.dumps([SEED_PROTOCOL, domain, base_seed, round_index, game_id, *parts], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_for(config: SelfPlayConfig, game_id: int) -> str:
    word = derive_seed(config.split_seed, config.round_index, game_id, "split")
    return "validation" if word / (1 << 64) < config.validation_fraction else "train"


def search_seeds(config: SelfPlayConfig, game_id: int) -> tuple[int, int]:
    return tuple(config.search_config.seed ^ derive_seed(config.seed, config.round_index, game_id, "search_world", p) for p in (0, 1))  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout)
    return {"git_revision": revision, "git_dirty": dirty, "implementation_sha256": {name: _sha256(root / name) for name in IMPLEMENTATION_FILES}}


def build_manifest(config: SelfPlayConfig) -> dict[str, Any]:
    model = load_frozen_actor_critic_checkpoint(Path(config.checkpoint), device="cpu")
    identity = {
        "path": config.checkpoint,
        "sha256": _sha256(Path(config.checkpoint)),
        "model_config": asdict(model.config),
    }
    versions = {
        "data": DATA_SCHEMA, "search": SEARCH_PROTOCOL, "seed": SEED_PROTOCOL,
        "split": SPLIT_PROTOCOL, "rules": RULESET_ID,
        "replay": REPLAY_SCHEMA_VERSION, "rng": RNG_ALGORITHM,
        "feature": FEATURE_SCHEMA_VERSION,
    }
    source = _source_identity()
    frozen = {"semantic_config": config.semantic_dict(), "checkpoint_identity": identity,
              "versions": versions, "implementation_sha256": source["implementation_sha256"]}
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"protocol": PROTOCOL, "semantic_fingerprint": fingerprint,
            **frozen, "source": source, "expected_game_ids": [0, config.num_games - 1],
            "created_utc": datetime.now(timezone.utc).isoformat()}


class _Recorder:
    def __init__(self, player_id: int, policy: PUCTSearchPolicy, config: SelfPlayConfig, game_id: int, records: list[dict[str, Any]]) -> None:
        self.player_id = player_id
        self.policy = policy
        self.config = config
        self.game_id = game_id
        self.records = records

    def choose(self, request: DecisionRequest):
        index = len(self.records)
        sampling_seed = derive_seed(self.config.seed, self.config.round_index, self.game_id, "search_sampling", self.player_id, index)
        temperature = self.config.temperature_schedule.for_request(request)
        encoded = encode_decision(request)
        result = self.policy.search(request, add_root_noise=self.config.add_root_noise,
                                    temperature=temperature, sampling_seed=sampling_seed)
        completed = request.observation.own_completed_turns + request.observation.opponent_completed_turns
        record: dict[str, Any] = {name: getattr(encoded, name).copy() for name in STATE_FIELDS}
        record.update({
            "candidate_features": encoded.candidate_features.copy(),
            "player_id": self.player_id, "decision_index": index,
            "phase": PHASE_ORDER.index(request.observation.phase),
            "completed_player_turns": completed,
            "chosen_candidate_index": result.chosen_candidate_index,
            "temperature": temperature, "sampling_seed": sampling_seed,
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
        self.records.append(record)
        return request.legal_actions[result.chosen_candidate_index]


def label_terminal_values(records: Sequence[dict[str, Any]], terminal: TerminalResult) -> None:
    for record in records:
        record["value_target"] = terminal.payoff(record["player_id"])


_WORKER_MODEL = None
_WORKER_CONFIG: SelfPlayConfig | None = None
_WORKER_ROOT: Path | None = None
_WORKER_FINGERPRINT: str | None = None
_WORKER_CHECKPOINT_SHA: str | None = None


def _worker_init(config: SelfPlayConfig, root: Path, checkpoint_sha: str, fingerprint: str) -> None:
    global _WORKER_MODEL, _WORKER_CONFIG, _WORKER_ROOT, _WORKER_FINGERPRINT, _WORKER_CHECKPOINT_SHA
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if _sha256(Path(config.checkpoint)) != checkpoint_sha:
        raise ValueError("checkpoint contents changed before worker loaded it")
    _WORKER_MODEL = load_frozen_actor_critic_checkpoint(Path(config.checkpoint), device=config.device)
    _WORKER_CONFIG, _WORKER_ROOT, _WORKER_FINGERPRINT, _WORKER_CHECKPOINT_SHA = config, root, fingerprint, checkpoint_sha


def _play_one(game_id: int) -> dict[str, Any]:
    if _WORKER_CONFIG is None or _WORKER_ROOT is None or _WORKER_MODEL is None or _WORKER_FINGERPRINT is None or _WORKER_CHECKPOINT_SHA is None:
        raise RuntimeError("self-play worker is not initialized")
    config, root = _WORKER_CONFIG, _WORKER_ROOT
    game_seed = derive_seed(config.seed, config.round_index, game_id, "game")
    seeds = search_seeds(config, game_id)
    records: list[dict[str, Any]] = []
    policies = tuple(PUCTSearchPolicy(_WORKER_MODEL, config=replace(config.search_config, seed=seeds[p]), device=config.device) for p in (0, 1))
    wrappers = tuple(_Recorder(p, policies[p], config, game_id, records) for p in (0, 1))
    started = time.perf_counter()
    match = run_match(
        wrappers[0], wrappers[1], game_seed=game_seed, policy_seeds=seeds,
        first_player=game_id % 2, policy_names=("selfplay-p0", "selfplay-p1"),
        max_completed_turns=config.max_completed_turns,
        failure_dir=root / "failures" / f"game-{game_id:06d}",
        adjudicate_deterministic_policy_cycles=False, raise_on_error=True, lean=True,
    )
    if match.status is not RunStatus.NATURAL_TERMINAL or match.terminal_result is None:
        raise RuntimeError(f"game {game_id} seed {game_seed}: {match.status.value}; scene={match.failure_scene}")
    if len(records) != match.decision_count:
        raise AssertionError("recorded decisions disagree with runner")
    terminal = match.terminal_result
    label_terminal_values(records, terminal)
    metadata: dict[str, Any] = {
        "data_schema": DATA_SCHEMA, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "semantic_fingerprint": _WORKER_FINGERPRINT,
        "checkpoint_sha256": _WORKER_CHECKPOINT_SHA,
        "run_id": config.run_id, "round_index": config.round_index,
        "game_id": game_id, "game_seed": game_seed, "search_seeds": list(seeds),
        "first_player": match.first_player, "split": split_for(config, game_id),
        "status": match.status.value, "terminal_result": to_primitive(terminal),
        "completed_player_turns": match.completed_player_turns,
        "decision_count": match.decision_count,
        "player_decisions": [sum(r["player_id"] == p for r in records) for p in (0, 1)],
        "policy_samples": sum(r["searchable"] for r in records),
        "mcts_stats": [policy.stats.to_dict() for policy in policies],
        "final_state_digest": match.final_state_digest,
        "replay": to_primitive(match.replay),
        "collection_seconds": time.perf_counter() - started,
    }
    arrays = pack_game(records, metadata, config.search_config.determinizations)
    validate_game_arrays({name: value for name, value in arrays.items() if name != "metadata_json"}, metadata, expected_game_id=game_id,
                         fingerprint=_WORKER_FINGERPRINT,
                         simulations=config.search_config.simulations,
                         worlds=config.search_config.determinizations,
                         split=metadata["split"])
    path = commit_game(root, game_id, arrays)
    return {"game_id": game_id, "decisions": len(records),
            "policy_samples": metadata["policy_samples"], "bytes": path.stat().st_size,
            "collection_seconds": metadata["collection_seconds"]}


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol") != PROTOCOL:
        raise ValueError("manifest protocol mismatch")
    return value


def _validate_committed(root: Path, manifest: Mapping[str, Any], config: SelfPlayConfig) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    checkpoint = manifest["checkpoint_identity"]
    if _sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("checkpoint contents changed")
    completed: dict[int, dict[str, Any]] = {}
    summary = {
        "expected_games": config.num_games, "completed_games": 0,
        "missing_games": 0, "failed_games": 0,
        "p0_wins": 0, "p1_wins": 0, "draws": 0,
        "first_player_counts": [0, 0], "samples": 0,
        "policy_samples": 0, "value_samples": 0,
        "single_candidate_samples": 0, "candidate_count": 0,
        "candidate_count_histogram": {},
        "game_turns_min": None, "game_turns_max": None,
        "game_decisions_min": None, "game_decisions_max": None,
        "phase_counts": {phase.value: 0 for phase in PHASE_ORDER},
        "split_games": {"train": 0, "validation": 0},
        "split_samples": {"train": 0, "validation": 0},
        "completed_player_turns": 0, "decisions": 0,
        "target_entropy_sum": 0.0, "behavior_entropy_sum": 0.0,
        "target_max_probability_sum": 0.0,
        "simulations": 0, "transition_cap_hits": 0,
        "model_batches": 0, "search_seconds": 0.0,
        "collection_seconds_sum": 0.0, "data_bytes": 0,
    }
    games_dir = root / "games"
    for path in sorted(games_dir.glob("game-*.npz")) if games_dir.exists() else ():
        try:
            game_id = int(path.stem.removeprefix("game-"))
        except ValueError as error:
            raise ValueError(f"invalid committed game filename: {path}") from error
        if not 0 <= game_id < config.num_games or game_id in completed or path != game_path(root, game_id):
            raise ValueError(f"unexpected or duplicate committed game file: {path}")
        arrays, metadata = load_game(path)
        counts = validate_game_arrays(arrays, metadata, expected_game_id=game_id,
                                      fingerprint=manifest["semantic_fingerprint"],
                                      simulations=config.search_config.simulations,
                                      worlds=config.search_config.determinizations,
                                      split=split_for(config, game_id))
        expected_seed = derive_seed(config.seed, config.round_index, game_id, "game")
        if metadata["game_seed"] != expected_seed or tuple(metadata["search_seeds"]) != search_seeds(config, game_id) or metadata["first_player"] != game_id % 2:
            raise ValueError(f"game {game_id}: seed or first-player mismatch")
        if metadata.get("checkpoint_sha256") != checkpoint["sha256"] or metadata.get("run_id") != config.run_id or metadata.get("round_index") != config.round_index:
            raise ValueError(f"game {game_id}: frozen identity mismatch")
        events = [event for event in metadata["replay"]["events"] if not event.get("automatic", False)]
        for i, event in enumerate(events):
            if int(arrays["player_id"][i]) != event["actor"] or PHASE_ORDER[int(arrays["phase"][i])].value != event["phase"]:
                raise ValueError(f"game {game_id} sample {i}: actor or phase disagrees with replay")
            expected_sampling = derive_seed(config.seed, config.round_index, game_id, "search_sampling", event["actor"], i)
            if int(arrays["sampling_seed"][i]) != expected_sampling:
                raise ValueError(f"game {game_id} sample {i}: sampling seed mismatch")
            expected_temperature = config.temperature_schedule.opening_temperature if int(arrays["completed_player_turns"][i]) < config.temperature_schedule.opening_completed_turns else config.temperature_schedule.late_temperature
            if not math.isclose(float(arrays["temperature"][i]), expected_temperature, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"game {game_id} sample {i}: temperature schedule mismatch")
        completed[game_id] = metadata
        summary["completed_games"] += 1
        summary["samples"] += counts["decisions"]
        summary["policy_samples"] += counts["policy_samples"]
        summary["value_samples"] += counts["decisions"]
        summary["decisions"] += counts["decisions"]
        summary["candidate_count"] += counts["candidates"]
        summary["simulations"] += counts["simulations"]
        summary["single_candidate_samples"] += counts["decisions"] - counts["policy_samples"]
        turns = metadata["completed_player_turns"]
        summary["completed_player_turns"] += turns
        for key, value in (("game_turns_min", turns), ("game_decisions_min", counts["decisions"])):
            summary[key] = value if summary[key] is None else min(summary[key], value)
        for key, value in (("game_turns_max", turns), ("game_decisions_max", counts["decisions"])):
            summary[key] = value if summary[key] is None else max(summary[key], value)
        summary["first_player_counts"][metadata["first_player"]] += 1
        summary["split_games"][metadata["split"]] += 1
        summary["split_samples"][metadata["split"]] += counts["decisions"]
        summary["collection_seconds_sum"] += metadata["collection_seconds"]
        summary["data_bytes"] += path.stat().st_size
        winners = metadata["terminal_result"]["winner_ids"]
        summary["draws" if len(winners) == 2 else ("p0_wins" if winners[0] == 0 else "p1_wins")] += 1
        for phase_index, phase in enumerate(PHASE_ORDER):
            summary["phase_counts"][phase.value] += int(np.sum(arrays["phase"] == phase_index))
        offsets = arrays["candidate_offsets"]
        for i in range(counts["decisions"]):
            sl = slice(int(offsets[i]), int(offsets[i+1]))
            candidate_width = sl.stop - sl.start
            histogram = summary["candidate_count_histogram"]
            histogram[str(candidate_width)] = histogram.get(str(candidate_width), 0) + 1
            for prefix, name in (("target", "policy_target"), ("behavior", "behavior_policy")):
                p = arrays[name][sl]
                entropy = -sum(float(x) * math.log(float(x)) for x in p if x > 0)
                summary[f"{prefix}_entropy_sum"] += entropy
            summary["target_max_probability_sum"] += float(max(arrays["policy_target"][sl]))
        for stats in metadata["mcts_stats"]:
            summary["transition_cap_hits"] += stats["transition_cap_hits"]
            summary["model_batches"] += stats["model_batches"]
            summary["search_seconds"] += stats["total_seconds"]
    summary["missing_games"] = config.num_games - len(completed)
    failure_records = list((root / "failures").glob("game-*.json")) if (root / "failures").exists() else []
    summary["failed_games"] = sum(int(p.stem.removeprefix("game-")) not in completed for p in failure_records)
    count = summary["samples"]
    summary["mean_candidates_per_sample"] = summary["candidate_count"] / count if count else 0.0
    summary["mean_completed_turns_per_game"] = summary["completed_player_turns"] / len(completed) if completed else 0.0
    summary["mean_decisions_per_game"] = count / len(completed) if completed else 0.0
    summary["single_candidate_fraction"] = summary["single_candidate_samples"] / count if count else 0.0
    summary["target_entropy_mean_per_decision"] = summary["target_entropy_sum"] / count if count else 0.0
    summary["behavior_entropy_mean_per_decision"] = summary["behavior_entropy_sum"] / count if count else 0.0
    summary["target_max_probability_mean_per_decision"] = summary["target_max_probability_sum"] / count if count else 0.0
    status_path = root / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary["wall_seconds"] = float(status.get("elapsed_seconds", 0.0))
        wall = summary["wall_seconds"]
        summary["games_per_wall_second"] = len(completed) / wall if wall else 0.0
        summary["policy_samples_per_wall_second"] = summary["policy_samples"] / wall if wall else 0.0
    return completed, summary


def validate_run(root: Path) -> dict[str, Any]:
    root = Path(root)
    manifest = _read_manifest(root)
    config = SelfPlayConfig.from_dict(manifest["semantic_config"] | {"workers": 1})
    _, summary = _validate_committed(root, manifest, config)
    summary["semantic_fingerprint"] = manifest["semantic_fingerprint"]
    return summary


def _lock(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(f"self-play run is already active: {root}") from error
    return descriptor


def generate(config: SelfPlayConfig, root: Path, *, resume: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    existing = root.exists() and any(root.iterdir())
    descriptor = _lock(root)
    started = time.perf_counter()
    try:
        current = build_manifest(config)
        if existing:
            if not resume:
                raise ValueError(f"nonempty output directory requires --resume: {root}")
            manifest = _read_manifest(root)
            if current["semantic_fingerprint"] != manifest["semantic_fingerprint"]:
                raise ValueError("resume semantic fingerprint mismatch")
        else:
            if resume:
                raise ValueError("cannot resume an empty self-play run")
            manifest = current
            _atomic_json(root / "manifest.json", manifest)
        completed, prior_summary = _validate_committed(root, manifest, config)
        missing = [i for i in range(config.num_games) if i not in completed]
        if not missing:
            _atomic_json(root / "summary.json", prior_summary)
            return prior_summary
        checkpoint_sha = manifest["checkpoint_identity"]["sha256"]
        done = len(completed)
        policy_samples = prior_summary["policy_samples"]
        previous_wall = prior_summary.get("wall_seconds", 0.0)
        def report(value: Mapping[str, Any]) -> None:
            nonlocal done, policy_samples
            done += 1
            policy_samples += value["policy_samples"]
            elapsed = previous_wall + time.perf_counter() - started
            print(json.dumps({"event": "selfplay_progress", "completed_games": done,
                              "expected_games": config.num_games, "policy_samples": policy_samples,
                              "elapsed_seconds": round(elapsed, 1)}, sort_keys=True), flush=True)
            _atomic_json(root / "status.json", {"completed_games": done,
                                                "expected_games": config.num_games,
                                                "policy_samples": policy_samples,
                                                "elapsed_seconds": elapsed})
        if config.workers == 1:
            _worker_init(config, root, checkpoint_sha, manifest["semantic_fingerprint"])
            for game_id in missing:
                try:
                    report(_play_one(game_id))
                except Exception as error:
                    game_seed = derive_seed(config.seed, config.round_index, game_id, "game")
                    _atomic_json(root / "failures" / f"game-{game_id:06d}.json",
                                 {"game_id": game_id, "game_seed": game_seed,
                                  "error_type": type(error).__name__, "detail": str(error),
                                  "notes": getattr(error, "__notes__", [])})
                    raise RuntimeError(f"game {game_id} seed {game_seed} failed: {error}") from error
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=config.workers, mp_context=context,
                                     initializer=_worker_init,
                                     initargs=(config, root, checkpoint_sha, manifest["semantic_fingerprint"])) as pool:
                iterator = iter(missing)
                first_ids = [next(iterator) for _ in range(min(config.workers, len(missing)))]
                futures = {pool.submit(_play_one, game_id): game_id for game_id in first_ids}
                while futures:
                    finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in finished:
                        failed_game_id = futures.pop(future)
                        try:
                            value = future.result()
                        except Exception as error:
                            for pending in futures:
                                pending.cancel()
                            game_seed = derive_seed(config.seed, config.round_index, failed_game_id, "game")
                            _atomic_json(root / "failures" / f"game-{failed_game_id:06d}.json",
                                         {"game_id": failed_game_id, "game_seed": game_seed,
                                          "error_type": type(error).__name__, "detail": str(error),
                                          "notes": getattr(error, "__notes__", [])})
                            raise RuntimeError(f"game {failed_game_id} seed {game_seed} failed: {error}") from error
                        report(value)
                        try:
                            game_id = next(iterator)
                        except StopIteration:
                            continue
                        futures[pool.submit(_play_one, game_id)] = game_id
        _, summary = _validate_committed(root, manifest, config)
        summary["session_wall_seconds"] = time.perf_counter() - started
        _atomic_json(root / "summary.json", summary)
        return summary
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-network MCTS self-play data")
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--resume", action="store_true")
    generate_parser.add_argument("--workers", type=int)
    for name in ("validate", "inspect"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("self-play config must be a JSON object")
        config = SelfPlayConfig.from_dict(raw)
        if args.workers is not None:
            config = replace(config, workers=_positive_int("workers", args.workers))
        summary = generate(config, args.output, resume=args.resume)
    else:
        summary = validate_run(args.input)
        if args.command == "validate":
            print(json.dumps({"event": "selfplay_validated", "completed_games": summary["completed_games"],
                              "missing_games": summary["missing_games"]}, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
