"""Numeric, game-atomic archives for MCTS self-play supervision."""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .features import EncodedDecisionBatch, FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, PHASE_ORDER

DATA_SCHEMA = "splendor-mcts-selfplay-v1"
DATA_SCHEMA_V2 = "splendor-mcts-selfplay-v2"
STATE_FIELDS = (
    "global_features", "market_t1", "market_t1_mask", "market_t2",
    "market_t2_mask", "market_t3", "market_t3_mask", "self_reserved",
    "self_reserved_mask", "opponent_reserved", "opponent_reserved_mask",
    "nobles", "nobles_mask", "pending_card", "pending_card_mask",
)
PROBABILITY_FIELDS = (
    "policy_target", "behavior_policy", "root_network_prior", "root_used_prior",
)
FLAT_FIELDS = (*PROBABILITY_FIELDS, "root_visit_counts")
SAMPLE_FIELDS = (
    "player_id", "decision_index", "phase", "completed_player_turns",
    "chosen_candidate_index", "temperature", "sampling_seed", "searchable",
    "policy_weight", "value_weight", "value_target", "valid_world_count",
    "simulations", "max_tree_depth", "transition_cap_hits",
)


def game_path(root: Path, game_id: int) -> Path:
    return root / "games" / f"game-{game_id:06d}.npz"


def _json_bytes(value: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"), dtype=np.uint8).copy()


def pack_game(records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], worlds: int) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("a completed game must contain decisions")
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for i, record in enumerate(records):
        offsets[i + 1] = offsets[i] + len(record["policy_target"])
    arrays: dict[str, np.ndarray] = {
        "metadata_json": _json_bytes(metadata),
        "candidate_offsets": offsets,
        "candidate_features": np.concatenate([r["candidate_features"] for r in records]).astype(np.float32, copy=True),
        "simulation_budgets": np.zeros((len(records), worlds), dtype=np.int32),
        "per_world_root_visit_counts": np.zeros((worlds, int(offsets[-1])), dtype=np.int32),
    }
    for name in STATE_FIELDS:
        arrays[name] = np.stack([r[name] for r in records]).copy()
    for name in FLAT_FIELDS:
        dtype = np.int32 if name == "root_visit_counts" else np.float32
        arrays[name] = np.concatenate([r[name] for r in records]).astype(dtype, copy=True)
    dtype_map = {
        "player_id": np.int8, "decision_index": np.int32, "phase": np.int8,
        "completed_player_turns": np.int32, "chosen_candidate_index": np.int32,
        "temperature": np.float32, "sampling_seed": np.uint64,
        "searchable": np.bool_, "policy_weight": np.float32,
        "value_weight": np.float32, "value_target": np.int8,
        "valid_world_count": np.int8, "simulations": np.int32,
        "max_tree_depth": np.int32, "transition_cap_hits": np.int32,
    }
    for name in SAMPLE_FIELDS:
        arrays[name] = np.asarray([r[name] for r in records], dtype=dtype_map[name])
    for i, r in enumerate(records):
        count = len(r["simulation_budgets"])
        if count:
            arrays["simulation_budgets"][i, :count] = r["simulation_budgets"]
            arrays["per_world_root_visit_counts"][:count, offsets[i]:offsets[i+1]] = r["per_world_root_visit_counts"]
    return arrays


def commit_game(root: Path, game_id: int, arrays: Mapping[str, np.ndarray]) -> Path:
    destination = game_path(root, game_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"game {game_id} is already committed")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"game {game_id} is already committed")
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_game(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        metadata = json.loads(arrays.pop("metadata_json").tobytes().decode("utf-8"))
    except (OSError, ValueError, KeyError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid game archive {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"game archive {path} has invalid metadata")
    return arrays, metadata


def _payoff(winners: Sequence[int], player: int) -> int:
    return 0 if len(winners) == 2 else (1 if player in winners else -1)


def _probability(values: np.ndarray, *, name: str, game_id: int, index: int) -> None:
    if not np.all(np.isfinite(values)) or np.any(values < 0) or abs(float(values.sum(dtype=np.float64)) - 1.0) > 1e-6:
        raise ValueError(f"game {game_id} sample {index}: invalid {name}")


def validate_game_arrays(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any],
    *, expected_game_id: int | None = None, fingerprint: str | None = None,
    simulations: int | None = None, worlds: int | None = None,
    split: str | None = None,
) -> dict[str, int | float]:
    game_id = metadata.get("game_id")
    if type(game_id) is not int or (expected_game_id is not None and game_id != expected_game_id):
        raise ValueError("archive game_id mismatch")
    schema = metadata.get("data_schema")
    if schema not in (DATA_SCHEMA, DATA_SCHEMA_V2) or metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"game {game_id}: schema mismatch")
    if fingerprint is not None and metadata.get("semantic_fingerprint") != fingerprint:
        raise ValueError(f"game {game_id}: semantic fingerprint mismatch")
    if split is not None and metadata.get("split") != split:
        raise ValueError(f"game {game_id}: split mismatch")
    if metadata.get("status") != "natural_terminal" or not isinstance(metadata.get("terminal_result"), dict):
        raise ValueError(f"game {game_id}: no formal terminal result")
    terminal = metadata["terminal_result"]
    winners = terminal.get("winner_ids")
    if winners not in ([0], [1], [0, 1]):
        raise ValueError(f"game {game_id}: invalid terminal winners")
    required = set(STATE_FIELDS) | set(FLAT_FIELDS) | set(SAMPLE_FIELDS) | {
        "candidate_offsets", "candidate_features", "simulation_budgets", "per_world_root_visit_counts"
    }
    if set(arrays) != required:
        raise ValueError(f"game {game_id}: archive fields mismatch")
    offsets = arrays["candidate_offsets"]
    length = int(metadata.get("decision_count", -1))
    if length <= 0 or offsets.dtype != np.int64 or offsets.shape != (length + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 1):
        raise ValueError(f"game {game_id}: invalid candidate offsets")
    candidates = int(offsets[-1])
    if arrays["candidate_features"].shape != (candidates, FEATURE_SCHEMA.action_dim) or arrays["candidate_features"].dtype != np.float32:
        raise ValueError(f"game {game_id}: invalid candidate features")
    if not np.all(np.isfinite(arrays["candidate_features"])):
        raise ValueError(f"game {game_id}: nonfinite candidate features")
    expected_shapes = {
        "global_features": (FEATURE_SCHEMA.global_dim,),
        "market_t1": (FEATURE_SCHEMA.market_capacity, FEATURE_SCHEMA.card_dim),
        "market_t2": (FEATURE_SCHEMA.market_capacity, FEATURE_SCHEMA.card_dim),
        "market_t3": (FEATURE_SCHEMA.market_capacity, FEATURE_SCHEMA.card_dim),
        "self_reserved": (FEATURE_SCHEMA.reserved_capacity, FEATURE_SCHEMA.card_dim),
        "opponent_reserved": (FEATURE_SCHEMA.reserved_capacity, FEATURE_SCHEMA.card_dim),
        "nobles": (FEATURE_SCHEMA.noble_capacity, FEATURE_SCHEMA.noble_dim),
        "pending_card": (FEATURE_SCHEMA.card_dim,),
    }
    for name in STATE_FIELDS:
        shape = expected_shapes.get(name)
        if shape is None:
            shape = expected_shapes[name.removesuffix("_mask")][:-1]
        value = arrays[name]
        expected_dtype = np.bool_ if name.endswith("_mask") else np.float32
        if value.shape != (length, *shape) or value.dtype != expected_dtype:
            raise ValueError(f"game {game_id}: invalid {name} shape/dtype")
        if value.dtype == np.float32 and not np.all(np.isfinite(value)):
            raise ValueError(f"game {game_id}: nonfinite {name}")
    for name in FLAT_FIELDS:
        value = arrays[name]
        dtype = np.int32 if name == "root_visit_counts" else np.float32
        if value.shape != (candidates,) or value.dtype != dtype:
            raise ValueError(f"game {game_id}: invalid {name}")
    sample_dtypes = {
        "player_id": np.int8, "decision_index": np.int32, "phase": np.int8,
        "completed_player_turns": np.int32, "chosen_candidate_index": np.int32,
        "temperature": np.float32, "sampling_seed": np.uint64,
        "searchable": np.bool_, "policy_weight": np.float32,
        "value_weight": np.float32, "value_target": np.int8,
        "valid_world_count": np.int8, "simulations": np.int32,
        "max_tree_depth": np.int32, "transition_cap_hits": np.int32,
    }
    for name in SAMPLE_FIELDS:
        if arrays[name].shape != (length,) or arrays[name].dtype != sample_dtypes[name]:
            raise ValueError(f"game {game_id}: invalid {name} shape/dtype")
    if np.any(~np.isfinite(arrays["temperature"])) or np.any(arrays["temperature"] < 0) or np.any(arrays["completed_player_turns"] < 0):
        raise ValueError(f"game {game_id}: invalid temperature or completed turns")
    if np.any((arrays["phase"] < 0) | (arrays["phase"] > 3)):
        raise ValueError(f"game {game_id}: invalid phase")
    if not np.array_equal(arrays["decision_index"], np.arange(length)):
        raise ValueError(f"game {game_id}: decision indices are not consecutive")
    if np.any((arrays["player_id"] < 0) | (arrays["player_id"] > 1)):
        raise ValueError(f"game {game_id}: invalid player id")
    if tuple(int(np.sum(arrays["player_id"] == p)) for p in (0, 1)) != tuple(metadata.get("player_decisions", ())):
        raise ValueError(f"game {game_id}: player decision counts mismatch")
    if np.any(arrays["value_target"] != np.array([_payoff(winners, int(p)) for p in arrays["player_id"]])):
        raise ValueError(f"game {game_id}: terminal value labels mismatch")
    if np.any(arrays["value_weight"] != 1) or np.any(arrays["policy_weight"] != arrays["searchable"].astype(np.float32)):
        raise ValueError(f"game {game_id}: sample weights mismatch")
    if arrays["simulation_budgets"].shape != (length, worlds) or arrays["per_world_root_visit_counts"].shape != (worlds, candidates):
        raise ValueError(f"game {game_id}: world statistics shape mismatch")
    for i in range(length):
        start, end = int(offsets[i]), int(offsets[i+1])
        width = end - start
        chosen = int(arrays["chosen_candidate_index"][i])
        if not 0 <= chosen < width:
            raise ValueError(f"game {game_id} sample {i}: chosen index out of range")
        for name in PROBABILITY_FIELDS:
            _probability(arrays[name][start:end], name=name, game_id=game_id, index=i)
        target = arrays["policy_target"][start:end]
        behavior = arrays["behavior_policy"][start:end]
        if behavior[chosen] <= 0:
            raise ValueError(f"game {game_id} sample {i}: chosen behavior probability is zero")
        if arrays["temperature"][i] == 0 and (behavior[chosen] < 1 - 1e-6 or np.count_nonzero(behavior > 1e-6) != 1):
            raise ValueError(f"game {game_id} sample {i}: temperature-zero behavior mismatch")
        visits = arrays["root_visit_counts"][start:end]
        world_visits = arrays["per_world_root_visit_counts"][:, start:end]
        budgets = arrays["simulation_budgets"][i]
        valid = int(arrays["valid_world_count"][i])
        if np.any(visits < 0) or np.any(world_visits < 0) or np.any(budgets < 0):
            raise ValueError(f"game {game_id} sample {i}: negative visits or budget")
        if not np.array_equal(visits, world_visits.sum(axis=0)):
            raise ValueError(f"game {game_id} sample {i}: root visits mismatch")
        if bool(arrays["searchable"][i]):
            expected_budgets = np.asarray([simulations // worlds + (j < simulations % worlds) for j in range(worlds)], dtype=np.int32)
            if width <= 1 or valid != worlds or not np.array_equal(budgets, expected_budgets) or int(visits.sum()) != simulations or int(arrays["simulations"][i]) != simulations:
                raise ValueError(f"game {game_id} sample {i}: simulation budget mismatch")
            if not np.array_equal(world_visits.sum(axis=1), budgets):
                raise ValueError(f"game {game_id} sample {i}: per-world visits mismatch")
            mean = (world_visits / budgets[:, None]).mean(axis=0)
            if not np.allclose(target, mean, rtol=0, atol=1e-6):
                raise ValueError(f"game {game_id} sample {i}: equal-world policy target mismatch")
        elif width != 1 or valid != 0 or np.any(budgets) or np.any(visits) or int(arrays["simulations"][i]) != 0 or target[0] != 1:
            raise ValueError(f"game {game_id} sample {i}: invalid single-candidate result")
    if int(np.sum(arrays["searchable"])) != metadata.get("policy_samples"):
        raise ValueError(f"game {game_id}: policy sample count mismatch")
    if schema == DATA_SCHEMA:
        replay = metadata.get("replay")
        if not isinstance(replay, dict) or replay.get("header", {}).get("game_seed") != metadata.get("game_seed") or replay.get("result") != terminal:
            raise ValueError(f"game {game_id}: replay metadata mismatch")
        events = replay.get("events")
        if not isinstance(events, list) or sum(not event.get("automatic", False) for event in events) != length:
            raise ValueError(f"game {game_id}: replay decision count mismatch")
        if events and events[-1].get("state_digest_after") != metadata.get("final_state_digest"):
            raise ValueError(f"game {game_id}: replay final digest mismatch")
        if not isinstance(metadata.get("final_state_digest"), str) or len(metadata["final_state_digest"]) != 64:
            raise ValueError(f"game {game_id}: missing final digest")
    else:
        trace = metadata.get("trace")
        if not isinstance(trace, dict) or trace.get("version") != "compact-action-trace-v2":
            raise ValueError(f"game {game_id}: missing compact action trace")
        if trace.get("game_seed") != metadata.get("game_seed") or trace.get("first_player") != metadata.get("first_player"):
            raise ValueError(f"game {game_id}: trace header mismatch")
        if trace.get("terminal_result") != terminal:
            raise ValueError(f"game {game_id}: trace terminal mismatch")
        actions = trace.get("actions")
        if not isinstance(actions, list) or len(actions) != length:
            raise ValueError(f"game {game_id}: trace decision count mismatch")
        for index, action in enumerate(actions):
            if (action.get("decision_index") != index
                    or action.get("player_id") != int(arrays["player_id"][index])
                    or action.get("phase") != PHASE_ORDER[int(arrays["phase"][index])].value
                    or action.get("candidate_index") != int(arrays["chosen_candidate_index"][index])
                    or not isinstance(action.get("action"), dict)):
                raise ValueError(f"game {game_id} sample {index}: trace action mismatch")
    return {"decisions": length, "policy_samples": int(np.sum(arrays["searchable"])), "candidates": candidates, "simulations": int(np.sum(arrays["simulations"]))}


@dataclass(frozen=True, slots=True)
class SelfPlaySample:
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    index: int

    @property
    def candidate_slice(self) -> slice:
        offsets = self.arrays["candidate_offsets"]
        return slice(int(offsets[self.index]), int(offsets[self.index + 1]))


@dataclass(frozen=True, slots=True)
class SelfPlayBatch:
    decisions: EncodedDecisionBatch
    policy_targets: np.ndarray
    behavior_policies: np.ndarray
    value_targets: np.ndarray
    policy_weights: np.ndarray
    value_weights: np.ndarray
    references: tuple[dict[str, int | str], ...]


def collate_selfplay_samples(samples: Sequence[SelfPlaySample]) -> SelfPlayBatch:
    if not samples:
        raise ValueError("cannot collate an empty self-play batch")
    width = max(sample.candidate_slice.stop - sample.candidate_slice.start for sample in samples)
    count = len(samples)
    candidates = np.zeros((count, width, FEATURE_SCHEMA.action_dim), dtype=np.float32)
    mask = np.zeros((count, width), dtype=np.bool_)
    targets = np.zeros((count, width), dtype=np.float32)
    behaviors = np.zeros((count, width), dtype=np.float32)
    for i, sample in enumerate(samples):
        candidate_slice = sample.candidate_slice
        n = candidate_slice.stop - candidate_slice.start
        candidates[i, :n] = sample.arrays["candidate_features"][candidate_slice]
        mask[i, :n] = True
        targets[i, :n] = sample.arrays["policy_target"][candidate_slice]
        behaviors[i, :n] = sample.arrays["behavior_policy"][candidate_slice]
    decisions = EncodedDecisionBatch(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        **{name: np.stack([s.arrays[name][s.index] for s in samples]) for name in STATE_FIELDS},
        candidate_features=candidates,
        candidate_mask=mask,
    )
    references = tuple({"game_id": int(s.metadata["game_id"]), "decision_index": int(s.arrays["decision_index"][s.index]), "player_id": int(s.arrays["player_id"][s.index])} for s in samples)
    return SelfPlayBatch(
        decisions, targets, behaviors,
        np.asarray([s.arrays["value_target"][s.index] for s in samples], dtype=np.float32),
        np.asarray([s.arrays["policy_weight"][s.index] for s in samples], dtype=np.float32),
        np.asarray([s.arrays["value_weight"][s.index] for s in samples], dtype=np.float32),
        references,
    )


class SelfPlayDataset:
    """Index games, loading only the current archive into memory."""

    def __init__(self, root: Path, *, split: str | None = None) -> None:
        if split not in (None, "train", "validation"):
            raise ValueError("split must be train, validation, or None")
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("protocol") not in ("mcts-selfplay-v1", "mcts-selfplay-v2") or manifest.get("versions", {}).get("data") not in (DATA_SCHEMA, DATA_SCHEMA_V2):
            raise ValueError("self-play manifest schema mismatch")
        config_key = "semantic_config" if manifest["protocol"] == "mcts-selfplay-v1" else "config"
        search = manifest[config_key]["search_config"]
        self.entries: list[tuple[Path, int]] = []
        self._cache: tuple[Path, dict[str, np.ndarray], dict[str, Any]] | None = None
        for path in sorted((self.root / "games").glob("game-*.npz")):
            arrays, metadata = load_game(path)
            game_id = int(path.stem.removeprefix("game-"))
            validate_game_arrays(
                arrays, metadata, expected_game_id=game_id,
                fingerprint=manifest["semantic_fingerprint"],
                simulations=search["simulations"],
                worlds=search["determinizations"],
            )
            if split is None or metadata["split"] == split:
                self.entries.extend((path, i) for i in range(int(metadata["decision_count"])))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> SelfPlaySample:
        path, sample_index = self.entries[index]
        if self._cache is None or self._cache[0] != path:
            arrays, metadata = load_game(path)
            self._cache = path, arrays, metadata
        return SelfPlaySample(self._cache[1], self._cache[2], sample_index)

    def __iter__(self) -> Iterator[SelfPlaySample]:
        for i in range(len(self)):
            yield self[i]
