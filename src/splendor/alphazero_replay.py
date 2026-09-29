"""Frozen cross-generation replay index over committed MCTS self-play runs.

The index keeps the most recent complete games across collection rounds.  A
game is identified by ``(round_index, game_id)``: every round restarts
``game_id`` at zero, so a plain ``game_id`` is not unique across runs.  Only
formally committed ``natural_terminal`` archives enter the index; failures,
truncations and temporary files never do, and eviction only drops an entry
from the active index while the archive file stays on disk for diagnostics.

Whole-game archives are validated once when they first enter the index.  The
validated result is persisted in a small JSON sidecar (``replay_index.json``)
keyed by path, size and mtime, so later rounds do not re-verify unchanged
archives and the training inner loop never re-reads a manifest per sample.
Random access goes through a bounded whole-game LRU cache; games are never
partially split between train and validation.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .rng import DeterministicRng
from .selfplay_data import (
    DATA_SCHEMA, DATA_SCHEMA_V2, SelfPlaySample, load_game, validate_game_arrays,
)

INDEX_PROTOCOL = "alphazero-replay-index-v1"
DEFAULT_CACHE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReplayRun:
    """One collection run that contributes complete games to the index."""

    round_index: int
    run_id: str
    root: Path

    def __post_init__(self) -> None:
        if type(self.round_index) is not int or self.round_index < 0:
            raise ValueError("round_index must be a nonnegative integer")
        if not self.run_id:
            raise ValueError("replay run_id must be nonempty")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)


@dataclass(frozen=True, slots=True)
class ReplayGame:
    round_index: int
    run_id: str
    game_id: int
    split: str
    decision_count: int
    policy_samples: int
    path: Path

    @property
    def key(self) -> tuple[int, int]:
        return (self.round_index, self.game_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "split": self.split,
            "decision_count": self.decision_count,
            "policy_samples": self.policy_samples,
            "path": str(self.path),
        }


@dataclass(frozen=True, slots=True)
class ReplayReference:
    """Position identity that stays unique across collection runs."""

    round_index: int
    run_id: str
    game_id: int
    decision_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "decision_index": self.decision_index,
        }


class _GameCache:
    """Bounded LRU over whole-game arrays."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("replay cache size must be positive")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[
            tuple[int, int], tuple[dict[str, np.ndarray], dict[str, Any], int]
        ] = OrderedDict()
        self._bytes = 0

    @property
    def bytes(self) -> int:
        return self._bytes

    @property
    def entries(self) -> int:
        return len(self._entries)

    def sample(self, game: ReplayGame, decision_index: int) -> SelfPlaySample:
        key = game.key
        entry = self._entries.get(key)
        if entry is None:
            arrays, metadata = load_game(game.path)
            size = sum(value.nbytes for value in arrays.values())
            entry = (arrays, metadata, size)
            if size <= self.max_bytes:
                # Only whole games that fit the budget are cached; a single
                # oversized archive would otherwise evict every other entry.
                self._entries[key] = entry
                self._bytes += size
                while self._bytes > self.max_bytes and self._entries:
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= evicted[2]
        else:
            self._entries.move_to_end(key)
        arrays, metadata, _ = entry
        return SelfPlaySample(arrays, metadata, decision_index)


class PositionPool:
    """Uniform-with-replacement position sampling inside one frozen pool."""

    def __init__(self, games: Sequence[ReplayGame], cache: _GameCache, label: str) -> None:
        self.label = label
        self._games = tuple(games)
        self._cache = cache
        counts = np.asarray([game.decision_count for game in self._games], dtype=np.int64)
        if len(counts):
            self._starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
            self._game_of_position = np.repeat(np.arange(len(self._games), dtype=np.int64), counts)
        else:
            self._starts = np.zeros(0, dtype=np.int64)
            self._game_of_position = np.zeros(0, dtype=np.int64)
        self._size = int(counts.sum()) if len(counts) else 0

    def __len__(self) -> int:
        return self._size

    @property
    def games(self) -> tuple[ReplayGame, ...]:
        return self._games

    def sample(self, rng: DeterministicRng) -> tuple[SelfPlaySample, ReplayReference]:
        if self._size == 0:
            raise ValueError(f"position pool {self.label!r} is empty")
        position = rng.randbelow(self._size)
        game_index = int(self._game_of_position[position])
        game = self._games[game_index]
        decision_index = int(position - self._starts[game_index])
        return (
            self._cache.sample(game, decision_index),
            ReplayReference(game.round_index, game.run_id, game.game_id, decision_index),
        )


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"replay run has no manifest: {root}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") not in ("mcts-selfplay-v1", "mcts-selfplay-v2"):
        raise ValueError(f"replay run protocol mismatch: {root}")
    if manifest.get("versions", {}).get("data") not in (DATA_SCHEMA, DATA_SCHEMA_V2):
        raise ValueError(f"replay run data schema mismatch: {root}")
    return manifest


def _read_game_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
    except (OSError, ValueError, KeyError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid game archive {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"game archive {path} has invalid metadata")
    return metadata


def _scan_run(run: ReplayRun) -> tuple[list[ReplayGame], dict[str, Any]]:
    manifest = _read_manifest(run.root)
    config_key = "semantic_config" if manifest["protocol"] == "mcts-selfplay-v1" else "config"
    search = manifest[config_key]["search_config"]
    verify = {
        "fingerprint": manifest["semantic_fingerprint"],
        "simulations": int(search["simulations"]),
        "worlds": int(search["determinizations"]),
    }
    games: list[ReplayGame] = []
    for path in sorted((run.root / "games").glob("game-*.npz")):
        game_id = int(path.stem.removeprefix("game-"))
        metadata = _read_game_metadata(path)
        if metadata.get("game_id") != game_id:
            raise ValueError(f"archived game id mismatch: {path}")
        if metadata.get("status") != "natural_terminal":
            raise ValueError(f"archived game is not a natural terminal: {path}")
        if metadata.get("split") not in ("train", "validation"):
            raise ValueError(f"archived game has no usable split: {path}")
        games.append(
            ReplayGame(
                round_index=run.round_index,
                run_id=run.run_id,
                game_id=game_id,
                split=str(metadata["split"]),
                decision_count=int(metadata["decision_count"]),
                policy_samples=int(metadata["policy_samples"]),
                path=path,
            )
        )
    return games, verify


def _load_index_state(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Derived cache: a damaged copy only forces re-verification.
        return {}
    if payload.get("protocol") != INDEX_PROTOCOL or not isinstance(payload.get("games"), list):
        return {}
    state: dict[tuple[int, int], dict[str, Any]] = {}
    for entry in payload["games"]:
        if isinstance(entry, Mapping) and "round_index" in entry and "game_id" in entry:
            state[(int(entry["round_index"]), int(entry["game_id"]))] = dict(entry)
    return state


def _write_index_state(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    payload = {"protocol": INDEX_PROTOCOL, "games": [dict(entry) for entry in entries]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _verify_game(game: ReplayGame, verify: Mapping[str, Any], entry: Mapping[str, Any] | None) -> None:
    """Full validation on first entry; an unchanged archive is trusted afterwards."""

    stat = game.path.stat()
    if entry is not None and entry.get("path") == str(game.path):
        if entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
            return
    arrays, metadata = load_game(game.path)
    validate_game_arrays(
        arrays,
        metadata,
        expected_game_id=game.game_id,
        fingerprint=str(verify["fingerprint"]),
        simulations=int(verify["simulations"]),
        worlds=int(verify["worlds"]),
        split=game.split,
    )


class ReplayIndex:
    """Frozen view of the active replay buffer for one training round."""

    def __init__(
        self,
        games: Sequence[ReplayGame],
        *,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
        evicted_games: int = 0,
    ) -> None:
        ordered = tuple(sorted(games, key=lambda game: game.key))
        keys = [game.key for game in ordered]
        if len(set(keys)) != len(keys):
            raise ValueError("replay games must be unique by (round_index, game_id)")
        self.games = ordered
        self.evicted_games = evicted_games
        self.cache = _GameCache(cache_bytes)

    @property
    def total_games(self) -> int:
        return len(self.games)

    def split_games(self, split: str) -> tuple[ReplayGame, ...]:
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        return tuple(game for game in self.games if game.split == split)

    def position_pool(
        self, *, split: str, before_round: int | None = None, round_index: int | None = None
    ) -> PositionPool:
        """Positions of one split, restricted to a round window."""

        if (before_round is None) == (round_index is None):
            raise ValueError("require exactly one of before_round or round_index")
        games = [
            game
            for game in self.split_games(split)
            if (
                game.round_index < before_round
                if before_round is not None
                else game.round_index == round_index
            )
        ]
        window = before_round if before_round is not None else round_index
        label = f"{split}-{'before' if before_round is not None else 'round'}-{window}"
        return PositionPool(games, self.cache, label)

    def identity(self) -> dict[str, Any]:
        """Lightweight frozen identity recorded in checkpoints and logs."""

        digest = hashlib.sha256()
        train_games = 0
        validation_games = 0
        train_positions = 0
        for game in self.games:
            digest.update(
                f"{game.round_index}:{game.game_id}:{game.decision_count}:{game.split}\n".encode("utf-8")
            )
            if game.split == "train":
                train_games += 1
                train_positions += game.decision_count
            else:
                validation_games += 1
        return {
            "protocol": INDEX_PROTOCOL,
            "total_games": len(self.games),
            "train_games": train_games,
            "validation_games": validation_games,
            "train_positions": train_positions,
            "evicted_games": self.evicted_games,
            "digest": digest.hexdigest(),
            "first_key": list(self.games[0].key) if self.games else None,
            "last_key": list(self.games[-1].key) if self.games else None,
        }


def build_replay_index(
    runs: Sequence[ReplayRun],
    *,
    max_games: int,
    cache_bytes: int = DEFAULT_CACHE_BYTES,
    state_path: Path | None = None,
) -> ReplayIndex:
    """Scan collection runs, keep the most recent ``max_games`` complete games."""

    if type(max_games) is not int or max_games <= 0:
        raise ValueError("max_games must be a positive integer")
    if cache_bytes <= 0:
        raise ValueError("cache_bytes must be positive")
    candidates: list[ReplayGame] = []
    verify_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for run in runs:
        games, verify = _scan_run(run)
        for game in games:
            verify_by_key[game.key] = verify
        candidates.extend(games)
    candidates.sort(key=lambda game: game.key)
    active = candidates[-max_games:]
    state = _load_index_state(state_path)
    for game in active:
        _verify_game(game, verify_by_key[game.key], state.get(game.key))
    if state_path is not None:
        entries: list[dict[str, Any]] = []
        for game in active:
            stat = game.path.stat()
            entry = dict(state.get(game.key, {}))
            entry.update(game.to_dict())
            entry["size"] = stat.st_size
            entry["mtime_ns"] = stat.st_mtime_ns
            entries.append(entry)
        _write_index_state(state_path, entries)
    return ReplayIndex(active, cache_bytes=cache_bytes, evicted_games=len(candidates) - len(active))


def load_index_state(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Read the derived verification cache; missing or damaged files return empty."""

    return _load_index_state(path)
