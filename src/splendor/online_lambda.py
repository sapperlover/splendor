"""Reuse λ targets from the search that actually produced each replay game."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .alphazero_recording import read_json, sha256, write_json
from .offline_train import _digest
from .selfplay_data import DATA_SCHEMA_V3
from .value_outcome import archived_search_lambda_targets

PROTOCOL = "online-archived-search-lambda-v1"
ROOT_FIELDS = ("global_features", "player_id", "value_target", "candidate_offsets",
               "valid_world_count", "per_world_root_visit_counts", "per_world_root_value_sums")


def _archive_identity(game):
    return _digest([game["path"], game["size"], game["mtime_ns"], game["decision_count"]])


def _write_source(folder, identity, games, teacher_sha):
    positions = sum(game["decision_count"] for game in games)
    targets = np.empty(positions, dtype=np.float32)
    mask = np.empty(positions, dtype=np.bool_)
    entries, source = {}, None
    offset = 0
    for game in games:
        path = game["path"]
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
            if metadata.get("data_schema") != DATA_SCHEMA_V3:
                raise ValueError(f"Online λ requires V3 search statistics: {path}")
            fields = ("round_index", "run_id", "game_id", "split", "decision_count")
            if any(metadata.get(key) != game[key] for key in fields):
                raise ValueError(f"Online λ source game identity changed: {path}")
            provenance = {key: metadata[key] for key in ("checkpoint_sha256", "search_training")}
            if teacher_sha is not None and provenance["checkpoint_sha256"] != teacher_sha:
                raise ValueError(f"Online λ collection teacher does not match the round: {path}")
            if source is None:
                source = provenance
            elif provenance != source:
                raise ValueError(f"Online λ source contains mixed teachers or search settings: {path}")
            arrays = {name: archive[name] for name in ROOT_FIELDS}
        labels, valid, _ = archived_search_lambda_targets(arrays, metadata, identity["outcome_lambda"])
        stop = offset + game["decision_count"]
        targets[offset:stop], mask[offset:stop] = labels, valid
        entries[str(game["game_id"])] = [offset, _archive_identity(game)]
        offset = stop

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{folder.name}-", dir=folder.parent) as temporary:
        pending = Path(temporary)
        hashes = {}
        for name, array in (("targets", targets), ("mask", mask)):
            path = pending / f"{name}.npy"
            with path.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            hashes[name] = sha256(path)
        write_json(pending / "manifest.json", {"identity": identity, **source,
                   "games": entries, "positions": positions, "sha256": hashes})
        pending.rename(folder)


def prepare_round_targets(root, manifest_path, snapshot, *, lam, collection_round, previous_sources):
    """Publish labels by source collection and bind the active Train rows to this round."""
    groups = {}
    for game in snapshot["games"]:
        if game["split"] == "train":
            groups.setdefault((game["round_index"], game["run_id"]), []).append(game)
    committed = read_json(manifest_path) if manifest_path.exists() else None
    positions = snapshot["value_positions"]
    targets = np.empty(positions, dtype=np.float32)
    mask = np.empty(positions, dtype=np.bool_)
    offsets, sources = {}, []
    offset = 0
    for (round_index, run_id), games in groups.items():
        identity = {"protocol": PROTOCOL, "outcome_lambda": lam, "round_index": round_index, "run_id": run_id}
        folder = root / f"{round_index:06d}-{_digest(run_id)}"
        teacher_sha = snapshot["teacher"]["sha256"] if round_index == collection_round else None
        source_path = folder / "manifest.json"
        expected_source = previous_sources.get(str(source_path))
        if not folder.exists():
            if committed is not None or expected_source is not None:
                raise ValueError(f"Committed round λ labels are missing: {folder}")
            _write_source(folder, identity, games, teacher_sha)
        source_sha = sha256(source_path)
        if expected_source is not None and expected_source != source_sha:
            raise ValueError(f"Frozen λ source manifest changed: {source_path}")
        source = read_json(source_path)
        if source["identity"] != identity or (teacher_sha is not None and source["checkpoint_sha256"] != teacher_sha):
            raise ValueError(f"Frozen λ source identity changed: {folder}")
        stored = []
        for name, dtype in (("targets", np.float32), ("mask", np.bool_)):
            path = folder / f"{name}.npy"
            if sha256(path) != source["sha256"][name]:
                raise ValueError(f"Frozen λ {name} changed: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != dtype or array.shape != (source["positions"],):
                raise ValueError(f"Frozen λ {name} has invalid dimensions or dtype: {path}")
            stored.append(array)
        source_targets, source_mask = stored
        first = offset
        for game in games:
            entry = source["games"].get(str(game["game_id"]))
            if entry is None or entry[1] != _archive_identity(game):
                raise ValueError(f"Frozen λ game identity changed: {game['path']}")
            start, count = entry[0], game["decision_count"]
            offsets[round_index, game["game_id"]] = offset
            targets[offset:offset+count] = source_targets[start:start+count]
            mask[offset:offset+count] = source_mask[start:start+count]
            offset += count
        sources.append({"path": str(source_path), "sha256": source_sha,
                        "checkpoint_sha256": source["checkpoint_sha256"], "games": len(games),
                        "positions": offset-first, "eligible_positions": int(np.count_nonzero(mask[first:offset]))})

    eligible = sum(source["eligible_positions"] for source in sources)
    if offset != positions or not eligible:
        raise ValueError("Online λ replay needs matching Train rows and at least one valid label")
    manifest = {"protocol": PROTOCOL, "outcome_lambda": lam, "snapshot_digest": snapshot["digest"],
                "sources": sources, "positions": positions, "eligible_positions": eligible,
                "target_coverage": eligible / positions}
    if committed is None:
        write_json(manifest_path, manifest)
    elif committed != manifest:
        raise ValueError("Frozen round λ labels changed on resume")
    return (targets, offsets, mask), manifest
