"""Freeze recent, original S1024/N1 attention self-play, without copying NPZs."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.selfplay_data import load_game, validate_game_arrays

HERE = Path(__file__).resolve().parent
SOURCES = {
    "attention_3x1000": "runs/entity_attention_10x1000_seed1",
    "hybrid_9x1000": "runs/entity_attention_hybrid_static_graph_seed1",
    "n1_50x200": "runs/entity_attention_hybrid_s1024_n1_from_round9_seed1",
    "split_20x200": "runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1",
    "behavior_m4_12x200": "runs/entity_attention_avg4_behavior_20261009/seed1_avg4",
    "steps50_2x200": "runs/entity_attention_iteration_trend_seed1_20261008/seed1_steps50",
    "native_probe_200": "runs/entity_attention_chance_training_probe_20261007/native",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def archive_record(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("metadata_json.npy") as handle:
            metadata = json.loads(np.load(handle, allow_pickle=False).tobytes())
        decoded_upper = sum(item.file_size for item in archive.infolist())
    stat = path.stat()
    if metadata["status"] != "natural_terminal" or metadata["split"] not in ("train", "validation"):
        raise ValueError(f"noncommitted game: {path}")
    if metadata["decision_count"] <= 0 or metadata["terminal_result"]["winner_ids"] not in ([0], [1], [0, 1]):
        raise ValueError(f"invalid terminal game: {path}")
    # Ignore run names/collection timing: the same seed, teacher and trajectory
    # may be present in independently named forks. Conflicting splits fail below.
    trajectory = {k: metadata[k] for k in ("game_seed", "first_player", "terminal_result")}
    trajectory["actions"] = metadata["trace"]["actions"]
    record = {k: metadata[k] for k in ("game_id", "run_id", "split", "decision_count", "policy_samples", "checkpoint_sha256", "game_seed")}
    record.update(path=str(path.resolve()), source_round_index=metadata["round_index"],
                  archive_sha256=sha256(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                  decoded_upper_bytes=decoded_upper, trajectory_sha256=digest(trajectory))
    return record, metadata


def prepare(output=HERE):
    output = Path(output)
    if (output / "pool_summary.json").exists():
        result = read_json(output / "pool_summary.json")
        if sha256(output / "pool_games.jsonl") != result["manifest_sha256"]:
            raise ValueError("frozen manifest changed")
        return result
    records, collections, excluded, duplicates, checks = [], [], [], [], []
    seen, teachers = {}, {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for source, root in SOURCES.items():
            root = Path(root)
            manifests = sorted(root.glob("rounds/*/selfplay/manifest.json")) or list(root.glob("selfplay/manifest.json"))
            for manifest_path in manifests:
                manifest = read_json(manifest_path)
                config = manifest.get("config", manifest.get("semantic_config"))
                search = config["search_config"]
                if search["simulations"] != 1024 or search["determinizations"] != 1 or manifest["versions"]["search"] != "determinized-puct-native-v1":
                    raise ValueError(f"wrong search: {manifest_path}")
                summary_path = manifest_path.with_name("summary.json")
                if not summary_path.exists():
                    excluded.append({"path": str(manifest_path.parent), "reason": "no completed collection summary"})
                    continue
                summary = read_json(summary_path)
                paths = sorted((manifest_path.parent / "games").glob("game-*.npz"))
                if not paths:
                    excluded.append({"path": str(manifest_path.parent), "reason": "empty collection"})
                    continue
                if summary["completed_games"] != len(paths) or summary["expected_games"] != len(paths):
                    raise ValueError(f"incomplete collection: {manifest_path}")
                if any(summary.get(k, 0) for k in ("failed_games", "exceptions", "engineering_truncations")):
                    excluded.append({"path": str(manifest_path.parent), "games": len(paths),
                        "reason": "collection has recorded failures; conservatively excluded even if subsequently completed",
                        "counters": {k: summary.get(k, 0) for k in ("failed_games", "exceptions", "engineering_truncations", "batch_failures")}})
                    continue
                teacher = manifest["checkpoint_identity"]
                if teacher["sha256"] not in teachers:
                    checkpoint = Path(teacher["path"])
                    if sha256(checkpoint) != teacher["sha256"]:
                        raise ValueError(f"teacher checksum mismatch: {checkpoint}")
                    payload = load_alphazero_checkpoint(checkpoint)
                    if payload.model_config["architecture"] != "entity_attention_v1":
                        raise ValueError(f"not attention: {checkpoint}")
                    teachers[teacher["sha256"]] = {**teacher, "model_config": payload.model_config}
                group = len(collections)
                accepted = 0
                for record, metadata in executor.map(archive_record, paths):
                    if metadata["semantic_fingerprint"] != manifest["semantic_fingerprint"] or metadata["checkpoint_sha256"] != teacher["sha256"]:
                        raise ValueError(f"archive/manifest mismatch: {record['path']}")
                    if metadata["run_id"] != config["run_id"] or metadata["round_index"] != config["round_index"]:
                        raise ValueError(f"archive source identity mismatch: {record['path']}")
                    key = record["trajectory_sha256"]
                    if key in seen:
                        original = seen[key]
                        if original["split"] != record["split"]:
                            raise ValueError("duplicate trajectory crosses original train/validation splits")
                        duplicates.append({"discarded": record["path"], "retained": original["path"], "split": record["split"]})
                        continue
                    record.update(source_id=source, round_index=group)
                    seen[key] = record
                    records.append(record)
                    accepted += 1
                # Existing production replay already validates all active games.
                # Check representative original archives again, not a new full replay.
                numeric_paths = paths if summary.get("batch_failures", 0) else dict.fromkeys((paths[0], paths[len(paths)//2], paths[-1]))
                for path in numeric_paths:
                    arrays, metadata = load_game(path)
                    checked = validate_game_arrays(arrays, metadata,
                        expected_game_id=int(path.stem.removeprefix("game-")),
                        fingerprint=manifest["semantic_fingerprint"], simulations=1024, worlds=1)
                    checks.append({"path": str(path), **checked})
                collections.append({"round_index": group, "source_id": source, "manifest": str(manifest_path.resolve()),
                    "manifest_sha256": sha256(manifest_path), "summary": str(summary_path.resolve()),
                    "summary_sha256": sha256(summary_path), "games": len(paths), "accepted": accepted,
                    "config": config, "versions": manifest["versions"], "teacher": teacher,
                    "historical_batch_failures": summary.get("batch_failures", 0),
                    "numeric_verification_games": len(numeric_paths)})
                print(f"入池 {source} {manifest_path.parent.parent.name}: {accepted}/{len(paths)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pool_games.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    by_source = {}
    for source in SOURCES:
        selected = [r for r in records if r["source_id"] == source]
        row = {"games": len(selected), "positions": sum(r["decision_count"] for r in selected),
               "policy_positions": sum(r["policy_samples"] for r in selected)}
        for split in ("train", "validation"):
            subset = [r for r in selected if r["split"] == split]
            row[split] = {"games": len(subset), "positions": sum(r["decision_count"] for r in subset),
                          "policy_positions": sum(r["policy_samples"] for r in subset)}
        by_source[source] = row
    totals = {k: sum(row[k] for row in by_source.values()) for k in ("games", "positions", "policy_positions")}
    for split in ("train", "validation"):
        totals[split] = {k: sum(row[split][k] for row in by_source.values()) for k in ("games", "positions", "policy_positions")}
    result = {"protocol": "recent-attention-s1024n1-pool-v1", "by_source": by_source, "totals": totals,
        "manifest_sha256": sha256(output / "pool_games.jsonl"), "collections": len(collections),
        "archive_bytes": sum(r["size"] for r in records), "decoded_upper_bytes": sum(r["decoded_upper_bytes"] for r in records),
        "duplicate_games": len(duplicates), "label": "original behavior-search policy_target; M4 sidecars excluded",
        "split": "original whole-game train/validation; cross-split duplicate trajectories fail",
        "excluded_categories": ["S512/N2", "chance/CPW search", "smokes", "evaluation games", "relabel-only duplicates", "empty/incomplete collections", "old DeepSets teacher data"],
        "verification": "all manifests, summaries, teacher hashes/configs, archive hashes/metadata; 3 numeric archive checks per collection, all games for recovered historical batch failures; no new full rules replay"}
    write_json(output / "sources.json", {"collections": collections, "teachers": teachers, "duplicates": duplicates, "empty_collections": excluded, "numeric_checks": checks})
    write_json(output / "pool_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    prepare(parser.parse_args().output)
