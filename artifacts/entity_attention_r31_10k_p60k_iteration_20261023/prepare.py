"""Prepare an R31 self-iteration branch with a 40k history and a growing 60k pool."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import _read_game_metadata
from splendor.catalog import load_default_catalog
from splendor.random_budget_iteration import Config, check

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
NAME = ART.name
OUTPUT = ROOT / "runs" / NAME
RUNTIME = ROOT / "runtime" / NAME
PARENT_NAME = "entity_attention_r20_s256n4_noble12_iteration_20261022"
PARENT = ROOT / "runs" / PARENT_NAME
PARENT_SOURCE = ROOT / "runtime" / PARENT_NAME / "source"
PARENT_ART = ROOT / "artifacts" / PARENT_NAME
BASE = ROOT / "configs/entity_attention_v1/random_budget_r20_s256n4_noble12_20261022.json"
LATEST_COLLECTION = ROOT / "artifacts/collection_tuning_20261022/configs/full_attention_best_4000.json"
CONFIG = ROOT / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json"
DRIVER = ROOT / "src/splendor/random_budget_iteration.py"
R31_SHA = "c57a2a8a7721fe65de90dbc1336483f5d0358d10c4ff4d5c893986e88e588f6c"


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen branch input changed: {path}")
    else:
        write_json(path, value)


def historical_inputs(completed, catalog_fingerprint):
    records, sources = [], []
    for iteration in range(22, 32):
        if iteration not in completed:
            raise ValueError(f"Parent iteration {iteration} is incomplete")
        stage = PARENT / "rounds" / f"round-{iteration-1:03d}"
        index_path = stage / "collection_index.json"
        manifest_path, summary_path = stage / "selfplay/manifest.json", stage / "selfplay/summary.json"
        manifest, summary = read_json(manifest_path), read_json(summary_path)
        if (manifest["checkpoint_identity"]["sha256"] != completed[iteration-1]["checkpoint"]["sha256"]
                or manifest["catalog_fingerprint"] != catalog_fingerprint):
            raise ValueError("Historical collection teacher or catalog differs")
        rows = read_json(index_path)["games"]
        if (len(rows) != 4000 or summary["completed_games"] != 4000 or summary["expected_games"] != 4000
                or summary["failed_games"] or summary["missing_games"]
                or len({r["game_id"] for r in rows}) != 4000
                or {r["round_index"] for r in rows} != {iteration+42}):
            raise ValueError(f"Historical iteration {iteration} lacks4000 complete unique games")
        for row in rows:
            stat = Path(row["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
                raise ValueError(f"Historical archive changed: {row['path']}")
        metadata = _read_game_metadata(Path(rows[0]["path"]))
        if metadata["trace"]["catalog_fingerprint"] != catalog_fingerprint:
            raise ValueError("Historical archive catalog differs")
        sources.append({"parent_iteration": iteration, "index": identity(index_path),
            "manifest": identity(manifest_path), "summary": identity(summary_path),
            "teacher": manifest["checkpoint_identity"], "catalog_fingerprint": catalog_fingerprint})
        records.extend(rows)
    return sorted(records, key=lambda g: (g["round_index"], g["game_id"])), sources


def freeze_source():
    """Use the parent's exact implementation, replacing only the pool driver."""
    destination, path = RUNTIME / "source", ART / "source.json"
    if not path.exists():
        parent_record = read_json(PARENT_ART / "source.json")
        inherited = {}
        for relative, expected in parent_record["snapshot_files"].items():
            source = PARENT_SOURCE / relative
            include = ((relative.startswith(("src/", "scripts/")) and source.suffix in (".py", ".cpp", ".so"))
                       or relative.startswith("data/"))
            if not include:
                continue
            if sha256(source) != expected:
                raise ValueError(f"Parent frozen source changed: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inherited[relative] = expected
        for source in (DRIVER, CONFIG, Path(__file__)):
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        files = {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
        changed = {relative: {"parent_sha256": expected, "branch_sha256": files[relative]}
            for relative, expected in inherited.items() if files[relative] != expected}
        if set(changed) != {str(DRIVER.relative_to(ROOT))}:
            raise ValueError("Only the partial-pool driver may differ from the parent implementation")
        freeze(path, {"base_source": identity(PARENT_ART / "source.json"), "changed_from_parent": changed,
            "snapshot_files": files, "workspace_at_preparation": source_identity(ROOT),
            "scope": "Parent R31 collection/model/learner/native/catalog frozen implementation, only random_budget_iteration.py replaced. Current unrelated model/auxiliary edits are not included."})
    for relative, expected in read_json(path)["snapshot_files"].items():
        if sha256(destination / relative) != expected:
            raise ValueError(f"Branch frozen source changed: {relative}")


def main():
    completed = {r["iteration"]: r for r in read_json(PARENT / "state.json")["completed_rounds"]}
    checkpoint = completed[31]["checkpoint"]
    if checkpoint["sha256"] != R31_SHA or sha256(Path(checkpoint["path"])) != R31_SHA:
        raise ValueError("Expected the original raw R31 full checkpoint")
    catalog = load_default_catalog()
    records, sources = historical_inputs(completed, catalog.data_fingerprint)
    manifest = OUTPUT / "inputs/history_r22_r31.jsonl"
    content = "".join(json.dumps(g, sort_keys=True, separators=(",", ":"))+"\n" for g in records)
    if manifest.exists():
        if manifest.read_text() != content:
            raise ValueError("Frozen R22-R31 history differs")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(content)
    raw = copy.deepcopy(read_json(BASE))
    raw.pop("continuation_state")
    raw.update(run_id=NAME, initial_checkpoint=checkpoint,
        history_manifest={"path": str(manifest), "sha256": sha256(manifest)},
        first_round=65, last_round=74, expected_games_per_round=4000,
        new_games_per_round=10000, replay_max_games=60000, updates_per_round=3000,
        cache_bytes=16 * 1024**3,
        validation_steps=[0, 100, 250, 500, 1000, 2000, 3000])
    raw["learner"]["learning_rate"] = 2e-5
    raw["selfplay"]["execution"] = copy.deepcopy(read_json(LATEST_COLLECTION)["execution"])
    raw["search_evaluation"]["execution"] = copy.deepcopy(raw["selfplay"]["execution"])
    raw["search_evaluation"]["execution"]["pipeline_slots"] = 1
    config = Config.from_dict(raw)
    _, preflight = check(config)
    assert preflight["first_pool_games"] == 50000 and preflight["first_new_policy_fraction"] == .2
    freeze(CONFIG, raw)
    splits = {split: {"games": sum(g["split"] == split for g in records),
        "positions": sum(g["decision_count"] for g in records if g["split"] == split),
        "policy_positions": sum(g["policy_samples"] for g in records if g["split"] == split)}
        for split in ("train", "validation")}
    schedule = []
    for local in range(1, 8):
        total = min(60000, 40000+10000*local)
        original = max(0, 40000-max(0, local-2)*10000)
        schedule.append({"branch_iteration": local, "parent_lineage_round": 31+local,
            "active_games": total, "initial_history_games": original, "new_games": total-original,
            "policy_new_fraction": (total-original)/total})
    freeze(ART / "plan.json", {"experiment": NAME, "status": "prepared_after_requested_reset",
        "config": identity(CONFIG), "runner": identity(Path(__file__)), "driver": identity(DRIVER),
        "base_config": identity(BASE), "collection_execution_source": identity(LATEST_COLLECTION),
        "reset_record": identity(ART / "reset_20261023/record.json"),
        "initial_checkpoint": checkpoint, "initial_total_train_step": preflight["initial_step"],
        "source_parent_iterations": [22, 31], "source_rounds": sources,
        "historical_games": len(records), "historical_splits": splits, "preflight": preflight,
        "pool_schedule": schedule, "catalog_fingerprint": catalog.data_fingerprint,
        "catalog_files": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "data").glob("*")) if p.is_file()},
        "round_numbering": "Branch iteration1/round-000 is lineage R32, data round_index74 and checkpoint round_index75; old completed-round state is not imported because the history is explicitly rebuilt.",
        "limit": "continuous; no round or session-update limit by default",
        "scope": "User-started first collection was interrupted before any completed games or learner updates; generated run/collection outputs deleted on explicit request. Prepared again without restarting. Original R31 model/AdamW/all RNG; full40000 games from R22-R31, original targets/splits. New10000/round, capacity60000, lr2e-5,3000updates, compact replay cache16GiB (not a process RSS hard cap). Latest attention execution W12/A2000/B500/local_microbatch128/pipeline2; search evaluation reuses it with owner pipeline1. Historical policy fixed25% subset and random-original-then-online-FIFO eviction remain unchanged. Original evaluation anchor R43 and greedy/search cadence remain branch-local. Parent frozen implementation retained except pool preflight driver."})
    freeze_source()
    print({"status": "prepared_after_requested_reset", **preflight}, flush=True)


if __name__ == "__main__":
    main()
