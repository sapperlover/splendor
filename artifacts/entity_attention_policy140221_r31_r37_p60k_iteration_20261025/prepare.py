"""Prepare a policy-auxiliary online branch from the selected offline model."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import _read_game_metadata
from splendor.catalog import load_default_catalog
from splendor.fixed_holdout import check_disjoint, read_manifest
from splendor.offline_train import _digest
from splendor.random_budget_iteration import Config, check, retain_games

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
NAME = ART.name
OUTPUT, RUNTIME = ROOT / "runs" / NAME, ROOT / "runtime" / NAME
OLD = ROOT / "runs/entity_attention_r20_s256n4_noble12_iteration_20261022"
ONLINE = ROOT / "runs/entity_attention_r31_10k_p60k_iteration_20261023"
OFFLINE = ROOT / "runs/entity_attention_128x2_opponent_policy_r21_r37_20261024"
OFFLINE_ART = ROOT / "artifacts" / OFFLINE.name
BASE = ROOT / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json"
CONFIG = ROOT / "configs/entity_attention_v1/random_budget_policy140221_r31_r37_p60k_20261025.json"
SELECTION_SHA = "e4edeb20e8caa3c1abec0146d3757594dd6e0428bd833badd0cb3915c2299a9f"


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Frozen preparation input changed: {path}")
    if not path.exists():
        write_json(path, value)


def selected_checkpoint():
    selection = read_json(OFFLINE / "selection.json")
    if selection != read_json(OFFLINE_ART / "selected_candidate.json"):
        raise ValueError("Offline selected checkpoint differs from pre-arena freeze")
    rows = []
    for candidate in selection["candidates"]:
        raw = read_json(OFFLINE / "full_validation" / f"step-{candidate['step']:09d}.json")
        if raw["checkpoint_sha256"] != candidate["checkpoint_sha256"] or raw["metrics"] != candidate["metrics"]:
            raise ValueError("Original full-holdout evidence changed")
        rows.append((raw["metrics"]["policy_kl"] + raw["metrics"]["value_loss"], raw["step"]))
    if min(rows) != (selection["selection_objective"], selection["selected_step"]):
        raise ValueError("Offline selection is not the full mainKL+MSE minimum")
    checkpoint = {"path": selection["checkpoint"], "sha256": selection["checkpoint_sha256"]}
    if (selection["selected_step"] != 140221 or checkpoint["sha256"] != SELECTION_SHA
            or sha256(Path(checkpoint["path"])) != SELECTION_SHA):
        raise ValueError("Expected the original policy-only step140221 full checkpoint")
    payload = load_alphazero_checkpoint(checkpoint["path"])
    auxiliary = payload.replay_identity["continuation_auxiliary"]
    if (payload.total_train_step != 140221 or auxiliary["shortterm_value_weight"] != 0
            or auxiliary["opponent_policy_weight"] != .15):
        raise ValueError("Selected model is not the policy-only offline model")
    return checkpoint, auxiliary, selection


def history(catalog):
    states = {run: {r["iteration"]: r for r in read_json(run / "state.json")["completed_rounds"]}
              for run in (OLD, ONLINE)}
    rows, sources, counts = [], [], {}
    for lineage in range(31, 38):
        run, local, expected = (OLD, 31, 4000) if lineage == 31 else (ONLINE, lineage-31, 10000)
        completed = states[run]
        if local not in completed:
            raise ValueError(f"Source R{lineage} is incomplete")
        teacher = (completed[local-1]["checkpoint"] if local > 1 else read_json(run / "config.json")["initial_checkpoint"])
        stage = run / "rounds" / f"round-{local-1:03d}"
        index, manifest, summary = (stage / "collection_index.json", stage / "selfplay/manifest.json", stage / "selfplay/summary.json")
        games, meta, result = read_json(index)["games"], read_json(manifest), read_json(summary)
        if (meta["checkpoint_identity"] != teacher or meta["catalog_fingerprint"] != catalog.data_fingerprint
                or len(games) != expected or result["completed_games"] != expected or result["expected_games"] != expected
                or result["failed_games"] or result["missing_games"]
                or {g["round_index"] for g in games} != {lineage+42}
                or len({g["game_id"] for g in games}) != expected):
            raise ValueError(f"R{lineage} collection count/lineage/catalog differs")
        for g in games:
            stat = Path(g["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (g["size"], g["mtime_ns"]):
                raise ValueError(f"Historical archive changed: {g['path']}")
        if _read_game_metadata(Path(games[0]["path"]))["trace"]["catalog_fingerprint"] != catalog.data_fingerprint:
            raise ValueError("Historical NPZ catalog differs")
        counts[str(lineage+43)] = expected
        rows.extend(games)
        sources.append({"lineage_round": lineage, "local_iteration": local, "games": expected,
            "index": identity(index), "manifest": identity(manifest), "summary": identity(summary), "teacher": teacher})
    return sorted(rows, key=lambda g: (g["round_index"], g["game_id"])), sources, counts, states[ONLINE][6]["checkpoint"]


def initial_holdout():
    source = ROOT / "artifacts/entity_attention_r31_10k_p60k_iteration_20261023/six_round_holdout_20261023/next_block_manifest.json"
    manifest = copy.deepcopy(read_json(source))
    for index in manifest["sources"]:
        if sha256(Path(index["path"])) != index["sha256"]:
            raise ValueError("R32-R37 holdout source index changed")
    for game in manifest["games"]:
        if sha256(Path(game["path"])) != game["archive_sha256"]:
            raise ValueError("R32-R37 holdout archive changed")
    manifest.update(source_run=str(ONLINE), source_lineage_rounds=list(range(32, 38)),
        evaluated_iterations=list(range(1, 7)), evaluated_lineage_rounds=list(range(38, 44)),
        selection="All original R32-R37 validation; new local1-6/lineageR38-R43. Source iterations are local to source_run. New block every6 rounds uses preceding6 branch rounds.")
    manifest["digest"] = _digest({k: v for k, v in manifest.items() if k != "digest"})
    destination = ART / "initial_holdout.json"
    freeze(destination, manifest)
    return {"manifest": {"path": str(destination), "sha256": sha256(destination)}, "batch_size": 512, "block_rounds": 6}, identity(source)


def freeze_source():
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        for path in (CONFIG, Path(__file__), ROOT / "src/splendor/native_search.cpp",
                     *ROOT.glob("src/splendor/_native_search*.so"), *(ROOT / "data").glob("splendor_*")):
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
        freeze(record_path, record)
    for path, expected in read_json(record_path)["snapshot_files"].items():
        if sha256(destination / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")


def main():
    checkpoint, auxiliary, selection = selected_checkpoint()
    catalog = load_default_catalog()
    rows, sources, counts, anchor = history(catalog)
    manifest = OUTPUT / "inputs/history_r31_r37.jsonl"
    content = "".join(json.dumps(g, sort_keys=True, separators=(",", ":"))+"\n" for g in rows)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError("Frozen64k history changed")
    if not manifest.exists():
        manifest.write_text(content)
    holdout, holdout_source = initial_holdout()
    raw = copy.deepcopy(read_json(BASE))
    raw.pop("continuation_state", None)
    raw.pop("fixed_holdout", None)
    raw.update(run_id=NAME, initial_checkpoint=checkpoint, history_manifest={"path": str(manifest), "sha256": sha256(manifest)},
        first_round=74, last_round=80, expected_games_per_round=counts, initial_history_eviction=True,
        continuation_auxiliary=auxiliary, encoder_grad_norm_every_steps=100, six_round_holdout=holdout,
        evaluation_anchor=anchor)
    for kind in ("greedy", "search"):
        panel = raw[f"{kind}_evaluation"]
        panel["policy_a"]["name"] = "policy_aux_online_candidate"
        panel["policy_b"].update(checkpoint=anchor["path"], name="raw_R37")
    if sha256(Path(anchor["path"])) != anchor["sha256"]:
        raise ValueError("Raw R37 anchor changed")
    config = Config.from_dict(raw)
    _, preflight = check(config)
    assert len(rows) == 64000 and preflight["first_pool_games"] == 60000
    assert preflight["first_evicted_history_games"] == 14000 and preflight["first_new_policy_fraction"] == 1/6
    read_manifest(holdout)
    check_disjoint(read_json(ART / "initial_holdout.json"), rows)
    freeze(CONFIG, raw)
    split_counts = {split: {"games": sum(g["split"] == split for g in rows),
        "positions": sum(g["decision_count"] for g in rows if g["split"] == split),
        "policy_positions": sum(g["policy_samples"] for g in rows if g["split"] == split)} for split in ("train", "validation")}
    freeze(ART / "plan.json", {"experiment": NAME, "status": "prepared_not_started", "config": identity(CONFIG),
        "prepare": identity(Path(__file__)), "base_iteration_config": identity(BASE),
        "initial_checkpoint": checkpoint, "offline_selection": identity(OFFLINE / "selection.json"),
        "offline_selected_objective": selection["selection_objective"], "initial_total_step": 140221,
        "source_lineage_rounds": [31, 37], "source_rounds": sources, "historical_games": len(rows), "historical_splits": split_counts,
        "preflight": preflight, "initial_holdout": identity(ART / "initial_holdout.json"), "holdout_source": holdout_source,
        "anchor": anchor, "catalog_fingerprint": catalog.data_fingerprint,
        "first_round": {"branch_local": 1, "lineage_label": 38, "data_round_index": 80, "checkpoint_round_index": 81,
            "source_history_games": 64000, "new_games": 10000, "evicted_original_games": 14000,
            "retained_history_games": 50000, "active_games": 60000, "policy_new_fraction": 1/6},
        "scope": "Prepared only, no production collection/training/arena launched. Offline policy-only minimum140221 full weights/AdamW/all RNG preserved; lr2e-5,3000updates/round, new10000/round, max60000. Full originalR31-R37 history64k, original splits/targets; first round randomly evicts14k original history using existing retention seed, then old-first random/FIFO. Historical policy25percent subset retained. Policy auxiliary0.15 continues, shorttermvalue off. Latest attention collectionW12/A2000/B500/mb128/pipeline2, searchA W12/A1000/B500/mb128/pipeline2; fixedrawR37 greedy1000pair each round/searchS1024N1 total1024 500pair every3local rounds. Every6rounds full prior6 validation, initialR32-R37; pooled2048 diagnostics unchanged. Streaming metrics/state-only value transfers/batched scalars/hash-only usage from current trainer. Online retains16GiB LRU/synchronous drawer; offline4worker default is not applied to changing online replay."})
    freeze_source()
    print({"status": "prepared_not_started", **preflight}, flush=True)


if __name__ == "__main__":
    main()
