"""Freeze completed P4-lineage S1024/N1 games, keeping original whole-game splits."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from splendor.alphazero_recording import read_json, sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / "artifacts/entity_attention_recent_pool_20261009/pool.py"
spec = importlib.util.spec_from_file_location("recent_pool_inventory", LEGACY)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
digest = legacy.digest
SOURCES = {
    "p4_r001_009": "runs/entity_attention_p4_pool_iteration_20261010",
    "continuation_r010_043": "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011",
}


def prepare():
    if (HERE / "pool_summary.json").exists():
        summary = read_json(HERE / "pool_summary.json")
        if sha256(HERE / "pool_games.jsonl") != summary["manifest_sha256"]:
            raise ValueError("frozen historical pool manifest changed")
        return summary
    snapshots = {}
    for source, root in SOURCES.items():
        path = ROOT / root / "hybrid_state.json"
        state = read_json(path)
        snapshots[source] = {
            "state_path": str(path.relative_to(ROOT)), "state_sha256": sha256(path),
            "completed_rounds": state["completed_rounds"],
            "absolute_offset": state.get("online_round_offset", 0),
        }
    if [len(snapshots[k]["completed_rounds"]) for k in SOURCES] != [9, 34]:
        raise ValueError("this experiment freezes exactly P4 rounds 1-43; source completion changed")
    legacy.SOURCES = {k: str(ROOT / p) for k, p in SOURCES.items()}
    summary = legacy.prepare(HERE)
    if summary["totals"]["games"] != 43000 or summary["collections"] != 43:
        raise ValueError("expected 43 complete distinct 1000-game collections")
    old_path = ROOT / "artifacts/entity_attention_recent_pool_20261009/pool_games.jsonl"
    old_records = [json.loads(line) for line in old_path.read_text().splitlines()]
    records = [json.loads(line) for line in (HERE / "pool_games.jsonl").read_text().splitlines()]
    old_seeds = {r["game_seed"] for r in old_records}
    new_seeds = [r["game_seed"] for r in records]
    if len(set(new_seeds)) != len(new_seeds) or old_seeds.intersection(new_seeds):
        raise ValueError("game seed reuse within new pool or across the old pool")
    if any(r["round_index"] != r["source_round_index"] for r in records):
        raise ValueError("pool generation IDs must equal original absolute generation IDs")
    write_json(HERE / "inventory.json", {
        "frozen_source_states": snapshots,
        "selected": summary["by_source"],
        "other_usable": {"old_recent_pool": {
            "games": len(old_records), "manifest": str(old_path.relative_to(ROOT)),
            "manifest_sha256": sha256(old_path),
            "use": "available earlier S1024/N1 attention pool; not mixed into this experiment",
        }},
        "total_distinct_games_available": len(old_records) + len(records),
        "old_new_seed_overlap": 0,
        "excluded": ["unfinished round44", "S512/N2", "chance/CPW", "smokes",
                     "arena games", "label sidecars and training-only replay copies"],
        "pool_builder": {"path": str(LEGACY.relative_to(ROOT)), "sha256": sha256(LEGACY)},
        "scope": "all original archive metadata and SHA; representative numeric checks; no full game replay",
    })
    return summary


if __name__ == "__main__":
    prepare()
