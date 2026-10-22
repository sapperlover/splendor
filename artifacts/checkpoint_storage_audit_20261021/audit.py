"""Read-only checkpoint inventory: exact hashes, consumers and suggested storage tiers."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
ACTIVE = "runs/entity_attention_128x2_offline_80k_20261021/"
MLP = "runs/entity_set_mlp_256x4_offline_80k_20261021/"
PATTERN = r'''(?:runs|runtime|play_checkpoints)/[^\s"'<>`\\,;\[\]{}()]+\.pth?'''


def read(path):
    return json.loads(path.read_text())


def local(path):
    path = str(path)
    prefix = str(ROOT) + "/"
    return path[len(prefix):] if path.startswith(prefix) else path.removeprefix("./")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_files():
    command = ["rg", "-o", "--with-filename", "--no-line-number", PATTERN,
        "artifacts", "configs", "docs", "scripts", "src", "README.md",
        "--glob", "*.json", "--glob", "*.jsonl", "--glob", "*.md", "--glob", "*.py",
        "--glob", "*.sh", "--glob", "!checkpoint_storage_audit_20261021/**"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    refs = defaultdict(set)
    for line in result.stdout.splitlines():
        filename, path = line.split(":", 1)
        if filename.startswith(str(ART.relative_to(ROOT)) + "/"):
            continue
        refs[path].add(filename)
    return refs


def main():
    refs = reference_files()
    selected = set()
    metadata_consumers = defaultdict(set)
    # Only run-level configs/selection/progress, not every sample or raw game.
    for run in sorted((ROOT / "runs").iterdir()):
        if not run.is_dir():
            continue
        for name in ("config.json", "selection.json", "progress.json", "state.json"):
            path = run / name
            if not path.is_file():
                continue
            raw = path.read_text()
            for target in re.findall(PATTERN, raw):
                metadata_consumers[target].add(str(path.relative_to(ROOT)))
            if name == "selection.json":
                selection = json.loads(raw)
                if "checkpoint" in selection:
                    selected.add(local(selection["checkpoint"]))
    ema = read(ROOT / "artifacts/entity_attention_ema_17_19_20261020/plan.json")
    ema_sources = {local(p["path"]) for p in ema["checkpoints"] + ema["boundaries"]}
    ema_sources.add(local(ema["initial_checkpoint"]["path"]))
    anchor_names = {
        "runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt",
        "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt",
        "runs/entity_attention_128x2_relations_no_aux_r24_r43_20261019/checkpoints/step-000037423.pt",
    }
    rows, links = [], []
    for top in ("runs", "runtime", "play_checkpoints"):
        for base, dirs, files in os.walk(ROOT / top, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ("selfplay", "references", "usage", "inputs", "source",
                "compile_cache", "triton_cache", "plot_dependencies", "__pycache__") and not d.endswith(".games")]
            for name in files:
                if not name.endswith((".pt", ".pth")):
                    continue
                path = Path(base) / name
                rel = str(path.relative_to(ROOT))
                if path.is_symlink():
                    links.append({"path": rel, "target": os.readlink(path), "available": path.exists()})
                    continue
                stat = path.stat()
                rows.append({"path": rel, "size": stat.st_size, "allocated_bytes": stat.st_blocks * 512,
                    "mtime_ns": stat.st_mtime_ns, "inode": [stat.st_dev, stat.st_ino],
                    "formal_references": sorted(refs.get(rel, ())),
                    "metadata_references": sorted(metadata_consumers.get(rel, ())),
                    "sha256": sha256(path)})
                if len(rows) % 400 == 0:
                    print(f"Hashed {len(rows)} checkpoints", flush=True)
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)

    def priority(row):
        path = row["path"]
        return (0 if path.startswith(ACTIVE) else 1 if path.startswith(MLP) or path in anchor_names
            else 2 if path in selected else 3 if row["formal_references"] else 4,
            0 if Path(path).name == "final.pt" else 1, path)

    duplicates = []
    for members in by_hash.values():
        if len(members) < 2:
            continue
        canonical = min(members, key=priority)
        duplicates.append({"sha256": canonical["sha256"], "canonical": canonical["path"],
            "members": [x["path"] for x in members]})
        for row in members:
            if row is not canonical:
                row["identical_to"] = canonical["path"]
    for row in rows:
        path, name = row["path"], Path(row["path"]).name
        config_reference = any(p.startswith("configs/") for p in row["formal_references"])
        if path.startswith(ACTIVE):
            category, reason = "keep_active", "All models in the active attention run; no pruning while training/selection is in progress."
        elif path.startswith(MLP):
            category, reason = "keep_mlp", "Nine models retained by explicit cleanup instruction: initial, seven validation nodes and committed97000."
        elif path in anchor_names or path in selected or config_reference:
            category, reason = "keep_anchor", "Current config/selected model or explicit R20/R43/37423 anchor; retain a local accessible copy."
        elif path in ema_sources:
            category, reason = "archive_ema_source", "Exact original sparse-EMA source from iterations12-19; preserve bytes, do not delete."
        elif "identical_to" in row and not row["formal_references"] and not row["metadata_references"]:
            category, reason = "deduplicate_candidate", "Byte-identical local canonical copy verified; no literal consumer found. Check dynamic paths before removing/replacing this alias."
        elif row["formal_references"] or row["metadata_references"]:
            category, reason = "archive_referenced", "Referenced historical checkpoint; externalize with manifest and usable original-path link, do not discard."
        elif path.startswith("runtime/"):
            category, reason = "archive_runtime_probe", "Unreferenced historical engineering/probe model; archive unless regeneration requirements are explicitly accepted."
        elif name.startswith("step-"):
            category, reason = "ordinary_step_candidate", "No literal consumer found; ordinary historical recovery node. May discard if old interruption/within-round reconstruction is no longer needed; dynamic uses are not ruled out."
        else:
            category, reason = "archive_other", "Historical model/boundary/export; retain by external archive rather than infer that it is unused."
        row.update(category=category, reason=reason)
    counts, totals = Counter(), Counter()
    per_run = defaultdict(lambda: {"count": 0, "allocated_bytes": 0, "categories": {}})
    for row in rows:
        category = row["category"]
        counts[category] += 1
        totals[category] += row["allocated_bytes"]
        key = "/".join(row["path"].split("/")[:2])
        group = per_run[key]
        group["count"] += 1
        group["allocated_bytes"] += row["allocated_bytes"]
        cat = group["categories"].setdefault(category, {"count": 0, "allocated_bytes": 0})
        cat["count"] += 1
        cat["allocated_bytes"] += row["allocated_bytes"]
    summary = {"status": "audit_only_no_additional_mutations", "recorded_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "scope": "Real .pt/.pth files under runs/runtime/play_checkpoints; no model loading or new experiments. Source snapshots, NPZ pools and already-externalized symlinks excluded from storage candidates.",
        "reference_scan": "Literal paths in artifacts/configs/docs/scripts/src/README plus run-level config/selection/progress/state; absence is not proof against dynamic consumers.",
        "active_run": ACTIVE, "real_checkpoint_count": len(rows),
        "allocated_bytes": sum(row["allocated_bytes"] for row in rows),
        "categories": {c: {"count": counts[c], "allocated_bytes": totals[c]} for c in counts},
        "exact_duplicate_groups": len(duplicates), "external_or_link_aliases": len(links),
        "broken_or_unmounted_links": sum(not p["available"] for p in links),
        "per_run": dict(sorted(per_run.items(), key=lambda kv: kv[1]["allocated_bytes"], reverse=True))}
    (ART / "checkpoints.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(rows, key=lambda r: r["path"])))
    tiers = {
        "keep_local": {"keep_active", "keep_mlp", "keep_anchor"},
        "deduplicate_candidates": {"deduplicate_candidate"},
        "ordinary_step_candidates": {"ordinary_step_candidate"},
        "externalize_candidates": {"archive_ema_source", "archive_referenced", "archive_other", "archive_runtime_probe"},
    }
    for name, categories in tiers.items():
        (ART / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
            for row in sorted(rows, key=lambda r: r["path"]) if row["category"] in categories))
    for name, content in (("inventory_summary.json", summary), ("duplicate_groups.json", duplicates), ("existing_links.json", links)):
        (ART / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_run"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
