"""Read-only verification of the resumed production process and shared mappings."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
NAME = "entity_attention_128x2_opponent_policy_r21_r37_20261024"


def read(path):
    return json.loads(path.read_text())


def main():
    run = ROOT / "runs" / NAME
    ready = read(run / "prefetch_progress.json")
    live = read(ROOT / "runtime" / NAME / "live.json")
    assert ready["status"] == "ready" and ready["workers"] == 4
    assert "source_prefetch_20261024" in " ".join(live["command"])
    rows = []
    with (run / "training_metrics.jsonl").open("rb") as handle:
        handle.seek(max(0, (run / "training_metrics.jsonl").stat().st_size-1500000))
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "train_step" and "prefetch" in row:
                rows.append(row)
    assert rows and rows[-1]["round_train_step"] > 102000
    rows = rows[-1000:]
    mappings, memory = [], []
    for pid in [live["child_pid"], *ready["worker_pids"]]:
        proc = Path(f"/proc/{pid}")
        maps = [line.split() for line in (proc / "maps").read_text().splitlines()
                if "/memfd:splendor-training-cache" in line]
        assert len(maps) == 1
        mappings.append({"pid": pid, "permissions": maps[0][1], "inode": maps[0][4]})
        stats = {}
        for line in (proc / "smaps_rollup").read_text().splitlines():
            fields = line.split()
            if fields[0] in ("Rss:", "Pss:", "Pss_Shmem:", "Private_Dirty:"):
                stats[fields[0][:-1]] = int(fields[1])
        memory.append({"pid": pid, **stats})
    assert all(row["permissions"].startswith("r--") for row in mappings[1:])
    assert len({row["inode"] for row in mappings}) == 1
    checkpoints = sorted(run.glob("checkpoints/step-*.pt"))
    checkpoints = [p for p in checkpoints if int(p.stem.split("-")[1]) > 102000]
    intervals = []
    for a, b in zip(checkpoints, checkpoints[1:]):
        lo, hi = (int(p.stem.split("-")[1]) for p in (a, b))
        if hi-lo == 1000:
            intervals.append({"steps": [lo, hi], "wall_seconds": b.stat().st_mtime-a.stat().st_mtime,
                              "updates_per_second": (hi-lo)/(b.stat().st_mtime-a.stat().st_mtime)})
    before = read(ART / "before.json")
    changed = [p for p, expected in before["protected"].items()
               if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != expected]
    assert not changed, changed
    out = {"time": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "status": "running_verified", "live": live,
        "queue": read(ROOT / "runtime/entity_attention_auxiliary_queue_20261024/state.json"),
        "shared_cache": ready, "shared_mappings": mappings, "memory_kib": memory,
        "process_tree_pss_GiB": sum(row["Pss"] for row in memory)/1024**2,
        "steps": [rows[0]["round_train_step"], rows[-1]["round_train_step"]],
        "recent_training": {"n": len(rows), "cache_misses": sum(r["cache_delta"]["misses"] for r in rows),
            "worker_pids_seen": sorted({r["prefetch"]["worker_pid"] for r in rows}),
            "mean_phase_seconds": {k: statistics.mean(r["seconds"][k] for r in rows)
                                   for k in ("draw_load", "collate", "update_host")}},
        "checkpoint_intervals": intervals, "protected_files_unchanged": len(before["protected"]),
        "meminfo": {line.split(":")[0]: line.split(":")[1].strip()
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:"))}}
    (ART / "running_verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("queue", "memory_kib", "live")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
