"""Execute only the cache/checkpoint deletions explicitly authorized by the user."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
MLP = ROOT / "runs/entity_set_mlp_256x4_offline_80k_20261021"
KEEP = {"initial.pt", *(f"step-{n:09d}.pt" for n in
    (13528, 27056, 40584, 54112, 67640, 81168, 94695, 97000))}
CACHE_RUNS = (
    "entity_attention_r20_chance_s128_vs_greedy_20261021",
    "entity_attention_r20_search_stability_128_20261021",
    "entity_attention_r20_s1024n1_vs_greedy_20261021",
    "entity_attention_r20_s256n4_vs_greedy_20261021",
    "entity_set_mlp_256x4_20261021",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write(record):
    path = ART / "cleanup.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def model_record(path):
    stat = path.stat()
    return {"path": str(path.relative_to(ROOT)), "size": stat.st_size,
        "allocated_bytes": stat.st_blocks * 512, "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path)}


def main():
    if (ART / "cleanup.json").exists():
        raise ValueError("Cleanup record already exists; inspect it instead of replaying deletion")
    with ExitStack() as locks:
        for path in (MLP / ".lock", ROOT / "runtime" / MLP.name / "chain.lock"):
            handle = locks.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = list((MLP / "checkpoints").iterdir())
        if len(files) != 105 or any(p.is_symlink() or not p.is_file() for p in files):
            raise ValueError("The authorized set of 105 original MLP models changed")
        preserved = [p for p in files if p.name in KEEP]
        deleted = [p for p in files if p.name not in KEEP]
        if len(preserved) != 9 or len(deleted) != 96:
            raise ValueError("Expected nine retained and 96 removed models")
        if any(not (p.name.startswith("step-") and p.suffix == ".pt") for p in deleted):
            raise ValueError("Unexpected model deletion target")
        progress = json.loads((MLP / "progress.json").read_text())
        if progress["step"] != 97000 or Path(progress["checkpoint"]).name not in KEEP:
            raise ValueError("MLP committed resume checkpoint changed")
        retained = [model_record(p) for p in sorted(preserved)]
        stop = json.loads((ROOT / "artifacts" / MLP.name / "user_stop.json").read_text())
        expected = {str(ROOT / n["checkpoint"]): n["checkpoint_sha256"] for n in stop["nodes"]}
        expected[str(ROOT / stop["latest_checkpoint"]["path"])] = stop["latest_checkpoint"]["sha256"]
        for item in retained:
            path = str(ROOT / item["path"])
            if path in expected and item["sha256"] != expected[path]:
                raise ValueError(f"Retained model identity changed: {path}")
        caches = []
        for name in CACHE_RUNS:
            for kind in ("compile_cache", "triton_cache"):
                path = ROOT / "runtime" / name / kind
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(f"Authorized cache directory changed: {path}")
                result = subprocess.run(["du", "-x", "-s", "-B1", str(path)],
                    check=True, capture_output=True, text=True)
                caches.append({"path": str(path.relative_to(ROOT)),
                    "allocated_bytes": int(result.stdout.split()[0])})
        record = {"status": "prepared", "started_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
            "authorization": "Delete the five named compile_cache/triton_cache pairs and exactly 96 nonessential MLP recovery checkpoints; audit other models only.",
            "caches": caches, "deleted_checkpoints": [model_record(p) for p in sorted(deleted)],
            "retained_checkpoints": retained, "deleted_cache_count": 0, "deleted_checkpoint_count": 0}
        write(record)
        for item in caches:
            shutil.rmtree(ROOT / item["path"])
            record["deleted_cache_count"] += 1
            write(record)
        for path in deleted:
            path.unlink()
            record["deleted_checkpoint_count"] += 1
        for item in retained:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Retained checkpoint changed during cleanup")
        if {p.name for p in (MLP / "checkpoints").iterdir()} != KEEP:
            raise ValueError("Retained checkpoint set differs")
        record.update(status="complete", ended_at=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
            removed_allocated_bytes=sum(x["allocated_bytes"] for x in caches + record["deleted_checkpoints"]),
            retained_sha256_verified=True)
        write(record)
        print(json.dumps({k: record[k] for k in ("status", "deleted_cache_count", "deleted_checkpoint_count",
            "removed_allocated_bytes", "retained_sha256_verified")}), flush=True)


if __name__ == "__main__":
    main()
