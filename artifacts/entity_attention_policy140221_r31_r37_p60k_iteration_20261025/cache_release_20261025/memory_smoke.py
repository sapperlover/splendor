"""Bounded CPU check of actual online-cache allocation, GC and heap trimming."""
import json
from pathlib import Path
import platform
import time

from splendor.compact_game_cache import CompactGameCache
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig
from splendor.hybrid_train import _release_inference_memory
from splendor.offline_train import make_index
from splendor.random_budget_iteration import _trim_cpu_heap

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def memory():
    return {key: int(value.split()[0])*1024
            for line in Path("/proc/self/status").read_text().splitlines() if ":" in line
            for key, value in [line.split(":", 1)] if key in ("VmRSS", "RssAnon")}


def main():
    run = ROOT / "runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025"
    records = json.loads((run / "rounds/round-011/replay_snapshot.json").read_text())["games"]
    records = [row for row in records if row["split"] == "train"][:1024]
    cache = CompactGameCache(512*1024**2, continuation_auxiliary=ContinuationAuxiliaryConfig(
        shortterm_value_weight=0., opponent_policy_weight=.15))
    index = make_index({"games": records}, cache.max_bytes)
    index.cache = cache
    _release_inference_memory()
    _trim_cpu_heap()
    before = memory()
    started = time.perf_counter()
    for game in index.games:
        sample = cache.sample(game, 0)
    filled = memory()
    cache_bytes = cache.bytes
    cache.retain(())
    del sample, index
    _release_inference_memory()
    after_gc = memory()
    trimmed = _trim_cpu_heap()
    after = memory()
    result = {"scope": "Existing 1024 training games, CPU only; no updates, search or arena.",
              "games": len(records), "libc": platform.libc_ver(), "cache_bytes": cache_bytes,
              "remaining_entries": cache.entries, "before": before, "filled": filled,
              "after_gc": after_gc, "after_release": after, "heap_trimmed": trimmed,
              "seconds": time.perf_counter()-started}
    (OUT / "memory_smoke.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
