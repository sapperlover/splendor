"""Run the four fixed eight-game search-acceleration collector profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splendor.selfplay_fast import FastSelfPlayConfig, generate, validate_run  # noqa: E402

PROFILES = ("serial_cpu_checked", "serial_cpu_fast", "batched_cpu", "batched_cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "search_accel" / "benchmark_8games")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    reports = {}
    for index, name in enumerate(PROFILES, 1):
        raw = json.loads((ROOT / "configs" / "search_accel" / f"{name}_8games.json").read_text())
        config = FastSelfPlayConfig.from_dict(raw)
        path = out / name
        print(f"[{index}/4] {name}: {path}", flush=True)
        reports[name] = generate(config, path, resume=args.resume)
        validate_run(path, replay=True)
        print(f"finished {name}: {reports[name]['games_per_wall_second']:.3f} games/s", flush=True)
    summary = {"profiles": reports, "game_ids": list(range(8)), "search": "S=256/N=2"}
    (out / "comparison.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
