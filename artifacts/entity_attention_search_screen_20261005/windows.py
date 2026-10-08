"""Two bounded production collection windows per S/N case, sequential GPU use."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_collect_throughput import run
from splendor.alphazero_recording import DiagnosticsConfig, InvocationRecorder, read_json, write_json
from splendor.selfplay_fast import FastSelfPlayConfig

ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_search_screen_20261005"
ORDER = ((512, 1), (1024, 1), (512, 2), (1024, 2),
         (1024, 2), (512, 2), (1024, 1), (512, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=35)
    parser.add_argument("--warmup-seconds", type=float, default=10)
    args = parser.parse_args()
    for ordinal, (s, n) in enumerate(ORDER):
        rep = 1 if ordinal < 4 else 2
        case = f"s{s}_n{n}_rep{rep}"
        output = RUNTIME / f"window_{case}.json"
        cfg = FastSelfPlayConfig.from_dict(read_json(ART / "configs" / f"collect_s{s}_n{n}.json"))
        recorder = InvocationRecorder(RUNTIME / "recording" / f"window_{case}", DiagnosticsConfig(str(RUNTIME)),
                                      {"case": case, "window_seconds": args.seconds, "warmup_seconds": args.warmup_seconds}, asdict(cfg))
        error = None
        print(json.dumps({"event": "window_start", "case": case}, ensure_ascii=False), flush=True)
        try:
            with recorder.stage("production_collection_window"):
                result = run(cfg, seconds=args.seconds, scratch=RUNTIME / "window_scratch",
                             warmup_seconds=args.warmup_seconds, label="real")
            result["screening_case"] = case
            result["screening_order"] = ordinal
            result["timing_scope"] = "100 active games, opening/early-game production window; excludes compile/setup and unfinished-game tail"
            write_json(output, result)
            clock = result["clock"]
            roots = sum(w["root_decisions"] / w["measured_seconds"] for w in result["workers"])
            print(json.dumps({"event": "window_complete", "case": case,
                              "simulations_per_second": clock["rates"]["simulations_per_second"],
                              "root_decisions_per_second": roots,
                              "mean_batch_size": clock["rates"]["mean_batch_size"],
                              "process_seconds": result["process_seconds"]}, ensure_ascii=False), flush=True)
        except BaseException as caught:
            error = caught
            raise
        finally:
            recorder.finish({"output": str(output), "completed": error is None}, error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
