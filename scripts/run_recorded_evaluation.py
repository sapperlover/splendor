"""Run one existing fixed panel with the AZ invocation/resource ledger."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from splendor.alphazero_recording import DiagnosticsConfig, InvocationRecorder, read_json, write_json
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    config = FixedEvaluationConfig.from_dict(read_json(args.config))
    output = args.output.resolve()
    ledger_root = output.parent / "recording" / output.stem
    previous = read_json(output) if output.exists() else {}
    baseline = previous.get("completed_pairs", 0)
    recorder = InvocationRecorder(ledger_root, DiagnosticsConfig(str(output.parent)),
                                  {"completed_pairs": baseline, "resume": args.resume}, config.to_dict())
    error = None
    summary = previous
    started = time.perf_counter()
    def progress(completed, total):
        elapsed = time.perf_counter() - started
        record = {"event": "anchor_progress", "panel": output.stem, "completed_pairs": completed, "total_pairs": total,
                  "new_pairs_this_session": max(completed - baseline, 0), "elapsed_seconds": elapsed}
        recorder.event(record)
        if completed % args.progress_every == 0 or completed == total:
            print(f"{output.stem}: {completed}/{total} pairs 本次新增 {record['new_pairs_this_session']}，{elapsed:.1f}s", flush=True)
    try:
        with recorder.stage("external_anchor_evaluation"):
            result = run_fixed_evaluation(config, output, resume=args.resume, progress_callback=progress)
            summary = result.summary
    except BaseException as caught:
        error = caught
        if output.exists():
            summary = read_json(output)
        raise
    finally:
        recorder.finish({"completed_pairs": summary.get("completed_pairs"), "passed": summary.get("passed"),
                         "new_completed_pairs": summary.get("completed_pairs", baseline) - baseline}, error)
        sessions = [read_json(p) for p in sorted((ledger_root / "recording/sessions").glob("*.json"))]
        write_json(ledger_root / "wall_summary.json", {"panel": output.stem, "invocations": len(sessions),
                   "recorded_wall_seconds": sum(s.get("wall_seconds", 0) for s in sessions),
                   "exact_wall_available": all("wall_seconds" in s for s in sessions),
                   "completed_pairs": summary.get("completed_pairs"), "summary_path": str(output),
                   "scope": "external anchor only; separate from training loop timing; resources in recording/resources.jsonl"})
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
