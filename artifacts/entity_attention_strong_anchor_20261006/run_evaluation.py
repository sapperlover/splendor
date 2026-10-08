"""Run the two authorized fixed-PPO panels; never collect or train."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from splendor.alphazero_recording import environment_identity, sha256, source_identity, write_json
from splendor.evaluation import _evaluation_config_from_file, run_fixed_evaluation

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("round9", "round50"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    specification = json.loads((ART / "specification.json").read_text())
    case = specification["cases"][args.case]
    for record in specification["inputs"]:
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"frozen evaluation input changed: {record['path']}")
    config_path = ART / case["config"]
    config, _ = _evaluation_config_from_file(config_path)
    session = str(time.time_ns())
    runtime = ROOT / "runtime/entity_attention_strong_anchor_20261006" / args.case
    snapshot = runtime / "source" / session
    source = source_identity(ROOT, snapshot)
    native_source = ROOT / "src/splendor/native_search.cpp"
    shutil.copy2(native_source, snapshot / "src/splendor/native_search.cpp")
    shutil.copy2(Path(__file__), snapshot / "run_evaluation.py")
    identity = {
        "protocol": "entity-attention-strong-anchor-panel-v1",
        "case": args.case,
        "session": session,
        "command": ["conda", "run", "--no-capture-output", "-n", "splendor", "env",
                    "PYTHONPATH=src", "TORCHINDUCTOR_COMPILE_THREADS=2", "python", "-u",
                    str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "native_source_sha256": sha256(native_source),
        "source": source,
        "source_snapshot": str(snapshot.relative_to(ROOT)),
        "environment": environment_identity(),
        "inputs": specification["inputs"],
        "panel_scope": "Repeated DEV seeds shared with the historical offline-versus-PPO panels; not FINAL.",
    }
    record_path = ART / "recording" / f"{args.case}-{session}.json"
    write_json(record_path, identity)

    def progress(completed: int, total: int) -> None:
        print(json.dumps({"event": "evaluation_progress", "case": args.case,
                          "completed_pairs": completed, "total_pairs": total}), flush=True)

    started = time.perf_counter()
    result = run_fixed_evaluation(
        config, ART / case["output"], store_dir=runtime / "games",
        failure_dir=runtime / "failures", resume=args.resume, require_scored_pairs=True,
        progress_every=10, summary_every=10, progress_callback=progress,
    )
    unchanged = all(sha256(ROOT / item["path"]) == item["sha256"]
                    for item in specification["inputs"])
    identity.update({"elapsed_seconds": time.perf_counter() - started,
                     "output_sha256": sha256(ART / case["output"]),
                     "inputs_unchanged": unchanged, "passed": result.summary["passed"]})
    write_json(record_path, identity)
    if not unchanged:
        raise ValueError("evaluation inputs changed during the run")
    print(json.dumps({"event": "evaluation_completed", "case": args.case,
                      "score": result.summary["win_rate_score"], "passed": result.summary["passed"],
                      "valid_games": result.summary["valid_games"],
                      "elapsed_seconds": identity["elapsed_seconds"]}), flush=True)
    return 0 if result.summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
