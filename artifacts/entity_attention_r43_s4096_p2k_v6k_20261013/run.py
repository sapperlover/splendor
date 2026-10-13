"""Time-bounded session of the existing continuous replay-fork driver."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from splendor.alphazero_recording import sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = ROOT / "configs/entity_attention_v1/r43_s4096_p2k_v6k_20261013.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, default=8 * 3600)
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    if args.max_seconds <= 0 or (args.max_steps_this_session is not None and args.max_steps_this_session <= 0):
        parser.error("session limits must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = HERE / f"session-{stamp}.json"
    command = ["timeout", "--signal=TERM", "--kill-after=120s", str(args.max_seconds),
               sys.executable, "-u", "-m", "splendor.pool_iteration", "run", "--config", str(PLAN)]
    if args.max_steps_this_session is not None:
        command += ["--max-steps-this-session", str(args.max_steps_this_session)]
    execution_environment = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                              "OPENBLAS_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS")}
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "max_seconds": args.max_seconds,
              "execution_environment": execution_environment,
              "command": command, "plan_sha256": sha256(PLAN), "status": "running"}
    write_json(path, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **execution_environment})
    record["timeout_pid"] = process.pid
    write_json(path, record)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        code = process.wait()
    record.update(finished_utc=datetime.now(timezone.utc).isoformat(), returncode=code,
                  status="time_limit" if code == 124 else "session_finished" if code == 0 else "interrupted_or_failed")
    write_json(path, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if code in (0, 124) else code


if __name__ == "__main__":
    raise SystemExit(main())
