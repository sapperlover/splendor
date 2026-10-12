"""Run the experiment and summarize it, or attach a blocking completion hook."""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
LOCK = ROOT / "runtime" / ART.name / "run.lock"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach", action="store_true",
                        help="wait for an already-running runner to release its lock; no polling")
    args = parser.parse_args()
    if args.attach:
        print("Waiting for the running experiment to finish; blocking on its file lock.", flush=True)
        with LOCK.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return subprocess.run([sys.executable, str(ART / "analyze.py")], cwd=ROOT).returncode
    result = subprocess.run([sys.executable, "-u", str(ART / "run.py"), "run"], cwd=ROOT)
    analysis = subprocess.run([sys.executable, str(ART / "analyze.py")], cwd=ROOT)
    return result.returncode or analysis.returncode


if __name__ == "__main__":
    raise SystemExit(main())
