"""Detach the serial arena queue and record its PID and exact command."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime" / ART.name
RUNTIME.mkdir(parents=True, exist_ok=True)
command = [sys.executable, "-u", str(ART / "run.py"), "run"]
environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1"}
with (RUNTIME / "queue.log").open("ab", buffering=0) as log:
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
record = {"pid": process.pid, "command": command, "start_unix": time.time(),
    "log": str((RUNTIME / "queue.log").relative_to(ROOT)), "detached": True}
(RUNTIME / "launch.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False), flush=True)
