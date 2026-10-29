"""Wait for the current complete chain, then run three complete chains serially."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / ART.name
CURRENT = "entity_attention_128x2_shortterm_value_r21_r37_20261023"
JOBS = (
    ("entity_attention_128x2_opponent_policy_r21_r37_20261024", "scripts/run_attention128_policy104k_20261024.sh"),
    ("entity_attention_128x2_continuation_r21_r37_20261024", "scripts/run_attention128_both104k_20261024.sh"),
    ("entity_attention_128x2_shortterm_value_r21_r31_vs_r37_20261024", "scripts/run_attention128_value44k_20261024.sh"),
)
child = None
stopping = False


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def require_complete(name):
    """An exited trainer, stopped queue, or failed arena is not a successful chain."""
    art, output = ROOT / "artifacts" / name, ROOT / "runs" / name
    sessions = read(art / "execution.json")["sessions"]
    progress, summary = read(output / "progress.json"), read(art / "summary.json")
    if (not sessions or sessions[-1]["status"] != "complete" or progress["status"] != "complete"
            or summary["status"] != "complete"):
        raise ValueError(f"Predecessor {name} did not successfully finish its entire train/select/arena chain")
    selection = read(output / "selection.json")
    if read(art / "selected_candidate.json") != selection or sha(Path(selection["checkpoint"])) != selection["checkpoint_sha256"]:
        raise ValueError(f"Predecessor {name} selection identity differs")
    panels = []
    for identifier, pairs in (("greedy_vs_R37", 1000), ("search_S1024N1_vs_R37", 500)):
        path = art / "panels" / f"{identifier}.json"
        result = read(path)
        config = read(art / "evaluation_configs" / f"{identifier}.json")
        if (not result["passed"] or result["completed_pairs"] != pairs or result["valid_games"] != 2*pairs
                or result["config"] != config
                or result["checkpoint_identity"]["policy_a"]["sha256"] != selection["checkpoint_sha256"]):
            raise ValueError(f"Predecessor {name} panel {identifier} is incomplete or differs")
        if any(result[k] for k in ("exceptions", "illegal_actions", "engineering_truncations", "invariant_failures")):
            raise ValueError(f"Predecessor {name} panel {identifier} failed engineering checks")
        panels.append({"panel": identifier, "pairs": pairs, "sha256": sha(path)})
    return {"experiment": name, "selected_step": selection["selected_step"],
            "checkpoint_sha256": selection["checkpoint_sha256"], "panels": panels}


def wait_for_predecessor(name, state):
    lock_path = ROOT / "runtime" / name / "chain.lock"
    if not lock_path.exists():
        raise ValueError(f"Missing predecessor chain lock: {name}")
    state.update(status="waiting", predecessor=name, active_experiment=None)
    write(RUNTIME / "state.json", state)
    # The existing supervisor holds this lock through BOTH arenas and its final
    # execution/summary commit. Blocking flock needs no progress/GPU polling.
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return require_complete(name)


def interrupt(signum, frame):
    global stopping
    stopping = True
    if child is None:
        raise KeyboardInterrupt
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)


def run_queue(state):
    global child
    predecessor = CURRENT
    for name, script in JOBS:
        completion = wait_for_predecessor(predecessor, state)
        state["predecessor_checks"].append(completion)
        if (ROOT / "artifacts" / name / "execution.json").exists():
            sessions = read(ROOT / "artifacts" / name / "execution.json")["sessions"]
            if sessions and sessions[-1]["status"] == "complete":
                state["completed"].append(require_complete(name))
                predecessor = name
                continue
        state.update(status="running", active_experiment=name, predecessor=predecessor)
        command = ["bash", script]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL)
        state.update(child_pid=child.pid, command=command)
        write(RUNTIME / "state.json", state)
        code = child.wait()
        child = None
        if stopping or code in (130, -signal.SIGINT):
            raise KeyboardInterrupt
        if code:
            raise subprocess.CalledProcessError(code, command)
        state["completed"].append(require_complete(name))
        write(RUNTIME / "state.json", state)
        predecessor = name
    state.update(status="complete", active_experiment=None, child_pid=None, ended_unix=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"), default="run", nargs="?")
    args = parser.parse_args()
    if args.action == "status":
        print(read(RUNTIME / "state.json"))
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupt)
        state = {"status": "starting", "pid": os.getpid(), "started_unix": time.time(),
                 "jobs": [name for name, _ in JOBS], "completed": [], "predecessor_checks": []}
        try:
            run_queue(state)
            return 0
        except KeyboardInterrupt:
            state.update(status="interrupted", ended_unix=time.time())
            return 130
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}", ended_unix=time.time())
            raise
        finally:
            write(RUNTIME / "state.json", state)
            write(ART / "execution.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
