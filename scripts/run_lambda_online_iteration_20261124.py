"""Freeze and run the λ161924 / FPU online iteration branch."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = "lambda161924_fpu_p60k_iteration_20261124"
CONFIG = Path("configs/entity_attention_v1") / f"{RUN}.json"
SOURCE = ROOT / "runtime" / RUN / "source_10000_steps"


def freeze_source():
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    with (SOURCE.parent / ".source.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if SOURCE.exists():
            return
        native = list(ROOT.glob("src/splendor/_native_search*.so"))
        if not native:
            raise FileNotFoundError("Build the FPU-capable native search extension before preparing this run")
        with TemporaryDirectory(prefix=".source-", dir=SOURCE.parent) as temporary:
            pending = Path(temporary)
            record = source_identity(ROOT, pending)
            paths = [ROOT / CONFIG, ROOT / "src/splendor/native_search.cpp", *native,
                     *(ROOT / "data").glob("splendor_*")]
            for path in paths:
                target = pending / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            record["snapshot_files"] = {
                str(path.relative_to(pending)): sha256(path)
                for path in sorted(pending.rglob("*")) if path.is_file()
            }
            write_json(pending / "source.json", record)
            pending.rename(SOURCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="freeze source and config only")
    mode.add_argument("--check", action="store_true", help="validate frozen inputs without training")
    parser.add_argument("--rounds", type=int, help="absolute completed-round target")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    if not args.check:
        freeze_source()
    root = SOURCE if SOURCE.exists() else ROOT
    if root == SOURCE:
        for name, expected in read_json(SOURCE / "source.json")["snapshot_files"].items():
            if sha256(SOURCE / name) != expected:
                raise ValueError(f"Frozen online source changed: {name}")
    if args.prepare:
        print(SOURCE)
        return
    os.chdir(root)
    env = dict(os.environ, PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1")
    command = [sys.executable, "-u", "-m", "splendor.random_budget_iteration",
               "--config", str(root / CONFIG), "--output", str(ROOT / "runs" / RUN)]
    if args.check:
        command.append("--check")
    if args.rounds is not None:
        command.extend(("--rounds", str(args.rounds)))
    if args.max_steps_this_session is not None:
        command.extend(("--max-steps-this-session", str(args.max_steps_this_session)))
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
