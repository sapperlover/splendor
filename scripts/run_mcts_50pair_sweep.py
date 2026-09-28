"""Run four fixed paired MCTS/search comparisons in the requested order."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "mcts_50pair_sweep"
OUTPUT_DIR = ROOT / "artifacts" / "mcts_50pair_sweep"
MATCHUPS = (
    "01_s256_n2_vs_s1024_n1",
    "02_s1024_n1_vs_s1024_n2",
    "03_s1024_n2_vs_s2048_n1",
    "04_s1024_n1_vs_ab_d6_top5",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="continue matching saved evaluations")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the four commands")
    args = parser.parse_args(argv)

    jobs = []
    for name in MATCHUPS:
        config_path = CONFIG_DIR / f"{name}.json"
        output_path = OUTPUT_DIR / f"{name}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("pairs") != 50 or config.get("seed_start") != 9800000:
            raise ValueError(f"unexpected paired seed schedule in {config_path}")
        for side in ("policy_a", "policy_b"):
            checkpoint = ROOT / config[side]["checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        if output_path.exists() and not args.resume and not args.dry_run:
            raise FileExistsError(f"{output_path} exists; use --resume")
        command = [
            sys.executable,
            "-m",
            "splendor.evaluation",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--progress-every",
            "1",
            "--summary-every",
            "5",
        ]
        if args.resume:
            command.append("--resume")
        jobs.append((name, command, output_path))

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    for index, (name, command, output_path) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {name}: {output_path}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        summary = json.loads(output_path.read_text(encoding="utf-8"))
        if not (
            summary.get("passed") is True
            and summary.get("completed_pairs") == 50
            and summary.get("valid_games") == 100
        ):
            raise RuntimeError(f"evaluation did not yield 50 valid pairs: {output_path}")
        print(
            f"finished {name}: {summary['a_wins']}W/{summary['draws']}D/"
            f"{summary['a_losses']}L, score={summary['win_rate_score']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
