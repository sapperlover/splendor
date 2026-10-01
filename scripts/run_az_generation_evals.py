"""Run per-generation fixed comparisons of AZ-loop checkpoints vs the initial net.

For every selected generation (theta1 = the checkpoint after AZ loop round 1) this
script runs two fixed matchups against the initial 5M checkpoint:

* ``greedy`` : theta_g greedy vs initial greedy, 500 paired seeds (1000 games);
* ``mcts``   : theta_g + PUCT S=1024/N=1 vs initial + PUCT S=1024/N=1, 50 paired seeds.

All generations share one fixed seed block (default 9,900,000) that is disjoint
from training (9.5M/9.51M/9.52M), the loop DEV panels (9.3M/9.31M/9.4M/9.41M),
the smoke run (9.6M) and the 50-pair sweep (9.8M), so every generation - including
older artifacts produced with the same block - is directly comparable per seed.

Per-pair summary JSON goes to ``artifacts/az_loop_seed1/evaluations/`` by default
so published conclusions remain auditable. Resumable ``*.games/`` sidecars are
git-ignored; the compact digest lives in ``artifacts/az_loop_seed1/generation_scores.json``.
Generated evaluation configs are written with repository-relative paths under the
git-ignored ``runtime/az_loop_seed1_evals/configs/``. Use ``--digest-only`` to rebuild
the digest from already finished outputs without running anything.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP_OUTPUT = ROOT / "runs" / "az_loop_seed1"
INITIAL_CHECKPOINT = (
    ROOT / "runs" / "5c_main_seed1" / "checkpoints" / "local_005000000_lifetime_005512112.pt"
)
DEFAULT_SEED_START = 9_900_000
DEFAULT_POLICY_SEED_BASE = 9_910_000
DEFAULT_BOOTSTRAP_SEED = 9_920_000
DEFAULT_OUTPUT_DIR = Path("artifacts/az_loop_seed1/evaluations")
DEFAULT_CONFIG_DIR = Path("runtime/az_loop_seed1_evals/configs")
DEFAULT_DIGEST = Path("artifacts/az_loop_seed1/generation_scores.json")
PANEL_ORDER = ("greedy", "mcts")
MCTS_SEARCH_CONFIG = {
    "simulations": 1024,
    "determinizations": 1,
    "c_puct": 1.5,
    "seed": 9_500_000,
    "fold_forced_phases": True,
    "max_simulation_transitions": 512,
    "root_noise_fraction": 0.0,
    "dirichlet_alpha": 0.3,
}
OUTCOME_SCORE = {"a_win": 1.0, "draw": 0.5, "a_loss": 0.0}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path) -> str:
    """Store repo-relative paths so generated configs stay portable."""

    absolute = path if path.is_absolute() else (ROOT / path)
    try:
        return absolute.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(absolute)


def checkpoint_for(generation: int) -> Path:
    """theta_g lives in round-(g-1) of the loop run."""

    return LOOP_OUTPUT / "rounds" / f"round-{generation - 1:03d}" / "checkpoints" / "final.pt"


def panel_names(panel: str, generation: int, pairs: int, simulations: int) -> tuple[str, str]:
    if panel == "greedy":
        name = f"greedy_theta{generation}_vs_initial_{pairs}pair"
    else:
        name = f"mcts_s{simulations}_theta{generation}_vs_initial_{pairs}pair"
    return name + ".json", name + ".json"


def build_config(
    panel: str,
    checkpoint: Path,
    *,
    pairs: int,
    simulations: int,
    workers: int,
    device: str,
    seed_start: int,
    policy_seed_base: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    checkpoint_text = repo_relative(checkpoint)
    initial_text = repo_relative(INITIAL_CHECKPOINT)
    if panel == "greedy":
        policy_a: dict[str, object] = {
            "kind": "greedy_checkpoint",
            "name": "theta_greedy",
            "checkpoint": checkpoint_text,
        }
        policy_b: dict[str, object] = {
            "kind": "greedy_checkpoint",
            "name": "initial5m_greedy",
            "checkpoint": initial_text,
        }
    else:
        search = dict(MCTS_SEARCH_CONFIG) | {"simulations": simulations}
        policy_a = {
            "kind": "mcts_checkpoint",
            "name": "theta_mcts",
            "checkpoint": checkpoint_text,
            "search_config": dict(search),
        }
        policy_b = {
            "kind": "mcts_checkpoint",
            "name": "initial5m_mcts",
            "checkpoint": initial_text,
            "search_config": dict(search),
        }
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "pairs": pairs,
        "seed_start": seed_start,
        "policy_seed_base": policy_seed_base,
        "workers": workers,
        "device": device,
        "max_completed_turns": 400,
        "adjudicate_deterministic_policy_cycles": True,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "progress_every": 25 if panel == "greedy" else 1,
        "summary_every": 50 if panel == "greedy" else 5,
        "metadata": {
            "purpose": f"az_loop_seed1_generations_vs_initial_{panel}",
            "panel": panel,
            "checkpoint": checkpoint_text,
            "note": "fixed seed block shared by all generations",
        },
    }


def load_summary(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pair_scores(summary: dict[str, object]) -> list[float]:
    return [
        sum(OUTCOME_SCORE[game["outcome_for_a"]] for game in pair["games"]) / 2.0
        for pair in summary.get("pairs", [])
    ]


def bootstrap_ci(values: list[float], *, samples: int = 20_000, seed: int = 7) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def row_from_summary(generation: int, panel: str, summary: dict[str, object], output: Path) -> dict[str, object]:
    bootstrap = summary.get("paired_bootstrap") or {}
    return {
        "generation": generation,
        "panel": panel,
        "score": summary.get("win_rate_score"),
        "wins": summary.get("a_wins"),
        "draws": summary.get("draws"),
        "losses": summary.get("a_losses"),
        "completed_pairs": summary.get("completed_pairs"),
        "total_pairs": summary.get("total_pairs"),
        "valid_games": summary.get("valid_games"),
        "passed": summary.get("passed"),
        "engineering_truncations": summary.get("engineering_truncations"),
        "exceptions": summary.get("exceptions"),
        "a_first": (summary.get("a_first") or {}).get("win_rate_score"),
        "a_second": (summary.get("a_second") or {}).get("win_rate_score"),
        "bootstrap_ci": [bootstrap.get("lower"), bootstrap.get("upper")] if bootstrap else None,
        "output": repo_relative(output),
    }


def format_row(row: dict[str, object]) -> str:
    ci = row.get("bootstrap_ci") or [None, None]
    ci_text = (
        f"[{ci[0]:.3f},{ci[1]:.3f}]"
        if isinstance(ci[0], (int, float)) and isinstance(ci[1], (int, float))
        else "n/a"
    )
    score = row.get("score")
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
    return (
        f"theta{row['generation']:<2} {str(row['panel']):<7} score {score_text} CI {ci_text}  "
        f"W/D/L {row['wins']}/{row['draws']}/{row['losses']}  "
        f"pairs {row['completed_pairs']}/{row['total_pairs']}  passed {row['passed']}  "
        f"seats first {row['a_first']} second {row['a_second']}"
    )


def paired_stats(panel: str, generations: list[int], summaries: dict[int, dict[str, object]]) -> dict[str, object]:
    """Per-generation means plus paired deltas vs the first generation (same seeds)."""

    available = [g for g in generations if isinstance(summaries.get(g), dict)]
    if len(available) < 2:
        return {}
    matrix = {g: pair_scores(summaries[g]) for g in available}
    width = min(len(values) for values in matrix.values())
    per_generation = {g: matrix[g][:width] for g in available}
    first = available[0]
    result: dict[str, object] = {
        "panel": panel,
        "generations": available,
        "seeds": width,
        "per_seed_std_across_generations": None,
        "pooled_mean": None,
        "pooled_ci": None,
        "vs_first": {},
    }
    stds = []
    for index in range(width):
        column = [per_generation[g][index] for g in available]
        mean = sum(column) / len(column)
        stds.append((sum((value - mean) ** 2 for value in column) / len(column)) ** 0.5)
    result["per_seed_std_across_generations"] = sum(stds) / len(stds) if stds else None
    pooled = [sum(per_generation[g][index] for g in available) / len(available) for index in range(width)]
    result["pooled_mean"] = sum(pooled) / len(pooled) if pooled else None
    result["pooled_ci"] = bootstrap_ci(pooled)
    for g in available[1:]:
        deltas = [per_generation[g][index] - per_generation[first][index] for index in range(width)]
        result["vs_first"][str(g)] = {
            "mean_delta": sum(deltas) / len(deltas) if deltas else None,
            "ci": bootstrap_ci(deltas),
            "seeds_up": sum(1 for value in deltas if value > 0) / len(deltas) if deltas else None,
            "seeds_down": sum(1 for value in deltas if value < 0) / len(deltas) if deltas else None,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="1-based generations to evaluate (theta_g = round g-1); default 1 2 3 4")
    parser.add_argument("--panels", default="greedy,mcts", help="greedy, mcts or greedy,mcts (default)")
    parser.add_argument("--greedy-pairs", type=int, default=500)
    parser.add_argument("--mcts-pairs", type=int, default=50)
    parser.add_argument("--mcts-simulations", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--policy-seed-base", type=int, default=DEFAULT_POLICY_SEED_BASE)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="auditable summaries (default: artifacts/az_loop_seed1/evaluations/); game sidecars are git-ignored")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR,
                        help="generated evaluation configs (default: runtime/az_loop_seed1_evals/configs/, git-ignored)")
    parser.add_argument("--digest-out", type=Path, default=DEFAULT_DIGEST,
                        help="compact aggregate meant for git (default: artifacts/az_loop_seed1/generation_scores.json)")
    parser.add_argument("--digest-only", action="store_true",
                        help="do not run evaluations; rebuild the digest from finished outputs")
    parser.add_argument("--resume", action="store_true", help="continue interrupted evaluations")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without running anything")
    args = parser.parse_args(argv)

    panels = [item.strip() for item in args.panels.split(",") if item.strip()]
    unknown = [p for p in panels if p not in PANEL_ORDER]
    if unknown:
        raise SystemExit(f"unknown panels: {unknown}")
    generations = sorted(set(args.rounds))
    if not generations or generations[0] < 1:
        raise SystemExit("--rounds must be positive 1-based generation numbers")
    if not INITIAL_CHECKPOINT.is_file():
        raise SystemExit(f"initial checkpoint missing: {INITIAL_CHECKPOINT}")

    output_dir = resolve(args.output_dir)
    config_dir = resolve(args.config_dir)
    digest_out = resolve(args.digest_out)

    jobs = []
    for generation in generations:
        checkpoint = checkpoint_for(generation)
        if not checkpoint.is_file():
            if generation in (1, 2, 3, 4, 5):
                raise SystemExit(f"theta{generation} checkpoint missing: {checkpoint}")
            continue
        for panel in PANEL_ORDER:
            if panel not in panels:
                continue
            pairs = args.greedy_pairs if panel == "greedy" else args.mcts_pairs
            config_name, output_name = panel_names(panel, generation, pairs, args.mcts_simulations)
            jobs.append({
                "generation": generation,
                "panel": panel,
                "pairs": pairs,
                "checkpoint": checkpoint,
                "config_path": config_dir / config_name,
                "output_path": output_dir / output_name,
            })

    greedy_jobs = sum(1 for job in jobs if job["panel"] == "greedy")
    mcts_jobs = len(jobs) - greedy_jobs
    if not args.digest_only:
        print(
            f"plan: {len(jobs)} evaluations ({greedy_jobs} greedy x {args.greedy_pairs} pairs, "
            f"{mcts_jobs} mcts x {args.mcts_pairs} pairs); "
            f"rough wall time {greedy_jobs} x 1-3 min + {mcts_jobs} x 25-40 min "
            f"(workers={args.workers}, device={args.device})",
            flush=True,
        )
        print(f"summaries -> {output_dir}; game sidecars are git-ignored; digest -> {digest_out}", flush=True)

    if not args.dry_run:
        config_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")

    rows: list[dict[str, object]] = []
    summaries: dict[int, dict[int, dict[str, object]]] = {panel: {} for panel in PANEL_ORDER}
    failures: list[str] = []
    for index, job in enumerate(jobs, start=1):
        generation, panel = job["generation"], job["panel"]
        config = build_config(
            panel,
            job["checkpoint"],
            pairs=job["pairs"],
            simulations=args.mcts_simulations,
            workers=args.workers,
            device=args.device,
            seed_start=args.seed_start,
            policy_seed_base=args.policy_seed_base,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        if not args.dry_run:
            job["config_path"].write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        command = [
            sys.executable, "-m", "splendor.evaluation",
            "--config", str(job["config_path"]),
            "--output", str(job["output_path"]),
            "--progress-every", str(config["progress_every"]),
            "--summary-every", str(config["summary_every"]),
        ]
        if args.resume:
            command.append("--resume")

        existing = load_summary(job["output_path"])
        complete = bool(existing) and existing.get("completed_pairs") == existing.get("total_pairs")
        print(f"[{index}/{len(jobs)}] theta{generation} {panel} -> {job['output_path']}", flush=True)
        if args.dry_run:
            print("    " + " ".join(command), flush=True)
            continue
        if args.digest_only:
            if existing is None:
                failures.append(f"theta{generation} {panel}: no summary at {job['output_path']}")
            elif not complete:
                failures.append(f"theta{generation} {panel}: incomplete summary")
        elif complete:
            print("    already complete", flush=True)
        elif existing is not None and not args.resume:
            failures.append(f"theta{generation} {panel}: incomplete output exists, rerun with --resume")
            print("    incomplete output exists; rerun with --resume", flush=True)
            continue
        else:
            result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if result.returncode not in (0, 1):
                failures.append(f"theta{generation} {panel}: evaluator exited with {result.returncode}")
                print(f"    evaluator exited with {result.returncode}", flush=True)
                continue
            existing = load_summary(job["output_path"])
        summary = existing
        if summary is None:
            if not args.digest_only:
                failures.append(f"theta{generation} {panel}: no summary written")
            print("    no summary written", flush=True)
            continue
        summaries.setdefault(panel, {})[generation] = summary
        row = row_from_summary(generation, panel, summary, job["output_path"])
        rows.append(row)
        if row["passed"] is not True:
            failures.append(f"theta{generation} {panel}: passed={row['passed']}")
        print("    " + format_row(row), flush=True)

    if args.dry_run:
        return 0

    digest = {
        "protocol": "az-loop-generations-vs-initial-v1",
        "loop_output": repo_relative(LOOP_OUTPUT),
        "initial_checkpoint": repo_relative(INITIAL_CHECKPOINT),
        "seed_start": args.seed_start,
        "policy_seed_base": args.policy_seed_base,
        "heavy_outputs": repo_relative(output_dir),
        "rows": rows,
        "paired": [
            paired_stats(panel, generations, summaries.get(panel, {}))
            for panel in PANEL_ORDER
            if summaries.get(panel)
        ],
        "failures": failures,
    }
    digest_out.parent.mkdir(parents=True, exist_ok=True)
    digest_out.write_text(json.dumps(digest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== summary (A = theta_g, B = initial 5M) ===", flush=True)
    for row in rows:
        print(format_row(row), flush=True)
    print(f"\ndigest: {digest_out}", flush=True)
    print(f"per-pair summaries: {output_dir}; game sidecars are git-ignored", flush=True)
    if failures:
        print("\nfailures:", flush=True)
        for item in failures:
            print("  - " + item, flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
