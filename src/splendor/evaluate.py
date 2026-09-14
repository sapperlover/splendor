from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import (
    baseline_policy_names,
    baseline_policy_spec,
    evaluate_policies,
    save_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired Splendor baseline evaluation")
    choices = baseline_policy_names()
    parser.add_argument("--player-a", choices=choices, required=True)
    parser.add_argument("--player-b", choices=choices, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--policy-seed-base", type=int, default=0)
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument("--failure-dir", type=Path, default=Path("evaluation_failures"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")

    result = evaluate_policies(
        baseline_policy_spec(args.player_a),
        baseline_policy_spec(args.player_b),
        range(args.seed_start, args.seed_start + args.seed_count),
        policy_seed_base=args.policy_seed_base,
        max_completed_turns=args.max_completed_turns,
        failure_dir=args.failure_dir,
    )
    if args.report is not None:
        save_evaluation(args.report, result)
    print(json.dumps(result.to_dict(include_matches=False), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
