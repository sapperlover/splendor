from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from splendor.evaluation import FixedEvaluationConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "mcts_50pair_sweep"
NAMES = (
    "01_s256_n2_vs_s1024_n1",
    "02_s1024_n1_vs_s1024_n2",
    "03_s1024_n2_vs_s2048_n1",
    "04_s1024_n1_vs_ab_d6_top5",
)


def test_sweep_configs_match_requested_comparisons() -> None:
    expected = (
        (("mcts_checkpoint", 256, 2), ("mcts_checkpoint", 1024, 1)),
        (("mcts_checkpoint", 1024, 1), ("mcts_checkpoint", 1024, 2)),
        (("mcts_checkpoint", 1024, 2), ("mcts_checkpoint", 2048, 1)),
        (("mcts_checkpoint", 1024, 1), ("search_checkpoint", 6, 5)),
    )
    checkpoints = set()
    for name, matchup in zip(NAMES, expected, strict=True):
        raw = json.loads((CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))
        config = FixedEvaluationConfig.from_dict(raw)
        assert config.game_seeds == tuple(range(9800000, 9800050))
        assert config.policy_seed_base == 9810000
        assert config.workers == 8 and config.device == "cpu"
        assert config.bootstrap_resamples == 100000
        for policy, (kind, first, second) in zip(
            (config.policy_a, config.policy_b), matchup, strict=True
        ):
            assert policy.kind == kind
            assert policy.search_config is not None
            if kind == "mcts_checkpoint":
                assert (policy.search_config["simulations"], policy.search_config["determinizations"]) == (first, second)
            else:
                assert (policy.search_config["depth"], policy.search_config["top_k"]) == (first, second)
                assert policy.search_config["fold_forced_phases"] is True
            assert policy.checkpoint is not None
            checkpoints.add(policy.checkpoint)
    assert len(checkpoints) == 1


def test_sweep_dry_run_lists_all_jobs_without_starting_games() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_mcts_50pair_sweep.py", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    headings = [line for line in result.stdout.splitlines() if line.startswith("[")]
    assert len(headings) == 4
    for index, name in enumerate(NAMES, start=1):
        assert headings[index - 1].startswith(f"[{index}/4] {name}:")
