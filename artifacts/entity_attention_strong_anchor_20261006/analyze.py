"""Summarize the completed PPO panels and the existing 50-round trajectory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from splendor.alphazero_recording import sha256, write_json
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
INPUTS = {}


def record(path: Path) -> Path:
    INPUTS[str(path.relative_to(ROOT))] = {"path": str(path.relative_to(ROOT)),
                                         "sha256": sha256(path), "bytes": path.stat().st_size}
    return path


def read(path: Path):
    return json.loads(record(path).read_text())


def main() -> None:
    paths = {
        "offline5000": ROOT / "runs/entity_attention_offline/evaluation/offline_step5000_mcts_s1024n1_vs_ppo_5_5m_100pairs.json",
        "offline_final": ROOT / "runs/entity_attention_offline/evaluation/mcts_s1024n1_vs_ppo_5_5m_100pairs.json",
        "round9": ART / "evaluations/round9_vs_ppo.json",
        "round50": ART / "evaluations/round50_vs_ppo.json",
    }
    panels = {name: read(path) for name, path in paths.items()}
    scores = {name: serialized_pair_scores(panel) for name, panel in panels.items()}
    reference = panels["round9"]["checkpoint_identity"]["policy_b"]
    for panel in panels.values():
        if panel["checkpoint_identity"]["policy_b"] != reference:
            raise ValueError("the PPO anchor identity differs between panels")
    summaries = {
        name: {key: panel[key] for key in (
            "win_rate_score", "a_wins", "a_losses", "draws", "valid_games", "passed",
            "paired_bootstrap", "checkpoint_identity", "elapsed_seconds_this_invocation",
        )} for name, panel in panels.items()
    }
    differences = {}
    for left, right in (("round50", "round9"), ("round9", "offline5000"),
                        ("round50", "offline5000"), ("round9", "offline_final"),
                        ("round50", "offline_final")):
        seed_a, score_a = scores[left]
        seed_b, score_b = scores[right]
        differences[f"{left}_minus_{right}"] = paired_bootstrap_difference(
            seed_a, score_a, seed_b, score_b, resamples=10000, seed=13530000,
        ).to_dict()
    run = ROOT / "runs/entity_attention_hybrid_s1024_n1_from_round9_seed1"
    trajectory = []
    for human in range(1, 51):
        stage = run / "rounds" / f"round-{human - 1:03d}"
        before = read(stage / "validation_before.json")["current"]
        after = read(stage / "validation_after.json")["current"]
        snapshot = read(stage / "replay_snapshot.json")
        quota = snapshot["quotas"][1]
        batches = {}
        for path in sorted((stage / "usage/online").glob("*.bin")):
            references = np.fromfile(record(path), dtype="<i8").reshape(-1, 4)
            for step in np.unique(references[:, 0]):
                block = references[references[:, 0] == step]
                if len(block) != quota:
                    raise ValueError("source ledger does not match its batch quota")
                if int(step) in batches and not np.array_equal(batches[int(step)], block):
                    raise ValueError("duplicate ledger attempts differ")
                batches[int(step)] = block
        if set(batches) != set(range(1, 101)):
            raise ValueError("expected 100 committed update steps")
        references = np.concatenate([batches[step] for step in range(1, 101)])
        fresh_draws = int(np.count_nonzero(references[:, 1] == snapshot["round_index"]))
        latest_games = [game for game in snapshot["games"]
                        if game["round_index"] == snapshot["round_index"]]
        latest_train_positions = sum(game["decision_count"] for game in latest_games
                                     if game["split"] == "train")
        collection = read(stage / "selfplay/summary.json")
        row = {
            "round": human, "quotas": snapshot["quotas"],
            "fresh_draws": fresh_draws, "fresh_batch_fraction": fresh_draws / 25600,
            "latest_train_positions": latest_train_positions,
            "latest_immediate_exposure": fresh_draws / latest_train_positions,
            "collection_wall_seconds": collection["wall_seconds"],
            "latest_holdout_positions": after["positions"],
            "policy_kl_before": before["policy_kl"], "policy_kl_after": after["policy_kl"],
            "policy_kl_delta": after["policy_kl"] - before["policy_kl"],
            "value_mse_before": before["value_loss"], "value_mse_after": after["value_loss"],
            "value_mse_delta": after["value_loss"] - before["value_loss"],
        }
        evaluation = stage / "evaluation/mcts_vs_theta5000.json"
        if evaluation.exists():
            panel = read(evaluation)
            if not panel["passed"]:
                raise ValueError("trajectory includes invalid engineering results")
            row["theta5000_score"] = panel["win_rate_score"]
        trajectory.append(row)
    result = {
        "protocol": "entity-attention-strong-anchor-analysis-v1",
        "scope": "Two newly evaluated panels; historical offline references and existing training trajectory.",
        "panels": summaries, "paired_differences": differences, "trajectory": trajectory,
        "inputs": list(INPUTS.values()), "analysis_script_sha256": sha256(Path(__file__)),
        "limitations": ["Single training seed; repeatedly inspected DEV panel, not FINAL.",
                        "A difference against a shared opponent is not a direct match between the candidates.",
                        "Historical panels were not rerun; their original execution versions remain historical."],
    }
    write_json(ART / "analysis.json", result)
    print(json.dumps({"panels": summaries, "paired_differences": differences}, ensure_ascii=False))


if __name__ == "__main__":
    main()
