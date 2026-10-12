"""200-pair S1024/N1 panels vs theta_5000 (shared seed domain).

Panels: replayed offline-9500, hybrid rounds 3/6/9 (6500/8000/9500 steps).
All four use the same 200 game seeds (14900000+), so panels are matched and can
be compared pairwise; both sides search S1024/N1 with no root noise at temp 0.

Usage (from the repository root):

    python artifacts/entity_attention_hybrid_offline9500_20261009/battles.py make-configs
    python -u artifacts/entity_attention_hybrid_offline9500_20261009/battles.py run
    python artifacts/entity_attention_hybrid_offline9500_20261009/battles.py analyze
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
CONFIG_DIR = ART / "battles/configs"
OUT_DIR = ART / "battles"
RUN = ROOT / "runs/entity_attention_hybrid_static_graph_seed1"
REPLAY = ROOT / "runs/entity_attention_offline_9500_replay"

PAIRS = 200
SEED_START = 14900000
POLICY_SEED_BASE = 14910000
BOOTSTRAP_SEED = 14920000
SEARCH_SEED = 14930000

PANELS = {
    "offline9500_vs_theta5000": REPLAY / "checkpoints/step-000009500.pt",
    "hybrid_r3_vs_theta5000": RUN / "rounds/round-002/checkpoints/final.pt",
    "hybrid_r6_vs_theta5000": RUN / "rounds/round-005/checkpoints/final.pt",
    "hybrid_r9_vs_theta5000": RUN / "rounds/round-008/checkpoints/final.pt",
}
ANCHOR = RUN / "checkpoints/theta_5000.pt"


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def policy_spec(checkpoint: Path, name: str):
    return {
        "kind": "mcts_checkpoint",
        "name": name,
        "checkpoint": str(checkpoint),
        "search_backend": "native",
        "search_config": {
            "simulations": 1024, "determinizations": 1, "c_puct": 1.5,
            "dirichlet_alpha": 0.3, "fold_forced_phases": True,
            "max_simulation_transitions": 512, "root_noise_fraction": 0.0,
            "seed": SEARCH_SEED,
        },
        "inference": {
            "mode": "compiled",
            "candidate_buckets": [16, 32, 48, 64],
            "native_transport": "reused", "compile_failure_policy": "fail_fast",
            "actor_projection": False,
        },
    }


def config_for(name: str, checkpoint: Path):
    return {
        "device": "cuda", "workers": 1,
        "pairs": PAIRS, "seed_start": SEED_START, "policy_seed_base": POLICY_SEED_BASE,
        "bootstrap_resamples": 10000, "bootstrap_seed": BOOTSTRAP_SEED,
        "max_completed_turns": 400, "adjudicate_deterministic_policy_cycles": True,
        "execution": {
            "backend": "native_batched", "active_games": 100, "max_batch_size": 64,
            "candidate_buckets": [16, 32, 48, 64], "inference_mode": "compiled",
            "actor_projection": False, "max_batch_wait_ms": 0.0,
        },
        "policy_a": policy_spec(checkpoint, name),
        "policy_b": policy_spec(ANCHOR, "theta_5000"),
        "metadata": {"experiment": "entity_attention_hybrid_offline9500_20261009",
                     "panel": name, "candidate": str(checkpoint.relative_to(ROOT)),
                     "seeds": "game 14900000+, policy 14910000+, search 14930000+"},
    }


def make_configs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in PANELS.items():
        path = CONFIG_DIR / f"{name}.json"
        payload = config_for(name, checkpoint)
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError(f"{path} exists with different content")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        emit("config", panel=name, path=str(path.relative_to(ROOT)),
             candidate=str(checkpoint.relative_to(ROOT)))


def run(only=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in PANELS:
        if only and name != only:
            continue
        output = OUT_DIR / f"{name}.json"
        command = [sys.executable, "-u", "-m", "splendor.evaluation",
                   "--config", str(CONFIG_DIR / f"{name}.json"),
                   "--output", str(output), "--summary-every", "50"]
        if output.exists():
            command.append("--resume")
        emit("panel_start", panel=name, output=str(output.relative_to(ROOT)))
        subprocess.run(command, cwd=ROOT, check=True)


def pair_scores(path: Path):
    data = json.loads(path.read_text())
    if not data.get("passed") or data["completed_pairs"] != data["total_pairs"]:
        return None, data
    scores = {}
    for pair in data["pairs"]:
        total = 0.0
        for game in pair["games"]:
            total += {"a_win": 1.0, "draw": 0.5, "a_loss": 0.0}[game["outcome_for_a"]]
        scores[int(pair["game_seed"])] = total / len(pair["games"])
    return scores, data


def analyze():
    summary = {"protocol": "hybrid-offline9500-battles-v1", "client_date": "2026-10-09",
               "panels": {}, "pooled": {}, "matched_contrasts": {}}
    scores = {}
    for name in PANELS:
        path = OUT_DIR / f"{name}.json"
        if not path.exists():
            summary["panels"][name] = {"status": "missing"}
            continue
        values, data = pair_scores(path)
        if values is None:
            summary["panels"][name] = {"status": "incomplete",
                                       "completed_pairs": data.get("completed_pairs")}
            continue
        scores[name] = values
        summary["panels"][name] = {
            "status": "complete", "pairs": len(values),
            "score": float(np.mean(list(values.values()))),
            "ci95": data["paired_bootstrap"],
            "output": str(path.relative_to(ROOT)),
        }
    for name, values in scores.items():
        rng = np.random.default_rng(20261015)
        array = np.asarray(list(values.values()), dtype=np.float64)
        boot = array[rng.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
        summary["pooled"][name] = {"pairs": len(array), "score": float(array.mean()),
                                   "lower": float(np.percentile(boot, 2.5)),
                                   "upper": float(np.percentile(boot, 97.5))}
    names = list(scores)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = sorted(set(scores[left]) & set(scores[right]))
            diff = np.asarray([scores[right][s] - scores[left][s] for s in shared])
            rng = np.random.default_rng(20261015 + i * 100 + names.index(right))
            boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
            summary["matched_contrasts"][f"{right}_minus_{left}"] = {
                "pairs": len(shared), "estimate": float(diff.mean()),
                "lower": float(np.percentile(boot, 2.5)),
                "upper": float(np.percentile(boot, 97.5))}
    (ART / "battle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pooled": summary["pooled"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("make-configs", "run", "analyze"))
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.action == "make-configs":
        make_configs()
    elif args.action == "run":
        run(args.only)
    else:
        analyze()


if __name__ == "__main__":
    main()
