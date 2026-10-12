"""Frozen-holdout KL / value MSE for the replayed offline-9500 checkpoint.

Evaluates the replay checkpoints 9000/9500/10000, the theta_5000 start and the
hybrid round-9 model on the frozen offline holdout plus the nine generation
online holdouts (rebuilding them through the shared trend-artifact code).

Usage (from the repository root):

    python -u artifacts/entity_attention_hybrid_offline9500_20261009/holds.py --device cpu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.offline_train import evaluate_validation

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
TREND = ROOT / "artifacts/entity_attention_hybrid_kl_trend_20261009"
RUN = ROOT / "runs/entity_attention_hybrid_static_graph_seed1"
REPLAY = ROOT / "runs/entity_attention_offline_9500_replay"

CHECKPOINTS = {
    "theta_5000": RUN / "checkpoints/theta_5000.pt",
    "offline_9000": REPLAY / "checkpoints/step-000009000.pt",
    "offline_9500": REPLAY / "checkpoints/step-000009500.pt",
    "offline_10000": REPLAY / "checkpoints/step-000010000.pt",
    "hybrid_r9_9500": RUN / "rounds/round-008/checkpoints/final.pt",
}
SETS = ("offline_fixed", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9")
METRICS = ("policy_kl", "policy_entropy", "target_entropy", "value_loss", "value_zero_mse",
           "positions", "policy_positions", "policy_weight_sum", "value_weight_sum")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-mib", type=int, default=384)
    parser.add_argument("--output", default=str(ART / "holdout_eval.json"))
    args = parser.parse_args()
    cache_bytes = args.cache_mib * 1024**2

    trend_run = load_module("offline9500_trend_run", TREND / "run.py")
    matrix = load_module("offline9500_matrix", TREND / "online_matrix.py")
    offline_index, offline_fixed, _, _ = trend_run.build_sets(cache_bytes)
    sets = {"offline_fixed": (offline_index, offline_fixed)}
    for r in matrix.GENERATIONS:
        sets[f"g{r + 1}"] = matrix.generation_set(r, cache_bytes)
    emit("sets_ready", sets=list(sets), offline_digest=offline_fixed["digest"][:12])

    rows = {}
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            emit("missing", name=name, path=str(path))
            continue
        learner, payload = trend_run.learner_for(path, args.device)
        entry = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                 "total_train_step": payload.total_train_step, "sets": {}}
        for set_name, (index, fixed) in sets.items():
            metrics = evaluate_validation(learner, index, fixed)
            entry["sets"][set_name] = {key: metrics[key] for key in METRICS}
        rows[name] = entry
        emit("evaluated", name=name, step=payload.total_train_step,
             offline_kl=entry["sets"]["offline_fixed"]["policy_kl"],
             g1_kl=entry["sets"]["g1"]["policy_kl"],
             offline_value=entry["sets"]["offline_fixed"]["value_loss"])
        del learner
        if args.device == "cuda":
            torch.cuda.empty_cache()

    summary = {"protocol": "hybrid-offline9500-holdout-eval-v1",
               "client_date": "2026-10-09",
               "metric": "weighted KL(target || student) and weighted value MSE on frozen sets",
               "freeze": {name: {"digest": sets[name][1]["digest"],
                                 "positions": sets[name][1]["positions"]} for name in sets},
               "checkpoints": rows}
    write_json(Path(args.output), summary)
    print(json.dumps({"rows": list(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
