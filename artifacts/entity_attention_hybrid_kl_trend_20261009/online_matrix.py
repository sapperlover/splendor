"""KL matrix: every checkpoint x every generation's frozen online holdout.

The published trend used a single fixed online holdout (round-1 validation).
Here each online round 1..9 gets its own frozen 2048-position holdout, built with
the same procedure/seed as `fixed_online_validation.json`
(`freeze_validation`, validation_seed=14640000), and every checkpoint is scored
on every generation's holdout with the shared `evaluate_validation` path.

Usage (from the repository root):

    python -u artifacts/entity_attention_hybrid_kl_trend_20261009/online_matrix.py \
        --device cpu --checkpoints parent          # 5000->9500, 10 checkpoints
    python -u artifacts/entity_attention_hybrid_kl_trend_20261009/online_matrix.py \
        --device cpu --checkpoints parent+n1       # + N1 fork rounds 1..50
"""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.offline_train import OfflineConfig, evaluate_validation, freeze_validation, make_index
from splendor.rng import DeterministicRng

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT / "runs/entity_attention_hybrid_static_graph_seed1"
FORK_N1 = ROOT / "runs/entity_attention_hybrid_s1024_n1_from_round9_seed1"
GENERATIONS = tuple(range(9))
POSITIONS = 2048
SEED = 14640000


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def generation_set(round_index: int, cache_bytes: int):
    snapshot = read_json(RUN / f"rounds/round-{round_index:03d}/replay_snapshot.json")
    games = sorted((g for g in snapshot["games"] if g["round_index"] == round_index),
                   key=lambda g: g["game_id"])
    index = make_index({"games": games}, cache_bytes)
    config = OfflineConfig(source_run="", source_artifacts="", model_config={}, max_steps=1,
                           rounds=(round_index,), validation_positions=POSITIONS,
                           validation_seed=SEED)
    frozen = freeze_validation(index, config)
    return index, frozen


def checkpoints(kind: str):
    series = [("parent", "step-05000", RUN / "checkpoints/theta_5000.pt")]
    series += [("parent", f"round-{i + 1:02d}-final",
                RUN / f"rounds/round-{i:03d}/checkpoints/final.pt") for i in range(9)]
    if kind == "parent+n1":
        for i in range(50):
            path = FORK_N1 / f"rounds/round-{i:03d}/checkpoints/final.pt"
            if path.exists():
                series.append(("n1", f"n1-round-{i + 1:02d}-final", path))
    return series


def learner_for(path: Path, device: str):
    payload = load_alphazero_checkpoint(path, device=device)
    config = replace(LearnerConfig(**payload.learner_config), device=device)
    learner = AlphazeroLearner(payload.build_model(device=device), config,
                               sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
    return learner, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoints", choices=("parent", "parent+n1"), default="parent")
    parser.add_argument("--cache-mib", type=int, default=384)
    parser.add_argument("--output", default=str(ART / "online_matrix.json"))
    args = parser.parse_args()
    cache_bytes = args.cache_mib * 1024**2

    sets = {}
    for r in GENERATIONS:
        index, frozen = generation_set(r, cache_bytes)
        sets[r] = (index, frozen)
        emit("froze_holdout", generation=r + 1, positions=frozen["positions"],
             digest=frozen["digest"][:12],
             policy_games=len([g for g in index.games if g.split == "validation"]))
    existing = read_json(RUN / "fixed_online_validation.json")
    check = {"generation1_digest_matches_fixed_online": sets[0][1]["digest"] == existing["digest"],
             "existing_digest": existing["digest"], "rebuilt_digest": sets[0][1]["digest"]}
    emit("reproduction_check", **check)

    rows = []
    for series, name, path in checkpoints(args.checkpoints):
        learner, payload = learner_for(path, args.device)
        values = {}
        for r in GENERATIONS:
            metrics = evaluate_validation(learner, *sets[r])
            values[str(r + 1)] = {"policy_kl": metrics["policy_kl"],
                                  "policy_entropy": metrics["policy_entropy"],
                                  "target_entropy": metrics["target_entropy"],
                                  "policy_weight_sum": metrics["policy_weight_sum"]}
        rows.append({"series": series, "name": name, "path": str(path.relative_to(ROOT)),
                     "sha256": sha256(path), "total_train_step": payload.total_train_step,
                     "holdouts": values})
        emit("evaluated", series=series, name=name, step=payload.total_train_step,
             min_holdout=min(values, key=lambda k: values[k]["policy_kl"]),
             min_kl=min(v["policy_kl"] for v in values.values()))
        del learner
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "protocol": "hybrid-online-holdout-matrix-v1",
        "client_date": "2026-10-09",
        "metric": "KL(target || student) per frozen online holdout; weighted policy KL",
        "holdouts": {str(r + 1): {"round_index": r, "positions": POSITIONS, "seed": SEED,
                                  "digest": sets[r][1]["digest"],
                                  "validation_games": len([g for g in sets[r][0].games
                                                           if g.split == "validation"])}
                     for r in GENERATIONS},
        "checkpoints": rows,
        "reproduction_check": check,
    }
    write_json(Path(args.output), summary)
    print(json.dumps({"check": check, "rows": len(rows)}))


if __name__ == "__main__":
    main()
