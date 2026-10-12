"""KL trend of every hybrid generation on the two frozen holdout sets.

Evaluates the 5000-step offline start (theta_5000) and the nine completed
self-iteration checkpoints of `runs/entity_attention_hybrid_static_graph_seed1`
on the run's two frozen diagnostic sets, using the same `evaluate_validation`
code path and the same metric as the training-time `validation_after.json`:

* offline_fixed: `offline/fixed_validation.json` (4096 positions, digest 60df109d,
  sampled from the validation split of the old theta5 10x1000 corpus);
* online_fixed: `fixed_online_validation.json` (2048 positions, digest c2ac6762,
  frozen from the first online round's validation games).

Usage (from the repository root):

    python -u artifacts/entity_attention_hybrid_kl_trend_20261009/run.py --device cpu
    python -u artifacts/entity_attention_hybrid_kl_trend_20261009/run.py --device cuda

`--extend-fork` additionally copies the stored per-round `validation_after.json`
values of the N1 fork (same frozen sets, same code path, rounds 1..50) into the
summary so the trend can be read up to step 14500.
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
from splendor.offline_train import evaluate_validation, make_index
from splendor.rng import DeterministicRng

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT / "runs/entity_attention_hybrid_static_graph_seed1"
FORK_N1 = ROOT / "runs/entity_attention_hybrid_s1024_n1_from_round9_seed1"
CHECKPOINTS = ([("step-05000", RUN / "checkpoints/theta_5000.pt")] +
               [(f"round-{i + 1:02d}-final", RUN / f"rounds/round-{i:03d}/checkpoints/final.pt")
                for i in range(9)])
METRICS = ("policy_kl", "policy_entropy", "target_entropy", "value_loss",
           "positions", "policy_positions", "policy_weight_sum")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def learner_for(path: Path, device: str):
    payload = load_alphazero_checkpoint(path, device=device)
    config = replace(LearnerConfig(**payload.learner_config), device=device)
    learner = AlphazeroLearner(payload.build_model(device=device), config,
                               sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
    return learner, payload


def build_sets(cache_bytes: int):
    dataset = read_json(RUN / "offline/dataset.json")
    offline_index = make_index(dataset, cache_bytes)
    offline_fixed = read_json(RUN / "offline/fixed_validation.json")
    online_dataset = read_json(RUN / "fixed_online_dataset.json")
    online_index = make_index(online_dataset, cache_bytes // 4)
    online_fixed = read_json(RUN / "fixed_online_validation.json")
    return offline_index, offline_fixed, online_index, online_fixed


def stored_values(run: Path, rounds: int):
    """Per-round stored validation values from validation_after.json."""
    series = []
    for i in range(rounds):
        path = run / f"rounds/round-{i:03d}/validation_after.json"
        if not path.exists():
            break
        payload = read_json(path)
        series.append({
            "name": f"round-{i + 1:02d}-final",
            "total_train_step": payload["total_train_step"],
            "offline_fixed": {k: payload["offline_fixed"][k] for k in METRICS},
            "online_fixed": {k: payload["online_fixed"][k] for k in METRICS},
            "source": str(path.relative_to(ROOT)),
        })
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--extend-fork", action="store_true")
    parser.add_argument("--output", default=str(ART / "trend.json"))
    args = parser.parse_args()

    offline_index, offline_fixed, online_index, online_fixed = build_sets(args.cache_bytes)
    rows = []
    for name, path in CHECKPOINTS:
        learner, payload = learner_for(path, args.device)
        offline = evaluate_validation(learner, offline_index, offline_fixed)
        online = evaluate_validation(learner, online_index, online_fixed)
        rows.append({
            "name": name, "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "total_train_step": payload.total_train_step,
            "offline_fixed": {k: offline[k] for k in METRICS},
            "online_fixed": {k: online[k] for k in METRICS},
        })
        emit("evaluated", name=name, step=payload.total_train_step,
             offline_kl=offline["policy_kl"], online_kl=online["policy_kl"])
        del learner
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Reproduce the stored round values as a verification of the evaluation path.
    stored = {row["name"]: row for row in stored_values(RUN, 9)}
    verification = {"max_abs_diff_offline": 0.0, "max_abs_diff_online": 0.0,
                    "compared": 0}
    for row in rows:
        if row["name"] not in stored:
            continue
        for key, field in (("offline_fixed", "max_abs_diff_offline"),
                           ("online_fixed", "max_abs_diff_online")):
            for metric in ("policy_kl", "value_loss", "target_entropy"):
                diff = abs(row[key][metric] - stored[row["name"]][key][metric])
                verification[field] = max(verification[field], diff)
        verification["compared"] += 1

    summary = {
        "protocol": "hybrid-kl-trend-v1",
        "client_date": "2026-10-09",
        "run": str(RUN.relative_to(ROOT)),
        "metric": "weighted KL(target || student) on frozen positions "
                  "(policy_kl from the shared evaluate_validation path)",
        "offline_fixed": {"path": "runs/entity_attention_hybrid_static_graph_seed1/offline/fixed_validation.json",
                          "digest": offline_fixed["digest"], "positions": offline_fixed["positions"],
                          "policy_positions": rows[0]["offline_fixed"]["policy_positions"]},
        "online_fixed": {"path": "runs/entity_attention_hybrid_static_graph_seed1/fixed_online_validation.json",
                         "digest": online_fixed["digest"], "positions": online_fixed["positions"],
                         "policy_positions": rows[0]["online_fixed"]["policy_positions"]},
        "checkpoints": rows,
        "verification_vs_stored_validation_after": verification,
    }
    if args.extend_fork:
        summary["fork_series"] = {
            "n1": {"run": str(FORK_N1.relative_to(ROOT)),
                   "offline_fixed_digest": read_json(FORK_N1 / "offline/fixed_validation.json")["digest"],
                   "online_fixed_digest": read_json(FORK_N1 / "fixed_online_validation.json")["digest"],
                   "rows": stored_values(FORK_N1, 50)},
        }
    write_json(Path(args.output), summary)
    print(json.dumps({"verification": verification}, ensure_ascii=False))


if __name__ == "__main__":
    main()
