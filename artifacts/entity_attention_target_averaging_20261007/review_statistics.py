"""Recompute the report revision from archived outputs; no search or training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
SEED = 16980000
RESAMPLES = 10000


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cluster_difference(rows, delta):
    keys = sorted({r["game_key"] for r in rows})
    mapping = {key: i for i, key in enumerate(keys)}
    index = np.asarray([mapping[r["game_key"]] for r in rows])
    counts = np.bincount(index, minlength=len(keys))
    sums = np.bincount(index, weights=delta, minlength=len(keys))
    draws = np.random.default_rng(SEED).integers(0, len(keys), size=(RESAMPLES, len(keys)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"difference": float(np.mean(delta)),
            "lower": float(np.quantile(values, .025)),
            "upper": float(np.quantile(values, .975)),
            "unit": "source_game_cluster", "games": len(keys), "roots": len(rows),
            "bootstrap_seed": SEED, "bootstrap_resamples": RESAMPLES}


def main():
    fit_path = ART / "fit.json"
    native_path = ROOT / "artifacts/entity_attention_chance_training_probe_20261007/train_native.json"
    fit = read(fit_path)["cases"]
    reference = fit["initial"]["per_root"]
    inputs = [identity(fit_path), identity(native_path), identity(Path(__file__))]
    for case in fit.values():
        assert [(r["id"], r["game_key"]) for r in case["per_root"]] == [
            (r["id"], r["game_key"]) for r in reference]
    kl = {}
    for step in (100, 300, 500):
        for other in (f"native{step}", "initial"):
            delta = np.asarray([a["kl_to_mean8"] - b["kl_to_mean8"] for a, b in zip(
                fit[f"avg{step}"]["per_root"], fit[other]["per_root"])])
            kl[f"avg{step}_minus_{other}"] = cluster_difference(reference, delta)

    historical = read(native_path)["curve"][0]["common"]["per_root"]
    assert [(r["root_id"], r["game_key"]) for r in historical] == [
        (r["id"], r["game_key"]) for r in reference]
    target = np.asarray([r["value_target"] for r in historical])
    errors = {name: (np.asarray([r["value"] for r in case["per_root"]]) - target) ** 2
              for name, case in fit.items()}
    value = {"archived_outcome_mse": {name: float(e.mean()) for name, e in errors.items()},
             "differences": {f"avg{s}_minus_native{s}": cluster_difference(
                 reference, errors[f"avg{s}"] - errors[f"native{s}"]) for s in (100, 300, 500)},
             "scope": "Archived game outcomes; not counterfactual returns under the updated student."}

    run = ROOT / "runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1"
    rounds = []
    for i in range(20):
        directory = run / "rounds" / f"round-{i:03d}"
        before_path, after_path = directory / "validation_before.json", directory / "validation_after.json"
        before, after = read(before_path), read(after_path)
        inputs.extend([identity(before_path), identity(after_path)])
        assert after["total_train_step"] - before["total_train_step"] == 100
        row = {"new_round": i + 1, "step_before": before["total_train_step"],
               "step_after": after["total_train_step"]}
        for bank in ("current", "recent"):
            assert before[bank]["positions"] == after[bank]["positions"]
            assert before[bank]["target_entropy_sum"] == after[bank]["target_entropy_sum"]
            row[bank] = {metric: {"before": before[bank][metric], "after": after[bank][metric],
                                 "delta": after[bank][metric] - before[bank][metric]}
                         for metric in ("policy_kl", "value_loss")}
        rounds.append(row)
    summary = {}
    for bank in ("current", "recent"):
        summary[bank] = {}
        for metric in ("policy_kl", "value_loss"):
            delta = np.asarray([r[bank][metric]["delta"] for r in rounds])
            summary[bank][metric] = {"rising_rounds": int((delta > 0).sum()), "rounds": len(rounds),
                                     "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta))}

    result = {"protocol": "target-averaging-report-review-v1", "review_date": "2026-10-07",
              "inputs": inputs, "kl_cluster_differences": kl, "value_diagnostics": value,
              "formal_split_training": {"run": str(run.relative_to(ROOT)), "rounds": rounds,
                                        "summary": summary,
                                        "scope": "Descriptive within-round before/after deltas; successive rounds are correlated, not independent repetitions."},
              "limitations": ["Original root-bootstrap analysis.json remains preserved.",
                              "Cluster intervals are conditional on this trained pair and eight reference searches; they do not cover training-seed variation.",
                              "No new collection, training, strength evaluation, or fresh-M1 comparison."]}
    (ART / "review_statistics.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"kl": kl, "formal_split_training": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
