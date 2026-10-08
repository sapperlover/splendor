"""Reproduce the bounded round 3/9 checkpoint comparison; never update weights."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.alphazero_train import AlphazeroLearner, metric_ratios
from splendor.hybrid_train import HybridConfig
from splendor.offline_train import evaluate_validation, make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores
from splendor.rng import DeterministicRng

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT / "runs/entity_attention_hybrid_static_graph_seed1"
RUNTIME = ROOT / "runtime/entity_attention_100_vs_500_20261005"
ROUNDS = (3, 9)  # Human-numbered rounds, selected before seeing step-100 scores.
FIELDS = ("policy_ce_sum", "policy_weight_sum", "value_sq_sum", "value_weight_sum",
          "entropy_sum", "positions", "target_entropy_sum", "policy_positions",
          "value_positions", "value_target_sum", "value_target_sq_sum")


def input_record(path):
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "hybrid-update-budget-comparison-v1", "rounds": ROUNDS,
                "selection": "early evaluated round 3 and latest evaluated round 9; no strength-based selection",
                "run": str(RUN), "analysis_script": input_record(Path(__file__)),
                "source": source_identity(ROOT, RUNTIME / "source"), "inputs": [],
                "reuse": "Existing 500-step panels reused; only candidate checkpoint changes for 100-step panels.",
                "commands": []}
    for path in (RUN / "config.json", RUN / "hybrid_state.json",
                 RUN / "offline/fixed_validation.json", RUN / "offline/dataset.json",
                 RUN / "fixed_online_validation.json", RUN / "fixed_online_dataset.json"):
        manifest["inputs"].append(input_record(path))
    shutil.copyfile(RUN / "config.json", ART / "training_config.json")
    for human in ROUNDS:
        stage = RUN / "rounds" / f"round-{human - 1:03d}"
        raw = read_json(stage / "evaluation/config.json")
        assert read_json(stage / "evaluation/mcts_vs_theta5000.json")["passed"]
        for step in (100, 500):
            config = copy.deepcopy(raw)
            if step == 100:
                config["policy_a"]["checkpoint"] = str((stage / "checkpoints/step-000000100.pt").resolve())
            name = f"round{human:02d}_step{step:03d}"
            target = ART / "configs" / f"{name}.json"
            write_json(target, config)
            manifest["inputs"].append(input_record(Path(config["policy_a"]["checkpoint"])))
            if step == 500:
                path = stage / "evaluation/mcts_vs_theta5000.json"
                shutil.copyfile(path, ART / f"{name}.json")
                manifest["inputs"].append(input_record(path))
                manifest["inputs"].append(input_record(stage / "evaluation/config.json"))
                header = Path(str(path) + ".games") / "header.json"
                manifest["inputs"].append(input_record(header))
                shutil.copyfile(header, ART / (name + "_header.json"))
            else:
                manifest["commands"].append(
                    "conda run --no-capture-output -n splendor env PYTHONPATH=src python -u "
                    f"scripts/run_recorded_evaluation.py --config {target.relative_to(ROOT)} "
                    f"--output {(RUNTIME / (name + '.json')).relative_to(ROOT)} --progress-every 10")
        for path in (stage / "replay_snapshot.json", stage / "validation_before.json",
                     stage / "validation_after.json", stage / "training_metrics.jsonl", stage / "source_usage.json"):
            manifest["inputs"].append(input_record(path))
    manifest["inputs"].append(input_record(Path(raw["policy_b"]["checkpoint"])))
    write_json(ART / "manifest.json", manifest)
    print(json.dumps({"prepared": list(ROUNDS), "commands": manifest["commands"]}, ensure_ascii=False), flush=True)


def fixed_metrics(learner, index, fixed):
    # Stream references to bound memory; retain the original held-out references exactly.
    games = {g.key: g for g in index.split_games("validation")}
    sums = Counter()
    refs = fixed["references"]
    for start in range(0, len(refs), learner.config.validation_batch_size):
        metrics = learner.evaluate_samples([
            index.cache.sample(games[(r["round_index"], r["game_id"])], r["decision_index"])
            for r in refs[start:start + learner.config.validation_batch_size]])
        sums.update({field: metrics[field] for field in FIELDS})
    return {"available": bool(sums["positions"]), **sums, **metric_ratios(sums)}


def training_data():
    all_rounds = []
    for human in range(1, 10):
        stage = RUN / "rounds" / f"round-{human - 1:03d}"
        rows = [json.loads(line) for line in (stage / "training_metrics.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["event"] == "train_step"]
        assert len(rows) == 500 and [r["round_train_step"] for r in rows] == list(range(1, 501))
        blocks = []
        for start in range(0, 500, 100):
            group = rows[start:start + 100]
            sums = Counter()
            for row in group:
                sums.update({field: row.get(field, row["batch_size"] if field == "positions" else 0) for field in FIELDS})
            seconds = Counter()
            for row in group:
                seconds.update(row["seconds"])
            blocks.append({"first_step": start + 1, "last_step": start + 100,
                           **metric_ratios(sums), "mean_total_loss": float(np.mean([r["total_loss"] for r in group])),
                           "mean_grad_norm": float(np.mean([r["grad_norm"] for r in group])),
                           "clipped_fraction": float(np.mean([r["grad_clipped"] for r in group])),
                           "seconds": dict(seconds), "draws": sum(r["batch_size"] for r in group),
                           "offline_draws": sum(r["source_positions"]["offline"] for r in group),
                           "online_draws": sum(r["source_positions"]["online"] for r in group),
                           "cache_delta": dict(sum((Counter(r["cache_delta"]) for r in group), Counter()))})
        usage = read_json(stage / "source_usage.json")
        before = read_json(stage / "validation_before.json")
        after = read_json(stage / "validation_after.json")
        collection = read_json(stage / "selfplay/summary.json")
        references = {}
        # Exact unique-position coverage of sampled training references, separately per source.
        for source in ("offline", "online"):
            batches = {}
            for path in sorted((stage / "usage" / source).glob("*.bin")):
                data = np.fromfile(path, dtype="<i8")
                assert len(data) % 4 == 0
                data = data.reshape(-1, 4)
                for step in np.unique(data[:, 0]):
                    block = data[data[:, 0] == step]
                    if step > 500:
                        continue
                    if int(step) in batches:
                        assert np.array_equal(batches[int(step)], block)
                    batches[int(step)] = block
            assert set(batches) == set(range(1, 501))
            references[source] = {}
            for budget in (100, 500):
                data = np.concatenate([batches[s] for s in range(1, budget + 1)])[:, 1:]
                _, counts = np.unique(data, axis=0, return_counts=True)
                new = data[data[:, 0] == human - 1] if source == "online" else np.empty((0, 3), dtype=np.int64)
                references[source][str(budget)] = {
                    "draws": len(data), "unique_positions": len(counts),
                    "pool_positions": usage[source]["pool_positions"],
                    "pool_coverage": len(counts) / usage[source]["pool_positions"],
                    "repeat_draw_fraction": 1 - len(counts) / len(data),
                    "mean_exposure": len(data) / usage[source]["pool_positions"],
                    "max_position_uses": int(counts.max()),
                    "current_round_draws": len(new) if source == "online" else None,
                    "current_round_unique_positions": len(np.unique(new, axis=0)) if source == "online" else None}
        all_rounds.append({"round": human, "blocks": blocks, "references": references,
                           "source_usage": usage, "original_before": before,
                           "original_after": after, "collection": collection,
                           "input_metrics": input_record(stage / "training_metrics.jsonl")})
    write_json(ART / "training_analysis.json", all_rounds)
    print("training log/reference analysis complete", flush=True)


def validation():
    torch.set_num_threads(1)
    cfg = HybridConfig.from_dict(read_json(RUN / "config.json"))
    learner_cfg = replace(cfg.offline.learner, device="cpu", validation_batch_size=256)
    old = make_index(read_json(RUN / "offline/dataset.json"), 512 * 1024**2)
    old_fixed = read_json(RUN / "offline/fixed_validation.json")
    online = make_index(read_json(RUN / "fixed_online_dataset.json"), 128 * 1024**2)
    online_fixed = read_json(RUN / "fixed_online_validation.json")
    records = []
    started = time.perf_counter()
    for human in ROUNDS:
        stage = RUN / "rounds" / f"round-{human - 1:03d}"
        snapshot = read_json(stage / "replay_snapshot.json")
        current = make_index({"games": [g for g in snapshot["games"] if g["round_index"] == human - 1]}, 128 * 1024**2)
        for step in (0, 100, 200, 300, 400, 500):
            path = stage / "checkpoints" / ("initial.pt" if step == 0 else f"step-{step:09d}.pt")
            model = load_frozen_actor_critic_checkpoint(path, device="cpu")
            learner = AlphazeroLearner(model, learner_cfg, sampling_rng=DeterministicRng(0))
            row = {"round": human, "step": step, "checkpoint": input_record(path),
                   "current": evaluate_validation(learner, current),
                   "offline_fixed": fixed_metrics(learner, old, old_fixed),
                   "online_fixed": fixed_metrics(learner, online, online_fixed)}
            records.append(row)
            write_json(ART / "validation_curves.json", {"device": "cpu", "dtype": "float32",
                       "threads": 1, "validation_batch_size": 256, "optimizer_updates": 0,
                       "fixed_offline_digest": old_fixed["digest"], "fixed_online_digest": online_fixed["digest"],
                       "rows": records, "elapsed_seconds": time.perf_counter() - started})
            print(json.dumps({"round": human, "step": step,
                              "metrics": {name: {k: row[name][k] for k in ("policy_kl", "value_loss")}
                                          for name in ("current", "offline_fixed", "online_fixed")}}, ensure_ascii=False), flush=True)
    # Check stream/CPU results against original CUDA diagnostics at existing endpoints.
    for row in records:
        stage = RUN / "rounds" / f"round-{row['round'] - 1:03d}"
        original = read_json(stage / ("validation_before.json" if row["step"] == 0 else "validation_after.json")) if row["step"] in (0, 500) else None
        if original is not None:
            for name in ("current",) if row["step"] == 0 else ("current", "offline_fixed", "online_fixed"):
                for key in ("policy_kl", "value_loss"):
                    assert abs(row[name][key] - original[name][key]) < 1e-5, (row["round"], row["step"], name, key)


def finalize():
    results = []
    for human in ROUNDS:
        names = [f"round{human:02d}_step{step:03d}" for step in (100, 500)]
        shutil.copyfile(RUNTIME / (names[0] + ".json"), ART / (names[0] + ".json"))
        configs = [read_json(ART / "configs" / (name + ".json")) for name in names]
        check = copy.deepcopy(configs[0])
        check["policy_a"]["checkpoint"] = configs[1]["policy_a"]["checkpoint"]
        assert check == configs[1]
        summaries = [read_json(ART / (name + ".json")) for name in names]
        pair_data = [serialized_pair_scores(s) for s in summaries]
        assert all(s["valid_games"] == 100 and s["completed_pairs"] == 50 for s in summaries)
        for summary, config in zip(summaries, configs):
            for side in ("policy_a", "policy_b"):
                assert summary["checkpoint_identity"][side]["sha256"] == sha256(Path(config[side]["checkpoint"]))
            for key in ("engineering_truncations", "exceptions", "illegal_actions", "invariant_failures", "forced_skips"):
                assert summary[key] == 0, key
        effective = copy.deepcopy(summaries[0]["config"])
        effective["policy_a"]["checkpoint"] = summaries[1]["config"]["policy_a"]["checkpoint"]
        assert effective == summaries[1]["config"]
        delta = paired_bootstrap_difference(*pair_data[1], *pair_data[0], resamples=10000, seed=14720000)
        results.append({"round": human, "score_difference_500_minus_100": delta.to_dict(),
                        "summary_100": {k: v for k, v in summaries[0].items() if k != "pairs"},
                        "summary_500": {k: v for k, v in summaries[1].items() if k != "pairs"},
                        "pairs_500_better": int(np.sum(pair_data[1][1] > pair_data[0][1])),
                        "pairs_equal": int(np.sum(pair_data[1][1] == pair_data[0][1])),
                        "pairs_100_better": int(np.sum(pair_data[1][1] < pair_data[0][1])),
                        "pair_scores_100": pair_data[0][1].tolist(), "pair_scores_500": pair_data[1][1].tolist()})
    write_json(ART / "strength_comparison.json", results)
    manifest = read_json(ART / "manifest.json")
    for name in (f"round{r:02d}_step100" for r in ROUNDS):
        ledger = RUNTIME / "recording" / name / "recording"
        sessions = list((ledger / "sessions").glob("*.json"))
        for path in sessions:
            manifest.setdefault("evaluation_sessions", []).append(input_record(path))
        for folder in ("sessions", "identities"):
            for path in (ledger / folder).glob("*.json"):
                target = ART / "recording" / name / folder / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        shutil.copyfile(Path(str(RUNTIME / (name + ".json")) + ".games") / "header.json", ART / (name + "_header.json"))
    manifest["analysis_script"] = input_record(Path(__file__))
    manifest["outputs"] = [input_record(path) for path in sorted(ART.rglob("*.json")) if path.name != "manifest.json"]
    manifest["status"] = "complete"
    write_json(ART / "manifest.json", manifest)
    print(json.dumps([{ "round": r["round"], "delta": r["score_difference_500_minus_100"]} for r in results], ensure_ascii=False), flush=True)


def plot():
    # Rendering uses the installed Matplotlib in base; all model analysis uses splendor.
    import subprocess
    subprocess.run(["conda", "run", "--no-capture-output", "-n", "base", "env",
                    "MPLCONFIGDIR=/tmp/splendor_comparison_matplotlib", "python",
                    str(ART / "plot.py")], check=True, cwd=ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "training", "validation", "finalize", "plot"))
    args = parser.parse_args()
    {"prepare": prepare, "training": training_data, "validation": validation, "finalize": finalize,
     "plot": plot}[args.phase]()
