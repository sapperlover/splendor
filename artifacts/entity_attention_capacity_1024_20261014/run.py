"""Capacity test: fit 1024 frozen positions with 128x2 vs 256x4 from random init.

Both architectures start from random weights, see the same 1024 positions every
optimizer step (two pre-collated 512-row micro-batches), share the same AdamW
(lr 5e-4, no weight decay, global clip 1) and step budget with early stopping.
Every step logs train policy KL and train value MSE; a fixed 1024-position
held-out probe is evaluated every 100 steps.

Usage (from the repository root):
    python -u artifacts/entity_attention_capacity_1024_20261014/run.py prepare
    python -u artifacts/entity_attention_capacity_1024_20261014/run.py train --device cuda
    python -u artifacts/entity_attention_capacity_1024_20261014/run.py analyze
"""
from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from splendor.actor_critic import build_actor_critic
from splendor.alphazero_train import (_torch_batch, LearnerConfig, alphazero_loss_sums)
from splendor.offline_train import make_index
from splendor.selfplay_data import collate_selfplay_samples

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
OUT = ROOT / "runs" / ART.name
DATASET = ROOT / "runs/entity_attention_256x4_aux_r24_r43_20261014/dataset.json"
POSITIONS = ART / "positions.json"
PLAN = ART / "plan.json"
CURVES = ART / "curves"

SAMPLES_PER_SET = 1024
SAMPLE_SEED = 37000000
INIT_SEED = 37000001
LR = 5e-4
WEIGHT_DECAY = 0.0
CLIP = 1.0
MAX_STEPS = 2500
EARLY_STOP_PATIENCE = 500
EARLY_STOP_MIN_DELTA = 1e-5
MICRO_BATCH = 512
EVAL_EVERY = 100
MODELS = {
    "small_128x2": {"architecture": "entity_attention_v1"},
    "big_256x4": {"architecture": "entity_attention_v1", "hidden": 256, "heads": 8,
                  "layers": 4, "ffn_hidden": 1024, "dropout": 0.0},
}
SUM_KEYS = ("policy_ce_sum", "policy_weight_sum", "value_sq_sum", "value_weight_sum", "entropy_sum")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def make_samples(index, split: str, count: int, seed: int):
    """Uniform without replacement; keep positions with policy weight > 0."""
    games = index.split_games(split)
    starts = np.cumsum([0] + [g.decision_count for g in games])
    total = int(starts[-1])
    rng = np.random.default_rng(seed)
    samples, identities, drawn = [], [], 0
    while len(samples) < count:
        batch = rng.choice(total, size=4096, replace=False)
        for position in batch:
            if len(samples) >= count:
                break
            ordinal = int(bisect.bisect_right(starts, int(position)) - 1)
            game, decision = games[ordinal], int(position) - int(starts[ordinal])
            sample = index.cache.sample(game, decision)
            drawn += 1
            if float(sample.arrays["policy_weight"][decision]) <= 0:
                continue
            samples.append(sample)
            identities.append({"round_index": int(game.round_index), "game_id": int(game.game_id),
                               "decision_index": decision, "file": game.path.name})
    return samples, identities, total, drawn


def decode(index, split: str, identities):
    games = {(int(g.round_index), int(g.game_id)): g for g in index.split_games(split)}
    return [index.cache.sample(games[(int(ref["round_index"]), int(ref["game_id"]))],
                               int(ref["decision_index"]))
            for ref in identities]


def label_totals(samples):
    weights = np.asarray([float(s.arrays["policy_weight"][s.index]) for s in samples], dtype=np.float64)
    value_weights = np.asarray([float(s.arrays["value_weight"][s.index]) for s in samples], dtype=np.float64)
    targets = []
    for sample in samples:
        a, b = sample.arrays["candidate_offsets"][sample.index], sample.arrays["candidate_offsets"][sample.index + 1]
        targets.append(np.asarray(sample.arrays["policy_target"][a:b], dtype=np.float64))
    entropy_sum = float(sum(-(t * np.log(np.maximum(t, 1e-300))).sum() * w
                            for t, w in zip(targets, weights)))
    return {"target_entropy_sum": entropy_sum,
            "policy_weight_sum": float(weights.sum()),
            "value_weight_sum": float(value_weights.sum())}


def ratios(sums, labels):
    return {
        "policy_kl": (sums["policy_ce_sum"] - labels["target_entropy_sum"]) / labels["policy_weight_sum"],
        "policy_ce": sums["policy_ce_sum"] / labels["policy_weight_sum"],
        "target_entropy": labels["target_entropy_sum"] / labels["policy_weight_sum"],
        "policy_entropy": sums["entropy_sum"] / labels["policy_weight_sum"],
        "value_mse": sums["value_sq_sum"] / labels["value_weight_sum"],
    }


def prepare():
    dataset = json.loads(DATASET.read_text())
    index = make_index(dataset, 4 * 1024**3)
    train, train_ids, train_total, train_drawn = make_samples(index, "train", SAMPLES_PER_SET, SAMPLE_SEED)
    valid, valid_ids, valid_total, valid_drawn = make_samples(index, "validation", SAMPLES_PER_SET, SAMPLE_SEED + 1)
    entropy, labels = [], []
    for sample in train + valid:
        a, b = sample.arrays["candidate_offsets"][sample.index], sample.arrays["candidate_offsets"][sample.index + 1]
        p = np.asarray(sample.arrays["policy_target"][a:b], dtype=np.float64)
        entropy.append(float(-(p * np.log(np.maximum(p, 1e-300))).sum()))
        labels.append(int(sample.arrays["value_target"][sample.index]))
    payload = {
        "protocol": "capacity-1024-v1", "client_date": "2026-10-14",
        "dataset": {"path": str(DATASET.relative_to(ROOT)), "digest": dataset["digest"]},
        "samples_per_set": SAMPLES_PER_SET, "sample_seed": SAMPLE_SEED,
        "train": {"identities": train_ids, "pool_positions": train_total, "drawn": train_drawn},
        "validation": {"identities": valid_ids, "pool_positions": valid_total, "drawn": valid_drawn},
        "stats": {"target_entropy_mean": float(np.mean(entropy)),
                  "value_label_mean": float(np.mean(labels)),
                  "value_zero_mse": float(np.mean(np.square(labels))),
                  "non_draw_fraction": float(np.mean([abs(v) > 0 for v in labels]))},
    }
    if POSITIONS.exists() and json.loads(POSITIONS.read_text()) != payload:
        raise ValueError("frozen positions changed")
    POSITIONS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    plan = {"protocol": "capacity-1024-v1", "client_date": "2026-10-14",
            "models": MODELS, "init_seed": INIT_SEED, "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY, "clip": CLIP, "max_steps": MAX_STEPS,
            "early_stop": {"patience": EARLY_STOP_PATIENCE, "min_delta": EARLY_STOP_MIN_DELTA},
            "micro_batch": MICRO_BATCH, "effective_batch": SAMPLES_PER_SET, "eval_every": EVAL_EVERY,
            "note": "random init; same 1024 positions every step; identical optimizer for both models"}
    if PLAN.exists() and json.loads(PLAN.read_text()) != plan:
        raise ValueError("frozen plan changed")
    PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    emit("prepared", train=train_total, validation=valid_total,
         target_entropy=payload["stats"]["target_entropy_mean"],
         value_zero_mse=payload["stats"]["value_zero_mse"])


def terms_of(model, batch, tensors):
    logits, values = model(batch.decisions)
    return alphazero_loss_sums(logits, values, candidate_mask=tensors["candidate_mask"],
                               policy_targets=tensors["policy_targets"],
                               value_targets=tensors["value_targets"],
                               policy_weights=tensors["policy_weights"],
                               value_weights=tensors["value_weights"])


def train_model(name, model_config, train_micro, valid_micro, labels, device):
    cfg = LearnerConfig(policy_loss_weight=1.0, value_loss_weight=1.0)
    train_tensors = [_torch_batch(b, device) for b in train_micro]
    valid_tensors = [_torch_batch(b, device) for b in valid_micro]
    torch.manual_seed(INIT_SEED)
    model = build_actor_critic(model_config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    metrics_path = folder / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    curve, best, best_step, stale = [], None, 0, 0
    started = time.time()
    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {key: 0.0 for key in SUM_KEYS}
        for batch, tensors in zip(train_micro, train_tensors, strict=True):
            terms = terms_of(model, batch, tensors)
            (terms.total(cfg) / len(train_micro)).backward()
            for key in SUM_KEYS:
                sums[key] += float(getattr(terms, key))
        grad_norm = float(clip_grad_norm_(model.parameters(), CLIP))
        optimizer.step()
        row = {"step": step, **ratios(sums, labels), "grad_norm": grad_norm}
        if step % EVAL_EVERY == 0 or step == 1:
            model.eval()
            vsums = {key: 0.0 for key in SUM_KEYS}
            with torch.no_grad():
                for batch, tensors in zip(valid_micro, valid_tensors, strict=True):
                    terms = terms_of(model, batch, tensors)
                    for key in SUM_KEYS:
                        vsums[key] += float(getattr(terms, key))
            valid = ratios(vsums, VALID_LABELS)
            row.update({"valid_" + key: value for key, value in valid.items()})
        curve.append(row)
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        objective = row["policy_kl"] + row["value_mse"]
        if best is None or objective < best - EARLY_STOP_MIN_DELTA:
            best, best_step, stale = objective, step, 0
            torch.save({"model": model.state_dict(), "step": step, "objective": objective,
                        "model_config": model_config}, folder / "best.pt")
        else:
            stale += 1
        if step % 50 == 0:
            emit("train_progress", model=name, step=step, policy_kl=round(row["policy_kl"], 5),
                 value_mse=round(row["value_mse"], 5), best=round(best, 5),
                 elapsed=round(time.time() - started, 1))
        if stale >= EARLY_STOP_PATIENCE:
            emit("early_stop", model=name, step=step, best_step=best_step)
            break
    torch.save({"model": model.state_dict(), "step": len(curve), "model_config": model_config},
               folder / "final.pt")
    CURVES.mkdir(exist_ok=True)
    (CURVES / f"{name}.json").write_text(json.dumps(
        {"model": name, "model_config": model_config, "parameters": parameters,
         "steps": len(curve), "best_objective": best, "best_step": best_step,
         "seconds": time.time() - started, "curve": curve}, indent=2) + "\n")
    return {"model": name, "parameters": parameters, "steps": len(curve),
            "best_objective": best, "best_step": best_step}


VALID_LABELS: dict = {}


def train(device_name):
    global VALID_LABELS
    positions = json.loads(POSITIONS.read_text())
    plan = json.loads(PLAN.read_text())
    dataset = json.loads(DATASET.read_text())
    index = make_index(dataset, 4 * 1024**3)
    device = torch.device(device_name)
    train_samples = decode(index, "train", positions["train"]["identities"])
    valid_samples = decode(index, "validation", positions["validation"]["identities"])
    labels = label_totals(train_samples)
    VALID_LABELS = label_totals(valid_samples)
    train_micro = [collate_selfplay_samples(train_samples[i:i + MICRO_BATCH])
                   for i in range(0, SAMPLES_PER_SET, MICRO_BATCH)]
    valid_micro = [collate_selfplay_samples(valid_samples[i:i + MICRO_BATCH])
                   for i in range(0, SAMPLES_PER_SET, MICRO_BATCH)]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, model_config in plan["models"].items():
        results.append(train_model(name, model_config, train_micro, valid_micro, labels, device))
    (ART / "train_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    emit("training_done", results=results)


def analyze():
    curves = {p.stem: json.loads(p.read_text()) for p in sorted(CURVES.glob("*.json"))}
    if not curves:
        raise SystemExit("no curves")
    stats = json.loads(POSITIONS.read_text())["stats"]
    lines = ["# 1024 固定局面过拟合容量对照（128×2 vs 256×4）", "",
             f"两边都从**随机初始化**开始，反复训练同一批 {SAMPLES_PER_SET} 个 Train 局面（全批 1024＝512×2）；"
             f"AdamW lr={LR}、wd=0、clip={CLIP}、最多 {MAX_STEPS} 步、早停 patience={EARLY_STOP_PATIENCE}；"
             f"另有 {SAMPLES_PER_SET} 个留出局面每 {EVAL_EVERY} 步探测。", "",
             f"固定集合：目标熵均值 {stats['target_entropy_mean']:.4f}；常数 0 预测的 value MSE = "
             f"{stats['value_zero_mse']:.4f}（非和棋比例 {stats['non_draw_fraction']:.3f}）。", "",
             "| 模型 | 参数 | 步数 | Train KL 最优/最终 | Train MSE 最优/最终 | KL<0.05 | MSE<0.05 | 留出 KL/MSE（末次） |",
             "| --- | ---: | ---: | --- | --- | ---: | ---: | --- |"]
    summary = {}
    for name, data in curves.items():
        rows = data["curve"]

        def first_below(key, threshold):
            for row in rows:
                if row.get(key) is not None and row[key] < threshold:
                    return row["step"]
            return None

        last = rows[-1]
        summary[name] = {
            "parameters": data["parameters"], "steps": data["steps"],
            "min_train_kl": min(r["policy_kl"] for r in rows),
            "final_train_kl": last["policy_kl"],
            "min_train_mse": min(r["value_mse"] for r in rows),
            "final_train_mse": last["value_mse"],
            "step_kl_lt_0p05": first_below("policy_kl", 0.05),
            "step_kl_lt_0p01": first_below("policy_kl", 0.01),
            "step_mse_lt_0p05": first_below("value_mse", 0.05),
            "step_mse_lt_0p01": first_below("value_mse", 0.01),
            "final_valid_kl": last.get("valid_policy_kl"),
            "final_valid_mse": last.get("valid_value_mse"),
            "seconds": data["seconds"]}
        entry = summary[name]
        lines.append(f"| {name} | {data['parameters']:,} | {data['steps']} | "
                     f"{entry['min_train_kl']:.5f} / {entry['final_train_kl']:.5f} | "
                     f"{entry['min_train_mse']:.5f} / {entry['final_train_mse']:.5f} | "
                     f"{entry['step_kl_lt_0p05']} | {entry['step_mse_lt_0p05']} | "
                     f"{entry['final_valid_kl']:.4f} / {entry['final_valid_mse']:.4f} |")
    lines += ["", "逐步明细 `curves/*.json`，训练日志 `runs/entity_attention_capacity_1024_20261014/<模型>/metrics.jsonl`。",
              "本实验只回答拟合容量，不代表棋力。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (ART / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "analyze"), nargs="?", default="analyze")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        if not POSITIONS.exists():
            prepare()
        train(args.device)
    else:
        analyze()


if __name__ == "__main__":
    main()
