"""Scale-up 2: current-M4 fixed-data learning-rate probe (A sampling, 2.5e-5 vs 1e-4).

Reuses the frozen inputs of `entity_attention_policy_history_20261009`: the same
round-4/8/12 pre-training checkpoints, the same four-generation stored M4 labels,
the same per-step policy/value batch ledger, the same inherited AdamW/RNG state,
the same clip1 split training.  The only intervention is the AdamW learning rate
2.5e-5 for every parameter group (the published arms A/B/C all used 1e-4), so the
new node series is directly comparable with the published A series.

Usage (from the repository root):

    python -u artifacts/entity_attention_scaleup_20261009/lr_probe.py freeze --device cuda
    python -u artifacts/entity_attention_scaleup_20261009/lr_probe.py run --device cuda
    python -u artifacts/entity_attention_scaleup_20261009/lr_probe.py run --key r004 --device cuda
    python -u artifacts/entity_attention_scaleup_20261009/lr_probe.py smoke
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

import common

PH = common.hist_module()
ARM = "A_lr2.5e-5"
SPEC_PATH = common.ART / "lr_specification.json"


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def source_spec():
    return common.read_json(PH.ART / "specification.json")


def freeze(device: str):
    spec = source_spec()
    payload = {
        "protocol": "fixed-m4-lr-probe-v1",
        "client_date": "2026-10-09",
        "arm": ARM,
        "learning_rate": common.LR_VALUE,
        "steps": common.LR_STEPS,
        "nodes": list(common.LR_NODES),
        "keys": list(common.KEYS),
        "intervention": "AdamW param-group lr override for every group after restoring the "
                        "frozen checkpoint and optimizer state; weight decay, clip1 split "
                        "training, batch ledgers, RNG and value rule unchanged",
        "source_experiment": str(PH.ART.relative_to(common.ROOT)),
        "source_specification_sha256": common.sha256(PH.ART / "specification.json"),
        "source_code_sha256": {
            name: common.sha256(PH.ART / name if name.endswith(".py") and "/" not in name
                                and not (PH.ROOT / "src/splendor" / name).exists()
                                else PH.ROOT / "src/splendor" / name)
            for name in spec["code"]},
        "subjects": {key: {
            "start": spec["subjects"][key]["start"],
            "formal_final": spec["subjects"][key]["formal_final"],
            "batch_plans": spec["subjects"][key]["batch_plans"],
            "dataset": spec["subjects"][key]["dataset"],
            "snapshot": spec["subjects"][key]["snapshot"],
            "recent_rounds": spec["subjects"][key]["recent_rounds"],
        } for key in common.KEYS},
        "source": common.git_identity([Path(__file__), common.ART / "common.py"]),
        "execution": {"device": device},
    }
    if SPEC_PATH.exists() and common.read_json(SPEC_PATH) != payload:
        saved = common.read_json(SPEC_PATH)
        ignore = {"source", "execution"}
        if {k: v for k, v in saved.items() if k not in ignore} != {
                k: v for k, v in payload.items() if k not in ignore}:
            raise ValueError("lr-probe specification changed; use a new experiment")
    if not SPEC_PATH.exists():
        common.write_json(SPEC_PATH, payload)
    emit("frozen", arm=ARM, lr=common.LR_VALUE)
    return payload


def subject_of(key: str):
    spec = source_spec()
    subject = spec["subjects"][key]
    for field in ("start", "snapshot", "dataset", "batch_plans"):
        path = Path(subject[field]["path"])
        if common.sha256(path) != subject[field]["sha256"]:
            raise ValueError(f"frozen input changed: {key}/{field}")
    return subject


def _set_lr(learner, lr):
    for group in learner.optimizer.param_groups:
        group["lr"] = lr
    return [float(group["lr"]) for group in learner.optimizer.param_groups]


def node(learner, subject, key, step, ctx, lr, device):
    """Same stored-target diagnostics as the published A/B/C nodes."""
    index, lookup, store, plan = ctx
    by_teacher, raw = {}, []
    for r in subject["recent_rounds"]:
        train = PH.evaluate_refs(learner, index, lookup, store, plan[f"train_{r}"])
        validation = []
        for game in index.split_games("validation"):
            if game.round_index != r:
                continue
            refs = np.asarray([(r, game.game_id, i) for i in range(game.decision_count)])
            metrics = PH.evaluate_refs(learner, index, lookup, store, refs)
            validation.append(metrics)
            raw.append({"round": r, "game_id": game.game_id,
                        **{k: metrics[k] for k in PH.SUM_FIELDS}})
        by_teacher[str(r)] = {"train_fixed_subset": train,
                              "holdout_full": PH.aggregate(validation),
                              "holdout_games": len(validation)}
    recent = {name: PH.aggregate([v[name] for v in by_teacher.values()])
              for name in ("train_fixed_subset", "holdout_full")}
    result = {"round": subject["round"], "arm": ARM, "step": step, "learning_rate": lr,
              "device": device, "total_train_step": learner.total_train_step,
              "by_teacher": by_teacher, "recent_four": recent,
              "latest": by_teacher[str(subject["recent_rounds"][-1])]}
    if step == 0:
        published = common.read_json(PH.ART / "nodes" / f"{key}_A_000000.json")
        for group in ("latest", "recent_four"):
            for metric in ("policy_kl", "value_loss"):
                got = result[group]["holdout_full"][metric]
                expected = published[group]["holdout_full"][metric]
                if abs(got - expected) > 2e-6:
                    raise ValueError(
                        f"{key}: node-0 {group}/{metric} differs from published A@0: "
                        f"{got} vs {expected}")
    common.write_json(common.LR_RUNTIME / key / f"holdout_games_{step:06d}.json", raw)
    common.write_json(common.ART / "nodes" / f"{key}_{ARM}_{step:06d}.json", result)
    emit("node", key=key, step=step,
         latest_train_KL=result["latest"]["train_fixed_subset"]["policy_kl"],
         latest_holdout_KL=result["latest"]["holdout_full"]["policy_kl"],
         recent_holdout_KL=recent["holdout_full"]["policy_kl"])
    return result


def run_key(key, device, max_new=None, do_nodes=True):
    subject = subject_of(key)
    identity = {"digest": common.sha256(SPEC_PATH), "subject": key, "arm": ARM,
                "lr": common.LR_VALUE, "device": device}
    folder = common.LR_RUN / key
    cps = folder / "checkpoints"
    cps.mkdir(parents=True, exist_ok=True)
    latest = sorted(cps.glob("step-*.pt"))
    start = latest[-1] if latest else Path(subject["start"]["path"])
    learner = PH.restore(start, device, identity, require_identity=bool(latest))
    _set_lr(learner, common.LR_VALUE)
    ctx = PH.context(subject)
    index, lookup, store, plan = ctx
    metrics_file = folder / "metrics.jsonl"
    committed = learner.round_train_step
    if metrics_file.exists():
        rows = [json.loads(line) for line in metrics_file.read_text().splitlines()]
        metrics_file.write_text("".join(json.dumps(row) + "\n" for row in rows
                                        if row["round_train_step"] <= committed))
    if do_nodes and not (common.ART / "nodes" / f"{key}_{ARM}_000000.json").exists():
        node(learner, subject, key, 0, ctx, common.LR_VALUE, device)
    limit = common.LR_STEPS if max_new is None else min(common.LR_STEPS, committed + max_new)
    for i in range(committed, limit):
        p_samples, p_refs = PH.resolve(index, lookup, plan["policy_A"][i], store)
        v_samples, v_refs = PH.resolve(index, lookup, plan["value"][i])
        metrics = learner.train_split_step(
            PH.collate_selfplay_samples(p_samples), PH.collate_selfplay_samples(v_samples),
            references=p_refs + v_refs,
            policy_max_grad_norm=subject["split_training"]["policy_max_grad_norm"],
            value_max_grad_norm=subject["split_training"]["value_max_grad_norm"])
        learner.sampling_rng = PH.DeterministicRng.restore(PH.RngState(
            "splitmix64-v1", int(plan["rng_state"][i]), int(plan["rng_draws"][i])))
        with metrics_file.open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        step = learner.round_train_step
        if step % 25 == 0 or step == limit:
            learner.save(cps / f"step-{step:06d}.pt",
                         metadata={"experiment": common.LR_IDENTIFIER, "arm": ARM,
                                   "source_round": subject["round"],
                                   "learning_rate": common.LR_VALUE})
            emit("training_progress", key=key, step=step, kl=metrics["policy_kl"])
        if do_nodes and step in common.LR_NODES:
            node(learner, subject, key, step, ctx, common.LR_VALUE, device)
    emit("training_job_finished", key=key, step=learner.round_train_step)


def run(device, keys, max_new=None):
    freeze(device)
    common.LR_RUNTIME.mkdir(parents=True, exist_ok=True)
    for key in keys:
        with (common.LR_RUNTIME / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = [sys.executable, "-u", str(Path(__file__)), "worker",
                       "--key", key, "--device", device]
            if max_new is not None:
                command += ["--max-new", str(max_new)]
            subprocess.run(command, cwd=common.ROOT, check=True)


def smoke():
    """Two CPU steps plus resume check on r012; no node evaluation."""
    subject = subject_of("r012")
    identity = {"digest": "smoke", "subject": "r012", "arm": ARM, "device": "cpu"}
    learner = PH.restore(subject["start"]["path"], "cpu", identity)
    _set_lr(learner, common.LR_VALUE)
    assert all(abs(g["lr"] - common.LR_VALUE) < 1e-18 for g in learner.optimizer.param_groups)
    ctx = PH.context(subject)
    index, lookup, store, plan = ctx
    for i in range(2):
        p_samples, p_refs = PH.resolve(index, lookup, plan["policy_A"][i], store)
        v_samples, v_refs = PH.resolve(index, lookup, plan["value"][i])
        metrics = learner.train_split_step(
            PH.collate_selfplay_samples(p_samples), PH.collate_selfplay_samples(v_samples),
            references=p_refs + v_refs,
            policy_max_grad_norm=subject["split_training"]["policy_max_grad_norm"],
            value_max_grad_norm=subject["split_training"]["value_max_grad_norm"])
        assert metrics["learning_rate"] == common.LR_VALUE, metrics
        assert math.isfinite(metrics["total_loss"]), metrics
    emit("smoke_done", steps=2, lr=common.LR_VALUE)


def worker(key, device, max_new):
    run_key(key, device, max_new=max_new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "worker", "smoke", "summary"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--key", choices=list(common.KEYS))
    parser.add_argument("--keys", nargs="+", choices=list(common.KEYS), default=list(common.KEYS))
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.key, args.device, args.max_new)
    elif args.action == "freeze":
        freeze(args.device)
    elif args.action == "run":
        run(args.device, tuple(args.keys), max_new=args.max_new)
    elif args.action == "smoke":
        smoke()
    else:
        result = {"protocol": "fixed-m4-lr-probe-summary-v1", "arm": ARM,
                  "learning_rate": common.LR_VALUE, "keys": {}}
        for key in common.KEYS:
            points = []
            for step in common.LR_NODES:
                path = common.ART / "nodes" / f"{key}_{ARM}_{step:06d}.json"
                if path.exists():
                    points.append(common.read_json(path))
            result["keys"][key] = points
        common.write_json(common.ART / "lr_results.json", result)
        print(json.dumps({"nodes": {k: len(v) for k, v in result["keys"].items()}}))


if __name__ == "__main__":
    main()
