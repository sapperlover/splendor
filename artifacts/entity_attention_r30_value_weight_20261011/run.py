"""Controlled round30 value-weight branches, with resumable training and arenas."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import (
    ReferenceWriter, environment_identity, read_json, sha256, source_identity, write_json,
)
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, run_training_round
from splendor.compact_game_cache import CompactGameCache
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.hybrid_train import draw_mixed_batch
from splendor.offline_train import _digest, make_index
from splendor.pool_fitting import RoundFitting

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUN = ROOT / "runs" / ART.name
RUNTIME = ROOT / "runtime" / ART.name
ORIGINAL = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-020"
PARENT = ORIGINAL.parent / "round-019/checkpoints/final.pt"
INITIAL = ORIGINAL / "checkpoints/initial.pt"
PLAN = ART / "plan.json"
POINTS = (0, 100, 250, 500, 1000)
BRANCHES = {"value025": 0.25, "value0": 0.0}
ARENAS = {"greedy": "greedy_vs_start", "search": "mcts_vs_fixed_anchor"}


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment changed: {path}")
    else:
        write_json(path, value)


def prepare():
    if sha256(PARENT) != "8918f75bc1dd098553fe4e1dbb9b4b6898620504b202ebeccaca869e4c918227":
        raise ValueError("round29 parent changed")
    payload = load_alphazero_checkpoint(INITIAL)
    parent = load_alphazero_checkpoint(PARENT)
    if payload.model_sha256 != parent.model_sha256:
        raise ValueError("round30 initial weights differ from round29 terminal weights")
    if (payload.round_index, payload.round_train_step, payload.total_train_step,
            payload.round_steps_target) != (30, 0, 58898, 1000):
        raise ValueError("start checkpoint counters differ")
    snapshot = read_json(ORIGINAL / "replay_snapshot.json")
    records = snapshot["games"]
    if (len(records) != 6000 or snapshot["quotas"] != [0, 256]
            or sorted({g["round_index"] for g in records}) != list(range(24, 30))
            or payload.replay_identity["digest"] != snapshot["digest"]):
        raise ValueError("original frozen pool differs")
    config = LearnerConfig(**payload.learner_config)
    if (config.learning_rate, config.batch_size, config.value_loss_weight) != (2e-5, 256, 1.0):
        raise ValueError("original training settings differ")
    freeze(ART / "replay_snapshot.json", snapshot)
    freeze(ART / "train_references.json", read_json(ORIGINAL / "fitting/train_references.json"))
    baseline = ART / "baseline"
    for step in POINTS:
        freeze(baseline / "fitting" / f"step-{step:09d}.json",
               read_json(ORIGINAL / "fitting" / f"step-{step:09d}.json"))
    templates = {}
    for mode, name in ARENAS.items():
        result = read_json(ORIGINAL / "evaluation" / f"{name}.json")
        if not result["passed"] or result["completed_pairs"] != 200:
            raise ValueError("original weight1 arena incomplete")
        freeze(baseline / "evaluation" / f"{mode}.json", result)
        templates[mode] = result["config"]
    plan = {
        "experiment": ART.name, "round29_parent": identity(PARENT),
        "initial_checkpoint": identity(INITIAL), "baseline_final": identity(ORIGINAL / "checkpoints/final.pt"),
        "base_learner": asdict(config), "replay_snapshot": identity(ART / "replay_snapshot.json"),
        "train_references": identity(ART / "train_references.json"), "target_steps": 1000,
        "fitting_steps": list(POINTS),
        "branches": {name: asdict(replace(config, value_loss_weight=weight))
                     for name, weight in BRANCHES.items()},
        "arena_templates": templates, "runner": identity(Path(__file__)),
        "scope": "Independent branches from round29 weights and original round30 AdamW/RNG; only value loss weight changes. Original weight1 round30 is reused. Fixed DEV data and opponent panels; not independent training-seed replication.",
        "zero_weight_semantics": "No value-loss gradient; existing AdamW moments/decay and shared encoder can still change critic predictions. No parameter freezing or optimizer reset.",
        "order": ["value025 train", "value025 greedy200pairs", "value025 search200pairs",
                  "value0 train", "value0 greedy200pairs", "value0 search200pairs"],
    }
    freeze(PLAN, plan)
    return plan


def train(branch, plan):
    config = LearnerConfig(**plan["branches"][branch])
    checkpoints = RUN / branch / "checkpoints"
    saved = sorted(checkpoints.glob("step-*.pt"))
    start = saved[-1] if saved else checkpoints / "initial.pt"
    original = load_alphazero_checkpoint(INITIAL)
    if start.exists():
        learner = AlphazeroLearner.resume(start, config, replay_identity=original.replay_identity,
            round_index=30, round_steps_target=1000)
    else:
        learner = AlphazeroLearner.resume(INITIAL, LearnerConfig(**original.learner_config),
            replay_identity=original.replay_identity, round_index=30, round_steps_target=1000)
        learner.config = config
        learner.save(start, metadata={"experiment": ART.name, "branch": branch,
                                     "source": plan["initial_checkpoint"]})
    snapshot = read_json(ART / "replay_snapshot.json")
    index = make_index(snapshot, 8 * 1024**3)
    index.cache = CompactGameCache(index.cache.max_bytes)
    pool = PositionPool(index.split_games("train"), index.cache, "online")
    fitting = RoundFitting({"steps": list(POINTS), "seed": 32600000,
        "train_positions_per_generation": 2048}, RUN, ART / branch,
        None, index, None, inherited_references=read_json(ART / "train_references.json")["references"])
    fitting(learner, start)
    session = str(time.time_ns())
    writer = ReferenceWriter(RUN / branch / "usage", session, config.batch_size, original.replay_identity)
    try:
        stats = run_training_round(learner, steps_target=1000, current_pool=pool, history_pool=None,
            validation_samples=(), validation_games=0, checkpoint_dir=checkpoints,
            metrics_path=RUN / branch / "training_metrics.jsonl", reference_writer=writer,
            checkpoint_callback=fitting,
            batch_drawer=lambda rng: draw_mixed_batch(pool, pool, rng, 0, config.batch_size, record_rounds=True),
            progress_callback=lambda row: print(json.dumps({"event": "train", "branch": branch,
                "step": row["round_train_step"], "policy_loss": row["policy_loss"],
                "value_loss": row["value_loss"], "grad_norm": row["grad_norm"]}), flush=True)
                if row["round_train_step"] % 100 == 0 else None)
    finally:
        writer.close()
    write_json(ART / branch / "training_sessions" / f"{session}.json", stats)
    original_rows = {r["round_train_step"]: r for r in
        map(json.loads, (ORIGINAL / "training_metrics.jsonl").read_text().splitlines()) if r["event"] == "train_step"}
    rows = [r for r in map(json.loads, (RUN / branch / "training_metrics.jsonl").read_text().splitlines())
            if r["event"] == "train_step"]
    # The original hybrid writer hashes a list of per-source hashes, even
    # when online is its only source; our single-source writer returns it raw.
    if len(rows) != 1000 or any(_digest([r["reference_sha256"]]) != original_rows[r["round_train_step"]]["reference_sha256"] for r in rows):
        raise ValueError("Training sample sequence differs from original weight1")
    end = load_alphazero_checkpoint(checkpoints / "step-000001000.pt")
    drift = {}
    for group in ("critic", "actor", "shared_encoder"):
        keys = [k for k in original.model_state_dict if
                (k.startswith(group + ".") if group != "shared_encoder"
                 else not k.startswith(("critic.", "actor.")))]
        diffs = [(end.model_state_dict[k].double() - original.model_state_dict[k].double()) for k in keys]
        drift[group] = {"tensor_count": len(keys), "delta_l2": float(sum((d*d).sum() for d in diffs).sqrt()),
                        "delta_max_abs": max(float(d.abs().max()) for d in diffs)}
    final = checkpoints / "final.pt"
    if not final.exists():
        shutil.copyfile(checkpoints / "step-000001000.pt", final)
    elif sha256(final) != sha256(checkpoints / "step-000001000.pt"):
        raise ValueError("final checkpoint differs")
    write_json(ART / branch / "training_verification.json", {
        "checkpoint": identity(final), "value_loss_weight": config.value_loss_weight,
        "steps": len(rows), "sample_draws": len(rows) * config.batch_size,
        "identical_sample_sequence_to_weight1": True, "parameter_drift": drift,
        "mean_preclip_grad_norm": sum(r["grad_norm"] for r in rows) / len(rows),
        "clip_fraction": sum(r["grad_norm"] > config.max_grad_norm for r in rows) / len(rows),
        "zero_weight_semantics": plan["zero_weight_semantics"]})
    print(json.dumps({"event": "training_complete", "branch": branch, "checkpoint": str(final)}), flush=True)


def arena(branch, mode, plan):
    config = dict(plan["arena_templates"][mode])
    config["policy_a"] = {**config["policy_a"],
        "checkpoint": str(RUN / branch / "checkpoints/final.pt"), "name": branch}
    config = FixedEvaluationConfig.from_dict(config).to_dict()
    freeze(ART / branch / "configs" / f"{mode}.json", config)
    output = ART / branch / "evaluation" / f"{mode}.json"
    run_fixed_evaluation(FixedEvaluationConfig.from_dict(config), output,
        resume=output.exists() or output.with_suffix(".json.games").exists(),
        progress_every=10, summary_every=10,
        progress_callback=lambda done, total: print(json.dumps({"event": "arena", "branch": branch,
            "mode": mode, "completed_pairs": done, "total_pairs": total}), flush=True),
        failure_dir=RUNTIME / branch / "failures")
    result = read_json(output)
    if not result["passed"] or result["completed_pairs"] != 200:
        raise ValueError("arena incomplete or invalid")
    print(json.dumps({"event": "arena_complete", "branch": branch, "mode": mode,
        "score": result["win_rate_score"], "ci": result["paired_bootstrap"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "train", "arena"), default="run", nargs="?")
    parser.add_argument("--branch", choices=tuple(BRANCHES))
    parser.add_argument("--mode", choices=tuple(ARENAS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = prepare()
    if args.action == "prepare":
        print(json.dumps({"plan": str(PLAN)}), flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen training/search settings")
    if args.action == "train":
        train(args.branch, plan)
        return 0
    if args.action == "arena":
        arena(args.branch, args.mode, plan)
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = str(time.time_ns())
        destination = RUNTIME / session
        source = source_identity(ROOT, destination)
        for path in (Path(__file__), PLAN, ART / "replay_snapshot.json", ART / "train_references.json"):
            shutil.copyfile(path, destination / path.name)
        record = ART / "execution.json"
        execution = read_json(record) if record.exists() else {"sessions": []}
        execution["sessions"].append({**source, "environment": environment_identity(),
            "command": [sys.executable, *sys.argv], "start_unix": time.time(),
            "runtime": str(destination.relative_to(ROOT)), "plan_sha256": sha256(PLAN)})
        write_json(record, execution)
        live = {"pid": os.getpid(), "status": "running", "plan_sha256": sha256(PLAN)}
        try:
            for branch in BRANCHES:
                commands = [("train", ["train", "--branch", branch])]
                commands += [(mode, ["arena", "--branch", branch, "--mode", mode]) for mode in ARENAS]
                for phase, extra in commands:
                    if phase == "train":
                        finished = (ART / branch / "training_verification.json").exists()
                    else:
                        path = ART / branch / "evaluation" / f"{phase}.json"
                        finished = path.exists() and read_json(path).get("passed", False) and read_json(path)["completed_pairs"] == 200
                    if finished:
                        continue
                    live.update(branch=branch, phase=phase, time_unix=time.time())
                    write_json(RUNTIME / "live.json", live)
                    command = [sys.executable, "-u", str(Path(__file__)), *extra]
                    print(json.dumps({"event": "start", "branch": branch, "phase": phase}), flush=True)
                    subprocess.run(command, cwd=ROOT, check=True)
            live.update(status="complete", time_unix=time.time())
            write_json(RUN / "complete.json", {"plan_sha256": sha256(PLAN), "time_unix": time.time()})
        except BaseException:
            live.update(status="interrupted_or_failed", time_unix=time.time())
            raise
        finally:
            write_json(RUNTIME / "live.json", live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
