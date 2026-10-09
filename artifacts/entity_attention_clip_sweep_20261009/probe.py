"""Policy-group clip-threshold response probe.

Replays the committed round-019 split-training ledger from the frozen
16500-step checkpoint.  Only the policy-group max gradient norm changes
across arms (0.5 / 1.0 / 2.0 / 4.0 / effectively disabled); the critic-group
clip stays at the committed 1.0 and the AdamW learning rate is 2.5e-5 or
1e-4.  Helpers, ledger and diagnostics are imported from the completed
``artifacts/entity_attention_formal_lr_step_20261007`` replay probe; nothing
in that directory (or elsewhere) is modified.

Outputs:
- ``summary.json`` and per-arm ``arm_<name>.json`` in this artifact directory;
- per-step metrics, node checkpoints and raw holdout outputs under the
  ignored ``runs/`` / ``runtime/`` trees.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_clip_sweep_20261009"
OUT = ROOT / "runs/entity_attention_clip_sweep_20261009"
PRIOR = ROOT / "artifacts/entity_attention_formal_lr_step_20261007"

_spec = importlib.util.spec_from_file_location("formal_lr_probe", PRIOR / "experiment.py")
exp = importlib.util.module_from_spec(_spec)
sys.modules["formal_lr_probe"] = exp
_spec.loader.exec_module(exp)

CLIPS = {"clip0p5": 0.5, "clip1": 1.0, "clip2": 2.0, "clip4": 4.0,
         "noclip": 1.0e9}
LRS = {"lr2p5e5": 2.5e-5, "lr1e4": 1.0e-4}
VALUE_CLIP = 1.0
STEPS = 100
NODES = (25, 50, 100)
START = exp.START


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": exp.sha256(path), "bytes": path.stat().st_size}


def git_identity():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                            capture_output=True, check=True).stdout
    return {"git_head": head,
            "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "script": record(Path(__file__)),
            "prior_experiment": record(PRIOR / "experiment.py"),
            "prior_specification": record(PRIOR / "specification.json")}


def sub_index(index, rounds):
    sub = exp.make_index({"games": [g.to_dict() for g in index.games
                                    if g.round_index in rounds]}, index.cache.max_bytes)
    sub.cache = index.cache
    return sub


def group_names(model):
    critic_ids = {id(param) for param in model.critic.parameters()}
    groups = {"policy": [], "critic": []}
    for name, param in model.named_parameters():
        groups["critic" if id(param) in critic_ids else "policy"].append(name)
    return groups


def displacement(groups, start_state, node_state):
    out = {}
    for group, names in groups.items():
        sq = 0.0
        max_abs = 0.0
        params = 0
        for name in names:
            delta = node_state[name].float() - start_state[name].float()
            sq += float(torch.sum(delta * delta))
            max_abs = max(max_abs, float(torch.max(torch.abs(delta))))
            params += delta.numel()
        out[group] = {"l2": math.sqrt(sq), "rms": math.sqrt(sq / params),
                      "max_abs": max_abs, "params": params}
    return out


def state_dict(checkpoint_path):
    payload = exp.load_alphazero_checkpoint(checkpoint_path, device="cpu")
    return payload.build_model(device="cpu").state_dict()


def aggregate(rows, policy_clip):
    norms = np.asarray([row["policy_grad_norm"] for row in rows], dtype=np.float64)
    value_norms = np.asarray([row["value_grad_norm"] for row in rows], dtype=np.float64)
    scales = np.minimum(1.0, float(policy_clip) / norms)
    return {
        "steps": len(rows),
        "policy_grad_norm": {"mean": float(norms.mean()), "median": float(np.median(norms)),
                             "min": float(norms.min()), "max": float(norms.max())},
        "policy_clip_scale": {"mean": float(scales.mean()),
                              "p10": float(np.percentile(scales, 10)),
                              "p50": float(np.percentile(scales, 50)),
                              "p90": float(np.percentile(scales, 90))},
        "policy_clip_frac": float(np.mean(norms > float(policy_clip))),
        "value_grad_norm_mean": float(value_norms.mean()),
        "value_clip_frac": float(np.mean(value_norms > VALUE_CLIP)),
        "total_loss_mean": float(np.mean([row["total_loss"] for row in rows])),
    }


def node_diagnostics(learner, checkpoint_path, recent_index, current_index, recent_games,
                     current_games, start_recent_raw, start_current_raw):
    recent_metrics = exp.evaluate_validation(learner, recent_index)
    current_metrics = exp.evaluate_validation(learner, current_index)
    raw_recent = exp.raw_outputs(learner, recent_index, recent_games)
    raw_current = exp.raw_outputs(learner, current_index, current_games)
    name = Path(checkpoint_path).parent.parent.name
    step = int(Path(checkpoint_path).stem.split("-")[1])
    prefix = f"{name}_step{step:03d}"
    exp.save_raw(RUNTIME / "diagnostics" / f"raw_{prefix}_recent.npz", raw_recent)
    exp.save_raw(RUNTIME / "diagnostics" / f"raw_{prefix}_current.npz", raw_current)
    return {
        "checkpoint": record(checkpoint_path),
        "holdout_recent": recent_metrics,
        "holdout_current": current_metrics,
        "fit_bank": exp.fit_bank(checkpoint_path),
        "change_vs_start_recent": exp.compare_raw(start_recent_raw, raw_recent),
        "change_vs_start_current": exp.compare_raw(start_current_raw, raw_current),
    }


def run_arm(name, lr, policy_clip, *, index, games, per_step_refs, recent_index,
            current_index, recent_games, current_games, start_recent_raw,
            start_current_raw, start_groups, start_state, prior_spec):
    result_path = ART / f"arm_{name}.json"
    if result_path.exists():
        emit(event="arm_skip", arm=name)
        return json.loads(result_path.read_text())
    learner, payload = exp.restore(START, lr, steps=STEPS)
    identity = exp.rng_identity(payload, learner)
    if identity != prior_spec["rng_identity"]:
        raise ValueError(f"{name}: restored RNG identity differs from the frozen record")
    state_sha = exp.model_state_sha256(learner.model.state_dict())
    if state_sha != prior_spec["start"]["model_state_sha256"]:
        raise ValueError(f"{name}: restored model state differs from the frozen record")
    optimizer_sha = exp.optimizer_state_sha256(learner.optimizer)
    if optimizer_sha != prior_spec["start"]["optimizer_state_sha256"]:
        raise ValueError(f"{name}: restored AdamW state differs from the frozen record")
    output = OUT / name
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    if metrics_path.exists():
        raise ValueError(f"{metrics_path} exists; refusing to append")
    rows = []
    nodes = {}
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        policy_refs = per_step_refs["policy_recent"][step]
        value_refs = per_step_refs["value_recent"][step] + per_step_refs["value_history"][step]
        policy_batch = exp.collate_selfplay_samples(
            exp.resolve_samples(index, games, policy_refs))
        value_batch = exp.collate_selfplay_samples(
            exp.resolve_samples(index, games, value_refs))
        row = learner.train_split_step(
            policy_batch, value_batch, references=policy_refs + value_refs,
            policy_max_grad_norm=float(policy_clip), value_max_grad_norm=VALUE_CLIP)
        exp.append_json(metrics_path, row)
        rows.append(row)
        if step in NODES:
            checkpoint_path = output / f"checkpoints/step-{step:06d}.pt"
            learner.save(checkpoint_path, metadata={
                "protocol": "clip-threshold-response-probe-v1", "arm": name,
                "learning_rate": lr, "policy_max_grad_norm": float(policy_clip),
                "value_max_grad_norm": VALUE_CLIP,
                "ledger": prior_spec["ledger"]["digest"], "start_checkpoint": record(START)})
            entry = node_diagnostics(learner, checkpoint_path, recent_index, current_index,
                                     recent_games, current_games, start_recent_raw,
                                     start_current_raw)
            entry["displacement"] = displacement(
                start_groups, start_state,
                state_dict(checkpoint_path))
            nodes[str(step)] = entry
            emit(event="node_done", arm=name, step=step,
                 recent_kl=round(entry["holdout_recent"]["policy_kl"], 6),
                 bank_kl=round(entry["fit_bank"]["mean_kl_to_mean8"], 6),
                 policy_l2=round(entry["displacement"]["policy"]["l2"], 6))
        if step % 25 == 0:
            emit(event="train_progress", arm=name, step=step,
                 elapsed=round(time.perf_counter() - started, 1))
    result = {
        "protocol": "clip-threshold-response-probe-v1",
        "arm": name, "learning_rate": lr, "policy_max_grad_norm": float(policy_clip),
        "value_max_grad_norm": VALUE_CLIP, "steps": STEPS, "nodes": list(NODES),
        "start": record(START), "start_model_state_sha256": state_sha,
        "start_optimizer_state_sha256": optimizer_sha, "rng_identity": identity,
        "training": aggregate(rows, policy_clip), "metrics": record(metrics_path),
        "checkpoints": [record(output / f"checkpoints/step-{step:06d}.pt") for step in NODES],
        "node_diagnostics": nodes,
        "wall_seconds": time.perf_counter() - started,
    }
    exp.write_json(result_path, result)
    del learner, payload
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    emit(event="arm_done", arm=name, seconds=round(result["wall_seconds"], 1))
    return result


def extension_per_step(context):
    rows, _ = exp.extension_rows()
    games = context["games"]
    per_step = {}
    for head, array in rows.items():
        items = [exp.ReplayReference(int(rr), games[(int(rr), int(gg))].run_id,
                                     int(gg), int(dd))
                 for _, rr, gg, dd in array]
        got = np.asarray([(ref.round_index, ref.game_id, ref.decision_index)
                          for ref in items], dtype="<i8")
        if not np.array_equal(got, array[:, 1:]):
            raise ValueError(f"extension ledger translation reordered {head}")
        per_step[head] = {step: items[(step - 1) * exp.HEADS[head]:step * exp.HEADS[head]]
                          for step in range(1, exp.EXT_STEPS + 1)}
    return per_step


def extend_arm(name, lr, policy_clip, *, context, ext_per_step):
    result_path = ART / f"ext_{name}.json"
    if result_path.exists():
        emit(event="ext_skip", arm=name)
        return json.loads(result_path.read_text())
    base_path = ART / f"arm_{name}.json"
    if not base_path.exists():
        raise ValueError(f"{name}: base arm result is missing")
    base = json.loads(base_path.read_text())
    start = OUT / name / "checkpoints/step-000100.pt"
    learner, payload = exp.restore(start, lr, steps=exp.EXT_STEPS)
    output = OUT / name
    metrics_path = output / "training_metrics_ext.jsonl"
    if metrics_path.exists():
        raise ValueError(f"{metrics_path} exists; refusing to append")
    rows = []
    nodes = {}
    started = time.perf_counter()
    for step in range(1, exp.EXT_STEPS + 1):
        policy_refs = ext_per_step["policy_recent"][step]
        value_refs = (ext_per_step["value_recent"][step]
                      + ext_per_step["value_history"][step])
        policy_batch = exp.collate_selfplay_samples(
            exp.resolve_samples(context["index"], context["games"], policy_refs))
        value_batch = exp.collate_selfplay_samples(
            exp.resolve_samples(context["index"], context["games"], value_refs))
        row = learner.train_split_step(
            policy_batch, value_batch, references=policy_refs + value_refs,
            policy_max_grad_norm=float(policy_clip), value_max_grad_norm=VALUE_CLIP)
        exp.append_json(metrics_path, row)
        rows.append(row)
        global_step = STEPS + step
        if global_step in (125, 150):
            checkpoint_path = output / f"checkpoints/step-{global_step:06d}.pt"
            learner.save(checkpoint_path, metadata={
                "protocol": "clip-threshold-response-probe-extension-v1", "arm": name,
                "learning_rate": lr, "policy_max_grad_norm": float(policy_clip),
                "value_max_grad_norm": VALUE_CLIP,
                "base_checkpoint": record(start)})
            entry = node_diagnostics(learner, checkpoint_path, context["recent_index"],
                                     context["current_index"], context["recent_games"],
                                     context["current_games"], context["start_recent_raw"],
                                     context["start_current_raw"])
            entry["displacement"] = displacement(
                context["start_groups"], context["start_state"],
                state_dict(checkpoint_path))
            nodes[str(global_step)] = entry
            emit(event="ext_node_done", arm=name, step=global_step,
                 recent_kl=round(entry["holdout_recent"]["policy_kl"], 6),
                 bank_kl=round(entry["fit_bank"]["mean_kl_to_mean8"], 6),
                 policy_l2=round(entry["displacement"]["policy"]["l2"], 6))
    result = {
        "protocol": "clip-threshold-response-probe-extension-v1",
        "arm": name, "learning_rate": lr, "policy_max_grad_norm": float(policy_clip),
        "value_max_grad_norm": VALUE_CLIP, "steps": exp.EXT_STEPS,
        "base": record(start), "base_100m": base["wall_seconds"],
        "training": aggregate(rows, policy_clip), "metrics": record(metrics_path),
        "node_diagnostics": nodes, "wall_seconds": time.perf_counter() - started,
    }
    exp.write_json(result_path, result)
    del learner, payload
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    emit(event="ext_done", arm=name, seconds=round(result["wall_seconds"], 1))
    return result


def summary(results):
    arms = {result["arm"]: result for result in results}
    node_keys = [str(step) for step in NODES]
    for name, arm in arms.items():
        ext_path = ART / f"ext_{name}.json"
        if ext_path.exists():
            extension = json.loads(ext_path.read_text())
            arm["node_diagnostics"].update(extension["node_diagnostics"])
            arm["extension"] = {"steps": extension["steps"],
                                "training": extension["training"],
                                "metrics": extension["metrics"]}
            node_keys = sorted(set(node_keys) | set(extension["node_diagnostics"]),
                               key=int)
    comparisons = {}
    for lr_name in LRS:
        base = arms.get(f"{lr_name}_clip1")
        if base is None:
            continue
        for clip_name in CLIPS:
            name = f"{lr_name}_{clip_name}"
            if name not in arms or clip_name == "clip1":
                continue
            arm = arms[name]
            comparisons[name] = {}
            for key in node_keys:
                b, a = base["node_diagnostics"][key], arm["node_diagnostics"][key]
                comparisons[name][key] = {
                    "recent_kl_minus_clip1": a["holdout_recent"]["policy_kl"]
                    - b["holdout_recent"]["policy_kl"],
                    "bank_kl_minus_clip1": a["fit_bank"]["mean_kl_to_mean8"]
                    - b["fit_bank"]["mean_kl_to_mean8"],
                    "policy_l2_minus_clip1": a["displacement"]["policy"]["l2"]
                    - b["displacement"]["policy"]["l2"],
                    "critic_l2_minus_clip1": a["displacement"]["critic"]["l2"]
                    - b["displacement"]["critic"]["l2"],
                    "change_recent_kl_minus_clip1":
                        a["change_vs_start_recent"]["policy_kl_after_before"]
                        - b["change_vs_start_recent"]["policy_kl_after_before"],
                }
    prior = json.loads((PRIOR / "analysis.json").read_text())
    reproduction = {}
    for lr_name, case in (("lr1e4", "lr1e4_100"), ("lr2p5e5", "lr2p5e5_100")):
        arm = arms.get(f"{lr_name}_clip1")
        if arm is None:
            continue
        old = prior["cases"][case]
        new = arm["node_diagnostics"]["100"]
        reproduction[lr_name] = {
            "recent_kl": {"prior": old["recent_kl"],
                          "clip1_rerun": new["holdout_recent"]["policy_kl"],
                          "abs_diff": abs(old["recent_kl"]
                                          - new["holdout_recent"]["policy_kl"])},
            "bank_kl": {"prior": old["bank_kl"],
                        "clip1_rerun": new["fit_bank"]["mean_kl_to_mean8"],
                        "abs_diff": abs(old["bank_kl"]
                                        - new["fit_bank"]["mean_kl_to_mean8"])},
            "change_recent_kl": {
                "prior": old["change_recent"]["policy_kl_after_before"],
                "clip1_rerun": new["change_vs_start_recent"]["policy_kl_after_before"],
                "abs_diff": abs(old["change_recent"]["policy_kl_after_before"]
                                - new["change_vs_start_recent"]["policy_kl_after_before"])},
        }
    output = {
        "protocol": "clip-threshold-response-probe-summary-v1",
        "client_date": "2026-10-09",
        "source": git_identity(),
        "environment": exp.environment_identity(),
        "design": {
            "start": record(START), "ledger": json.loads((PRIOR / "ledger_summary.json").read_text()),
            "policy_clip_values": CLIPS, "learning_rates": LRS, "value_clip": VALUE_CLIP,
            "steps": STEPS, "nodes": list(NODES),
            "replay": "committed round-019 split ledger, same AdamW/RNG per arm",
            "diagnostics": "stored-target recent/current KL, 8-search empirical-mean bank KL, "
                           "policy/value displacement, policy change vs start",
        },
        "arms": arms,
        "comparisons_vs_clip1": comparisons,
        "clip1_reproduction_of_prior_probe": reproduction,
        "limitations": [
            "Single committed start (16500), single frozen ledger, one sampling stream.",
            "No strength panels were run; KL/displacement are not chess strength.",
            "Value clip stays at 1.0 and is almost never active; only the policy group is varied.",
            "fit_bank compares to targets produced by the 16500 model, not new search labels.",
        ],
    }
    exp.write_json(ART / "summary.json", output)
    return output


def build_context():
    index = exp.build_index(exp.TRAIN_CACHE_BYTES)
    games = {(game.round_index, game.game_id): game for game in index.games}
    per_step_refs = exp.ledger_references(games)
    snapshot = exp.read_json(exp.SNAPSHOT)
    recent_rounds = list(snapshot["split_training"]["recent_round_indices"])
    current_round = max(recent_rounds)
    recent_index = sub_index(index, recent_rounds)
    current_index = sub_index(index, [current_round])
    recent_games = sorted((g for g in recent_index.games if g.split == "validation"),
                          key=lambda g: g.key)
    current_games = sorted((g for g in current_index.games if g.split == "validation"),
                           key=lambda g: g.key)
    start_learner, _ = exp.restore(START, 1e-4, steps=STEPS)
    start_recent_raw = exp.raw_outputs(start_learner, recent_index, recent_games)
    start_current_raw = exp.raw_outputs(start_learner, current_index, current_games)
    exp.save_raw(RUNTIME / "diagnostics/raw_start_recent.npz", start_recent_raw)
    exp.save_raw(RUNTIME / "diagnostics/raw_start_current.npz", start_current_raw)
    start_model = exp.load_alphazero_checkpoint(START, device="cpu").build_model(device="cpu")
    start_groups = group_names(start_model)
    start_state = start_model.state_dict()
    del start_learner, start_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return dict(index=index, games=games, per_step_refs=per_step_refs,
                recent_index=recent_index, current_index=current_index,
                recent_games=recent_games, current_games=current_games,
                start_recent_raw=start_recent_raw, start_current_raw=start_current_raw,
                start_groups=start_groups, start_state=start_state)


def run(selected_lrs, selected_clips):
    prior_spec = exp.read_json(PRIOR / "specification.json")
    context = build_context()
    results = []
    for lr_name in selected_lrs:
        for clip_name in selected_clips:
            name = f"{lr_name}_{clip_name}"
            result = run_arm(name, LRS[lr_name], CLIPS[clip_name], prior_spec=prior_spec,
                             **context)
            results.append(result)
    return summary(results)


def run_extension(selected_lrs, selected_clips):
    context = build_context()
    ext_per_step = extension_per_step(context)
    results = []
    for lr_name in selected_lrs:
        for clip_name in selected_clips:
            name = f"{lr_name}_{clip_name}"
            result = extend_arm(name, LRS[lr_name], CLIPS[clip_name],
                                context=context, ext_per_step=ext_per_step)
            results.append(result)
    return summary([json.loads(path.read_text()) for path in sorted(ART.glob("arm_*.json"))])


def smoke():
    """Two CPU steps of one arm to exercise the full path without GPU."""
    prior_spec = exp.read_json(PRIOR / "specification.json")
    learner, payload = exp.restore(START, 1e-4, device="cpu", steps=STEPS)
    index = exp.build_index(exp.TRAIN_CACHE_BYTES)
    games = {(game.round_index, game.game_id): game for game in index.games}
    per_step_refs = exp.ledger_references(games)
    for step in (1, 2):
        policy_refs = per_step_refs["policy_recent"][step]
        value_refs = per_step_refs["value_recent"][step] + per_step_refs["value_history"][step]
        row = learner.train_split_step(
            exp.collate_selfplay_samples(exp.resolve_samples(index, games, policy_refs)),
            exp.collate_selfplay_samples(exp.resolve_samples(index, games, value_refs)),
            references=policy_refs + value_refs,
            policy_max_grad_norm=2.0, value_max_grad_norm=1.0)
        emit(event="smoke_step", step=step, policy_grad_norm=round(row["policy_grad_norm"], 6),
             policy_clip_frac=float(row["policy_grad_norm"] > 2.0),
             value_grad_norm=round(row["value_grad_norm"], 6),
             policy_kl=round(row["policy_kl"], 6))
    emit(event="smoke_done", rng_matches="not checked", start_sha=exp.sha256(START),
         ledger_digest=prior_spec["ledger"]["digest"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "extend", "summarize", "smoke"))
    parser.add_argument("--lrs", nargs="*", choices=tuple(LRS), default=list(LRS))
    parser.add_argument("--clips", nargs="*", choices=tuple(CLIPS), default=list(CLIPS))
    args = parser.parse_args()
    exp.settings()
    if args.action == "smoke":
        smoke()
    elif args.action == "extend":
        run_extension(args.lrs, args.clips)
    elif args.action == "summarize":
        paths = sorted(ART.glob("arm_*.json"))
        summary([json.loads(path.read_text()) for path in paths])
        emit(event="summary_done", arms=len(paths))
    else:
        run(args.lrs, args.clips)
