"""Step / effective-LR / value-branch / label-averaging dose response.

All arms restore the same 16500-step checkpoint, replay the exact native batch
ledger (same states, same value batches, same optimizer moments) and differ only
in the controlled axis.  Checkpoints are prefixes of one trajectory, so nodes
(25/50/100/200/300/500) are directly comparable.

Axes:
- label:   4-search averaged labels vs the stored single-sample labels
- steps:   25/50/100/200/300/500 optimizer steps on the fixed 200-game batch
- lr:      1e-4 vs 2.5e-5, including 400 steps at 2.5e-5 (same lr*steps as
            100 steps at 1e-4)
- value:   critic updated on detached features vs critic head frozen
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint, save_alphazero_checkpoint
from splendor.alphazero_recording import append_json, environment_identity, read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.offline_train import make_index
from splendor.selfplay_data import collate_selfplay_samples

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_step_dose_20261007"
OUT = ROOT / "runs/entity_attention_step_dose_20261007"
AVG_ART = ROOT / "artifacts/entity_attention_target_averaging_20261007"
PROBE_ART = ROOT / "artifacts/entity_attention_chance_training_probe_20261007"

_spec = importlib.util.spec_from_file_location("target_averaging_probe", AVG_ART / "experiment.py")
avg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avg)

INITIAL = avg.INITIAL
INITIAL_SHA = avg.INITIAL_SHA
DEV_SEEDS = avg.DEV_SEEDS

ARMS = {
    "avg4_detach": {"label": "avg4", "value": "detach", "lr": 1e-4, "steps": 500,
                    "nodes": (25, 50, 100, 200, 300, 500)},
    "single_detach": {"label": "single", "value": "detach", "lr": 1e-4, "steps": 500,
                      "nodes": (25, 50, 100, 200, 300, 500)},
    "avg4_frozen": {"label": "avg4", "value": "frozen", "lr": 1e-4, "steps": 200,
                    "nodes": (25, 50, 100, 200)},
    "avg4_lr25e5": {"label": "avg4", "value": "detach", "lr": 2.5e-5, "steps": 400,
                    "nodes": (25, 50, 100, 200, 400)},
}

PANELS = (
    ("avg4_25_native", "avg4_detach", 25),
    ("avg4_50_native", "avg4_detach", 50),
    ("avg4_200_native", "avg4_detach", 200),
    ("single_50_native", "single_detach", 50),
    ("single_200_native", "single_detach", 200),
    ("avg4_frozen_100_native", "avg4_frozen", 100),
    ("avg4_lr25e5_100_native", "avg4_lr25e5", 100),
    ("avg4_lr25e5_400_native", "avg4_lr25e5", 400),
)

REFERENCES = {
    "avg4_100_native": AVG_ART / "evaluations/avg_100_native.json",
    "avg4_300_native": AVG_ART / "evaluations/avg_300_native.json",
    "single_100_native": PROBE_ART / "evaluations/native_100_native.json",
    "single_500_native": PROBE_ART / "evaluations/native_500_native.json",
}


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def checkpoint(arm, step):
    return OUT / arm / f"checkpoints/step-{step:06d}.pt"


def settings():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare():
    assert not (ART / "specification.json").exists(), "specification already frozen"
    assert sha256(INITIAL) == INITIAL_SHA
    inputs = [record(INITIAL), record(avg.NATIVE_DATASET), record(avg.NATIVE_LEDGER),
              record(AVG_ART / "labels_summary.json"), record(AVG_ART / "README.md")]
    inputs.extend(record(path) for path in REFERENCES.values() if path.exists())
    panels = {}
    for name, arm, step in PANELS:
        panels[name] = panel_spec(name, arm, step)
    for name, raw in panels.items():
        write_json(ART / f"configs/{name}.json", raw)
    source = source_identity(ROOT, RUNTIME / "source")
    extras = [ROOT / "src/splendor/native_search.cpp", ROOT / "setup.py",
              *[p for p in (ROOT / "data").glob("**/*") if p.is_file()]]
    source["additional_files"] = {}
    for path in extras:
        target = RUNTIME / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source["additional_files"][str(path.relative_to(ROOT))] = sha256(path)
    for path in ROOT.glob("src/splendor/_native_search*.so"):
        shutil.copyfile(path, RUNTIME / "source" / path.relative_to(ROOT))
    spec = {"protocol": "step-dose-probe-v1", "status": "prepared", "client_date": "2026-10-07",
            "initial": record(INITIAL), "arms": ARMS, "panels": [
                {"id": name, "arm": arm, "step": step, "pairs": 50} for name, arm, step in PANELS],
            "references": {k: record(v) for k, v in REFERENCES.items() if v.exists()},
            "inputs": inputs, "source": source,
            "scope": ("Fixed 200-game native batch, exact native ledger replay; axes are label type, "
                      "stopping step, learning rate and value-branch control. Single start/collection seed; "
                      "50-pair DEV strength panels plus a 200-root fit bank."),
            "commands": {}}
    prefix = "conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u "
    spec["commands"] = {name: prefix + str(Path(__file__).relative_to(ROOT)) + f" {name}"
                        for name in ("prepare", "train-all", "fit", "analyze", "verify")}
    write_json(ART / "specification.json", spec)
    emit(event="prepared", arms=list(ARMS), panels=list(panels))


def panel_spec(name, arm, step):
    search = {"simulations": 1024, "determinizations": 1, "seed": 16850000,
              "root_noise_fraction": 0.0, "dirichlet_alpha": 0.3, "c_puct": 1.5,
              "fold_forced_phases": True, "max_simulation_transitions": 512}
    inference = {"mode": "eager", "actor_projection": False, "candidate_buckets": [16, 32, 48, 64]}
    return {"policy_a": {"kind": "mcts_checkpoint", "name": f"{arm}_{step}", "checkpoint": str(checkpoint(arm, step)),
                         "search_backend": "native", "search_config": search, "inference": inference},
            "policy_b": {"kind": "mcts_checkpoint", "name": "frozen_16500_native", "checkpoint": str(INITIAL),
                         "search_backend": "native", "search_config": search.copy(), "inference": inference.copy()},
            "pairs": 50, "seed_start": DEV_SEEDS["game"], "policy_seed_base": DEV_SEEDS["policy"],
            "bootstrap_seed": DEV_SEEDS["bootstrap"], "bootstrap_resamples": 10000, "device": "cuda",
            "workers": 1, "max_completed_turns": 400, "adjudicate_deterministic_policy_cycles": True,
            "execution": {"backend": "native_batched", "active_games": 100, "max_batch_size": 200,
                          "candidate_buckets": [16, 32, 48, 64], "actor_projection": False,
                          "inference_mode": "eager", "max_batch_wait_ms": 0},
            "progress_every": 5, "summary_every": 5,
            "metadata": {"protocol": "step-dose-probe-v1", "arm": arm, "step": step}}


def train_arm(arm):
    cfg = ARMS[arm]
    spec = read_json(ART / "specification.json")
    assert sha256(INITIAL) == INITIAL_SHA
    labels = avg.load_labels() if cfg["label"] == "avg4" else None
    dataset = read_json(avg.NATIVE_DATASET)
    index = make_index(dataset, 1024 ** 3)
    pool = index.position_pool(split="train", round_index=0)
    assert len(pool) == 14477
    learner, payload = avg.restore()
    for group in learner.optimizer.param_groups:
        group["lr"] = float(cfg["lr"])
    ledger = np.load(avg.NATIVE_LEDGER)
    critic_params = list(learner.model.critic.parameters())
    critic_snapshot = [param.detach().clone() for param in critic_params] if cfg["value"] == "frozen" else None
    output = OUT / arm
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    started = time.perf_counter()
    saved = []
    replaced = kept = 0
    for step in range(int(cfg["steps"])):
        draws = [[pool.sample(learner.sampling_rng) for _ in range(256)] for _ in range(2)]
        for head in range(2):
            got = [(int(x[1].game_id), int(x[1].decision_index)) for x in draws[head]]
            expected = [tuple(int(v) for v in row) for row in ledger[step, head]]
            assert got == expected, (step, head)
        policy_samples = [sample for sample, _ in draws[0]]
        if cfg["label"] == "avg4":
            new_samples = []
            for sample, ref in draws[0]:
                row = labels.get((int(ref.game_id), int(ref.decision_index)))
                if row is None:
                    assert sample.candidate_slice.stop - sample.candidate_slice.start == 1
                    new_samples.append(sample)
                    kept += 1
                    continue
                target = row["fresh"].mean(axis=0)
                assert len(target) == sample.candidate_slice.stop - sample.candidate_slice.start
                new_samples.append(avg.replace_target(sample, target))
                replaced += 1
            policy_samples = new_samples
        policy_batch = collate_selfplay_samples(policy_samples)
        value_batch = collate_selfplay_samples([sample for sample, _ in draws[1]])
        references = [ref for batch in draws for _, ref in batch]
        row = learner.train_split_step(policy_batch, value_batch, references=references,
                                       policy_max_grad_norm=1.0, value_max_grad_norm=1.0)
        if critic_snapshot is not None:
            with torch.no_grad():
                for param, snapshot in zip(critic_params, critic_snapshot):
                    param.copy_(snapshot)
        append_json(metrics_path, row)
        if step + 1 in cfg["nodes"]:
            path = checkpoint(arm, step + 1)
            save_alphazero_checkpoint(path, model=learner.model, optimizer=learner.optimizer,
                learner_config=asdict(learner.config), round_index=learner.round_index,
                round_train_step=step + 1, total_train_step=learner.total_train_step, round_steps_target=500,
                parent_checkpoint=record(INITIAL),
                replay_identity={"dataset_sha256": sha256(avg.NATIVE_DATASET),
                                 "labels_sha256": None if labels is None else sha256(avg.RUNTIME / "labels.npz")},
                sampling_rng_state=learner.sampling_rng.snapshot(),
                metadata={"diagnostic": "step-dose-probe-v1", "arm": arm, "label": cfg["label"],
                          "value_branch": cfg["value"], "learning_rate": cfg["lr"]})
            saved.append(record(path))
        if (step + 1) % 50 == 0:
            emit(event="train_progress", arm=arm, step=step + 1, elapsed=round(time.perf_counter() - started, 1))
    result = {"arm": arm, **{k: v for k, v in cfg.items() if k != "nodes"},
              "checkpoints": saved, "wall_seconds": time.perf_counter() - started,
              "metrics": record(metrics_path), "policy_targets_replaced": replaced,
              "single_candidate_kept": kept, "environment": environment_identity()}
    write_json(ART / f"train_{arm}.json", result)
    del learner, payload
    torch.cuda.empty_cache()
    emit(event="train_done", arm=arm, steps=cfg["steps"], seconds=round(result["wall_seconds"], 2),
         saved=len(saved), replaced=replaced)


def train_all():
    for arm in ARMS:
        train_arm(arm)
    check_single_control()


def check_single_control():
    out = {}
    for step in (100, 300, 500):
        ours = load_alphazero_checkpoint(checkpoint("single_detach", step), device="cpu").build_model().state_dict()
        theirs = load_alphazero_checkpoint(avg.NATIVE_CKPT / f"step-{step:06d}.pt", device="cpu").build_model().state_dict()
        diffs = {k: float(torch.max(torch.abs(ours[k].float() - theirs[k].float()))) for k in ours}
        out[str(step)] = {"max_abs_diff": max(diffs.values()), "identical": bool(max(diffs.values()) == 0.0)}
    write_json(ART / "single_control_reproduction.json",
               {"protocol": "step-dose-control-v1", "compare": out})
    emit(event="single_control", **out)


def fit():
    bank = read_json(avg.BANK)
    rows = bank["positions"]
    requests = [avg.decode_request(row["request"]) for row in rows]
    raw = np.load(avg.RAW)["policy_target"]
    mean8 = raw[3].mean(axis=0)
    cases = {"start": INITIAL}
    for arm, cfg in ARMS.items():
        for step in cfg["nodes"]:
            cases[f"{arm}_{step}"] = checkpoint(arm, step)
    output = {}
    for name, path in cases.items():
        model = avg.load_frozen_actor_critic_checkpoint(path, device="cuda").eval()
        evaluator = avg.FrozenSearchInference(model, device="cuda", config=avg.inference_config())
        results = evaluator.policy_value(requests)
        per_root = []
        for i, row in enumerate(rows):
            logit, value = results[i]
            logit = np.asarray(logit, dtype=np.float64)
            logp = logit - np.logaddexp.reduce(logit)
            width = len(logit)
            pi = mean8[i][:width]
            mask = pi > 0
            per_root.append({"id": int(row["id"]), "game_key": row["game_key"],
                             "kl_to_mean8": float(np.sum(pi[mask] * (np.log(pi[mask]) - logp[mask]))),
                             "top1": bool(int(np.argmax(logit)) == int(np.argmax(pi)))})
        output[name] = {"checkpoint": record(path), "mean_kl_to_mean8": float(np.mean([r["kl_to_mean8"] for r in per_root])),
                        "top1_match_rate": float(np.mean([r["top1"] for r in per_root])), "per_root": per_root}
        del model, evaluator
        torch.cuda.empty_cache()
        emit(event="fit_done", name=name, kl=round(output[name]["mean_kl_to_mean8"], 5),
             top1=round(output[name]["top1_match_rate"], 4))
    write_json(ART / "fit.json", {"protocol": "step-dose-fit-v1", "bank": record(avg.BANK),
                                  "raw_targets": record(avg.RAW), "cases": output})


def evaluate(case):
    raw = read_json(ART / f"configs/{case}.json")
    config = FixedEvaluationConfig.from_dict(raw)
    output = ART / f"evaluations/{case}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    emit(event="evaluation_start", case=case, pairs=len(config.game_seeds))
    result = run_fixed_evaluation(config, output, resume=output.exists(), require_scored_pairs=True,
        progress_callback=lambda done, total: emit(event="evaluation_progress", case=case,
                                                   completed_pairs=done, pairs=total))
    assert result.summary["passed"]
    write_json(ART / f"evaluations/{case}_recording.json",
               {"config": record(ART / f"configs/{case}.json"), "output": record(output),
                "checkpoint_a": record(Path(config.policy_a.checkpoint)),
                "checkpoint_b": record(Path(config.policy_b.checkpoint)),
                "environment": environment_identity()})
    emit(event="evaluation_done", case=case, score=result.summary["win_rate_score"])


def analyze():
    spec = read_json(ART / "specification.json")
    fit_data = read_json(ART / "fit.json")
    panels = {}
    for name, arm, step in PANELS:
        path = ART / f"evaluations/{name}.json"
        if path.exists():
            result = read_json(path)
            assert result["passed"] and result["valid_games"] == 100
            panels[name] = {k: result[k] for k in ("win_rate_score", "paired_bootstrap", "a_wins", "a_losses",
                                                   "draws", "valid_games", "elapsed_seconds_this_invocation")}
    for name, path in REFERENCES.items():
        if path.exists():
            result = read_json(path)
            panels[name] = {k: result[k] for k in ("win_rate_score", "paired_bootstrap", "a_wins", "a_losses",
                                                   "draws", "valid_games", "elapsed_seconds_this_invocation")}
    differences = {}
    def diff(name_a, name_b, key):
        if name_a in panels and name_b in panels:
            a = read_json(ART / f"evaluations/{name_a}.json") if (ART / f"evaluations/{name_a}.json").exists() \
                else read_json(REFERENCES[name_a])
            b = read_json(ART / f"evaluations/{name_b}.json") if (ART / f"evaluations/{name_b}.json").exists() \
                else read_json(REFERENCES[name_b])
            differences[key] = avg.bootstrap_difference(a, b, DEV_SEEDS["bootstrap"])
    diff("avg4_100_native", "single_100_native", "avg4_100_minus_single_100")
    diff("avg4_50_native", "single_50_native", "avg4_50_minus_single_50")
    diff("avg4_50_native", "avg4_100_native", "avg4_50_minus_avg4_100")
    diff("avg4_200_native", "avg4_100_native", "avg4_200_minus_avg4_100")
    diff("avg4_300_native", "avg4_100_native", "avg4_300_minus_avg4_100")
    diff("avg4_frozen_100_native", "avg4_100_native", "frozen_100_minus_detach_100")
    diff("avg4_lr25e5_100_native", "avg4_100_native", "lr25e5_100_minus_lr1e4_100")
    diff("avg4_lr25e5_400_native", "avg4_100_native", "lr25e5_400_minus_lr1e4_100")
    fit_summary = {name: {"mean_kl_to_mean8": case["mean_kl_to_mean8"], "top1": case["top1_match_rate"]}
                   for name, case in fit_data["cases"].items()}
    output = {"protocol": "step-dose-analysis-v1", "fit": fit_summary, "panels": panels,
              "panel_differences": differences,
              "limitations": ["Single start/collection/training seed; 50-pair DEV panels; "
                              "fit bank is a separate 200-root held-out set.",
                              "Frozen value freezes only the critic head; the shared trunk still changes."]}
    write_json(ART / "analysis.json", output)
    emit(event="analysis_done", panels=list(panels), differences={k: round(v["difference"], 4)
                                                                  for k, v in differences.items()})


def verify():
    spec = read_json(ART / "specification.json")
    frozen = {item["path"]: item["sha256"] for item in spec["inputs"]}
    checks = {"initial": sha256(INITIAL) == INITIAL_SHA,
              "dataset": sha256(avg.NATIVE_DATASET) == frozen[str(avg.NATIVE_DATASET.resolve())],
              "ledger": sha256(avg.NATIVE_LEDGER) == frozen[str(avg.NATIVE_LEDGER.resolve())],
              "labels": sha256(avg.RUNTIME / "labels.npz") == read_json(AVG_ART / "labels_summary.json")["labels"]["sha256"]}
    control = read_json(ART / "single_control_reproduction.json")["compare"] if (ART / "single_control_reproduction.json").exists() else {}
    checks["single_control_identical"] = bool(control) and all(v["identical"] for v in control.values())
    for arm, cfg in ARMS.items():
        for step in cfg["nodes"]:
            path = checkpoint(arm, step)
            if not path.exists():
                checks[f"checkpoint_{arm}_{step}"] = False
    for name, _, _ in PANELS:
        path = ART / f"evaluations/{name}.json"
        if path.exists():
            result = read_json(path)
            checks[f"panel_{name}"] = bool(result["passed"] and result["valid_games"] == 100
                                           and not result["exceptions"] and not result["illegal_actions"]
                                           and not result["invariant_failures"] and not result["engineering_truncations"])
    output = {"protocol": "step-dose-verify-v1", "passed": all(checks.values()), "checks": checks}
    write_json(ART / "verification.json", output)
    emit(event="verify_done", passed=output["passed"], checks=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "train-all", "fit", "evaluate", "analyze", "verify"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--case")
    args = parser.parse_args()
    settings()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        assert args.arm
        train_arm(args.arm)
    elif args.action == "train-all":
        train_all()
    elif args.action == "fit":
        fit()
    elif args.action == "evaluate":
        assert args.case
        evaluate(args.case)
    elif args.action == "analyze":
        analyze()
    elif args.action == "verify":
        verify()
