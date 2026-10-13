"""R24 historical-pool offline fit, fixed screening, and independent opener confirmation."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from pool import HERE, ROOT, prepare
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.paired_statistics import serialized_pair_scores

NAME = HERE.name
OUTPUT = ROOT / "runs" / NAME
RUNTIME = ROOT / "runtime" / NAME
INITIAL = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-014/checkpoints/final.pt"
TEMPLATE = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-014/evaluation/mcts_vs_fixed_anchor.json"
child = None
stopping = False


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen experiment changed: {path}")
    else:
        write_json(path, value)


def prepare_plan():
    summary = prepare()
    original = read_json(ROOT / "artifacts/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-000024.json")["checkpoint"]
    if sha256(INITIAL) != original["sha256"]:
        raise ValueError("original round24 checkpoint SHA mismatch")
    parent = load_alphazero_checkpoint(INITIAL)
    learner = dict(parent.learner_config)
    learner.update(device="cuda", learning_rate=2e-5, checkpoint_every_steps=1000,
                   batch_size=256, validation_batch_size=256, max_steps_per_round=None)
    train_positions = summary["totals"]["train"]["positions"]
    milestones = {str(e): math.ceil(e * train_positions / 256) for e in (.5, 1, 2, 4)}
    config = {
        "protocol": "r24-fixed-history-offline-v1",
        "manifest": str(HERE / "pool_games.jsonl"), "manifest_sha256": summary["manifest_sha256"],
        "output": str(OUTPUT), "artifacts": str(HERE),
        "initial_checkpoint": {"path": str(INITIAL), "sha256": original["sha256"],
                               "model_sha256": parent.model_sha256,
                               "total_train_step": parent.total_train_step},
        "preserve_optimizer": True, "sampling_seed": 34410000, "validation_seed": 34420000,
        "positions_per_source_split": 512, "cache_mib": 8192,
        "draw_exposures": 4, "full_holdout_exposures": [0, .5, 1, 2, 4],
        "full_holdout_steps": [1000],
        "lr_drop_exposure": 4, "late_lr_multiplier": 1., "learner": learner,
        "milestones": milestones,
        "training_loop_origin": {
            "path": "artifacts/entity_attention_recent_pool_20261009/train.py",
            "sha256": sha256(ROOT / "artifacts/entity_attention_recent_pool_20261009/train.py"),
            "changes": "warm R24 weights/AdamW; per-generation diagnostics; explicit full-holdout steps",
        },
        "strength_plan": {
            "opponent": "original complete round24 model for all panels",
            "greedy": {"steps": [1000] + list(milestones.values()), "pairs": 200, "seed_start": 34500000},
            "search_dev": {"steps": list(milestones.values()), "pairs": 200, "seed_start": 34600000},
            "confirmation": {"pairs": 500, "seed_start": 34700000,
                             "selection": "highest completed search DEV score; ties choose earlier step"},
            "search_template": {"path": str(TEMPLATE), "sha256": sha256(TEMPLATE)},
            "scope": "single training seed; best-node DEV scores are exploratory; separate opener bank confirms selection",
        },
    }
    freeze(HERE / "config.json", config)
    return config


def checkpoint(step):
    return OUTPUT / "checkpoints" / f"step-{step:09d}.pt"


def evaluation_config(config, step, kind, *, confirm=False):
    plan = config["strength_plan"]
    initial_round = config.get("initial_round", 24)
    if sha256(TEMPLATE) != plan["search_template"]["sha256"]:
        raise ValueError("search evaluation template changed")
    raw = copy.deepcopy(read_json(TEMPLATE)["config"])
    settings = plan["confirmation" if confirm else "search_dev" if kind == "search" else "greedy"]
    seed = settings["seed_start"]
    raw.pop("pairs", None)
    raw.pop("seed_start", None)
    raw.update(game_seeds=list(range(seed, seed + settings["pairs"])),
               policy_seed_base=seed + 10000, bootstrap_seed=seed + 20000,
               bootstrap_resamples=10000, metadata={
                   "experiment": NAME, "additional_training_step": step,
                   "candidate_sha256": sha256(checkpoint(step)),
                   "opponent_sha256": config["initial_checkpoint"]["sha256"],
                   "panel": "confirmation" if confirm else f"{kind}_DEV",
                   "selection_scope": plan["confirmation"]["selection"] if confirm else "predeclared checkpoint",
               })
    for side, path, name in (("policy_a", checkpoint(step), f"R{initial_round}_history_step{step}"),
                             ("policy_b", INITIAL, f"original_R{initial_round}")):
        if kind == "search":
            raw[side].update(checkpoint=str(path), name=name)
            raw[side]["search_config"].update(simulations=1024, determinizations=1,
                                               root_noise_fraction=0., seed=seed + 30000)
        else:
            raw[side] = {"checkpoint": str(path), "name": name, "kind": "greedy_checkpoint"}
    if kind == "greedy":
        raw.update(device="cpu", workers=8)
        raw.pop("execution", None)
    return FixedEvaluationConfig.from_dict(raw).to_dict()


def run_child(command, phase):
    global child
    if stopping:
        raise KeyboardInterrupt
    write_json(RUNTIME / "live.json", {"phase": phase, "status": "running", "time_unix": time.time(), "command": command})
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    code = child.wait()
    child = None
    if stopping:
        raise KeyboardInterrupt
    if code:
        raise subprocess.CalledProcessError(code, command)


def run_panel(config, step, kind, *, confirm=False):
    name = "confirmation" if confirm else f"{kind}_step{step:09d}"
    raw = evaluation_config(config, step, kind, confirm=confirm)
    config_path = HERE / "evaluation_configs" / f"{name}.json"
    output = HERE / "panels" / f"{name}.json"
    freeze(config_path, raw)
    if output.exists():
        result = read_json(output)
        if result["config"] != raw:
            raise ValueError("saved panel configuration differs from the frozen plan")
        if result.get("passed") and result.get("completed_pairs") == len(raw["game_seeds"]):
            serialized_pair_scores(result)
            return result
    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(config_path),
               "--output", str(output), "--progress-every", "10", "--summary-every", "10"]
    if output.exists() or Path(str(output) + ".games").exists():
        command.append("--resume")
    run_child(command, name)
    result = read_json(output)
    serialized_pair_scores(result)
    if result["completed_pairs"] != len(raw["game_seeds"]):
        raise ValueError("evaluation incomplete")
    return result


def analyze(config):
    fitting = []
    matrix = []
    for path in sorted(HERE.glob("full_holdout_step_*.json")):
        raw = read_json(path)
        fitting.append({"step": raw["step"], "metrics": raw["evaluation"]["pool_weighted"]["validation"],
                        "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        for key, metrics in raw["evaluation"]["metrics"].items():
            generation, split, stage = key.split("/")
            if generation.startswith("g"):
                matrix.append({"step": raw["step"], "generation": generation, "split": split, "stage": stage,
                               **{name: metrics[name] for name in ("positions", "policy_kl", "value_mse",
                                                                  "teacher_top1_agreement", "value_bias", "value_correlation")}})
    curve_path = HERE / "curve.jsonl"
    if curve_path.exists():
        for line in curve_path.read_text().splitlines():
            point = json.loads(line)
            for key, metrics in point["evaluation"]["metrics"].items():
                generation, split, stage = key.split("/")
                if generation.startswith("g"):
                    matrix.append({"step": point["step"], "generation": generation, "split": f"{split}_fixed512", "stage": stage,
                                   **{name: metrics[name] for name in ("positions", "policy_kl", "value_mse",
                                                                      "teacher_top1_agreement", "value_bias", "value_correlation")}})
    if matrix:
        with (HERE / "generation_stage_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
            writer.writeheader()
            writer.writerows(matrix)
    panels = []
    for path in sorted((HERE / "panels").glob("*.json")):
        raw = read_json(path)
        panels.append({"panel": path.stem, "step": raw["config"]["metadata"]["additional_training_step"],
                       "passed": raw["passed"], "completed_pairs": raw["completed_pairs"],
                       "total_pairs": raw["total_pairs"], "score": raw["win_rate_score"],
                       "ci95": raw["paired_bootstrap"], "path": str(path.relative_to(ROOT)),
                       "sha256": sha256(path)})
    result = {"experiment": NAME, "training": read_json(HERE / "progress.json") if (HERE / "progress.json").exists() else {"status": "prepared_not_started"},
              "full_holdouts": fitting, "panels": panels,
              "scope": "stored-label DEV fitting; selection on fixed search DEV; separate 500-pair openers confirm one chosen node; single training seed"}
    write_json(HERE / "results.json", result)
    initial_round = config.get("initial_round", 24)
    lines = [f"# 第{initial_round}轮模型：4.3万局历史池离线训练", "", f"训练状态：{result['training']['status']}。正式实验是否有收益以完成的对战为准。", "",
             "| 追加步数 | 完整留出 KL | 完整留出 MSE |", "| ---: | ---: | ---: |"]
    for row in fitting:
        metrics = row["metrics"]
        lines.append(f"| {row['step']} | {metrics['policy_kl']:.6f} | {metrics['value_mse']:.6f} |")
    lines += ["", "| 面板 | 追加步数 | 完成pair | 得分率 | 95% CI |", "| --- | ---: | ---: | ---: | --- |"]
    for row in panels:
        ci = row["ci95"]
        interval = f"[{ci['lower']:.2%}, {ci['upper']:.2%}]" if ci else "未完成"
        score = f"{row['score']:.2%}" if row["score"] is not None else "未完成"
        lines.append(f"| {row['panel']} | {row['step']} | {row['completed_pairs']}/{row['total_pairs']} | {score} | {interval} |")
    lines += ["", f"对手均为原第{initial_round}轮完整模型。搜索为双方S1024/N1、无根噪声、互换先后手；和棋半分。",
              "DEV用于选节点；独立开局确认仍限定本训练seed、对手和预算。未完成面板不能作棋力结论。", ""]
    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run(config, *, training_only=False, max_steps_this_session=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("formal training needs CUDA; no CPU fallback")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "chain.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = RUNTIME / str(time.time_ns())
        provenance = source_identity(ROOT, session / "source")
        for path in HERE.glob("*.py"):
            shutil.copyfile(path, session / path.name)
        write_json(session / "identity.json", {**provenance, "command": sys.argv, "config_sha256": sha256(HERE / "config.json")})
        progress = HERE / "progress.json"
        if not progress.exists() or read_json(progress)["status"] != "complete":
            command = [sys.executable, "-u", config.get("training_script", str(HERE / "train.py")),
                       "--config", str(HERE / "config.json")]
            if max_steps_this_session is not None:
                command += ["--max-steps-this-session", str(max_steps_this_session)]
            run_child(command, "training")
        analyze(config)
        if training_only or read_json(progress)["status"] != "complete":
            return
        for step in config["strength_plan"]["greedy"]["steps"]:
            run_panel(config, step, "greedy")
            analyze(config)
        dev = []
        for step in config["strength_plan"]["search_dev"]["steps"]:
            raw = run_panel(config, step, "search")
            dev.append((step, raw))
            analyze(config)
        best_step, best = max(dev, key=lambda item: (item[1]["win_rate_score"], -item[0]))
        freeze(HERE / "selection.json", {"step": best_step, "score_DEV": best["win_rate_score"],
                                        "checkpoint_sha256": sha256(checkpoint(best_step)),
                                        "rule": config["strength_plan"]["confirmation"]["selection"]})
        run_panel(config, best_step, "search", confirm=True)
        analyze(config)
        write_json(RUNTIME / "live.json", {"status": "complete", "phase": "complete", "time_unix": time.time()})


def request_stop(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "train", "status", "analyze"), nargs="?", default="run")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    if args.max_steps_this_session is not None and args.max_steps_this_session <= 0:
        parser.error("--max-steps-this-session must be positive")
    if args.action == "status":
        progress = read_json(HERE / "progress.json") if (HERE / "progress.json").exists() else {"status": "not_started"}
        result = {"training": {key: progress.get(key) for key in ("status", "step", "steps_target", "best_validation_kl_step")},
                  "latest_fitting": progress.get("latest", {}).get("evaluation", {}).get("pool_weighted"),
                  "live": read_json(RUNTIME / "live.json") if (RUNTIME / "live.json").exists() else None}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    config = prepare_plan()
    if args.action == "prepare":
        print(json.dumps({"steps_target": config["milestones"]["4"], "milestones": config["milestones"],
                          "pool": read_json(HERE / "pool_summary.json")["totals"]}, ensure_ascii=False, indent=2))
        analyze(config)
    elif args.action == "analyze":
        analyze(config)
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)
        try:
            run(config, training_only=args.action == "train", max_steps_this_session=args.max_steps_this_session)
        except KeyboardInterrupt:
            analyze(config)
            print("已中断；同一命令恢复已提交训练checkpoint或对战pair。", flush=True)


if __name__ == "__main__":
    main()
