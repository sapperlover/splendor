"""Train new 128x2 relations network without auxiliary heads, select full-holdout KL+MSE minimum, then face R43."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.offline_scratch import ScratchOfflineConfig, validation_schedule
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
OUTPUT = ROOT / "runs" / ART.name
RUNTIME = ROOT / "runtime" / ART.name
CONFIG = ROOT / "configs/entity_attention_v1/offline_128x2_relations_no_aux_r24_r43_20261019.json"
OPPONENT = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt"
OPPONENT_SHA = "b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14"
TEMPLATE = ROOT / "artifacts/entity_attention_r43_search_variants_arena_20261014/configs/S1024_vs_greedy.json"
PANELS = (("greedy_vs_R43", "greedy", 1000, 36600000),
          ("search_S1024_vs_R43", "search", 200, 36700000))
child = None
stopping = False


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment changed: {path}")
    else:
        write_json(path, value)


def prepare():
    config = ScratchOfflineConfig.from_dict(read_json(CONFIG))
    if sha256(Path(config.manifest)) != config.manifest_sha256 or sha256(OPPONENT) != OPPONENT_SHA:
        raise ValueError("Historical manifest or original R43 identity changed")
    with Path(config.manifest).open() as handle:
        games = [r for line in handle if config.first_round-1 <= (r := json.loads(line))["round_index"] < config.last_round]
    counts = Counter(g["round_index"] for g in games)
    if set(counts) != set(range(config.first_round-1, config.last_round)) or set(counts.values()) != {config.expected_games_per_round}:
        raise ValueError("Historical pool generation counts differ")
    splits = {split: {"games": sum(g["split"] == split for g in games),
        "positions": sum(g["decision_count"] for g in games if g["split"] == split)}
        for split in ("train", "validation")}
    steps = math.ceil(splits["train"]["positions"] * config.draw_exposures / config.learner.batch_size)
    plan = {"experiment": ART.name, "config": identity(CONFIG), "runner": identity(Path(__file__)),
        "opponent": identity(OPPONENT), "search_template": identity(TEMPLATE),
        "absolute_rounds": [config.first_round, config.last_round], "games": len(games), "splits": splits,
        "initialization": "random 128x2 relations model; no auxiliary heads; fresh AdamW; R43 is opponent only",
        "model_config": config.model_config, "steps_target": steps, "draw_exposures": config.draw_exposures,
        "validation_percent_step": config.validation_percent_step, "validation_steps": validation_schedule(config, steps),
        "selection": "minimum full-holdout policy KL + win/loss value MSE at the percentage checkpoints; auxiliary losses excluded; ties earlier step",
        "panels": [{"id": name, "kind": kind, "pairs": pairs, "seed_start": seed}
                   for name, kind, pairs, seed in PANELS],
        "scope": "Prepared pipeline only until run is explicitly invoked. Single training seed; no checkpoint selection by match results; no PSW; original archived targets and whole-game splits."}
    freeze(ART / "plan.json", plan)
    return plan


def evaluation_config(plan, selection, panel):
    raw = copy.deepcopy(read_json(TEMPLATE))
    seed = panel["seed_start"]
    raw.update(game_seeds=list(range(seed, seed+panel["pairs"])), policy_seed_base=seed+10000,
        bootstrap_seed=seed+20000, bootstrap_resamples=10000)
    candidate = Path(selection["checkpoint"])
    if sha256(candidate) != selection["checkpoint_sha256"] or sha256(OPPONENT) != plan["opponent"]["sha256"]:
        raise ValueError("Evaluation checkpoint identity changed")
    raw["metadata"] = {"experiment": ART.name, "panel": panel["id"],
        "selected_step": selection["selected_step"], "selection": plan["selection"],
        "candidate_sha256": selection["checkpoint_sha256"], "opponent_sha256": plan["opponent"]["sha256"]}
    search_spec = copy.deepcopy(raw["policy_a"])
    for side, path, name in (("policy_a", candidate, "selected_128x2_relations_no_aux"), ("policy_b", OPPONENT, "original_R43")):
        if panel["kind"] == "greedy":
            raw[side] = {"kind": "greedy_checkpoint", "name": name, "checkpoint": str(path)}
        else:
            spec = copy.deepcopy(search_spec)
            spec.update(checkpoint=str(path), name=name)
            spec["search_config"].update(simulations=1024, determinizations=1, root_noise_fraction=0., seed=seed+30000)
            raw[side] = spec
    if panel["kind"] == "greedy":
        raw.update(device="cpu", workers=8)
        raw.pop("execution")
    return FixedEvaluationConfig.from_dict(raw).to_dict()


def summarize_panel(panel):
    path = ART / "panels" / f"{panel['id']}.json"
    row = {**panel, "status": "pending", "completed_pairs": 0}
    if not path.exists():
        return row
    raw = read_json(path)
    config = read_json(ART / "evaluation_configs" / f"{panel['id']}.json")
    if raw["config"] != config:
        raise ValueError("Evaluation configuration differs")
    row.update(status="incomplete", completed_pairs=raw["completed_pairs"])
    if raw["completed_pairs"] != panel["pairs"]:
        return row
    seeds, scores = serialized_pair_scores(raw)
    if seeds != tuple(config["game_seeds"]) or raw["valid_games"] != 2*panel["pairs"]:
        raise ValueError("Evaluation population differs")
    ci = paired_bootstrap_mean(scores, resamples=config["bootstrap_resamples"], seed=config["bootstrap_seed"]).to_dict()
    if ci != raw["paired_bootstrap"]:
        raise ValueError("Paired confidence interval differs")
    for side, key in (("policy_a", "candidate_sha256"), ("policy_b", "opponent_sha256")):
        if raw["checkpoint_identity"][side]["sha256"] != config["metadata"][key]:
            raise ValueError("Evaluated model differs")
    for pair in raw["pairs"]:
        if [g["first_player"] for g in pair["games"]] != [0,1]:
            raise ValueError("Pair seats not swapped")
        for game in pair["games"]:
            for side in ("policy_a", "policy_b"):
                stats = game[f"{side}_stats"]
                if panel["kind"] == "search":
                    if stats["simulations"] != 1024*stats["searchable_decisions"] or stats["determinizations"] != stats["searchable_decisions"]:
                        raise ValueError("Actual search budget differs")
                elif stats.get("simulations",0) or stats.get("determinizations",0):
                    raise ValueError("Greedy unexpectedly searched")
    row.update(status="complete", score=raw["win_rate_score"], ci95=ci, result=identity(path),
        **{k:raw[k] for k in ("a_wins","draws","a_losses","exceptions","illegal_actions",
            "engineering_truncations","invariant_failures","policy_cycle_draws")})
    return row


def report(plan):
    progress_path = OUTPUT / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"status": "prepared_not_started", "step": 0}
    selection_path = OUTPUT / "selection.json"
    selection = read_json(selection_path) if selection_path.exists() else None
    full = [read_json(p) for p in sorted((OUTPUT / "full_validation").glob("step-*.json"))]
    panels = [summarize_panel(p) for p in plan["panels"]]
    status = ("complete" if all(p["status"] == "complete" for p in panels) else
              "training_complete_evaluation_pending" if progress["status"] == "complete" else progress["status"])
    result = {"status": status,
        "training": progress, "full_validation": full, "selection": selection, "panels": panels,
        "scope": "Validation-selected complete checkpoint; greedy and S1024 each directly versus original R43; no training or strength claim before execution."}
    write_json(ART / "summary.json", result)
    lines = ["# 新128×2 无辅助头：R24–R43离线训练与R43对战", "", f"状态：`{result['status']}`；训练 {progress['step']}/{plan['steps_target']} 步。", "",
        "新网络从随机初始化开始，lr1e-4，policy surprise与两个辅助目标关闭，市场及双方明保留购买关系开启；原R24–R43的20000局与原整局split。每5%保存并评估完整留出；每1000步仅保存续跑checkpoint。按完整留出KL＋胜负MSE最小值选整个checkpoint。", "",
        "| 步数 | Policy KL | 胜负MSE | KL＋MSE |",
        "| ---: | ---: | ---: | ---: |"]
    for point in full:
        m = point["metrics"]
        lines.append(f"| {point['step']} | {m['policy_kl']:.6f} | {m['value_loss']:.6f} | {m['policy_kl']+m['value_loss']:.6f} |")
    if selection is not None:
        lines += ["", f"选中 step={selection['selected_step']}，KL＋MSE={selection['selection_objective']:.6f}；checkpoint SHA `{selection['checkpoint_sha256']}`。"]
    lines += ["", "| 对战 | pair | 胜／和／负 | 候选得分 | 95%配对区间 |", "| --- | ---: | --- | ---: | --- |"]
    for p in panels:
        if p["status"] == "complete":
            ci=p["ci95"]
            lines.append(f"| {p['id']} | {p['pairs']} | {p['a_wins']}/{p['draws']}/{p['a_losses']} | {100*p['score']:.3f}% | [{100*ci['lower']:.3f}%,{100*ci['upper']:.3f}%] |")
        else:
            lines.append(f"| {p['id']} | {p['completed_pairs']}/{p['pairs']} | — | — | — |")
    lines += ["", "greedy每组1000pair；搜索为双方S1024/N1、native、c_puct1.5、根噪声0、T0，每组200pair。交换先后手、和棋半分，整pair bootstrap10000次。日志与未完成对局持续保存，汇总在阶段结束或status命令时刷新。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def interrupt(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def run_child(command, phase):
    global child
    print(f"START {phase}", flush=True)
    write_json(RUNTIME / "live.json", {"status":"running", "phase":phase, "pid":os.getpid(), "command":command})
    with (RUNTIME / f"{phase}.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        for line in child.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = child.wait()
        child = None
    if stopping or code == 130:
        raise KeyboardInterrupt
    if code:
        raise subprocess.CalledProcessError(code, command)


def verify_source(source):
    for path, expected in source["source_files"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Source changed during pipeline: {path}")
    for path, expected in source["native_extensions"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Native extension changed during pipeline: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","run","train","status"), nargs="?", default="run")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        result = report(read_json(ART / "plan.json"))
        print({"status":result["status"],"step":result["training"]["step"],
            "completed_panels":sum(p["status"] == "complete" for p in result["panels"])})
        return 0
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        report(plan)
        if args.action == "prepare":
            print({"status":"prepared_not_started","steps":plan["steps_target"],"validation_steps":plan["validation_steps"]})
            return 0
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        session = RUNTIME / str(time.time_ns())
        source = source_identity(ROOT, session)
        for p in (Path(__file__), CONFIG, ART / "plan.json"):
            shutil.copyfile(p, session / p.name)
        execution_path = ART / "execution.json"
        execution = read_json(execution_path) if execution_path.exists() else {"sessions":[]}
        record = {**source, "command":[sys.executable,*sys.argv],"start_unix":time.time(),"status":"running"}
        execution["sessions"].append(record)
        write_json(execution_path,execution)
        try:
            command = [sys.executable,"-u","-m","splendor.offline_scratch","--config",str(CONFIG),"--output",str(OUTPUT)]
            run_child(command,"train")
            verify_source(source)
            run_child(command+["--select-only"],"select")
            selection=read_json(OUTPUT / "selection.json")
            if args.action == "run":
                for panel in plan["panels"]:
                    verify_source(source)
                    raw=evaluation_config(plan,selection,panel)
                    path=ART / "evaluation_configs" / f"{panel['id']}.json"
                    output=ART / "panels" / f"{panel['id']}.json"
                    freeze(path,raw)
                    if summarize_panel(panel)["status"] == "complete":
                        continue
                    command=[sys.executable,"-u","-m","splendor.evaluation","--config",str(path),
                        "--output",str(output),"--progress-every","10","--summary-every","10"]
                    if output.exists() or output.with_suffix(".json.games").exists():
                        command.append("--resume")
                    run_child(command,panel["id"])
                    verify_source(source)
                    if summarize_panel(panel)["status"] != "complete":
                        raise ValueError("Evaluation stopped before completion")
                    report(plan)
            result=report(plan)
            record.update(status="complete" if args.action == "run" else "training_complete",end_unix=time.time())
            write_json(RUNTIME / "live.json",{"status":record["status"],"pid":os.getpid()})
            return 0
        except KeyboardInterrupt:
            record.update(status="interrupted",end_unix=time.time())
            write_json(RUNTIME / "live.json",{"status":"interrupted","pid":os.getpid()})
            report(plan)
            return 130
        except Exception as error:
            record.update(status="failed",end_unix=time.time(),error=f"{type(error).__name__}: {error}")
            write_json(RUNTIME / "live.json",{"status":"failed","error":record["error"]})
            report(plan)
            raise
        finally:
            write_json(execution_path,execution)


if __name__ == "__main__":
    raise SystemExit(main())
