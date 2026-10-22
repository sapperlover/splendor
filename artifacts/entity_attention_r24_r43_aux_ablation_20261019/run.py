"""Matched R43 warm-start plain/turns-only offline fits and serial strength panels."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.offline_objectives import OfflineObjectiveConfig, freeze_inputs
from splendor.paired_statistics import paired_bootstrap_difference, paired_bootstrap_mean, serialized_pair_scores

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / ART.name
RUNS = ROOT / "runs" / ART.name
BASE_CONFIG = ROOT / "configs/entity_attention_v1/offline_r24_r43_value_auxiliary_20261013.json"
CONFIGS = {"plain": ROOT / "configs/entity_attention_v1/offline_r24_r43_no_aux_20261019.json",
           "turns_only": ROOT / "configs/entity_attention_v1/offline_r24_r43_turns_only_20261019.json"}
OLD = ROOT / "runs/entity_attention_r24_r43_value_auxiliary_offline_20261013"
OLD_GREEDY = ROOT / "artifacts/entity_attention_r24_r43_offline_objectives_20261013/greedy_vs_r43_1000pair/value_auxiliary.json"
SEARCH_TEMPLATE = ROOT / "artifacts/entity_attention_r43_search_variants_arena_20261014/configs/S1024_vs_greedy.json"
PANELS = ("greedy", "search")
child = None
stopping = False


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen input changed: {path}")
    else:
        write_json(path, value)


def prepare():
    original = read_json(BASE_CONFIG)
    branches = {}
    for name, path in CONFIGS.items():
        expected = copy.deepcopy(original)
        expected["value_auxiliary"] = None if name == "plain" else {
            **original["value_auxiliary"], "score_loss_weight": 0.0}
        if read_json(path) != expected:
            raise ValueError(f"Branch changes more than the auxiliary objective: {name}")
        config = OfflineObjectiveConfig.from_dict(expected)
        _, _, _, fixed, plan = freeze_inputs(config, RUNS / name)
        if fixed != read_json(OLD / "fixed_validation.json"):
            raise ValueError("Validation positions differ from the original dual-head run")
        if read_json(RUNS / name / "dataset.json")["games"] != read_json(OLD / "dataset.json")["games"]:
            raise ValueError("Historical pool or whole-game split differs")
        if plan["steps_target"] != 22681:
            raise ValueError("Original training budget differs")
        branches[name] = {"config": identity(path), "steps_target": plan["steps_target"],
                          "splits": plan["splits"], "output": str((RUNS / name).relative_to(ROOT))}
    plan = {"experiment": ART.name, "runner": identity(Path(__file__)), "base_config": identity(BASE_CONFIG),
            "branches": branches, "old_dual_training_metrics": identity(OLD / "training_metrics.jsonl"),
            "old_dual_full_validation": identity(OLD / "full_validation.json"),
            "old_dual_greedy": identity(OLD_GREEDY), "search_template": identity(SEARCH_TEMPLATE),
            "initial_checkpoint": identity(ROOT / original["initial_checkpoint"]),
            "old_dual_initial": identity(OLD / "checkpoints/initial.pt"),
            "selection": "final22681 only, matching the previous small-model experiment; no DEV checkpoint selection",
            "evaluation": {"greedy": {"pairs": 1000, "seed_start": 36200000, "policy_seed_base": 36210000},
                           "search": {"pairs": 200, "seed_start": 39100000, "policy_seed_base": 39110000}},
            "scope": "Two new training branches only, ordinary joint policy/value versus remaining-turns auxiliary0.25. Same R43 weights/AdamW/RNG, historical targets, pool, splits, sampling and budget as the old dual-head run. Greedy reuses the old DEV openers for paired contrasts; search uses a common new panel. No new data collection."}
    freeze(ART / "plan.json", plan)
    return plan


def evaluation_config(plan, branch, kind):
    raw = copy.deepcopy(read_json(SEARCH_TEMPLATE if kind == "search" else
        OLD_GREEDY.parent / "value_auxiliary_combined_config.json"))
    settings = plan["evaluation"][kind]
    seed = settings["seed_start"]
    candidate = RUNS / branch / "checkpoints/final.pt"
    opponent = ROOT / plan["initial_checkpoint"]["path"]
    raw.update(game_seeds=list(range(seed, seed + settings["pairs"])),
               policy_seed_base=settings["policy_seed_base"], bootstrap_seed=seed + 20000,
               bootstrap_resamples=10000)
    search_spec = copy.deepcopy(raw["policy_a"])
    for side, path, name in (("policy_a", candidate, branch), ("policy_b", opponent, "original_R43")):
        if kind == "greedy":
            raw[side] = {"kind": "greedy_checkpoint", "name": name, "checkpoint": str(path)}
        else:
            spec = copy.deepcopy(search_spec)
            spec.update(checkpoint=str(path), name=name)
            spec["search_config"].update(simulations=1024, determinizations=1,
                                         root_noise_fraction=0., seed=seed + 30000)
            raw[side] = spec
    if kind == "greedy":
        raw.update(device="cpu", workers=8)
        raw.pop("execution", None)
    raw["metadata"] = {"experiment": ART.name, "branch": branch, "kind": kind,
                       "checkpoint": identity(candidate), "opponent": identity(opponent),
                       "selection": plan["selection"], "scope": plan["scope"]}
    return FixedEvaluationConfig.from_dict(raw).to_dict()


def panel_result(branch, kind, plan):
    path = ART / "panels" / f"{branch}_{kind}.json"
    if not path.exists():
        return {"status": "pending", "completed_pairs": 0}
    raw = read_json(path)
    config = read_json(ART / "evaluation_configs" / f"{branch}_{kind}.json")
    if raw["config"] != config:
        raise ValueError("Evaluation configuration differs")
    count = plan["evaluation"][kind]["pairs"]
    if raw["completed_pairs"] != count:
        return {"status": "incomplete", "completed_pairs": raw["completed_pairs"]}
    seeds, scores = serialized_pair_scores(raw)
    ci = paired_bootstrap_mean(scores, resamples=10000, seed=config["bootstrap_seed"]).to_dict()
    if seeds != tuple(config["game_seeds"]) or ci != raw["paired_bootstrap"] or raw["valid_games"] != 2 * count:
        raise ValueError("Evaluation population or paired interval differs")
    for side, key in (("policy_a", "checkpoint"), ("policy_b", "opponent")):
        if raw["checkpoint_identity"][side]["sha256"] != config["metadata"][key]["sha256"]:
            raise ValueError("Evaluated checkpoint differs")
    if not raw["passed"]:
        raise ValueError("Evaluation reported faults")
    for pair in raw["pairs"]:
        if [g["first_player"] for g in pair["games"]] != [0, 1]:
            raise ValueError("Pair seats differ")
        for game in pair["games"]:
            for side in ("policy_a", "policy_b"):
                stats = game[f"{side}_stats"]
                if kind == "search" and (stats["simulations"] != 1024 * stats["searchable_decisions"]
                        or stats["determinizations"] != stats["searchable_decisions"]):
                    raise ValueError("Actual search budget differs")
                if kind == "greedy" and stats.get("simulations", 0):
                    raise ValueError("Greedy unexpectedly searched")
    return {"status": "complete", "completed_pairs": count, "score": raw["win_rate_score"],
            "ci95": ci, "result": identity(path), **{k: raw[k] for k in
            ("a_wins", "draws", "a_losses", "policy_cycle_draws", "engineering_truncations", "exceptions", "illegal_actions")}}


def sampling_digest(path):
    records = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "train_step":
                continue
            step, h = row["round_train_step"], row["reference_sha256"]
            if step in records and records[step] != h:
                raise ValueError("Resumed step changed its sample references")
            records[step] = h
    if set(records) != set(range(1, 22682)):
        raise ValueError("Training sample ledger is incomplete")
    return records


def report(plan):
    branches, panels = {}, {}
    for branch in CONFIGS:
        output = RUNS / branch
        progress = read_json(output / "progress.json") if (output / "progress.json").exists() else {"status": "pending", "step": 0}
        row = {"status": progress["status"], "step": progress["step"]}
        if (output / "full_validation.json").exists():
            row["full_final"] = read_json(output / "full_validation.json")
            row["checkpoint"] = identity(output / "checkpoints/final.pt")
        curve = output / "validation_curve.jsonl"
        if curve.exists():
            row["fixed_validation_curve"] = [json.loads(x) for x in curve.read_text().splitlines() if x]
        branches[branch] = row
        panels[branch] = {kind: panel_result(branch, kind, plan) for kind in PANELS}
    differences = {}
    for kind in PANELS:
        available = {b: read_json(ART / "panels" / f"{b}_{kind}.json") for b in CONFIGS
                     if panels[b][kind]["status"] == "complete"}
        if kind == "greedy":
            available["old_dual"] = read_json(OLD_GREEDY)
        for left, right in (("turns_only", "plain"), ("old_dual", "plain"), ("old_dual", "turns_only")):
            if left in available and right in available:
                ls, lv = serialized_pair_scores(available[left]); rs, rv = serialized_pair_scores(available[right])
                differences[f"{kind}/{left}_minus_{right}"] = paired_bootstrap_difference(
                    ls, lv, rs, rv, resamples=10000, seed=39140000).to_dict()
    complete = all(p["status"] == "complete" for branch in panels.values() for p in branch.values())
    result = {"status": "complete" if complete else "incomplete", "plan": identity(ART / "plan.json"),
              "training": branches, "panels": panels, "paired_common_opponent_differences": differences,
              "old_dual_full_final": read_json(OLD / "full_validation.json"), "scope": plan["scope"]}
    write_json(ART / "summary.json", result)
    lines = ["# R24–43小模型：无辅助与仅剩余回合辅助", "", f"状态：`{result['status']}`。", "",
             "同R43起点、原AdamW、20000局原split、同抽样；每组22681步，PSW关闭，评测final。", "",
             "| 分支 | 步数 | 全留出KL | 胜负MSE | 剩余回合MAE |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in branches.items():
        v = row.get("full_final")
        lines.append(f"| {name} | {row['step']} | {v['policy_kl']:.6f} | {v['value_loss']:.6f} | {v.get('remaining_turns_mae', '—')} |" if v else
                     f"| {name} | {row['step']} | — | — | — |")
    lines += ["", "| 分支 | 对战 | pair | 得分 | 95%区间 |", "| --- | --- | ---: | ---: | --- |"]
    for branch, ps in panels.items():
        for kind, p in ps.items():
            lines.append(f"| {branch} | {kind}对R43 | {p['completed_pairs']} | {p['score']:.3%} | [{p['ci95']['lower']:.3%}, {p['ci95']['upper']:.3%}] |" if p['status']=='complete' else
                         f"| {branch} | {kind}对R43 | {p['completed_pairs']} | 待完成 | — |")
    for name, d in differences.items():
        lines.append(f"\n- {name}：{100*d['estimate']:+.3f}pp，95%区间[{100*d['lower']:+.3f},{100*d['upper']:+.3f}]pp。")
    lines += ["", "差值是共同R43对手、共同开局的配对得分差，不是候选直接互打。旧双辅助greedy面板为已查看DEV的复用，未重新训练或补跑旧模型搜索。单训练seed，区间未作多重比较校正。", ""]
    (ART / "results.md").write_text("\n".join(lines))
    return result


def interrupt(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def run_child(command, phase):
    global child
    print(f"START {phase}", flush=True)
    with (RUNTIME / f"{phase}.log").open("a") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(RUNTIME / "live.json", {"status": "running", "phase": phase, "pid": os.getpid(),
                                          "child_pid": child.pid, "command": command})
        code = child.wait()
        child = None
    if stopping or code == 130:
        raise KeyboardInterrupt
    if code:
        raise subprocess.CalledProcessError(code, command)


def verify_source(source, plan):
    for path, digest in source["source_files"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Source changed during experiment: {path}")
    for path, digest in source["native_extensions"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Native extension changed: {path}")
    for item in [plan[k] for k in ("runner", "base_config", "initial_checkpoint", "old_dual_initial",
                                  "old_dual_training_metrics", "old_dual_full_validation", "old_dual_greedy", "search_template")]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen evidence changed: {item['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status"), default="run", nargs="?")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        print(report(read_json(ART / "plan.json"))["status"])
        return 0
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(); report(plan)
        if args.action == "prepare":
            print({"status": "prepared", "branches": list(CONFIGS), "steps_each": 22681})
            return 0
        signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
        session = RUNTIME / str(time.time_ns())
        source = source_identity(ROOT, session)
        record = {"status": "running", "pid": os.getpid(), "source": source,
                  "command": [sys.executable, *sys.argv], "start_unix": time.time(), "plan": identity(ART / "plan.json")}
        execution_path = ART / "execution" / f"{session.name}.json"
        write_json(execution_path, record)
        try:
            original_samples = sampling_digest(OLD / "training_metrics.jsonl")
            for branch, path in CONFIGS.items():
                if stopping: raise KeyboardInterrupt
                verify_source(source, plan)
                run_child([sys.executable, "-u", "-m", "splendor.offline_objectives", "--config", str(path),
                           "--output", str(RUNS / branch)], f"train_{branch}")
                if sampling_digest(RUNS / branch / "training_metrics.jsonl") != original_samples:
                    raise ValueError("Training did not use the exact original sample stream")
                write_json(ART / f"{branch}_sampling_verification.json", {
                    "passed": True, "committed_steps": 22681, "matches_old_dual_per_step_sample_hashes": True,
                    "metrics": identity(RUNS / branch / "training_metrics.jsonl")})
                report(plan)
            for branch in CONFIGS:
                for kind in PANELS:
                    if stopping: raise KeyboardInterrupt
                    verify_source(source, plan)
                    config_path = ART / "evaluation_configs" / f"{branch}_{kind}.json"
                    output = ART / "panels" / f"{branch}_{kind}.json"
                    freeze(config_path, evaluation_config(plan, branch, kind))
                    if panel_result(branch, kind, plan)["status"] == "complete": continue
                    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(config_path),
                               "--output", str(output), "--progress-every", "50", "--summary-every", "50"]
                    if output.exists() or Path(str(output) + ".games").exists(): command.append("--resume")
                    run_child(command, f"{branch}_{kind}"); report(plan)
                    if panel_result(branch, kind, plan)["status"] != "complete":
                        raise ValueError("Evaluation incomplete")
            verify_source(source, plan)
            record.update(status="complete", end_unix=time.time())
            return 0
        except KeyboardInterrupt:
            record.update(status="interrupted", end_unix=time.time()); return 130
        except Exception as error:
            record.update(status="failed", end_unix=time.time(), error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(execution_path, record)
            write_json(RUNTIME / "live.json", {"status": record["status"], "pid": os.getpid(),
                                              "execution": identity(execution_path)})
            report(plan)


if __name__ == "__main__":
    raise SystemExit(main())
