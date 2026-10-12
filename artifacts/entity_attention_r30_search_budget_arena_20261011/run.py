"""Round30: S1024 versus greedy, then S4096 versus S1024; resumable."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime" / ART.name
PLAN = ART / "plan.json"
LIVE = RUNTIME / "live.json"
CHECKPOINT = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-020/checkpoints/final.pt"
CHECKPOINT_SHA = "9fc258b211954cf32cd7b688d5d534a502441304ea7cc92a7a35e059a7c91b0f"
TEMPLATE = ROOT / "artifacts/entity_attention_r30_value_weight_20261011/baseline/evaluation/search.json"
PANELS = (
    ("S01_S1024_vs_greedy", "第30轮 S1024/N1 vs 无搜索greedy", 1024, None, 32900000),
    ("S02_S4096_vs_S1024", "第30轮 S4096/N1 vs S1024/N1", 4096, 1024, 33000000),
)
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
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Original round30 checkpoint changed")
    source = identity(CHECKPOINT)
    template = read_json(TEMPLATE)["config"]
    panels = []
    for name, label, a_budget, b_budget, seed in PANELS:
        config = copy.deepcopy(template)
        config.update(game_seeds=list(range(seed, seed + 200)), policy_seed_base=seed + 10000,
                      bootstrap_seed=seed + 20000)
        for side, budget in (("policy_a", a_budget), ("policy_b", b_budget)):
            if budget is None:
                config[side] = {"kind": "greedy_checkpoint", "name": "round30_greedy",
                                "checkpoint": str(CHECKPOINT),
                                "inference": copy.deepcopy(config["policy_a"]["inference"])}
            else:
                config[side].update(checkpoint=str(CHECKPOINT), name=f"round30_S{budget}_N1")
                config[side]["search_config"].update(simulations=budget, determinizations=1,
                                                      seed=seed + 30000)
        config["metadata"] = {"experiment": ART.name, "panel": name, "model": source,
            "absolute_round": 30, "global_step": 59898,
            "scope": "Both sides use the same original round30 complete network; only search budget changes"}
        config = FixedEvaluationConfig.from_dict(config).to_dict()
        path = ART / "configs" / f"{name}.json"
        freeze(path, config)
        panels.append({"id": name, "label": label, "pairs": 200, "config": identity(path),
            "output": str((ART / "panels" / f"{name}.json").relative_to(ROOT))})
    plan = {"experiment": ART.name, "model": source, "absolute_round": 30, "global_step": 59898,
        "panels": panels, "runner": identity(Path(__file__)), "template": identity(TEMPLATE),
        "settings": "N1, c_puct1.5, no root noise, native_batched/CUDA compiled search, active100, max_batch64; greedy argmax without search; separate sequential subprocesses",
        "scope": "Two direct 200-pair seat-swapped arenas on fresh separate opener banks, using the original weight1 round30 complete network on every side. S01 measures S1024 search gain over greedy; S02 measures additional gain from S4096 over S1024. Draws score one half; 10000 whole-pair bootstrap resamples. One checkpoint and fixed DEV panels, no multiplicity correction."}
    freeze(PLAN, plan)
    return plan


def summarize(panel, model):
    path = ROOT / panel["output"]
    row = {"id": panel["id"], "label": panel["label"], "status": "pending", "completed_pairs": 0}
    if not path.exists():
        return row
    raw = read_json(path)
    config = read_json(ROOT / panel["config"]["path"])
    if raw["config"] != config:
        raise ValueError("Saved arena configuration differs")
    row.update(status="incomplete", completed_pairs=raw["completed_pairs"])
    if raw["completed_pairs"] != panel["pairs"]:
        return row
    seeds, scores = serialized_pair_scores(raw)
    if seeds != tuple(config["game_seeds"]) or raw["valid_games"] != 2 * panel["pairs"]:
        raise ValueError("Completed arena population differs")
    for side in ("policy_a", "policy_b"):
        if raw["checkpoint_identity"][side]["sha256"] != model["sha256"]:
            raise ValueError("Evaluated checkpoint differs")
    ci = paired_bootstrap_mean(scores, resamples=config["bootstrap_resamples"],
                               seed=config["bootstrap_seed"]).to_dict()
    if ci != raw["paired_bootstrap"]:
        raise ValueError("Paired confidence interval differs")
    for pair in raw["pairs"]:
        for game in pair["games"]:
            for side in ("policy_a", "policy_b"):
                spec, stats = config[side], game[f"{side}_stats"]
                if spec["kind"] == "greedy_checkpoint":
                    if stats.get("simulations", 0) or stats.get("determinizations", 0):
                        raise ValueError("Greedy side unexpectedly searched")
                else:
                    search = spec["search_config"]
                    if (stats["simulations"] != search["simulations"] * stats["searchable_decisions"]
                            or stats["determinizations"] != search["determinizations"] * stats["searchable_decisions"]):
                        raise ValueError("Actual search budget differs from side configuration")
    row.update(status="complete", score=raw["win_rate_score"], ci95=ci, result=identity(path),
        **{key: raw[key] for key in ("a_wins", "draws", "a_losses", "a_first", "a_second",
            "exceptions", "illegal_actions", "engineering_truncations", "invariant_failures",
            "elapsed_seconds_this_invocation")})
    return row


def aggregate(plan):
    rows = [summarize(panel, plan["model"]) for panel in plan["panels"]]
    count = sum(row["status"] == "complete" for row in rows)
    result = {"status": "complete" if count == len(rows) else "incomplete", "completed_panels": count,
              "plan_sha256": sha256(PLAN), "panels": rows, "scope": plan["scope"]}
    write_json(ART / "summary.json", result)
    lines = ["# 原第30轮搜索预算对照", "", f"状态：`{result['status']}`；完成{count}/2组。", "",
        "每组200 pair／400局，交换先后手、和棋半分，95% CI按整pair bootstrap 10000次。所有侧均使用原配置第30轮完整网络（global 59898）。得分按左侧A计算。", "",
        "| 顺序／组合 | 完整pair | 胜／和／负 | A得分 | 95% CI |",
        "| --- | ---: | --- | ---: | --- |"]
    for row in rows:
        if row["status"] == "complete":
            ci = row["ci95"]
            lines.append(f"| {row['label']} | 200 | {row['a_wins']}/{row['draws']}/{row['a_losses']} | {100*row['score']:.3f}% | [{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%] |")
        else:
            lines.append(f"| {row['label']} | {row['completed_pairs']} | — | — | — |")
    lines += ["", "第一组检验S1024相对greedy的搜索收益；第二组检验增加到S4096的边际收益。两组各自直接对战，使用独立的新开局面板，不计算跨组配对差值。区间包含50%时，200pair不足以确认该预算差的正负；不能据此断言两者等强。", "",
        "结果限定本checkpoint、N1和本次固定DEV面板；未校正多重比较。评测每10pair保存一次汇总，表格在每组结束／链路退出时更新；逐pair持久化文件用于断点续跑。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    index_path = ROOT / "artifacts/experiment_index.json"
    index = read_json(index_path)
    entry_id = "entity-attention-r30-search-budget-arena-2026-10-11"
    entry = next((e for e in index["entries"] if e["id"] == entry_id), None)
    if entry is None:
        entry = {"id": entry_id}
        index["entries"].append(entry)
    evidence = sorted([*ART.glob("*.json"), *ART.glob("*.py"), *ART.glob("*.md"),
                       *ART.glob("configs/*.json"), *ART.glob("panels/*.json")])
    entry.update(status="valid" if result["status"] == "complete" else "incomplete",
        report=str((ART / "results.md").relative_to(ROOT)), scope=plan["scope"],
        evidence=[str(p.relative_to(ROOT)) for p in evidence],
        evidence_hashes={str(p.relative_to(ROOT)): sha256(p) for p in evidence})
    write_json(index_path, index)
    return result


def interrupt(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def main():
    global child
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"), default="run", nargs="?")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        try:
            plan = prepare()
            result = aggregate(plan)
            if args.action != "run":
                print({"status": result["status"], "completed_panels": result["completed_panels"]}, flush=True)
                return 0
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for the frozen search configuration")
            session = RUNTIME / str(time.time_ns())
            source = source_identity(ROOT, session)
            for path in (Path(__file__), PLAN, *sorted((ART / "configs").glob("*.json"))):
                target = session / path.relative_to(ART)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            record = ART / "execution.json"
            execution = read_json(record) if record.exists() else {"sessions": []}
            execution["sessions"].append({**source, "environment": environment_identity(),
                "command": [sys.executable, *sys.argv], "runtime": str(session.relative_to(ROOT)),
                "start_unix": time.time(), "plan_sha256": sha256(PLAN)})
            write_json(record, execution)
            for panel in plan["panels"]:
                if stopping:
                    break
                if summarize(panel, plan["model"])["status"] == "complete":
                    continue
                if sha256(ROOT / panel["config"]["path"]) != panel["config"]["sha256"]:
                    raise ValueError("Frozen panel configuration changed")
                output = ROOT / panel["output"]
                command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config",
                    str(ROOT / panel["config"]["path"]), "--output", str(output),
                    "--progress-every", "10", "--summary-every", "10"]
                if output.exists() or output.with_suffix(".json.games").exists():
                    command.append("--resume")
                write_json(session / f"command-{panel['id']}.json", {"command": command, "time_unix": time.time()})
                log_path = RUNTIME / f"{panel['id']}.log"
                print(f"START {panel['id']}: 200 pairs", flush=True)
                with log_path.open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    write_json(LIVE, {"status": "running", "pid": os.getpid(), "child_pid": child.pid,
                        "current_panel": panel["id"], "log": str(log_path.relative_to(ROOT)), "time_unix": time.time()})
                    code = child.wait()
                    child = None
                result = aggregate(plan)
                if stopping or code == 130:
                    break
                if code:
                    raise RuntimeError(f"Evaluation exited {code}: {log_path}")
                row = next(row for row in result["panels"] if row["id"] == panel["id"])
                if row["status"] != "complete":
                    raise ValueError("Evaluator exited before completing 200 valid pairs")
                print(f"DONE {panel['id']}: score={row['score']:.5f}", flush=True)
            result = aggregate(plan)
            status = "complete" if result["status"] == "complete" else "interrupted"
            write_json(LIVE, {"status": status, "pid": os.getpid(), "time_unix": time.time()})
            return 0 if status == "complete" else 130
        except Exception as error:
            write_json(LIVE, {"status": "failed", "pid": os.getpid(),
                "error": f"{type(error).__name__}: {error}", "time_unix": time.time()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
