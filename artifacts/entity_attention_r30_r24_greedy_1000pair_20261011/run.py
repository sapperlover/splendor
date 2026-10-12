"""Original round30 versus round24, greedy only, 1000 paired seeds."""
from __future__ import annotations

import argparse
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
CONFIG = ART / "config.json"
PLAN = ART / "plan.json"
OUTPUT = ART / "evaluation.json"
LIVE = RUNTIME / "live.json"
SOURCES = {
    "policy_a": ("runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-020/checkpoints/final.pt",
                 "9fc258b211954cf32cd7b688d5d534a502441304ea7cc92a7a35e059a7c91b0f"),
    "policy_b": ("runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-014/checkpoints/final.pt",
                 "15424dc571c15dcbff0aeb3d6e698cdbea952583eb482f6f5826fc23b00f5bdb"),
}
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
    models = {}
    for side, (relative, expected) in SOURCES.items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise ValueError(f"Source checkpoint changed: {side}")
        models[side] = identity(path)
    config = {side: {"kind": "greedy_checkpoint", "name": name,
                    "checkpoint": str(ROOT / SOURCES[side][0])}
              for side, name in (("policy_a", "original_round30"), ("policy_b", "round24"))}
    config.update(game_seeds=list(range(33300000, 33301000)), policy_seed_base=33310000,
        workers=8, device="cpu", max_completed_turns=400,
        adjudicate_deterministic_policy_cycles=True, bootstrap_resamples=10000,
        bootstrap_seed=33320000, metadata={"experiment": ART.name, "models": models,
            "scope": "1000 fresh paired seeds, original round30(A) vs round24(B), greedy argmax only; no training or search"})
    freeze(CONFIG, FixedEvaluationConfig.from_dict(config).to_dict())
    plan = {"experiment": ART.name, "models": models, "config": identity(CONFIG),
        "runner": identity(Path(__file__)), "pairs": 1000, "games": 2000,
        "absolute_rounds": {"policy_a": 30, "policy_b": 24},
        "global_steps": {"policy_a": 59898, "policy_b": 53898},
        "scope": "Direct greedy matchup on fresh independent opener seeds; score is for original round30. No search, no value-weight branches or model compositions. Paired CI covers opener variation for these two checkpoints, not retraining variability."}
    freeze(PLAN, plan)
    return plan


def analyze(plan):
    result = {"status": "pending", "completed_pairs": 0, "total_pairs": 1000,
              "plan_sha256": sha256(PLAN), "scope": plan["scope"]}
    if OUTPUT.exists():
        raw = read_json(OUTPUT)
        config = read_json(CONFIG)
        if raw["config"] != config:
            raise ValueError("Saved evaluation configuration differs")
        result.update(status="incomplete", completed_pairs=raw["completed_pairs"])
        if raw["completed_pairs"] == 1000:
            seeds, scores = serialized_pair_scores(raw)
            if seeds != tuple(config["game_seeds"]) or raw["valid_games"] != 2000:
                raise ValueError("Evaluation population differs")
            for side in SOURCES:
                if raw["checkpoint_identity"][side]["sha256"] != plan["models"][side]["sha256"]:
                    raise ValueError("Evaluated checkpoint differs")
            for pair in raw["pairs"]:
                for game in pair["games"]:
                    for side in SOURCES:
                        stats = game[f"{side}_stats"]
                        if stats.get("simulations", 0) or stats.get("determinizations", 0):
                            raise ValueError("Greedy unexpectedly searched")
            ci = paired_bootstrap_mean(scores, resamples=config["bootstrap_resamples"],
                                       seed=config["bootstrap_seed"]).to_dict()
            if ci != raw["paired_bootstrap"]:
                raise ValueError("Paired confidence interval differs")
            result.update(status="complete", score=raw["win_rate_score"], ci95=ci,
                result=identity(OUTPUT), **{k: raw[k] for k in ("a_wins", "draws", "a_losses",
                    "a_first", "a_second", "exceptions", "illegal_actions", "invariant_failures",
                    "engineering_truncations", "policy_cycle_draws", "elapsed_seconds_this_invocation")})
    write_json(ART / "summary.json", result)
    lines = ["# 原第30轮 vs 第24轮：1000pair greedy直接对战", "",
        f"状态：`{result['status']}`；完整pair {result['completed_pairs']}/1000。", "",
        "双方均为合法动作logits的argmax，无搜索。A为原配置第30轮（global 59898），B为第24轮（global 53898）。每个开局交换先后手，和棋半分，95% CI按整pair bootstrap 10000次。", ""]
    if result["status"] == "complete":
        ci = result["ci95"]
        lines += ["| A胜／和／负 | 第30轮得分率 | 95% CI | 本次调用耗时 |",
                  "| --- | ---: | --- | ---: |",
                  f"| {result['a_wins']}/{result['draws']}/{result['a_losses']} | {100*result['score']:.3f}% | [{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%] | {result['elapsed_seconds_this_invocation']:.2f}秒 |", "",
                  f"先手得分{100*result['a_first']['win_rate_score']:.3f}%，后手得分{100*result['a_second']['win_rate_score']:.3f}%。", ""]
        conclusion = ("本面板支持第30轮greedy强于第24轮。" if ci["lower"] > .5 else
                      "本面板支持第30轮greedy弱于第24轮。" if ci["upper"] < .5 else
                      "本面板尚未确认第30轮与第24轮greedy的强弱，区间跨50%；不代表严格等强。")
        lines += [conclusion, "",
            f"异常{result['exceptions']}、非法动作{result['illegal_actions']}、不变量失败{result['invariant_failures']}、工程截断{result['engineering_truncations']}、循环裁和{result['policy_cycle_draws']}。模型SHA、2000有效局、双方零搜索及CI已核对。", ""]
    lines += ["greedy结果反映policy及其encoder的整局表现，不使用value决定动作。与已有S1024的24轮对30轮200pair使用不同开局面板，不能做逐pair差值，也不能仅凭两种对战结果将差异归因于value。", "",
              "限定这两个checkpoint与本次开局面板，不覆盖训练种子变异。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    index_path = ROOT / "artifacts/experiment_index.json"
    index = read_json(index_path)
    entry_id = "entity-attention-r30-r24-greedy-1000pair-2026-10-11"
    entry = next((e for e in index["entries"] if e["id"] == entry_id), None)
    if entry is None:
        entry = {"id": entry_id}
        index["entries"].append(entry)
    evidence = sorted([*ART.glob("*.json"), *ART.glob("*.py"), *ART.glob("*.md")])
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
    parser.add_argument("action", choices=("prepare", "run", "analyze"), nargs="?", default="run")
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
            result = analyze(plan)
            if args.action != "run" or result["status"] == "complete":
                print(result, flush=True)
                return 0
            session = RUNTIME / str(time.time_ns())
            source = source_identity(ROOT, session)
            for path in (Path(__file__), CONFIG, PLAN):
                shutil.copyfile(path, session / path.name)
            command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(CONFIG),
                       "--output", str(OUTPUT), "--progress-every", "100", "--summary-every", "100"]
            if OUTPUT.exists() or OUTPUT.with_suffix(".json.games").exists():
                command.append("--resume")
            record = ART / "execution.json"
            execution = read_json(record) if record.exists() else {"sessions": []}
            execution["sessions"].append({**source, "environment": environment_identity(),
                "command": [sys.executable, *sys.argv], "evaluation_command": command,
                "runtime": str(session.relative_to(ROOT)), "start_unix": time.time(),
                "plan_sha256": sha256(PLAN)})
            write_json(record, execution)
            if stopping:
                return 130
            with (RUNTIME / "evaluation.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                write_json(LIVE, {"status": "running", "pid": os.getpid(), "child_pid": child.pid,
                                  "time_unix": time.time()})
                print("START original round30 vs round24 greedy: 1000 pairs", flush=True)
                code = child.wait()
                child = None
            result = analyze(plan)
            status = "complete" if result["status"] == "complete" else "interrupted"
            if code and code != 130 and not stopping:
                raise RuntimeError(f"Evaluator exited {code}; see {RUNTIME / 'evaluation.log'}")
            if not code and status != "complete":
                raise ValueError("Evaluator exited before completing 1000 valid pairs")
            write_json(LIVE, {"status": status, "pid": os.getpid(), "time_unix": time.time()})
            print(result, flush=True)
            return 0 if status == "complete" else 130
        except Exception as error:
            write_json(LIVE, {"status": "failed", "pid": os.getpid(),
                             "error": f"{type(error).__name__}: {error}", "time_unix": time.time()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
