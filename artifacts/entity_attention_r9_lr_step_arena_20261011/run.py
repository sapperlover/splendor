"""Resumable 200-pair arena: round-nine lr2e-5/1000 vs lr1e-4/500."""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime" / ART.name
PLAN = ART / "plan.json"
CONFIG = ART / "config.json"
OUTPUT = ART / "arena.json"
LIVE = RUNTIME / "live.json"
SOURCE = ROOT / "artifacts/entity_attention_r9_ofat_20261010/summary.json"
TEMPLATE = ROOT / "artifacts/entity_attention_p4_p9_crossplay_20261010/configs/S01_newP9_oldV4.json"


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment changed: {path}")
    else:
        write_json(path, value)


def prepare():
    source = read_json(SOURCE)
    a = next(p for p in source["panels"] if p["id"] == "lr_2e-05")
    a = next(p for p in a["points"] if p["step"] == 1000)
    b = next(p for p in source["baseline"] if p["step"] == 500)
    config = read_json(TEMPLATE)
    identities = {}
    for side, name, point, lr in (("policy_a", "lr2e-5_step1000", a, 2e-5),
                                  ("policy_b", "lr1e-4_step500", b, 1e-4)):
        checkpoint = ROOT / point["checkpoint"]["path"]
        if sha256(checkpoint) != point["checkpoint"]["sha256"]:
            raise ValueError(f"Checkpoint changed: {side}")
        config[side].update(checkpoint=str(checkpoint), name=name)
        config[side]["search_config"]["seed"] = 31130000
        identities[side] = {"name": name, "checkpoint": point["checkpoint"],
                            "learning_rate": lr, "round_steps": point["step"],
                            "global_step": 38398 + point["step"],
                            "validation_policy_kl": point["policy_kl"],
                            "validation_value_mse": point["value_mse"]}
    config.update(game_seeds=list(range(31100000, 31100200)), policy_seed_base=31110000,
                  bootstrap_seed=31120000, metadata={"experiment": ART.name,
                  "scope": "Direct complete-network arena; fresh 200 paired openers; no training",
                  "models": identities})
    config = FixedEvaluationConfig.from_dict(config).to_dict()
    freeze(CONFIG, config)
    plan = {"experiment": ART.name, "models": identities,
            "config": {"path": str(CONFIG.relative_to(ROOT)), "sha256": sha256(CONFIG)},
            "training_results": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
            "settings_template": {"path": str(TEMPLATE.relative_to(ROOT)), "sha256": sha256(TEMPLATE)},
            "runner_sha256": sha256(Path(__file__)), "pairs": 200, "games": 400,
            "settings": "Both S1024/N1, native_batched, compiled, active100, max_batch64, no root noise; seat swap per seed",
            "scope": "Same round-eight start, AdamW, 6k pool and B256; learning rate AND update count differ. Measures these complete checkpoints, not an isolated learning-rate effect. Single training trajectory; fresh arena seeds, pair-bootstrap 95% CI."}
    freeze(PLAN, plan)
    return plan


def aggregate(plan):
    result = {"status": "pending", "completed_pairs": 0, "total_pairs": 200,
              "plan_sha256": sha256(PLAN), "scope": plan["scope"]}
    if OUTPUT.exists():
        raw = read_json(OUTPUT)
        if raw["config"] != read_json(CONFIG):
            raise ValueError("Saved arena config differs")
        for side, model in plan["models"].items():
            if raw["checkpoint_identity"][side]["sha256"] != model["checkpoint"]["sha256"]:
                raise ValueError(f"Evaluated checkpoint differs: {side}")
        keys = ("completed_pairs", "completed_games", "valid_games", "a_wins", "draws", "a_losses",
                "win_rate_score", "paired_bootstrap", "engineering_truncations", "exceptions",
                "illegal_actions", "invariant_failures", "policy_cycle_draws", "a_first", "a_second",
                "elapsed_seconds_this_invocation")
        result.update({key: raw[key] for key in keys})
        result["status"] = "complete" if raw["passed"] else "incomplete"
        if raw["completed_pairs"] == 200:
            seeds, scores = serialized_pair_scores(raw)
            config = read_json(CONFIG)
            if list(seeds) != config["game_seeds"] or raw["completed_games"] != 400:
                raise ValueError("Arena population differs")
            interval = paired_bootstrap_mean(scores, resamples=config["bootstrap_resamples"],
                                             seed=config["bootstrap_seed"]).to_dict()
            if interval != raw["paired_bootstrap"]:
                raise ValueError("Paired confidence interval differs")
            for pair in raw["pairs"]:
                for game in pair["games"]:
                    for side in ("policy_a_stats", "policy_b_stats"):
                        stats = game[side]
                        if (stats["simulations"] != 1024 * stats["searchable_decisions"]
                                or stats["determinizations"] != stats["searchable_decisions"]):
                            raise ValueError("Actual search budget differs from S1024/N1")
            result["verification"] = "400 valid games, seeds, checkpoint identities, actual S1024/N1 budgets and pair-bootstrap independently checked"
        result["arena"] = {"path": str(OUTPUT.relative_to(ROOT)), "sha256": sha256(OUTPUT)}
    write_json(ART / "summary.json", result)
    lines = ["# lr2e-5 × 1000步 vs lr1e-4 × 500步", "",
             f"状态：`{result['status']}`；完成{result['completed_pairs']}/200 pair。", "",
             "A：2e-5训练1000步；B：1e-4训练500步。共同第八轮终点，6k在线池、B256；均使用完整网络做S1024/N1搜索。每个新开局种子交换先后手，共400局，和棋计半分。", ""]
    if result["status"] == "complete":
        ci = result["paired_bootstrap"]
        lines += [f"A胜／和／负：{result['a_wins']}/{result['draws']}/{result['a_losses']}。",
                  f"A得分：**{100*result['win_rate_score']:.3f}%**；按整pair bootstrap的95% CI：**[{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%]**。", "",
                  f"A先手得分：{100*result['a_first']['win_rate_score']:.3f}%；A后手得分：{100*result['a_second']['win_rate_score']:.3f}%。", "",
                  "400局均为有效完局，checkpoint、实际搜索预算及区间复算通过；异常／非法动作／工程截断均为0。", ""]
    lines += ["两个配置同时改变学习率和更新步数，结果只比较这两个完整checkpoint。单条训练轨迹与一个200-pair面板，不能据此确定全局最优参数。", "",
              "逐局配对结果见[arena.json](arena.json)，配置见[config.json](config.json)，身份见[plan.json](plan.json)。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze", "status", "stop"), nargs="?", default="run")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "stop":
        live = read_json(LIVE)
        if live["status"] == "running":
            command = Path(f"/proc/{live['pid']}/cmdline").read_bytes().split(b"\0")
            if not any(item and Path(os.fsdecode(item)).resolve() == Path(__file__) for item in command):
                raise ValueError("Recorded PID is not this runner")
            os.kill(live["pid"], signal.SIGINT)
        return 0
    with (RUNTIME / "run.lock").open("a") as lock:
        if args.action != "status":
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        plan = prepare()
        if args.action != "run":
            result = aggregate(plan)
            print({"status": result["status"], "completed_pairs": result["completed_pairs"]}, flush=True)
            return 0
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen arena config")
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        session = RUNTIME / str(time.time_ns())
        identity = source_identity(ROOT, session)
        for path in (Path(__file__), CONFIG, PLAN):
            shutil.copyfile(path, session / path.name)
        execution_path = ART / "execution.json"
        execution = read_json(execution_path) if execution_path.exists() else {"sessions": []}
        execution["sessions"].append({**identity, "environment": environment_identity(),
            "command": [sys.executable, *sys.argv], "session": str(session.relative_to(ROOT)),
            "start_unix": time.time(), "plan_sha256": sha256(PLAN)})
        write_json(execution_path, execution)
        write_json(LIVE, {"status": "running", "pid": os.getpid(), "time_unix": time.time()})
        try:
            config = FixedEvaluationConfig.from_dict(read_json(CONFIG))
            run_fixed_evaluation(config, OUTPUT,
                resume=OUTPUT.exists() or OUTPUT.with_suffix(".json.games").exists(),
                progress_every=10, summary_every=10,
                progress_callback=lambda done, total: print({"completed_pairs": done, "total_pairs": total}, flush=True),
                failure_dir=RUNTIME / "failures")
            result = aggregate(plan)
            if result["status"] != "complete":
                raise ValueError("Arena did not complete all 200 valid pairs")
            write_json(LIVE, {"status": "complete", "pid": os.getpid(), "time_unix": time.time()})
            print(result, flush=True)
            return 0
        except KeyboardInterrupt:
            aggregate(plan)
            write_json(LIVE, {"status": "interrupted", "pid": os.getpid(), "time_unix": time.time()})
            return 130
        except Exception as error:
            write_json(LIVE, {"status": "failed", "pid": os.getpid(), "error": str(error), "time_unix": time.time()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
