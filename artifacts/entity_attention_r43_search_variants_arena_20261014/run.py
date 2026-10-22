"""Original R43: native S1024/S4096, chance S1024, CPW S1024 versus greedy; serial queue."""
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
from splendor.paired_statistics import paired_bootstrap_difference, paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime" / ART.name
PLAN = ART / "plan.json"
CHECKPOINT = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt"
CHECKPOINT_SHA = "b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14"
TEMPLATE = ROOT / "artifacts/entity_attention_r43_low_search_budget_arena_20261014/configs/S128_vs_greedy.json"
PAIRS = 500
SEED = 36300000
PANELS = (
    ("S1024_vs_greedy", "确定化 S1024", 1024, "native", None),
    ("S4096_vs_greedy", "确定化 S4096", 4096, "native", None),
    ("S1024_chance_vs_greedy", "随机节点 S1024", 1024, "native_chance", None),
    ("S1024_cpw_vs_greedy", "CPW S1024", 1024, "native_chance", {"k": 0.25, "alpha": 0.5}),
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
        raise ValueError("Original R43 checkpoint changed")
    model = identity(CHECKPOINT)
    panels = []
    for name, label, budget, backend, widening in PANELS:
        config = copy.deepcopy(read_json(TEMPLATE))
        config.update(game_seeds=list(range(SEED, SEED + PAIRS)), policy_seed_base=SEED + 10000,
                      bootstrap_seed=SEED + 20000)
        for side in ("policy_a", "policy_b"):
            config[side].update(checkpoint=str(CHECKPOINT),
                name=f"R43_{name}" if side == "policy_a" else "R43_greedy")
        config["policy_a"]["search_backend"] = backend
        if widening is not None:
            config["policy_a"]["chance_widening"] = copy.deepcopy(widening)
        config["policy_a"]["search_config"].update(simulations=budget, determinizations=1,
            root_noise_fraction=0.0, seed=SEED + 30000)
        config["metadata"] = {"experiment": ART.name, "panel": name, "model": model,
            "absolute_round": 43, "global_step": 72898,
            "scope": "Original R43 complete network on both sides; search versus greedy; shared seeds with previous S128/S256/S512 panels",
            "search_variant": label}
        config = FixedEvaluationConfig.from_dict(config).to_dict()
        path = ART / "configs" / f"{name}.json"
        freeze(path, config)
        panels.append({"id": name, "label": label, "budget": budget, "backend": backend,
            "chance_widening": widening, "pairs": PAIRS, "config": identity(path),
            "output": str((ART / "panels" / f"{name}.json").relative_to(ROOT))})
    plan = {"experiment": ART.name, "model": model, "absolute_round": 43, "global_step": 72898,
        "panels": panels, "runner": identity(Path(__file__)), "template": identity(TEMPLATE),
        "difference_bootstrap_seed": SEED + 40000,
        "settings": "N1, c_puct1.5, no root noise, native_batched/CUDA compiled, active100, max_batch64; greedy argmax; sequential subprocesses",
        "scope": "Four serial 500-pair seat-swapped arenas: native S1024, native S4096, native_chance S1024 without widening, native_chance S1024 with CPW k0.25/alpha0.5, all versus greedy using the original R43 complete network. Shared opening/policy/search seeds match previous S128/S256/S512 panels. N1 is one fixed world for native and one shared statistics pool with per-simulation sampling for native_chance. Draws half score; 10000 whole-pair bootstrap resamples, unadjusted 95% intervals. Cross-panel differences compare scores versus common greedy, not direct search-versus-search games."}
    freeze(PLAN, plan)
    return plan


def summarize(panel, model):
    path = ROOT / panel["output"]
    row = {"id": panel["id"], "label": panel["label"], "budget": panel["budget"],
        "backend": panel["backend"], "chance_widening": panel["chance_widening"],
        "status": "pending", "completed_pairs": 0}
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
    if seeds != tuple(config["game_seeds"]) or raw["valid_games"] != 2 * panel["pairs"] or not raw["passed"]:
        raise ValueError("Completed arena population or validity differs")
    for side in ("policy_a", "policy_b"):
        if raw["checkpoint_identity"][side]["sha256"] != model["sha256"]:
            raise ValueError("Evaluated checkpoint differs")
    ci = paired_bootstrap_mean(scores, resamples=config["bootstrap_resamples"],
                               seed=config["bootstrap_seed"]).to_dict()
    if ci != raw["paired_bootstrap"]:
        raise ValueError("Paired confidence interval differs")
    for pair in raw["pairs"]:
        if [game["first_player"] for game in pair["games"]] != [0, 1]:
            raise ValueError("Pair seats were not swapped")
        for game in pair["games"]:
            i = pair["pair_id"]
            if game["policy_seeds"] != [config["policy_seed_base"] + 2*i,
                                        config["policy_seed_base"] + 2*i + 1]:
                raise ValueError("Policy seed schedule differs")
            a, b = game["policy_a_stats"], game["policy_b_stats"]
            if (a["simulations"] != panel["budget"] * a["searchable_decisions"]
                    or a["determinizations"] != a["searchable_decisions"]):
                raise ValueError("Actual N1 search budget differs")
            if b.get("simulations", 0) or b.get("determinizations", 0):
                raise ValueError("Greedy side unexpectedly searched")
    row.update(status="complete", score=raw["win_rate_score"], ci95=ci, result=identity(path),
        **{key: raw[key] for key in ("a_wins", "draws", "a_losses", "a_first", "a_second",
            "exceptions", "illegal_actions", "engineering_truncations", "invariant_failures",
            "policy_cycle_draws", "elapsed_seconds_this_invocation", "policy_a_stats", "policy_b_stats")})
    return row


def aggregate(plan):
    rows = [summarize(panel, plan["model"]) for panel in plan["panels"]]
    differences = []
    for left, right in ((1, 0), (2, 0), (3, 0), (3, 2)):
        if rows[left]["status"] == rows[right]["status"] == "complete":
            ls, lv = serialized_pair_scores(read_json(ROOT / plan["panels"][left]["output"]))
            rs, rv = serialized_pair_scores(read_json(ROOT / plan["panels"][right]["output"]))
            differences.append({"left": rows[left]["id"], "right": rows[right]["id"],
                "ci95": paired_bootstrap_difference(ls, lv, rs, rv, resamples=10000,
                    seed=plan["difference_bootstrap_seed"]).to_dict()})
    count = sum(row["status"] == "complete" for row in rows)
    result = {"status": "complete" if count == len(rows) else "incomplete", "completed_panels": count,
        "plan_sha256": sha256(PLAN), "panels": rows, "common_opponent_differences": differences,
        "scope": plan["scope"], "verification": "Every complete panel: 1000 valid games, checkpoint SHA, paired CI, per-game exact search budget and greedy zero search checked."}
    write_json(ART / "summary.json", result)
    lines = ["# R43 高预算与随机节点／CPW对 greedy", "", f"状态：`{result['status']}`；完成 {count}/4 组。", "",
        "每组 500 pair／1000 局，同开局交换先后手，四组共用开局种子，与上一批 S128–S512 相同。双方均用原 R43 完整网络（global 72898）；A 按表搜索，B 为 greedy。c_puct=1.5、无根噪声、T0；和棋半分，95% CI 按整 pair bootstrap 10000 次。", "",
        "native 的 N1 为一个确定化世界；native_chance 的 N1 为一个共享统计池，仍按 simulation 重采隐藏信息／随机转移。随机节点组不扩宽，CPW 为 k=0.25、α=0.5。", "",
        "| 搜索配置 | 完整 pair | 胜／和／负 | 搜索方得分 | 95% CI |", "| --- | ---: | --- | ---: | --- |"]
    for row in rows:
        if row["status"] == "complete":
            ci = row["ci95"]
            lines.append(f"| {row['label']} | {PAIRS} | {row['a_wins']}/{row['draws']}/{row['a_losses']} | {100*row['score']:.3f}% | [{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%] |")
        else:
            lines.append(f"| {row['label']} | {row['completed_pairs']} | — | — | — |")
    if differences:
        lines += ["", "共同 greedy 对手下的配对得分差（百分点），不是不同搜索预算之间的直接对战胜率：", "",
            "| 得分差 | 点估计 | 95% CI |", "| --- | ---: | --- |"]
        for diff in differences:
            ci = diff["ci95"]
            lines.append(f"| {diff['left']} − {diff['right']} | {100*ci['estimate']:+.3f} | [{100*ci['lower']:+.3f}, {100*ci['upper']:+.3f}] |")
    lines += ["", "质量核验：逐局检查搜索预算与 N1 世界数、greedy 无搜索、checkpoint SHA 及配对区间。", ""]
    for row in rows:
        if row["status"] == "complete":
            lines.append(f"- {row['label']}：异常 {row['exceptions']}，非法行动 {row['illegal_actions']}，工程截断 {row['engineering_truncations']}，不变量失败 {row['invariant_failures']}，策略循环判和 {row['policy_cycle_draws']}；本次 evaluator 耗时 {row['elapsed_seconds_this_invocation']:.1f} 秒。")
    complete = [row for row in rows if row["status"] == "complete"]
    if complete:
        lines += ["", "以下搜索统计来自各组实际经过的不同状态，不能单独作为棋力差异的因果解释。逻辑深度不计折叠 forced 动作。", "",
            "| 搜索配置 | 总 simulation | 平均逻辑深度 | 精确终局命中比例 | transition cap命中 |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for row in complete:
            stats = row["policy_a_stats"]
            sims = stats["simulations"]
            lines.append(f"| {row['label']} | {sims} | {stats['tree_depth_sum']/sims:.3f} | {100*stats['terminal_hits']/sims:.4f}% | {stats['transition_cap_hits']} |")
    lines += ["", "这些结果限定本 R43、N1 和固定开局面板；区间未做多重比较校正。比赛衡量搜索收益，不直接证明对应预算用于自对弈采集后的训练收益或最终迭代效率。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    update_index(result, plan)
    return result


def update_index(result, plan):
    path = ROOT / "artifacts/experiment_index.json"
    index = read_json(path)
    entry_id = "entity-attention-r43-search-variants-arena-2026-10-14"
    entry = next((item for item in index["entries"] if item["id"] == entry_id), None)
    if entry is None:
        entry = {"id": entry_id}
        index["entries"].append(entry)
    files = sorted([*ART.glob("*.json"), *ART.glob("*.py"), *ART.glob("*.md"),
                    *ART.glob("configs/*.json"), *ART.glob("panels/*.json")])
    entry.update(status="valid" if result["status"] == "complete" else "incomplete",
        report=str((ART / "results.md").relative_to(ROOT)), scope=plan["scope"],
        evidence=[str(p.relative_to(ROOT)) for p in files],
        evidence_hashes={str(p.relative_to(ROOT)): sha256(p) for p in files})
    write_json(path, index)


def verify_source(source):
    for path, expected in source["source_files"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Production source changed during the frozen queue: {path}")
    for path, expected in source["native_extensions"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Native extension changed during the frozen queue: {path}")


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
        entry = {**source, "environment": environment_identity(), "command": [sys.executable, *sys.argv],
            "runtime": str(session.relative_to(ROOT)), "start_unix": time.time(), "plan_sha256": sha256(PLAN)}
        execution["sessions"].append(entry)
        write_json(record, execution)
        update_index(result, plan)
        try:
            for panel in plan["panels"]:
                if stopping:
                    break
                if summarize(panel, plan["model"])["status"] == "complete":
                    continue
                verify_source(source)
                if sha256(CHECKPOINT) != CHECKPOINT_SHA:
                    raise ValueError("Original R43 checkpoint changed during the queue")
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
                print(f"START {panel['id']}: {panel['pairs']} pairs", flush=True)
                with log_path.open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    write_json(RUNTIME / "live.json", {"status": "running", "pid": os.getpid(),
                        "child_pid": child.pid, "current_panel": panel["id"], "time_unix": time.time()})
                    code = child.wait()
                    child = None
                verify_source(source)
                result = aggregate(plan)
                if stopping or code == 130:
                    break
                if code:
                    raise RuntimeError(f"Evaluation exited {code}: {log_path}")
                row = next(row for row in result["panels"] if row["id"] == panel["id"])
                if row["status"] != "complete":
                    raise ValueError("Evaluator exited before completing 500 valid pairs")
                print(f"DONE {panel['id']}: score={row['score']:.5f}", flush=True)
            result = aggregate(plan)
            entry.update(end_unix=time.time(), status=result["status"])
            write_json(record, execution)
            update_index(result, plan)
            write_json(RUNTIME / "live.json", {"status": result["status"], "time_unix": time.time()})
            return 0 if result["status"] == "complete" else 130
        except Exception as error:
            entry.update(end_unix=time.time(), status="failed", error=f"{type(error).__name__}: {error}")
            write_json(record, execution)
            update_index(result, plan)
            write_json(RUNTIME / "live.json", {"status": "failed", "error": entry["error"], "time_unix": time.time()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
