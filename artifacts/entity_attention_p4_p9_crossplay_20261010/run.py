"""Two sequential, resumable 200-pair P4/P9 policy-value compositions."""
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
from dataclasses import asdict
from pathlib import Path

import torch

from splendor.actor_critic import MODEL_WEIGHTS_FORMAT_VERSION
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.entity_attention import EntityAttentionPolicyValue
from splendor.evaluation import FixedEvaluationConfig
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.paired_statistics import paired_bootstrap_difference, paired_bootstrap_mean, serialized_pair_scores
from splendor.replay import RULESET_ID

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime" / ART.name
MODELS = ROOT / "runs" / ART.name / "compositions"
PLAN = ART / "plan.json"
LIVE = RUNTIME / "live.json"
BASELINE = ROOT / "runs/entity_attention_p4_pool_iteration_20261010/rounds/round-008/evaluation/mcts_vs_fixed_anchor.json"
SOURCES = {
    "P4V4": ("runs/entity_attention_recent_pool_20261009/checkpoints/step-000034398.pt", "94f4ce89d27f42ec0c39e72055b460405ae36b8f9931051cbe360ee145ace5ec", 34398),
    "P9V9": ("runs/entity_attention_p4_pool_iteration_20261010/rounds/round-008/checkpoints/final.pt", "ff34660fece12590762da93d031523e4ae09db4fa6af5d2284c4af0dcd57237b", 38898),
}
MATCHUPS = (("S01_newP9_oldV4", "P9V9", "P4V4"), ("S02_oldP4_newV9", "P4V4", "P9V9"))
child = None
stopping = False


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen experiment identity changed: {path}")
    else:
        write_json(path, value)


def prepare():
    identities, parents = {}, {}
    for name, (relative, expected, step) in SOURCES.items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise ValueError(f"source checkpoint changed: {name}")
        identities[name] = {"checkpoint": relative, "sha256": expected, "global_step": step}
        parents[name] = load_frozen_actor_critic_checkpoint(path)
    baseline = read_json(BASELINE)
    seeds, scores = serialized_pair_scores(baseline)
    if seeds != tuple(range(30900000, 30900200)) or baseline["completed_pairs"] != 200:
        raise ValueError("baseline is not the completed ninth-round 200-pair panel")
    for side, name in (("policy_a", "P9V9"), ("policy_b", "P4V4")):
        if baseline["checkpoint_identity"][side]["sha256"] != identities[name]["sha256"]:
            raise ValueError("baseline checkpoint identity differs")
    panels = []
    for name, policy, value in MATCHUPS:
        path = MODELS / f"{name}.pt"
        sources = {"policy": identities[policy], "value": identities[value]}
        if not path.exists():
            model = EntityAttentionPolicyValue().eval().requires_grad_(False)
            model.policy_model.load_state_dict(parents[policy].state_dict(), strict=True)
            model.value_model.load_state_dict(parents[value].state_dict(), strict=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"checkpoint_schema": "splendor-frozen-actor-critic-v1",
                "format_version": MODEL_WEIGHTS_FORMAT_VERSION, "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "config": asdict(model.config), "state_dict": model.state_dict(),
                "metadata": {"rules_version": RULESET_ID, "sources": sources, "inference_only": True,
                    "composition": "complete separate source encoders; policy from one network and value from the other"}}, path)
        model = load_frozen_actor_critic_checkpoint(path)
        for branch, parent in (("policy", policy), ("value", value)):
            actual, expected = getattr(model, f"{branch}_model").state_dict(), parents[parent].state_dict()
            if actual.keys() != expected.keys() or any(not torch.equal(actual[k], expected[k]) for k in expected):
                raise ValueError(f"composition weights differ: {name}/{branch}")
        identity = {"checkpoint": str(path.relative_to(ROOT)), "sha256": sha256(path), "sources": sources}
        config = copy.deepcopy(baseline["config"])
        config["policy_a"].update(checkpoint=str(path), name=name)
        config["metadata"] = {"experiment": ART.name, "panel": name, "composition": identity,
            "seed_scope": "same 200 DEV openers and search RNG seeds as completed P9V9 vs P4V4"}
        config = FixedEvaluationConfig.from_dict(config).to_dict()
        config_path = ART / "configs" / f"{name}.json"
        freeze(config_path, config)
        panels.append({"id": name, "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path), "model": identity, "pairs": 200,
            "output": str((ART / "panels" / f"{name}.json").relative_to(ROOT))})
    plan = {"experiment": ART.name, "sources": identities, "panels": panels,
        "runner_sha256": sha256(Path(__file__)), "baseline": {"path": str(BASELINE.relative_to(ROOT)),
            "sha256": sha256(BASELINE), "score": baseline["win_rate_score"], "ci95": baseline["paired_bootstrap"],
            "game_seeds": list(seeds), "pair_scores": scores.tolist()},
        "settings": "S1024/N1; native_batched; compiled; active100; batch64; no root noise; both seats; two panels sequential",
        "scope": "One fixed training trajectory and opponent; reused DEV openers; exploratory comparisons without multiplicity adjustment. Separate complete source encoders, no training."}
    freeze(PLAN, plan)
    return plan


def summarize(panel):
    path = ROOT / panel["output"]
    row = {"id": panel["id"], "status": "pending", "completed_pairs": 0}
    if not path.exists():
        return row
    raw = read_json(path)
    config = read_json(ROOT / panel["config"])
    if raw["config"] != config:
        raise ValueError("saved evaluation config differs")
    row.update(status="incomplete", completed_pairs=raw["completed_pairs"])
    if raw["completed_pairs"] != 200:
        return row
    seeds, scores = serialized_pair_scores(raw)
    if seeds != tuple(config["game_seeds"]) or raw["completed_games"] != 400:
        raise ValueError("completed panel population differs")
    for side in ("policy_a", "policy_b"):
        identity = raw["checkpoint_identity"][side]
        if identity["path"] != config[side]["checkpoint"] or identity["sha256"] != sha256(Path(identity["path"])):
            raise ValueError("evaluated checkpoint differs")
    ci = paired_bootstrap_mean(scores, resamples=config["bootstrap_resamples"], seed=config["bootstrap_seed"]).to_dict()
    if ci != raw["paired_bootstrap"]:
        raise ValueError("paired confidence interval differs")
    for pair in raw["pairs"]:
        for game in pair["games"]:
            for side in ("policy_a_stats", "policy_b_stats"):
                stats = game[side]
                if stats["simulations"] != stats["searchable_decisions"] * 1024 or stats["determinizations"] != stats["searchable_decisions"]:
                    raise ValueError("actual search budget differs from S1024/N1")
    row.update(status="complete", score=raw["win_rate_score"], ci95=ci, result=panel["output"],
        result_sha256=sha256(path), elapsed_seconds_this_invocation=raw["elapsed_seconds_this_invocation"],
        **{k: raw[k] for k in ("a_wins", "draws", "a_losses", "engineering_truncations", "exceptions", "illegal_actions", "invariant_failures", "policy_cycle_draws")})
    return row


def aggregate(plan):
    rows = [summarize(panel) for panel in plan["panels"]]
    complete = [row for row in rows if row["status"] == "complete"]
    contrasts = {}
    for row in complete:
        seeds, scores = serialized_pair_scores(read_json(ROOT / row["result"]))
        contrasts[row["id"] + "_minus_P9V9"] = paired_bootstrap_difference(seeds, scores,
            plan["baseline"]["game_seeds"], plan["baseline"]["pair_scores"], resamples=10000, seed=30940000).to_dict()
    if len(complete) == 2:
        a_seeds, a = serialized_pair_scores(read_json(ROOT / complete[0]["result"]))
        b_seeds, b = serialized_pair_scores(read_json(ROOT / complete[1]["result"]))
        contrasts["newP9_oldV4_minus_oldP4_newV9"] = paired_bootstrap_difference(a_seeds, a, b_seeds, b,
            resamples=10000, seed=30940001).to_dict()
    result = {"status": "complete" if len(complete) == 2 else "incomplete", "completed_panels": len(complete),
        "plan_sha256": sha256(PLAN), "panels": rows, "paired_score_differences": contrasts,
        "existing_P9V9_vs_P4V4": plan["baseline"], "scope": plan["scope"]}
    write_json(ART / "summary.json", result)
    lines = ["# 新 P9 与旧 P4 的 policy/value 交换对战", "", f"状态：`{result['status']}`；完成 {len(complete)}/2 组。",
        "", "两组均对旧 P4V4，各200 pair／400局；S1024/N1，和棋半分，95% CI按整pair bootstrap。", "",
        "| 组合 | 完整pair | 胜／和／负 | 得分 | 95% CI |", "| --- | ---: | --- | ---: | --- |"]
    for row in rows:
        if row["status"] == "complete":
            ci = row["ci95"]
            lines.append(f"| [{row['id']}](panels/{row['id']}.json) | 200 | {row['a_wins']}/{row['draws']}/{row['a_losses']} | {100*row['score']:.3f}% | [{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%] |")
        else:
            lines.append(f"| {row['id']} | {row['completed_pairs']} | — | — | — |")
    baseline = plan["baseline"]
    ci = baseline["ci95"]
    lines += [f"| P9V9（已有对照） | 200 | 234/2/164 | {100*baseline['score']:.3f}% | [{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%] |",
        "", "同开局配对得分差（百分点）：", "", "| 比较 | 差值 | 95% CI |", "| --- | ---: | --- |"]
    for name, ci in contrasts.items():
        lines.append(f"| {name} | {100*ci['estimate']:+.3f} | [{100*ci['lower']:+.3f}, {100*ci['upper']:+.3f}] |")
    lines += ["", "比较的是各组合对固定 P4V4 的得分差，不是组合之间直接互打。复用第九轮DEV开局，只有一条训练轨迹；未作多重比较校正。policy/value各自保留完整来源encoder，结果包含编码表示与搜索交互，不能拆成可相加的head贡献。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def interrupt(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def main():
    global child
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "analyze", "stop"), nargs="?", default="run")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        print({"live": read_json(LIVE) if LIVE.exists() else None,
            "panels": [summarize(p) for p in read_json(PLAN)["panels"]]})
        return 0
    if args.action == "stop":
        live = read_json(LIVE)
        if live["status"] != "running":
            print(live["status"])
            return 0
        command = Path(f"/proc/{live['pid']}/cmdline").read_bytes().split(b"\0")
        if not any(item and Path(os.fsdecode(item)).resolve() == Path(__file__) for item in command):
            raise ValueError("recorded PID is not this experiment runner")
        os.kill(live["pid"], signal.SIGTERM)
        return 0
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
                print({"status": result["status"], "completed_panels": result["completed_panels"]})
                return 0
            session = RUNTIME / str(time.time_ns())
            identity = source_identity(ROOT, session)
            for path in (Path(__file__), PLAN, *sorted((ART / "configs").glob("*.json"))):
                target = session / path.relative_to(ART)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            execution_path = ART / "execution.json"
            execution = read_json(execution_path) if execution_path.exists() else {"sessions": []}
            execution["sessions"].append({**identity, "session": str(session.relative_to(ROOT)),
                "environment": environment_identity(), "command": [sys.executable, *sys.argv], "start_unix": time.time(), "plan_sha256": sha256(PLAN)})
            write_json(execution_path, execution)
            for panel in plan["panels"]:
                if stopping:
                    break
                if summarize(panel)["status"] == "complete":
                    print(f"SKIP {panel['id']}", flush=True)
                    continue
                if sha256(ROOT / panel["config"]) != panel["config_sha256"]:
                    raise ValueError("frozen panel config changed")
                output = ROOT / panel["output"]
                command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(ROOT / panel["config"]),
                    "--output", str(output), "--progress-every", "10", "--summary-every", "10"]
                if output.exists() or output.with_suffix(output.suffix + ".games").exists():
                    command.append("--resume")
                write_json(session / f"command-{panel['id']}.json", {"command": command, "start_unix": time.time()})
                logfile = RUNTIME / f"{panel['id']}.log"
                print(f"START {panel['id']} (200 pairs)", flush=True)
                with logfile.open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    write_json(LIVE, {"status": "running", "pid": os.getpid(), "child_pid": child.pid,
                        "current_panel": panel["id"], "log": str(logfile), "time_unix": time.time()})
                    code = child.wait()
                    child = None
                if stopping or code == 130:
                    break
                if code:
                    raise RuntimeError(f"evaluation exited {code}: {logfile}")
                result = aggregate(plan)
                row = next(r for r in result["panels"] if r["id"] == panel["id"])
                if row["status"] != "complete":
                    raise ValueError("evaluation exited before all pairs completed")
                print(f"DONE {panel['id']}: score={row['score']:.5f}, CI={row['ci95']['lower']:.5f}..{row['ci95']['upper']:.5f}", flush=True)
            result = aggregate(plan)
            status = "complete" if result["status"] == "complete" else "interrupted"
            write_json(LIVE, {"status": status, "pid": os.getpid(), "time_unix": time.time()})
            return 0 if status == "complete" else 130
        except Exception as error:
            write_json(LIVE, {"status": "failed", "pid": os.getpid(), "error": f"{type(error).__name__}: {error}", "time_unix": time.time()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
