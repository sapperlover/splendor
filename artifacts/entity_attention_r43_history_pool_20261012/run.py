"""R24 offline candidate vs R43, then the identical fixed-pool fit from R43."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHARED = ROOT / "artifacts/entity_attention_r24_history_pool_20261012"
CROSS = ROOT / "artifacts/entity_attention_r24_offline_vs_r43_20261012"
sys.path.insert(0, str(SHARED))
spec = importlib.util.spec_from_file_location("r24_history_runner", SHARED / "run.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.paired_statistics import serialized_pair_scores

NAME = HERE.name
INITIAL = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt"
CANDIDATE = ROOT / "runs/entity_attention_r24_history_pool_20261012/checkpoints/step-000024388.pt"
base.HERE = HERE
base.NAME = NAME
base.OUTPUT = ROOT / "runs" / NAME
base.RUNTIME = ROOT / "runtime" / NAME
base.INITIAL = INITIAL


def prepare_plan():
    reference = read_json(SHARED / "config.json")
    summary = read_json(SHARED / "pool_summary.json")
    if sha256(Path(reference["manifest"])) != reference["manifest_sha256"]:
        raise ValueError("original fixed 43000-game manifest changed")
    original = read_json(ROOT / "artifacts/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-000043.json")["checkpoint"]
    selected = read_json(SHARED / "selection.json")
    if selected["step"] != 24388 or sha256(CANDIDATE) != selected["checkpoint_sha256"]:
        raise ValueError("R24 +24388 candidate identity changed")
    if sha256(INITIAL) != original["sha256"]:
        raise ValueError("original round43 checkpoint identity changed")
    parent = load_alphazero_checkpoint(INITIAL)
    r24 = load_alphazero_checkpoint(reference["initial_checkpoint"]["path"])
    if parent.model_config != r24.model_config:
        raise ValueError("R43 and R24 architectures differ")
    config = copy.deepcopy(reference)
    config.update(protocol="r43-fixed-history-offline-v1", initial_round=43,
                  output=str(base.OUTPUT), artifacts=str(HERE),
                  training_script=str(SHARED / "train.py"),
                  initial_checkpoint={"path": str(INITIAL), "sha256": original["sha256"],
                                      "model_sha256": parent.model_sha256,
                                      "total_train_step": parent.total_train_step,
                                      "absolute_round": 43},
                  reference_experiment={"config": str(SHARED / "config.json"),
                                        "config_sha256": sha256(SHARED / "config.json"),
                                        "method": "same frozen pool, Train/validation splits, sampler/subset seeds, AdamW continuation and fit/evaluation schedule"},
                  training_loop_origin={"path": str((SHARED / "train.py").relative_to(ROOT)),
                                        "sha256": sha256(SHARED / "train.py"),
                                        "changes": "identical training math; explicit round43 parent identity"})
    plan = config["strength_plan"]
    plan["opponent"] = "original complete round43 model for all offline-fit panels"
    for kind, seed in (("greedy", 35500000), ("search_dev", 35600000), ("confirmation", 35700000)):
        plan[kind]["seed_start"] = seed
    config["shared_sources"] = {str((SHARED / name).relative_to(ROOT)): sha256(SHARED / name)
                                for name in ("run.py", "train.py", "pool.py")}
    config["pre_training_crossplay"] = {
        "config": str(CROSS / "config.json"), "output": str(CROSS / "evaluation.json"),
        "candidate": str(CANDIDATE), "candidate_sha256": selected["checkpoint_sha256"],
        "opponent": str(INITIAL), "opponent_sha256": original["sha256"],
        "pairs": 200, "seed_start": 35800000}
    base.freeze(HERE / "config.json", config)
    for name in ("run.py", "train.py", "pool.py"):
        source = SHARED / name
        target = HERE / "shared_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"frozen shared-source snapshot changed: {target}")
        if not target.exists():
            target.write_bytes(source.read_bytes())
    raw = copy.deepcopy(read_json(SHARED / "evaluation_configs/confirmation.json"))
    first = config["pre_training_crossplay"]
    seed = first["seed_start"]
    raw.update(game_seeds=list(range(seed, seed + first["pairs"])),
               policy_seed_base=seed + 10000, bootstrap_seed=seed + 20000,
               metadata={"experiment": CROSS.name, "candidate_sha256": first["candidate_sha256"],
                         "opponent_sha256": first["opponent_sha256"],
                         "panel": "predeclared independent-openers R24+24388 vs original R43",
                         "additional_training_step": 24388})
    raw["policy_a"].update(checkpoint=str(CANDIDATE), name="R24_history_step24388")
    raw["policy_b"].update(checkpoint=str(INITIAL), name="original_R43")
    for side in ("policy_a", "policy_b"):
        raw[side]["search_config"].update(seed=seed + 30000, simulations=1024,
                                           determinizations=1, root_noise_fraction=0.)
    base.freeze(CROSS / "config.json", FixedEvaluationConfig.from_dict(raw).to_dict())
    if summary["totals"]["train"]["positions"] != 3121619 or config["milestones"]["4"] != 48776:
        raise ValueError("expected the same original train population and step budget")
    return config


def cross_report():
    output = CROSS / "evaluation.json"
    raw = read_json(output) if output.exists() else None
    lines = ["# 第24轮＋24388步 vs 原第43轮", "",
             "双方完整网络、S1024/N1、无根噪声，独立开局35800000起，互换先后手；和棋半分。", ""]
    if raw is None:
        lines += ["状态：准备完成，正式对战未开始。"]
    else:
        ci = raw["paired_bootstrap"]
        interval = f"[{ci['lower']:.3%}, {ci['upper']:.3%}]" if ci else "未完成"
        score = f"{raw['win_rate_score']:.3%}" if raw["win_rate_score"] is not None else "未完成"
        lines += [f"完成：{raw['completed_pairs']}/{raw['total_pairs']} pair；R24离线候选得分{score}；95% CI {interval}。",
                  f"有效对局：{raw['valid_games']}；passed={raw['passed']}。"]
    lines += ["", "本面板单独判断此前离线候选是否超过R43；不用于选择R43离线训练节点。",
              "原始配置与逐pair结果见本目录config.json、evaluation.json；恢复入口见[串行训练链路](../entity_attention_r43_history_pool_20261012/README.md)。", ""]
    (CROSS / "results.md").write_text("\n".join(lines), encoding="utf-8")


def run_crossplay(config):
    first = config["pre_training_crossplay"]
    output = Path(first["output"])
    raw_config = read_json(Path(first["config"]))
    if output.exists():
        raw = read_json(output)
        if raw["config"] != raw_config:
            raise ValueError("saved crossplay config differs from frozen plan")
        if raw["passed"] and raw["completed_pairs"] == first["pairs"]:
            serialized_pair_scores(raw)
            cross_report()
            return
    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", first["config"],
               "--output", str(output), "--progress-every", "10", "--summary-every", "10"]
    if output.exists() or Path(str(output) + ".games").exists():
        command.append("--resume")
    try:
        base.run_child(command, "R24_offline_vs_R43")
    finally:
        cross_report()
    raw = read_json(output)
    serialized_pair_scores(raw)
    if raw["config"] != raw_config or raw["completed_pairs"] != first["pairs"]:
        raise ValueError("crossplay incomplete or configuration changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "train", "status", "analyze"), nargs="?", default="run")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    if args.max_steps_this_session is not None and args.max_steps_this_session <= 0:
        parser.error("--max-steps-this-session must be positive")
    if args.action == "status":
        progress = read_json(HERE / "progress.json") if (HERE / "progress.json").exists() else {}
        cross = read_json(CROSS / "evaluation.json") if (CROSS / "evaluation.json").exists() else {}
        print(json.dumps({"training": {k: progress.get(k) for k in ("status", "step", "steps_target")},
                          "crossplay": {k: cross.get(k) for k in ("passed", "completed_pairs", "total_pairs", "win_rate_score")},
                          "live": read_json(base.RUNTIME / "live.json") if (base.RUNTIME / "live.json").exists() else None}, ensure_ascii=False, indent=2))
        return
    config = prepare_plan()
    if args.action in ("prepare", "analyze"):
        base.analyze(config)
        cross_report()
        print(json.dumps({"train_steps": config["milestones"]["4"], "crossplay_pairs": 200,
                          "manifest_sha256": config["manifest_sha256"]}, ensure_ascii=False))
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, base.request_stop)
    base.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (base.OUTPUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = base.RUNTIME / f"queue-{time.time_ns()}"
        provenance = source_identity(ROOT, session / "source")
        write_json(session / "identity.json", {**provenance, "command": sys.argv,
                   "entrypoint_sha256": sha256(Path(__file__)), "config_sha256": sha256(HERE / "config.json"),
                   "shared_sources": config["shared_sources"]})
        try:
            run_crossplay(config)
            base.run(config, training_only=args.action == "train", max_steps_this_session=args.max_steps_this_session)
        except KeyboardInterrupt:
            base.analyze(config)
            cross_report()
            print("已中断；同一命令续跑未完成对战／训练。", flush=True)


if __name__ == "__main__":
    main()
