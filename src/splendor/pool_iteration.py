"""One-command finite P4 experiments or continuous online replay forks."""
from __future__ import annotations
import argparse
import fcntl
import json
import signal
import sys
import time
from pathlib import Path
import torch
from .alphazero_checkpoint import load_alphazero_checkpoint
from .alphazero_recording import ReferenceWriter, read_json, sha256, write_json
from .alphazero_replay import PositionPool
from .alphazero_train import AlphazeroLearner, LearnerConfig, run_training_round
from .compact_game_cache import CompactGameCache
from .evaluation import FixedEvaluationConfig, run_fixed_evaluation
from .hybrid_train import (HybridConfig, _identity, _release_inference_memory,
                           frozen_offline_dataset, prepare_initialization, run_hybrid)
from .offline_train import _copy_checkpoint, _digest, make_index
from .pool_fitting import evaluate
from .rng import DeterministicRng
from .opponent_pool import load_frozen_actor_critic_checkpoint

PROTOCOL = "p4-pool-self-iteration-v1"


def load_plan(path):
    plan = read_json(path)
    forever = plan.get("forever", False)
    if plan["protocol"] != PROTOCOL or type(forever) is not bool:
        raise ValueError("invalid pool iteration plan")
    if forever:
        if "rounds" in plan or set(plan.get("fork_from", {})) != {
                "run", "round", "checkpoint_sha256", "replay_snapshot_sha256"}:
            raise ValueError("forever requires a frozen replay fork and no rounds target")
    elif type(plan.get("rounds")) is not int or plan["rounds"] <= 0:
        raise ValueError("finite pool iteration requires positive rounds")
    config = HybridConfig.from_dict(read_json(Path(plan["hybrid_config"])))
    if config.fitting_diagnostics is None:
        raise ValueError("pool iteration requires fitting diagnostics")
    if forever:
        if not config.online_only or config.evaluation_anchor is None:
            raise ValueError("forever requires online_only and a fixed evaluation anchor")
    elif config.initialization is None or config.online_only:
        raise ValueError("finite pool iteration requires explicit offline initialization")
    return plan, config


def check(plan, config):
    if plan.get("forever", False):
        fork = plan["fork_from"]
        parent = Path(fork["run"])
        state = read_json(parent / "hybrid_state.json")
        chosen = fork["round"]
        if type(chosen) is not int or not 1 <= chosen <= len(state["completed_rounds"]):
            raise ValueError("fork round must be a completed parent round")
        stage = parent / "rounds" / f"round-{chosen-1:03d}"
        checkpoint, snapshot_path = stage / "checkpoints/final.pt", stage / "replay_snapshot.json"
        for path, expected in ((checkpoint, fork["checkpoint_sha256"]),
                               (snapshot_path, fork["replay_snapshot_sha256"]),
                               (Path(config.evaluation_anchor["path"]), config.evaluation_anchor["sha256"])):
            if sha256(path) != expected:
                raise ValueError(f"frozen continuation source changed: {path}")
        payload = load_alphazero_checkpoint(checkpoint)
        if payload.learner_config != config.to_dict()["offline"]["learner"]:
            raise ValueError("fork source learner configuration changed")
        if payload.round_train_step != payload.round_steps_target:
            raise ValueError("fork source training is incomplete")
        games = read_json(snapshot_path)["games"]
        if len(games) != config.replay_max_games or max(g["round_index"] for g in games) != payload.round_index-1:
            raise ValueError("continuation requires the complete selected online window")
        if not all(Path(g["path"]).is_file() for g in games):
            raise ValueError("inherited online game archive is missing")
        if config.online_validation_positions:
            for name in ("fixed_online_dataset.json", "fixed_online_validation.json"):
                read_json(parent / name)
        read_json(stage / "fitting/train_references.json")
        FixedEvaluationConfig.from_dict(config.evaluation)
        greedy = FixedEvaluationConfig.from_dict(plan["greedy_evaluation"])
        if Path(greedy.policy_b.checkpoint).resolve() != Path(config.evaluation_anchor["path"]).resolve():
            raise ValueError("greedy opponent differs from the fixed anchor")
        result = {"forever": True, "parent_round": chosen, "first_absolute_round": payload.round_index+1,
            "source_games": len(games), "train_games": sum(g["split"] == "train" for g in games),
            "validation_games": sum(g["split"] == "validation" for g in games),
            "initial_step": payload.total_train_step, "effective_learning_rate": config.learning_rate_override,
            "new_games_per_round": config.new_games_per_round, "updates_per_round": config.updates_per_round,
            "offline_batch": 0, "online_batch": config.offline.learner.batch_size,
            "online_cache_bytes": config.offline.replay_cache_bytes,
            "teacher_update": "each round's complete final checkpoint, unconditional on metrics/scores"}
        if config.policy_replay_max_games is not None:
            result["joint_head_pools"] = {"policy_games": config.policy_replay_max_games,
                "value_games": config.replay_max_games, "batch_per_head": config.offline.learner.batch_size}
        return result
    dataset = prepare_initialization(config)
    for identity in [config.evaluation_anchor, plan["strong_anchor"]]:
        if sha256(Path(identity["path"])) != identity["sha256"]:
            raise ValueError("evaluation anchor changed")
    FixedEvaluationConfig.from_dict(config.evaluation)
    FixedEvaluationConfig.from_dict(plan["greedy_evaluation"])
    return {"source_games": len(dataset["games"]), "train_games": sum(g["split"] == "train" for g in dataset["games"]),
        "validation_games": sum(g["split"] == "validation" for g in dataset["games"]),
        "initial_step": load_alphazero_checkpoint(config.initialization["checkpoint"]["path"]).total_train_step,
        "rounds": plan["rounds"], "new_games": plan["rounds"] * config.new_games_per_round,
        "updates": plan["rounds"] * config.updates_per_round,
        "quotas_by_round": [{"round": r + 1, "offline": config.quotas(r)[0],
                            "online": config.quotas(r)[1]} for r in range(plan["rounds"])],
        "collection_batching": {name: config.selfplay.execution[name]
                                for name in ("max_batch_size", "batch_buckets")},
        "teacher_update": "every completed training round, unconditional on metrics/scores"}


def panel(template, checkpoint, opponent, output, *, opponent_name="P4V4"):
    raw = json.loads(json.dumps(template))
    raw["policy_a"]["checkpoint"] = str(Path(checkpoint).resolve())
    raw["policy_b"]["checkpoint"] = str(Path(opponent).resolve())
    raw["policy_b"]["name"] = opponent_name
    write_json(output.with_name(output.stem + "_config.json"), raw)
    result = run_fixed_evaluation(FixedEvaluationConfig.from_dict(raw), output,
        resume=output.exists() or Path(str(output) + ".games").exists(), require_scored_pairs=True)
    if not result.summary["passed"]:
        raise RuntimeError(f"evaluation engineering failure: {output}")
    return result


def offline_control(plan, config):
    root = Path(plan["output"]) / "offline_control"
    dataset = frozen_offline_dataset(config)
    index = make_index(dataset, config.offline.replay_cache_bytes // 2)
    index.cache = CompactGameCache(index.cache.max_bytes)
    pool = PositionPool(index.split_games("train"), index.cache, "offline-control")
    target = plan["rounds"] * config.updates_per_round
    identity = {"digest": _digest([dataset["digest"], "offline-control", target]),
                "scope": "same old pool, same initial AdamW, batch and total optimizer updates"}
    cps = root / "checkpoints"
    steps = sorted(cps.glob("step-*.pt"))
    original = Path(config.initialization["checkpoint"]["path"])
    if steps or (cps / "initial.pt").exists():
        learner = AlphazeroLearner.resume(steps[-1] if steps else cps / "initial.pt", config.offline.learner,
            replay_identity=identity, round_index=0, round_steps_target=target)
    else:
        payload = load_alphazero_checkpoint(original)
        learner = AlphazeroLearner.resume(original, LearnerConfig(**payload.learner_config),
            replay_identity=payload.replay_identity, round_index=payload.round_index,
            round_steps_target=payload.round_steps_target)
        learner.config = config.offline.learner
        learner.round_index, learner.round_train_step, learner.round_steps_target = 0, 0, target
        learner.replay_identity, learner.parent_checkpoint = identity, _identity(original)
        learner.save(cps / "initial.pt")
    refs = Path(plan["output"]) / "offline/fitting_references.json"
    fixed = read_json(refs)

    def observe(learner, checkpoint):
        if learner.round_train_step not in {0, target // 3, 2 * target // 3, target}:
            return
        path = root / "fitting" / f"step-{learner.round_train_step:09d}.json"
        if not path.exists():
            write_json(path, {"step": learner.round_train_step, "total_train_step": learner.total_train_step,
                "checkpoint_sha256": sha256(checkpoint), "evaluation": evaluate(learner, index, dataset["games"], fixed=fixed)})

    observe(learner, steps[-1] if steps else cps / "initial.pt")
    writer = ReferenceWriter(root / "usage", str(time.time_ns()), config.offline.learner.batch_size, identity)
    try:
        run_training_round(learner, steps_target=target, current_pool=pool, history_pool=None,
            validation_samples=(), validation_games=0, checkpoint_dir=cps,
            metrics_path=root / "training_metrics.jsonl", reference_writer=writer,
            checkpoint_callback=observe)
    finally:
        writer.close()
    final = cps / "final.pt"
    _copy_checkpoint(cps / f"step-{target:09d}.pt", final)
    del learner, index, pool
    _release_inference_memory()
    return final


def final_offline_holdouts(plan, config, checkpoints):
    dataset = frozen_offline_dataset(config)
    index = make_index(dataset, config.offline.replay_cache_bytes // 2)
    index.cache = CompactGameCache(index.cache.max_bytes)
    for name, checkpoint in checkpoints.items():
        path = Path(plan["output"]) / "final_fitting" / f"{name}_old_pool_full.json"
        if path.exists():
            if read_json(path)["checkpoint_sha256"] != sha256(checkpoint):
                raise ValueError("final fitting result checkpoint changed")
            continue
        learner = AlphazeroLearner(load_frozen_actor_critic_checkpoint(checkpoint, device=config.offline.learner.device),
                                  config.offline.learner, sampling_rng=DeterministicRng(0))
        write_json(path, {"checkpoint_sha256": sha256(checkpoint), "manifest": config.initialization["manifest"],
                         "evaluation": evaluate(learner, index, dataset["games"])})
        del learner
    del index
    _release_inference_memory()


def _online_round_record(plan, state, local_round):
    """Persist one committed generation; never retain all historical game panels."""
    root = Path(plan["output"])
    absolute_round = state["online_round_offset"] + local_round
    stage = root / "rounds" / f"round-{local_round-1:03d}"
    path = Path(plan["artifacts"]) / "rounds" / f"round-{absolute_round:06d}.json"
    if path.exists():
        return read_json(path)
    points, panels = [], []
    for raw_path in sorted((stage / "fitting").glob("step-*.json")):
        point = read_json(raw_path)
        points.append({"round_step": point["round_step"], "total_train_step": point["total_train_step"],
            "checkpoint_sha256": point["checkpoint_sha256"], "raw_path": str(raw_path),
            "online_train_fixed": point["online_train_fixed"]["metrics"],
            "online_validation_full": point["online_validation_full"]["metrics"]})
    fields = ("passed", "completed_pairs", "total_pairs", "win_rate_score", "paired_bootstrap")
    for raw_path in sorted((stage / "evaluation").glob("*.json")):
        value = read_json(raw_path)
        if "win_rate_score" in value:
            panels.append({"path": str(raw_path), "summary": {k: value[k] for k in fields}})
    result = {"round": absolute_round, "local_round": local_round, "fitting": points, "panels": panels,
        "source_usage": read_json(stage / "source_usage.json"),
        "checkpoint": _identity(stage / "checkpoints/final.pt")}
    write_json(path, result)
    return result


def _online_summary(plan, state):
    completed = len(state["completed_rounds"])
    latest = _online_round_record(plan, state, completed) if completed else None
    result = {"status": "in_progress", "completed_rounds": completed, "target_rounds": None,
        "parent_round": plan["fork_from"]["round"], "parent_absolute_round": state["online_round_offset"],
        "last_absolute_round": state["online_round_offset"]+completed,
        "latest_teacher": state["latest_checkpoint"], "latest_round": latest,
        "history": "rounds/round-*.json; one immutable record per committed absolute generation",
        "scope": "online-only; single training seed; stored-label DEV fitting and paired strength panels"}
    artifacts = Path(plan["artifacts"])
    write_json(artifacts / "summary.json", result)
    lines = [f"# 第{result['parent_absolute_round']}轮终点长期在线续训", "", f"完成续训{completed}轮；最新绝对轮号{result['last_absolute_round']}；无轮数终点。",
        "", "教师每轮完整终点无条件升级；历史矩阵和对战摘要逐轮保存于 rounds/，原始结果保留在runs。", ""]
    if latest:
        lines += ["| 轮 | 本轮步数 | 最新代完整留出KL | 最新代完整留出MSE |", "|---|---:|---:|---:|"]
        key = f"g{latest['round']:03d}/validation/all"
        for point in latest["fitting"]:
            if key in point["online_validation_full"]:
                metric = point["online_validation_full"][key]
                lines.append(f"| {latest['round']} | {point['round_step']} | {metric['policy_kl']:.6f} | {metric['value_mse']:.6f} |")
        lines += ["", "| 对战面板 | 完成pair | 得分 | 95%配对区间 |", "|---|---:|---:|---:|"]
        for item in latest["panels"]:
            s, label = item["summary"], Path(item["path"]).name
            ci = s["paired_bootstrap"]
            lines.append(f"| {label} | {s['completed_pairs']}/{s['total_pairs']} | {s['win_rate_score']:.2%} | [{ci['lower']:.2%},{ci['upper']:.2%}] |")
    (artifacts / "results.md").write_text("\n".join(lines)+"\n")
    return result


def analyze(plan):
    if plan.get("forever", False):
        state_path = Path(plan["output"]) / "hybrid_state.json"
        if not state_path.exists():
            return {"status": "not_started", "completed_rounds": 0, "target_rounds": None}
        state = read_json(state_path)
        # Repair any summary interrupted after the durable teacher commit, one round at a time.
        for local_round in range(1, len(state["completed_rounds"])+1):
            _online_round_record(plan, state, local_round)
        return _online_summary(plan, state)
    root, artifacts = Path(plan["output"]), Path(plan["artifacts"])
    state_path = root / "hybrid_state.json"
    state = read_json(state_path) if state_path.exists() else {"completed_rounds": []}
    rows, panels, source_usage = [], [], []
    for stage in sorted((root / "rounds").glob("round-*")):
        if (stage / "source_usage.json").exists():
            source_usage.append({"round": int(stage.name.split("-")[-1]) + 1,
                                 "usage": read_json(stage / "source_usage.json")})
        for path in sorted((stage / "fitting").glob("step-*.json")):
            point = read_json(path)
            rows.append({"round": int(stage.name.split("-")[-1]) + 1,
                "round_step": point["round_step"], "total_train_step": point["total_train_step"],
                "checkpoint_sha256": point["checkpoint_sha256"], "raw_path": str(path),
                "offline_fixed": point["offline_fixed"]["pool_weighted"],
                "online_train_fixed": point["online_train_fixed"]["metrics"],
                "online_validation_full": point["online_validation_full"]["metrics"]})
        for path in sorted((stage / "evaluation").glob("*.json")):
            value = read_json(path)
            if "win_rate_score" in value:
                panels.append({"path": str(path), "summary": value})
    for path in sorted((root / "final_evaluation").glob("*.json")):
        value = read_json(path)
        if "win_rate_score" in value:
            panels.append({"path": str(path), "summary": value})
    final_fitting = {path.stem: {"raw_path": str(path), "pool_weighted": read_json(path)["evaluation"]["pool_weighted"]}
                     for path in sorted((root / "final_fitting").glob("*.json"))}
    result = {"status": "complete" if (root / "complete.json").exists() else "in_progress" if state_path.exists() else "not_started",
        "completed_rounds": len(state["completed_rounds"]), "target_rounds": plan["rounds"],
        "fitting": rows, "final_fitting": final_fitting, "source_usage": source_usage, "panels": panels,
        "scope": "single training seed; stored-label DEV fitting and paired strength panels"}
    write_json(artifacts / "summary.json", result)
    lines = ["# P4大池起点自迭代", "", f"状态：{result['status']}；完成 {result['completed_rounds']}/{plan['rounds']} 轮。", "",
        "教师每轮末无条件升级。预算以冻结配置为准；拟合与棋力结果用于诊断。", "",
        "| 轮 | 本轮步数 | 最新代完整留出KL | 最新代完整留出MSE |", "|---|---:|---:|---:|"]
    for row in rows:
        key = f"g{row['round']:03d}/validation/all"
        if key in row["online_validation_full"]:
            m = row["online_validation_full"][key]
            lines.append(f"| {row['round']} | {row['round_step']} | {m['policy_kl']:.6f} | {m['value_mse']:.6f} |")
    lines += ["", "| 对战面板 | 完成pair | 得分 | 95%配对区间 |", "|---|---:|---:|---:|"]
    for item in panels:
        s = item["summary"]
        label = str(Path(item["path"]).relative_to(root))
        if s["passed"] and s["completed_pairs"] == s["total_pairs"]:
            ci = s["paired_bootstrap"]
            lines.append(f"| {label} | {s['completed_pairs']} | {s['win_rate_score']:.2%} | [{ci['lower']:.2%},{ci['upper']:.2%}] |")
        else:
            lines.append(f"| {label} | {s['completed_pairs']}/{s['total_pairs']} | 未完成有效面板 | — |")
    lines += ["", "棋力逐面板summary、阶段指标、实际采样份额及历史代矩阵见 summary.json；原始引用和逐局统计保留在runs。", ""]
    (artifacts / "results.md").write_text("\n".join(lines))
    return result


def run(plan, config, plan_path, *, max_steps_this_session=None):
    torch.set_num_threads(1)
    if config.offline.learner.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("this training configuration requires an available CUDA device")
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = root / "plan.json"
        if frozen.exists() and read_json(frozen) != plan:
            raise ValueError("frozen iteration plan changed")
        write_json(frozen, plan)
        write_json(Path(plan["artifacts"]) / "plan.json", plan)
        check(plan, config)
        if (root / "complete.json").exists():
            completed = read_json(root / "complete.json")
            for name in ("checkpoint", "control"):
                if _identity(Path(completed[name]["path"])) != completed[name]:
                    raise ValueError("completed experiment checkpoint changed")
            return analyze(plan)
        cfg_marker = root / "config.json"
        if cfg_marker.exists() and read_json(cfg_marker) != config.to_dict():
            raise ValueError("frozen hybrid configuration changed")

        def observe_round(round_number, checkpoint, stage):
            opponent = config.evaluation_anchor if config.online_only else config.initialization["checkpoint"]
            panel(plan["greedy_evaluation"], checkpoint, opponent["path"],
                  stage / "evaluation/greedy_vs_start.json")
            if not plan.get("forever", False):
                analyze(plan)

        if plan.get("forever", False):
            if cfg_marker.exists():
                analyze(plan)
            fork = plan["fork_from"]
            try:
                state = run_hybrid(config, root, resume=cfg_marker.exists(), forever=True,
                    fork_from=None if cfg_marker.exists() else fork["run"],
                    fork_round=None if cfg_marker.exists() else fork["round"],
                    max_steps_this_session=max_steps_this_session, round_callback=observe_round,
                    round_commit_callback=lambda r, cp, stage, state: _online_summary(plan, state))
                return _online_summary(plan, state)
            finally:
                if (root / "hybrid_state.json").exists():
                    _online_summary(plan, read_json(root / "hybrid_state.json"))

        try:
            state = run_hybrid(config, root, resume=cfg_marker.exists(), online_rounds=plan["rounds"],
                max_steps_this_session=max_steps_this_session, round_callback=observe_round)
            if len(state["completed_rounds"]) < plan["rounds"]:
                return analyze(plan)
            if max_steps_this_session is not None:
                # A bounded smoke/session must not silently launch the control or final matches.
                return analyze(plan)
            final = Path(state["latest_checkpoint"]["path"])
            control = offline_control(plan, config)
            final_offline_holdouts(plan, config, {"online": final, "offline_control": control})
            devices = list(range(torch.cuda.device_count())) if config.offline.learner.device == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                try:
                    panel(config.evaluation, final, control, root / "final_evaluation/online_vs_offline_control.json",
                          opponent_name="offline_control")
                    _release_inference_memory()
                    panel(config.evaluation, final, plan["strong_anchor"]["path"], root / "final_evaluation/online_vs_P12V4.json",
                          opponent_name="P12V4")
                finally:
                    _release_inference_memory()
            write_json(root / "complete.json", {"rounds": plan["rounds"], "checkpoint": _identity(final),
                "control": _identity(control), "plan_sha256": sha256(plan_path)})
            return analyze(plan)
        finally:
            analyze(plan)


def status(plan, config):
    root = Path(plan["output"])
    if not (root / "hybrid_state.json").exists():
        return {"status": "not_started"}
    state = read_json(root / "hybrid_state.json")
    completed = len(state["completed_rounds"])
    result = {"completed_rounds": completed, "target_rounds": plan.get("rounds"),
              "latest_teacher": state["latest_checkpoint"]}
    if (root / "complete.json").exists():
        result["status"] = "complete"
    elif plan.get("forever", False) or completed < plan["rounds"]:
        stage = root / "rounds" / f"round-{completed:03d}"
        steps = sorted((stage / "checkpoints").glob("step-*.pt"))
        result.update(status="online", current_round=completed + 1,
            committed_new_games=len(list((stage / "selfplay/games").glob("game-*.npz"))),
            new_games_target=config.new_games_per_round,
            checkpoint_step=int(steps[-1].stem.split("-")[-1]) if steps else 0,
            updates_target=config.updates_per_round)
        if config.online_only:
            result["current_round"] += state["online_round_offset"]
            result["local_round"] = completed+1
            result["phase"] = "evaluation_or_commit" if result["checkpoint_step"] == config.updates_per_round else (
                "training" if (stage / "checkpoints/initial.pt").exists() else "collection")
    else:
        steps = sorted((root / "offline_control/checkpoints").glob("step-*.pt"))
        result.update(status="offline_control_or_final_evaluation",
                      control_checkpoint_step=int(steps[-1].stem.split("-")[-1]) if steps else 0)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check", "status", "analyze"), nargs="?", default="run")
    parser.add_argument("--config", type=Path, default=Path("configs/entity_attention_v1/p4_pool_iteration_20261010.json"))
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args(argv)
    plan, config = load_plan(args.config)
    if args.action == "check":
        result = check(plan, config)
    elif args.action == "status":
        result = status(plan, config)
    elif args.action == "analyze":
        result = analyze(plan)
    else:
        def interrupted(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            result = run(plan, config, args.config, max_steps_this_session=args.max_steps_this_session)
        except KeyboardInterrupt:
            print("已中断；重新执行同一命令恢复已提交对局、checkpoint和评测pair。", flush=True)
            return 130
    if args.action in ("run", "analyze"):
        result = {key: result[key] for key in ("status", "completed_rounds", "target_rounds")}
        result["summary"] = str(Path(plan["artifacts"]) / "summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
