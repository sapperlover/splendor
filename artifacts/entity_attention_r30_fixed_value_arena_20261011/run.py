"""Three fixed-value policy panels, then round24 versus round30; resumable."""
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
from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples, load_game

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime" / ART.name
MODELS = ROOT / "runs" / ART.name / "compositions"
PLAN = ART / "plan.json"
LIVE = RUNTIME / "live.json"
PREVIOUS = ROOT / "artifacts/entity_attention_r30_value_weight_20261011"
SOURCES = {
    "P4": ("runs/entity_attention_recent_pool_20261009/checkpoints/step-000034398.pt",
           "94f4ce89d27f42ec0c39e72055b460405ae36b8f9931051cbe360ee145ace5ec"),
    "round24": ("runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-014/checkpoints/final.pt",
                "15424dc571c15dcbff0aeb3d6e698cdbea952583eb482f6f5826fc23b00f5bdb"),
    "round30": ("runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-020/checkpoints/final.pt",
                "9fc258b211954cf32cd7b688d5d534a502441304ea7cc92a7a35e059a7c91b0f"),
    "value025": ("runs/entity_attention_r30_value_weight_20261011/value025/checkpoints/final.pt",
                 "3e870b839ca1c9dddc6853c7bc8fca14efea428afdebc6b425578a192ee475e9"),
    "value0": ("runs/entity_attention_r30_value_weight_20261011/value0/checkpoints/final.pt",
               "273758eb745e06344f5c7ee602f70bd3a0ce4494b3bf9d5b37b49884950d3d59"),
}
POLICY_PANELS = (
    ("S01_P1_V30_vs_P4", "round30", "权重1 policy＋原第30轮value vs P4"),
    ("S02_P025_V30_vs_P4", "value025", "权重0.25 policy＋原第30轮value vs P4"),
    ("S03_P0_V30_vs_P4", "value0", "权重0 policy＋原第30轮value vs P4"),
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


def verify_outputs(parents, models):
    records = read_json(PREVIOUS / "replay_snapshot.json")["games"]
    games = [g for g in records if g["round_index"] == 29 and g["split"] == "validation"][:3]
    samples, sources = [], []
    for game in games:
        path = Path(game["path"])
        arrays, metadata = load_game(path)
        positions = sorted({0, game["decision_count"] // 2, game["decision_count"] - 1})
        samples.extend(SelfPlaySample(arrays, metadata, i) for i in positions)
        sources.append({**identity(path), "decision_indices": positions})
    batch = collate_selfplay_samples(samples).decisions
    with torch.inference_mode():
        expected_value = parents["round30"](batch)[1]
        for name, source, _ in POLICY_PANELS:
            logits, value = models[name](batch)
            torch.testing.assert_close(logits, parents[source](batch)[0], rtol=0, atol=0)
            torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
    freeze(ART / "composition_verification.json", {
        "positions": len(samples), "sources": sources, "device": "cpu", "rtol": 0, "atol": 0,
        "policy_matches_source_exactly": True, "all_three_values_match_original_round30_exactly": True,
        "scope": "Real beginning/middle/end decisions from three original whole-game holdouts; complete independent encoders, no head transplant or training."})


def prepare():
    parents, identities, models = {}, {}, {}
    for name, (relative, expected_sha) in SOURCES.items():
        path = ROOT / relative
        if sha256(path) != expected_sha:
            raise ValueError(f"Source checkpoint changed: {name}")
        identities[name] = identity(path)
        parents[name] = load_frozen_actor_critic_checkpoint(path).eval().requires_grad_(False)
    template_path = PREVIOUS / "baseline/evaluation/search.json"
    template = read_json(template_path)["config"]
    panels = []
    for name, policy, label in POLICY_PANELS:
        path = MODELS / f"{name}.pt"
        sources = {"policy": identities[policy], "value": identities["round30"]}
        if not path.exists():
            model = EntityAttentionPolicyValue().eval().requires_grad_(False)
            model.policy_model.load_state_dict(parents[policy].state_dict(), strict=True)
            model.value_model.load_state_dict(parents["round30"].state_dict(), strict=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"checkpoint_schema": "splendor-frozen-actor-critic-v1",
                "format_version": MODEL_WEIGHTS_FORMAT_VERSION, "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "config": asdict(model.config), "state_dict": model.state_dict(),
                "metadata": {"rules_version": RULESET_ID, "sources": sources, "inference_only": True,
                    "composition": "complete independent policy encoder and value encoder; value fixed to original round30"}}, path)
        model = load_frozen_actor_critic_checkpoint(path).eval()
        for branch, parent in (("policy", policy), ("value", "round30")):
            actual, expected = getattr(model, f"{branch}_model").state_dict(), parents[parent].state_dict()
            if actual.keys() != expected.keys() or any(not torch.equal(actual[k], expected[k]) for k in expected):
                raise ValueError(f"Composition source parameters differ: {name}/{branch}")
        models[name] = model
        config = copy.deepcopy(template)
        config["policy_a"].update(checkpoint=str(path), name=name)
        config.update(game_seeds=list(range(32700000, 32700200)), policy_seed_base=32710000,
                      bootstrap_seed=32720000)
        for side in ("policy_a", "policy_b"):
            config[side]["search_config"]["seed"] = 32730000
        panels.append((name, label, config, {"candidate": identity(path), "sources": sources,
                                           "opponent": identities["P4"]}))
    verify_outputs(parents, models)
    config = copy.deepcopy(template)
    config["policy_a"].update(checkpoint=str(ROOT / SOURCES["round24"][0]), name="round24_full")
    config["policy_b"].update(checkpoint=str(ROOT / SOURCES["round30"][0]), name="round30_full")
    config.update(game_seeds=list(range(32800000, 32800200)), policy_seed_base=32810000,
                  bootstrap_seed=32820000)
    for side in ("policy_a", "policy_b"):
        config[side]["search_config"]["seed"] = 32830000
    panels.append(("S04_R24_vs_R30", "第24轮完整网络 vs 原第30轮完整网络", config,
                   {"candidate": identities["round24"], "opponent": identities["round30"]}))
    frozen_panels = []
    for name, label, config, model_identity in panels:
        config["metadata"] = {"experiment": ART.name, "panel": name, "models": model_identity,
            "scope": "Fresh shared DEV openers for S01-S03; separate fresh direct-arena openers for S04; no training"}
        config = FixedEvaluationConfig.from_dict(config).to_dict()
        path = ART / "configs" / f"{name}.json"
        freeze(path, config)
        frozen_panels.append({"id": name, "label": label, "pairs": 200, "config": identity(path),
            "models": model_identity, "output": str((ART / "panels" / f"{name}.json").relative_to(ROOT))})
    plan = {"experiment": ART.name, "sources": identities, "panels": frozen_panels,
        "runner": identity(Path(__file__)), "template": identity(template_path),
        "composition_verification": identity(ART / "composition_verification.json"),
        "settings": "All S1024/N1, c_puct1.5, no root noise, native_batched/compiled, active100, max_batch64, paired seat swaps; four panels run sequentially in separate processes",
        "scope": "S01-S03 compare policies trained with value weights1/.25/0 using the same original round30 COMPLETE value network. Each is rerun on fresh shared 200-pair DEV openers versus P4. S04 directly tests round24(A) versus original round30(B) on separate fresh 200 pairs. Single training trajectory; exploratory paired comparisons without multiplicity correction."}
    freeze(PLAN, plan)
    return plan


def summarize(panel):
    path = ROOT / panel["output"]
    row = {"id": panel["id"], "label": panel["label"], "status": "pending", "completed_pairs": 0}
    if not path.exists():
        return row
    raw = read_json(path)
    config = read_json(ROOT / panel["config"]["path"])
    if raw["config"] != config:
        raise ValueError("Saved arena configuration differs")
    row.update(status="incomplete", completed_pairs=raw["completed_pairs"])
    if raw["completed_pairs"] != 200:
        return row
    seeds, scores = serialized_pair_scores(raw)
    if seeds != tuple(config["game_seeds"]) or raw["valid_games"] != 400:
        raise ValueError("Completed arena population differs")
    for side, source in (("policy_a", "candidate"), ("policy_b", "opponent")):
        if raw["checkpoint_identity"][side]["sha256"] != panel["models"][source]["sha256"]:
            raise ValueError("Evaluated checkpoint differs")
    ci = paired_bootstrap_mean(scores, resamples=10000, seed=config["bootstrap_seed"]).to_dict()
    if ci != raw["paired_bootstrap"]:
        raise ValueError("Paired confidence interval differs")
    for pair in raw["pairs"]:
        for game in pair["games"]:
            for side in ("policy_a_stats", "policy_b_stats"):
                s = game[side]
                if s["simulations"] != 1024 * s["searchable_decisions"] or s["determinizations"] != s["searchable_decisions"]:
                    raise ValueError("Actual S1024/N1 budget differs")
    row.update(status="complete", score=raw["win_rate_score"], ci95=ci, result=identity(path),
               **{key: raw[key] for key in ("a_wins", "draws", "a_losses", "a_first", "a_second",
                    "exceptions", "illegal_actions", "engineering_truncations", "invariant_failures",
                    "elapsed_seconds_this_invocation")})
    return row


def aggregate(plan):
    rows = [summarize(panel) for panel in plan["panels"]]
    complete = {row["id"]: row for row in rows if row["status"] == "complete"}
    contrasts = {}
    for a, b in ((1, 0), (2, 0), (2, 1)):
        a_id, b_id = POLICY_PANELS[a][0], POLICY_PANELS[b][0]
        if a_id in complete and b_id in complete:
            a_seeds, a_scores = serialized_pair_scores(read_json(ROOT / complete[a_id]["result"]["path"]))
            b_seeds, b_scores = serialized_pair_scores(read_json(ROOT / complete[b_id]["result"]["path"]))
            contrasts[f"{a_id}_minus_{b_id}"] = paired_bootstrap_difference(
                a_seeds, a_scores, b_seeds, b_scores, resamples=10000, seed=32740000).to_dict()
    result = {"status": "complete" if len(complete) == 4 else "incomplete", "completed_panels": len(complete),
              "plan_sha256": sha256(PLAN), "panels": rows, "paired_score_differences": contrasts,
              "scope": plan["scope"]}
    write_json(ART / "summary.json", result)
    lines = ["# 固定第30轮value的policy对照，以及第24轮对第30轮", "",
             f"状态：`{result['status']}`；完成{len(complete)}/4组。", "",
             "每组200 pair／400局，S1024/N1；和棋半分，95% CI按整pair bootstrap。前三组重新跑相同的新开局面板；第四组是直接对战，得分按第24轮计算。", "",
             "| 顺序／组合 | 完整pair | 胜／和／负 | A得分 | 95% CI |",
             "| --- | ---: | --- | ---: | --- |"]
    for row in rows:
        if row["status"] == "complete":
            ci = row["ci95"]
            lines.append(f"| {row['label']} | 200 | {row['a_wins']}/{row['draws']}/{row['a_losses']} | {100*row['score']:.3f}% | [{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%] |")
        else:
            lines.append(f"| {row['label']} | {row['completed_pairs']} | — | — | — |")
    lines += ["", "前三组同开局、同P4对手面板的配对得分差（不是候选之间直接互打）：", ""]
    for name, ci in contrasts.items():
        lines.append(f"- `{name}`：{100*ci['estimate']:+.3f}个百分点，95% CI [{100*ci['lower']:+.3f}, {100*ci['upper']:+.3f}]。")
    lines += ["", "value固定完整原第30轮encoder＋head。policy保留各自完整encoder及动作查询模块；不做head移植，也不训练新参数。权重1同样使用双完整网络组合执行，且本次重新对战，未复用旧200pair结果。", "",
              "前三组检验在同一value评估器下policy分支的搜索表现；第四组检验第24→30轮两个完整网络的直接强弱。仍限定单条训练轨迹与本次开局面板，未校正多重比较。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    index_path = ROOT / "artifacts/experiment_index.json"
    index = read_json(index_path)
    entry_id = "entity-attention-r30-fixed-value-arena-2026-10-11"
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
            for path in (Path(__file__), PLAN, ART / "composition_verification.json", *sorted((ART / "configs").glob("*.json"))):
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
                if summarize(panel)["status"] == "complete":
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
