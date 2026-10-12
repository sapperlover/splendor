"""Sequential, resumable search ablations and greedy policy comparisons."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
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
PREVIOUS = ROOT / "artifacts/entity_attention_recent_pool_20261009"
RUNTIME = ROOT / "runtime" / ART.name
MODELS = ROOT / "runs" / ART.name / "compositions"
PLAN = ART / "plan.json"
LIVE = RUNTIME / "live.json"
SUMMARY = ART / "summary.json"
PROTOCOL = "recent-pool-crossplay-20261010-v1"
BASES = {
    "P4V4": ("runs/entity_attention_recent_pool_20261009/checkpoints/step-000034398.pt", "94f4ce89d27f42ec0c39e72055b460405ae36b8f9931051cbe360ee145ace5ec"),
    "P8V8": ("runs/entity_attention_recent_pool_20261009/checkpoints/step-000068796.pt", "81f1183efc6c550a79204d8e1d1e5b7705e3306bf912041d4d7b9a47d2ec1dba"),
    "P12V12": ("runs/entity_attention_recent_pool_20261009/checkpoints/final.pt", "9faa66af917c597fbc62b823eeffea14e8403391415ce552d7a4114481d2784a"),
    "theta5000": ("runs/entity_attention_hybrid_static_graph_seed1/checkpoints/theta_5000.pt", "6fec2039cfe5fa7e8775ec5b5692a5defadfa3ab21bdb398e400696be17b08db"),
    "offline9500": ("runs/entity_attention_offline_9500_replay/checkpoints/step-000009500.pt", "a179d043b35b302a4c5edaff7a37f00a2f77a8510925817c19f4c4b1d72835b0"),
    "hybrid_r6": ("runs/entity_attention_hybrid_static_graph_seed1/rounds/round-005/checkpoints/final.pt", "909c5007beb9d5018f3e3e9f20c5c1f3edca34d4f3dabbd48b4d03ed33b4d68c"),
    "hybrid_r9": ("runs/entity_attention_hybrid_static_graph_seed1/rounds/round-008/checkpoints/final.pt", "af3b2b64007705cc9adcbb3e8a6ad68239f4dc967c31c515e857b1240479e44a"),
}
SEARCH_PANELS = (("S01_P12V4_vs_P4V4", "P12V4", "P4V4"),
                 ("S02_P4V12_vs_theta5000", "P4V12", "theta5000"),
                 ("S03_P8V4_vs_theta5000", "P8V4", "theta5000"))
GREEDY_PANELS = tuple((f"G{i:02d}_{a}_vs_{b}", a, b) for i, (a, b) in enumerate((
    ("P4V4", "theta5000"), ("P8V8", "theta5000"), ("P12V12", "theta5000"),
    ("offline9500", "theta5000"), ("hybrid_r6", "theta5000"), ("hybrid_r9", "theta5000"),
    ("P12V12", "P4V4"), ("P8V8", "P4V4"), ("P12V12", "P8V8"), ("hybrid_r9", "hybrid_r6"),
), 1))
child = None
stop_requested = False


def freeze(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"frozen experiment identity changed: {path}")
    if not path.exists():
        write_json(path, value)


def prepare_models():
    identities, parents = {}, {}
    for name, (relative, expected) in BASES.items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise ValueError(f"source checkpoint changed: {name}")
        identities[name] = {"checkpoint": relative, "sha256": expected}
        parents[name] = load_frozen_actor_critic_checkpoint(path)
    for name, policy, value in (("P12V4", "P12V12", "P4V4"),
                                ("P4V12", "P4V4", "P12V12"), ("P8V4", "P8V8", "P4V4")):
        path = (ROOT / "runs/entity_attention_recent_pool_20261009/compositions/final_policy_early_value.pt"
                if name == "P12V4" else MODELS / f"{name}.pt")
        sources = {"policy": identities[policy], "value": identities[value]}
        if not path.exists():
            model = EntityAttentionPolicyValue().eval().requires_grad_(False)
            model.policy_model.load_state_dict(parents[policy].state_dict(), strict=True)
            model.value_model.load_state_dict(parents[value].state_dict(), strict=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"checkpoint_schema": "splendor-frozen-actor-critic-v1", "format_version": MODEL_WEIGHTS_FORMAT_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION, "config": asdict(model.config),
                "state_dict": model.state_dict(), "metadata": {"rules_version": RULESET_ID, "sources": sources,
                "inference_only": True, "composition": "complete separate encoders; policy from one network, value from the other"}}, path)
        model = load_frozen_actor_critic_checkpoint(path)
        for branch, parent in (("policy", policy), ("value", value)):
            actual, expected = getattr(model, f"{branch}_model").state_dict(), parents[parent].state_dict()
            if actual.keys() != expected.keys() or any(not torch.equal(actual[k], expected[k]) for k in expected):
                raise ValueError(f"composition source weights differ: {name}/{branch}")
        identities[name] = {"checkpoint": str(path.relative_to(ROOT)), "sha256": sha256(path), "sources": sources}
    return identities


def prepare():
    models = prepare_models()
    template_path = PREVIOUS / "early_battle_config.json"
    template = read_json(template_path)
    panels = []
    for kind, matchups in (("search", SEARCH_PANELS), ("greedy", GREEDY_PANELS)):
        for name, a, b in matchups:
            config = copy.deepcopy(template)
            fresh = kind == "greedy" or name.startswith("S01")
            config.update(seed_start=30500000 if fresh else 14900000,
                policy_seed_base=30510000 if fresh else 14910000, bootstrap_seed=30520000 if fresh else 14920000)
            if kind == "greedy":
                config.pop("execution")
                config.update(device="cpu", workers=8)
            for side, model in (("policy_a", a), ("policy_b", b)):
                if kind == "search":
                    config[side]["checkpoint"] = str(ROOT / models[model]["checkpoint"])
                    config[side]["name"] = model
                else:
                    config[side] = {"kind": "greedy_checkpoint", "name": model,
                                    "checkpoint": str(ROOT / models[model]["checkpoint"])}
            config["metadata"] = {"experiment": ART.name, "protocol": PROTOCOL, "panel": name,
                "kind": kind, "models": {"policy_a": models[a], "policy_b": models[b]},
                "seed_scope": "fresh fixed crossplay seeds" if fresh else "same DEV seeds as existing theta5000 search panels"}
            FixedEvaluationConfig.from_dict(config)
            config_path = ART / "configs" / f"{name}.json"
            freeze(config_path, config)
            panels.append({"id": name, "kind": kind, "a": a, "b": b, "pairs": 200,
                "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
                "output": str((ART / "panels" / f"{name}.json").relative_to(ROOT))})
    manifest = PREVIOUS / "pool_games.jsonl"
    if sha256(manifest) != "d5d49d8e019767b90e6b2eb5457f0a5f74dbab1f116b10ecbcce06fc3675fe6b":
        raise ValueError("original pool manifest changed")
    pool_seeds = {json.loads(line)["game_seed"] for line in manifest.open()}
    if pool_seeds.intersection((*range(30500000, 30500200), *range(14900000, 14900200))):
        raise ValueError("evaluation seed overlaps original pool")
    plan = {"protocol": PROTOCOL, "models": models, "panels": panels, "script_sha256": sha256(Path(__file__)),
        "template_sha256": sha256(template_path), "manifest_sha256": sha256(manifest), "pool_seed_overlap": 0,
        "fresh_seed_range": [30500000, 30500199], "reused_dev_seed_range": [14900000, 14900199],
        "search": "native S1024/N1, compiled, no root noise, temperature0; one CUDA owner, active100, batch64",
        "greedy": "argmax policy; CPU8 workers, one Torch thread each; value variants with identical policy are deduplicated",
        "scope": "fixed matchups; one training trajectory; theta5000 contrasts reuse DEV, direct search and greedy use fresh openers; no training"}
    freeze(PLAN, plan)
    return plan


def summarize_panel(panel):
    path = ROOT / panel["output"]
    if not path.exists():
        return {"id": panel["id"], "status": "pending", "kind": panel["kind"], "a": panel["a"], "b": panel["b"], "completed_pairs": 0}
    raw = read_json(path)
    config = FixedEvaluationConfig.from_dict(read_json(ROOT / panel["config"])).to_dict()
    if raw["config"] != config:
        raise ValueError(f"result config differs: {panel['id']}")
    row = {"id": panel["id"], "kind": panel["kind"], "a": panel["a"], "b": panel["b"],
        "completed_pairs": raw["completed_pairs"], "completed_games": raw["completed_games"],
        "score": raw["win_rate_score"], "status": "incomplete"}
    if raw["completed_pairs"] != panel["pairs"]:
        return row
    seeds, values = serialized_pair_scores(raw)
    if seeds != tuple(config["game_seeds"]) or raw["completed_games"] != 2 * panel["pairs"]:
        raise ValueError("completed panel population mismatch")
    for side, identity in raw["checkpoint_identity"].items():
        if identity["path"] != config[side]["checkpoint"] or sha256(Path(identity["path"])) != identity["sha256"]:
            raise ValueError("completed panel checkpoint mismatch")
    ci = paired_bootstrap_mean(values, resamples=config["bootstrap_resamples"], seed=config["bootstrap_seed"]).to_dict()
    if ci != raw["paired_bootstrap"]:
        raise ValueError("completed panel confidence interval mismatch")
    if panel["kind"] == "search":
        for pair in raw["pairs"]:
            for game in pair["games"]:
                for side in ("policy_a_stats", "policy_b_stats"):
                    stats = game[side]
                    if stats["simulations"] != stats["searchable_decisions"] * 1024 or stats["determinizations"] != stats["searchable_decisions"]:
                        raise ValueError("actual S1024/N1 budget mismatch")
    row.update(status="complete", ci95=ci, result=panel["output"], result_sha256=sha256(path),
        completed_at=raw["updated_at"], elapsed_seconds_this_invocation=raw["elapsed_seconds_this_invocation"],
        **{k: raw[k] for k in ("a_wins", "draws", "a_losses", "a_first", "a_second", "engineering_truncations",
                              "exceptions", "illegal_actions", "invariant_failures", "policy_cycle_draws")})
    return row


def aggregate(plan):
    rows = [summarize_panel(panel) for panel in plan["panels"]]
    result = {"protocol": PROTOCOL, "plan_sha256": sha256(PLAN),
        "status": "complete" if all(row["status"] == "complete" for row in rows) else "incomplete",
        "completed_panels": sum(row["status"] == "complete" for row in rows), "total_panels": len(rows), "panels": rows,
        "scope": plan["scope"], "matched_theta5000_search_contrasts": {}}
    for row in rows:
        if row["status"] != "complete" or row["kind"] != "search" or row["b"] != "theta5000":
            continue
        raw = read_json(ROOT / row["result"])
        seeds, values = serialized_pair_scores(raw)
        contrasts = {}
        for i, filename in enumerate(("early_step000034398_vs_theta5000.json", "final103194_vs_theta5000.json", "final_policy_early_value_vs_theta5000.json")):
            previous_path = PREVIOUS / "battles" / filename
            old = read_json(previous_path)
            if old["checkpoint_identity"]["policy_b"] != raw["checkpoint_identity"]["policy_b"]:
                raise ValueError("matched historical opponent differs")
            for key in ("policy_seed_base", "game_seeds", "execution", "max_completed_turns", "adjudicate_deterministic_policy_cycles"):
                if old["config"][key] != raw["config"][key]:
                    raise ValueError(f"matched historical configuration differs: {key}")
            for side in ("policy_a", "policy_b"):
                if old["config"][side]["search_config"] != raw["config"][side]["search_config"]:
                    raise ValueError("matched historical search differs")
            old_seeds, old_values = serialized_pair_scores(old)
            contrasts[previous_path.stem] = {**paired_bootstrap_difference(seeds, values, old_seeds, old_values,
                resamples=10000, seed=30530000+i).to_dict(), "source": str(previous_path.relative_to(ROOT)), "source_sha256": sha256(previous_path)}
        result["matched_theta5000_search_contrasts"][row["id"]] = contrasts
    write_json(SUMMARY, result)
    publish(result)
    return result


def publish(result):
    lines = ["# 固定对战队列结果", "", f"状态：`{result['status']}`；完成 {result['completed_panels']}/{result['total_panels']} 组。",
        "", "每组200 pair／400局，和棋半分；CI按整pair bootstrap。Greedy只取决于policy。未完成组不报告最终分数。", "",
        "| 组 | A | B | 完整pair | A胜／和／负 | A得分 | 95% CI | 循环裁和 |", "| --- | --- | --- | ---: | --- | ---: | --- | ---: |"]
    for row in result["panels"]:
        if row["status"] == "complete":
            ci = row["ci95"]
            label = f"[{row['id']}](panels/{row['id']}.json)"
            cells = (f"{row['a_wins']}/{row['draws']}/{row['a_losses']}", f"{100*row['score']:.3f}%",
                     f"[{100*ci['lower']:.3f}%,{100*ci['upper']:.3f}%]", str(row["policy_cycle_draws"]))
        else:
            label, cells = row["id"], ("—", "—", "—", "—")
        lines.append(f"| {label} | {row['a']} | {row['b']} | {row['completed_pairs']}/200 | {' | '.join(cells)} |")
    lines += ["", "## 对θ5000的同种子搜索得分差", "", "差值为新面板减旧面板，对手均为θ5000。S01直接互打单独见上表。", "",
              "| 新面板 | 旧面板 | 得分差（百分点） | 配对差95% CI（百分点） |", "| --- | --- | ---: | --- |"]
    for panel, comparisons in result["matched_theta5000_search_contrasts"].items():
        for name, ci in comparisons.items():
            lines.append(f"| {panel} | {name} | {100*ci['estimate']:+.3f} | [{100*ci['lower']:+.3f},{100*ci['upper']:+.3f}] |")
    lines += ["", "固定训练轨迹与对手的探索性结果；θ5000搜索对照复用DEV，未作多重比较校正。新开局互打独立于旧DEV及原训练池，但没有覆盖训练seed变异。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    index_path = ROOT / "artifacts/experiment_index.json"
    index = read_json(index_path)
    entry_id = "entity-attention-recent-pool-crossplay-2026-10-10"
    entry = next((e for e in index["entries"] if e["id"] == entry_id), None)
    if entry is None:
        entry = {"id": entry_id}
        index["entries"].append(entry)
    evidence = [ART / "README.md", Path(__file__), PLAN, SUMMARY, ART / "results.md", *sorted((ART / "configs").glob("*.json"))]
    evidence += [p for p in (ART / "smoke.json", ART / "execution.json") if p.exists()]
    evidence += [ROOT / r["result"] for r in result["panels"] if r["status"] == "complete"]
    paths = [str(p.relative_to(ROOT)) for p in evidence]
    entry.update(status="valid" if result["status"] == "complete" else "incomplete",
        report=str((ART / "README.md").relative_to(ROOT)), evidence=paths,
        evidence_hashes={str(p.relative_to(ROOT)): sha256(p) for p in evidence},
        scope=f"Fixed queue of three 200-pair native S1024/N1 search panels and ten 200-pair CPU8 greedy panels; {result['completed_panels']}/13 complete. "
              "P12V4 versus P4V4 directly uses fresh openers; P4V12 and P8V4 versus theta5000 reuse prior DEV seeds. "
              "Greedy covers P4/P8/P12, offline9500, hybrid r6/r9 versus theta5000, three P4/P8/P12 head-to-heads, and hybrid r9 versus r6. "
              "Independent complete source encoders for policy/value composition; no training, no value-only greedy duplicates. "
              "Finished results validate model/config identities, swapped seats, paired bootstrap and actual search budgets; unfinished panels are not strength evidence. "
              "Results and matched theta5000 contrasts auto-published, exploratory without multiplicity correction or training-seed uncertainty.")
    write_json(index_path, index)


def interrupt(signum, frame):
    global stop_requested
    stop_requested = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def evaluate(config, output, logfile, session):
    global child
    if stop_requested:
        return 130
    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(config), "--output", str(output),
               "--progress-every", "10", "--summary-every", "10"]
    if output.exists() or output.with_suffix(output.suffix + ".games").exists():
        command.append("--resume")
    write_json(session / f"command-{output.stem}-{time.time_ns()}.json", {"command": command, "start_unix": time.time()})
    with logfile.open("ab", buffering=0) as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(LIVE, {"protocol": PROTOCOL, "status": "running", "pid": os.getpid(), "child_pid": child.pid,
                         "current_panel": output.stem, "log": str(logfile), "time_unix": time.time()})
        code = child.wait()
        child = None
    if stop_requested or code == 130:
        return 130
    if code:
        raise RuntimeError(f"evaluation exited {code}; inspect {logfile}")
    return 0


def snapshot():
    session = RUNTIME / str(time.time_ns())
    identity = source_identity(ROOT, session)
    for path in (Path(__file__), PLAN, *sorted((ART / "configs").glob("*.json"))):
        target = session / path.relative_to(ART)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    record = {**identity, "session": str(session.relative_to(ROOT)), "environment": environment_identity(),
              "command": [sys.executable, *sys.argv], "start_unix": time.time(), "plan_sha256": sha256(PLAN)}
    write_json(session / "identity.json", record)
    execution = read_json(ART / "execution.json") if (ART / "execution.json").exists() else {"protocol": PROTOCOL, "sessions": []}
    execution["sessions"].append(record)
    write_json(ART / "execution.json", execution)
    return session


def smoke(plan, session):
    results = []
    for kind, panel_index in (("search", 1), ("greedy", 3)):
        config = read_json(ROOT / plan["panels"][panel_index]["config"])
        config.update(pairs=2, seed_start=30600000, bootstrap_resamples=1000)
        config["metadata"] = {"protocol": PROTOCOL, "purpose": "finite smoke; excluded from formal panels"}
        if kind == "search":
            # Exercise both newly composed branches in the actual native/compiled path.
            config["policy_b"] = copy.deepcopy(config["policy_a"])
            config["policy_b"].update(name="P8V4", checkpoint=str(ROOT / plan["models"]["P8V4"]["checkpoint"]))
            for side in ("policy_a", "policy_b"):
                config[side]["search_config"]["simulations"] = 16
            config["execution"]["active_games"] = 4
        config_path, output = RUNTIME / f"smoke-{kind}-config.json", RUNTIME / f"smoke-{kind}.json"
        freeze(config_path, config)
        if evaluate(config_path, output, RUNTIME / f"smoke-{kind}.log", session):
            return 130
        raw = read_json(output)
        serialized_pair_scores(raw)
        if raw["completed_pairs"] != 2 or raw["completed_games"] != 4:
            raise ValueError("finite smoke did not complete")
        before = [(p["game_seed"], [g["final_state_digest"] for g in p["games"]]) for p in raw["pairs"]]
        if evaluate(config_path, output, RUNTIME / f"smoke-{kind}.log", session):
            return 130
        after = read_json(output)
        if before != [(p["game_seed"], [g["final_state_digest"] for g in p["games"]]) for p in after["pairs"]]:
            raise ValueError("completed smoke pairs changed on resume")
        results.append({"kind": kind, "pairs": 2, "passed": True, "resume_preserves_games": True,
                        "config": config, "output": str(output.relative_to(ROOT)), "output_sha256": sha256(output)})
    write_json(ART / "smoke.json", {"protocol": PROTOCOL, "plan_sha256": sha256(PLAN), "passed": True, "results": results})
    print("Finite native/compiled composition and CPU8 greedy smoke passed; completed-pair resume preserved.", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "run", "status", "analyze", "stop"), nargs="?", default="run")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        plan = read_json(PLAN)
        rows = [summarize_panel(p) for p in plan["panels"]]
        print(json.dumps({"live": read_json(LIVE) if LIVE.exists() else None, "panels": rows}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "stop":
        live = read_json(LIVE)
        if live["status"] != "running":
            print(f"runner status: {live['status']}")
            return 0
        command = Path(f"/proc/{live['pid']}/cmdline").read_bytes().split(b"\0")
        if not any(item and Path(os.fsdecode(item)).resolve() == Path(__file__).resolve() for item in command):
            raise ValueError("recorded PID is not this experiment runner")
        os.kill(live["pid"], signal.SIGTERM)
        print("Graceful stop requested; completed pairs are preserved.")
        return 0
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        try:
            plan = prepare()
            if args.action == "prepare":
                aggregate(plan)
                return 0
            if args.action == "analyze":
                print(json.dumps(aggregate(plan), ensure_ascii=False, indent=2))
                return 0
            session = snapshot()
            if args.action == "smoke":
                code = smoke(plan, session)
                aggregate(plan)
                write_json(LIVE, {"protocol": PROTOCOL, "status": "smoke_complete" if code == 0 else "interrupted",
                                  "pid": os.getpid(), "time_unix": time.time()})
                return code
            if not (ART / "smoke.json").exists() or read_json(ART / "smoke.json")["plan_sha256"] != sha256(PLAN):
                code = smoke(plan, session)
                if code:
                    write_json(LIVE, {"protocol": PROTOCOL, "status": "interrupted", "pid": os.getpid(), "time_unix": time.time()})
                    return code
            aggregate(plan)
            for panel in plan["panels"]:
                if stop_requested:
                    break
                row = summarize_panel(panel)
                if row["status"] == "complete":
                    print(f"SKIP complete {panel['id']}: {row['score']:.5f}", flush=True)
                    continue
                print(f"START {panel['id']} (200 pairs)", flush=True)
                if sha256(ROOT / panel["config"]) != panel["config_sha256"]:
                    raise ValueError("frozen panel config changed")
                code = evaluate(ROOT / panel["config"], ROOT / panel["output"], RUNTIME / f"{panel['id']}.log", session)
                result = aggregate(plan)
                if code:
                    break
                row = next(r for r in result["panels"] if r["id"] == panel["id"])
                if row["status"] != "complete":
                    raise ValueError("evaluator exited before all pairs completed")
                print(f"DONE {panel['id']}: score={row['score']:.5f} CI={row['ci95']['lower']:.5f}..{row['ci95']['upper']:.5f}", flush=True)
            result = aggregate(plan)
            status = "complete" if result["status"] == "complete" else "interrupted"
            write_json(LIVE, {"protocol": PROTOCOL, "status": status, "pid": os.getpid(), "time_unix": time.time()})
            return 0 if status == "complete" else 130
        except Exception as error:
            write_json(LIVE, {"protocol": PROTOCOL, "status": "failed", "pid": os.getpid(),
                             "error": f"{type(error).__name__}: {error}", "time_unix": time.time()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
