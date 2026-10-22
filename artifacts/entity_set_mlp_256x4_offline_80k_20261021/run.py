"""Prepare or run current set-MLP offline fitting, validation selection and R20 arenas."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter

from splendor.actor_critic import build_actor_critic
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import _read_game_metadata
from splendor.catalog import load_default_catalog
from splendor.evaluation import FixedEvaluationConfig
from splendor.offline_mixed_budget import MixedBudgetOfflineConfig
from splendor.offline_scratch import validation_schedule
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
OUTPUT = ROOT / "runs" / ART.name
RUNTIME = ROOT / "runtime" / ART.name
SOURCE_RUN = ROOT / "runs/entity_attention_37423_random_budget_p6k_v24k_20261019"
CONFIG = ROOT / "configs/entity_set_mlp_v1/offline_80k_lr1e4_20261021.json"
PRESET = ROOT / "configs/entity_set_mlp_v1/model_256x4.json"
PREVIOUS_CONFIG = ROOT / "configs/entity_attention_v1/offline_128x2_relations_no_aux_r24_r43_20261019.json"
OPPONENT = SOURCE_RUN / "rounds/round-019/checkpoints/final.pt"
OPPONENT_SHA = "fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd"
TEMPLATE = ROOT / "artifacts/entity_attention_r20_s1024n1_vs_greedy_20261021/config.json"
PANELS = (
    {"id": "greedy_vs_R20", "kind": "greedy", "pairs": 1000, "seed_start": 43200000},
    {"id": "search_S1024N1_vs_R20", "kind": "search", "pairs": 500, "seed_start": 43300000},
)
child = None
stopping = False


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment input changed: {path}")
    else:
        write_json(path, value)


def source_games():
    completed = {r["iteration"]: r for r in read_json(SOURCE_RUN / "state.json")["completed_rounds"]}
    records, sources = [], []
    for iteration in range(1, 21):
        if iteration not in completed:
            raise ValueError(f"Source iteration {iteration} is incomplete")
        stage = SOURCE_RUN / "rounds" / f"round-{iteration-1:03d}"
        index_path = stage / "collection_index.json"
        manifest_path, summary_path = stage / "selfplay/manifest.json", stage / "selfplay/summary.json"
        manifest, summary = read_json(manifest_path), read_json(summary_path)
        expected_teacher = (read_json(SOURCE_RUN / "config.json")["initial_checkpoint"] if iteration == 1
            else completed[iteration - 1]["checkpoint"])
        if manifest["checkpoint_identity"]["sha256"] != expected_teacher["sha256"]:
            raise ValueError("Collection teacher differs from the completed source lineage")
        rows = read_json(index_path)["games"]
        if (len(rows) != 4000 or summary["completed_games"] != 4000
                or summary["expected_games"] != 4000 or summary["failed_games"] or summary["missing_games"]):
            raise ValueError(f"Source iteration {iteration} lacks 4000 complete games")
        if {r["round_index"] for r in rows} != {iteration + 42}:
            raise ValueError("Original absolute generation identity differs")
        if len({r["game_id"] for r in rows}) != 4000:
            raise ValueError("Duplicate source game identity")
        for row in rows:
            stat = Path(row["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
                raise ValueError(f"Source archive changed: {row['path']}")
        metadata = _read_game_metadata(Path(rows[0]["path"]))
        sources.append({"iteration": iteration,
            "index": identity(index_path), "manifest": identity(manifest_path), "summary": identity(summary_path),
            "teacher": manifest["checkpoint_identity"],
            "historical_catalog_fingerprint": metadata["trace"]["catalog_fingerprint"]})
        records.extend(rows)
    return sorted(records, key=lambda r: (r["round_index"], r["game_id"])), sources


def prepare():
    if sha256(OPPONENT) != OPPONENT_SHA:
        raise ValueError("R20 opponent checkpoint identity changed")
    records, sources = source_games()
    manifest = OUTPUT / "inputs/pool_games.jsonl"
    content = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records)
    if manifest.exists():
        if manifest.read_text() != content:
            raise ValueError("Frozen 80k pool differs")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(content)
    raw = read_json(PREVIOUS_CONFIG)
    raw.update(manifest=str(manifest), manifest_sha256=sha256(manifest),
        model_config=read_json(PRESET), first_round=44, last_round=63, expected_games_per_round=4000,
        initialization_seed=43100000, sampling_seed=43110000, validation_seed=43120000,
        value_auxiliary=None, sampling_protocol="policy-high-value-all-v1", cache_mib=16384)
    raw["learner"]["learning_rate"] = 1e-4
    freeze(CONFIG, raw)
    config = MixedBudgetOfflineConfig.from_dict(raw)
    counts = Counter(r["round_index"] for r in records)
    if set(counts.values()) != {config.expected_games_per_round}:
        raise ValueError("Source generation count differs")
    splits = {split: {"games": sum(g["split"] == split for g in records),
        "positions": sum(g["decision_count"] for g in records if g["split"] == split),
        "policy_positions": sum(g["policy_samples"] for g in records if g["split"] == split)}
        for split in ("train", "validation")}
    steps = math.ceil(config.draw_exposures * splits["train"]["positions"] / config.learner.batch_size)
    catalog = load_default_catalog()
    if len(catalog.nobles) != 12:
        raise ValueError("Arenas require the current 12-noble rules")
    catalog_files = {}
    for path in sorted((ROOT / "data").glob("splendor_*")):
        if path.suffix not in (".csv", ".json"):
            continue
        target = ART / "catalog" / path.name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if sha256(target) != sha256(path):
                raise ValueError("Frozen evaluation catalog changed")
        else:
            shutil.copyfile(path, target)
        catalog_files[path.name] = sha256(target)
    plan = {"experiment": ART.name, "config": identity(CONFIG), "runner": identity(Path(__file__)),
        "preset": identity(PRESET), "model_implementation": identity(ROOT / "src/splendor/entity_set_mlp.py"),
        "opponent": identity(OPPONENT), "search_template": identity(TEMPLATE), "source_iterations": [1, 20],
        "absolute_rounds": [44, 63], "games": len(records), "splits": splits, "source_rounds": sources,
        "model_config": config.model_config, "parameters": build_actor_critic(config.model_config).trainable_parameter_count(),
        "initialization": "random current 256x4 set-MLP, nonlinear action branch and multiplicative interaction; fresh AdamW",
        "steps_target": steps, "draw_exposures": config.draw_exposures, "cache_mib": config.cache_mib,
        "sampling": {"policy": "256 positive-weight high-budget decisions", "value": "256 all-budget decisions",
            "update": "one joint update, shared encoder, one global clip",
            "policy_exposure": steps*256/splits["train"]["policy_positions"],
            "value_exposure": steps*256/splits["train"]["positions"]},
        "validation_percent_step": config.validation_percent_step, "validation_steps": validation_schedule(config, steps),
        "selection": "minimum full-holdout policy KL + win/loss value MSE over the 20 percentage checkpoints; ties earlier step",
        "panels": list(PANELS), "catalog_fingerprint": catalog.data_fingerprint, "catalog_files": catalog_files,
        "scope": "Prepared only until run; original archived targets and whole-game splits, no relabeling, no PSW or auxiliary heads. Historical games use the 10-noble pool; arenas use the frozen current 12-noble pool."}
    freeze(ART / "plan.json", plan)
    return plan


def evaluation_config(plan, selection, panel):
    candidate = Path(selection["checkpoint"])
    if sha256(candidate) != selection["checkpoint_sha256"] or sha256(OPPONENT) != plan["opponent"]["sha256"]:
        raise ValueError("Evaluation checkpoint identity changed")
    raw = copy.deepcopy(read_json(TEMPLATE))
    seed = panel["seed_start"]
    raw.update(game_seeds=list(range(seed, seed + panel["pairs"])), policy_seed_base=seed+10000,
        bootstrap_seed=seed+20000, bootstrap_resamples=10000)
    raw.pop("pairs", None)
    raw.pop("seed_start", None)
    for side, path, name in (("policy_a", candidate, "selected_set_mlp"), ("policy_b", OPPONENT, "R20_raw")):
        if panel["kind"] == "greedy":
            raw[side] = {"kind": "greedy_checkpoint", "name": name, "checkpoint": str(path)}
        else:
            spec = copy.deepcopy(read_json(TEMPLATE)["policy_a"])
            spec.update(checkpoint=str(path), name=name)
            spec["search_config"].update(simulations=1024, determinizations=1, root_noise_fraction=0.,
                seed=seed+30000, forced_playouts=0., policy_target_pruning=False)
            raw[side] = spec
    if panel["kind"] == "greedy":
        raw.update(device="cpu", workers=8)
        raw.pop("execution", None)
    else:
        raw["execution"]["pipeline_slots"] = 1
    raw["metadata"] = {"experiment": ART.name, "panel": panel["id"], "selected_step": selection["selected_step"],
        "selection": plan["selection"], "candidate_sha256": selection["checkpoint_sha256"],
        "opponent_sha256": plan["opponent"]["sha256"], "catalog_fingerprint": plan["catalog_fingerprint"]}
    return FixedEvaluationConfig.from_dict(raw).to_dict()


def summarize_panel(panel):
    path = ART / "panels" / f"{panel['id']}.json"
    row = {**panel, "status": "pending", "completed_pairs": 0}
    if not path.exists():
        return row
    result = read_json(path)
    config = read_json(ART / "evaluation_configs" / f"{panel['id']}.json")
    if result["config"] != config or result["catalog_fingerprint"] != config["metadata"]["catalog_fingerprint"]:
        raise ValueError("Arena configuration or rules differ")
    row.update(status="incomplete", completed_pairs=result["completed_pairs"])
    if result["completed_pairs"] != panel["pairs"]:
        return row
    seeds, scores = serialized_pair_scores(result)
    if seeds != tuple(config["game_seeds"]) or result["valid_games"] != 2*panel["pairs"]:
        raise ValueError("Arena population differs")
    ci = paired_bootstrap_mean(scores, resamples=config["bootstrap_resamples"], seed=config["bootstrap_seed"]).to_dict()
    if ci != result["paired_bootstrap"]:
        raise ValueError("Paired confidence interval differs")
    for side, key in (("policy_a", "candidate_sha256"), ("policy_b", "opponent_sha256")):
        if result["checkpoint_identity"][side]["sha256"] != config["metadata"][key]:
            raise ValueError("Evaluated model differs")
    for pair in result["pairs"]:
        if [g["first_player"] for g in pair["games"]] != [0, 1]:
            raise ValueError("Pair seats not swapped")
        for game in pair["games"]:
            for side in ("policy_a", "policy_b"):
                stats = game[f"{side}_stats"]
                if panel["kind"] == "search":
                    if (stats["simulations"] != 1024*stats["searchable_decisions"]
                            or stats["determinizations"] != stats["searchable_decisions"]):
                        raise ValueError("Actual search budget differs")
                elif stats.get("simulations", 0) or stats.get("determinizations", 0):
                    raise ValueError("Greedy unexpectedly searched")
    row.update(status="complete", score=result["win_rate_score"], ci95=ci, result=identity(path),
        **{k: result[k] for k in ("a_wins", "draws", "a_losses", "exceptions", "illegal_actions",
            "engineering_truncations", "invariant_failures", "policy_cycle_draws")})
    return row


def report(plan):
    progress = read_json(OUTPUT / "progress.json") if (OUTPUT / "progress.json").exists() else {"status": "prepared_not_started", "step": 0}
    selection = read_json(OUTPUT / "selection.json") if (OUTPUT / "selection.json").exists() else None
    full = [read_json(p) for p in sorted((OUTPUT / "full_validation").glob("step-*.json"))]
    panels = [summarize_panel(p) for p in plan["panels"]]
    status = ("complete" if all(p["status"] == "complete" for p in panels) else
        "training_complete_evaluation_pending" if progress["status"] == "complete" else progress["status"])
    result = {"status": status, "training": progress, "full_validation": full, "selection": selection,
        "panels": panels, "scope": "Validation-selected full checkpoint; no formal result before execution."}
    write_json(ART / "summary.json", result)
    lines = ["# 当前256×4实体MLP：8万局离线训练与R20对战", "", f"状态：`{status}`；训练 {progress['step']}/{plan['steps_target']} 步。", "",
        "从零初始化，lr=1e-4，16GiB共享整局缓存；保留非线性动作旁路和乘积，PSW及辅助目标关闭。每5%对同一4096子集及完整留出评估；按完整留出KL+MSE选择整个checkpoint。", "",
        "| 步数 | Policy KL | 胜负MSE | KL＋MSE |", "| ---: | ---: | ---: | ---: |"]
    for point in full:
        m = point["metrics"]
        lines.append(f"| {point['step']} | {m['policy_kl']:.6f} | {m['value_loss']:.6f} | {m['policy_kl']+m['value_loss']:.6f} |")
    if selection:
        lines += ["", f"选中 step={selection['selected_step']}，KL＋MSE={selection['selection_objective']:.6f}；SHA `{selection['checkpoint_sha256']}`。"]
    lines += ["", "| 对战 | 已完成／计划pair | 候选得分 | 95%配对区间 |", "| --- | ---: | ---: | --- |"]
    for p in panels:
        if p["status"] == "complete":
            ci = p["ci95"]
            lines.append(f"| {p['id']} | {p['pairs']}/{p['pairs']} | {100*p['score']:.3f}% | [{100*ci['lower']:.3f}%,{100*ci['upper']:.3f}%] |")
        else:
            lines.append(f"| {p['id']} | {p['completed_pairs']}/{p['pairs']} | — | — |")
    lines += ["", "Greedy 1000pair，搜索双方S1024N1（总1024）500pair；无根噪声、T0、FP/PTP关闭。评测12贵族，历史训练数据10贵族。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def frozen_source():
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        for path in [ROOT / "src/splendor/native_search.cpp", *ROOT.glob("src/splendor/_native_search*.so")]:
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for path in (CONFIG, PRESET, ART / "plan.json", Path(__file__)):
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copytree(ART / "catalog", destination / "data", dirs_exist_ok=True)
        record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob("*")) if p.is_file()}
        freeze(record_path, record)
    record = read_json(record_path)
    for path, expected in record["snapshot_files"].items():
        if sha256(destination / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    return destination


def interrupt(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)


def run_child(command, phase, source):
    global child
    print(f"START {phase}", flush=True)
    write_json(RUNTIME / "live.json", {"status": "running", "phase": phase, "pid": os.getpid(), "command": command})
    env = {**os.environ, "PYTHONPATH": str(source / "src")}
    with (RUNTIME / f"{phase}.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        for line in child.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = child.wait()
        child = None
    if stopping or code in (130, -signal.SIGINT):
        raise KeyboardInterrupt
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "train", "status"), nargs="?", default="run")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        result = report(read_json(ART / "plan.json"))
        print({"status": result["status"], "step": result["training"]["step"],
            "completed_panels": sum(p["status"] == "complete" for p in result["panels"])})
        return 0
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        report(plan)
        if args.action == "prepare":
            print({"status": "prepared_not_started", "steps": plan["steps_target"], "validation_steps": plan["validation_steps"]})
            return 0
        source = frozen_source()
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        execution_path = ART / "execution.json"
        execution = read_json(execution_path) if execution_path.exists() else {"sessions": []}
        record = {"command": [sys.executable, *sys.argv], "start_unix": time.time(), "status": "running",
            "source": identity(ART / "source.json"), "environment": environment_identity()}
        execution["sessions"].append(record)
        write_json(execution_path, execution)
        try:
            command = [sys.executable, "-u", "-m", "splendor.offline_mixed_budget", "--config",
                str(source / CONFIG.relative_to(ROOT)), "--output", str(OUTPUT)]
            if not (OUTPUT / "progress.json").exists() or read_json(OUTPUT / "progress.json")["status"] != "complete":
                run_child(command, "train", source)
            run_child(command + ["--select-only"], "select", source)
            selection = read_json(OUTPUT / "selection.json")
            if args.action == "run":
                for panel in plan["panels"]:
                    config_path = ART / "evaluation_configs" / f"{panel['id']}.json"
                    output = ART / "panels" / f"{panel['id']}.json"
                    freeze(config_path, evaluation_config(plan, selection, panel))
                    if summarize_panel(panel)["status"] == "complete":
                        continue
                    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(config_path),
                        "--output", str(output), "--progress-every", "10", "--summary-every", "10"]
                    if output.exists() or output.with_suffix(".json.games").exists():
                        command.append("--resume")
                    run_child(command, panel["id"], source)
                    if summarize_panel(panel)["status"] != "complete":
                        raise ValueError("Arena stopped before completion")
                    report(plan)
            record.update(status="complete" if args.action == "run" else "training_complete", end_unix=time.time())
            report(plan)
            return 0
        except KeyboardInterrupt:
            record.update(status="interrupted", end_unix=time.time())
            report(plan)
            return 130
        except Exception as error:
            record.update(status="failed", end_unix=time.time(), error=f"{type(error).__name__}: {error}")
            report(plan)
            raise
        finally:
            write_json(execution_path, execution)
            write_json(RUNTIME / "live.json", {"status": record["status"], "pid": os.getpid(), "error": record.get("error")})


if __name__ == "__main__":
    raise SystemExit(main())
