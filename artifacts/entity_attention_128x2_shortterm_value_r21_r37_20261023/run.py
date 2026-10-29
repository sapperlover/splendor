"""Train fresh attention128 + shortterm value on R21–R37, then two fixed R37 arenas."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import _read_game_metadata
from splendor.catalog import load_default_catalog
from splendor.evaluation import FixedEvaluationConfig
from splendor.offline_continuation import ContinuationOfflineConfig, prepare_inputs
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
OUTPUT = ROOT / "runs" / ART.name
RUNTIME = ROOT / "runtime" / ART.name
SOURCE = ROOT / "runs/entity_attention_r20_s256n4_noble12_iteration_20261022"
BASE_CONFIG = ROOT / "configs/entity_attention_v1/offline_shortterm_value_r21_r31_20261023.json"
CONFIG = ROOT / "configs/entity_attention_v1/offline_shortterm_value_r21_r37_20261023.json"
CONTINUATION_SOURCE = ROOT / "runs/entity_attention_r31_10k_p60k_iteration_20261023"
EVALUATION_CONFIGS = {kind: ROOT / f"configs/entity_attention_v1/evaluation_{suffix}_r43.json"
    for kind, suffix in (("greedy", "greedy"), ("search", "search_a"))}
PANELS = (
    {"id": "greedy_vs_R37", "kind": "greedy", "opponent": "R37", "pairs": 1000, "seed_start": 37000000},
    {"id": "search_S1024N1_vs_R37", "kind": "search", "opponent": "R37", "pairs": 500, "seed_start": 37100000},
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


def prepare():
    states = {root: {r["iteration"]: r for r in read_json(root / "state.json")["completed_rounds"]}
        for root in (SOURCE, CONTINUATION_SOURCE)}
    catalog = load_default_catalog()
    opponent = states[CONTINUATION_SOURCE][6]["checkpoint"]
    if sha256(Path(opponent["path"])) != opponent["sha256"]:
        raise ValueError("Raw R37 checkpoint identity changed")
    records, sources, counts = [], [], {}
    for lineage in range(21, 38):
        root, local, expected = ((SOURCE, lineage, 4000) if lineage <= 31
                                 else (CONTINUATION_SOURCE, lineage - 31, 10000))
        completed = states[root]
        if local not in completed:
            raise ValueError(f"Source R{lineage} is incomplete")
        teacher = (completed[local-1]["checkpoint"] if local > 1
                   else read_json(root / "config.json")["initial_checkpoint"])
        stage = root / "rounds" / f"round-{local-1:03d}"
        index_path = stage / "collection_index.json"
        manifest_path, summary_path = stage / "selfplay/manifest.json", stage / "selfplay/summary.json"
        manifest, summary = read_json(manifest_path), read_json(summary_path)
        if (manifest["checkpoint_identity"] != teacher
                or manifest["catalog_fingerprint"] != catalog.data_fingerprint):
            raise ValueError(f"R{lineage} teacher lineage or catalog differs")
        rows = read_json(index_path)["games"]
        if (len(rows) != expected or summary["completed_games"] != expected
                or summary["expected_games"] != expected or summary["failed_games"] or summary["missing_games"]
                or {r["round_index"] for r in rows} != {lineage+42}
                or len({r["game_id"] for r in rows}) != expected):
            raise ValueError(f"Source R{lineage} lacks {expected} unique complete games")
        for row in rows:
            stat = Path(row["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
                raise ValueError(f"Source archive changed: {row['path']}")
        metadata = _read_game_metadata(Path(rows[0]["path"]))
        if metadata["trace"]["catalog_fingerprint"] != catalog.data_fingerprint:
            raise ValueError("Source game catalog differs")
        sources.append({"lineage_round": lineage, "local_iteration": local, "games": expected,
            "index": identity(index_path), "manifest": identity(manifest_path), "summary": identity(summary_path),
            "teacher": teacher, "catalog_fingerprint": manifest["catalog_fingerprint"]})
        counts[str(lineage+43)] = expected  # Config round numbers are one-based; NPZ IDs are zero-based.
        records.extend(rows)
    records.sort(key=lambda r: (r["round_index"], r["game_id"]))
    manifest = OUTPUT / "inputs/pool_games.jsonl"
    content = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records)
    if manifest.exists():
        if manifest.read_text() != content:
            raise ValueError("Frozen104k pool differs")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(content)
    raw = read_json(BASE_CONFIG)
    raw.update(manifest=str(manifest), manifest_sha256=sha256(manifest), first_round=64, last_round=80,
        expected_games_per_round=counts)
    freeze(CONFIG, raw)
    config = ContinuationOfflineConfig.from_dict(raw)
    _, _, _, fixed, training = prepare_inputs(config, OUTPUT)
    freeze(ART / "evaluation_templates.json", {kind: read_json(path) for kind, path in EVALUATION_CONFIGS.items()})
    catalog_files = {}
    for path in sorted((ROOT / "data").glob("splendor_*")):
        if path.suffix in (".csv", ".json"):
            target = ART / "catalog" / path.name
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(path, target)
            if sha256(target) != sha256(path):
                raise ValueError("Frozen catalog changed")
            catalog_files[path.name] = sha256(target)
    plan = {"experiment": ART.name, "config": identity(CONFIG), "runner": identity(Path(__file__)),
        "baseline_config": identity(BASE_CONFIG), "source_iterations": [21, 37], "source_rounds": sources,
        "absolute_rounds": [64, 80], "games": len(records), "splits": training["splits"],
        "dataset_digest": read_json(OUTPUT / "dataset.json")["digest"],
        "fixed_validation_digest": fixed["digest"], "model_config": config.model_config,
        "parameters": 650757, "continuation_auxiliary": config.continuation_auxiliary,
        "initialization": "random base model and shortterm heads, fresh AdamW; previous offline attention128 seeds",
        "steps_target": training["steps_target"], "validation_steps": training["validation_steps"],
        "sampling": training["sampling"], "cache_mib": config.cache_mib,
        "recording": "per-step training_metrics.jsonl + reference SHA/identity; no .bin/.steps.jsonl or duplicate session metrics; checkpoints every1000 plus all5% validation nodes",
        "early_stopping": {"consecutive_strict_increases": 3, "metric": "full-holdout policy KL + win/loss value MSE",
            "schedule": "5%,10%,...,100% of12 value draw exposures; step0 excluded; ties/decreases reset; earliest stop20%"},
        "selection": "minimum main full-holdout KL+MSE among all reached percentage checkpoints; auxiliary loss excluded; ties earlier step; freeze entire checkpoint before any arena",
        "opponents": {"R37": opponent}, "panels": list(PANELS), "catalog_fingerprint": catalog.data_fingerprint,
        "catalog_files": catalog_files, "evaluation_templates": identity(ART / "evaluation_templates.json"),
        "evaluation_sources": {kind: identity(path) for kind, path in EVALUATION_CONFIGS.items()},
        "scope": "Original104000 games, labels and whole-game splits; v3 search-root EMA H2/H6/H14 shortterm value weight0.25, opponent policy auxiliary0; PSW/old auxiliary disabled. CPU8 greedy1000pair then latest searchA500pair, both S1024N1 total1024,12nobles. Single training seed; no strength result before execution."}
    freeze(ART / "plan.json", plan)
    return plan


def evaluation_config(plan, selection, panel):
    opponent = plan["opponents"][panel["opponent"]]
    for record in (selection, opponent):
        path = Path(record.get("checkpoint", record.get("path", "")))
        expected = record.get("checkpoint_sha256", record.get("sha256"))
        if sha256(path) != expected:
            raise ValueError("Arena model identity changed")
    raw = copy.deepcopy(read_json(ART / "evaluation_templates.json")[panel["kind"]])
    seed = panel["seed_start"]
    raw.pop("pairs", None)
    raw.pop("seed_start", None)
    raw.update(game_seeds=list(range(seed, seed+panel["pairs"])), policy_seed_base=seed+10000,
        bootstrap_seed=seed+20000, bootstrap_resamples=10000)
    for side, path, name in (("policy_a", selection["checkpoint"], "selected_attention128_shortterm_value"),
                              ("policy_b", opponent["path"], panel["opponent"])):
        raw[side].update(checkpoint=path, name=name)
        if panel["kind"] == "search":
            raw[side]["search_config"].update(simulations=1024, determinizations=1, seed=seed+30000,
                root_noise_fraction=0., forced_playouts=0., policy_target_pruning=False)
    raw["metadata"] = {"experiment": ART.name, "panel": panel["id"], "selected_step": selection["selected_step"],
        "candidate_sha256": selection["checkpoint_sha256"], "opponent_sha256": opponent["sha256"],
        "catalog_fingerprint": plan["catalog_fingerprint"], "selection": plan["selection"]}
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
    if (seeds != tuple(config["game_seeds"]) or result["valid_games"] != 2*panel["pairs"]
            or not result["passed"]):
        raise ValueError("Arena population or engineering completion differs")
    ci = paired_bootstrap_mean(scores, resamples=config["bootstrap_resamples"], seed=config["bootstrap_seed"]).to_dict()
    if ci != result["paired_bootstrap"]:
        raise ValueError("Paired interval differs")
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
    panels = [summarize_panel(p) for p in PANELS]
    status = ("complete" if all(p["status"] == "complete" for p in panels) else
        "training_complete_evaluation_pending" if progress["status"] == "complete" else progress["status"])
    result = {"status": status, "training": progress, "selection": selection, "full_validation": full, "panels": panels}
    write_json(ART / "summary.json", result)
    lines = ["# Attention128：第21–37轮固定池短期value离线训练", "", f"状态：`{status}`；已提交 {progress['step']}/{plan['steps_target']} 步。", "",
        "每5%完整留出KL+MSE连续三次严格上升即停止，持平或下降重置；用已到达节点内联合指标最低的完整checkpoint依次对战。", "",
        "| 步数 | 完整留出KL | 胜负MSE | KL+MSE |", "| ---: | ---: | ---: | ---: |"]
    for p in full:
        m = p["metrics"]
        lines.append(f"| {p['step']} | {m['policy_kl']:.6f} | {m['value_loss']:.6f} | {m['policy_kl']+m['value_loss']:.6f} |")
    if selection:
        lines += ["", f"预先选中 step={selection['selected_step']}，SHA `{selection['checkpoint_sha256']}`。"]
    lines += ["", "| 对战 | 已完成/计划pair | 候选得分 | 95%配对区间 |", "| --- | ---: | ---: | --- |"]
    for p in panels:
        score = f"{100*p['score']:.3f}%" if p["status"] == "complete" else "—"
        ci = f"[{100*p['ci95']['lower']:.3f}%,{100*p['ci95']['upper']:.3f}%]" if p["status"] == "complete" else "—"
        lines.append(f"| {p['id']} | {p['completed_pairs']}/{p['pairs']} | {score} | {ci} |")
    lines += ["", "先对raw R37 greedy1000pair，随后双方S1024N1搜索500pair（每世界1024次、1世界、总1024），最新配置A／owner双槽；无根噪声/T0，12贵族，单训练seed。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def frozen_source():
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        for path in (CONFIG, Path(__file__), ROOT / "src/splendor/native_search.cpp",
                     *ROOT.glob("src/splendor/_native_search*.so"), *(ART / "catalog").iterdir()):
            relative = Path("data") / path.name if path.parent == ART / "catalog" else path.relative_to(ROOT)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
        freeze(record_path, record)
    for path, expected in read_json(record_path)["snapshot_files"].items():
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
    if stopping:
        raise KeyboardInterrupt
    env = {**os.environ, "PYTHONPATH": str(source / "src")}
    with (RUNTIME / f"{phase}.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        write_json(RUNTIME / "live.json", {"status": "running", "phase": phase, "pid": os.getpid(),
            "child_pid": child.pid, "command": command})
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
        print(report(read_json(ART / "plan.json"))["status"], flush=True)
        return 0
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        report(plan)
        source = frozen_source()
        if args.action == "prepare":
            print({"status": "prepared_not_started", "games": plan["games"], "steps_target": plan["steps_target"]}, flush=True)
            return 0
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        path = ART / "execution.json"
        execution = read_json(path) if path.exists() else {"sessions": []}
        record = {"command": [sys.executable, *sys.argv], "start_unix": time.time(), "status": "running",
            "source": identity(ART / "source.json"), "environment": environment_identity()}
        execution["sessions"].append(record)
        write_json(path, execution)
        try:
            command = [sys.executable, "-u", "-m", "splendor.offline_continuation", "--config",
                str(source / CONFIG.relative_to(ROOT)), "--output", str(OUTPUT)]
            if not (OUTPUT / "progress.json").exists() or read_json(OUTPUT / "progress.json")["status"] != "complete":
                run_child(command, "train", source)
            run_child(command + ["--select-only"], "select", source)
            selection = read_json(OUTPUT / "selection.json")
            freeze(ART / "selected_candidate.json", selection)
            if args.action == "run":
                for panel in PANELS:
                    config = ART / "evaluation_configs" / f"{panel['id']}.json"
                    output = ART / "panels" / f"{panel['id']}.json"
                    freeze(config, evaluation_config(plan, selection, panel))
                    if summarize_panel(panel)["status"] == "complete":
                        continue
                    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(config),
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
            write_json(path, execution)
            write_json(RUNTIME / "live.json", {"status": record["status"], "pid": os.getpid(), "error": record.get("error")})


if __name__ == "__main__":
    raise SystemExit(main())
