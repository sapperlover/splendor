"""Five selected offline models vs raw R37: all greedy panels, then all S128N1."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import (DiagnosticsConfig, InvocationRecorder, read_json, sha256,
                                         source_identity, write_json)
from splendor.catalog import load_default_catalog
from splendor.evaluation import FixedEvaluationConfig
from splendor.paired_statistics import serialized_pair_scores

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / ART.name
MODELS = (
    ("value", "entity_attention_128x2_shortterm_value_r21_r37_20261023", (True, False, False)),
    ("value_policy", "entity_attention_128x2_continuation_r21_r37_20261024", (True, True, False)),
    ("policy", "entity_attention_128x2_opponent_policy_r21_r37_20261024", (False, True, False)),
    ("noaux", "entity_attention_128x2_noaux_r21_r37_20261026", (False, False, False)),
    ("policy_score", "entity_attention_128x2_policy_score_r21_r37_20261026", (False, True, True)),
)
PAIRS, SEED_START, POLICY_SEED_BASE = 10000, 47270000, 47300000
SEARCH_SEED, BOOTSTRAP_SEED = 47330000, 47340000
PROGRESS_EVERY, SUMMARY_EVERY = 100, 1000
child = None
stopping = False


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen evaluation input changed: {path}")
    else:
        write_json(path, value)


def selected_model(key, name, flags):
    art, output = ROOT / "artifacts" / name, ROOT / "runs" / name
    selection = read_json(art / "selected_candidate.json")
    if selection != read_json(output / "selection.json") or read_json(output / "progress.json")["status"] != "complete":
        raise ValueError(f"Offline selection is not final: {key}")
    # Other checkpoints may be archived; only the selected checkpoint is needed locally.
    candidates = []
    for row in selection["candidates"]:
        original = output / "full_validation" / f"step-{row['step']:09d}.json"
        if read_json(original) != {k: v for k, v in row.items() if k != "selection_objective"}:
            raise ValueError(f"Original full validation differs: {key}, step {row['step']}")
        candidates.append((row["metrics"]["policy_kl"] + row["metrics"]["value_loss"], row["step"]))
    if min(candidates) != (selection["selection_objective"], selection["selected_step"]):
        raise ValueError(f"Checkpoint is not the main KL+MSE minimum: {key}")
    path = Path(selection["checkpoint"])
    if sha256(path) != selection["checkpoint_sha256"]:
        raise ValueError(f"Selected checkpoint changed: {key}")
    payload = load_alphazero_checkpoint(path)
    config = payload.model_config
    actual = (config.get("predict_shortterm_value", config["architecture"] == "entity_attention_continuation_v1"),
              config.get("predict_opponent_policy", False), config.get("predict_score_ownership", False))
    if actual != flags or config["hidden"] != 128 or config["layers"] != 2:
        raise ValueError(f"Wrong selected model architecture: {key}")
    if payload.total_train_step != selection["selected_step"]:
        raise ValueError(f"Selected step differs: {key}")
    reference = read_json(art / "plan.json")
    if reference["games"] != 104000 or reference["source_iterations"] != [21, 37]:
        raise ValueError(f"Expected the original 104k offline experiment: {key}")
    freeze(ART / "selections" / f"{key}.json", selection)
    return {"key": key, "experiment": name, "checkpoint": str(path), "sha256": selection["checkpoint_sha256"],
        "selected_step": selection["selected_step"], "selection_objective": selection["selection_objective"],
        "model_config": config, "dataset_digest": reference["dataset_digest"],
        "selection": identity(art / "selected_candidate.json"), "run_selection": identity(output / "selection.json"),
        "training_plan": identity(art / "plan.json")}


def panel_config(model, opponent, kind, templates, fingerprint):
    raw = copy.deepcopy(templates[kind])
    raw.pop("pairs", None)
    raw.pop("seed_start", None)
    raw.update(game_seeds=list(range(SEED_START, SEED_START + PAIRS)), policy_seed_base=POLICY_SEED_BASE,
        bootstrap_resamples=10000, bootstrap_seed=BOOTSTRAP_SEED)
    for side, record, name in (("policy_a", model, model["key"]), ("policy_b", opponent, "raw_R37")):
        raw[side].update(checkpoint=record["checkpoint"], name=name)
        if kind == "search":
            raw[side]["search_config"].update(simulations=128, determinizations=1, seed=SEARCH_SEED,
                root_noise_fraction=0., forced_playouts=0., policy_target_pruning=False)
    raw["metadata"] = {"experiment": ART.name, "model": model["key"], "mode": kind,
        "selected_step": model["selected_step"], "candidate_sha256": model["sha256"],
        "opponent_sha256": opponent["sha256"], "catalog_fingerprint": fingerprint,
        "search_budget": "both S128N1 = 128 per world * 1 world = total128" if kind == "search" else "zero search",
        "selection": "Pre-existing main full-holdout KL+MSE minimum, fixed before these arenas"}
    return FixedEvaluationConfig.from_dict(raw).to_dict()


def prepare():
    models = [selected_model(*record) for record in MODELS]
    if len({m["dataset_digest"] for m in models}) != 1:
        raise ValueError("The five models did not use the same original offline pool")
    reference = read_json(ROOT / "artifacts" / MODELS[0][1] / "plan.json")
    anchor = reference["opponents"]["R37"]
    opponent = {"checkpoint": anchor["path"], "sha256": anchor["sha256"]}
    if sha256(Path(opponent["checkpoint"])) != opponent["sha256"]:
        raise ValueError("Raw R37 checkpoint changed")
    payload = load_alphazero_checkpoint(opponent["checkpoint"])
    if payload.model_config["architecture"] != "entity_attention_v1":
        raise ValueError("Expected the original raw R37 model")
    opponent["model_config"] = payload.model_config
    templates = {kind: read_json(ROOT / "configs/entity_attention_v1" / filename) for kind, filename in
        (("greedy", "evaluation_greedy_r43.json"), ("search", "evaluation_search_a_r43.json"))}
    freeze(ART / "evaluation_templates.json", templates)
    catalog = load_default_catalog()
    if catalog.data_fingerprint != reference["catalog_fingerprint"]:
        raise ValueError("Current catalog differs from the original 12-noble experiment")
    catalog_files = {}
    for filename, expected in reference["catalog_files"].items():
        source = ROOT / "data" / filename
        if sha256(source) != expected:
            raise ValueError("Original catalog changed")
        target = ART / "catalog" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256(target) != expected:
            raise ValueError("Frozen catalog changed")
        catalog_files[filename] = expected
    panels = []
    for kind in ("greedy", "search"):
        for model in models:
            panel_id = f"{len(panels)+1:02d}_{kind}_{model['key']}_vs_R37"
            config = ART / "configs" / f"{panel_id}.json"
            freeze(config, panel_config(model, opponent, kind, templates, catalog.data_fingerprint))
            panels.append({"id": panel_id, "kind": kind, "model": model["key"], "pairs": PAIRS,
                "config": identity(config), "output": str((ART / "panels" / f"{panel_id}.json").relative_to(ROOT))})
    plan = {"experiment": ART.name, "runner": identity(Path(__file__)), "models": models, "opponent": opponent,
        "catalog_fingerprint": catalog.data_fingerprint, "catalog_files": catalog_files,
        "templates": identity(ART / "evaluation_templates.json"), "panels": panels,
        "order": "Five greedy panels in user order, then five both-S128N1 panels in the same order; strictly serial",
        "pairs_per_panel": PAIRS, "total_pairs": 10*PAIRS, "total_games": 20*PAIRS,
        "seeds": {"game_seed_start": SEED_START, "game_seed_end": SEED_START+PAIRS-1,
            "policy_seed_base": POLICY_SEED_BASE, "search_seed": SEARCH_SEED, "bootstrap_seed": BOOTSTRAP_SEED},
        "progress_every": PROGRESS_EVERY, "summary_every": SUMMARY_EVERY,
        "scope": "Preselected best full checkpoints on the same104k pool. No training or new selection. Common fresh openings/policy seeds across all models and both modes. New comparison, not an independent final acceptance claim. Resume full pairs using native fixed evaluation; failures stop subsequent panels."}
    freeze(ART / "plan.json", plan)
    return plan


def frozen_source():
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        paths = [Path(__file__), ART / "validate.py", ROOT / "scripts/run_offline5_vs_r37_10k_20261027.sh",
            ROOT / "src/splendor/native_search.cpp", *ROOT.glob("src/splendor/_native_search*.so"),
            *(ART / "configs").glob("*.json")]
        for source in paths:
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for source in (ART / "catalog").iterdir():
            target = destination / "data" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
        freeze(record_path, record)
    for relative, expected in read_json(record_path)["snapshot_files"].items():
        if sha256(destination / relative) != expected:
            raise ValueError(f"Frozen evaluation source changed: {relative}")
    return destination


def panel_status(panel, plan):
    output = ROOT / panel["output"]
    row = {"id": panel["id"], "model": panel["model"], "kind": panel["kind"], "status": "pending", "completed_pairs": 0}
    if not output.exists():
        return row
    result, config = read_json(output), read_json(ROOT / panel["config"]["path"])
    if result["config"] != config or result["catalog_fingerprint"] != plan["catalog_fingerprint"]:
        raise ValueError(f"Stored panel configuration/catalog differs: {panel['id']}")
    row.update(status="incomplete", completed_pairs=result["completed_pairs"])
    if result["completed_pairs"] != panel["pairs"]:
        return row
    if not result["passed"] or result["valid_games"] != 2*panel["pairs"]:
        raise ValueError(f"Completed panel failed engineering checks: {panel['id']}")
    seeds, scores = serialized_pair_scores(result)
    if list(seeds) != config["game_seeds"]:
        raise ValueError("Panel opening seeds differ")
    for side, key in (("policy_a", "candidate_sha256"), ("policy_b", "opponent_sha256")):
        if result["checkpoint_identity"][side]["sha256"] != config["metadata"][key]:
            raise ValueError("Evaluated checkpoint differs")
    for pair in result["pairs"]:
        if [g["first_player"] for g in pair["games"]] != [0, 1]:
            raise ValueError("Pair seats were not swapped")
        for game in pair["games"]:
            for side in ("policy_a", "policy_b"):
                stats = game[f"{side}_stats"]
                if panel["kind"] == "search":
                    if (stats["simulations"] != 128*stats["searchable_decisions"]
                            or stats["determinizations"] != stats["searchable_decisions"]):
                        raise ValueError("Actual search budget differs from S128N1 total128")
                elif stats.get("simulations", 0) or stats.get("determinizations", 0):
                    raise ValueError("Greedy unexpectedly searched")
    ci = result["paired_bootstrap"]
    if (ci["pair_count"] != panel["pairs"] or ci["resamples"] != config["bootstrap_resamples"]
            or ci["seed"] != config["bootstrap_seed"] or ci["estimate"] != float(scores.mean())):
        raise ValueError("Paired bootstrap identity differs")
    row.update(status="complete", score=result["win_rate_score"], ci95=ci, result=identity(output),
        **{k: result[k] for k in ("a_wins", "draws", "a_losses", "exceptions", "illegal_actions",
            "engineering_truncations", "invariant_failures", "policy_cycle_draws")})
    return row


def write_report(plan, rows):
    complete = all(row["status"] == "complete" for row in rows)
    write_json(ART / "summary.json", {"status": "complete" if complete else "incomplete",
        "panels": rows, "planned_pairs": plan["total_pairs"], "completed_pairs": sum(r["completed_pairs"] for r in rows)})
    lines = ["# 五个离线best对raw R37：10k pair greedy及双方S128N1", "",
        "顺序：先全部greedy，再全部搜索；每组10000pair／20000局。每个pair同开局换先后手。", "",
        "| 面板 | 完成pair | 得分率 | 95%整pair区间 |", "| --- | ---: | ---: | --- |"]
    for row in rows:
        score = f"{100*row['score']:.3f}%" if row["status"] == "complete" else "—"
        ci = f"[{100*row['ci95']['lower']:.3f}%, {100*row['ci95']['upper']:.3f}%]" if row["status"] == "complete" else "—"
        lines.append(f"| {row['id']} | {row['completed_pairs']}/10000 | {score} | {ci} |")
    lines += ["", "搜索为双方S128N1，每世界128×1世界＝总128；配置A、无根噪声、T0，12贵族。",
        "未完成面板不能作为完整棋力结论。模型在本次对战前已按原完整留出主KL＋MSE选定。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")


def panel_command(panel, source):
    output = ROOT / panel["output"]
    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config",
        str(source / panel["config"]["path"]), "--output", str(output),
        "--progress-every", str(PROGRESS_EVERY), "--summary-every", str(SUMMARY_EVERY)]
    if output.exists() or Path(str(output)+".games").exists():
        command.append("--resume")
    return command


def interrupt(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
    else:
        raise KeyboardInterrupt


def run_panel(panel, plan, source):
    global child
    command = panel_command(panel, source)
    recorder = InvocationRecorder(RUNTIME / "panels" / panel["id"], DiagnosticsConfig(str(ART)),
        {"panel": panel["id"], "resume": "--resume" in command},
        {"panel": panel, "command": command, "execution_source": identity(ART / "source.json")})
    error = None
    try:
        with recorder.stage(panel["id"]), (RUNTIME / f"{panel['id']}.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=source, env={**os.environ, "PYTHONPATH": str(source / "src")},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
            write_json(RUNTIME / "live.json", {"status": "running", "panel": panel["id"],
                "pid": os.getpid(), "child_pid": child.pid, "command": command})
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
    except BaseException as caught:
        error = caught
        raise
    finally:
        recorder.finish({"panel": panel["id"], "command": command}, error)


def run_sequence(plan, source):
    rows = [panel_status(p, plan) for p in plan["panels"]]
    write_report(plan, rows)
    for index, panel in enumerate(plan["panels"]):
        if stopping:
            raise KeyboardInterrupt
        if rows[index]["status"] == "complete":
            continue
        run_panel(panel, plan, source)
        rows[index] = panel_status(panel, plan)
        write_report(plan, rows)
        if rows[index]["status"] != "complete":
            raise ValueError(f"Panel stopped before completion: {panel['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="run", choices=("prepare", "run", "status"))
    args = parser.parse_args()
    if args.action == "status":
        plan = read_json(ART / "plan.json")
        rows = [panel_status(p, plan) for p in plan["panels"]]
        print({"status": "complete" if all(row["status"] == "complete" for row in rows) else "incomplete",
            "panels": rows, "planned_pairs": plan["total_pairs"],
            "completed_pairs": sum(row["completed_pairs"] for row in rows)}, flush=True)
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        source = frozen_source()
        if args.action == "prepare":
            write_report(plan, [panel_status(p, plan) for p in plan["panels"]])
            print({"status": "prepared_not_started", "panels": 10, "pairs": plan["total_pairs"]}, flush=True)
            return 0
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupt)
        path = ART / "execution.json"
        execution = read_json(path) if path.exists() else {"sessions": []}
        record = {"command": [sys.executable, *sys.argv], "start_unix": time.time(), "status": "running",
            "source": identity(ART / "source.json")}
        execution["sessions"].append(record)
        write_json(path, execution)
        try:
            run_sequence(plan, source)
            record["status"] = "complete"
            return 0
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            return 130
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["end_unix"] = time.time()
            write_json(path, execution)
            write_json(RUNTIME / "live.json", {"status": record["status"], "pid": os.getpid(), "error": record.get("error")})


if __name__ == "__main__":
    raise SystemExit(main())
