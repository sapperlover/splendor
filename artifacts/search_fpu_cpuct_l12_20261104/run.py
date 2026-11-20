"""FPU / c_puct screening (stages 2a, 2b): L12 new search vs L12 historical search, both S128N1.

2a fixes c_puct=1.5 and screens fpu_reduction r in {0.1, 0.2, 0.3}. 2b keeps the
2a winner (highest score, ties to the smaller r) and screens c_puct in {1.0, 2.5};
its configs are frozen only after 2a completes. Side B is always the historical
search (unvisited Q=0, c_puct=1.5). The confirmation stage 2c is not scripted here.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from splendor.alphazero_recording import (DiagnosticsConfig, InvocationRecorder, read_json, sha256,
                                         source_identity, write_json)
from splendor.catalog import load_default_catalog
from splendor.evaluation import FixedEvaluationConfig
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / ART.name
LAUNCHER = ROOT / "scripts/run_search_fpu_cpuct_l12_20261104.sh"
# Config-A execution template: the completed L12 graft panel (same machine, same L12 checkpoint).
TEMPLATE = ROOT / "artifacts/graft_policy_value_l0_l12_20261104/configs/01_search_L12_vs_R37.json"
L12 = ("runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt",
       "2a6304e3470cab3dad78ec0a2789ee892892e741f3731cafc2197cde9bd2d45d")
PAIRS, SMOKE_PAIRS = 2000, 2
GAME_SEED_START, POLICY_SEED_BASE, SEARCH_SEED = 47420000, 47430000, 47440000
BOOTSTRAP_SEED, DIFFERENCE_BOOTSTRAP_SEED = 47450000, 47450001
STAGE_A = tuple((r, 1.5) for r in (0.1, 0.2, 0.3))
STAGE_B_CPUCT = (1.0, 2.5)
PROGRESS_EVERY, SUMMARY_EVERY = 100, 500
child = None
stopping = False


def identity(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment input changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)


def panel_key(reduction, c_puct):
    return f"fpu{reduction:g}_c{c_puct:g}"


def panel_config(stage, reduction, c_puct, template, fingerprint, pairs=PAIRS):
    raw = copy.deepcopy(template)
    checkpoint = str(ROOT / L12[0])
    historical = {**template["policy_b"]["search_config"], "seed": SEARCH_SEED}
    raw.update(game_seeds=list(range(GAME_SEED_START, GAME_SEED_START + pairs)),
               policy_seed_base=POLICY_SEED_BASE, bootstrap_seed=BOOTSTRAP_SEED)
    raw["policy_a"].update(checkpoint=checkpoint, name=f"L12_{panel_key(reduction, c_puct)}",
                           search_config={**historical, "c_puct": c_puct, "fpu_reduction": reduction})
    raw["policy_b"].update(checkpoint=checkpoint, name="L12_historical_search", search_config=historical)
    raw["metadata"] = {"experiment": ART.name, "stage": stage, "mode": "search",
        "candidate_sha256": L12[1], "opponent_sha256": L12[1], "catalog_fingerprint": fingerprint,
        "search_budget": "both S128N1 = 128 per world * 1 world = total128",
        "candidate_search": f"fpu_reduction={reduction:g} (unvisited Q = parent Q - r), c_puct={c_puct:g}",
        "opponent_search": "historical: unvisited Q=0, c_puct=1.5"}
    return FixedEvaluationConfig.from_dict(raw).to_dict()


def panel_record(index, stage, reduction, c_puct, template, fingerprint):
    panel_id = f"{index:02d}_{stage}_{panel_key(reduction, c_puct)}"
    config = ART / "configs" / f"{panel_id}.json"
    freeze(config, panel_config(stage, reduction, c_puct, template, fingerprint))
    return {"id": panel_id, "stage": stage, "fpu_reduction": reduction, "c_puct": c_puct, "pairs": PAIRS,
            "config": identity(config), "output": str((ART / "panels" / f"{panel_id}.json").relative_to(ROOT))}


def load_template():
    template = read_json(TEMPLATE)
    side, execution = template["policy_a"]["search_config"], template["execution"]
    expected = {"backend": "native_cuda_pool", "search_workers": 12, "total_active_games": 1000,
                "max_batch_size": 500, "local_microbatch": 128, "pipeline_slots": 2,
                "batch_buckets": [64, 128, 256, 500], "candidate_buckets": [32, 48, 64],
                "max_batch_wait_ms": 5, "shared_slots": 2, "precision": "float32",
                "compile_mode": "native-cudagraph"}
    if (any(execution[k] != v for k, v in expected.items()) or side != template["policy_b"]["search_config"]
            or side["simulations"] != 128 or side["determinizations"] != 1 or side["c_puct"] != 1.5
            or side["root_noise_fraction"] != 0. or "fpu_reduction" in side):
        raise ValueError("Template is not the expected config-A S128N1 historical-search panel")
    return template


def prepare():
    if sha256(ROOT / L12[0]) != L12[1]:
        raise ValueError("L12 checkpoint changed")
    template = load_template()
    catalog = load_default_catalog()
    if catalog.data_fingerprint != template["metadata"]["catalog_fingerprint"]:
        raise ValueError("Current catalog differs from the template")
    panels = [panel_record(i + 1, "2a", r, c, template, catalog.data_fingerprint) for i, (r, c) in enumerate(STAGE_A)]
    freeze(ART / "configs" / "smoke_fpu0.2_c1.5.json",
           panel_config("smoke", 0.2, 1.5, template, catalog.data_fingerprint, pairs=SMOKE_PAIRS))
    plan = {"experiment": ART.name, "runner": identity(Path(__file__)), "template": identity(TEMPLATE),
        "checkpoint": {"path": L12[0], "sha256": L12[1]}, "catalog_fingerprint": catalog.data_fingerprint,
        "pairs_per_panel": PAIRS, "stage_a": panels,
        "stage_b_rule": f"keep the 2a reduction with the highest score (ties: smaller r); c_puct in {list(STAGE_B_CPUCT)}",
        "seeds": {"game_seed_start": GAME_SEED_START, "game_seed_end": GAME_SEED_START + PAIRS - 1,
            "policy_seed_base": POLICY_SEED_BASE, "search_seed": SEARCH_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED, "difference_bootstrap_seed": DIFFERENCE_BOOTSTRAP_SEED},
        "scope": "Search-only screening on one model (L12) against its own historical search; no training. "
                 "2a/2b select by point estimate and need the 2c confirmation on fresh seeds before any use in training."}
    freeze(ART / "plan.json", plan)
    return plan


def stage_b(plan, rows):
    """Freeze 2b after 2a; returns None while 2a is incomplete."""
    path = ART / "stage_b.json"
    if path.exists():
        return read_json(path)
    if any(row["status"] != "complete" for row in rows[:len(plan["stage_a"])]):
        return None
    best = max(zip(plan["stage_a"], rows), key=lambda item: (item[1]["score"], -item[0]["fpu_reduction"]))[0]
    template, fingerprint = load_template(), plan["catalog_fingerprint"]
    panels = [panel_record(len(plan["stage_a"]) + i + 1, "2b", best["fpu_reduction"], c, template, fingerprint)
              for i, c in enumerate(STAGE_B_CPUCT)]
    freeze(path, {"selected_from": best["id"], "fpu_reduction": best["fpu_reduction"],
                  "rule": plan["stage_b_rule"], "panels": panels})
    return read_json(path)


def all_panels(plan):
    path = ART / "stage_b.json"
    return plan["stage_a"] + (read_json(path)["panels"] if path.exists() else [])


def frozen_source():
    """Snapshot of the current tree with the FPU-enabled native extension."""
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        if destination.exists():
            raise ValueError("Unrecorded frozen source directory exists")
        record = source_identity(ROOT, destination)
        paths = [Path(__file__), ART / "mechanism.py", LAUNCHER, ROOT / "src/splendor/native_search.cpp",
                 *ROOT.glob("src/splendor/_native_search*.so"), *ROOT.glob("data/*"),
                 *(ART / "configs").glob("*.json")]
        for source in paths:
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
        freeze(record_path, record)
    record = read_json(record_path)
    for relative, expected in record["snapshot_files"].items():
        if sha256(destination / relative) != expected:
            raise ValueError(f"Frozen evaluation source changed: {relative}")
    # 2b configs are written after the snapshot; copy them in and check against the frozen stage record.
    for config in (ART / "configs").glob("*.json"):
        target = destination / config.relative_to(ROOT)
        if not target.exists():
            shutil.copyfile(config, target)
        elif sha256(target) != sha256(config):
            raise ValueError(f"Frozen config differs from artifact config: {config.name}")
    return destination


def panel_status(panel, plan, pairs=None):
    pairs = pairs or panel["pairs"]
    output = ROOT / panel["output"]
    row = {"id": panel["id"], "stage": panel["stage"], "status": "pending", "completed_pairs": 0}
    if not output.exists():
        return row
    result, config = read_json(output), read_json(ROOT / panel["config"]["path"])
    if result["config"] != config or result["catalog_fingerprint"] != plan["catalog_fingerprint"]:
        raise ValueError(f"Stored panel configuration/catalog differs: {panel['id']}")
    row.update(status="incomplete", completed_pairs=result["completed_pairs"])
    if result["completed_pairs"] != pairs:
        return row
    if not result["passed"] or result["valid_games"] != 2 * pairs:
        raise ValueError(f"Completed panel failed engineering checks: {panel['id']}")
    seeds, scores = serialized_pair_scores(result)
    if list(seeds) != config["game_seeds"]:
        raise ValueError("Panel opening seeds differ")
    for side in ("policy_a", "policy_b"):
        if result["checkpoint_identity"][side]["sha256"] != L12[1]:
            raise ValueError("Evaluated checkpoint differs")
    for pair in result["pairs"]:
        if [g["first_player"] for g in pair["games"]] != [0, 1]:
            raise ValueError("Pair seats were not swapped")
        for game in pair["games"]:
            for side in ("policy_a", "policy_b"):
                stats = game[f"{side}_stats"]
                if (stats["simulations"] != 128 * stats["searchable_decisions"]
                        or stats["determinizations"] != stats["searchable_decisions"]):
                    raise ValueError("Actual search budget differs from S128N1 total128")
    ci = result["paired_bootstrap"]
    if ci["pair_count"] != pairs or ci["estimate"] != float(scores.mean()):
        raise ValueError("Paired bootstrap identity differs")
    row.update(status="complete", score=result["win_rate_score"], ci95=ci, result=identity(output),
        seeds=list(seeds), pair_scores=scores.tolist(),
        **{k: result[k] for k in ("a_wins", "draws", "a_losses", "exceptions", "illegal_actions",
            "engineering_truncations", "invariant_failures", "policy_cycle_draws")})
    return row


def differences(panels, rows):
    done = [(p, r) for p, r in zip(panels, rows) if r["status"] == "complete"]
    out = []
    for i, (p, a) in enumerate(done):
        for q, b in done[i + 1:]:
            if a["seeds"] != b["seeds"]:
                raise ValueError("Panels do not share opening seeds")
            boot = asdict(paired_bootstrap_mean(np.asarray(b["pair_scores"]) - np.asarray(a["pair_scores"]),
                                                resamples=10000, seed=DIFFERENCE_BOOTSTRAP_SEED))
            out.append({"name": f"{q['id']} - {p['id']}", **{k: boot[k] for k in ("estimate", "lower", "upper")}})
    return out


def write_report(plan, panels, rows):
    selection = read_json(ART / "stage_b.json") if (ART / "stage_b.json").exists() else None
    complete = len(rows) == len(plan["stage_a"]) + len(STAGE_B_CPUCT) and all(r["status"] == "complete" for r in rows)
    diffs = differences(panels, rows)
    compact = [{k: v for k, v in r.items() if k not in ("seeds", "pair_scores")} for r in rows]
    write_json(ART / "summary.json", {"status": "complete" if complete else "incomplete", "panels": compact,
        "stage_b": selection, "paired_differences": diffs,
        "planned_pairs": PAIRS * (len(plan["stage_a"]) + len(STAGE_B_CPUCT)),
        "completed_pairs": sum(r["completed_pairs"] for r in rows)})
    lines = ["# FPU／c_puct 初筛：L12 新搜索 vs L12 旧搜索（双方 S128N1）", "",
        f"状态：{'complete' if complete else 'incomplete'}。每组 {PAIRS} pair／{2*PAIRS} 局，同开局换先后手，各组共用开局。",
        "A 方未访问边 Q = 父节点 Q − r；B 方为历史搜索（未访问 Q=0、c_puct=1.5）。50% 即无差异。", "",
        "| 面板 | 阶段 | r | c_puct | 完成pair | 胜／和／负 | A方得分 | 95%整pair区间 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for panel, row in zip(panels, rows):
        done = row["status"] == "complete"
        wdl = f"{row['a_wins']}/{row['draws']}/{row['a_losses']}" if done else "—"
        score = f"{100*row['score']:.3f}%" if done else "—"
        ci = f"[{100*row['ci95']['lower']:.3f}%, {100*row['ci95']['upper']:.3f}%]" if done else "—"
        lines.append(f"| {row['id']} | {panel['stage']} | {panel['fpu_reduction']:g} | {panel['c_puct']:g} | "
                     f"{row['completed_pairs']}/{PAIRS} | {wdl} | {score} | {ci} |")
    if selection is None:
        lines += ["", f"2b 尚未冻结：{plan['stage_b_rule']}。"]
    else:
        lines += ["", f"2b 沿用 2a 得分最高的 r={selection['fpu_reduction']:g}（{selection['selected_from']}）。"]
    if diffs:
        lines += ["", "| 配对得分差（百分点） | 估计 | 95%区间 |", "| --- | ---: | --- |"]
        lines += [f"| {d['name']} | {100*d['estimate']:+.3f} | [{100*d['lower']:+.3f}, {100*d['upper']:+.3f}] |" for d in diffs]
    lines += ["", "配置A，无根噪声，T0，12贵族；同一 L12 checkpoint。初筛按点估计选择，存在选择偏差；"
              "进入训练前须在全新种子上做 2c 确认（本脚本不含 2c）。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")


def evaluation_command(config, output, source, resume):
    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(source / config),
        "--output", str(output), "--progress-every", str(PROGRESS_EVERY), "--summary-every", str(SUMMARY_EVERY)]
    return command + (["--resume"] if resume else [])


def interrupt(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
    else:
        raise KeyboardInterrupt


def run_child(name, command, source):
    global child
    recorder = InvocationRecorder(RUNTIME / "panels" / name, DiagnosticsConfig(str(ART)),
        {"panel": name, "resume": "--resume" in command},
        {"command": command, "execution_source": identity(ART / "source.json")})
    error = None
    try:
        with recorder.stage(name), (RUNTIME / f"{name}.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=source, env={**os.environ, "PYTHONPATH": str(source / "src")},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
            write_json(RUNTIME / "live.json", {"status": "running", "panel": name,
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
        recorder.finish({"panel": name, "command": command}, error)


def smoke(plan, source):
    """Two pairs of r=0.2 through the exact frozen config-A path; written only under runtime."""
    name = "smoke_fpu0.2_c1.5"
    output = RUNTIME / "smoke" / f"{name}.json"
    if output.exists():
        output.unlink()
    shutil.rmtree(Path(str(output) + ".games"), ignore_errors=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = ART / "configs" / f"{name}.json"
    run_child(name, evaluation_command(config.relative_to(ROOT), output, source, False), source)
    panel = {"id": name, "stage": "smoke", "pairs": SMOKE_PAIRS, "config": identity(config),
             "output": str(output.relative_to(ROOT))}
    row = panel_status(panel, plan)
    if row["status"] != "complete":
        raise ValueError("Smoke did not complete")
    write_json(ART / "smoke.json", {"pairs": SMOKE_PAIRS, "status": "passed", "result": identity(output),
        "checks": "valid games, seat swap, actual S128N1 budget on both sides, checkpoint SHA",
        **{k: row[k] for k in ("a_wins", "draws", "a_losses", "exceptions", "illegal_actions",
            "engineering_truncations", "invariant_failures")}})
    return row


def run_panels(plan, panels, rows, source):
    for index, panel in enumerate(panels):
        if stopping:
            raise KeyboardInterrupt
        if rows[index]["status"] == "complete":
            continue
        output = ROOT / panel["output"]
        resume = output.exists() or Path(str(output) + ".games").exists()
        run_child(panel["id"], evaluation_command(panel["config"]["path"], output, source, resume), source)
        rows[index] = panel_status(panel, plan)
        write_report(plan, panels, rows)
        if rows[index]["status"] != "complete":
            raise ValueError(f"Panel stopped before completion: {panel['id']}")


def run_sequence(plan):
    rows = [panel_status(p, plan) for p in plan["stage_a"]]
    write_report(plan, plan["stage_a"], rows)
    run_panels(plan, plan["stage_a"], rows, frozen_source())
    selection = stage_b(plan, rows)
    panels = all_panels(plan)
    rows += [panel_status(p, plan) for p in selection["panels"]]
    write_report(plan, panels, rows)
    run_panels(plan, panels, rows, frozen_source())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="run", choices=("prepare", "smoke", "run", "status"))
    args = parser.parse_args()
    if args.action == "status":
        plan = read_json(ART / "plan.json")
        rows = [panel_status(p, plan) for p in all_panels(plan)]
        print({"panels": [{k: r[k] for k in ("id", "status", "completed_pairs")} for r in rows],
               "stage_b_frozen": (ART / "stage_b.json").exists()}, flush=True)
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        if args.action == "prepare":
            frozen_source()
            write_report(plan, all_panels(plan), [panel_status(p, plan) for p in all_panels(plan)])
            print({"status": "prepared_not_started", "stage_a_panels": len(plan["stage_a"]), "pairs": PAIRS}, flush=True)
            return 0
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupt)
        if args.action == "smoke":
            print(smoke(plan, frozen_source())["status"], flush=True)
            return 0
        if read_json(ART / "smoke.json")["status"] != "passed":
            raise ValueError("Run the smoke action first")
        path = ART / "execution.json"
        execution = read_json(path) if path.exists() else {"sessions": []}
        record = {"command": [sys.executable, *sys.argv], "start_unix": time.time(), "status": "running",
            "source": identity(ART / "source.json")}
        execution["sessions"].append(record)
        write_json(path, execution)
        try:
            run_sequence(plan)
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
