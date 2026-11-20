"""L12 and the two L0/L12 policy-value grafts vs raw R37, all both-S128N1.

The offline L0 reference is the completed 10k-pair panel
offline5_vs_r37_10k_20261027/08_search_policy_vs_R37. These panels reuse its
frozen evaluation source (only the composition class is patched), its config
template and the first 2000 of its opening seeds.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from splendor.actor_critic import MODEL_WEIGHTS_FORMAT_VERSION
from splendor.alphazero_recording import DiagnosticsConfig, InvocationRecorder, read_json, sha256, write_json
from splendor.catalog import load_default_catalog
from splendor.entity_attention import EntityAttentionPolicyValue, EntityAttentionPolicyValueConfig
from splendor.evaluation import FixedEvaluationConfig
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores
from splendor.replay import RULESET_ID
from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples, load_game

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / ART.name
MODELS_DIR = ROOT / "runs" / ART.name / "compositions"
REFERENCE = ROOT / "artifacts" / "offline5_vs_r37_10k_20261027"
REFERENCE_PANEL = "08_search_policy_vs_R37"
REFERENCE_SOURCE = ROOT / "runtime" / "offline5_vs_r37_10k_20261027" / "source"
PATCHED = ("src/splendor/entity_attention.py",)
SNAPSHOT_HOLDOUT = ROOT / "runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-012/replay_snapshot.json"
SOURCES = {
    "L0": ("runs/entity_attention_128x2_opponent_policy_r21_r37_20261024/checkpoints/step-000140221.pt",
           "e4edeb20e8caa3c1abec0146d3757594dd6e0428bd833badd0cb3915c2299a9f"),
    "L12": ("runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt",
            "2a6304e3470cab3dad78ec0a2789ee892892e741f3731cafc2197cde9bd2d45d"),
    "R37": ("runs/entity_attention_r31_10k_p60k_iteration_20261023/rounds/round-005/checkpoints/final.pt",
            "611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045"),
}
# (panel key, policy source, value source); L12 is evaluated as its original checkpoint.
CANDIDATES = (("L12", "L12", "L12"), ("C1_policyL12_valueL0", "L12", "L0"), ("C2_policyL0_valueL12", "L0", "L12"))
PAIRS, SMOKE_PAIRS = 2000, 2
DIFFERENCE_BOOTSTRAP_SEED = 47340001
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


def load_sources():
    parents = {}
    for name, (relative, expected) in SOURCES.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Source checkpoint changed: {name}")
        parents[name] = load_frozen_actor_critic_checkpoint(ROOT / relative).eval().requires_grad_(False)
    return parents


def build_composition(key, policy, value, parents):
    path = MODELS_DIR / f"{key}.pt"
    sources = {"policy": identity(ROOT / SOURCES[policy][0]), "value": identity(ROOT / SOURCES[value][0])}
    if not path.exists():
        model = EntityAttentionPolicyValue(EntityAttentionPolicyValueConfig(
            policy_branch=asdict(parents[policy].config), value_branch=asdict(parents[value].config))).eval()
        model.policy_model.load_state_dict(parents[policy].state_dict(), strict=True)
        model.value_model.load_state_dict(parents[value].state_dict(), strict=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"checkpoint_schema": "splendor-frozen-actor-critic-v1",
            "format_version": MODEL_WEIGHTS_FORMAT_VERSION, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "config": asdict(model.config), "state_dict": model.state_dict(),
            "metadata": {"rules_version": RULESET_ID, "sources": sources, "inference_only": True,
                "composition": f"complete independent encoders: policy {policy}, value {value}"}}, path)
    model = load_frozen_actor_critic_checkpoint(path).eval()
    for branch, parent in (("policy", policy), ("value", value)):
        actual, expected = getattr(model, f"{branch}_model").state_dict(), parents[parent].state_dict()
        if actual.keys() != expected.keys() or any(not torch.equal(actual[k], expected[k]) for k in expected):
            raise ValueError(f"Composition parameters differ from source: {key}/{branch}")
    return model, {"checkpoint": str(path), "sha256": sha256(path), "sources": sources}


def verify_compositions(parents, models):
    games = [g for g in read_json(SNAPSHOT_HOLDOUT)["games"] if g["split"] == "validation"][:3]
    samples, sources = [], []
    for game in games:
        arrays, metadata = load_game(Path(game["path"]))
        positions = sorted({0, game["decision_count"] // 2, game["decision_count"] - 1})
        samples.extend(SelfPlaySample(arrays, metadata, i) for i in positions)
        sources.append({**identity(game["path"]), "decision_indices": positions})
    batch = collate_selfplay_samples(samples).decisions
    with torch.inference_mode():
        for model, policy, value in models.values():
            logits, values = model(batch)
            torch.testing.assert_close(logits, parents[policy](batch)[0], rtol=0, atol=0)
            torch.testing.assert_close(values, parents[value](batch)[1], rtol=0, atol=0)
    freeze(ART / "composition_verification.json", {"positions": len(samples), "sources": sources,
        "compositions": sorted(models), "device": "cpu", "rtol": 0, "atol": 0,
        "policy_matches_policy_source_exactly": True, "value_matches_value_source_exactly": True,
        "scope": "Real start/middle/end decisions from three round-012 validation games; no training."})


def panel_config(key, checkpoint, sha, template, fingerprint, pairs=PAIRS):
    raw = copy.deepcopy(template)
    raw["game_seeds"] = template["game_seeds"][:pairs]
    raw["policy_a"].update(checkpoint=checkpoint, name=key)
    raw["metadata"] = {"experiment": ART.name, "model": key, "mode": "search",
        "candidate_sha256": sha, "opponent_sha256": SOURCES["R37"][1], "catalog_fingerprint": fingerprint,
        "search_budget": "both S128N1 = 128 per world * 1 world = total128",
        "reference": f"{REFERENCE.name}/{REFERENCE_PANEL}: same template, first {pairs} of its opening seeds"}
    return FixedEvaluationConfig.from_dict(raw).to_dict()


def prepare():
    parents = load_sources()
    template = read_json(REFERENCE / "configs" / f"{REFERENCE_PANEL}.json")
    side = template["policy_a"]
    if (side["checkpoint"] != str(ROOT / SOURCES["L0"][0]) or template["policy_b"]["checkpoint"] != str(ROOT / SOURCES["R37"][0])
            or side["search_config"]["simulations"] != 128 or side["search_config"]["determinizations"] != 1
            or side["search_config"]["root_noise_fraction"] != 0. or len(template["game_seeds"]) != 10000):
        raise ValueError("Reference panel template is not the expected L0 vs R37 S128N1 panel")
    reference = [p for p in read_json(REFERENCE / "summary.json")["panels"] if p["id"] == REFERENCE_PANEL][0]
    if reference["status"] != "complete" or reference["completed_pairs"] != 10000:
        raise ValueError("Reference L0 panel is not complete")
    catalog = load_default_catalog()
    if catalog.data_fingerprint != template["metadata"]["catalog_fingerprint"]:
        raise ValueError("Current catalog differs from the reference panel")
    models, records = {}, []
    for key, policy, value in CANDIDATES:
        if key == "L12":
            record = {"checkpoint": str(ROOT / SOURCES["L12"][0]), "sha256": SOURCES["L12"][1],
                      "sources": {"full": identity(ROOT / SOURCES["L12"][0])}}
        else:
            model, record = build_composition(key, policy, value, parents)
            models[key] = (model, policy, value)
        panel_id = f"{len(records)+1:02d}_search_{key}_vs_R37"
        config = ART / "configs" / f"{panel_id}.json"
        freeze(config, panel_config(key, record["checkpoint"], record["sha256"], template, catalog.data_fingerprint))
        records.append({"id": panel_id, "model": key, "policy": policy, "value": value, "pairs": PAIRS,
            "config": identity(config), "output": str((ART / "panels" / f"{panel_id}.json").relative_to(ROOT)), **record})
    verify_compositions(parents, models)
    smoke = ART / "configs" / "smoke_C1.json"
    c1 = next(r for r in records if r["model"] == "C1_policyL12_valueL0")
    freeze(smoke, panel_config("C1_policyL12_valueL0", c1["checkpoint"], c1["sha256"], template,
                               catalog.data_fingerprint, pairs=SMOKE_PAIRS))
    plan = {"experiment": ART.name, "runner": identity(Path(__file__)), "panels": records,
        "opponent": {"checkpoint": str(ROOT / SOURCES["R37"][0]), "sha256": SOURCES["R37"][1]},
        "reference_L0": {"panel": REFERENCE_PANEL, "summary": identity(REFERENCE / "summary.json"),
            "config": identity(REFERENCE / "configs" / f"{REFERENCE_PANEL}.json"),
            "score": reference["score"], "ci95": reference["ci95"], "pairs": 10000},
        "catalog_fingerprint": catalog.data_fingerprint, "pairs_per_panel": PAIRS,
        "seeds": {"game_seed_start": template["game_seeds"][0], "game_seed_end": template["game_seeds"][PAIRS-1],
            "policy_seed_base": template["policy_seed_base"], "search_seed": side["search_config"]["seed"],
            "bootstrap_seed": template["bootstrap_seed"], "difference_bootstrap_seed": DIFFERENCE_BOOTSTRAP_SEED},
        "order": "L12, C1(policy L12 + value L0), C2(policy L0 + value L12); strictly serial",
        "scope": "Inference-only decomposition of the L0->L12 S128N1 gain into policy and value branches. "
                 "No training. L0 is the existing 10k reference; differences against it are unpaired normal approximations."}
    freeze(ART / "plan.json", plan)
    return plan


def frozen_source():
    """Copy the reference panel's frozen source; replace only the composition module."""
    destination, record_path = RUNTIME / "source", ART / "source.json"
    reference = read_json(REFERENCE / "source.json")
    for relative, expected in reference["snapshot_files"].items():
        if sha256(REFERENCE_SOURCE / relative) != expected:
            raise ValueError(f"Reference frozen source changed: {relative}")
    if not record_path.exists():
        if destination.exists():
            raise ValueError("Unrecorded frozen source directory exists")
        shutil.copytree(REFERENCE_SOURCE, destination, ignore=shutil.ignore_patterns("__pycache__"))
        for relative in PATCHED:
            shutil.copyfile(ROOT / relative, destination / relative)
        for source in (Path(__file__), ROOT / "scripts/run_graft_policy_value_l0_l12_20261104.sh",
                       *(ART / "configs").glob("*.json")):
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        files = {str(p.relative_to(destination)): sha256(p)
                 for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
        changed = sorted(k for k, v in files.items() if reference["snapshot_files"].get(k) != v)
        freeze(record_path, {"base": identity(REFERENCE / "source.json"), "patched_modules": list(PATCHED),
            "patch_purpose": "EntityAttentionPolicyValueConfig carries branch configs (continuation/market-relation models)",
            "changed_or_added_vs_reference": changed, "snapshot_files": files})
    for relative, expected in read_json(record_path)["snapshot_files"].items():
        if sha256(destination / relative) != expected:
            raise ValueError(f"Frozen evaluation source changed: {relative}")
    return destination


def panel_status(panel, plan, pairs=None):
    pairs = pairs or panel["pairs"]
    output = ROOT / panel["output"]
    row = {"id": panel["id"], "model": panel["model"], "status": "pending", "completed_pairs": 0}
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
    for side, key in (("policy_a", "candidate_sha256"), ("policy_b", "opponent_sha256")):
        if result["checkpoint_identity"][side]["sha256"] != config["metadata"][key]:
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


def _se(ci):
    return (ci["upper"] - ci["lower"]) / (2 * 1.959963984540054)


def decomposition(plan, rows):
    by = {r["model"]: r for r in rows}
    if any(by[k]["status"] != "complete" for k, _, _ in CANDIDATES):
        return None
    l0 = plan["reference_L0"]
    l0_score, l0_se = l0["score"], _se(l0["ci95"])
    scores = {k: np.asarray(by[k]["pair_scores"]) for k, _, _ in CANDIDATES}
    if not (by["L12"]["seeds"] == by["C1_policyL12_valueL0"]["seeds"] == by["C2_policyL0_valueL12"]["seeds"]):
        raise ValueError("Candidate panels do not share opening seeds")
    l12, c1, c2 = scores["L12"], scores["C1_policyL12_valueL0"], scores["C2_policyL0_valueL12"]

    def vs_l0(name, vector, sign_l0):
        # Unpaired against the 10k L0 reference: pairing reduced variance by only ~1-5% in earlier panels.
        boot = paired_bootstrap_mean(vector, resamples=10000, seed=DIFFERENCE_BOOTSTRAP_SEED)
        estimate = boot.estimate + sign_l0 * l0_score
        se = math.hypot(_se(asdict(boot)), l0_se)
        return {"name": name, "estimate": estimate, "lower": estimate - 1.959963984540054 * se,
                "upper": estimate + 1.959963984540054 * se, "method": "candidate whole-pair bootstrap + 10k L0 CI, normal combination"}

    def paired(name, vector):
        boot = asdict(paired_bootstrap_mean(vector, resamples=10000, seed=DIFFERENCE_BOOTSTRAP_SEED))
        return {"name": name, **{k: boot[k] for k in ("estimate", "lower", "upper")}, "method": "paired whole-pair bootstrap, shared seeds"}

    return [vs_l0("total: L12 - L0", l12, -1), vs_l0("policy: C1 - L0", c1, -1), vs_l0("value: C2 - L0", c2, -1),
            vs_l0("interaction: L12 - C1 - C2 + L0", l12 - c1 - c2, +1),
            paired("value given new policy: L12 - C1", l12 - c1), paired("policy given new value: L12 - C2", l12 - c2),
            paired("policy graft vs value graft: C1 - C2", c1 - c2)]


def write_report(plan, rows):
    complete = all(row["status"] == "complete" for row in rows)
    parts = decomposition(plan, rows) if complete else None
    compact = [{k: v for k, v in r.items() if k not in ("seeds", "pair_scores")} for r in rows]
    write_json(ART / "summary.json", {"status": "complete" if complete else "incomplete", "panels": compact,
        "reference_L0": plan["reference_L0"], "decomposition": parts,
        "planned_pairs": PAIRS * len(rows), "completed_pairs": sum(r["completed_pairs"] for r in rows)})
    l0 = plan["reference_L0"]
    lines = ["# L12 与 L0/L12 policy・value 嫁接对 raw R37（双方 S128N1）", "",
        "每组 2000 pair／4000 局，同开局换先后手；开局为 L0 参照面板前 2000 个种子。", "",
        "| 面板 | policy | value | 完成pair | 得分率 | 95%整pair区间 |", "| --- | --- | --- | ---: | ---: | --- |",
        f"| L0 参照（{l0['panel']}，10000 pair） | L0 | L0 | 10000/10000 | {100*l0['score']:.3f}% | "
        f"[{100*l0['ci95']['lower']:.3f}%, {100*l0['ci95']['upper']:.3f}%] |"]
    for panel, row in zip(plan["panels"], rows):
        done = row["status"] == "complete"
        score = f"{100*row['score']:.3f}%" if done else "—"
        ci = f"[{100*row['ci95']['lower']:.3f}%, {100*row['ci95']['upper']:.3f}%]" if done else "—"
        lines.append(f"| {row['id']} | {panel['policy']} | {panel['value']} | {row['completed_pairs']}/{PAIRS} | {score} | {ci} |")
    if parts:
        lines += ["", "| 分解（百分点） | 估计 | 95%区间 | 方法 |", "| --- | ---: | --- | --- |"]
        lines += [f"| {p['name']} | {100*p['estimate']:+.3f} | [{100*p['lower']:+.3f}, {100*p['upper']:+.3f}] | {p['method']} |" for p in parts]
    lines += ["", "双方 S128N1（每世界128×1世界＝总128），配置A，无根噪声，T0，12贵族；对手 raw R37。",
        "与 L0 的差值为非配对正态合成（L0 用其 10k 面板区间）；三组候选之间共用开局，按整 pair 配对 bootstrap。",
        "单一训练轨迹的推理期嫁接，不训练；未完成面板不作结论。", ""]
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
    """Two pairs of the C1 graft through the exact frozen config-A path; written only under runtime."""
    output = RUNTIME / "smoke" / "smoke_C1.json"
    if output.exists():
        output.unlink()
    shutil.rmtree(Path(str(output) + ".games"), ignore_errors=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_child("smoke_C1", evaluation_command("artifacts/graft_policy_value_l0_l12_20261104/configs/smoke_C1.json",
        output, source, False), source)
    panel = {"id": "smoke_C1", "model": "C1_policyL12_valueL0", "pairs": SMOKE_PAIRS,
        "config": identity(ART / "configs" / "smoke_C1.json"), "output": str(output.relative_to(ROOT))}
    row = panel_status(panel, plan)
    if row["status"] != "complete":
        raise ValueError("Smoke did not complete")
    write_json(ART / "smoke.json", {"pairs": SMOKE_PAIRS, "status": "passed", "result": identity(output),
        "checks": "valid games, seat swap, actual S128N1 budget on both sides, checkpoint SHA",
        **{k: row[k] for k in ("a_wins", "draws", "a_losses", "exceptions", "illegal_actions",
            "engineering_truncations", "invariant_failures")}})
    return row


def run_sequence(plan, source):
    rows = [panel_status(p, plan) for p in plan["panels"]]
    write_report(plan, rows)
    for index, panel in enumerate(plan["panels"]):
        if stopping:
            raise KeyboardInterrupt
        if rows[index]["status"] == "complete":
            continue
        output = ROOT / panel["output"]
        resume = output.exists() or Path(str(output) + ".games").exists()
        run_child(panel["id"], evaluation_command(panel["config"]["path"], output, source, resume), source)
        rows[index] = panel_status(panel, plan)
        write_report(plan, rows)
        if rows[index]["status"] != "complete":
            raise ValueError(f"Panel stopped before completion: {panel['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="run", choices=("prepare", "smoke", "run", "status"))
    args = parser.parse_args()
    if args.action == "status":
        plan = read_json(ART / "plan.json")
        rows = [panel_status(p, plan) for p in plan["panels"]]
        print({"status": "complete" if all(r["status"] == "complete" for r in rows) else "incomplete",
            "panels": [{k: r[k] for k in ("id", "status", "completed_pairs")} for r in rows]}, flush=True)
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        source = frozen_source()
        if args.action == "prepare":
            write_report(plan, [panel_status(p, plan) for p in plan["panels"]])
            print({"status": "prepared_not_started", "panels": len(plan["panels"]), "pairs": PAIRS}, flush=True)
            return 0
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupt)
        if args.action == "smoke":
            print(smoke(plan, source)["status"], flush=True)
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
