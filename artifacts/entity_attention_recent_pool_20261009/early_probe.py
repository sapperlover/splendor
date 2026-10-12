"""Evaluate three earlier checkpoints, then one selected 200-pair search panel."""
from __future__ import annotations

import argparse
import fcntl
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from compact_cache import CompactGameCache
from train import evaluate, load_records
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_train import LearnerConfig
from splendor.evaluation import FixedEvaluationConfig
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.paired_statistics import paired_bootstrap_difference, paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime/entity_attention_recent_pool_20261009/early_probe"
PROTOCOL = ART / "early_probe_config.json"
SELECTION = ART / "early_selection.json"
BATTLE_CONFIG = ART / "early_battle_config.json"


def prepare():
    train_config = read_json(ART / "config.json")
    checkpoints = ROOT / train_config["output"] / "checkpoints"
    available = sorted((int(p.stem.split("-")[1]), p) for p in checkpoints.glob("step-*.pt"))
    targets = [34398 * fraction / 4 for fraction in (1, 2, 3)]
    samples = []
    for target in targets:
        step, path = min(available, key=lambda item: (abs(item[0] - target), item[0]))
        samples.append({"target_step": target, "step": step, "checkpoint": str(path.relative_to(ROOT)),
                        "sha256": sha256(path)})
    protocol = {"protocol": "recent-pool-early-probe-v1", "interval": [0, 34398],
        "points": samples, "existing_endpoint_step": 34398,
        "manifest_sha256": train_config["manifest_sha256"], "training_config_sha256": sha256(ART / "config.json"),
        "selection_rule": "minimum full-heldout policy_KL + value_MSE among the three points and existing step34398; original training weights are 1:1; ties prefer earlier step",
        "selection_scope": "adaptive DEV selection by offline fitting only; not established strength optimum",
        "battle": "one selected checkpoint vs theta5000, shared DEV 200 paired seeds, both native S1024/N1; identical to final103194 panel apart from candidate and metadata"}
    if PROTOCOL.exists() and read_json(PROTOCOL) != protocol:
        raise ValueError("frozen early-probe protocol changed")
    write_json(PROTOCOL, protocol)
    return protocol


def fitting(protocol):
    config = read_json(ART / "config.json")
    manifest = Path(config["manifest"])
    if sha256(manifest) != protocol["manifest_sha256"]:
        raise ValueError("frozen pool manifest changed")
    records = load_records(manifest)
    validation = [r for r in records if r["split"] == "validation"]
    for r in validation:
        stat = Path(r["path"]).stat()
        if stat.st_size != r["size"] or stat.st_mtime_ns != r["mtime_ns"]:
            raise ValueError(f"validation archive changed: {r['path']}")
    index = make_index({"games": records}, config["cache_mib"] * 1024**2)
    index.cache = CompactGameCache(config["cache_mib"] * 1024**2)
    cfg = LearnerConfig(**config["learner"])
    reference = read_json(ART / "full_holdout_step_000034398.json")
    rows = []
    for point in [*protocol["points"], {"step": 34398}]:
        step = point["step"]
        output = ART / f"full_holdout_step_{step:09d}.json"
        if not output.exists():
            started = time.perf_counter()
            checkpoint = ROOT / point["checkpoint"]
            model = load_frozen_actor_critic_checkpoint(checkpoint, device=cfg.device)
            learner = SimpleNamespace(model=model, device=torch.device(cfg.device), config=cfg)
            result = evaluate(learner, index, records)
            write_json(output, {"step": step, "manifest_sha256": config["manifest_sha256"],
                "checkpoint": point["checkpoint"], "checkpoint_sha256": point["sha256"],
                "early_probe_config_sha256": sha256(PROTOCOL), "evaluation": result,
                "seconds": time.perf_counter() - started})
            del learner, model
        full = read_json(output)
        if full["manifest_sha256"] != config["manifest_sha256"] or full["step"] != step:
            raise ValueError("heldout result identity mismatch")
        if step != 34398 and full["checkpoint_sha256"] != point["sha256"]:
            raise ValueError("heldout checkpoint identity mismatch")
        result = full["evaluation"]
        if result["per_game"].keys() != reference["evaluation"]["per_game"].keys():
            raise ValueError("heldout games or stages changed")
        all_metrics = result["metrics"]["__all_observed__/validation/all"]
        if all_metrics["positions"] != 247453 or all_metrics["policy_weight"] != 195067:
            raise ValueError("full heldout population mismatch")
        for key, metrics in result["per_game"].items():
            expected = reference["evaluation"]["per_game"][key]
            if metrics["positions"] != expected["positions"] or metrics["policy_weight"] != expected["policy_weight"]:
                raise ValueError("per-game population mismatch")
        metrics = result["pool_weighted"]["validation"]
        # In the full set these weighted metrics must equal direct ratio-of-sums.
        if any(not math.isclose(metrics[k], all_metrics[k], abs_tol=1e-12) for k in metrics):
            raise ValueError("full metric aggregation mismatch")
        row = {"step": step, **metrics, "joint_objective": metrics["policy_kl"] + metrics["value_mse"],
            "result": str(output.relative_to(ROOT)), "result_sha256": sha256(output),
            "stage_metrics": {stage: {k: result["metrics"][f"__all_observed__/validation/{stage}"][k]
                for k in ("positions", "policy_kl", "value_mse", "value_correlation")}
                for stage in ("early_mid", "late")}}
        rows.append(row)
        print(f"完整留出 step={step}: KL={row['policy_kl']:.6f} MSE={row['value_mse']:.6f} sum={row['joint_objective']:.6f}", flush=True)
    selected = min(rows, key=lambda row: (row["joint_objective"], row["step"]))
    checkpoint = ROOT / config["output"] / "checkpoints" / f"step-{selected['step']:09d}.pt"
    selection = {"protocol": protocol["protocol"], "config_sha256": sha256(PROTOCOL),
        "selection_rule": protocol["selection_rule"], "rows": rows, "selected_step": selected["step"],
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
        "full_holdout_games": len(validation), "full_holdout_positions": 247453,
        "scope": protocol["selection_scope"]}
    if SELECTION.exists() and read_json(SELECTION) != selection:
        raise ValueError("frozen early checkpoint selection changed")
    write_json(SELECTION, selection)
    return selection


def battle_config(selection):
    config = read_json(ART / "battle_config.json")
    config["policy_a"]["checkpoint"] = str(ROOT / selection["checkpoint"])
    config["policy_a"]["name"] = f"recent_pool_early{selection['selected_step']}"
    config["metadata"] = {"experiment": "entity_attention_recent_pool_20261009", "panel": "selected_early_vs_theta5000",
        "candidate_step": selection["selected_step"], "candidate_sha256": selection["checkpoint_sha256"],
        "selection_sha256": sha256(SELECTION), "selection_rule": selection["selection_rule"],
        "template_sha256": sha256(ART / "battle_config.json"), "seeds": "shared DEV with final103194 and previous offline9500 panels"}
    FixedEvaluationConfig.from_dict(config)
    if BATTLE_CONFIG.exists() and read_json(BATTLE_CONFIG) != config:
        raise ValueError("frozen early battle configuration changed")
    write_json(BATTLE_CONFIG, config)
    return ART / "battles" / f"early_step{selection['selected_step']:09d}_vs_theta5000.json"


def analyze_battle(output, *, extended_selection=None):
    raw = read_json(output)
    seeds, values = serialized_pair_scores(raw)
    if seeds != tuple(range(14900000, 14900200)) or raw["completed_games"] != 400:
        raise ValueError("selected battle has incomplete paired seeds")
    ci = paired_bootstrap_mean(values, resamples=10000, seed=14920000).to_dict()
    if raw["paired_bootstrap"] != ci:
        raise ValueError("battle bootstrap recomputation mismatch")
    config = FixedEvaluationConfig.from_dict(read_json(BATTLE_CONFIG)).to_dict()
    if raw["config"] != config:
        raise ValueError("battle config identity mismatch")
    for name, identity in raw["checkpoint_identity"].items():
        if sha256(Path(identity["path"])) != identity["sha256"] or identity["path"] != config[name]["checkpoint"]:
            raise ValueError("battle model identity mismatch")
    result = {"protocol": "recent-pool-selected-early-battle-v1", "status": "complete",
        "candidate_step": read_json(SELECTION)["selected_step"], "pairs": 200, "games": 400,
        "score": raw["win_rate_score"], "ci95": ci,
        "completed_at": raw["updated_at"], "elapsed_seconds": raw["elapsed_seconds_this_invocation"],
        **{key: raw[key] for key in ("a_wins", "draws", "a_losses", "a_first", "a_second", "engineering_truncations", "exceptions", "illegal_actions", "invariant_failures", "policy_cycle_draws")},
        "output": str(output.relative_to(ROOT)), "output_sha256": sha256(output),
        "selection_sha256": sha256(SELECTION), "verification": "production paired statistics recomputed; config, model files, seeds and both seats validated", "matched_contrasts": {},
        "scope": "one fitting-selected early checkpoint, single opponent and budget, shared DEV seeds; differences are opponent-score contrasts, exploratory with no multiplicity correction; not isolated critic causality"}
    if extended_selection is not None:
        result["extended_selection_sha256"] = sha256(extended_selection)
    previous = [("final103194", ART / "battles/final103194_vs_theta5000.json"),
        ("offline9500", ROOT / "artifacts/entity_attention_hybrid_offline9500_20261009/battles/offline9500_vs_theta5000.json"),
        ("hybrid_r9", ROOT / "artifacts/entity_attention_hybrid_offline9500_20261009/battles/hybrid_r9_vs_theta5000.json")]
    for i, (name, path) in enumerate(previous):
        old = read_json(path)
        old_seeds, old_values = serialized_pair_scores(old)
        if old["checkpoint_identity"]["policy_b"] != raw["checkpoint_identity"]["policy_b"]:
            raise ValueError("old opponent identity changed")
        for key in ("policy_b", "execution", "game_seeds", "policy_seed_base", "max_completed_turns", "adjudicate_deterministic_policy_cycles"):
            if old["config"][key] != config[key]:
                raise ValueError(f"old comparison config differs: {key}")
        if old["config"]["policy_a"]["search_config"] != config["policy_a"]["search_config"]:
            raise ValueError("old candidate search differs")
        result["matched_contrasts"][f"early_minus_{name}"] = {
            **paired_bootstrap_difference(seeds, values, old_seeds, old_values, resamples=10000, seed=30300000+i).to_dict(),
            "source": str(path.relative_to(ROOT)), "source_sha256": sha256(path)}
    write_json(ART / "early_battle_summary.json", result)
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "run", "analyze"), default="run", nargs="?")
    parser.add_argument("--extended-selection", action="store_true", help="verify the exposure5-to8 extension still selects this candidate")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare()
        if args.action == "prepare":
            return
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        session = RUNTIME / str(time.time_ns())
        identity = source_identity(ROOT, session)
        for name in ("early_probe.py", "early_probe_config.json", "train.py", "compact_cache.py"):
            shutil.copyfile(ART / name, session / name)
        write_json(session / "identity.json", {**identity, "environment": environment_identity(),
            "command": [sys.executable, *sys.argv], "start_unix": time.time(), "probe_config_sha256": sha256(PROTOCOL)})
        if args.action == "analyze":
            selection = read_json(SELECTION)
        else:
            selection = fitting(protocol)
        extended_selection = ART / "extended_selection.json" if args.extended_selection else None
        if extended_selection is not None:
            extended = read_json(extended_selection)
            if (extended["selected_step"], extended["checkpoint_sha256"]) != (selection["selected_step"], selection["checkpoint_sha256"]):
                raise ValueError("extended fitting selection requires a different battle candidate")
        output = battle_config(selection)
        if args.action == "evaluate":
            return
        shutil.copyfile(BATTLE_CONFIG, session / BATTLE_CONFIG.name)
        shutil.copyfile(SELECTION, session / SELECTION.name)
        if extended_selection is not None:
            shutil.copyfile(extended_selection, session / extended_selection.name)
        if args.action == "run":
            command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(BATTLE_CONFIG),
                "--output", str(output), "--progress-every", "10", "--summary-every", "10"]
            if output.exists() or output.with_suffix(output.suffix + ".games").exists():
                command.append("--resume")
            write_json(session / "battle_command.json", {"command": command, "start_unix": time.time()})
            try:
                subprocess.run(command, cwd=ROOT, check=True)
            except subprocess.CalledProcessError as error:
                if error.returncode == 130:
                    raise SystemExit(130) from None
                raise
        analyze_battle(output, extended_selection=extended_selection)


if __name__ == "__main__":
    main()
