"""Extend heldout fitting checks to mean exposures 5, 6, 7 and existing 8."""
from __future__ import annotations

import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from compact_cache import CompactGameCache
from train import evaluate, load_records
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_train import LearnerConfig
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]


def main():
    config = read_json(ART / "config.json")
    manifest = Path(config["manifest"])
    if sha256(manifest) != config["manifest_sha256"]:
        raise ValueError("frozen pool manifest changed")
    records = load_records(manifest)
    ntrain = sum(r["decision_count"] for r in records if r["split"] == "train")
    checkpoints = ROOT / config["output"] / "checkpoints"
    available = [(int(p.stem.split("-")[1]), p) for p in checkpoints.glob("step-*.pt")]
    points = []
    for exposure in (5, 6, 7):
        target = exposure * ntrain / config["learner"]["batch_size"]
        step, path = min(available, key=lambda item: (abs(item[0]-target), item[0]))
        points.append({"target_exposure": exposure, "target_step": target, "step": step,
            "actual_exposure": step * config["learner"]["batch_size"] / ntrain,
            "checkpoint": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    protocol = {"protocol": "recent-pool-exposure5-to8-probe-v1", "points": points,
        "existing_exposure8_step": 68796, "manifest_sha256": config["manifest_sha256"],
        "training_config_sha256": sha256(ART / "config.json"),
        "prior_selection_sha256": sha256(ART / "early_selection.json"),
        "selection_rule": "minimum full-heldout policy_KL + value_MSE over all eight measured points from step9000 through step68796; 1:1 original training weights; ties prefer earlier step",
        "scope": "same complete 2935-game DEV holdout, adaptive fitting selection; no additional training"}
    protocol_path = ART / "mid_probe_config.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("frozen exposure5-to8 protocol changed")
    write_json(protocol_path, protocol)
    session = ROOT / "runtime/entity_attention_recent_pool_20261009/mid_probe" / str(time.time_ns())
    identity = source_identity(ROOT, session)
    for name in ("mid_probe.py", "mid_probe_config.json", "train.py", "compact_cache.py"):
        shutil.copyfile(ART / name, session / name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    write_json(session / "identity.json", {**identity, "environment": environment_identity(),
        "command": [sys.executable, *sys.argv], "start_unix": time.time(), "protocol_sha256": sha256(protocol_path)})
    for record in records:
        if record["split"] != "validation":
            continue
        stat = Path(record["path"]).stat()
        if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError("original validation archive changed")
    index = make_index({"games": records}, config["cache_mib"] * 1024**2)
    index.cache = CompactGameCache(config["cache_mib"] * 1024**2)
    cfg = LearnerConfig(**config["learner"])
    reference = read_json(ART / "full_holdout_step_000034398.json")["evaluation"]
    for point in points:
        output = ART / f"full_holdout_step_{point['step']:09d}.json"
        if not output.exists():
            started = time.perf_counter()
            model = load_frozen_actor_critic_checkpoint(ROOT / point["checkpoint"], device=cfg.device)
            result = evaluate(SimpleNamespace(model=model, device=torch.device(cfg.device), config=cfg), index, records)
            write_json(output, {"step": point["step"], "manifest_sha256": config["manifest_sha256"],
                "checkpoint": point["checkpoint"], "checkpoint_sha256": point["sha256"],
                "mid_probe_config_sha256": sha256(protocol_path), "evaluation": result,
                "seconds": time.perf_counter()-started})
            del model
        full = read_json(output)
        if full["checkpoint_sha256"] != point["sha256"] or full["manifest_sha256"] != config["manifest_sha256"]:
            raise ValueError("full holdout identity mismatch")
        result = full["evaluation"]
        if result["per_game"].keys() != reference["per_game"].keys():
            raise ValueError("full holdout games changed")
        for key, metrics in result["per_game"].items():
            if any(metrics[field] != reference["per_game"][key][field] for field in ("positions", "policy_weight")):
                raise ValueError("full holdout per-game population changed")
        print(f"完整留出 exposure≈{point['target_exposure']} step={point['step']}: {result['pool_weighted']['validation']}", flush=True)
    rows = []
    for step in sorted([9000, 17000, 26000, 34398, *[p["step"] for p in points], 68796]):
        output = ART / f"full_holdout_step_{step:09d}.json"
        result = read_json(output)["evaluation"]
        metrics = result["pool_weighted"]["validation"]
        direct = result["metrics"]["__all_observed__/validation/all"]
        if direct["positions"] != 247453 or direct["policy_weight"] != 195067:
            raise ValueError("full heldout population mismatch")
        if any(not math.isclose(metrics[k], direct[k], abs_tol=1e-12) for k in metrics):
            raise ValueError("full metric aggregation mismatch")
        rows.append({"step": step, "actual_exposure": step*cfg.batch_size/ntrain, **metrics,
            "joint_objective": metrics["policy_kl"]+metrics["value_mse"],
            "result": str(output.relative_to(ROOT)), "result_sha256": sha256(output),
            "stage_metrics": {stage: {field: result["metrics"][f"__all_observed__/validation/{stage}"][field]
                for field in ("positions", "policy_kl", "value_mse", "value_correlation")}
                for stage in ("early_mid", "late")}})
    selected = min(rows, key=lambda row: (row["joint_objective"], row["step"]))
    checkpoint = checkpoints / f"step-{selected['step']:09d}.pt"
    selection = {"protocol": protocol["protocol"], "config_sha256": sha256(protocol_path),
        "selection_rule": protocol["selection_rule"], "rows": rows, "selected_step": selected["step"],
        "minimum_mse_step": min(rows, key=lambda row: row["value_mse"])["step"],
        "minimum_kl_step": min(rows, key=lambda row: row["policy_kl"])["step"],
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint), "scope": protocol["scope"]}
    selection_path = ART / "extended_selection.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError("frozen extended selection changed")
    write_json(selection_path, selection)
    print(__import__("json").dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
