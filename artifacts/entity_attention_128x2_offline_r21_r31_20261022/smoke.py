"""Finite real-data/GPU training and four two-game arena checks, separate from production."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run as experiment
from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.offline_early_stopping import EarlyStoppingOfflineConfig, run
from splendor.offline_mixed_budget import select_checkpoint


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires a real CUDA device")
    root = experiment.RUNTIME / "gpu_smoke"
    root.mkdir(parents=True, exist_ok=True)
    rows = read_json(experiment.SOURCE / "rounds/round-020/collection_index.json")["games"]
    selected = [g for split in ("train", "validation") for g in [r for r in rows if r["split"] == split][:2]]
    manifest = root / "pool.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in selected))
    config = EarlyStoppingOfflineConfig.from_dict(read_json(experiment.CONFIG))
    train_positions = sum(r["decision_count"] for r in selected if r["split"] == "train")
    config = replace(config, manifest=str(manifest), manifest_sha256=sha256(manifest), first_round=64, last_round=64,
        expected_games_per_round=4, cache_mib=64, validation_positions=32,
        draw_exposures=19.5*config.learner.batch_size/train_positions,
        learner=replace(config.learner, checkpoint_every_steps=5))
    run(config, root / "fit")
    selection = select_checkpoint(config, root / "fit")
    plan = read_json(experiment.ART / "plan.json")
    panels = []
    for panel in experiment.PANELS:
        raw = experiment.evaluation_config(plan, selection, {**panel, "pairs": 1})
        raw["workers"] = 1
        if panel["kind"] == "search":
            raw["execution"].update(search_workers=2, total_active_games=4, local_microbatch=4,
                max_batch_size=8, batch_buckets=[], inference_mode="eager", compile_mode=None)
            for side in ("policy_a", "policy_b"):
                raw[side]["search_config"]["simulations"] = 2
        raw["metadata"]["scope"] = "engineering smoke only;1pair; searchS2N1 total2; eager small GPU pool"
        raw = FixedEvaluationConfig.from_dict(raw).to_dict()
        config_path, output = root / f"{panel['id']}-config.json", root / f"{panel['id']}.json"
        write_json(config_path, raw)
        command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(config_path), "--output", str(output)]
        subprocess.run(command, cwd=experiment.ROOT, env={**os.environ, "PYTHONPATH": "src"}, check=True)
        result = read_json(output)
        if (not result["passed"] or result["completed_pairs"] != 1 or result["valid_games"] != 2
                or any(result[k] for k in ("exceptions", "illegal_actions", "engineering_truncations", "invariant_failures"))):
            raise ValueError("Arena smoke failed")
        for pair in result["pairs"]:
            for game in pair["games"]:
                for side in ("policy_a", "policy_b"):
                    stats = game[f"{side}_stats"]
                    if panel["kind"] == "search":
                        if stats["simulations"] != 2*stats["searchable_decisions"]:
                            raise ValueError("Smoke search budget differs")
                    elif stats.get("simulations", 0):
                        raise ValueError("Smoke greedy unexpectedly searched")
        panels.append({"id": panel["id"], "command": command, "config_sha256": sha256(config_path),
            "result_sha256": sha256(output), "valid_games": result["valid_games"], "passed": result["passed"],
            "exceptions": result["exceptions"], "illegal_actions": result["illegal_actions"],
            "engineering_truncations": result["engineering_truncations"]})
    write_json(experiment.ART / "gpu_smoke.json", {"scope": "finite engineering verification, no strength conclusion",
        "command": [sys.executable, *sys.argv], "environment": environment_identity(),
        "training": read_json(root / "fit/progress.json"), "selected_step": selection["selected_step"],
        "panels": panels, "passed": True})
    print({"passed": True, "panels": len(panels), "games": sum(p["valid_games"] for p in panels)}, flush=True)


if __name__ == "__main__":
    main()
