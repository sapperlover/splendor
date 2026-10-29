"""Finite real-data scratch/shortterm CUDA fitting, resume and two one-pair engineering checks."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run as experiment
from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.offline_continuation import ContinuationOfflineConfig, initialize_learner, run, select_checkpoint
from splendor.alphazero_checkpoint import load_alphazero_checkpoint


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires a real CUDA device")
    root = experiment.RUNTIME / "gpu_smoke"
    root.mkdir(parents=True, exist_ok=True)
    cfg = read_json(experiment.CONFIG)
    rows = [json.loads(line) for line in Path(cfg["manifest"]).read_text().splitlines()]
    manifest = root / "pool.jsonl"
    # One train and one validation game per generation exercise all17 data sources.
    selected = [g for r in range(63, 80) for split in ("train", "validation")
                for g in [row for row in rows if row["round_index"] == r and row["split"] == split][:1]]
    manifest.write_text("".join(json.dumps(r)+"\n" for r in selected))
    config = ContinuationOfflineConfig.from_dict(cfg)
    train_positions = sum(r["decision_count"] for r in selected if r["split"] == "train")
    config = replace(config, manifest=str(manifest), manifest_sha256=sha256(manifest),
        expected_games_per_round=2, cache_mib=128, validation_positions=32,
        draw_exposures=5.5*config.learner.batch_size/train_positions,
        learner=replace(config.learner, checkpoint_every_steps=2))
    initial = initialize_learner(config, {"smoke": True}, 6)
    assert initial.model.trainable_parameter_count() == 650757
    assert not hasattr(initial.model, "opponent_policy_head") and not initial.optimizer.state
    del initial
    torch.cuda.empty_cache()
    run(config, root / "fit", max_steps_this_session=2)
    run(config, root / "fit")
    selection = select_checkpoint(config, root / "fit")
    plan = read_json(experiment.ART / "plan.json")
    panels = []
    for panel in experiment.PANELS:
        raw = experiment.evaluation_config(plan, selection, {**panel, "pairs": 1})
        raw["workers"] = 1
        if panel["kind"] == "search":
            raw["execution"].update(search_workers=2, total_active_games=2, local_microbatch=2,
                max_batch_size=8, batch_buckets=[2, 4, 8])
            for side in ("policy_a", "policy_b"):
                raw[side]["search_config"]["simulations"] = 2
        raw["metadata"]["scope"] = "engineering smoke only;1pair; searchS2N1 total2; compiled native-cudagraph/pipeline2 small GPU pool"
        raw = FixedEvaluationConfig.from_dict(raw).to_dict()
        config_path, output = root / f"{panel['id']}-config.json", root / f"{panel['id']}.json"
        write_json(config_path, raw)
        command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(config_path), "--output", str(output)]
        subprocess.run(command, cwd=experiment.RUNTIME / "source", env=os.environ.copy(), check=True)
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
    from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig, game_continuation_targets
    from splendor.selfplay_data import load_game
    label_sources = []
    for row in selected:
        arrays, metadata = load_game(Path(row["path"]))
        targets = game_continuation_targets(arrays, metadata, ContinuationAuxiliaryConfig(**config.continuation_auxiliary))
        label_sources.append({"path": row["path"], "sha256": sha256(Path(row["path"])),
            "round_index": row["round_index"], "split": row["split"],
            "main_positions": int(targets["shortterm_main_mask"].sum()),
            "shortterm_positions": int(targets["shortterm_value_mask"].sum())})
    updates = [json.loads(line) for line in (root / "fit/training_metrics.jsonl").read_text().splitlines()
               if json.loads(line)["event"] == "train_step"]
    assert sum(r["shortterm_positions"] for r in updates) > 0
    assert not list((root / "fit").rglob("*.bin"))
    assert not list((root / "fit").rglob("*.steps.jsonl"))
    assert not list((root / "fit/references").rglob("*.metrics.jsonl"))
    payload = load_alphazero_checkpoint(root / "fit/checkpoints/final.pt")
    assert payload.model_config["architecture"] == "entity_attention_continuation_v1"
    assert payload.model_config["predict_opponent_policy"] is False
    write_json(experiment.ART / "gpu_smoke.json", {"scope": "finite engineering verification, no strength conclusion",
        "command": [sys.executable, *sys.argv], "environment": environment_identity(),
        "config": asdict(config), "label_sources": label_sources,
        "training": read_json(root / "fit/progress.json"), "selected_step": selection["selected_step"],
        "resume_after_steps": 2, "parameters": 650757, "lightweight_recording_checked": True,
        "panels": panels, "passed": True})
    print({"passed": True, "panels": len(panels), "games": sum(p["valid_games"] for p in panels)}, flush=True)


if __name__ == "__main__":
    main()
