"""Finite CPU warm-start/resume and evaluation-plan check for the R43 fork."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

import run
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json


def verify():
    config = run.prepare_plan()
    reference = read_json(run.SHARED / "config.json")
    keys = ("manifest", "manifest_sha256", "learner", "sampling_seed", "validation_seed",
            "preserve_optimizer", "positions_per_source_split", "cache_mib", "draw_exposures",
            "full_holdout_exposures", "full_holdout_steps", "lr_drop_exposure", "late_lr_multiplier", "milestones")
    assert all(config[key] == reference[key] for key in keys)
    completed_panel = copy.deepcopy(config)
    completed_panel["pre_training_crossplay"].update(
        config=str(run.SHARED / "evaluation_configs/confirmation.json"),
        output=str(run.SHARED / "panels/confirmation.json"), pairs=500)
    run.run_crossplay(completed_panel)
    records = [json.loads(line) for line in Path(config["manifest"]).read_text().splitlines()]
    tiny = [next(r for r in records if r["round_index"] == 0 and r["split"] == split)
            for split in ("train", "validation")]
    parent = load_alphazero_checkpoint(run.INITIAL)
    with tempfile.TemporaryDirectory(prefix="splendor-r43-offline-", dir="/tmp") as directory:
        temporary = Path(directory)
        manifest = temporary / "games.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in tiny))

        def make(name):
            cfg = copy.deepcopy(config)
            cfg.update(manifest=str(manifest), manifest_sha256=sha256(manifest),
                       output=str(temporary / name / "run"), artifacts=str(temporary / name / "artifacts"),
                       draw_exposures=7.5 / tiny[0]["decision_count"], full_holdout_exposures=[0],
                       full_holdout_steps=[1, 2], positions_per_source_split=4, cache_mib=16)
            cfg["learner"].update(device="cpu", batch_size=4, history_batch_size=2,
                                  validation_batch_size=32, checkpoint_every_steps=1)
            path = temporary / name / "config.json"
            write_json(path, cfg)
            return cfg, path

        resumed, resume_path = make("resumed")
        continuous, continuous_path = make("continuous")

        def execute(path, limit=None):
            command = [sys.executable, "-u", config["training_script"], "--config", str(path)]
            if limit is not None:
                command += ["--max-steps-this-session", str(limit)]
            result = subprocess.run(command, cwd=run.ROOT, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)

        execute(resume_path, 1)
        initial = load_alphazero_checkpoint(Path(resumed["output"]) / "checkpoints/initial.pt")
        assert initial.model_sha256 == parent.model_sha256
        assert initial.parent_checkpoint["absolute_round"] == 43
        assert initial.total_train_step == parent.total_train_step == 72898
        assert initial.optimizer_state_dict["state"].keys() == parent.optimizer_state_dict["state"].keys()
        for key, state in initial.optimizer_state_dict["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, parent.optimizer_state_dict["state"][key][name], rtol=0, atol=0)
        execute(resume_path)
        execute(continuous_path)
        a = load_alphazero_checkpoint(Path(resumed["output"]) / "checkpoints/final.pt")
        b = load_alphazero_checkpoint(Path(continuous["output"]) / "checkpoints/final.pt")
        assert a.round_train_step == b.round_train_step == 2
        assert a.total_train_step == parent.total_train_step + 2
        assert a.sampling_rng_state == b.sampling_rng_state
        assert torch.equal(a.torch_cpu_rng_state, b.torch_cpu_rng_state)
        for key, value in a.model_state_dict.items():
            torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
        for key, state in a.optimizer_state_dict["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, b.optimizer_state_dict["state"][key][name], rtol=0, atol=0)
        assert (Path(resumed["output"]) / "references.i64").read_bytes() == (Path(continuous["output"]) / "references.i64").read_bytes()
        full = read_json(Path(resumed["artifacts"]) / "full_holdout_step_000000002.json")
        assert full["evaluation"]["metrics"]["__all_observed__/validation/all"]["positions"] == tiny[1]["decision_count"]
        old_output = run.base.OUTPUT
        run.base.OUTPUT = Path(resumed["output"])
        try:
            for kind in ("greedy", "search"):
                raw = run.base.evaluation_config(config, 2, kind)
                assert raw["policy_b"]["checkpoint"] == str(run.INITIAL)
                assert raw["policy_b"]["name"] == "original_R43"
                if kind == "search":
                    assert all(raw[s]["search_config"]["root_noise_fraction"] == 0 for s in ("policy_a", "policy_b"))
            confirmation = run.base.evaluation_config(config, 2, "search", confirm=True)
            assert len(confirmation["game_seeds"]) == 500
        finally:
            run.base.OUTPUT = old_output
    cross = read_json(run.CROSS / "config.json")
    assert len(cross["game_seeds"]) == 200
    assert cross["policy_a"]["checkpoint"] == str(run.CANDIDATE)
    assert cross["policy_b"]["checkpoint"] == str(run.INITIAL)
    seeds = [set(range(p["seed_start"], p["seed_start"] + p["pairs"]))
             for p in [config["pre_training_crossplay"], *config["strength_plan"].values()]
             if isinstance(p, dict) and "seed_start" in p]
    assert all(not left.intersection(right) for i, left in enumerate(seeds) for right in seeds[i+1:])
    result = {"status": "passed", "scope": "two-game temporary CPU smoke, four optimizer updates; no formal search or GPU training",
              "warm_model_and_AdamW_match_R43": True, "initial_absolute_round": 43,
              "continuous_vs_resumed": "model/AdamW/Torch RNG/sampling RNG/reference ledger bitwise identical",
              "same_pool_and_training_fields": list(keys), "full_holdout_count_matches_original_validation": True,
              "crossplay_pairs": 200, "greedy_and_search_plans_validate": True, "confirmation_pairs": 500,
              "new_evaluation_opener_banks_disjoint": True, "config_sha256": sha256(run.HERE / "config.json"),
              "completed_crossplay_resume_path_checked": True,
              "entrypoint_sha256": sha256(run.HERE / "run.py"), "shared_sources": config["shared_sources"]}
    write_json(run.HERE / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
