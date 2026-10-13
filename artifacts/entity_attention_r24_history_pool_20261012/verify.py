"""Four CPU updates on two original games; never runs formal training or arenas."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from pool import HERE, ROOT
from run import evaluation_config
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json


def verify():
    config = read_json(HERE / "config.json")
    records = [json.loads(line) for line in Path(config["manifest"]).read_text().splitlines()]
    tiny = [next(r for r in records if r["round_index"] == 0 and r["split"] == split)
            for split in ("train", "validation")]
    parent = load_alphazero_checkpoint(config["initial_checkpoint"]["path"])
    with tempfile.TemporaryDirectory(prefix="splendor-r24-offline-", dir="/tmp") as directory:
        temporary = Path(directory)
        manifest = temporary / "games.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in tiny))
        def make(name):
            cfg = copy.deepcopy(config)
            cfg.update(manifest=str(manifest), manifest_sha256=sha256(manifest),
                       output=str(temporary / name / "run"), artifacts=str(temporary / name / "artifacts"),
                       draw_exposures=7.5 / tiny[0]["decision_count"],
                       full_holdout_exposures=[0], full_holdout_steps=[1, 2],
                       positions_per_source_split=4, cache_mib=16,
                       baseline_checkpoints={})
            cfg["learner"].update(device="cpu", batch_size=4, history_batch_size=2,
                                   validation_batch_size=32, checkpoint_every_steps=1)
            path = temporary / name / "config.json"
            write_json(path, cfg)
            return cfg, path
        resumed, resumed_path = make("resumed")
        continuous, continuous_path = make("continuous")
        def execute(path, limit=None):
            command = [sys.executable, "-u", str(HERE / "train.py"), "--config", str(path)]
            if limit is not None:
                command += ["--max-steps-this-session", str(limit)]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        execute(resumed_path, 1)
        initial = load_alphazero_checkpoint(Path(resumed["output"]) / "checkpoints/initial.pt")
        assert initial.model_sha256 == parent.model_sha256
        assert initial.total_train_step == parent.total_train_step
        assert initial.optimizer_state_dict["state"].keys() == parent.optimizer_state_dict["state"].keys()
        for key, state in initial.optimizer_state_dict["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, parent.optimizer_state_dict["state"][key][name], rtol=0, atol=0)
        for path in (Path(resumed["output"]) / "training_metrics.jsonl",
                     Path(resumed["artifacts"]) / "curve.jsonl"):
            with path.open("a") as handle:
                handle.write('{"step":2,"uncommitted":')
        with (Path(resumed["output"]) / "references.i64").open("ab") as handle:
            handle.write(b"uncommitted reference tail")
        execute(resumed_path)
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
        ledger = Path(resumed["output"]) / "references.i64"
        assert ledger.read_bytes() == (Path(continuous["output"]) / "references.i64").read_bytes()
        rows = np.fromfile(ledger, dtype="<i8").reshape(-1, 4)
        assert rows.shape == (8, 4)
        assert set(map(tuple, rows[:, 1:3])) == {(tiny[0]["round_index"], tiny[0]["game_id"])}
        full = read_json(Path(resumed["artifacts"]) / "full_holdout_step_000000002.json")
        assert full["evaluation"]["metrics"]["__all_observed__/validation/all"]["positions"] == tiny[1]["decision_count"]
        import run
        original_output = run.OUTPUT
        run.OUTPUT = Path(resumed["output"])
        try:
            for kind in ("greedy", "search"):
                raw = evaluation_config(config, 2, kind)
                assert raw["policy_b"]["checkpoint"] == config["initial_checkpoint"]["path"]
                if kind == "search":
                    for side in ("policy_a", "policy_b"):
                        assert raw[side]["search_config"]["simulations"] == 1024
                        assert raw[side]["search_config"]["root_noise_fraction"] == 0
            confirmation = evaluation_config(config, 2, "search", confirm=True)
            assert len(confirmation["game_seeds"]) == 500
        finally:
            run.OUTPUT = original_output
        verification = {
            "status": "passed", "device": "cpu", "optimizer_updates_total": 4,
            "scope": "two-game temporary smoke only; formal training and arenas not started",
            "warm_model_matches_R24": True, "initial_AdamW_moments_match_R24": True,
            "two_steps_equal_one_plus_one": "model/AdamW/Torch RNG/sampling RNG/reference ledger bitwise identical",
            "uncommitted_and_torn_log_ledger_tails_reconciled": True,
            "training_draws_only_original_Train": True, "full_holdout_positions_match_original_validation": True,
            "greedy_and_search_configs_validate": True, "confirmation_pairs": 500,
            "source_sha256": {p.name: sha256(p) for p in HERE.glob("*.py")},
        }
        write_json(HERE / "verification.json", verification)
        print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
