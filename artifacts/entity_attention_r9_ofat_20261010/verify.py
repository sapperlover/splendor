"""CPU verification of the completed OFAT experiment and actual sample streams."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import read_json, sha256, write_json

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUN = ROOT / "runs" / ART.name


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def reference_rows(directory, batch_size, first, last):
    steps = {}
    for path in sorted(directory.glob("*.bin")):
        data = np.frombuffer(path.read_bytes(), dtype="<i8").reshape(-1, 4)
        boundaries = np.r_[0, np.flatnonzero(np.diff(data[:, 0])) + 1, len(data)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if left == right:
                continue
            step = data[left, 0]
            if first <= step <= last:
                block = data[left:right]
                assert len(block) == batch_size, (path, step, len(block))
                if step in steps:
                    assert np.array_equal(steps[step], block), (path, "resumed samples changed", step)
                steps[step] = block
    assert set(steps) == set(range(first, last + 1)), directory
    return np.concatenate([steps[s] for s in range(first, last + 1)])


def main():
    torch.set_num_threads(1)
    plan = read_json(ART / "plan.json")
    summary = read_json(ART / "summary.json")
    assert summary["status"] == "complete" and summary["completed_panels"] == 10
    for key in ("initial_checkpoint", "dataset", "latest_train_references", "runner", "statistics_helper"):
        assert sha256(ROOT / plan[key]["path"]) == plan[key]["sha256"]
    records = read_json(ART / "dataset.json")["games"]
    source = torch.load(ROOT / plan["initial_checkpoint"]["path"], map_location="cpu", weights_only=True)
    parent = torch.load(plan["start"]["path"], map_location="cpu", weights_only=True)
    assert equal(source["model_state_dict"], parent["model_state_dict"])
    assert equal(source["optimizer_state_dict"], parent["optimizer_state_dict"])
    original = ROOT / "runs/entity_attention_p4_pool_iteration_20261010/rounds/round-008/usage/online"
    extension = ROOT / "runs/entity_attention_r9_step_extension_20261010/usage"
    baseline = np.concatenate((reference_rows(original, 256, 1, 500), reference_rows(extension, 256, 501, 2000)))
    streams, panels, total_draws = {}, [], 0
    preserved = ("model_state_dict", "model_sha256", "sampling_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_states", "parent_checkpoint")
    for panel in plan["panels"]:
        name = panel["id"]
        root = RUN / name
        initial = torch.load(root / "checkpoints/initial.pt", map_location="cpu", weights_only=True)
        assert all(equal(initial[k], source[k]) for k in preserved), name
        optimizer = copy.deepcopy(source["optimizer_state_dict"])
        for group in optimizer["param_groups"]:
            group["lr"] = panel["learner"]["learning_rate"]
        assert equal(initial["optimizer_state_dict"], optimizer), name
        assert initial["learner_config"] == panel["learner"]
        assert initial["round_train_step"] == 0 and initial["total_train_step"] == 38398
        assert initial["round_steps_target"] == 2000
        changes = {k for k in panel["learner"] if panel["learner"][k] != plan["base_learner"][k]}
        expected = {panel["axis"]} if panel["axis"] != "pool_games" else set()
        if name == "batch_64":
            expected.add("history_batch_size")  # Explicit zero-history drawer never reads this field.
        assert changes == expected, (name, changes, expected)
        rows = reference_rows(root / "usage", panel["learner"]["batch_size"], 1, 2000)
        selected = records[-panel["pool_games"]:]
        allowed = np.zeros(9000, dtype=bool)
        counts = np.zeros(9000, dtype=np.int64)
        for game in selected:
            key = game["round_index"] * 1000 + game["game_id"]
            allowed[key] = game["split"] == "train"
            counts[key] = game["decision_count"]
        keys = rows[:, 1] * 1000 + rows[:, 2]
        assert allowed[keys].all(), name
        assert ((rows[:, 3] >= 0) & (rows[:, 3] < counts[keys])).all(), name
        total_draws += len(rows)
        if panel["axis"] == "learning_rate":
            assert np.array_equal(rows, baseline), (name, "learning-rate sample stream differs")
        if panel["axis"] == "batch_size":
            assert np.array_equal(rows[:128000, 1:], baseline[:128000, 1:]), (name, "equal-draw sample stream differs")
        generation_draws = {str(r): int((rows[:, 1] == r-1).sum()) for r in panel["generations"]}
        panels.append({"id": name, "draws": len(rows), "draws_by_generation": generation_draws,
            "start_model_and_rng_exact": True, "start_adamw_moments_exact": True,
            "functional_config_changes": sorted(changes), "holdout_draws": 0})
        del rows, initial, optimizer
    points = 0
    for panel_id, steps in [("baseline", [b["step"] for b in plan["baselines"]]), *[(p["id"], p["steps"]) for p in plan["panels"]]]:
        for step in steps:
            data = read_json(ART / "fitting" / panel_id / f"step-{step:09d}.json")
            checkpoint = ROOT / data["checkpoint"]["path"]
            assert sha256(checkpoint) == data["checkpoint"]["sha256"]
            raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert raw["total_train_step"] == 38398 + step
            assert raw["round_train_step"] == step
            per_game = {k: v for k, v in data["validation"]["per_game"].items() if k.endswith("/all")}
            assert len(per_game) == 100
            assert sum(g["positions"] for g in per_game.values()) == 8174
            assert sum(g["policy_weight"] for g in per_game.values()) == 6688
            aggregate = data["validation"]["metrics"]["g009/validation/all"]
            for metric, numerator, denominator in (("policy_kl", "kl", "policy_weight"), ("value_mse", "sq_error", "positions")):
                value = sum(g[numerator] for g in per_game.values()) / sum(g[denominator] for g in per_game.values())
                assert abs(value - aggregate[metric]) < 1e-12
            train = data["latest_train"]["metrics"]["g009/train/all"]
            assert train["positions"] == 2048 and train["policy_weight"] == 1652
            points += 1
    sessions = [read_json(p) for p in (ART / "training_sessions").glob("*/*.json")]
    updates = sum(s["steps_this_session"] for s in sessions)
    assert updates == 20000
    report = {"passed": True, "new_training_updates": updates, "new_sample_draws": total_draws,
        "verified_points": points, "panels": panels, "start_matches_round8_weights_and_adamw": True,
        "lr_streams_equal_original_2000_step_stream": True, "batch_equal_draw_streams_exact": True,
        "validation_draws": 0, "training_seconds": sum(s["training_seconds"] for s in sessions),
        "plan_sha256": sha256(ART / "plan.json"), "summary_sha256": sha256(ART / "summary.json")}
    write_json(ART / "verification.json", report)
    print({k: v for k, v in report.items() if k != "panels"})


if __name__ == "__main__":
    main()
