"""Finite CPU real-data check; run from the chosen frozen source directory."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
import sys

import torch

from artifacts.entity_attention_r37_104k_offline_lr_20261027.train import (
    R37OfflineConfig, initialize_learner, run, select_checkpoint)
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.offline_scratch import initialize_learner as scratch_initializer
from splendor.rng import DeterministicRng

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("lr1e4", "lr2e5"))
    tag = parser.parse_args().variant
    torch.set_num_threads(1)
    name = f"entity_attention_r37_104k_{tag}_20261027"
    work = ROOT / "runtime" / name / "cpu_validation"
    formal = R37OfflineConfig.from_dict(read_json(ROOT / f"configs/entity_attention_v1/offline_r37_104k_{tag}_20261027.json"))
    rows = [json.loads(line) for line in Path(formal.manifest).read_text().splitlines()]
    chosen = [r for split in ("train", "validation") for r in
        [r for r in rows if r["round_index"] == 79 and r["split"] == split and r["policy_samples"] > 0][:2]]
    assert len(chosen) == 4
    work.mkdir(parents=True, exist_ok=True)
    manifest = work / "games.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in chosen))
    positions = sum(r["decision_count"] for r in chosen if r["split"] == "train")
    cfg = replace(formal, manifest=str(manifest), manifest_sha256=sha256(manifest), first_round=80,
        last_round=80, expected_games_per_round=4, cache_mib=8, training_workers=0,
        draw_exposures=(4*8-.01)/positions, validation_positions=16, validation_percent_step=50,
        learner=replace(formal.learner, device="cpu", batch_size=8, history_batch_size=4,
                        validation_batch_size=32, checkpoint_every_steps=2))
    saved = load_alphazero_checkpoint(cfg.initial_checkpoint)
    initial = initialize_learner(cfg, {"engineering": True}, 4)
    assert initial.total_train_step == initial.round_train_step == 0 and not initial.optimizer.state
    assert initial.model.trainable_parameter_count() == 583682
    assert initial.sampling_rng.snapshot() == DeterministicRng(cfg.sampling_seed).snapshot()
    assert all(g["lr"] == cfg.learner.learning_rate for g in initial.optimizer.param_groups)
    for key, value in initial.model.state_dict().items():
        torch.testing.assert_close(value, saved.model_state_dict[key], rtol=0, atol=0)
    rng_after_warm_init = torch.get_rng_state().clone()
    scratch_initializer(cfg, {"engineering": True}, 4)
    torch.testing.assert_close(rng_after_warm_init, torch.get_rng_state(), rtol=0, atol=0)
    serial, resumed = work / "serial", work / "resumed"
    run(cfg, serial)
    parallel = replace(cfg, training_workers=4)
    run(parallel, resumed, max_steps_this_session=2)
    execution = work / "config.json"
    write_json(execution, asdict(parallel))
    command = [sys.executable, "-u", "-m", "artifacts.entity_attention_r37_104k_offline_lr_20261027.train",
               "--config", str(execution), "--output", str(resumed)]
    subprocess.run(command, check=True)
    subprocess.run(command+["--select-only"], check=True)
    a, b = [load_alphazero_checkpoint(p / "checkpoints/final.pt") for p in (serial, resumed)]
    assert a.total_train_step == b.total_train_step == 4
    assert a.replay_identity == b.replay_identity and a.sampling_rng_state == b.sampling_rng_state
    torch.testing.assert_close(a.torch_cpu_rng_state, b.torch_cpu_rng_state, rtol=0, atol=0)
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    assert a.optimizer_state_dict["param_groups"] == b.optimizer_state_dict["param_groups"]
    for pid, state in a.optimizer_state_dict["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)
    metrics = read_json(resumed / "full_validation/step-000000004.json")["metrics"]
    assert metrics == read_json(serial / "full_validation/step-000000004.json")["metrics"]
    assert not any(k.startswith(("opponent_", "shortterm_", "score_ownership_")) for k in metrics)
    selection = read_json(resumed / "selection.json")
    assert selection == select_checkpoint(parallel, resumed)
    assert selection["selection_objective"] == min(p["metrics"]["policy_kl"]+p["metrics"]["value_loss"] for p in selection["candidates"])
    assert sha256(Path(cfg.initial_checkpoint)) == cfg.initial_checkpoint_sha256
    write_json(work / "verification.json", {"variant": tag, "scope": "Finite CPU verification only, no formal training",
        "source_checkpoint_sha256": cfg.initial_checkpoint_sha256, "source_step": saved.total_train_step,
        "same_R37_weights": True, "fresh_AdamW": True, "same_offline_sampling_rng": True,
        "same_offline_torch_initial_rng": True, "steps_per_path": 4,
        "serial_vs_4worker_resume_bitwise": ["weights", "AdamW", "sampling_rng", "Torch_rng", "full_validation"],
        "lr": cfg.learner.learning_rate, "metrics": metrics, "selection": selection,
        "prefetch": read_json(resumed / "prefetch_progress.json"),
        "games": [{"path": r["path"], "sha256": sha256(Path(r["path"]))} for r in chosen],
        "imported_training_entry": sys.modules[R37OfflineConfig.__module__].__file__,
        "imported_core": sys.modules["splendor.offline_objectives"].__file__})
    print({"variant": tag, "status": "passed", "steps_per_path": 4, "fresh_AdamW": True})


if __name__ == "__main__":
    main()
