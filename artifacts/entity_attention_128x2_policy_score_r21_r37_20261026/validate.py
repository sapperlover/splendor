"""Finite real-data CPU check of the frozen policy + ownership offline entry."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig, game_continuation_targets
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.offline_continuation import (ContinuationOfflineConfig, initialize_learner, run,
                                          select_checkpoint)
from splendor.selfplay_data import load_game
from scripts.validate_continuation_auxiliary import checked_score_ownership_oracle

ROOT = Path.cwd()
NAME = "entity_attention_128x2_policy_score_r21_r37_20261026"
ART = ROOT / "artifacts" / NAME
WORK = ROOT / "runtime" / NAME / "cpu_validation"


def main():
    torch.set_num_threads(1)
    formal = ContinuationOfflineConfig.from_dict(read_json(ROOT /
        "configs/entity_attention_v1/offline_policy_score_r21_r37_20261026.json"))
    rows = [json.loads(line) for line in Path(formal.manifest).read_text().splitlines()]
    selected = [row for split in ("train", "validation")
        for row in [r for r in rows if r["round_index"] == 79 and r["split"] == split
                    and r["policy_samples"] > 0][:2]]
    assert len(selected) == 4
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = WORK / "games.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in selected))
    auxiliary = ContinuationAuxiliaryConfig(**formal.continuation_auxiliary)
    games = []
    for row in selected:
        arrays, metadata = load_game(Path(row["path"]))
        labels = game_continuation_targets(arrays, metadata, auxiliary)
        assert not any(key.startswith("shortterm_") for key in labels)
        np.testing.assert_array_equal(labels["score_ownership_target"],
                                      checked_score_ownership_oracle(arrays, metadata))
        games.append({**row, "sha256": sha256(Path(row["path"])), "score_oracle": True})
    train_positions = sum(r["decision_count"] for r in selected if r["split"] == "train")
    cfg = replace(formal, manifest=str(manifest), manifest_sha256=sha256(manifest), first_round=80,
        last_round=80, expected_games_per_round=4, cache_mib=8, training_workers=0,
        draw_exposures=(4*8-.01)/train_positions, validation_positions=16,
        validation_percent_step=50, encoder_grad_norm_every_steps=2,
        learner=replace(formal.learner, device="cpu", batch_size=8, history_batch_size=4,
                        validation_batch_size=32, checkpoint_every_steps=2))
    fresh = initialize_learner(cfg, {"engineering_check": True}, 4)
    assert fresh.total_train_step == 0 and not fresh.optimizer.state
    assert fresh.model.trainable_parameter_count() == 852498
    assert not hasattr(fresh.model, "shortterm_value_head")
    serial, resumed = WORK / "serial", WORK / "resumed"
    run(cfg, serial)
    run(replace(cfg, training_workers=4), resumed, max_steps_this_session=2)
    run(replace(cfg, training_workers=4), resumed)
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
    assert metrics["opponent_positions"] > 0 and metrics["score_ownership_positions"] > 0
    assert metrics["score_ownership_phase_payment_positions"] > 0
    assert "score_ownership_constant_mse_points" in metrics
    assert not any("shortterm" in key for key in metrics)
    prefetch = read_json(resumed / "prefetch_progress.json")
    assert prefetch["workers"] == 4 and 0 < prefetch["arena_bytes"] < 8*1024**2
    selection = select_checkpoint(replace(cfg, training_workers=4), resumed)
    assert selection["selection_objective"] == min(
        p["metrics"]["policy_kl"]+p["metrics"]["value_loss"] for p in selection["candidates"])
    model = b.build_model()
    def reject_auxiliary(*_):
        raise AssertionError("Inference executed an auxiliary head")
    hooks = [head.register_forward_pre_hook(reject_auxiliary)
             for head in (model.opponent_policy_head, model.score_ownership_head)]
    searches = []
    for seed in (902601, 902602, 902603):
        result = NativePUCTSearchPolicy(model, config=MCTSConfig(simulations=8, determinizations=1,
            seed=seed, root_noise_fraction=0.)).search(GameEngine.new(seed, mode="fast").decision())
        assert result.simulations == 8
        searches.append({"seed": seed, "simulations": result.simulations})
    for hook in hooks:
        hook.remove()
    write_json(ART / "cpu_validation.json", {"scope": "Finite CPU engineering check only; no strength result",
        "formal_config_sha256": sha256(ROOT / "configs/entity_attention_v1/offline_policy_score_r21_r37_20261026.json"),
        "games": games, "fresh_model_and_adamw": True, "parameters": 852498,
        "serial_vs_four_worker_resume_bitwise": ["model", "AdamW", "sampling_rng", "Torch_rng", "full_validation"],
        "steps_per_path": 4, "prefetch": prefetch, "metrics": metrics, "selection": selection,
        "native_searches": searches, "inference_auxiliary_calls": 0,
        "imported_modules": {name: __import__("splendor."+name, fromlist=[name]).__file__
                             for name in ("offline_continuation", "offline_objectives", "score_ownership",
                                          "continuation_auxiliary", "training_prefetch", "alphazero_train")}})
    print({"status": "passed", "games": len(games), "steps_per_path": 4, "native_simulations": 24})


if __name__ == "__main__":
    main()
