"""Finite real-data CPU updates and native roots; leave the active GPU run alone."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import torch

from run_experiment import ART, ROOT, VARIANTS, configure
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig, game_continuation_targets, continuation_targets
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.offline_continuation import ContinuationOfflineConfig, initialize_learner
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples, load_game


def main():
    torch.set_num_threads(1)
    parent_initial = ROOT / "runs/entity_attention_128x2_shortterm_value_r21_r37_20261023/checkpoints/initial.pt"
    original = load_alphazero_checkpoint(parent_initial)
    original_model = original.build_model(device="cpu")
    assert original_model.config.predict_shortterm_value is True  # Legacy config lacks this field.
    results = []
    for key, arm in VARIANTS.items():
        entry = configure(key)
        config = ContinuationOfflineConfig.from_dict(read_json(entry.CONFIG))
        config = replace(config, learner=replace(config.learner, device="cpu"))
        auxiliary = ContinuationAuxiliaryConfig(**config.continuation_auxiliary)
        learner = initialize_learner(config, {"engineering_smoke": key}, 1)
        assert learner.model.trainable_parameter_count() == arm["parameters"]
        assert not learner.optimizer.state
        # The new switch does not perturb original main/value auxiliary initialization.
        for name, value in learner.model.state_dict().items():
            if name in original.model_state_dict:
                torch.testing.assert_close(value, original.model_state_dict[name], rtol=0, atol=0)
        rows = [json.loads(line) for line in Path(config.manifest).read_text().splitlines()]
        selected = [r for r in rows if r["split"] == "train"][-3:]
        samples, games = [], []
        for row in selected:
            path = Path(row["path"])
            arrays, metadata = load_game(path)
            targets = game_continuation_targets(arrays, metadata, auxiliary)
            assert bool(any(k.startswith("shortterm_") for k in targets)) == bool(arm["value"])
            enriched = arrays | targets
            samples.extend(SelfPlaySample(enriched, metadata, int(i))
                for i in np.linspace(0, row["decision_count"]-1, 16, dtype=int))
            games.append({"path": row["path"], "sha256": sha256(path), "round_index": row["round_index"]})
        policies = [s for s in samples if s.arrays["policy_weight"][s.index] > 0][:16]
        values = samples[::3][:16]
        policy_batch, value_batch = collate_selfplay_samples(policies), collate_selfplay_samples(values)
        metrics = learner.train_step(policy_batch, value_batch=value_batch, references=[],
            continuation_auxiliary=auxiliary, continuation_labels=continuation_targets(values, auxiliary),
            record_encoder_grad_norms=True)
        if arm["value"]:
            assert metrics["shortterm_positions"] > 0
        else:
            assert not any("shortterm" in name for name in metrics)
        if arm["policy"]:
            assert metrics["opponent_positions"] > 0 and metrics["encoder_grad_norm_opponent_policy"] > 0
        checkpoint = ROOT / "runtime" / ART.name / "validation" / key / "model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        learner.save(checkpoint)
        restored = load_frozen_actor_critic_checkpoint(checkpoint, device="cpu")
        with torch.inference_mode():
            for a, b in zip(learner.model(policy_batch.decisions), restored(policy_batch.decisions)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        def forbidden(*_):
            raise AssertionError("Inference executed a training-only auxiliary head")
        hooks = [getattr(restored, head).register_forward_pre_hook(forbidden)
                 for head in ("shortterm_value_head", "opponent_policy_head") if hasattr(restored, head)]
        searches = []
        for seed in (902301, 902302, 902303):
            result = NativePUCTSearchPolicy(restored, config=MCTSConfig(simulations=8, determinizations=1,
                seed=seed, root_noise_fraction=0)).search(GameEngine.new(seed, mode="fast").decision())
            assert result.simulations == 8
            searches.append({"seed": seed, "simulations": result.simulations})
        for hook in hooks:
            hook.remove()
        results.append({"variant": key, "parameters": arm["parameters"], "config_sha256": sha256(entry.CONFIG),
            "games": games, "metrics": metrics, "native_searches": searches,
            "checkpoint_sha256": sha256(checkpoint), "strict_reload": True,
            "inference_auxiliary_calls": 0, "original_initialization_bitwise_preserved": True})
    write_json(ART / "cpu_validation.json", {"passed": True, "scope": "3 CPU updates/9 native S8N1 roots (total72 simulations), engineering only; no GPU use or formal strength result",
        "legacy_initial_checkpoint_sha256": sha256(parent_initial), "variants": results})
    print({"passed": True, "updates": 3, "native_roots": 9, "total_simulations": 72}, flush=True)


if __name__ == "__main__":
    main()
