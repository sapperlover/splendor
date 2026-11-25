"""Strict scalar checkpoint loading and one CPU S128N1 root per model; no full games."""
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.engine import GameEngine
from splendor.features import collate_encoded_decisions, encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy

ROOT = Path.cwd()
NAME = "lambda161924_search3000_20261123"


def main():
    torch.set_num_threads(1)
    plan = read_json(ROOT / "artifacts" / NAME / "plan.json")
    results = []
    for key, record in plan["models"].items():
        assert sha256(Path(record["checkpoint"])) == record["sha256"]
        model = load_alphazero_checkpoint(record["checkpoint"]).build_model(device="cpu")
        model.eval()
        assert model.config.architecture == "entity_attention_v1" and model.trainable_parameter_count() == 583682
        request = GameEngine.new(plan["seeds"]["game_seed_start"], mode="fast").decision()
        batch = collate_encoded_decisions([encode_decision(request)])
        with torch.inference_mode():
            logits, value = model(batch)
        assert value.numel() == 1 and torch.isfinite(value).all() and torch.isfinite(logits[batch.candidate_mask]).all()
        result = NativePUCTSearchPolicy(model, config=MCTSConfig(simulations=128, determinizations=1,
            seed=plan["seeds"]["search_seed"], root_noise_fraction=0., forced_playouts=0.,
            policy_target_pruning=False)).search(request)
        assert result.simulations == 128
        results.append({"model": key, "checkpoint_sha256": record["sha256"], "strict_load": True,
                        "scalar_value": float(value.item()), "native_simulations": result.simulations})
    write_json(ROOT / "runtime" / NAME / "cpu_validation.json", {"scope": "Three CPU roots,384 simulations,zero complete games; no CUDA/strength claim",
        "models": results, "imported_core": __import__("splendor.evaluation", fromlist=["evaluation"]).__file__,
        "imported_actor": __import__("splendor.actor_critic", fromlist=["actor_critic"]).__file__})
    print({"status": "passed", "models": 3, "native_simulations": 384, "complete_games": 0})


if __name__ == "__main__":
    main()
