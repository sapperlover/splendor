"""CPU-only checkpoint/greedy/S128N1 root checks; no complete match is played."""
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.engine import GameEngine
from splendor.features import collate_encoded_decisions, encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy

ROOT = Path.cwd()
ART = ROOT / "artifacts/offline5_vs_r37_10k_20261027"


def main():
    torch.set_num_threads(1)
    plan = read_json(ART / "plan.json")
    records = [*plan["models"], {**plan["opponent"], "key": "raw_R37"}]
    results = []
    for record in records:
        assert sha256(Path(record["checkpoint"])) == record["sha256"]
        payload = load_alphazero_checkpoint(record["checkpoint"])
        model = payload.build_model(device="cpu")
        model.eval()
        def reject_auxiliary(*_):
            raise AssertionError("Evaluation executed a training-only auxiliary head")
        hooks = [getattr(model, name).register_forward_pre_hook(reject_auxiliary)
            for name in ("shortterm_value_head", "opponent_policy_head", "score_ownership_head")
            if hasattr(model, name)]
        request = GameEngine.new(plan["seeds"]["game_seed_start"], mode="fast").decision()
        batch = collate_encoded_decisions([encode_decision(request)])
        with torch.inference_mode():
            logits, value = model(batch)
        assert torch.isfinite(value).all() and torch.isfinite(logits[batch.candidate_mask]).all()
        search = NativePUCTSearchPolicy(model, config=MCTSConfig(simulations=128, determinizations=1,
            seed=plan["seeds"]["search_seed"], root_noise_fraction=0., forced_playouts=0.,
            policy_target_pruning=False)).search(request)
        assert search.simulations == 128
        for hook in hooks:
            hook.remove()
        results.append({"key": record["key"], "checkpoint_sha256": record["sha256"],
            "parameters": model.trainable_parameter_count(), "strict_load": True, "greedy_root_forward": True,
            "native_simulations": search.simulations, "training_auxiliary_inference_calls": 0})
    write_json(ART / "cpu_validation.json", {"scope": "Finite CPU root checks only; zero full matches/pairs; no strength or CUDA performance claim",
        "models": results, "total_native_simulations": sum(r["native_simulations"] for r in results),
        "imported_modules": {name: __import__("splendor."+name, fromlist=[name]).__file__
            for name in ("evaluation", "native_policy", "entity_attention", "continuation_auxiliary")}})
    print({"status": "passed", "models": len(results), "total_simulations": 768, "complete_games": 0})


if __name__ == "__main__":
    main()
