"""CPU root checks only; do not play complete candidate matches."""
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.engine import GameEngine
from splendor.features import collate_encoded_decisions, encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy

ROOT = Path.cwd()
ART = ROOT / "artifacts/r37_lr2e5_step140221_vs_r37_10k_20261029"


def main():
    torch.set_num_threads(1)
    plan = read_json(ART / "plan.json")
    results = []
    for record in [*plan["models"], plan["opponent"]]:
        assert sha256(Path(record["checkpoint"])) == record["sha256"]
        model = load_alphazero_checkpoint(record["checkpoint"]).build_model(device="cpu")
        model.eval()
        request = GameEngine.new(plan["seeds"]["game_seed_start"], mode="fast").decision()
        batch = collate_encoded_decisions([encode_decision(request)])
        with torch.inference_mode():
            logits, values = model(batch)
        assert torch.isfinite(values).all() and torch.isfinite(logits[batch.candidate_mask]).all()
        result = NativePUCTSearchPolicy(model, config=MCTSConfig(simulations=128, determinizations=1,
            seed=plan["seeds"]["search_seed"], root_noise_fraction=0., forced_playouts=0.,
            policy_target_pruning=False)).search(request)
        assert result.simulations == 128
        results.append({"key": record["key"], "checkpoint_sha256": record["sha256"],
            "strict_load": True, "greedy_root_forward": True, "native_simulations": result.simulations})
    write_json(ROOT / "runtime" / ART.name / "cpu_validation.json", {
        "scope": "Two CPU model/root checks,256 simulations,zero complete games; no strength or CUDA claim",
        "models": results, "imported_evaluation": __import__("splendor.evaluation", fromlist=["evaluation"]).__file__,
        "imported_native_policy": __import__("splendor.native_policy", fromlist=["native_policy"]).__file__})
    print({"status": "passed", "models": 2, "native_simulations": 256, "complete_games": 0})


if __name__ == "__main__":
    main()
