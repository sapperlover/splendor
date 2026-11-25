"""Strict loading and one real CPU FPU/S128N1 root for each fixed checkpoint."""
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy

ROOT=Path.cwd()
NAME="lambda_online_r3_comparisons_20261125"


def main():
    torch.set_num_threads(1)
    plan=read_json(ROOT / "artifacts" / NAME / "plan.json")
    panel=plan["panels"]["lambda"]
    config=read_json(ROOT / panel["config"]["path"])
    search=MCTSConfig(**config["policy_a"]["search_config"])
    assert search.fpu_reduction==.3 and search.c_puct==1.
    results=[]
    for key,record in plan["models"].items():
        assert sha256(Path(record["checkpoint"]))==record["sha256"]
        model=load_alphazero_checkpoint(record["checkpoint"]).build_model(device="cpu")
        model.eval()
        assert model.config.architecture=="entity_attention_v1" and model.trainable_parameter_count()==583682
        result=NativePUCTSearchPolicy(model,config=search).search(
            GameEngine.new(plan["seeds"]["game_seed_start"],mode="fast").decision())
        assert result.simulations==128
        results.append({"key":key,"checkpoint_sha256":record["sha256"],"strict_load":True,
            "native_simulations":result.simulations,"fpu_reduction":search.fpu_reduction,"c_puct":search.c_puct})
    write_json(ROOT / "runtime" / NAME / "cpu_validation.json",{
        "scope":"Four real-model CPU FPU/S128N1 roots,512simulations,zero complete games; no CUDA/strength claim",
        "models":results,"imported_core":__import__("splendor.evaluation",fromlist=["evaluation"]).__file__,
        "imported_native":__import__("splendor.native_policy",fromlist=["native_policy"]).__file__})
    print({"status":"passed","models":4,"native_simulations":512,"complete_games":0})


if __name__=="__main__":main()
