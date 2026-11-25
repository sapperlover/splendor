"""CPU roots through real configured FPU/historical policies; zero full matches."""
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, write_json
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy

ROOT=Path.cwd()
NAME="lambda161924_vs_l12_fpu2000_20261123"


def main():
    torch.set_num_threads(1)
    plan=read_json(ROOT / "artifacts" / NAME / "plan.json")
    model_cache={};results=[]
    for key,panel in plan["panels"].items():
        config=read_json(ROOT / panel["config"]["path"])
        for side in ("policy_a","policy_b"):
            spec=config[side]
            if spec["checkpoint"] not in model_cache:
                model=load_alphazero_checkpoint(spec["checkpoint"]).build_model(device="cpu")
                model.eval();model_cache[spec["checkpoint"]]=model
            model=model_cache[spec["checkpoint"]]
            def reject_auxiliary(*_):raise AssertionError("Search called a training-only auxiliary head")
            hooks=[getattr(model,name).register_forward_pre_hook(reject_auxiliary) for name in
                   ("shortterm_value_head","opponent_policy_head","score_ownership_head") if hasattr(model,name)]
            search=MCTSConfig(**spec["search_config"])
            result=NativePUCTSearchPolicy(model,config=search).search(
                GameEngine.new(plan["seeds"]["game_seed_start"],mode="fast").decision())
            assert result.simulations==128
            for hook in hooks:hook.remove()
            results.append({"variant":key,"side":side,"simulations":result.simulations,
                "fpu_reduction":search.fpu_reduction,"c_puct":search.c_puct,"auxiliary_inference_calls":0})
    assert len(model_cache)==2
    write_json(ROOT / "runtime" / NAME / "cpu_validation.json",{
        "scope":"Two real models,fourCPU S128N1 roots,512simulations,zero full matches; no CUDA/strength claim",
        "roots":results,"strict_model_loads":2,
        "imported_native":__import__("splendor.native_policy",fromlist=["native_policy"]).__file__,
        "imported_mcts":__import__("splendor.mcts_policy",fromlist=["mcts_policy"]).__file__})
    print({"status":"passed","roots":4,"native_simulations":512,"complete_games":0})


if __name__=="__main__":main()
