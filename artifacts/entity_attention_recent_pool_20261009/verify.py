"""Finite real-data checks: streaming metrics and exact interrupted resume."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from pool import HERE
from train import evaluate, load_records
from compact_cache import CompactGameCache
from splendor.actor_critic import build_actor_critic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import sha256, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.offline_train import evaluate_validation, make_index
from splendor.rng import DeterministicRng
from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples, load_game


def verify(manifest, output):
    output.mkdir(parents=True, exist_ok=False)
    records = load_records(manifest)
    chosen = []
    for split in ("train", "validation"):
        chosen.extend([r for r in records if r["split"] == split][:2])
    subset = output / "pool_games.jsonl"
    subset.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in chosen))
    index = make_index({"games": chosen}, 2*1024**2)
    for game in index.games:
        arrays,metadata=load_game(game.path)
        raw=collate_selfplay_samples([SelfPlaySample(arrays,metadata,i) for i in range(game.decision_count)])
        encoded=CompactGameCache(2*1024**2)
        decoded=collate_selfplay_samples([encoded.sample(game,i) for i in range(game.decision_count)])
        for field in raw.decisions.__dataclass_fields__:
            a,b=getattr(raw.decisions,field),getattr(decoded.decisions,field)
            assert np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b
        for field in ("policy_targets","behavior_policies","value_targets","policy_weights","value_weights"):
            assert np.array_equal(getattr(raw,field),getattr(decoded,field))
    torch.set_num_threads(1)
    torch.manual_seed(991)
    cfg = LearnerConfig(device="cpu", batch_size=4, history_batch_size=2, validation_batch_size=32, checkpoint_every_steps=2)
    learner = AlphazeroLearner(build_actor_critic({"architecture":"entity_attention_v1"}), cfg, sampling_rng=DeterministicRng(992))
    streaming = evaluate(learner, index, chosen)
    reference = evaluate_validation(learner, index)
    measured = streaming["pool_weighted"]["validation"]
    assert abs(measured["policy_kl"]-reference["policy_kl"]) < 2e-6
    assert abs(measured["value_mse"]-reference["value_loss"]) < 2e-6
    totals = streaming["metrics"]["__all_observed__/validation/all"]
    early = streaming["metrics"]["__all_observed__/validation/early_mid"]
    late = streaming["metrics"]["__all_observed__/validation/late"]
    assert early["positions"] + late["positions"] == totals["positions"]
    positions = sum(r["decision_count"] for r in chosen if r["split"]=="train")
    base = {"manifest":str(subset.resolve()), "manifest_sha256":sha256(subset),
        "initialization_seed":991,"sampling_seed":992,"validation_seed":993,
        "model_config":{"architecture":"entity_attention_v1"},"positions_per_source_split":8,
        "draw_exposures":16/positions,"full_holdout_exposures":[],"cache_mib":2,
        "lr_drop_exposure":100,"late_lr_multiplier":.25,"learner":{**cfg.__dict__} if hasattr(cfg,'__dict__') else None}
    from dataclasses import asdict
    base["learner"] = asdict(cfg)
    for name in ("continuous","resumed"):
        config = {**base,"output":str((output/name).resolve()),"artifacts":str((output/f"{name}_metrics").resolve())}
        path = output / f"{name}.json"
        write_json(path,config)
        command = [sys.executable,str(HERE/"train.py"),"--config",str(path)]
        if name=="resumed":
            subprocess.run(command+["--max-steps-this-session","2"],check=True)
            # Mimic a crash after metrics/ledger writes but before checkpoint 3.
            ledger = Path(config["output"])/"references.i64"
            with ledger.open("ab") as handle:
                handle.write(bytes(32*cfg.batch_size))
            with (Path(config["output"])/"training_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"step":3,"uncommitted_probe":True})+"\n")
        subprocess.run(command,check=True)
    checkpoints = [load_alphazero_checkpoint(output/n/"checkpoints/final.pt") for n in ("continuous","resumed")]
    a,b = checkpoints
    assert a.round_train_step==b.round_train_step==4
    assert a.model_sha256==b.model_sha256
    assert a.model_sha256=="af27aecdc8193649cd67655fd3ffed31cb80bafc01360b7089287c1f058c553b", "compact cache differs from the previously verified raw-cache run"
    assert a.sampling_rng_state==b.sampling_rng_state
    assert torch.equal(a.torch_cpu_rng_state,b.torch_cpu_rng_state)
    for key in a.optimizer_state_dict["state"]:
        for field, value in a.optimizer_state_dict["state"][key].items():
            other=b.optimizer_state_dict["state"][key][field]
            assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
    assert (output/"continuous/references.i64").read_bytes()==(output/"resumed/references.i64").read_bytes()
    assert read_json(output/"continuous_metrics/usage.json")==read_json(output/"resumed_metrics/usage.json")
    result={"passed":True,"device":"cpu","real_games":len(chosen),"steps":4,
        "checks":["every collated input/label equals the original float32 data", "4-step model equals prior raw-cache model bitwise", "streaming full validation agrees with production metrics", "score-10 stages partition positions", "4 steps equals 2+2 resume bitwise for model/AdamW/RNG/references/usage", "uncommitted metric and ledger tail discarded"],
        "streaming_policy_kl":measured["policy_kl"],"production_policy_kl":reference["policy_kl"],
        "streaming_value_mse":measured["value_mse"],"production_value_mse":reference["value_loss"],
        "model_sha256":a.model_sha256,"manifest_sha256":sha256(subset)}
    write_json(output/"verification.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    from splendor.alphazero_recording import read_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=HERE/"pool_games.jsonl")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    verify(args.manifest,args.output)
