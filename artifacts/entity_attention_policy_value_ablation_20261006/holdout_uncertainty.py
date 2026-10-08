"""CPU per-game paired bootstrap for the fixed25-game holdout."""
import json
from pathlib import Path
import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.alphazero_recording import sha256, write_json
from splendor.offline_train import make_index
from splendor.rng import DeterministicRng
from train import check_inputs

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
spec=json.loads((ART/"specification.json").read_text())
check_inputs(spec)
torch.set_num_threads(1)
index=make_index({"games":spec["games"]},512*1024**2)
paths={"initial":ROOT/spec["initial"]["path"]}
for arm in spec["arms"]:
    for step in (100,300,500):
        paths[f"{arm}_{step}"]=ROOT/spec["output"]/arm/f"step-{step:06d}.pt"
by_game={}
for name,path in paths.items():
    payload=load_alphazero_checkpoint(path)
    cfg=LearnerConfig(**(spec["learner_config"]|{"device":"cpu"}))
    learner=AlphazeroLearner(payload.build_model(device="cpu"),cfg,sampling_rng=DeterministicRng(0))
    rows=[]
    for game in index.split_games("validation"):
        metrics=learner.evaluate_samples([index.cache.sample(game,i) for i in range(game.decision_count)])
        rows.append({"game_id":game.game_id,"policy_kl_sum":metrics["policy_ce_sum"]-metrics["target_entropy_sum"],
                     "policy_weight_sum":metrics["policy_weight_sum"],
                     "value_sq_sum":metrics["value_sq_sum"],"value_weight_sum":metrics["value_weight_sum"]})
    by_game[name]=rows
    print(json.dumps({"event":"holdout_game_metrics","candidate":name}),flush=True)
    del learner,payload
rng=np.random.default_rng(14852000)
draws=rng.integers(0,25,size=(10000,25))
def compare(left,right):
    a,b=by_game[left],by_game[right]
    assert [r["game_id"] for r in a]==[r["game_id"] for r in b]
    result={}
    for label,num,den in (("policy_kl","policy_kl_sum","policy_weight_sum"),
                           ("value_mse","value_sq_sum","value_weight_sum")):
        change=np.array([x[num]-y[num] for x,y in zip(a,b)])
        weight=np.array([r[den] for r in a])
        assert np.array_equal(weight,[r[den] for r in b])
        boot=change[draws].sum(axis=1)/weight[draws].sum(axis=1)
        result[label]={"estimate":float(change.sum()/weight.sum()),
                       "lower":float(np.quantile(boot,.025)),"upper":float(np.quantile(boot,.975)),
                       "confidence":.95,"unit":"whole heldout game","games":25,"resamples":10000,"seed":14852000}
    return result
differences={name+"_minus_initial":compare(name,"initial") for name in paths if name!="initial"}
for arm in ("policy_only","value_detached"):
    for step in (100,300,500):
        differences[f"{arm}_{step}_minus_joint"]=compare(f"{arm}_{step}",f"joint_{step}")
check_inputs(spec)
write_json(ART/"holdout_uncertainty.json",{"protocol":"fixed-holdout-game-bootstrap-v1","per_game":by_game,
          "differences":differences,"checkpoint_hashes":{str(p.relative_to(ROOT)):sha256(p) for p in paths.values()},
          "execution":"CPU FP32; same frozen whole-game split; point metrics may differ from GPU by rounding.",
          "script_sha256":sha256(Path(__file__)),"inputs_unchanged":True})
