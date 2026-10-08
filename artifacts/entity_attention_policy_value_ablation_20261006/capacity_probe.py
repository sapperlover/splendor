"""Exploratory CPU sanity probe: can policy memorize 256 fixed searchable positions?"""
import json
import time
from pathlib import Path
import torch

from splendor.alphazero_recording import environment_identity, sha256, write_json
from splendor.offline_train import make_index
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples
from core import diagnostic_step, restored_learner
from train import check_inputs

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
spec=json.loads((ART/"specification.json").read_text())
check_inputs(spec)
torch.set_num_threads(1)
index=make_index({"games":spec["games"]},512*1024**2)
pool=index.position_pool(split="train",round_index=spec["data_round"])
rng=DeterministicRng(14851000)
subset=[]
references=[]
seen=set()
while len(subset)<256:
    sample,ref=pool.sample(rng)
    key=(ref.game_id,ref.decision_index)
    if key not in seen and sample.arrays["policy_weight"][sample.index]>0:
        seen.add(key)
        subset.append(sample)
        references.append(ref.to_dict())
holdout=[index.cache.sample(g,i) for g in index.split_games("validation") for i in range(g.decision_count)]
learner,_=restored_learner(ROOT/spec["initial"]["path"],"cpu")
learner.model.critic.requires_grad_(False)
started=time.perf_counter()
curve=[]
for step in range(1001):
    if step in (0,100,500,1000):
        row={"step":step,"train_subset":learner.evaluate_samples(subset),
             "full_latest_holdout":learner.evaluate_samples(holdout)}
        curve.append(row)
        print(json.dumps({"event":"capacity_probe","step":step,
                          "subset_kl":row["train_subset"]["policy_kl"],
                          "holdout_kl":row["full_latest_holdout"]["policy_kl"]}),flush=True)
    if step==1000:
        break
    batch=collate_selfplay_samples([subset[learner.sampling_rng.randbelow(256)] for _ in range(64)])
    diagnostic_step(learner,batch,"policy_only")
check_inputs(spec)
write_json(ART/"capacity_probe.json",{"protocol":"exploratory-policy-capacity-probe-v1",
    "scope":"Added after main fitting curves; fitting sanity only, not a new strength arm or equal-budget comparison.",
    "initial":spec["initial"],"positions":256,"reference_selection_seed":14851000,
    "references":references,"batch_size":64,"steps":1000,"optimizer":"same inherited AdamW; critic frozen",
    "curve":curve,"elapsed_seconds":time.perf_counter()-started,"inputs_unchanged":True,
    "environment":environment_identity(),"script_sha256":sha256(Path(__file__)),"core_sha256":sha256(ART/"core.py")})
