"""Single-position, same-total-budget diagnostics; not a playing-strength test."""
from dataclasses import asdict,replace
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.native_policy import NativeBatchRunner,NativePUCTSearchPolicy,NativeChancePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state
from splendor.search_inference import FrozenSearchInference

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name
a.identity_check();torch.set_num_threads(1)
catalog=load_default_catalog()
model=load_frozen_actor_critic_checkpoint(a.MODEL,device='cpu')
evaluator=FrozenSearchInference(model,device='cpu')
runner=NativeBatchRunner(evaluator,catalog,capacity=128,buckets=(16,32,48,64),slots=1)
state=_decode_state(next(r['state'] for r in read_json(RAW/'states.json') if r['event']==69))
request=a.fast(state,catalog).decision()
cases=[]
for label,cls,worlds in [('S1024N4',NativePUCTSearchPolicy,4),
                          ('S256N16',NativePUCTSearchPolicy,16),
                          ('S128N32',NativePUCTSearchPolicy,32),
                          ('chance4096_shared',NativeChancePUCTSearchPolicy,1)]:
    cfg=replace(a.CFG,determinizations=worlds)
    policy=cls(model,config=cfg,catalog=catalog,evaluator=evaluator)
    cases.append(dict(label=label,policy=policy,config=cfg,
                       job=policy.start_job(request,add_root_noise=False,temperature=0.)))
started=time.perf_counter()
while any(not r['job'].done for r in cases):
    assert runner.run([r['job'] for r in cases if not r['job'].done])>0
rows=[]
for row in cases:
    result=row['job'].result
    record=a.root_record(request,result)
    write_json(RAW/f"variant_{row['label']}.json",record)
    rows.append(dict(label=row['label'],config=asdict(row['config']),chosen=record['chosen'],top=record['top'],
                     note='N=1 is one shared statistics pool, fresh chance draws during simulations' if row['label'].startswith('chance') else 'independent fixed worlds'))
    print(rows[-1],flush=True)
write_json(ART/'search_variants.json',dict(scope='Exactly one root, seed0, no noise. Total4096 simulations each. '
           'Not a strength result or a recommended replacement preset.',rows=rows,seconds=time.perf_counter()-started,
           script_sha256=sha256(Path(__file__)),checkpoint_sha256=sha256(a.MODEL),device='cpu',torch_threads=1))
