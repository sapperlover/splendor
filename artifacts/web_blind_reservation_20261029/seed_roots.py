"""Same first root, same S1024N4; vary only the hidden-world base seed."""
import sys,time
from collections import Counter
from dataclasses import replace
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
import analyze as a
from splendor.alphazero_recording import read_json,write_json,sha256
from splendor.catalog import load_default_catalog
from splendor.native_policy import NativeBatchRunner,NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state
from splendor.search_inference import FrozenSearchInference
from splendor.search_policy import _request_seed,determinize_request

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
a.identity_check()
catalog=load_default_catalog()
root=read_json(a.RAW/'roots.json')[0]
request=a.fast(_decode_state(root['state']),catalog).decision()
model=load_frozen_actor_critic_checkpoint(a.MODEL,device='cuda')
evaluator=FrozenSearchInference(model,device='cuda')
runner=NativeBatchRunner(evaluator,catalog,capacity=128,buckets=(16,32,48,64),slots=1)
policies=[NativePUCTSearchPolicy(model,config=replace(a.CFG,seed=seed),catalog=catalog,evaluator=evaluator) for seed in range(32)]
started=time.perf_counter()
jobs=[p.start_job(request,add_root_noise=False,temperature=0.) for p in policies]
while any(not j.done for j in jobs):
 assert runner.run([j for j in jobs if not j.done])>0
rows=[]
for seed,job in enumerate(jobs):
 row=a.root_record(request,job.result)
 tops=[determinize_request(request,seed=_request_seed(request,seed,j),catalog=catalog,engine_mode='fast')._debug_state().decks.tier_three[-1] for j in range(4)]
 row.update(seed=seed,world_tops=tops,white_worlds=sum(x in a.WHITE for x in tops))
 rows.append(row)
report=dict(scope='One root, no root noise; seed sensitivity only, not playing strength.',device='cuda',precision='float32',tf32=False,search='S1024N4 (4096 total)',rows=rows,seconds=time.perf_counter()-started,script_sha256=sha256(Path(__file__)))
raw_path=a.RAW/'seed_roots_raw.json'
write_json(raw_path,report)
compact={**report,'raw_source':str(raw_path.relative_to(a.ROOT)),'raw_sha256':sha256(raw_path)}
compact['rows']=[{k:v for k,v in row.items() if k in ('seed','world_tops','white_worlds','chosen','top')} for row in rows]
write_json(a.ART/'seed_roots.json',compact)
print('complete',report['seconds'],dict(Counter(str(r['chosen']) for r in rows)),flush=True)
