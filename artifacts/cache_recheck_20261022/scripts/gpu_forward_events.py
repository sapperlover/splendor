"""GPU-event timing of the production-shaped compiled policy/value forward.

This measures actual GPU elapsed time for one B500 compiled forward, not host
launch time. It uses the same model, candidate buckets, batch buckets and
native-cudagraph compile mode as collection, but not the C++ shared-memory
transport. Scope: model forward only.
"""
from __future__ import annotations
import argparse, importlib.util, json, statistics, time
from pathlib import Path
import torch
from splendor.features import encode_decision
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig

spec=importlib.util.spec_from_file_location('bench_large','scripts/benchmark_large_batch_inference.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--batch',type=int,default=500); ap.add_argument('--repeats',type=int,default=20)
    args=ap.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    requests=mod.collect_requests(args.batch,seed=44000000)
    batch=[encode_decision(r) for r in requests]
    model=load_frozen_actor_critic_checkpoint(args.checkpoint,device='cuda')
    cfg=InferenceConfig(mode='compiled',actor_projection=False,candidate_buckets=(16,32,48,64),
        compile_mode='native-cudagraph',compile_failure_policy='fail_fast',native_transport='reused',
        batch_buckets=(16,32,48,64,128,192,256,384,500))
    ev=FrozenSearchInference(model,device='cuda',config=cfg)
    for _ in range(3):
        ev.policy_value_encoded(batch)
    torch.cuda.synchronize()
    times=[]
    for _ in range(args.repeats):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); ev.policy_value_encoded(batch); end.record(); end.synchronize()
        times.append(start.elapsed_time(end))
    rows=len(batch)
    out={'checkpoint':str(Path(args.checkpoint).resolve()),'batch':rows,'repeats':args.repeats,
         'scope':'GPU event elapsed for compiled model forward only; excludes H2D/gather/dispatch and native C++ transport',
         'compiled':ev.stats.to_dict().get('compile',{}).get('effective_mode'),
         'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times),
         'median_us_per_row':statistics.median(times)*1000/rows,'all_ms':times}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('checkpoint','batch','median_ms','median_us_per_row','min_ms','max_ms')},indent=2))

if __name__=='__main__': main()
