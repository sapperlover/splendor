"""Time only the compiled CUDA-graph dispatch, excluding Python encoding/H2D."""
from __future__ import annotations
import argparse, importlib.util, json, statistics
from pathlib import Path
import torch
from splendor.features import encode_decision, collate_encoded_decisions
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig, _POLICY_FIELDS
spec=importlib.util.spec_from_file_location('bench_large','scripts/benchmark_large_batch_inference.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--batch',type=int,default=500); ap.add_argument('--repeats',type=int,default=20); ap.add_argument('--calls',type=int,default=1)
args=ap.parse_args(); assert torch.cuda.is_available(); torch.set_num_threads(1)
req=mod.collect_requests(args.batch,seed=44000000); enc=[encode_decision(r) for r in req]
model=load_frozen_actor_critic_checkpoint(args.checkpoint,device='cuda')
cfg=InferenceConfig(mode='compiled',actor_projection=False,candidate_buckets=(16,32,48,64),compile_mode='native-cudagraph',compile_failure_policy='fail_fast',native_transport='reused',batch_buckets=(16,32,48,64,128,192,256,384,500))
ev=FrozenSearchInference(model,device='cuda',config=cfg)
batch=ev._tensor_batch(collate_encoded_decisions(enc), state_only=False)
width=ev.compiled.width_for(batch.candidate_features.shape[1]); assert width is not None
prepared=ev.compiled._pad_batch(ev.compiled._pad_width(batch,width))
name=ev.compiled._graph_name(f'policy_value:{width}',prepared.global_features.shape[0])
tensors=ev.compiled._tensors(prepared,_POLICY_FIELDS)
# warm up graph construction and capture
for _ in range(3):
    ev.compiled._dispatch(name,width,ev.compiled._policy_value_impl,tensors)
torch.cuda.synchronize()
times=[]
for _ in range(args.repeats):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.calls):
        ev.compiled._dispatch(name,width,ev.compiled._policy_value_impl,tensors)
    e.record(); e.synchronize()
    times.append(s.elapsed_time(e)/args.calls)
out={'checkpoint':str(Path(args.checkpoint).resolve()),'batch':args.batch,'width':width,'graph':name,'repeats':args.repeats,'calls_per_event':args.calls,
     'scope':'CUDA event elapsed around compiled graph dispatch only; excludes Python encode, collate, H2D and owner gather/dispatch',
     'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times),'median_us_per_row':statistics.median(times)*1000/args.batch,'all_ms':times}
Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('batch','width','graph','median_ms','median_us_per_row','min_ms','max_ms')},indent=2))
