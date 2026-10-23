"""Time the real NativePackedTransfer path, including CUDAGraph capture/replay.

This wraps FrozenSearchInference.new_native_transfer and records one CUDA event
pair per begin/finish, so the elapsed interval includes H2D, the production
native CUDAGraph replay (search_inference.NativePackedTransfer._direct_graph),
D2H and the final submit.  Scope: one native search job through NativeBatchRunner,
not a full collection round.
"""
from __future__ import annotations
import argparse, importlib.util, json, statistics
from pathlib import Path
import torch
from splendor.catalog import load_default_catalog
from splendor.features import encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig
spec=importlib.util.spec_from_file_location('bench_large','scripts/benchmark_large_batch_inference.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class TimedHandle:
    def __init__(self, inner, start, owner, rows): self.inner,self.start,self.owner,self.rows=inner,start,owner,rows
    def done(self): return self.inner.done()
    def discard(self): return self.inner.discard()
    def finish(self):
        self.inner.finish()
        end=torch.cuda.Event(enable_timing=True); end.record(); end.synchronize()
        self.owner.times.append({'rows':self.rows,'ms':self.start.elapsed_time(end)})
class TimedTransfer:
    def __init__(self, inner): self.inner=inner; self.times=[]
    def _begin(self, fn, sink):
        start=torch.cuda.Event(enable_timing=True); start.record()
        return TimedHandle(fn(sink), start, self, sink.row_count())
    def begin_policy_value(self, sink): return self._begin(self.inner.begin_policy_value, sink)
    def begin_value_only(self, sink): return self._begin(self.inner.begin_value_only, sink)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--simulations',type=int,default=1024)
    args=ap.parse_args(); assert torch.cuda.is_available(); torch.set_num_threads(1)
    catalog=load_default_catalog(); reqs=mod.collect_requests(500,seed=44000000)
    model=load_frozen_actor_critic_checkpoint(args.checkpoint,device='cuda')
    cfg=InferenceConfig(mode='compiled',actor_projection=False,candidate_buckets=(16,32,48,64),compile_mode='native-cudagraph',compile_failure_policy='fail_fast',native_transport='reused',batch_buckets=(16,32,48,64,128,192,256,384,500))
    ev=FrozenSearchInference(model,device='cuda',config=cfg)
    orig=ev.new_native_transfer; transfers=[]
    def new_native_transfer(**kwargs):
        inner=orig(**kwargs); t=TimedTransfer(inner); transfers.append(t); return t
    ev.new_native_transfer=new_native_transfer
    policy=NativePUCTSearchPolicy(model,config=MCTSConfig(simulations=args.simulations,determinizations=1,seed=44000000,root_noise_fraction=0.0),device='cuda',catalog=catalog,evaluator=ev)
    jobs=[policy.start_job(req) for req in reqs]
    runner=NativeBatchRunner(ev,catalog,capacity=500,buckets=(16,32,48,64),slots=1)
    rows=0
    while jobs:
        rows+=runner.run(jobs)
        jobs=[job for job in jobs if not job.done]
    times=[t for tr in transfers for t in tr.times]
    from collections import Counter
    hist=Counter(t['rows'] for t in times)
    full=[t['ms'] for t in times if t['rows']==500]
    vals=[t['ms'] for t in times]
    out={'checkpoint':str(Path(args.checkpoint).resolve()),'simulations':args.simulations,'rows':rows,
         'batches_timed':len(times),'rows_histogram':dict(sorted(hist.items())),'batches_500':len(full),
         'native_graph_captures':ev.stats.native_graph_captures,'native_graph_replays':ev.stats.native_graph_replays,
         'scope':'real NativePackedTransfer begin/finish: H2D + production CUDAGraph replay + D2H; one native search job',
         'median_ms_all':statistics.median(vals) if vals else None,'min_ms_all':min(vals) if vals else None,'max_ms_all':max(vals) if vals else None,
         'median_ms_b500':statistics.median(full) if full else None,'min_ms_b500':min(full) if full else None,'max_ms_b500':max(full) if full else None,
         'all':times}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('simulations','rows','batches_timed','rows_histogram','batches_500','native_graph_captures','native_graph_replays','median_ms_all','median_ms_b500','min_ms_b500','max_ms_b500')},indent=2))

if __name__=='__main__': main()
