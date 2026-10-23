"""Time production CUDAGraph replay and total NativePackedTransfer path.

Patches NativePackedTransfer._direct_graph in-process to record CUDA events
around the actual `graph.replay()` (the production native-cudagraph path at
search_inference.py:589), and records total begin/finish (H2D + replay + D2H).
Uses a real NativeBatchRunner with many concurrent jobs to obtain B500 batches.
"""
from __future__ import annotations
import argparse, importlib.util, json, statistics
from pathlib import Path
from collections import Counter
import torch
from splendor.catalog import load_default_catalog
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig, NativePackedTransfer
spec=importlib.util.spec_from_file_location('bench_large','scripts/benchmark_large_batch_inference.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

GRAPH=[]; TOTAL=[]
orig_direct=NativePackedTransfer._direct_graph
def timed_direct(self, prepared, rows):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); ok=orig_direct(self,prepared,rows); e.record(); e.synchronize()
    GRAPH.append({'rows':rows,'width':self.width,'ms':s.elapsed_time(e),'capture':ok and False})
    return ok
NativePackedTransfer._direct_graph=timed_direct

class TimedHandle:
    def __init__(self, inner,start,owner,rows,width): self.inner,self.start,self.owner,self.rows,self.width=inner,start,owner,rows,width
    def done(self): return self.inner.done()
    def discard(self): return self.inner.discard()
    def finish(self):
        self.inner.finish(); e=torch.cuda.Event(enable_timing=True); e.record(); e.synchronize()
        self.owner.times.append({'rows':self.rows,'width':self.width,'ms':self.start.elapsed_time(e)})
class TimedTransfer:
    def __init__(self,inner): self.inner=inner; self.times=[]
    def _begin(self,fn,sink):
        s=torch.cuda.Event(enable_timing=True); s.record()
        return TimedHandle(fn(sink),s,self,sink.row_count(),getattr(self.inner,'width',0))
    def begin_policy_value(self,sink): return self._begin(self.inner.begin_policy_value,sink)
    def begin_value_only(self,sink): return self._begin(self.inner.begin_value_only,sink)

def med(vals):
    return statistics.median(vals) if vals else None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--width',type=int,required=True); ap.add_argument('--jobs',type=int,default=1000); ap.add_argument('--simulations',type=int,default=8)
    args=ap.parse_args(); assert torch.cuda.is_available(); torch.set_num_threads(1)
    catalog=load_default_catalog()
    reqs=[r for r in mod.collect_requests(max(args.jobs,512),seed=44000000) if len(r.legal_actions)<=args.width]
    if not reqs: raise SystemExit('no requests at width')
    reqs=(reqs*((args.jobs+len(reqs)-1)//len(reqs)))[:args.jobs]
    model=load_frozen_actor_critic_checkpoint(args.checkpoint,device='cuda')
    cfg=InferenceConfig(mode='compiled',actor_projection=False,candidate_buckets=(16,32,48,64),compile_mode='native-cudagraph',compile_failure_policy='fail_fast',native_transport='reused',batch_buckets=(16,32,48,64,128,192,256,384,500))
    ev=FrozenSearchInference(model,device='cuda',config=cfg)
    orig=ev.new_native_transfer; transfers=[]
    def new_transfer(**kw):
        inner=orig(**kw); t=TimedTransfer(inner); transfers.append(t); return t
    ev.new_native_transfer=new_transfer
    policy=NativePUCTSearchPolicy(model,config=MCTSConfig(simulations=args.simulations,determinizations=1,seed=44000000,root_noise_fraction=0.0),device='cuda',catalog=catalog,evaluator=ev)
    jobs=[policy.start_job(r) for r in reqs]; runner=NativeBatchRunner(ev,catalog,capacity=500,buckets=(16,32,48,64),slots=1)
    rows=0
    while jobs:
        rows+=runner.run(jobs); jobs=[j for j in jobs if not j.done]
    totals=[x for tr in transfers for x in tr.times]
    graph=[x for x in GRAPH if x['rows']==500]
    total500=[x for x in totals if x['rows']==500]
    # exclude obvious first-capture outliers (>100 ms) for medians
    graph_steady=[x['ms'] for x in graph if x['ms']<100]
    total_steady=[x['ms'] for x in total500 if x['ms']<100]
    out={'checkpoint':str(Path(args.checkpoint).resolve()),'requested_width':args.width,'jobs':args.jobs,'simulations':args.simulations,
         'rows':rows,'native_graph_captures':ev.stats.native_graph_captures,'native_graph_replays':ev.stats.native_graph_replays,
         'b500_graph_count':len(graph_steady),'b500_total_count':len(total_steady),
         'graph_median_ms':med(graph_steady),'graph_min_ms':min(graph_steady) if graph_steady else None,'graph_max_ms':max(graph_steady) if graph_steady else None,
         'total_median_ms':med(total_steady),'total_min_ms':min(total_steady) if total_steady else None,'total_max_ms':max(total_steady) if total_steady else None,
         'rows_hist':dict(sorted(Counter(x['rows'] for x in totals).items())),
         'width_hist_graph':dict(sorted(Counter(x['width'] for x in GRAPH).items())),
         'width_hist_total':dict(sorted(Counter(x['width'] for x in totals).items())),
         'scope':'production NativePackedTransfer: graph_median is event around _direct_graph graph.replay(); total is begin/finish incl H2D+D2H; B500 subset'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('requested_width','b500_graph_count','b500_total_count','graph_median_ms','total_median_ms','native_graph_captures','native_graph_replays')},indent=2))
if __name__=='__main__': main()
