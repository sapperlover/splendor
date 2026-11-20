"""Measure one live root at a time; full matrices remain under runtime/."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from splendor.kirameki_live import DEFAULT_CHECKPOINT, request_from_snapshot
from splendor.compact_trace import decode_request
from splendor.parallel_recommend import BatchedWorldSearch
from splendor.recommend import LocalAdvisor
from splendor.search_inference import InferenceConfig

ROOT=Path('runtime/single_tree_batch_20261031')
MATRIX=Path('runtime/kirameki_budget_matrix_20261030')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--inference',choices=('eager','compiled'),default='eager')
    parser.add_argument('--compile-mode',default='default')
    parser.add_argument('--budget',type=int,default=8192)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--cases',nargs='+',default=['g03_r002_main','g04_r013_main','g05_r024_main'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source_root=Path(__import__('splendor').__file__).parent
    source_paths=[source_root/name for name in ('native_search.cpp','native_policy.py',
        'parallel_recommend.py','search_inference.py','recommend.py','kirameki_live.py')]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=InferenceConfig(mode=args.inference,candidate_buckets=(16,32,48,64),
        compile_mode=args.compile_mode,compile_failure_policy='fail_fast',
        batch_buckets=tuple(sorted({1,args.batch})) if args.inference=='compiled' else ())
    advisor=LocalAdvisor(DEFAULT_CHECKPOINT,device=args.device,inference_config=config)
    search=BatchedWorldSearch(advisor,simulations_per_world=args.budget,worlds=1,
        seed=0,search_backend='native_chance',leaf_batch_size=args.batch)
    warmup=search.runner.warmup()
    payload=dict(config=vars(args)|{'output':str(args.output)},warmup_seconds=warmup,
        checkpoint_sha256=advisor.checkpoint_sha256,torch=torch.__version__,
        device=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
        package=__import__('splendor').__file__,results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save()
    for case in args.cases:
        spec=json.loads((MATRIX/'results'/f'{case}.json').read_text())['spec']
        path=MATRIX/spec['input_directory']/'snapshot.json'
        if spec['kind']=='main':
            request=request_from_snapshot(json.loads(path.read_text()),advisor.catalog)
        else:
            original=json.loads((MATRIX/spec['input_directory']/'recommendation.json').read_text())
            request=decode_request(original['followup']['request'])
        for repeat in range(args.repeats):
            before=advisor.evaluator.stats.to_dict()
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            started=time.perf_counter()
            result=search.search(request)
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            seconds=time.perf_counter()-started
            after=advisor.evaluator.stats.to_dict()
            chance=search.last_search_diagnostics['chance']
            assert result.simulations==sum(result.root_visit_counts)==args.budget
            assert chance['root_hidden_samples']==args.budget
            assert chance['pending_simulations']==chance['virtual_visits']==0
            record=dict(case=case,repeat=repeat,seconds=seconds,
                simulations_per_second=args.budget/seconds,result=asdict(result),
                stats=search.last_search_diagnostics['stats'],chance=chance,
                inference_delta={k:after[k]-before[k] for k in ('physical_model_batches',
                    'logical_model_positions','feature_seconds','forward_seconds','h2d_seconds',
                    'd2h_seconds','wait_seconds')},inference=after)
            payload['results'].append(record);save()
            print(json.dumps(dict(case=case,repeat=repeat,seconds=seconds,
                choice=result.chosen_candidate_index,share=max(result.policy_target),
                collisions=chance['leaf_collisions'],stats=record['inference_delta'])),flush=True)
    assert source_hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    payload['source_sha256']=source_hashes
    payload['binary_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_root.glob('_native_search*.so')}
    payload['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
    payload['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    save()


if __name__=='__main__':main()
