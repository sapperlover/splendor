"""Freeze noise-free R43 S1024 root rankings and same-world child values."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.search_policy import determinize_request, _request_seed

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'/ART.name
BANK=ROOT/'artifacts/entity_attention_r43_value_signal_64_20261014/root_bank.json'
PREVIOUS=ROOT/'artifacts/entity_attention_r43_value_signal_64_20261014/plan.json'
SEARCH_SEED=38100000


def child(engine, index):
    """Apply root candidate then fold forced decisions, preserving action indices."""
    response=engine.apply_candidate(index,state_version=engine.state_version)
    forced=[]
    while isinstance(response,DecisionRequest) and len(response.legal_actions)==1:
        forced.append(0)
        response=engine.apply_candidate(0,state_version=engine.state_version)
        if len(forced)>=512:
            raise RuntimeError('Forced-child transition cap reached')
    return response,forced


def root_value(value,actor):
    return float(np.clip(value,-1,1))*(1 if actor==0 else -1)


def payoff(terminal):
    return 0 if len(terminal.winner_ids)==2 else 1 if 0 in terminal.winner_ids else -1


def main():
    if (ART/'search_roots.json').exists():
        raise ValueError('Search roots already frozen; use the saved search evidence')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    RUNTIME.mkdir(parents=True,exist_ok=True)
    previous=read_json(PREVIOUS);checkpoint=ROOT/previous['checkpoint']['path']
    if sha256(checkpoint)!=previous['checkpoint']['sha256']:raise ValueError('R43 identity changed')
    roots=read_json(BANK)['roots'];requests=[decode_request(r['request']) for r in roots]
    source=source_identity(ROOT,RUNTIME/'search_source')
    source['producer_sha256']=sha256(Path(__file__))
    config={**previous['search'],'seed':SEARCH_SEED,'root_noise_fraction':0.}
    model=load_frozen_actor_critic_checkpoint(checkpoint,device='cuda')
    inference=InferenceConfig(mode='compiled',actor_projection=False,candidate_buckets=(16,32,48,64),compile_failure_policy='fail_fast')
    evaluator=FrozenSearchInference(model,device='cuda',config=inference)
    policy=NativePUCTSearchPolicy(model,evaluator=evaluator,config=MCTSConfig(**config))
    runner=NativeBatchRunner(evaluator,policy.catalog,capacity=64,buckets=(16,32,48,64))
    started=time.perf_counter()
    jobs=[policy.start_job(r,add_root_noise=False,temperature=0.) for r in requests]
    while any(not j.done for j in jobs):
        if runner.run([j for j in jobs if not j.done])<=0:raise RuntimeError('Root search stalled')
    rows=[];child_requests=[];references=[]
    for root,request,job in zip(roots,requests,jobs,strict=True):
        result=job.result;diag=job.core.chance_diagnostics()
        visits=list(result.root_visit_counts);sums=list(diag['root_value_sums'])
        if sum(visits)!=1024 or len(visits)!=len(sums) or tuple(result.root_network_prior)!=tuple(result.root_used_prior):
            raise ValueError('Root search is not complete noise-free S1024/N1')
        world_seed=_request_seed(request,SEARCH_SEED,0)
        if job.failure_context_for_world(0)['determinization_seed']!=world_seed:raise ValueError('Search-world seed differs')
        order=sorted(range(len(visits)),key=lambda i:(-visits[i],i))
        prior=list(result.root_network_prior);greedy=int(np.argmax(prior))
        row={'root_id':root['id'],'world_seed':world_seed,'visits':visits,'root_value_sums':sums,
            'q':[s/n if n else None for s,n in zip(sums,visits,strict=True)],'prior':prior,
            'visit_order':order,'mcts_choice':result.chosen_candidate_index,'greedy_choice':greedy,
            'child_value':[None]*len(visits),'child_raw_value':[None]*len(visits),'child_actor':[None]*len(visits),
            'actions':[to_primitive(a) for a in request.legal_actions]}
        for index in range(len(visits)):
            engine=determinize_request(request,seed=world_seed,engine_mode='fast')
            response,_=child(engine,index)
            if isinstance(response,TerminalResult):
                row['child_value'][index]=row['child_raw_value'][index]=payoff(response)
            else:
                row['child_actor'][index]=engine.current_player
                references.append((row,index,engine.current_player));child_requests.append(response)
        rows.append(row)
    for start in range(0,len(child_requests),64):
        for (row,index,actor),value in zip(references[start:start+64],evaluator.value_only(child_requests[start:start+64]),strict=True):
            row['child_raw_value'][index]=float(value)*(1 if actor==0 else -1)
            row['child_value'][index]=root_value(value,actor)
    for path,digest in source['source_files'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Source changed during root search')
    output={'checkpoint':previous['checkpoint'],'source_bank':{'path':str(BANK.relative_to(ROOT)),'sha256':sha256(BANK)},
        'producer':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha256(Path(__file__))},
        'search':config,'root_noise':False,'temperature':0.,'source':source,'environment':environment_identity(),
        'inference':inference.to_dict(),'command':sys.argv,'seconds':time.perf_counter()-started,'rows':rows,
        'scope':'One frozen S1024/N1 search per existing root. Q=sum/visits in root-player perspective; unvisited Q is undefined. Child V uses same determinization, forced-choice folding and native [-1,1] clipping; raw V retained.'}
    write_json(ART/'search_roots.json',output)
    conflicts=[r for r in rows if r['mcts_choice']!=r['greedy_choice']]
    print(json.dumps({'roots':len(rows),'search_policy_conflicts':len(conflicts),'seconds':output['seconds']}),flush=True)


if __name__=='__main__':main()
