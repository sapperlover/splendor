"""Original paired-action experiment, changing only continuation to R43 S1024."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass,field
import fcntl
import importlib.util
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity,read_json,sha256,source_identity,write_json
from splendor.catalog import load_default_catalog
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner,NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import state_digest,to_primitive
from splendor.result import TerminalResult
from splendor.rules import player_score
from splendor.search_inference import FrozenSearchInference,InferenceConfig
from splendor.search_policy import determinize_request
from splendor.selfplay import derive_seed

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'/ART.name
PARENT=ROOT/'artifacts/entity_attention_r43_action_ranking_64_20261015'

# Reuse the frozen statistical/replay implementation without changing its files.
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('r43_greedy_action_reference',PARENT/'run.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
sys.path.remove(str(PARENT))
base.ART=ART;base.RUNTIME=RUNTIME
SEARCH_SEED_BASE=38500000


def identity(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}


def prepare():
    previous=read_json(PARENT/'plan.json')
    plan=dict(previous)
    search=dict(read_json(ROOT/previous['search_evidence']['path'])['search'])
    plan.update(protocol='r43-action-ranking-64-paired-worlds-s1024-v1',runner=identity(Path(__file__)),
        parent_plan=identity(PARENT/'plan.json'),parent_statistics=identity(PARENT/'run.py'),
        continuation='Both sides frozen original R43 native S1024/N1, c_puct1.5, root noise0, T0; first comparison action forced; no training.',
        continuation_search=search,continuation_search_seed_base=SEARCH_SEED_BASE,
        scope=previous['scope'].replace('R43 Greedy continuation','R43 native S1024/N1 continuation'),
        change_scope='Same original64 roots,32/32 partition,forced candidate pairs,hidden-world seeds,posterior/bootstrap seeds,32->128 uncertainty rule,methods,400-turn cap,CPU eager2threads and128active arms. Only post-forced-action policy changes from Greedy to S1024; expansion decisions are recomputed from the new32-world returns.',
        search_seed_pairing='Per(root_id,world,actor) base shared across both arms. Each subsequent safe request derives its own search determinization via existing public-request hashing; no actual-world private state passed to search.')
    base.freeze(ART/'plan.json',plan)
    print({'prepared':True,'device':plan['device'],'continuation_simulations':search['simulations'],'roots':64},flush=True)


def verify_identity(plan):
    for key in ('checkpoint','bank','search_evidence','runner','search_producer','parent_plan','parent_statistics'):
        item=plan[key]
        if sha256(ROOT/item['path'])!=item['sha256']:raise ValueError(f'Frozen {key} changed')


def policy_seed(root_id,world,actor):
    return derive_seed(SEARCH_SEED_BASE,0,root_id,'paired_continuation_search',world,actor)


@dataclass
class SearchArm:
    root:dict
    comparison:dict
    world:int
    arm:int
    model:object
    evaluator:object
    search:dict
    actions:list=field(default_factory=list)
    search_decision_indices:list=field(default_factory=list)
    simulations:int=0
    determinizations:int=0
    child_value:float|None=None
    child_raw_value:float|None=None
    child_pending:bool=True
    status:str|None=None
    job:object=None

    def __post_init__(self):
        self.seed=base.world_seed(self.root['id'],self.world)
        self.engine=determinize_request(decode_request(self.root['request']),seed=self.seed,engine_mode='fast')
        self.initial_digest=state_digest(self.engine._debug_state())
        self.response=self.engine.decision()
        self.apply(self.comparison['a' if self.arm==0 else 'b'])
        while isinstance(self.response,DecisionRequest) and len(self.response.legal_actions)==1:
            self.apply(0)
            if len(self.actions)>=512:raise RuntimeError('Forced child cap')
        self.initial_forced_actions=len(self.actions)
        self.policies=tuple(NativePUCTSearchPolicy(self.model,evaluator=self.evaluator,
            config=MCTSConfig(**{**self.search,'seed':policy_seed(self.root['id'],self.world,actor)})) for actor in (0,1))

    def apply(self,index):
        self.actions.append(int(index))
        self.response=self.engine.apply_candidate(index,state_version=self.engine.state_version)

    def finish_if_ready(self):
        if isinstance(self.response,TerminalResult):
            self.status='natural_terminal';self.z=base.payoff(self.response)
            if self.child_pending:self.child_value=self.child_raw_value=float(self.z);self.child_pending=False
            return True
        if sum(self.engine._debug_state().completed_turns)>=400:
            self.status='engineering_truncation';self.z=None;return True
        return False

    def set_child_value(self,value):
        self.child_value=base.root_value(value,self.engine.current_player)
        self.child_raw_value=float(value)*(1 if self.engine.current_player==0 else -1)
        self.child_pending=False

    def ensure_job(self):
        while self.job is None and not self.finish_if_ready():
            if len(self.response.legal_actions)==1:
                self.apply(0)
            else:
                self.job=self.policies[self.engine.current_player].start_job(self.response,add_root_noise=False,temperature=0.)

    def consume(self):
        result=self.job.result
        if (not result.searchable or result.simulations!=self.search['simulations'] or
                tuple(result.simulation_budgets)!=(self.search['simulations'],) or
                sum(result.root_visit_counts)!=self.search['simulations'] or
                tuple(result.root_network_prior)!=tuple(result.root_used_prior)):
            raise ValueError('Continuation is not complete noise-free S1024/N1')
        self.search_decision_indices.append(len(self.actions))
        self.simulations+=result.simulations;self.determinizations+=1
        self.apply(result.chosen_candidate_index);self.job=None

    def record(self):
        state=self.engine._debug_state()
        return {'root_id':self.root['id'],'world':self.world,'arm':self.arm,'world_seed':self.seed,
            'initial_state_digest':self.initial_digest,'first_action':self.comparison['a' if self.arm==0 else 'b'],
            'status':self.status,'z':self.z,'scores':[player_score(p,self.engine.catalog) for p in state.players],
            'completed_turns':list(state.completed_turns),'purchased_counts':[len(p.purchased_card_ids) for p in state.players],
            'child_value':self.child_value,'child_raw_value':self.child_raw_value,'actions':self.actions,
            'initial_forced_actions':self.initial_forced_actions,'search_decision_indices':self.search_decision_indices,
            'search_simulations':self.simulations,'search_determinizations':self.determinizations,
            'simulations_per_decision':self.search['simulations'],
            'search_seed_bases':[policy_seed(self.root['id'],self.world,a) for a in (0,1)]}


def drive(tasks,model,evaluator,search,active_limit=128):
    waiting=deque(tasks);active=[];completed=0;started=time.perf_counter();tick=started
    runner=NativeBatchRunner(evaluator,load_default_catalog(),capacity=128,buckets=(16,32,48,64))
    while waiting or active:
        while waiting and len(active)<active_limit:
            active.append(SearchArm(*waiting.popleft(),model,evaluator,search))
        keep=[]
        for arm in active:
            if arm.finish_if_ready():write_json(base.arm_path(arm.root['id'],arm.world,arm.arm),arm.record());completed+=1
            else:keep.append(arm)
        active=keep
        pending=[a for a in active if a.child_pending]
        if pending:
            for arm,value in zip(pending,evaluator.value_only([a.response for a in pending]),strict=True):arm.set_child_value(value)
        for arm in active:arm.ensure_job()
        jobs=[a.job for a in active if a.job is not None]
        if jobs:
            if runner.run(jobs)<=0:raise RuntimeError('Continuation searches stalled')
            for arm in active:
                if arm.job is not None and arm.job.done:arm.consume()
        if time.perf_counter()-tick>=30:
            print({'event':'S1024_progress','completed':completed,'queued':len(waiting),'active':len(active),'seconds':time.perf_counter()-started},flush=True)
            tick=time.perf_counter()
    return {'arms':completed,'seconds':time.perf_counter()-started,'native_batches':runner.batches,'native_model_rows':runner.rows}


def verify_arm(root,row,checked=False):
    if row['world_seed']!=base.world_seed(root['id'],row['world']):raise ValueError('World seed differs')
    engine=determinize_request(decode_request(root['request']),seed=row['world_seed'],engine_mode='checked' if checked else 'fast')
    if state_digest(engine._debug_state())!=row['initial_state_digest']:raise ValueError('Initial world differs')
    searched=[]
    for i,index in enumerate(row['actions']):
        request=engine.decision()
        if not isinstance(request,DecisionRequest):raise ValueError('Action follows terminal')
        if i>=row['initial_forced_actions'] and len(request.legal_actions)>1:searched.append(i)
        engine.apply(request.legal_actions[index])
    terminal=engine.decision()
    if row['status']=='natural_terminal':
        if not isinstance(terminal,TerminalResult) or base.payoff(terminal)!=row['z'] or list(terminal.scores)!=row['scores'] or list(terminal.completed_turns)!=row['completed_turns']:
            raise ValueError('Continuation terminal replay differs')
    elif isinstance(terminal,TerminalResult) or row['z'] is not None:raise ValueError('Censored continuation mislabelled')
    if searched!=row['search_decision_indices'] or row['search_determinizations']!=len(searched) or row['search_simulations']!=row['simulations_per_decision']*len(searched):
        raise ValueError('Actual per-decision search coverage differs')
    if row['search_seed_bases']!=[policy_seed(root['id'],row['world'],a) for a in (0,1)]:raise ValueError('Continuation search seeds differ')


def run(smoke=False):
    plan=read_json(ART/'plan.json');verify_identity(plan)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(plan['torch_threads']);torch.set_num_interop_threads(1)
        name=('smoke-' if smoke else 'formal-')+str(time.time_ns())
        execution={'source':source_identity(ROOT,RUNTIME/name),'environment':environment_identity(),'plan':identity(ART/'plan.json'),
            'status':'running','command':sys.argv,'start_unix':time.time()}
        shutil.copyfile(__file__,RUNTIME/name/'run.py')
        write_json(ART/'execution'/f'{name}.json',execution)
        try:
            roots=read_json(ROOT/plan['bank']['path'])['roots'];comparisons=plan['comparisons']
            model=load_frozen_actor_critic_checkpoint(ROOT/plan['checkpoint']['path'],device=plan['device'])
            evaluator=FrozenSearchInference(model,device=plan['device'],config=InferenceConfig(**plan['inference']))
            if smoke:
                c=next(c for c in comparisons if c['same_action_control']);r=roots[c['root_id']]
                search={**plan['continuation_search'],'simulations':16}
                timing=drive([(r,c,10000,a) for a in (0,1)],model,evaluator,search,active_limit=2)
                pair=[read_json(base.arm_path(c['root_id'],10000,a)) for a in (0,1)]
                if pair[0]['actions']!=pair[1]['actions'] or pair[0]['z']!=pair[1]['z']:raise ValueError('Search duplicate differs')
                for arm in pair:verify_arm(r,arm,checked=True)
                # Also exercise the exact formal1024 budget on a real root.
                policy=NativePUCTSearchPolicy(model,evaluator=evaluator,config=MCTSConfig(**plan['continuation_search']))
                job=policy.start_job(decode_request(r['request']),add_root_noise=False,temperature=0.)
                runner=NativeBatchRunner(evaluator,load_default_catalog(),capacity=128,buckets=(16,32,48,64))
                while not job.done:
                    if runner.run([job])<=0:raise RuntimeError('Formal-budget smoke stalled')
                if job.result.simulations!=1024 or tuple(job.result.simulation_budgets)!=(1024,):raise ValueError('Formal budget differs')
                write_json(ART/'smoke.json',{'passed':True,'full_pair_simulations':16,'formal_root_simulations':1024,'timing':timing,
                    'pair':pair,'formal_root_result':job.result.to_dict(),'scope':'S16 full paired continuations plus one real S1024 root; engineering verification, excluded from formal statistics.'})
            else:
                timing=[]
                for phase,n in (('initial',32),('expanded',128)):
                    selected=comparisons if n==32 else [c for c in comparisons if not c['same_action_control'] and not base.assess(c,32)['resolved']]
                    tasks=[(roots[c['root_id']],c,w,a) for w in range(n) for c in selected for a in (0,1) if not base.arm_path(c['root_id'],w,a).exists()]
                    timing.append({'phase':phase,'roots':len(selected),**drive(tasks,model,evaluator,plan['continuation_search'],active_limit=plan['active_arms'])})
                    for c in selected:base.commit_root(c['root_id'],n)
                    write_json(ART/'progress.json',{'status':phase+'_complete','expanded_roots':[c['root_id'] for c in selected] if n==128 else [],'timing':timing})
                    print({'event':phase+'_complete','roots':len(selected),'timing':timing[-1]},flush=True)
                execution['timing']=timing
                verify_identity(plan)
                for path,digest in execution['source']['source_files'].items():
                    if sha256(ROOT/path)!=digest:raise ValueError('Source changed during execution')
            execution.update(status='complete',end_unix=time.time(),inference_stats=evaluator.stats.to_dict())
        except BaseException as error:
            execution.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',end_unix=time.time(),error=f'{type(error).__name__}: {error}')
            raise
        finally:write_json(ART/'execution'/f'{name}.json',execution)


def analyze():
    base.verify_identity=verify_identity;base.verify_arm=verify_arm
    base.analyze()
    summary=read_json(ART/'summary.json');simulations=decisions=0
    for s in summary['refined_per_root']:
        for pair in read_json(ART/'raw'/f"root-{s['root_id']:02d}-n{s['worlds']:03d}.json")['pairs']:
            for arm in pair:
                if arm['simulations_per_decision']!=1024:raise ValueError('Nonformal search budget found')
                simulations+=arm['search_simulations'];decisions+=arm['search_determinizations']
    summary['continuation_budget']={'simulations_per_searchable_decision':1024,'determinizations':1,'simulations':simulations,'searchable_decisions':decisions}
    write_json(ART/'summary.json',summary)
    path=ART/'results.md';text=path.read_text()
    text=text.replace('后续双方只用R43 Greedy','后续双方固定R43 native S1024/N1、无根噪声、T0')
    text=text.replace('而参考值来自Greedy续局；二者差异可能包含续局策略不一致','参考值也来自双方S1024续局；单棵根树与跨世界完整续局仍是不同估计')
    text+='\n实际续局搜索预算核验：'+str(summary['continuation_budget'])+'。固定32份及自适应结果可与Greedy分支在同一根／世界种子下比较。\n'
    path.write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','smoke','run','analyze'))
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='analyze':analyze()
    else:run(smoke=args.action=='smoke')
