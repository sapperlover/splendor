"""Paired hidden-world action ranking, R43 greedy continuations, 32 -> 128."""
from __future__ import annotations

import argparse
from collections import Counter,deque
from dataclasses import dataclass,field
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity,read_json,sha256,source_identity,write_json
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.native_policy import _extension,_native_catalog
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import state_digest,to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference,InferenceConfig
from splendor.search_policy import determinize_request
from splendor.selfplay import derive_seed
from splendor.selfplay_data import STATE_FIELDS
from search import child,payoff,root_value

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'/ART.name
BANK=ROOT/'artifacts/entity_attention_r43_value_signal_64_20261014/root_bank.json'
SEEDS={'selection':38000000,'world':38200000,'posterior':38300000,'bootstrap':38400000}


def identity(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}


def freeze(path,value):
    if path.exists():
        if read_json(path)!=value:raise ValueError(f'Frozen experiment changed: {path}')
    else:write_json(path,value)


def prepare():
    search=read_json(ART/'search_roots.json');roots=read_json(BANK)['roots']
    rows={r['root_id']:r for r in search['rows']}
    conflicts=[r for r in search['rows'] if r['mcts_choice']!=r['greedy_choice']]
    if len(conflicts)!=28:raise ValueError('Frozen 28-conflict population changed')
    agreements=sorted(set(range(64))-{r['root_id'] for r in conflicts})
    rng=np.random.default_rng(SEEDS['selection'])
    controls=set(map(int,rng.choice(agreements,4,replace=False)))
    group_b={r['root_id'] for r in conflicts}|controls
    comparisons=[]
    for root in roots:
        r=rows[root['id']];a=r['mcts_choice']
        b=r['greedy_choice'] if root['id'] in group_b else next(i for i in r['visit_order'] if i!=a)
        comparisons.append({'root_id':root['id'],'group':'search_vs_policy' if root['id'] in group_b else 'search_top2',
            'a':a,'b':b,'same_action_control':a==b,'stage':root['stage'],
            'conflict_prior_gap':r['prior'][r['greedy_choice']]-r['prior'][a],
            'conflict_visit_gap':(r['visits'][a]-r['visits'][r['greedy_choice']])/1024,
            'v_search_world':[r['child_value'][a],r['child_value'][b]],
            'q':[r['q'][a],r['q'][b]],'visits':[r['visits'][a],r['visits'][b]],
            'greedy_choice':r['greedy_choice']})
    plan={'protocol':'r43-action-ranking-64-paired-worlds-v1','client_date':'2026-10-15',
        'checkpoint':search['checkpoint'],'bank':identity(BANK),'search_evidence':identity(ART/'search_roots.json'),
        'runner':identity(Path(__file__)),'search_producer':identity(ART/'search.py'),
        'roots':64,'comparisons':comparisons,'initial_worlds':32,'expanded_worlds':128,'seeds':SEEDS,
        'continuation':'Both sides frozen original R43 Policy Greedy, candidate-index-first argmax ties; no further MCTS, no noise, no action sampling.',
        'pairing':'Each root/world draws one full hidden state; both forced-action arms start in exactly that same world and retain its tier deck orders.',
        'uncertainty':'Dirichlet(.5,.5,.5,.5,.5) posterior for paired z_A-z_B in {-2,-1,0,1,2};20000 deterministic posterior draws;95% equal-tail credible interval containing0 -> add96 worlds. Same-action controls stay32. Censored outcomes widen intervals using worst-case +/-2 per missing pair. Bayesian credible intervals, not frequentist coverage claims.',
        'primary_methods':['V_same_search_world','root_Q','MCTS_visit_choice'],
        'auxiliary_methods':['V_mean_paired_worlds','PolicyGreedy_when_in_pair'],
        'scope':'Original64 roots partition32/32, second group28 conflicts+4 same-action controls as approved. Controls excluded from ranking/correlation/regret. ChildV in original search world is primary; mean childV across evaluation worlds is auxiliary and shares sampling with targets. Reference action value is conditional on R43 Greedy continuation and existing uniform hidden-world sampler; not optimal play or proof of full-search strength.',
        'max_completed_player_turns':400,'censoring':'Engineering turn-limit/nonterminal is recorded with null z, never called draw; unresolved missing pairs reported with worst-case bounds.',
        'active_arms':256,'inference':search['inference']}
    freeze(ART/'plan.json',plan)
    print({'groups':dict(Counter(x['group'] for x in comparisons)),'conflicts':28,'controls':sorted(controls)},flush=True)


def verify_identity(plan):
    for key in ('checkpoint','bank','search_evidence','runner','search_producer'):
        item=plan[key]
        if sha256(ROOT/item['path'])!=item['sha256']:raise ValueError(f'Frozen {key} changed')


def world_seed(root_id,world):return derive_seed(SEEDS['world'],0,root_id,'paired_action_world',world)


def collate_native(encoded):
    if not encoded:raise ValueError('Empty inference batch')
    width=max(x['candidate_features'].shape[0] for x in encoded)
    arrays={name:np.stack([x[name] for x in encoded]) for name in STATE_FIELDS}
    dims=encoded[0]['candidate_features'].shape[1]
    arrays['candidate_features']=np.zeros((len(encoded),width,dims),dtype=np.float32)
    arrays['candidate_mask']=np.zeros((len(encoded),width),dtype=np.bool_)
    for i,row in enumerate(encoded):
        n=len(row['candidate_features']);arrays['candidate_features'][i,:n]=row['candidate_features'];arrays['candidate_mask'][i,:n]=True
    arrays['feature_schema_version']=FEATURE_SCHEMA_VERSION
    return arrays


@dataclass
class Arm:
    root:dict
    comparison:dict
    world:int
    arm:int
    catalog:object
    game:object=field(init=False)
    actions:list=field(default_factory=list)
    child_value:float|None=None
    child_raw_value:float|None=None
    child_pending:bool=True
    status:str|None=None

    def __post_init__(self):
        self.seed=world_seed(self.root['id'],self.world)
        engine=determinize_request(decode_request(self.root['request']),seed=self.seed,engine_mode='fast')
        self.initial_digest=state_digest(engine._debug_state())
        self.game=_extension().NativeGame(_native_catalog(self.catalog),engine._debug_state())
        if not self.game.verify_actions(decode_request(self.root['request'])):raise ValueError('Native root candidates differ')
        self.apply(self.comparison['a' if self.arm==0 else 'b'])
        while self.game.snapshot()['phase']!=4 and len(self.game.actions())==1:
            self.apply(0)
            if len(self.actions)>=512:raise RuntimeError('Forced child cap')

    def apply(self,index):self.game.apply_index(index);self.actions.append(int(index))

    def finish_if_ready(self):
        snapshot=self.game.snapshot()
        if snapshot['phase']==4:
            self.status='natural_terminal'
            self.z=0 if snapshot['winner']==-1 else 1 if snapshot['winner']==0 else -1
            if self.child_pending:self.child_value=self.child_raw_value=float(self.z);self.child_pending=False
            return True
        if sum(snapshot['turns'])>=400:
            self.status='engineering_truncation';self.z=None
            return True
        return False

    def infer(self,logits,value):
        if self.child_pending:
            actor=self.game.snapshot()['actor']
            self.child_value=root_value(value,actor)
            self.child_raw_value=float(value)*(1 if actor==0 else -1)
            self.child_pending=False
        self.apply(int(np.argmax(logits)))

    def record(self):
        snapshot=self.game.snapshot()
        scores=[sum(self.catalog.cards[i].points for i in p['purchased'])+sum(self.catalog.nobles[i].points for i in p['nobles']) for p in snapshot['players']]
        return {'root_id':self.root['id'],'world':self.world,'arm':self.arm,'world_seed':self.seed,
            'initial_state_digest':self.initial_digest,'first_action':self.comparison['a' if self.arm==0 else 'b'],
            'status':self.status,'z':self.z,'scores':scores,'completed_turns':list(snapshot['turns']),
            'purchased_counts':[len(p['purchased']) for p in snapshot['players']],
            'child_value':self.child_value,'child_raw_value':self.child_raw_value,'actions':self.actions}


def arm_path(root_id,world,arm):return RUNTIME/'work.games'/f'root-{root_id:02d}-world-{world:03d}-arm-{arm}.json'


def drive(tasks,evaluator,catalog,active_limit=256):
    waiting=deque(tasks);active=[];completed=0;started=time.perf_counter();tick=started
    while waiting or active:
        while waiting and len(active)<active_limit:
            active.append(Arm(*waiting.popleft(),catalog))
        keep=[]
        for arm in active:
            if arm.finish_if_ready():write_json(arm_path(arm.root['id'],arm.world,arm.arm),arm.record());completed+=1
            else:keep.append(arm)
        active=keep
        for start in range(0,len(active),256):
            batch=active[start:start+256]
            logits,values=evaluator.policy_value_native(collate_native([a.game.encoded() for a in batch]))
            for a,x,v in zip(batch,logits,values,strict=True):a.infer(x[:len(a.game.actions())],v)
        if time.perf_counter()-tick>=30:
            print(json.dumps({'event':'greedy_progress','completed':completed,'queued':len(waiting),'active':len(active),'seconds':time.perf_counter()-started}),flush=True);tick=time.perf_counter()
    return {'arms':completed,'seconds':time.perf_counter()-started}


def commit_root(root_id,n):
    records=[]
    for w in range(n):
        pair=[read_json(arm_path(root_id,w,a)) for a in (0,1)]
        if pair[0]['initial_state_digest']!=pair[1]['initial_state_digest'] or pair[0]['world_seed']!=pair[1]['world_seed']:
            raise ValueError('Arms do not share the same hidden world')
        records.append(pair)
    write_json(ART/'raw'/f'root-{root_id:02d}-n{n:03d}.json',{'root_id':root_id,'worlds':n,'pairs':records})


def posterior(differences,root_id,n,missing=0,draws=20000):
    d=np.asarray(differences,dtype=np.int64)
    if not np.isin(d,[-2,-1,0,1,2]).all():raise ValueError('Invalid paired terminal difference')
    counts=np.bincount(d+2,minlength=5)
    rng=np.random.default_rng(derive_seed(SEEDS['posterior'],0,root_id,'posterior',n))
    samples=rng.dirichlet(counts+.5,draws)@np.arange(-2,3)
    # Missing outcomes remain bounded, rather than being fabricated as draws.
    lo,hi=np.quantile(samples,[.025,.975]);observed=len(d);total=observed+missing
    widened=[float((lo*observed-2*missing)/total),float((hi*observed+2*missing)/total)]
    return {'counts':counts.tolist(),'complete_pairs':observed,'missing_pairs':missing,
        'delta_complete_pairs':float(d.mean()) if observed else None,'delta_worst_case_bounds':[float((d.sum()-2*missing)/total),float((d.sum()+2*missing)/total)],
        'credible_interval95':widened,'resolved':widened[0]>0 or widened[1]<0,
        'p_a_better':float(np.mean(samples>0)) if not missing else None,
        'expected_regret_a':float(np.maximum(-samples,0).mean()) if not missing else None,
        'expected_regret_b':float(np.maximum(samples,0).mean()) if not missing else None}


def assess(comparison,n):
    raw=read_json(ART/'raw'/f"root-{comparison['root_id']:02d}-n{n:03d}.json")
    valid=[p for p in raw['pairs'] if all(a['z'] is not None for a in p)]
    stat=posterior([p[0]['z']-p[1]['z'] for p in valid],comparison['root_id'],n,missing=n-len(valid))
    if comparison['same_action_control']:
        if any(a!=b and (a['actions']!=b['actions'] or a['z']!=b['z']) for a,b in raw['pairs']):
            raise ValueError('Same-action paired control differs')
        stat['resolved']=True
    stat.update(root_id=comparison['root_id'],group=comparison['group'],worlds=n,same_action_control=comparison['same_action_control'],
        v_mean_worlds=[float(np.mean([p[a]['child_value'] for p in raw['pairs']])) for a in (0,1)],
        natural_arms=sum(a['status']=='natural_terminal' for p in raw['pairs'] for a in p),
        censored_arms=sum(a['status']!='natural_terminal' for p in raw['pairs'] for a in p))
    return stat


def run(smoke=False):
    plan=read_json(ART/'plan.json');verify_identity(plan)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        name=('smoke-' if smoke else 'formal-')+str(time.time_ns())
        execution={'source':source_identity(ROOT,RUNTIME/name),'environment':environment_identity(),'plan':identity(ART/'plan.json'),
            'status':'running','command':sys.argv,'start_unix':time.time()}
        shutil.copyfile(__file__,RUNTIME/name/'run.py');write_json(ART/'execution'/f'{name}.json',execution)
        try:
            roots=read_json(BANK)['roots'];comparisons=plan['comparisons']
            from splendor.catalog import load_default_catalog
            catalog=load_default_catalog();model=load_frozen_actor_critic_checkpoint(ROOT/plan['checkpoint']['path'],device='cuda')
            evaluator=FrozenSearchInference(model,device='cuda',config=InferenceConfig(**plan['inference']))
            if smoke:
                c=next(c for c in comparisons if c['same_action_control']);r=roots[c['root_id']]
                # Distinct world id avoids colliding with formal32/128 records.
                timing=drive([(r,c,10000,a) for a in (0,1)],evaluator,catalog,active_limit=2)
                pair=[read_json(arm_path(c['root_id'],10000,a)) for a in (0,1)]
                if pair[0]['actions']!=pair[1]['actions'] or pair[0]['z']!=pair[1]['z']:raise ValueError('Greedy duplicate differs')
                for a in pair:verify_arm(r,a,checked=True)
                write_json(ART/'smoke.json',{'passed':True,'same_action_world_reproducible':True,'timing':timing,'pair':pair})
            else:
                timing=[]
                for phase,n in (('initial',32),('expanded',128)):
                    selected=comparisons if n==32 else [c for c in comparisons if not c['same_action_control'] and not assess(c,32)['resolved']]
                    tasks=[]
                    for w in range(n):
                        for c in selected:
                            for a in (0,1):
                                path=arm_path(c['root_id'],w,a)
                                if not path.exists():tasks.append((roots[c['root_id']],c,w,a))
                    timing.append({'phase':phase,'roots':len(selected),**drive(tasks,evaluator,catalog)})
                    for c in selected:commit_root(c['root_id'],n)
                    write_json(ART/'progress.json',{'status':phase+'_complete','expanded_roots':[c['root_id'] for c in selected] if n==128 else [],'timing':timing})
                    print(json.dumps({'event':phase+'_complete','roots':len(selected),'timing':timing[-1]}),flush=True)
                for path,digest in execution['source']['source_files'].items():
                    if sha256(ROOT/path)!=digest:raise ValueError('Source changed during greedy continuations')
                verify_identity(plan)
                execution['timing']=timing
            execution.update(status='complete',end_unix=time.time(),inference_stats=evaluator.stats.to_dict())
        except BaseException as error:
            execution.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',end_unix=time.time(),error=f'{type(error).__name__}: {error}')
            raise
        finally:write_json(ART/'execution'/f'{name}.json',execution)


def verify_arm(root,row,checked=False):
    if row['world_seed']!=world_seed(root['id'],row['world']):raise ValueError('World seed differs')
    engine=determinize_request(decode_request(root['request']),seed=row['world_seed'],engine_mode='checked' if checked else 'fast')
    if state_digest(engine._debug_state())!=row['initial_state_digest']:raise ValueError('Initial world differs')
    for index in row['actions']:
        request=engine.decision()
        if not isinstance(request,DecisionRequest):raise ValueError('Action follows terminal')
        engine.apply(request.legal_actions[index])
    response=engine.decision()
    if row['status']=='natural_terminal':
        if not isinstance(response,TerminalResult) or payoff(response)!=row['z'] or list(response.scores)!=row['scores'] or list(response.completed_turns)!=row['completed_turns']:
            raise ValueError('Native/Python continuation terminal differs')
    elif isinstance(response,TerminalResult) or row['z'] is not None:
        raise ValueError('Censored continuation incorrectly labelled')


def method_delta(comparison,stat,name):
    if name=='V_same_search_world':return comparison['v_search_world'][0]-comparison['v_search_world'][1]
    if name=='root_Q':return comparison['q'][0]-comparison['q'][1] if None not in comparison['q'] else None
    if name=='V_mean_paired_worlds':return stat['v_mean_worlds'][0]-stat['v_mean_worlds'][1]
    if name=='MCTS_visit_choice':return (comparison['visits'][0]-comparison['visits'][1])/1024
    if name=='PolicyGreedy_when_in_pair':
        if comparison['greedy_choice']==comparison['a']:return 1.
        if comparison['greedy_choice']==comparison['b']:return -1.
        return None
    raise ValueError(name)


def metrics(comparisons,stats,method):
    rows=[]
    for c,s in zip(comparisons,stats,strict=True):
        pred=method_delta(c,s,method);delta=s['delta_complete_pairs']
        if c['same_action_control'] or pred is None or delta is None or s['missing_pairs']:continue
        choice=1 if method=='MCTS_visit_choice' else int(np.sign(pred))
        truth=int(np.sign(delta));accuracy=.5 if choice==0 or truth==0 else float(choice==truth)
        regret=(abs(delta)-choice*delta)/2
        expected_regret=(s['expected_regret_a'] if choice==1 else s['expected_regret_b'] if choice==-1 else (s['expected_regret_a']+s['expected_regret_b'])/2)
        p=s['p_a_better'];expected_accuracy=p if choice==1 else 1-p if choice==-1 else .5
        rows.append((pred,delta,accuracy,regret,expected_regret,expected_accuracy,s['resolved']))
    if not rows:return {'cases':0}
    x=np.asarray(rows,dtype=float);resolved=x[:,6].astype(bool)
    correlation=float(np.corrcoef(x[:,0],x[:,1])[0,1]) if x[:,0].std()>0 and x[:,1].std()>0 else None
    rng=np.random.default_rng(SEEDS['bootstrap']);indices=rng.integers(len(x),size=(10000,len(x)))
    bx=x[indices,0];by=x[indices,1];bx=bx-bx.mean(axis=1,keepdims=True);by=by-by.mean(axis=1,keepdims=True)
    denominator=np.sqrt((bx*bx).sum(axis=1)*(by*by).sum(axis=1));valid=denominator>0
    correlations=(bx*by).sum(axis=1)[valid]/denominator[valid]
    return {'cases':len(x),'resolved_cases':int(resolved.sum()),'empirical_accuracy_ties_half':float(x[:,2].mean()),
        'accuracy_ci95_roots':[float(t) for t in np.quantile(x[indices,2].mean(axis=1),[.025,.975])],
        'resolved_accuracy':float(x[resolved,2].mean()) if resolved.any() else None,
        'delta_correlation':correlation,'delta_correlation_ci95_roots':[float(t) for t in np.quantile(correlations,[.025,.975])] if len(correlations) else None,
        'mean_empirical_regret_z':float(x[:,3].mean()),'empirical_regret_ci95_roots':[float(t) for t in np.quantile(x[indices,3].mean(axis=1),[.025,.975])],
        'mean_posterior_regret_z':float(x[:,4].mean()),'mean_posterior_accuracy':float(x[:,5].mean()),
        'mean_empirical_regret_score_pp':float(50*x[:,3].mean()),'scope':'Censored roots excluded from these point metrics and listed separately; controls excluded; empirical rankings and regrets use finite MC estimates, not known optimal action values. Visit difference correlation is a confidence proxy, not a value difference.'}


def analyze():
    plan=read_json(ART/'plan.json');verify_identity(plan)
    roots=read_json(BANK)['roots'];comparisons=plan['comparisons'];initial=[assess(c,32) for c in comparisons]
    refined=[assess(c,128) if (ART/'raw'/f"root-{c['root_id']:02d}-n128.json").exists() else assess(c,32) for c in comparisons]
    replays=checked=0
    for c,s in zip(comparisons,refined,strict=True):
        raw=read_json(ART/'raw'/f"root-{c['root_id']:02d}-n{s['worlds']:03d}.json")
        for w,pair in enumerate(raw['pairs']):
            for arm in pair:
                verify_arm(roots[c['root_id']],arm,checked=w==0);replays+=1;checked+=int(w==0)
    methods=plan['primary_methods']+plan['auxiliary_methods'];panels={}
    for label,stats in (('fixed32',initial),('adaptive',refined)):
        panels[label]={}
        for group in ('all','search_top2','search_vs_policy'):
            indices=[i for i,c in enumerate(comparisons) if group=='all' or c['group']==group]
            panels[label][group]={m:metrics([comparisons[i] for i in indices],[stats[i] for i in indices],m) for m in methods}
    result={'status':'complete','plan':identity(ART/'plan.json'),'panels':panels,'initial_per_root':initial,'refined_per_root':refined,
        'counts':{'roots':64,'distinct_action_comparisons':60,'same_action_controls':4,'expanded_roots':sum(s['worlds']==128 for s in refined),
            'world_pairs':sum(s['worlds'] for s in refined),'arms':replays,'checked_python_replays':checked,
            'censored_arms':sum(s['censored_arms'] for s in refined),'unresolved_distinct_roots':sum(not s['resolved'] for s in refined if not s['same_action_control'])},
        'scope':plan['scope'],'uncertainty':plan['uncertainty']}
    write_json(ART/'summary.json',result)
    lines=['# R43：64根配对动作排序诊断','', '根部无噪声S1024/N1；实际访问首选为A。第一组32根A对访问次选；第二组28根A对Policy Greedy首选，加4根同选控制。每根先32份同世界配对，95%后验区间跨0者补到128；后续双方只用R43 Greedy。', '',f"完成：{result['counts']}。",'','| 面板 | 方法 | 有效根 | 经验排序准确率 | 已分辨根准确率 | 差值相关 | 错选损失z | 后验期望损失z |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group,methods_data in panels['adaptive'].items():
        for method,row in methods_data.items():
            if not row['cases']:continue
            fmt=lambda x:'—' if x is None else f'{x:.4f}'
            lines.append(f"| {group} | {method} | {row['cases']} | {row['empirical_accuracy_ties_half']:.4f} | {fmt(row['resolved_accuracy'])} | {fmt(row['delta_correlation'])} | {row['mean_empirical_regret_z']:.4f} | {row['mean_posterior_regret_z']:.4f} |")
    lines+=['','四个同选对照不计排序、相关与损失；未分辨局面不当作已知真值。固定32份完整对照、后验期望正确率、按根bootstrap区间、逐根差值及补样记录见summary.json。经验错选损失为两动作间基于MC均值的损失，后验期望损失考虑剩余不确定性，均非最优博弈遗憾值。z损失乘50为续局得分率百分点。', '', '子节点V主指标来自原S1024同一隐藏世界，按root玩家转换符号，并与native一致裁剪到[-1,1]；终局直接用真实payoff。V_mean_paired_worlds为额外平均世界诊断，其预测与MC目标共享抽样，只作辅助。MCTS的差值相关使用访问份额差，不是价值差。根Q是单世界搜索树自身策略的估计，而参考值来自Greedy续局；二者差异可能包含续局策略不一致，不能仅凭排序误差归罪value或搜索。', '', '隐藏世界采样沿用现有均匀重建器，非历史行为后验。本样本按搜索冲突分组，不是总体棋力评测。非终局工程上限不算和棋，若存在会单独列出并给收益差的最坏情况范围。']
    (ART/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'counts':result['counts'],'adaptive':panels['adaptive']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','smoke','run','analyze'))
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='analyze':analyze()
    else:run(smoke=args.action=='smoke')
