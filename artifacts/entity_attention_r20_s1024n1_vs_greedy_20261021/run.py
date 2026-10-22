"""Frozen R20 S1024 per world x N1 versus its own greedy, 1000 pairs."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from splendor.alphazero_recording import (environment_identity, read_json, sha256,
    source_identity, write_json)
from splendor.evaluation import FixedEvaluationConfig
from splendor.paired_statistics import paired_bootstrap_mean, paired_bootstrap_difference, serialized_pair_scores

ROOT = Path(os.environ.get('SPLENDOR_ARENA_ROOT', Path(__file__).resolve().parents[2]))
ART = ROOT/'artifacts/entity_attention_r20_s1024n1_vs_greedy_20261021'
RUNTIME = ROOT/'runtime'/ART.name
CHECKPOINT = ROOT/'runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt'
MODEL_SHA = 'fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd'
OUTPUT = ART/'evaluation.json'
CONFIG = ART/'config.json'
PAIRS = 1000
SEED = 42500000
EXPERIMENT_ID = 'entity-attention-r20-s1024n1-vs-greedy-2026-10-21'


def freeze(path, value):
    if path.exists():
        assert read_json(path) == value, f'Frozen input changed: {path}'
    else:
        write_json(path,value)


def prepare():
    if (ART/'plan.json').exists():
        verify();return read_json(CONFIG)
    assert sha256(CHECKPOINT)==MODEL_SHA
    source=read_json(ROOT/'configs/entity_attention_v1/random_budget_37423_p6k_v24k_20261019.json')
    collection=source['selfplay']['execution']
    inference={'mode':collection['inference_mode'],'actor_projection':collection['actor_projection'],
        'candidate_buckets':collection['candidate_buckets'],'batch_buckets':collection['batch_buckets'],
        'compile_mode':collection['compile_mode'],'compile_failure_policy':'fail_fast',
        'native_transport':collection['native_transport']}
    search=copy.deepcopy(source['selfplay']['search_config'])
    search.update(simulations=1024,determinizations=1,root_noise_fraction=0.,seed=SEED+30000,
        forced_playouts=0.,policy_target_pruning=False)
    raw={'policy_a':{'kind':'mcts_checkpoint','name':'R20_S1024N1',
            'checkpoint':str(CHECKPOINT),'search_backend':'native',
            'search_config':search,'inference':inference},
        'policy_b':{'kind':'greedy_checkpoint','name':'R20_greedy',
            'checkpoint':str(CHECKPOINT),'inference':inference},
        'device':'cuda','workers':1,'pairs':PAIRS,'seed_start':SEED,
        'policy_seed_base':SEED+10000,'bootstrap_resamples':10000,'bootstrap_seed':SEED+20000,
        'max_completed_turns':400,'adjudicate_deterministic_policy_cycles':True,
        'execution':{**collection,'pipeline_slots':1},
        'metadata':{'experiment':ART.name,'round':20,'model_sha256':MODEL_SHA,
            'scope':'Native independent determinizations: 1024 simulations in one independent world = 1024 total. Greedy is root-only batched logit argmax with zero simulations. Current twelve-noble rules.'}}
    config=FixedEvaluationConfig.from_dict(raw).to_dict()
    assert config['policy_a'].get('chance_widening') is None
    comparison=read_json(ROOT/'artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/config.json')
    for field in ('game_seeds','policy_seed_base','execution','policy_b'):
        assert config[field]==comparison[field],field
    assert {k:v for k,v in config['policy_a']['search_config'].items() if k!='determinizations'} == {
        k:v for k,v in comparison['policy_a']['search_config'].items() if k!='determinizations'}
    freeze(CONFIG,config)
    from splendor.catalog import load_default_catalog
    catalog=load_default_catalog()
    assert len(catalog.nobles)==12
    (ART/'catalog').mkdir(exist_ok=True)
    for path in (ROOT/'data').glob('splendor_*'):
        if path.suffix in ('.csv','.json'):shutil.copyfile(path,ART/'catalog'/path.name)
    frozen=RUNTIME/'source'
    identity=source_identity(ROOT,frozen)
    for path in [ROOT/'src/splendor/native_search.cpp',*ROOT.glob('src/splendor/_native_search*.so'),*ART.glob('*.py')]:
        target=frozen/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    shutil.copytree(ART/'catalog',frozen/'data',dirs_exist_ok=True)
    identity['snapshot_files']={str(p.relative_to(frozen)):sha256(p) for p in frozen.rglob('*') if p.is_file() and p.name!='dirty.patch'}
    freeze(ART/'plan.json',{'experiment':EXPERIMENT_ID,'pairs':PAIRS,'games':2*PAIRS,
        'checkpoint':str(CHECKPOINT.relative_to(ROOT)),'checkpoint_sha256':MODEL_SHA,
        'config_sha256':sha256(CONFIG),'runner_sha256':sha256(Path(__file__)),
        'source_collection_config_sha256':sha256(ROOT/'configs/entity_attention_v1/random_budget_37423_p6k_v24k_20261019.json'),
        'search':'native S1024N1: 1024 simulations in one independent world, total 1024; no widening',
        'execution':'native_cuda_pool, collection 12 workers/active1000/B500/local64/shared slots2; owner pipeline_slots1, wait5ms, native-cudagraph',
        'catalog_fingerprint':catalog.data_fingerprint,'nobles':len(catalog.nobles),
        'catalog_files':{p.name:sha256(p) for p in (ART/'catalog').iterdir()},'source':identity,
        'comparison_evaluation':{'path':'artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/evaluation.json',
            'sha256':sha256(ROOT/'artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/evaluation.json')},
        'statistics':'10000 whole-pair bootstrap resamples, unadjusted 95% interval; draws half score.'})
    return config


def verify():
    plan=read_json(ART/'plan.json')
    assert sha256(CHECKPOINT)==MODEL_SHA
    assert sha256(CONFIG)==plan['config_sha256']
    assert sha256(ART/'run.py')==plan['runner_sha256']
    for path,expected in plan['source']['snapshot_files'].items():assert sha256(RUNTIME/'source'/path)==expected,path
    for path,expected in plan['catalog_files'].items():assert sha256(ART/'catalog'/path)==expected,path
    return plan


def update_index(status):
    files=sorted(p for p in ART.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and '.games' not in ''.join(p.parts))
    entry={'id':EXPERIMENT_ID,'report':str((ART/'README.md').relative_to(ROOT)),
        'status':'valid' if status=='complete' else 'incomplete',
        'scope':f'R20 raw checkpoint {MODEL_SHA}; native S1024N1 (total 1024) versus same-model zero-simulation greedy, 1000 seat-swapped pairs under current 12-noble rules. Collector 12-worker GPU pool. Status: {status}. Strength on this panel does not establish training improvement.',
        'evidence':[str(p.relative_to(ROOT)) for p in files],
        'evidence_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in files}}
    path=ROOT/'artifacts/experiment_index.json';text=path.read_text();needle='  "entries": [\n'
    decoder=json.JSONDecoder();position=text.index(needle)+len(needle)
    while position<len(text):
        while text[position] in ' \n\r\t,':position+=1
        if text[position]==']':break
        current,end=decoder.raw_decode(text,position)
        if current['id']==EXPERIMENT_ID:
            lines=json.dumps(entry,ensure_ascii=False,indent=2).splitlines()
            path.write_text(text[:position]+lines[0]+'\n'+'\n'.join('    '+s for s in lines[1:])+text[end:])
            return
        position=end
    body='\n'.join('    '+s for s in json.dumps(entry,ensure_ascii=False,indent=2).splitlines())
    path.write_text(text.replace(needle,needle+body+',\n',1))


def analyze():
    raw=read_json(OUTPUT)
    assert raw['config']==read_json(CONFIG)
    assert raw['completed_pairs']==PAIRS and raw['valid_games']==2*PAIRS and raw['passed']
    seeds,scores=serialized_pair_scores(raw)
    assert seeds==tuple(read_json(CONFIG)['game_seeds'])
    ci=paired_bootstrap_mean(scores,resamples=10000,seed=SEED+20000).to_dict()
    assert ci==raw['paired_bootstrap']
    for side in ('policy_a','policy_b'):
        assert raw['checkpoint_identity'][side]['sha256']==MODEL_SHA
    assert raw['catalog_fingerprint']==read_json(ART/'plan.json')['catalog_fingerprint']
    assert raw['execution_stats']['backend']=='native_cuda_pool'
    for pair in raw['pairs']:
        assert [g['first_player'] for g in pair['games']]==[0,1]
        for game in pair['games']:
            a,b=game['policy_a_stats'],game['policy_b_stats']
            assert a['simulations']==1024*a['searchable_decisions']
            assert a['determinizations']==a['searchable_decisions']
            assert b.get('simulations',0)==0
            assert b['determinizations']==b['state_transitions']==0
    summary={k:raw[k] for k in ('completed_pairs','valid_games','a_wins','draws','a_losses',
        'win_rate_score','paired_bootstrap','a_first','a_second','exceptions','illegal_actions',
        'engineering_truncations','invariant_failures','policy_cycle_draws','policy_a_stats',
        'policy_b_stats','execution_stats','elapsed_seconds_this_invocation')}
    summary.update(status='complete',evaluation_sha256=sha256(OUTPUT),
        validation='All 1000 swapped pairs, both checkpoint hashes, per-game total S1024 / N1, greedy zero search, and paired CI verified.')
    write_json(ART/'summary.json',summary)
    previous=read_json(ART/'plan.json')['comparison_evaluation']
    assert sha256(ROOT/previous['path'])==previous['sha256']
    n4=read_json(ROOT/previous['path'])
    assert raw['catalog_fingerprint']==n4['catalog_fingerprint']
    assert raw['checkpoint_identity']==n4['checkpoint_identity']
    n4_seeds,n4_scores=serialized_pair_scores(n4)
    difference=paired_bootstrap_difference(n4_seeds,n4_scores,seeds,scores,resamples=10000,seed=SEED+20001).to_dict()
    write_json(ART/'comparison_vs_s256n4.json',{'scope':'Same model/rules/ordered opening seeds; difference in scores against common greedy opponent, not direct N4 versus N1 play.',
        'direction':'S256N4 minus S1024N1','paired_difference':difference,
        's256n4_score':n4['win_rate_score'],'s1024n1_score':raw['win_rate_score'],
        's256n4_evaluation_sha256':previous['sha256'],'s1024n1_evaluation_sha256':sha256(OUTPUT)})
    write_json(ART/'paired_outcomes.json',{'game_seeds':seeds,'pair_scores':scores.tolist(),
        'pairs':[{'pair_id':p['pair_id'],'games':[{
            k:g[k] for k in ('first_player','outcome_for_a','status')} for g in p['games']]} for p in raw['pairs']]})
    (ART/'results.md').write_text(
        '# R20 S1024N1对Greedy\n\n'
        f'完成1000 pair／2000局；搜索方胜／和／负：{raw["a_wins"]}／{raw["draws"]}／{raw["a_losses"]}。\n\n'
        f'搜索方得分 **{raw["win_rate_score"]:.3%}**，按pair bootstrap的95%区间 '
        f'**[{ci["lower"]:.3%}, {ci["upper"]:.3%}]**；和棋半分。\n\n'
        f'异常{raw["exceptions"]}，非法行动{raw["illegal_actions"]}，工程截断{raw["engineering_truncations"]}，'
        f'不变量失败{raw["invariant_failures"]}，确定性循环判和{raw["policy_cycle_draws"]}。\n\n'
        '双方为相同第20轮原模型。native独立1个世界、总1024次；当前12贵族规则；'
        '无根噪声、T0、FP/PTP关闭、c_puct1.5。该结果衡量对同模型Greedy的搜索收益，尚未测试改作教师后的训练收益。\n')
    with (ART/'results.md').open('a') as f:
        f.write(f'\n同开局对共同Greedy对手的得分差，S256N4减S1024N1：{difference["estimate"]*100:+.3f}个百分点，'
            f'配对95%区间[{difference["lower"]*100:+.3f}, {difference["upper"]*100:+.3f}]；不是N4对N1直接胜率。\n')
    return summary


def run():
    config=prepare()
    assert torch.cuda.is_available(),'CUDA required; do not silently switch execution device.'
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        session=RUNTIME/str(time.time_ns())
        session.mkdir(parents=True,exist_ok=True)
        identity=read_json(ART/'plan.json')['source']
        shutil.copyfile(Path(__file__),session/'run.py')
        shutil.copyfile(CONFIG,session/'config.json')
        command=[sys.executable,'-u','-m','splendor.evaluation','--config',str(CONFIG),
            '--output',str(OUTPUT),'--progress-every','20','--summary-every','10']
        if OUTPUT.exists() or OUTPUT.with_suffix('.json.games').exists():command.append('--resume')
        execution={'start_unix':time.time(),'command':command,'pid':os.getpid(),
            'session':str(session.relative_to(ROOT)),'source':identity,'environment':environment_identity()}
        write_json(ART/'execution.json',execution)
        write_json(RUNTIME/'live.json',{'status':'running','pid':os.getpid(),'output':str(OUTPUT)})
        update_index('running')
        try:
            subprocess.run(command,cwd=ROOT,check=True)
            assert sha256(CHECKPOINT)==MODEL_SHA
            verify()
            summary=analyze()
        except BaseException as error:
            execution.update(status='failed',end_unix=time.time(),error=f'{type(error).__name__}: {error}')
            write_json(ART/'execution.json',execution)
            write_json(RUNTIME/'live.json',{'status':'failed','error':execution['error']})
            update_index('failed')
            raise
        execution.update(status='complete',end_unix=time.time())
        write_json(ART/'execution.json',execution)
        write_json(RUNTIME/'live.json',{'status':'complete','score':summary['win_rate_score']})
        update_index('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run','analyze'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.action=='prepare':
        prepare();update_index('prepared');print('Prepared 1000-pair S1024N1 total1024 versus same-model R20 greedy.')
    elif args.action=='analyze':
        analyze();update_index('complete')
    else:
        run()
