"""Reuse the frozen ABCD bank and metric definitions, changing only the search backend."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import read_json, write_json, sha256, source_identity
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy, NativeChancePUCTSearchPolicy, NativeBatchRunner
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime/entity_attention_chance_search_20261007'
BASE = ROOT / 'artifacts/entity_attention_target_noise_20261007'
spec = importlib.util.spec_from_file_location('frozen_abcd_metrics', BASE / 'probe.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ART = ART
base.RUNTIME = RUNTIME


def prepare():
    assert not (ART / 'manifest.json').exists()
    original = read_json(BASE / 'manifest.json')
    assert original['status'] == 'complete'
    shutil.copyfile(BASE / 'root_bank.json', ART / 'root_bank.json')
    source = source_identity(ROOT, RUNTIME / 'source')
    extra = [ROOT / 'src/splendor/native_search.cpp', ROOT / 'setup.py', *sorted((ROOT / 'data').glob('**/*'))]
    source['additional_files'] = {}
    for path in extra:
        if path.is_file():
            source['additional_files'][str(path.relative_to(ROOT))] = sha256(path)
            target = RUNTIME / 'source' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(path, target)
    manifest = {k: original[k] for k in ('checkpoint', 'total_train_step', 'completed_new_round', 'seeds',
                                        'search_config', 'positions', 'repeats', 'arms')}
    manifest.update(protocol='attention-sampled-chance-abcd-v1', status='prepared', client_date='2026-10-07',
        search_backend='native_chance', source=source, script=base.record(BASE / 'probe.py'),
        adapter_script=base.record(__file__), baseline_manifest=base.record(BASE / 'manifest.json'),
        inputs=[base.record(BASE / f) for f in ('root_bank.json', 'raw_targets.npz', 'summary.json', 'probe.py')],
        build=read_json(RUNTIME / 'build.json'), interpretation='Every simulation resamples root blind cards; future draws are sampled lazily without replacement. Decision Q/N is shared by acting-player observation history. A/B fix the complete sampling seed schedule, not one world; C/D vary it. Same checkpoint, bank, seeds, total S1024 and metric code as the baseline.',
        commands={c: f'conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u artifacts/entity_attention_chance_search_20261007/compare.py {c}'
                  for c in ('prepare', 'smoke', 'baseline_check', 'run', 'analyze', 'verify')})
    write_json(ART / 'manifest.json', manifest)
    print(json.dumps({'prepared': True, 'roots': 200, 'backend': 'native_chance'}), flush=True)


def baseline_check():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=read_json(ART/'manifest.json')
    model=load_frozen_actor_critic_checkpoint(Path(manifest['checkpoint']['path']),device='cuda')
    evaluator=FrozenSearchInference(model,device='cuda')
    requests=[decode_request(r['request']) for r in read_json(ART/'root_bank.json')['positions']]
    catalog=NativePUCTSearchPolicy(model,evaluator=evaluator).catalog
    runner=NativeBatchRunner(evaluator,catalog,capacity=200,buckets=(16,32,48,64),slots=1)
    started=time.perf_counter()
    with np.load(BASE/'raw_targets.npz',allow_pickle=False) as raw:
        for arm in ('A','B'):
            cfg=MCTSConfig(**dict(manifest['search_config'],seed=manifest['seeds']['world']))
            policy=NativePUCTSearchPolicy(model,evaluator=evaluator,config=cfg)
            jobs=[policy.start_job(r,add_root_noise=arm=='B',temperature=0,sampling_seed=manifest['seeds']['noise']) for r in requests]
            while any(not j.done for j in jobs):assert runner.run([j for j in jobs if not j.done])>0
            ai=base.ARMS.index(arm)
            for ri,job in enumerate(jobs):
                result=job.result;size=len(requests[ri].legal_actions)
                for field in ('policy_target','root_network_prior','root_used_prior','root_visit_counts'):
                    assert np.array_equal(getattr(result,field),raw[field][ai,0,ri,:size]),(arm,ri,field)
                assert result.chosen_candidate_index==raw['chosen_candidate_index'][ai,0,ri]
            print(json.dumps({'baseline_arm':arm,'roots':200,'bitwise_identical':True}),flush=True)
    write_json(ART/'baseline_equivalence.json',{'passed':True,'searches':400,'arms':['A','B'],'repeat':0,
        'target_prior_used_prior_counts_and_choices_bitwise_identical':True,'seconds':time.perf_counter()-started})


def compare_summary():
    old=read_json(BASE/'summary.json');new=read_json(ART/'summary.json')
    bank=read_json(ART/'root_bank.json')['positions']
    before=read_json(BASE/'per_root.json');after=read_json(ART/'per_root.json')
    keys=('pairwise_js','top1_agreement','top3_overlap','multiway_js','target_kl_to_network',
          'mean_target_kl_to_network','decisive_flip_fraction','network_top1_agreement')
    games=sorted(set(r['game_key'] for r in bank))
    indices=[np.asarray([i for i,r in enumerate(bank) if r['game_key']==g]) for g in games]
    rng=np.random.default_rng(base.SEEDS['bootstrap'])
    choices=rng.integers(len(games),size=(10000,len(games)))
    weights=np.stack([np.bincount(c,minlength=len(games)) for c in choices]).astype(float)
    denominator=weights@np.asarray([len(i) for i in indices])
    comparison={}
    for arm in base.ARMS:
        group={}
        for key in keys:
            a=np.asarray([r[key] for r in after if r['arm']==arm])
            b=np.asarray([r[key] for r in before if r['arm']==arm])
            sampled=weights@np.asarray([(a[i]-b[i]).sum() for i in indices])/denominator
            lo,hi=np.quantile(sampled,[.025,.975])
            group[key]={'baseline':float(b.mean()),'chance':float(a.mean()),'difference':float((a-b).mean()),
                        'ci95':[float(lo),float(hi)]}
        group['js_reduction_fraction']=1-group['pairwise_js']['chance']/group['pairwise_js']['baseline'] if arm!='A' else None
        group['randomness_fraction_of_mean_kl_before']=old['groups'][arm]['randomness_fraction_of_mean_kl']
        group['randomness_fraction_of_mean_kl_after']=new['groups'][arm]['randomness_fraction_of_mean_kl']
        comparison[arm]=group
    result={'protocol':'attention-chance-versus-determinized-fixed-targets-v1','groups':comparison,
        'bootstrap':{'clusters':len(games),'resamples':10000,'seed':base.SEEDS['bootstrap']},
        'search_seconds':{'baseline':read_json(BASE/'manifest.json')['search_seconds'],
                          'chance':read_json(ART/'manifest.json')['search_seconds']},
        'scope':'Same frozen checkpoint,200 roots andS1024 total simulations; seed-sensitive target diagnostics only. More stable/closer-to-prior targets are not strength proof.'}
    return result


def analyze():
    base.analyze()
    result=compare_summary();write_json(ART/'comparison.json',result)
    manifest=read_json(ART/'manifest.json')
    manifest['outputs'] += [base.record(ART/f) for f in ('comparison.json','baseline_equivalence.json')]
    write_json(ART/'manifest.json',manifest)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def verify():
    base.verify()
    assert compare_summary()==read_json(ART/'comparison.json')
    manifest=read_json(ART/'manifest.json')
    assert sha256(Path(__file__))==manifest['adapter_script']['sha256']
    assert sha256(ART/'root_bank.json')==sha256(BASE/'root_bank.json')
    assert read_json(ART/'baseline_equivalence.json')['passed']
    print(json.dumps({'comparison_recomputed':True,'same_frozen_bank_verified':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','smoke','baseline_check','run','analyze','verify'))
    args=parser.parse_args()
    if args.command in ('run','smoke'):
        m=read_json(ART/'manifest.json');assert sha256(Path(__file__))==m['adapter_script']['sha256']
        base.NativePUCTSearchPolicy=NativeChancePUCTSearchPolicy
        base.run(smoke=args.command=='smoke')
    else:globals()[args.command]()
