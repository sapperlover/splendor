import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('cache_rate_probe', Path(__file__).with_name('run.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def packed(width):
    floats = np.arange(2 + width * 2, dtype=np.float32)
    return dict(state_only=False, capacity=1, width=width, float_stride=len(floats),
        mask_stride=1+width, float_blob=floats, mask_blob=np.ones(1+width,dtype=np.uint8),
        float_fields=[('state',0,2,(2,)),('candidate_features',2,width*2,(width,2))],
        mask_fields=[('state_mask',0,1,(1,)),('candidate_mask',1,width,(width,))])


def test_key_ignores_padding_but_preserves_candidate_order_and_state():
    a,b=packed(2),packed(4)
    b['float_blob'][6:]=99; b['mask_blob'][3:]=0
    assert probe.row_payload(a,0,2)==probe.row_payload(b,0,2)
    b['float_blob'][2:6]=[4,5,2,3]
    assert probe.row_payload(a,0,2)!=probe.row_payload(b,0,2)
    b=packed(2); b['float_blob'][0]+=1
    assert probe.row_payload(a,0,2)!=probe.row_payload(b,0,2)
    assert probe.row_payload(a,0,1)!=probe.row_payload(a,0,2)


def test_duplicate_partition_and_namespace_isolation():
    s=probe.Shadow()
    assert s.observe(b'a',0,1,10)=='unique'
    assert s.observe(b'a',1,1,10)=='inflight_duplicate'
    s.completed([b'a'])
    assert s.observe(b'a',0,1,10)=='ready_hit'
    assert s.observe(b'b',1,1,10)=='unique'
    assert s.observe(b'b',1,1,10)=='inflight_duplicate'
    d=s.result()
    assert [d[k] for k in ('requests','unique_inputs','within_world_duplicates','cross_world_duplicates','ready_hit','inflight_duplicate')]==[5,2,2,1,1,2]
    assert probe.Shadow().observe(b'a',0,1,10)=='unique'


@pytest.mark.parametrize('backend,worlds', [('native',1),('native',4),('native_chance',1)])
def test_observer_keeps_real_native_results_and_counts(backend,worlds):
    from splendor.actor_critic import ActorCritic
    torch.set_num_threads(1); torch.manual_seed(45200000)
    catalog=probe.load_catalog(probe.BANK/'search_catalog')
    model=ActorCritic().eval()
    evaluator=probe.FrozenSearchInference(model,device='cpu')
    root=probe.selected_roots()[32]
    results=[]
    for observed in (False,True):
        cls=probe.NativePUCTSearchPolicy if backend=='native' else probe.NativeChancePUCTSearchPolicy
        policy=cls(model,evaluator=evaluator,catalog=catalog,
            config=probe.MCTSConfig(simulations=16,determinizations=worlds,seed=42100000,root_noise_fraction=0.))
        job=policy.start_job(probe.decode_request(root['request']),add_root_noise=False)
        runner_cls=probe.ObservedRunner if observed else probe.NativeBatchRunner
        runner=runner_cls(evaluator,catalog,capacity=16,buckets=probe.BUCKETS,slots=1)
        if observed:
            runner.shadows[job]=probe.Shadow();runner.labels[job]=[root['id'],42100000]
        while not job.done:
            assert runner.run([job])>0
        results.append(job.result.to_dict())
        if observed:
            d=runner.shadows[job].result()
            assert d['requests']==policy.stats.model_positions==runner.rows
            assert d['kind_counts']['0']==1
    assert results[0]==results[1]
