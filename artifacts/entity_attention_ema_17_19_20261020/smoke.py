"""Actual-data metric and native-root checks; never a strength result."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from splendor.alphazero_recording import read_json, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.compact_game_cache import CompactGameCache
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.offline_train import make_index
from splendor.benchmark_search import fixed_requests
from splendor.catalog import load_default_catalog
from splendor.search_inference import FrozenSearchInference, InferenceConfig

ART = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ema_runner_smoke', ART/'run.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def main():
    torch.set_num_threads(1)
    plan = read_json(probe.DEFAULT_OUTPUT/'plan.json')
    models = probe.build_emas(plan,probe.DEFAULT_OUTPUT,ART)
    p = next(g for g in plan['panels'] if g['iteration']==19)
    index = make_index({'games':p['games']},256*1024**2)
    index.cache = CompactGameCache(256*1024**2)
    by_key = {g.key:g for g in index.games}
    refs = p['references'][:64]
    samples = [index.cache.sample(by_key[(r['round_index'],r['game_id'])],r['decision_index']) for r in refs]
    models_to_check = [{'half_life_steps':None,**p['raw_checkpoint']}] + [m for m in models['models'] if m['iteration']==19]
    cases = fixed_requests()
    checks=[]
    for source in models_to_check:
        model=load_frozen_actor_critic_checkpoint(probe.checked(source),device='cpu')
        actual,_=probe.evaluate_samples(model,samples,np.arange(64),64,32)
        reference=AlphazeroLearner.evaluate_samples(SimpleNamespace(model=model,device=torch.device('cpu'),
            config=LearnerConfig(device='cpu',validation_batch_size=32)),samples)
        differences={key:actual[key]-reference[old] for key,old in
            (('policy_kl','policy_kl'),('policy_ce','policy_loss'),('value_mse','value_loss'))}
        assert max(map(abs,differences.values()))<5e-5
        evaluator=FrozenSearchInference(model,device='cpu',config=InferenceConfig(actor_projection=False))
        policies=[NativePUCTSearchPolicy(model,device='cpu',evaluator=evaluator,
            config=MCTSConfig(simulations=2,determinizations=1,root_noise_fraction=0.,seed=39700000+i))
            for i in range(len(cases))]
        jobs=[policy.start_job(request,add_root_noise=False,temperature=0.)
              for policy,(_,request,_) in zip(policies,cases)]
        runner=NativeBatchRunner(evaluator,load_default_catalog(),capacity=4,buckets=(16,32,48,64))
        while not all(job.done for job in jobs):
            assert runner.run([job for job in jobs if not job.done])>0
        for job,(_,request,_) in zip(jobs,cases):
            assert job.result.simulations==(2 if len(request.legal_actions)>1 else 0)
            assert 0<=job.result.chosen_candidate_index<len(request.legal_actions)
        checks.append({'half_life_steps':source['half_life_steps'],'positions':64,
            'metric_differences':differences,'native_s2_roots':len(jobs)})
    record={'passed':True,'scope':'Engineering only;64 real validation samples per model and3 native S2 roots; no optimizer updates or strength games',
        'models':checks,'plan_digest':plan['digest'],'validation_reference_digest':p['reference_digest'],
        'runner_sha256':probe.sha256(ART/'run.py'),'smoke_sha256':probe.sha256(Path(__file__))}
    write_json(ART/'smoke.json',record)
    print({'passed':True,'models':len(checks),'native_roots':sum(c['native_s2_roots'] for c in checks)})


if __name__=='__main__':
    main()
