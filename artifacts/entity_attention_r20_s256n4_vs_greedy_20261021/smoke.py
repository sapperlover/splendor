"""Finite real-model GPU checks, excluded from the 1000-pair arena."""
import copy
from dataclasses import asdict
import gc
import time

import torch

from run import ART,ROOT,RUNTIME,CHECKPOINT,MODEL_SHA,SEED,prepare
from splendor.alphazero_recording import environment_identity,read_json,sha256,write_json
from splendor.compact_trace import decode_request
from splendor.evaluation import FixedEvaluationConfig,run_fixed_evaluation
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner,NativeGreedyPolicy,NativePUCTSearchPolicy
from splendor.neural_policy import GreedyActorCriticPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference,InferenceConfig


def main():
    assert torch.cuda.is_available() and sha256(CHECKPOINT)==MODEL_SHA
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=prepare();started=time.perf_counter()
    model=load_frozen_actor_critic_checkpoint(CHECKPOINT,device='cuda')
    evaluator=FrozenSearchInference(model,device='cuda',config=InferenceConfig(**raw['policy_a']['inference']))
    greedy=NativeGreedyPolicy(model,evaluator=evaluator)
    bank=read_json(ROOT/'artifacts/entity_attention_r20_search_stability_128_20261021/root_bank.json')['roots']
    requests=[decode_request(r['request']) for r in bank]
    eager=GreedyActorCriticPolicy(model,device='cuda')
    expected=[r.legal_actions.index(eager.choose(r)) for r in requests]
    jobs=[greedy.start_job(r) for r in requests]
    runner=NativeBatchRunner(evaluator,greedy.catalog,capacity=500,buckets=(16,32,48,64),slots=1)
    assert runner.run(jobs)==128
    assert all(j.done and j.result.chosen_candidate_index==i for j,i in zip(jobs,expected,strict=True))
    assert greedy.stats.simulations==greedy.stats.determinizations==greedy.stats.state_transitions==0
    policy=NativePUCTSearchPolicy(model,evaluator=evaluator,
        config=MCTSConfig(simulations=1024,determinizations=4,seed=SEED+90000,root_noise_fraction=0.))
    job=policy.start_job(requests[0],add_root_noise=False,temperature=0.)
    while not job.done:assert runner.run([job])>0
    result=asdict(job.result)
    assert result['simulation_budgets']==[256]*4 and result['simulations']==1024
    assert sum(result['root_visit_counts'])==1024 and result['transition_cap_hits']==0
    del job,jobs,policy,greedy,eager,runner,evaluator,model
    gc.collect();torch.cuda.empty_cache()
    config=copy.deepcopy(raw)
    config['game_seeds']=list(range(SEED+91000,SEED+91006))
    config['policy_a']['search_config']['simulations']=8
    config['execution']['total_active_games']=12
    config['bootstrap_resamples']=100
    write_json(ART/'smoke_config.json',config)
    out=RUNTIME/'smoke_evaluation.json'
    evaluation=run_fixed_evaluation(FixedEvaluationConfig.from_dict(config),out,
        resume=out.exists(),progress_every=6,summary_every=6).summary
    assert evaluation['passed'] and evaluation['completed_pairs']==6 and evaluation['valid_games']==12
    assert evaluation['execution_stats']['actual_search_workers']==12
    assert evaluation['execution_stats']['owner_pipeline_slots']==1
    assert all(not w['cuda_initialized'] for w in evaluation['execution_stats']['workers'])
    for pair in evaluation['pairs']:
        for game in pair['games']:
            a,b=game['policy_a_stats'],game['policy_b_stats']
            assert a['simulations']==8*a['searchable_decisions']
            assert a['determinizations']==4*a['searchable_decisions']
            assert b['simulations']==b['determinizations']==b['state_transitions']==0
    write_json(ART/'smoke.json',dict(passed=True,
        scope='Engineering only: 128 greedy roots versus GPU eager, one full-budget S256N4 root, six small-budget pairs through all 12 workers; excluded from formal arena.',
        greedy_matches_eager=128,full_budget_root=result,
        six_pair_smoke=dict(completed_pairs=6,valid_games=12,passed=True,actual_workers=12,pipeline_slots=1,
                           evaluation=str(out.relative_to(ROOT)),sha256=sha256(out)),
        elapsed_seconds=time.perf_counter()-started,environment=environment_identity()))
    print('CUDA engineering checks passed; 128 greedy choices matched, S256N4 allocation verified, 12-worker mixed arena completed.',flush=True)


if __name__=='__main__':main()
