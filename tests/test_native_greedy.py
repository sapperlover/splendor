"""Root-only native jobs must remain exact logit argmax with no tree work."""
import numpy as np
import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.engine import GameEngine
from splendor.native_policy import NativeBatchRunner, NativeGreedyPolicy
from splendor.neural_policy import GreedyActorCriticPolicy
from splendor.search_inference import FrozenSearchInference, InferenceConfig


def test_native_greedy_root_logits_ties_and_no_world_initialization():
    request=GameEngine.new(91,first_player=0,mode='checked').decision()
    policy=NativeGreedyPolicy(ActorCritic())
    for chosen in (0,3):
        job=policy.start_job(request)
        pending=job.pending_requests()
        assert len(pending)==1 and pending[0].world_index==-1
        logits=[-4.]*len(request.legal_actions)
        logits[chosen]=logits[chosen+1]=1.
        job.submit(pending[0].request_id,tuple(logits),.7)
        assert job.done and job.result.chosen_candidate_index==chosen
        assert job.result.simulations==0 and not job.result.searchable
        assert not job.result.simulation_budgets and sum(job.result.root_visit_counts)==0
        assert not job.pending_requests()
        assert job.core.chance_diagnostics()['decision_nodes']==0
    assert policy.stats.simulations==policy.stats.determinizations==policy.stats.state_transitions==0
    assert policy.stats.searchable_decisions==0 and policy.live_simulations()==0


def test_native_greedy_batch_matches_eager_policy():
    torch.manual_seed(32);torch.set_num_threads(1)
    model=ActorCritic().eval();evaluator=FrozenSearchInference(model,config=InferenceConfig(mode='eager'))
    policy=NativeGreedyPolicy(model,evaluator=evaluator)
    requests=[GameEngine.new(s,first_player=s%2,mode='checked').decision() for s in range(8)]
    jobs=[policy.start_job(r) for r in requests]
    runner=NativeBatchRunner(evaluator,policy.catalog,capacity=8,buckets=(16,32,48,64))
    assert runner.run(jobs)==8
    eager=GreedyActorCriticPolicy(model)
    for job,request in zip(jobs,requests,strict=True):
        assert job.done
        assert request.legal_actions[job.result.chosen_candidate_index]==eager.choose(request)
        assert job.result.simulations==0
    assert evaluator.stats.positions==8 and runner.root_rows==8


def test_native_greedy_rejects_search_randomness():
    policy=NativeGreedyPolicy(ActorCritic())
    request=GameEngine.new(17,first_player=0,mode='checked').decision()
    with pytest.raises(ValueError,match='greedy root inference'):
        policy.start_job(request,add_root_noise=True)
    with pytest.raises(ValueError,match='greedy root inference'):
        policy.start_job(request,temperature=1.)
