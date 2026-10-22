import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.native_policy import NativeBatchRunner
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference,InferenceConfig

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'artifacts/entity_attention_r43_action_ranking_64_s1024_20261015/run.py'
spec=importlib.util.spec_from_file_location('action_ranking_s1024_probe',PATH)
probe=importlib.util.module_from_spec(spec);sys.modules[spec.name]=probe;spec.loader.exec_module(probe)


def test_plan_changes_only_continuation_recipe(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'ART',tmp_path)
    monkeypatch.setattr(probe.base,'ART',tmp_path)
    previous=probe.read_json(probe.PARENT/'plan.json')
    probe.prepare();plan=probe.read_json(tmp_path/'plan.json')
    for key in ('checkpoint','bank','search_evidence','search_producer','roots','comparisons','initial_worlds','expanded_worlds','seeds',
                'uncertainty','primary_methods','auxiliary_methods','max_completed_player_turns','censoring','active_arms','device','torch_threads','inference'):
        assert plan[key]==previous[key]
    assert plan['continuation_search']['simulations']==1024
    assert plan['continuation_search']['determinizations']==1
    assert plan['continuation_search']['root_noise_fraction']==0


def test_new_search_randomness_does_not_change_paired_hidden_worlds():
    for root in (0,15,63):
        for world in (0,31,127):
            assert probe.base.world_seed(root,world)==probe.derive_seed(38200000,0,root,'paired_action_world',world)
    assert probe.policy_seed(3,4,0)==probe.policy_seed(3,4,0)
    assert len({probe.policy_seed(3,w,a) for w in range(128) for a in (0,1)})==256


def test_forced_action_prefix_matches_greedy_reference():
    request=GameEngine.new(42,first_player=1,mode='fast').decision()
    root={'id':7,'request':to_primitive(request)};comparison={'a':0,'b':0}
    catalog=load_default_catalog();model=ActorCritic()
    evaluator=FrozenSearchInference(model,device='cpu',config=InferenceConfig())
    search={'simulations':8,'determinizations':1,'root_noise_fraction':0.,'seed':0}
    a=probe.SearchArm(root,comparison,3,0,model,evaluator,search)
    b=probe.base.Arm(root,comparison,3,1,catalog)
    assert a.initial_digest==b.initial_digest and a.actions==b.actions
    a.set_child_value(.4)
    assert a.child_value==pytest.approx(.4 if a.engine.current_player==0 else -.4)


def test_each_nonforced_continuation_decision_uses_full_configured_search_budget():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        request=GameEngine.new(17,first_player=0,mode='fast').decision()
        root={'id':1,'request':to_primitive(request)};comparison={'a':0,'b':0}
        model=ActorCritic();evaluator=FrozenSearchInference(model,device='cpu',config=InferenceConfig())
        search={'simulations':8,'determinizations':1,'root_noise_fraction':0.,'seed':0}
        arm=probe.SearchArm(root,comparison,0,0,model,evaluator,search)
        arm.set_child_value(0.);arm.ensure_job()
        runner=NativeBatchRunner(evaluator,load_default_catalog(),capacity=2,buckets=(16,32,48,64))
        while not arm.job.done:assert runner.run([arm.job])>0
        prefix=len(arm.actions);arm.consume()
        assert arm.simulations==8 and arm.determinizations==1
        assert arm.search_decision_indices==[prefix]
        assert len(arm.actions)==prefix+1
    finally:torch.set_num_threads(previous)
