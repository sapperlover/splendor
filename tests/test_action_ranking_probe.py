import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import encode_decision
from splendor.native_policy import _extension,_native_catalog
from splendor.replay import state_digest
from splendor.search_policy import determinize_request
from splendor.replay import to_primitive

DIRECTORY=Path(__file__).resolve().parents[1]/'artifacts/entity_attention_r43_action_ranking_64_20261015'
sys.path.insert(0,str(DIRECTORY))
spec=importlib.util.spec_from_file_location('action_ranking_probe',DIRECTORY/'run.py')
probe=importlib.util.module_from_spec(spec);sys.modules[spec.name]=probe;spec.loader.exec_module(probe)
sys.path.remove(str(DIRECTORY))


def test_value_sign_and_native_clipping():
    assert probe.root_value(1.2,0)==1
    assert probe.root_value(.7,1)==-.7
    assert probe.root_value(-1.2,1)==1


def test_paired_seeds_unique_and_deterministic():
    seeds=[probe.world_seed(i,w) for i in range(64) for w in range(128)]
    assert len(set(seeds))==8192
    assert probe.world_seed(2,7)==probe.world_seed(2,7)


def test_native_batch_padding_preserves_features():
    catalog=load_default_catalog();engine=GameEngine.new(17,first_player=0,mode='fast')
    game=_extension().NativeGame(_native_catalog(catalog),engine._debug_state())
    request=engine.decision();native=game.encoded();batch=probe.collate_native([native,native])
    encoded=encode_decision(request)
    for name in probe.STATE_FIELDS:
        np.testing.assert_allclose(batch[name][0],getattr(encoded,name),rtol=0,atol=1e-6)
    np.testing.assert_allclose(batch['candidate_features'][0],encoded.candidate_features,rtol=0,atol=1e-6)
    assert batch['candidate_mask'].all()


def test_paired_arms_start_identical_and_forced_child_matches_python():
    request=GameEngine.new(42,first_player=1,mode='fast').decision()
    root={'id':7,'request':to_primitive(request)};comparison={'a':0,'b':0};catalog=load_default_catalog()
    a=probe.Arm(root,comparison,3,0,catalog);b=probe.Arm(root,comparison,3,1,catalog)
    assert a.initial_digest==b.initial_digest and a.actions==b.actions
    engine=determinize_request(request,seed=probe.world_seed(7,3),engine_mode='fast')
    response,forced=probe.child(engine,0)
    assert a.actions==[0]+forced
    if isinstance(response,DecisionRequest):
        native=a.game.encoded();encoded=encode_decision(response)
        for name in probe.STATE_FIELDS:np.testing.assert_allclose(native[name],getattr(encoded,name),rtol=0,atol=1e-6)


def test_zero_observed_differences_still_uncertain():
    p=probe.posterior([0]*32,1,32)
    assert not p['resolved']
    assert p['credible_interval95'][0]<0<p['credible_interval95'][1]
    assert p['delta_complete_pairs']==0


def test_strong_paired_advantage_resolves_and_censoring_widens_bounds():
    p=probe.posterior([2]*32,1,32)
    assert p['resolved'] and p['p_a_better']>.99
    censored=probe.posterior([2]*10,1,32,missing=22)
    assert not censored['resolved'] and censored['p_a_better'] is None
    assert censored['delta_worst_case_bounds']==[-.75,2.]


def test_metric_ties_regret_orientation_and_controls_excluded():
    comparisons=[{'same_action_control':False,'v_search_world':[.8,0],'visits':[700,300]},
                 {'same_action_control':False,'v_search_world':[-.8,0],'visits':[700,300]},
                 {'same_action_control':True,'v_search_world':[0,0],'visits':[1,1]}]
    stats=[{'delta_complete_pairs':.4,'missing_pairs':0,'resolved':True,'p_a_better':.9,'expected_regret_a':.01,'expected_regret_b':.4},
           {'delta_complete_pairs':.2,'missing_pairs':0,'resolved':True,'p_a_better':.8,'expected_regret_a':.02,'expected_regret_b':.2},
           {'delta_complete_pairs':0,'missing_pairs':0,'resolved':True}]
    m=probe.metrics(comparisons,stats,'V_same_search_world')
    assert m['cases']==2 and m['empirical_accuracy_ties_half']==.5
    assert m['mean_empirical_regret_z']==pytest.approx(.1)
    assert m['mean_empirical_regret_score_pp']==pytest.approx(5)
    assert m['delta_correlation']==pytest.approx(1)
    mcts=probe.metrics(comparisons,stats,'MCTS_visit_choice')
    assert mcts['empirical_accuracy_ties_half']==1 and mcts['mean_empirical_regret_z']==0
