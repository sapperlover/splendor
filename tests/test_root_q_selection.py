from dataclasses import asdict
import pytest
import torch

from splendor.actor_critic import ActorCritic, MODEL_WEIGHTS_FORMAT_VERSION
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig, MCTSSearchResult
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.parallel_recommend import BatchedWorldSearch, WorldParallelSearch
from splendor.recommend import LocalAdvisor
from splendor.root_q_selection import (RootQSettings, root_coverage_shortlist,
    select_root_by_world_mean_q)
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.replay import RULESET_ID


def make_result(visits,values,prior=None):
    width=len(visits[0]);prior=prior or tuple(1/width for _ in range(width))
    counts=tuple(sum(row[i] for row in visits) for i in range(width))
    budgets=tuple(sum(row) for row in visits);total=sum(budgets)
    chosen=max(range(width),key=lambda i:(counts[i],prior[i],-i))
    return MCTSSearchResult(chosen,True,prior,prior,budgets,tuple(visits),counts,
        tuple(n/total for n in counts),tuple(float(i==chosen) for i in range(width)),total,3,0,
        tuple(visits),tuple(values),tuple((0,)*width for _ in visits))


def test_equal_world_q_can_disagree_with_both_visits_and_pooled_q():
    r=make_result(((90,10),(10,90)),((72.,3.),(-9.,27.)),(.6,.4))
    assert r.chosen_candidate_index==0
    assert sum(row[0] for row in r.per_world_root_value_sums)/r.root_visit_counts[0]>.3
    selected=select_root_by_world_mean_q(r,RootQSettings(10,2),(0,1),())
    assert selected.world_mean_q==pytest.approx((-.05,.3))
    assert selected.chosen_candidate_index==1
    assert selected.behavior_policy==(0.,1.)
    assert selected.root_visit_counts==r.root_visit_counts
    assert selected.simulations==200


def test_recommendation_can_overrule_visit_winner_and_show_equal_q(catalog):
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    request=GameEngine.new(17).decision();width=len(request.legal_actions)
    visits=((90,10)+(0,)*(width-2),(10,90)+(0,)*(width-2))
    sums=((72.,5.)+(0.,)*(width-2),(-9.,9.)+(0.,)*(width-2))
    result=make_result(visits,sums,(.6,.4)+(0.,)*(width-2))
    selected=select_root_by_world_mean_q(result,RootQSettings(10,2),(0,1),())
    advisor=object.__new__(LocalAdvisor)
    advisor.catalog=catalog;advisor.mode='mcts';advisor.checkpoint=Path('test-only.pt');advisor.checkpoint_sha256='test-only'
    advisor.evaluator=SimpleNamespace(policy_value=lambda requests:[(np.zeros(width),.1)])
    advisor.search_policy=SimpleNamespace(search=lambda request,**kwargs:selected)
    output=advisor.recommend(request)
    assert output['recommended_candidate_index']==1
    assert output['candidates'][0]['candidate_index']==1
    assert output['root_q_selection']['visit_candidate_index']==0
    assert output['root_q_selection']['changed_from_visit_choice'] is True
    assert output['candidates'][0]['world_mean_q_actor']==pytest.approx(.3)
    assert output['candidates'][0]['search_q_actor']==pytest.approx(.14)
    assert output['estimated_win_rate_actor']==pytest.approx(.65)
    assert output['win_rate_basis']=='world_mean_q_actor'
    assert output['candidates'][1]['search_q_actor']==pytest.approx(.63)
    assert output['candidates'][1]['world_mean_q_actor']==pytest.approx(-.05)


def test_floor_is_per_world_and_missing_world_is_not_zero_or_skipped():
    r=make_result(((1000,10),(0,10)),((900.,1.),(0.,1.)))
    selected=select_root_by_world_mean_q(r,RootQSettings(10,1),(1,),())
    assert selected.world_mean_q==(None,.1)
    assert selected.eligible_root_candidates==(1,)
    assert selected.chosen_candidate_index==1
    with pytest.raises(ValueError,match='coverage'):
        select_root_by_world_mean_q(r,RootQSettings(10,1),(0,),())


def test_naturally_qualified_actions_outside_shortlist_can_win():
    r=make_result(((30,20,10),(30,20,10)),((0.,2.,8.),(0.,2.,8.)))
    selected=select_root_by_world_mean_q(r,RootQSettings(10,1),(0,),())
    assert selected.chosen_candidate_index==2
    assert selected.eligible_root_candidates==(0,1,2)


def test_shortlist_is_global_and_budget_reservation_is_explicit():
    settings=RootQSettings(32,8)
    assert settings.initial_limit(4096,30)==3840
    assert settings.initial_limit(4096,2)==4032
    with pytest.raises(ValueError,match='预算'):
        settings.initial_limit(128,30)
    assert root_coverage_shortlist(((100,0,3),(0,120,3)),(.6,.2,.2),RootQSettings(2,2))==(1,0)


def drive_native(core,request,stop):
    while not stop():
        pending=core.pending()
        assert pending
        for row in pending:
            width=0 if row['kind']=='value_only' else len(row['encoded']['candidate_features'])
            core.submit(row['world'],(0.,)*width,.1)
            if row['world']==-1:core.initialize_worlds(request)


def test_native_pause_and_coverage_reuse_trees_and_exact_budget(catalog):
    from splendor import _native_search as native
    request=GameEngine.new(42).decision()
    core=native.SearchJob(native.Catalog(catalog),request,MCTSConfig(simulations=40,determinizations=2),42,0,False,0.)
    core.set_root_search_limit(14)
    drive_native(core,request,core.root_phase_complete)
    assert not core.done()
    before=core.root_statistics()
    assert [sum(row) for row in before['visits']]==[14,14]
    with pytest.raises(ValueError,match='remaining world budget'):
        core.resume_with_root_coverage([0,29],20)
    with pytest.raises(ValueError,match='duplicated'):
        core.resume_with_root_coverage([0,0],1)
    core.resume_with_root_coverage([0,29],3)
    drive_native(core,request,core.done)
    result=core.result()
    assert result['simulations']==40
    assert [sum(row) for row in result['per_world_root_visit_counts']]==[20,20]
    assert all(row[0]>=3 and row[29]>=3 for row in result['per_world_root_visit_counts'])
    assert all(a<=b for old,new in zip(before['visits'],result['per_world_root_visit_counts']) for a,b in zip(old,new))


@pytest.fixture
def small_advisor(tmp_path):
    torch.manual_seed(13);torch.set_num_threads(1)
    model=ActorCritic().eval()
    path=tmp_path/'model.pt'
    torch.save({'checkpoint_schema':'splendor-frozen-actor-critic-v1',
        'format_version':MODEL_WEIGHTS_FORMAT_VERSION,'feature_schema_version':FEATURE_SCHEMA_VERSION,
        'config':asdict(model.config),'state_dict':model.state_dict(),
        'metadata':{'rules_version':RULESET_ID}},path)
    return LocalAdvisor(path)


def assert_coverage(result,worlds,budget,minimum):
    assert result.simulations==worlds*budget
    assert len(result.per_world_root_visit_counts)==worlds
    assert all(sum(row)==budget for row in result.per_world_root_visit_counts)
    assert all(row[c]>=minimum for c in result.coverage_candidates for row in result.per_world_root_visit_counts)
    assert result.chosen_candidate_index in result.eligible_root_candidates
    assert result.world_mean_q[result.chosen_candidate_index]==max(result.world_mean_q[i] for i in result.eligible_root_candidates)


def test_batched_advisor_ranks_by_equal_q_and_preserves_diagnostics(small_advisor):
    request=GameEngine.new(17).decision()
    search=BatchedWorldSearch(small_advisor,simulations_per_world=24,worlds=4,q_settings=RootQSettings(3,3))
    result=search.search(request);assert_coverage(result,4,24,3)
    small_advisor.search_policy=search;small_advisor.mode='mcts'
    output=small_advisor.recommend(request,top_k=5)
    assert output['selection_rule']=='world_mean_q'
    assert output['recommended_candidate_index']==output['candidates'][0]['candidate_index']
    assert output['candidates'][0]['world_mean_q_actor']==max(q for i,q in enumerate(output['root_q_selection']['world_mean_q']) if i in output['root_q_selection']['eligible_candidates'])
    assert len(output['per_world_root_value_sums'])==4
    assert output['root_q_selection']['minimum_visits_per_world']==3


def test_parallel_q_barrier_persists_trees_across_two_roots(small_advisor):
    search=WorldParallelSearch(small_advisor.checkpoint,simulations_per_world=24,worlds=4,workers=2,
                               q_settings=RootQSettings(3,3))
    try:
        for seed in (17,19):
            result=search.search(GameEngine.new(seed).decision())
            assert_coverage(result,4,24,3)
    finally:search.close()


@pytest.mark.parametrize('phase',['payment','return_tokens'])
def test_q_search_uses_native_payment_and_return_candidates(small_advisor,phase):
    import json
    from pathlib import Path
    from splendor.actions import BuyMarket, BuyReserved
    from splendor.kirameki_live import request_from_snapshot
    from splendor.observation import build_observation
    from splendor.rules import apply_legal_action
    from splendor.search_policy import _determinized_state, request_from_observation
    cases=json.loads((Path(__file__).parent/'fixtures/kirameki_public_cases.json').read_text())
    request=None
    for snapshot in cases:
        root=request_from_snapshot(snapshot)
        if phase=='return_tokens' and root.observation.phase.value==phase:
            request=root;break
        if phase=='payment' and root.observation.phase.value=='main_action':
            for action in root.legal_actions:
                if not isinstance(action,(BuyMarket,BuyReserved)):continue
                state=apply_legal_action(_determinized_state(root.observation,small_advisor.catalog,seed=0),action,small_advisor.catalog)
                follow=request_from_observation(build_observation(state,0,small_advisor.catalog),catalog=small_advisor.catalog)
                if len(follow.legal_actions)>1:request=follow;break
        if request is not None:break
    assert request is not None and request.observation.phase.value==phase
    result=BatchedWorldSearch(small_advisor,simulations_per_world=24,worlds=4,q_settings=RootQSettings(3,3)).search(request)
    assert_coverage(result,4,24,3)


def test_legacy_visit_mode_is_unchanged(small_advisor):
    request=GameEngine.new(17).decision()
    reference=NativePUCTSearchPolicy(small_advisor.model,evaluator=small_advisor.evaluator,
        config=MCTSConfig(simulations=96,determinizations=4,root_noise_fraction=0.))
    expected=reference.search(request)
    actual=BatchedWorldSearch(small_advisor,simulations_per_world=24,worlds=4).search(request)
    assert actual.root_visit_counts==expected.root_visit_counts
    assert actual.chosen_candidate_index==expected.chosen_candidate_index


def test_interactive_chance_backend_uses_one_shared_tree_and_saves_sampling_diagnostics(small_advisor):
    from splendor.native_policy import NativeChancePUCTSearchPolicy
    request=GameEngine.new(17).decision()
    reference=NativeChancePUCTSearchPolicy(small_advisor.model,evaluator=small_advisor.evaluator,
        config=MCTSConfig(simulations=33,determinizations=1,seed=91,root_noise_fraction=0.))
    expected=reference.search(request)
    search=BatchedWorldSearch(small_advisor,simulations_per_world=33,worlds=1,seed=91,
                              search_backend='native_chance')
    assert search.policy.stochastic and search.policy.chance_widening is None
    actual=search.search(request)
    assert actual.root_visit_counts==expected.root_visit_counts
    assert actual.chosen_candidate_index==expected.chosen_candidate_index
    assert actual.simulations==33 and len(actual.per_world_root_visit_counts)==1
    small_advisor.search_policy=search;small_advisor.mode='mcts'
    output=small_advisor.recommend(request)
    assert output['search_backend']=='native_chance' and output['selection_rule']=='visits'
    assert 'root_q_selection' not in output
    assert output['search_diagnostics']['chance']['root_hidden_samples']==33
    assert output['search_diagnostics']['chance']['chance_draws']>0
    assert output['search_diagnostics']['stats']['simulations']==33
    assert sum(output['per_world_root_visit_counts'][0])==33
    chosen=output['recommended_candidate_index']
    q=output['per_world_root_value_sums'][0][chosen]/output['per_world_root_visit_counts'][0][chosen]
    assert output['estimated_win_rate_actor']==pytest.approx((q+1)/2)
    assert output['win_rate_basis']=='search_q_actor'


def test_interactive_chance_rejects_q_coverage_and_independent_worlds(small_advisor):
    with pytest.raises(ValueError,match='world_mean_q'):
        BatchedWorldSearch(small_advisor,simulations_per_world=24,worlds=1,
                           search_backend='native_chance',q_settings=RootQSettings())
    with pytest.raises(ValueError,match='determinizations=1'):
        BatchedWorldSearch(small_advisor,simulations_per_world=24,worlds=2,search_backend='native_chance')


def test_single_candidate_keeps_zero_search_and_does_not_require_coverage(small_advisor):
    from splendor.benchmark_search import micro_requests
    single=next(r for r in micro_requests() if len(r.legal_actions)==1)
    search=BatchedWorldSearch(small_advisor,simulations_per_world=1,worlds=1,q_settings=RootQSettings())
    result=search.search(single)
    assert result.simulations==0
    assert result.chosen_candidate_index==0
    small_advisor.search_policy=search;small_advisor.mode='mcts'
    output=small_advisor.recommend(single)
    assert output['win_rate_basis']=='value_actor_raw'
    assert output['estimated_win_rate_actor']==pytest.approx((output['value_actor_raw']+1)/2)
