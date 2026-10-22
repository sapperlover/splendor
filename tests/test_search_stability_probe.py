"""Metric/protocol guards for the user-requested fixed-root search experiment."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

FILE=Path(__file__).resolve().parents[1]/'artifacts/entity_attention_r20_search_stability_128_20261021/run.py'
spec=importlib.util.spec_from_file_location('search_stability_probe',FILE)
probe=importlib.util.module_from_spec(spec);sys.modules[spec.name]=probe;spec.loader.exec_module(probe)


def row(p, chosen, visits=None, sums=None):
    return dict(policy=p,chosen=chosen,visits=visits,
                world_visits=[] if visits is None else [visits],
                world_value_sums=[] if sums is None else [sums])


def test_user_budget_convention_and_seed_split():
    panels={p['name']:p for p in probe.PANELS}
    assert panels['S256N4']['total_simulations']==1024
    assert panels['S1024N4']['total_simulations']==4096
    assert panels['reference_S16384N4']['total_simulations']==65536
    assert panels['reference_chance_S65536']['total_simulations']==65536
    assert not set(probe.SCREEN_SEEDS)&{probe.SEED_BASE+i for i in probe.REPEATS}
    a=set(probe.repeat_seeds(panels['S256N4']))
    b=set(probe.repeat_seeds(panels['reference_S16384N4']))
    c=set(probe.repeat_seeds(panels['reference_chance_S65536']))
    assert not a&b and not a&c and not b&c


def test_js_with_zero_mass_and_known_extremes():
    assert probe.js([1,0],[1,0])==0
    assert probe.js([1,0],[0,1])==pytest.approx(np.log(2))
    assert probe.js([.8,.2],[.2,.8])==pytest.approx(probe.js([.2,.8],[.8,.2]))


def test_pairwise_agreement_differs_from_modal_fraction():
    rows=[row([1,0],0)]*4+[row([0,1],1)]*4
    stats=probe.seed_stability(rows)
    assert stats['top1_pairwise_agreement']==pytest.approx(12/28)
    assert stats['top1_modal_fraction']==.5
    assert not stats['unanimous']
    assert stats['policy_pairwise_js']==pytest.approx(16/28*np.log(2))


def test_reference_q_averages_worlds_equally_and_checks_coverage():
    rows=[row([.8,.2],0,[100,20],[100,0]),row([.8,.2],0,[20,100],[0,100])]
    ref=probe.reference(rows)
    assert ref['q']==[.5,.5]  # pooled W/N would yield .8333 for each, a different estimand
    assert ref['policy_top2_gap_score_pp']==0
    rows[1]['world_visits'][0][0]=15
    ref=probe.reference(rows)
    assert ref['q'][0] is None and ref['policy_top2_gap_score_pp'] is None


def test_coverage_zero_visits_greedy_na_and_signed_q_regret():
    ref=dict(policy=[.6,.3,.1],chosen=0,q=[.4,.6,None],policy_top2_gap_score_pp=-10.)
    # Zero-mass candidates must not count as covered through arbitrary ties.
    result=probe.compare([row([0,1,0],1,[0,128,0])],ref)
    assert result['reference_top1_hit']==0
    assert result['reference_action_zero_visit']==1
    assert result['reference_top1_in_method_top3']==0
    assert result['reference_topk_set_coverage_3']==pytest.approx(1/3)
    assert result['q_regret_score_pp']==pytest.approx(-10)
    greedy=probe.compare([row([.6,.3,.1],0)],ref)
    assert greedy['reference_action_zero_visit'] is None
    assert greedy['q_regret_seed_coverage']==1


def test_old_game_replay_and_conditional_new_catalog_equivalence():
    from splendor.catalog import load_catalog,load_default_catalog
    from splendor.search_policy import determinize_request
    from splendor.selfplay_data import load_game
    old=load_catalog(probe.ROOT/'artifacts/noble_pool_bga12_20261021/previous_catalog')
    path=probe.RUN/'rounds/round-019/selfplay/games/game-000000.npz'
    arrays,meta=load_game(path)
    roots=probe.replay_game(arrays,meta,old)
    assert roots
    for _,request in (roots[0],roots[len(roots)//2],roots[-1]):
        a=determinize_request(request,seed=199,catalog=old,engine_mode='checked')
        b=determinize_request(request,seed=199,catalog=load_default_catalog(),engine_mode='checked')
        assert a.decision()==b.decision()==request
        # Full future conditional game is identical under the same hidden draw.
        for _ in range(8):
            if not isinstance(a.decision(),probe.DecisionRequest):break
            assert a.decision()==b.decision()
            assert a.apply(a.decision().legal_actions[0])==b.apply(b.decision().legal_actions[0])


def test_full_summary_preserves_empty_importance_strata_and_portable_raw(tmp_path,monkeypatch):
    import gzip
    import json
    art=tmp_path/'art';work=tmp_path/'work';art.mkdir();work.mkdir()
    monkeypatch.setattr(probe,'ROOT',tmp_path);monkeypatch.setattr(probe,'ART',art);monkeypatch.setattr(probe,'WORK',work)
    plan=dict(formal_repeat_seeds=[probe.SEED_BASE+i for i in probe.REPEATS],
              panel_repeat_seeds={p['name']:probe.repeat_seeds(p) for p in probe.PANELS})
    monkeypatch.setattr(probe,'verify_plan',lambda:plan)
    roots=[dict(id=i,request_sha256=str(i),stage=probe.STAGES[i%3],
                group='baseline' if i<3 else 'high_disagreement') for i in range(4)]
    probe.write_json(art/'root_bank.json',dict(roots=roots));probe.write_json(art/'plan.json',plan)
    for p in probe.PANELS:
        for root in roots:
            for seed in probe.repeat_seeds(p):
                r=row([.8,.2],0,None if p['backend']=='greedy' else [80,20],[40,0])
                if p['backend']=='greedy':r['world_value_sums']=[]
                r.update(root_id=root['id'],seed=seed,request_sha256=root['request_sha256'])
                probe.write_json(work/'raw'/p['name']/f"root-{root['id']:03d}-seed-{seed}.json",r)
    probe.summarize()
    summary=probe.read_json(art/'summary.json')
    assert summary['status']=='complete' and summary['reference_consensus_agreement_roots']==4
    assert next(g for g in summary['groups'] if g['stratum']=='references_disagree')['stability']['roots']==0
    assert len([json.loads(line) for line in gzip.open(art/'raw_searches.jsonl.gz','rt')])==4*9*8
