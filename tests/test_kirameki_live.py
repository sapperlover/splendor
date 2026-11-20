"""Public website fixtures and information/transport boundary regressions."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from splendor.catalog import load_default_catalog
from splendor.actions import BuyMarket, BuyReserved
from splendor.kirameki_live import (SNAPSHOT_JS, SITE_CARD_IDS, SITE_NOBLE_IDS,
    StateUnavailable, identity, recommend_snapshot, request_from_snapshot, save_audit, site_action, watch)
from splendor.model import ReservationSource
from splendor.parallel_recommend import merge_world_results, world_base_seed
from splendor.search_policy import _request_public_word, _seed_from_public_word

FIXTURES=Path(__file__).parent/'fixtures'
CASES=json.loads((FIXTURES/'kirameki_public_cases.json').read_text())


def test_complete_card_and_noble_mapping():
    data=json.loads((FIXTURES/'kirameki_catalog.json').read_text())
    cat=load_default_catalog()
    assert set(SITE_CARD_IDS)=={c.card_id for c in cat.cards}
    assert set(SITE_NOBLE_IDS)=={n.noble_id for n in cat.nobles}
    for c in data['cards']:
        local=cat.card(SITE_CARD_IDS[c['id']])
        assert (local.tier,local.points,list(local.cost),('white','blue','green','red','black').index(local.bonus.value))==(c['level'],c['points'],c['cost'],c['bonus'])
    for n in data['nobles']:
        local=cat.noble(SITE_NOBLE_IDS[n['id']])
        assert (local.points,list(local.requirement))==(n['points'],n['requirement'])


@pytest.mark.parametrize('snapshot',CASES,ids=lambda s:str(s['fixture_index']))
def test_site_and_native_candidates_and_roundtrip(snapshot):
    request=request_from_snapshot(snapshot)
    viewer=snapshot['perspective']; own=snapshot['state']['players'][viewer]
    assert tuple(request.observation.self_public.tokens)==tuple(own['gems'])
    assert request.observation.self_public.score==own['points']
    if not snapshot['state']['board']['waiting_return']:
        assert set(request.legal_actions)=={site_action(a) for a in snapshot['state']['legal_actions']}
    assert request.observation.self_is_first_player==(viewer==0)
    b=snapshot['state']['board']
    turns=(b['turn']+int(b['current_player']==1),b['turn'])
    assert request.observation.own_completed_turns==turns[viewer]
    assert request.observation.opponent_completed_turns==turns[1-viewer]


def test_own_blind_identity_and_opponent_hidden_tier():
    own=next(s for s in CASES if s['fixture_index']==3)
    request=request_from_snapshot(own)
    assert len(request.observation.self_private.blind_reserved_cards)==1
    assert request.observation.self_public.reserved_cards[0].source is ReservationSource.DECK
    opponent=next(s for s in CASES if s['fixture_index']==2)
    request=request_from_snapshot(opponent)
    assert len(request.observation.opponent_public.reserved_cards)==1
    assert request.observation.opponent_public.reserved_cards[0].card is None


def test_browser_reader_masks_secrets_before_returning_to_python():
    s=deepcopy(CASES[2]);viewer=s['perspective']
    # Deliberately inject an opponent identity that a full client might expose.
    s['state']['players'][1-viewer]['reserved_cards'][0]=61
    props={'state':s['state'],'perspective':viewer,'isHumanTurn':True,
           'publicReservedCardIds':s['publicReservedCardIds'],'hiddenReservedLevels':s['hiddenReservedLevels']}
    source='const props='+json.dumps(props)+';'
    source+='const view={state:props.state,moves:[],human_seat:props.perspective,seed:"SECRET_ACTUAL_SEED"};'
    source+='const fiber={memoizedProps:props,memoizedState:{memoizedState:view,next:null},return:null};'
    source+='global.document={querySelector:()=>({"__reactFiber$test":fiber})};'
    source+='global.localStorage={getItem:()=>JSON.stringify({ranked:false,moves:[],setup:{seed:"SECRET_ACTUAL_SEED"},token:"SECRET_ACCOUNT_TOKEN"})};'
    source+='process.stdout.write(JSON.stringify(('+SNAPSHOT_JS+')()));'
    result=subprocess.run(['node','-e',source],text=True,capture_output=True,check=True)
    safe=json.loads(result.stdout)
    assert safe['state']['players'][1-viewer]['reserved_cards'][0]==-1
    assert 'SECRET_' not in result.stdout
    assert safe['state']['legal_actions']==s['state']['legal_actions']


def test_world_merge_preserves_raw_visits_and_q_numerators():
    from splendor.mcts_policy import MCTSSearchResult
    def one(counts,sums):
        return MCTSSearchResult(chosen_candidate_index=0,searchable=True,
            root_network_prior=(.6,.4),root_used_prior=(.6,.4),simulation_budgets=(4,),
            per_world_root_visit_counts=(counts,),root_visit_counts=counts,
            policy_target=tuple(c/4 for c in counts),behavior_policy=(1.,0.),simulations=4,
            max_tree_depth=2,transition_cap_hits=0,per_world_policy_target_counts=(counts,),
            per_world_root_value_sums=(sums,),per_world_forced_playouts=((0,0),))
    merged=merge_world_results([(1,one((1,3),(.5,-1.5))),(0,one((4,0),(2.,0.)))],2)
    assert merged.root_visit_counts==(5,3)
    assert merged.policy_target==(.625,.375)
    assert merged.simulations==8
    assert merged.per_world_root_value_sums==((2.,0.),(.5,-1.5))
    with pytest.raises(ValueError,match='missing or duplicated'):
        merge_world_results([(0,one((4,0),(2.,0.)))]*2,2)


class SelectAdvisor:
    def __init__(self,main,follow):
        self.catalog=load_default_catalog(); self.main=main; self.follow=follow
    def recommend(self,request,top_k=3):
        if request.observation.phase.value=='main_action':chosen=self.main
        else:chosen=next(a for a in request.legal_actions if tuple(a.tokens)==tuple(self.follow))
        return {'recommended_candidate_index':request.legal_actions.index(chosen),
                'description':str(chosen),'submitted_action':False}


@pytest.mark.parametrize('kind',['payment','take_return'])
def test_atomic_website_action_uses_conditioned_native_followup(kind):
    if kind=='payment':
        s=next(s for s in CASES if any(a['type']==4 and a['pay'][5]>0 for a in s['state']['legal_actions']))
        actual=next(a for a in s['state']['legal_actions'] if a['type']==4 and a['pay'][5]>0)
        tokens=actual['pay']
    else:
        s=next(s for s in CASES if any(a['type'] in (0,1) and any(a['return_gems']) for a in s['state']['legal_actions']))
        actual=next(a for a in s['state']['legal_actions'] if a['type'] in (0,1) and any(a['return_gems']))
        tokens=actual['return_gems']
    result,request=recommend_snapshot(s,SelectAdvisor(site_action(actual),tokens))
    assert result['site_usi']==actual['usi']
    assert result['followup']['request']['observation']['phase'] in ('payment','return_tokens')
    assert result['submitted_action'] is False


def test_ranked_wrong_actor_and_terminal_are_rejected():
    for mutate in (lambda s:s.update(ranked=True),
                   lambda s:s['state']['board'].update(current_player=1-s['perspective']),
                   lambda s:s['state']['board'].update(game_over=True)):
        snapshot=deepcopy(CASES[0]);mutate(snapshot)
        with pytest.raises(StateUnavailable):request_from_snapshot(snapshot)


def test_multiple_nobles_after_purchase_requires_real_source_context():
    # Synthetic complete public position: three bonuses in W/U/G/R, with
    # WUG333 and UGR333 both available. No fabricated future/private cards.
    s=deepcopy(CASES[0]); cat=load_default_catalog()
    bought=[]
    for color in ('white','blue','green','red'):
        bought.extend(c.card_id for c in [c for c in cat.cards if c.bonus.value==color and c.points==0][:3])
    owned=set(bought);inverse={c:i for i,c in enumerate(SITE_CARD_IDS)}
    market=[];counts=[]
    for tier in (1,2,3):
        remaining=[c.card_id for c in cat.cards if int(c.tier)==tier and c.card_id not in owned]
        market.append([inverse[c] for c in remaining[:4]]);counts.append(len(remaining)-4)
    b=s['state']['board'];b.update(visible_cards=market,deck_counts=counts,nobles=[7,9,1],turn=15,waiting_noble=True)
    own=s['state']['players'][0];own.update(purchased_cards=[inverse[c] for c in bought],bonuses=[3,3,3,3,0],points=0)
    s['state']['legal_actions']=[{'type':5,'noble_choice':i,'usi':f'noble:N{i}'} for i in (7,9)]
    s['moves']=[f'buy:C{inverse[bought[-1]]}/pay:W0U0G0R0K0D0']
    with pytest.raises(StateUnavailable,match='购买前'):
        request_from_snapshot(s)
    previous=SimpleNamespace(observation=SimpleNamespace(
        self_public=SimpleNamespace(purchased_card_ids=tuple(bought[:-1]),reserved_cards=()),
        self_private=SimpleNamespace(blind_reserved_cards=())))
    request=request_from_snapshot(s,previous=previous)
    assert isinstance(request.observation.pending.main_action,BuyMarket)
    assert len(request.legal_actions)==2
    previous.observation.self_private.blind_reserved_cards=(cat.card(bought[-1]),)
    request=request_from_snapshot(s,previous=previous)
    assert isinstance(request.observation.pending.main_action,BuyReserved)


def test_world_seed_partition_preserves_all_sixteen_original_worlds():
    request=request_from_snapshot(CASES[0]);word=_request_public_word(request)
    for base in (0,42,-3):
        expected=[_seed_from_public_word(word,base,j) for j in range(16)]
        actual=[_seed_from_public_word(word,world_base_seed(base,j),0) for j in range(16)]
        assert actual==expected
        assert len(set(actual))==16


def test_each_advice_keeps_its_snapshot_when_latest_is_replaced(tmp_path):
    first=deepcopy(CASES[0]);second=deepcopy(CASES[1])
    p=save_audit(tmp_path,first,{'description':'first','submitted_action':False})
    q=save_audit(tmp_path,second,{'description':'second','submitted_action':False})
    assert p!=q
    assert json.loads((p/'snapshot.json').read_text())==first
    assert json.loads((p/'recommendation.json').read_text())['description']=='first'


def test_watch_survives_navigation_and_never_submits(monkeypatch,tmp_path):
    from playwright.async_api import Error
    import playwright.async_api
    safe=deepcopy(CASES[0]);safe['state']['legal_actions']=[{'type':6,'usi':'pass'}]
    class Page:
        url='https://kirameki-lab.com/play/game'
        def __init__(self):self.calls=0
        async def evaluate(self,script):
            assert script==SNAPSHOT_JS
            self.calls+=1
            if self.calls==1:raise Error('Execution context was destroyed, most likely because of a navigation')
            return safe
    page=Page();browser=SimpleNamespace(contexts=[SimpleNamespace(pages=[page])])
    closed=[]
    async def close():closed.append(True)
    browser.close=close
    async def connect(endpoint,**kwargs):
        assert endpoint=='http://127.0.0.1:9222'
        assert kwargs=={'no_defaults':True,'timeout':60000}
        return browser
    class PW:
        async def __aenter__(self):return SimpleNamespace(chromium=SimpleNamespace(connect_over_cdp=connect))
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(playwright.async_api,'async_playwright',PW)
    args=SimpleNamespace(cdp='http://127.0.0.1:9222',profile=None,poll=.001,settle=0,
        output=tmp_path,once=True,top_k=3,simulations_per_world=4096,determinizations=16,search_backend='native')
    advisor=SimpleNamespace(catalog=load_default_catalog(),mode='mcts')
    asyncio.run(watch(args,advisor))
    assert page.calls>=4
    assert closed==[True]
    assert json.loads((tmp_path/'latest.json').read_text())['submitted_action'] is False


def test_watch_reports_browser_connection_timeout_before_reading_or_searching(monkeypatch,capsys):
    import playwright.async_api
    async def connect(endpoint,**kwargs):
        assert endpoint=='http://127.0.0.1:9222'
        assert kwargs=={'no_defaults':True,'timeout':60000}
        raise playwright.async_api.TimeoutError('BrowserType.connect_over_cdp: Timeout 60000ms exceeded')
    class PW:
        async def __aenter__(self):return SimpleNamespace(chromium=SimpleNamespace(connect_over_cdp=connect))
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(playwright.async_api,'async_playwright',PW)
    args=SimpleNamespace(cdp='http://127.0.0.1:9222',profile=None)
    with pytest.raises(SystemExit,match='尚未读取盘面或开始搜索') as error:
        asyncio.run(watch(args,object()))
    assert 'edge://inspect/#remote-debugging' in str(error.value)
    assert 'http://127.0.0.1:9222' in str(error.value)
    assert '最长等待60秒' in capsys.readouterr().out


@pytest.mark.parametrize('options',[
    ['--search-backend','native_chance','--determinizations','16'],
    ['--search-backend','native_chance','--determinizations','1','--root-selection','world_mean_q'],
])
def test_chance_cli_rejects_world_ensembles_and_world_mean_q_before_loading_model(options,capsys):
    from splendor.kirameki_live import main
    with pytest.raises(SystemExit) as error:
        main(options)
    assert error.value.code==2
    assert 'native_chance' in capsys.readouterr().err


@pytest.mark.parametrize('options',[
    ['--leaf-batch-size','8'],
    ['--compile-mode','native-cudagraph'],
    ['--device','cpu','--inference-mode','compiled','--compile-mode','native-cudagraph','--search-workers','1'],
    ['--inference-mode','compiled','--search-workers','4'],
])
def test_cli_rejects_incompatible_leaf_and_inference_modes_before_loading_model(options):
    from splendor.kirameki_live import main
    with pytest.raises(SystemExit) as error:main(options)
    assert error.value.code==2
