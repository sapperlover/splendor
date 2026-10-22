from __future__ import annotations
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from splendor.bga_live import ReservationHistory,StateUnavailable,request_from_snapshot,semantic_identity
from splendor.bga_watch import cdp_endpoint,estimate_snapshot,evaluation_line,evaluation_observations,find_game
from splendor.features import encode_decision_state,encode_observation_state

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def snapshot():
    return json.loads((ROOT/'tests/fixtures/bga_live_step8_934625108.json').read_text())

@pytest.fixture
def opponent_turn(snapshot):
    snapshot['public_logs']=[]
    gd=snapshot['gamedatas'];other='100878205'
    gd['gamestate']['active_player']=other
    gd['players'][other]['coins']={'C':0,'S':0,'E':0,'R':0,'O':0,'G':3}
    gd['market']['pool']={'C':4,'S':4,'E':4,'R':4,'O':4,'G':2}
    return snapshot


def test_own_encoding_matches_existing_features(snapshot):
    request,_,_=request_from_snapshot(snapshot,ReservationHistory.start(snapshot))
    a,b=encode_decision_state(request),encode_observation_state(request.observation)
    for field in a.__dataclass_fields__:
        x,y=getattr(a,field),getattr(b,field)
        if isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
        else:assert x==y


def test_opponent_view_turn_order_and_hidden_hand_sampling(opponent_turn):
    rows,catalog,actor,seat=evaluation_observations(opponent_turn,ReservationHistory.start(opponent_turn),samples=8,seed=9)
    assert actor=='100878205' and seat==1 and len(rows)==8
    for row in rows:
        assert row.self_is_first_player and row.own_completed_turns==row.opponent_completed_turns==3
        assert len(row.self_private.blind_reserved_cards)==3
        assert all(c.card_id not in ('008','025') for c in row.self_private.blind_reserved_cards)
        assert len(row.opponent_public.reserved_cards)==1
        assert row.opponent_public.reserved_cards[0].card is None
    assert len({tuple(c.card_id for c in row.self_private.blind_reserved_cards) for row in rows})>1
    assert len(catalog.cards)==90


def test_opponent_value_is_averaged_and_negated(opponent_turn):
    class Evaluator:
        def value_only_encoded(self,states):
            assert len(states)==4
            return (.1,.3,.5,.7)
    r=estimate_snapshot(opponent_turn,ReservationHistory.start(opponent_turn),Evaluator(),samples=4)
    assert r['value_actor_mean']==pytest.approx(.4) and r['value_viewer_raw']==pytest.approx(-.4)
    assert r['viewer_score_rate']==pytest.approx(.3) and r['opponent_score_rate']==pytest.approx(.7)
    assert r['submitted_action'] is False


def test_own_value_needs_no_hidden_world_sampling(snapshot):
    class Evaluator:
        def value_only_encoded(self,states):
            assert len(states)==1
            return (.6,)
    r=estimate_snapshot(snapshot,ReservationHistory.start(snapshot),Evaluator(),samples=10)
    assert r['value_samples']==1 and r['viewer_score_rate']==pytest.approx(.8)


def test_game_end_does_not_guess_resignation_from_points(snapshot):
    snapshot['gamedatas']['gamestate']={'name':'gameEnd'}
    with pytest.raises(StateUnavailable,match='BGA结算'):
        estimate_snapshot(snapshot,ReservationHistory.start(snapshot),None)


def test_invalid_values_not_published(snapshot):
    class Evaluator:
        def value_only_encoded(self,states):return (float('nan'),)
    with pytest.raises(ValueError,match='value'):
        estimate_snapshot(snapshot,ReservationHistory.start(snapshot),Evaluator())


def test_acknowledgments_do_not_invent_board_updates(snapshot):
    before=semantic_identity(snapshot)
    p=snapshot['gamedatas']['players']['92773015'];p['ack']='new';p['avatar']='changed'
    assert semantic_identity(snapshot)==before
    p['coins']['C']=1
    assert semantic_identity(snapshot)!=before


def test_cdp_endpoint_reads_actual_websocket_without_http_discovery(tmp_path):
    (tmp_path/'DevToolsActivePort').write_text('9222\n/devtools/browser/test\n')
    assert cdp_endpoint(profile=tmp_path)=='ws://127.0.0.1:9222/devtools/browser/test'
    (tmp_path/'DevToolsActivePort').write_text('bad\ninvalid\n')
    with pytest.raises(ValueError,match='无效'):cdp_endpoint(profile=tmp_path)
    with pytest.raises(ValueError,match='本地'):cdp_endpoint('http://remote.example:9222')


def test_auto_selection_only_reads_existing_tabs(snapshot):
    class Frame:
        def __init__(self,tid,focus=False):self.url=f'https://boardgamearena.com/13/splendor?table={tid}';self.focus=focus
        async def evaluate(self,script):
            s=copy.deepcopy(snapshot);s['url']=self.url;s['page_focused']=self.focus;s['page_visible']=False;return s
    class Page:
        url='https://boardgamearena.com/tableview'
        def __init__(self,f):self.frames=[f]
        # No goto/new_page/close methods: discovery must never call them.
    a,b=Frame(1),Frame(2)
    browser=SimpleNamespace(contexts=[SimpleNamespace(pages=[Page(a)])])
    assert asyncio.run(find_game(browser))[0] is a
    browser.contexts[0].pages.append(Page(b))
    with pytest.raises(StateUnavailable,match='多张'):asyncio.run(find_game(browser))
    b.focus=True
    assert asyncio.run(find_game(browser))[0] is b
    assert asyncio.run(find_game(browser,'https://boardgamearena.com/table?table=1'))[0] is a


def test_separate_action_and_reveal_output(snapshot):
    class Evaluator:
        def value_only_encoded(self,states):return (.2,)
    value=estimate_snapshot(snapshot,ReservationHistory.start(snapshot),Evaluator())
    lines=[evaluation_line(value,e) for e in [
        {'kind':'buyCard','actor_id':str(snapshot['player_id'])},
        {'kind':'revealCard','card_id':'40'},
        {'kind':'coins','actor_id':'100878205'}]]
    assert '我方购买完成' in lines[0] and '翻牌' in lines[1] and '对方拿筹码完成' in lines[2]
    assert all('我方胜率估计 60.0%' in line for line in lines)


def test_navigation_detached_frame_is_a_transient_read(snapshot):
    from playwright.async_api import Error
    class Frame:
        url='https://boardgamearena.com/13/splendor?table=934625108'
        async def evaluate(self,script):raise Error('Frame was detached')
    page=SimpleNamespace(url='https://boardgamearena.com/tableview?table=934625108',frames=[Frame()])
    browser=SimpleNamespace(contexts=[SimpleNamespace(pages=[page])])
    assert asyncio.run(find_game(browser)) is None


def test_external_rate_is_invalidated_on_change_but_not_opponent_turn(tmp_path):
    from splendor.bga_watch import Monitor
    from splendor.bga_live import save_json
    monitor=Monitor(SimpleNamespace(output=tmp_path))
    path=tmp_path/'latest_evaluation.json'
    save_json(path,{'status':'ready','valid':True,'viewer_score_rate':.6})
    monitor.status('等待轮到你',1)
    assert json.loads(path.read_text())['valid'] is True
    monitor.status('正在核对局面是否稳定',1)
    invalid=json.loads(path.read_text())
    assert invalid['valid'] is False and 'viewer_score_rate' not in invalid


def test_browser_reader_captures_payment_and_pending_discard_actions():
    import subprocess
    from playwright._impl._driver import compute_driver_executable
    from splendor.bga_live import SNAPSHOT_JS
    from splendor.bga_watch import INSTALL_JS
    program=r'''
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const zeros=()=>Object.fromEntries('CSEROG'.split('').map(c=>[c,0]));
const player=()=>({...zeros(),boughtCards:[],boughtNobles:[],storedCards:[],hidden_cards:{1:0,2:0,3:0}});
const model={gamestorage:{...zeros(),cards:[],nobles:[],drawcounts:{1:36,2:26,3:16}},
 playerstorage:{1:player(),2:player()},tempstorage:{...zeros(),card:0},paying:zeros(),
 carddb:{},nobledb:{},getPossibleCardBuys:()=>[33]};
global.window={};global.location={href:'https://boardgamearena.com/1/splendor?table=1'};
global.gameui={player_id:1,animated:false,gamedatas:{players:{1:{},2:{}},market:{},
 gamestate:{name:'playerTurn',active_player:1,args:{turn:4}}}};
global.require=()=>model;global.g_gamethemeurl='fixture/';
let paymentButton=true;
global.document={visibilityState:'visible',hasFocus:()=>true,querySelectorAll:()=>[],
 getElementById:id=>id==='btn_sendpay'&&paymentButton?{}:null};
global.getComputedStyle=()=>({display:'none'});
const capture=(new Function('return ('+input.capture+')'))();
model.selectedcard_id=33;model.paying.S=2;
const payment=capture().gamedatas.decision;
const handlers={};global.dojo={subscribe:(name,cb)=>{handlers[name]=cb;return name;}};
global.setTimeout=()=>1;global.clearTimeout=()=>{};
(new Function('return ('+input.install+')'))()();
model.selectedcard_id=null;paymentButton=false;model.paying=zeros();
handlers.coins({args:{player_id:1,gap:{C:1,S:1,E:1,R:0,O:0,G:0}}});
gameui.gamedatas.gamestate={name:'playerDiscard',active_player:1,args:null};
const discard=capture().gamedatas.decision;
model.paying.E=1;
const partial=capture().gamedatas.decision;
gameui.gamedatas.gamestate={name:'playerTurn',active_player:1,args:{turn:5}};
handlers.reserveCard({args:{player_id:1,card:{id:33,location:'market_1'}}});
gameui.gamedatas.gamestate={name:'playerDiscard',active_player:1,args:null};
const reserve=capture().gamedatas.decision;
gameui.gamedatas.gamestate={name:'playerTurn',active_player:1,args:{turn:6}};capture();
handlers.reserveCard({args:{player_id:1,card:{id:70,location:'draw_2'}}});
gameui.gamedatas.gamestate={name:'playerDiscard',active_player:1,args:null};
const blind=capture().gamedatas.decision;
window.__splendorLocalAdvisor.lastMain=null;
const missing=capture().gamedatas.decision;
process.stdout.write(JSON.stringify({payment,discard,partial,reserve,blind,missing}));
'''
    node,_=compute_driver_executable()
    completed=subprocess.run([node,'-e',program],input=json.dumps({'capture':SNAPSHOT_JS,'install':INSTALL_JS}),
                             text=True,capture_output=True,check=False,timeout=15)
    assert completed.returncode==0,completed.stderr
    rows=json.loads(completed.stdout)
    assert rows['payment']['decision_type']=='payment'
    assert rows['payment']['card_id']=='33' and rows['payment']['selected_tokens']['S']==2
    assert rows['discard']['decision_type']=='return_tokens' and rows['discard']['turn']==4
    assert rows['discard']['main_action']=={'kind':'take','income':dict(zip('CSEROG',(1,1,1,0,0,0)))}
    assert rows['partial']['selected_tokens']['E']==1
    assert rows['reserve']['main_action']=={'kind':'reserve_market','card_id':'33'}
    assert rows['reserve']['turn']==5
    assert rows['blind']['main_action']=={'kind':'reserve_deck','tier':2}
    assert rows['missing']['main_action'] is None


def test_opponent_discard_value_uses_the_acting_players_wallet(opponent_turn):
    from splendor.model import DecisionPhase
    gd=opponent_turn['gamedatas'];actor=gd['gamestate']['active_player']
    gd['players'][actor]['coins']=dict(zip('CSEROG',(3,3,3,2,2,0)))
    gd['market']['pool']=dict(zip('CSEROG',(1,1,1,2,2,5)))
    gd['gamestate'].update(name='playerDiscard',args=None)
    gd['decision']={'decision_type':'return_tokens','turn':4,
                    'main_action':{'kind':'take','income':dict(zip('CSEROG',(1,1,1,0,0,0)))},
                    'selected_tokens':dict(zip('CSEROG',(0,0,0,0,0,0)))}
    rows,_,_,seat=evaluation_observations(opponent_turn,ReservationHistory.start(opponent_turn),samples=2)
    assert seat==1 and len(rows)==2
    assert all(row.phase is DecisionPhase.RETURN_TOKENS and row.self_public.tokens.total==13
               and row.pending.tokens_to_return==3 for row in rows)
