"""Mutation guards, exact payment transport, acknowledgements and terminal records."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from splendor.kirameki_autoplay import (AUTO_SNAPSHOT_JS, LOBBY_JS, SUBMIT_JS,
    finish_game, game_directory, validate_game, wait_for_ack)
from splendor.kirameki_live import StateUnavailable

CASES=json.loads((Path(__file__).parent/'fixtures/kirameki_public_cases.json').read_text())


def sample():
    s=deepcopy(CASES[0])
    s.update(game_id='casual-123',site_mode='human-vs-ai',opponent_level='bga1000-v5',
             human_seat=s['perspective'],moves=[])
    return s


@pytest.mark.parametrize('change',[
    lambda s:s.update(ranked=True),lambda s:s.update(opponent_level='bga900-v2'),
    lambda s:s.update(site_mode='ai-vs-ai'),lambda s:s.update(human_seat=1-s['perspective']),
])
def test_wrong_match_is_rejected(change):
    s=sample();change(s)
    with pytest.raises(StateUnavailable):validate_game(s,'bga1000-v5')


def test_terminal_own_match_can_be_saved_and_game_id_cannot_escape_output(tmp_path):
    s=sample();s['state']['board'].update(game_over=True,winner=s['perspective'])
    validate_game(s,'bga1000-v5')
    path=game_directory(tmp_path,s);r=finish_game(path,s)
    assert r['outcome']=='win' and r['moves']==[]
    assert json.loads((path/'result.json').read_text())['terminal_snapshot']==s
    s['game_id']='../outside'
    with pytest.raises(ValueError):game_directory(tmp_path,s)


@pytest.mark.parametrize('viewer,winner,expected',[(0,-2,'draw'),(1,-2,'draw'),
    (0,0,'win'),(1,1,'win'),(0,1,'loss'),(1,0,'loss')])
def test_site_terminal_winner_codes_and_raw_draw_are_preserved(tmp_path,viewer,winner,expected):
    s=sample();s['perspective']=s['human_seat']=viewer
    s['state']['board'].update(game_over=True,winner=winner)
    result=finish_game(tmp_path,s)
    assert result['winner']==winner and result['outcome']==expected
    assert json.loads((tmp_path/'result.json').read_text())['terminal_snapshot']==s


@pytest.mark.parametrize('game_over,winner',[(False,-1),(False,-2),(False,0),(True,-1),(True,2)])
def test_incomplete_or_unknown_terminal_codes_are_rejected(tmp_path,game_over,winner):
    s=sample();s['state']['board'].update(game_over=game_over,winner=winner)
    with pytest.raises(ValueError,match='终局结果不完整'):finish_game(tmp_path,s)
    assert not (tmp_path/'result.json').exists()


class Page:
    def __init__(self,rows):self.rows=iter(rows);self.calls=0
    async def evaluate(self,script):
        assert script==AUTO_SNAPSHOT_JS
        self.calls+=1;return next(self.rows)


def test_ack_accepts_opponent_already_replied():
    before=sample();before['moves']=['take:WUG'];after=deepcopy(before)
    after['moves']+=['reserve:C3','take:UGR']
    page=Page([before,after])
    assert asyncio.run(wait_for_ack(page,before,'reserve:C3',1))==after
    assert page.calls==2


@pytest.mark.parametrize('change',[
    lambda s:s.update(game_id='casual-other'),
    lambda s:s.update(moves=['take:WRK']),
    lambda s:s.update(unavailable='导航中'),
])
def test_ack_ambiguity_stops_without_any_resubmit(change):
    before=sample();after=deepcopy(before);change(after)
    with pytest.raises(RuntimeError):asyncio.run(wait_for_ack(Page([after]),before,'take:WUG',1))


def run_node(source):
    return json.loads(subprocess.run(['node','-e',source],text=True,capture_output=True,check=True).stdout)


@pytest.mark.parametrize('ranked,selected,seat,expected',[(True,'bga1000-v5','random',False),
    (False,'bga900-v2','random',False),(False,'bga1000-v5','random',True),
    (False,'bga1000-v5','0',False),(False,'bga1000-v5','1',False)])
def test_lobby_checks_live_ranked_opponent_and_random_seat_before_click(ranked,selected,seat,expected):
    source='''let clicks=0;
global.location={hostname:'kirameki-lab.com',pathname:'/play'};
const ranked={checked:RANKED,closest:()=>({textContent:'Ranked game (counts toward your ELO)'})};
const opponent={value:SELECTED,options:[{value:'bga1000-v5'},{value:'bga900-v2'}]};
const seat={value:SEAT,options:[{value:'0'},{value:'1'},{value:'random'}],'__reactProps$test':{value:SEAT}};
const button={textContent:'Start game',disabled:false,click:()=>clicks++};
global.document={querySelectorAll:s=>s.startsWith('input')?[ranked]:s==='select'?[opponent,seat]:[button]};
const result=(SCRIPT)({level:'bga1000-v5',start:true});
process.stdout.write(JSON.stringify({result,clicks}));
'''.replace('RANKED',json.dumps(ranked)).replace('SELECTED',json.dumps(selected)).replace('SEAT',json.dumps(seat)).replace('SCRIPT',LOBBY_JS)
    r=run_node(source)
    assert r['result']['ready']==expected and r['clicks']==int(expected)


def test_reader_uses_nonsecret_start_time_for_casual_id_and_masks_opponent():
    s=deepcopy(CASES[2]);viewer=s['perspective'];s['state']['players'][1-viewer]['reserved_cards'][0]=61
    props={'state':s['state'],'perspective':viewer,'isHumanTurn':True,
           'publicReservedCardIds':s['publicReservedCardIds'],'hiddenReservedLevels':s['hiddenReservedLevels']}
    source='const props='+json.dumps(props)+''';
const view={state:props.state,moves:[],human_seat:props.perspective,mode:'human-vs-ai',
 level_ids:props.perspective===0?[null,'bga1000-v5']:['bga1000-v5',null],seed:'SECRET_DECK_SEED'};
const fiber={memoizedProps:props,memoizedState:{memoizedState:view,next:null},return:null};
global.document={querySelector:()=>({'__reactFiber$test':fiber})};
global.localStorage={getItem:()=>JSON.stringify({ranked:false,moves:[],setup:{seed:'SECRET_SEED',
 startedAt:'2026-10-31T00:00:00Z'},token:'SECRET_TOKEN'})};
process.stdout.write(JSON.stringify((SCRIPT)()));
'''.replace('SCRIPT',AUTO_SNAPSHOT_JS)
    r=run_node(source)
    assert r['game_id']=='casual-1793404800000'
    assert r['opponent_level']=='bga1000-v5'
    assert r['state']['players'][1-viewer]['reserved_cards'][0]==-1
    assert 'SECRET' not in json.dumps(r)


def test_exact_extra_gold_payment_waits_for_confirm_button_and_refuses_duplicate():
    s,chosen=next((deepcopy(s),a) for s in CASES for a in s['state']['legal_actions'] if a['type']==4
                 and any(b['type']==4 and b['card_id']==a['card_id'] and b['pay'][5]<a['pay'][5]
                         for b in s['state']['legal_actions']))
    source='const input='+json.dumps(s)+';const target='+json.dumps(chosen)+''';
const props={state:input.state,perspective:input.perspective,isHumanTurn:true,
 publicReservedCardIds:input.publicReservedCardIds,hiddenReservedLevels:input.hiddenReservedLevels};
const view={state:props.state,moves:[],human_seat:props.perspective,mode:'human-vs-ai',
 level_ids:props.perspective===0?[null,'bga1000-v5']:['bga1000-v5',null]};
let sent=[],clicks=0;
const panel={buttons:[],busy:false,canCancel:false,onCancel:()=>{panel.canCancel=false;}};
const choose=id=>{props.selectedCardId=id;panel.canCancel=true;
 const pays=props.state.legal_actions.filter(a=>a.type===4&&a.card_id===id).map(a=>a.pay);
 props.payingGems=pays[0];panel.buttons=[{tone:'primary',enabled:true,onClick:()=>{
  const a=props.state.legal_actions.find(a=>a.type===4&&a.card_id===id&&JSON.stringify(a.pay)===JSON.stringify(props.payingGems));
  if(!a)throw Error('invalid payment');sent.push(a.usi);
 }}];
 props.onResourceClick=c=>{
  clicks++;const old=props.payingGems,values=[...new Set(pays.map(v=>v[c]))].sort((a,b)=>a-b);
  const next=values[(values.indexOf(old[c])+1)%values.length],distance=v=>v.reduce((s,n,i)=>s+(i===c?0:Math.abs(n-old[i])),0);
  props.payingGems=pays.filter(v=>v[c]===next).reduce((best,v)=>distance(v)<distance(best)?v:best);
  panel.buttons[0].enabled=false;setTimeout(()=>panel.buttons[0].enabled=true,40);
 };
};
props.onCardClick=props.onReservedCardClick=choose;
const boardEl={'__reactFiber$test':{memoizedProps:props,memoizedState:{memoizedState:view,next:null},return:null}};
const toolbar={'__reactFiber$test':{memoizedProps:panel,return:null}};
global.document={querySelector:s=>s.includes('board-players')?boardEl:null,querySelectorAll:s=>s==='button'?[toolbar]:[]};
global.localStorage={getItem:()=>JSON.stringify({ranked:false,setup:{startedAt:'2026-10-31T00:00:00Z'}})};
global.window={};
(async()=>{
 const before=(SNAPSHOT)();const submit=(SUBMIT);
 await submit({snapshot:before,usi:target.usi});let duplicate=false;
 try{await submit({snapshot:before,usi:target.usi});}catch(e){duplicate=/禁止重发/.test(e.message);}
 process.stdout.write(JSON.stringify({sent,duplicate,clicks}));
})().catch(e=>{console.error(e);process.exit(1)});
'''.replace('SNAPSHOT',AUTO_SNAPSHOT_JS).replace('SUBMIT',SUBMIT_JS)
    r=run_node(source)
    assert r['sent']==[chosen['usi']] and r['duplicate'] and r['clicks']>0


@pytest.mark.parametrize('kind',['take','pair','take_return','reserve','deck','noble','pass','return'])
def test_board_callbacks_preserve_atomic_action_and_returns(kind):
    s=sample();actions=s['state']['legal_actions']
    if kind=='take_return':
        s=deepcopy(next(s for s in CASES if any(a['type']==0 and any(a['return_gems']) for a in s['state']['legal_actions'])))
        chosen=next(a for a in s['state']['legal_actions'] if a['type']==0 and any(a['return_gems']))
    elif kind=='return':
        s=deepcopy(next(s for s in CASES if s['state']['board']['waiting_return']))
        chosen=s['state']['legal_actions'][0]
    elif kind in ('noble','pass'):
        chosen={'type':5,'noble_choice':0,'usi':'noble:N0'} if kind=='noble' else {'type':6,'usi':'pass'}
        s['state']['legal_actions']=[chosen]
    else:
        chosen=next(a for a in actions if a['type']=={'take':0,'pair':1,'reserve':2,'deck':3}[kind])
    if kind in ('reserve','deck'):s['state']['board']['bank'][5]=0  # no-gold confirmation
    source='const input='+json.dumps(s)+';const target='+json.dumps(chosen)+''';
const props={state:input.state,perspective:input.perspective,isHumanTurn:true,
 publicReservedCardIds:input.publicReservedCardIds,hiddenReservedLevels:input.hiddenReservedLevels,
 selectedGems:[],returningGems:[0,0,0,0,0,0]};
const view={state:props.state,moves:[],human_seat:props.perspective,mode:'human-vs-ai',
 level_ids:props.perspective===0?[null,'bga1000-v5']:['bga1000-v5',null]};
let sent=[],confirmed=0,warning=null;
const panel={buttons:[],busy:false,canCancel:false,onCancel:()=>{panel.canCancel=false;}};
const send=()=>{if(target.type===0||target.type===1){
 const take=Array(6).fill(0);for(const c of props.selectedGems)take[c]++;
 if(JSON.stringify(take)!==JSON.stringify(target.take))throw Error('wrong take');}
 if(target.return_gems?.some(Boolean)&&JSON.stringify(target.return_gems)!==JSON.stringify(props.returningGems))throw Error('wrong return');
 sent.push(target.usi);};
const returning=()=>{props.isReturnSelectable=()=>true;props.onResourceClick=c=>props.returningGems[c]++;
 panel.buttons=[{tone:'danger',enabled:true,onClick:send}];};
const act=()=>{if(target.return_gems?.some(Boolean))returning();else send();};
const reserve=()=>{warning={onYes:()=>{confirmed++;warning=null;act();},onNo:()=>{}};};
props.onGemClick=c=>{props.selectedGems.push(c);panel.canCancel=true;panel.buttons=[{tone:'primary',enabled:true,onClick:act}];};
props.onCardClick=c=>{props.selectedCardId=c;panel.canCancel=true;panel.buttons=[{tone:'reserve',enabled:true,onClick:reserve}];};
props.onDeckClick=c=>{props.selectedDeckLevel=c;panel.canCancel=true;panel.buttons=[{tone:'reserve',enabled:true,onClick:reserve}];};
props.onNobleClick=c=>{if(c!==target.noble_choice)throw Error('wrong noble');send();};
if(target.type===6)panel.buttons=[{tone:'neutral',enabled:true,onClick:send}];
if(target.type===7)returning();
const boardEl={'__reactFiber$test':{memoizedProps:props,memoizedState:{memoizedState:view,next:null},return:null}};
const toolbar={'__reactFiber$test':{memoizedProps:panel,return:null}};
const modal={'__reactFiber$test':{get memoizedProps(){return warning;},return:null}};
global.document={querySelector:s=>s.includes('board-players')?boardEl:null,
 querySelectorAll:s=>s==='button'?[toolbar]:warning?[modal]:[]};
global.localStorage={getItem:()=>JSON.stringify({ranked:false,setup:{startedAt:'2026-10-31T00:00:00Z'}})};
global.window={};
(async()=>{const before=(SNAPSHOT)();await (SUBMIT)({snapshot:before,usi:target.usi});
 process.stdout.write(JSON.stringify({sent,confirmed}));})().catch(e=>{console.error(e);process.exit(1)});
'''.replace('SNAPSHOT',AUTO_SNAPSHOT_JS).replace('SUBMIT',SUBMIT_JS)
    r=run_node(source)
    assert r['sent']==[chosen['usi']]
    assert r['confirmed']==int(kind in ('reserve','deck'))


@pytest.mark.parametrize('shared_connection',[False,True])
@pytest.mark.parametrize('old_winner',[0,-2])
def test_watch_restarts_old_terminal_with_random_and_counts_only_new_game(monkeypatch,tmp_path,shared_connection,old_winner):
    import playwright.async_api
    import splendor.kirameki_autoplay as auto
    old=sample();old['state']['board'].update(game_over=True,winner=old_winner)
    fresh=sample();fresh['game_id']='casual-456'
    class GamePage:
        url='https://kirameki-lab.com/play/game'
        snapshot=old
        menu_clicks=0;random_settings=0;sends=0;seat='0'
        async def evaluate(self,script,arg=None):
            if script==AUTO_SNAPSHOT_JS:return deepcopy(self.snapshot)
            if script==LOBBY_JS:
                if arg.get('start'):
                    assert self.seat=='random'
                    self.url='https://kirameki-lab.com/play/game';self.snapshot=fresh
                return {'ready':True,'started':arg.get('start',False)}
            assert script==SUBMIT_JS
            assert arg['snapshot']==fresh and arg['usi']=='pass'
            self.sends+=1
            self.snapshot=deepcopy(fresh);self.snapshot['moves']=['pass']
            self.snapshot['state']['board'].update(game_over=True,winner=-2)
        def get_by_role(self,role,**kwargs):
            assert role=='button';return self
        async def click(self,**kwargs):self.menu_clicks+=1;self.url='https://kirameki-lab.com/play?level=bga1000-v5'
        def locator(self,selector):
            assert selector=='select:has(option[value="random"])';return self
        async def select_option(self,value):
            assert value=='random';self.seat=value;self.random_settings+=1
        async def wait_for_function(self,*args,**kwargs):assert self.seat=='random'
        async def wait_for_url(self,*args,**kwargs):pass
    page=GamePage();closed=[]
    async def close():closed.append(True)
    browser=SimpleNamespace(contexts=[SimpleNamespace(pages=[page])],close=close)
    async def connect(*args):return browser
    class PW:
        async def __aenter__(self):return object()
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(playwright.async_api,'async_playwright',PW)
    monkeypatch.setattr(auto,'connect_browser',connect)
    if shared_connection:
        async def unexpected_connect(*args):raise AssertionError('借用套件连接时不得再次连接浏览器')
        monkeypatch.setattr(auto,'connect_browser',unexpected_connect)
    monkeypatch.setattr(auto,'recommend_snapshot',lambda *args,**kwargs:({'description':'pass','site_usi':'pass','submitted_action':False},None))
    args=SimpleNamespace(output=tmp_path,opponent_level='bga1000-v5',games=1,once=False,
                         restart_delay=0,poll=.001,settle=0,top_k=3,action_timeout=1)
    finished=[]
    asyncio.run(auto.watch_autoplay(args,SimpleNamespace(checkpoint_sha256='test-model'),
                                   browser=browser if shared_connection else None,
                                   on_game_finished=lambda:finished.append(True)))
    assert (page.menu_clicks,page.random_settings,page.sends)==(1,1,1)
    assert closed==([] if shared_connection else [True])
    assert len(finished)==2  # initial old terminal and the new completed game
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['completed_games']==1 and summary['draw']==1 and summary['games']==['casual-456']
    assert (tmp_path/'games'/'casual-123'/'result.json').is_file()


@pytest.mark.parametrize('options',[['--games','-1'],['--action-timeout','0'],['--snapshot','unused.json']])
def test_invalid_autoplay_options_fail_before_model_load(options):
    from splendor.kirameki_autoplay import main
    with pytest.raises(SystemExit) as error:main(options)
    assert error.value.code==2
