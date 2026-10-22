"""Attach-only Chrome monitor; evaluate settled action/reveal events locally."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
import os
import json
import math
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import torch

from .bga_live import (SNAPSHOT_JS, ReservationHistory, StateUnavailable,
    observation_from_snapshot, request_from_snapshot, report_text,
    save_json, semantic_identity, table_id, vector)
from .features import encode_observation_state
from .actions import ReturnTokens
from .invariants import assert_game_state
from .observation import build_observation
from .recommend import LocalAdvisor, decision_identity, describe_action
from .model import Color, DecisionPhase, TokenVector
from .replay import to_primitive
from .search_policy import _determinized_state

# Subscribe after BGA's own handlers, and only write to our private event buffer.
# Browser-side settling preserves intermediate *completed turns* while Python is
# busy searching. A purchase and its reveal get separate output lines, tied to
# the same complete post-animation snapshot rather than a half-refilled market.
INSTALL_JS='''() => {
 if(window.__splendorLocalAdvisor)return true;
 if(typeof dojo==='undefined'||typeof gameui==='undefined')return false;
 const capture=('''+SNAPSHOT_JS+''');
 const o={pending:[],batches:[],handles:[],timer:null,previous:null,next:0,lost:0,lastAt:0,
  turn:Number(gameui.gamedatas.gamestate.args?.turn),lastMain:null};
 const signature=s=>JSON.stringify([s.player_id,s.move_text,s.gamedatas.players,
  s.gamedatas.cards_hidden,s.gamedatas.market,s.gamedatas.first_player,
  s.gamedatas.roundnumber,s.gamedatas.gamestate.name,s.gamedatas.gamestate.active_player,
  s.gamedatas.gamestate.args,s.gamedatas.decision]);
 function settle(){
  o.timer=null;
  if(!o.pending.length)return;
  const s=capture();
  if(!s||s.updating||!['none','payment','return_tokens'].includes(s.gamedatas.decision.decision_type)||
    !['playerTurn','playerDiscard','gameEnd'].includes(s.gamedatas.gamestate.name)){
   o.previous=null;o.timer=setTimeout(settle,125);return;
  }
  const sig=signature(s);
  if(sig!==o.previous||Date.now()-o.lastAt<250){
   o.previous=sig;o.timer=setTimeout(settle,125);return;
  }
  if(o.batches.length>=64){o.batches.shift();o.lost++;}
  o.batches.push({sequence:++o.next,events:o.pending.splice(0),snapshot:s});o.previous=null;
 }
 for(const topic of ['coins','buyCard','reserveCard','revealCard','claimNoble']){
  o.handles.push(dojo.subscribe(topic,n=>{
   const a=n.args||{};
   let action=null;
   if(topic==='coins'&&Object.entries(a.gap||{}).some(([c,v])=>c!=='G'&&Number(v)>0)){
    action={kind:'take',income:Object.fromEntries('CSEROG'.split('').map(c=>[c,Math.max(0,Number(a.gap[c]||0))]))};
   }
   if(topic==='reserveCard'){
    if(a.card?.location?.startsWith('market_'))action={kind:'reserve_market',card_id:String(a.card.id)};
    else if(a.card?.location?.startsWith('draw_'))action={kind:'reserve_deck',tier:Number(a.card.location.split('_')[1])};
    else if(a.drawpile)action={kind:'reserve_deck',tier:Number(a.drawpile)};
   }
   if(action){
    const turn=gameui.gamedatas.gamestate.args?.turn;
    if(turn!=null)o.turn=Number(turn);
    o.lastMain={actor_id:String(a.player_id),turn:o.turn,action};
   }
   if(topic==='coins'&&!Object.entries(a.gap||{}).some(([c,v])=>c!=='G'&&Number(v)>0))return;
   if(topic==='revealCard'&&a.card?.location==='hand')return;
   o.pending.push({kind:topic,actor_id:a.player_id==null?null:String(a.player_id),
    card_id:a.card?.id==null?null:String(a.card.id),
    public_reservation:topic==='reserveCard'&&!!a.card?.location?.startsWith('market_'),
    notification_uid:n.uid==null?null:String(n.uid),observed_at_ms:Date.now()});
   o.lastAt=Date.now();if(!o.timer)o.timer=setTimeout(settle,125);
  }));
 }
 window.__splendorLocalAdvisor=o;return true;
}'''
DRAIN_JS="""()=>{const o=window.__splendorLocalAdvisor;if(!o)return {batches:[],lost:0};
 const result={batches:o.batches.splice(0),lost:o.lost};o.lost=0;return result;}"""
CLEANUP_JS="""()=>{const o=window.__splendorLocalAdvisor;if(!o)return;
 if(o.timer)clearTimeout(o.timer);o.handles.forEach(h=>dojo.unsubscribe(h));delete window.__splendorLocalAdvisor;}"""


def cdp_endpoint(explicit=None,profile=None):
    if explicit:
        parsed=urlparse(explicit)
        if parsed.scheme not in ('http','https','ws','wss') or parsed.hostname not in ('localhost','127.0.0.1','::1'):
            raise ValueError('CDP地址必须是这台电脑上的本地Chrome调试地址')
        return explicit
    config=Path(os.environ.get('XDG_CONFIG_HOME',str(Path.home()/'.config')))
    roots=[profile] if profile else [config/name for name in ('google-chrome','microsoft-edge','chromium')]
    for directory in roots:
        port_file=directory/'DevToolsActivePort'
        if not port_file.exists():continue
        lines=port_file.read_text().splitlines()
        if len(lines)<2 or not lines[0].isdigit() or not 0<int(lines[0])<65536 or not lines[1].startswith('/devtools/browser/'):
            raise ValueError(f'浏览器调试端口文件无效：{port_file}；请通过--cdp指定地址')
        return f'ws://127.0.0.1:{int(lines[0])}{lines[1]}'
    return 'http://127.0.0.1:9222'


def evaluation_observations(snapshot,history,*,samples=4,seed=0):
    """Only sample information unknown to the logged-in viewer.

    When the opponent acts, evaluate its correctly oriented observations across
    sampled private hands and negate the mean value for the viewer. Never use
    BGA's opponent storedCards field or invent one exact opponent hand as truth.
    """
    observation,catalog,_=observation_from_snapshot(snapshot,history,require_self_turn=False)
    actor=str(snapshot['gamedatas']['gamestate']['active_player'])
    viewer=str(snapshot['player_id']);seat=0 if actor==viewer else 1
    # The viewer's state features cannot depend on sampled opponent secrets.
    count=samples if seat==1 and any(r.card is None for r in observation.opponent_public.reserved_cards) else 1
    values=[]
    for i in range(count):
        state=_determinized_state(observation,catalog,seed=seed+i)
        state=replace(state,current_player=seat)
        assert_game_state(state,catalog)
        values.append(build_observation(state,seat,catalog))
    return values,catalog,actor,seat


def estimate_snapshot(snapshot,history,evaluator,*,samples=4,seed=0):
    gd=snapshot['gamedatas'];viewer=str(snapshot['player_id'])
    if viewer not in gd['players'] or len(gd['players'])!=2:
        raise StateUnavailable('请在已有Chrome中登录本局双人玩家账号')
    if gd['gamestate']['name']=='gameEnd':
        # Resignation/abandonment cannot be inferred from development points.
        raise StateUnavailable('对局已结束；胜负以BGA结算为准，停止模型估算')
    observations,_,actor,seat=evaluation_observations(snapshot,history,samples=samples,seed=seed)
    raw=evaluator.value_only_encoded(tuple(encode_observation_state(o) for o in observations))
    if len(raw)!=len(observations) or not all(math.isfinite(v) for v in raw):raise ValueError('模型输出非有限或数量不符的value')
    actor_value=sum(raw)/len(raw);value=actor_value if seat==0 else -actor_value
    count=len(raw);source='model'
    rate=(max(-1.,min(1.,value))+1.)/2.
    return {'schema':'bga-live-evaluation-v1','table':table_id(snapshot['url']),
        'move':snapshot.get('move_text'),'viewer_id':viewer,'evaluation_actor_id':actor,
        'value_actor_mean':actor_value,'value_viewer_raw':value,'viewer_score_rate':rate,
        'opponent_score_rate':1.-rate,'value_samples':count,'source':source,
        'estimate_kind':'expected_score_draw_half_uncalibrated',
        'snapshot_sha256':semantic_identity(snapshot),'submitted_action':False}


def event_label(event,viewer):
    if event['kind']=='revealCard':return '翻牌'+(f"（BGA牌{event['card_id']}）" if event.get('card_id') else '')
    if event['kind']=='attach':return '接入局面'
    if event['kind']=='update':return '局面更新'
    who='我方' if str(event.get('actor_id'))==str(viewer) else '对方'
    verb={'coins':'拿筹码完成','buyCard':'购买完成','reserveCard':'保留完成','claimNoble':'领取贵族'}[event['kind']]
    return who+verb


def evaluation_line(value,event):
    stamp=event.get('observed_at_ms')
    clock=datetime.fromtimestamp(stamp/1000,ZoneInfo('Asia/Singapore')).strftime('%H:%M:%S') if stamp else datetime.now(ZoneInfo('Asia/Singapore')).strftime('%H:%M:%S')
    label=event_label(event,value['viewer_id'])
    suffix='最终结果' if value['source']=='result' else '模型估计，平局折半'
    return (f"[{clock}] 桌{value['table']} 第{value['move']}步 · {label} | "
            f"我方胜率估计 {value['viewer_score_rate']:.1%} · 对方 {value['opponent_score_rate']:.1%}（{suffix}）")


async def find_game(browser,url=None):
    """Read only BGA game frames; never navigate or inspect unrelated pages."""
    from playwright.async_api import Error
    wanted=table_id(url) if url else None
    found=[]
    for context in browser.contexts:
        for page in context.pages:
            host=urlparse(page.url).hostname or ''
            if host!='boardgamearena.com' and not host.endswith('.boardgamearena.com'):continue
            for frame in page.frames:
                try:tid=table_id(frame.url)
                except ValueError:continue
                if wanted is not None and tid!=wanted:continue
                try:snapshot=await frame.evaluate(SNAPSHOT_JS)
                except Error as error:
                    if any(t in str(error) for t in ('Execution context was destroyed','Cannot find context','Frame was detached','Target page, context or browser has been closed')):continue
                    raise
                if snapshot:found.append((frame,snapshot))
    players=[item for item in found if str(item[1]['player_id']) in item[1]['gamedatas']['players']]
    if players:found=players
    live=[item for item in found if item[1]['gamedatas']['gamestate']['name']!='gameEnd']
    if live:found=live
    focused=[item for item in found if item[1].get('page_focused')]
    visible=[item for item in found if item[1].get('page_visible')]
    if len(focused)==1:return focused[0]
    if len(visible)==1:return visible[0]
    if len(found)==1:return found[0]
    if not found:return None
    raise StateUnavailable('有多张BGA牌桌，请切换到需要分析的Chrome牌桌标签页；也可用--url限定')


class Monitor:
    def __init__(self,args):
        self.args=args;self.output=args.output.resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.histories={};self.advisor=None;self.last_message=None
        self.last_value={};self.last_request=None;self.last_result=None;self.current=None

    def status(self,message,tid=None):
        if (message,tid)==self.last_message:return
        value={'status':'waiting','message':message,'table':tid,'submitted_action':False}
        save_json(self.output/'latest.json',value)
        (self.output/'latest.md').write_text(report_text(value))
        # Search/waiting-for-opponent does not invalidate the already evaluated
        # board. All other transitions must invalidate the externally read rate.
        if message not in ('等待轮到你','局面已读取，正在计算推荐'):
            save_json(self.output/'latest_evaluation.json',{'status':'waiting','valid':False,
                'message':message,'table':tid,'submitted_action':False})
        quiet=('正在核对局面是否稳定','等待页面完成更新','你正在网页内选择动作；等待确认或取消后重新读取')
        if message not in quiet:print(message,flush=True)
        self.last_message=(message,tid)

    def history(self,snapshot):
        key=(table_id(snapshot['url']),str(snapshot['player_id']))
        if key not in self.histories:
            path=self.output/f'history_{key[0]}.json'
            if path.exists():
                history=ReservationHistory(**json.loads(path.read_text()))
                if history.viewer!=key[1]:
                    path=self.output/f'history_{key[0]}_{key[1]}.json'
                    history=ReservationHistory(**json.loads(path.read_text())) if path.exists() else ReservationHistory.start(snapshot)
            else:history=ReservationHistory.start(snapshot)
            self.histories[key]=(history,path)
        return self.histories[key]

    def get_advisor(self,snapshot):
        from .bga_live import runtime_catalog
        catalog,_,_=runtime_catalog(snapshot['gamedatas'])
        if self.advisor is None or self.advisor.catalog.data_fingerprint!=catalog.data_fingerprint:
            self.advisor=LocalAdvisor(self.args.checkpoint,catalog=catalog,mode=self.args.mode,device='cpu',
                simulations=self.args.simulations,determinizations=self.args.determinizations,seed=self.args.seed)
        return self.advisor

    def evaluate(self,snapshot,events):
        if snapshot['gamedatas']['gamestate']['name']=='gameEnd':
            raise StateUnavailable('对局已结束；胜负以BGA结算为准，停止模型估算')
        history,path=self.history(snapshot)
        for event in events:
            if event['kind']=='reserveCard' and event.get('public_reservation'):
                key=str(event['card_id']);owner=str(event['actor_id'])
                if key in history.market_reserves and history.market_reserves[key]!=owner:
                    raise StateUnavailable('公开保留事件与历史拥有者冲突')
                history.market_reserves[key]=owner
        history.observe(snapshot);save_json(path,history.to_dict())
        advisor=self.get_advisor(snapshot)
        value=estimate_snapshot(snapshot,history,advisor.evaluator,samples=self.args.value_samples,seed=self.args.seed)
        value.update(status='ready',valid=True,checkpoint=str(advisor.checkpoint),checkpoint_sha256=advisor.checkpoint_sha256)
        save_json(self.output/'latest_evaluation.json',value)
        save_json(self.output/'evaluation_snapshot.json',snapshot)
        for event in events:
            row={**value,'event':event,'snapshot_timing':'settled_after_action_and_reveals'}
            with (self.output/'evaluations.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            print(evaluation_line(value,event),flush=True)
        key=(value['table'],value['viewer_id']);self.last_value[key]=value['snapshot_sha256']
        return value


async def watch_existing(args):
    from playwright.async_api import async_playwright, Error
    torch.set_num_threads(args.threads)
    monitor=Monitor(args);attached={};previous=None;stable=0
    print('只连接已有Chrome；自动识别BGA牌桌，不新建窗口、不跳转网页、不执行招法。',flush=True)
    print('胜率显示为(V+1)/2的模型估计，平局折半，未经概率校准。',flush=True)
    async with async_playwright() as p:
        browser=None
        try:
            while True:
                if browser is None or not browser.is_connected():
                    try:browser=await p.chromium.connect_over_cdp(cdp_endpoint(args.cdp),no_defaults=True,timeout=5000)
                    except Error:
                        monitor.status('等待已有Chrome的本地调试连接；请打开已启用远程调试的Chrome（默认端口9222）')
                        if args.once:return
                        await asyncio.sleep(2);continue
                    monitor.status('已连接现有Chrome，正在寻找已登录的BGA牌桌')
                try:
                    picked=await find_game(browser,args.url)
                    if picked is None:
                        monitor.status('等待在现有Chrome中打开BGA牌桌');await asyncio.sleep(args.interval);continue
                    frame,snapshot=picked
                    tid=table_id(snapshot['url']);key=(tid,str(snapshot['player_id']))
                    if key!=monitor.current:
                        monitor.current=key;monitor.last_request=None;monitor.last_value.pop(key,None);previous=None;stable=0
                        monitor.status(f'已识别牌桌 {tid}，直接读取现有标签页',tid)
                    # The Frame object survives navigation; reinstall per document.
                    if await frame.evaluate(INSTALL_JS):attached[frame]=True
                    drained=await frame.evaluate(DRAIN_JS)
                    if drained['lost']:print(f"事件缓冲曾溢出，丢失{drained['lost']}组历史更新；当前局面继续读取。",flush=True)
                    for batch in drained['batches']:
                        try:monitor.evaluate(batch['snapshot'],batch['events'])
                        except StateUnavailable as error:monitor.status(str(error),tid)
                    identity=semantic_identity(snapshot)
                    stable=stable+1 if identity==previous else 1;previous=identity
                    if stable<2:
                        monitor.status('正在核对局面是否稳定',tid);await asyncio.sleep(args.interval);continue
                    save_json(monitor.output/'observed_snapshot.json',snapshot)
                    if monitor.last_value.get(key)!=identity:
                        monitor.evaluate(snapshot,[{'kind':'attach' if key not in monitor.last_value else 'update'}])
                    if args.once:return
                    history,_=monitor.history(snapshot)
                    request,catalog,mapping=request_from_snapshot(snapshot,history)
                    request_key=decision_identity(request)
                    if monitor.last_request==request_key:
                        if monitor.last_message is not None:
                            save_json(monitor.output/'latest.json',monitor.last_result)
                            (monitor.output/'latest.md').write_text(report_text(monitor.last_result));monitor.last_message=None
                        await asyncio.sleep(args.interval);continue
                    advisor=monitor.get_advisor(snapshot)
                    monitor.status('局面已读取，正在计算推荐',tid)
                    result=advisor.recommend(request,top_k=args.top_k)
                    fresh=await frame.evaluate(SNAPSHOT_JS)
                    if fresh is None or semantic_identity(fresh)!=identity:
                        monitor.last_request=None;monitor.status('计算期间局面已变化，重新读取',tid);continue
                    result.update(status='ready',table=tid,move=snapshot.get('move_text'),viewer_id=snapshot['player_id'],
                        value_viewer_raw=result['value_actor_raw'],expected_score_viewer=(max(-1.,min(1.,result['value_actor_raw']))+1.)/2.)
                    if request.observation.phase is DecisionPhase.RETURN_TOKENS:
                        selected=vector(snapshot['gamedatas']['decision']['selected_tokens'])
                        remaining=TokenVector(*(result['recommended_action']['tokens'][color.value] for color in Color))
                        if selected.total:
                            result['selected_return_description']=describe_action(ReturnTokens(selected),catalog)
                            result['total_return_description']=describe_action(ReturnTokens(selected.plus(remaining)),catalog)
                    action=result['recommended_action']
                    if 'card_id' in action:result['bga_card_id']=mapping['cards'][action['card_id']]
                    if 'noble_id' in action:result['bga_noble_id']=mapping['nobles'][action['noble_id']]
                    save_json(monitor.output/'latest.json',result)
                    (monitor.output/'latest.md').write_text(report_text(result))
                    save_json(monitor.output/'current_request.json',{'request':to_primitive(request),'bga_mapping':mapping})
                    save_json(monitor.output/'current_snapshot.json',snapshot)
                    save_json(monitor.output/'current_nobles.json',[to_primitive(n) for n in catalog.nobles])
                    print(report_text(result),flush=True)
                    monitor.last_request=request_key;monitor.last_result=result;monitor.last_message=None
                except StateUnavailable as error:
                    monitor.last_request=None;monitor.status(str(error),monitor.current[0] if monitor.current else None)
                    if args.once and '对局已结束' in str(error):return
                except Error as error:
                    if not browser.is_connected():
                        monitor.last_request=None;monitor.status('Chrome连接已断开，等待重新连接');continue
                    if any(t in str(error) for t in ('Execution context was destroyed','Cannot find context','Frame was detached','Target page, context or browser has been closed')):
                        monitor.last_request=None;monitor.status('网页正在跳转或已关闭，重新识别牌桌');continue
                    raise
                await asyncio.sleep(args.interval)
        finally:
            for frame in attached:
                try:await frame.evaluate(CLEANUP_JS)
                except Error:pass  # A closed/navigated frame owns no usable observer.
            if not args.once:monitor.status('推荐器已停止，旧推荐及胜率失效')
            # Stop the driver only. Never close a user-owned browser/context/tab.
