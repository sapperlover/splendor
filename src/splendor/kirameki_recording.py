"""Read-only game/website-value recording, isolated from local search inputs."""
from __future__ import annotations

import asyncio
import gzip
import json
import time

from .kirameki_autoplay import AUTO_SNAPSHOT_JS, game_directory, validate_game
from .kirameki_live import identity


# Observe existing UI client promises, returning each original promise unchanged.
# No additional website evaluation/search is requested. Only explicit public
# fields and self-visible cards are projected before anything leaves the page.
INSTALL_RECORDER_JS = r'''() => {
 const el=document.querySelector('[data-testid="board-players"]');if(!el)return false;
 let f=el[Object.keys(el).find(k=>k.startsWith('__reactFiber$'))],client=null;
 for(;f;f=f.return)for(let h=f.memoizedState;h;h=h.next){const v=h.memoizedState?.current;
  if(typeof v?.evaluate==='function'&&typeof v?.humanAction==='function'&&typeof v?.aiAction==='function')client=v;
 }
 if(!client)return false;
 if(window.__splendorGameRecorder?.client===client)return true;
 const recorder={client,events:[],pending:0,dropped:0,originals:{},wrappers:{}};
 const metadata=()=>{const r=JSON.parse(localStorage.getItem('lab.current')||'null');
  const t=Date.parse(r?.setup?.startedAt);return {game_id:Number.isFinite(t)?'casual-'+t:null,
   ranked:r?.ranked,human_seat:r?.setup?.humanSeat,moves:r?.moves?.slice()??[]};};
 const push=e=>{if(recorder.events.length>=256){recorder.events.shift();recorder.dropped++;}recorder.events.push(e);};
 const project=(v,m)=>{
  if(!v?.state?.board||!Array.isArray(v.moves)||m.ranked!==false)return null;
  const seat=v.human_seat;if(seat!==0&&seat!==1)return null;
  const pub=v.public_reserved_card_ids||[],hidden=v.hidden_reserved_levels||{},b=v.state.board,board={};
  for(const k of ['bank','visible_cards','deck_counts','nobles','current_player','turn','waiting_noble','waiting_return','game_over','winner'])board[k]=b[k];
  const players=v.state.players.map((p,i)=>({index:p.index,gems:p.gems,bonuses:p.bonuses,points:p.points,
   purchased_cards:p.purchased_cards,acquired_nobles:p.acquired_nobles,
   reserved_cards:p.reserved_cards.map(id=>i===seat||pub.includes(id)?id:-1)}));
  return {schema:'kirameki-visible-v1',ranked:false,game_id:m.game_id,perspective:seat,human_seat:seat,
   isHumanTurn:!b.game_over&&b.current_player===seat,site_mode:v.mode,opponent_level:v.level_ids?.[1-seat]??null,
   moves:v.moves.map(x=>x.usi),publicReservedCardIds:pub,hiddenReservedLevels:hidden,
   state:{board,players,legal_actions:b.current_player===seat?v.state.legal_actions:[]}};
 };
 for(const name of ['evaluate','humanAction','aiAction','checkClock']){
  const original=client[name];if(typeof original!=='function')continue;recorder.originals[name]=original;
  client[name]=function(...args){
   const m=metadata(),requested_at_ms=Date.now(),promise=original.apply(this,args);recorder.pending++;
   promise.then(value=>{
    const event={game_id:m.game_id,requested_at_ms,completed_at_ms:Date.now(),api:name};
    if(name==='evaluate'&&m.ranked===false){
     if(Number.isFinite(value?.p0))push({...event,kind:'site_evaluation',moves:m.moves,human_seat:m.human_seat,p0:value.p0});
    }else{const snapshot=project(value,m);if(snapshot)push({...event,kind:'action_state',snapshot});}
   },error=>push({kind:'site_api_error',game_id:m.game_id,api:name,error:String(error),at_ms:Date.now()}))
    .catch(error=>{recorder.failure=String(error);})
    .finally(()=>recorder.pending--);
   return promise;
  };
  recorder.wrappers[name]=client[name];
 }
 window.__splendorGameRecorder=recorder;return true;
}'''

CLEANUP_RECORDER_JS = r'''() => {
 const r=window.__splendorGameRecorder;if(!r)return;
 for(const [name,fn] of Object.entries(r.originals))if(r.client[name]===r.wrappers[name])r.client[name]=fn;
 delete window.__splendorGameRecorder;
}'''

READ_PROGRESS_JS = r'''() => {
 const snapshot=('''+AUTO_SNAPSHOT_JS+r''')();
 const r=window.__splendorGameRecorder;
 const events=r?r.events.splice(0):[],dropped=r?.dropped??0;if(r)r.dropped=0;
 let bar=null;
 const el=document.querySelector('[data-testid="board-players"]');
 let f=el?.[Object.keys(el).find(k=>k.startsWith('__reactFiber$'))];
 for(;f;f=f.return){const p=f.memoizedProps;if(p?.state?.board&&p.state.players){
  if(p.showAnalysisBar&&Array.isArray(p.winRates))bar={p0:p.winRates[0]/100,p1:p.winRates[1]/100,
   source:'displayed_bar',position_alignment:'unverified_async'};break;
 }}
 return {snapshot,bar,events,dropped,pending:r?.pending??0,failure:r?.failure??null,at_ms:Date.now()};
}'''


def append_json(path,row,*,compressed=False):
    opener=gzip.open if compressed else open
    with opener(path,'at',encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


class ProgressRecorder:
    def __init__(self,output,level):
        self.output=output;self.level=level;self.seen=set();self.last_bar={}

    def snapshot(self,snapshot,at_ms,origin):
        if snapshot.get('unavailable'):return
        validate_game(snapshot,self.level)
        sig=identity(snapshot)
        if sig in self.seen:return
        self.seen.add(sig)
        path=game_directory(self.output,snapshot)
        append_json(path/'observations.jsonl.gz',{'at_ms':at_ms,'origin':origin,'snapshot_sha256':sig,
                    'ply':len(snapshot['moves']),'snapshot':snapshot},compressed=True)

    def consume(self,data):
        if data.get('failure'):raise RuntimeError('网站过程记录失败：'+data['failure'])
        if data['dropped']:raise RuntimeError('网站过程记录缓冲溢出，停止以免保存不完整数据')
        snapshot=data['snapshot'];at=data['at_ms']
        self.snapshot(snapshot,at,'poll')
        if not snapshot.get('unavailable') and data.get('bar'):
            game=snapshot['game_id'];key=(len(snapshot['moves']),data['bar']['p0'],data['bar']['p1'])
            if self.last_bar.get(game)!=key:
                append_json(game_directory(self.output,snapshot)/'displayed_bar.jsonl',
                            {'at_ms':at,'ply':len(snapshot['moves']),'human_seat':snapshot['perspective'],**data['bar']})
                self.last_bar[game]=key
        for event in data['events']:
            if event['kind']=='action_state':
                self.snapshot(event['snapshot'],event['completed_at_ms'],event['api'])
            elif event['kind']=='site_evaluation':
                seat=event['human_seat'];p0=event['p0']
                if seat not in (0,1) or not 0<=p0<=1:raise ValueError('网站胜率数值或座位无效')
                path=game_directory(self.output,{'game_id':event['game_id']})
                append_json(path/'site_evaluations.jsonl',{**event,'ply':len(event['moves']),
                    'viewer_rate':p0 if seat==0 else 1-p0,'source':'existing_site_evaluate',
                    'position_alignment':'request_moves'})
            elif event['kind']=='site_api_error':
                append_json(self.output/'site_api_errors.jsonl',event)

    async def sample(self,page):
        await page.evaluate(INSTALL_RECORDER_JS)
        data=await page.evaluate(READ_PROGRESS_JS);self.consume(data);return data

    async def close(self,page):
        await page.evaluate(CLEANUP_RECORDER_JS)

    async def observe(self,page):
        from playwright.async_api import Error
        while True:
            try:await self.sample(page)
            except Error as error:
                if 'Execution context was destroyed' not in str(error):raise
            await asyncio.sleep(.1)

    async def flush(self,page):
        # Drain the site's already-running evaluation before leaving a terminal
        # page. Never request another evaluation or make the local move wait for it.
        deadline=time.monotonic()+10
        await asyncio.sleep(.1)
        while True:
            data=await self.sample(page)
            if not data['pending'] or time.monotonic()>=deadline:return
            await asyncio.sleep(.1)

    def local(self,snapshot,result):
        path=game_directory(self.output,snapshot)
        value=result.get('value_actor_raw')
        append_json(path/'local_evaluations.jsonl',{
            'at_ms':round(time.time()*1000),'game_id':snapshot['game_id'],'ply':len(snapshot['moves']),
            'human_seat':snapshot['perspective'],'snapshot_sha256':identity(snapshot),'site_usi':result['site_usi'],
            'network_value':value,'network_rate':None if value is None else max(0.,min(1.,(value+1)/2)),
            'decision_rate':result.get('estimated_win_rate_actor'),'basis':result.get('win_rate_basis'),
            'search_seconds':result.get('total_elapsed_seconds',0.),'submitted_action':True})
