"""Play local-model moves through Kirameki's UI, then restart casual games."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

from .kirameki_live import (SNAPSHOT_JS, StateUnavailable, connect_browser, identity,
    recommend_snapshot, recommendation_text, request_from_snapshot, request_from_snapshot_gate, save_audit, save_result)


# Add only visible match metadata; never copy a worker, seed, deck or account token.
AUTO_SNAPSHOT_JS = r'''() => {
 const snapshot=('''+SNAPSHOT_JS+r''')();
 if(snapshot.unavailable)return snapshot;
 const el=document.querySelector('[data-testid="board-players"]');
 let f=el[Object.keys(el).find(k=>k.startsWith('__reactFiber$'))],view=null;
 for(;f;f=f.return)for(let h=f.memoizedState;h;h=h.next){
  const v=h.memoizedState;if(v?.state?.board&&Array.isArray(v.moves))view=v;
 }
 const record=JSON.parse(localStorage.getItem('lab.current')||'null');
 const started=Date.parse(record?.setup?.startedAt);
 return {...snapshot,game_id:Number.isFinite(started)?'casual-'+started:null,site_mode:view?.mode??null,
  opponent_level:view?.level_ids?.[1-snapshot.perspective]??null};
}'''

LOBBY_JS = r'''({level,start=false}) => {
 if(location.hostname!=='kirameki-lab.com'||location.pathname.replace(/\/$/,'')!=='/play')
  throw Error('不在Kirameki对局菜单');
 const ranked=[...document.querySelectorAll('input[type="checkbox"]')].find(e=>
  /ELO/i.test(e.closest('label')?.textContent||e.parentElement?.textContent||''));
 const opponent=[...document.querySelectorAll('select')].find(e=>[...e.options].some(o=>o.value===level));
 const seat=[...document.querySelectorAll('select')].find(e=>[...e.options].some(o=>o.value==='random'));
 const button=[...document.querySelectorAll('button')].find(e=>/^(Start game|対局開始)$/.test(e.textContent.trim()));
 if(!ranked||!opponent||!seat||!button)return {ready:false,message:'等待菜单中的非排位／对手设置和开始按钮'};
 if(ranked.checked)return {ready:false,message:'请先取消Ranked game／ELO排位'};
 if(opponent.value!==level)return {ready:false,message:'请先选择指定对手：'+level};
 if(button.disabled)return {ready:false,message:'等待开始按钮可用'};
 const renderedSeat=seat[Object.keys(seat).find(k=>k.startsWith('__reactProps$'))]?.value??null;
 if(start&&(seat.value!=='random'||renderedSeat!=='random'))return {ready:false,message:'等待Random先后手设置生效'};
 if(start)button.click();
 return {ready:true,ranked:false,opponent_level:opponent.value,human_seat:seat.value,rendered_human_seat:renderedSeat,started:start};
}'''

# Use the existing board callbacks and toolbar buttons. In particular the exact
# payment is selected before Buy; the website's default gold payment is not used
# as a substitute. A synchronous final snapshot check precedes every callback.
SUBMIT_JS = r'''async ({snapshot,usi}) => {
 const read=('''+AUTO_SNAPSHOT_JS+r''');
 const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 const canonical=x=>Array.isArray(x)?x.map(canonical):x&&typeof x==='object'?
  Object.fromEntries(Object.keys(x).sort().map(k=>[k,canonical(x[k])])):x;
 const same=(a,b)=>equal(canonical(a),canonical(b));
 const check=()=>{const s=read();if(!same(s,snapshot))throw Error('提交前盘面已变化，停止本次操作');
  if(s.ranked!==false||s.site_mode!=='human-vs-ai'||s.state.board.game_over||
    s.state.board.current_player!==s.perspective||s.isHumanTurn!==true)throw Error('当前不是非排位本人回合');};
 const board=()=>{const el=document.querySelector('[data-testid="board-players"]');
  let f=el?.[Object.keys(el).find(k=>k.startsWith('__reactFiber$'))];
  for(;f;f=f.return)if(f.memoizedProps?.state?.board&&f.memoizedProps.state.players)return f.memoizedProps;
  throw Error('无法取得网站棋盘回调');};
 const component=(test,selector='button')=>{for(const el of document.querySelectorAll(selector)){
  let f=el[Object.keys(el).find(k=>k.startsWith('__reactFiber$'))];
  for(;f;f=f.return)if(test(f.memoizedProps))return f.memoizedProps;
 }return null;};
 const panel=()=>component(p=>Array.isArray(p?.buttons)&&typeof p.onCancel==='function');
 const confirm=()=>component(p=>typeof p?.onYes==='function'&&typeof p?.onNo==='function','[data-testid="play-confirm"]');
 const wait=async test=>{for(let i=0;i<80;i++){check();if(test())return;await new Promise(r=>setTimeout(r,25));}
  throw Error('网站操作面板未按预期更新');};
 const invoke=fn=>{check();if(typeof fn!=='function')throw Error('缺少网站动作回调');fn();};
 const press=async tone=>{await wait(()=>{const p=panel();return p&&!p.busy&&p.buttons.filter(b=>b.tone===tone&&b.enabled).length===1;});
  invoke(panel().buttons.find(b=>b.tone===tone&&b.enabled).onClick);};
 check();
 const actions=snapshot.state.legal_actions,found=actions.filter(a=>a.usi===usi);
 if(found.length!==1)throw Error('建议不对应唯一合法网站动作');
 const a=found[0],key=JSON.stringify([snapshot.game_id,snapshot.moves,usi]);
 if(window.__splendorAutoAttempt===key)throw Error('本次动作已经尝试提交，禁止重发');
 window.__splendorAutoAttempt=key;
 let p=panel();if(p?.canCancel){invoke(p.onCancel);await wait(()=>!panel()?.canCancel);}
 if(a.type===5){invoke(()=>board().onNobleClick(a.noble_choice));return {attempted:true,usi};}
 if(a.type===6){await press('neutral');return {attempted:true,usi};}
 if(a.type===0||a.type===1){
  for(let c=0;c<5;c++)for(let n=0;n<a.take[c];n++){
   const old=(board().selectedGems||[]).length;invoke(()=>board().onGemClick(c));
   await wait(()=>(board().selectedGems||[]).length===old+1);
  }
 }else if(a.type===2||a.type===4){
  invoke(()=>a.from_reserved?board().onReservedCardClick(a.card_id):board().onCardClick(a.card_id));
  await wait(()=>board().selectedCardId===a.card_id);
 }else if(a.type===3){
  invoke(()=>board().onDeckClick(a.deck_level+1));
  await wait(()=>board().selectedDeckLevel===a.deck_level+1);
 }else if(a.type!==7)throw Error('不支持的网站动作类型：'+a.type);
 if(a.type===4){
  // Search the finite payment selector graph, using the website's next-value /
  // minimum-other-colour-change rule. This also reaches extra-gold payments.
  const pays=actions.filter(x=>x.type===4&&x.card_id===a.card_id).map(x=>x.pay);
  const start=board().payingGems,target=a.pay;
  if(!Array.isArray(start))throw Error('网站未显示购买支付');
  const queue=[{pay:start,path:[]}],seen=new Set([JSON.stringify(start)]);let plan=null;
  for(let i=0;i<queue.length;i++){
   const item=queue[i];if(equal(item.pay,target)){plan=item.path;break;}
   for(let c=0;c<6;c++){
    const values=[...new Set(pays.map(v=>v[c]))].sort((a,b)=>a-b);
    const next=values[(values.indexOf(item.pay[c])+1)%values.length];
    const candidates=pays.filter(v=>v[c]===next);
    const distance=v=>v.reduce((sum,n,k)=>sum+(k===c?0:Math.abs(n-item.pay[k])),0);
    const pay=candidates.reduce((best,v)=>distance(v)<distance(best)?v:best);
    const id=JSON.stringify(pay);if(!seen.has(id)){seen.add(id);queue.push({pay,path:[...item.path,{c,pay}]});}
   }
  }
  if(plan===null)throw Error('网站支付选择器无法到达模型指定的支付');
  for(const step of plan){invoke(()=>board().onResourceClick(step.c));await wait(()=>equal(board().payingGems,step.pay));}
  if(!equal(board().payingGems,target))throw Error('网站支付与模型指定不一致');
  await press('primary');return {attempted:true,usi};
 }
 const returns=a.return_gems||[0,0,0,0,0,0];
 if(a.type!==7){
  await press(a.type===2||a.type===3?'reserve':'primary');
  if((a.type===2||a.type===3)&&snapshot.state.board.bank[5]===0){
   await wait(()=>confirm()!==null);invoke(confirm().onYes);
  }
  if(!returns.some(Boolean))return {attempted:true,usi};
 }
 await wait(()=>typeof board().isReturnSelectable==='function');
 for(let c=0;c<6;c++)for(let n=0;n<returns[c];n++){
  const old=board().returningGems[c];invoke(()=>board().onResourceClick(c));
  await wait(()=>board().returningGems[c]===old+1);
 }
 if(!equal(board().returningGems,returns))throw Error('网站归还与模型指定不一致');
 await press('danger');return {attempted:true,usi};
}'''


def validate_game(snapshot,level):
    if snapshot.get('unavailable'):raise StateUnavailable(snapshot['unavailable'])
    if snapshot.get('ranked') is not False:raise StateUnavailable('只操作已确认的非排位对局')
    if snapshot.get('site_mode')!='human-vs-ai':raise StateUnavailable('只支持本人对网站AI的对局')
    if snapshot.get('opponent_level')!=level:raise StateUnavailable('网站对手不是指定等级：'+level)
    if snapshot.get('human_seat')!=snapshot.get('perspective') or snapshot.get('perspective') not in (0,1):
        raise StateUnavailable('无法确认本人座位')
    if not isinstance(snapshot.get('game_id'),str) or not snapshot['game_id']:
        raise ValueError('网站未提供对局ID')


def game_directory(output,snapshot):
    # IDs are file names supplied by the site, never paths.
    game_id=snapshot['game_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',game_id):raise ValueError('网站对局ID格式无效')
    path=output/'games'/game_id;path.mkdir(parents=True,exist_ok=True)
    return path


def append_event(path,event):
    with (path/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'at':time.time(),**event},ensure_ascii=False,allow_nan=False)+'\n')


async def wait_for_ack(page,before,usi,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        after=await page.evaluate(AUTO_SNAPSHOT_JS)
        if after.get('unavailable'):raise RuntimeError('提交后无法读取原对局；为防止重复落子，已停止')
        if after.get('game_id')!=before['game_id']:raise RuntimeError('提交期间切换了对局，已停止')
        moves=after.get('moves',[]);prefix=before['moves']
        if moves[:len(prefix)]!=prefix:raise RuntimeError('提交后棋谱前缀改变，已停止')
        if len(moves)>len(prefix):
            if moves[len(prefix)]!=usi:raise RuntimeError('网站实际落子与模型建议不一致，已停止')
            return after
        await asyncio.sleep(.05)
    raise RuntimeError('动作提交后未在时限内记入网站棋谱；已停止，不自动重发，请检查网页')


def finish_game(path,snapshot):
    board=snapshot['state']['board'];winner=board['winner'];viewer=snapshot['perspective']
    # Site engine: -1 is ongoing, -2 is a completed draw; 0/1 are winners.
    if not board['game_over'] or winner not in (-2,0,1):raise ValueError('网站终局结果不完整')
    result={'game_id':snapshot['game_id'],'ranked':False,'opponent_level':snapshot['opponent_level'],
            'human_seat':viewer,'winner':winner,'outcome':'draw' if winner==-2 else 'win' if winner==viewer else 'loss',
            'points':[p['points'] for p in snapshot['state']['players']],
            'moves':snapshot['moves'],'terminal_snapshot':snapshot}
    (path/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def recover_previous(path,snapshot,catalog):
    choices=[]
    for file in (path/'decisions').glob('*/recommendation.json'):
        result=json.loads(file.read_text())
        if not result.get('submitted_action') or result.get('request',{}).get('observation',{}).get('phase')!='main_action':continue
        before=json.loads((file.parent/'snapshot.json').read_text());n=len(before['moves'])
        if snapshot['moves'][:n+1]==before['moves']+[result['site_usi']]:choices.append((n,before))
    return request_from_snapshot(max(choices,key=lambda x:x[0])[1],catalog) if choices else None


# Exact inputs from the release before shared CDP ownership. This adapter
# upgrade also corrects site draw decoding; search/model/recording must match.
_BEFORE_SHARED_CONNECTION={
    'kirameki_suite.py':'3501a73c8cc89762f81e44d55d8f32601fbc942edc25e7bb0da16853f34f05e4',
    'kirameki_autoplay.py':'c3fa10f2f47c0b80dd73a095d6803efedba3503f105557e3482387d1d36bec06',
    'kirameki_live.py':'6cfc95d242bf5c061a65e7c7be1c0e26ed535ec6352db23fe04867366e61e9c5'}


def upgrade_connection_sources(path,old,sources):
    """Permit the known adapter upgrade; retain the original identity."""
    before=old.get('source_sha256',{})
    if before.keys()!=sources.keys():return False
    changed=[name for name in before if before[name]!=sources[name]]
    if not changed or any(before[name]!=_BEFORE_SHARED_CONNECTION.get(name) for name in changed):return False
    backup=path.with_name(path.stem+'.before_shared_connection.json')
    if backup.exists() and json.loads(backup.read_text())!=old:
        raise ValueError('连接升级的原配置备份已存在且内容不符：'+str(backup))
    backup.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path.write_text(json.dumps({**old,'source_sha256':sources},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('已升级为共享浏览器连接，原配置保存在：'+str(backup),flush=True)
    return True


def prepare_session(args,advisor):
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(checkpoint_sha256=advisor.checkpoint_sha256,argv=sys.argv,started_at=time.time())
    sources=['kirameki_autoplay.py','kirameki_live.py','parallel_recommend.py','native_policy.py',
             'native_search.cpp','search_inference.py','recommend.py']
    if getattr(args,'record_progress',False):sources.append('kirameki_recording.py')
    config['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
    manifest=args.output/'session.json';assigned_path=args.output/'assigned_games.json'
    assigned=set();finished={}
    if getattr(args,'resume',False) and manifest.exists():
        old=json.loads(manifest.read_text())
        fields=('checkpoint_sha256','mode','simulations_per_world','determinizations','leaf_batch_size','root_selection',
                'seed','search_backend','inference_mode','compile_mode','device','opponent_level','record_progress',
                'clock','source_sha256','fresh_games_only')
        changed=[k for k in fields if old.get(k)!=config.get(k)]
        if changed==['source_sha256'] and upgrade_connection_sources(manifest,old,config['source_sha256']):changed=[]
        if changed:raise ValueError('恢复目录的模型／搜索／源码配置不一致：'+','.join(changed))
        assigned=set(json.loads(assigned_path.read_text())) if assigned_path.exists() else set()
        for game_id in assigned:
            result_path=game_directory(args.output,{'game_id':game_id})/'result.json'
            if result_path.exists():finished[game_id]=json.loads(result_path.read_text())
    manifest.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return assigned,finished


def write_summary(output,finished):
    counts={kind:sum(r['outcome']==kind for r in finished.values()) for kind in ('win','draw','loss')}
    n=len(finished)
    summary={'completed_games':n,**counts,'games':list(finished),
             'win_rate':counts['win']/n if n else None,
             'score_rate':(counts['win']+.5*counts['draw'])/n if n else None}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return counts


async def pick_page(browser):
    pages=[p for c in browser.contexts for p in c.pages if urlparse(p.url).hostname=='kirameki-lab.com'
           and urlparse(p.url).path.rstrip('/') in ('/play','/play/game')]
    if len(pages)>1:
        pages=[p for p in pages if await p.evaluate('() => document.visibilityState === "visible"')]
    if len(pages)!=1:raise StateUnavailable('请只保留一个目标对局，或切换到目标对局标签页')
    return pages[0]


@asynccontextmanager
async def browser_session(args,browser=None):
    """Only the creator disconnects; suite groups borrow one persistent client."""
    from playwright.async_api import async_playwright
    if browser is not None:
        yield browser
        return
    async with async_playwright() as p:
        owned=await connect_browser(p,args)
        try:yield owned
        finally:await owned.close()


async def watch_autoplay(args,advisor,*,browser=None,on_game_finished=None):
    from playwright.async_api import Error as PlaywrightError
    args.output.mkdir(parents=True,exist_ok=True)
    assigned,finished=prepare_session(args,advisor)
    if args.games and len(finished)>=args.games:
        write_summary(args.output,finished);print('本组已完成目标局数。',flush=True);return
    previous=None;current_game=None;notice=None;processed=set(finished);seen_active=set(assigned)
    recorder=None;record_task=None;record_page=None;launched=False
    if getattr(args,'record_progress',False):
        from .kirameki_recording import ProgressRecorder
        recorder=ProgressRecorder(args.output,args.opponent_level)
    async with browser_session(args,browser) as browser:
        try:
            while True:
                try:
                    page=await pick_page(browser)
                    if record_task is not None and record_task.done():record_task.result()
                    if recorder is not None and record_page is not page:
                        if record_task is not None:
                            record_task.cancel()
                            with suppress(asyncio.CancelledError):await record_task
                        record_page=page;record_task=asyncio.create_task(recorder.observe(page))
                    if urlparse(page.url).path.rstrip('/')=='/play':
                        lobby=await page.evaluate(LOBBY_JS,{'level':args.opponent_level})
                        if not lobby['ready']:raise StateUnavailable(lobby['message'])
                        if getattr(args,'clock','keep')=='unlimited':
                            await page.locator('select:has(option[value="unlimited"])').select_option('unlimited')
                        if recorder is not None:
                            await page.get_by_role('checkbox',name=re.compile(r'^(Show the win-rate bar|勝率バーを表示する)$')).check()
                        await page.locator('select:has(option[value="random"])').select_option('random')
                        await page.wait_for_function('cfg=>{const s=('+LOBBY_JS+")(cfg);return s.human_seat==='random'&&s.rendered_human_seat==='random';}",
                                                     arg={'level':args.opponent_level},timeout=5000)
                        print('开始非排位对局，对手：'+args.opponent_level+'；先后手：Random',flush=True)
                        started=await page.evaluate(LOBBY_JS,{'level':args.opponent_level,'start':True})
                        if not started.get('started'):raise StateUnavailable(started['message'])
                        launched=True
                        await page.wait_for_url('**/play/game',timeout=30000)
                        previous=None;current_game=None;notice=None
                        continue
                    snapshot=await page.evaluate(AUTO_SNAPSHOT_JS)
                    validate_game(snapshot,args.opponent_level)
                    if snapshot['game_id']!=current_game:
                        current_game=snapshot['game_id'];previous=None
                        if getattr(args,'resume',False):previous=recover_previous(game_directory(args.output,snapshot),snapshot,advisor.catalog)
                    path=game_directory(args.output,snapshot)
                    if snapshot['state']['board']['game_over']:
                        if recorder is not None:await recorder.flush(page)
                        if current_game not in processed:
                            result=finish_game(path,snapshot);processed.add(current_game)
                            if current_game in seen_active:finished[current_game]=result
                            append_event(path,{'event':'finished','outcome':result['outcome'],'points':result['points']})
                            counts=write_summary(args.output,finished)
                            print(f'终局：{result["outcome"]}，分数{result["points"]}；累计{counts}',flush=True)
                            if on_game_finished is not None:on_game_finished()
                        if args.games and len(finished)>=args.games:return
                        await asyncio.sleep(args.restart_delay)
                        # The site's Menu preserves the selected opponent/clock in its URL.
                        await page.get_by_role('button',name=re.compile(r'^(Menu|メニューへ)$')).click(timeout=5000)
                        await page.wait_for_url(re.compile(r'https://kirameki-lab\.com/play(?:\?.*)?$'),timeout=10000)
                        previous=None;current_game=None;notice=None
                        continue
                    if current_game not in assigned:
                        if getattr(args,'fresh_games_only',False) and not launched and len(snapshot['moves'])>snapshot['perspective']:
                            raise RuntimeError('当前棋局已中途开始且不属于本组；请完成原局或回菜单后再启动，避免混合策略')
                        assigned.add(current_game)
                        (args.output/'assigned_games.json').write_text(json.dumps(sorted(assigned))+'\n')
                        launched=False
                    seen_active.add(current_game)
                    request_from_snapshot_gate(snapshot)
                    sig=identity(snapshot)
                    await asyncio.sleep(args.settle)
                    if identity(await page.evaluate(AUTO_SNAPSHOT_JS))!=sig:continue
                    print(f'[{time.strftime("%H:%M:%S")}] 对局{current_game}，第{len(snapshot["moves"])+1}步，搜索中…',flush=True)
                    result,request=await asyncio.to_thread(recommend_snapshot,snapshot,advisor,previous=previous,top_k=args.top_k)
                    if identity(await page.evaluate(AUTO_SNAPSHOT_JS))!=sig:
                        result['status']='discarded_stale';save_audit(path,snapshot,result);continue
                    if record_task is not None and record_task.done():record_task.result()
                except StateUnavailable as error:
                    if str(error)!=notice:print(str(error),flush=True);notice=str(error)
                    await asyncio.sleep(args.poll);continue
                except PlaywrightError as error:
                    if 'Execution context was destroyed' not in str(error):raise
                    await asyncio.sleep(args.poll);continue
                # No navigation/error retry around submission: an ambiguous send must
                # stop, so the same action cannot be applied a second time.
                result.update(game_id=current_game,site='kirameki-lab.com',observed_at=time.time(),status='prepared')
                save_audit(path,snapshot,result)
                append_event(path,{'event':'prepared','snapshot_sha256':sig,'site_usi':result['site_usi']})
                print('执行：'+recommendation_text(result),flush=True)
                try:
                    await page.evaluate(SUBMIT_JS,{'snapshot':snapshot,'usi':result['site_usi']})
                    after=await wait_for_ack(page,snapshot,result['site_usi'],args.action_timeout)
                except BaseException as error:
                    append_event(path,{'event':'submission_unconfirmed','site_usi':result['site_usi'],'error':str(error)})
                    raise
                result.update(submitted_action=True,status='submitted',acknowledged_moves=len(after['moves']))
                save_audit(path,snapshot,result);save_result(args.output,result)
                if recorder is not None:recorder.local(snapshot,result)
                append_event(path,{'event':'submitted','site_usi':result['site_usi'],'snapshot_sha256':sig})
                previous=request;notice=None
                if args.once:return
                await asyncio.sleep(args.poll)
        finally:
            try:
                if record_task is not None:
                    record_task.cancel()
                    with suppress(asyncio.CancelledError):await record_task
            finally:
                if recorder is not None and record_page is not None:
                    with suppress(PlaywrightError):await recorder.close(record_page)


def main(argv=None):
    from .kirameki_live import main as live_main
    live_main(argv,autoplay=True)


if __name__=='__main__':main()
