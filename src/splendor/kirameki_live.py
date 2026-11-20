"""Attach to an existing Kirameki casual game and recommend, never submit."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import torch

from .actions import (BuyMarket, BuyReserved, ClaimNoble, Pay, ReserveDeck,
                      ReserveMarket, ReturnTokens, TakeDistinct, TakePair)
from .bga_watch import cdp_endpoint
from .catalog import load_default_catalog
from .contracts import (MarketView, Observation, PendingObservation,
    PlayerPrivateState, PlayerPublicState, PublicReservedCard)
from .model import (DecisionPhase, GEM_COLORS, GemVector, ReservationSource,
                    Tier, TokenVector)
from .observation import _card_view, _noble_view, build_observation
from .recommend import LocalAdvisor, describe_action
from .replay import to_primitive
from .rules import apply_legal_action
from .search_policy import _determinized_state, request_from_observation

DEFAULT_CHECKPOINT = Path('runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-008/checkpoints/final.pt')
SITE_CARD_IDS = (
    '015','013','014','011','012','009','010','016',
    '039','037','038','035','036','033','034','040',
    '007','005','006','003','004','001','002','008',
    '023','021','022','019','020','017','018','024',
    '031','029','030','027','028','025','026','032',
    '047','048','051','049','050','052',
    '065','066','069','068','067','070',
    '041','042','045','044','043','046',
    '053','054','057','056','055','058',
    '060','059','063','062','061','064',
    '075','076','077','078','087','088','089','090',
    '071','072','073','074','079','080','081','082',
    '083','084','085','086')
SITE_NOBLE_IDS = ('N01','N08','N05','N04','N03','N12',
                  'N02','N07','N10','N06','N09','N11')

# Only copy public data and the viewer's own hand. Never return a seed, deck
# order, full React view, localStorage record, or opponent blind identity.
SNAPSHOT_JS = r'''() => {
 const el=document.querySelector('[data-testid="board-players"]');
 if(!el)return {unavailable:'等待打开对局盘面'};
 let f=el[Object.keys(el).find(k=>k.startsWith('__reactFiber$'))];
 let props=null,view=null;
 for(;f;f=f.return){
  const p=f.memoizedProps;
  if(!props&&p?.state?.board&&p.state.players)props=p;
  for(let h=f.memoizedState;h;h=h.next){
   const v=h.memoizedState;
   if(v?.state?.board&&Array.isArray(v.moves))view=v;
  }
 }
 if(!props)return {unavailable:'无法识别当前盘面组件'};
 const record=JSON.parse(localStorage.getItem('lab.current')||'null');
 if(!record||typeof record.ranked!=='boolean')return {unavailable:'无法确认非排位状态'};
 if(record.ranked)return {unavailable:'排位对局不提供辅助'};
 const s=props.state,b=s.board,viewer=props.perspective;
 const pub=props.publicReservedCardIds||[],hidden=props.hiddenReservedLevels||{};
 const players=s.players.map((p,seat)=>({index:p.index,gems:p.gems,
  bonuses:p.bonuses,points:p.points,purchased_cards:p.purchased_cards,
  acquired_nobles:p.acquired_nobles,
  reserved_cards:p.reserved_cards.map((id,slot)=>
    seat===viewer||pub.includes(id)?id:-1)}));
 const board={};
 for(const k of ['bank','visible_cards','deck_counts','nobles','current_player',
  'turn','waiting_noble','waiting_return','game_over','winner'])board[k]=b[k];
 return {schema:'kirameki-visible-v1',ranked:false,perspective:viewer,
  human_seat:view?.human_seat??null,isHumanTurn:props.isHumanTurn,
  game_id:record.id??null,moves:view?view.moves.map(m=>m.usi):record.moves,
  publicReservedCardIds:pub,hiddenReservedLevels:hidden,
  state:{board,players,legal_actions:b.current_player===viewer?s.legal_actions:[]}};
}'''


class StateUnavailable(ValueError):
    """An expected waiting state; no recommendation may be issued."""


def identity(snapshot):
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


def site_action(action):
    """Project a site atomic action onto the corresponding native decision."""
    kind=action['type']
    if kind==0:
        return TakeDistinct(tuple(c for c,n in zip(GEM_COLORS,action['take'][:5]) if n))
    if kind==1:
        return TakePair(next(c for c,n in zip(GEM_COLORS,action['take'][:5]) if n))
    if kind==2:return ReserveMarket(SITE_CARD_IDS[action['card_id']])
    if kind==3:return ReserveDeck(Tier(action['deck_level']+1))
    if kind==4:
        return (BuyReserved if action['from_reserved'] else BuyMarket)(SITE_CARD_IDS[action['card_id']])
    if kind==5:return ClaimNoble(SITE_NOBLE_IDS[action['noble_choice']])
    if kind==7:return ReturnTokens(TokenVector(*action['return_gems']))
    raise ValueError(f'不支持的网站动作类型：{kind}')


def last_main(snapshot, previous=None):
    for usi in reversed(snapshot.get('moves',[])):
        head=usi.split('/')[0]
        if head.startswith('return:') or head.startswith('noble:'):continue
        if head.startswith('take:'):
            letters=head.split(':')[1]
            colors=tuple(GEM_COLORS['WUGRK'.index(c)] for c in letters)
            return TakePair(colors[0]) if len(colors)==2 and colors[0]==colors[1] else TakeDistinct(colors)
        if head.startswith('reserve:L'):return ReserveDeck(Tier(int(head[9:])))
        if head.startswith('reserve:C'):return ReserveMarket(SITE_CARD_IDS[int(head[9:])])
        if head.startswith('buy:C'):
            card=SITE_CARD_IDS[int(head[5:])]
            if previous is not None:
                own=previous.observation.self_public
                current=snapshot['state']['players'][snapshot['perspective']]
                purchased={SITE_CARD_IDS[i] for i in current['purchased_cards']}
                if purchased!=set(own.purchased_card_ids)|{card} or card in own.purchased_card_ids:
                    raise StateUnavailable('缺少本次购买前的完整盘面，请手动完成当前贵族选择')
                reserved={r.card.card_id for r in own.reserved_cards if r.card is not None}
                reserved.update(c.card_id for c in previous.observation.self_private.blind_reserved_cards)
                return (BuyReserved if card in reserved else BuyMarket)(card)
            raise StateUnavailable('请手动完成当前贵族选择；随后即可推荐（缺少购买前的来源记录）')
        break
    raise StateUnavailable('无法恢复本手主行动，请手动完成当前归还／贵族阶段')


def request_from_snapshot(snapshot, catalog=None, *, previous=None):
    if snapshot.get('unavailable'):raise StateUnavailable(snapshot['unavailable'])
    if snapshot.get('ranked') is not False:raise StateUnavailable('仅支持已确认的非排位对局')
    catalog=catalog or load_default_catalog()
    s=snapshot['state']; b=s['board']; viewer=snapshot['perspective']
    if viewer not in (0,1) or snapshot.get('human_seat',viewer) not in (None,viewer):
        raise StateUnavailable('请打开本人参与的双人对局')
    if b['game_over']:raise StateUnavailable('对局已结束，结果以网站为准')
    if b['current_player']!=viewer:raise StateUnavailable('等待对手行动')
    if snapshot.get('isHumanTurn') is False:raise StateUnavailable('等待网站完成当前行动')
    public=set(snapshot['publicReservedCardIds']); hidden=snapshot['hiddenReservedLevels']
    cv=lambda i:_card_view(catalog.card(SITE_CARD_IDS[i]))
    nv=lambda i:_noble_view(catalog.noble(SITE_NOBLE_IDS[i]))
    players=[]; own_blind=[]
    for seat,p in enumerate(s['players']):
        reservations=[]
        for slot,card in enumerate(p['reserved_cards']):
            level=hidden.get(f'{seat}:{slot}')
            if card>=0 and card in public:
                view=cv(card); reservations.append(PublicReservedCard(view.tier,ReservationSource.MARKET,view))
            elif level is not None or (seat==viewer and card>=0):
                # The site only annotates hidden slots for the opponent;
                # the viewer knows the identity (and tier) of their own card.
                tier=Tier(level) if level is not None else cv(card).tier
                reservations.append(PublicReservedCard(tier,ReservationSource.DECK,None))
                if seat==viewer:
                    if card<0:raise StateUnavailable('本人盲保留牌尚未显示')
                    view=cv(card)
                    if view.tier!=tier:raise ValueError('保留牌等级不一致')
                    own_blind.append(view)
            elif card>=0:
                raise ValueError('保留牌缺少公开来源／盲留等级')
        reservations.sort(key=lambda r:(r.source.value,r.tier.value,r.card.card_id if r.card else ''))
        players.append(PlayerPublicState(TokenVector(*p['gems']),GemVector(*p['bonuses']),p['points'],
            tuple(sorted(SITE_CARD_IDS[i] for i in p['purchased_cards'])),
            tuple(sorted(SITE_NOBLE_IDS[i] for i in p['acquired_nobles'])),tuple(reservations)))
    phase=DecisionPhase.RETURN_TOKENS if b['waiting_return'] else (
        DecisionPhase.CHOOSE_NOBLE if b['waiting_noble'] else DecisionPhase.MAIN_ACTION)
    pending=None
    if phase is not DecisionPhase.MAIN_ACTION:
        main=last_main(snapshot,previous)
        eligible=tuple(sorted((nv(a['noble_choice']) for a in s['legal_actions'] if a['type']==5),key=lambda n:n.noble_id))
        pending=PendingObservation(main,tokens_to_return=max(0,players[viewer].tokens.total-10),eligible_nobles=eligible)
    counts=(b['turn']+int(b['current_player']==1),b['turn'])
    purchased=set(players[0].purchased_card_ids+players[1].purchased_card_ids)
    # The first player can trigger the final reply only after completing a turn.
    triggered=b['current_player']==1 and players[0].score>=15
    streak=0
    for usi in reversed(snapshot.get('moves',[])):
        if usi=='pass':streak+=1
        elif usi.startswith(('return:','noble:')):continue
        else:break
    obs=Observation(phase,viewer==0,TokenVector(*b['bank']),tuple(b['deck_counts']),
        MarketView(*(tuple(sorted((cv(i) for i in row if i>=0),key=lambda c:c.card_id)) for row in b['visible_cards'])),
        tuple(sorted((nv(i) for i in b['nobles']),key=lambda n:n.noble_id)),
        players[viewer],PlayerPrivateState(tuple(sorted(own_blind,key=lambda c:c.card_id))),players[1-viewer],
        tuple(c.card_id in purchased for c in catalog.cards),counts[viewer],counts[1-viewer],streak,
        triggered,triggered and viewer==0,pending)
    request=request_from_observation(obs,catalog=catalog)
    if phase is not DecisionPhase.RETURN_TOKENS:
        projected={site_action(a) for a in s['legal_actions'] if a['type']!=6}
        if projected!=set(request.legal_actions):
            raise ValueError(f'网站与本地合法动作不一致：网站额外{projected-set(request.legal_actions)}；本地额外{set(request.legal_actions)-projected}')
    else:
        actual={tuple(a['return_gems']) for a in s['legal_actions']}
        expected={tuple(int(i==j) for i in range(6)) for j,n in enumerate(obs.self_public.tokens) if n}
        if actual!=expected:raise ValueError('网站归还单枚候选与持有筹码不一致')
    return request


def recommend_snapshot(snapshot, advisor, *, previous=None, top_k=3):
    started=time.perf_counter()
    actions=snapshot['state']['legal_actions']; catalog=advisor.catalog
    # A site forced pass or unique noble needs no neural decision.
    if len(actions)==1 and actions[0]['type'] in (5,6):
        request_from_snapshot_gate(snapshot)
        a=actions[0]
        description='跳过（唯一合法行动）' if a['type']==6 else describe_action(site_action(a),catalog)
        return {'description':description,'site_usi':a['usi'],'submitted_action':False,'forced':True},None
    request=request_from_snapshot(snapshot,catalog,previous=previous)
    result=advisor.recommend(request,top_k=top_k)
    result['request']=to_primitive(request)
    chosen=request.legal_actions[result['recommended_candidate_index']]
    matches=[a for a in actions if site_action(a)==chosen] if not isinstance(chosen,ReturnTokens) else []
    if isinstance(chosen,(BuyMarket,BuyReserved,TakeDistinct,TakePair)):
        state=_determinized_state(request.observation,catalog,seed=0)
        follow=apply_legal_action(state,chosen,catalog)
        if follow.current_player==0 and follow.phase in (DecisionPhase.PAYMENT,DecisionPhase.RETURN_TOKENS):
            follow_request=request_from_observation(build_observation(follow,0,catalog),catalog=catalog)
            extra=advisor.recommend(follow_request,top_k=top_k)
            extra['request']=to_primitive(follow_request)
            detail=follow_request.legal_actions[extra['recommended_candidate_index']]
            key='pay' if isinstance(detail,Pay) else 'return_gems'
            matches=[a for a in matches if tuple(a[key])==tuple(detail.tokens)]
            result['followup']=extra
            result['description']+='；'+extra['description']
    if isinstance(chosen,ReturnTokens):
        # The website returns one token per click. Recompute after each real click.
        color=next(i for i,n in enumerate(chosen.tokens) if n)
        matches=[a for a in actions if a['return_gems'][color]==1]
        result['description']+='（网站逐枚归还；当前先归还'+('白','蓝','绿','红','黑','金')[color]+'1）'
    if not matches:raise ValueError('模型建议无法映射到当前网站合法动作')
    result.update(site_usi=matches[0]['usi'],site_action=matches[0],snapshot_sha256=identity(snapshot),
                  total_elapsed_seconds=time.perf_counter()-started)
    return result,request


def request_from_snapshot_gate(snapshot):
    if snapshot.get('ranked') is not False:raise StateUnavailable('仅支持非排位')
    b=snapshot['state']['board']
    if b['game_over'] or b['current_player']!=snapshot['perspective'] or snapshot.get('isHumanTurn') is False:
        raise StateUnavailable('等待本人行动')


def recommendation_text(result):
    lines=[result['description']]
    if 'site_usi' in result:lines.append('网站动作：'+result['site_usi'])
    if 'estimated_win_rate_actor' in result:
        basis={'world_mean_q_actor':'推荐动作等权Q',
               'search_q_actor':'推荐动作搜索Q','value_actor_raw':'网络V'}[result['win_rate_basis']]
        lines.append(f'预估胜率（本人，{basis}，和棋折半）：{result["estimated_win_rate_actor"]:.1%}')
    if 'value_actor_raw' in result:
        lines.append(f'V={result["value_actor_raw"]:+.4f}；推荐总耗时{result["total_elapsed_seconds"]:.1f}秒')
    for row in result.get('candidates',[]):
        share=row.get('visit_share')
        q=row.get('world_mean_q_actor')
        quality=(f' · 等权Q={q:+.4f} · 每世界最少{min(row["per_world_visits"])}次' if q is not None else '')
        lines.append('  '+row['description']+quality+(f' · 访问份额{share:.1%}' if share is not None else ''))
    return '\n'.join(lines)


def save_result(directory, result):
    directory.mkdir(parents=True,exist_ok=True)
    for name,content in [('latest.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),
                         ('latest.md',recommendation_text(result)+'\n')]:
        path=directory/name; temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(content,encoding='utf-8'); temp.replace(path)


def save_audit(directory,snapshot,result=None):
    path=directory/'decisions'/identity(snapshot)
    path.mkdir(parents=True,exist_ok=True)
    (path/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if result is not None:
        (path/'recommendation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


async def connect_browser(playwright,args):
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    endpoint=cdp_endpoint(args.cdp,args.profile)
    print('正在连接已有浏览器（最长等待60秒）；若浏览器弹出调试连接授权，请确认本机推荐器的连接。',flush=True)
    try:
        browser=await playwright.chromium.connect_over_cdp(endpoint,no_defaults=True,timeout=60000)
    except PlaywrightTimeoutError:
        raise SystemExit('连接已有浏览器超时，尚未读取盘面或开始搜索。请检查 Edge 的调试连接授权；'
                         '也可在 edge://inspect/#remote-debugging 检查远程调试状态，再重跑原命令。'
                         f'\n连接地址：{endpoint}') from None
    print('已连接已有浏览器，等待可推荐的盘面。',flush=True)
    return browser


async def watch(args,advisor):
    from playwright.async_api import async_playwright
    from playwright.async_api import Error as PlaywrightError
    async with async_playwright() as p:
        browser=await connect_browser(p,args)
        previous=None; last=None; notice=None
        try:
            while True:
                pages=[page for c in browser.contexts for page in c.pages
                       if urlparse(page.url).hostname=='kirameki-lab.com' and urlparse(page.url).path.startswith('/play')]
                games=[page for page in pages if urlparse(page.url).path.rstrip('/')=='/play/game']
                pages=games or pages
                try:
                    if len(pages)>1:
                        visible=[page for page in pages if await page.evaluate('() => document.visibilityState === "visible"')]
                        pages=visible
                    if len(pages)!=1:raise StateUnavailable('请只保留一个目标对局，或切换到目标对局标签页')
                    page=pages[0]; snapshot=await page.evaluate(SNAPSHOT_JS)
                    if snapshot.get('unavailable'):raise StateUnavailable(snapshot['unavailable'])
                    request_from_snapshot_gate(snapshot)
                    sig=identity(snapshot)
                    if sig==last:
                        await asyncio.sleep(args.poll);continue
                    await asyncio.sleep(args.settle)
                    if identity(await page.evaluate(SNAPSHOT_JS))!=sig:continue
                    audit_path=save_audit(args.output,snapshot)
                    forced=len(snapshot['state']['legal_actions'])==1 and snapshot['state']['legal_actions'][0]['type'] in (5,6)
                    if not forced:
                        model_request=request_from_snapshot(snapshot,advisor.catalog,previous=previous)
                        (audit_path/'request.json').write_text(json.dumps(to_primitive(model_request),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                    own=snapshot['state']['players'][snapshot['perspective']]
                    held=[SITE_CARD_IDS[i] for i in own['reserved_cards'] if i>=0]
                    print(f'盘面：本人P{snapshot["perspective"]}，第{snapshot["state"]["board"]["turn"]+1}轮，筹码{own["gems"]}，保留{held}；快照{sig[:12]}',flush=True)
                    chance=advisor.mode=='mcts' and args.search_backend=='native_chance'
                    label=(f'随机节点搜索中：单共享树，总预算{args.simulations_per_world}' if chance else
                           f'搜索中：S{args.simulations_per_world}N{args.determinizations}，总预算{args.simulations_per_world*args.determinizations}'
                           if advisor.mode=='mcts' else 'greedy推理中（零搜索）')
                    print(f'[{time.strftime("%H:%M:%S")}] {label}…',flush=True)
                    # Leave CDP/event processing responsive while native search runs.
                    result,request=await asyncio.to_thread(recommend_snapshot,snapshot,advisor,previous=previous,top_k=args.top_k)
                    if identity(await page.evaluate(SNAPSHOT_JS))!=sig:
                        result['status']='discarded_stale';save_audit(args.output,snapshot,result)
                        print('盘面已变化，丢弃过期建议。',flush=True);continue
                    result.update(game_id=snapshot.get('game_id'),site='kirameki-lab.com',observed_at=time.time())
                    save_audit(args.output,snapshot,result)
                    result['audit_directory']=str(audit_path)
                    save_result(args.output,result)
                    print('\n推荐：'+recommendation_text(result),flush=True)
                    print('由你在网页手动执行；等待盘面更新。\n',flush=True)
                    previous=request;last=sig;notice=None
                    if args.once:return
                except StateUnavailable as error:
                    message=str(error)
                    if message!=notice:
                        save_result(args.output,{'description':message,'submitted_action':False,'status':'waiting'})
                        print(message,flush=True);notice=message
                except PlaywrightError as error:
                    # Navigation replaces the JS context; inspect the new page
                    # on the next poll instead of terminating the adviser.
                    if not any(text in str(error) for text in (
                        'Execution context was destroyed',
                        'Cannot find context with specified id',
                        'Target page, context or browser has been closed')):
                        raise
                    message='页面正在切换，等待新盘面…'
                    if notice!=message:print(message,flush=True);notice=message
                await asyncio.sleep(args.poll)
        finally:
            await browser.close()  # disconnect this client; keep the browser/game open


def parse_args(argv=None,*,autoplay=False):
    parser=argparse.ArgumentParser(description=('Kirameki非排位自动对弈并重开；连接已有浏览器。' if autoplay else
                                              'Kirameki非排位只读推荐；连接已有浏览器，不执行动作。'))
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--mode',choices=('greedy','mcts'),default='mcts')
    parser.add_argument('--simulations-per-world',type=int,default=4096)
    parser.add_argument('--determinizations',type=int,default=16)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--search-workers',type=int,default=4,help='CPU搜索进程数（最多世界数）；CUDA请设1以批量推理')
    parser.add_argument('--search-backend',choices=('native','native_chance'),default='native',help='独立世界或随机节点共享树；后者要求--determinizations 1')
    parser.add_argument('--leaf-batch-size',type=int,default=1,help='随机节点单树内最多同时在途的模拟数；1为串行参考路径')
    parser.add_argument('--inference-mode',choices=('eager','compiled'),default='eager')
    parser.add_argument('--compile-mode',choices=('default','reduce-overhead','native-cudagraph'),default='default')
    parser.add_argument('--root-selection',choices=('world_mean_q','visits'),help='默认native用world_mean_q，native_chance用visits')
    parser.add_argument('--root-q-min-visits',type=int,default=32,help='Q选点要求每个世界至少此访问数')
    parser.add_argument('--root-q-candidates',type=int,default=8,help='初始搜索后按全世界总访问数选取补评候选数')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--profile',type=Path,default=Path.home()/'.config/microsoft-edge')
    parser.add_argument('--cdp')
    parser.add_argument('--output',type=Path,default=Path('runtime/kirameki_advisor'))
    parser.add_argument('--poll',type=float,default=.5)
    parser.add_argument('--settle',type=float,default=.3)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--snapshot',type=Path,help='离线工程验证：读取已保存的安全快照，不连接浏览器')
    if autoplay:
        parser.add_argument('--games',type=int,default=0,help='完成局数后停止；0持续对弈，Ctrl-C停止')
        parser.add_argument('--opponent-level',default='bga1000-v5',help='预期网站对手ID；默认宝神/Gem Deity')
        parser.add_argument('--action-timeout',type=float,default=30,help='提交后等待动作记入棋谱的秒数；超时停止，不重发')
        parser.add_argument('--restart-delay',type=float,default=2,help='终局保存后等待多少秒再回菜单重开')
        parser.add_argument('--record-progress',action='store_true',help='保存双方过程盘面、网站胜率条及本地估值')
        parser.add_argument('--resume',action='store_true',help='恢复同配置输出目录，累计已完成局数')
        parser.add_argument('--fresh-games-only',action='store_true',help='拒绝接入未登记的中途棋局，避免混合不同策略')
        parser.add_argument('--clock',choices=('keep','unlimited'),default='keep',help='重开时保留时钟或设置不限时')
        parser.set_defaults(search_backend='native_chance',device='cuda',search_workers=1,determinizations=1,
                            simulations_per_world=65536,leaf_batch_size=32,inference_mode='compiled',
                            compile_mode='native-cudagraph',output=Path('runtime/kirameki_autoplay'))
    args=parser.parse_args(argv)
    if autoplay and (args.games<0 or args.action_timeout<=0 or args.restart_delay<0 or args.snapshot):
        parser.error('自动对弈要求games非负、action-timeout为正、restart-delay非负，且不使用--snapshot')
    if args.root_selection is None:
        args.root_selection='visits' if args.search_backend=='native_chance' else 'world_mean_q'
    if min(args.simulations_per_world,args.determinizations,args.threads,args.top_k,args.search_workers,
           args.root_q_min_visits,args.root_q_candidates,args.leaf_batch_size)<1 or args.poll<=0 or args.settle<0:
        parser.error('预算、世界数、线程数、top-k、poll须为正，settle不能为负')
    args.search_workers=min(args.search_workers,args.determinizations)
    if args.mode=='mcts' and args.search_backend=='native_chance':
        if args.determinizations!=1:
            parser.error('native_chance随机节点搜索要求--determinizations 1：共享树，总预算等于--simulations-per-world')
        if args.root_selection!='visits':
            parser.error('native_chance共享树使用--root-selection visits，不适用world_mean_q补评')
    if args.device!='cpu' and args.search_workers>1:
        parser.error('GPU模式请设置--search-workers 1，使用多世界批量推理')
    if args.leaf_batch_size!=1 and (args.mode!='mcts' or args.search_backend!='native_chance'):
        parser.error('--leaf-batch-size>1要求随机节点单树MCTS')
    if args.inference_mode=='compiled' and args.search_workers>1:
        parser.error('编译推理使用--search-workers 1')
    if args.compile_mode!='default' and args.inference_mode!='compiled':
        parser.error('--compile-mode要求--inference-mode compiled')
    if args.compile_mode=='native-cudagraph' and not args.device.startswith('cuda'):
        parser.error('native-cudagraph要求CUDA设备')
    return args


def create_advisor(args,*,autoplay=False):
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    budget=(f'随机节点单共享树（总预算{args.simulations_per_world}）' if args.mode=='mcts' and args.search_backend=='native_chance' else
            f'S{args.simulations_per_world}N{args.determinizations}（总预算{args.simulations_per_world*args.determinizations}）'
            if args.mode=='mcts' else '零搜索')
    print(f'{"非排位自动对弈" if autoplay else "只读推荐"} · {args.mode} · {budget}',flush=True)
    from .search_inference import InferenceConfig
    capacity=args.determinizations*args.leaf_batch_size
    inference_config=InferenceConfig(mode=args.inference_mode,
        candidate_buckets=(16,32,48,64),compile_mode=args.compile_mode,
        compile_failure_policy='fail_fast',
        batch_buckets=tuple(sorted({1,capacity})) if args.inference_mode=='compiled' else ())
    advisor=LocalAdvisor(args.checkpoint,device=args.device,mode='greedy',inference_config=inference_config)
    if args.mode=='greedy' and args.inference_mode=='compiled':
        print('正在预热greedy编译推理…',flush=True)
        advisor.evaluator.warmup_requests(inference_config.candidate_buckets)
    if args.mode=='mcts':
        from .parallel_recommend import WorldParallelSearch,BatchedWorldSearch
        from .root_q_selection import RootQSettings
        settings=(RootQSettings(args.root_q_min_visits,args.root_q_candidates)
                  if args.root_selection=='world_mean_q' else None)
        if args.search_workers>1:
            advisor.search_policy=WorldParallelSearch(args.checkpoint,simulations_per_world=args.simulations_per_world,
                worlds=args.determinizations,workers=args.search_workers,seed=args.seed,threads=args.threads,
                q_settings=settings)
        else:
            advisor.search_policy=BatchedWorldSearch(advisor,simulations_per_world=args.simulations_per_world,
                worlds=args.determinizations,seed=args.seed,q_settings=settings,search_backend=args.search_backend,
                leaf_batch_size=args.leaf_batch_size)
        advisor.mode='mcts'
        layout=('单共享树，逐模拟采样隐藏牌及随机翻牌，CPW关闭' if args.search_backend=='native_chance' else '各进程批量评估世界叶子')
        print(f'搜索：{args.search_backend}，{args.search_workers}进程，设备{args.device}，{layout}',flush=True)
        print(f'叶子批量：{args.leaf_batch_size}；推理：{args.inference_mode}/{args.compile_mode}',flush=True)
        print(f'选点：{args.root_selection}'+(f'；每世界门槛{settings.min_visits}、补评候选最多{settings.candidates}，补评计入原预算' if settings else ''),flush=True)
        if args.inference_mode=='compiled':
            print('正在预热编译推理；首次编译可能较慢…',flush=True)
            seconds=advisor.search_policy.runner.warmup()
            seconds+=advisor.evaluator.warmup_requests(inference_config.candidate_buckets)
            print(f'推理预热完成：{seconds:.1f}秒；后续搜索复用已预热缓冲。',flush=True)
    print(f'权重：{advisor.checkpoint}\nSHA256：{advisor.checkpoint_sha256}',flush=True)
    return advisor


def main(argv=None,*,autoplay=False):
    args=parse_args(argv,autoplay=autoplay)
    advisor=create_advisor(args,autoplay=autoplay)
    try:
        if args.snapshot:
            snapshot=json.loads(args.snapshot.read_text())
            result,_=recommend_snapshot(snapshot,advisor,top_k=args.top_k)
            save_result(args.output,result);save_audit(args.output,snapshot,result)
            print('推荐：'+recommendation_text(result))
            return
        if autoplay:
            from .kirameki_autoplay import watch_autoplay
            asyncio.run(watch_autoplay(args,advisor))
        else:
            asyncio.run(watch(args,advisor))
    except KeyboardInterrupt:print('\n已停止，浏览器和对局保持打开。')
    finally:
        if advisor.search_policy is not None:advisor.search_policy.close()


if __name__=='__main__':main()
