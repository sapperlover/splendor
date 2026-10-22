"""Read-only BGA Splendor state adapter and local-weight live adviser."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from .actions import BuyMarket, BuyReserved, ReserveDeck, ReserveMarket, TakeDistinct, TakePair
from .catalog import Catalog, load_default_catalog
from .contracts import (CardView, MarketView, NobleView, Observation,
                        PendingObservation, PlayerPrivateState, PlayerPublicState, PublicReservedCard)
from .model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from .recommend import DEFAULT_SIMULATIONS, LocalAdvisor, decision_identity
from .replay import to_primitive
from .search_policy import request_from_observation

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT=ROOT/'runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt'
LETTERS='CSEROG'

# Only reads game state and the public action log; no calls to game actions.
SNAPSHOT_JS=r'''() => {
 if (typeof gameui === 'undefined' || !gameui.gamedatas) return null;
 const gd=gameui.gamedatas;
 const fields=['players','cards_hidden','market','first_player','roundnumber','carddb','nobledb',
  'expansion_cities','expansion_tradings_post','expansion_orient','expansion_strongholds',
  'gamestate','playerorder','notifications','decision'];
 const selected=JSON.parse(JSON.stringify(Object.fromEntries(fields.map(k=>[k,gd[k]]))));
 // This AMD module is already loaded by the game. Reading it sends no request.
 // gamedatas is only the initial payload; notifications update these storages.
 if(typeof require==='undefined'||typeof g_gamethemeurl==='undefined')return null;
 const model=require(g_gamethemeurl+'modules/spl_game.js');
 if(!model.gamestorage||!model.playerstorage)return null;
 const asMap=cards=>Object.fromEntries(cards.map(c=>[String(c.id),c]));
 const bank=model.gamestorage;
 selected.market.pool=Object.fromEntries('CSEROG'.split('').map(c=>[c,bank[c]]));
 for(const tier of [1,2,3]){
  selected.market['row_'+tier]={cards:asMap(bank.cards.filter(c=>Number(model.carddb[c.id].lvl)===tier)),count:bank.drawcounts[tier]};
 }
 selected.market.nobles=asMap(bank.nobles);
 for(const pid of Object.keys(selected.players)){
  const ps=model.playerstorage[pid],dest=selected.players[pid];
  dest.coins=Object.fromEntries('CSEROG'.split('').map(c=>[c,ps[c]]));
  dest.cards_played=asMap(ps.boughtCards);
  dest.nobles_played=asMap(ps.boughtNobles);
  dest.score=ps.boughtCards.reduce((sum,c)=>sum+Number(model.carddb[c.id].points),0)
   +ps.boughtNobles.reduce((sum,c)=>sum+Number(model.nobledb[c.id].points),0);
  // Opponent storedCards is initialized with the viewer's private cards by BGA;
  // never use it. Only hidden_cards counts are authoritative for the opponent.
  dest.nb_cards_hidden=Object.fromEntries([1,2,3].map(tier=>[String(tier),
   String(pid)===String(gameui.player_id)?ps.storedCards.filter(c=>Number(model.carddb[c.id].lvl)===tier).length:ps.hidden_cards[tier]]));
 }
 if(model.playerstorage[gameui.player_id])selected.cards_hidden=asMap(model.playerstorage[gameui.player_id].storedCards);
 const observer=window.__splendorLocalAdvisor;
 if(selected.gamestate.name==='playerTurn'){
  selected.roundnumber=selected.gamestate.args.turn;
  if(observer)observer.turn=Number(selected.roundnumber);
 }
 const choosing=model.selectedcard_id!=null||Object.values(model.tempstorage).some(v=>Number(v)!==0)
   ||Object.values(model.paying).some(v=>Number(v)!==0);
 selected.decision={decision_type:choosing?'local_selection':'none'};
 const paid=Object.fromEntries('CSEROG'.split('').map(c=>[c,Number(model.paying[c])]));
 if(selected.gamestate.name==='playerTurn'&&model.selectedcard_id!=null&&
   document.getElementById('btn_sendpay')&&
   model.getPossibleCardBuys(gameui.player_id).map(String).includes(String(model.selectedcard_id))){
  selected.decision={decision_type:'payment',card_id:String(model.selectedcard_id),selected_tokens:paid};
 }
 if(selected.gamestate.name==='playerDiscard'){
  const pending=observer?.lastMain;
  const valid=pending&&pending.actor_id===String(selected.gamestate.active_player)&&pending.turn===observer.turn;
  if(valid)selected.roundnumber=pending.turn;
  selected.decision={decision_type:'return_tokens',main_action:valid?pending.action:null,
   turn:valid?pending.turn:null,selected_tokens:String(gameui.player_id)===String(selected.gamestate.active_player)?paid:
    Object.fromEntries('CSEROG'.split('').map(c=>[c,0]))};
 }
 const action=document.getElementById('gameaction_status_wrap');
 return {schema:'bga-splendor-visible-state-v2',url:location.href,player_id:gameui.player_id,
  gamedatas:JSON.parse(JSON.stringify(selected)),move_text:document.getElementById('move_nbr')?.textContent,
  title:document.getElementById('pagemaintitletext')?.innerText,
  page_visible:document.visibilityState==='visible',page_focused:document.hasFocus(),
  updating:!!gameui.animated || (!!action && getComputedStyle(action).display!=='none'),
  public_logs:Array.from(document.querySelectorAll('#logs .log')).map(e=>({
   id:e.id,text:e.textContent,
   player_names:Array.from(e.querySelectorAll('.playername')).map(p=>p.textContent.trim()),
   card_ids:Array.from(e.querySelectorAll('.spl_notif-inner-tooltip[data-id]')).map(p=>p.dataset.id)
  }))};
}'''


class StateUnavailable(ValueError):
    """A transient state or required visible information is missing."""


def vector(value, cls=TokenVector):
    return cls(*(int(value[c]) for c in LETTERS[:6 if cls is TokenVector else 5]))


def cost(value):
    counts=Counter(value)
    if set(counts)-set(LETTERS[:5]):
        raise StateUnavailable('卡牌费用包含基础版之外的符号')
    return GemVector(*(counts[c] for c in LETTERS[:5]))


def records(value):
    # BGA represents an empty card collection as [] and nonempty sets as maps.
    if isinstance(value, dict):return list(value.values())
    if isinstance(value,list) and not value:return []
    raise StateUnavailable('BGA卡牌集合结构与已验证结构不符')


def ids(value):
    return [str(x['type']) for x in records(value)]


def table_id(url):
    parsed=urlparse(url)
    if not (parsed.hostname=='boardgamearena.com' or (parsed.hostname or '').endswith('.boardgamearena.com')):
        raise ValueError('只接受BGA对局网址')
    values=parse_qs(parsed.query).get('table',[])
    if len(values)!=1 or not values[0].isdigit():raise ValueError('网址缺少有效的table编号')
    return int(values[0])


def semantic_identity(snapshot):
    gd=snapshot['gamedatas']
    state=gd['gamestate']
    semantic={k:gd[k] for k in ('cards_hidden','market','first_player','roundnumber','decision')}
    semantic['players']={pid:{k:p[k] for k in ('coins','cards_played','nobles_played','nb_cards_hidden','score')}
                         for pid,p in gd['players'].items()}
    semantic.update(player_id=snapshot['player_id'],state={k:state.get(k) for k in ('id','name','active_player','args')},
                    move=snapshot.get('move_text'),updating=snapshot.get('updating'))
    return hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def runtime_catalog(gd,base=None):
    """Map BGA cards and selected nobles to the normative 12-noble catalog."""
    base=base or load_default_catalog()
    mapped={}
    for bga_id,raw in gd['carddb'].items():
        if int(raw['lvl']) in (11,12,13):continue  # Client merges disabled Orient definitions into carddb.
        matches=[c for c in base.cards if int(c.tier)==int(raw['lvl'])
                 and c.bonus==tuple(Color)[int(raw['type'])] and c.points==int(raw['points']) and c.cost==cost(raw['cost'])]
        if len(matches)!=1:raise StateUnavailable(f'卡牌{bga_id}无法唯一映射到本地90张基础卡')
        mapped[str(bga_id)]=matches[0]
    if len(mapped)!=90 or len({c.card_id for c in mapped.values()})!=90:
        raise StateUnavailable('BGA卡表与本地90张卡并非一一对应')
    noble_ids={str(n['type']) for n in records(gd['market']['nobles'])}
    for player in gd['players'].values():noble_ids.update(ids(player['nobles_played']))
    if len(noble_ids)!=3:raise StateUnavailable('当前基础双人局的贵族总数不是3')
    noble_map={}
    for bga_id in sorted(noble_ids,key=int):
        raw=gd['nobledb'][bga_id]
        requirement=cost(raw['cost'])
        matches=[n for n in base.nobles if n.points==int(raw['points']) and n.requirement==requirement]
        if len(matches)!=1:raise StateUnavailable(f'贵族{bga_id}无法唯一映射到本地规范贵族池')
        noble_map[bga_id]=matches[0]
    if len({n.noble_id for n in noble_map.values()})!=3:raise StateUnavailable('本局贵族定义重复')
    return base,mapped,noble_map


@dataclass
class ReservationHistory:
    table: int
    viewer: str
    market_reserves: dict[str,str]

    @classmethod
    def start(cls,snapshot):
        # Attach at any turn. observe() restores identities from available
        # public logs; request_from_snapshot() treats remaining slots as blind.
        return cls(table_id(snapshot['url']),str(snapshot['player_id']),{})

    def observe(self,snapshot):
        if table_id(snapshot['url'])!=self.table or str(snapshot['player_id'])!=self.viewer:
            raise StateUnavailable('历史记录与当前桌号或登录玩家不符')
        players=snapshot['gamedatas']['players']
        names={p['name']:str(pid) for pid,p in players.items()}
        for log in snapshot['public_logs']:
            if not log['card_ids'] or '保留' not in log['text']:continue
            owners={names[name] for name in log['player_names'] if name in names}
            if len(owners)!=1 or len(log['card_ids'])!=1:
                raise StateUnavailable('公开保留记录中的玩家或目标不唯一')
            key=str(log['card_ids'][0]);owner=owners.pop()
            if key in self.market_reserves and self.market_reserves[key]!=owner:
                raise StateUnavailable('同一张公开保留牌出现不同拥有者')
            self.market_reserves[key]=owner

    def to_dict(self):return {'table':self.table,'viewer':self.viewer,'market_reserves':self.market_reserves}


def observation_from_snapshot(snapshot,history,*,require_self_turn=True):
    gd=snapshot['gamedatas'];viewer=str(snapshot['player_id'])
    if viewer not in gd['players']:raise StateUnavailable('请在脚本Chrome窗口登录本局玩家账号')
    if len(gd['players'])!=2:raise StateUnavailable('目前仅支持双人基础版')
    if any(int(gd[k])!=2 for k in ('expansion_cities','expansion_tradings_post','expansion_orient','expansion_strongholds')):
        raise StateUnavailable('本地模型只支持关闭四个扩展的基础版')
    if snapshot.get('updating'):raise StateUnavailable('等待页面完成更新')
    state=gd['gamestate']
    if state['name']=='gameEnd':raise StateUnavailable('对局已结束')
    actor=str(state.get('active_player'))
    if actor not in gd['players']:raise StateUnavailable('等待行动方确定')
    if require_self_turn and actor!=viewer:raise StateUnavailable('等待轮到你')
    if state['name'] not in ('playerTurn','playerDiscard'):
        raise StateUnavailable(f"当前网页阶段{state['name']}尚未验证；请手动完成该阶段")
    decision=gd['decision'];kind=decision['decision_type']
    if kind not in ('none','payment','return_tokens'):
        raise StateUnavailable('你正在网页内选择动作；等待确认或取消后重新读取')
    history.observe(snapshot)
    catalog,cards,nobles=runtime_catalog(gd)
    def cv(card):return CardView(card.card_id,card.tier,card.bonus,card.points,card.cost)
    def nv(n):return NobleView(n.noble_id,n.points,n.requirement)
    purchased={pid:tuple(sorted(cards[c].card_id for c in ids(p['cards_played']))) for pid,p in gd['players'].items()}
    all_purchased={c for values in purchased.values() for c in values}
    hidden=records(gd['cards_hidden'])
    own_cards={str(c['type']):cards[str(c['type'])] for c in hidden}
    def public(pid):
        p=gd['players'][pid]
        bought=purchased[pid]
        reservations=[]
        known=[]
        for key,owner in history.market_reserves.items():
            if owner==pid and cards[key].card_id not in all_purchased:known.append(key)
        known_counts=Counter(int(cards[c].tier) for c in known)
        for key in known:
            if pid==viewer and key not in own_cards:raise StateUnavailable('本人公开保留记录与手牌不一致')
            reservations.append(PublicReservedCard(cards[key].tier,ReservationSource.MARKET,cv(cards[key])))
        for tier in (1,2,3):
            count=int(p['nb_cards_hidden'][str(tier)])-known_counts[tier]
            if count<0:raise StateUnavailable('公开保留牌历史与当前手牌数量不一致')
            reservations.extend(PublicReservedCard(Tier(tier),ReservationSource.DECK,None) for _ in range(count))
        reservations.sort(key=lambda r:(r.source.value,int(r.tier),r.card.card_id if r.card else ''))
        bonuses=GemVector(*(sum(catalog.card(c).bonus==color for c in bought) for color in tuple(Color)[:5]))
        return PlayerPublicState(vector(p['coins']),bonuses,int(p['score']),bought,
             tuple(sorted(nobles[n].noble_id for n in ids(p['nobles_played']))),tuple(reservations))
    own=public(viewer);other_id=next(pid for pid in gd['players'] if pid!=viewer);opponent=public(other_id)
    if len(hidden)!=len(own.reserved_cards):raise StateUnavailable('本人保留牌身份不完整')
    known_own={r.card.card_id for r in own.reserved_cards if r.card is not None}
    private=PlayerPrivateState(tuple(sorted((cv(c) for c in own_cards.values() if c.card_id not in known_own),key=lambda c:c.card_id)))
    supply=vector(gd['market']['pool'])
    phase=DecisionPhase.MAIN_ACTION;pending=None
    if kind=='payment':
        key=str(decision['card_id'])
        if key in own_cards:main=BuyReserved(cards[key].card_id)
        elif any(key in ids(gd['market'][f'row_{t}']['cards']) for t in (1,2,3)):
            main=BuyMarket(cards[key].card_id)
        else:raise StateUnavailable('支付目标已不在市场或本人保留区，重新读取')
        phase=DecisionPhase.PAYMENT;pending=PendingObservation(main,payment_target=cv(cards[key]))
    elif kind=='return_tokens':
        raw=decision['main_action']
        if raw is None:
            raise StateUnavailable('弃筹码阶段缺少本手行动记录；请保持脚本运行，下次拿筹码或保留后会推荐弃法')
        if raw['kind']=='take':
            income=vector(raw['income'])
            colors=tuple(color for color,n in zip(tuple(Color),income) if n)
            main=TakePair(colors[0]) if income.total==2 and len(colors)==1 else TakeDistinct(colors)
        elif raw['kind']=='reserve_market':main=ReserveMarket(cards[str(raw['card_id'])].card_id)
        elif raw['kind']=='reserve_deck':main=ReserveDeck(Tier(int(raw['tier'])))
        else:raise StateUnavailable('本手行动不是可触发弃筹码的基础版行动')
        selected=vector(decision['selected_tokens'])
        acting=own if actor==viewer else opponent
        if any(n<0 or n>held for n,held in zip(selected,acting.tokens)):
            raise StateUnavailable('已选归还筹码与持有数量尚未同步')
        acting=replace(acting,tokens=acting.tokens.minus(selected))
        if actor==viewer:own=acting
        else:opponent=acting
        supply=supply.plus(selected)
        excess=acting.tokens.total-catalog.constants.max_tokens_at_turn_end
        if excess<=0:raise StateUnavailable('等待归还筹码提交完成')
        phase=DecisionPhase.RETURN_TOKENS;pending=PendingObservation(main,tokens_to_return=excess)
    roundnumber=int(gd['roundnumber'])
    if kind=='return_tokens':
        if decision['turn'] is None:raise StateUnavailable('等待弃筹码回合信息')
        roundnumber=int(decision['turn'])
    state_args=state['args'] or {}
    if state_args.get('turn') is not None and int(state_args['turn'])!=roundnumber:
        raise StateUnavailable('回合号尚未同步')
    first=str(gd['first_player']);self_first=first==viewer
    completed={pid:roundnumber-1+int(pid==first and actor!=first) for pid in gd['players']}
    turns=(completed[viewer],completed[other_id])
    market=MarketView(*(tuple(sorted((cv(cards[str(c['type'])]) for c in records(gd['market'][f'row_{t}']['cards'])),key=lambda c:c.card_id)) for t in (1,2,3)))
    trigger=own.score>=15 or opponent.score>=15
    obs=Observation(phase,self_first,supply,
          tuple(int(gd['market'][f'row_{t}']['count']) for t in (1,2,3)),market,
          tuple(sorted((nv(nobles[str(n['type'])]) for n in records(gd['market']['nobles'])),key=lambda n:n.noble_id)),
          own,private,opponent,tuple(c.card_id in all_purchased for c in catalog.cards),
          *turns,0,trigger,own.score>=15,pending)
    mapping={'cards':{c.card_id:int(k) for k,c in cards.items()},'nobles':{n.noble_id:int(k) for k,n in nobles.items()}}
    return obs,catalog,mapping


def request_from_snapshot(snapshot,history):
    observation,catalog,mapping=observation_from_snapshot(snapshot,history)
    return request_from_observation(observation,catalog=catalog),catalog,mapping


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def report_text(result):
    if result.get('status') not in ('ready','value_only'):return '# BGA 只读招法推荐\n\n'+result['message']+'\n'
    lines=['# BGA 只读招法推荐','',f"桌号 {result['table']} · 当前步数 {result['move']} · 混合预算第20轮",'']
    if 'expected_score_viewer' in result:
        lines += [f"模型期望得分估计：{result['expected_score_viewer']:.1%}（非校准胜率）",'']
    if result['status']=='value_only':
        return '\n'.join(lines+['对手行动中；轮到你后计算推荐。',''])
    lines += [
           '**推荐：'+result['description']+'**','',f"模式：{result['mode']}；搜索模拟 {result['simulations']} 次；耗时 {result['elapsed_seconds']:.2f} 秒。",'']
    if result['phase']=='payment':
        lines+=['支付建议为完整总额；若与网页预选不同，请取消卡牌选择后按建议重选。','']
    elif result['phase']=='return_tokens' and result.get('selected_return_description'):
        lines+=['已选：'+result['selected_return_description']+'；上方建议是接下来还需归还的筹码。',
                '本次合计：'+result['total_return_description'],'']
    lines+=['候选招法：','']
    for c in result['candidates']:
        suffix=f"；访问量 {c['visits']}" if 'visits' in c else ''
        lines.append(f"- {c['description']}（策略权重 {c['policy_probability']:.1%}{suffix}）")
    lines+=['','由你手动执行。局面变化时，上一条推荐失效。','']
    return '\n'.join(lines)


async def watch(args):
    # Attach to a browser already owned by the user. No browser launch/navigation.
    from .bga_watch import watch_existing
    await watch_existing(args)


def main(argv=None):
    parser=argparse.ArgumentParser(description='连接已有Chrome/Edge，自动识别BGA牌桌，只读推荐及模型期望得分估计')
    parser.add_argument('--url',help='可选：仅分析这一桌；省略时自动识别已有BGA标签页')
    parser.add_argument('--cdp',help='已有Chrome/Edge的本地CDP地址；默认读取浏览器端口文件或连接127.0.0.1:9222')
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output',type=Path,default=ROOT/'runtime/bga_advisor')
    parser.add_argument('--mode',choices=('greedy','mcts'),default='mcts')
    parser.add_argument('--simulations',type=int,default=DEFAULT_SIMULATIONS)
    parser.add_argument('--determinizations',type=int,default=1)
    parser.add_argument('--value-samples',type=int,default=4,help='对手行动时未知手牌的value采样数')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--interval',type=float,default=.25)
    parser.add_argument('--once',action='store_true',help='输出一次有效评估后断开，不关闭浏览器')
    args=parser.parse_args(argv)
    if args.interval<=0 or args.threads<1 or args.value_samples<1:parser.error('间隔、线程数和value采样数必须为正数')
    if args.url:table_id(args.url)
    try:asyncio.run(watch(args))
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
