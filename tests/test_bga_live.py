from __future__ import annotations
import copy
import json
from pathlib import Path

import pytest

from splendor.bga_live import (ReservationHistory,StateUnavailable,request_from_snapshot,
                              runtime_catalog,semantic_identity,table_id)
from splendor.compact_trace import decode_request
from splendor.model import DecisionPhase, ReservationSource
from splendor.search_policy import request_from_observation,determinize_request

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def snapshot():
    return json.loads((ROOT/'tests/fixtures/bga_live_initial_934625108.json').read_text())


def test_live_root_preserves_public_reserve_memory_and_extra_noble(snapshot):
    history=ReservationHistory.start(snapshot)
    request,catalog,mapping=request_from_snapshot(snapshot,history)
    assert len(catalog.cards)==90 and len(catalog.nobles)==12
    assert len(request.legal_actions)==30
    assert request.observation.self_is_first_player is False
    assert (request.observation.own_completed_turns,request.observation.opponent_completed_turns)==(0,1)
    reserve=request.observation.opponent_public.reserved_cards
    assert len(reserve)==1 and reserve[0].source==ReservationSource.MARKET
    assert reserve[0].card.card_id=='013' and mapping['cards']['013']==9
    extra=next(n for n in request.observation.nobles_in_play if n.noble_id=='N11')
    assert tuple(extra.requirement)==(0,3,3,0,3)
    assert determinize_request(request,seed=123,catalog=catalog,engine_mode='fast').decision()==request
    # Raw visible catalogue can contain disabled Orient definitions; they never
    # become market identities or model cards.
    assert all(str(k) not in mapping['cards'] for k in range(201,231))


def test_extra_noble_works_in_existing_native_catalog(snapshot):
    from splendor.native_policy import _native_catalog
    catalog,_,_=runtime_catalog(snapshot['gamedatas'])
    assert _native_catalog(catalog) is not None


def test_bga_maps_both_added_nobles_to_normative_ids(snapshot):
    from splendor.catalog import load_default_catalog
    gd=snapshot['gamedatas']
    gd['market']['nobles']={str(k):{'id':str(k),'type':str(k)} for k in (1,11,12)}
    gd['nobledb']['12']={'id':12,'cost':'RRRRCCCC','points':3,'name':'Hino Tomiko'}
    base=load_default_catalog()
    catalog,_,mapping=runtime_catalog(gd,base)
    assert catalog is base
    assert mapping['11'].noble_id=='N11' and mapping['12'].noble_id=='N12'


def test_bga_rejects_nobles_outside_the_normative_pool(snapshot):
    key=next(iter(snapshot['gamedatas']['market']['nobles']))
    snapshot['gamedatas']['nobledb'][key]['cost']='CCSS'
    with pytest.raises(StateUnavailable,match='规范贵族池'):
        runtime_catalog(snapshot['gamedatas'])


@pytest.mark.parametrize('change,match',[
    ('visitor','登录'),('opponent','轮到'),('updating','更新'),('expansion','扩展'),('selection','选择'),
])
def test_no_advice_for_incomplete_or_wrong_turn(snapshot,change,match):
    history=ReservationHistory.start(snapshot)
    if change=='visitor':snapshot['player_id']=-1
    elif change=='opponent':snapshot['gamedatas']['gamestate']['active_player']='100878205'
    elif change=='updating':snapshot['updating']=True
    elif change=='expansion':snapshot['gamedatas']['expansion_orient']=1
    elif change=='selection':snapshot['gamedatas']['decision']['decision_type']='selecting'
    with pytest.raises(StateUnavailable,match=match):request_from_snapshot(snapshot,history)


def test_reservation_memory_survives_disappearing_old_logs(snapshot):
    history=ReservationHistory.start(snapshot)
    first,_,_=request_from_snapshot(snapshot,history)
    restored=ReservationHistory(**json.loads(json.dumps(history.to_dict())))
    snapshot['public_logs']=[]
    second,_,_=request_from_snapshot(snapshot,restored)
    assert second==first


def test_semantic_identity_ignores_clock_but_tracks_moves(snapshot):
    before=semantic_identity(snapshot)
    snapshot['gamedatas']['gamestate']['reflexion']={'clock':999}
    assert semantic_identity(snapshot)==before
    snapshot['move_text']='3'
    assert semantic_identity(snapshot)!=before


def test_first_attach_midgame_without_logs_uses_blind_reserves():
    snapshot=json.loads((ROOT/'tests/fixtures/bga_live_step8_934625108.json').read_text())
    snapshot['public_logs']=[]
    history=ReservationHistory.start(snapshot)
    request,catalog,_=request_from_snapshot(snapshot,history)
    obs=request.observation
    assert (obs.own_completed_turns,obs.opponent_completed_turns)==(3,4)
    assert obs.self_public.purchased_card_ids==('025',)
    assert len(obs.opponent_public.reserved_cards)==3
    assert len(obs.self_public.reserved_cards)==1
    assert all(r.source==ReservationSource.DECK and r.card is None
               for r in (*obs.self_public.reserved_cards,*obs.opponent_public.reserved_cards))
    # Missing public history does not discard the viewer's visible hand.
    assert [c.card_id for c in obs.self_private.blind_reserved_cards]==['008']
    assert history.market_reserves=={}
    for seed in (0,1,2):
        assert determinize_request(request,seed=seed,catalog=catalog,engine_mode='fast').decision()==request


def test_midgame_partial_logs_preserve_known_and_blind_reserves():
    snapshot=json.loads((ROOT/'tests/fixtures/bga_live_step4_934625108.json').read_text())
    logs=snapshot['public_logs']
    snapshot['public_logs']=[log for log in logs if log['card_ids']==['27']]
    history=ReservationHistory.start(snapshot)
    request,catalog,mapping=request_from_snapshot(snapshot,history)
    reserves=request.observation.opponent_public.reserved_cards
    known=[r for r in reserves if r.card is not None]
    blind=[r for r in reserves if r.card is None]
    assert len(known)==len(blind)==1
    assert known[0].source==ReservationSource.MARKET and mapping['cards'][known[0].card.card_id]==27
    assert blind[0].source==ReservationSource.DECK
    assert determinize_request(request,seed=0,catalog=catalog,engine_mode='fast').decision()==request
    # Later visible logs can restore identities that were unknown on attach.
    snapshot['public_logs']=logs
    complete,_,_=request_from_snapshot(snapshot,history)
    assert all(r.card is not None for r in complete.observation.opponent_public.reserved_cards)
    assert not complete.observation.self_private.blind_reserved_cards
    restored=ReservationHistory(**json.loads(json.dumps(history.to_dict())))
    snapshot['public_logs']=[]
    resumed,_,_=request_from_snapshot(snapshot,restored)
    assert resumed==complete


def test_all_88_captured_observations_rebuild_exact_candidates():
    source=ROOT/'artifacts/bga_927500015_20261020/model_decisions.jsonl'
    for line in source.read_text().splitlines():
        request=decode_request(json.loads(line)['request'])
        assert request_from_observation(request.observation)==request


@pytest.mark.parametrize('url', ['https://example.com/?table=1','https://boardgamearena.com/table?table=x'])
def test_url_scope(url):
    with pytest.raises(ValueError):table_id(url)


def test_real_two_move_transition_keeps_both_public_reserves():
    snapshot=json.loads((ROOT/'tests/fixtures/bga_live_step4_934625108.json').read_text())
    history=ReservationHistory(934625108,'92773015',{'9':'100878205','33':'92773015','27':'100878205'})
    request,catalog,mapping=request_from_snapshot(snapshot,history)
    obs=request.observation
    assert snapshot['schema']=='bga-splendor-visible-state-v2'
    assert obs.deck_counts==(33,26,16)
    assert obs.supply.gold==2 and obs.self_public.tokens.gold==1 and obs.opponent_public.tokens.gold==2
    assert (obs.own_completed_turns,obs.opponent_completed_turns)==(1,2)
    assert [r.card.card_id for r in obs.self_public.reserved_cards]==['008']
    assert {mapping['cards'][r.card.card_id] for r in obs.opponent_public.reserved_cards}=={9,27}
    assert not obs.self_private.blind_reserved_cards
    assert determinize_request(request,seed=456,catalog=catalog,engine_mode='fast').decision()==request


def test_initial_attach_without_reservation_logs_uses_blind_reserves(snapshot):
    snapshot['public_logs']=[]
    history=ReservationHistory.start(snapshot)
    request,catalog,_=request_from_snapshot(snapshot,history)
    reserves=request.observation.opponent_public.reserved_cards
    assert len(reserves)==1
    assert reserves[0].source==ReservationSource.DECK and reserves[0].card is None
    assert determinize_request(request,seed=0,catalog=catalog,engine_mode='fast').decision()==request


@pytest.mark.parametrize('step,turns,tokens,green,bought',[
    (6,(2,3),(1,1,0,0,1,1),0,()),
    (8,(3,4),(0,0,0,0,0,0),1,('025',)),
])
def test_live_tokens_and_purchase_without_page_reload(step,turns,tokens,green,bought):
    snapshot=json.loads((ROOT/f'tests/fixtures/bga_live_step{step}_934625108.json').read_text())
    # Each public reserve is retained even though the reloaded DOM lost older logs.
    public={'9':'100878205','33':'92773015','27':'100878205'}
    history=ReservationHistory(934625108,'92773015',public)
    history.observe(snapshot)
    request,catalog,_=request_from_snapshot(snapshot,history)
    obs=request.observation
    assert (obs.own_completed_turns,obs.opponent_completed_turns)==turns
    assert tuple(obs.self_public.tokens)==tokens
    assert obs.self_public.bonuses.green==green
    assert obs.self_public.purchased_card_ids==bought
    assert len(obs.opponent_public.reserved_cards)==3
    assert all(r.source==ReservationSource.MARKET for r in obs.opponent_public.reserved_cards)
    assert len(obs.market.tier_one)==4
    assert determinize_request(request,seed=789,catalog=catalog,engine_mode='fast').decision()==request


def test_game_end_invalidates_advice(snapshot):
    history=ReservationHistory.start(snapshot)
    snapshot['gamedatas']['gamestate']={'id':99,'name':'gameEnd'}
    with pytest.raises(StateUnavailable,match='对局已结束'):
        request_from_snapshot(snapshot,history)


def _midgame_with_tokens(held):
    snapshot=json.loads((ROOT/'tests/fixtures/bga_live_step8_934625108.json').read_text())
    gd=snapshot['gamedatas'];viewer=str(snapshot['player_id'])
    other=next(pid for pid in gd['players'] if pid!=viewer)
    gd['players'][viewer]['coins']=dict(zip('CSEROG',held))
    gd['players'][other]['coins']=dict(zip('CSEROG',(0,0,0,0,0,3)))
    gd['market']['pool']={c:(5 if c=='G' else 4)-gd['players'][viewer]['coins'][c]-
                         gd['players'][other]['coins'][c] for c in 'CSEROG'}
    return snapshot


@pytest.mark.parametrize('target,source',[('33','BuyReserved'),('30','BuyMarket')])
def test_selected_purchase_enumerates_payment_combinations(target,source):
    snapshot=_midgame_with_tokens((3,3,0,0,0,2))
    snapshot['gamedatas']['decision']={'decision_type':'payment','card_id':target,
                                     'selected_tokens':dict(zip('CSEROG',(0,0,0,0,0,0)))}
    request,catalog,_=request_from_snapshot(snapshot,ReservationHistory.start(snapshot))
    assert request.observation.phase is DecisionPhase.PAYMENT
    assert type(request.observation.pending.main_action).__name__==source
    assert len(request.legal_actions)==2
    assert {a.tokens.gold for a in request.legal_actions}=={1,2}
    assert all(a.tokens.total==4 for a in request.legal_actions)
    assert determinize_request(request,seed=0,catalog=catalog,engine_mode='fast').decision()==request
    # Local payment previews do not subtract from the real wallet or public bank.
    snapshot['gamedatas']['decision']['selected_tokens']['G']=1
    again,_,_=request_from_snapshot(snapshot,ReservationHistory.start(snapshot))
    assert again==request


@pytest.mark.parametrize('selected',(0,1))
def test_return_stage_recommends_only_remaining_tokens(selected):
    snapshot=_midgame_with_tokens((3,3,3,2,2,0));gd=snapshot['gamedatas']
    gd['gamestate'].update(name='playerDiscard',args=None)
    gd['decision']={'decision_type':'return_tokens','turn':4,
                    'main_action':{'kind':'take','income':dict(zip('CSEROG',(1,1,1,0,0,0)))},
                    'selected_tokens':dict(zip('CSEROG',(0,0,selected,0,0,0)))}
    request,catalog,_=request_from_snapshot(snapshot,ReservationHistory.start(snapshot))
    obs=request.observation
    assert obs.phase is DecisionPhase.RETURN_TOKENS
    assert obs.self_public.tokens.total==13-selected
    assert obs.pending.tokens_to_return==3-selected
    assert obs.supply.green==1+selected
    assert len(request.legal_actions)>1
    assert all(a.tokens.total==3-selected for a in request.legal_actions)
    engine=determinize_request(request,seed=0,catalog=catalog,engine_mode='fast')
    assert engine.decision()==request
    engine.apply(request.legal_actions[0])
    assert engine._debug_state().players[0].tokens.total==10
    # The reader never changes the captured web state.
    assert sum(gd['players'][str(snapshot['player_id'])]['coins'].values())==13


def test_return_stage_needs_real_pending_action_and_waits_for_submission():
    snapshot=_midgame_with_tokens((3,3,3,2,2,0));gd=snapshot['gamedatas']
    gd['gamestate'].update(name='playerDiscard',args=None)
    gd['decision']={'decision_type':'return_tokens','turn':4,'main_action':None,
                    'selected_tokens':dict(zip('CSEROG',(0,0,0,0,0,0)))}
    with pytest.raises(StateUnavailable,match='本手行动记录'):
        request_from_snapshot(snapshot,ReservationHistory.start(snapshot))
    gd['decision']['main_action']={'kind':'take','income':dict(zip('CSEROG',(1,1,1,0,0,0)))}
    gd['decision']['selected_tokens']['E']=3
    with pytest.raises(StateUnavailable,match='提交完成'):
        request_from_snapshot(snapshot,ReservationHistory.start(snapshot))


@pytest.mark.parametrize('blind',(False,True))
def test_discard_after_reserving_and_taking_gold(blind):
    snapshot=_midgame_with_tokens((2,2,2,2,2,1));gd=snapshot['gamedatas']
    if blind:snapshot['public_logs']=[log for log in snapshot['public_logs'] if log['card_ids']!=['33']]
    gd['gamestate'].update(name='playerDiscard',args=None)
    gd['decision']={'decision_type':'return_tokens','turn':4,
                    'main_action':{'kind':'reserve_deck','tier':1} if blind else
                                  {'kind':'reserve_market','card_id':'33'},
                    'selected_tokens':dict(zip('CSEROG',(0,0,0,0,0,0)))}
    request,catalog,_=request_from_snapshot(snapshot,ReservationHistory.start(snapshot))
    assert request.observation.pending.tokens_to_return==1
    assert type(request.observation.pending.main_action).__name__==('ReserveDeck' if blind else 'ReserveMarket')
    assert any(action.tokens.gold==1 for action in request.legal_actions)
    assert determinize_request(request,seed=0,catalog=catalog,engine_mode='fast').decision()==request


def test_online_default_search_budget_is_16384(monkeypatch):
    from splendor import bga_live
    from splendor.recommend import DEFAULT_SIMULATIONS
    captured=[]
    async def watch(args):captured.append(args)
    monkeypatch.setattr(bga_live,'watch',watch)
    bga_live.main([])
    assert DEFAULT_SIMULATIONS==captured[0].simulations==16384


def test_return_report_distinguishes_remaining_and_total():
    from splendor.bga_live import report_text
    text=report_text({'status':'ready','table':1,'move':'8','phase':'return_tokens',
        'mode':'mcts','simulations':16384,'elapsed_seconds':1.,'description':'归还绿2',
        'expected_score_viewer':.5,'selected_return_description':'归还绿1',
        'total_return_description':'归还绿3','candidates':[]})
    assert '**推荐：归还绿2**' in text and '已选：归还绿1' in text and '本次合计：归还绿3' in text
    assert text.index('**推荐：')<text.index('已选：')<text.index('候选招法：')
