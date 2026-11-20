"""Rebuild the finished BGA game from its saved browser replay notifications.

Unrevealed deck suffixes are synthetic. Analysis exports actor-safe requests only.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from splendor._state import DeckState, MarketState
from splendor.actions import BuyMarket, BuyReserved, Pay, ReserveDeck, ReserveMarket, ReturnTokens, TakeDistinct, TakePair
from splendor.bga_live import runtime_catalog
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import encode_decision
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, Tier, TokenVector
from splendor.observation import build_observation
from splendor.replay import to_primitive
from splendor.rules import apply_legal_action, legal_actions, player_score

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RAW = ROOT / 'runtime' / ART.name
LETTERS = 'CSEROG'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct():
    capture = json.loads((RAW / 'capture_zero.json').read_text())
    gd = capture['initial']['gamedatas']
    cat, cards, nobles = runtime_catalog(gd)
    assert all(int(gd[k]) == 2 for k in ('expansion_cities', 'expansion_tradings_post', 'expansion_orient', 'expansion_strongholds'))
    ids = list(map(str, gd['playerorder']))
    assert ids[0] == str(gd['first_player'])
    seats = {pid:i for i,pid in enumerate(ids)}
    logs = capture['logs']
    public = [p for p in logs if p['channel'] == '/table/t939469760']
    assert [int(p['move_id']) for p in public] == list(range(1,64))
    private = {}
    for packet in logs:
        if packet['channel'].startswith('/player/'):
            for n in packet['data']:
                if n['type'] == 'reserveCard':
                    private[int(packet['move_id'])] = n['args']
    assert set(private) == {22,51}
    market = {t:[str(c['type']) for c in gd['market'][f'row_{t}']['cards'].values()] for t in (1,2,3)}
    draws = {t:[] for t in (1,2,3)}
    for packet in public[1:]:
        move = int(packet['move_id'])
        for n in packet['data']:
            a = n['args']
            if n['type'] == 'reserveCard' and 'drawpile' in a:
                identity = private[move]
                assert identity['player_id'] == a['player_id']
                assert identity['card']['location'] == f"draw_{a['drawpile']}"
                draws[int(a['drawpile'])].append(str(identity['card']['type']))
            if n['type'] == 'revealCard':
                c = a['card']; tier = int(c['location'].split('_')[1])
                draws[tier].append(str(c['type']))
    decks = []
    for t in (1,2,3):
        known = [cards[c].card_id for c in draws[t]]
        visible = [cards[c].card_id for c in market[t]]
        assert len(set(known+visible)) == len(known+visible)
        rest = [c.card_id for c in cat.cards if int(c.tier)==t and c.card_id not in known+visible]
        decks.append(tuple(rest + list(reversed(known))))
    base = GameEngine.new(0, first_player=0, catalog=cat, mode='fast')._debug_state()
    chosen_nobles = tuple(n.noble_id for n in nobles.values())
    initial = replace(base, market=MarketState(*(tuple(cards[c].card_id for c in market[t]) for t in (1,2,3))),
        decks=DeckState(*decks), nobles_in_play=chosen_nobles,
        nobles_out_of_game=tuple(n.noble_id for n in cat.nobles if n.noble_id not in chosen_nobles))
    state = initial
    rows, states, checks = [], [], []
    tokens = [[0]*6 for _ in ids]
    scores = [0,0]
    bought = [set(),set()]
    reserved = [set(),set()]
    deck_counts = [int(gd['market'][f'row_{t}']['count']) for t in (1,2,3)]
    api = GameEngine(initial, cat, None, RAW/'failures', mode='fast')

    def apply(action, move):
        nonlocal state
        assert_game_state(state,cat)
        candidates = legal_actions(state,cat)
        assert action in candidates,(move,state.phase,action,candidates)
        req = DecisionRequest(build_observation(state,state.current_player,cat),candidates)
        assert api.decision() == req
        alt = replace(state,decks=DeckState(*(tuple(reversed(state.decks.for_tier(t))) for t in Tier)))
        assert build_observation(alt,state.current_player,cat) == req.observation
        assert legal_actions(alt,cat) == candidates
        assert all(r.card is None for r in req.observation.opponent_public.reserved_cards if r.source.value=='deck')
        assert decode_request(to_primitive(req)) == req
        encode_decision(req)
        rows.append(dict(decision_index=len(rows),bga_move=move,actor=state.current_player,
            turn=state.completed_turns[state.current_player]+1,phase=state.phase.value,
            request=to_primitive(req),action=to_primitive(action),candidate_index=candidates.index(action)))
        states.append(to_primitive(state))
        state = apply_legal_action(state,action,cat)
        api.apply(action)
        assert api._debug_state() == state

    for packet in public[1:]:
        move = int(packet['move_id'])
        for notification in packet['data']:
            kind,a = notification['type'],notification['args']
            if kind not in ('reserveCard','buyCard','coins','revealCard','updateScore'):
                continue
            actor = seats[str(a['player_id'])] if 'player_id' in a else None
            if kind == 'reserveCard':
                if 'drawpile' in a:
                    key = str(private[move]['card']['type'])
                    t = int(a['drawpile'])
                    action = ReserveDeck(Tier(t)); deck_counts[t-1] -= 1
                else:
                    key = str(a['card']['type']); t = int(cards[key].tier)
                    market[t].remove(key)
                    action = ReserveMarket(cards[key].card_id)
                assert state.current_player == actor
                apply(action,move); reserved[actor].add(key)
            elif kind == 'buyCard':
                key = str(a['card']['type']); t = int(cards[key].tier)
                if a['card']['location'] == 'hand':
                    action = BuyReserved(cards[key].card_id); reserved[actor].remove(key)
                else:
                    assert a['card']['location'] == f'market_{t}'
                    action = BuyMarket(cards[key].card_id); market[t].remove(key)
                assert state.current_player == actor
                apply(action,move); bought[actor].add(key)
            elif kind == 'coins':
                gap = [int(a['gap'][c]) for c in LETTERS]
                if state.phase is DecisionPhase.PAYMENT:
                    assert not a['income_coins']
                    apply(Pay(TokenVector(*[-v for v in gap])),move)
                elif state.phase is DecisionPhase.RETURN_TOKENS:
                    assert not a['income_coins']
                    apply(ReturnTokens(TokenVector(*[-v for v in gap])),move)
                elif any(gap[:5]):
                    assert min(gap)>=0 and gap[-1]==0
                    colors=tuple(tuple(Color)[i] for i,v in enumerate(gap) if v)
                    action=TakePair(colors[0]) if max(gap)==2 else TakeDistinct(colors)
                    assert state.current_player == actor
                    apply(action,move)
                else:
                    assert gap == [0,0,0,0,0,1],(move,gap)
                tokens[actor] = [x+y for x,y in zip(tokens[actor],gap)]
            elif kind == 'revealCard':
                c=a['card']; t=int(c['location'].split('_')[1]); key=str(c['type'])
                market[t].append(key); deck_counts[t-1]-=1
            elif kind == 'updateScore':
                scores[actor]=int(a['score'])
        assert_game_state(state,cat)
        assert list(state.supply) == [tuple(cat.constants.initial_tokens)[i]-sum(p[i] for p in tokens) for i in range(6)]
        assert [len(state.decks.for_tier(t)) for t in Tier] == deck_counts,(move,'deck')
        for t in (1,2,3):
            assert set(state.market.for_tier(Tier(t))) == {cards[c].card_id for c in market[t]},(move,'market')
        for p in (0,1):
            assert list(state.players[p].tokens)==tokens[p],(move,'tokens',p)
            assert player_score(state.players[p],cat)==scores[p],(move,'score',p)
            assert set(state.players[p].purchased_card_ids)=={cards[c].card_id for c in bought[p]}
            assert {r.card_id for r in state.players[p].reserved_cards}=={cards[c].card_id for c in reserved[p]}
        # Compare BGA's announced next actor and round when a turn settles.
        changes=[n['args'] for n in packet['data'] if n['type']=='gameStateChange']
        last=changes[-1]
        if int(last.get('id',0))==25:
            assert state.phase is DecisionPhase.MAIN_ACTION
            assert state.current_player == seats[str(last['active_player'])]
            assert int(last['args']['turn']) == state.completed_turns[state.current_player]+1
        elif int(last.get('id',0))==40:
            assert state.phase is DecisionPhase.RETURN_TOKENS
        elif int(last.get('id',0))==99:
            assert state.result is not None
        checks.append(dict(move=move,status='all_notification_fields_matched'))
    assert state.result.scores==(13,16) and state.result.completed_turns==(29,29)
    assert api.decision()==state.result
    assert Counter(r['phase'] for r in rows)=={'main_action':58,'payment':24,'return_tokens':4}
    write(RAW/'reconstruction.json',dict(initial_state=to_primitive(initial),final_state=to_primitive(state),
        warning='Unseen deck completion is synthetic, never a model input. Original BGA shuffle/seed unavailable.'))
    write(RAW/'states.json',states)
    (RAW/'model_decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write(RAW/'card_id_mapping.json',dict(cards={k:c.card_id for k,c in cards.items()},nobles={k:n.noble_id for k,n in nobles.items()}))
    write(RAW/'reconstruction_checks.json',checks)
    for p in (ROOT/'data').glob('splendor_*'):
        if p.suffix in ('.csv','.json'):shutil.copyfile(p, RAW/p.name)
    summary=dict(table=939469760,source_url='https://boardgamearena.com/gamereview?table=939469760',
        players=[dict(seat=i,id=pid,name=gd['players'][pid]['name']) for i,pid in enumerate(ids)],
        result=to_primitive(state.result),packets=len(logs),public_packets=len(public),private_blind_records=len(private),
        decisions=len(rows),phase_counts=dict(Counter(r['phase'] for r in rows)),catalog_fingerprint=cat.data_fingerprint,
        validation=dict(all_62_moves_matched=True,engine_api_replay=True,all_actions_legal=True,
            unseen_deck_order_invariant=True,opponent_blind_masked=True,feature_encoding=True),
        raw_directory=str(RAW.relative_to(ROOT)),unavailable=['original_shuffle_seed','unrevealed_deck_order'])
    write(ART/'capture_summary.json',summary)
    write(RAW/'SHA256SUMS.json',{p.name:sha(p) for p in sorted(RAW.iterdir()) if p.is_file() and p.name not in ('SHA256SUMS.json','replay_backup.zip')})
    with zipfile.ZipFile(RAW/'replay_backup.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RAW.iterdir()):
            if p.is_file() and p.name!='replay_backup.zip':z.write(p,p.name)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':reconstruct()
