"""Convert captured public BGA board states with the existing Splendor rule core.

The unseen deck remainder is a deterministic completion, not BGA's real shuffle.
Only player-safe DecisionRequest objects are exported as model inputs.
"""
import argparse
import copy
import json
import sys
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
REPO = Path('/home/lc/下载/splendor')
sys.path.insert(0, str(REPO / 'src'))

from splendor._state import DeckState, MarketState
from splendor.actions import BuyMarket, BuyReserved, ClaimNoble, Pay, ReserveDeck, ReserveMarket, ReturnTokens, TakeDistinct, TakePair
from splendor.catalog import load_default_catalog
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import encode_decision
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, Tier, TokenVector
from splendor.observation import build_observation
from splendor.replay import to_primitive
from splendor.rules import apply_legal_action, legal_actions, player_bonuses, player_score

COLORS = 'CSEROG'
CAT = load_default_catalog()

def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def normalize_capture(s):
    s = copy.deepcopy(s)
    if isinstance(s['bank'], dict):
        s['bank'] = [int(s['bank'][c]) for c in COLORS]
        s['decks'] = list(map(int, s['decks']))
        for card in s['cards']:
            card['id'] = int(card['id'].split('_')[-1])
            card['cost'] = [card['cost'].get(c, 0) for c in COLORS[:5]]
        for noble in s['nobles']:
            noble['id'] = int(noble['id'].split('_')[-1])
            noble['cost'] = [noble['cost'].get(c, 0) for c in COLORS[:5]]
        for p in s['players']:
            p['score'] = int(p['score'])
            p['tokens'] = [x['n'] for x in p['tokens']]
            p['bonus'] = [0] * 5
    return s

def market(s):
    return {c['id']: int(c['container'][-1]) for c in s['cards'] if c['container'] in ('row_1','row_2','row_3')}

def prepare(raw_path):
    states = {s['move']: normalize_capture(s) for s in json.loads(raw_path.read_text())}
    logs = json.loads((OUT / 'raw_replay_logs.json').read_text())['logs']
    assert set(states) == set(range(1,58)), sorted(set(range(1,58)) - set(states))
    # The browser exposes move numbers before score/replacement animations end.
    # Preserve raw captures; repair these evidenced transient fields separately.
    recheck = json.loads((OUT / 'recheck_after_move_53.json').read_text())
    assert recheck['located_after_bga_move'] == 53
    assert recheck['bank'] == states[53]['bank'] and recheck['decks'] == states[53]['decks']
    assert set(market(recheck)) == set(market(states[53]))
    assert [p['tokens'] for p in recheck['players']] == [p['tokens'] for p in states[53]['players']]
    assert [p['bonus'] for p in recheck['players']] == [p['bonus'] for p in states[53]['players']]
    assert recheck['players'][1]['score'] == states[54]['players'][1]['score'] == 11
    states[53]['players'][1]['score'] = recheck['players'][1]['score']
    assert states[55]['players'][0]['score'] == 14
    states[54]['players'][0]['score'] = states[55]['players'][0]['score']
    added_54 = [c for c in states[55]['cards'] if c['container']=='row_3' and c['id'] not in market(states[54])]
    added_55 = [c for c in states[56]['cards'] if c['container']=='row_1' and c['id'] not in market(states[55])]
    assert len(added_54) == len(added_55) == 1
    assert added_54[0]['id'] == 89 and added_55[0]['id'] == 38
    states[54]['cards'].extend(copy.deepcopy(added_54))
    states[55]['cards'].extend(copy.deepcopy(added_55))
    write('capture_normalization.json', {
        'raw_snapshots_preserved':True,
        'reason':'BGA DOM animations update scores and replacement cards after the move counter changes.',
        'corrections':[
            {'move':53,'field':'players[1].score','raw':7,'settled':11,
             'evidence':['recheck_after_move_53.json: opposite player perspective, bank/tokens/bonuses/market/decks agree', 'raw move 54 opponent score 11'],
             'recheck_note':'BGA move counter was blank while interface showed updating; the sought position was after 53, and the board state was cross-checked.'},
            {'move':54,'field':'players[0].score','raw':12,'settled':14,
             'evidence':['raw move 55 score 14','move 55 belongs to the other player','move 54 log awards 3 points']},
            {'move':54,'field':'market tier 3 replacement','raw':'not yet rendered','settled_bga_card_id':89,
             'evidence':['raw move 55 card 89','move 55 only buys a tier 1 card; tier 3 is unchanged']},
            {'move':55,'field':'market tier 1 replacement','raw':'not yet rendered','settled_bga_card_id':38,
             'evidence':['raw move 56 card 38','move 56 only buys a tier 2 card; tier 1 is unchanged']},
        ],
        'bonus_counter_note':'Captures 2–15 used an unchanging UI data-count attribute. Their bonus arrays are not used as evidence; bonuses are reconstructed from exact bought cards. From move 16 onwards the displayed numeric bonus is read.',
    })
    write('normalized_board_snapshots.json', [states[m] for m in sorted(states)])
    cards = {}
    nobles = {}
    for s in states.values():
        for c in s['cards']:
            if c['type'] is None:
                continue
            matches = [card for card in CAT.cards if card.bonus.value == ['white','blue','green','red','black'][COLORS.index(c['type'])]
                       and card.points == c['points'] and list(card.cost) == c['cost']]
            assert len(matches) == 1, c
            mapped = matches[0]
            if c['id'] in cards:
                assert cards[c['id']] == mapped, c
            cards[c['id']] = mapped
        for n in s['nobles']:
            if not sum(n['cost']):
                continue
            matches = [x for x in CAT.nobles if x.points == n['points'] and list(x.requirement) == n['cost']]
            assert len(matches) == 1, n
            nobles[n['id']] = matches[0]
    # The owner's second-perspective capture is the authority for the blind card.
    private = json.loads((OUT / 'blind_reservation_owner_view.json').read_text())
    blind = [c for c in private['cards'] if c['container'] == 'player_reserve']
    assert len(blind) == 1, blind
    b = blind[0]
    matches = [c for c in CAT.cards if c.points == b['points'] and list(c.cost) == b['cost']
               and c.bonus.value == ['white','blue','green','red','black'][COLORS.index(b['type'])]]
    assert len(matches) == 1
    cards[b['id']] = matches[0]
    blind_id = b['id']

    turns = []
    draws = {1: [], 2: [], 3: []}
    reserves = {0: [], 1: []}
    for log in logs:
        m = log['move']
        actor = (m - 2) % 2
        before, after = states[m-1], states[m]
        removed = set(market(before)) - set(market(after))
        added = set(market(after)) - set(market(before))
        for c in added:
            draws[market(after)[c]].append(c)
        text = log['entries'][0]['text']
        payment = [0]*6
        take = [0]*6
        if '保留' in text:
            if '牌堆' in text:
                action = ReserveDeck(Tier.ONE)
                reserves[actor].append(blind_id)
                draws[1].append(blind_id)
            else:
                assert len(removed) == 1, (m, removed)
                target = removed.pop()
                action = ReserveMarket(cards[target].card_id)
                reserves[actor].append(target)
        elif '购买' in text:
            if '手中' in text:
                previous_ids = set(c['id'] for c in before['cards'] if c['container'] == 'player_reserve')
                next_ids = set(c['id'] for c in after['cards'] if c['container'] == 'player_reserve')
                if m == 15:
                    target = blind_id
                elif actor == 0:
                    removed_reserved = previous_ids - next_ids
                    assert len(removed_reserved) == 1, (m, removed_reserved)
                    target = removed_reserved.pop()
                else:
                    delta = [b-a for a,b in zip(before['players'][actor]['bonus'], after['players'][actor]['bonus'])]
                    targets = [c for c in reserves[actor] if delta[list(Color).index(cards[c].bonus)] == 1]
                    assert len(targets) == 1, (m, targets)
                    target = targets[0]
                assert target in reserves[actor], (m, target, reserves)
                reserves[actor].remove(target)
                action = BuyReserved(cards[target].card_id)
            else:
                assert len(removed) == 1, (m, removed)
                target = removed.pop()
                action = BuyMarket(cards[target].card_id)
            for entry in log['entries']:
                if '花费' in entry['text']:
                    for icon in entry['icons']:
                        payment[COLORS.index(icon[-1])] += 1
        elif '拿取' in text:
            for icon in log['entries'][0]['icons']:
                take[COLORS.index(icon[-1])] += 1
            if max(take) == 2:
                action = TakePair(tuple(Color)[take.index(2)])
            else:
                action = TakeDistinct(tuple(tuple(Color)[i] for i,n in enumerate(take) if n))
        else:
            raise ValueError((m, text))
        claimed = {n['id'] for n in before['nobles'] if n['container']=='noblesbar'} - {n['id'] for n in after['nobles'] if n['container']=='noblesbar'}
        turns.append({'bga_move':m, 'actor':actor, 'action':action, 'payment':TokenVector(*payment),
                      'noble_ids':tuple(nobles[n].noble_id for n in sorted(claimed)), 'source_time':log['time']})

    initial = states[1]
    base = GameEngine.new(0, first_player=0, catalog=CAT, mode='fast')._debug_state()
    markets = [tuple(cards[c].card_id for c,t in market(initial).items() if t==tier) for tier in (1,2,3)]
    deck_lists = []
    for tier in (1,2,3):
        known = [cards[c].card_id for c in draws[tier]]
        assert len(known) == len(set(known)), known
        remaining = [c.card_id for c in CAT.cards if int(c.tier)==tier and c.card_id not in known and c.card_id not in markets[tier-1]]
        deck_lists.append(tuple(remaining + list(reversed(known))))
    in_play = tuple(nobles[n['id']].noble_id for n in initial['nobles'] if n['container']=='noblesbar')
    state = replace(base, market=MarketState(*markets), decks=DeckState(*deck_lists),
                    nobles_in_play=in_play, nobles_out_of_game=tuple(n.noble_id for n in CAT.nobles if n.noble_id not in in_play))
    assert_game_state(state, CAT)
    return states, cards, nobles, turns, state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures', type=Path, default=OUT / 'raw_board_snapshots.json')
    args = parser.parse_args()
    states, cards, nobles, turns, state = prepare(args.captures)
    samples = []
    encoded_arrays = {}
    validations = []
    phase_counts = Counter()
    initial = state
    def apply(action, turn, automatic=False):
        nonlocal state
        candidates = legal_actions(state, CAT)
        assert action in candidates, (turn['bga_move'], state.phase, action, candidates)
        request = DecisionRequest(build_observation(state, state.current_player, CAT), candidates)
        alternate_decks = DeckState(*(tuple(reversed(state.decks.for_tier(t))) for t in Tier))
        alternate = replace(state, decks=alternate_decks)
        assert build_observation(alternate, state.current_player, CAT) == request.observation
        assert legal_actions(alternate, CAT) == candidates
        assert all(r.card is None for r in request.observation.opponent_public.reserved_cards if r.source.value == 'deck')
        primitive = to_primitive(request)
        assert decode_request(primitive) == request
        encoded = encode_decision(request)
        i = len(samples)
        for field in fields(encoded):
            value = getattr(encoded, field.name)
            if hasattr(value, 'dtype'):
                encoded_arrays[f'{i:03d}/{field.name}'] = value
        phase_counts[state.phase.value] += 1
        samples.append({'decision_index':i, 'bga_move':turn['bga_move'], 'actor':state.current_player,
                        'request':primitive, 'action':to_primitive(action),
                        'candidate_index':candidates.index(action), 'automatic':automatic,
                        'outcome_value':1 if state.current_player==0 else -1})
        state = apply_legal_action(state, action, CAT)
        assert_game_state(state, CAT)

    for turn in turns:
        m, actor = turn['bga_move'], turn['actor']
        assert state.current_player == actor and state.phase == DecisionPhase.MAIN_ACTION
        apply(turn['action'], turn)
        if state.phase == DecisionPhase.PAYMENT:
            apply(Pay(turn['payment']), turn)
        if state.phase == DecisionPhase.RETURN_TOKENS:
            raise ValueError(f'Explicit return event requires capture reconstruction at move {m}')
        if state.phase == DecisionPhase.CHOOSE_NOBLE:
            assert len(turn['noble_ids']) == 1
            apply(ClaimNoble(turn['noble_ids'][0]), turn, automatic=len(legal_actions(state,CAT))==1)
        observed = states[m]
        assert list(state.supply) == observed['bank'], (m,'bank',list(state.supply),observed['bank'])
        assert [len(state.decks.for_tier(t)) for t in Tier] == observed['decks'], (m,'decks')
        for tier in Tier:
            expected = {cards[c].card_id for c,t in market(observed).items() if t==int(tier)}
            assert set(state.market.for_tier(tier)) == expected, (m,'market',tier)
        for p in (0,1):
            assert list(state.players[p].tokens) == observed['players'][p]['tokens'], (m,'tokens',p)
            assert player_score(state.players[p],CAT) == observed['players'][p]['score'], (m,'score',p)
            if m >= 17:
                assert list(player_bonuses(state.players[p],CAT)) == observed['players'][p]['bonus'], (m,'bonuses',p)
        validations.append({'bga_move':m,'status':'matched','bank':True,'players':True,'markets':True,'deck_counts':True})
    assert state.result is not None
    assert state.result.scores == (15,13)
    assert state.result.completed_turns == (28,28)
    api_engine = GameEngine(initial, CAT, None, ROOT / 'work/reconstruction_failures', mode='fast')
    for sample in samples:
        if sample['automatic']:
            continue
        actual = api_engine.decision()
        assert actual == decode_request(sample['request'])
        api_engine.apply(actual.legal_actions[sample['candidate_index']])
    assert api_engine.decision() == state.result
    write('actions.json', {'table_id':927500015,'turns':[{**t,'action':to_primitive(t['action']),'payment':to_primitive(t['payment'])} for t in turns]})
    (OUT / 'model_decisions.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in samples),encoding='utf-8')
    import numpy as np
    encoded_arrays.update({
        'labels/decision_index':np.arange(len(samples),dtype=np.int64),
        'labels/bga_move':np.array([s['bga_move'] for s in samples],dtype=np.int64),
        'labels/actor':np.array([s['actor'] for s in samples],dtype=np.int64),
        'labels/candidate_index':np.array([s['candidate_index'] for s in samples],dtype=np.int64),
        'labels/outcome_value':np.array([s['outcome_value'] for s in samples],dtype=np.float32),
        'labels/automatic':np.array([s['automatic'] for s in samples],dtype=np.bool_),
    })
    np.savez_compressed(OUT / 'model_features.npz', **encoded_arrays)
    write('reconstruction.json', {'format':'bga-observed-game-v1','table_id':927500015,
                                 'initial_state_with_synthetic_unseen_deck_completion':to_primitive(initial),
                                 'warning':'Not a BGA seed or shuffle. Unobserved deck remainder is a canonical completion for checked reconstruction only; never pass this state to a model.',
                                 'terminal_result':to_primitive(state.result),
                                 'catalog_fingerprint':CAT.data_fingerprint})
    write('card_id_mapping.json', {'cards':{str(k):to_primitive(v) for k,v in sorted(cards.items())},
                                   'nobles':{str(k):to_primitive(v) for k,v in sorted(nobles.items())}})
    write('validation.json', {'status':'passed','turns':len(turns),'decision_samples':len(samples),
                              'phase_counts':dict(phase_counts),'terminal_result':to_primitive(state.result),
                              'checked_steps':validations,'catalog_fingerprint':CAT.data_fingerprint,
                              'decode_request_roundtrip':True,'feature_encoding':True,
                              'engine_api_replay':True,
                              'unseen_deck_order_invariance_at_every_decision':True,
                              'opponent_blind_reservations_masked':True,
                              'input_hidden_information_boundary':'Existing build_observation and encode_decision only'})
    write('features_manifest.json', {
        'format':'per-decision-numpy-arrays-v1','schema':'splendor-observation-action-v1',
        'load':'numpy.load("model_features.npz", allow_pickle=False)',
        'array_key_pattern':'{decision_index:03d}/{feature_field}',
        'labels':{k:str(v.dtype) for k,v in encoded_arrays.items() if k.startswith('labels/')},
        'first_decision_arrays':{k.split('/',1)[1]:{'shape':list(v.shape),'dtype':str(v.dtype)}
                                 for k,v in encoded_arrays.items() if k.startswith('000/')},
        'ragged_candidates':'Each decision stores exactly its own legal candidate count.',
        'sample_labels':'Human choices and final outcome only. No MCTS policy targets or teacher values.',
    })
    print(json.dumps({'turns':len(turns),'decisions':len(samples),'phases':dict(phase_counts),'result':to_primitive(state.result)},ensure_ascii=False))

if __name__ == '__main__':
    main()
