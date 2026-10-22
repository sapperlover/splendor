"""Describe conditional future choices without supplying future cards to search."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = ROOT / 'artifacts/bga_934713322_move20_061_20261021'


def save(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def early_nodes(result):
    evidence=result['early_nodes_evidence']
    path=ROOT/evidence['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==evidence['sha256']
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        nodes=json.load(stream)
    assert len(nodes)==evidence['nodes']
    return nodes


def signature(state, hide_own_blind=False):
    result = {k:state[k] for k in ('phase','actor','first','turns','supply','trigger','return_count')}
    result['deck_counts'] = [len(d) for d in state['decks']]
    result['market'] = [sorted(d) for d in state['market']]
    result['nobles'] = sorted(state['nobles'])
    result['players'] = []
    for seat,p in enumerate(state['players']):
        result['players'].append({'tokens':p['tokens'],'purchased':sorted(p['purchased']),
            'nobles':sorted(p['nobles']),'reserved':sorted(
                ((-1 if source==1 and (seat==1 or hide_own_blind) else card),source)
                for card,source in p['reserved'])})
    return result


def main():
    greedy = json.loads((OUT/'greedy.json').read_text())
    old = json.loads((OUT/'reference_move20.json').read_text())
    target = early_nodes(old)[0]['state']
    target_public = signature(target,True)
    target_exact = signature(target)
    root_request = json.loads((OUT/'root18_request.json').read_text())
    # Catalog order is fixed; decode the few IDs needed for matching statistics.
    from splendor.bga_live import runtime_catalog
    from splendor.recommend import describe_action
    from splendor.native_policy import NativeSearchJob
    from types import SimpleNamespace
    snap = json.loads((OUT/'root18_snapshot.json').read_text())
    catalog, cardmap, _ = runtime_catalog(snap['gamedatas'])
    buy_card = next(i for i,c in enumerate(catalog.cards) if c.card_id=='061')
    reserve_card = next(i for i,c in enumerate(catalog.cards) if c.card_id==cardmap['45'].card_id)
    dummy = SimpleNamespace(policy=SimpleNamespace(catalog=catalog))
    def describe(raw):
        return describe_action(NativeSearchJob._action_from_native(dummy,raw),catalog)
    def node_data(n):
        actions=n['actions'];visits=n['visits'];prior=n['prior']
        ranking=sorted(range(len(actions)),key=lambda i:(-visits[i],-prior[i],i))
        buy=next((i for i,a in enumerate(actions) if a[0]==4 and a[1]==buy_card),None)
        chosen=ranking[0]
        return {'world':n['world'],'node':n['node'],'outgoing_visits':sum(visits),
            'branch':[describe(a) for a in n['branch']],
            'choice':describe(actions[chosen]),'buy061_legal':buy is not None,
            'buy061_rank':ranking.index(buy)+1 if buy is not None else None,
            'buy061_visits':visits[buy] if buy is not None else 0,
            'buy061_prior':prior[buy] if buy is not None else None,
            'buy061_top1':buy==chosen,
            'buy061_q':n['value_sums_root_actor'][buy]/visits[buy] if buy is not None and visits[buy] else None,
            'own_blind_cards':[catalog.cards[card].card_id for card,source in n['state']['players'][0]['reserved'] if source==1],
            'market_tier2':[catalog.cards[c].card_id for c in n['state']['market'][1]],
            'top3':[{'action':describe(actions[i]),'visits':visits[i],'prior':prior[i],
                'q_root_actor':n['value_sums_root_actor'][i]/visits[i] if visits[i] else None} for i in ranking[:3]]}
    def aggregate(nodes):
        result={'expanded_nodes':len(nodes),'thresholds':{}}
        for threshold in (1,8,32,128):
            selected=[n for n in nodes if sum(n['visits'])>=threshold]
            data=[node_data(n) for n in selected]
            result['thresholds'][str(threshold)]={'nodes':len(data),
                'buy061_top1_nodes':sum(n['buy061_top1'] for n in data),
                'sum_outgoing_visits':sum(n['outgoing_visits'] for n in data),
                'buy061_visits':sum(n['buy061_visits'] for n in data),
                'choice_counts':dict(Counter(n['choice'] for n in data))}
        return result
    validations=[];summaries=[];details={}
    for path in sorted((OUT/'search').glob('*.json')):
        result=json.loads(path.read_text());nodes=early_nodes(result)
        root=nodes[0]
        assert root['node']==0 and root['world']==0 and not root['branch']
        assert sum(result['root_visit_counts'])==result['simulations']
        assert result['transition_cap_hits']==0
        for n in nodes:
            assert len(n['visits'])==len(n['actions'])==len(n['prior'])
            assert all(abs(v)<=count+1e-6 for v,count in zip(n['value_sums_root_actor'],n['visits']))
        baseline=OUT/'baseline'/path.name
        control=OUT/'controls'/path.name
        if baseline.exists():
            b=json.loads(baseline.read_text())
            c=json.loads(control.read_text())
            assert c['root_visit_counts']==b['root_visit_counts'],path
            assert c['per_world_root_value_sums']==b['per_world_root_value_sums'],path
            validations.append({'panel':path.stem,'same_batch_control_visits_exactly_equal':True,
                'same_batch_control_value_sums_exactly_equal':True,
                'matrix_different_batch_root_visits_equal':result['root_visit_counts']==b['root_visit_counts']})
        next_turn=[n for n in nodes if n['state']['actor']==0 and n['state']['turns']==target['turns']]
        blind=[n for n in next_turn if n['branch'] and n['branch'][0][0]==3 and n['branch'][0][2]==0]
        same_actions=[n for n in blind if len(n['branch'])==2 and n['branch'][1][0]==2 and n['branch'][1][1]==reserve_card]
        public=[n for n in next_turn if signature(n['state'],True)==target_public]
        exact=[n for n in next_turn if signature(n['state'])==target_exact]
        root_ranking=sorted(range(len(greedy['prior'])),key=lambda i:(-result['root_visit_counts'][i],-greedy['prior'][i],i))
        buy=root_request['legal_actions'].index(next(a for a in root_request['legal_actions'] if a.get('card_id')=='061' and a['__type__']=='BuyMarket'))
        summaries.append({'panel':path.stem,
            'root_choice':greedy['actions'][result['chosen_candidate_index']],
            'root_top3':[{'action':greedy['actions'][i],'visits':result['root_visit_counts'][i]} for i in root_ranking[:3]],
            'root_buy061_rank':root_ranking.index(buy)+1,
            'root_buy061_visits':result['root_visit_counts'][buy],
            'next_own_turn_all':aggregate(next_turn),
            'after_blind_reserve_tier1':aggregate(blind),
            'after_same_two_actions':aggregate(same_actions),
            'same_public_position_any_own_blind':aggregate(public),
            'exact_move20_information_set':aggregate(exact)})
        details[path.stem]={'after_same_two_actions':[node_data(n) for n in same_actions],
            'same_public_position_any_own_blind':[node_data(n) for n in public],
            'exact_move20_information_set':[node_data(n) for n in exact],
            'after_blind_reserve_32plus':[node_data(n) for n in blind if sum(n['visits'])>=32]}
    previous=json.loads((PREVIOUS/'reference_S1024_seed0.json').read_text())
    assert old['root_visit_counts']==previous['root_visit_counts']
    assert len(summaries)==24
    save('validation.json',{'root18_three_baselines':validations,
        'move20_original_full_visit_vector_reproduced':True,
        'all_24_searches_complete':True,'zero_transition_caps':True,
        'q_sign':'All value sums use the root actor view; selected future nodes are also that actor.',
        'matching':'Deck order excluded; own blind identity excluded only in the explicitly named public-position group.',
        'opponent_public_reserve_card':cardmap['45'].card_id})
    save('summary.json',{'root18_greedy_choice':greedy['actions'][greedy['ranking'][0]],
        'root18_greedy_buy061_prior':greedy['prior'][buy],
        'root18_greedy_buy061_rank':greedy['ranking'].index(buy)+1,'rows':summaries})
    save('conditional_nodes.json',details)
    for row in summaries:
        print(row['panel'],row['root_choice'],
            'blind',row['after_blind_reserve_tier1']['thresholds']['32'],
            'same_actions',row['after_same_two_actions']['thresholds']['1'],
            'exact',row['exact_move20_information_set']['thresholds']['1'])


if __name__=='__main__':
    main()
