"""Inspect four disjoint groups of four saved worlds; no extra simulations."""
import json
import numpy as np
from splendor.alphazero_recording import write_json
from analyze_n16 import ART, RAW, identity
import analyze as base


def main():
    identity()
    rows=base.records();saved={r['decision_index']:r for r in json.loads((RAW/'search.json').read_text())}
    original={r['decision_index']:r for r in json.loads((base.RAW/'search.json').read_text())}
    assert len(saved)==71
    results=[]
    for row in rows:
        if row['phase']!='main_action':continue
        i=row['decision_index'];r=saved[i];v=np.asarray(r['per_world_visits']);actual=row['candidate_index']
        choices=[];shares=[];ranks=[]
        for start in (0,4,8,12):
            counts=v[start:start+4].sum(axis=0)
            order=sorted(range(len(counts)),key=lambda j:(-counts[j],-r['prior'][j],j))
            choices.append(order[0]);shares.append(float(counts[actual]/4096));ranks.append(order.index(actual)+1)
        if r['move'] in (8,21,26,40):
            print(json.dumps(dict(move=r['move'],group_choices=choices,actual_shares=shares,actual_ranks=ranks),ensure_ascii=False),flush=True)
        results.append(dict(move=row['bga_move'],actor=row['actor'],actual_index=actual,choices=choices,
            actual_shares=shares,actual_ranks=ranks,unique_group_choices=len(set(choices)),
            n16_agrees=r['choice']==actual,first_group_reproduces_old=choices[0]==original[i]['choice']))
    write_json(RAW/'world_groups.json',results)
    summary=dict(method='Four disjoint saved groups of worlds 0-3,4-7,8-11,12-15; 1024 simulations/world. No extra search. Groups are not four independent global seeds.',
        main_roots=58,group_top1_changes=sum(r['unique_group_choices']>1 for r in results),
        mean_distinct_choices=float(np.mean([r['unique_group_choices'] for r in results])),
        first_group_reproduces_old_roots=sum(r['first_group_reproduces_old'] for r in results),players={})
    for actor,player in enumerate(base.PLAYERS):
        group=[r for r in results if r['actor']==actor]
        summary['players'][player]=dict(n=29,
            group_top1=[sum(r['choices'][j]==r['actual_index'] for r in group) for j in range(4)],
            group_top3=[sum(r['actual_ranks'][j]<=3 for r in group) for j in range(4)],
            unanimous_group_choices=sum(r['unique_group_choices']==1 for r in group),
            actual_chosen_in_any_group=sum(r['actual_index'] in r['choices'] for r in group))
    write_json(ART/'world_groups_summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
