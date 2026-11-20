"""Check the saved N16 panel independently without running another search."""
import json
import numpy as np

from splendor.alphazero_recording import write_json
from analyze_n16 import ART, RAW, TOTAL, identity
import analyze as base


def main():
    identity();plan=json.loads((ART/'plan.json').read_text())
    for name,digest in plan['input_sha256'].items():assert base.sha(base.ROOT/name)==digest,name
    rows=base.records();greedy=json.loads((base.RAW/'greedy.json').read_text())
    old={r['decision_index']:r for r in json.loads((base.RAW/'search.json').read_text())}
    saved=json.loads((RAW/'search.json').read_text())
    assert len(saved)==len(old)==71
    assert {r['decision_index'] for r in saved}==set(old)
    checked=[]
    for r in saved:
        i=r['decision_index'];v=np.asarray(r['visits'],dtype=np.int64)
        wv=np.asarray(r['per_world_visits'],dtype=np.int64);ws=np.asarray(r['per_world_value_sums'],dtype=float)
        assert wv.shape==ws.shape==(16,len(v))
        assert np.all(np.isfinite(ws)) and np.all(wv>=0)
        assert v.sum()==TOTAL and tuple(r['simulation_budgets'])==(1024,)*16
        assert np.all(wv.sum(axis=1)==1024)
        np.testing.assert_array_equal(v,wv.sum(axis=0))
        np.testing.assert_allclose(v/TOTAL,(wv/1024).mean(axis=0),rtol=0,atol=1e-14)
        np.testing.assert_allclose(r['prior'],greedy[i]['prior'],rtol=0,atol=2e-6)
        best=max(range(len(v)),key=lambda j:(v[j],r['prior'][j],-j))
        assert r['choice']==best
        for j,n in enumerate(v):
            if n:assert abs(r['q'][j]-ws[:,j].sum()/n)<1e-12
            else:assert r['q'][j] is None
        checked.append(dict(index=i,prefix_equal=np.array_equal(np.asarray(old[i]['per_world_visits']),wv[:4])))
    summary=json.loads((ART/'summary.json').read_text())
    main=[r for r in rows if r['phase']=='main_action']
    by_id={r['decision_index']:r for r in saved}
    assert len(main)==58
    for actor,player in enumerate(base.PLAYERS):
        group=[r for r in main if r['actor']==actor]
        expected=[]
        for row in group:
            r=by_id[row['decision_index']];actual=row['candidate_index']
            rank=sorted(range(len(r['visits'])),key=lambda j:(-r['visits'][j],-r['prior'][j],j)).index(actual)+1
            expected.append((r['choice']==actual,rank<=3))
        stats=summary['stats'][player]['S1024N16']
        assert stats['n']==29 and stats['top1']==sum(a for a,b in expected) and stats['top3']==sum(b for a,b in expected)
    assert summary['complete'] and summary['total_simulations']==71*TOTAL
    assert sum(x['prefix_equal'] for x in checked)==summary['prefix_exact_roots']
    result=dict(status='passed',roots=71,per_world_simulations=1024,worlds=16,total_per_root=TOTAL,
        total_simulations=71*TOTAL,world_normalization=True,choice_tie_break=True,prior_matches_greedy=True,
        pooled_q_verified=True,summary_counts_verified=True,prefix_exact_roots=summary['prefix_exact_roots'],
        n4_input_shas_unchanged=True,source_model_request_shas_verified=True,
        search_sha256=base.sha(RAW/'search.json'),summary_sha256=base.sha(ART/'summary.json'),
        raw_path=str(RAW.relative_to(base.ROOT)))
    write_json(ART/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
