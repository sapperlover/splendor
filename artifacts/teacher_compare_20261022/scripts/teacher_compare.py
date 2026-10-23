import gzip,json,math,collections,statistics
from pathlib import Path
ART=Path('artifacts/entity_attention_r20_search_stability_128_20261021')
bank=json.load(open(ART/'root_bank.json')); meta={r['id']:r for r in bank['roots']}
recs=collections.defaultdict(lambda: collections.defaultdict(dict))
with gzip.open(ART/'raw_searches.jsonl.gz','rt') as f:
    for line in f:
        d=json.loads(line); recs[d['panel']][d['root_id']][d['seed']]=d
def js(p,q):
    m=[(a+b)/2 for a,b in zip(p,q)]
    def kl(a,b): return sum(x*math.log(x/y) for x,y in zip(a,b) if x>0 and y>0)
    return 0.5*kl(p,m)+0.5*kl(q,m)
def mode(vals):
    c=collections.Counter(vals); v,n=c.most_common(1)[0]; return v,n/len(vals)
out={}
for panel in ('S256N4','S1024N1'):
    rows=[]
    for rid, seedmap in recs[panel].items():
        seeds=sorted(seedmap); chosen=[seedmap[s]['chosen'] for s in seeds]
        greedy=recs['greedy'][rid]; gchosen={x['chosen'] for x in greedy.values()}; assert len(gchosen)==1; gchosen=next(iter(gchosen))
        ref=recs['reference_S16384N4'][rid]; ref_mode,_=mode([x['chosen'] for x in ref.values()])
        pol=[seedmap[s]['policy'] for s in seeds]
        mean_pol=[sum(p[i] for p in pol)/len(pol) for i in range(len(pol[0]))]
        onehot=[1.0 if i==gchosen else 0.0 for i in range(len(mean_pol))]
        m,mfrac=mode(chosen); stable=len(set(chosen))==1
        rows.append(dict(rid=rid,group=meta[rid]['group'],stage=meta[rid]['stage'],stable=stable,mode=m,mode_frac=mfrac,
            match_greedy=(m==gchosen),seed_match_greedy=sum(c==gchosen for c in chosen)/len(chosen),
            stable_same_greedy=(stable and m==gchosen),stable_diff_greedy=(stable and m!=gchosen),
            match_ref_mode=(m==ref_mode),seed_match_ref=sum(c==ref_mode for c in chosen)/len(chosen),
            mass_greedy=mean_pol[gchosen],js_greedy=js(mean_pol,onehot),
            entropy=-sum(x*math.log(x) for x in mean_pol if x>0)))
    out[panel]=rows
def agg(rows):
    n=len(rows); return {
      'roots':n,'stable_rate':sum(r['stable'] for r in rows)/n,
      'mode_match_greedy':sum(r['match_greedy'] for r in rows)/n,
      'seed_match_greedy':sum(r['seed_match_greedy'] for r in rows)/n,
      'stable_same_greedy':sum(r['stable_same_greedy'] for r in rows)/n,
      'stable_diff_greedy':sum(r['stable_diff_greedy'] for r in rows)/n,
      'stable_same_n':sum(r['stable_same_greedy'] for r in rows),
      'stable_diff_n':sum(r['stable_diff_greedy'] for r in rows),
      'stable_n':sum(r['stable'] for r in rows),
      'mode_match_ref_S16384':sum(r['match_ref_mode'] for r in rows)/n,
      'seed_match_ref_S16384':sum(r['seed_match_ref'] for r in rows)/n,
      'mass_greedy':statistics.mean(r['mass_greedy'] for r in rows),
      'js_greedy':statistics.mean(r['js_greedy'] for r in rows),
      'entropy':statistics.mean(r['entropy'] for r in rows)}
res={}
for panel,rows in out.items():
    res[panel]={}
    for group in ('baseline','high_disagreement','all'):
        res[panel][group]=agg([r for r in rows if group=='all' or r['group']==group])
print(json.dumps(res,indent=2))
Path('runtime/teacher_compare_summary.json').write_text(json.dumps(res,indent=2)+'\n')
