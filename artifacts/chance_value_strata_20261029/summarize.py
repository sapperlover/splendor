"""Summarize outcome spread, independent-checkpoint ordering and sampling variance."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.alphazero_recording import read_json,sha256,write_json

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name


def ranks(x):
    _,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts)
    return ((ends-counts+ends-1)/2)[inverse]


def distribution(values):
    x=np.asarray([v for v in values if v is not None],dtype=float)
    return None if not len(x) else dict(n=len(x),mean=float(x.mean()),p10=float(np.quantile(x,.1)),
          p25=float(np.quantile(x,.25)),median=float(np.median(x)),p75=float(np.quantile(x,.75)),p90=float(np.quantile(x,.9)))


def metrics(row):
    x=np.asarray(row['value9'],dtype=float);y=np.asarray(row['value12'],dtype=float)
    p=(np.clip(x,-1,1)+1)*50;q=(np.clip(y,-1,1)+1)*50
    groups=np.array_split(np.argsort(x,kind='stable'),4)
    means=np.asarray([p[g].mean() for g in groups]);weights=np.asarray([len(g)/len(x) for g in groups])
    iid=float(np.var(p)/4);strat=float(sum(w*w*np.var(p[g]) for w,g in zip(weights,groups)))
    iid12=float(np.var(q)/4);strat12=float(sum(w*w*np.var(q[g]) for w,g in zip(weights,groups)))
    rx=ranks(x);ry=ranks(y)
    rho=float(np.corrcoef(rx,ry)[0,1]) if np.std(rx)>0 and np.std(ry)>0 else None
    k=(len(x)+3)//4
    overlap=len(set(np.argsort(x,kind='stable')[-k:])&set(np.argsort(y,kind='stable')[-k:]))/k
    return {**{key:row[key] for key in ('id','game_id','stage','mode','tier','action_kind')},
            'outcomes':len(x),'range_pp':float(np.ptp(p)),
            'p90_p10_pp':float(np.quantile(p,.9)-np.quantile(p,.1)),
            'top_bottom_quartile_gap_pp':float(means[-1]-means[0]),
            'minimum_adjacent_quartile_gap_pp':float(np.diff(means).min()),
            'quartile_means_pp':means.tolist(),'stratum_probability':weights.tolist(),
            'rank_correlation_9_12':rho,'top_quartile_overlap_9_12':overlap,
            'iid4_variance_9':iid,'stratified4_variance_9':strat,
            'without_replacement4_variance_9':iid*(len(x)-4)/(len(x)-1),
            'variance_ratio_9':strat/iid if iid>1e-12 else None,
            'variance_ratio_12_using9strata':strat12/iid12 if iid12>1e-12 else None,
            'clipped_outcomes_fraction':float(np.mean(np.abs(x)>1))}


def summary(rows):
    names=('range_pp','p90_p10_pp','top_bottom_quartile_gap_pp','minimum_adjacent_quartile_gap_pp',
           'rank_correlation_9_12','top_quartile_overlap_9_12','variance_ratio_9',
           'variance_ratio_12_using9strata','outcomes','clipped_outcomes_fraction')
    return dict(cases=len(rows),unique_games=len({r['game_id'] for r in rows}),
                metrics={key:distribution([r[key] for r in rows]) for key in names},
                fractions=dict(range_ge1pp=float(np.mean([r['range_pp']>=1 for r in rows])),
                    range_ge5pp=float(np.mean([r['range_pp']>=5 for r in rows])),
                    range_ge10pp=float(np.mean([r['range_pp']>=10 for r in rows])),
                    range_ge20pp=float(np.mean([r['range_pp']>=20 for r in rows])),
                    p90_p10_ge5pp=float(np.mean([r['p90_p10_pp']>=5 for r in rows])),
                    quartile_gap_ge5pp=float(np.mean([r['top_bottom_quartile_gap_pp']>=5 for r in rows])),
                    four_layers_adjacent_gap_ge2pp=float(np.mean([r['minimum_adjacent_quartile_gap_pp']>=2 for r in rows])),
                    ranking_rho_ge05=float(np.mean([r['rank_correlation_9_12'] is not None and r['rank_correlation_9_12']>=.5 for r in rows])),
                    stratification12_variance_lower=float(np.mean([r['variance_ratio_12_using9strata'] is not None and r['variance_ratio_12_using9strata']<1 for r in rows]))))


def main():
    execution=read_json(ART/'execution.json')
    assert sha256(RAW/'outcomes.json')==execution['outcomes_sha256']
    rows=[metrics(row) for row in read_json(RAW/'outcomes.json')]
    assert len(rows)==1200
    grouped=defaultdict(list);tier_groups=defaultdict(list);kind_groups=defaultdict(list)
    for row in rows:
        grouped[f"{row['mode']}_{row['stage']}"].append(row)
        tier_groups[f"{row['mode']}_{row['stage']}_t{row['tier']}"].append(row)
        kind_groups[f"{row['action_kind']}_{row['stage']}"].append(row)
    result=dict(scope='Within-position outcome differences, not calibrated win probabilities or true ranking accuracy.',
                groups={k:summary(v) for k,v in grouped.items()},
                by_tier={k:summary(v) for k,v in tier_groups.items()},
                by_action={k:summary(v) for k,v in kind_groups.items()},
                script_sha256=sha256(Path(__file__)),execution=execution)
    write_json(ART/'summary.json',result)
    write_json(RAW/'case_metrics.json',rows)
    for name,s in sorted(result['groups'].items()):
        m=s['metrics'];f=s['fractions']
        print(name,'n',s['cases'],'range',m['range_pp']['median'],'p90p10',m['p90_p10_pp']['median'],
              'range>=5',f['range_ge5pp'],'4layers',f['four_layers_adjacent_gap_ge2pp'],
              'rho',m['rank_correlation_9_12']['median'],'ratio12',m['variance_ratio_12_using9strata']['median'],flush=True)


if __name__=='__main__':main()
