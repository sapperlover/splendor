"""Summarize collection windows without mixing cumulative and window counters.

For native_cuda_pool, host forward submission time, gather, dispatch, H2D and
wait deltas are read from ``clock.owner_timings`` and logical positions from
``clock.totals``.  ``gpu_owner.inference`` is cumulative over the whole process
and is intentionally not used as a window value.
"""
import glob, json, os
from pathlib import Path
rows=[]
for p in sorted(glob.glob('runtime/collection_tuning_20261022/results/*.json')):
    if p.endswith('summary.json'): continue
    try: d=json.load(open(p))
    except Exception: continue
    c=d.get('clock',{}); rates=c.get('rates',{}); ex=d.get('execution',{}); totals=c.get('totals',{})
    if not c: continue
    name=Path(p).stem; backend=d.get('backend'); window=float(c.get('common_seconds') or 0)
    proc=float(d.get('process_seconds') or 0); sims=float(rates.get('simulations_per_second') or 0)
    meanb=float(rates.get('mean_batch_size') or 0); pos=int(totals.get('network_positions') or 0)
    go=d.get('gpu_owner')
    if go:
        t=c.get('owner_timings',{})
        host_fwd=float(t.get('forward_seconds') or 0)
        gather=float(t.get('gather_seconds') or 0); dispatch=float(t.get('dispatch_seconds') or 0)
        wait=float(t.get('wait_seconds') or 0); h2d=float(t.get('h2d_seconds') or 0)
        cumulative_fwd=float(go.get('inference',{}).get('forward_seconds') or 0)
        forward_kind='host_forward_window'
    else:
        host_fwd=float(totals.get('forward_seconds') or 0); gather=dispatch=wait=h2d=0.0
        cumulative_fwd=None; forward_kind='cpu_forward_window'
    rows.append(dict(name=name,backend=backend,checkpoint=os.path.basename(str(d.get('checkpoint',''))),
        workers=ex.get('workers'),search_workers=ex.get('search_workers'),
        active=ex.get('total_active_games') if backend=='native_cuda_pool' else (ex.get('active_games_per_worker',0)*ex.get('workers',0)),
        batch=ex.get('max_batch_size'),pipeline=ex.get('pipeline_slots'),micro=ex.get('local_microbatch'),
        wait_ms=ex.get('max_batch_wait_ms'),shared_width=ex.get('shared_candidate_width'),shared_slots=ex.get('shared_slots'),
        sims_s=sims,mean_batch=meanb,process_s=proc,window_s=window,network_positions=pos,
        forward_kind=forward_kind,host_forward_s=host_fwd,
        host_forward_us_per_pos=(1e6*host_fwd/pos if pos else None),
        host_forward_fraction=(host_fwd/window if window and go else None),
        cumulative_forward_s=cumulative_fwd,
        gather_s=gather,dispatch_s=dispatch,wait_s=wait,h2d_s=h2d,
        worker_cpu_s=float(c.get('cpu_seconds') or 0)))
rows.sort(key=lambda r:(r['checkpoint'],r['name']))
Path('runtime/collection_tuning_20261022/results/summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print('| run | backend | w/a | B | mb | sim/s | meanB | host fwd ms/pos | host fwd % | gather s | wait s |')
print('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    w=f"{r['search_workers'] or r['workers']}/{r['active']}"
    us=r['host_forward_us_per_pos']; frac=r['host_forward_fraction']
    fs = 'n/a' if frac is None else f'{frac*100:.2f}%'
    print(f"| {r['name']} | {r['backend']} | {w} | {r['batch']} | {r['micro']} | {r['sims_s']:.1f} | {r['mean_batch']:.1f} | {us:.3f} | {fs} | {r['gather_s']:.2f} | {r['wait_s']:.2f} |")
