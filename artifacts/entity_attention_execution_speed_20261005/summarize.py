"""Keep measured scopes and invalid windows explicit."""
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

ROOT=Path('artifacts/entity_attention_execution_speed_20261005')
variants={
 'baseline':[1,2,3], 'static':[1,2], 'double':[1],
 'baseline_shared2':[1,2,3], 'double_shared2':[1,2], 'combined_shared2':[1,2],
 'static500_shared2':[2], 'direct':[1,3], 'direct_double':[1,3],
}
summary={'scope':'15 s opening windows, 3 s warmup; not whole-game/whole-round speed',
         'checkpoint':'runs/entity_attention_offline/checkpoints/final.pt',
         'variants':{},'primary_repeat3':{},'gpu_checks':{}}
for name,repeats in variants.items():
 rows=[]
 for repeat in repeats:
  path=ROOT/f'{name}_r{repeat}.json';x=json.loads(path.read_text());o=x['gpu_owner'];w=o['benchmark']['owner_window'];first,last=w['start'],w['end']
  row={'repeat':repeat,'raw':str(path),'rates':x['clock']['rates'],'owner_timings':x['clock']['owner_timings'],
       'compile_warmup_seconds':o['benchmark']['compile_warmup_seconds'],'process_seconds':x['process_seconds'],
       'mean_batch_size':x['clock']['rates']['mean_batch_size'],'gpu_peak_bytes':o['gpu_peak_bytes'],
       'batches_prepared_while_inflight':last.get('batches_prepared_while_inflight',0)-first.get('batches_prepared_while_inflight',0),
       'gather_while_inflight_seconds':last.get('overlapped_gather_seconds',0)-first.get('overlapped_gather_seconds',0),
       'batch_padding_positions':last.get('batch_padding_positions',0)-first.get('batch_padding_positions',0),
       'disabled_graphs':o['inference']['compile']['disabled_graphs']}
  rows.append(row)
  if repeat==3: summary['primary_repeat3'][name]=row
 summary['variants'][name]={'windows':rows,'mean_simulations_per_second':statistics.mean(r['rates']['simulations_per_second'] for r in rows)}
for name in ('baseline','static','direct'):
 x=json.loads((ROOT/f'gpu_{name}.json').read_text());events=json.loads(Path(x['trace']).read_text())['traceEvents']
 memcpy=Counter(e['name'] for e in events if e.get('name','').startswith('Memcpy '))
 summary['gpu_checks'][name]={'numeric_checks':x['numeric_checks'],'value_only':x['value_only'],
    'runtime_counts_for_10_batches':x['cuda_runtime_counts_for_10_batches'],'memcpy_events_for_10_batches':dict(memcpy)}
summary['micro_timing_note']='Diagnostic only; final direct numeric/profiler check ran alongside CPU regressions. Throughput conclusions use isolated run_windows outputs.'
for name in ('smoke','smoke_direct'):
 x=json.loads((ROOT/f'gpu_complete_{name}.json').read_text())
 summary[name]={'completed_games':x['checked_replay']['completed_games'],'failed_games':x['checked_replay']['failed_games'],
                'graph_replays':x['gpu_owner']['inference'].get('native_graph_replays',0),
                'prepared_while_inflight':x['gpu_owner']['batches_prepared_while_inflight']}
summary['cpu_tests']=ET.parse(ROOT/'cpu_tests.xml').getroot().find('testsuite').attrib
base=summary['primary_repeat3']['baseline']['rates']['simulations_per_second'];control=summary['primary_repeat3']['baseline_shared2']['rates']['simulations_per_second']
for name,row in summary['primary_repeat3'].items():
 row['relative_to_original']=row['rates']['simulations_per_second']/base
 row['relative_to_shared2_control']=row['rates']['simulations_per_second']/control
summary['invalid_window']=json.loads((ROOT/'invalid_window.json').read_text())
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({name: {k:r[k] for k in ('rates','relative_to_original','relative_to_shared2_control')} for name,r in summary['primary_repeat3'].items()},indent=2))
