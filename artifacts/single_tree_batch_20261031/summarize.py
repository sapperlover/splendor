"""Audit saved single-root benchmarks and keep only compact results in Git."""
from pathlib import Path
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET

ROOT=Path('runtime/single_tree_batch_20261031')
OUT=Path(__file__).resolve().parent


def main():
    modes={}
    reference=None
    for batch in (1,32,64,128):
        path=ROOT/f'deep_cuda_graph_b{batch}.json'
        data=json.loads(path.read_text());rows=[r for r in data['results'] if r['repeat']==1]
        assert len(rows)==3
        cases={}
        for row in rows:
            repeated=[r['result'] for r in data['results'] if r['case']==row['case']]
            assert len(repeated)==2 and repeated[0]==repeated[1]
            result=row['result'];diag=row['chance'];stats=row['stats']
            assert result['simulations']==sum(result['root_visit_counts'])==65536
            assert diag['root_hidden_samples']==65536
            assert diag['pending_simulations']==diag['virtual_visits']==0
            assert stats['model_positions']+diag['leaf_collisions']+stats['terminal_hits']==65537
            assert row['inference']['compile']['eager_fallbacks']==0
            assert row['inference']['compile']['compile_failure_fallbacks']==0
            cases[row['case']]=dict(seconds=row['seconds'],
                choice=result['chosen_candidate_index'],share=max(result['policy_target']),
                collisions=diag['leaf_collisions'],collision_fraction=diag['leaf_collisions']/65536,
                mean_depth=stats['tree_depth_sum']/65536,
                mean_inference_batch=row['inference_delta']['logical_model_positions']/
                                     row['inference_delta']['physical_model_batches'])
        if reference is None:reference=cases
        modes[str(batch)]=dict(cases=cases,
            median_seconds=statistics.median(c['seconds'] for c in cases.values()),
            matched_serial_choices=sum(c['choice']==reference[k]['choice'] for k,c in cases.items()),
            median_speedup_over_same_gpu_serial=statistics.median(
                reference[k]['seconds']/c['seconds'] for k,c in cases.items()),
            cuda_peak_allocated_bytes=data['cuda_peak_allocated_bytes'],
            raw_path=str(path),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    suite=ET.parse(ROOT/'final_tests.xml').getroot()
    tests={k:sum(int(s.attrib[k]) for s in suite.iter('testsuite'))
           for k in ('tests','failures','errors','skipped')}
    assert tests['failures']==tests['errors']==tests['skipped']==0
    cli=json.loads((ROOT/'cli_smoke_final/latest.json').read_text())
    decisions=[cli]+([cli['followup']] if 'followup' in cli else [])
    for row in decisions:
        assert row['simulations']==65536
        assert row['search_diagnostics']['chance']['root_hidden_samples']==65536
        assert row['search_diagnostics']['chance']['virtual_visits']==0
        assert row['selection_rule']=='visits' and row['submitted_action'] is False
    assert decisions[0]['search_diagnostics']['inference_cumulative']['compile']['dynamo_delta']['unique_graphs']==\
           decisions[-1]['search_diagnostics']['inference_cumulative']['compile']['dynamo_delta']['unique_graphs']
    numerical=json.loads((ROOT/'inference_verification.json').read_text())
    assert numerical['passed'] and numerical['max_abs_logit_error']<1e-5 and numerical['max_abs_value_error']<1e-5
    output=dict(status='complete',scope='Engineering latency and accounting, no formal strength match',
        budget_per_searchable_decision=65536,seed=0,root_noise=False,cpw=False,
        checkpoint_sha256=cli['checkpoint_sha256'],recommended_leaf_batch_size=64,
        numerical_verification={k:numerical[k] for k in ('rows','max_abs_logit_error','max_abs_value_error')},
        gpu='RTX 4050 Laptop',dtype='FP32; TF32 disabled',modes=modes,tests=tests,
        cli=dict(path=str(ROOT/'cli_smoke_final/latest.json'),decision_count=len(decisions),
                 total_simulations=sum(d['simulations'] for d in decisions),
                 total_elapsed_seconds=cli['total_elapsed_seconds'],
                 phase_seconds=[d['elapsed_seconds'] for d in decisions]),
        source_identity=str(ROOT/'final_source_identity.json'),
        delivery_source_identity=str(ROOT/'delivery_identity.json'),
        raw_storage='local runtime, not archived to T7')
    (OUT/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(tests=tests,median_seconds={k:v['median_seconds'] for k,v in modes.items()},
                         cli=output['cli']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
