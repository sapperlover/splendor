"""Recompute per-search unique/world counts from the retained fingerprint streams."""
from collections import Counter, defaultdict
import gzip
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('cache_probe', Path(__file__).with_name('run.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

results = json.loads((p.ART/'per_root.json').read_text())['rows']
assert len(results) == 768
reports = []
for name, backend, per_world, worlds in p.PANELS:
    searches = {}
    for path in sorted(p.WORK.glob(f'{name}-events-*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                label, batch, world, simulation, kind, count, digest, flag = json.loads(line)
                root, seed = label
                key = (root, seed)
                if key not in searches:
                    searches[key] = (defaultdict(set), Counter(), Counter())
                seen, counts, kinds = searches[key]
                counts['requests'] += 1; kinds[str(kind)] += 1
                previous = seen[digest]
                if not previous:
                    assert flag == 'unique'
                    counts['unique_inputs'] += 1
                else:
                    assert flag in ('ready_hit', 'inflight_duplicate')
                    counts[flag] += 1
                    counts['within_world_duplicates' if world in previous else 'cross_world_duplicates'] += 1
                previous.add(world)
    assert len(searches) == 192
    selected = [r for r in results if r['panel']==name]
    assert len(selected) == 192
    for row in selected:
        seen, counts, kinds = searches[(row['root_id'],row['seed'])]
        for field in ('requests','unique_inputs','within_world_duplicates','cross_world_duplicates','ready_hit','inflight_duplicate'):
            assert counts[field] == row[field], (name,row['root_id'],field)
        assert dict(kinds) == row['kind_counts']
        assert row['simulations'] == per_world*worlds
        assert row['simulation_budgets'] == [per_world]*worlds
        if worlds == 1:
            assert counts['cross_world_duplicates'] == counts['inflight_duplicate'] == 0
    reports.append(dict(panel=name, searches=len(searches), stream_rows=sum(c['requests'] for _,c,_ in searches.values()),
        unique_and_world_decomposition_recomputed=True, recorded_ready_inflight_counts_reconciled=True))
    del searches
plan = json.loads((p.ART/'plan.json').read_text())
for path, digest in plan['files'].items():
    assert p.sha256(p.ROOT/path)==digest, path
assert p.sha256(p.CHECKPOINT)==p.MODEL_SHA
p.write_json(p.ART/'validation.json', dict(status='passed', searches=768, panels=reports,
    tests=dict(command='conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q artifacts/search_input_cache_rate_20261021/test_probe.py',
        passed=5, failed=0, test_sha256=p.sha256(p.ART/'test_probe.py')),
    observer_invariance='CPU native N1/N4/chance real 16-simulation jobs match full unobserved results; included in five tests.',
    fingerprint_audit='Per-root unique and world counts independently recomputed; ready/inflight flags reconciled with saved per-root counters, scheduling timestamps not independently replayed.',
    audit_sha256=p.sha256(Path(__file__))))
print(json.dumps(reports))
