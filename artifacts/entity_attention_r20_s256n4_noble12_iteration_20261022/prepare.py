"""Freeze an R20 continuation; does not collect, train, or delete old data."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.catalog import load_default_catalog
from splendor.random_budget_iteration import Config, check

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
NAME = 'entity_attention_r20_s256n4_noble12_iteration_20261022'
OUTPUT = ROOT / 'runs' / NAME
RUNTIME = ROOT / 'runtime' / NAME
PARENT = ROOT / 'runs/entity_attention_37423_random_budget_p6k_v24k_20261019'
BASE = ROOT / 'configs/entity_attention_v1/random_budget_37423_p6k_v24k_20261019.json'
CONFIG = ROOT / 'configs/entity_attention_v1/random_budget_r20_s256n4_noble12_20261022.json'


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen branch input changed: {path}')
    else:
        write_json(path, value)


def main():
    source_state = PARENT / 'state.json'
    state = read_json(source_state)
    if len(state['completed_rounds']) != 20:
        raise ValueError('Expected exactly 20 completed parent rounds')
    checkpoint = state['latest_checkpoint']
    if checkpoint['sha256'] != 'fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd':
        raise ValueError('Expected raw R20 checkpoint')
    parent_copy = OUTPUT / 'inputs/parent_state.json'
    parent_copy.parent.mkdir(parents=True, exist_ok=True)
    if not parent_copy.exists():
        shutil.copyfile(source_state, parent_copy)
    if sha256(source_state) != sha256(parent_copy):
        raise ValueError('Frozen parent state differs')
    raw = copy.deepcopy(read_json(BASE))
    raw.update(run_id=NAME, initial_checkpoint=checkpoint,
        continuation_state={'path': str(parent_copy), 'sha256': sha256(parent_copy)})
    raw['selfplay']['search_config']['determinizations'] = 4
    raw['selfplay']['budget_randomization']['low_determinizations'] = 1
    # Internal simulations is the total: S256N4 = 1024; low S128N1 = 128.
    assert raw['selfplay']['search_config']['simulations'] == 1024
    for key in ('history_manifest', 'evaluation_anchor'):
        raw[key]['path'] = str((ROOT / raw[key]['path']).resolve())
    for key in ('greedy_evaluation', 'search_evaluation'):
        raw[key]['policy_b']['checkpoint'] = raw['evaluation_anchor']['path']
    config = Config.from_dict(raw)
    _, preflight = check(config)
    freeze(CONFIG, raw)
    catalog = load_default_catalog()
    if len(catalog.nobles) != 12:
        raise ValueError('Expected the 12-noble catalog')
    collection = read_json(PARENT / 'rounds/round-019/collection_config.json')['config']
    if collection['execution'] != raw['selfplay']['execution']:
        raise ValueError('Execution differs from the actual R20 collection')
    plan = {'experiment': NAME, 'config': {'path': str(CONFIG.relative_to(ROOT)), 'sha256': sha256(CONFIG)},
        'parent_state': {'path': str(source_state.relative_to(ROOT)), 'sha256': sha256(source_state)},
        'parent_checkpoint': checkpoint, 'parent_replay': state['completed_rounds'][-1]['snapshot'],
        'preflight': preflight, 'catalog_fingerprint': catalog.data_fingerprint,
        'catalog_files': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'data').glob('*')) if p.is_file()},
        'high_search': {'fraction': .25, 'simulations_per_world': 256, 'worlds': 4, 'total_simulations': 1024},
        'low_search': {'fraction': .75, 'simulations_per_world': 128, 'worlds': 1, 'total_simulations': 128},
        'changes': ['high-budget worlds1->4, total1024 unchanged', 'low-budget worlds explicitly1',
                    'new collection and evaluation use frozen12-noble catalog'],
        'historical_data': 'Inherited R15-R20 pool:24000 games from old10-noble catalog; retained without relabeling. FIFO eviction4000 per new round.',
        'limit': 'continuous; no round or update-session limit',
        'runner': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha256(Path(__file__))}}
    freeze(ART/'plan.json', plan)
    destination = RUNTIME/'source'
    record_path = ART/'source.json'
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        for path in [CONFIG, Path(__file__), ROOT/'src/splendor/native_search.cpp',
                     *ROOT.glob('src/splendor/_native_search*.so'), *sorted((ROOT/'data').glob('*'))]:
            if path.is_file():
                target = destination/path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        record['snapshot_files'] = {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob('*')) if p.is_file()}
        freeze(record_path, record)
    for name, expected in read_json(record_path)['snapshot_files'].items():
        if sha256(destination/name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    print({'status':'prepared', **preflight, 'catalog_fingerprint':catalog.data_fingerprint}, flush=True)


if __name__ == '__main__':
    main()
