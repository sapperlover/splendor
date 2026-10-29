"""Information-safe, resumable analysis of the 88 exported BGA decisions."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import fcntl
import gc
import html
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.catalog import load_catalog
from splendor.compact_trace import decode_request
from splendor.features import encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import ChanceWideningConfig, NativeBatchRunner, NativeChancePUCTSearchPolicy, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.search_policy import _request_seed, determinize_request

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
DATA = ROOT / 'artifacts/bga_927500015_20261020'
RUNTIME = ROOT / 'runtime' / ART.name
CHECKPOINT = ROOT / 'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt'
SEEDS = [39700000, 39701000, 39702000]
COLORS = {'white': '白', 'blue': '蓝', 'green': '绿', 'red': '红', 'black': '黑', 'gold': '金'}
PLAYERS = ['VnEpCxmR', 'FourDimensional']
CHUNK = 16


def load_default_catalog():
    # Reuse the original game's frozen card/rule catalog for paired comparison.
    return load_catalog(DATA / 'catalog')


def panels():
    # MCTSConfig.simulations is TOTAL across worlds, not per-world budget.
    return [dict(name=f'{method}_S{budget}_N{worlds}', method=method, simulations=budget, determinizations=worlds)
            for budget in (1024, 16384)
            for method, worlds in [('native', 1), ('native', 4), ('chance', 1), ('cpw', 1)]]


def records():
    return [json.loads(line) for line in (DATA / 'model_decisions.jsonl').read_text().splitlines()]


def choice(values, prior=None):
    # Match native T=0: visit ties use network prior, then candidate order.
    return max(range(len(values)), key=lambda i: (values[i], prior[i] if prior is not None else 0., -i))


def fmt_action(action, catalog=None):
    kind = action['__type__']
    if kind == 'TakeDistinct':
        return '拿' + ''.join(COLORS[c] for c in action['colors'])
    if kind == 'TakePair':
        return '拿2' + COLORS[action['color']]
    if kind == 'ReserveDeck':
        return f"盲保留{action['tier']}级"
    if kind in ('BuyMarket', 'BuyReserved', 'ReserveMarket'):
        card = (catalog or load_default_catalog()).card(action['card_id'])
        verb = {'BuyMarket': '买市场', 'BuyReserved': '买保留', 'ReserveMarket': '保留'}[kind]
        return f'{verb}{card.card_id}({int(card.tier)}级{COLORS[card.bonus.value]}+/{card.points}分)'
    if kind in ('Pay', 'ReturnTokens'):
        tokens = action['tokens']
        return ('支付' if kind == 'Pay' else '退还') + (''.join(f'{COLORS[c]}{v}' for c, v in tokens.items() if c != '__type__' and v) or '0')
    if kind == 'ClaimNoble':
        return '贵族' + action['noble_id']
    raise ValueError(kind)


def identity_check(plan):
    for relative, digest in plan['inputs'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f'Frozen input changed: {relative}')
    for relative, digest in plan['source']['source_files'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f'Frozen source changed: {relative}')
    for absolute, digest in plan['source']['native_extensions'].items():
        if sha256(Path(absolute)) != digest:
            raise ValueError(f'Native extension changed: {absolute}')


def prepare():
    if (ART / 'plan.json').exists():
        identity_check(read_json(ART / 'plan.json'))
        return
    RUNTIME.mkdir(parents=True, exist_ok=True)
    shas = read_json(DATA / 'SHA256SUMS.json')
    for name, digest in shas.items():
        if sha256(DATA / name) != digest:
            raise ValueError(f'Capture SHA mismatch: {name}')
    catalog = load_default_catalog()
    if catalog.data_fingerprint != load_catalog(DATA / 'catalog').data_fingerprint:
        raise ValueError('Captured/current catalog differs')
    rows = records()
    if len(rows) != 88 or sum(r['request']['observation']['phase'] == 'main_action' for r in rows) != 56:
        raise ValueError('Unexpected captured decision count')
    checks = []
    with np.load(DATA / 'model_features.npz', allow_pickle=False) as frozen:
        for i, row in enumerate(rows):
            request = decode_request(row['request'])
            assert row['decision_index'] == i
            assert to_primitive(request) == row['request']
            assert to_primitive(request.legal_actions[row['candidate_index']]) == row['action']
            assert request.observation.self_is_first_player == (row['actor'] == 0)
            encoded = encode_decision(request)
            fields = [key.split('/', 1)[1] for key in frozen.files if key.startswith(f'{i:03d}/')]
            for field in fields:
                np.testing.assert_array_equal(getattr(encoded, field), frozen[f'{i:03d}/{field}'])
            # Only the safe request enters the simulator. Never load reconstruction.json.
            # Engine.decision automatically resolves compulsory noble claims;
            # those exported one-candidate requests need no search roundtrip.
            roundtrip_seeds = [] if row['automatic'] else SEEDS[:2]
            for seed in roundtrip_seeds:
                engine = determinize_request(request, seed=_request_seed(request, seed, 0), engine_mode='fast', catalog=catalog)
                assert engine.current_player == 0  # simulator normalizes root actor to 0
                assert engine.decision() == request
            checks.append({'decision_index': i, 'feature_fields': len(fields), 'safe_world_roundtrips': len(roundtrip_seeds)})
    source = source_identity(ROOT, RUNTIME / 'source')
    paths = [CHECKPOINT, Path(__file__), ROOT / 'src/splendor/native_search.cpp', DATA / 'model_decisions.jsonl',
             DATA / 'model_features.npz', DATA / 'SHA256SUMS.json', DATA / 'table_metadata.json', ART / 'summarize.py']
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device='cpu')
    inference = FrozenSearchInference(model, device='cpu', config=InferenceConfig(actor_projection=False))
    outputs = inference.policy_value([decode_request(r['request']) for r in rows])
    greedy = []
    for row, (logits, value) in zip(rows, outputs, strict=True):
        logits = np.asarray(logits, dtype=np.float64)
        prior = np.exp(logits - logits.max()); prior /= prior.sum()
        greedy.append({'decision_index': row['decision_index'], 'prior': prior.tolist(), 'logits': logits.tolist(),
                       'choice': choice(prior), 'value_actor_raw': value,
                       'value_first_player_raw': value * (1 if row['actor'] == 0 else -1)})
    plan = {'schema': 'bga-root-analysis-v1', 'table': 927500015, 'round': 12,
            'checkpoint': str(CHECKPOINT.relative_to(ROOT)), 'model_config': asdict(model.config),
            'parameters': sum(p.numel() for p in model.parameters()), 'source': source,
            'inputs': {str(p.relative_to(ROOT)): sha256(p) for p in paths},
            'catalog_fingerprint': catalog.data_fingerprint, 'seeds': SEEDS, 'panels': panels(),
            'search_common': dict(c_puct=1.5, fold_forced_phases=True, max_simulation_transitions=512,
                                  root_noise_fraction=0., dirichlet_alpha=.3, forced_playouts=0., policy_target_pruning=False),
            'cpw': {'k': .25, 'alpha': .5}, 'chunk_roots': CHUNK, 'capacity': 64, 'pipeline_slots': 1,
            'inference': InferenceConfig(mode='compiled', actor_projection=False, candidate_buckets=(16, 32, 48, 64),
                                         compile_mode='native-cudagraph', batch_buckets=(16, 64), compile_failure_policy='fail_fast').to_dict(),
            'captured_files_checked': len(shas), 'decisions': len(rows), 'searchable': sum(len(r['request']['legal_actions']) > 1 for r in rows),
            'scope': 'One observed game. Every non-forced decision; 3 common seeds per panel. Total S split equally over N. T=0, no root noise. No future reconstructed deck or final outcome enters inference/search. Q is model/search opinion, not verified human regret.',
            'command': sys.argv}
    write_json(ART / 'greedy.json', {'rows': greedy})
    write_json(ART / 'input_validation.json', {'catalog_fingerprint': catalog.data_fingerprint, 'capture_files_checked': len(shas), 'rows': checks})
    write_json(ART / 'plan.json', plan)
    report()
    print(json.dumps({'prepared': True, 'decisions': len(rows), 'searchable': plan['searchable']}), flush=True)


def make_policy(model, evaluator, panel, seed, plan, budget=None):
    config = MCTSConfig(**plan['search_common'], simulations=budget or panel['simulations'],
                        determinizations=panel['determinizations'], seed=seed)
    cls = NativePUCTSearchPolicy if panel['method'] == 'native' else NativeChancePUCTSearchPolicy
    kwargs = {'chance_widening': ChanceWideningConfig(**plan['cpw'])} if panel['method'] == 'cpw' else {}
    return cls(model, evaluator=evaluator, config=config, catalog=load_default_catalog(), **kwargs)


def search_row(job, row, prior, panel, seed, total):
    result = job.result
    visits = np.asarray(result.root_visit_counts, dtype=np.int64)
    world_visits = np.asarray(result.per_world_root_visit_counts, dtype=np.int64)
    world_sums = np.asarray(result.per_world_root_value_sums, dtype=np.float64)
    # chance_diagnostics.root_value_sums covers ONLY world 0; never use it for N>1.
    sums = world_sums.sum(axis=0)
    assert int(visits.sum()) == total == sum(result.simulation_budgets)
    np.testing.assert_array_equal(visits, world_visits.sum(axis=0))
    np.testing.assert_array_equal(world_visits.sum(axis=1), result.simulation_budgets)
    assert len(result.simulation_budgets) == panel['determinizations']
    assert tuple(result.root_network_prior) == tuple(result.root_used_prior)
    np.testing.assert_allclose(result.root_network_prior, prior, atol=2e-5, rtol=2e-5)
    assert np.all(np.isfinite(sums)) and np.all(np.abs(sums) <= visits + 1e-6)
    assert choice(visits, result.root_network_prior) == result.chosen_candidate_index
    diagnostics = job.core.chance_diagnostics()
    diagnostics.pop('root_value_sums', None)
    if panel['method'] == 'native':
        # Completed multiworld failure_context returns the last active world,
        # not a historical per-world archive. Derive the seeds via the contract.
        seeds = [_request_seed(job.request, seed, i) for i in range(panel['determinizations'])]
        if panel['determinizations'] == 1:
            assert job.failure_context_for_world(0)['determinization_seed'] == seeds[0]
    else:
        seeds = None  # chance mode resamples hidden cards per simulation; base seed retained below
    return {'decision_index': row['decision_index'], 'seed': seed, 'choice': result.chosen_candidate_index,
            'visits': visits.tolist(), 'value_sums': sums.tolist(),
            'q': [float(s / n) if n else None for s, n in zip(sums, visits, strict=True)],
            'per_world_visits': world_visits.tolist(), 'per_world_value_sums': world_sums.tolist(),
            'simulation_budgets': list(result.simulation_budgets), 'world_seeds': seeds,
            'prior': list(result.root_network_prior), 'max_tree_depth': result.max_tree_depth,
            'transition_cap_hits': result.transition_cap_hits, 'diagnostics': diagnostics}


def smoke(device):
    prepare()
    plan = read_json(ART / 'plan.json'); identity_check(plan)
    rows = records(); greedy = read_json(ART / 'greedy.json')['rows']
    torch.set_num_threads(1)
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device=device)
    inf = FrozenSearchInference(model, device=device, config=InferenceConfig(actor_projection=False))
    # Both original seats, a payment choice, an unknown opponent blind reserve,
    # and a late root. The real blind card is never supplied to the opponent.
    ids = [0, 1, next(r['decision_index'] for r in rows if r['request']['observation']['phase'] == 'payment' and len(r['request']['legal_actions']) > 1),
           next(r['decision_index'] for r in rows if r['bga_move'] == 14 and r['request']['observation']['phase'] == 'main_action'), 60]
    runner = NativeBatchRunner(inf, load_default_catalog(), capacity=16, buckets=(16, 32, 48, 64))
    checked = []
    for panel in panels()[:4]:
        policy = make_policy(model, inf, panel, SEEDS[0], plan, budget=16)
        jobs = [policy.start_job(decode_request(rows[i]['request']), add_root_noise=False, temperature=0.) for i in ids]
        while any(not j.done for j in jobs):
            assert runner.run([j for j in jobs if not j.done]) > 0
        for i, job in zip(ids, jobs, strict=True):
            result = search_row(job, rows[i], greedy[i]['prior'], panel, SEEDS[0], 16)
            checked.append({'panel': panel['name'], 'decision_index': i, 'total_visits': sum(result['visits']),
                            'world_budgets': result['simulation_budgets'], 'chance': result['diagnostics']['stochastic']})
    write_json(ART / f'smoke_{device}.json', {'scope': 'Engineering validation only, not game quality', 'rows': checked})
    print(json.dumps({'smoke': device, 'roots': len(checked)}), flush=True)


def run():
    prepare()
    plan = read_json(ART / 'plan.json'); identity_check(plan)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for full analysis')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rows = records(); greedy = read_json(ART / 'greedy.json')['rows']
    searchable = [r for r in rows if len(r['request']['legal_actions']) > 1]
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device='cuda')
    cfg = dict(plan['inference'])
    for key in ('candidate_buckets', 'batch_buckets'):
        cfg[key] = tuple(cfg[key])
    evaluator = FrozenSearchInference(model, device='cuda', config=InferenceConfig(**cfg))
    runner = NativeBatchRunner(evaluator, load_default_catalog(), capacity=plan['capacity'], buckets=(16, 32, 48, 64), slots=1)
    write_json(ART / 'execution.json', {'environment': environment_identity(), 'command': sys.argv, 'started_unix': time.time()})
    started = time.perf_counter()
    for panel in plan['panels']:
        for seed in plan['seeds']:
            path = ART / 'search' / panel['name'] / f'{seed}.json'
            saved = read_json(path) if path.exists() else {'panel': panel, 'seed': seed, 'rows': [], 'chunks': []}
            done = {r['decision_index'] for r in saved['rows']}
            pending = [r for r in searchable if r['decision_index'] not in done]
            if not pending:
                continue
            policy = make_policy(model, evaluator, panel, seed, plan)
            for start in range(0, len(pending), CHUNK):
                group = pending[start:start + CHUNK]
                tick = time.perf_counter()
                jobs = [policy.start_job(decode_request(r['request']), add_root_noise=False, temperature=0.) for r in group]
                while any(not j.done for j in jobs):
                    if runner.run([j for j in jobs if not j.done]) <= 0:
                        raise RuntimeError('Search stalled')
                for row, job in zip(group, jobs, strict=True):
                    saved['rows'].append(search_row(job, row, greedy[row['decision_index']]['prior'], panel, seed, panel['simulations']))
                saved['chunks'].append({'decisions': [r['decision_index'] for r in group], 'seconds': time.perf_counter() - tick})
                write_json(path, saved)
                status = {'stage': 'search', 'panel': panel['name'], 'seed': seed, 'done_in_seed': len(saved['rows']),
                          'total_in_seed': len(searchable), 'elapsed_this_session': time.perf_counter() - started,
                          'updated_unix': time.time(), 'inference': evaluator.stats.to_dict()}
                write_json(ART / 'live.json', status)
                print(json.dumps({k: v for k, v in status.items() if k != 'inference'}), flush=True)
                del jobs, job
                gc.collect()
            del policy
            report()
    identity_check(plan)
    summary = report()
    assert summary['complete']
    write_json(ART / 'live.json', {'stage': 'complete', 'updated_unix': time.time(), 'elapsed_this_session': time.perf_counter() - started,
                                 'inference': evaluator.stats.to_dict(), 'searches': summary['completed_searches']})
    print(json.dumps({'complete': True, 'searches': summary['completed_searches']}), flush=True)


def combine(rows):
    """Equal seed weight for visit shares and Q; do not manufacture missing Q."""
    visits = np.asarray([r['visits'] for r in rows], dtype=np.float64)
    shares = visits / visits.sum(axis=1, keepdims=True)
    mean_shares = shares.mean(axis=0)
    q, q_min, q_max, coverage = [], [], [], []
    for i in range(visits.shape[1]):
        values = [r['q'][i] for r in rows if r['q'][i] is not None]
        q.append(statistics.mean(values) if values else None)
        q_min.append(min(values) if values else None); q_max.append(max(values) if values else None)
        coverage.append(len(values))
    best = choice(mean_shares, rows[0]['prior'])
    return {'choice': best, 'visit_share': mean_shares.tolist(), 'q': q, 'q_min': q_min, 'q_max': q_max,
            'q_seed_coverage': coverage, 'minimum_visits': visits.min(axis=0).astype(int).tolist(),
            'seed_choices': [r['choice'] for r in rows], 'choice_support': sum(r['choice'] == best for r in rows),
            'seeds_completed': len(rows), 'max_depth': max(r['max_tree_depth'] for r in rows),
            'transition_cap_hits': sum(r['transition_cap_hits'] for r in rows),
            'raw': rows}


def js(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / a.sum(), b / b.sum(); m = (a + b) / 2
    def kl(p):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / m[mask])))
    return (kl(a) + kl(b)) / 2


def report():
    if not (ART / 'greedy.json').exists():
        raise ValueError('prepare first')
    plan = read_json(ART / 'plan.json')
    rows = records(); greedy = read_json(ART / 'greedy.json')['rows']
    estimates = {}; raw = {}; completed = 0
    for panel in plan['panels']:
        by_id = {}
        for seed in plan['seeds']:
            path = ART / 'search' / panel['name'] / f'{seed}.json'
            if not path.exists():
                continue
            data = read_json(path)
            for row in data['rows']:
                by_id.setdefault(row['decision_index'], []).append(row)
                raw[panel['name'], seed, row['decision_index']] = row
                completed += 1
        estimates[panel['name']] = {i: combine(rs) for i, rs in by_id.items()}
    expected = plan['searchable'] * len(plan['seeds']) * len(plan['panels'])
    status_label = '全部完成' if completed == expected else '待执行（正式搜索尚未启动）' if completed == 0 else '部分完成'
    summary = {'complete': completed == expected, 'status': status_label, 'completed_searches': completed, 'expected_searches': expected,
               'main_decisions': 56, 'choice_payments': 20, 'forced_decisions': 12, 'panels': {}, 'comparisons': {}}
    main_ids = [r['decision_index'] for r in rows if r['request']['observation']['phase'] == 'main_action']
    panel_names = ['greedy'] + [p['name'] for p in plan['panels']]
    for name in panel_names:
        stats = {}
        for actor in (0, 1, None):
            ids = [i for i in main_ids if actor is None or rows[i]['actor'] == actor]
            if name == 'greedy':
                data = [(i, greedy[i]['choice']) for i in ids]
            else:
                data = [(i, estimates[name][i]['choice']) for i in ids if i in estimates[name] and estimates[name][i]['seeds_completed'] == 3]
            stats[str(actor)] = {'decisions': len(data), 'human_agree': sum(c == rows[i]['candidate_index'] for i, c in data),
                                 'greedy_agree': sum(c == greedy[i]['choice'] for i, c in data)}
        if name != 'greedy':
            es = [estimates[name][i] for i in main_ids if i in estimates[name] and estimates[name][i]['seeds_completed'] == 3]
            stats['seed_instability'] = {'decisions': len(es), 'any_top1_change': sum(len(set(e['seed_choices'])) > 1 for e in es)}
        summary['panels'][name] = stats
    comparisons = [(f'native_S1024_N1', 'native_S16384_N1')]
    comparisons += [('native_S16384_N1', f'{m}_S16384_N{n}') for m, n in [('native', 4), ('chance', 1), ('cpw', 1)]]
    for a, b in comparisons:
        paired = [(raw[a, s, i], raw[b, s, i]) for s in SEEDS for i in main_ids if (a, s, i) in raw and (b, s, i) in raw]
        summary['comparisons'][f'{a}__{b}'] = {'paired_root_seeds': len(paired),
            'top1_changes': sum(x['choice'] != y['choice'] for x, y in paired),
            'mean_js_nats': statistics.mean(js(x['visits'], y['visits']) for x, y in paired) if paired else None}
    write_json(ART / 'summary.json', summary)
    serial = {name: {str(i): {k: v for k, v in e.items() if k != 'raw'} for i, e in es.items()} for name, es in estimates.items()}
    write_json(ART / 'aggregates.json', serial)
    catalog = load_default_catalog()
    decision_csv, action_csv, blocks = [], [], []
    for i, row in enumerate(rows):
        actions = row['request']['legal_actions']; actual = row['candidate_index']; obs = row['request']['observation']
        labels = [fmt_action(a, catalog) for a in actions]
        order = sorted(range(len(actions)), key=lambda j: (-greedy[i]['prior'][j], j))
        actual_rank = order.index(actual) + 1
        heading = f"BGA {row['bga_move']} · {PLAYERS[row['actor']]} · {obs['phase']} · 决策{i}"
        lines = [f'### {heading}', '', f"实际：**{labels[actual]}**；Greedy：{labels[greedy[i]['choice']]}。实际走法policy概率{greedy[i]['prior'][actual]:.1%}，排名{actual_rank}/{len(actions)}；原始V（行动者视角）{greedy[i]['value_actor_raw']:+.3f}。", '',
                 '| 方法（总预算） | 模型首选 | 3种子首选支持 | 实际访问份额 | Q实际 | Q首选−Q实际 |', '| --- | --- | ---: | ---: | ---: | ---: |']
        detail_tables = []
        for name in panel_names:
            if name == 'greedy':
                best = greedy[i]['choice']; e = None
            else:
                e = estimates[name].get(i)
                if e is None:
                    continue
                best = e['choice']
            qs = e['q'] if e else [None] * len(actions)
            paired_gaps = ([r['q'][best] - r['q'][actual] for r in e['raw'] if r['q'][best] is not None and r['q'][actual] is not None] if e else [])
            gap = statistics.mean(paired_gaps) if paired_gaps else None
            support = e['choice_support'] if e else None
            stable = bool(e and len(e['raw']) == 3 and support == 3 and best != actual and len(paired_gaps) == 3
                          and min(paired_gaps) > 0 and e['minimum_visits'][actual] >= 16)
            entry = {'decision_index': i, 'bga_move': row['bga_move'], 'actor': row['actor'], 'player': PLAYERS[row['actor']],
                     'phase': obs['phase'], 'forced': len(actions) == 1, 'panel': name, 'actual': labels[actual], 'recommendation': labels[best],
                     'agree': best == actual, 'policy_actual_probability': greedy[i]['prior'][actual], 'policy_actual_rank': actual_rank,
                     'value_actor_raw': greedy[i]['value_actor_raw'], 'actual_visit_share': e['visit_share'][actual] if e else None,
                     'q_actual': qs[actual], 'q_recommended': qs[best], 'paired_seed_q_gap': gap,
                     'paired_seed_q_gap_min': min(paired_gaps) if paired_gaps else None,
                     'paired_seed_q_gap_max': max(paired_gaps) if paired_gaps else None,
                     'q_gap_seed_count': len(paired_gaps), 'actual_minimum_visits': e['minimum_visits'][actual] if e else None,
                     'choice_support': support, 'seeds_completed': e['seeds_completed'] if e else None, 'stable_disagreement': stable}
            decision_csv.append(entry)
            for j, label in enumerate(labels):
                action_csv.append({'decision_index': i, 'bga_move': row['bga_move'], 'actor': row['actor'], 'phase': obs['phase'],
                                   'panel': name, 'candidate_index': j, 'action': label, 'is_actual': j == actual,
                                   'is_recommended': j == best, 'prior': greedy[i]['prior'][j],
                                   'visit_share': e['visit_share'][j] if e else None, 'q': qs[j],
                                   'q_min': e['q_min'][j] if e else None, 'q_max': e['q_max'][j] if e else None,
                                   'q_seed_coverage': e['q_seed_coverage'][j] if e else None,
                                   'minimum_visits': e['minimum_visits'][j] if e else None})
            if e:
                def qstr(v):
                    return '未访问' if v is None else f'{v:+.3f}'
                lines.append(f"| {name} | {'✓ ' if best == actual else ''}{labels[best]} | {support}/{e['seeds_completed']} | {e['visit_share'][actual]:.1%} | {qstr(qs[actual])} | {qstr(gap)} |")
                top = sorted(range(len(actions)), key=lambda j: (-e['visit_share'][j], j))[:5]
                detail_tables.append(f'<h4>{html.escape(name)} 前5候选</h4><ul>' + ''.join(f'<li>{html.escape(labels[j])}：访问{e["visit_share"][j]:.1%}，Q {qstr(qs[j])}（种子范围{qstr(e["q_min"][j])}～{qstr(e["q_max"][j])}；覆盖{e["q_seed_coverage"][j]}/3）</li>' for j in top) + '</ul>')
        if len(actions) == 1:
            lines += ['', '强制决策：仅1个候选，不执行搜索，不计入选择一致率。']
        blocks.append({'actor': row['actor'], 'phase': obs['phase'], 'heading': heading, 'lines': lines,
                       'observation': obs, 'details': ''.join(detail_tables)})
    for name, data in [('decisions.csv', decision_csv), ('action_details.csv', action_csv)]:
        with (ART / name).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    intro = ['# BGA #927500015：本地第12轮模型逐步分析', '',
             f"状态：{status_label}；已完成{completed}/{expected}次根搜索。", '',
             '使用混合预算在线分支本地第12轮原始模型（128×2、583682参数），不是旧R20或EMA。总S平分给N个世界；无根噪声、T=0；每组3个种子。主行动56次、可选支付20次、强制12次。', '',
             '首选取三个种子的平均访问份额argmax，同时保留各次首选。Q为各个有访问种子的等权均值；Q差仅在同一种子两动作都被访问时配对。Q范围不是置信区间，低访问Q不可靠。高S和与人类一致都不证明决策更优。', '',
             '只使用行动者当时可见的DecisionRequest；真实后续补牌、对手暗牌和终局结果不进入模型／搜索。每个局面的V相对于当时行动者，不能直接把相邻双方V相减当作失误损失。', '',
             '| 方法 | VnEpCxmR一致 | FourDimensional一致 | 合计 | 三种子内首选变化 |', '| --- | ---: | ---: | ---: | ---: |']
    for name, stats in summary['panels'].items():
        count = lambda d: f"{d['human_agree']}/{d['decisions']}"
        instability = stats.get('seed_instability')
        intro.append(f"| {name} | {count(stats['0'])} | {count(stats['1'])} | {count(stats['None'])} | " + (f"{instability['any_top1_change']}/{instability['decisions']}" if instability else '—') + ' |')
    (ART / 'results.md').write_text('\n'.join(intro + ['', '逐步全部候选见action_details.csv；逐种子／逐世界访问和value_sum见search/。', ''] + list(itertools.chain.from_iterable(b['lines'] + [''] for b in blocks))), encoding='utf-8')
    def board_text(obs):
        def tokens(t):
            return ' '.join(COLORS[c] + str(v) for c, v in t.items() if c != '__type__')
        text = []
        for key, label in [('self_public', '行动者'), ('opponent_public', '对手')]:
            p = obs[key]
            text.append(f'{label}：分数{p["score"]}；筹码 {tokens(p["tokens"])}；加成 {tokens(p["bonuses"])}；保留 ' + json.dumps(p['reserved_cards'], ensure_ascii=False))
        for tier, cards in obs['market'].items():
            if tier == '__type__':
                continue
            text.append(tier + '：' + '；'.join(f'{c["card_id"]} {COLORS[c["bonus"]]}+/{c["points"]}分 成本[{tokens(c["cost"])}]' for c in cards))
        text.append('己方已知盲保留：' + json.dumps(obs['self_private']['blind_reserved_cards'], ensure_ascii=False))
        return html.escape('\n'.join(text))
    articles = []
    for b in blocks:
        # Markdown's six-column method table rendered directly, no external JS/CSS.
        table_lines = [line for line in b['lines'][4:] if line.startswith('|') and not line.startswith('| ---')]
        table = '<table>' + ''.join('<tr>' + ''.join('<td>' + html.escape(cell.strip()) + '</td>' for cell in line.strip('|').split('|')) + '</tr>' for line in table_lines) + '</table>'
        articles.append(f'<article data-actor="{b["actor"]}" data-phase="{b["phase"]}"><h2>{html.escape(b["heading"])}</h2><p>{html.escape(b["lines"][2])}</p>{table}<details><summary>当时可见局面与各方法前5候选</summary><pre>{board_text(b["observation"])}</pre>{b["details"]}</details></article>')
    page = '''<!doctype html><meta charset="utf-8"><title>BGA 927500015 · local12</title>
<style>body{font:15px system-ui;max-width:1450px;margin:25px auto;padding:0 18px;background:#fafafa;color:#222}article{background:white;padding:18px;margin:18px 0;border:1px solid #ddd;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}td{padding:7px;border-bottom:1px solid #eee}pre{white-space:pre-wrap;font:13px system-ui}summary{cursor:pointer;padding:12px 0}select{padding:8px;margin-right:12px}</style>
<h1>BGA #927500015 · 本地第12轮模型逐步分析</h1><p>STATUS。每组3个种子；S为总预算，N4平分。Q和推荐只是模型意见；低访问估计不稳定，高预算不是正确答案。仅使用当时可见信息。</p>
<select id="actor"><option value="all">双方</option><option value="0">VnEpCxmR</option><option value="1">FourDimensional</option></select><select id="phase"><option value="main_action">主行动</option><option value="all">全部88决策</option><option value="payment">支付</option></select>
ARTICLES<script>function filter(){document.querySelectorAll('article').forEach(x=>x.hidden=!((actor.value==='all'||x.dataset.actor===actor.value)&&(phase.value==='all'||x.dataset.phase===phase.value)))}actor.onchange=phase.onchange=filter;filter()</script>'''
    page = page.replace('STATUS', status_label + f'，根搜索{completed}/{expected}').replace('ARTICLES', ''.join(articles))
    (ART / 'report.html').write_text(page, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['prepare', 'smoke', 'run', 'report'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare': prepare()
        elif args.command == 'smoke': smoke(args.device)
        elif args.command == 'run': run()
        else: report()


if __name__ == '__main__':
    main()
