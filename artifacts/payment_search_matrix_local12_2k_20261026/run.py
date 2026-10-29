"""Matched payment roots: local12, three native budgets, with/without root noise."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, fields
import fcntl
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.catalog import load_default_catalog
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import EncodedDecision, encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_action, to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.search_policy import _request_seed, determinize_request

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime' / ART.name
SOURCE = ROOT / 'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-012/selfplay'
CHECKPOINT = ROOT / 'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt'
MODEL_SHA = '2a6304e3470cab3dad78ec0a2789ee892892e741f3731cafc2197cde9bd2d45d'
SAMPLE_SEED = 2026102601
SEARCH_SEED = 2026102602
NOISE_SEED_BASE = 2026103000
COUNT = 2000
CHUNK = 64
CAPACITY = 256
BUCKETS = (16, 32, 48, 64)


def panels():
    return [dict(name=f'{label}_noise{int(noise)}', label=label, simulations=total,
                 determinizations=worlds, noise=noise)
            for label, total, worlds in [('S256N4', 1024, 4), ('S1024N1', 1024, 1), ('S4096N4', 16384, 4)]
            for noise in (False, True)]


def identity_check(plan):
    for path, digest in plan['inputs'].items():
        assert sha256(ROOT / path) == digest, path
    for path, digest in plan['source']['source_files'].items():
        assert sha256(ROOT / path) == digest, path
    for path, digest in plan['source']['native_extensions'].items():
        assert sha256(Path(path)) == digest, path
    assert load_default_catalog().data_fingerprint == plan['catalog_fingerprint']


def requests():
    return [json.loads(line) for line in (RUNTIME / 'roots.jsonl').read_text().splitlines()]


def prepare():
    if (ART / 'plan.json').exists():
        identity_check(read_json(ART / 'plan.json'))
        print(json.dumps({'prepared': True, 'reused': True}), flush=True)
        return
    started = time.perf_counter()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    assert sha256(CHECKPOINT) == MODEL_SHA
    catalog = load_default_catalog()
    population = []
    paths = sorted((SOURCE / 'games').glob('game-*.npz'))
    assert len(paths) == 10000
    for game_id, path in enumerate(paths):
        assert path.name == f'game-{game_id:06d}.npz'
        with np.load(path, allow_pickle=False) as z:
            ids = np.flatnonzero((z['phase'] == 1) & (np.diff(z['candidate_offsets']) > 1))
            population.extend((game_id, int(i)) for i in ids)
        if (game_id + 1) % 2500 == 0:
            print(json.dumps({'stage': 'inventory', 'games': game_id + 1, 'payment_roots': len(population)}), flush=True)
    assert len(population) == 89027
    population = np.asarray(population, dtype=np.int32)
    np.savez_compressed(RUNTIME / 'population.npz', game_decision=population)
    selected = np.sort(np.random.default_rng(SAMPLE_SEED).choice(len(population), COUNT, replace=False))
    by_game = defaultdict(dict)
    for root_id, population_index in enumerate(selected):
        game_id, decision_index = map(int, population[population_index])
        by_game[game_id][decision_index] = (root_id, int(population_index))
    roots = [None] * COUNT
    refs = [None] * COUNT
    selected_files = {}
    checked_features = 0
    feature_fields = [f.name for f in fields(EncodedDecision)
                      if f.name not in ('feature_schema_version', 'candidate_features', 'candidate_mask', 'candidate_actions')]
    for number, (game_id, selected_decisions) in enumerate(sorted(by_game.items()), 1):
        path = paths[game_id]
        selected_files[str(path.relative_to(ROOT))] = sha256(path)
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(z['metadata_json'].tobytes().decode())
            assert metadata['checkpoint_sha256'] == MODEL_SHA
            trace = metadata['trace']
            assert trace['catalog_fingerprint'] == catalog.data_fingerprint
            assert len(trace['actions']) == len(z['phase'])
            # Only engine.decision() is exported to search, never authoritative hidden state.
            engine = GameEngine.new(trace['game_seed'], first_player=trace['first_player'], catalog=catalog, mode='checked')
            current = engine.decision()
            arrays = {name: z[name] for name in feature_fields}
            offsets, candidates = z['candidate_offsets'], z['candidate_features']
            turns, players = z['completed_player_turns'], z['player_id']
            for i, item in enumerate(trace['actions']):
                assert isinstance(current, DecisionRequest)
                assert item['decision_index'] == i and item['player_id'] == engine.current_player
                action = _decode_action(item['action'])
                assert current.legal_actions[item['candidate_index']] == action
                if i in selected_decisions:
                    root_id, population_index = selected_decisions[i]
                    encoded = encode_decision(current)
                    for name in feature_fields:
                        np.testing.assert_array_equal(getattr(encoded, name), arrays[name][i])
                    lo, hi = map(int, offsets[i:i + 2])
                    np.testing.assert_array_equal(encoded.candidate_features, candidates[lo:hi])
                    assert encoded.candidate_mask.all()
                    assert current.observation.phase.value == 'payment' and hi - lo > 1
                    gold = [a.tokens.gold for a in current.legal_actions]
                    assert gold.count(min(gold)) == 1
                    obs = current.observation
                    completed = obs.own_completed_turns + obs.opponent_completed_turns
                    assert completed == int(turns[i])
                    stage = ('opening_first10turns' if completed < 10 else
                             'late_maxscore_ge10' if max(obs.self_public.score, obs.opponent_public.score) >= 10 else 'middle')
                    primitive = to_primitive(current)
                    assert decode_request(primitive) == current
                    # Information-safe reconstruction also preserves candidate order.
                    if root_id < 8:
                        world = determinize_request(current, seed=_request_seed(current, SEARCH_SEED, 0), engine_mode='fast', catalog=catalog)
                        assert world.decision() == current
                    terminal = i == len(trace['actions']) - 1
                    if terminal:
                        assert trace['terminal_result']['reason'] == 'final_round'
                    ref = dict(root_id=root_id, population_index=population_index, game_id=game_id,
                               decision_index=i, game_seed=trace['game_seed'], player_id=int(players[i]),
                               completed_player_turns=completed, stage=stage, direct_terminal=terminal,
                               self_score=obs.self_public.score, opponent_score=obs.opponent_public.score,
                               card_tier=int(obs.pending.payment_target.tier), candidates=hi - lo,
                               gold=gold, minimum_gold=min(gold),
                               sampling_seed=NOISE_SEED_BASE + root_id)
                    refs[root_id] = ref
                    roots[root_id] = dict(**ref, request=primitive)
                    checked_features += len(feature_fields) + 1
                current = engine.apply(action)
            assert isinstance(current, TerminalResult)
            assert to_primitive(current) == trace['terminal_result']
        if number % 300 == 0:
            print(json.dumps({'stage': 'checked_replay', 'games': number, 'roots': sum(r is not None for r in roots)}), flush=True)
    assert all(r is not None for r in roots)
    for filename, rows in [(RUNTIME / 'roots.jsonl', roots), (ART / 'root_refs.jsonl', refs)]:
        filename.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))
    frozen = RUNTIME / 'source'
    source = source_identity(ROOT, frozen)
    for p in [Path(__file__), ROOT / 'src/splendor/native_search.cpp', *sorted((ROOT / 'data').glob('*.json'))]:
        target = frozen / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    inputs = [CHECKPOINT, Path(__file__), ROOT / 'src/splendor/native_search.cpp',
              RUNTIME / 'roots.jsonl', RUNTIME / 'population.npz', ART / 'root_refs.jsonl', SOURCE / 'manifest.json',
              *sorted((ROOT / 'data').glob('*.json'))]
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device='cpu')
    plan = dict(schema='payment-root-matrix-v1', checkpoint=str(CHECKPOINT.relative_to(ROOT)), model_sha256=MODEL_SHA,
                model_config=asdict(model.config), source_dataset=str(SOURCE.relative_to(ROOT)),
                catalog_fingerprint=catalog.data_fingerprint, games=10000, population_roots=len(population), roots=COUNT,
                sample_seed=SAMPLE_SEED, search_seed=SEARCH_SEED, sampling_seed_base=NOISE_SEED_BASE,
                distinct_sampled_games=len(by_game), panels=panels(), source=source,
                inputs={str(p.relative_to(ROOT)): sha256(p) for p in inputs}, selected_game_inputs=selected_files,
                search_common=dict(c_puct=1.5, root_noise_fraction=.25, dirichlet_alpha=.3,
                                   fold_forced_phases=True, max_simulation_transitions=512,
                                   forced_playouts=0., policy_target_pruning=False),
                chunk_roots=CHUNK, capacity=CAPACITY, pipeline_slots=2,
                inference=InferenceConfig(mode='compiled', actor_projection=False, candidate_buckets=BUCKETS,
                                          compile_mode='native-cudagraph', batch_buckets=(64, 128, 256),
                                          compile_failure_policy='fail_fast').to_dict(),
                definitions=dict(extra_gold='strictly above unique legal minimum gold at the same root',
                                 rank='descending visit/probability, unnoised network prior, candidate index',
                                 top3_extra_mass='extra-gold normalized mass in first min(3,candidates), not renormalized',
                                 top3_extra_fraction='per-root extra mass / total mass within top3',
                                 stage='opening: completed player turns<10; late: otherwise max pre-payment score>=10; middle: otherwise',
                                 direct_terminal='checked recorded payment immediately returns final_round TerminalResult',
                                 greedy_noise='argmax of exact native noisy used prior from matched S256N4 root, no simulations',
                                 pairing='same 2000 roots and per-root noise draw; same config seed and world0 across budgets'),
                sampling='uniform without replacement from all 89027 multiple-payment roots; no model-dependent screening',
                repeats='one common search seed per root and one common Dirichlet draw per root; no multi-seed repeat',
                feature_arrays_checked=checked_features, prepare_seconds=time.perf_counter() - started,
                prepare_command=sys.argv)
    write_json(ART / 'plan.json', plan)
    counts = Counter(r['stage'] for r in refs)
    counts['direct_terminal'] = sum(r['direct_terminal'] for r in refs)
    print(json.dumps({'prepared': True, 'roots': COUNT, 'games': len(by_game), 'groups': counts,
                      'seconds': plan['prepare_seconds']}), flush=True)


def search_row(job, ref, panel, total):
    r = job.result
    visits = np.asarray(r.root_visit_counts, dtype=np.int64)
    worlds = np.asarray(r.per_world_root_visit_counts, dtype=np.int64)
    sums = np.asarray(r.per_world_root_value_sums, dtype=np.float64)
    assert visits.sum() == total == sum(r.simulation_budgets)
    np.testing.assert_array_equal(visits, worlds.sum(axis=0))
    np.testing.assert_array_equal(worlds.sum(axis=1), r.simulation_budgets)
    assert len(worlds) == panel['determinizations']
    assert np.all(np.isfinite(sums)) and np.all(np.abs(sums) <= worlds + 1e-6)
    p, u = np.asarray(r.root_network_prior), np.asarray(r.root_used_prior)
    assert np.isclose(p.sum(), 1.) and np.isclose(u.sum(), 1.)
    if not panel['noise']:
        np.testing.assert_array_equal(p, u)
    else:
        eta = (u - .75 * p) / .25
        assert eta.min() >= -1e-6 and np.isclose(eta.sum(), 1.)
    order = np.lexsort((np.arange(len(visits)), -p, -visits))
    assert int(order[0]) == r.chosen_candidate_index
    return dict(root_id=ref['root_id'], choice=r.chosen_candidate_index, visits=visits.tolist(),
                per_world_visits=worlds.tolist(), per_world_value_sums=sums.tolist(),
                prior=p.tolist(), used_prior=u.tolist(), simulation_budgets=list(r.simulation_budgets),
                world_seeds=[_request_seed(job.request, SEARCH_SEED, i) for i in range(len(worlds))],
                sampling_seed=ref['sampling_seed'], max_tree_depth=r.max_tree_depth,
                transition_cap_hits=r.transition_cap_hits)


def run(smoke=False):
    plan = read_json(ART / 'plan.json'); identity_check(plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device='cuda')
    config = dict(plan['inference'])
    for key in ('batch_buckets', 'candidate_buckets'):
        config[key] = tuple(config[key])
    evaluator = FrozenSearchInference(model, device='cuda', config=InferenceConfig(**config))
    runner = NativeBatchRunner(evaluator, load_default_catalog(), capacity=CAPACITY, buckets=BUCKETS, slots=2)
    roots = requests()
    if smoke:
        # Stage and terminal coverage; engineering validation only.
        ids = list(dict.fromkeys([0, 1] + [next(r['root_id'] for r in roots if r['stage'] == s)
                                          for s in ('opening_first10turns', 'middle', 'late_maxscore_ge10')]
                                + [next(r['root_id'] for r in roots if r['direct_terminal'])]))
        roots = [roots[i] for i in ids]
    started = time.perf_counter()
    if not smoke:
        write_json(RUNTIME / f'execution_{time.time_ns()}.json', dict(environment=environment_identity(), command=sys.argv, started_unix=time.time()))
    smoke_rows = []
    for panel in plan['panels']:
        total = 32 if smoke else panel['simulations']
        cfg = MCTSConfig(**plan['search_common'], simulations=total, determinizations=panel['determinizations'], seed=SEARCH_SEED)
        policy = NativePUCTSearchPolicy(model, evaluator=evaluator, config=cfg, catalog=load_default_catalog())
        for start in range(0, len(roots), CHUNK):
            group = roots[start:start + CHUNK]
            path = RUNTIME / 'search' / panel['name'] / f'{start:04d}.json'
            if path.exists() and not smoke:
                saved = read_json(path)
                assert [r['root_id'] for r in saved['rows']] == [r['root_id'] for r in group]
                assert saved['panel'] == panel and saved['plan_sha256'] == sha256(ART / 'plan.json')
                continue
            tick = time.perf_counter()
            jobs = [policy.start_job(decode_request(r['request']), add_root_noise=panel['noise'],
                                     temperature=0., sampling_seed=r['sampling_seed']) for r in group]
            while any(not j.done for j in jobs):
                assert runner.run([j for j in jobs if not j.done]) > 0
            result = [search_row(j, r, panel, total) for j, r in zip(jobs, group, strict=True)]
            elapsed = time.perf_counter() - tick
            if smoke:
                smoke_rows.append(dict(panel=panel, rows=result, seconds=elapsed))
            else:
                write_json(path, dict(panel=panel, plan_sha256=sha256(ART / 'plan.json'), rows=result,
                                      seconds=elapsed, finished_unix=time.time()))
                status = dict(stage='search', panel=panel['name'], panel_roots_done=start + len(group), panel_roots_total=COUNT,
                              chunk_seconds=elapsed, chunk_simulations=total * len(group),
                              simulations_per_second=total * len(group) / elapsed,
                              session_seconds=time.perf_counter() - started, updated_unix=time.time(),
                              inference=evaluator.stats.to_dict(), runner_wide_batches=runner.wide_batches,
                              runner_rows=runner.rows, runner_batches=runner.batches,
                              cuda_peak_bytes=torch.cuda.max_memory_allocated())
                write_json(RUNTIME / 'live.json', status)
                print(json.dumps({k: v for k, v in status.items() if k != 'inference'}), flush=True)
            del jobs
            gc.collect()
        del policy
    identity_check(plan)
    stats = dict(inference=evaluator.stats.to_dict(), runner_rows=runner.rows, runner_batches=runner.batches,
                 runner_wide_batches=runner.wide_batches, seconds=time.perf_counter() - started,
                 cuda_peak_bytes=torch.cuda.max_memory_allocated())
    if smoke:
        # Native noise is identical across N/budgets for a paired root.
        for i in range(len(roots)):
            for name in ('prior', 'used_prior'):
                for arm in (3, 5):
                    np.testing.assert_allclose(smoke_rows[1]['rows'][i][name], smoke_rows[arm]['rows'][i][name], atol=2e-5, rtol=2e-5)
        write_json(ART / 'smoke.json', dict(scope='engineering only, S=32 total; not formal matrix', chunks=smoke_rows, **stats))
    else:
        write_json(RUNTIME / 'complete.json', stats)
        write_json(RUNTIME / 'live.json', dict(stage='complete', updated_unix=time.time(), **stats))
    print(json.dumps({'complete': True, 'smoke': smoke, **stats}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'smoke', 'run'])
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            prepare()
        else:
            run(smoke=args.command == 'smoke')
