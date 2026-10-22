"""Recover move18 and inspect next-turn decisions inside that search."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

from splendor.actions import BuyMarket, ReserveDeck
from splendor.bga_live import DEFAULT_CHECKPOINT, ReservationHistory, request_from_snapshot, runtime_catalog
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.model import Tier
from splendor.native_policy import (ChanceWideningConfig, NativeBatchRunner,
    NativeChancePUCTSearchPolicy, NativePUCTSearchPolicy)
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.recommend import decision_identity, describe_action
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.search_policy import determinize_request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = ROOT / 'artifacts/bga_934713322_move20_061_20261021'
SEEDS = (0, 1, 2)
BUDGETS = (1024, 16384)
METHODS = (('native', 1), ('native', 4), ('chance', 1), ('cpw', 1))
SHA = 'fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd'


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model():
    assert sha(DEFAULT_CHECKPOINT) == SHA
    torch.set_num_threads(1)
    model = load_frozen_actor_critic_checkpoint(DEFAULT_CHECKPOINT, device='cpu')
    return model, FrozenSearchInference(model, device='cpu',
        config=InferenceConfig(mode='eager', actor_projection=False))


def prepare():
    captured = json.loads((OUT / 'captured_snapshot.json').read_text())
    assert captured['move_text'] == '14'
    original = json.loads((PREVIOUS / 'root19_snapshot.json').read_text())
    snapshot = json.loads(json.dumps(original))
    gd = snapshot['gamedatas']
    gd['cards_hidden'].pop('20')
    own = gd['players']['100878205']
    own['nb_cards_hidden']['1'] -= 1
    own['coins']['G'] -= 1
    gd['market']['pool']['G'] += 1
    gd['market']['row_1']['count'] += 1
    gd['roundnumber'] = '9'
    gd['gamestate']['args']['turn'] = '9'
    gd['gamestate']['active_player'] = 100878205
    snapshot['move_text'] = '18'
    snapshot['public_logs'] = [row for row in snapshot['public_logs']
                               if row['id'] not in ('log_25', 'log_26')]
    events = json.loads((PREVIOUS / 'source_events.json').read_text())
    history = ReservationHistory.start(snapshot)
    for row in events:
        event = row['event']
        if int(row['move']) <= 18 and event.get('kind') == 'reserveCard' and event.get('public_reservation'):
            history.market_reserves[str(event['card_id'])] = str(event['actor_id'])
    request, catalog, _ = request_from_snapshot(snapshot, history)

    # Independently replay visible wallet changes from the newly captured move14.
    raw = json.loads((ROOT / 'runtime/bga_934713322_061_20261021/live_dom.json').read_text())[0]
    logs = {r['id']: r for r in raw['logs']}
    letters = 'CSEROG'
    wallets = {pid: dict(p['coins']) for pid, p in captured['gamedatas']['players'].items()}
    transfers = []
    for logid, pid, sign in [('log_20', '100878205', -1), ('log_21', '99846887', 1),
                             ('log_23', '100878205', -1), ('log_24', '99846887', 1)]:
        counts = Counter(re.findall(r'spl_log_coin icon_([CSEROG])', logs[logid]['html']))
        for color in letters:
            wallets[pid][color] += sign * counts[color]
        transfers.append({'log': logid, 'actor': pid, 'sign': sign, 'tokens': dict(counts)})
    assert wallets == {pid: p['coins'] for pid, p in gd['players'].items()}
    assert set(captured['gamedatas']['players']['100878205']['cards_played']) | {'9', '40'} == set(own['cards_played'])
    assert set(captured['gamedatas']['cards_hidden']) == set(gd['cards_hidden']) == {'51'}
    assert all(captured['gamedatas']['market'][f'row_{t}'] == gd['market'][f'row_{t}'] for t in (2, 3))
    for seed in SEEDS:
        assert determinize_request(request, seed=seed, catalog=catalog, engine_mode='checked').decision() == request
    model, evaluator = load_model()
    logits, value = evaluator.policy_value((request,))[0]
    assert abs(value - .4264373779296875) < 1e-7, value
    prior = np.exp(np.asarray(logits) - max(logits)); prior /= prior.sum()
    actions = [describe_action(a, catalog) for a in request.legal_actions]
    save('root18_snapshot.json', snapshot)
    save('root18_request.json', to_primitive(request))
    save('root18_history.json', history.to_dict())
    save('source_logs14_18.json', [logs[k] for k in ('log_19','log_20','log_21','log_22','log_23','log_24')])
    save('greedy.json', {'actions': actions, 'prior': prior.tolist(), 'value': value,
        'ranking': sorted(range(len(prior)), key=lambda i: (-prior[i], i))})
    save('input_validation.json', {'capture_move':14, 'recovered_move':18,
        'provenance':'Undo move18 blind reserve from saved move19; independently check wallets against move14 and public coin-icon logs.',
        'wallet_transfers14_18':transfers, 'wallets_match':True,
        'original_logged_value':.4264373779296875, 'recovered_value':value,
        'decision_sha256':decision_identity(request), 'hidden_card20_removed':True,
        'future_reveal53_not_in_root':True, 'checkpoint_sha256':SHA,
        'three_checked_worlds':True, 'previous_inputs':{
            str(p.relative_to(ROOT)):sha(p) for p in [PREVIOUS/'root19_snapshot.json',
                PREVIOUS/'source_events.json', ROOT/'runtime/bga_934713322_061_20261021/live_dom.json']}})
    buy = request.legal_actions.index(BuyMarket('061'))
    print(json.dumps({'prepared':True, 'value':value,
        'greedy':actions[int(np.argmax(prior))], 'buy061_prior':float(prior[buy]),
        'buy061_rank':1+int(sum(prior>prior[buy]))}, ensure_ascii=False), flush=True)


def load_probe():
    assert 'splendor._native_search' not in sys.modules
    build = json.loads((OUT / 'probe_build.json').read_text())
    binary = ROOT / build['binary']
    assert sha(binary) == build['binary_sha256']
    spec = importlib.util.spec_from_file_location('splendor._native_search', binary)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def policy_for(model, evaluator, catalog, method, worlds, budget, seed):
    cls = NativePUCTSearchPolicy if method == 'native' else NativeChancePUCTSearchPolicy
    kw = {'chance_widening':ChanceWideningConfig(k=.25, alpha=.5)} if method == 'cpw' else {}
    return cls(model, evaluator=evaluator, catalog=catalog, config=MCTSConfig(
        simulations=budget, determinizations=worlds, seed=seed, root_noise_fraction=0.), **kw)


def run(probe):
    if probe:
        load_probe()
    request = decode_request(json.loads((OUT / 'root18_request.json').read_text()))
    catalog, _, _ = runtime_catalog(json.loads((OUT / 'root18_snapshot.json').read_text())['gamedatas'])
    model, evaluator = load_model()
    runner = NativeBatchRunner(evaluator, catalog, capacity=12, buckets=(16,32,48,64), slots=1)
    jobs = []
    for budget in BUDGETS if probe else (1024,):
        for method, worlds in METHODS:
            if not probe and worlds != 1:
                continue
            for seed in SEEDS if probe else (0,):
                name = f'{method}_S{budget}_N{worlds}_seed{seed}'
                path = f'search/{name}.json' if probe else f'baseline/{name}.json'
                if (OUT / path).exists():
                    continue
                policy = policy_for(model,evaluator,catalog,method,worlds,budget,seed)
                jobs.append((path, policy.start_job(request,add_root_noise=False,temperature=0.)))
    if probe:
        # This root has an exactly reproduced online reference and tests the probe path too.
        old = decode_request(json.loads((PREVIOUS/'root_request.json').read_text()))
        policy = policy_for(model,evaluator,catalog,'native',1,1024,0)
        jobs.append(('reference_move20.json',policy.start_job(old,add_root_noise=False,temperature=0.)))
    save('execution_probe.json' if probe else 'execution_baseline.json', {
        'command':sys.argv, 'python':sys.version, 'torch':torch.__version__, 'numpy':np.__version__,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'checkpoint':str(DEFAULT_CHECKPOINT), 'checkpoint_sha256':SHA,
        'probe':probe, 'budgets':BUDGETS if probe else (1024,), 'seeds':SEEDS if probe else (0,),
        'methods':METHODS, 'execution':{'device':'cpu','threads':1,'capacity':12,'slots':1},
        'source_sha256':{p:sha(ROOT/p) for p in ('src/splendor/native_search.cpp',
            'src/splendor/native_policy.py','src/splendor/features.py','src/splendor/search_inference.py')},
        'script_sha256':sha(Path(__file__)), 'pending':len(jobs)})
    start = time.perf_counter();last = 0.
    while jobs:
        assert runner.run([j for _,j in jobs]) > 0
        for path, job in list(jobs):
            if not job.done:
                continue
            result = asdict(job.result)
            assert sum(result['root_visit_counts']) == result['simulations']
            if probe:
                result['early_nodes'] = job.core.debug_early_nodes()
            save(path, result)
            jobs.remove((path,job))
        elapsed = time.perf_counter()-start
        if elapsed-last > 15 or not jobs:
            print(json.dumps({'probe':probe,'pending':len(jobs),'elapsed_seconds':round(elapsed,2)}),flush=True)
            save('progress.json',{'probe':probe,'pending':len(jobs),'elapsed_seconds':elapsed})
            last = elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','baseline','probe'))
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        run(args.stage == 'probe')
