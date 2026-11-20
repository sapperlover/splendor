"""Actual-game private-view value curve and bounded root diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze as a
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.model import DecisionPhase
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state, to_primitive
from splendor.search_inference import FrozenSearchInference

ART = Path(__file__).resolve().parent
RAW = a.RAW / 'advantage'


def replay_roots():
    saved = GameEngine.restore(a.REPLAY)
    record = saved.replay_record
    engine = GameEngine.new(record.header.game_seed, record.header.first_player)
    rows, requests, states = [], [], []
    for event in record.events:
        if event.automatic:
            continue
        request = engine.decision()
        state = engine._debug_state()
        assert engine.current_player == event.actor
        assert event.action in request.legal_actions
        if request.observation.phase is DecisionPhase.MAIN_ACTION:
            rows.append(dict(event=event.sequence, actor=event.actor,
                             turn=state.completed_turns[event.actor] + 1,
                             scores=[engine.observation_for(i).self_public.score for i in (0, 1)],
                             action=to_primitive(event.action),
                             private_reservations=[[r.card_id for r in p.reserved_cards] for p in state.players]))
            requests.append(request)
            states.append(to_primitive(state))
        engine.apply(event.action)
    assert engine.current_state_digest == saved.current_state_digest
    return rows, requests, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('curve', 'roots'))
    parser.add_argument('--events', type=int, nargs='+', default=[])
    args = parser.parse_args()
    a.identity_check()
    torch.set_num_threads(1)
    RAW.mkdir(parents=True, exist_ok=True)
    catalog = load_default_catalog()
    model = load_frozen_actor_critic_checkpoint(a.MODEL, device='cpu')
    evaluator = FrozenSearchInference(model, device='cpu')
    rows, requests, states = replay_roots()
    started = time.perf_counter()
    if args.mode == 'curve':
        for row, request, (logits, value) in zip(rows, requests, evaluator.policy_value(requests), strict=True):
            row.update(network_value=float(value), estimated_score_rate=(float(np.clip(value, -1, 1)) + 1) / 2,
                       greedy=to_primitive(request.legal_actions[int(np.argmax(logits))]))
        report = dict(scope='Each player evaluated only at their own actual MAIN_ACTION decision. '
                      'Private views differ; these are uncalibrated network estimates, not objective win probabilities.',
                      model_sha256=sha256(a.MODEL), replay_sha256=sha256(a.REPLAY),
                      script_sha256=sha256(Path(__file__)), frozen_source=read_json(a.ART/'plan.json')['source_snapshot'],
                      device='cpu', torch_threads=1, rows=rows, seconds=time.perf_counter()-started)
        write_json(ART/'curve.json', report)
        write_json(RAW/'states.json', [dict(event=r['event'], state=s) for r, s in zip(rows, states, strict=True)])
        for r in rows:
            print(r['event'], 'H' if r['actor']==0 else 'A', r['turn'], r['scores'],
                  round(r['estimated_score_rate']*100, 2), r['action'], flush=True)
    else:
        selected = [(r, q) for r, q in zip(rows, requests, strict=True) if r['event'] in args.events]
        assert len(selected) == len(set(args.events))
        policy = NativePUCTSearchPolicy(model, config=a.CFG, catalog=catalog, evaluator=evaluator)
        runner = NativeBatchRunner(evaluator, catalog, capacity=128, buckets=(16, 32, 48, 64), slots=1)
        results = a.search_all(policy, runner, [q for _, q in selected])
        summary, raw = [], []
        for (row, req), result in zip(selected, results, strict=True):
            detail = dict(**row, search=a.root_record(req, result))
            raw.append(detail)
            summary.append(dict(**row, chosen=detail['search']['chosen'], top=detail['search']['top'],
                                reproduces_actual=detail['search']['chosen']==row['action']))
            print(row['event'], summary[-1], flush=True)
        suffix = '_'.join(map(str, args.events))
        write_json(RAW/f'roots_{suffix}.json', raw)
        write_json(ART/f'roots_{suffix}.json', dict(search=asdict(a.CFG), device='cpu', torch_threads=1,
                   seconds=time.perf_counter()-started, script_sha256=sha256(Path(__file__)), rows=summary))


if __name__ == '__main__':
    main()
