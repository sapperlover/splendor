"""Measure Python exact-input key cost vs model forward cost.

This is a microbenchmark for the feasibility of an input-result cache; it does
not modify the production search path.  The key must hash every row, so its
cost is paid even on cache hits; compare it with the forward saving at the
observed ready-hit rate (~15% for the current mixed budget).
"""
from __future__ import annotations

import argparse, hashlib, json, statistics, time
from pathlib import Path
import numpy as np
import torch
import importlib.util
spec=importlib.util.spec_from_file_location('bench_large','scripts/benchmark_large_batch_inference.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
collect_requests=mod.collect_requests
from splendor.features import encode_decision
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference

STATE_FIELDS = (
    'global_features','market_t1','market_t1_mask','market_t2','market_t2_mask',
    'market_t3','market_t3_mask','self_reserved','self_reserved_mask',
    'opponent_reserved','opponent_reserved_mask','nobles','nobles_mask',
    'pending_card','pending_card_mask',
)

def key_row(enc) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for name in STATE_FIELDS:
        a = getattr(enc, name)
        h.update(np.ascontiguousarray(a).tobytes())
    count = int(enc.candidate_mask.sum())
    h.update(np.ascontiguousarray(enc.candidate_features[:count]).tobytes())
    h.update(np.ascontiguousarray(enc.candidate_mask[:count]).tobytes())
    h.update(count.to_bytes(4, 'little'))
    return h.digest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--requests', type=int, default=64)
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--key-repeats', type=int, default=2000)
    args = ap.parse_args()
    torch.set_num_threads(1)
    requests = collect_requests(args.requests, seed=44000000)
    encoded = [encode_decision(r) for r in requests]
    # key cost: measure each row once per repeat; report per-row median
    key_times=[]
    for _ in range(args.key_repeats):
        t=time.perf_counter()
        key_row(encoded[_ % len(encoded)])
        key_times.append(time.perf_counter()-t)
    key_us=statistics.median(key_times)*1e6
    out={'checkpoint':str(Path(args.checkpoint).resolve()),'requests':len(encoded),
         'key_us_per_row':key_us,'key_repeats':args.key_repeats,'forward':{}}
    for device in ('cpu','cuda'):
        if device=='cuda' and not torch.cuda.is_available():
            continue
        model=load_frozen_actor_critic_checkpoint(args.checkpoint, device=device)
        ev=FrozenSearchInference(model, device=device)
        batch=encoded*((256+len(encoded)-1)//len(encoded))
        batch=batch[:256]
        for _ in range(3): ev.policy_value_encoded(batch)
        if device=='cuda': torch.cuda.synchronize()
        ts=[]
        for _ in range(args.repeats):
            t=time.perf_counter(); ev.policy_value_encoded(batch)
            if device=='cuda': torch.cuda.synchronize()
            ts.append(time.perf_counter()-t)
        med=statistics.median(ts)
        out['forward'][device]={'batch':len(batch),'median_s':med,'us_per_row':med/len(batch)*1e6,
            'key_vs_forward':key_us/(med/len(batch)*1e6),
            'save_at_15pct_us_per_row':0.15*(med/len(batch)*1e6),
            'net_us_per_row':0.15*(med/len(batch)*1e6)-key_us}
        del ev, model
        if device=='cuda': torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
