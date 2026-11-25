"""Exact compact-array census, one game per worker; no cache warmup or training."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

from splendor.compact_game_cache import DictionaryArray, compact
from splendor.selfplay_data import load_game
from splendor.alphazero_recording import sha256, write_json

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent


def measure(record):
    p=Path(record['path']);s=p.stat()
    if (s.st_size,s.st_mtime_ns)!=(record['size'],record['mtime_ns']):
        raise ValueError(f'Pool game changed: {p}')
    arrays,meta=load_game(p)
    packed,_,payload=compact(arrays,meta)
    sizes=[]
    for a in packed.values():
        sizes.extend([a.values.nbytes,a.codes.nbytes] if isinstance(a,DictionaryArray) else [a.nbytes])
    return payload,sum((n+63)//64*64 for n in sizes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    manifest=ROOT/'runtime/value_outcome_60k_20261120/pool_games.jsonl'
    games=[json.loads(l) for l in manifest.read_text().splitlines()]
    train=[g for g in games if g['split']=='train'];payload=aligned=0;start=time.monotonic()
    with mp.get_context('spawn').Pool(args.workers) as workers:
        for i,(p,a) in enumerate(workers.imap(measure,train,chunksize=8),1):
            payload+=p;aligned+=a
            if i%2000==0:print(f'缓存容量核算 {i}/{len(train)}',flush=True)
    positions=sum(g['decision_count'] for g in train)
    capacity_mib=json.loads((ROOT/'configs/entity_attention_v1/outcome_60k_scalar_mc_20261120.json').read_text())['cache_mib']
    report=dict(status='complete',manifest_sha256=sha256(manifest),games=len(train),positions=positions,
        payload_bytes=payload,arena_aligned_upper_bound_bytes=aligned,
        capacity_mib=capacity_mib,capacity_bytes=capacity_mib*1024**2,fits=aligned<=capacity_mib*1024**2,
        parent_lambda_target_bytes=positions*3*4,temporary_teacher_prediction_bytes=positions*3*4,
        seconds=time.monotonic()-start,workers=args.workers,
        scope='All train archives decoded one at a time; exact dictionary/array byte census with per-array alignment upper bound. No arena allocation, model, optimizer or training.')
    write_json(ART/'cache_capacity.json',report)
    if not report['fits']:raise ValueError('Configured shared cache is too small')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
