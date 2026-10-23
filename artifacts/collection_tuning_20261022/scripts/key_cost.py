import hashlib, json, statistics, time
from pathlib import Path
import numpy as np
import importlib.util, sys
spec=importlib.util.spec_from_file_location('bench_large','scripts/benchmark_large_batch_inference.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
collect_requests=mod.collect_requests
from splendor.features import encode_decision
STATE=('global_features','market_t1','market_t1_mask','market_t2','market_t2_mask','market_t3','market_t3_mask','self_reserved','self_reserved_mask','opponent_reserved','opponent_reserved_mask','nobles','nobles_mask','pending_card','pending_card_mask')
def key(enc):
    h=hashlib.blake2b(digest_size=16)
    for n in STATE:
        h.update(np.ascontiguousarray(getattr(enc,n)).tobytes())
    c=int(enc.candidate_mask.sum())
    h.update(np.ascontiguousarray(enc.candidate_features[:c]).tobytes())
    h.update(np.ascontiguousarray(enc.candidate_mask[:c]).tobytes())
    h.update(c.to_bytes(4,'little'))
    return h.digest()
reqs=collect_requests(64,seed=44000000); enc=[encode_decision(r) for r in reqs]
for _ in range(100): key(enc[_%64])
ts=[]
for i in range(2000):
    t=time.perf_counter(); key(enc[i%64]); ts.append(time.perf_counter()-t)
out={'requests':len(enc),'key_us_median':statistics.median(ts)*1e6,'key_us_p95':sorted(ts)[int(.95*len(ts))]*1e6}
Path('runtime/collection_tuning_20261022/results/key_cost.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
