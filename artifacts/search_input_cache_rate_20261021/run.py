"""Observe exact native model inputs without caching or timing search performance."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, sha256, source_identity, write_json
from splendor.catalog import load_catalog
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy, NativeChancePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
WORK = ROOT / 'runtime' / ART.name
BANK = ROOT / 'artifacts/entity_attention_r20_search_stability_128_20261021'
CHECKPOINT = ROOT / 'runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt'
MODEL_SHA = 'fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd'
SEEDS = (42100000, 42100001)
PANELS = (('S1024N1', 'native', 1024, 1), ('S256N4', 'native', 256, 4),
          ('S1024N4', 'native', 1024, 4), ('chance_S1024', 'native_chance', 1024, 1))
BUCKETS = (16, 32, 48, 64)


def row_payload(packed, row: int, count: int) -> bytes:
    """Exact FP32/mask bytes, retaining field/candidate order, excluding padding."""
    state_only = bool(packed['state_only'])
    floats = np.asarray(packed['float_blob']).reshape(packed['capacity'], packed['float_stride'])
    masks = np.asarray(packed['mask_blob']).reshape(packed['capacity'], packed['mask_stride'])
    fields = {str(name): (int(offset), int(size)) for name, offset, size, _ in packed['float_fields']}
    mask_fields = {str(name): (int(offset), int(size)) for name, offset, size, _ in packed['mask_fields']}
    if state_only:
        f_end, m_end = packed['float_stride'], packed['mask_stride']
    else:
        off, size = fields['candidate_features']
        action_dim = size // packed['width']
        f_end = off + count * action_dim
        m_end = mask_fields['candidate_mask'][0] + count
    return (struct.pack('<BI', state_only, 0 if state_only else count)
            + floats[row, :f_end].astype('<f4', copy=False).tobytes()
            + masks[row, :m_end].astype('u1', copy=False).tobytes())


class Shadow:
    """One root/seed/panel namespace; no model response is replaced."""
    def __init__(self):
        self.worlds = {}
        self.ready = set()
        self.counts = Counter()
        self.kind_counts = Counter()
        self.input_bytes = 0

    def observe(self, key: bytes, world: int, kind: int, payload_size: int):
        self.counts['requests'] += 1
        self.kind_counts[str(kind)] += 1
        self.input_bytes += payload_size
        previous = self.worlds.get(key)
        if previous is None:
            self.counts['unique_inputs'] += 1
            self.worlds[key] = {world}
            classification = 'unique'
        else:
            if world in previous:
                self.counts['within_world_duplicates'] += 1
            else:
                self.counts['cross_world_duplicates'] += 1
            previous.add(world)
            classification = 'ready_hit' if key in self.ready else 'inflight_duplicate'
            self.counts[classification] += 1
        return classification

    def completed(self, keys):
        self.ready.update(keys)

    def result(self):
        names = ('requests', 'unique_inputs', 'within_world_duplicates', 'cross_world_duplicates',
                 'ready_hit', 'inflight_duplicate')
        c = {k: self.counts[k] for k in names}
        assert c['requests'] - c['unique_inputs'] == c['within_world_duplicates'] + c['cross_world_duplicates']
        assert c['requests'] - c['unique_inputs'] == c['ready_hit'] + c['inflight_duplicate']
        return dict(**c, logical_input_bytes=self.input_bytes, kind_counts=dict(self.kind_counts))


class ObservedRunner(NativeBatchRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shadows = {}
        self.labels = {}
        self.event_writer = None

    def _fill(self, *args, **kwargs):
        # _fill starts the real transfer; the input buffer remains immutable until _consume.
        record = super()._fill(*args, **kwargs)
        packed = record['buffer'].packed()
        observed = []
        for start, count, job in record['segments']:
            shadow = self.shadows[job]
            keys = []
            for row in range(start, start + count):
                world, simulation, kind, candidates = map(int, record['identity'][row])
                payload = row_payload(packed, row, candidates)
                digest = hashlib.sha256(payload).digest()
                classification = shadow.observe(digest, world, kind, len(payload))
                keys.append(digest)
                if self.event_writer is not None:
                    self.event_writer.write(json.dumps([self.labels[job], record['batch'], world,
                        simulation, kind, candidates, digest.hex(), classification], separators=(',', ':')) + '\n')
            observed.append((shadow, keys))
        record['shadow_inputs'] = observed
        return record

    def _consume(self, record):
        super()._consume(record)
        for shadow, keys in record['shadow_inputs']:
            shadow.completed(keys)


def selected_roots():
    bank = json.loads((BANK / 'root_bank.json').read_text())
    roots = [r for r in bank['roots'] if r['group'] == 'baseline']
    assert len(roots) == 96
    assert Counter(r['stage'] for r in roots) == dict(early=32, middle=32, late=32)
    assert len({r['request_sha256'] for r in roots}) == 96
    return roots


def inference_config():
    return InferenceConfig(mode='compiled', actor_projection=False, candidate_buckets=BUCKETS,
        compile_mode='native-cudagraph', compile_failure_policy='fail_fast', native_transport='reused',
        batch_buckets=(16, 32, 64, 128, 256, 500))


def prepare():
    if (ART / 'plan.json').exists():
        raise FileExistsError('Existing plan: use run to resume, not prepare')
    ART.mkdir(exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    assert sha256(CHECKPOINT) == MODEL_SHA
    roots = selected_roots()
    write_json(ART / 'roots.json', {'roots': roots, 'selection': 'All 96 un-enriched baseline roots; 32 per trajectory third, distinct source games.'})
    shutil.copytree(BANK / 'search_catalog', ART / 'catalog')
    frozen = WORK / 'source'
    source = source_identity(ROOT, frozen)
    for p in [ART / 'run.py', ROOT / 'src/splendor/native_search.cpp', *ROOT.glob('src/splendor/_native_search*.so')]:
        target = frozen / p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    shutil.copytree(ART / 'catalog', frozen / 'data')
    files = [ART / 'run.py', ART / 'roots.json', *sorted((ART / 'catalog').glob('*'))]
    write_json(ART / 'plan.json', dict(scope='Exact-input shadow-cache potential, no cached responses and no speed measurement',
        checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=MODEL_SHA,
        seeds=SEEDS, panels=[dict(name=n, backend=b, per_world_simulations=s, worlds=w, total_simulations=s*w)
                           for n,b,s,w in PANELS], active_jobs=16, capacity=500, pipeline_slots=1,
        root_noise=False, temperature=0, cpw=False, forced_playouts=0, policy_target_pruning=False,
        precision='float32', tf32=False, cache_scope='one root, seed, panel; unlimited capacity',
        key='SHA256 of exact little-endian FP32 and mask bytes, logical candidates only, ordered; kind and count included. Model SHA and schema frozen by plan namespace.',
        no_cross_seed_or_cross_root_cache=True, no_cross_move_measurement=True,
        files={str(p.relative_to(ROOT)): sha256(p) for p in files}, source=source,
        command=['conda','run','--no-capture-output','-n','splendor','env','PYTHONPATH=src','OMP_NUM_THREADS=1',
                 'MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','python','-u',str((ART/'run.py').relative_to(ROOT)),'run'] ))


def rates(rows):
    names = ('requests', 'unique_inputs', 'within_world_duplicates', 'cross_world_duplicates', 'ready_hit', 'inflight_duplicate')
    sums = {k: sum(r[k] for r in rows) for k in names}
    n = sums['requests']
    return dict(searches=len(rows), **sums,
        potential_rate=(n-sums['unique_inputs'])/n,
        ready_rate=sums['ready_hit']/n, inflight_rate=sums['inflight_duplicate']/n,
        within_world_rate=sums['within_world_duplicates']/n, cross_world_rate=sums['cross_world_duplicates']/n,
        root_mean_rate=float(np.mean([(r['requests']-r['unique_inputs'])/r['requests'] for r in rows])),
        root_rate_quantiles=dict(zip(('min','p25','p50','p75','max'),
            np.quantile([(r['requests']-r['unique_inputs'])/r['requests'] for r in rows], [0,.25,.5,.75,1]).tolist())))


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted((WORK/'raw').glob('*/*.json'))]
    summary = dict(complete=len(rows)==96*len(SEEDS)*len(PANELS), searches=len(rows),
        note='Input reuse potential along uncached baseline trajectories. Ready/inflight split depends on this scheduler. No speed inference or cross-move estimate.', panels={})
    for name, _, _, _ in PANELS:
        subset = [r for r in rows if r['panel']==name]
        if subset:
            summary['panels'][name] = dict(all=rates(subset), stages={stage:rates([r for r in subset if r['stage']==stage])
                    for stage in ('early','middle','late') if any(r['stage']==stage for r in subset)})
    write_json(ART/'per_root.json', dict(rows=rows))
    write_json(ART/'summary.json', summary)
    return summary


def run():
    plan=json.loads((ART/'plan.json').read_text())
    for path, digest in plan['files'].items():
        assert sha256(ROOT/path)==digest, path
    assert sha256(ROOT/plan['checkpoint'])==plan['checkpoint_sha256']
    assert torch.cuda.is_available(), 'Real CUDA required; do not silently change backend'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    env=environment_identity(); write_json(WORK/'environment.json', env)
    model=load_frozen_actor_critic_checkpoint(ROOT/plan['checkpoint'], device='cuda')
    evaluator=FrozenSearchInference(model, device='cuda', config=inference_config())
    catalog=load_catalog(ART/'catalog')
    roots=json.loads((ART/'roots.json').read_text())['roots']
    for name, backend, s, worlds in PANELS:
        directory=WORK/'raw'/name; directory.mkdir(parents=True,exist_ok=True)
        runner=ObservedRunner(evaluator,catalog,capacity=500,buckets=BUCKETS,slots=1)
        todo=[(r,seed) for r in roots for seed in SEEDS if not (directory/f"root-{r['id']:03d}-seed-{seed}.json").exists()]
        jobs=[]; index=0; completed=192-len(todo)
        with gzip.open(WORK/f'{name}-events-{completed:03d}.jsonl.gz', 'xt', compresslevel=1) as events:
            runner.event_writer=events
            while index<len(todo) or jobs:
                while index<len(todo) and len(jobs)<plan['active_jobs']:
                    root, seed=todo[index]; index+=1
                    cls=NativePUCTSearchPolicy if backend=='native' else NativeChancePUCTSearchPolicy
                    policy=cls(model,evaluator=evaluator,catalog=catalog,model_version=MODEL_SHA,
                        config=MCTSConfig(simulations=s*worlds,determinizations=worlds,seed=seed,
                            root_noise_fraction=0.,forced_playouts=0.,policy_target_pruning=False))
                    job=policy.start_job(decode_request(root['request']),add_root_noise=False,temperature=0.)
                    runner.shadows[job]=Shadow(); runner.labels[job]=[root['id'],seed]
                    jobs.append((root,seed,job))
                assert runner.run([j for _,_,j in jobs])>0
                remaining=[]
                for root,seed,job in jobs:
                    if not job.done:
                        remaining.append((root,seed,job)); continue
                    result=job.result
                    assert result.simulations==s*worlds and tuple(result.simulation_budgets)==(s,)*worlds
                    assert sum(result.root_visit_counts)==s*worlds and result.transition_cap_hits==0
                    shadow=runner.shadows.pop(job).result(); runner.labels.pop(job)
                    assert shadow['requests']==job.policy.stats.model_positions
                    row=dict(root_id=root['id'],stage=root['stage'],seed=seed,panel=name,
                        request_sha256=root['request_sha256'],simulations=result.simulations,
                        simulation_budgets=list(result.simulation_budgets),chosen=result.chosen_candidate_index,
                        visits=list(result.root_visit_counts),max_tree_depth=result.max_tree_depth,
                        diagnostics=job.core.chance_diagnostics(),**shadow)
                    write_json(directory/f"root-{root['id']:03d}-seed-{seed}.json",row)
                    completed+=1
                    if completed%16==0 or completed==192:
                        progress=dict(panel=name,completed=completed,expected=192,status='running')
                        write_json(WORK/'live.json',progress);print(json.dumps(progress),flush=True)
                jobs=remaining
        del runner;gc.collect()
        summarize()
    summary=summarize(); assert summary['complete']
    write_json(ART/'execution.json', dict(environment=env,inference=evaluator.stats.to_dict(),
        raw_events=[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p),bytes=p.stat().st_size)
                    for p in sorted(WORK.glob('*events*.jsonl.gz'))],
        plan_sha256=sha256(ART/'plan.json'), script_sha256=sha256(ART/'run.py')))
    write_json(WORK/'live.json',dict(status='complete',searches=summary['searches']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','summarize'))
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'summarize':summarize}[args.command]()
