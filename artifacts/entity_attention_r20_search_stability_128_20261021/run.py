"""Frozen information-safe roots, independent-seed searches and proxy comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import itertools
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.catalog import load_catalog
from splendor.compact_trace import decode_request, TRACE_VERSION
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.model import DecisionPhase
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy, NativeChancePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import RULESET_ID, to_primitive
from splendor.result import TerminalResult
from splendor.rng import RNG_ALGORITHM
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.search_policy import determinize_request
from splendor.selfplay_data import STATE_FIELDS, load_game

ROOT = Path(os.environ.get('SPLENDOR_PROBE_ROOT', Path(__file__).resolve().parents[2]))
ART = ROOT/'artifacts/entity_attention_r20_search_stability_128_20261021'
WORK = ROOT/'runtime'/ART.name
RUN = ROOT/'runs/entity_attention_37423_random_budget_p6k_v24k_20261019'
CHECKPOINT = RUN/'rounds/round-019/checkpoints/final.pt'
MODEL_SHA = 'fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd'
STAGES = ('early', 'middle', 'late')
REPEATS = tuple(range(8))
SEED_BASE = 42100000
SCREEN_SEEDS = tuple(range(42010000, 42010004))
MIN_Q_VISITS = 16
TOP_K = (1, 3, 5)
IMPORTANCE_PP = (0.5, 1., 2., 5.)


def panel(name, backend, s, n=1, reference=False):
    # User convention: S is PER WORLD. Internal MCTSConfig.simulations is TOTAL.
    return dict(name=name, backend=backend, per_world_simulations=s,
                worlds=n, total_simulations=s*n, reference=reference)


PANELS = (
    panel('S1024N1', 'native', 1024),
    panel('chance_S1024', 'native_chance', 1024),
    panel('S256N4', 'native', 256, 4),
    panel('S1024N4', 'native', 1024, 4),
    panel('greedy', 'greedy', 0),
    panel('S128N1', 'native', 128),
    panel('chance_S128', 'native_chance', 128),
    panel('reference_S16384N4', 'native', 16384, 4, True),
    panel('reference_chance_S65536', 'native_chance', 65536, 1, True),
)
REFS = tuple(p['name'] for p in PANELS if p['reference'])


def repeat_seeds(p):
    base = 42200000 if p['name'] == REFS[0] else 42300000 if p['name'] == REFS[1] else SEED_BASE
    return [base+i for i in REPEATS]


def object_sha(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def js(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.shape != q.shape or min(p.sum(), q.sum()) <= 0 or np.any(p < 0) or np.any(q < 0):
        raise ValueError('JS requires aligned nonnegative nonempty distributions')
    p, q = p/p.sum(), q/q.sum()
    m = (p+q)/2
    def kl(x):
        mask = x > 0
        return float(np.sum(x[mask]*np.log(x[mask]/m[mask])))
    return (kl(p)+kl(q))/2


def order(p):
    return sorted(range(len(p)), key=lambda i: (-p[i], i))


def positive_top(p, k):
    return [i for i in order(p)[:k] if p[i] > 0]


def seed_stability(rows):
    pairs = list(itertools.combinations(rows, 2))
    counts = Counter(r['chosen'] for r in rows)
    return dict(top1_pairwise_agreement=float(np.mean([a['chosen'] == b['chosen'] for a, b in pairs])),
                top1_modal_fraction=max(counts.values())/len(rows), unanimous=len(counts) == 1,
                policy_pairwise_js=float(np.mean([js(a['policy'], b['policy']) for a, b in pairs])))


def reference(rows, minimum=MIN_Q_VISITS):
    """Equal weight for every independent world, never pool N4 visits into one Q."""
    p = np.mean([r['policy'] for r in rows], axis=0)
    visits = np.concatenate([r['world_visits'] for r in rows], axis=0).astype(float)
    sums = np.concatenate([r['world_value_sums'] for r in rows], axis=0).astype(float)
    supported = np.all(visits >= minimum, axis=0)
    q = [float(np.mean(sums[:, i]/visits[:, i])) if supported[i] else None for i in range(len(p))]
    top = order(p)
    a, b = top[:2]
    gap = (q[a]-q[b])*50 if q[a] is not None and q[b] is not None else None
    return dict(policy=p.tolist(), chosen=a, q=q, policy_top2_gap_score_pp=gap,
                q_min_visits_per_world=visits.min(axis=0).astype(int).tolist(),
                stability=seed_stability(rows))


def compare(rows, ref):
    target = ref['chosen']
    values = dict(reference_top1_hit=[], reference_policy_js=[], reference_action_zero_visit=[],
                  reference_action_le1_visit=[], reference_action_le4_visits=[], q_regret_score_pp=[])
    for k in TOP_K:
        for prefix in ('reference_top1_in_method_top', 'method_top1_in_reference_top',
                       'reference_mass_in_method_top', 'reference_topk_set_coverage_'):
            values[prefix+str(k)] = []
    for r in rows:
        p = r['policy']; chosen = r['chosen']
        values['reference_top1_hit'].append(chosen == target)
        values['reference_policy_js'].append(js(p, ref['policy']))
        if r['visits'] is not None:
            for label, threshold in (('reference_action_zero_visit', 0), ('reference_action_le1_visit', 1),
                                     ('reference_action_le4_visits', 4)):
                values[label].append(r['visits'][target] <= threshold)
        qa, qb = ref['q'][target], ref['q'][chosen]
        if qa is not None and qb is not None:
            values['q_regret_score_pp'].append((qa-qb)*50)
        for k in TOP_K:
            mine, theirs = positive_top(p, k), positive_top(ref['policy'], k)
            values['reference_top1_in_method_top'+str(k)].append(target in mine)
            values['method_top1_in_reference_top'+str(k)].append(chosen in theirs)
            values['reference_mass_in_method_top'+str(k)].append(sum(ref['policy'][i] for i in mine))
            values['reference_topk_set_coverage_'+str(k)].append(len(set(mine)&set(theirs))/len(theirs))
    result = {k: float(np.mean(v)) if v else None for k, v in values.items()}
    result['q_regret_seed_coverage'] = len(values['q_regret_score_pp'])/len(rows)
    result['importance_gap_score_pp'] = ref['policy_top2_gap_score_pp']
    return result


def catalog(kind):
    return load_catalog(ART/('historical_catalog' if kind == 'historical' else 'search_catalog'))


def replay_game(arrays, meta, old):
    trace = meta['trace']
    if (trace['catalog_fingerprint'] != old.data_fingerprint or trace['version'] != TRACE_VERSION
            or trace['ruleset_id'] != RULESET_ID or trace['rng_algorithm'] != RNG_ALGORITHM):
        raise ValueError('Historical trace identity mismatch')
    engine = GameEngine.new(meta['game_seed'], first_player=meta['first_player'], catalog=old, mode='checked')
    roots = []
    for i, action in enumerate(trace['actions']):
        req = engine.decision(); chosen = int(action['candidate_index'])
        if (not isinstance(req, DecisionRequest) or action['decision_index'] != i
                or action['phase'] != req.observation.phase.value or action['player_id'] != engine.current_player
                or int(arrays['chosen_candidate_index'][i]) != chosen
                or to_primitive(req.legal_actions[chosen]) != action['action']):
            raise ValueError(f'Archived choice mismatch at {i}')
        if req.observation.phase is DecisionPhase.MAIN_ACTION and len(req.legal_actions) > 1:
            roots.append((i, req))
        engine.apply(req.legal_actions[chosen])
    terminal = engine.decision()
    if not isinstance(terminal, TerminalResult) or to_primitive(terminal) != meta['terminal_result']:
        raise ValueError('Historical natural terminal mismatch')
    return roots


def prepare():
    ART.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    if (ART/'plan.json').exists():
        verify_plan(); return
    if sha256(CHECKPOINT) != MODEL_SHA:
        raise ValueError('R20 model changed')
    for name, source in (('historical_catalog', ROOT/'artifacts/noble_pool_bga12_20261021/previous_catalog'),
                         ('search_catalog', ROOT/'data')):
        (ART/name).mkdir(exist_ok=True)
        for f in ('splendor_cards.csv', 'splendor_nobles.csv', 'splendor_config.json'):
            shutil.copyfile(source/f, ART/name/f)
    rng = random.Random(42000000)
    candidates = []; sources = []; indexes = []
    group_counts = Counter()
    # 288 validation games, one root per game; 48 games from each of rounds 15..20.
    for rnd in range(14, 20):
        index = RUN/f'rounds/round-{rnd:03d}/collection_index.json'
        records = read_json(index)['games']
        available = sorted((g for g in records if g['split'] == 'validation'), key=lambda g: g['game_id'])
        selected = rng.sample(available, 48)
        indexes.append(dict(path=str(index.relative_to(ROOT)), sha256=sha256(index)))
        for slot, game in enumerate(selected):
            path = Path(game['path'])
            if 'archive_sha256' in game and sha256(path) != game['archive_sha256']:
                raise ValueError('Source archive changed')
            arrays, meta = load_game(path)
            roots = replay_game(arrays, meta, catalog('historical'))
            group = 'baseline' if slot < 16 else 'screen_candidate'
            stage = STAGES[group_counts[group] % 3]; group_counts[group] += 1
            part = STAGES.index(stage)
            eligible = roots[len(roots)*part//3:len(roots)*(part+1)//3]
            if not eligible:
                raise ValueError('No main-action roots in requested trajectory third')
            decision, req = rng.choice(eligible)
            encoded = encode_decision(req)
            for field in STATE_FIELDS:
                np.testing.assert_array_equal(getattr(encoded, field), arrays[field][decision])
            lo, hi = map(int, arrays['candidate_offsets'][decision:decision+2])
            np.testing.assert_array_equal(encoded.candidate_features, arrays['candidate_features'][lo:hi])
            # Unplayed nobles cannot change this conditional game; validate both reconstructions.
            old_world = determinize_request(req, seed=42009000, catalog=catalog('historical'), engine_mode='checked')
            new_world = determinize_request(req, seed=42009000, catalog=catalog('search'), engine_mode='checked')
            if old_world.decision() != req or new_world.decision() != req:
                raise ValueError('Information-safe root differs under frozen catalogs')
            row = dict(id=len(candidates), stage=stage, group=group,
                       source_round=rnd+1, source_game_id=game['game_id'], decision_index=decision,
                       source_path=str(path.relative_to(ROOT)), source_sha256=sha256(path),
                       source_game_seed=meta['game_seed'], request=to_primitive(req),
                       completed_player_turns=req.observation.own_completed_turns+req.observation.opponent_completed_turns,
                       scores=[req.observation.self_public.score, req.observation.opponent_public.score],
                       candidate_count=len(req.legal_actions))
            row['request_sha256'] = object_sha(row['request'])
            candidates.append(row)
            sources.append(dict(candidate_id=row['id'], archive=row['source_path'], sha256=row['source_sha256'], trace=meta['trace']))
    paired = sorted(zip(candidates, sources, strict=True), key=lambda x: x[0]['group'] != 'baseline')
    candidates, sources = [x[0] for x in paired], [x[1] for x in paired]
    for i, (r, s) in enumerate(zip(candidates, sources, strict=True)):
        r['id'] = s['candidate_id'] = i
    write_json(ART/'candidate_bank.json', dict(roots=candidates))
    with (ART/'source_games.jsonl').open('w') as f:
        for s in sources:
            f.write(json.dumps(s, ensure_ascii=False, separators=(',', ':'))+'\n')
    frozen = WORK/'source'
    identity = source_identity(ROOT, frozen)
    for p in [ROOT/'src/splendor/native_search.cpp', *ROOT.glob('src/splendor/_native_search*.so'), *ART.glob('*.py')]:
        target = frozen/p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    # Default loader (if reached) must see frozen normative data too.
    shutil.copytree(ART/'search_catalog', frozen/'data', dirs_exist_ok=True)
    files = [*frozen.rglob('*.py'), *frozen.rglob('*.so'), frozen/'src/splendor/native_search.cpp', *frozen.glob('data/*')]
    identity['snapshot_files'] = {str(p.relative_to(frozen)):sha256(p) for p in files}
    plan = dict(protocol='search-stability-fixed128-v1', checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                checkpoint_sha256=MODEL_SHA, panels=PANELS, formal_repeat_seeds=[SEED_BASE+i for i in REPEATS],
                panel_repeat_seeds={p['name']:repeat_seeds(p) for p in PANELS},
                screening_seeds=SCREEN_SEEDS, selection_seed=42000000, positions=128,
                baseline_positions=96, screening_candidates=192, high_disagreement_positions=32,
                source_indexes=indexes, historical_catalog=catalog('historical').data_fingerprint,
                search_catalog=catalog('search').data_fingerprint, source=identity,
                inputs={p.name:sha256(p) for p in (ART/'candidate_bank.json',ART/'source_games.jsonl')},
                scripts={p.name:sha256(p) for p in ART.glob('*.py')},
                catalog_files={str(p.relative_to(ART)):sha256(p) for name in ('historical_catalog','search_catalog') for p in (ART/name).iterdir()},
                execution=dict(device='cuda', batch_size=500, low_budget_active=64, high_budget_active=16,
                               pipeline_slots=1, fp32=True, tf32=False, root_noise=False, temperature=0,
                               forced_playouts=False, policy_target_pruning=False, cpw=False),
                inference=InferenceConfig(mode='compiled', actor_projection=False, compile_mode='native-cudagraph',
                    batch_buckets=(16,32,64,128,256,500),candidate_buckets=(16,32,48,64),compile_failure_policy='fail_fast').to_dict(),
                q_minimum_per_world=MIN_Q_VISITS, importance_threshold_score_pp=IMPORTANCE_PP,
                total_formal_searches=128*8*8, total_formal_simulations=128*8*sum(p['total_simulations'] for p in PANELS),
                screening_searches=192*4, screening_simulations=192*4*1024,
                scope='Fixed roots from old ten-noble games, valid conditional states under current twelve-noble rules; no new-noble starting states. References are noisy model-search proxies, not ground truth. Root-equal macro statistics, stage-balanced baseline and deliberately enriched disagreement group reported separately.',
                command=sys.argv)
    write_json(ART/'plan.json', plan)
    print(json.dumps(dict(prepared=True,candidates=len(candidates),total_formal_simulations=plan['total_formal_simulations'])),flush=True)


def verify_plan():
    p = read_json(ART/'plan.json')
    checks = [(ROOT/p['checkpoint'],p['checkpoint_sha256'])]
    checks += [(ART/f,s) for f,s in {**p['inputs'],**p['scripts'],**p['catalog_files']}.items()]
    checks += [(WORK/'source'/f,s) for f,s in p['source']['snapshot_files'].items()]
    for path, digest in checks:
        if sha256(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')
    return p


def make_policy(p, seed, model, evaluator):
    cls = NativePUCTSearchPolicy if p['backend'] == 'native' else NativeChancePUCTSearchPolicy
    return cls(model, evaluator=evaluator, catalog=catalog('search'), config=MCTSConfig(
        simulations=p['total_simulations'], determinizations=p['worlds'], seed=seed,
        root_noise_fraction=0., forced_playouts=0., policy_target_pruning=False))


def result_row(job, root, p, seed):
    r = job.result
    if r.simulations != p['total_simulations'] or sum(r.root_visit_counts) != r.simulations or r.transition_cap_hits:
        raise ValueError('Search incomplete or hit transition cap')
    if tuple(r.simulation_budgets) != (p['per_world_simulations'],)*p['worlds']:
        raise ValueError('S*N budget allocation mismatch')
    counts = np.asarray(r.per_world_root_visit_counts)
    np.testing.assert_array_equal(counts.sum(axis=0), r.root_visit_counts)
    if r.root_network_prior != r.root_used_prior:
        raise ValueError('Unexpected root noise')
    return dict(root_id=root['id'], request_sha256=root['request_sha256'], panel=p['name'], seed=seed,
                chosen=r.chosen_candidate_index, policy=list(r.policy_target), visits=list(r.root_visit_counts),
                world_visits=counts.tolist(), world_value_sums=[list(x) for x in r.per_world_root_value_sums],
                prior=list(r.root_network_prior), simulation_budgets=list(r.simulation_budgets),
                simulations=r.simulations, max_tree_depth=r.max_tree_depth,
                transition_cap_hits=r.transition_cap_hits, diagnostics=job.core.chance_diagnostics())


def progress(stage, **kw):
    row = dict(status='running',stage=stage,unix=time.time(),max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**kw)
    write_json(WORK/'live.json',row)
    print(json.dumps(row),flush=True)


def search_panel(roots, p, seeds, model, evaluator, directory, active):
    directory.mkdir(parents=True,exist_ok=True)
    todo = [(r,s) for r in roots for s in seeds if not (directory/f"root-{r['id']:03d}-seed-{s}.json").exists()]
    runner = NativeBatchRunner(evaluator,catalog('search'),capacity=500,buckets=(16,32,48,64),slots=1)
    jobs = []; index = 0; completed = 0; start = time.perf_counter(); last = 0
    while index < len(todo) or jobs:
        while index < len(todo) and len(jobs) < active:
            root,seed = todo[index]; index += 1
            policy = make_policy(p,seed,model,evaluator)
            job = policy.start_job(decode_request(root['request']),add_root_noise=False,temperature=0.)
            jobs.append((root,seed,job))
        if runner.run([j for _,_,j in jobs]) <= 0:
            raise RuntimeError('Native search stalled')
        unfinished=[]
        for root,seed,job in jobs:
            if job.done:
                write_json(directory/f"root-{root['id']:03d}-seed-{seed}.json",result_row(job,root,p,seed))
                completed += 1
            else:
                unfinished.append((root,seed,job))
        jobs = unfinished
        now = time.perf_counter()-start
        if now-last >= 10 or (not jobs and index == len(todo)):
            progress(p['name'], completed_this_session=completed,pending=len(todo)-completed,elapsed_seconds=now,
                     native_batches=runner.batches,native_rows=runner.rows)
            last=now
    # Break the policy->active_job->policy reference cycles without accumulating trees.
    import gc
    gc.collect()


def load_rows(root, name, seeds, directory=None):
    base = directory if directory else WORK/'raw'/name
    rows = [read_json(base/f"root-{root['id']:03d}-seed-{s}.json") for s in seeds]
    for row, seed in zip(rows, seeds, strict=True):
        if row['root_id'] != root['id'] or row['seed'] != seed or row['request_sha256'] != root['request_sha256']:
            raise ValueError('Saved result/root identity mismatch')
    return rows


def select_bank():
    candidates=read_json(ART/'candidate_bank.json')['roots']
    ranked=[]
    for root in candidates[96:]:
        rows=load_rows(root,'screen',SCREEN_SEEDS,WORK/'screen')
        stats=seed_stability(rows)
        ranked.append(dict(candidate_id=root['id'],stage=root['stage'],**stats))
    ranked.sort(key=lambda r:(r['top1_pairwise_agreement'],-r['policy_pairwise_js'],r['candidate_id']))
    selected=ranked[:32]
    roots=[dict(r) for r in candidates[:96]]
    for item in selected:
        row=dict(candidates[item['candidate_id']]);row['group']='high_disagreement';row['screening']=item
        roots.append(row)
    if len({r['source_path'] for r in roots}) != 128 or len({r['request_sha256'] for r in roots}) != 128:
        raise ValueError('Frozen roots must be distinct and from independent source games')
    bank=dict(roots=roots,screen_ranking=ranked,
              selection='Baseline: first 96 preassigned roots, 32 per trajectory third. Enriched: 32/192 smallest screening top1 pairwise agreement, JS descending tiebreak. Four screening seeds are disjoint from eight formal seeds.',
              chosen_disagreeing_count=sum(r['top1_pairwise_agreement']<1 for r in selected))
    write_json(ART/'root_bank.json',bank)
    return roots


def mean_metrics(rows):
    if not rows:
        return dict(roots=0,metrics={})
    keys=[k for k,v in rows[0].items() if isinstance(v,(int,float,bool)) or v is None]
    result={}
    for k in keys:
        vs=[r[k] for r in rows if r.get(k) is not None]
        result[k]=dict(mean=float(np.mean(vs)) if vs else None, roots=len(vs))
    return dict(roots=len(rows),metrics=result)


def summarize():
    plan=verify_plan(); roots=read_json(ART/'root_bank.json')['roots']; seeds=plan['formal_repeat_seeds']
    references={ref:{r['id']:reference(load_rows(r,ref,plan['panel_repeat_seeds'][ref])) for r in roots} for ref in REFS}
    detailed=[]; stability=[]
    for root in roots:
        rid=root['id'];agree=references[REFS[0]][rid]['chosen']==references[REFS[1]][rid]['chosen']
        for p in PANELS:
            rows=load_rows(root,p['name'],plan['panel_repeat_seeds'][p['name']])
            stable=seed_stability(rows)
            base=dict(root_id=rid,panel=p['name'],group=root['group'],stage=root['stage'],references_agree=agree)
            stability.append(dict(**base,**stable))
            for ref in REFS:
                proxy=references[ref][rid]
                detailed.append(dict(**base,reference=ref,**compare(rows,proxy)))
    # Strata are fixed before reading method performance. All128 is deliberately selection-biased.
    strata={'all_128':lambda r:True,'baseline_96':lambda r:r['group']=='baseline',
            'high_disagreement_32':lambda r:r['group']=='high_disagreement',
            'references_agree':lambda r:r['references_agree'], 'references_disagree':lambda r:not r['references_agree']}
    for stage in STAGES:
        strata['baseline_'+stage]=lambda r,s=stage:r['group']=='baseline' and r['stage']==s
    groups=[]
    for p in PANELS:
        for group,condition in strata.items():
            s=[r for r in stability if r['panel']==p['name'] and condition(r)]
            groups.append(dict(panel=p['name'],stratum=group,stability=mean_metrics([
                {k:r[k] for k in ('top1_pairwise_agreement','top1_modal_fraction','unanimous','policy_pairwise_js')} for r in s])))
            for ref in REFS:
                selected=[r for r in detailed if r['panel']==p['name'] and r['reference']==ref and condition(r)]
                # Reference compared to itself is descriptive and not an independent accuracy test.
                groups[-1].setdefault('references',{})[ref]=dict(all=mean_metrics([
                    {k:v for k,v in r.items() if k not in ('root_id','references_agree')} for r in selected]))
                for t in IMPORTANCE_PP:
                    important=[r for r in selected if r['importance_gap_score_pp'] is not None and r['importance_gap_score_pp']>=t]
                    groups[-1]['references'][ref][f'important_ge_{t:g}pp']=mean_metrics([
                        {k:v for k,v in r.items() if k not in ('root_id','references_agree')} for r in important])
    write_json(ART/'reference_roots.json',references)
    write_json(ART/'per_root_metrics.json',dict(stability=stability,comparison=detailed))
    # Compact raw evidence is portable and sufficient to recompute every metric.
    import gzip
    raw=ART/'raw_searches.jsonl.gz';tmp=raw.with_suffix('.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8') as f:
        for p in PANELS:
            for root in roots:
                for row in load_rows(root,p['name'],plan['panel_repeat_seeds'][p['name']]):
                    f.write(json.dumps(row,separators=(',',':'))+'\n')
    tmp.replace(raw)
    with gzip.open(ART/'screen_searches.jsonl.gz','wt',encoding='utf-8') as f:
        for path in sorted((WORK/'screen').glob('root-*-seed-*.json')):
            f.write(json.dumps(read_json(path),separators=(',',':'))+'\n')
    summary=dict(status='complete',groups=groups,reference_consensus_agreement_roots=sum(
        references[REFS[0]][r['id']]['chosen']==references[REFS[1]][r['id']]['chosen'] for r in roots),
        reference_disclaimer='Model-search proxy only, not true optimality. Self-reference metrics use overlapping seeds and are not unbiased accuracy estimates.',
        raw_sha256=sha256(raw), root_bank_sha256=sha256(ART/'root_bank.json'),plan_sha256=sha256(ART/'plan.json'))
    write_json(ART/'summary.json',summary)
    lines=['# 128个固定局面搜索稳定性','', '两种高预算参考各由8个seed的平均访问分布定义首选；这是搜索参考吻合度，不是真实最优准确率。']
    for group in ('baseline_96','high_disagreement_32','all_128'):
        lines += ['',f'样本组：{group}。','',
                  '| 方法 | 跨seed首选一致率 | 平均两两JS | 命中S16384N4参考 | 命中随机节点S65536参考 |',
                  '|---|---:|---:|---:|---:|']
        for p in PANELS:
            row=next(g for g in groups if g['panel']==p['name'] and g['stratum']==group)
            s=row['stability']['metrics'];a=row['references'][REFS[0]]['all']['metrics'];b=row['references'][REFS[1]]['all']['metrics']
            lines.append(f"| {p['name']} | {s['top1_pairwise_agreement']['mean']:.2%} | {s['policy_pairwise_js']['mean']:.5f} | {a['reference_top1_hit']['mean']:.2%} | {b['reference_top1_hit']['mean']:.2%} |")
    lines+=['', '完整指标、重要决策阈值、参考一致/分歧分组、Top-k覆盖与有效Q样本数见summary.json及per_root_metrics.json。',
            'Greedy的零访问/少访问指标为null：它没有搜索。重要性用参考访问首选与访问次选的Q差（期望得分百分点），要求每个seed的每个世界对两个动作均至少16次访问；不满足即不判定。',
            '负Q regret保留，表示参考访问首选的Q未必最大；Q不能当作实测行动收益。JS单位nat、范围[0,ln2]。96个分阶段根与32个筛选根分别解释，128根平均数不代表自然对局分布。','']
    (ART/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(WORK/'live.json',dict(status='complete',unix=time.time(),summary=str((ART/'summary.json').relative_to(ROOT))))
    print(json.dumps(dict(status='complete',reference_agreement=summary['reference_consensus_agreement_roots'])),flush=True)


def execute(smoke=False):
    plan=verify_plan()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_frozen_actor_critic_checkpoint(CHECKPOINT,device='cuda')
    config=InferenceConfig(**{**plan['inference'],'candidate_buckets':tuple(plan['inference']['candidate_buckets']),
                              'batch_buckets':tuple(plan['inference']['batch_buckets'])})
    evaluator=FrozenSearchInference(model,device='cuda',config=config)
    candidates=read_json(ART/'candidate_bank.json')['roots']
    if smoke:
        root=candidates[0]
        for p in (panel('smoke_native', 'native', 8, 4),panel('smoke_chance','native_chance',32)):
            search_panel([root],p,[42009900],model,evaluator,WORK/'smoke'/p['name'],1)
        a=load_rows(root,'smoke_native',[42009900],WORK/'smoke/smoke_native')[0]
        b=load_rows(root,'smoke_chance',[42009900],WORK/'smoke/smoke_chance')[0]
        assert a['simulation_budgets']==[8]*4 and b['diagnostics']['stochastic']
        assert b['diagnostics']['root_hidden_samples']==32
        write_json(ART/'smoke.json',dict(scope='Two CUDA compiled engineering roots; excluded from formal statistics.',
                   passed=True,environment=environment_identity(),native=a,chance=b))
        return
    with (WORK/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(ART/'execution.json',dict(environment=environment_identity(),plan_sha256=sha256(ART/'plan.json'),
                   script_sha256=sha256(Path(__file__)),command=sys.argv,pid=os.getpid(),started_unix=time.time(),
                   frozen_source=str((WORK/'source').relative_to(ROOT))))
        search_panel(candidates[96:],panel('screen_S1024N1','native',1024),SCREEN_SEEDS,
                     model,evaluator,WORK/'screen',64)
        if (ART/'root_bank.json').exists():
            roots=read_json(ART/'root_bank.json')['roots']
        else:
            roots=select_bank()
        bank_identity = dict(sha256=sha256(ART/'root_bank.json'))
        if (WORK/'bank_identity.json').exists() and read_json(WORK/'bank_identity.json') != bank_identity:
            raise ValueError('Frozen root bank changed on resume')
        write_json(WORK/'bank_identity.json',bank_identity)
        for p in PANELS:
            if p['backend']=='greedy':
                directory=WORK/'raw'/p['name'];directory.mkdir(parents=True,exist_ok=True)
                for start in range(0,len(roots),64):
                    chunk=roots[start:start+64]
                    predictions=evaluator.policy_value(tuple(decode_request(r['request']) for r in chunk))
                    for root,(logits,v) in zip(chunk,predictions,strict=True):
                        prob=np.exp(np.asarray(logits,dtype=float)-max(logits));prob/=prob.sum()
                        for seed in plan['formal_repeat_seeds']:
                            write_json(directory/f"root-{root['id']:03d}-seed-{seed}.json",dict(root_id=root['id'],
                                request_sha256=root['request_sha256'],panel='greedy',seed=seed,chosen=int(np.argmax(prob)),
                                policy=prob.tolist(),visits=None,world_visits=[],world_value_sums=[],value=v,
                                deterministic_reused_prediction=True))
            else:
                search_panel(roots,p,plan['panel_repeat_seeds'][p['name']],model,evaluator,WORK/'raw'/p['name'],
                             16 if p['reference'] else 64)
        summarize()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','smoke','run','summarize'))
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='summarize':summarize()
    else:
        try:execute(smoke=args.stage=='smoke')
        except BaseException as error:
            write_json(WORK/'failure.json',dict(error_type=type(error).__name__,error=str(error),unix=time.time()))
            write_json(WORK/'live.json',dict(status='failed',error=str(error),unix=time.time()))
            raise


if __name__=='__main__':
    main()
