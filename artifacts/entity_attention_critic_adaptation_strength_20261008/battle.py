"""Prepared, opt-in paired search matches for the frozen-policy critic adaptation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint, model_state_sha256
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.mcts_policy import MCTSStats
from splendor.native_policy import NativePUCTSearchPolicy, NativeSearchJob, NativeBatchRunner
from splendor.paired_statistics import serialized_pair_scores, paired_bootstrap_mean

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
TEMP = ROOT / 'runtime/entity_attention_critic_adaptation_strength_20261008'


def phase_for(request):
    o = request.observation
    return 'late' if max(o.self_public.score, o.opponent_public.score)>=10 else 'early'


def use_adapted(mode, candidate, phase):
    assert mode in ('full', 'early', 'late') and phase in ('early', 'late')
    return candidate and (mode=='full' or mode==phase)


@dataclass(slots=True)
class PhaseStats(MCTSStats):
    early_decisions: int = 0
    late_decisions: int = 0
    early_simulations: int = 0
    late_simulations: int = 0
    early_tree_depth_sum: int = 0
    late_tree_depth_sum: int = 0
    early_terminal_hits: int = 0
    late_terminal_hits: int = 0
    early_changed_action_count: int = 0
    late_changed_action_count: int = 0
    early_adapted_searchable_roots: int = 0
    late_adapted_searchable_roots: int = 0
    early_original_searchable_roots: int = 0
    late_original_searchable_roots: int = 0
    late_entry_count: int = 0
    late_entry_completed_turns_sum: int = 0
    late_entry_score_gap_sum: int = 0


class PhaseJob(NativeSearchJob):
    def __init__(self, *args, phase, adapted, **kwargs):
        self.phase, self.use_adapted = phase, adapted
        super().__init__(*args, **kwargs)

    def _collect_result(self):
        if self._finished:
            return
        counts = dict(self.core.stats())
        super()._collect_result()
        for key in ('simulations', 'tree_depth_sum', 'terminal_hits'):
            name = f'{self.phase}_{key}'
            setattr(self.policy.stats, name, getattr(self.policy.stats, name)+counts[key])
        result = self.result
        if result.searchable:
            raw = max(range(len(result.root_network_prior)), key=lambda i: (result.root_network_prior[i], -i))
            name = f'{self.phase}_changed_action_count'
            setattr(self.policy.stats, name, getattr(self.policy.stats, name)+int(raw!=result.chosen_candidate_index))


@contextmanager
def install_routing(mode, candidate_sha, base_critic_sha):
    """Route every leaf of a root's job through the root-selected frozen critic."""
    import splendor.evaluation_batch as batch
    original_policy, original_runner = batch.NativePUCTSearchPolicy, batch.NativeBatchRunner
    shared = {}

    class PhasePolicy(NativePUCTSearchPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.candidate = self.model_version==candidate_sha
            self.stats = PhaseStats()
            self.entered_late = False

        def start_job(self, request, *, add_root_noise=False, temperature=0., sampling_seed=None, root_encoded=None):
            if add_root_noise or temperature!=0:
                raise ValueError('Frozen paired matches require zero root noise and temperature')
            phase = phase_for(request)
            adapted = use_adapted(mode, self.candidate, phase)
            key = f'{phase}_decisions'
            setattr(self.stats, key, getattr(self.stats, key)+1)
            if len(request.legal_actions)>1:
                key = f'{phase}_{"adapted" if adapted else "original"}_searchable_roots'
                setattr(self.stats, key, getattr(self.stats, key)+1)
            if phase=='late' and not self.entered_late:
                self.entered_late = True
                o = request.observation
                self.stats.late_entry_count += 1
                self.stats.late_entry_completed_turns_sum += o.own_completed_turns+o.opponent_completed_turns
                self.stats.late_entry_score_gap_sum += o.self_public.score-o.opponent_public.score
            return PhaseJob(self, request, phase=phase, adapted=adapted, add_root_noise=False,
                            temperature=0., sampling_seed=sampling_seed, root_encoded=root_encoded)

    class RoutingRunner:
        def __init__(self, evaluator, catalog, **kwargs):
            self.evaluator, self.catalog, self.kwargs = evaluator, catalog, kwargs
            self.primary = original_runner(evaluator, catalog, **kwargs)
            self.base_runner = None
            self.adapted = model_state_sha256(evaluator.model.critic.state_dict())!=base_critic_sha
            if not self.adapted:
                shared['base_evaluator'] = evaluator
            self.batch_records = []
            self.last_runner = self.primary

        def run(self, jobs, *, context=None):
            self.batch_records = []
            if not self.adapted:
                count = self.primary.run(jobs, context=context)
                self.batch_records.extend(self.primary.batch_records)
                return count
            count = 0
            for adapted in (True, False):
                group = [job for job in jobs if job.use_adapted==adapted]
                if not group:
                    continue
                if adapted:
                    runner = self.primary
                else:
                    if self.base_runner is None:
                        self.base_runner = original_runner(shared['base_evaluator'], self.catalog, **self.kwargs)
                    runner = self.base_runner
                self.last_runner = runner
                count += runner.run(group, context=context)
                self.batch_records.extend(runner.batch_records)
            return count

        def failure_report(self, error):
            return self.last_runner.failure_report(error)

        def __getattr__(self, name):
            if name in ('rows', 'batches', 'wide_batches', 'root_rows', 'policy_rows', 'value_rows',
                        'fill_seconds', 'consume_seconds'):
                return getattr(self.primary, name)+(getattr(self.base_runner, name) if self.base_runner is not None else 0)
            raise AttributeError(name)

    batch.NativePUCTSearchPolicy, batch.NativeBatchRunner = PhasePolicy, RoutingRunner
    try:
        yield
    finally:
        batch.NativePUCTSearchPolicy, batch.NativeBatchRunner = original_policy, original_runner
        shared.clear()


def code_hashes():
    paths = [Path(__file__), *sorted((ROOT/'src/splendor').glob('*.py')), ROOT/'src/splendor/native_search.cpp',
             *sorted((ROOT/'src/splendor').glob('_native_search*.so'))]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def prepare():
    plan = read_json(ART/'battle_plan.json')
    hashes = code_hashes()
    for item in [plan['mse'], *plan['checkpoints'].values()]:
        assert sha256(Path(item['path']))==item['sha256'], f'Input changed: {item["path"]}'
    base_hash = None
    for step in ('0', '50', '200', '500'):
        p = load_alphazero_checkpoint(plan['checkpoints'][step]['path'], device='cpu')
        frozen = model_state_sha256({k: v for k, v in p.model_state_dict.items() if not k.startswith('critic.')})
        if base_hash is None:
            base_hash = frozen
        assert frozen==base_hash, 'Encoder/policy differ between candidate checkpoints'
    manifest_path = ART/'battle_manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest['plan']==plan and manifest['code_hashes']==hashes, 'Frozen battle plan or code changed'
    else:
        source = source_identity(ROOT, TEMP/'battle_source')
        destination = TEMP/'battle_source'/Path(__file__).relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(__file__, destination)
        manifest = {'plan': plan, 'code_hashes': hashes, 'source': source, 'preparation_environment': environment_identity(),
                    'status': 'prepared_not_started', 'frozen_encoder_policy_sha256': base_hash,
                    'scope': 'Full MSE complete; no matches executed during preparation.'}
        write_json(manifest_path, manifest)
    for step in (50, 200, 500):
        write_json(ART/f'battle_configs/dev-{step}.json', make_config(plan, step, 'dev', 'full').to_dict())
    return plan


def make_config(plan, step, split, mode):
    section = plan[split]
    pairs = section['pairs'] if mode=='full' else section['stage_pairs']
    execution = plan['execution']
    inference = {'mode': 'compiled', 'actor_projection': False, 'candidate_buckets': execution['candidate_buckets'],
                 'native_transport': 'reused', 'compile_failure_policy': 'fail_fast',
                 'compile_mode': 'native-cudagraph', 'batch_buckets': [16, 32, 64, 128, 200]}
    policies = {name: {'name': f'{name}_critic_{number}', 'kind': 'mcts_checkpoint',
                      'checkpoint': plan['checkpoints'][str(number)]['path'], 'search_backend': 'native',
                      'search_config': {**plan['search_config'], 'seed': section['search_seed']}, 'inference': inference}
                for name, number in (('policy_a', step), ('policy_b', 0))}
    return FixedEvaluationConfig.from_dict({**policies, 'pairs': pairs, 'seed_start': section['game_seed_start'],
        'policy_seed_base': section['policy_seed_base'], 'workers': 1, 'device': 'cuda',
        'max_completed_turns': 400, 'bootstrap_resamples': 10000, 'bootstrap_seed': section['bootstrap_seed'],
        'execution': execution, 'metadata': {'protocol': plan['protocol'], 'critic_mode': mode, 'extra_steps': step,
            'late_score_threshold': 10, 'root_critic_route': 'one frozen critic throughout each root search tree',
            'required_runner': str(Path(__file__).relative_to(ROOT)),
            'performance_scope': 'Stage routing shares the original evaluator; per-model timing fields can overlap, do not sum them.'}})


def select(plan):
    candidates = []
    mse = read_json(plan['mse']['path'])
    for step in (50, 200, 500):
        summary = read_json(ART/f'battles/dev-{step}.json')
        assert summary['passed'] and summary['completed_pairs']==plan['dev']['pairs']
        holdout = next(n for n in mse['nodes'] if n['step']==step)['aggregates']['all']['validation']['all']['MSE']
        candidates.append({'step': step, 'dev_score': summary['win_rate_score'], 'holdout_mse': holdout})
    winner = min(candidates, key=lambda c: (-c['dev_score'], c['holdout_mse'], c['step']))
    selection = {'selected_step': winner['step'], 'candidates': candidates,
                 'rule': 'Highest DEV score; exact ties choose lower holdout MSE, then fewer steps. Confirm regardless of DEV score.',
                 'scope': 'Selection uses DEV only; HEAD results are never used to choose the step.'}
    path = ART/'selection.json'
    if path.exists():
        assert read_json(path)==selection
    else:
        write_json(path, selection)
    return winner['step']


def run_panel(plan, label):
    if label.startswith('dev-'):
        step, split, mode = int(label.split('-')[1]), 'dev', 'full'
    else:
        step, split, mode = read_json(ART/'selection.json')['selected_step'], 'head', label.split('-')[1]
    config = make_config(plan, step, split, mode)
    write_json(ART/f'battle_configs/{label}.json', config.to_dict())
    output = ART/f'battles/{label}.json'
    base = load_alphazero_checkpoint(plan['checkpoints']['0']['path'], device='cpu')
    base_critic_sha = model_state_sha256({k.removeprefix('critic.'): v for k, v in base.model_state_dict.items() if k.startswith('critic.')})
    write_json(ART/f'invocations/{label}-{time.time_ns()}.json', {'command': sys.argv, 'environment': environment_identity()})
    def progress(n, total):
        row = {'panel': label, 'completed_pairs': n, 'pairs': total,
               'rss_bytes': int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')))*1024,
               'cuda_allocated_bytes': torch.cuda.memory_allocated(), 'cuda_reserved_bytes': torch.cuda.memory_reserved()}
        print(json.dumps(row), flush=True)
        write_json(TEMP/f'{label}_memory_latest.json', row)
    with install_routing(mode, plan['checkpoints'][str(step)]['sha256'], base_critic_sha):
        result = run_fixed_evaluation(config, output, resume=Path(str(output)+'.games').exists(),
            failure_dir=TEMP/f'failures/{label}', progress_every=25, summary_every=25, progress_callback=progress)
    assert result.summary['passed'] and result.summary['completed_pairs']==len(config.game_seeds)
    for name in ('policy_a_stats', 'policy_b_stats'):
        stats = result.summary[name]
        assert stats['simulations']==1024*stats['searchable_decisions']
        assert stats['early_simulations']+stats['late_simulations']==stats['simulations']
        if name=='policy_b_stats':
            assert stats['early_adapted_searchable_roots']+stats['late_adapted_searchable_roots']==0
    stats = result.summary['policy_a_stats']
    if mode!='full':
        wrong = 'late' if mode=='early' else 'early'
        assert stats[f'{wrong}_adapted_searchable_roots']==0
    print(json.dumps({'event': 'panel_complete', 'panel': label, 'score': result.summary['win_rate_score'],
                      'ci95': result.summary['paired_bootstrap']}), flush=True)


def analyze(plan):
    panels, values, seeds = {}, {}, {}
    for label in ('dev-50', 'dev-200', 'dev-500', 'head-full', 'head-early', 'head-late'):
        path = ART/f'battles/{label}.json'
        if not path.exists():
            continue
        summary = read_json(path)
        expected = plan['dev']['pairs'] if label.startswith('dev') else plan['head']['pairs' if label=='head-full' else 'stage_pairs']
        if summary['completed_pairs']!=expected or not summary['passed']:
            continue
        seeds[label], values[label] = serialized_pair_scores(summary)
        panels[label] = {'score': summary['win_rate_score'], 'ci95': summary['paired_bootstrap'],
                         'wins': summary['a_wins'], 'draws': summary['draws'], 'losses': summary['a_losses'],
                         'policy_cycle_draws': summary['policy_cycle_draws'],
                         'candidate_stats': summary['policy_a_stats'], 'control_stats': summary['policy_b_stats']}
        if label in ('head-early', 'head-late'):
            panels[label]['secondary_ci99'] = paired_bootstrap_mean(values[label], confidence=.99,
                resamples=10000, seed=plan['head']['bootstrap_seed']+7, batch_size=256).to_dict()
    comparisons = {}
    for left, right in (('head-full', 'head-early'), ('head-full', 'head-late'), ('head-late', 'head-early')):
        if left not in values or right not in values:
            continue
        n = min(len(values[left]), len(values[right]))
        assert seeds[left][:n]==seeds[right][:n]
        difference = values[left][:n]-values[right][:n]
        comparisons[f'{left}_minus_{right}'] = {
            'ci95': paired_bootstrap_mean(difference, resamples=10000, seed=plan['head']['bootstrap_seed']+8, batch_size=256).to_dict(),
            'secondary_ci99': paired_bootstrap_mean(difference, confidence=.99, resamples=10000,
                                                   seed=plan['head']['bootstrap_seed']+8, batch_size=256).to_dict()}
    result = {'status': 'complete' if len(panels)==6 else 'incomplete', 'panels': panels, 'comparisons': comparisons,
              'primary': 'HEAD full selected by DEV; 2000 pairs,95%CI compared to 50%.',
              'secondary': 'Five stage hypotheses use 99%CI (Bonferroni family .05);95%CI also reported descriptively.',
              'scope': 'Matched policy/encoder/search settings; phase interventions change trajectories. No additive phase decomposition or independent training-seed replication.'}
    write_json(ART/'battle_analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--run-search', action='store_true', help='Explicitly launch search matches; no implicit launch')
    parser.add_argument('--panel', choices=('dev-50', 'dev-200', 'dev-500', 'head-full', 'head-early', 'head-late'))
    args = parser.parse_args()
    if sum((args.prepare_only, args.analyze_only, args.run_search))!=1 or (args.panel and not args.run_search):
        parser.error('Choose exactly one of --prepare-only/--analyze-only/--run-search; --panel requires --run-search')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    TEMP.mkdir(parents=True, exist_ok=True)
    plan = prepare()
    if args.prepare_only:
        print(json.dumps({'status': 'prepared_not_started', 'dev_pairs_each': plan['dev']['pairs'],
                          'head_full_pairs': plan['head']['pairs'], 'head_stage_pairs_each': plan['head']['stage_pairs']}), flush=True)
        return
    if args.analyze_only:
        print(json.dumps(analyze(plan), ensure_ascii=False), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('This frozen battle plan requires CUDA; use a CUDA-enabled execution environment')
    if args.panel:
        run_panel(plan, args.panel)
        return
    with (TEMP/'battle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for label in ('dev-50', 'dev-200', 'dev-500', 'head-full', 'head-early', 'head-late'):
            if label=='head-full':
                select(plan)
            subprocess.run([sys.executable, str(Path(__file__)), '--run-search', '--panel', label], cwd=ROOT, check=True)
            analyze(plan)
        print(json.dumps(analyze(plan), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
