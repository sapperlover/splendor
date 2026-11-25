"""Prepare or execute one of four frozen-source scratch comparison arms."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.offline_outcome import OutcomeOfflineConfig, prepare_inputs

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'/ART.name
SOURCE=RUNTIME/'source'
BRANCH=ROOT/'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025'
ARMS=('scalar_mc','scalar_lambda','wdl_mc','wdl_lambda')
ACTIVE_ARMS=('scalar_mc','scalar_lambda')


def config_path(arm):
    return ROOT/'configs/entity_attention_v1'/f'outcome_60k_{arm}_20261120.json'


def output_path(arm):
    return ROOT/'runs'/f'outcome_60k_{arm}_20261120'


def prepare_pool():
    manifest=RUNTIME/'pool_games.jsonl';identity=ART/'pool_identity.json'
    if identity.exists():
        record=read_json(identity)
        if sha256(manifest)!=record['manifest_sha256']:
            raise ValueError('Frozen60k manifest changed')
        return record
    state=read_json(BRANCH/'state.json')
    snapshot=Path(state['completed_rounds'][-1]['snapshot']['path'])
    games=read_json(snapshot)['games']
    if len(games)!=60000 or sorted({g['round_index'] for g in games})!=list(range(87,93)):
        raise ValueError('This preset fixes local8-13 / archive round_index87-92, not a moving window')
    if len({(g['round_index'],g['game_id']) for g in games})!=60000:
        raise ValueError('Duplicate60k game identity')
    RUNTIME.mkdir(parents=True,exist_ok=True)
    manifest.write_text(''.join(json.dumps(g,ensure_ascii=False,sort_keys=True)+'\n'
                                for g in sorted(games,key=lambda g:(g['round_index'],g['game_id']))))
    record=dict(status='frozen',games=60000,source_snapshot=str(snapshot),
        source_snapshot_sha256=sha256(snapshot),source_rounds=list(range(87,93)),
        local_online_rounds=[8,13],manifest=str(manifest),manifest_sha256=sha256(manifest),
        splits={s:dict(games=sum(g['split']==s for g in games),
            positions=sum(g['decision_count'] for g in games if g['split']==s),
            policy_positions=sum(g['policy_samples'] for g in games if g['split']==s))
            for s in ('train','validation')},
        split_rule='original game-level splits, no reshuffle; training targets never use held-out games')
    write_json(identity,record)
    return record


def prepare(arm):
    pool=prepare_pool();raw=read_json(config_path(arm))
    common=[]
    for name in ARMS:
        other=read_json(config_path(name))
        if other.pop('outcome_objective')!=name:
            raise ValueError('Preset filename and outcome objective disagree')
        for key in ('outcome_teacher_checkpoint','outcome_teacher_selection','outcome_teacher_sha256','outcome_lambda_source'):
            other.pop(key,None)
        common.append(other)
    if not all(c==common[0] for c in common):
        raise ValueError('The four presets may differ only in outcome objective and fixed teacher source')
    if raw['manifest_sha256']!=pool['manifest_sha256'] or raw['manifest']!=pool['manifest']:
        raise ValueError('Preset and frozen60k identity differ')
    config=OutcomeOfflineConfig.from_dict(raw)
    capacity=read_json(ART/'cache_capacity.json')
    if (capacity['manifest_sha256']!=pool['manifest_sha256'] or
            capacity['arena_aligned_upper_bound_bytes']>config.cache_mib*1024**2):
        raise ValueError('Audited full training arena does not fit this preset')
    opponent=Path(read_json(ART/'evaluation_templates.json')['greedy']['policy_b']['checkpoint'])
    opponent_identity=dict(path=str(opponent),sha256=sha256(opponent))
    frozen_opponent=ART/'opponent.json'
    if frozen_opponent.exists() and read_json(frozen_opponent)!=opponent_identity:
        raise ValueError('Frozen raw R37 opponent changed')
    if not frozen_opponent.exists():write_json(frozen_opponent,opponent_identity)
    prepared=prepare_inputs(config,output_path(arm))
    write_json(ART/f'plan_{arm}.json',dict(status='prepared_not_started',arm=arm,
        config=str(config_path(arm)),config_sha256=sha256(config_path(arm)),
        output=str(output_path(arm)),pool_identity_sha256=sha256(ART/'pool_identity.json'),
        steps_target=prepared[-1]['steps_target'],validation_steps=prepared[-1]['validation_steps'],
        lambda_target_policy=prepared[-1]['lambda_target_policy'],cache_mib=config.cache_mib,
        outcome_teacher_checkpoint=config.outcome_teacher_checkpoint,
        outcome_teacher_selection=config.outcome_teacher_selection,
        outcome_lambda_source=config.outcome_lambda_source,
        execution_plan='active' if arm in ACTIVE_ARMS else 'deferred',
        training_workers=config.training_workers,initialization='random; fresh AdamW; no checkpoint inheritance',
        formal_evaluation='explicit evaluate phase: greedy1000pair then both S128N1 searchA500pair'))
    if not (ART/'source.json').exists():
        record=source_identity(ROOT,SOURCE)
        for p in [*ROOT.glob('src/splendor/_native_search*.so'),*ROOT.glob('data/*'),
                  Path(__file__),ART/'audit_cache.py',ART/'evaluation_templates.json',*(config_path(a) for a in ARMS),
                  *(ROOT/'scripts'/f'run_outcome_60k_{a}_20261120.sh' for a in ARMS)]:
            if p.is_file():
                target=SOURCE/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,target)
        record['snapshot_files']={str(p.relative_to(SOURCE)):sha256(p)
            for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        write_json(ART/'source.json',record)
    for name,expected in read_json(ART/'source.json')['snapshot_files'].items():
        if sha256(SOURCE/name)!=expected:
            raise ValueError(f'Frozen execution source changed: {name}')
    if sha256(SOURCE/config_path(arm).relative_to(ROOT))!=sha256(config_path(arm)):
        raise ValueError('Working config differs from frozen execution config')
    for p in (Path(__file__),ART/'evaluation_templates.json',
              ROOT/'scripts'/f'run_outcome_60k_{arm}_20261120.sh'):
        if sha256(SOURCE/p.relative_to(ROOT))!=sha256(p):
            raise ValueError(f'Execution entry/template differs from frozen source: {p}')
    return prepared[-1]


def evaluation_configs(arm, selection):
    if sha256(Path(selection['checkpoint']))!=selection['checkpoint_sha256']:
        raise ValueError('Selected evaluation checkpoint changed')
    templates=read_json(ART/'evaluation_templates.json')
    result={}
    for name,template in templates.items():
        raw=copy.deepcopy(template)
        raw['policy_a'].update(checkpoint=selection['checkpoint'],name=arm)
        raw['metadata']=dict(experiment=ART.name,arm=arm,
            selected_step=selection['selected_step'],checkpoint_sha256=selection['checkpoint_sha256'])
        result[name]=raw
    return result


def report():
    rows=[]
    for arm in ARMS:
        output=output_path(arm);status=read_json(output/'progress.json') if (output/'progress.json').exists() else {}
        row=dict(arm=arm,status=status.get('status','prepared_not_started') if arm in ACTIVE_ARMS else 'deferred',
            step=status.get('step',0),execution_plan='active' if arm in ACTIVE_ARMS else 'deferred')
        labels=output/'lambda_targets/archived_search/manifest.json'
        if labels.exists():
            manifest=read_json(labels)
            row['archived_search_targets']={k:manifest[k] for k in ('counts','target_coverage','root_coverage','root_budget_counts')}
        if (output/'selection.json').exists():
            s=read_json(output/'selection.json');row.update(selected_step=s['selected_step'],
                selected_checkpoint=s['checkpoint'],checkpoint_sha256=s['checkpoint_sha256'],
                selection_objective=s['selection_objective'])
            selected=read_json(output/'full_validation'/f'step-{s["selected_step"]:09d}.json')
            row['validation']={k:selected['metrics'][k] for k in ('policy_kl','value_loss','wdl_nll','wdl_brier','wdl_ece')
                               if k in selected['metrics']}
        row['evaluations']={}
        for name in ('greedy','search_S128N1'):
            p=ART/'panels'/f'{arm}_{name}.json'
            if p.exists():
                o=read_json(p)
                row['evaluations'][name]={k:o[k] for k in ('completed_pairs','passed','win_rate_score','paired_bootstrap') if k in o}
        rows.append(row)
    write_json(ART/'summary.json',dict(status='complete' if all(r['status']=='complete' for r in rows if r['arm'] in ACTIVE_ARMS) else 'prepared_or_incomplete',
        active_arms=list(ACTIVE_ARMS),deferred_arms=[a for a in ARMS if a not in ACTIVE_ARMS],
        arms=rows,scope='Single training seed, same frozen pool; validation comparison is not a playing-strength conclusion.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=ARMS)
    parser.add_argument('phase',choices=('prepare','targets','train','select','evaluate'),nargs='?',default='train')
    parser.add_argument('--max-steps-this-session',type=int)
    args=parser.parse_args()
    if args.max_steps_this_session is not None and args.phase!='train':
        parser.error('--max-steps-this-session applies only to train')
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(args.arm)
        env=os.environ|{'PYTHONPATH':str(SOURCE/'src')}
        command=[sys.executable,'-u','-m','splendor.offline_outcome',
            '--config',str(config_path(args.arm)),'--output',str(output_path(args.arm))]
        if args.phase in ('targets','train','select'):
            if args.phase=='select':command.append('--select-only')
            if args.phase=='targets':command.append('--generate-targets-only')
            if args.max_steps_this_session is not None:
                command += ['--max-steps-this-session',str(args.max_steps_this_session)]
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            if args.phase=='train' and read_json(output_path(args.arm)/'progress.json')['status']=='complete':
                subprocess.run(command[:]+['--select-only'],cwd=ROOT,env=env,check=True)
        if args.phase=='evaluate':
            if read_json(output_path(args.arm)/'progress.json')['status']!='complete':
                raise ValueError('Evaluation requires complete training and fixed selection')
            selection=read_json(output_path(args.arm)/'selection.json')
            for name,raw in evaluation_configs(args.arm,selection).items():
                p=ART/'configs'/f'{args.arm}_{name}.json'
                if p.exists() and read_json(p)!=raw:
                    raise ValueError('Frozen evaluation configuration changed')
                write_json(p,raw)
                result=ART/'panels'/f'{args.arm}_{name}.json'
                subprocess.run([sys.executable,'-u','-m','splendor.evaluation','--config',str(p),
                    '--output',str(result),'--resume','--progress-every','100'],cwd=ROOT,env=env,check=True)
                if not read_json(result)['passed']:
                    raise ValueError('Evaluation did not pass; stopping the chain')
        report()


if __name__=='__main__':main()
