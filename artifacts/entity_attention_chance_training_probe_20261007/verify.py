"""Recompute raw pair summaries and audit finite training/data/source identities."""
from pathlib import Path
import ast
import json

import numpy as np

from splendor.alphazero_recording import read_json, write_json, sha256
from splendor.evaluation import FixedEvaluationConfig, _evaluation_summary_from_records
from experiment import ART, ROOT, OUT, RUNTIME, ARMS, INITIAL_SHA, check_source, record


if __name__=='__main__':
    spec=check_source()
    analysis=read_json(ART/'analysis.json')
    assert analysis['complete_main_panels']
    source=spec['source']
    for name,expected in {**source['source_files'],**source['additional_files']}.items():
        assert sha256(RUNTIME/'source'/name)==expected,name
    assert sha256(RUNTIME/'source/dirty.patch')==source['dirty_diff_sha256']
    amendments=spec.get('report_only_amendments',[])+spec.get('execution_only_amendments',[])
    for i,item in enumerate(amendments):
        assert sha256(Path(item['original_script_snapshot']))==item['before']['sha256']
        expected=amendments[i+1]['before']['sha256'] if i+1<len(amendments) else sha256(Path(item['after']['path']))
        assert expected==item['after']['sha256']
        excluded={'analyze'} if item in spec.get('report_only_amendments',[]) else {'eval_config'}
        def behavior(path):
            tree=ast.parse(Path(path).read_text())
            tree.body=[node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name not in excluded]
            return ast.dump(tree)
        after_path=amendments[i+1]['original_script_snapshot'] if i+1<len(amendments) else item['after']['path']
        assert behavior(item['original_script_snapshot'])==behavior(after_path)
    for amendment in spec.get('execution_only_amendments',[]):
        for item in amendment['configs']:
            snapshot=Path(item['original_config_snapshot'])
            assert sha256(snapshot)==item['before']['sha256']
            original=read_json(snapshot)
            assert original.pop('execution')['backend']=='serial'
            assert original==read_json(Path(item['after']['path']))
            assert sha256(Path(item['after']['path']))==item['after']['sha256']
    outcomes={'a_win':1.,'a_loss':0.,'draw':.5}
    verified_panels=[]
    for panel in spec['panels']+spec.get('confirmation_panels',[]):
        name=panel['id'];path=ART/f'evaluations/{name}.json'
        saved=read_json(path)
        cfg=FixedEvaluationConfig.from_dict(read_json(ART/f'configs/{name}.json'))
        assert saved['passed'] and saved['valid_games']==2*len(cfg.game_seeds)
        assert not any(saved[k] for k in ('exceptions','illegal_actions','invariant_failures','engineering_truncations'))
        records={}
        logs=Path(saved['game_log_root'])
        paths=sorted(logs.glob('pair-*.jsonl'));assert len(paths)==len(cfg.game_seeds)
        for p in paths:
            pair_id=int(p.stem.split('-')[1]);rows=[json.loads(x) for x in p.read_text().splitlines()]
            rows.sort(key=lambda x:x['game_in_pair']);assert len(rows)==2
            for seat,row in enumerate(rows):
                assert row['game_seed']==cfg.game_seeds[pair_id] and row['first_player']==seat
                assert row['policy_seeds']==[cfg.policy_seed_base+2*pair_id,cfg.policy_seed_base+2*pair_id+1]
                assert row['status'] in ('natural_terminal','policy_cycle_draw')
                assert row['error_type'] is None and row['error_detail'] is None
                for which,policy in (('policy_a_stats',cfg.policy_a),('policy_b_stats',cfg.policy_b)):
                    stats=row[which]
                    if policy.kind=='greedy_checkpoint':assert stats=={}
                    else:
                        assert stats['simulations']==1024*stats['searchable_decisions']
                        assert stats['transition_cap_hits']==0
            records[pair_id]=rows
        rebuilt=_evaluation_summary_from_records(cfg,records,output_path=path,game_log_root=logs,
                                               elapsed_seconds=saved['elapsed_seconds_this_invocation'])
        for key,value in rebuilt.items():
            if key!='updated_at':assert saved[key]==value,(name,key)
        for which,policy in (('policy_a',cfg.policy_a),('policy_b',cfg.policy_b)):
            assert saved['checkpoint_identity'][which]['sha256']==sha256(Path(policy.checkpoint))
        verified_panels.append({'id':name,'games':saved['valid_games'],'summary':record(path),
                                'raw_game_log_hashes':{str(p):sha256(p) for p in paths}})
    splits=[]
    for arm in ARMS:
        data=read_json(ART/f'dataset_{arm}.json')
        split={g['game_id']:g['split'] for g in data['games']}
        assert len(split)==200
        splits.append(split)
        for game in data['games']:assert sha256(Path(game['path']))==game['sha256']
        training=read_json(ART/f'train_{arm}.json')
        assert sha256(Path(training['batch_ledger']['path']))==training['batch_ledger']['sha256']
        ledger=np.load(training['batch_ledger']['path']);assert ledger.shape==(500,2,256,2)
        assert all(split[int(gid)]=='train' for gid in np.unique(ledger[...,0]))
        rows=[json.loads(x) for x in (ART/f'metrics_{arm}.jsonl').read_text().splitlines()]
        assert len(rows)==500
        assert [r['round_train_step'] for r in rows]==list(range(1,501))
        assert all(r['policy_batch_size']==r['value_batch_size']==256 and r['policy_max_grad_norm']==r['value_max_grad_norm']==1 for r in rows)
        assert all(r['training_mode']=='split_policy_detached_value' for r in rows)
    assert splits[0]==splits[1]==splits[2]
    output={'passed':True,'protocol':'chance-training-probe-audit-v1',
            'verified_panels':verified_panels,'total_main_evaluation_games':sum(p['games'] for p in verified_panels if not p['id'].startswith('confirm_')),
            'total_confirmation_evaluation_games':sum(p['games'] for p in verified_panels if p['id'].startswith('confirm_')),
            'collection_games':600,'training_updates':1500,'source_snapshot_verified':True,
            'same_game_id_split':True,'no_holdout_in_training':True,'initial_sha256':INITIAL_SHA,
            'script':record(Path(__file__))}
    write_json(ART/'verification.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='verified_panels'},ensure_ascii=False))
