from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from test_offline_mixed_budget import mixed
from splendor.actor_critic import build_actor_critic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json
from splendor.engine import GameEngine
from splendor.features import EncodedStateBatch, collate_encoded_decisions, encode_decision
from splendor.offline_outcome import OutcomeOfflineConfig, initialize, prepare_inputs, run
from splendor.value_outcome import (archived_search_lambda_targets, scalar_lambda_targets,
                                    terminal_one_hot, vector_lambda_targets)


@pytest.fixture(autouse=True, scope='module')
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_vector_returns_keep_player_information_and_ignore_payment_clock():
    players=np.array([0,0,1,0,1,1]);main=np.array([1,0,1,1,1,0],dtype=bool)
    terminal=np.array([1,1,-1,1,-1,-1])
    p=np.array([[.2,.3,.5],[np.nan]*3,[.1,.8,.1],[.4,.2,.4],[.3,.1,.6],[np.nan]*3])
    result=vector_lambda_targets(p,players,main,terminal,.5)
    np.testing.assert_allclose(result[[0,1]],[[.7,.1,.2]]*2,atol=1e-7)
    np.testing.assert_allclose(result[2],[.15,.05,.8],atol=1e-7)
    np.testing.assert_array_equal(result[[3,4,5]],terminal_one_hot(terminal[[3,4,5]]))
    np.testing.assert_allclose(result.sum(1),1)
    assert (result>=0).all()
    np.testing.assert_array_equal(vector_lambda_targets(p,players,main,terminal,1),terminal_one_hot(terminal))
    np.testing.assert_allclose(vector_lambda_targets(p,players,main,terminal,0)[0],p[3])
    with pytest.raises(ValueError,match='probability'):
        vector_lambda_targets(p*2,players,main,terminal,.8)
    with pytest.raises(ValueError,match='exact terminal'):
        terminal_one_hot([.25])
    scalar=scalar_lambda_targets(p[:,0]-p[:,2],players,main,terminal,.5)
    np.testing.assert_allclose(scalar,result[:,0]-result[:,2],atol=1e-7)
    np.testing.assert_array_equal(scalar_lambda_targets(p[:,0]-p[:,2],players,main,terminal,1),terminal)
    with pytest.raises(ValueError,match='finite'):
        scalar_lambda_targets(np.full(len(players),np.nan),players,main,terminal,.8)
    assert scalar_lambda_targets([0,1.2],[0,0],[True,True],[1,1],0)[0]==1


def test_wdl_search_api_and_shared_initialization_are_preserved():
    config=dict(architecture='entity_attention_v1',hidden=16,heads=2,layers=1,ffn_hidden=32,
                market_relations=True,value_readout='global_players_market')
    torch.manual_seed(5);scalar=build_actor_critic(config)
    torch.manual_seed(5);wdl=build_actor_critic(config|{'architecture':'entity_attention_wdl_v1'})
    for name,v in scalar.state_dict().items():
        if not name.startswith('critic.'):
            torch.testing.assert_close(v,wdl.state_dict()[name],rtol=0,atol=0)
    batch=collate_encoded_decisions([encode_decision(GameEngine.new(3).decision())])
    wdl.eval()
    logits,value,outcomes=wdl.forward_with_wdl(batch)
    p=outcomes.softmax(-1)
    torch.testing.assert_close(value,p[:,0]-p[:,2],rtol=0,atol=0)
    torch.testing.assert_close(wdl(batch)[0],logits)
    torch.testing.assert_close(wdl(batch)[1],value)
    state=EncodedStateBatch(**{k:getattr(batch,k) for k in EncodedStateBatch.__dataclass_fields__})
    torch.testing.assert_close(wdl.value_from_state(state),value)
    assert value.shape==(1,) and -1<=value.item()<=1


def config_for(mixed,objective,workers=0,teacher=None):
    raw=asdict(mixed)
    raw.update(outcome_objective=objective,
        outcome_teacher_checkpoint=str(teacher or '/pending-teacher.pt') if objective.endswith('_lambda') else None,
        target_batch_size=4,draw_exposures=4.,training_workers=workers,
        model_config=dict(architecture='entity_attention_v1',hidden=8,heads=2,layers=1,ffn_hidden=16,
                          market_relations=True,value_readout='global_players_market'))
    return OutcomeOfflineConfig.from_dict(raw)


def save_teacher(mixed,tmp_path,objective='wdl_mc'):
    cfg=config_for(mixed,objective)
    cfg=replace(cfg,initialization_seed=cfg.initialization_seed+91)
    learner=initialize(cfg,{'digest':'external-fixed-teacher'},8)
    path=tmp_path/'teacher.pt';learner.save(path)
    return path


def assert_same_checkpoint(a,b):
    a=load_alphazero_checkpoint(a);b=load_alphazero_checkpoint(b)
    assert a.sampling_rng_state==b.sampling_rng_state
    assert a.round_train_step==b.round_train_step
    for key in a.model_state_dict:
        torch.testing.assert_close(a.model_state_dict[key],b.model_state_dict[key],rtol=0,atol=0)
    for pid,state in a.optimizer_state_dict['state'].items():
        for key,v in state.items():
            torch.testing.assert_close(v,b.optimizer_state_dict['state'][pid][key],rtol=0,atol=0)


@pytest.mark.parametrize('objective',['scalar_lambda','wdl_lambda'])
def test_fixed_lambda_exact_resume_and_immutable_labels(mixed,tmp_path,monkeypatch,objective):
    import splendor.offline_outcome as entry
    monkeypatch.setattr(entry,'source_identity',lambda *_:{'test':True})
    teacher=save_teacher(mixed,tmp_path,'scalar_mc' if objective=='scalar_lambda' else 'wdl_mc')
    config=config_for(mixed,objective,teacher=teacher)
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    # Explicit label generation performs inference only, before any student exists.
    original=entry.initialize
    monkeypatch.setattr(entry,'initialize',lambda *_:pytest.fail('label generation initialized a student'))
    rng=torch.get_rng_state().clone()
    run(config,resumed,generate_targets_only=True)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert not (resumed/'checkpoints').exists()
    manifest_path=resumed/'lambda_targets/fixed/manifest.json'
    frozen_manifest=manifest_path.read_bytes()
    monkeypatch.setattr(entry,'initialize',original)
    run(config,full)
    run(config,resumed,max_steps_this_session=3)
    run(config,resumed,max_steps_this_session=1)
    run(config,resumed)
    assert_same_checkpoint(full/'checkpoints/final.pt',resumed/'checkpoints/final.pt')
    assert manifest_path.read_bytes()==frozen_manifest
    manifests=list((full/'lambda_targets').glob('*/manifest.json'))
    assert len(manifests)==1
    assert all(read_json(p)['identity']['train_only'] for p in manifests)
    for p in manifests:
        targets=np.load(p.parent/'targets.npy')
        assert targets.shape==((8,3) if objective=='wdl_lambda' else (8,))
        if objective=='wdl_lambda':
            np.testing.assert_allclose(targets.sum(1),1,atol=1e-6)
        else:
            assert np.isfinite(targets).all() and (np.abs(targets)<=1).all()
    point=read_json(full/'full_validation/step-000000008.json')
    assert point['metrics']['positions']==8
    if objective=='wdl_lambda':
        assert point['metrics']['wdl_nll']>=0 and point['metrics']['wdl_brier']>=0
    else:
        assert 'wdl_nll' not in point['metrics']
    assert not list(full.rglob('*.bin')) and not list(full.rglob('*.steps.jsonl'))
    from splendor.offline_scratch import select_checkpoint
    selected=select_checkpoint(config,full,input_preparer=prepare_inputs)
    assert selected['selected_step'] in [2,4,6,8]
    rows=[json.loads(l) for l in (full/'training_metrics.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='train_step']
    assert [r['outcome_teacher_step'] for r in updates]==[0]*8
    assert all(r['value_objective']==objective for r in updates)
    metric='wdl_train_ce' if objective=='wdl_lambda' else 'lambda_train_mse'
    assert all(metric in r for r in updates)
    # Students start from the declared random initialization, not teacher weights.
    expected_student=initialize(config,{},8)
    initial=load_alphazero_checkpoint(full/'checkpoints/initial.pt')
    for name,value in expected_student.model.state_dict().items():
        torch.testing.assert_close(value,initial.model_state_dict[name],rtol=0,atol=0)
    # Labels are immutable and checked on resume/use, rather than silently rebuilt.
    from splendor.offline_outcome import lambda_targets
    index,*_=prepare_inputs(config,full)
    # Interrupt-before-publish recovery recreates only an owned partial stage,
    # without changing the student's global torch random stream.
    import shutil
    teacher_identity=read_json(full/'fixed_teacher.json')
    stage=full/'lambda_targets/fixed'
    expected=np.load(stage/'targets.npy').copy()
    stage_identity=read_json(stage/'manifest.json')['identity']
    shutil.rmtree(stage)
    partial=stage.with_name(stage.name+'.partial');partial.mkdir()
    (partial/'identity.json').write_text(json.dumps(stage_identity))
    (partial/'targets.npy').write_bytes(b'unpublished partial')
    rng=torch.get_rng_state().clone()
    rebuilt,_=lambda_targets(config,full,teacher_identity,index)
    np.testing.assert_array_equal(rebuilt,expected)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    p=sorted((full/'lambda_targets').glob('*/targets.npy'))[0]
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='identity changed'):
        lambda_targets(config,full,teacher_identity,index)


@pytest.mark.parametrize('objective',['scalar_mc','scalar_lambda','wdl_mc','wdl_lambda'])
def test_worker_pool_and_serial_updates_are_identical(mixed,tmp_path,monkeypatch,objective):
    import splendor.offline_outcome as entry
    monkeypatch.setattr(entry,'source_identity',lambda *_:{'test':True})
    teacher=save_teacher(mixed,tmp_path) if objective.endswith('_lambda') else None
    config=config_for(mixed,objective,teacher=teacher)
    serial=tmp_path/'serial';shared=tmp_path/'shared'
    run(config,serial)
    if objective.endswith('_lambda'):
        run(replace(config,training_workers=4),shared,max_steps_this_session=3)
    run(replace(config,training_workers=4),shared)
    assert_same_checkpoint(serial/'checkpoints/final.pt',shared/'checkpoints/final.pt')
    assert read_json(shared/'prefetch_progress.json')['workers']==4
    assert read_json(shared/'prefetch_progress.json')['arena_bytes']<config.cache_mib*1024**2


def test_prepare_does_not_allocate_model_or_start_training(mixed,tmp_path,monkeypatch):
    import splendor.offline_outcome as entry
    monkeypatch.setattr(entry,'initialize',lambda *_:pytest.fail('prepare initialized a model'))
    cfg=config_for(mixed,'wdl_lambda');out=tmp_path/'prepare'
    plan=run(cfg,out,prepare_only=True)
    assert plan['steps_target']==8 and 'stage_boundaries' not in plan
    assert 'immutable reuse' in plan['lambda_target_policy']
    assert not (out/'checkpoints').exists()
    with pytest.raises(ValueError,match='exactly one'):
        replace(cfg,outcome_teacher_checkpoint=None)


def test_lambda_one_matches_scalar_mc_updates(mixed,tmp_path,monkeypatch):
    import splendor.offline_outcome as entry
    monkeypatch.setattr(entry,'source_identity',lambda *_:{'test':True})
    teacher=save_teacher(mixed,tmp_path,'scalar_mc')
    cfg=replace(config_for(mixed,'scalar_lambda',teacher=teacher),outcome_lambda=1.)
    soft=tmp_path/'lambda_one';mc=tmp_path/'mc'
    run(cfg,soft)
    run(replace(cfg,outcome_objective='scalar_mc'),mc)
    assert_same_checkpoint(soft/'checkpoints/final.pt',mc/'checkpoints/final.pt')
    updates=[json.loads(s) for s in (soft/'training_metrics.jsonl').read_text().splitlines()
             if json.loads(s)['event']=='train_step']
    assert all(r['lambda_train_mse']==r['value_loss'] for r in updates)


def test_fixed_teacher_selection_and_changed_identity_fail_before_cache(mixed,tmp_path,monkeypatch):
    import splendor.offline_outcome as entry
    from splendor.alphazero_recording import sha256,write_json
    teacher=save_teacher(mixed,tmp_path)
    selection=tmp_path/'selection.json'
    selected=dict(checkpoint=str(teacher),checkpoint_sha256=sha256(teacher))
    write_json(selection,selected)
    cfg=replace(config_for(mixed,'wdl_lambda'),outcome_teacher_checkpoint=None,
                outcome_teacher_selection=str(selection),training_workers=4)
    out=tmp_path/'fit';prepare_inputs(cfg,out)
    first=entry.fixed_teacher(cfg,out)
    assert first['checkpoint_sha256']==selected['checkpoint_sha256']
    monkeypatch.setattr(entry,'SharedReadonlyPrefetch',lambda *_a,**_k:pytest.fail('allocated full cache'))
    selected['checkpoint_sha256']='0'*64;write_json(selection,selected)
    with pytest.raises(ValueError,match='SHA changed'):
        run(cfg,out)
    assert not (out/'checkpoints').exists()
    # A missing WDL dependency also fails before cache/model initialization.
    selection.unlink()
    with pytest.raises(FileNotFoundError):
        run(cfg,out)
    # Scalar teachers are accepted for scalar lambda, never fabricated into WDL.
    scalar=save_teacher(mixed,tmp_path,'scalar_mc')
    with pytest.raises(ValueError,match='require a WDL teacher'):
        entry.fixed_teacher(replace(cfg,outcome_teacher_selection=None,
                                    outcome_teacher_checkpoint=str(scalar)),out)


def test_wdl_checkpoint_runs_native_search_and_compiled_tensor_core(mixed,tmp_path,monkeypatch):
    from splendor.native_policy import NativeChancePUCTSearchPolicy
    from splendor.mcts_policy import MCTSConfig
    from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
    from splendor.search_inference import FrozenSearchInference,InferenceConfig
    cfg=config_for(mixed,'wdl_mc')
    learner=initialize(cfg,{'digest':'search-test'},8)
    path=tmp_path/'wdl.pt';learner.save(path)
    model=load_frozen_actor_critic_checkpoint(path)
    engine=GameEngine.new(42);request=engine.decision()
    policy=NativeChancePUCTSearchPolicy(model,config=MCTSConfig(simulations=8,determinizations=1,seed=71))
    result=policy.search(request)
    assert len(result.policy_target)==len(encode_decision(request).candidate_features)
    assert np.isfinite(result.policy_target).all() and result.simulations==8
    original=torch.compile
    monkeypatch.setattr(torch,'compile',lambda function,**kw:
        original(function,backend='eager',dynamic=kw.get('dynamic')))
    encoded=[encode_decision(request)]
    reference=FrozenSearchInference(model,device='cpu',config=InferenceConfig(mode='eager'))
    compiled=FrozenSearchInference(model,device='cpu',config=InferenceConfig(mode='compiled',
        candidate_buckets=(32,64),batch_buckets=(1,),compile_failure_policy='fail_fast'))
    a=reference.policy_value_encoded(encoded)[0];b=compiled.policy_value_encoded(encoded)[0]
    np.testing.assert_allclose(a[0],b[0],atol=1e-6)
    assert abs(a[1]-b[1])<1e-6


def test_four_presets_and_optional_evaluation_keep_shared_protocol(tmp_path):
    import importlib.util
    from pathlib import Path
    from splendor.alphazero_recording import sha256
    from splendor.evaluation import FixedEvaluationConfig
    root=Path(__file__).resolve().parents[1]
    runner=root/'artifacts/value_outcome_60k_20261120/run.py'
    spec=importlib.util.spec_from_file_location('outcome_comparison_entry',runner)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    configurations=[]
    for arm in entry.ARMS:
        raw=read_json(entry.config_path(arm));assert raw.pop('outcome_objective')==arm
        if arm=='scalar_lambda':
            assert raw['outcome_lambda_source']=='archive_search' and raw['outcome_teacher_checkpoint'] is None
        for key in ('outcome_teacher_checkpoint','outcome_teacher_selection','outcome_teacher_sha256','outcome_lambda_source'):
            raw.pop(key,None)
        configurations.append(raw)
    assert len(configurations)==4 and all(c==configurations[0] for c in configurations)
    assert configurations[0]['training_workers']==4 and configurations[0]['cache_mib']==10240
    assert entry.ACTIVE_ARMS==('scalar_mc','scalar_lambda')
    checkpoint=tmp_path/'selected.pt';checkpoint.write_bytes(b'identity-only engineering fixture')
    configs=entry.evaluation_configs('wdl_mc',dict(checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),selected_step=5000))
    greedy=FixedEvaluationConfig.from_dict(configs['greedy'])
    search=FixedEvaluationConfig.from_dict(configs['search_S128N1'])
    assert len(greedy.game_seeds)==1000 and len(search.game_seeds)==500
    assert tuple(greedy.game_seeds[:500])==tuple(search.game_seeds)
    assert configs['search_S128N1']['policy_a']['search_config']['simulations']==128
    assert configs['search_S128N1']['execution']['total_active_games']==1000


def test_archived_roots_use_equal_worlds_own_next_main_and_missing_masks():
    from splendor.features import GLOBAL_FEATURE_NAMES as G
    from splendor.continuation_auxiliary import archived_root_values
    players=np.array([0,0,1,0,1,1]);main=np.array([1,0,1,1,1,0],bool)
    global_features=np.zeros((6,len(G)),np.float32)
    global_features[main,G.index('phase_main_action')]=1
    visits=np.array([[1]*6,[9]*6])
    q=np.array([[.1,0,-.8,.3,.1,0],[.9,0,-.2,.9,.7,0]])
    arrays=dict(global_features=global_features,player_id=players,value_target=np.array([1,1,-1,1,-1,-1]),
                candidate_offsets=np.arange(7),valid_world_count=np.full(6,2),
                per_world_root_visit_counts=visits,per_world_root_value_sums=q*visits)
    values,available=archived_root_values(arrays)
    assert values[3]==pytest.approx(.6)  # Equal world mean, not visit-weighted .84.
    targets,mask,_=archived_search_lambda_targets(arrays,{'status':'natural_terminal'},.8)
    np.testing.assert_allclose(targets,[.92,.92,-.72,1,-1,-1],atol=1e-7)
    assert mask.all() and available.tolist()==main.tolist()
    # A missing own successor root masks the earlier target; the other player is unaffected.
    arrays['valid_world_count'][3]=0
    _,mask,_=archived_search_lambda_targets(arrays,{'status':'natural_terminal'},.8)
    assert mask.tolist()==[False,False,True,True,True,True]
    labels,mask,_=archived_search_lambda_targets(arrays,{'status':'natural_terminal'},1.)
    assert mask.all();np.testing.assert_array_equal(labels,arrays['value_target'])
    # Flipping seats preserves actor-relative roots and labels, without using the opponent's private view.
    arrays['player_id']=1-players
    labels2,mask2,_=archived_search_lambda_targets(arrays,{'status':'natural_terminal'},1.)
    np.testing.assert_array_equal(labels2,labels);np.testing.assert_array_equal(mask2,mask)
    chain=dict(global_features=np.tile(global_features[0],(3,1)),player_id=np.zeros(3,int),
        value_target=np.ones(3,int),candidate_offsets=np.arange(4),valid_world_count=np.array([2,2,0]),
        per_world_root_visit_counts=np.ones((2,3),int),per_world_root_value_sums=np.full((2,3),.4))
    assert archived_search_lambda_targets(chain,{'status':'natural_terminal'},.8)[1].tolist()==[False,False,True]
    assert archived_search_lambda_targets(chain,{'status':'natural_terminal'},0.)[1].tolist()==[True,False,True]
    arrays['per_world_root_value_sums'][0,4]=np.nan
    with pytest.raises(ValueError,match='outside'):
        archived_search_lambda_targets(arrays,{'status':'natural_terminal'},.8)


@pytest.fixture
def archived_config(mixed):
    """Add raw search statistics to synthetic games, without modifying their feature dictionaries."""
    from pathlib import Path
    from splendor.alphazero_recording import sha256
    manifest=Path(mixed.manifest);rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    for row in rows:
        path=Path(row['path'])
        with np.load(path) as archive:arrays={k:archive[k] for k in archive.files}
        metadata=json.loads(arrays['metadata_json'].tobytes());metadata['status']='natural_terminal'
        arrays['metadata_json']=np.frombuffer(json.dumps(metadata).encode(),np.uint8)
        total=int(arrays['candidate_offsets'][-1]);visits=np.zeros((2,total),np.int32)
        sums=np.zeros((2,total),np.float64)
        for i in range(4):
            start=int(arrays['candidate_offsets'][i]);visits[:,start]=[1,3];sums[:,start]=[.2,1.8]
        arrays.update(per_world_root_visit_counts=visits,per_world_root_value_sums=sums,
                      valid_world_count=np.array([2,0,2,2]),simulations=np.array([4,0,4,4]))
        np.savez_compressed(path,**arrays)
        stat=path.stat();row.update(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    base=replace(mixed,manifest_sha256=sha256(manifest))
    return replace(config_for(base,'scalar_lambda'),outcome_teacher_checkpoint=None,outcome_lambda_source='archive_search')


def test_archived_generation_has_no_model_and_training_resumes_with_same_masks(archived_config,tmp_path,monkeypatch):
    import splendor.offline_outcome as entry
    monkeypatch.setattr(entry,'source_identity',lambda *_:{'test':True})
    cfg=archived_config;full=tmp_path/'full';resumed=tmp_path/'resumed'
    original=entry.initialize
    monkeypatch.setattr(entry,'initialize',lambda *_:pytest.fail('label generation initialized student'))
    monkeypatch.setattr(entry,'load_alphazero_checkpoint',lambda *_:pytest.fail('archived labels loaded a teacher'))
    rng=torch.get_rng_state().clone()
    run(cfg,resumed,generate_targets_only=True)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert not (resumed/'checkpoints').exists() and not (resumed/'fixed_teacher.json').exists()
    folder=resumed/'lambda_targets/archived_search'
    manifest=read_json(folder/'manifest.json')
    assert manifest['identity']['train_only'] and manifest['counts']['positions']==8
    assert manifest['target_coverage']==.75
    frozen=(folder/'manifest.json').read_bytes()
    monkeypatch.setattr(entry,'initialize',original)
    run(cfg,full)
    run(replace(cfg,training_workers=4),resumed,max_steps_this_session=3)
    run(replace(cfg,training_workers=4),resumed)
    assert_same_checkpoint(full/'checkpoints/final.pt',resumed/'checkpoints/final.pt')
    assert (folder/'manifest.json').read_bytes()==frozen
    rows=[json.loads(s) for s in (full/'training_metrics.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='train_step']
    assert all(r['outcome_target_source']=='archive_search' and r['outcome_teacher_step'] is None for r in updates)
    assert all(r['value_weight_sum']==4 and r['outcome_train_weight_sum']<=4 for r in updates)
    assert read_json(full/'full_validation/step-000000008.json')['metrics']['value_weight_sum']==8
    # Only an unpublished partial with this exact identity may be regenerated.
    import shutil
    stage=full/'lambda_targets/archived_search'
    expected=np.load(stage/'targets.npy').copy();expected_mask=np.load(stage/'mask.npy').copy()
    identity=read_json(stage/'manifest.json')['identity'];shutil.rmtree(stage)
    partial=stage.with_name(stage.name+'.partial');partial.mkdir()
    (partial/'identity.json').write_text(json.dumps(identity));(partial/'targets.npy').write_bytes(b'partial')
    index,*_=prepare_inputs(cfg,full)
    rebuilt,_,rebuilt_mask=entry.archive_lambda_targets(cfg,full,index)
    np.testing.assert_array_equal(rebuilt,expected);np.testing.assert_array_equal(rebuilt_mask,expected_mask)
    # Corruption is detected for the mask as well as the target values.
    mask_path=folder/'mask.npy';mask_path.write_bytes(mask_path.read_bytes()+b'corrupt')
    index,*_=prepare_inputs(cfg,resumed)
    with pytest.raises(ValueError,match='identity changed'):
        entry.archive_lambda_targets(cfg,resumed,index)


def test_archived_all_masked_batch_is_policy_only_and_true_mse_is_preserved(archived_config,tmp_path):
    from splendor.offline_mixed_budget import make_batch_drawer
    from splendor.selfplay_data import collate_selfplay_samples
    from splendor.rng import DeterministicRng
    cfg=archived_config;index,pool,identity,_,plan=prepare_inputs(cfg,tmp_path/'fit')
    learner=initialize(cfg,identity,plan['steps_target'])
    learner.outcome_targets=np.ones(8,np.float32);learner.outcome_target_mask=np.zeros(8,bool)
    learner.outcome_offsets={(0,0):0,(1,0):4}
    draw=make_batch_drawer(index,pool,4)(DeterministicRng(92))
    row=learner.train_step(collate_selfplay_samples(draw.samples),references=draw.references,
                           value_batch=collate_selfplay_samples(draw.value_samples))
    assert row['outcome_train_coverage']==0 and row['lambda_train_mse'] is None
    assert row['total_loss']==pytest.approx(row['policy_loss'])
    assert np.isfinite(row['value_loss']) and row['value_weight_sum']==4


def test_archived_lambda_one_matches_mc_without_mask_loss(archived_config,tmp_path,monkeypatch):
    import splendor.offline_outcome as entry
    monkeypatch.setattr(entry,'source_identity',lambda *_:{'test':True})
    cfg=replace(archived_config,outcome_lambda=1.)
    full=tmp_path/'soft';mc=tmp_path/'mc'
    run(cfg,full)
    run(replace(cfg,outcome_objective='scalar_mc',outcome_lambda_source='network'),mc)
    assert_same_checkpoint(full/'checkpoints/final.pt',mc/'checkpoints/final.pt')


@pytest.mark.parametrize('objective',['scalar_mc','scalar_lambda'])
@pytest.mark.parametrize('crash_stage',['before_terminal_validation','before_final','before_complete'])
def test_terminal_checkpoint_recovery_only_finishes_committed_training(
        archived_config,tmp_path,monkeypatch,objective,crash_stage):
    import splendor.offline_outcome as entry
    from splendor.alphazero_recording import sha256
    from splendor.offline_scratch import select_checkpoint
    monkeypatch.setattr(entry,'source_identity',lambda *_:{'test':True})
    cfg=archived_config if objective=='scalar_lambda' else replace(
        archived_config,outcome_objective='scalar_mc',outcome_lambda_source='network')
    reference=tmp_path/'reference';output=tmp_path/'crashed'
    run(cfg,reference)
    original_evaluate=entry.evaluate;original_copy=entry.shutil.copyfile;original_write=entry.write_json

    class InterruptedFinalization(RuntimeError):pass

    def evaluate(current,index):
        if crash_stage=='before_terminal_validation' and current.round_train_step==8:
            raise InterruptedFinalization()
        return original_evaluate(current,index)

    def copy(source,destination,*args,**kwargs):
        if crash_stage=='before_final' and destination.name in ('final.pt','final.pt.tmp'):
            destination.write_bytes(b'interrupted partial final publication')
            raise InterruptedFinalization()
        return original_copy(source,destination,*args,**kwargs)

    def write(path,value):
        if crash_stage=='before_complete' and path.name=='progress.json' and value.get('status')=='complete':
            raise InterruptedFinalization()
        return original_write(path,value)

    with monkeypatch.context() as patch:
        patch.setattr(entry,'evaluate',evaluate);patch.setattr(entry.shutil,'copyfile',copy)
        patch.setattr(entry,'write_json',write)
        with pytest.raises(InterruptedFinalization):run(cfg,output)
    checkpoint=output/'checkpoints/step-000000008.pt'
    assert checkpoint.exists() and read_json(output/'progress.json')['status']=='incomplete'
    metrics=(output/'training_metrics.jsonl').read_bytes()
    checkpoint_sha=sha256(checkpoint)
    if crash_stage=='before_final':
        assert not (output/'checkpoints/final.pt').exists()
    with monkeypatch.context() as patch:
        patch.setattr(entry,'run_training_round',lambda *_a,**_k:pytest.fail('trained at zero remaining steps'))
        patch.setattr(entry,'SharedReadonlyPrefetch',lambda *_a,**_k:pytest.fail('warmed completed training pool'))
        patch.setattr(entry,'archive_lambda_targets',lambda *_a,**_k:pytest.fail('regenerated completed labels'))
        patch.setattr(entry,'initialize',lambda *_a,**_k:pytest.fail('initialized a new student'))
        if crash_stage!='before_terminal_validation':
            patch.setattr(entry,'evaluate',lambda *_a,**_k:pytest.fail('repeated committed validation'))
            patch.setattr(entry.OutcomeLearner,'resume',lambda *_a,**_k:pytest.fail('allocated completed learner'))
        assert run(replace(cfg,training_workers=4),output)=={'status':'complete','step':8}
    assert sha256(output/'checkpoints/final.pt')==checkpoint_sha
    assert not (output/'checkpoints/final.pt.tmp').exists()
    assert (output/'training_metrics.jsonl').read_bytes()==metrics
    assert read_json(output/'progress.json')['status']=='complete'
    assert_same_checkpoint(reference/'checkpoints/final.pt',output/'checkpoints/final.pt')
    assert select_checkpoint(cfg,output,input_preparer=prepare_inputs)['selected_step'] in [2,4,6,8]
    curve=(output/'validation_curve.jsonl').read_bytes()
    assert run(cfg,output)=={'status':'complete','step':8}
    assert (output/'validation_curve.jsonl').read_bytes()==curve


@pytest.mark.parametrize('changed_field',['checkpoint_sha256','dataset_digest'])
def test_terminal_recovery_rejects_changed_validation_identity(mixed,tmp_path,monkeypatch,changed_field):
    import splendor.offline_outcome as entry
    from splendor.alphazero_recording import write_json
    monkeypatch.setattr(entry,'source_identity',lambda *_:{'test':True})
    cfg=config_for(mixed,'scalar_mc');out=tmp_path/'fit';run(cfg,out)
    write_json(out/'progress.json',dict(status='incomplete',step=8,steps_target=8))
    path=out/'full_validation/step-000000008.json';row=read_json(path)
    row[changed_field]='changed';write_json(path,row)
    with pytest.raises(ValueError,match='Terminal validation'):
        run(cfg,out)
    assert read_json(out/'progress.json')['status']=='incomplete'


def test_terminal_recovery_keeps_checkpoint_replay_identity_checks(mixed,tmp_path,monkeypatch):
    import splendor.offline_outcome as entry
    from splendor.alphazero_recording import write_json,sha256
    monkeypatch.setattr(entry,'source_identity',lambda *_:{'test':True})
    cfg=config_for(mixed,'scalar_mc');out=tmp_path/'fit';run(cfg,out)
    write_json(out/'progress.json',dict(status='incomplete',step=8,steps_target=8))
    checkpoint=out/'checkpoints/step-000000008.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    payload['replay_identity']['digest']='different-training-identity';torch.save(payload,checkpoint)
    path=out/'full_validation/step-000000008.json';row=read_json(path)
    row['checkpoint_sha256']=sha256(checkpoint);write_json(path,row)
    with pytest.raises(ValueError,match='Terminal checkpoint'):
        run(cfg,out)
    assert read_json(out/'progress.json')['status']=='incomplete'
