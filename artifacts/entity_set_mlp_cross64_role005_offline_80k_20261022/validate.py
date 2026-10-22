"""Finite correctness checks of the actual preset; no training/strength benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from splendor.actor_critic import ActorCritic,build_actor_critic,tensorize_batch
from splendor.alphazero_recording import environment_identity,read_json,sha256,source_identity,write_json
from splendor.alphazero_train import AlphazeroLearner,LearnerConfig
from splendor.compact_trace import decode_request
from splendor.features import encode_decision,encode_decision_state,collate_encoded_decisions
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner,NativePUCTSearchPolicy,NativeChancePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.rng import DeterministicRng
from splendor.search_inference import FrozenSearchInference,InferenceConfig
from splendor.selfplay_data import SelfPlaySample,collate_selfplay_samples,load_game
from splendor.value_auxiliary import ValueAuxiliaryConfig,add_auxiliary_heads,auxiliary_targets

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
PRESET=ROOT/'configs/entity_set_mlp_v1/model_256x4.json'
ARCHIVE=ROOT/'runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/selfplay/games/game-000000.npz'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(43100000)
    if args.device=='cuda':
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();work=ROOT/'runtime'/ART.name/'smoke'/args.device;work.mkdir(parents=True,exist_ok=True)
    model=build_actor_critic(read_json(PRESET))
    capacity=dict(config=asdict(model.config),parameters=model.trainable_parameter_count(),
        dual_auxiliary_parameters=add_auxiliary_heads(model).trainable_parameter_count(),
        turns_only_parameters=add_auxiliary_heads(model,predict_score=False).trainable_parameter_count(),
        state_input_dim=model.state_input_dim,action_input_dim=model.action_representation_dim,
        by_module={**{name:sum(p.numel() for p in child.parameters()) for name,child in model.named_children()},
                   **{name:p.numel() for name,p in model.named_parameters(recurse=False)}})
    assert capacity['parameters']==906434 and sum(capacity['by_module'].values())==906434
    bank=read_json(ROOT/'artifacts/entity_attention_r20_search_stability_128_20261021/root_bank.json')['roots']
    scale_roots=[next(r for r in bank if r['group']=='baseline' and r['stage']==stage) for stage in ('early','middle','late')]
    scale_batch=tensorize_batch(collate_encoded_decisions([encode_decision(decode_request(r['request'])) for r in scale_roots]))
    with torch.no_grad():
        state=model.encode_state(scale_batch)
        role=torch.stack([model.action_cross._role(2,0,t) for t in (1,2,3)])
        content_rms=float(state.cards[:,:12].square().mean().sqrt())
        role_rms=float(role.square().mean().sqrt())
    assert .25 < role_rms/content_rms < 4
    role_scale={'seed':43100000,'root_ids':[r['id'] for r in scale_roots],
        'individual_role_std':0.05,'combined_role_rms':role_rms,'card_content_rms':content_rms,
        'ratio':role_rms/content_rms,'scope':'initial market embeddings on three fixed early/middle/late roots'}
    arrays,meta=load_game(ARCHIVE)
    indices=[int(np.flatnonzero(arrays['phase']==p)[0]) for p in np.unique(arrays['phase'])]
    indices=(indices+[i for i in range(len(arrays['phase'])) if i not in indices])[:4]
    samples=[SelfPlaySample(arrays,meta,i) for i in indices];batch=collate_selfplay_samples(samples)
    config=LearnerConfig(device=args.device,batch_size=4,history_batch_size=2,learning_rate=1e-4,checkpoint_every_steps=1)
    learner=AlphazeroLearner(model,config,sampling_rng=DeterministicRng(45310000))
    learner.round_steps_target=2;learner.replay_identity={'digest':sha256(ARCHIVE)}
    new_names=('action_nonlinear.0.weight','action_nonlinear.2.weight','interaction_scale',
               'action_cross.query_state.weight','action_cross.query_action.weight',
               'action_cross.key.weight','action_cross.value.weight','action_cross.output.weight')
    before={name:model.state_dict()[name].clone() for name in new_names}
    primary=learner.train_step(batch,references=[])
    assert all(math.isfinite(primary[k]) for k in ('policy_loss','value_loss','total_loss','grad_norm'))
    updates={name:float((model.state_dict()[name]-before[name]).abs().max()) for name in new_names}
    assert all(math.isfinite(delta) and delta>0 for delta in updates.values())
    checkpoint=work/'step-1.pt';learner.save(checkpoint)
    restored=load_frozen_actor_critic_checkpoint(checkpoint,device=args.device)
    for key,value in learner.model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    resume=AlphazeroLearner.resume(checkpoint,config,replay_identity=learner.replay_identity,round_index=0,round_steps_target=2)
    if args.device=='cpu':
        learner.train_step(batch,references=[]);resume.train_step(batch,references=[])
        for key,value in learner.model.state_dict().items():torch.testing.assert_close(value,resume.model.state_dict()[key],rtol=0,atol=0)
    pure=work/'weights.pt';restored.save_weights(pure)
    pure_model=ActorCritic.load_weights(pure,map_location='cpu')
    assert asdict(pure_model.config)==asdict(restored.config)
    auxiliary=AlphazeroLearner(add_auxiliary_heads(restored),config,sampling_rng=DeterministicRng(45320000))
    aux_metrics=auxiliary.train_step(batch,references=[],value_auxiliary=ValueAuxiliaryConfig(),
                                    auxiliary_labels=auxiliary_targets(samples,ValueAuxiliaryConfig()))
    assert all(math.isfinite(aux_metrics[k]) for k in ('score_loss','remaining_turns_loss','total_loss'))
    auxiliary.save(work/'aux-step-1.pt')
    assert load_frozen_actor_critic_checkpoint(work/'aux-step-1.pt',device=args.device).config.architecture=='entity_set_mlp_aux_v1'
    bank=read_json(ROOT/'artifacts/entity_attention_r20_search_stability_128_20261021/root_bank.json')['roots']
    roots=[next(r for r in bank if r['group']=='baseline' and r['stage']==stage) for stage in ('early','middle','late')]
    requests=[decode_request(r['request']) for r in roots]
    encoded=[encode_decision(r) for r in requests];states=[encode_decision_state(r) for r in requests]
    eager=FrozenSearchInference(restored,device=args.device,config=InferenceConfig(actor_projection=False))
    inference=InferenceConfig(mode='compiled',actor_projection=True,
        candidate_buckets=(16,32,48,64),compile_mode='native-cudagraph' if args.device=='cuda' else None,
        batch_buckets=(4,) if args.device=='cuda' else (),compile_failure_policy='fail_fast')
    evaluator=FrozenSearchInference(restored,device=args.device,config=inference)
    for (la,va),(lb,vb) in zip(eager.policy_value_encoded(encoded),evaluator.policy_value_encoded(encoded),strict=True):
        torch.testing.assert_close(torch.tensor(la),torch.tensor(lb),rtol=2e-4,atol=2e-5)
        torch.testing.assert_close(torch.tensor(va),torch.tensor(vb),rtol=2e-4,atol=2e-5)
    for la,lb in zip(eager.policy_only(requests),evaluator.policy_only(requests),strict=True):
        torch.testing.assert_close(torch.tensor(la),torch.tensor(lb),rtol=2e-4,atol=2e-5)
    torch.testing.assert_close(torch.tensor(eager.value_only_encoded(states)),torch.tensor(evaluator.value_only_encoded(states)),rtol=2e-4,atol=2e-5)
    policies=[cls(restored,evaluator=evaluator,config=MCTSConfig(simulations=8,determinizations=n,
        seed=45330000,root_noise_fraction=0.)) for cls,n in ((NativePUCTSearchPolicy,1),(NativePUCTSearchPolicy,4),(NativeChancePUCTSearchPolicy,1))]
    jobs=[p.start_job(r,add_root_noise=False,temperature=0.) for p,r in zip(policies,requests,strict=True)]
    runner=NativeBatchRunner(evaluator,policies[0].catalog,capacity=4,buckets=(16,32,48,64),slots=1)
    while any(not j.done for j in jobs):assert runner.run([j for j in jobs if not j.done])>0
    for job in jobs:
        assert job.result.simulations==8 and sum(job.result.root_visit_counts)==8 and job.result.transition_cap_hits==0
    stats=evaluator.stats.to_dict()
    compile_stats=evaluator.compiled.stats.to_dict()
    assert compile_stats['compile_failure_fallbacks']==0 and sum(compile_stats['compiled_calls'].values())>0
    source=source_identity(ROOT,work/'source')
    source['producer_sha256']=sha256(Path(__file__))
    write_json(ART/f'{args.device}_validation.json',dict(scope='Four archived samples, finite updates/checkpoint round trips and three S8 roots only; no throughput or strength conclusion.',
        passed=True,device=args.device,capacity=capacity,role_scale=role_scale,preset_sha256=sha256(PRESET),
        archive=dict(path=str(ARCHIVE.relative_to(ROOT)),sha256=sha256(ARCHIVE),indices=indices,phases=[int(arrays['phase'][i]) for i in indices]),
        primary_step=primary,new_parameter_updates=updates,auxiliary_step=aux_metrics,cpu_exact_resume=args.device=='cpu',
        policy_only_checked=True,
        checkpoint=dict(path=str(checkpoint.relative_to(ROOT)),sha256=sha256(checkpoint)),
        inference=inference.to_dict(),inference_stats=stats,
        searches=[dict(stage=r['stage'],backend='chance' if p.stochastic else 'native',result=asdict(j.result)) for r,p,j in zip(roots,policies,jobs,strict=True)],
        elapsed_seconds=time.perf_counter()-started,source=source,environment=environment_identity(),command=sys.argv))
    write_json(ART/'capacity.json',capacity)
    print(f'{args.device}: cross64x2 preset training/checkpoint/compiled search checks passed.',flush=True)


if __name__=='__main__':main()
