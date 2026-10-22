from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic, TorchStateBatch, build_actor_critic, tensorize_batch
from splendor.entity_set_mlp import EntitySetMLP, EntitySetMLPConfig, EntitySetMLPState
from splendor.engine import GameEngine
from splendor.features import CARD_FEATURE_NAMES, collate_encoded_decisions, encode_decision
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.value_auxiliary import add_auxiliary_heads


@pytest.fixture(autouse=True, scope='module')
def one_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def batch():
    request=GameEngine.new(45200000,first_player=0,mode='checked').decision()
    return tensorize_batch(collate_encoded_decisions([encode_decision(request)]))


@pytest.fixture(params=[{}, {'action_bottleneck':32}, {'multiplicative_interaction':True},
    {'action_bottleneck':32,'multiplicative_interaction':True},
    {'action_bottleneck':32,'multiplicative_interaction':True,'market_relations':False},
    {'action_cross_hidden':64},
    {'action_bottleneck':32,'action_cross_hidden':64},
    {'action_bottleneck':32,'multiplicative_interaction':True,'action_cross_hidden':64},
    {'action_cross_hidden':32,'action_cross_heads':1,'market_relations':False}],
    ids=['legacy','nonlinear','multiply','both','both-no-relations','cross-only','cross-no-product','cross-all','cross32-no-relations'])
def config(request):
    return EntitySetMLPConfig(**request.param)


def test_preset_capacity_and_explicit_factory():
    import json
    preset=json.loads((Path(__file__).resolve().parents[1]/'configs/entity_set_mlp_v1/model_256x4.json').read_text())
    model=build_actor_critic(preset)
    assert type(model) is EntitySetMLP
    assert model.trainable_parameter_count()==906434
    assert model.config.action_bottleneck==32 and model.config.multiplicative_interaction
    assert model.action_nonlinear[0].in_features==90
    assert model.action_nonlinear[0].out_features==32
    assert model.action_nonlinear[2].out_features==128
    torch.testing.assert_close(model.interaction_scale,torch.ones(128))
    assert model.state_input_dim==922 and model.action_representation_dim==90
    assert len(model.state_encoder)==6  # input linear, ReLU, four residual blocks
    assert model.config.action_cross_hidden==64 and model.config.action_cross_heads==2
    assert not any(isinstance(x, (torch.nn.LayerNorm,torch.nn.MultiheadAttention)) for x in model.state_encoder.modules())


@pytest.mark.parametrize('seed', [43100000, 45400001, 45400002])
def test_cross_role_initialization_matches_content_scale(batch, seed):
    torch.manual_seed(seed)
    model=EntitySetMLP(EntitySetMLPConfig(action_cross_hidden=64))
    with torch.no_grad():
        cards=model.encode_state(batch).cards[:,:12]
        roles=torch.stack([model.action_cross._role(2,0,t) for t in (1,2,3)])
        ratio=roles.square().mean().sqrt()/cards.square().mean().sqrt()
    assert .25 < ratio < 4  # role identity must not swamp within-tier card content


def test_loading_existing_cross_weights_does_not_rescale_roles(batch, tmp_path):
    model=EntitySetMLP(EntitySetMLPConfig(action_cross_hidden=64)).eval()
    with torch.no_grad():
        for module in (model.action_cross.type_embedding,model.action_cross.owner_embedding,model.action_cross.tier_embedding):
            module.weight.mul_(20)  # a checkpoint produced by the previous std1 initializer
    path=tmp_path/'previous-cross.pt'
    model.save_weights(path)
    loaded=ActorCritic.load_weights(path).eval()
    for name,weight in model.state_dict().items():
        torch.testing.assert_close(weight,loaded.state_dict()[name],rtol=0,atol=0)
    for a,b in zip(model(batch),loaded(batch),strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_shapes_state_only_projection_and_candidate_permutation(batch,config):
    model=EntitySetMLP(config).eval()
    state=TorchStateBatch(**{k:getattr(batch,k) for k in TorchStateBatch.__dataclass_fields__})
    with torch.inference_mode():
        logits,value=model(batch)
        torch.testing.assert_close(value,model.value_from_state(state))
        torch.testing.assert_close(value,model.value_only(batch))
        torch.testing.assert_close(logits,model.policy_logits(batch))
        projected=model._forward_prepared(batch,projection=model.build_actor_projection())
        for a,b in zip((logits,value),projected,strict=True):torch.testing.assert_close(a,b)
        perm=torch.arange(logits.shape[1]-1,-1,-1)
        other=model(replace(batch,candidate_features=batch.candidate_features[:,perm],candidate_mask=batch.candidate_mask[:,perm]))
        torch.testing.assert_close(other[0],logits[:,perm]);torch.testing.assert_close(other[1],value)


def test_set_permutation_padding_and_empty_max_are_safe(batch,config):
    model=EntitySetMLP(config).eval()
    changes={}
    for field in ('market_t1','market_t2','market_t3','self_reserved','opponent_reserved','nobles'):
        changes[field]=getattr(batch,field).flip(1)
        changes[field+'_mask']=getattr(batch,field+'_mask').flip(1)
    with torch.no_grad():
        baseline=model(batch)
        other=model(replace(batch,**changes))
        for a,b in zip(baseline,other,strict=True):torch.testing.assert_close(a,b)
        poison=replace(batch,self_reserved=torch.full_like(batch.self_reserved,float('nan')),
            opponent_reserved=torch.full_like(batch.opponent_reserved,float('nan')),
            pending_card=torch.full_like(batch.pending_card,float('nan')))
        for a,b in zip(baseline,model(poison),strict=True):torch.testing.assert_close(a,b)
        empty=replace(poison,market_t1_mask=torch.zeros_like(batch.market_t1_mask),
            market_t2_mask=torch.zeros_like(batch.market_t2_mask),market_t3_mask=torch.zeros_like(batch.market_t3_mask),
            nobles_mask=torch.zeros_like(batch.nobles_mask))
        assert all(torch.isfinite(x).all() for x in model(empty))
    value=model(empty)[1];value.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_actor_directly_reads_raw_target_even_with_frozen_state(batch,config):
    model=EntitySetMLP(config)
    index=model.action_meta_dim+CARD_FEATURE_NAMES.index('cost_white_scaled')
    first,_,last=model.actor
    with torch.no_grad():
        first.weight.zero_();first.bias.fill_(1.)
        first.weight[0,model.config.state_hidden+index]=1.
        last.weight.zero_();last.bias.zero_();last.weight[0,0]=1.
        if model.action_nonlinear is not None:
            model.action_nonlinear[2].weight.zero_();model.action_nonlinear[2].bias.zero_()
        if model.interaction_scale is not None:model.interaction_scale.zero_()
        if model.action_cross is not None:
            model.action_cross.output.weight.zero_();model.action_cross.output.bias.zero_()
    fixed=torch.zeros(1,model.config.state_hidden)
    if model.action_cross is not None:
        fixed=model.encode_state(batch)._replace(shared=fixed)
    other=batch.candidate_features.clone();other[0,0,index]+=.5
    a=model._actor_logits(batch,fixed);b=model._actor_logits(replace(batch,candidate_features=other),fixed)
    assert float((b[0,0]-a[0,0]).detach())==pytest.approx(.5)
    torch.testing.assert_close(a[:,1:],b[:,1:])
    a.sum().backward()
    assert first.weight.grad[0,model.config.state_hidden+index]!=0


def test_public_target_relations_equal_entity_relations_and_exclude_blind(batch):
    model=EntitySetMLP();opponent=batch.global_features[:,model.opponent_indices]
    raw=batch.candidate_features.clone();meta=model.action_meta_dim
    raw[0,:3,meta:meta+29]=batch.market_t1[0,:3]
    raw[0,1,meta+10:meta+12]=torch.tensor([0.,1.])  # known own blind card
    raw[0,2,meta:meta+29]=0
    raw[0,2,meta]=raw[0,2,meta+2]=raw[0,2,meta+11]=1.  # unknown blind card
    out=model._action_inputs(replace(batch,candidate_features=raw))
    expected=model._card_inputs(raw[:,:3,meta:meta+29],batch.candidate_mask[:,:3],opponent)
    torch.testing.assert_close(out[:,:3,-8:],expected[:,:3,-8:])
    assert torch.count_nonzero(out[:,1:3,-8:])==0


@pytest.mark.parametrize('predict_score',[False,True])
def test_auxiliary_upgrade_preserves_base_and_inference_skips_heads(batch,predict_score,tmp_path,config):
    base=EntitySetMLP(config);model=add_auxiliary_heads(base,predict_score=predict_score)
    for name,value in base.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    assert [name for name,_ in model.named_parameters()][:len(list(base.parameters()))]==[name for name,_ in base.named_parameters()]
    logits,v,score,turns=model.forward_with_auxiliary(batch)
    assert (score is not None)==predict_score and (turns>0).all()
    loss=logits.square().mean()+v.square().mean()+turns.square().mean()
    if score is not None:loss=loss+score.square().mean()
    loss.backward()
    assert model.remaining_turns_head[0].weight.grad is not None
    hooks=[model.remaining_turns_head.register_forward_pre_hook(lambda *_:pytest.fail('inference ran auxiliary head'))]
    if predict_score:hooks.append(model.score_head.register_forward_pre_hook(lambda *_:pytest.fail('inference ran auxiliary head')))
    model(batch);model.value_only(batch)
    for hook in hooks:hook.remove()
    path=tmp_path/'model.pt';model.save_weights(path)
    restored=ActorCritic.load_weights(path)
    assert asdict(restored.config)==asdict(model.config)
    for a,b in zip(model(batch),restored(batch),strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('changes',[{'residual_blocks':0},{'state_hidden':True},{'market_relations':1},
    {'feature_schema_version':'wrong'},{'action_bottleneck':-1},{'action_bottleneck':True},
    {'action_bottleneck':1.5},{'multiplicative_interaction':1},
    {'action_cross_hidden':-1},{'action_cross_hidden':True},{'action_cross_heads':0},
    {'action_cross_heads':True},{'action_cross_hidden':63,'action_cross_heads':2}])
def test_invalid_config_fails(changes):
    with pytest.raises(ValueError):EntitySetMLPConfig(**changes)


def test_eager_search_inference_projection(batch,config):
    model=EntitySetMLP(config).eval()
    eager=FrozenSearchInference(model,config=InferenceConfig(mode='eager',actor_projection=True))
    with torch.inference_mode():
        expected=model(batch)
        observed=eager._forward(batch)
    for a,b in zip(expected,observed,strict=True):torch.testing.assert_close(a,b)


def test_action_branch_is_nonlinear_and_trainable_with_fixed_state(batch):
    model=EntitySetMLP(EntitySetMLPConfig(action_bottleneck=32))
    index=model.action_meta_dim+CARD_FEATURE_NAMES.index('cost_white_scaled')
    with torch.no_grad():
        for parameter in model.actor.parameters():parameter.zero_()
        for parameter in model.action_nonlinear.parameters():parameter.zero_()
        model.actor[0].bias.fill_(1.)
        model.actor[2].weight[0,0]=1.
        model.action_nonlinear[0].weight[0,index]=1.
        model.action_nonlinear[2].weight[0,0]=1.
    raw=torch.zeros_like(batch.candidate_features)
    raw[0,:3,index]=torch.tensor([-.5,0.,.5])
    logits=model._actor_logits(replace(batch,candidate_features=raw),torch.zeros(1,256))
    torch.testing.assert_close(logits[0,:3],torch.tensor([1.,1.,1.5]))
    logits.sum().backward()
    assert model.action_nonlinear[0].weight.grad[0,index]>0
    assert model.action_nonlinear[2].weight.grad[0,0]>0


def test_product_changes_action_preference_as_state_changes(batch):
    model=EntitySetMLP(EntitySetMLPConfig(multiplicative_interaction=True))
    index=model.action_meta_dim+CARD_FEATURE_NAMES.index('cost_white_scaled')
    with torch.no_grad():
        for parameter in model.actor.parameters():parameter.zero_()
        model.actor[0].weight[0,0]=1.
        model.actor[0].weight[0,256+index]=1.
        model.actor[2].weight[0,0]=1.
    raw=torch.zeros_like(batch.candidate_features);raw[0,:2,index]=torch.tensor([1.,2.])
    prepared=replace(batch,candidate_features=raw)
    state=torch.zeros(1,256);state[0,0]=1.
    first=model._actor_logits(prepared,state)
    state2=state.clone();state2[0,0]=2.
    second=model._actor_logits(prepared,state2)
    torch.testing.assert_close(first[0,:2],torch.tensor([3.,5.]))
    torch.testing.assert_close(second[0,:2],torch.tensor([5.,8.]))
    (first.sum()+second.sum()).backward()
    assert model.interaction_scale.grad[0]>0
    with torch.no_grad():model.interaction_scale.zero_()
    for z in (state,state2):
        additive=model._actor_logits(prepared,z)
        torch.testing.assert_close(additive[0,1]-additive[0,0],torch.tensor(1.))


def test_legacy_weights_without_new_config_fields_load_strictly(batch,tmp_path):
    legacy=EntitySetMLP()
    path=tmp_path/'legacy.pt';legacy.save_weights(path)
    payload=torch.load(path,weights_only=False)
    for name in ('action_bottleneck','multiplicative_interaction','action_cross_hidden','action_cross_heads'):
        payload['config'].pop(name)
    torch.save(payload,path)
    restored=ActorCritic.load_weights(path)
    assert restored.trainable_parameter_count()==848418
    assert restored.action_nonlinear is None and restored.interaction_scale is None
    for a,b in zip(legacy(batch),restored(batch),strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
    upgraded=EntitySetMLP(EntitySetMLPConfig(action_bottleneck=32,multiplicative_interaction=True))
    with pytest.raises(RuntimeError,match='Missing key'):
        upgraded.load_state_dict(payload['state_dict'],strict=True)


def test_value_only_does_not_run_action_branch(batch):
    model=EntitySetMLP(EntitySetMLPConfig(action_bottleneck=32,multiplicative_interaction=True,action_cross_hidden=64))
    hook=model.action_nonlinear.register_forward_pre_hook(lambda *_:pytest.fail('value ran action branch'))
    cross_hook=model.action_cross.register_forward_pre_hook(lambda *_:pytest.fail('value ran cross attention'))
    model.value_only(batch)
    hook.remove()
    cross_hook.remove()


def test_upgraded_actor_ignores_poisoned_candidate_padding(batch):
    model=EntitySetMLP(EntitySetMLPConfig(action_bottleneck=32,multiplicative_interaction=True,action_cross_hidden=64)).eval()
    padded=replace(batch,candidate_features=torch.cat((batch.candidate_features,
        torch.full_like(batch.candidate_features[:,:1],float('nan'))),dim=1),
        candidate_mask=torch.cat((batch.candidate_mask,torch.zeros_like(batch.candidate_mask[:,:1])),dim=1))
    with torch.no_grad():
        logits,value=model(batch);other_logits,other_value=model(padded)
        torch.testing.assert_close(logits,other_logits[:,:-1])
        torch.testing.assert_close(value,other_value)
        probabilities=model.distribution(padded)[0].probs
        assert probabilities[0,-1]==0 and torch.isfinite(probabilities).all()


def test_cross_reads_unpooled_entities_with_fixed_shared_state(batch):
    torch.manual_seed(45400001)
    model=EntitySetMLP(EntitySetMLPConfig(action_cross_hidden=64))
    with torch.no_grad():
        for parameter in model.actor.parameters():parameter.zero_()
        model.actor[0].bias.fill_(5.)
        model.actor[2].weight[0,0]=1.
        model.action_cross.output.weight.zero_();model.action_cross.output.bias.zero_()
        model.action_cross.output.weight[0,0]=1.
    encoded=model.encode_state(batch)
    cards=encoded.cards.detach().clone().requires_grad_()
    fixed=EntitySetMLPState(torch.zeros_like(encoded.shared),cards,encoded.nobles.detach())
    logits=model._actor_logits(batch,fixed)
    other_cards=cards.detach().clone();other_cards[:,0,0]+=.7
    other=model._actor_logits(batch,fixed._replace(cards=other_cards))
    assert not torch.allclose(logits,other,rtol=0,atol=1e-6)
    logits[:,0].sum().backward()
    assert cards.grad[:,:12].abs().sum()>0
    assert torch.count_nonzero(cards.grad[:,12:])==0  # empty reserves/pending
    poisoned=cards.detach().clone();poisoned[:,12:]=float('nan')
    torch.testing.assert_close(model._actor_logits(batch,fixed._replace(cards=poisoned)),logits)


def test_cross_query_reads_state_and_action_and_projects_memory_once(batch):
    torch.manual_seed(45400002)
    model=EntitySetMLP(EntitySetMLPConfig(action_cross_hidden=64))
    state=model.encode_state(batch);raw=model._action_inputs(batch)
    observed={name:[] for name in ('key','value','query_state','query_action')}
    hooks=[getattr(model.action_cross,name).register_forward_pre_hook(
        lambda module,args,name=name:observed[name].append(tuple(args[0].shape))) for name in observed]
    context=model.action_cross(batch,state,raw)
    for hook in hooks:hook.remove()
    assert observed['key']==observed['value']==[(1,25,64)]
    assert observed['query_state']==[(1,256)]
    assert observed['query_action']==[(1,raw.shape[1],90)]
    z=state.shared.detach().clone();z[:,0]+=.5
    other_raw=raw.detach().clone();other_raw[:,0,0]+=.5
    assert not torch.allclose(context,model.action_cross(batch,state._replace(shared=z),raw),rtol=0,atol=1e-6)
    assert not torch.allclose(context[:,0],model.action_cross(batch,state,other_raw)[:,0],rtol=0,atol=1e-6)
    torch.testing.assert_close(context[:,1:],model.action_cross(batch,state,other_raw)[:,1:])


def test_zero_cross_residual_recovers_previous_actor_and_value(batch):
    config=EntitySetMLPConfig(action_bottleneck=32,multiplicative_interaction=True)
    torch.manual_seed(45400003);base=EntitySetMLP(config)
    torch.manual_seed(45400003);model=EntitySetMLP(replace(config,action_cross_hidden=64))
    for name,value in base.state_dict().items():
        torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    with torch.no_grad():
        model.action_cross.output.weight.zero_();model.action_cross.output.bias.zero_()
        for a,b in zip(base(batch),model(batch),strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_previous_actor12_weights_without_cross_fields_load_strictly(batch,tmp_path):
    base=EntitySetMLP(EntitySetMLPConfig(action_bottleneck=32,multiplicative_interaction=True))
    path=tmp_path/'actor12.pt';base.save_weights(path)
    payload=torch.load(path,weights_only=False)
    for name in ('action_cross_hidden','action_cross_heads'):payload['config'].pop(name)
    torch.save(payload,path)
    restored=ActorCritic.load_weights(path)
    assert restored.action_cross is None and restored.trainable_parameter_count()==855682
    for a,b in zip(base(batch),restored(batch),strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
