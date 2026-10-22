from dataclasses import replace

import pytest
import torch

from test_alphazero_training import az_runs
from test_offline_scratch import scratch
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.offline_scratch import initialize_learner,run
from splendor.rng import DeterministicRng


@pytest.fixture(autouse=True,scope='module')
def one_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('auxiliary',['none','turns','dual'])
def test_real_scratch_entry_training_and_exact_resume(scratch,tmp_path,auxiliary):
    config=replace(scratch,model_config={'architecture':'entity_set_mlp_v1','card_hidden':8,'noble_hidden':4,
        'state_hidden':16,'residual_blocks':2,'actor_hidden':8,'critic_hidden':8,'market_relations':True,
        'action_bottleneck':4,'multiplicative_interaction':True,'action_cross_hidden':8,'action_cross_heads':2},
        draw_exposures=scratch.draw_exposures*2.25/20.75,validation_percent_step=50,
        learner=replace(scratch.learner,checkpoint_every_steps=1))
    if auxiliary=='none':config=replace(config,value_auxiliary=None)
    elif auxiliary=='turns':config=replace(config,value_auxiliary={**config.value_auxiliary,'score_loss_weight':0.})
    uninterrupted=tmp_path/'full';resumed=tmp_path/'resume'
    run(config,uninterrupted)
    run(config,resumed,max_steps_this_session=1)
    run(config,resumed)
    a=load_alphazero_checkpoint(uninterrupted/'checkpoints/final.pt')
    b=load_alphazero_checkpoint(resumed/'checkpoints/final.pt')
    assert a.total_train_step==b.total_train_step==3
    for key,value in a.model_state_dict.items():torch.testing.assert_close(value,b.model_state_dict[key],rtol=0,atol=0)
    for pid,state in a.optimizer_state_dict['state'].items():
        for key,value in state.items():torch.testing.assert_close(value,b.optimizer_state_dict['state'][pid][key],rtol=0,atol=0)
    assert a.sampling_rng_state==b.sampling_rng_state
    assert a.build_model().config.architecture==('entity_set_mlp_v1' if auxiliary=='none' else 'entity_set_mlp_aux_v1')
    assert a.build_model().config.action_bottleneck==4
    assert a.build_model().config.multiplicative_interaction
    assert a.build_model().config.action_cross_hidden==8
    initial=initialize_learner(config,{'test':True},3).model.state_dict()
    for key in ('action_nonlinear.0.weight','action_nonlinear.2.weight','interaction_scale',
                'action_cross.query_state.weight','action_cross.query_action.weight','action_cross.key.weight',
                'action_cross.value.weight','action_cross.output.weight'):
        assert not torch.equal(a.model_state_dict[key],initial[key])
