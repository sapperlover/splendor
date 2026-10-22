"""Public purchase relations and direct value readout, independent of archives."""
from dataclasses import replace

import pytest
import torch

from splendor.actor_critic import build_actor_critic
from splendor.entity_attention import PlayersMarketValueHead, market_purchase_relations
from splendor.value_auxiliary import add_auxiliary_heads


@pytest.fixture(autouse=True, scope='module')
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_purchase_relations_gold_discounts_unknown_and_padding():
    cards = torch.zeros(1, 4, 29)
    cards[:, :2, :2] = 1
    cards[0, 0, 13:18] = torch.tensor([4, 3, 2, 0, 1]) / 7
    cards[0, 1, 13:18] = torch.tensor([2, 2, 2, 0, 0]) / 7
    cards[0, 1, 28] = 1 / 14
    cards[0, 2, 0] = 1  # A valid unknown card is not a free known card.
    cards[0, 2, 13:] = float('nan')
    cards[0, 3] = float('nan')
    opponent = torch.zeros(1, 14)
    opponent[0, :6] = torch.tensor([1/4, 0, 0, 0, 0, 2/5])
    opponent[0, 6:11] = torch.tensor([1, 1, 2, 0, 0]) / 18
    actual = market_purchase_relations(cards, torch.tensor([[True, True, True, False]]), opponent)
    expected = torch.tensor([[[2/7, 2/7, 0, 0, 1/7, 3/14, 1, 0],
                              [0, 1/7, 0, 0, 0, 0, 0, 1], [0]*8, [0]*8]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('bonus', range(19))
def test_exact_affordability_survives_normalization(bonus):
    cards = torch.zeros(1, 1, 29)
    cards[..., :2] = 1
    cards[..., 13:18] = bonus / 7
    opponent = torch.zeros(1, 14)
    opponent[:, 6:11] = bonus / 18
    actual = market_purchase_relations(cards, torch.ones(1, 1, dtype=torch.bool), opponent)
    torch.testing.assert_close(actual[..., :6], torch.zeros(1, 1, 6), atol=0, rtol=0)
    assert actual[0, 0, 7] == 1


def test_value_directly_reads_both_players_and_only_valid_market():
    torch.manual_seed(39140000)
    head = PlayersMarketValueHead(8)
    tokens = torch.randn(2, 25, 8, requires_grad=True)
    valid = torch.ones(2, 25, dtype=torch.bool)
    valid[:, 5] = False
    head((tokens, valid)).sum().backward()
    assert all(tokens.grad[:, i].abs().sum() > 0 for i in (0, 1, 2, 3, 14))
    assert tokens.grad[:, 5].abs().sum() == 0
    assert tokens.grad[:, 15:].abs().sum() == 0
    with torch.no_grad():
        reference = head((tokens, valid))
        changed = tokens.clone()
        changed[:, 5] = float('nan')
        changed[:, 15:] = float('nan')
        torch.testing.assert_close(head((changed, valid)), reference)
        valid[:, 3:15] = False
        changed[:, 3:15] = float('nan')
        assert torch.isfinite(head((changed, valid))).all()


@pytest.mark.parametrize('predict_score', [False, True])
def test_auxiliary_upgrade_retains_new_options_and_all_weights(predict_score):
    config = {'architecture': 'entity_attention_v1', 'market_relations': True,
              'value_readout': 'global_players_market'}
    model = build_actor_critic(config)
    upgraded = add_auxiliary_heads(model, predict_score=predict_score)
    assert upgraded.config.market_relations
    assert upgraded.config.value_readout == 'global_players_market'
    assert isinstance(upgraded.critic, PlayersMarketValueHead)
    assert isinstance(upgraded.remaining_turns_head, PlayersMarketValueHead)
    if predict_score:
        assert isinstance(upgraded.score_head, PlayersMarketValueHead)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[name], value, atol=0, rtol=0)


def test_relation_branch_and_readout_receive_training_gradients(initial_state, catalog):
    from splendor.actor_critic import tensorize_batch
    from splendor.contracts import DecisionRequest
    from splendor.features import collate_encoded_decisions, encode_decision
    from splendor.observation import build_observation
    from splendor.rules import legal_actions

    model = build_actor_critic({'architecture': 'entity_attention_v1', 'market_relations': True,
                               'value_readout': 'global_players_market'})
    request = DecisionRequest(build_observation(initial_state, 0, catalog), legal_actions(initial_state, catalog))
    batch = tensorize_batch(collate_encoded_decisions([encode_decision(request)]))
    logits, value = model(batch)
    loss = logits[:, 0].square().mean() + (value - 1).square().mean()
    loss.backward()
    assert model.market_relation_encoder.weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize('dimensions,expected_parameters', [
    ({'hidden': 128, 'heads': 4, 'layers': 2, 'ffn_hidden': 512}, 583682),
    ({'hidden': 256, 'heads': 8, 'layers': 4, 'ffn_hidden': 1024}, 3828226),
])
def test_public_reservations_share_relations_and_exclude_blind_cards(
        initial_state, catalog, dimensions, expected_parameters):
    from splendor.actor_critic import tensorize_batch
    from splendor.contracts import DecisionRequest
    from splendor.features import GLOBAL_FEATURE_NAMES, collate_encoded_decisions, encode_decision
    from splendor.observation import build_observation
    from splendor.rules import legal_actions

    model = build_actor_critic({'architecture': 'entity_attention_v1', **dimensions,
                               'market_relations': True, 'value_readout': 'global_players_market'})
    assert model.trainable_parameter_count() == expected_parameters
    request = DecisionRequest(build_observation(initial_state, 0, catalog), legal_actions(initial_state, catalog))
    batch = tensorize_batch(collate_encoded_decisions([encode_decision(request)]))
    own, opponent = batch.market_t1[:, :3].clone(), batch.market_t2[:, :3].clone()
    # One public card, one deck reservation and one poisoned padding slot per side.
    own[:, 1, 10:12] = torch.tensor([0., 1.])  # self privately knows this blind card
    opponent[:, 1] = 0
    opponent[:, 1, 0] = opponent[:, 1, 2] = opponent[:, 1, 11] = 1
    own[:, 2] = opponent[:, 2] = float('nan')
    valid = torch.tensor([[True, True, False]])
    global_features = batch.global_features.clone()
    # Asymmetric resources distinguish current actor from reservation owner.
    for name, value in [('opponent_token_white_scaled', .25),
                        ('opponent_token_gold_scaled', .4), ('opponent_bonus_blue_scaled', 1/18)]:
        global_features[:, GLOBAL_FEATURE_NAMES.index(name)] = value
    batch = replace(batch, global_features=global_features, self_reserved=own, opponent_reserved=opponent,
                    self_reserved_mask=valid, opponent_reserved_mask=valid)
    inputs, outputs = [], []

    def capture(module, args, output):
        inputs.append(args[0].detach().clone())
        output.retain_grad()
        outputs.append(output)

    hook = model.market_relation_encoder.register_forward_hook(capture)
    logits, value = model(batch)
    hook.remove()
    assert len(inputs) == 5  # three market tiers, self reserves, opponent reserves
    for cards, actual in zip((own, opponent), inputs[-2:]):
        public = torch.tensor([[True, False, False]])
        expected = market_purchase_relations(cards, public, global_features[:, model.opponent_indices])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert actual[:, 0, :5].sum() > 0
        assert torch.count_nonzero(actual[:, 1:]) == 0
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    (logits[:, 0].square().sum() + (value - 1).square().sum()).backward()
    assert all(output.grad[:, 0].abs().sum() > 0 for output in outputs[-2:])


@pytest.mark.parametrize('changes', [{'market_relations': 1}, {'market_relations': 'true'},
                                    {'value_readout': 'unknown'}])
@pytest.mark.parametrize('architecture', ['entity_attention_v1', 'entity_attention_aux_v1'])
def test_invalid_relation_options_fail(changes, architecture):
    with pytest.raises(ValueError):
        build_actor_critic({'architecture': architecture, **changes})
