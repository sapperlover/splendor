from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch

import splendor.actor_critic as actor_critic_module
from splendor.actions import (
    BuyMarket,
    ClaimNoble,
    Pay,
    ReserveMarket,
    ReturnTokens,
    TakeDistinct,
)
from splendor.actor_critic import (
    ActorCritic,
    ActorCriticConfig,
    CardEncoder,
    NobleEncoder,
    TorchDecisionBatch,
    tensorize_batch,
)
from splendor.contracts import DecisionRequest, MarketView, PendingObservation
from splendor.engine import GameEngine
from splendor.features import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_VERSION,
    collate_encoded_decisions,
    encode_decision,
)
from splendor.model import Color, DecisionPhase, TokenVector
from splendor.observation import build_observation
from splendor.random_policy import RandomPolicy
from splendor.result import TerminalResult
from splendor.rules import legal_actions


def _main_encoded(initial_state, catalog, *, one_candidate: bool = False):
    observation = build_observation(initial_state, 0, catalog)
    actions = legal_actions(initial_state, catalog)
    if one_candidate:
        actions = (TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),)
    return encode_decision(DecisionRequest(observation, actions))


def _four_phase_encoded(initial_state, catalog):
    base = build_observation(initial_state, 0, catalog)
    reserve = next(
        action
        for action in legal_actions(initial_state, catalog)
        if isinstance(action, ReserveMarket)
    )
    main = encode_decision(DecisionRequest(base, (reserve,)))

    payment_card = base.market.tier_one[0]
    payment_observation = replace(
        base,
        phase=DecisionPhase.PAYMENT,
        pending=PendingObservation(
            main_action=BuyMarket(payment_card.card_id),
            payment_target=payment_card,
        ),
    )
    payment = encode_decision(
        DecisionRequest(payment_observation, (Pay(TokenVector()),))
    )

    return_observation = replace(
        base,
        phase=DecisionPhase.RETURN_TOKENS,
        pending=PendingObservation(
            main_action=TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
            tokens_to_return=1,
        ),
    )
    returning = encode_decision(
        DecisionRequest(return_observation, (ReturnTokens(TokenVector(white=1)),))
    )

    noble = base.nobles_in_play[0]
    noble_observation = replace(
        base,
        phase=DecisionPhase.CHOOSE_NOBLE,
        pending=PendingObservation(
            main_action=TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
            eligible_nobles=(noble,),
        ),
    )
    choosing = encode_decision(
        DecisionRequest(noble_observation, (ClaimNoble(noble.noble_id),))
    )
    return main, payment, returning, choosing


def _fill_linear_parameters_positive(model: ActorCritic) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(0.01)
                module.bias.fill_(0.01)


def test_architecture_dimensions_and_parameter_count() -> None:
    model = ActorCritic()

    assert isinstance(model.card_encoder, CardEncoder)
    assert isinstance(model.noble_encoder, NobleEncoder)
    assert model.card_encoder.network[0].in_features == 29
    assert model.card_encoder.network[0].out_features == 64
    assert model.card_encoder.network[2].in_features == 64
    assert model.card_encoder.network[2].out_features == 64
    assert model.noble_encoder.network[0].in_features == 17
    assert model.noble_encoder.network[0].out_features == 32
    assert model.noble_encoder.network[2].in_features == 32
    assert model.noble_encoder.network[2].out_features == 32
    assert model.state_input_dim == 570
    assert model.action_meta_dim == 36
    assert model.action_representation_dim == 132
    assert model.actor_input_dim == 388
    assert model.trainable_parameter_count() == 302_626


def test_tensorize_and_cpu_forward_support_ragged_batch(initial_state, catalog) -> None:
    torch.manual_seed(4)
    one = _main_encoded(initial_state, catalog, one_candidate=True)
    many = _main_encoded(initial_state, catalog)
    numpy_batch = collate_encoded_decisions((one, many))
    batch = tensorize_batch(numpy_batch)

    assert isinstance(batch, TorchDecisionBatch)
    assert batch.global_features.dtype is torch.float32
    assert batch.candidate_mask.dtype is torch.bool
    assert batch.candidate_features.shape[1] == len(many.candidate_actions)
    assert not hasattr(batch, "candidate_actions")

    model = ActorCritic().eval()
    logits, values = model(batch)
    assert logits.shape == batch.candidate_mask.shape
    assert values.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()

    distribution, distribution_values = model.distribution(batch)
    probabilities = distribution.probs
    torch.testing.assert_close(distribution_values, values)
    assert probabilities[0, 0].item() == 1.0
    assert torch.count_nonzero(probabilities[0, 1:]).item() == 0
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(2), rtol=0, atol=1e-7
    )


def test_candidate_padding_mask_rejects_values_and_zero_candidate_rows(
    initial_state, catalog
) -> None:
    torch.manual_seed(5)
    one = _main_encoded(initial_state, catalog, one_candidate=True)
    many = _main_encoded(initial_state, catalog)
    batch = tensorize_batch(collate_encoded_decisions((one, many)))
    model = ActorCritic().eval()

    base_distribution, base_values = model.distribution(batch)
    changed_features = batch.candidate_features.clone()
    changed_features[~batch.candidate_mask] = 1_000_000.0
    changed = replace(batch, candidate_features=changed_features)
    changed_distribution, changed_values = model.distribution(changed)

    assert torch.count_nonzero(changed_distribution.probs[~batch.candidate_mask]).item() == 0
    torch.testing.assert_close(
        changed_distribution.probs[batch.candidate_mask],
        base_distribution.probs[batch.candidate_mask],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(changed_values, base_values, rtol=0, atol=0)

    empty = replace(
        batch,
        candidate_features=batch.candidate_features[:, :0],
        candidate_mask=batch.candidate_mask[:, :0],
    )
    with pytest.raises(ValueError, match="at least one legal candidate"):
        model(empty)


def test_all_set_and_pending_padding_is_output_invariant(initial_state, catalog) -> None:
    base = build_observation(initial_state, 0, catalog)
    empty_observation = replace(
        base,
        market=MarketView((), (), ()),
        nobles_in_play=(),
    )
    encoded = encode_decision(
        DecisionRequest(
            empty_observation,
            (TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),),
        )
    )
    batch = tensorize_batch(collate_encoded_decisions((encoded,)))
    model = ActorCritic().eval()
    base_logits, base_values = model(batch)

    replacements = {}
    for feature_name, mask_name in (
        ("market_t1", "market_t1_mask"),
        ("market_t2", "market_t2_mask"),
        ("market_t3", "market_t3_mask"),
        ("self_reserved", "self_reserved_mask"),
        ("opponent_reserved", "opponent_reserved_mask"),
        ("nobles", "nobles_mask"),
    ):
        features = getattr(batch, feature_name).clone()
        features[~getattr(batch, mask_name)] = 1_000_000.0
        replacements[feature_name] = features
    pending = batch.pending_card.clone()
    pending[~batch.pending_card_mask] = 1_000_000.0
    replacements["pending_card"] = pending
    changed = replace(batch, **replacements)

    changed_logits, changed_values = model(changed)
    torch.testing.assert_close(changed_logits, base_logits, rtol=0, atol=0)
    torch.testing.assert_close(changed_values, base_values, rtol=0, atol=0)


def test_all_decision_phases_pending_and_unknown_card_forward(initial_state, catalog) -> None:
    encoded = _four_phase_encoded(initial_state, catalog)
    batch = tensorize_batch(collate_encoded_decisions(encoded))
    model = ActorCritic().eval()
    logits, values = model(batch)

    assert logits.shape == batch.candidate_mask.shape
    assert values.shape == (4,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    assert batch.pending_card_mask.tolist() == [False, True, False, False]


def test_real_51_candidate_request_forward(tmp_path) -> None:
    engine = GameEngine.new(24, failure_dir=tmp_path)
    policies = (RandomPolicy(49), RandomPolicy(50))
    largest = None
    while True:
        decision = engine.decision()
        if isinstance(decision, TerminalResult):
            break
        if largest is None or len(decision.legal_actions) > len(largest.legal_actions):
            largest = decision
        engine.apply(policies[engine.current_player].choose(decision))

    assert largest is not None
    assert largest.observation.phase is DecisionPhase.RETURN_TOKENS
    assert len(largest.legal_actions) == 51
    batch = collate_encoded_decisions((encode_decision(largest),))
    logits, values = ActorCritic().eval()(batch)
    assert logits.shape == (1, 51)
    assert values.shape == (1,)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()


def test_critic_does_not_read_candidates(initial_state, catalog) -> None:
    encoded = _main_encoded(initial_state, catalog)
    batch = tensorize_batch(collate_encoded_decisions((encoded,)))
    model = ActorCritic().eval()
    _, value = model(batch)
    changed = replace(
        batch,
        candidate_features=batch.candidate_features + 100.0,
    )
    _, changed_value = model(changed)
    torch.testing.assert_close(changed_value, value, rtol=0, atol=0)


def test_shared_encoders_are_used_for_all_state_and_action_regions(
    initial_state, catalog
) -> None:
    main, _, _, noble = _four_phase_encoded(initial_state, catalog)
    batch = tensorize_batch(collate_encoded_decisions((main, noble)))
    model = ActorCritic().eval()
    card_calls: list[int] = []
    noble_calls: list[int] = []
    card_hook = model.card_encoder.register_forward_hook(
        lambda module, inputs, output: card_calls.append(id(module))
    )
    noble_hook = model.noble_encoder.register_forward_hook(
        lambda module, inputs, output: noble_calls.append(id(module))
    )
    try:
        model(batch)
    finally:
        card_hook.remove()
        noble_hook.remove()

    assert card_calls == [id(model.card_encoder)] * 7
    assert noble_calls == [id(model.noble_encoder)] * 2


def test_absent_action_targets_have_exactly_zero_learned_embeddings(
    initial_state, catalog
) -> None:
    encoded = _main_encoded(initial_state, catalog, one_candidate=True)
    batch = tensorize_batch(collate_encoded_decisions((encoded,)))
    model = ActorCritic().eval()
    representations = model.encode_actions(batch.candidate_features)

    card_start = model.action_meta_dim
    noble_start = card_start + model.config.card_hidden
    assert torch.count_nonzero(representations[..., card_start:noble_start]).item() == 0
    assert torch.count_nonzero(representations[..., noble_start:]).item() == 0


def test_gradient_reaches_every_module_and_candidate_card_path(initial_state, catalog) -> None:
    main, _, _, noble = _four_phase_encoded(initial_state, catalog)
    batch = tensorize_batch(collate_encoded_decisions((main, noble)))
    model = ActorCritic()
    _fill_linear_parameters_positive(model)
    logits, values = model(batch)
    loss = logits.masked_select(batch.candidate_mask).sum() + values.sum()
    loss.backward()

    for module in (
        model.card_encoder,
        model.noble_encoder,
        model.state_encoder,
        model.actor,
        model.critic,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0

    model.zero_grad(set_to_none=True)
    action_embeddings = model.encode_actions(batch.candidate_features)
    card_slice = slice(
        model.action_meta_dim,
        model.action_meta_dim + model.config.card_hidden,
    )
    action_embeddings[..., card_slice].sum().backward()
    assert sum(
        parameter.grad.abs().sum().item()
        for parameter in model.card_encoder.parameters()
        if parameter.grad is not None
    ) > 0


def test_save_load_preserves_config_schema_and_outputs(initial_state, catalog, tmp_path) -> None:
    torch.manual_seed(9)
    batch = collate_encoded_decisions((_main_encoded(initial_state, catalog),))
    model = ActorCritic().eval()
    expected_logits, expected_values = model(batch)
    path = tmp_path / "actor-critic-weights.pt"
    model.save_weights(path)
    restored = ActorCritic.load_weights(path, map_location="cpu").eval()
    actual_logits, actual_values = restored(batch)

    assert restored.config == model.config
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)


def test_schema_mismatch_and_model_import_boundary(initial_state, catalog) -> None:
    with pytest.raises(ValueError, match="schema version"):
        ActorCritic(
            ActorCriticConfig(feature_schema_version="obsolete-feature-schema")
        )

    batch = tensorize_batch(
        collate_encoded_decisions((_main_encoded(initial_state, catalog),))
    )
    mismatched = replace(batch, feature_schema_version="obsolete-feature-schema")
    with pytest.raises(ValueError, match="schema version"):
        ActorCritic()(mismatched)

    source = inspect.getsource(actor_critic_module)
    for forbidden in (
        "from ._state",
        "from .engine",
        "from .rules",
        "from .rng",
        "from .replay",
        "GameState",
    ):
        assert forbidden not in source
    assert FEATURE_SCHEMA_VERSION == "splendor-observation-action-v1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_and_distribution(initial_state, catalog) -> None:
    one = _main_encoded(initial_state, catalog, one_candidate=True)
    many = _main_encoded(initial_state, catalog)
    batch = collate_encoded_decisions((one, many))
    model = ActorCritic().cuda().eval()

    logits, values = model(batch)
    distribution, distribution_values = model.distribution(batch)
    assert logits.is_cuda and values.is_cuda
    assert distribution.probs.is_cuda and distribution_values.is_cuda
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    assert distribution.probs[0, 0].item() == 1.0
    assert torch.count_nonzero(distribution.probs[0, 1:]).item() == 0
