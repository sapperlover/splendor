"""Diagnostic-only updates and an explicitly composed frozen policy/value model."""
from __future__ import annotations

import math
from dataclasses import asdict

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from splendor.alphazero_checkpoint import load_alphazero_checkpoint, move_optimizer_state
from splendor.alphazero_train import (AlphazeroLearner, LearnerConfig, _torch_batch,
                                      alphazero_loss_sums, label_statistics, metric_ratios)
from splendor.rng import DeterministicRng

ARMS = ("joint", "policy_only", "value_detached")
POLICY_PREFIXES = ("action_encoder.", "query_norm.", "memory_norm.", "cross_attention.", "actor.")


def restored_learner(checkpoint, device):
    payload = load_alphazero_checkpoint(checkpoint, device=device)
    config = LearnerConfig(**(payload.learner_config | {"device": device}))
    learner = AlphazeroLearner(payload.build_model(device=device), config,
                               sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
    learner.optimizer.load_state_dict(payload.optimizer_state_dict)
    move_optimizer_state(learner.optimizer, learner.device)
    learner.round_index = payload.round_index
    learner.total_train_step = payload.total_train_step
    learner.round_train_step = 0
    learner.round_steps_target = 500
    torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
    if device == "cuda":
        torch.cuda.set_rng_state_all([s.cpu() for s in payload.torch_cuda_rng_states])
    return learner, payload


def terms_for(learner, batch, *, detach_value=False):
    model = learner.model
    tensors = _torch_batch(batch, learner.device)
    prepared = model._prepare(batch.decisions)
    state = model.encode_state(prepared)
    logits = model._actor_logits(prepared, state)
    value_state = (state[0].detach(), state[1]) if detach_value else state
    values = model.critic(value_state).squeeze(-1)
    terms = alphazero_loss_sums(
        logits, values, candidate_mask=tensors["candidate_mask"],
        policy_targets=tensors["policy_targets"], value_targets=tensors["value_targets"],
        policy_weights=tensors["policy_weights"], value_weights=tensors["value_weights"])
    return terms, logits, values


def diagnostic_step(learner, batch, arm):
    assert arm in ARMS
    learner.model.train()
    terms, logits, values = terms_for(learner, batch, detach_value=arm == "value_detached")
    loss = terms.policy_loss if arm == "policy_only" else terms.total(learner.config)
    assert bool(torch.isfinite(logits).all() & torch.isfinite(values).all() & torch.isfinite(loss))
    learner.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(clip_grad_norm_(learner.model.parameters(), learner.config.max_grad_norm))
    assert math.isfinite(grad_norm)
    learner.optimizer.step()
    learner.round_train_step += 1
    learner.total_train_step += 1
    scalars = torch.stack((terms.policy_ce_sum, terms.policy_weight_sum, terms.value_sq_sum,
                           terms.value_weight_sum, terms.entropy_sum, loss)).detach().cpu().tolist()
    row = dict(zip(("policy_ce_sum", "policy_weight_sum", "value_sq_sum", "value_weight_sum",
                    "entropy_sum", "objective_loss"), scalars)) | label_statistics(batch)
    return row | metric_ratios(row) | {"step": learner.round_train_step,
        "total_train_step": learner.total_train_step, "grad_norm": grad_norm,
        "grad_clipped": grad_norm > learner.config.max_grad_norm}


def shared_parameters(model):
    return [p for name, p in model.named_parameters()
            if not name.startswith(("critic.", *POLICY_PREFIXES))]


def gradient_probe(learner, batch):
    """Hypothetical unblocked joint gradient, including on the detached arm."""
    params = shared_parameters(learner.model)
    terms, _, _ = terms_for(learner, batch)
    gp = torch.autograd.grad(terms.policy_loss, params, retain_graph=True)
    gv = torch.autograd.grad(terms.value_loss, params)
    dot = sum((a * b).sum() for a, b in zip(gp, gv))
    npol = torch.sqrt(sum((a * a).sum() for a in gp))
    nval = torch.sqrt(sum((a * a).sum() for a in gv))
    cosine = dot / (npol * nval).clamp_min(1e-20)
    return dict(zip(("dot", "policy_norm", "value_norm", "cosine"),
                    torch.stack((dot, npol, nval, cosine)).detach().cpu().tolist()))


class FixedCritic(nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.head = critic

    def forward(self, state):
        return self.head(state[1])


class FixedValueModel(nn.Module):
    """Use one complete policy network and another complete frozen value function."""
    def __init__(self, policy_model, value_model):
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model
        self.critic = FixedCritic(value_model.critic)
        self.config = policy_model.config
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_state(self, prepared):
        return self.policy_model.encode_state(prepared), self.value_model.encode_state(prepared)

    def _actor_logits(self, prepared, state):
        return self.policy_model._actor_logits(prepared, state[0])

    def _forward_prepared(self, prepared, *, projection=None):
        assert projection is None
        state = self.encode_state(prepared)
        return self._actor_logits(prepared, state), self.critic(state).squeeze(-1)

    def _policy_logits_prepared(self, prepared, *, projection=None):
        assert projection is None
        return self.policy_model._policy_logits_prepared(prepared)

    def build_actor_projection(self):
        raise ValueError("fixed-value attention diagnostic requires actor_projection=false")

    def forward(self, batch):
        return self._forward_prepared(self.policy_model._prepare(batch))


def actual_learner_config(learner, arm):
    return asdict(learner.config) | {"value_loss_weight": 0.0 if arm == "policy_only" else 1.0}
