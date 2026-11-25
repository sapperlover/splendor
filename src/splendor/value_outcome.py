"""Scalar/WDL main value and same-player, next-main-action lambda returns.

Class order is always win, draw, loss. Search consumes E[z] = p(win)-p(loss).
Historical scalar root Qs are never interpreted as three-class probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .actor_critic import tensorize_state
from .alphazero_train import (AlphazeroLearner, NonFiniteTrainingError, _cpu_scalar_sums,
    _torch_batch, alphazero_loss_sums, label_statistics, metric_ratios)
from .entity_attention import EntityAttention, EntityAttentionConfig
from .features import EncodedStateBatch, GLOBAL_FEATURE_NAMES

WDL_ORDER = ('win', 'draw', 'loss')


def terminal_one_hot(values):
    values = np.asarray(values)
    if not np.isin(values, (-1, 0, 1)).all():
        raise ValueError('WDL labels require exact terminal payoffs in {-1,0,1}')
    return np.eye(3, dtype=np.float32)[(1-values).astype(np.int64)]


def vector_lambda_targets(probabilities, players, main_actions, terminal_values, lam):
    """Bootstrap only at a strictly later MAIN_ACTION of the same player.

    Payment/return/noble rows share the next-own-main-action clock. If no later
    own main action exists, the exact terminal outcome absorbs the entire tail.
    Input probabilities are frozen teacher predictions, detached from training.
    """
    p = np.asarray(probabilities, dtype=np.float32)
    players, main = np.asarray(players), np.asarray(main_actions, dtype=bool)
    n = len(players)
    if (p.shape != (n, 3) or main.shape != (n,) or not 0 <= lam <= 1
            or not np.isin(players, (0, 1)).all()):
        raise ValueError('Invalid vector lambda trajectory shapes/player/lambda')
    if (not np.isfinite(p[main]).all() or (p[main] < 0).any()
            or not np.allclose(p[main].sum(1), 1, atol=1e-5)):
        raise ValueError('Teacher WDL predictions must be probability vectors')
    onehot = terminal_one_hot(terminal_values)
    if onehot.shape != (n, 3):
        raise ValueError('Terminal payoff must be provided for each actor row')
    return _lambda_recursion(p, players, main, onehot, lam)[0]


def scalar_lambda_targets(values, players, main_actions, terminal_values, lam):
    """Fixed scalar returns, clipping the unbounded critic to the payoff range."""
    values = np.asarray(values, dtype=np.float32)
    players, main = np.asarray(players), np.asarray(main_actions, dtype=bool)
    terminal = np.asarray(terminal_values, dtype=np.float32)
    n = len(players)
    if (values.shape != (n,) or terminal.shape != (n,) or main.shape != (n,)
            or not np.isin(players, (0, 1)).all() or not 0 <= lam <= 1):
        raise ValueError('Invalid scalar lambda trajectory shapes/player/lambda')
    if not np.isin(terminal, (-1, 0, 1)).all():
        raise ValueError('Scalar lambda requires exact terminal payoffs')
    if not np.isfinite(values[main]).all():
        raise ValueError('Teacher scalar values must be finite')
    return _lambda_recursion(np.clip(values, -1, 1), players, main, terminal, lam)[0]


def archived_search_lambda_targets(arrays, metadata, lam):
    """Next-own-main lambda returns; missing roots mask dependent earlier targets."""
    from .continuation_auxiliary import archived_root_values
    if metadata.get('status') != 'natural_terminal':
        raise ValueError('Archived search lambda requires a natural terminal')
    if not 0 <= lam <= 1:
        raise ValueError('lambda must be in [0,1]')
    values, root_valid = archived_root_values(arrays)
    players = arrays['player_id']
    main = arrays['global_features'][:, GLOBAL_FEATURE_NAMES.index('phase_main_action')] > .5
    terminal = np.asarray(arrays['value_target'], dtype=np.float64)
    if (terminal.shape != values.shape or not np.isin(terminal, (-1, 0, 1)).all()
            or not np.isin(players, (0, 1)).all()):
        raise ValueError('Invalid archived player/terminal trajectory')
    targets, mask = _lambda_recursion(values, players, main, terminal, lam, root_valid)
    return targets.astype(np.float32), mask, root_valid


def _lambda_recursion(predictions, players, main, terminal, lam, bootstrap_valid=None):
    result = np.empty_like(predictions)
    valid = np.ones(len(players), dtype=np.bool_)
    future = {}
    for i in range(len(players)-1, -1, -1):
        actor = int(players[i])
        if actor in future:
            j = future[actor]
            if not np.array_equal(terminal[i], terminal[j]):
                raise ValueError('Terminal outcome changed within one player trajectory')
            if bootstrap_valid is not None:
                valid[i] = lam == 1 or (bootstrap_valid[j] and (lam == 0 or valid[j]))
            if not valid[i] or lam == 1:
                # Placeholder values for masked rows never enter the training loss.
                result[i] = terminal[i]
            else:
                result[i] = (1-lam)*predictions[j] + lam*result[j]
        else:
            result[i] = terminal[i]
        if main[i]:
            future[actor] = i
    return result, valid


@dataclass(frozen=True)
class EntityAttentionWDLConfig(EntityAttentionConfig):
    architecture: str = 'entity_attention_wdl_v1'

    def __post_init__(self):
        if self.architecture != 'entity_attention_wdl_v1':
            raise ValueError('Invalid WDL architecture')
        EntityAttentionConfig(**(asdict(self) | {'architecture': 'entity_attention_v1'}))


class WDLReadout(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        # Preserve the base initialization of trunk, actor and critic hidden
        # layers. Only the final scalar projection changes shape.
        last = scalar.network[-1]
        scalar.network[-1] = nn.Linear(last.in_features, 3)
        self.readout = scalar

    def logits(self, state):
        return self.readout(state)

    def forward(self, state):
        p = self.logits(state).softmax(-1)
        return (p[:, 0]-p[:, 2]).unsqueeze(-1)


class EntityAttentionWDL(EntityAttention):
    def __init__(self, config=None):
        config = config or EntityAttentionWDLConfig()
        super().__init__(EntityAttentionConfig(**(asdict(config) | {'architecture': 'entity_attention_v1'})))
        self.critic = WDLReadout(self.critic)
        self.config = config

    def wdl_logits_from_state(self, batch):
        state = tensorize_state(batch, device=self.device)
        if state.feature_schema_version != self.config.feature_schema_version:
            raise ValueError('WDL state feature schema mismatch')
        return self.critic.logits(self.encode_state(state))

    def forward_with_wdl(self, batch):
        prepared = self._prepare(batch)
        state = self.encode_state(prepared)
        logits = self.critic.logits(state)
        p = logits.softmax(-1)
        return self._actor_logits(prepared, state), p[:, 0]-p[:, 2], logits


class OutcomeLearner(AlphazeroLearner):
    """Fixed scalar-lambda/WDL update using the split-batch/streaming driver."""
    outcome_targets = None
    outcome_offsets = None
    outcome_target_mask = None
    outcome_target_source = 'network'
    outcome_teacher_step = None

    def train_step(self, batch, *, references, value_batch=None, **kwargs):
        wdl = self.model.config.architecture == 'entity_attention_wdl_v1'
        if not wdl and self.outcome_targets is None:
            return super().train_step(batch, references=references, value_batch=value_batch, **kwargs)
        if kwargs or value_batch is None:
            raise ValueError('Outcome fitting requires separate batches and no auxiliary objectives')
        self.model.train()
        p = _torch_batch(batch, self.device)
        actor_logits = self.model.policy_logits(batch.decisions)
        terms = alphazero_loss_sums(actor_logits, torch.zeros_like(p['value_targets']), **p)
        state = EncodedStateBatch(**{k: getattr(value_batch.decisions, k)
                                     for k in EncodedStateBatch.__dataclass_fields__})
        if wdl:
            outcome_logits = self.model.wdl_logits_from_state(state)
            probabilities = outcome_logits.softmax(-1)
            values = probabilities[:, 0]-probabilities[:, 2]
            labels = terminal_one_hot(value_batch.value_targets)
        else:
            values = self.model.value_from_state(state)
        if self.outcome_targets is not None:
            refs = references[len(batch.value_targets):]
            ids = [self.outcome_offsets[(r.round_index, r.game_id)]+r.decision_index for r in refs]
            labels = np.asarray(self.outcome_targets[ids])
        target = torch.as_tensor(labels, device=self.device)
        weight = torch.as_tensor(value_batch.value_weights, device=self.device)
        train_weight = weight
        if self.outcome_target_mask is not None:
            train_weight = weight * torch.as_tensor(self.outcome_target_mask[ids], device=self.device)
        denominator = train_weight.sum().clamp_min(torch.finfo(torch.float32).tiny)
        if wdl:
            train_sum = (-(target*outcome_logits.log_softmax(-1)).sum(-1)*train_weight).sum()
            metric = 'wdl_train_ce'
        else:
            train_sum = ((values-target).square()*train_weight).sum()
            metric = 'lambda_train_mse'
        z = torch.as_tensor(value_batch.value_targets, device=self.device)
        mse_sum = ((values-z).square()*weight).sum()
        total = self.config.policy_loss_weight*terms.policy_loss + self.config.value_loss_weight*train_sum/denominator
        finite = torch.isfinite(outcome_logits).all() if wdl else torch.isfinite(values).all()
        if not bool(torch.isfinite(actor_logits).all() & finite & torch.isfinite(total)):
            raise NonFiniteTrainingError('Nonfinite outcome update', details={'step': self.round_train_step})
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = float(clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm))
        if not np.isfinite(norm):
            raise NonFiniteTrainingError('Nonfinite outcome gradient', details={'step': self.round_train_step})
        self.optimizer.step()
        self.round_train_step += 1
        self.total_train_step += 1
        sums = _cpu_scalar_sums(dict(policy_ce_sum=terms.policy_ce_sum,
            policy_weight_sum=terms.policy_weight_sum, entropy_sum=terms.entropy_sum,
            value_sq_sum=mse_sum, value_weight_sum=weight.sum(), **{metric+'_sum': train_sum},
            outcome_train_weight_sum=train_weight.sum(),
            total_loss=total))
        sums.update(label_statistics(batch))
        sums.update({k:v for k,v in label_statistics(value_batch).items() if k.startswith('value_')})
        ratios = metric_ratios(sums)
        return sums | ratios | dict(event='train_step', round_index=self.round_index,
            round_train_step=self.round_train_step, total_train_step=self.total_train_step,
            grad_norm=norm, grad_clipped=norm>self.config.max_grad_norm,
            learning_rate=float(self.optimizer.param_groups[0]['lr']),
            batch_size=len(batch.value_targets)+len(value_batch.value_targets),
            policy_batch_size=len(batch.value_targets), value_batch_size=len(value_batch.value_targets),
            training_mode='joint_separate_batches',
            **{metric:sums[metric+'_sum']/sums['outcome_train_weight_sum'] if sums['outcome_train_weight_sum'] else None},
            outcome_train_coverage=sums['outcome_train_weight_sum']/sums['value_weight_sum'],
            outcome_target_source=self.outcome_target_source,
            value_objective=('wdl_lambda' if self.outcome_targets is not None else 'wdl_mc') if wdl else 'scalar_lambda',
            outcome_teacher_step=self.outcome_teacher_step)
