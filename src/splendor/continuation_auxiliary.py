"""Training-only continuation targets; public policy/value inference is unchanged."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .entity_attention import EntityAttention, EntityAttentionConfig, _value_head
from .actor_critic import tensorize_state
from .features import EncodedStateBatch, GLOBAL_FEATURE_NAMES as G

HORIZONS = (2, 6, 14)
LAMBDAS = np.asarray(HORIZONS, dtype=np.float64) / (1 + np.asarray(HORIZONS))
PROTOCOL = "continuation-ema-h2-h6-h14-opponent-main-v1"


@dataclass(frozen=True)
class ContinuationAuxiliaryConfig:
    shortterm_value_weight: float = .25
    opponent_policy_weight: float = 0.
    protocol: str = PROTOCOL
    score_ownership_weight: float = 0.

    def __post_init__(self):
        if self.protocol != PROTOCOL:
            raise ValueError("unknown continuation auxiliary protocol")
        if (isinstance(self.shortterm_value_weight, bool)
                or not isinstance(self.shortterm_value_weight, (int, float))
                or not math.isfinite(self.shortterm_value_weight) or self.shortterm_value_weight < 0):
            raise ValueError("shortterm_value_weight must be finite and nonnegative")
        if (isinstance(self.opponent_policy_weight, bool)
                or not isinstance(self.opponent_policy_weight, (int, float))
                or not math.isfinite(self.opponent_policy_weight) or self.opponent_policy_weight < 0):
            raise ValueError("opponent_policy_weight must be finite and nonnegative")
        if (isinstance(self.score_ownership_weight, bool)
                or not isinstance(self.score_ownership_weight, (int, float))
                or not math.isfinite(self.score_ownership_weight) or self.score_ownership_weight < 0):
            raise ValueError("score_ownership_weight must be finite and nonnegative")
        if self.shortterm_value_weight == self.opponent_policy_weight == self.score_ownership_weight == 0:
            raise ValueError("at least one continuation auxiliary must be enabled")


def continuation_config_identity(config):
    """Disabled new tasks do not change existing checkpoint/replay identities."""
    result = asdict(config)
    if not config.score_ownership_weight:
        result.pop("score_ownership_weight")
    return result


@dataclass(frozen=True)
class EntityAttentionContinuationConfig(EntityAttentionConfig):
    architecture: str = "entity_attention_continuation_v1"
    predict_opponent_policy: bool = False
    predict_shortterm_value: bool = True
    predict_score_ownership: bool = False

    def __post_init__(self):
        if self.architecture != "entity_attention_continuation_v1":
            raise ValueError("invalid continuation attention architecture")
        if type(self.predict_opponent_policy) is not bool:
            raise ValueError("predict_opponent_policy must be boolean")
        if type(self.predict_shortterm_value) is not bool:
            raise ValueError("predict_shortterm_value must be boolean")
        if type(self.predict_score_ownership) is not bool:
            raise ValueError("predict_score_ownership must be boolean")
        if not (self.predict_opponent_policy or self.predict_shortterm_value or self.predict_score_ownership):
            raise ValueError("at least one continuation head must be enabled")
        EntityAttentionConfig(**_base_config(self))


def _base_config(config):
    return {k: v for k, v in asdict(config).items()
            if k not in ("predict_opponent_policy", "predict_shortterm_value", "predict_score_ownership")} | {"architecture": "entity_attention_v1"}


class EntityAttentionContinuation(EntityAttention):
    def __init__(self, config=None):
        config = config or EntityAttentionContinuationConfig()
        super().__init__(EntityAttentionConfig(**_base_config(config)))
        self.config = config
        # Consume the same initialization draws in every arm, so opponent heads
        # match the existing dual-head preset. Disabled value parameters are discarded.
        shortterm_head = _value_head(config.hidden, config.value_readout, len(HORIZONS))
        if config.predict_shortterm_value:
            self.shortterm_value_head = shortterm_head
        if config.predict_opponent_policy:
            from .opponent_policy import OpponentPolicyHead
            self.opponent_policy_head = OpponentPolicyHead(config.hidden, config.value_readout)
        if config.predict_score_ownership:
            self.score_ownership_head = _value_head(config.hidden, config.value_readout, 8)

    def _continuation_predictions(self, prepared, state):
        result = {}
        if self.config.predict_shortterm_value:
            result["shortterm_value"] = self.shortterm_value_head(state).tanh()
        if self.config.predict_opponent_policy:
            result.update(self.opponent_policy_head(prepared, state))
        if self.config.predict_score_ownership:
            result["score_ownership"] = self.score_ownership_head(state)
        return result

    def value_with_continuation(self, batch):
        prepared = self._prepare(batch)
        state = self.encode_state(prepared)
        return self.critic(state).squeeze(-1), self._continuation_predictions(prepared, state)

    def value_with_continuation_from_state(self, batch):
        """Value/auxiliary path without transferring unused policy candidates."""
        prepared = (tensorize_state(batch, device=self.device)
                    if isinstance(batch, EncodedStateBatch) else batch.to(self.device))
        if prepared.feature_schema_version != self.config.feature_schema_version:
            raise ValueError("batch feature schema version does not match model configuration")
        state = self.encode_state(prepared)
        return self.critic(state).squeeze(-1), self._continuation_predictions(prepared, state)

    def forward_with_continuation(self, batch):
        prepared = self._prepare(batch)
        state = self.encode_state(prepared)
        return (self._actor_logits(prepared, state), self.critic(state).squeeze(-1),
                self._continuation_predictions(prepared, state))


def add_continuation_heads(base, *, predict_opponent_policy=False, predict_shortterm_value=True, predict_score_ownership=False):
    if type(base) is not EntityAttention:
        raise ValueError("continuation heads require base entity attention")
    model = EntityAttentionContinuation(EntityAttentionContinuationConfig(
        **(asdict(base.config) | {"architecture": "entity_attention_continuation_v1",
                                "predict_opponent_policy": predict_opponent_policy,
                                "predict_shortterm_value": predict_shortterm_value,
                                "predict_score_ownership": predict_score_ownership})))
    state = model.state_dict()
    state.update(base.state_dict())
    model.load_state_dict(state, strict=True)
    return model


def game_shortterm_value_targets(arrays, metadata):
    """Exact finite KataGo EMA, with terminal payoff absorbing the remaining tail.

    H is the mean forward offset: lambda=H/(H+1), including the current root.
    Only MAIN_ACTION boundaries count as player turns. A missing root/turn makes
    every earlier EMA unavailable, rather than substituting a network/terminal
    estimate. Valid suffixes are retained and coverage is reported by the loss.
    """
    if metadata.get("status") != "natural_terminal":
        raise ValueError("shortterm value labels require a natural terminal")
    if "per_world_root_value_sums" not in arrays:
        raise ValueError("shortterm value requires v3 per_world_root_value_sums")
    players = arrays["player_id"]
    turns = arrays["completed_player_turns"]
    main = np.flatnonzero(arrays["global_features"][:, G.index("phase_main_action")] > .5)
    terminal_turn = sum(metadata["terminal_result"]["completed_turns"])
    if not len(main) or np.any(np.diff(turns) < 0) or np.any(np.diff(turns[main]) <= 0) or terminal_turn <= turns[-1]:
        raise ValueError("invalid completed-turn timeline for shortterm value")
    sign = 1 - 2 * players.astype(np.int64)
    terminal = float(arrays["value_target"][-1] * sign[-1])
    tail = np.full(len(HORIZONS), terminal, dtype=np.float64)
    labels = np.zeros((len(players), len(HORIZONS)), dtype=np.float32)
    mask = np.zeros(len(players), dtype=np.bool_)
    available = True
    next_turn = terminal_turn
    for i in reversed(main):
        start, end = arrays["candidate_offsets"][i:i + 2]
        worlds = int(arrays["valid_world_count"][i])
        visits = arrays["per_world_root_visit_counts"][:worlds, start:end].sum(axis=1)
        available = available and turns[i] + 1 == next_turn and worlds > 0 and bool(np.all(visits > 0))
        if available:
            sums = arrays["per_world_root_value_sums"][:worlds, start:end].sum(axis=1)
            root = float(np.mean(sums / visits)) * sign[i]
            if not math.isfinite(root) or abs(root) > 1 + 1e-9:
                raise ValueError("root MCTS value is outside [-1, 1]")
            tail = (1 - LAMBDAS) * root + LAMBDAS * tail
            labels[i] = tail * sign[i]
            mask[i] = True
        next_turn = turns[i]
    main_mask = np.zeros(len(players), dtype=np.bool_)
    main_mask[main] = True
    return {"shortterm_value_target": labels, "shortterm_value_mask": mask, "shortterm_main_mask": main_mask}


def game_continuation_targets(arrays, metadata, config):
    result = game_shortterm_value_targets(arrays, metadata) if config.shortterm_value_weight > 0 else {}
    if config.opponent_policy_weight > 0:
        from .opponent_policy import game_opponent_policy_targets
        result.update(game_opponent_policy_targets(arrays, metadata))
    if config.score_ownership_weight > 0:
        from .score_ownership import game_score_ownership_targets
        result.update(game_score_ownership_targets(arrays, metadata))
    return result


def continuation_targets(samples, config):
    keys = (["shortterm_value_target", "shortterm_value_mask", "shortterm_main_mask"]
            if config.shortterm_value_weight > 0 else [])
    if config.opponent_policy_weight > 0:
        keys.extend(("opponent_action_target", "opponent_buy_target", "opponent_reserve_target", "opponent_policy_weight"))
    if config.score_ownership_weight > 0:
        keys.append("score_ownership_target")
    return {key: np.stack([s.arrays[key][s.index] for s in samples])
            for key in keys}


def continuation_loss_sums(predictions, labels, value_weights, config):
    sums = {}
    if config.shortterm_value_weight > 0:
        sums.update(shortterm_loss_sums(predictions, labels, value_weights))
    if config.opponent_policy_weight > 0:
        from .opponent_policy import opponent_loss_sums
        sums.update(opponent_loss_sums(predictions, labels, value_weights))
    if config.score_ownership_weight > 0:
        from .score_ownership import score_ownership_loss_sums
        sums.update(score_ownership_loss_sums(predictions, labels, value_weights))
    return sums


def shortterm_loss_sums(predictions, labels, value_weights):
    prediction = predictions["shortterm_value"]
    target = torch.as_tensor(labels["shortterm_value_target"], device=prediction.device)
    mask = torch.as_tensor(labels["shortterm_value_mask"], device=prediction.device)
    main_mask = torch.as_tensor(labels["shortterm_main_mask"], device=prediction.device)
    if (target.shape != prediction.shape or target.shape != (len(value_weights), len(HORIZONS))
            or mask.shape != value_weights.shape or main_mask.shape != value_weights.shape):
        raise ValueError("shortterm targets differ from the value batch")
    weight = value_weights * mask
    sums = {**{f"shortterm_h{h}_sq_sum": ((prediction[:, j] - target[:, j]).square() * weight).sum()
                for j, h in enumerate(HORIZONS)}, "shortterm_weight_sum": weight.sum(),
            "shortterm_eligible_weight_sum": (value_weights * main_mask).sum()}
    return sums


def continuation_loss(sums, config):
    loss = None
    if config.shortterm_value_weight > 0:
        loss = config.shortterm_value_weight * sum(sums[f"shortterm_h{h}_sq_sum"] for h in HORIZONS) / (
            len(HORIZONS) * sums["shortterm_weight_sum"].clamp_min(1e-12))
    if config.opponent_policy_weight > 0:
        opponent = config.opponent_policy_weight * sum(sums[f"opponent_{name}_ce_sum"] for name in ("action", "buy", "reserve")) / sums["opponent_weight_sum"].clamp_min(1e-12)
        loss = opponent if loss is None else loss + opponent
    if config.score_ownership_weight > 0:
        from .score_ownership import score_ownership_loss
        ownership = config.score_ownership_weight * score_ownership_loss(sums)
        loss = ownership if loss is None else loss + ownership
    return loss


def continuation_gradient_tasks(sums, config):
    """Raw normalized task losses and their coefficients in the training loss.

    Aggregate tasks overlap their components and are for diagnostics only.
    Training continues to use continuation_loss with its original arithmetic.
    """
    tasks = {}
    if config.shortterm_value_weight > 0:
        n = sums["shortterm_weight_sum"].clamp_min(1e-12)
        tasks.update({f"shortterm_h{h}": (sums[f"shortterm_h{h}_sq_sum"] / n,
                                        config.shortterm_value_weight / len(HORIZONS)) for h in HORIZONS})
        tasks["shortterm_value"] = (sum(sums[f"shortterm_h{h}_sq_sum"] for h in HORIZONS) / (len(HORIZONS)*n),
                                    config.shortterm_value_weight)
    if config.opponent_policy_weight > 0:
        n = sums["opponent_weight_sum"].clamp_min(1e-12)
        tasks.update({f"opponent_{name}": (sums[f"opponent_{name}_ce_sum"] / n, config.opponent_policy_weight)
                      for name in ("action", "buy", "reserve")})
        tasks["opponent_policy"] = (sum(sums[f"opponent_{name}_ce_sum"] for name in ("action", "buy", "reserve")) / n,
                                    config.opponent_policy_weight)
    if config.score_ownership_weight > 0:
        from .score_ownership import score_ownership_gradient_tasks
        tasks.update(score_ownership_gradient_tasks(sums, config.score_ownership_weight))
    return tasks


def continuation_metrics(sums, config):
    result = {}
    if config.shortterm_value_weight > 0:
        n = max(float(sums["shortterm_weight_sum"]), 1e-12)
        metrics = {f"shortterm_h{h}_mse": float(sums[f"shortterm_h{h}_sq_sum"]) / n for h in HORIZONS}
        result.update(metrics | {"shortterm_loss": sum(metrics.values()) / len(HORIZONS),
                            "shortterm_positions": float(sums["shortterm_weight_sum"]),
                            "shortterm_eligible_positions": float(sums["shortterm_eligible_weight_sum"]),
                            "shortterm_coverage": float(sums["shortterm_weight_sum"]) / max(float(sums["shortterm_eligible_weight_sum"]), 1e-12)})
    if config.opponent_policy_weight > 0:
        from .opponent_policy import opponent_metrics
        result.update(opponent_metrics(sums))
    if config.score_ownership_weight > 0:
        from .score_ownership import score_ownership_metrics
        result.update(score_ownership_metrics(sums))
    return result
