"""Short-horizon progress supervision, derived only from archived continuations.

Protocol v1: MAIN_ACTION samples, horizons 2/4 completed player turns; scores
0..16 and per-color acquired bonuses 0..2. Early natural terminals are absorbing.
Labels are training-only arrays and never enter the model's state inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from .entity_attention import EntityAttention, EntityAttentionConfig, _value_head
from .features import GLOBAL_FEATURE_NAMES as G, SCALES

HORIZONS = (2, 4)
SCORE_CLASSES = 17
BONUS_CLASSES = 3
PROGRESS_OUTPUTS = len(HORIZONS) * 2 * (SCORE_CLASSES + 5 * BONUS_CLASSES)
PROTOCOL = "short-progress-main-h2-h4-v1"


@dataclass(frozen=True)
class ProgressAuxiliaryConfig:
    loss_weight: float = .05
    protocol: str = PROTOCOL

    def __post_init__(self):
        if self.protocol != PROTOCOL:
            raise ValueError("unknown progress auxiliary protocol")
        if (isinstance(self.loss_weight, bool) or not isinstance(self.loss_weight, (int, float))
                or not math.isfinite(self.loss_weight) or self.loss_weight <= 0):
            raise ValueError("progress loss_weight must be finite and positive")


@dataclass(frozen=True)
class EntityAttentionProgressConfig(EntityAttentionConfig):
    architecture: str = "entity_attention_progress_v1"

    def __post_init__(self):
        if self.architecture != "entity_attention_progress_v1":
            raise ValueError("invalid progress attention architecture")
        EntityAttentionConfig(**(asdict(self) | {"architecture": "entity_attention_v1"}))


class EntityAttentionProgress(EntityAttention):
    def __init__(self, config=None):
        config = config or EntityAttentionProgressConfig()
        super().__init__(EntityAttentionConfig(**(asdict(config) | {"architecture": "entity_attention_v1"})))
        self.config = config
        self.progress_head = _value_head(config.hidden, config.value_readout, PROGRESS_OUTPUTS)

    def value_with_progress(self, batch):
        state = self.encode_state(self._prepare(batch))
        return self.critic(state).squeeze(-1), self.progress_head(state)

    def forward_with_progress(self, batch):
        prepared = self._prepare(batch)
        state = self.encode_state(prepared)
        return self._actor_logits(prepared, state), self.critic(state).squeeze(-1), self.progress_head(state)


def add_progress_head(base):
    if type(base) is not EntityAttention:
        raise ValueError("progress head requires base entity attention")
    model = EntityAttentionProgress(EntityAttentionProgressConfig(
        **(asdict(base.config) | {"architecture": "entity_attention_progress_v1"})))
    state = model.state_dict()
    state.update(base.state_dict())
    model.load_state_dict(state, strict=True)
    return model


def game_progress_targets(arrays, metadata):
    """Build all labels once per decoded game, before compact-cache compression.

    A main-action snapshot is at a completed-turn boundary. Missing boundaries
    caused by automatic forced skips have unchanged resources, so the next main
    snapshot supplies that boundary. The last snapshot's bonuses are advanced
    through a final Pay, if present; terminal scores come from terminal_result.
    """
    g = arrays["global_features"]
    players = arrays["player_id"]
    turns = arrays["completed_player_turns"]
    if metadata.get("status") != "natural_terminal":
        raise ValueError("progress labels require a natural terminal")
    terminal = metadata["terminal_result"]
    if np.any(np.diff(turns) < 0) or sum(terminal["completed_turns"]) <= turns[-1]:
        raise ValueError("invalid completed-turn timeline")
    scores = np.empty((len(g), 2), dtype=np.int64)
    bonuses = np.empty((len(g), 2, 5), dtype=np.int64)
    for p in (0, 1):
        own = players == p
        scores[:, p] = np.rint(np.where(own, g[:, G.index("self_score_scaled")],
                                       g[:, G.index("opponent_score_scaled")]) * SCALES.player_score)
        for role, mask in (("self", own), ("opponent", ~own)):
            start = G.index(role + "_bonus_white_scaled")
            bonuses[mask, p] = np.rint(g[mask, start:start + 5] * SCALES.player_bonus)
    final_bonuses = bonuses[-1].copy()
    if g[-1, G.index("phase_payment")] > .5:
        # The purchase is applied by Pay, after the final recorded observation.
        pending = arrays["pending_card"][-1]
        if not arrays["pending_card_mask"][-1] or pending[1] != 1:
            raise ValueError("final payment lacks a known pending card")
        final_bonuses[players[-1]] += np.rint(pending[5:10]).astype(np.int64)
    final_scores = np.asarray(terminal["scores"], dtype=np.int64)
    main = np.flatnonzero(g[:, G.index("phase_main_action")] > .5)
    score_labels = np.zeros((len(g), 2, 2), dtype=np.int64)
    bonus_labels = np.zeros((len(g), 2, 2, 5), dtype=np.int64)
    mask = np.zeros(len(g), dtype=np.bool_)
    mask[main] = True
    for hindex, horizon in enumerate(HORIZONS):
        lookup = np.searchsorted(turns[main], turns[main] + horizon)
        future_scores = np.broadcast_to(final_scores, (len(main), 2)).copy()
        future_bonuses = np.broadcast_to(final_bonuses, (len(main), 2, 5)).copy()
        available = lookup < len(main)
        future_scores[available] = scores[main[lookup[available]]]
        future_bonuses[available] = bonuses[main[lookup[available]]]
        score_delta, bonus_delta = future_scores - scores[main], future_bonuses - bonuses[main]
        for relative in (0, 1):
            actor = (players[main] + relative) % 2
            score_labels[main, hindex, relative] = score_delta[np.arange(len(main)), actor]
            bonus_labels[main, hindex, relative] = bonus_delta[np.arange(len(main)), actor]
    if (np.any(score_labels < 0) or np.any(score_labels >= SCORE_CLASSES)
            or np.any(bonus_labels < 0) or np.any(bonus_labels >= BONUS_CLASSES)):
        raise ValueError("short progress labels violate the 2/4-turn bounds")
    return {"progress_score": score_labels.astype(np.int8),
            "progress_bonus": bonus_labels.astype(np.int8), "progress_mask": mask}


def progress_targets(samples):
    return tuple(np.stack([s.arrays[key][s.index] for s in samples])
                 for key in ("progress_score", "progress_bonus", "progress_mask"))


def progress_loss_sums(logits, labels, value_weights):
    score, bonus, mask = [torch.as_tensor(x, device=logits.device) for x in labels]
    expected = (logits.shape[0], len(HORIZONS), 2)
    if score.shape != expected or bonus.shape != (*expected, 5) or mask.shape != (expected[0],):
        raise ValueError("progress target shapes differ from the value batch")
    predictions = logits.reshape(*expected, SCORE_CLASSES + 5 * BONUS_CLASSES)
    score_logits = predictions[..., :SCORE_CLASSES]
    bonus_logits = predictions[..., SCORE_CLASSES:].reshape(*expected, 5, BONUS_CLASSES)
    score_ce = F.cross_entropy(score_logits.reshape(-1, SCORE_CLASSES), score.long().reshape(-1),
                              reduction="none").reshape(expected).mean((1, 2))
    bonus_ce = F.cross_entropy(bonus_logits.reshape(-1, BONUS_CLASSES), bonus.long().reshape(-1),
                              reduction="none").reshape(*expected, 5).mean((1, 2, 3))
    weight = value_weights * mask
    return {"progress_score_ce_sum": (score_ce * weight).sum(),
            "progress_bonus_ce_sum": (bonus_ce * weight).sum(),
            "progress_weight_sum": weight.sum()}


def progress_metrics(sums):
    denominator = max(float(sums["progress_weight_sum"]), 1e-12)
    score = float(sums["progress_score_ce_sum"]) / denominator
    bonus = float(sums["progress_bonus_ce_sum"]) / denominator
    return {"progress_score_ce": score, "progress_bonus_ce": bonus,
            "progress_loss": (score + bonus) / 2,
            "progress_positions": float(sums["progress_weight_sum"])}
