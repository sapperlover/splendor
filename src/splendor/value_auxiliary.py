"""Terminal-only score and duration labels, never model input features."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np

from .entity_attention import EntityAttention, EntityAttentionAux, EntityAttentionAuxConfig


@dataclass(frozen=True)
class ValueAuxiliaryConfig:
    score_scale: float = 15.0
    turns_scale: float = 40.0
    score_loss_weight: float = 0.25
    turns_loss_weight: float = 0.25

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def auxiliary_targets(samples, config: ValueAuxiliaryConfig):
    """Current actor's final score difference; unfinished player turns to terminal.

    A turn is one player's complete main turn, including any payment/discard/
    noble decisions. Pending subdecisions do not advance the duration label.
    Both labels follow the archived continuation, just like terminal value z.
    """
    scores, turns = [], []
    for sample in samples:
        terminal = sample.metadata["terminal_result"]
        player = int(sample.arrays["player_id"][sample.index])
        remaining = sum(terminal["completed_turns"]) - int(sample.arrays["completed_player_turns"][sample.index])
        if remaining < 0:
            raise ValueError("terminal turn count precedes archived position")
        scores.append((terminal["scores"][player] - terminal["scores"][1 - player]) / config.score_scale)
        turns.append(remaining / config.turns_scale)
    return np.asarray(scores, dtype=np.float32), np.asarray(turns, dtype=np.float32)


def add_auxiliary_heads(base: EntityAttention):
    """Explicitly upgrade a base model, preserving every existing parameter."""
    if type(base) is not EntityAttention:
        raise ValueError("auxiliary warm start requires entity_attention_v1")
    config = asdict(base.config) | {"architecture": "entity_attention_aux_v1"}
    model = EntityAttentionAux(EntityAttentionAuxConfig(**config))
    state = model.state_dict()
    state.update(base.state_dict())
    model.load_state_dict(state, strict=True)
    return model


def extend_optimizer_state(old_state, base, model, optimizer):
    """Retain old AdamW moments; new heads start with empty optimizer state."""
    old_names = [name for name, _ in base.named_parameters()]
    new_names = [name for name, _ in model.named_parameters()]
    if new_names[:len(old_names)] != old_names or len(old_state["param_groups"]) != 1:
        raise ValueError("auxiliary upgrade requires the unchanged base parameter order and one AdamW group")
    state = copy.deepcopy(old_state)
    group = state["param_groups"][0]
    new_ids = optimizer.state_dict()["param_groups"][0]["params"]
    old_ids = group["params"]
    if len(old_ids) != len(old_names):
        raise ValueError("base optimizer parameter count mismatch")
    mapping = dict(zip(old_ids, new_ids[:len(old_names)], strict=True))
    state["state"] = {mapping[key]: value for key, value in state["state"].items()}
    group["params"] = new_ids
    optimizer.load_state_dict(state)
