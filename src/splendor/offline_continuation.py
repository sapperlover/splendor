"""Fixed-pool attention fitting with shortterm value and optional opponent policy."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .actor_critic import build_actor_critic
from .alphazero_recording import read_json
from .alphazero_train import AlphazeroLearner
from .continuation_auxiliary import (ContinuationAuxiliaryConfig, EntityAttentionContinuationConfig,
                                     add_continuation_heads, continuation_config_identity)
from .offline_early_stopping import EarlyStoppingOfflineConfig
from .offline_mixed_budget import make_batch_drawer, prepare_sampling
from .offline_objectives import freeze_history_inputs, run as run_objectives
from .offline_scratch import select_checkpoint as select_scratch_checkpoint, validation_schedule
from .rng import DeterministicRng


@dataclass(frozen=True)
class ContinuationOfflineConfig(EarlyStoppingOfflineConfig):
    continuation_auxiliary: dict = field(default_factory=lambda: continuation_config_identity(ContinuationAuxiliaryConfig()))
    encoder_grad_norm_every_steps: int = 100

    def __post_init__(self):
        super().__post_init__()
        if self.model_config.get("architecture") != "entity_attention_v1":
            raise ValueError("continuation fitting requires a base entity_attention_v1 config")
        ContinuationAuxiliaryConfig(**self.continuation_auxiliary)
        if type(self.encoder_grad_norm_every_steps) is not int or self.encoder_grad_norm_every_steps < 0:
            raise ValueError("encoder_grad_norm_every_steps must be a nonnegative integer")


def model_config(config):
    # Parse defaults without allocating weights or consuming the initialization RNG.
    from .actor_critic import model_config_from_dict
    base = asdict(model_config_from_dict(config.model_config))
    auxiliary = ContinuationAuxiliaryConfig(**config.continuation_auxiliary)
    return EntityAttentionContinuationConfig(**(base | {"architecture": "entity_attention_continuation_v1",
        "predict_opponent_policy": auxiliary.opponent_policy_weight > 0,
        "predict_shortterm_value": auxiliary.shortterm_value_weight > 0,
        "predict_score_ownership": auxiliary.score_ownership_weight > 0}))


def initialization(config):
    return {"kind": "random_initialization", "seed": config.initialization_seed,
            "auxiliary_seed": config.auxiliary_initialization_seed, "model_config": asdict(model_config(config)),
            "optimizer": "fresh AdamW"}


def prepare_inputs(config, output):
    prepared = freeze_history_inputs(config, output, initialization={"initialization": initialization(config)})
    prepared[-1]["validation_steps"] = validation_schedule(config, prepared[-1]["steps_target"])
    return prepare_sampling(config, output, prepared)


def initialize_learner(config, identity, steps):
    torch.manual_seed(config.initialization_seed)
    base = build_actor_critic(config.model_config)
    torch.manual_seed(config.auxiliary_initialization_seed)
    auxiliary = ContinuationAuxiliaryConfig(**config.continuation_auxiliary)
    model = add_continuation_heads(base, predict_opponent_policy=auxiliary.opponent_policy_weight > 0,
                                  predict_shortterm_value=auxiliary.shortterm_value_weight > 0,
                                  predict_score_ownership=auxiliary.score_ownership_weight > 0)
    learner = AlphazeroLearner(model, config.learner, sampling_rng=DeterministicRng(config.sampling_seed))
    learner.replay_identity = identity
    learner.round_steps_target = steps
    learner.parent_checkpoint = initialization(config)
    return learner


def run(config, output, *, prepare_only=False, max_steps_this_session=None):
    return run_objectives(config, output, prepare_only=prepare_only,
        max_steps_this_session=max_steps_this_session, input_preparer=prepare_inputs,
        learner_initializer=initialize_learner, validation_percent_step=config.validation_percent_step,
        batch_drawer_factory=lambda index, value: make_batch_drawer(index, value, config.learner.batch_size),
        early_stopping_increases=config.early_stopping_increases,
        encoder_grad_norm_every_steps=config.encoder_grad_norm_every_steps)


def select_checkpoint(config, output):
    return select_scratch_checkpoint(config, output, input_preparer=prepare_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    config = ContinuationOfflineConfig.from_dict(read_json(args.config))
    if args.select_only:
        print(select_checkpoint(config, args.output), flush=True)
    else:
        run(config, args.output, prepare_only=args.prepare_only, max_steps_this_session=args.max_steps_this_session)


if __name__ == "__main__":
    main()
