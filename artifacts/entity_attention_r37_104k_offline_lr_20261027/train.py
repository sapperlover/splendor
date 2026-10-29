"""R37 weights with fresh offline AdamW/RNG; original percentage validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from splendor.actor_critic import build_actor_critic, model_config_from_dict
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json
from splendor.alphazero_train import AlphazeroLearner
from splendor.offline_early_stopping import EarlyStoppingOfflineConfig
from splendor.offline_mixed_budget import make_batch_drawer, prepare_sampling
from splendor.offline_objectives import freeze_history_inputs, run as run_objectives
from splendor.offline_scratch import select_checkpoint as select_scratch, validation_schedule
from splendor.rng import DeterministicRng


@dataclass(frozen=True)
class R37OfflineConfig(EarlyStoppingOfflineConfig):
    initial_checkpoint: str = ""
    initial_checkpoint_sha256: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.initial_checkpoint or len(self.initial_checkpoint_sha256) != 64:
            raise ValueError("R37 checkpoint path and SHA256 are required")
        if self.model_config.get("architecture") != "entity_attention_v1":
            raise ValueError("R37 offline fitting uses the base attention model without auxiliaries")


def prepare_inputs(config, output):
    prepared = freeze_history_inputs(config, output, initialization={
        "initial_checkpoint": config.initial_checkpoint, "initial_checkpoint_sha256": config.initial_checkpoint_sha256,
        "initialization": {"kind": "checkpoint_weights_only", "optimizer": "fresh AdamW",
            "preserve_optimizer": False, "preserve_sampling_rng": False, "preserve_torch_rng": False,
            "initialization_seed": config.initialization_seed, "sampling_seed": config.sampling_seed,
            "model_config": config.model_config}})
    prepared[-1]["validation_steps"] = validation_schedule(config, prepared[-1]["steps_target"])
    return prepare_sampling(config, output, prepared)


def initialize_learner(config, identity, steps):
    # Consume the same initialization RNG as the offline base model, then load R37 weights.
    torch.manual_seed(config.initialization_seed)
    model = build_actor_critic(config.model_config)
    parent = load_alphazero_checkpoint(config.initial_checkpoint)
    if asdict(model_config_from_dict(parent.model_config)) != asdict(model.config):
        raise ValueError("R37 model architecture differs from the frozen offline model")
    model.load_state_dict(parent.model_state_dict, strict=True)
    learner = AlphazeroLearner(model, config.learner, sampling_rng=DeterministicRng(config.sampling_seed))
    learner.replay_identity = identity
    learner.round_steps_target = steps
    learner.parent_checkpoint = {"path": config.initial_checkpoint, "sha256": config.initial_checkpoint_sha256,
        "kind": "checkpoint_weights_only", "source_total_train_step": parent.total_train_step,
        "optimizer": "fresh AdamW", "preserve_optimizer": False, "preserve_sampling_rng": False,
        "preserve_torch_rng": False, "initialization_seed": config.initialization_seed,
        "sampling_seed": config.sampling_seed}
    return learner


def run(config, output, *, prepare_only=False, max_steps_this_session=None):
    return run_objectives(config, output, prepare_only=prepare_only, max_steps_this_session=max_steps_this_session,
        input_preparer=prepare_inputs, learner_initializer=initialize_learner,
        validation_percent_step=config.validation_percent_step,
        batch_drawer_factory=lambda index, value: make_batch_drawer(index, value, config.learner.batch_size),
        early_stopping_increases=config.early_stopping_increases)


def select_checkpoint(config, output):
    return select_scratch(config, output, input_preparer=prepare_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    config = R37OfflineConfig.from_dict(read_json(args.config))
    if args.select_only:
        result = select_checkpoint(config, args.output)
        print({"selected_step": result["selected_step"], "objective": result["selection_objective"]}, flush=True)
    else:
        run(config, args.output, prepare_only=args.prepare_only, max_steps_this_session=args.max_steps_this_session)


if __name__ == "__main__":
    main()
