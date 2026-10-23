"""Continue mixed-budget fitting with inherited AdamW/RNG and interval validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from .alphazero_checkpoint import load_alphazero_checkpoint
from .alphazero_recording import read_json, write_json
from .alphazero_train import AlphazeroLearner, LearnerConfig
from .offline_mixed_budget import make_batch_drawer, prepare_sampling
from .offline_objectives import OfflineObjectiveConfig, freeze_inputs, run as run_objectives
from .offline_scratch import select_checkpoint as select_scratch_checkpoint


@dataclass(frozen=True)
class FinetuneOfflineConfig(OfflineObjectiveConfig):
    early_stopping_increases: int = 3
    validation_interval_steps: int = 2000
    sampling_protocol: str = "policy-high-value-all-v1"

    def __post_init__(self):
        super().__post_init__()
        if self.policy_surprise_weighting is not None or self.value_auxiliary is not None:
            raise ValueError("mixed-budget continuation requires PSW and auxiliary targets disabled")
        if self.sampling_protocol != "policy-high-value-all-v1":
            raise ValueError("unknown continuation sampling protocol")
        for name in ("early_stopping_increases", "validation_interval_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


def prepare_inputs(config, output):
    prepared = prepare_sampling(config, output, freeze_inputs(config, output))
    plan = prepared[-1]
    steps, interval = plan["steps_target"], config.validation_interval_steps
    plan.update(validation_steps=list(range(interval, steps, interval)) + [steps],
        validation_interval_steps=interval,
        initialization={"kind": "checkpoint_continuation", "preserve_optimizer": True,
                        "preserve_sampling_rng": True, "preserve_torch_rng": True,
                        "budget": "additional draw exposures; local step resets, total step continues"})
    write_json(output / "plan.json", plan)
    return prepared


def initialize_learner(config, identity, steps):
    parent = load_alphazero_checkpoint(config.initial_checkpoint)
    requested = asdict(config.learner)
    if any(value != parent.learner_config[name] for name, value in requested.items() if name != "learning_rate"):
        raise ValueError("continuation must preserve parent learner settings except learning_rate")
    learner = AlphazeroLearner.resume(config.initial_checkpoint, LearnerConfig(**parent.learner_config),
        replay_identity=parent.replay_identity, round_index=parent.round_index,
        round_steps_target=parent.round_steps_target)
    learner.config = config.learner
    for group in learner.optimizer.param_groups:
        group["lr"] = config.learner.learning_rate
    learner.round_index = 0
    learner.round_train_step = 0
    learner.round_steps_target = steps
    learner.replay_identity = identity
    learner.parent_checkpoint = {"path": config.initial_checkpoint, "sha256": config.initial_checkpoint_sha256,
        "total_train_step": parent.total_train_step, "round_train_step": parent.round_train_step,
        "preserve_optimizer": True, "preserve_sampling_rng": True, "preserve_torch_rng": True,
        "learning_rate": config.learner.learning_rate}
    return learner


def run(config, output, *, prepare_only=False, max_steps_this_session=None):
    return run_objectives(config, output, prepare_only=prepare_only,
        max_steps_this_session=max_steps_this_session, input_preparer=prepare_inputs,
        learner_initializer=initialize_learner, validation_interval_steps=config.validation_interval_steps,
        batch_drawer_factory=lambda index, value: make_batch_drawer(index, value, config.learner.batch_size),
        early_stopping_increases=config.early_stopping_increases)


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
    config = FinetuneOfflineConfig.from_dict(read_json(args.config))
    if args.select_only:
        selection = select_checkpoint(config, args.output)
        print({"selected_step": selection["selected_step"], "objective": selection["selection_objective"]}, flush=True)
    else:
        run(config, args.output, prepare_only=args.prepare_only, max_steps_this_session=args.max_steps_this_session)
        if not args.prepare_only and read_json(args.output / "progress.json")["status"] == "complete":
            selection = select_checkpoint(config, args.output)
            print({"selected_step": selection["selected_step"], "objective": selection["selection_objective"]}, flush=True)


if __name__ == "__main__":
    main()
