"""Mixed-budget scratch fitting with consecutive full-holdout deterioration stopping."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .alphazero_recording import read_json
from .offline_mixed_budget import MixedBudgetOfflineConfig, make_batch_drawer, prepare_inputs, select_checkpoint
from .offline_objectives import run as run_objectives
from .offline_scratch import initialize_learner


@dataclass(frozen=True)
class EarlyStoppingOfflineConfig(MixedBudgetOfflineConfig):
    early_stopping_increases: int = 3

    def __post_init__(self):
        super().__post_init__()
        if type(self.early_stopping_increases) is not int or self.early_stopping_increases <= 0:
            raise ValueError("early_stopping_increases must be a positive integer")


def run(config, output, *, prepare_only=False, max_steps_this_session=None):
    return run_objectives(config, output, prepare_only=prepare_only,
        max_steps_this_session=max_steps_this_session, input_preparer=prepare_inputs,
        learner_initializer=initialize_learner, validation_percent_step=config.validation_percent_step,
        batch_drawer_factory=lambda index, value: make_batch_drawer(index, value, config.learner.batch_size),
        early_stopping_increases=config.early_stopping_increases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    config = EarlyStoppingOfflineConfig.from_dict(read_json(args.config))
    if args.select_only:
        selected = select_checkpoint(config, args.output)
        print({"selected_step": selected["selected_step"], "objective": selected["selection_objective"]}, flush=True)
    else:
        run(config, args.output, prepare_only=args.prepare_only,
            max_steps_this_session=args.max_steps_this_session)


if __name__ == "__main__":
    main()
