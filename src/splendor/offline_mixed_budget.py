"""Random-initialized fixed-pool fitting: high-budget policy, all-budget value."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .alphazero_recording import read_json, write_json
from .alphazero_train import DrawnBatch
from .offline_objectives import run as run_objectives
from .offline_scratch import (ScratchOfflineConfig, initialize_learner,
    prepare_inputs as prepare_scratch_inputs, select_checkpoint as select_scratch_checkpoint)
from .random_budget_iteration import PolicyPool


@dataclass(frozen=True)
class MixedBudgetOfflineConfig(ScratchOfflineConfig):
    value_auxiliary: dict | None = None
    sampling_protocol: str = "policy-high-value-all-v1"

    def __post_init__(self):
        super().__post_init__()
        if self.sampling_protocol != "policy-high-value-all-v1":
            raise ValueError("unknown mixed-budget offline sampling protocol")
        if self.value_auxiliary is not None:
            raise ValueError("mixed-budget offline fitting requires auxiliary targets disabled")


def prepare_inputs(config, output):
    return prepare_sampling(config, output, prepare_scratch_inputs(config, output))


def prepare_sampling(config, output, prepared):
    """Describe the shared policy-high/value-all sampling over frozen inputs."""
    index, value, identity, fixed, plan = prepared
    for split, counts in plan["splits"].items():
        counts["policy_positions"] = sum(g.policy_samples for g in index.split_games(split))
    policy_positions = plan["splits"]["train"]["policy_positions"]
    if not policy_positions:
        raise ValueError("mixed-budget training pool has no positive-weight policy positions")
    draws = plan["steps_target"] * config.learner.batch_size
    plan["sampling"] = {
        "protocol": config.sampling_protocol,
        "policy_batch_size": config.learner.batch_size,
        "value_batch_size": config.learner.batch_size,
        "policy_draw_exposure": draws / policy_positions,
        "value_draw_exposure": draws / len(value),
        "update": "one joint AdamW update and one global gradient clip per policy/value batch pair",
    }
    write_json(output / "plan.json", plan)
    return index, value, identity, fixed, plan


def make_batch_drawer(index, value, batch_size):
    policy = PolicyPool(index.split_games("train"), index.cache)

    def draw(rng):
        policies = [policy.sample(rng) for _ in range(batch_size)]
        values = [value.sample(rng) for _ in range(batch_size)]
        return DrawnBatch(
            tuple(sample for sample, _ in policies),
            tuple(ref for _, ref in (*policies, *values)),
            2 * batch_size, 0,
            {"policy_high": batch_size, "value_all": batch_size},
            tuple(sample for sample, _ in values),
        )

    return draw


def run(config, output, *, prepare_only=False, max_steps_this_session=None):
    return run_objectives(config, output, prepare_only=prepare_only,
        max_steps_this_session=max_steps_this_session,
        input_preparer=prepare_inputs, learner_initializer=initialize_learner,
        validation_percent_step=config.validation_percent_step,
        batch_drawer_factory=lambda index, value: make_batch_drawer(index, value, config.learner.batch_size))


def select_checkpoint(config, output):
    return select_scratch_checkpoint(config, output, input_preparer=prepare_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-steps-this-session", type=int)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    config = MixedBudgetOfflineConfig.from_dict(read_json(args.config))
    if args.select_only:
        selected = select_checkpoint(config, args.output)
        print({"selected_step": selected["selected_step"], "objective": selected["selection_objective"]}, flush=True)
    else:
        run(config, args.output, prepare_only=args.prepare_only,
            max_steps_this_session=args.max_steps_this_session)


if __name__ == "__main__":
    main()
