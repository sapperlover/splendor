"""Random-initialized attention or set-MLP fitting on a frozen pool, with optional auxiliary targets."""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .actor_critic import build_actor_critic, model_config_from_dict
from .alphazero_recording import read_json, sha256, write_json
from .alphazero_train import AlphazeroLearner, LearnerConfig
from .offline_objectives import freeze_history_inputs, run as run_objectives
from .rng import DeterministicRng
from .value_auxiliary import ValueAuxiliaryConfig, add_auxiliary_heads


@dataclass(frozen=True)
class ScratchOfflineConfig:
    manifest: str
    manifest_sha256: str
    model_config: dict
    first_round: int = 24
    last_round: int = 43
    expected_games_per_round: int = 1000
    draw_exposures: float = 12.0
    cache_mib: int = 8192
    initialization_seed: int = 36500000
    auxiliary_initialization_seed: int = 36100000
    sampling_seed: int = 34410000
    validation_seed: int = 34420000
    validation_positions: int = 4096
    policy_surprise_weighting: None = None
    value_auxiliary: dict | None = field(default_factory=lambda: asdict(ValueAuxiliaryConfig()))
    validation_percent_step: int = 5
    learner: LearnerConfig = field(default_factory=lambda: LearnerConfig(
        learning_rate=2e-5, checkpoint_every_steps=1000))

    def __post_init__(self):
        if not 1 <= self.first_round <= self.last_round or self.expected_games_per_round <= 0:
            raise ValueError("invalid absolute source generation range/count")
        if not math.isfinite(self.draw_exposures) or self.draw_exposures <= 0 or self.cache_mib <= 0:
            raise ValueError("exposures and cache must be positive")
        if not 0 < self.validation_positions <= 4096:
            raise ValueError("validation_positions must be in [1, 4096]")
        if self.learner.max_steps_per_round is not None:
            raise ValueError("offline budget is determined by draw_exposures")
        if self.policy_surprise_weighting is not None:
            raise ValueError("scratch fitting requires policy surprise weighting disabled")
        if self.value_auxiliary is not None:
            ValueAuxiliaryConfig(**self.value_auxiliary)
        model_config_from_dict(self.model_config)
        if self.model_config.get("architecture") not in ("entity_attention_v1", "entity_set_mlp_v1"):
            raise ValueError("scratch fitting requires a base entity_attention_v1 or entity_set_mlp_v1 config")
        if (type(self.validation_percent_step) is not int or not 0 < self.validation_percent_step <= 100
                or 100 % self.validation_percent_step):
            raise ValueError("validation_percent_step must be a positive integer divisor of 100")

    @classmethod
    def from_dict(cls, raw):
        values = dict(raw)
        values["learner"] = LearnerConfig(**values.get("learner", {}))
        return cls(**values)


def prepare_inputs(config, output):
    prepared = freeze_history_inputs(config, output, initialization={"initialization": {
        "kind": "random_initialization", "seed": config.initialization_seed,
        "auxiliary_seed": config.auxiliary_initialization_seed if config.value_auxiliary is not None else None,
        "model_config": config.model_config}})
    prepared[-1]["validation_steps"] = validation_schedule(config, prepared[-1]["steps_target"])
    write_json(output / "plan.json", prepared[-1])
    return prepared


def validation_schedule(config, steps):
    return sorted({math.ceil(steps*p/100) for p in range(config.validation_percent_step, 101,
                                                       config.validation_percent_step)})


def initialize_learner(config, identity, steps):
    torch.manual_seed(config.initialization_seed)
    base = build_actor_critic(config.model_config)
    model = base
    if config.value_auxiliary is not None:
        torch.manual_seed(config.auxiliary_initialization_seed)
        model = add_auxiliary_heads(base, predict_score=ValueAuxiliaryConfig(**config.value_auxiliary).score_loss_weight > 0)
    learner = AlphazeroLearner(model, config.learner, sampling_rng=DeterministicRng(config.sampling_seed))
    learner.replay_identity = identity
    learner.round_steps_target = steps
    learner.parent_checkpoint = {"kind": "random_initialization", "seed": config.initialization_seed,
        "auxiliary_seed": config.auxiliary_initialization_seed if config.value_auxiliary is not None else None,
        "model_config": asdict(model.config),
        "optimizer": "fresh AdamW"}
    return learner


def checkpoint_path(output, step):
    return output / "checkpoints" / ("initial.pt" if step == 0 else f"step-{step:09d}.pt")


def select_checkpoint(config, output, *, input_preparer=prepare_inputs):
    """Full-holdout KL+MSE minimum over the predeclared percentage checkpoints."""
    progress = read_json(output / "progress.json")
    if progress["status"] != "complete":
        raise ValueError("checkpoint selection requires completed training")
    _, _, _, _, plan = input_preparer(config, output)
    candidates = plan["validation_steps"]
    rows = []
    for step in candidates:
        checkpoint = checkpoint_path(output, step)
        path = output / "full_validation" / f"step-{step:09d}.json"
        expected_sha = sha256(checkpoint)
        row = read_json(path)
        if row["checkpoint_sha256"] != expected_sha or row["dataset_digest"] != read_json(output / "dataset.json")["digest"]:
            raise ValueError("full-validation checkpoint/data identity differs")
        if row["metrics"]["positions"] != plan["splits"]["validation"]["positions"]:
            raise ValueError("full-validation population differs")
        row["selection_objective"] = row["metrics"]["policy_kl"] + row["metrics"]["value_loss"]
        rows.append(row)
    best = min(rows, key=lambda p: (p["selection_objective"], p["step"]))
    selection = {"criterion": "full_validation.policy_kl + full_validation.value_loss; auxiliary losses excluded; ties earlier step",
        "candidate_rule": "all post-update percentage checkpoints, including final",
        "validation_percent_step": config.validation_percent_step, "candidates": rows,
        "selected_step": best["step"], "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"], "selection_objective": best["selection_objective"],
        "scope": "single training seed; validation-selected full checkpoint, no policy/value composition"}
    write_json(output / "selection.json", selection)
    return selection


def run(config, output, *, prepare_only=False, max_steps_this_session=None):
    return run_objectives(config, output, prepare_only=prepare_only,
        max_steps_this_session=max_steps_this_session, input_preparer=prepare_inputs,
        learner_initializer=initialize_learner, validation_percent_step=config.validation_percent_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-steps-this-session", type=int)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    config = ScratchOfflineConfig.from_dict(read_json(args.config))
    if args.select_only:
        selection = select_checkpoint(config, args.output)
        print({"selected_step": selection["selected_step"], "objective": selection["selection_objective"]}, flush=True)
    else:
        run(config, args.output, prepare_only=args.prepare_only, max_steps_this_session=args.max_steps_this_session)


if __name__ == "__main__":
    main()
