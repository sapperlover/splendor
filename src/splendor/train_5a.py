"""Work-package 5A fixed-Random full-game PPO experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import resource
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .actions import (
    BuyMarket,
    BuyReserved,
    ClaimNoble,
    Pay,
    ReserveDeck,
    ReserveMarket,
    ReturnTokens,
    TakeDistinct,
    TakePair,
)
from .actor_critic import ActorCritic
from .evaluation import EvaluationResult, PolicySpec, baseline_policy_spec, evaluate_policies
from .features import FEATURE_SCHEMA_VERSION
from .neural_policy import GreedyActorCriticPolicy
from .ppo import (
    PPOConfig,
    PPOTrainer,
    PolicySampler,
    RolloutBatch,
    RolloutCollector,
    RolloutSample,
    compute_gae,
)
from .random_policy import RandomPolicy
from .replay import RULESET_ID, ForcedSkipEvent, to_primitive
from .rng import DeterministicRng, RNG_ALGORITHM
from .training_env import LearningEnvironment


FIVE_A_RUN_FORMAT_VERSION = 1
_SEED_STRIDE = 10_000_000
_POLICY_STREAM_OFFSET = 1_000_000_000_000_000
_OPPONENT_STREAM_OFFSET = 2_000_000_000_000_000
_SEAT_STREAM_OFFSET = 3_000_000_000_000_000
_FIRST_PLAYER_STREAM_OFFSET = 4_000_000_000_000_000

_MAIN_ACTION_TYPES: tuple[tuple[type[object], str], ...] = (
    (TakeDistinct, "TakeDistinct"),
    (TakePair, "TakePair"),
    (ReserveMarket, "ReserveMarket"),
    (ReserveDeck, "ReserveDeck"),
    (BuyMarket, "BuyMarket"),
    (BuyReserved, "BuyReserved"),
)
_PHASE_NAMES = ("main_action", "payment", "return_tokens", "choose_noble")


@dataclass(frozen=True, slots=True)
class FiveATrainingConfig:
    total_learning_decisions: int = 100_000
    training_seed: int = 1
    opponent: str = "random"
    reward: str = "terminal"
    reward_shaping_alpha: float = 0.0
    device: str = "cuda"
    rollout_steps: int = 8192
    eval_interval: int = 20_000
    checkpoint_interval: int = 50_000
    eval_pairs: int = 50
    final_eval_pairs: int = 200
    max_completed_turns: int = 400

    def __post_init__(self) -> None:
        if self.total_learning_decisions < 0:
            raise ValueError("total_learning_decisions must be non-negative")
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")
        if self.opponent != "random":
            raise ValueError("work package 5A only permits the fixed RandomPolicy opponent")
        if self.reward != "terminal" or self.reward_shaping_alpha != 0.0:
            raise ValueError("work package 5A only permits unshaped terminal reward")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if min(
            self.rollout_steps,
            self.eval_interval,
            self.checkpoint_interval,
            self.eval_pairs,
            self.final_eval_pairs,
            self.max_completed_turns,
        ) <= 0:
            raise ValueError("training sizes, intervals, and watchdog must be positive")


@dataclass(frozen=True, slots=True)
class EpisodePlan:
    episode_index: int
    game_seed: int
    learning_policy_seed: int
    opponent_seed: int
    learning_player: int
    first_player: int


def training_game_seed(training_seed: int, episode_index: int) -> int:
    return training_seed * _SEED_STRIDE + episode_index


def dev_eval_game_seeds(training_seed: int, count: int) -> tuple[int, ...]:
    """The negative DEV domain is disjoint from all non-negative TRAIN seeds."""

    return tuple(-(training_seed * _SEED_STRIDE + index + 1) for index in range(count))


def episode_plan(training_seed: int, episode_index: int) -> EpisodePlan:
    base = training_seed * _SEED_STRIDE + episode_index
    learning_player = DeterministicRng(_SEAT_STREAM_OFFSET + base).randbelow(2)
    first_player = DeterministicRng(_FIRST_PLAYER_STREAM_OFFSET + base).randbelow(2)
    return EpisodePlan(
        episode_index=episode_index,
        game_seed=training_game_seed(training_seed, episode_index),
        learning_policy_seed=_POLICY_STREAM_OFFSET + base,
        opponent_seed=_OPPONENT_STREAM_OFFSET + base,
        learning_player=learning_player,
        first_player=first_player,
    )


@dataclass(slots=True)
class TrainingProgress:
    games_finished: int = 0
    games_completed: int = 0
    engineering_truncations: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    forced_skips: int = 0
    total_completed_player_turns: int = 0
    total_episode_learning_decisions: int = 0
    purchased_development_cards: int = 0
    nobles_acquired: int = 0
    final_score_total: int = 0
    final_score_games: int = 0
    learning_player_games: list[int] = field(default_factory=lambda: [0, 0])
    main_actions: dict[str, int] = field(
        default_factory=lambda: {name: 0 for _, name in _MAIN_ACTION_TYPES}
    )
    phase_decisions: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _PHASE_NAMES}
    )
    evaluation_steps: list[int] = field(default_factory=list)
    best_score_rate: float | None = None
    best_checkpoint: str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainingProgress:
        progress = cls()
        for name in cls.__dataclass_fields__:
            if name in values:
                setattr(progress, name, values[name])
        return progress


@dataclass(frozen=True, slots=True)
class FiveATrainingResult:
    output_dir: str
    final_checkpoint: str
    best_checkpoint: str | None
    global_learning_decisions: int
    ppo_updates: int
    games_completed: int
    engineering_truncations: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_METRIC_FIELDS = (
    "record_type",
    "eval_kind",
    "step",
    "update_count",
    "checkpoint_id",
    "wall_seconds",
    "policy_loss",
    "value_loss",
    "total_loss",
    "entropy",
    "explained_variance",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "learning_rate",
    "epochs_completed",
    "minibatches_processed",
    "gradient_clipped_minibatches",
    "kl_early_stopped",
    "train_win_rate",
    "train_draw_rate",
    "train_loss_rate",
    "mean_reward",
    "mean_valid_candidates",
    "single_candidate_fraction",
    "games_finished",
    "games_completed",
    "engineering_truncations",
    "forced_skips",
    "mean_completed_player_turns",
    "mean_learning_decisions_per_game",
    "payment_decisions",
    "return_tokens_decisions",
    "choose_noble_decisions",
    "TakeDistinct_count",
    "TakePair_count",
    "ReserveMarket_count",
    "ReserveDeck_count",
    "BuyMarket_count",
    "BuyReserved_count",
    "TakeDistinct_fraction",
    "TakePair_fraction",
    "ReserveMarket_fraction",
    "ReserveDeck_fraction",
    "BuyMarket_fraction",
    "BuyReserved_fraction",
    "mean_buys_per_game",
    "mean_reserves_per_game",
    "mean_token_taking_per_game",
    "mean_purchased_development_cards",
    "mean_nobles_acquired",
    "mean_final_score",
    "learning_decisions_per_second",
    "games_per_second",
    "rollout_seconds",
    "environment_encoding_seconds",
    "model_inference_seconds",
    "gae_seconds",
    "ppo_update_seconds",
    "evaluation_seconds",
    "cuda_peak_allocated_mib",
    "cpu_max_rss_mib",
    "paired_seeds",
    "total_games",
    "wins",
    "draws",
    "losses",
    "score_rate",
    "win_rate",
    "draw_rate",
    "first_player_score",
    "second_player_score",
    "median_completed_player_turns",
    "illegal_actions",
    "invariant_failures",
    "exceptions",
)


class MetricWriter:
    def __init__(self, output_dir: Path) -> None:
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.csv_path = output_dir / "metrics.csv"

    def append(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({name: record.get(name, "") for name in _METRIC_FIELDS})


def _git_metadata() -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": bool(command("git", "status", "--porcelain")),
    }


def runtime_metadata(device: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": cuda_available,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git": _git_metadata(),
        "rules_version": RULESET_ID,
        "rng_algorithm": RNG_ALGORITHM,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_actor_critic_vs_random(
    model: ActorCritic,
    *,
    training_seed: int,
    paired_seed_count: int,
    device: str,
    max_completed_turns: int,
    failure_dir: Path,
) -> tuple[EvaluationResult, float]:
    """Paired, greedy, frozen-model DEV evaluation."""

    parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())
    was_training = model.training
    model.eval()
    policy = PolicySpec(
        "ppo-greedy",
        lambda seed: GreedyActorCriticPolicy(model, device=device),
    )
    started = time.perf_counter()
    result = evaluate_policies(
        policy,
        baseline_policy_spec("random"),
        dev_eval_game_seeds(training_seed, paired_seed_count),
        policy_seed_base=-(training_seed * _SEED_STRIDE + 5_000_000),
        max_completed_turns=max_completed_turns,
        failure_dir=failure_dir,
    )
    elapsed = time.perf_counter() - started
    model.train(was_training)
    if not all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    ):
        raise AssertionError("evaluation modified model parameters")
    return result, elapsed


def _action_name(action: object) -> str | None:
    for action_type, name in _MAIN_ACTION_TYPES:
        if isinstance(action, action_type):
            return name
    return None


def _phase_name(action: object) -> str:
    if _action_name(action) is not None:
        return "main_action"
    if isinstance(action, Pay):
        return "payment"
    if isinstance(action, ReturnTokens):
        return "return_tokens"
    if isinstance(action, ClaimNoble):
        return "choose_noble"
    raise TypeError(f"unknown action type: {type(action).__name__}")


def _episode_summary(environment: LearningEnvironment) -> dict[str, Any]:
    transitions = environment.transitions
    if not transitions:
        raise AssertionError("completed learning environment has no transition")
    result = environment.terminal_result
    reward = transitions[-1].reward
    main_actions = Counter(
        name
        for transition in transitions
        if (name := _action_name(transition.chosen_action)) is not None
    )
    noble_count = sum(
        event.actor == environment.learning_player
        and isinstance(event.action, ClaimNoble)
        for event in environment.replay_record.events
    )
    return {
        "episode_id": environment.episode_id,
        "learning_player": environment.learning_player,
        "terminated": environment.terminated,
        "truncated": environment.truncated,
        "reward": reward,
        "completed_player_turns": sum(transitions[-1].completed_turns_after),
        "learning_decisions": len(transitions),
        "opponent_decisions": environment.opponent_decision_count,
        "forced_skips": sum(
            isinstance(event, ForcedSkipEvent)
            for event in environment.replay_record.events
        ),
        "main_actions": dict(main_actions),
        "purchased_development_cards": (
            None
            if result is None
            else result.purchased_card_counts[environment.learning_player]
        ),
        "nobles_acquired": noble_count,
        "final_score": None if result is None else result.scores[environment.learning_player],
    }


def summarize_rollout(
    rollout: RolloutBatch,
    completed_environments: Sequence[LearningEnvironment],
) -> dict[str, Any]:
    samples = rollout.samples
    valid_counts = [int(np.count_nonzero(sample.encoded.candidate_mask)) for sample in samples]
    action_counts = Counter(
        name
        for sample in samples
        if (name := _action_name(
            sample.encoded.candidate_actions[sample.chosen_candidate_index]
        ))
        is not None
    )
    phase_counts = Counter(
        _phase_name(sample.encoded.candidate_actions[sample.chosen_candidate_index])
        for sample in samples
    )
    episodes = [_episode_summary(environment) for environment in completed_environments]
    terminal = [episode for episode in episodes if episode["terminated"]]
    wins = sum(episode["reward"] > 0 for episode in terminal)
    draws = sum(episode["reward"] == 0 for episode in terminal)
    losses = sum(episode["reward"] < 0 for episode in terminal)
    main_total = sum(action_counts.values())

    def average(values: Sequence[int | float]) -> float:
        return float(mean(values)) if values else 0.0

    record: dict[str, Any] = {
        "train_win_rate": wins / len(terminal) if terminal else 0.0,
        "train_draw_rate": draws / len(terminal) if terminal else 0.0,
        "train_loss_rate": losses / len(terminal) if terminal else 0.0,
        "mean_reward": average([sample.reward for sample in samples]),
        "mean_valid_candidates": average(valid_counts),
        "single_candidate_fraction": sum(count == 1 for count in valid_counts)
        / len(valid_counts),
        "games_finished": len(episodes),
        "games_completed": len(terminal),
        "engineering_truncations": sum(episode["truncated"] for episode in episodes),
        "forced_skips": sum(episode["forced_skips"] for episode in episodes),
        "mean_completed_player_turns": average(
            [episode["completed_player_turns"] for episode in episodes]
        ),
        "mean_learning_decisions_per_game": average(
            [episode["learning_decisions"] for episode in episodes]
        ),
        "payment_decisions": phase_counts["payment"],
        "return_tokens_decisions": phase_counts["return_tokens"],
        "choose_noble_decisions": phase_counts["choose_noble"],
        "mean_buys_per_game": average(
            [
                episode["main_actions"].get("BuyMarket", 0)
                + episode["main_actions"].get("BuyReserved", 0)
                for episode in episodes
            ]
        ),
        "mean_reserves_per_game": average(
            [
                episode["main_actions"].get("ReserveMarket", 0)
                + episode["main_actions"].get("ReserveDeck", 0)
                for episode in episodes
            ]
        ),
        "mean_token_taking_per_game": average(
            [
                episode["main_actions"].get("TakeDistinct", 0)
                + episode["main_actions"].get("TakePair", 0)
                for episode in episodes
            ]
        ),
        "mean_purchased_development_cards": average(
            [
                episode["purchased_development_cards"]
                for episode in terminal
                if episode["purchased_development_cards"] is not None
            ]
        ),
        "mean_nobles_acquired": average(
            [episode["nobles_acquired"] for episode in terminal]
        ),
        "mean_final_score": average(
            [episode["final_score"] for episode in terminal if episode["final_score"] is not None]
        ),
        "_episodes": episodes,
        "_phase_counts": dict(phase_counts),
    }
    for _, name in _MAIN_ACTION_TYPES:
        record[f"{name}_count"] = action_counts[name]
        record[f"{name}_fraction"] = (
            action_counts[name] / main_total if main_total else 0.0
        )
    return record


def _update_progress(progress: TrainingProgress, rollout_metrics: dict[str, Any]) -> None:
    for _, name in _MAIN_ACTION_TYPES:
        progress.main_actions[name] += int(rollout_metrics[f"{name}_count"])
    for phase, count in rollout_metrics["_phase_counts"].items():
        progress.phase_decisions[phase] += int(count)
    for episode in rollout_metrics["_episodes"]:
        progress.games_finished += 1
        progress.learning_player_games[episode["learning_player"]] += 1
        progress.total_completed_player_turns += episode["completed_player_turns"]
        progress.total_episode_learning_decisions += episode["learning_decisions"]
        progress.forced_skips += episode["forced_skips"]
        if episode["truncated"]:
            progress.engineering_truncations += 1
            continue
        progress.games_completed += 1
        if episode["reward"] > 0:
            progress.wins += 1
        elif episode["reward"] < 0:
            progress.losses += 1
        else:
            progress.draws += 1
        progress.purchased_development_cards += episode["purchased_development_cards"]
        progress.nobles_acquired += episode["nobles_acquired"]
        progress.final_score_total += episode["final_score"]
        progress.final_score_games += 1


def _tensorboard_rollout(
    writer: SummaryWriter,
    record: dict[str, Any],
    step: int,
) -> None:
    for key in (
        "train_win_rate",
        "train_draw_rate",
        "mean_reward",
        "mean_valid_candidates",
        "single_candidate_fraction",
        "games_completed",
    ):
        writer.add_scalar(f"rollout/{key}", record[key], step)
    for _, name in _MAIN_ACTION_TYPES:
        writer.add_scalar(f"behavior/{name}_fraction", record[f"{name}_fraction"], step)
    for key in (
        "mean_buys_per_game",
        "mean_reserves_per_game",
        "mean_token_taking_per_game",
    ):
        writer.add_scalar(f"behavior/{key}", record[key], step)


def _evaluation_record(
    result: EvaluationResult,
    *,
    kind: str,
    step: int,
    update_count: int,
    checkpoint_id: str,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "record_type": "evaluation",
        "eval_kind": kind,
        "step": step,
        "update_count": update_count,
        "checkpoint_id": checkpoint_id,
        "paired_seeds": result.total_pairs,
        "total_games": result.total_games,
        "wins": result.a_wins,
        "draws": result.draws,
        "losses": result.a_losses,
        "score_rate": result.win_rate_score,
        "win_rate": result.a_wins / result.valid_games if result.valid_games else None,
        "draw_rate": result.draws / result.valid_games if result.valid_games else None,
        "first_player_score": result.a_first.win_rate_score,
        "second_player_score": result.a_second.win_rate_score,
        "mean_completed_player_turns": result.mean_completed_player_turns,
        "median_completed_player_turns": result.median_completed_player_turns,
        "forced_skips": result.forced_skips,
        "engineering_truncations": result.engineering_truncations,
        "illegal_actions": result.illegal_actions,
        "invariant_failures": result.invariant_failures,
        "exceptions": result.exceptions,
        "evaluation_seconds": elapsed,
    }


def _tensorboard_evaluation(
    writer: SummaryWriter,
    record: dict[str, Any],
    step: int,
    *,
    prefix: str = "eval",
) -> None:
    for tag, key in (
        ("random_score_rate", "score_rate"),
        ("random_win_rate", "win_rate"),
        ("random_draw_rate", "draw_rate"),
        ("random_first_player_score", "first_player_score"),
        ("random_second_player_score", "second_player_score"),
        ("mean_turns", "mean_completed_player_turns"),
    ):
        if record[key] is not None:
            writer.add_scalar(f"{prefix}/{tag}", record[key], step)


def _checkpoint_metadata(
    config: FiveATrainingConfig,
    progress: TrainingProgress,
    collector: RolloutCollector,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if collector.active_environment is not None:
        raise ValueError("5A checkpoints must be written at an episode boundary")
    return {
        "run_format_version": FIVE_A_RUN_FORMAT_VERSION,
        "training_config": asdict(config),
        "progress": progress.to_dict(),
        "next_episode_index": collector.next_episode_index,
        "collector_at_episode_boundary": True,
        "runtime": metadata,
        "rules_version": RULESET_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def _save_checkpoint(
    trainer: PPOTrainer,
    sampler: PolicySampler,
    collector: RolloutCollector,
    config: FiveATrainingConfig,
    progress: TrainingProgress,
    metadata: dict[str, Any],
    path: Path,
) -> None:
    trainer.save_checkpoint(
        path,
        sampler=sampler,
        metadata=_checkpoint_metadata(config, progress, collector, metadata),
    )


def _finish_active_episode(
    collector: RolloutCollector,
    rollout: RolloutBatch,
) -> RolloutBatch:
    samples = list(rollout.samples)
    while collector.active_environment is not None:
        samples.extend(collector.collect(1).samples)
    return RolloutBatch(tuple(samples))


def _save_truncation(output_dir: Path, environment: LearningEnvironment) -> None:
    state = environment._engine._debug_state()
    _write_json(
        output_dir / "logs" / f"{environment.episode_id}-truncation.json",
        {
            "episode_id": environment.episode_id,
            "learning_player": environment.learning_player,
            "state": to_primitive(state),
            "replay": to_primitive(environment.replay_record),
        },
    )


def _immutable_resume_config(config: FiveATrainingConfig) -> dict[str, Any]:
    values = asdict(config)
    for key in ("total_learning_decisions", "final_eval_pairs"):
        values.pop(key)
    return values


def run_five_a_training(
    config: FiveATrainingConfig,
    *,
    output_dir: Path,
    resume_checkpoint: Path | None = None,
) -> FiveATrainingResult:
    """Run or resume the single-process fixed-Random 5A experiment."""

    output_dir = output_dir.resolve()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resume_checkpoint is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {output_dir}")
    for name in ("checkpoints", "eval", "logs", "tensorboard"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    runtime = runtime_metadata(config.device)
    if resume_checkpoint is None:
        random.seed(config.training_seed)
        np.random.seed(config.training_seed)
        torch.manual_seed(config.training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training_seed)
        model = ActorCritic()
        ppo_config = PPOConfig(rollout_steps=config.rollout_steps)
        trainer = PPOTrainer(
            model,
            ppo_config,
            device=config.device,
            shuffle_seed=_POLICY_STREAM_OFFSET + config.training_seed,
        )
        sampler = PolicySampler(
            trainer.model,
            seed=_POLICY_STREAM_OFFSET + config.training_seed,
            device=config.device,
        )
        progress = TrainingProgress()
        start_episode_index = 0
        _write_json(
            output_dir / "config.json",
            {
                "run_format_version": FIVE_A_RUN_FORMAT_VERSION,
                "training": asdict(config),
                "ppo": asdict(ppo_config),
                "model": asdict(model.config),
                "runtime": runtime,
                "train_seed_domain": "non-negative",
                "dev_eval_seed_domain": "negative",
            },
        )
    else:
        trainer, sampler, checkpoint_metadata = PPOTrainer.load_checkpoint(
            resume_checkpoint, device=config.device
        )
        if checkpoint_metadata.get("run_format_version") != FIVE_A_RUN_FORMAT_VERSION:
            raise ValueError("resume checkpoint is not a work-package 5A run")
        saved_config = FiveATrainingConfig(**checkpoint_metadata["training_config"])
        if _immutable_resume_config(saved_config) != _immutable_resume_config(config):
            raise ValueError("resume configuration changes an immutable 5A setting")
        if checkpoint_metadata.get("rules_version") != RULESET_ID:
            raise ValueError("resume checkpoint rules version mismatch")
        progress = TrainingProgress.from_dict(checkpoint_metadata["progress"])
        start_episode_index = int(checkpoint_metadata["next_episode_index"])
        if not checkpoint_metadata.get("collector_at_episode_boundary"):
            raise ValueError("resume checkpoint was not saved at an episode boundary")
        _write_json(
            output_dir
            / f"resume_config_step_{trainer.global_learning_decision_step:09d}.json",
            {"training": asdict(config), "runtime": runtime},
        )

    def environment_factory(index: int) -> LearningEnvironment:
        plan = episode_plan(config.training_seed, index)
        sampler.manual_seed(plan.learning_policy_seed)
        return LearningEnvironment.new(
            plan.game_seed,
            first_player=plan.first_player,
            learning_player=plan.learning_player,
            opponent_policy=RandomPolicy(plan.opponent_seed),
            episode_id=f"train-{config.training_seed}-{index:08d}",
            max_completed_turns=config.max_completed_turns,
            failure_dir=output_dir / "logs" / f"failure-{index:08d}",
        )

    collector = RolloutCollector(
        sampler,
        environment_factory,
        trainer.config,
        start_episode_index=start_episode_index,
    )
    metric_writer = MetricWriter(output_dir)
    tensorboard = SummaryWriter(
        log_dir=str(output_dir / "tensorboard"),
        purge_step=(
            trainer.global_learning_decision_step + 1
            if resume_checkpoint is not None
            else None
        ),
    )
    run_started = time.perf_counter()
    previous_elapsed = progress.elapsed_seconds

    def checkpoint_path(step: int) -> Path:
        return output_dir / "checkpoints" / f"step_{step:09d}.pt"

    def perform_evaluation(kind: str, paired_seeds: int) -> dict[str, Any]:
        step = trainer.global_learning_decision_step
        path = checkpoint_path(step)
        result, elapsed = evaluate_actor_critic_vs_random(
            trainer.model,
            training_seed=config.training_seed,
            paired_seed_count=paired_seeds,
            device=config.device,
            max_completed_turns=config.max_completed_turns,
            failure_dir=output_dir / "eval" / f"failures-step-{step:09d}-{kind}",
        )
        record = _evaluation_record(
            result,
            kind=kind,
            step=step,
            update_count=trainer.update_count,
            checkpoint_id=path.name,
            elapsed=elapsed,
        )
        metric_writer.append(record)
        _tensorboard_evaluation(
            tensorboard,
            record,
            step,
            prefix="eval" if kind == "dev" else "eval_key",
        )
        _write_json(
            output_dir / "eval" / f"step_{step:09d}_{kind}.json",
            {
                "global_learning_decisions": step,
                "checkpoint_id": path.name,
                "training_config": asdict(config),
                "runtime": runtime,
                "rules_version": RULESET_ID,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "evaluation": result.to_dict(include_matches=True),
                "evaluation_seconds": elapsed,
            },
        )
        if result.exceptions or result.illegal_actions or result.invariant_failures:
            raise RuntimeError("DEV evaluation contained an engine or policy error")
        if result.win_rate_score is not None and (
            progress.best_score_rate is None
            or result.win_rate_score > progress.best_score_rate
        ):
            progress.best_score_rate = result.win_rate_score
            progress.best_checkpoint = str(path)
        if step not in progress.evaluation_steps:
            progress.evaluation_steps.append(step)
        return record

    try:
        if resume_checkpoint is None:
            initial_path = checkpoint_path(0)
            _save_checkpoint(
                trainer, sampler, collector, config, progress, runtime, initial_path
            )
            perform_evaluation("dev", config.eval_pairs)
            progress.elapsed_seconds = previous_elapsed + (time.perf_counter() - run_started)
            _save_checkpoint(
                trainer, sampler, collector, config, progress, runtime, initial_path
            )

        next_eval = (
            trainer.global_learning_decision_step // config.eval_interval + 1
        ) * config.eval_interval
        next_checkpoint = (
            trainer.global_learning_decision_step // config.checkpoint_interval + 1
        ) * config.checkpoint_interval

        while trainer.global_learning_decision_step < config.total_learning_decisions:
            current = trainer.global_learning_decision_step
            target = min(config.rollout_steps, config.total_learning_decisions - current)
            boundary_due = current + target >= min(
                next_eval,
                next_checkpoint,
                config.total_learning_decisions,
            )
            sampler.reset_timing()
            rollout_started = time.perf_counter()
            rollout = collector.collect(target)
            if boundary_due:
                rollout = _finish_active_episode(collector, rollout)
            rollout_seconds = time.perf_counter() - rollout_started
            completed = collector.take_completed_environments()
            rollout_metrics = summarize_rollout(rollout, completed)

            gae_started = time.perf_counter()
            prepared = compute_gae(
                rollout,
                gamma=trainer.config.gamma,
                gae_lambda=trainer.config.gae_lambda,
            )
            gae_seconds = time.perf_counter() - gae_started
            if config.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            update_started = time.perf_counter()
            stats = trainer.update(prepared)
            if config.device == "cuda":
                torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            step = trainer.global_learning_decision_step
            _update_progress(progress, rollout_metrics)

            environment_seconds = max(0.0, rollout_seconds - sampler.inference_seconds)
            games_finished = int(rollout_metrics["games_finished"])
            update_record = {
                "record_type": "update",
                "step": step,
                "update_count": trainer.update_count,
                "wall_seconds": previous_elapsed + (time.perf_counter() - run_started),
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "total_loss": stats.total_loss,
                "entropy": stats.entropy,
                "explained_variance": stats.explained_variance,
                "approx_kl": stats.approx_kl,
                "clip_fraction": stats.clip_fraction,
                "grad_norm": stats.grad_norm,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "epochs_completed": stats.epochs_completed,
                "minibatches_processed": stats.minibatches_processed,
                "gradient_clipped_minibatches": stats.gradient_clipped_minibatches,
                "kl_early_stopped": stats.early_stopped,
                "learning_decisions_per_second": len(rollout) / rollout_seconds,
                "games_per_second": games_finished / rollout_seconds,
                "rollout_seconds": rollout_seconds,
                "environment_encoding_seconds": environment_seconds,
                "model_inference_seconds": sampler.inference_seconds,
                "gae_seconds": gae_seconds,
                "ppo_update_seconds": update_seconds,
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                    if config.device == "cuda"
                    else 0.0
                ),
                "cpu_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
                **{
                    key: value
                    for key, value in rollout_metrics.items()
                    if not key.startswith("_")
                },
            }
            metric_writer.append(update_record)
            for key in (
                "policy_loss",
                "value_loss",
                "total_loss",
                "entropy",
                "explained_variance",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
                "learning_rate",
            ):
                tensorboard.add_scalar(f"ppo/{key}", update_record[key], step)
            _tensorboard_rollout(tensorboard, update_record, step)
            for key in (
                "learning_decisions_per_second",
                "games_per_second",
                "rollout_seconds",
                "ppo_update_seconds",
                "cuda_peak_allocated_mib",
            ):
                tensorboard.add_scalar(f"system/{key}", update_record[key], step)
            tensorboard.flush()

            truncated_environments = [
                environment for environment in completed if environment.truncated
            ]
            for environment in truncated_environments:
                _save_truncation(output_dir, environment)

            evaluation_due = step >= next_eval
            checkpoint_due = step >= next_checkpoint
            final_due = step >= config.total_learning_decisions
            if evaluation_due:
                perform_evaluation("dev", config.eval_pairs)
                while next_eval <= step:
                    next_eval += config.eval_interval
            if evaluation_due or checkpoint_due or final_due or truncated_environments:
                progress.elapsed_seconds = previous_elapsed + (
                    time.perf_counter() - run_started
                )
                _save_checkpoint(
                    trainer,
                    sampler,
                    collector,
                    config,
                    progress,
                    runtime,
                    checkpoint_path(step),
                )
                while next_checkpoint <= step:
                    next_checkpoint += config.checkpoint_interval
            if truncated_environments:
                raise RuntimeError(
                    "engineering truncation encountered; replay/state saved for investigation"
                )

        final_step = trainer.global_learning_decision_step
        perform_evaluation("key", config.final_eval_pairs)
        progress.elapsed_seconds = previous_elapsed + (time.perf_counter() - run_started)
        final_path = checkpoint_path(final_step)
        _save_checkpoint(
            trainer, sampler, collector, config, progress, runtime, final_path
        )
        _write_json(
            output_dir / "summary.json",
            {
                "training_config": asdict(config),
                "ppo_config": asdict(trainer.config),
                "runtime": runtime,
                "global_learning_decisions": final_step,
                "ppo_updates": trainer.update_count,
                "progress": progress.to_dict(),
                "final_checkpoint": str(final_path),
            },
        )
        return FiveATrainingResult(
            output_dir=str(output_dir),
            final_checkpoint=str(final_path),
            best_checkpoint=progress.best_checkpoint,
            global_learning_decisions=final_step,
            ppo_updates=trainer.update_count,
            games_completed=progress.games_completed,
            engineering_truncations=progress.engineering_truncations,
            elapsed_seconds=progress.elapsed_seconds,
        )
    finally:
        tensorboard.flush()
        tensorboard.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Work-package 5A full-game PPO training against fixed RandomPolicy"
    )
    parser.add_argument("--total-learning-steps", type=int, default=100_000)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--opponent", choices=("random",), default="random")
    parser.add_argument("--reward", choices=("terminal",), default="terminal")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rollout-size", type=int, default=8192)
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--eval-pairs", type=int, default=50)
    parser.add_argument("--final-eval-pairs", type=int, default=200)
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = FiveATrainingConfig(
            total_learning_decisions=args.total_learning_steps,
            training_seed=args.training_seed,
            opponent=args.opponent,
            reward=args.reward,
            device=args.device,
            rollout_steps=args.rollout_size,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            eval_pairs=args.eval_pairs,
            final_eval_pairs=args.final_eval_pairs,
            max_completed_turns=args.max_completed_turns,
        )
    except ValueError as error:
        parser.error(str(error))
    result = run_five_a_training(
        config,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
