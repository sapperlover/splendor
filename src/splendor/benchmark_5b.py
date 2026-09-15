"""Serial versus multi-environment rollout benchmark for work package 5B."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .features import FEATURE_SCHEMA_VERSION
from .ppo import (
    BatchedRolloutCollector,
    PPOTrainer,
    PolicySampler,
    RolloutCollector,
    compute_gae,
)
from .replay import RULESET_ID
from .train_5b import fixed_mix_episode_plan, fixed_opponent
from .training_env import LearningEnvironment


def _environment(
    training_seed: int,
    index: int,
    failure_dir: Path,
) -> tuple[LearningEnvironment, int]:
    plan = fixed_mix_episode_plan(training_seed, index)
    return (
        LearningEnvironment.new(
            plan.game_seed,
            first_player=plan.first_player,
            learning_player=plan.learning_player,
            opponent_policy=fixed_opponent(plan.opponent_kind, plan.opponent_seed),
            episode_id=f"benchmark-{training_seed}-{index:08d}",
            max_completed_turns=400,
            failure_dir=failure_dir / f"episode-{index:08d}",
        ),
        plan.learning_policy_seed,
    )


def _one_benchmark(
    checkpoint: Path,
    *,
    device: str,
    decisions: int,
    training_seed: int,
    active_envs: int | None,
    failure_dir: Path,
) -> dict[str, Any]:
    trainer, sampler, _ = PPOTrainer.load_checkpoint(checkpoint, device=device)
    before_rollout = tuple(
        parameter.detach().clone() for parameter in trainer.model.parameters()
    )

    if active_envs is None:
        def serial_factory(index: int) -> LearningEnvironment:
            environment, policy_seed = _environment(
                training_seed, index, failure_dir / "serial"
            )
            sampler.manual_seed(policy_seed)
            return environment

        collector = RolloutCollector(sampler, serial_factory, trainer.config)
        label = "serial"
    else:
        def batched_factory(index: int) -> tuple[LearningEnvironment, int]:
            return _environment(
                training_seed, index, failure_dir / f"active-{active_envs}"
            )

        collector = BatchedRolloutCollector(
            sampler,
            batched_factory,
            active_envs=active_envs,
        )
        label = f"batched-{active_envs}"

    sampler.reset_timing()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    rollout_started = time.perf_counter()
    rollout = collector.collect(decisions)
    if device == "cuda":
        torch.cuda.synchronize()
    rollout_seconds = time.perf_counter() - rollout_started
    completed = collector.take_completed_environments()
    if any(environment.truncated for environment in completed):
        raise RuntimeError(f"{label} benchmark encountered engineering truncation")
    if not all(
        torch.equal(before, after)
        for before, after in zip(
            before_rollout, trainer.model.parameters(), strict=True
        )
    ):
        raise AssertionError(f"{label} rollout modified the learning model")

    gae_started = time.perf_counter()
    prepared = compute_gae(
        rollout,
        gamma=trainer.config.gamma,
        gae_lambda=trainer.config.gae_lambda,
    )
    gae_seconds = time.perf_counter() - gae_started
    update_started = time.perf_counter()
    trainer.update(prepared)
    if device == "cuda":
        torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_started
    return {
        "label": label,
        "active_envs": 1 if active_envs is None else active_envs,
        "decisions": len(rollout),
        "completed_games": len(completed),
        "rollout_seconds": rollout_seconds,
        "decisions_per_second": len(rollout) / rollout_seconds,
        "games_per_second": len(completed) / rollout_seconds,
        "environment_encoding_seconds": max(
            0.0, rollout_seconds - sampler.inference_seconds
        ),
        "gpu_inference_seconds": sampler.inference_seconds,
        "inference_forward_calls": sampler.inference_calls,
        "mean_inference_batch_size": sampler.mean_inference_batch_size,
        "gae_seconds": gae_seconds,
        "ppo_update_seconds": update_seconds,
        "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if device == "cuda"
            else 0.0
        ),
        "cpu_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
    }


def benchmark_samplers(
    checkpoint: Path,
    *,
    device: str,
    decisions: int,
    training_seed: int,
    active_env_counts: Sequence[int] = (1, 8, 16, 32),
    output_path: Path | None = None,
) -> dict[str, Any]:
    if decisions <= 0:
        raise ValueError("decisions must be positive")
    counts = tuple(active_env_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("active environment counts must be positive")
    failure_dir = (
        output_path.parent / "benchmark_failures"
        if output_path is not None
        else Path.cwd() / "benchmark_failures"
    )
    rows = [
        _one_benchmark(
            checkpoint,
            device=device,
            decisions=decisions,
            training_seed=training_seed,
            active_envs=None,
            failure_dir=failure_dir,
        )
    ]
    rows.extend(
        _one_benchmark(
            checkpoint,
            device=device,
            decisions=decisions,
            training_seed=training_seed,
            active_envs=count,
            failure_dir=failure_dir,
        )
        for count in counts
    )
    serial_rate = float(rows[0]["decisions_per_second"])
    for row in rows:
        row["speedup_vs_serial"] = row["decisions_per_second"] / serial_rate
    result = {
        "checkpoint": str(checkpoint),
        "device": device,
        "decisions_per_case": decisions,
        "training_seed": training_seed,
        "rules_version": RULESET_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "results": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5B batched rollout benchmark")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--decisions", type=int, default=8192)
    parser.add_argument("--training-seed", type=int, default=902)
    parser.add_argument("--active-envs", type=int, nargs="+", default=(1, 8, 16, 32))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = benchmark_samplers(
        args.checkpoint,
        device=args.device,
        decisions=args.decisions,
        training_seed=args.training_seed,
        active_env_counts=args.active_envs,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
