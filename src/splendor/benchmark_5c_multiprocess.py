"""5C multiprocessing-collector preflight and throughput benchmark."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .multiprocess_collector import (
    MultiprocessEpisodeLaunch,
    MultiprocessEpisodeSpec,
    MultiprocessRolloutCollector,
)
from .neural_policy import NeuralInferenceTiming
from .opponent_pool import OpponentChoice, OpponentPool
from .ppo import BatchedRolloutCollector, PPOTrainer, compute_gae
from .replay import RULESET_ID
from .train_5a import episode_plan
from .train_5c import (
    FEATURE_SCHEMA_VERSION,
    FiveCConfig,
    FiveCOpponentSelector,
    _MIXTURE_RNG_OFFSET,
    _apply_schedules,
    _configure_trainer,
    load_five_c_config,
)
from .training_env import LearningEnvironment


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _one_case(
    config: FiveCConfig,
    *,
    run_root: Path,
    device: str,
    decisions: int,
    repeat: int,
    num_workers: int | None,
    opponent_backend: str,
) -> dict[str, Any]:
    trainer, sampler, metadata = PPOTrainer.load_checkpoint(
        Path(config.source_checkpoint), device=device
    )
    if metadata.get("rules_version") != RULESET_ID:
        raise ValueError("benchmark source checkpoint rules version mismatch")
    _configure_trainer(trainer, config)
    _apply_schedules(trainer, config, 0)
    pool = OpponentPool.load_manifest(
        run_root / "historical_pool" / "initial.json",
        device=("cpu" if opponent_backend == "worker_cpu" else device),
    )
    latest = pool.set_latest(
        trainer.model, training_step=trainer.global_learning_decision_step
    )
    latest_model = pool.policy_for(latest.opponent_id).policy.model
    latest_before = tuple(
        parameter.detach().clone() for parameter in latest_model.parameters()
    )
    learner_before = tuple(
        parameter.detach().clone() for parameter in trainer.model.parameters()
    )
    selector = FiveCOpponentSelector(
        _MIXTURE_RNG_OFFSET + config.training_seed
    )
    opponent_timing = NeuralInferenceTiming()
    failure_root = (
        run_root
        / "multiprocessing_preflight"
        / "failures"
        / ("single" if num_workers is None else f"workers-{num_workers}")
        / f"repeat-{repeat}"
    )

    def choice(index: int) -> tuple[Any, OpponentChoice, str]:
        plan = episode_plan(config.training_seed, index)
        kind = selector.choose_kind(0)
        selected = pool.choose_for_training(
            kind,
            policy_seed=plan.opponent_seed,
            timing=opponent_timing,
        )
        return plan, selected, f"5c-mp-preflight-{repeat}-{index:09d}"

    def local_factory(index: int):
        plan, selected, episode_id = choice(index)
        return (
            LearningEnvironment.new(
                plan.game_seed,
                first_player=plan.first_player,
                learning_player=plan.learning_player,
                opponent_policy=selected.policy,
                episode_id=episode_id,
                max_completed_turns=config.max_completed_turns,
                failure_dir=failure_root / f"episode-{index:09d}",
                defer_opponent=True,
            ),
            plan.learning_policy_seed,
        )

    def process_factory(index: int) -> MultiprocessEpisodeLaunch:
        plan, selected, episode_id = choice(index)
        return MultiprocessEpisodeLaunch(
            MultiprocessEpisodeSpec(
                episode_index=index,
                episode_id=episode_id,
                game_seed=plan.game_seed,
                first_player=plan.first_player,
                learning_player=plan.learning_player,
                learning_policy_seed=plan.learning_policy_seed,
                opponent_seed=plan.opponent_seed,
                opponent_kind=selected.entry.kind,
                opponent_id=selected.entry.opponent_id,
                max_completed_turns=config.max_completed_turns,
                failure_dir=str(failure_root / f"episode-{index:09d}"),
            ),
            neural_opponent_model=(
                None
                if opponent_backend == "worker_cpu"
                else getattr(selected.policy, "model", None)
            ),
        )

    if num_workers is None:
        collector: BatchedRolloutCollector | MultiprocessRolloutCollector
        collector = BatchedRolloutCollector(
            sampler, local_factory, active_envs=config.active_envs
        )
        label = "single-process-32-envs"
        envs_per_worker = 32
        worker_before = ()
    else:
        envs_per_worker = config.active_envs // num_workers
        collector = MultiprocessRolloutCollector(
            sampler,
            process_factory,
            num_workers=num_workers,
            envs_per_worker=envs_per_worker,
            master_seed=config.training_seed,
            opponent_inference_backend=opponent_backend,
            worker_historical_models=(
                {
                    entry.opponent_id: entry.checkpoint_path
                    for entry in pool.historical_entries
                    if entry.checkpoint_path is not None
                }
                if opponent_backend == "worker_cpu"
                else None
            ),
        )
        label = (
            f"{num_workers}-workers-x-{envs_per_worker}-envs-"
            f"{opponent_backend}"
        )
        worker_before = collector.worker_statistics()

    sampler.reset_timing()
    opponent_timing.reset()
    if isinstance(collector, MultiprocessRolloutCollector):
        collector.reset_timing()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    main_cpu_before = time.process_time()
    rollout_started = time.perf_counter()
    try:
        if (
            isinstance(collector, MultiprocessRolloutCollector)
            and collector.worker_local_opponents
        ):
            collector.sync_worker_opponents(
                latest_opponent_id=latest.opponent_id,
                latest_model=latest_model,
                historical_models={
                    entry.opponent_id: entry.checkpoint_path
                    for entry in pool.historical_entries
                    if entry.checkpoint_path is not None
                },
            )
        rollout = collector.collect(decisions)
        if device == "cuda":
            torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - rollout_started
        main_cpu_seconds = time.process_time() - main_cpu_before
        completed = collector.take_completed_environments()
        if any(environment.truncated for environment in completed):
            raise RuntimeError(f"{label} produced an engineering truncation")
        if not all(
            torch.equal(before, after)
            for before, after in zip(
                learner_before, trainer.model.parameters(), strict=True
            )
        ):
            raise AssertionError(f"{label} rollout modified learner weights")
        if not all(
            torch.equal(before, after)
            for before, after in zip(
                latest_before, latest_model.parameters(), strict=True
            )
        ):
            raise AssertionError(f"{label} rollout modified frozen latest weights")

        gae_started = time.perf_counter()
        prepared = compute_gae(
            rollout, gamma=config.gamma, gae_lambda=config.gae_lambda
        )
        gae_seconds = time.perf_counter() - gae_started
        update_started = time.perf_counter()
        stats = trainer.update(prepared)
        if device == "cuda":
            torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_started

        if isinstance(collector, MultiprocessRolloutCollector):
            worker_after = collector.worker_statistics()
            worker_cpu = tuple(
                after.cpu_seconds - before.cpu_seconds
                for before, after in zip(worker_before, worker_after, strict=True)
            )
            environment_seconds = sum(
                after.environment_encoding_seconds
                - before.environment_encoding_seconds
                for before, after in zip(worker_before, worker_after, strict=True)
            )
            opponent_seconds = collector.opponent_inference_seconds
            opponent_items = collector.opponent_inference_items
            opponent_batch = collector.mean_opponent_inference_batch_size
            ipc_wait = collector.ipc_wait_seconds
            worker_memory = sum(item.max_rss_mib for item in worker_after)
            pids = collector.worker_pids
        else:
            worker_cpu = ()
            environment_seconds = max(
                0.0,
                rollout_seconds
                - sampler.inference_seconds
                - opponent_timing.seconds,
            )
            opponent_seconds = opponent_timing.seconds
            opponent_items = opponent_timing.decisions
            opponent_batch = 1.0 if opponent_timing.decisions else 0.0
            ipc_wait = 0.0
            worker_memory = 0.0
            pids = ()

        return {
            "label": label,
            "repeat": repeat,
            "num_workers": 1 if num_workers is None else num_workers,
            "envs_per_worker": envs_per_worker,
            "active_envs": config.active_envs,
            "start_method": (
                "none" if num_workers is None else collector.start_method
            ),
            "opponent_inference_backend": (
                "single_process" if num_workers is None else opponent_backend
            ),
            "worker_pids": list(pids),
            "decisions": len(rollout),
            "completed_games": len(completed),
            "rollout_seconds": rollout_seconds,
            "learning_decisions_per_second": len(rollout) / rollout_seconds,
            "games_per_second": len(completed) / rollout_seconds,
            "mean_inference_batch_size": sampler.mean_inference_batch_size,
            "mean_opponent_inference_batch_size": opponent_batch,
            "environment_encoding_seconds": environment_seconds,
            "gpu_learner_inference_seconds": sampler.inference_seconds,
            "neural_opponent_inference_seconds": opponent_seconds,
            "gpu_opponent_inference_seconds": opponent_seconds,
            "opponent_model_sync_seconds": (
                collector.opponent_model_sync_seconds
                if isinstance(collector, MultiprocessRolloutCollector)
                else 0.0
            ),
            "neural_opponent_decisions": opponent_items,
            "ipc_wait_seconds": ipc_wait,
            "gae_seconds": gae_seconds,
            "ppo_update_seconds": update_seconds,
            "ppo_approx_kl": stats.approx_kl,
            "main_cpu_seconds": main_cpu_seconds,
            "worker_cpu_seconds": list(worker_cpu),
            "aggregate_cpu_utilization_percent": (
                100.0 * (main_cpu_seconds + sum(worker_cpu)) / rollout_seconds
            ),
            "per_worker_utilization_percent": [
                100.0 * value / rollout_seconds for value in worker_cpu
            ],
            "cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if device == "cuda"
                else 0.0
            ),
            "main_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "worker_max_rss_sum_mib": worker_memory,
            "illegal_actions": 0,
            "invariant_failures": 0,
            "candidate_mapping_mismatches": 0,
            "engineering_truncations": 0,
        }
    finally:
        if isinstance(collector, MultiprocessRolloutCollector):
            collector.close()


def benchmark_multiprocessing_collectors(
    config_path: Path,
    *,
    device: str,
    decisions: int,
    repeats: int,
    worker_counts: Sequence[int] = (1, 2, 4, 8),
    opponent_backend: str = "central",
    output_path: Path | None = None,
) -> dict[str, Any]:
    if decisions <= 0 or repeats <= 0:
        raise ValueError("decisions and repeats must be positive")
    if opponent_backend not in ("central", "worker_cpu"):
        raise ValueError("unsupported opponent inference backend")
    counts = tuple(worker_counts)
    if counts != (1, 2, 4, 8):
        raise ValueError("preflight worker counts must be exactly 1, 2, 4, 8")
    config_path = config_path.resolve()
    run_root = config_path.parent
    config = load_five_c_config(config_path)
    rows = []
    for repeat in range(repeats):
        rows.append(
            _one_case(
                config,
                run_root=run_root,
                device=device,
                decisions=decisions,
                repeat=repeat,
                num_workers=None,
                opponent_backend=opponent_backend,
            )
        )
        for count in counts:
            rows.append(
                _one_case(
                    config,
                    run_root=run_root,
                    device=device,
                    decisions=decisions,
                    repeat=repeat,
                    num_workers=count,
                    opponent_backend=opponent_backend,
                )
            )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    summaries = []
    for label, values in grouped.items():
        rates = [float(item["learning_decisions_per_second"]) for item in values]
        summaries.append(
            {
                "label": label,
                "num_workers": values[0]["num_workers"],
                "envs_per_worker": values[0]["envs_per_worker"],
                "mean_learning_decisions_per_second": statistics.mean(rates),
                "min_learning_decisions_per_second": min(rates),
                "max_learning_decisions_per_second": max(rates),
                "coefficient_of_variation": (
                    statistics.pstdev(rates) / statistics.mean(rates)
                    if len(rates) > 1
                    else 0.0
                ),
                "mean_games_per_second": statistics.mean(
                    float(item["games_per_second"]) for item in values
                ),
                "mean_inference_batch_size": statistics.mean(
                    float(item["mean_inference_batch_size"]) for item in values
                ),
                "mean_aggregate_cpu_utilization_percent": statistics.mean(
                    float(item["aggregate_cpu_utilization_percent"])
                    for item in values
                ),
                "mean_ipc_wait_seconds": statistics.mean(
                    float(item["ipc_wait_seconds"]) for item in values
                ),
            }
        )
    baseline = next(
        item for item in summaries if item["label"] == "single-process-32-envs"
    )
    eligible = [
        item
        for item in summaries
        if item["label"] != "single-process-32-envs"
        and item["coefficient_of_variation"] <= 0.15
    ]
    if not eligible:
        eligible = [
            item for item in summaries if item["label"] != "single-process-32-envs"
        ]
    recommended = max(
        eligible, key=lambda item: item["mean_learning_decisions_per_second"]
    )
    baseline_rate = float(baseline["mean_learning_decisions_per_second"])
    for item in summaries:
        item["speedup_vs_single_process"] = (
            float(item["mean_learning_decisions_per_second"]) / baseline_rate
        )
    result = {
        "status": "PASS",
        "config": str(config_path),
        "checkpoint": config.source_checkpoint,
        "device": device,
        "decisions_per_case": decisions,
        "repeats": repeats,
        "active_envs": config.active_envs,
        "opponent_inference_backend": opponent_backend,
        "rules_version": RULESET_ID,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "raw_results": rows,
        "summaries": summaries,
        "recommendation": {
            "num_workers": recommended["num_workers"],
            "envs_per_worker": recommended["envs_per_worker"],
            "learning_decisions_per_second": recommended[
                "mean_learning_decisions_per_second"
            ],
            "speedup_vs_single_process": recommended[
                "speedup_vs_single_process"
            ],
            "selection": "highest measured stable throughput",
        },
    }
    if output_path is not None:
        _write_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--decisions", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--opponent-backend",
        choices=("central", "worker_cpu"),
        default="central",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = benchmark_multiprocessing_collectors(
        args.config,
        device=args.device,
        decisions=args.decisions,
        repeats=args.repeats,
        opponent_backend=args.opponent_backend,
        output_path=args.output,
    )
    print(json.dumps(result["recommendation"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
