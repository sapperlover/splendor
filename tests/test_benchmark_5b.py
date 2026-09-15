from __future__ import annotations

from pathlib import Path

from splendor.actor_critic import ActorCritic
from splendor.benchmark_5b import benchmark_samplers
from splendor.ppo import PPOTrainer, PolicySampler
from splendor.replay import RULESET_ID


def test_small_cpu_benchmark_records_serial_and_batched_metrics(
    tmp_path: Path,
) -> None:
    model = ActorCritic()
    trainer = PPOTrainer(model)
    sampler = PolicySampler(model, seed=1)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(
        checkpoint,
        sampler=sampler,
        metadata={"rules_version": RULESET_ID},
    )
    output = tmp_path / "benchmark.json"

    result = benchmark_samplers(
        checkpoint,
        device="cpu",
        decisions=8,
        training_seed=50,
        active_env_counts=(2,),
        output_path=output,
    )

    assert output.exists()
    assert [row["label"] for row in result["results"]] == ["serial", "batched-2"]
    assert all(row["decisions"] == 8 for row in result["results"])
    assert all(row["decisions_per_second"] > 0 for row in result["results"])
    assert result["results"][1]["mean_inference_batch_size"] > 1.0
