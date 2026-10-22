"""Bounded correctness checks for the 256x4 preset; never a strength experiment."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from splendor.actor_critic import ActorCritic, build_actor_critic
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import ReplayGame, ReplayIndex
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.benchmark_search import fixed_requests
from splendor.catalog import load_default_catalog
from splendor.entity_attention import EntityAttention
from splendor.features import encode_decision, encode_decision_state
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.rng import DeterministicRng
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_data import collate_selfplay_samples
from splendor.value_auxiliary import add_auxiliary_heads

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
PRESET = ROOT / "configs/entity_attention_v1/model_256x4.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(36400000)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runtime = ROOT / "runtime" / ART.name / args.device
    runtime.mkdir(parents=True, exist_ok=True)
    config = read_json(PRESET)
    model = build_actor_critic(config)
    capacity = {"model_config": asdict(model.config),
        "parameters": model.trainable_parameter_count(),
        "legacy_parameters": EntityAttention().trainable_parameter_count(),
        "auxiliary_parameters": add_auxiliary_heads(model).trainable_parameter_count(),
        "by_module": {name: sum(p.numel() for p in module.parameters())
                      for name, module in model.named_children()}}
    capacity["fp32_weights_bytes"] = capacity["parameters"] * 4
    assert sum(capacity["by_module"].values()) == capacity["parameters"]

    # Four real positions, using the established archived search and terminal labels.
    manifest = ROOT / "artifacts/entity_attention_r24_history_pool_20261012/pool_games.jsonl"
    with manifest.open() as source:
        record = next(row for line in source if (row := json.loads(line))["round_index"] == 23
                      and row["split"] == "train")
    archive = Path(record["path"])
    assert sha256(archive) == record["archive_sha256"]
    fields = ("round_index", "run_id", "game_id", "split", "decision_count", "policy_samples")
    game = ReplayGame(**{name: record[name] for name in fields}, path=archive)
    index = ReplayIndex([game], cache_bytes=64 * 1024**2)
    indices = [0, 1, 2, 3]
    batch = collate_selfplay_samples([index.cache.sample(game, i) for i in indices])
    learner_config = LearnerConfig(device=args.device, batch_size=4, history_batch_size=2,
                                   learning_rate=2e-5, checkpoint_every_steps=1)
    learner = AlphazeroLearner(model, learner_config, sampling_rng=DeterministicRng(36410000))
    learner.round_steps_target = 2
    learner.replay_identity = {"digest": record["archive_sha256"]}
    first = learner.train_step(batch, references=[])
    assert all(math.isfinite(first[key]) for key in ("total_loss", "policy_loss", "value_loss"))
    assert all(torch.isfinite(p).all() for p in learner.model.parameters())
    checkpoint = runtime / "step-1.pt"
    learner.save(checkpoint)
    frozen = load_frozen_actor_critic_checkpoint(checkpoint, device=args.device)
    for name, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, frozen.state_dict()[name], rtol=0, atol=0)
    resumed = AlphazeroLearner.resume(checkpoint, learner_config,
        replay_identity=learner.replay_identity, round_index=0, round_steps_target=2)
    assert resumed.total_train_step == 1
    resume_exact = None
    if args.device == "cpu":
        learner.train_step(batch, references=[])
        resumed.train_step(batch, references=[])
        for name, value in learner.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
        for key, state in learner.optimizer.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
        resume_exact = True
    pure = runtime / "weights.pt"
    learner.model.save_weights(pure)
    restored = ActorCritic.load_weights(pure, map_location="cpu")
    assert asdict(restored.config) == asdict(model.config)

    # Shared compiled/eager interfaces and native contiguous batches, only S8 per root.
    model = frozen
    cases = fixed_requests()
    encoded = [encode_decision(request) for _, request, _ in cases]
    states = [encode_decision_state(request) for _, request, _ in cases]
    eager = FrozenSearchInference(model, device=args.device, config=InferenceConfig(actor_projection=False))
    evaluator = (eager if args.device == "cpu" else FrozenSearchInference(model, device=args.device,
        config=InferenceConfig(mode="compiled", actor_projection=False, candidate_buckets=(16,32,48,64),
            native_transport="reused", compile_failure_policy="fail_fast")))
    expected = eager.policy_value_encoded(encoded)
    actual = evaluator.policy_value_encoded(encoded)
    for (left_policy, left_value), (right_policy, right_value) in zip(expected, actual):
        torch.testing.assert_close(torch.tensor(left_policy), torch.tensor(right_policy), atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(torch.tensor(left_value), torch.tensor(right_value), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(torch.tensor(eager.value_only_encoded(states)),
                               torch.tensor(evaluator.value_only_encoded(states)), atol=2e-5, rtol=2e-4)
    policies = [NativePUCTSearchPolicy(model, device=args.device, evaluator=evaluator,
        config=MCTSConfig(simulations=8, determinizations=1, root_noise_fraction=0., seed=36430000+i))
        for i in range(len(cases))]
    jobs = [policy.start_job(request) for policy, (_, request, _) in zip(policies, cases)]
    runner = NativeBatchRunner(evaluator, load_default_catalog(), capacity=4, buckets=(16,32,48,64))
    while not all(job.done for job in jobs):
        assert runner.run([job for job in jobs if not job.done]) > 0
    for job, (_, request, _) in zip(jobs, cases):
        expected_sims = 8 if len(request.legal_actions) > 1 else 0
        assert job.result.simulations == expected_sims
        assert 0 <= job.result.chosen_candidate_index < len(request.legal_actions)
    compile_stats = None if evaluator.compiled is None else evaluator.compiled.stats.to_dict()
    if args.device == "cuda":
        assert sum(compile_stats["compiled_calls"].values()) > 0 and compile_stats["compile_failure_fallbacks"] == 0
        torch.cuda.synchronize()
    result = {"status": "passed", "scope": "Finite engineering validation, not training or strength evidence",
        "capacity": capacity, "device": args.device, "seed": 36400000,
        "preset_sha256": sha256(PRESET), "validator_sha256": sha256(Path(__file__)),
        "source": source_identity(ROOT, runtime / "source"), "environment": environment_identity(),
        "training": {"archive": str(archive), "archive_sha256": record["archive_sha256"],
            "decision_indices": indices, "batch_size": 4, "first_step": first,
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
            "az_save_frozen_load": True, "resume_loaded": True, "cpu_resume_exact": resume_exact,
            "pure_weights_load": True},
        "search": {"roots": len(cases), "simulations_per_searchable_root": 8,
            "mode": "compiled" if args.device == "cuda" else "eager", "eager_output_match": True,
            "state_only_output_match": True, "native_batched": True, "compile": compile_stats},
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "seconds": time.perf_counter() - started}
    write_json(ART / f"{args.device}_validation.json", result)
    print({"status": result["status"], "parameters": capacity["parameters"],
        "auxiliary_parameters": capacity["auxiliary_parameters"], "device": args.device,
        "native_search_roots": len(cases), "seconds": result["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
