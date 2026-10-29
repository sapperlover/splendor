"""Two finite real-checkpoint CPU updates and three native roots, no collection."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

import prepare as experiment
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import run_training_round
from splendor.compact_game_cache import CompactGameCache
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.random_budget_iteration import Config, PolicyPool, draw_batch, start_learner


def main():
    torch.set_num_threads(1)
    config = Config.from_dict(read_json(experiment.CONFIG))
    smoke_config = replace(config, learner=replace(config.learner, device="cpu", batch_size=16, history_batch_size=8))
    auxiliary = ContinuationAuxiliaryConfig(**config.continuation_auxiliary)
    rows = [json.loads(line) for line in Path(config.history_manifest["path"]).read_text().splitlines()]
    selected = [next(g for g in rows if g["round_index"] == r and g["split"] == "train") for r in range(73, 80)]
    index = make_index({"games": selected}, 64*1024**2)
    index.cache = CompactGameCache(64*1024**2, continuation_auxiliary=auxiliary)
    value = PositionPool(index.split_games("train"), index.cache, "real-history-subset")
    old_policy = PolicyPool(index.split_games("train"), index.cache, fraction=config.historical_policy_fraction,
        seed=config.historical_policy_seed)
    empty_new = PolicyPool([], index.cache)
    identity = {"protocol": "policy-online-cpu-engineering-v1", "digest": "real-checkpoint-seven-source-games",
                "continuation_auxiliary": asdict(auxiliary)}
    teacher = load_alphazero_checkpoint(config.initial_checkpoint["path"])
    root = experiment.RUNTIME / "cpu_validation"
    root.mkdir(parents=True, exist_ok=True)
    # The real checkpoint requests CUDA. For a CPU-only engineering check, copy
    # its payload and change only execution-device metadata; production uses the original.
    cpu_payload = torch.load(config.initial_checkpoint["path"], map_location="cpu", weights_only=True)
    cpu_payload["learner_config"]["device"] = "cpu"
    cpu_checkpoint = root / "teacher_cpu_execution.pt"
    torch.save(cpu_payload, cpu_checkpoint)
    cpu_identity = {"path": str(cpu_checkpoint), "sha256": sha256(cpu_checkpoint)}
    learner = start_learner(cpu_identity, smoke_config, identity, config.last_round)
    assert learner.total_train_step == 140221 and learner.round_train_step == 0
    assert learner.sampling_rng.snapshot() == teacher.sampling_rng_state
    for key, value_tensor in teacher.model_state_dict.items():
        torch.testing.assert_close(learner.model.state_dict()[key].cpu(), value_tensor, rtol=0, atol=0)
    for pid, states in teacher.optimizer_state_dict["state"].items():
        for key, tensor in states.items():
            torch.testing.assert_close(learner.optimizer.state_dict()["state"][pid][key], tensor, rtol=0, atol=0)
    original_keys = {g.key for g in index.games}
    result = run_training_round(learner, steps_target=config.updates_per_round, current_pool=value, history_pool=None,
        validation_samples=[], validation_games=0, checkpoint_dir=root / "checkpoints", metrics_path=root / "training_metrics.jsonl",
        batch_drawer=lambda rng: draw_batch(old_policy, empty_new, value, rng, 16,
            learner.round_train_step+1, 0, len(selected), original_keys),
        joint_separate_batches=True, max_steps_this_session=2, continuation_auxiliary=auxiliary,
        encoder_grad_norm_every_steps=1)
    rows_metrics = [json.loads(line) for line in (root / "training_metrics.jsonl").read_text().splitlines()]
    updates = [row for row in rows_metrics if row["event"] == "train_step"]
    assert len(updates) == 2 and sum(row["opponent_positions"] for row in updates) > 0
    assert all("shortterm_loss" not in row and row["encoder_grad_norm_opponent_policy"] > 0 for row in updates)
    checkpoint = root / "checkpoints/step-000000002.pt"
    restored = load_frozen_actor_critic_checkpoint(checkpoint, device="cpu")
    assert not hasattr(restored, "shortterm_value_head")
    def forbidden(*_):
        raise AssertionError("Inference executed opponent auxiliary head")
    hook = restored.opponent_policy_head.register_forward_pre_hook(forbidden)
    searches = []
    for seed in (902501, 902502, 902503):
        result_search = NativePUCTSearchPolicy(restored, config=MCTSConfig(simulations=8, determinizations=1,
            seed=seed, root_noise_fraction=0)).search(GameEngine.new(seed, mode="fast").decision())
        assert result_search.simulations == 8
        searches.append({"seed": seed, "simulations": result_search.simulations})
    hook.remove()
    assert sha256(Path(config.initial_checkpoint["path"])) == config.initial_checkpoint["sha256"]
    write_json(experiment.ART / "cpu_validation.json", {"passed": True,
        "scope": "Engineering only: real selected checkpoint, seven original source games,2 CPU B16 updates using only historical policy25percent subset;3 native S8N1 roots,total24. No collection, production training or formal arena.",
        "initial_checkpoint": config.initial_checkpoint, "source_games": [{"path": g["path"], "sha256": sha256(Path(g["path"])),
            "round_index": g["round_index"]} for g in selected], "initial_weights_optimizer_rng_exact": True,
        "learning_rate": learner.config.learning_rate, "training": result, "native_roots": searches,
        "cpu_execution_copy": cpu_identity, "cpu_copy_changes": {"learner_config.device": "cuda -> cpu only; weights/AdamW/all RNG unchanged"},
        "inference_auxiliary_calls": 0, "updates": updates})
    print({"passed": True, "initial_step": 140221, "new_CPU_steps": 2, "native_roots": 3, "total_simulations": 24}, flush=True)


if __name__ == "__main__":
    main()
