"""Bounded real-data engineering check; never writes the production run."""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import statistics
import time

import torch

from splendor.alphazero_recording import read_json, write_json
from splendor.alphazero_replay import PositionPool, ReplayGame
from splendor.alphazero_train import AlphazeroLearner
from splendor.compact_game_cache import CompactGameCache
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig, continuation_targets
from splendor.offline_continuation import ContinuationOfflineConfig
from splendor.offline_mixed_budget import make_batch_drawer
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples
from splendor.training_prefetch import SharedReadonlyPrefetch

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    cfg = ContinuationOfflineConfig.from_dict(read_json(ROOT / "configs/entity_attention_v1/offline_opponent_policy_r21_r37_20261024.json"))
    run = ROOT / "runs/entity_attention_128x2_opponent_policy_r21_r37_20261024"
    progress = read_json(run / "progress.json")
    checkpoint = Path(progress["checkpoint"])
    selected = defaultdict(list)
    for line in Path(cfg.manifest).read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "train" and len(selected[row["round_index"]]) < 4:
            selected[row["round_index"]].append(ReplayGame(**{key: (Path(row[key]) if key == "path" else row[key])
                for key in ("round_index", "run_id", "game_id", "split", "decision_count", "policy_samples", "path")}))
    games = [game for group in selected.values() for game in group]
    auxiliary = ContinuationAuxiliaryConfig(**cfg.continuation_auxiliary)
    cache = CompactGameCache(18*1024**3, continuation_auxiliary=auxiliary)
    index = SimpleNamespace(cache=cache, split_games=lambda split: games if split == "train" else [])
    value = PositionPool(games, cache, "smoke")
    drawer = make_batch_drawer(index, value, cfg.learner.batch_size)
    prefetch = SharedReadonlyPrefetch(index, drawer, 4)
    original_children = [worker.pid for worker in prefetch.pool._pool]

    def learner():
        return AlphazeroLearner.resume(checkpoint, cfg.learner,
            replay_identity=progress["replay_identity"], round_index=0,
            round_steps_target=progress["steps_target"])

    def update(model, parallel):
        if parallel:
            drawn, state, batch, value_batch, labels, _, _ = prefetch.next()
            model.sampling_rng = DeterministicRng.restore(state)
        else:
            drawn = drawer(model.sampling_rng)
            batch, value_batch = collate_selfplay_samples(drawn.samples), collate_selfplay_samples(drawn.value_samples)
            labels = continuation_targets(drawn.value_samples, auxiliary)
        return model.train_step(batch, references=drawn.references, value_batch=value_batch,
            continuation_auxiliary=auxiliary, continuation_labels=labels,
            record_encoder_grad_norms=(model.total_train_step+1) % 100 == 0)

    try:
        a = learner()
        serial_rows = [update(a, False) for _ in range(8)]
        b = learner()
        prefetch.start(b.sampling_rng, 8)
        parallel_rows = [update(b, True) for _ in range(8)]
        assert a.sampling_rng.snapshot() == b.sampling_rng.snapshot()
        for key, tensor in a.model.state_dict().items():
            torch.testing.assert_close(tensor, b.model.state_dict()[key], rtol=0, atol=0)
        for key, state in a.optimizer.state_dict()["state"].items():
            for field, tensor in state.items():
                torch.testing.assert_close(tensor, b.optimizer.state_dict()["state"][key][field], rtol=0, atol=0)
        del a, b
        gc.collect()
        timings = []
        for parallel in (False, True):
            model = learner()
            count, warmup = 160, 20
            if parallel:
                prefetch.start(model.sampling_rng, count+warmup)
            for _ in range(warmup):
                update(model, parallel)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            for _ in range(count):
                update(model, parallel)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            timings.append({"workers": 4 if parallel else 0, "updates": count,
                "warmup_updates": warmup, "seconds": seconds, "updates_per_second": count/seconds})
            del model
            gc.collect()
        result = {"scope": "Engineering smoke on 68 real train games, hot shared cache; not whole-pool throughput or playing strength",
            "checkpoint": str(checkpoint), "source_games": len(games), "batch_size": cfg.learner.batch_size,
            "device": torch.cuda.get_device_name(), "cache_capacity_bytes": cache.max_bytes,
            "shared_payload_bytes": cache.bytes, "arena_bytes": prefetch.used_bytes,
            "worker_pids": original_children, "bitwise_model_optimizer_rng_after_8_updates": True,
            "serial_losses": [row["total_loss"] for row in serial_rows],
            "parallel_losses": [row["total_loss"] for row in parallel_rows], "timings": timings,
            "speed_ratio": timings[1]["updates_per_second"]/timings[0]["updates_per_second"]}
        write_json(ART / "gpu_smoke.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        prefetch.close()


if __name__ == "__main__":
    main()
