"""Bounded old/new CUDA comparison; never writes the production run."""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import gc
import importlib.util
import json
import sys
import time

import torch

from splendor.alphazero_recording import read_json, write_json
from splendor.alphazero_replay import PositionPool, ReplayGame
from splendor.alphazero_train import AlphazeroLearner
from splendor.compact_game_cache import CompactGameCache
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig
from splendor.offline_continuation import ContinuationOfflineConfig
from splendor.offline_mixed_budget import make_batch_drawer
from splendor.rng import DeterministicRng
from splendor.training_prefetch import SharedReadonlyPrefetch

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
NAME = "entity_attention_128x2_opponent_policy_r21_r37_20261024"


def main():
    torch.set_num_threads(1)
    baseline = ROOT / "runtime" / NAME / "source_prefetch_20261024/src/splendor/alphazero_train.py"
    spec = importlib.util.spec_from_file_location("splendor._streaming_baseline_train", baseline)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = ContinuationOfflineConfig.from_dict(read_json(ROOT / "configs/entity_attention_v1/offline_opponent_policy_r21_r37_prefetch_20261024.json"))
    progress = read_json(ROOT / "runs" / NAME / "progress.json")
    checkpoint = ROOT / "runs" / NAME / "checkpoints/step-000115000.pt"
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
    drawer = make_batch_drawer(index, PositionPool(games, cache, "smoke"), cfg.learner.batch_size)
    prefetch = SharedReadonlyPrefetch(index, drawer, 4)

    def learner(cls):
        return cls.resume(checkpoint, cfg.learner, replay_identity=progress["replay_identity"],
            round_index=0, round_steps_target=progress["steps_target"])

    def update(model, *, diagnostic=None):
        drawn, state, batch, value_batch, labels, _, _ = prefetch.next()
        model.sampling_rng = DeterministicRng.restore(state)
        return model.train_step(batch, references=drawn.references, value_batch=value_batch,
            continuation_auxiliary=auxiliary, continuation_labels=labels,
            record_encoder_grad_norms=((model.total_train_step+1) % 100 == 0 if diagnostic is None else diagnostic))

    try:
        a, b = learner(module.AlphazeroLearner), learner(AlphazeroLearner)
        prefetch.start(a.sampling_rng, 8)
        old_rows = [update(a, diagnostic=i in (0, 7)) for i in range(8)]
        old_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        prefetch.start(b.sampling_rng, 8)
        new_rows = [update(b, diagnostic=i in (0, 7)) for i in range(8)]
        assert old_rows == new_rows
        assert a.sampling_rng.snapshot() == b.sampling_rng.snapshot()
        assert torch.equal(old_rng[0], torch.get_rng_state()) and torch.equal(old_rng[1], torch.cuda.get_rng_state())
        for key, tensor in a.model.state_dict().items():
            torch.testing.assert_close(tensor, b.model.state_dict()[key], rtol=0, atol=0)
        for pa, pb in zip(a.model.parameters(), b.model.parameters()):
            if pa.grad is not None:
                torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
        for key, state in a.optimizer.state_dict()["state"].items():
            for field, tensor in state.items():
                torch.testing.assert_close(tensor, b.optimizer.state_dict()["state"][key][field], rtol=0, atol=0)
        del a, b
        gc.collect()
        timings, scalar_reads = [], []
        # ABBA order reduces simple warmup/temperature ordering bias.
        for name, cls in (("old", module.AlphazeroLearner), ("new", AlphazeroLearner),
                          ("new", AlphazeroLearner), ("old", module.AlphazeroLearner)):
            model = learner(cls)
            count, warmup = 100, 20
            prefetch.start(model.sampling_rng, count+warmup+1)
            for _ in range(warmup):
                update(model)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            for _ in range(count):
                update(model)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            timings.append(dict(implementation=name, updates=count, warmup_updates=warmup,
                seconds=elapsed, updates_per_second=count/elapsed))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                update(model, diagnostic=False)
            scalar_reads.append(dict(implementation=name, local_scalar_dense=sum(
                event.count for event in profile.key_averages() if event.key == "aten::_local_scalar_dense")))
            del model
            gc.collect()
        result = dict(scope="Engineering smoke: 68 real train games, B256+256, hot shared cache, 4 workers; not whole-pool throughput or playing strength",
            checkpoint=str(checkpoint), device=torch.cuda.get_device_name(), source_games=len(games),
            bitwise_model_optimizer_gradients_metrics_rng_after_8_updates=True,
            diagnostics_tested_at_updates=[1, 8], losses=[row["total_loss"] for row in new_rows],
            timings=timings, scalar_reads=scalar_reads)
        result["scalar_read_scope"] = "Profiler aten::_local_scalar_dense count includes CPU AdamW step counters; not a count of CUDA synchronization barriers."
        result["speed_ratio"] = sum(r["seconds"] for r in timings if r["implementation"] == "old") / sum(
            r["seconds"] for r in timings if r["implementation"] == "new")
        write_json(ART / "gpu_smoke.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        prefetch.close()


if __name__ == "__main__":
    main()
