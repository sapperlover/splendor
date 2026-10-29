"""Three real-data CUDA updates per remaining arm; no production writes."""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import gc
import importlib.util
import json
import sys
import tempfile

import torch

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_replay import PositionPool, ReplayGame
from splendor.alphazero_train import run_training_round
from splendor.compact_game_cache import CompactGameCache
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig, continuation_targets
from splendor.offline_continuation import ContinuationOfflineConfig, initialize_learner
from splendor.offline_mixed_budget import make_batch_drawer
from splendor.selfplay_data import collate_selfplay_samples
from splendor.training_prefetch import SharedReadonlyPrefetch

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    results = {}
    for key in ("both104k", "value44k"):
        release = read_json(ART / key / "source.json")
        cfg = ContinuationOfflineConfig.from_dict(read_json(ROOT / release["config"]))
        baseline = Path(release["base_source"]) / "src/splendor/alphazero_train.py"
        spec = importlib.util.spec_from_file_location(f"splendor._offline_default_baseline_{key}", baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        selected = defaultdict(list)
        for line in Path(cfg.manifest).read_text().splitlines():
            row = json.loads(line)
            if row["split"] == "train" and len(selected[row["round_index"]]) < 4:
                selected[row["round_index"]].append(ReplayGame(**{k: Path(row[k]) if k == "path" else row[k]
                    for k in ("round_index", "run_id", "game_id", "split", "decision_count", "policy_samples", "path")}))
        games = [game for group in selected.values() for game in group]
        auxiliary = ContinuationAuxiliaryConfig(**cfg.continuation_auxiliary)
        cache = CompactGameCache(18*1024**3, continuation_auxiliary=auxiliary)
        index = SimpleNamespace(cache=cache, split_games=lambda split: games if split == "train" else [])
        value = PositionPool(games, cache, "smoke")
        drawer = make_batch_drawer(index, value, cfg.learner.batch_size)
        prefetch = SharedReadonlyPrefetch(index, drawer, 4)
        identity = {"continuation_auxiliary": asdict(auxiliary)}
        try:
            original = initialize_learner(cfg, identity, 3)
            old = module.AlphazeroLearner(original.model, cfg.learner, sampling_rng=original.sampling_rng)
            old.replay_identity, old.round_steps_target = identity, 3
            old_rows = []
            for i in range(3):
                drawn = drawer(old.sampling_rng)
                old_rows.append(old.train_step(collate_selfplay_samples(drawn.samples), references=drawn.references,
                    value_batch=collate_selfplay_samples(drawn.value_samples), continuation_auxiliary=auxiliary,
                    continuation_labels=continuation_targets(drawn.value_samples, auxiliary), record_encoder_grad_norms=i == 0))
            old_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            new = initialize_learner(cfg, identity, 3)
            with tempfile.TemporaryDirectory(prefix=f"splendor-{key}-default-smoke-") as directory:
                output = Path(directory)
                summary = run_training_round(new, steps_target=3, current_pool=value, history_pool=None,
                    validation_samples=[], validation_games=0, checkpoint_dir=output / "checkpoints",
                    metrics_path=output / "training_metrics.jsonl", batch_drawer=drawer, joint_separate_batches=True,
                    continuation_auxiliary=auxiliary, encoder_grad_norm_every_steps=100, batch_prefetch=prefetch)
                new_rows = [json.loads(line) for line in (output / "training_metrics.jsonl").read_text().splitlines()
                            if json.loads(line).get("event") == "train_step"]
                for a, b in zip(old_rows, new_rows, strict=True):
                    assert a == {k: b[k] for k in a}
                assert old.sampling_rng.snapshot() == new.sampling_rng.snapshot()
                assert torch.equal(old_rng[0], torch.get_rng_state()) and torch.equal(old_rng[1], torch.cuda.get_rng_state())
                for name, tensor in old.model.state_dict().items():
                    torch.testing.assert_close(tensor, new.model.state_dict()[name], rtol=0, atol=0)
                for pa, pb in zip(old.model.parameters(), new.model.parameters()):
                    if pa.grad is not None:
                        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
                for pid, state in old.optimizer.state_dict()["state"].items():
                    for name, tensor in state.items():
                        torch.testing.assert_close(tensor, new.optimizer.state_dict()["state"][pid][name], rtol=0, atol=0)
                assert summary["steps_this_session"] == 3
                results[key] = dict(scope="Engineering verification, no playing-strength conclusion", games=len(games),
                    device=torch.cuda.get_device_name(), batch_sizes=[256, 256], workers=4,
                    cache_capacity_bytes=cache.max_bytes, arena_bytes=prefetch.used_bytes,
                    baseline_source_sha256=sha256(baseline), release_source_sha256=sha256(ART / key / "source.json"),
                    initialization_seed=cfg.initialization_seed, sampling_seed=cfg.sampling_seed,
                    bitwise_model_optimizer_gradients_metrics_rng_after_3_updates=True,
                    diagnostics_tested_at_update=1, summary=summary, raw_new_rows=new_rows)
            del original, old, new
        finally:
            prefetch.close()
        gc.collect()
    write_json(ART / "gpu_smoke.json", results)
    print(json.dumps({key: {k: v for k, v in result.items() if k not in ("summary", "raw_new_rows")}
                      for key, result in results.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
