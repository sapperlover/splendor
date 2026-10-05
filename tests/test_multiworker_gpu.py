"""Real spawned search and shared slot tests with a deterministic owner."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import uuid

import numpy as np
import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.benchmark_search import fixed_requests
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.mcts_policy import MCTSConfig
from splendor.multiworker_gpu import SlotNames, _Slot, _gather, collect
from splendor.native_policy import _default_catalog, _extension, _native_catalog
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID
from splendor.search_inference import FixedValueSearchInference, InferenceStats
from splendor.selfplay_data import game_path, load_game
from splendor.selfplay_fast import (
    ExecutionConfig, FastSelfPlayConfig, _collect_group, _sha256,
)


class _FixedHandle:
    def __init__(self, evaluator, sink, state_only, width):
        self.evaluator = evaluator
        self.sink = sink
        self.state_only = state_only
        self.width = width

    def finish(self):
        rows = self.sink.row_count()
        if self.state_only:
            self.sink.submit(None, np.zeros(rows, dtype=np.float32))
        else:
            self.sink.submit_fused(np.zeros((rows, self.width + 1), dtype=np.float32))
        stats = self.evaluator.stats
        stats.batches += 1
        stats.positions += rows
        stats.batch_sizes[rows] += 1


class _FixedTransfer:
    def __init__(self, evaluator, capacity, width, float_stride, mask_stride):
        self.evaluator = evaluator
        self.width = width
        self.float_staging = np.empty(capacity * float_stride, dtype=np.float32)
        self.mask_staging = np.empty(capacity * mask_stride, dtype=np.bool_)

    def begin_policy_value(self, sink):
        return _FixedHandle(self.evaluator, sink, False, self.width)

    def begin_value_only(self, sink):
        return _FixedHandle(self.evaluator, sink, True, self.width)


class _FixedOwner:
    device = torch.device("cpu")

    def __init__(self):
        self.stats = InferenceStats(config={"mode": "fixed"})

    def new_native_transfer(self, *, state_only, capacity, width, float_stride, mask_stride):
        return _FixedTransfer(self, capacity, width, float_stride, mask_stride)


def _checkpoint(path: Path) -> Path:
    model = ActorCritic()
    torch.save({
        "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "metadata": {"rules_version": RULESET_ID},
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
    }, path)
    return path


def _config(checkpoint: Path, *, backend="native_cuda_pool", games=3, workers=3,
            active=2, batch=2, micro=2, slots=2, simulations=3, determinizations=2):
    return FastSelfPlayConfig(
        protocol="mcts-selfplay-v2", run_id="pool-unit", round_index=0,
        checkpoint=str(checkpoint), num_games=games, seed=43434,
        search_config=MCTSConfig(
            simulations=simulations, determinizations=determinizations,
            seed=9931, root_noise_fraction=0.25, dirichlet_alpha=0.3,
            fold_forced_phases=True, max_simulation_transitions=128),
        execution=ExecutionConfig(
            backend=backend, workers=1, engine_mode="fast", batch_interface="native",
            max_batch_size=batch, active_games_per_worker=active,
            search_workers=workers, total_active_games=active,
            local_microbatch=micro, shared_slots=slots, shared_candidate_width=128,
            max_batch_wait_ms=1),
        validation_fraction=0.0,
    )


def test_shared_layout_multiple_real_rows_and_padding():
    catalog = _native_catalog(_default_catalog())
    requests = [item[1] for item in fixed_requests()[:2]]
    jobs = [_extension().SearchJob(catalog, request, MCTSConfig(
        simulations=2, determinizations=1, root_noise_fraction=0.0),
        123 + index, 101, False, 0.0) for index, request in enumerate(requests)]
    src_probe = _extension().NativeBatch(catalog, False, 1, 32)
    dst_probe = _extension().NativeBatch(catalog, False, 1, 48)
    tag = f"spgpu_test_{uuid.uuid4().hex}"
    names = SlotNames(tag + "_f", tag + "_m", tag + "_r",
                      2 * src_probe.float_stride(), 2 * src_probe.mask_stride(), 2 * 33)
    slot = _Slot(names, create=True)
    try:
        batch = _extension().NativeBatch(catalog, False, 2, 32, slot.floats, slot.masks)
        assert sum(batch.fill(job) for job in jobs) == 2
        original = batch.arrays()
        dest = _FixedTransfer(_FixedOwner(), 2, 48, dst_probe.float_stride(), dst_probe.mask_stride())
        layout = dst_probe.packed()
        base_float = next(item[1] for item in layout["float_fields"] if item[0] == "candidate_features")
        base_mask = next(item[1] for item in layout["mask_fields"] if item[0] == "candidate_mask")
        _gather(slot, 32, dest, "policy", 0, 0, 2,
                src_probe.float_stride(), src_probe.mask_stride(),
                dst_probe.float_stride(), dst_probe.mask_stride(),
                base_float, base_mask, 82)
        float_rows = dest.float_staging.reshape(2, dst_probe.float_stride())
        mask_rows = dest.mask_staging.reshape(2, dst_probe.mask_stride())
        for name, offset, count, shape in layout["float_fields"]:
            actual = float_rows[:, offset:offset + count].reshape(2, *shape)
            expected = np.asarray(original[name])
            np.testing.assert_array_equal(actual[:, :expected.shape[1]] if name == "candidate_features" else actual,
                                          expected)
            if name == "candidate_features":
                assert not actual[:, 32:].any()
        for name, offset, count, shape in layout["mask_fields"]:
            actual = mask_rows[:, offset:offset + count].reshape(2, *shape)
            expected = np.asarray(original[name])
            np.testing.assert_array_equal(actual[:, :expected.shape[1]] if name == "candidate_mask" else actual,
                                          expected)
            if name == "candidate_mask":
                assert not actual[:, 32:].any()
        del original, batch
    finally:
        slot.close()
        slot.unlink()


def test_spawned_pool_matches_single_worker_fixed_evaluator(tmp_path):
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    pool = _config(checkpoint, games=3, workers=3, active=2, batch=2,
                   micro=2, slots=2, simulations=3, determinizations=2)
    reference = _config(checkpoint, backend="native_cpu", games=3, workers=1, active=2,
                        batch=2, simulations=3, determinizations=2)
    pooled_root = tmp_path / "pooled"
    reference_root = tmp_path / "reference"
    results, owner = collect(pool, pooled_root, (0, 1, 2), "controlled", _sha256(checkpoint),
                             evaluator=_FixedOwner())
    _collect_group(reference, reference_root, (0, 1, 2), "controlled", _sha256(checkpoint),
                   evaluator=FixedValueSearchInference())
    assert sum(len(item["completed"]) for item in results) == 3
    assert owner["actual_search_workers"] == 2
    assert owner["actual_active_limit"] == 2
    assert owner["inference"]["physical_model_batches"] > 0
    assert set(game_id for worker in owner["workers"] for game_id in worker["game_ids"]) == {0, 1, 2}
    for game_id in range(3):
        actual, actual_metadata = load_game(game_path(pooled_root, game_id))
        expected, expected_metadata = load_game(game_path(reference_root, game_id))
        assert actual_metadata["trace"] == expected_metadata["trace"]
        for name in ("root_visit_counts", "simulation_budgets", "policy_target",
                     "chosen_candidate_index", "per_world_root_visit_counts"):
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize("workers,active,games,batch", [
    (1, 1, 2, 3), (3, 7, 8, 5), (3, 7, 3, 5), (4, 2, 2, 3),
])
def test_generic_worker_and_active_limits(tmp_path, workers, active, games, batch):
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = _config(checkpoint, games=games, workers=workers, active=active,
                     batch=batch, micro=2, simulations=1, determinizations=1)
    results, owner = collect(config, tmp_path / "games", tuple(range(games)),
                             "controlled", _sha256(checkpoint), evaluator=_FixedOwner())
    assert sum(len(item["completed"]) for item in results) == games
    assert owner["actual_search_workers"] == min(workers, active, games)
    assert owner["actual_active_limit"] == min(active, games)
    assert sum(item["peak_active_games"] for item in results) <= min(active, games)


def test_full_round_admission_is_independent_of_gpu_batch_limit(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = _config(checkpoint, games=1000, workers=8, active=1000,
                     batch=500, micro=64, slots=1, simulations=1, determinizations=1)
    monkeypatch.setattr(_FixedOwner, "warmup_native", lambda self, buckets: 0.0, raising=False)
    results, owner = collect(config, tmp_path / "games", tuple(range(1000)),
                             "admission-window", _sha256(checkpoint), evaluator=_FixedOwner(),
                             benchmark_seconds=0.25, warmup_seconds=0.1)
    assert owner["actual_active_limit"] == 1000
    assert owner["global_max_batch"] == 500
    assert [item["peak_active_games"] for item in sorted(results, key=lambda item: item["worker_id"])] == [125] * 8


def test_value_only_n_gt_one_matches_reference(tmp_path):
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    from dataclasses import replace
    pool = _config(checkpoint, games=2, workers=2, active=2, batch=3,
                   micro=1, slots=2, simulations=4, determinizations=2)
    pool = replace(pool, search_config=replace(pool.search_config,
                                              max_simulation_transitions=1))
    reference = replace(pool, execution=replace(pool.execution, backend="native_cpu"))
    pooled_root = tmp_path / "pooled"
    reference_root = tmp_path / "reference"
    results, owner = collect(pool, pooled_root, (0, 1), "controlled", _sha256(checkpoint),
                             evaluator=_FixedOwner())
    _collect_group(reference, reference_root, (0, 1), "controlled", _sha256(checkpoint),
                   evaluator=FixedValueSearchInference())
    assert sum(item["counters"]["native_value_rows"] for item in results) > 0
    for game_id in range(2):
        actual, _ = load_game(game_path(pooled_root, game_id))
        expected, _ = load_game(game_path(reference_root, game_id))
        for name in ("root_visit_counts", "simulation_budgets", "policy_target",
                     "chosen_candidate_index", "per_world_root_visit_counts"):
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


class _NaNHandle(_FixedHandle):
    def finish(self):
        rows = self.sink.row_count()
        if self.state_only:
            self.sink.submit(None, np.full(rows, np.nan, dtype=np.float32))
        else:
            self.sink.submit_fused(np.full((rows, self.width + 1), np.nan, dtype=np.float32))


class _NaNTransfer(_FixedTransfer):
    def begin_policy_value(self, sink):
        return _NaNHandle(self.evaluator, sink, False, self.width)

    def begin_value_only(self, sink):
        return _NaNHandle(self.evaluator, sink, True, self.width)


class _NaNOwner(_FixedOwner):
    def new_native_transfer(self, *, state_only, capacity, width, float_stride, mask_stride):
        return _NaNTransfer(self, capacity, width, float_stride, mask_stride)


def test_owner_failure_cleans_processes_and_shared_segments(tmp_path):
    import multiprocessing as mp
    from dataclasses import replace
    import glob
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = _config(checkpoint, games=2, workers=2, active=2,
                     simulations=1, determinizations=1)
    from splendor.selfplay_fast import _atomic_json, build_manifest, validate_run
    root = tmp_path / "games"
    _atomic_json(root / "manifest.json", build_manifest(config))
    before = set(glob.glob("/dev/shm/spgpu_*"))
    with pytest.raises(ValueError, match="nonfinite pooled output"):
        collect(config, root, (0, 1), "controlled", _sha256(checkpoint),
                evaluator=_NaNOwner())
    assert (tmp_path / "games" / "failures" / "owner.json").is_file()
    worker_scenes = list((tmp_path / "games" / "failures").glob("worker-*.json"))
    assert worker_scenes
    import json
    reports = [json.loads(path.read_text()) for path in worker_scenes]
    assert any(slot["rows"] for report in reports for slot in report["outstanding"])
    validation = validate_run(root)
    assert validation["batch_failures"] == 1
    assert validation["failed_games"] == validation["exceptions"] > 0
    assert len(validation["batch_failure_game_ids"]) == len(set(validation["batch_failure_game_ids"]))
    assert set(validation["batch_failure_game_ids"]).issubset({0, 1})
    assert set(glob.glob("/dev/shm/spgpu_*")) == before
    assert not any(child.name.startswith("native-search-") for child in mp.active_children())


def test_worker_failure_preserves_original_error_and_cleans_up(tmp_path):
    import multiprocessing as mp
    from dataclasses import replace
    import glob
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = replace(_config(checkpoint, games=2, workers=2, active=2,
                             simulations=1, determinizations=1),
                     max_completed_turns=1)
    before = set(glob.glob("/dev/shm/spgpu_*"))
    with pytest.raises(RuntimeError, match="engineering turn limit"):
        collect(config, tmp_path / "games", (0, 1), "controlled", _sha256(checkpoint),
                evaluator=_FixedOwner())
    assert list((tmp_path / "games" / "failures").glob("worker-*.json"))
    assert set(glob.glob("/dev/shm/spgpu_*")) == before
    assert not any(child.name.startswith("native-search-") for child in mp.active_children())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_pool_complete_games_and_checked_replay(tmp_path):
    from splendor.selfplay_fast import generate, validate_run
    checkpoint = Path("runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt").resolve()
    if not checkpoint.is_file():
        pytest.skip("production checkpoint unavailable")
    config = _config(checkpoint, games=2, workers=2, active=2, batch=3,
                     micro=2, simulations=2, determinizations=1)
    root = tmp_path / "cuda"
    summary = generate(config, root)
    assert summary["completed_games"] == 2
    assert summary["inference"]["physical_model_batches"] > 0
    assert validate_run(root, replay=True)["failed_games"] == 0
    status = __import__("json").loads((root / "status.json").read_text())
    assert status["gpu_owner"]["actual_search_workers"] == 2
    assert len(status["worker_compile"]) == 1


def test_above_compile_bucket_uses_bounded_fallback(tmp_path):
    from dataclasses import replace
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = _config(checkpoint, games=1, workers=1, active=1, batch=3,
                     simulations=1, determinizations=1)
    config = replace(config, execution=replace(config.execution, candidate_buckets=(16,)))
    results, owner = collect(config, tmp_path / "games", (0,), "controlled",
                             _sha256(checkpoint), evaluator=_FixedOwner())
    assert len(results[0]["completed"]) == 1
    assert "128" in owner["width_histogram"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_gpu_pooled_logits_and_values_match_reference(tmp_path):
    from splendor.multiworker_gpu import _OutputSink
    from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
    from splendor.search_inference import FrozenSearchInference, InferenceConfig
    checkpoint = Path("runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt").resolve()
    if not checkpoint.is_file():
        pytest.skip("production checkpoint unavailable")
    catalog = _native_catalog(_default_catalog())
    requests = [item[1] for item in fixed_requests()[:2]]
    jobs = [_extension().SearchJob(catalog, request, MCTSConfig(
        simulations=2, determinizations=1, root_noise_fraction=0.0),
        919 + index, 101, False, 0.0) for index, request in enumerate(requests)]
    source = _extension().NativeBatch(catalog, False, 1, 32)
    target = _extension().NativeBatch(catalog, False, 1, 48)
    tag = f"spgpu_test_{uuid.uuid4().hex}"
    names = SlotNames(tag + "_f", tag + "_m", tag + "_r",
                      2 * source.float_stride(), 2 * source.mask_stride(), 2 * 33)
    slot = _Slot(names, create=True)
    try:
        batch = _extension().NativeBatch(catalog, False, 2, 32, slot.floats, slot.masks)
        assert sum(batch.fill(job) for job in jobs) == 2
        model = load_frozen_actor_critic_checkpoint(checkpoint, device="cuda")
        evaluator = FrozenSearchInference(model, device="cuda",
                                          config=InferenceConfig(mode="eager", native_transport="reused"))
        transfer = evaluator.new_native_transfer(
            state_only=False, capacity=2, width=48,
            float_stride=target.float_stride(), mask_stride=target.mask_stride())
        layout = target.packed()
        base_float = next(item[1] for item in layout["float_fields"] if item[0] == "candidate_features")
        base_mask = next(item[1] for item in layout["mask_fields"] if item[0] == "candidate_mask")
        _gather(slot, 32, transfer, "policy", 0, 0, 2,
                source.float_stride(), source.mask_stride(),
                target.float_stride(), target.mask_stride(),
                base_float, base_mask, 82)
        sink = _OutputSink(2, layout)
        transfer.begin_policy_value(sink).finish()
        assert sink.output is not None
        expected_logits, expected_values = evaluator.policy_value_native(batch.arrays())
        mask = batch.arrays()["candidate_mask"]
        for row in range(2):
            count = int(mask[row].sum())
            np.testing.assert_allclose(sink.output[row, :count],
                                       expected_logits[row, :count], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(sink.output[row, 48], expected_values[row],
                                       rtol=1e-5, atol=1e-5)
        del batch, transfer, sink
    finally:
        slot.close()
        slot.unlink()


def test_slot_generation_rejects_stale_and_duplicate_replies():
    from splendor.multiworker_gpu import _checked_result
    outstanding = {0: (2, object())}
    assert _checked_result(outstanding, 4, 0, 2) == outstanding[0]
    with pytest.raises(RuntimeError, match="stale or duplicate"):
        _checked_result(outstanding, 4, 0, 1)
    del outstanding[0]
    with pytest.raises(RuntimeError, match="stale or duplicate"):
        _checked_result(outstanding, 4, 0, 2)


def test_microbatch_split_across_owner_batches_preserves_search(tmp_path):
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    pool = _config(checkpoint, games=2, workers=1, active=2, batch=1,
                   micro=4, slots=2, simulations=3, determinizations=3)
    reference = _config(checkpoint, backend="native_cpu", games=2, workers=1,
                        active=2, batch=1, simulations=3, determinizations=3)
    pooled_root = tmp_path / "pooled"
    reference_root = tmp_path / "reference"
    results, owner = collect(pool, pooled_root, (0, 1), "controlled", _sha256(checkpoint),
                             evaluator=_FixedOwner())
    _collect_group(reference, reference_root, (0, 1), "controlled", _sha256(checkpoint),
                   evaluator=FixedValueSearchInference())
    assert owner["inference"]["physical_model_batches"] > results[0]["counters"]["native_batch_batches"]
    for game_id in range(2):
        actual, _ = load_game(game_path(pooled_root, game_id))
        expected, _ = load_game(game_path(reference_root, game_id))
        for name in ("root_visit_counts", "simulation_budgets", "policy_target",
                     "chosen_candidate_index", "per_world_root_visit_counts"):
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def test_formal_collection_reports_each_committed_game(tmp_path):
    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = _config(checkpoint, games=2, workers=2, active=2,
                     simulations=1, determinizations=1)
    progress = []
    collect(config, tmp_path / "games", (0, 1), "controlled", _sha256(checkpoint),
            progress_callback=lambda done, total: progress.append((done, total)),
            evaluator=_FixedOwner())
    assert progress == [(1, 2), (2, 2)]


def test_gather_two_worker_layouts_with_different_row_strides():
    catalog = _native_catalog(_default_catalog())
    requests = [item[1] for item in fixed_requests()[:2]]
    slots = []
    batches = []
    try:
        for worker_id, width in enumerate((32, 48)):
            probe = _extension().NativeBatch(catalog, False, 1, width)
            tag = f"spgpu_test_{uuid.uuid4().hex}_{worker_id}"
            names = SlotNames(tag + "_f", tag + "_m", tag + "_r",
                              probe.float_stride(), probe.mask_stride(), width + 1)
            slot = _Slot(names, create=True)
            slots.append(slot)
            batch = _extension().NativeBatch(catalog, False, 1, width,
                                             slot.floats, slot.masks)
            job = _extension().SearchJob(
                catalog, requests[worker_id],
                MCTSConfig(simulations=2, determinizations=1, root_noise_fraction=0.0),
                100 + worker_id, 222, False, 0.0)
            assert batch.fill(job) == 1
            batches.append(batch)
        target = _extension().NativeBatch(catalog, False, 1, 48)
        dest = _FixedTransfer(_FixedOwner(), 2, 48, target.float_stride(), target.mask_stride())
        layout = target.packed()
        base_float = next(item[1] for item in layout["float_fields"] if item[0] == "candidate_features")
        base_mask = next(item[1] for item in layout["mask_fields"] if item[0] == "candidate_mask")
        for row, width in enumerate((32, 48)):
            source = batches[row]
            _gather(slots[row], width, dest, "policy", 0, row, 1,
                    source.float_stride(), source.mask_stride(),
                    target.float_stride(), target.mask_stride(),
                    base_float, base_mask, 82)
        float_rows = dest.float_staging.reshape(2, target.float_stride())
        mask_rows = dest.mask_staging.reshape(2, target.mask_stride())
        for row, batch in enumerate(batches):
            arrays = batch.arrays()
            for name, offset, count, _shape in layout["float_fields"]:
                expected = np.asarray(arrays[name])[0].reshape(-1)
                np.testing.assert_array_equal(float_rows[row, offset:offset + len(expected)],
                                              expected, err_msg=f"worker {row} {name}")
            for name, offset, count, _shape in layout["mask_fields"]:
                expected = np.asarray(arrays[name])[0].reshape(-1)
                np.testing.assert_array_equal(mask_rows[row, offset:offset + len(expected)],
                                              expected, err_msg=f"worker {row} {name}")
        assert not float_rows[0, base_float + 32 * 82:].any()
        assert not mask_rows[0, base_mask + 32:].any()
        del arrays, batch, source, batches
    finally:
        for slot in slots:
            slot.close()
            slot.unlink()


def test_large_unclaimed_queue_failure_exits_process(tmp_path):
    """The first failed game must not leave a feeder flushing 20,000 IDs."""
    from dataclasses import replace
    import json
    import subprocess
    import sys

    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = replace(_config(checkpoint, games=20_000, workers=1, active=1,
                             simulations=1, determinizations=1), max_completed_turns=1)
    settings = tmp_path / "config.json"
    settings.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    program = """
import json
import sys
from pathlib import Path
from splendor.multiworker_gpu import FixedPoolOwner, collect
from splendor.selfplay_fast import FastSelfPlayConfig, build_manifest
config = FastSelfPlayConfig.from_dict(json.loads(Path(sys.argv[1]).read_text()))
manifest = build_manifest(config)
try:
    collect(config, Path(sys.argv[2]), tuple(range(config.num_games)),
            manifest['semantic_fingerprint'], manifest['checkpoint_identity']['sha256'],
            evaluator=FixedPoolOwner())
except RuntimeError as error:
    assert 'engineering turn limit' in str(error)
    print('CAUGHT', flush=True)
else:
    raise AssertionError('expected turn-limit failure')
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(settings), str(tmp_path / "collection")],
        capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CAUGHT" in completed.stdout


def test_pool_failure_reports_are_validated_once_per_game(tmp_path):
    from dataclasses import replace
    from splendor.multiworker_gpu import FixedPoolOwner
    from splendor.selfplay_fast import _atomic_json, build_manifest, validate_run

    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = replace(_config(checkpoint, games=3, workers=1, active=1,
                             simulations=1, determinizations=1), max_completed_turns=1)
    root = tmp_path / "collection"
    manifest = build_manifest(config)
    _atomic_json(root / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="engineering turn limit"):
        collect(config, root, tuple(range(config.num_games)),
                manifest["semantic_fingerprint"], manifest["checkpoint_identity"]["sha256"],
                evaluator=FixedPoolOwner())
    assert (root / "failures" / "owner.json").exists()
    assert (root / "failures" / "worker-0.json").exists()
    _atomic_json(root / "failures" / "game-0.json", {
        "error_detail": "engineering turn limit reached without formal terminal"})
    summary = validate_run(root)
    assert summary["failed_games"] == 1
    assert summary["engineering_truncations"] == 1
    assert summary["exceptions"] == 0
    assert summary["batch_failures"] == 0
    assert summary["missing_games"] == 3
    # Old pool reports lacked origin and could carry a stale request from the
    # previous batch even though the worker, not inference, failed.
    _atomic_json(root / "failures" / "owner.json", {
        "error_detail": "search worker 0: engineering turn limit reached without formal terminal",
        "requests": [{"worker_id": 0, "slot": 0, "generation": 1,
                      "row_offset": 0, "rows": 1}],
    })
    assert validate_run(root)["batch_failures"] == 0
    import json
    worker_scene = json.loads((root / "failures" / "worker-0.json").read_text())
    worker_scene["games"].append({"game_id": 1})
    worker_scene["failed_game_id"] = 0
    _atomic_json(root / "failures" / "worker-0.json", worker_scene)
    with_interrupted = validate_run(root)
    assert with_interrupted["failed_games"] == 2
    assert with_interrupted["engineering_truncations"] == 1
    assert with_interrupted["exceptions"] == 1


def test_pool_window_rates_use_matching_intervals(tmp_path, monkeypatch):
    from scripts import benchmark_collect_throughput as bench

    checkpoint = _checkpoint(tmp_path / "weights.pt")
    config = _config(checkpoint, games=9, workers=2, active=2,
                     simulations=1, determinizations=1)

    def inference(positions, batches):
        return {"logical_model_positions": positions, "physical_model_batches": batches,
                "batch_size_histogram": {"10": batches}, "h2d_seconds": 0.0,
                "forward_seconds": 0.0, "d2h_seconds": 0.0, "wait_seconds": 0.0}

    def worker(wid, start, end, simulations, completed):
        first = {"at": start, "cpu_seconds": 0.0, "simulations": 0,
                 "exported_rows": 0, "batches": 0, "wait_result_seconds": 0.0,
                 "queue_backpressure_seconds": 0.0, "active_games": 1,
                 "root_decisions": 0, "completed_games": 4}
        last = {**first, "at": end, "cpu_seconds": 2.0,
                "simulations": simulations, "completed_games": 4 + completed}
        return {"worker_id": wid, "start": first, "end": last}

    owner_start = {"at": 10.0, "cpu_seconds": 0.0, "gather_seconds": 0.0,
                   "dispatch_seconds": 0.0, "wait_input_seconds": 0.0,
                   "wait_coalesce_seconds": 0.0, "inference": inference(0, 0)}
    owner_end = {**owner_start, "at": 20.0, "cpu_seconds": 3.0,
                 "wait_input_seconds": 2.0, "wait_coalesce_seconds": 1.0,
                 "inference": inference(100, 10)}
    owner = {"benchmark": {"owner_window": {"start": owner_start, "end": owner_end},
                           "worker_windows": [worker(0, 12.0, 17.0, 50, 1),
                                              worker(1, 11.0, 19.0, 80, 2)]},
             "inference": {"compile": {}}, "actual_search_workers": 2}
    results = [{"wall_seconds": 20.0, "initialization_seconds": 1.0,
                "completed": [{}] * 9}] * 2
    monkeypatch.setattr("splendor.multiworker_gpu.collect", lambda *args, **kwargs: (results, owner))
    clock = bench._run_pool(config, seconds=10.0, scratch=tmp_path,
                            warmup_seconds=0.0, label="fixed")["clock"]
    assert clock["common_seconds"] == 5.0
    assert clock["rates"]["simulations_per_second"] == 20.0
    assert clock["rates"]["network_positions_per_second"] == 10.0
    assert clock["rates"]["physical_batches_per_second"] == 1.0
    assert clock["totals"]["completed_games"] == 3
    assert clock["cpu_us_per_simulation"] is None
    assert clock["worker_cpu_us_per_simulation"] == 1e6 * 4 / 130
    assert clock["owner_cpu_us_per_network_position"] == 30_000.0
    assert clock["owner_timings"]["wait_input_seconds"] == 2.0
    assert clock["owner_timings"]["wait_coalesce_seconds"] == 1.0
