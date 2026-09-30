"""Native contiguous batch input/output against the per-leaf reference path."""

import numpy as np
import pytest

from splendor.benchmark_search import fixed_requests
from splendor.catalog import load_default_catalog
from splendor.mcts_policy import MCTSConfig

native = pytest.importorskip("splendor._native_search")

BUCKETS = (16, 32, 48)
CAPACITY = 8
FIELD_NAMES = ("global_features", "market_t1", "market_t1_mask", "market_t2", "market_t2_mask",
               "market_t3", "market_t3_mask", "self_reserved", "self_reserved_mask",
               "opponent_reserved", "opponent_reserved_mask", "nobles", "nobles_mask",
               "pending_card", "pending_card_mask")


@pytest.fixture(scope="module")
def native_catalog():
    return native.Catalog(load_default_catalog())


def _config(**overrides):
    values = {"simulations": 8, "determinizations": 2, "root_noise_fraction": 0.0, "seed": 9500000}
    values.update(overrides)
    return MCTSConfig(**values)


def _logits(count):
    return [0.05 * (index + 1) for index in range(count)]


def _run_per_leaf(native_catalog, request, config, word):
    job = native.SearchJob(native_catalog, request, config, word, 12345, False, 0.0)
    while not job.done():
        for item in job.pending():
            encoded = item["encoded"]
            if item["kind"] == "value_only":
                job.submit(item["world"], (), 0.0)
                continue
            job.submit(item["world"], _logits(encoded["candidate_features"].shape[0]), 0.0)
            if item["world"] == -1:
                job.initialize_worlds(request)
    return job.result()


def _run_batch(native_catalog, request, config, word):
    job = native.SearchJob(native_catalog, request, config, word, 12345, False, 0.0)
    buffers = {}
    while not job.done():
        progressed = False
        for state_only in (False, True):
            width = job.pending_width(state_only)
            if not width:
                continue
            bucket = 1 if state_only else next(b for b in BUCKETS if b >= width)
            key = (state_only, bucket)
            if key not in buffers:
                buffers[key] = native.NativeBatch(native_catalog, state_only, CAPACITY, bucket)
            buffer = buffers[key]
            root_row = job.root_pending_now()
            buffer.fill(job)
            rows = buffer.row_count()
            if not rows:
                continue
            arrays = buffer.arrays()
            if state_only:
                buffer.submit(None, np.zeros(rows, dtype=np.float32))
            else:
                logits = np.zeros((rows, buffer.padded_width()), dtype=np.float32)
                for index, mask_row in enumerate(arrays["candidate_mask"]):
                    count = int(np.asarray(mask_row).sum())
                    logits[index, :count] = _logits(count)
                buffer.submit(logits, np.zeros(rows, dtype=np.float32))
            if root_row:
                job.initialize_worlds(request)
            progressed = True
        assert progressed, "native search made no progress"
    return job.result()


@pytest.mark.parametrize("name,decision,_trace", fixed_requests())
def test_batch_rows_match_the_reference_encoder(native_catalog, name, decision, _trace):
    word = 0x1234 + len(name)
    reference_job = native.SearchJob(native_catalog, decision, _config(), word, 12345, False, 0.0)
    batch_job = native.SearchJob(native_catalog, decision, _config(), word, 12345, False, 0.0)
    batch = native.NativeBatch(native_catalog, False, 1, 32)
    assert batch.fill(batch_job) == 1
    arrays = batch.arrays()
    assert arrays["feature_schema_version"] == "splendor-observation-action-v1"
    encoded = reference_job.pending()[0]["encoded"]
    for field in FIELD_NAMES:
        left, right = np.asarray(encoded[field]), np.asarray(arrays[field])[0]
        assert left.shape == right.shape, field
        if left.dtype.kind == "f":
            np.testing.assert_allclose(right, left, rtol=0, atol=0, err_msg=field)
        else:
            assert np.array_equal(right, left), field
    candidates, padded = np.asarray(encoded["candidate_features"]), np.asarray(arrays["candidate_features"])[0]
    assert padded.shape[0] == 32 and candidates.shape[0] <= 32
    np.testing.assert_allclose(padded[:candidates.shape[0]], candidates, rtol=0, atol=0)
    assert np.all(padded[candidates.shape[0]:] == 0)
    mask = np.asarray(arrays["candidate_mask"])[0]
    assert mask.sum() == candidates.shape[0] and not mask[candidates.shape[0]:].any()


@pytest.mark.parametrize("config", [_config(), _config(fold_forced_phases=True, max_simulation_transitions=1)])
def test_batch_submission_matches_per_leaf_search(native_catalog, config):
    for name, decision, _trace in fixed_requests():
        word = 0x1234 + len(name)
        expected = _run_per_leaf(native_catalog, decision, config, word)
        actual = _run_batch(native_catalog, decision, config, word)
        assert actual["chosen_candidate_index"] == expected["chosen_candidate_index"]
        assert actual["root_visit_counts"] == expected["root_visit_counts"]
        assert actual["policy_target"] == expected["policy_target"]


def test_batch_rows_are_claimed_once(native_catalog):
    _name, decision, _trace = fixed_requests()[0]
    job = native.SearchJob(native_catalog, decision, _config(), 0x99, 12345, False, 0.0)
    batch = native.NativeBatch(native_catalog, False, 4, 32)
    assert batch.fill(job) == 1
    assert batch.fill(job) == 0
    batch.submit(np.zeros((1, 32), dtype=np.float32), np.zeros(1, dtype=np.float32))
    assert batch.row_count() == 0
    with pytest.raises(ValueError):
        batch.submit(np.zeros((0, 32), dtype=np.float32), np.zeros(1, dtype=np.float32))


def test_batch_rejects_nonfinite_values(native_catalog):
    _name, decision, _trace = fixed_requests()[0]
    job = native.SearchJob(native_catalog, decision, _config(), 0x77, 12345, False, 0.0)
    batch = native.NativeBatch(native_catalog, False, 4, 32)
    batch.fill(job)
    rows = batch.row_count()
    logits = np.zeros((rows, 32), dtype=np.float32)
    values = np.zeros(rows, dtype=np.float32)
    values[0] = np.nan
    with pytest.raises(ValueError):
        batch.submit(logits, values)


def _run_batch_forcecast(native_catalog, request, config, word):
    """Same schedule as ``_run_batch`` but submits float64 and strided outputs."""

    job = native.SearchJob(native_catalog, request, config, word, 12345, False, 0.0)
    buffers = {}
    while not job.done():
        progressed = False
        for state_only in (False, True):
            width = job.pending_width(state_only)
            if not width:
                continue
            bucket = 1 if state_only else next(b for b in BUCKETS if b >= width)
            key = (state_only, bucket)
            if key not in buffers:
                buffers[key] = native.NativeBatch(native_catalog, state_only, CAPACITY, bucket)
            buffer = buffers[key]
            root_row = job.root_pending_now()
            buffer.fill(job)
            rows = buffer.row_count()
            if not rows:
                continue
            if state_only:
                buffer.submit(None, np.zeros(rows, dtype=np.float64))
            else:
                padded = buffer.padded_width()
                wide = np.zeros((rows, padded * 2), dtype=np.float64)
                for index, item in enumerate(buffer.row_ids()):
                    wide[index, :item[3] * 2:2] = _logits(item[3])
                buffer.submit(wide[:, ::2], np.zeros(rows, dtype=np.float64))
            if root_row:
                job.initialize_worlds(request)
            progressed = True
        assert progressed, "native search made no progress"
    return job.result()


def test_forcecast_and_strided_batch_outputs_match_the_reference(native_catalog):
    for name, decision, _trace in fixed_requests():
        word = 0x1234 + len(name)
        expected = _run_per_leaf(native_catalog, decision, _config(), word)
        actual = _run_batch_forcecast(native_catalog, decision, _config(), word)
        assert actual["chosen_candidate_index"] == expected["chosen_candidate_index"]
        assert actual["root_visit_counts"] == expected["root_visit_counts"]
        assert actual["policy_target"] == expected["policy_target"]


def test_batch_validates_every_row_before_applying_any(native_catalog):
    config = _config()
    jobs = [native.SearchJob(native_catalog, decision, config, 0x300 + index, 12345, False, 0.0)
            for index, (_name, decision, _trace) in enumerate(fixed_requests()[:2])]
    batch = native.NativeBatch(native_catalog, False, 4, 32)
    for job in jobs:
        batch.fill(job)
    rows = batch.row_count()
    assert rows == 2
    logits = np.zeros((rows, 32), dtype=np.float32)
    values = np.zeros(rows, dtype=np.float32)
    values[1] = np.nan
    with pytest.raises(ValueError, match="batch row 1"):
        batch.submit(logits, values)
    assert batch.row_count() == rows
    assert all(job.root_pending_now() for job in jobs)
    assert not any(job.done() for job in jobs)


def test_fused_batch_submission_matches_split_submission(native_catalog):
    config = _config()
    expected_job = native.SearchJob(native_catalog, fixed_requests()[0][1], config, 0x1234, 12345, False, 0.0)
    actual_job = native.SearchJob(native_catalog, fixed_requests()[0][1], config, 0x1234, 12345, False, 0.0)
    for job, fused in ((expected_job, False), (actual_job, True)):
        while not job.done():
            for state_only in (False, True):
                width = job.pending_width(state_only)
                if not width:
                    continue
                bucket = 1 if state_only else next(b for b in BUCKETS if b >= width)
                batch = native.NativeBatch(native_catalog, state_only, CAPACITY, bucket)
                root_row = job.root_pending_now()
                batch.fill(job)
                rows = batch.row_count()
                if not rows:
                    continue
                if state_only:
                    batch.submit(None, np.zeros(rows, dtype=np.float32))
                else:
                    padded = batch.padded_width()
                    logits = np.zeros((rows, padded), dtype=np.float32)
                    for index, item in enumerate(batch.row_ids()):
                        logits[index, :item[3]] = _logits(item[3])
                    if fused:
                        joined = np.concatenate([logits, np.zeros((rows, 1), dtype=np.float32)], axis=1)
                        batch.submit_fused(joined)
                    else:
                        batch.submit(logits, np.zeros(rows, dtype=np.float32))
                if root_row:
                    job.initialize_worlds(fixed_requests()[0][1])
    assert actual_job.result()["root_visit_counts"] == expected_job.result()["root_visit_counts"]
    assert actual_job.result()["chosen_candidate_index"] == expected_job.result()["chosen_candidate_index"]


def test_external_staging_is_written_and_matches_internal_layout(native_catalog):
    _name, decision, _trace = fixed_requests()[0]
    probe = native.NativeBatch(native_catalog, False, 1, 32)
    float_staging = np.zeros(4 * probe.float_stride(), dtype=np.float32)
    mask_staging = np.zeros(4 * probe.mask_stride(), dtype=np.uint8)
    batch = native.NativeBatch(native_catalog, False, 4, 32, float_staging, mask_staging)
    job = native.SearchJob(native_catalog, decision, _config(), 0x444, 12345, False, 0.0)
    assert batch.fill(job) == 1
    reference = native.SearchJob(native_catalog, decision, _config(), 0x444, 12345, False, 0.0).pending()[0]["encoded"]
    arrays = batch.arrays()
    np.testing.assert_allclose(arrays["global_features"][0], reference["global_features"], rtol=0, atol=0)
    packed = batch.packed()
    offsets = {name: offset for name, offset, _count, _shape in packed["float_fields"]}
    np.testing.assert_allclose(float_staging[offsets["global_features"]:offsets["global_features"] + 148],
                               reference["global_features"], rtol=0, atol=0)
    mask_offsets = {name: offset for name, offset, _count, _shape in packed["mask_fields"]}
    count = int(reference["candidate_mask"].sum())
    np.testing.assert_array_equal(
        mask_staging[mask_offsets["candidate_mask"]:mask_offsets["candidate_mask"] + count].astype(bool),
        reference["candidate_mask"])
    with pytest.raises(ValueError, match="float staging"):
        native.NativeBatch(native_catalog, False, 4, 32, float_staging.astype(np.float64), mask_staging)
    with pytest.raises(ValueError, match="mask staging"):
        native.NativeBatch(native_catalog, False, 4, 32, float_staging, mask_staging[:-1])


def test_row_ids_report_world_generation_and_candidate_width(native_catalog):
    _name, decision, _trace = fixed_requests()[0]
    job = native.SearchJob(native_catalog, decision, _config(simulations=8, determinizations=2),
                           0x555, 12345, False, 0.0)
    batch = native.NativeBatch(native_catalog, False, 8, 32)
    assert batch.fill(job) == 1
    assert batch.row_ids() == [(-1, -1, 0, len(decision.legal_actions))]
    batch.submit(np.zeros((1, 32), dtype=np.float32), np.zeros(1, dtype=np.float32))
    job.initialize_worlds(decision)
    leaves = native.NativeBatch(native_catalog, False, 8, 32)
    assert leaves.fill(job) == 2
    rows = leaves.row_ids()
    assert [item[0] for item in rows] == [0, 1]
    assert [item[1] for item in rows] == [0, 0]
    assert [item[2] for item in rows] == [1, 1]
    assert all(item[3] > 0 for item in rows)


def test_stale_batch_rows_are_rejected_without_touching_the_tree(native_catalog):
    _name, decision, _trace = fixed_requests()[0]
    job = native.SearchJob(native_catalog, decision, _config(), 0x666, 12345, False, 0.0)
    batch = native.NativeBatch(native_catalog, False, 4, 32)
    assert batch.fill(job) == 1
    job.submit(-1, list(_logits(len(decision.legal_actions))), 0.0)
    with pytest.raises(ValueError, match="batch row 0"):
        batch.submit(np.zeros((1, 32), dtype=np.float32), np.zeros(1, dtype=np.float32))
    assert not job.done()


def test_value_only_batches_reject_logits(native_catalog):
    batch = native.NativeBatch(native_catalog, True, 2, 1)
    with pytest.raises(ValueError, match="value-only"):
        batch.submit(np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.float32))

@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_cuda_reused_multiline_staging_and_real_model_output(native_catalog):
    import torch
    from pathlib import Path
    from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
    from splendor.search_inference import FrozenSearchInference, InferenceConfig

    checkpoint = Path(__file__).resolve().parents[1] / "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt"
    model = load_frozen_actor_critic_checkpoint(checkpoint, device="cuda")
    evaluator = FrozenSearchInference(model, device="cuda", config=InferenceConfig(
        mode="eager", actor_projection=True, native_transport="reused"))
    requests = [decision for _name, decision, _trace in fixed_requests()[:2]]
    jobs = [native.SearchJob(native_catalog, request, _config(), 0x500 + index,
                             12345, False, 0.0) for index, request in enumerate(requests)]
    probe = native.NativeBatch(native_catalog, False, 1, 32)
    transfer = evaluator.new_native_transfer(state_only=False, capacity=2, width=32,
        float_stride=probe.float_stride(), mask_stride=probe.mask_stride())
    batch = native.NativeBatch(native_catalog, False, 2, 32,
                               transfer.float_staging, transfer.mask_staging)
    for job in jobs:
        assert batch.fill(job) == 1
    arrays = batch.arrays()
    assert [int(row.sum()) for row in arrays["candidate_mask"]] == [30, 17]
    packed = batch.packed()
    transfer._stage_inputs(packed, 2)
    staged = transfer._device_batch(packed, 2)
    for name in FIELD_NAMES + ("candidate_features", "candidate_mask"):
        actual = getattr(staged, name).cpu().numpy()
        np.testing.assert_array_equal(actual, arrays[name], err_msg=name)
    reference = evaluator._tensor_batch_native(arrays, state_only=False)
    with torch.inference_mode():
        expected_logits, expected_values = evaluator._forward(reference)
    handle = transfer.begin_policy_value(batch)
    handle._event.synchronize()
    actual = transfer.out_numpy[:2].copy()
    for row, count in enumerate((30, 17)):
        np.testing.assert_allclose(actual[row, :count], expected_logits[row, :count].cpu().numpy(),
                                   rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual[:, 32], expected_values.cpu().numpy(), rtol=1e-4, atol=1e-4)
    handle.finish()
    # The same slot handles a shorter tail and then a full batch again.
    for indices in ((1,), (0, 1)):
        fresh = [native.SearchJob(native_catalog, requests[index], _config(),
                                  0x900 + index, 12345, False, 0.0) for index in indices]
        for job in fresh:
            batch.fill(job)
        rows = len(indices)
        arrays = batch.arrays()
        packed = batch.packed()
        transfer._stage_inputs(packed, rows)
        staged = transfer._device_batch(packed, rows)
        for name in FIELD_NAMES + ("candidate_features", "candidate_mask"):
            np.testing.assert_array_equal(getattr(staged, name).cpu().numpy(), arrays[name], err_msg=name)
        transfer.begin_policy_value(batch).finish()
    # A different candidate bucket must preserve its own row stride.
    late = fixed_requests()[2][1]
    narrow_probe = native.NativeBatch(native_catalog, False, 1, 16)
    narrow = evaluator.new_native_transfer(state_only=False, capacity=1, width=16,
        float_stride=narrow_probe.float_stride(), mask_stride=narrow_probe.mask_stride())
    narrow_batch = native.NativeBatch(native_catalog, False, 1, 16,
                                      narrow.float_staging, narrow.mask_staging)
    narrow_batch.fill(native.SearchJob(native_catalog, late, _config(), 0xa00, 12345, False, 0.0))
    narrow._stage_inputs(narrow_batch.packed(), 1)
    narrowed = narrow._device_batch(narrow_batch.packed(), 1)
    for name, value in narrow_batch.arrays().items():
        if name != "feature_schema_version":
            np.testing.assert_array_equal(getattr(narrowed, name).cpu().numpy(), value, err_msg=name)
    narrow.begin_policy_value(narrow_batch).finish()
    # The packed control must count its real forward call, not start the clock
    # after readback has already completed.
    from splendor.search_inference import NativeArrayPackedTransfer
    packed_batch = native.NativeBatch(native_catalog, False, 2, 32)
    for index, request in enumerate(requests):
        packed_batch.fill(native.SearchJob(native_catalog, request, _config(),
                                           0xb00 + index, 12345, False, 0.0))
    before = evaluator.stats.forward_seconds
    NativeArrayPackedTransfer(evaluator).begin_policy_value(packed_batch).finish()
    assert evaluator.stats.forward_seconds > before

@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_cuda_reused_value_only_multiline_fields(native_catalog):
    import torch
    from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
    from splendor.search_inference import FrozenSearchInference, InferenceConfig
    from pathlib import Path

    checkpoint = Path(__file__).resolve().parents[1] / "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt"
    model = load_frozen_actor_critic_checkpoint(checkpoint, device="cuda")
    evaluator = FrozenSearchInference(model, device="cuda", config=InferenceConfig(mode="eager"))
    requests = [decision for _name, decision, _trace in fixed_requests()[:2]]
    jobs = [native.SearchJob(native_catalog, request,
                             _config(fold_forced_phases=True, max_simulation_transitions=1),
                             0x700 + index, 12345, False, 0.0)
            for index, request in enumerate(requests)]
    root = native.NativeBatch(native_catalog, False, 2, 32)
    for job in jobs:
        root.fill(job)
    root.submit(np.zeros((2, 32), dtype=np.float32), np.zeros(2, dtype=np.float32))
    for job, request in zip(jobs, requests):
        job.initialize_worlds(request)
    assert all(job.pending_width(True) for job in jobs)
    probe = native.NativeBatch(native_catalog, True, 1, 1)
    transfer = evaluator.new_native_transfer(state_only=True, capacity=4, width=1,
        float_stride=probe.float_stride(), mask_stride=probe.mask_stride())
    batch = native.NativeBatch(native_catalog, True, 4, 1,
                               transfer.float_staging, transfer.mask_staging)
    assert all(batch.fill(job) >= 1 for job in jobs)
    rows = batch.row_count()
    assert rows >= 2
    arrays = batch.arrays()
    packed = batch.packed()
    transfer._stage_inputs(packed, rows)
    staged = transfer._device_batch(packed, rows)
    for name in FIELD_NAMES:
        np.testing.assert_array_equal(getattr(staged, name).cpu().numpy(), arrays[name], err_msg=name)
    reference = evaluator._tensor_batch_native(arrays, state_only=True)
    with torch.inference_mode():
        expected = evaluator._forward_value(reference).cpu().numpy()
    handle = transfer.begin_value_only(batch)
    handle._event.synchronize()
    np.testing.assert_allclose(transfer.values_numpy[:rows], expected, rtol=1e-4, atol=1e-4)
    handle.finish()

@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_cuda_compiled_reused_multiline_matches_eager(native_catalog):
    import torch
    from pathlib import Path
    from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
    from splendor.search_inference import FrozenSearchInference, InferenceConfig

    checkpoint = Path(__file__).resolve().parents[1] / "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt"
    model = load_frozen_actor_critic_checkpoint(checkpoint, device="cuda")
    eager = FrozenSearchInference(model, device="cuda", config=InferenceConfig(
        mode="eager", actor_projection=True))
    compiled = FrozenSearchInference(model, device="cuda", config=InferenceConfig(
        mode="compiled", actor_projection=True, native_transport="reused"))
    probe = native.NativeBatch(native_catalog, False, 1, 32)
    transfer = compiled.new_native_transfer(state_only=False, capacity=2, width=32,
        float_stride=probe.float_stride(), mask_stride=probe.mask_stride())
    batch = native.NativeBatch(native_catalog, False, 2, 32,
                               transfer.float_staging, transfer.mask_staging)
    for index, (_name, request, _trace) in enumerate(fixed_requests()[:2]):
        batch.fill(native.SearchJob(native_catalog, request, _config(),
                                    0xc00 + index, 12345, False, 0.0))
    with torch.inference_mode():
        logits, values = eager._forward(eager._tensor_batch_native(batch.arrays(), state_only=False))
    handle = transfer.begin_policy_value(batch)
    handle._event.synchronize()
    actual = transfer.out_numpy[:2].copy()
    for row, count in enumerate((30, 17)):
        np.testing.assert_allclose(actual[row, :count], logits[row, :count].cpu().numpy(),
                                   rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(actual[:, 32], values.cpu().numpy(), rtol=2e-4, atol=2e-4)
    handle.finish()


def test_row_ids_array_matches_row_ids(native_catalog):
    _name, decision, _trace = fixed_requests()[0]
    config = _config(simulations=8, determinizations=2)
    job = native.SearchJob(native_catalog, decision, config, 0x777, 12345, False, 0.0)
    batch = native.NativeBatch(native_catalog, False, 8, 32)
    assert batch.fill(job) == 1
    batch.submit(np.zeros((1, 32), dtype=np.float32), np.zeros(1, dtype=np.float32))
    job.initialize_worlds(decision)
    assert batch.fill(job) == 2
    readable = batch.row_ids()
    compact = batch.row_ids_array()
    assert compact.shape == (len(readable), 4)
    for row, item in zip(compact, readable):
        assert tuple(int(value) for value in row) == item
