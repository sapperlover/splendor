"""The profiler must preserve the native CUDA staging ownership contract."""

import numpy as np
import pytest
import torch

from scripts.profile_inference_pipeline import RecordingInference
from splendor.actor_critic import ActorCritic
from splendor.benchmark_search import fixed_requests
from splendor.catalog import load_default_catalog
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner
from splendor.search_inference import FrozenSearchInference, InferenceConfig

native = pytest.importorskip("splendor._native_search")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_profiler_forwards_reused_staging_to_native_batch():
    evaluator = FrozenSearchInference(ActorCritic().eval(), device="cuda",
        config=InferenceConfig(mode="eager", native_transport="reused"))
    recorder = RecordingInference(evaluator, limit=1)
    runner = NativeBatchRunner(recorder, load_default_catalog(), capacity=2,
                               buckets=(16, 32), slots=1)
    buffer, transfer = runner._buffer(0, False, 30)
    assert transfer.float_staging is transfer.inner.float_staging
    assert transfer.mask_staging is transfer.inner.mask_staging
    assert np.shares_memory(buffer.packed()["float_blob"], transfer.float_staging)
    assert np.shares_memory(buffer.packed()["mask_blob"], transfer.mask_staging)

    for index, (_name, request, _trace) in enumerate(fixed_requests()[:2]):
        job = native.SearchJob(runner.native_catalog, request,
                               MCTSConfig(simulations=2, determinizations=1,
                                          root_noise_fraction=0.0),
                               100 + index, 12345, False, 0.0)
        buffer.fill(job)
    expected = np.array(buffer.arrays()["global_features"])
    transfer.begin_policy_value(buffer).finish()
    np.testing.assert_array_equal(recorder.records[0]["global_features"], expected)


def _record(request, width):
    from splendor.features import collate_encoded_decisions, encode_decision
    from scripts.profile_inference_pipeline import POLICY_FIELDS

    batch = collate_encoded_decisions((encode_decision(request),))
    record = {name: np.asarray(getattr(batch, name)).copy() for name in POLICY_FIELDS}
    pad = width - record["candidate_features"].shape[1]
    record["candidate_features"] = np.pad(record["candidate_features"], ((0, 0), (0, pad), (0, 0)))
    record["candidate_mask"] = np.pad(record["candidate_mask"], ((0, 0), (0, pad)))
    record["feature_schema_version"] = batch.feature_schema_version
    return record


def test_profiler_merges_mixed_candidate_widths_and_keeps_tail():
    from scripts.profile_inference_pipeline import _regroup, concat_records

    requests = fixed_requests()
    narrow = _record(requests[2][1], 16)
    wide = _record(requests[0][1], 32)
    merged = concat_records([narrow, wide])
    assert merged["candidate_features"].shape == (2, 32, 82)
    assert merged["candidate_mask"].shape == (2, 32)
    np.testing.assert_array_equal(merged["candidate_features"][0, :16], narrow["candidate_features"][0])
    assert not merged["candidate_features"][0, 16:].any()
    assert not merged["candidate_mask"][0, 16:].any()
    np.testing.assert_array_equal(merged["candidate_features"][1], wide["candidate_features"][0])
    assert len(_regroup([narrow], target_rows=256)) == 1


def test_profiler_uses_configured_game_count(tmp_path):
    import json
    from pathlib import Path
    from scripts.profile_inference_pipeline import load_capture_config

    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "configs/native_search/throughput_cuda_256.json").read_text())
    raw["num_games"] = 500
    raw["execution"]["active_games_per_worker"] = 500
    raw["execution"]["max_batch_size"] = 500
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    checkpoint = root / "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt"
    config = load_capture_config(path, checkpoint)
    assert config.num_games == 500
    assert config.execution.active_games_per_worker == 500
    assert load_capture_config(path, checkpoint, games=512).num_games == 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_profiler_times_pack_and_transfers_separately():
    from scripts.profile_inference_pipeline import measure

    request = fixed_requests()[0][1]
    arrays = _record(request, 32)
    evaluator = FrozenSearchInference(ActorCritic().eval(), device="cuda",
        config=InferenceConfig(mode="eager"))
    result = measure(evaluator, arrays, repeats=1)
    assert result["rows"] == 1 and result["padded_width"] == 32
    assert result["pack_seconds"] > 0
    assert result["h2d_seconds"] > 0
    assert result["output_device_copy_seconds"] > 0
    assert result["d2h_seconds"] > 0
    assert result["full_seconds"] > 0
    assert "d2h_share" not in result
