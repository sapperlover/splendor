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
