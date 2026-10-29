from dataclasses import asdict, replace
import multiprocessing as mp

import numpy as np
import pytest
import torch

from test_continuation_auxiliary import continuation_pool
from test_alphazero_training import az_runs
from test_offline_mixed_budget import mixed
from test_offline_scratch import scratch
from test_value_auxiliary import setup as objective_setup
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.offline_continuation import ContinuationOfflineConfig, prepare_inputs, run
from splendor.offline_mixed_budget import make_batch_drawer
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples
from splendor.training_prefetch import SharedReadonlyPrefetch


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_four_workers_match_batches_and_commit_only_consumed_rng(continuation_pool, tmp_path):
    cfg = replace(continuation_pool, training_workers=4)
    index, value, *_ = prepare_inputs(cfg, tmp_path / "inputs")
    drawer = make_batch_drawer(index, value, cfg.learner.batch_size)
    reference_rng = DeterministicRng(cfg.sampling_seed)
    consumed_rng = DeterministicRng(cfg.sampling_seed)
    before_children = {p.pid for p in mp.active_children()}
    prefetch = SharedReadonlyPrefetch(index, drawer, 4)
    pids = {p.pid for p in prefetch.pool._pool}
    assert len(pids) == 4
    try:
        for arrays, _, _ in index.cache._entries.values():
            for value in arrays.values():
                if hasattr(value, "codes"):
                    assert not value.codes.flags.writeable and not value.values.flags.writeable
                else:
                    assert not value.flags.writeable
        prefetch.start(consumed_rng, 8)
        assert consumed_rng.snapshot() == DeterministicRng(cfg.sampling_seed).snapshot()
        for _ in range(3):
            expected = drawer(reference_rng)
            drawn, state, batch, value_batch, labels, seconds, pid = prefetch.next()
            assert state == reference_rng.snapshot()
            assert drawn.references == expected.references and pid in pids
            for actual, samples in [(batch, expected.samples), (value_batch, expected.value_samples)]:
                wanted = collate_selfplay_samples(samples)
                np.testing.assert_array_equal(actual.decisions.candidate_features, wanted.decisions.candidate_features)
                np.testing.assert_array_equal(actual.policy_targets, wanted.policy_targets)
                np.testing.assert_array_equal(actual.value_targets, wanted.value_targets)
            assert seconds > 0 and labels
    finally:
        prefetch.close()
    assert not ({p.pid for p in mp.active_children()} - before_children)


def test_serial_parallel_and_changed_execution_resume_are_bitwise_identical(continuation_pool, tmp_path):
    cfg = continuation_pool
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    run(cfg, serial)
    run(cfg, parallel, max_steps_this_session=1)
    raw = asdict(cfg)
    raw.pop("training_workers")
    parallel_cfg = replace(ContinuationOfflineConfig.from_dict(raw), cache_mib=18)
    assert parallel_cfg.training_workers == 4
    run(parallel_cfg, parallel)
    a, b = [load_alphazero_checkpoint(p / "checkpoints/final.pt") for p in (serial, parallel)]
    assert a.replay_identity == b.replay_identity
    assert a.sampling_rng_state == b.sampling_rng_state
    torch.testing.assert_close(a.torch_cpu_rng_state, b.torch_cpu_rng_state, rtol=0, atol=0)
    for name, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[name], rtol=0, atol=0)
    for identifier, state in a.optimizer_state_dict["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][identifier][name], rtol=0, atol=0)
    assert not ({p.pid for p in mp.active_children()})


def assert_same_checkpoint(a, b):
    assert a.replay_identity == b.replay_identity
    assert a.sampling_rng_state == b.sampling_rng_state
    torch.testing.assert_close(a.torch_cpu_rng_state, b.torch_cpu_rng_state, rtol=0, atol=0)
    for name, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[name], rtol=0, atol=0)
    for identifier, state in a.optimizer_state_dict["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][identifier][name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["mixed", "legacy_auxiliary", "psw"])
def test_default_workers_execute_shared_and_separate_batches_and_resume(mode, request, tmp_path):
    from splendor.offline_mixed_budget import MixedBudgetOfflineConfig, run as run_mixed
    from splendor.offline_scratch import ScratchOfflineConfig, run as run_scratch
    from splendor.offline_objectives import OfflineObjectiveConfig, run as run_objective
    if mode == "mixed":
        cls, runner, serial_cfg = MixedBudgetOfflineConfig, run_mixed, request.getfixturevalue("mixed")
    elif mode == "legacy_auxiliary":
        cls, runner, serial_cfg = ScratchOfflineConfig, run_scratch, request.getfixturevalue("scratch")
    else:
        cls, runner = OfflineObjectiveConfig, run_objective
        serial_cfg = replace(request.getfixturevalue("objective_setup")[0], value_auxiliary=None,
                             policy_surprise_weighting={"fraction": .5})
    raw = asdict(serial_cfg)
    raw.pop("training_workers")
    parallel_cfg = cls.from_dict(raw)
    assert parallel_cfg.training_workers == 4
    serial, parallel = tmp_path / "serial_default", tmp_path / "parallel_default"
    runner(serial_cfg, serial)
    runner(serial_cfg, parallel, max_steps_this_session=1)
    runner(parallel_cfg, parallel)
    a, b = [load_alphazero_checkpoint(p / "checkpoints/final.pt") for p in (serial, parallel)]
    assert_same_checkpoint(a, b)
    import json
    ready = json.loads((parallel / "prefetch_progress.json").read_text())
    assert ready["workers"] == len(ready["worker_pids"]) == 4
    assert not mp.active_children()


def test_legacy_offline_default_workers_resume_old_config_without_execution_field(mixed, tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import splendor.offline_train as entry
    from splendor.alphazero_recording import read_json, write_json
    rows = [json.loads(line) for line in Path(mixed.manifest).read_text().splitlines()]
    data = dict(protocol="az-offline-data-v1", games=rows, sources=[], rounds=[0, 1])
    data["digest"] = entry._digest(data)
    monkeypatch.setattr(entry, "freeze_dataset", lambda cfg: data)
    monkeypatch.setattr(entry, "source_identity", lambda *args: {})
    raw = dict(source_run="unused", source_artifacts="unused", rounds=[0, 1], model_config=mixed.model_config,
        max_steps=5, validation_positions=8, replay_cache_bytes=8*1024**2, learner=asdict(mixed.learner))
    default = entry.OfflineConfig.from_dict(raw)
    assert default.training_workers == 4
    serial_cfg = replace(default, training_workers=0)
    serial, parallel = tmp_path / "old_serial", tmp_path / "old_parallel"
    entry.run_offline(serial_cfg, serial)
    entry.run_offline(serial_cfg, parallel, max_steps_this_session=2)
    saved = read_json(parallel / "config.json")
    saved.pop("training_workers")
    write_json(parallel / "config.json", saved)
    result = entry.run_offline(default, parallel, resume=True)
    assert_same_checkpoint(*[load_alphazero_checkpoint(p / "checkpoints/final.pt") for p in (serial, parallel)])
    assert read_json(parallel / "prefetch_progress.json")["workers"] == 4
    assert result["replay_cache"]["capacity_bytes"] == 8*1024**2
    assert result["validation_cache"]["capacity_bytes"] == 256*1024**2
    assert not mp.active_children()


@pytest.mark.parametrize("workers", [-1, True, 1.5])
def test_offline_worker_counts_reject_invalid_values(mixed, workers):
    from splendor.offline_objectives import OfflineObjectiveConfig
    from splendor.offline_train import OfflineConfig
    with pytest.raises(ValueError, match="training_workers"):
        replace(mixed, training_workers=workers)
    with pytest.raises(ValueError, match="training_workers"):
        OfflineObjectiveConfig(manifest="", manifest_sha256="", initial_checkpoint="",
            initial_checkpoint_sha256="", training_workers=workers)
    with pytest.raises(ValueError, match="training_workers"):
        OfflineConfig(source_run="", source_artifacts="", model_config={}, max_steps=1, training_workers=workers)


def test_early_stop_and_finetune_inherit_offline_default(mixed):
    from splendor.offline_early_stopping import EarlyStoppingOfflineConfig
    from splendor.offline_finetune import FinetuneOfflineConfig
    raw = asdict(mixed)
    raw.pop("training_workers")
    assert EarlyStoppingOfflineConfig.from_dict(raw).training_workers == 4
    for name in ("model_config", "initialization_seed", "validation_percent_step"):
        raw.pop(name)
    raw.update(initial_checkpoint="unused", initial_checkpoint_sha256="unused")
    assert FinetuneOfflineConfig.from_dict(raw).training_workers == 4


def test_shared_pool_capacity_failure_is_explicit_and_reclaims_loaders(mixed, tmp_path):
    from splendor.compact_game_cache import CompactGameCache
    from splendor.offline_mixed_budget import prepare_inputs as prepare_mixed
    index, value, *_ = prepare_mixed(mixed, tmp_path / "too_small")
    index.cache = CompactGameCache(64)
    with pytest.raises(ValueError, match="exceeds shared cache capacity"):
        SharedReadonlyPrefetch(index, make_batch_drawer(index, value, 4), 4)
    assert not mp.active_children()
    assert index.cache.entries == 0
