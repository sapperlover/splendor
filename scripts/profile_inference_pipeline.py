"""Profile the frozen inference pipeline: real batches, replayed outside MCTS.

Phase 1 records real native batch buffers while a production collection window
runs (the recording wrapper only copies arrays; search behaviour is unchanged).
Phase 2 replays those rows outside the search through the NumPy packed path.
It measures CPU packing, H2D, forward, output copies and D2H separately,
plus the packed round trip for eager and compiled modes. The replay is a
microbenchmark; the capture window is the live production search path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_collect_throughput import run as run_window  # noqa: E402

from splendor.actor_critic import TorchDecisionBatch  # noqa: E402
from splendor.features import FEATURE_SCHEMA_VERSION  # noqa: E402
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint  # noqa: E402
from splendor.search_inference import FrozenSearchInference, InferenceConfig  # noqa: E402
from splendor.selfplay_fast import FastSelfPlayConfig  # noqa: E402

POLICY_FIELDS = ("global_features", "market_t1", "market_t1_mask", "market_t2", "market_t2_mask",
                 "market_t3", "market_t3_mask", "self_reserved", "self_reserved_mask",
                 "opponent_reserved", "opponent_reserved_mask", "nobles", "nobles_mask",
                 "pending_card", "pending_card_mask", "candidate_features", "candidate_mask")


class _RecordingTransfer:
    """Record the encoded batch, then run the real (possibly reused) transfer."""

    def __init__(self, recorder: "RecordingInference", inner: Any) -> None:
        self.recorder = recorder
        self.inner = inner
        self.float_staging = getattr(inner, "float_staging", None)
        self.mask_staging = getattr(inner, "mask_staging", None)

    def begin_policy_value(self, buffer):
        self.recorder.capture(buffer)
        return self.inner.begin_policy_value(buffer)

    def begin_value_only(self, buffer):
        return self.inner.begin_value_only(buffer)


class RecordingInference:
    """Delegate to a real evaluator while copying the first N batch buffers."""

    def __init__(self, inner: FrozenSearchInference, limit: int) -> None:
        self.inner = inner
        self.limit = limit
        self.records: list[dict[str, np.ndarray]] = []
        self.stats = inner.stats
        self.model = inner.model
        self.device = inner.device
        self.config = inner.config

    def capture(self, buffer) -> None:
        if len(self.records) >= self.limit:
            return
        arrays = buffer.arrays()
        self.records.append({
            name: (np.array(value) if isinstance(value, np.ndarray) and name != "feature_schema_version"
                   else value)
            for name, value in arrays.items()
        })

    def warmup_native(self, widths, *, rows: int = 2) -> float:
        return self.inner.warmup_native(widths, rows=rows)

    def new_native_transfer(self, **kwargs: Any) -> _RecordingTransfer:
        return _RecordingTransfer(self, self.inner.new_native_transfer(**kwargs))

    def policy_value_native(self, arrays):
        if len(self.records) < self.limit:
            self.records.append({
                name: (np.array(value) if isinstance(value, np.ndarray) and name != "feature_schema_version"
                       else value)
                for name, value in arrays.items()
            })
        return self.inner.policy_value_native(arrays)

    def value_only_native(self, arrays):
        return self.inner.value_only_native(arrays)


def concat_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    width = max(int(np.asarray(record["candidate_features"]).shape[1]) for record in records)

    def stack(name: str) -> np.ndarray:
        sample = np.asarray(records[0][name])
        if sample.ndim == 1:
            return np.concatenate([np.asarray(record[name]) for record in records])
        if sample.ndim == 2:
            if name.startswith("candidate"):
                padded = []
                for record in records:
                    value = np.asarray(record[name])
                    pad = width - value.shape[1]
                    padded.append(np.pad(value, ((0, 0), (0, pad))) if pad else value)
                return np.concatenate(padded)
            return np.concatenate([np.asarray(record[name]) for record in records])
        if name == "candidate_features":
            padded = []
            for record in records:
                value = np.asarray(record[name])
                pad = width - value.shape[1]
                padded.append(np.pad(value, ((0, 0), (0, pad), (0, 0))) if pad else value)
            return np.concatenate(padded)
        return np.concatenate([np.asarray(record[name]) for record in records])

    return {
        "feature_schema_version": records[0]["feature_schema_version"],
        **{name: stack(name) for name in POLICY_FIELDS},
    }


def _regroup(records: list[dict[str, np.ndarray]], target_rows: int) -> list[dict[str, np.ndarray]]:
    """Concatenate consecutive recorded batches until each group reaches target rows."""

    groups: list[dict[str, np.ndarray]] = []
    pending: list[dict[str, np.ndarray]] = []
    rows = 0
    for record in records:
        pending.append(record)
        rows += int(np.asarray(record["global_features"]).shape[0])
        if rows >= target_rows:
            groups.append(concat_records(pending))
            pending, rows = [], 0
    if pending:
        groups.append(concat_records(pending))
    return groups


def build(checkpoint: Path, device: str, inference_config: InferenceConfig) -> FrozenSearchInference:
    model = load_frozen_actor_critic_checkpoint(checkpoint, device=device)
    return FrozenSearchInference(model, device=device, config=inference_config)


def measure(evaluator: FrozenSearchInference, arrays, repeats: int) -> dict[str, float]:
    """Time replay stages separately; only full_seconds includes the whole call."""
    rows = int(np.asarray(arrays["global_features"]).shape[0])
    device = evaluator.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    shapes = {name: np.asarray(arrays[name]).shape for name in POLICY_FIELDS}
    mask_names = {name for name in POLICY_FIELDS if np.asarray(arrays[name]).dtype == np.bool_}
    packed = np.concatenate([
        np.ascontiguousarray(arrays[name]).reshape(-1).astype(np.float32, copy=False)
        for name in POLICY_FIELDS
    ])
    host = torch.from_numpy(packed)
    pack = time.perf_counter() - started

    started = time.perf_counter()
    flat = host.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    h2d = time.perf_counter() - started if device.type == "cuda" else 0.0
    values: dict[str, torch.Tensor] = {}
    offset = 0
    for name in POLICY_FIELDS:
        count = int(np.prod(shapes[name]))
        view = flat.narrow(0, offset, count).view(shapes[name])
        values[name] = view.bool() if name in mask_names else view
        offset += count
    prepared = TorchDecisionBatch(feature_schema_version=FEATURE_SCHEMA_VERSION, **values)

    with torch.inference_mode():
        evaluator._forward(prepared)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            logits, critics = evaluator._forward(prepared)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward = (time.perf_counter() - started) / repeats

        if device.type == "cuda":
            width = int(logits.shape[1])
            fused_dev = torch.empty((rows, width + 1), dtype=torch.float32, device=device)
            fused_host = torch.empty((rows, width + 1), dtype=torch.float32, pin_memory=True)
            started = time.perf_counter()
            fused_dev[:, :width].copy_(logits)
            fused_dev[:, width].copy_(critics)
            torch.cuda.synchronize()
            output_copy = time.perf_counter() - started
            started = time.perf_counter()
            fused_host.copy_(fused_dev, non_blocking=True)
            torch.cuda.synchronize()
            d2h = time.perf_counter() - started
        else:
            output_copy = 0.0
            d2h = 0.0

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        evaluator.policy_value_native(arrays)
        timings.append(time.perf_counter() - started)
    full = statistics.median(timings)
    return {
        "rows": rows,
        "padded_width": int(np.asarray(arrays["candidate_features"]).shape[1]),
        "pack_seconds": pack,
        "h2d_seconds": h2d,
        "forward_seconds": forward,
        "output_device_copy_seconds": output_copy,
        "d2h_seconds": d2h,
        "full_seconds": full,
        "forward_positions_per_second": rows / forward if forward else 0.0,
        "full_positions_per_second": rows / full if full else 0.0,
    }


def load_capture_config(path: Path, checkpoint: Path, games: int | None = None) -> FastSelfPlayConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if games is not None:
        raw["num_games"] = games
    raw["checkpoint"] = str(checkpoint.resolve())
    config = FastSelfPlayConfig.from_dict(raw)
    if config.execution.workers != 1:
        raise ValueError("profiling capture requires one worker to retain recorded batches")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("runtime/inference_throughput/scratch_profile"))
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--games", type=int, default=None,
                        help="override config num_games for the capture window")
    parser.add_argument("--capture", type=int, default=96, help="recorded policy batches")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--devices", default="cuda")
    parser.add_argument("--row-targets", default="64,256,1024,4096",
                        help="replay batch sizes in rows; recorded batches are regrouped to these targets")
    args = parser.parse_args()
    torch.set_num_threads(1)

    config = load_capture_config(args.config, args.checkpoint, args.games)
    reports: dict[str, Any] = {
        "config": asdict(config.execution), "num_games": config.num_games,
        "max_possible_active_games": min(config.num_games, config.execution.active_games_per_worker),
        "replay_transport": "numpy_packed_microbenchmark",
        "timing_clocks": "wall seconds; CUDA stages synchronize on completion; stages are measured separately",
        "capture": {}, "replay": {},
    }

    # Phase 1: real collection window with recording.
    device = "cuda" if config.execution.backend.endswith("cuda") else "cpu"
    inference_config = InferenceConfig(
        mode=config.execution.inference_mode,
        actor_projection=config.execution.actor_projection,
        candidate_buckets=config.execution.candidate_buckets,
        compile_mode=config.execution.compile_mode,
        compile_failure_policy=config.execution.compile_failure_policy,
        native_transport=config.execution.native_transport,
    )
    evaluator = build(args.checkpoint, device, inference_config)
    recorder = RecordingInference(evaluator, args.capture)
    window = run_window(config, seconds=args.seconds, scratch=args.scratch, evaluator=recorder,
                        label=config.execution.inference_mode)
    reports["capture"] = {
        "window": {key: window[key] for key in
                   ("process_seconds", "worker_inference", "clock")},
        "recorded_batches": len(recorder.records),
        "recorded_rows": int(sum(np.asarray(record["global_features"]).shape[0]
                                 for record in recorder.records)),
        "recorded_widths": sorted({int(np.asarray(record["candidate_features"]).shape[1])
                                   for record in recorder.records}),
        "recorded_batch_sizes": sorted({int(np.asarray(record["global_features"]).shape[0])
                                        for record in recorder.records}),
    }

    # Phase 2: replay recorded rows without any search.
    rows_total = np.concatenate([np.asarray(record["global_features"])
                                 for record in recorder.records]).shape[0]
    for device_name in args.devices.split(","):
        for mode in ("eager", "compiled"):
            replay_evaluator = build(args.checkpoint, device_name,
                                     replace(inference_config, mode=mode))
            for target in (int(part) for part in args.row_targets.split(",")):
                grouped = _regroup(recorder.records, target)
                if not grouped:
                    continue
                samples = [measure(replay_evaluator, arrays, args.repeats) for arrays in grouped]
                key = f"{device_name}:{mode}:rows={target}"
                reports["replay"][key] = {
                    "batches": len(samples),
                    "rows": int(sum(sample["rows"] for sample in samples)),
                    "mean_rows": statistics.fmean(sample["rows"] for sample in samples),
                    "padded_width": max(sample["padded_width"] for sample in samples),
                    "pack_seconds": statistics.fmean(sample["pack_seconds"] for sample in samples),
                    "h2d_seconds": statistics.fmean(sample["h2d_seconds"] for sample in samples),
                    "forward_seconds": statistics.fmean(sample["forward_seconds"] for sample in samples),
                    "output_device_copy_seconds": statistics.fmean(
                        sample["output_device_copy_seconds"] for sample in samples),
                    "d2h_seconds": statistics.fmean(sample["d2h_seconds"] for sample in samples),
                    "full_seconds": statistics.fmean(sample["full_seconds"] for sample in samples),
                    "forward_positions_per_second": statistics.fmean(
                        sample["forward_positions_per_second"] for sample in samples),
                    "full_positions_per_second": statistics.fmean(
                        sample["full_positions_per_second"] for sample in samples),
                    "compile": replay_evaluator.stats.to_dict()["compile"],
                }
                print(key, json.dumps({name: round(reports["replay"][key][name], 5) for name in
                                       ("mean_rows", "pack_seconds", "h2d_seconds", "forward_seconds",
                                        "output_device_copy_seconds", "d2h_seconds", "full_seconds",
                                        "forward_positions_per_second", "full_positions_per_second")}))
    reports["recorded_rows_total"] = int(rows_total)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
