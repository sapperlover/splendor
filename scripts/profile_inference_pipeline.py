"""Profile the frozen inference pipeline: real batches, replayed outside MCTS.

Phase 1 records real native batch buffers while a production collection window
runs (the recording wrapper only copies arrays; search behaviour is unchanged).
Phase 2 replays those rows outside the search and measures H2D, pure forward and
the full round trip, for the eager and compiled paths and for several batch
groupings.  This answers "how much of the wall clock is GPU inference, and does
a bigger batch still help?" without MCTS in the loop.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_collect_throughput import run as run_window  # noqa: E402

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
    rows = sum(int(np.asarray(record["global_features"]).shape[0]) for record in records)

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
    if pending and len(pending) > 1:
        groups.append(concat_records(pending))
    return groups


def build(checkpoint: Path, device: str, mode: str, buckets) -> FrozenSearchInference:
    model = load_frozen_actor_critic_checkpoint(checkpoint, device=device)
    return FrozenSearchInference(
        model, device=device,
        config=InferenceConfig(mode=mode, actor_projection=True, candidate_buckets=tuple(buckets)))


def measure(evaluator: FrozenSearchInference, arrays, repeats: int) -> dict[str, float]:
    rows = int(np.asarray(arrays["global_features"]).shape[0])
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    prepared = evaluator._tensor_batch_native(arrays, state_only=False)
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize()
    h2d = time.perf_counter() - started

    with torch.inference_mode():
        evaluator._forward(prepared)
        if evaluator.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            evaluator._forward(prepared)
        if evaluator.device.type == "cuda":
            torch.cuda.synchronize()
        forward = (time.perf_counter() - started) / repeats

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        logits, values = evaluator.policy_value_native(arrays)
        timings.append(time.perf_counter() - started)
    full = statistics.median(timings)
    return {
        "rows": rows,
        "padded_width": int(np.asarray(arrays["candidate_features"]).shape[1]),
        "h2d_seconds": h2d,
        "forward_seconds": forward,
        "full_seconds": full,
        "forward_positions_per_second": rows / forward if forward else 0.0,
        "full_positions_per_second": rows / full if full else 0.0,
        "d2h_share": max(0.0, 1.0 - (h2d + forward) / full) if full else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("runtime/inference_throughput/scratch_profile"))
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--capture", type=int, default=96, help="recorded policy batches")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--devices", default="cuda")
    parser.add_argument("--row-targets", default="64,256,1024,4096",
                        help="replay batch sizes in rows; recorded batches are regrouped to these targets")
    args = parser.parse_args()
    torch.set_num_threads(1)

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    raw["num_games"] = 64
    config = FastSelfPlayConfig.from_dict(raw)
    reports: dict[str, Any] = {"config": asdict(config.execution), "capture": {}, "replay": {}}

    # Phase 1: real collection window with recording.
    device = "cuda" if config.execution.backend.endswith("cuda") else "cpu"
    evaluator = build(args.checkpoint, device, config.execution.inference_mode,
                      config.execution.candidate_buckets)
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
            replay_evaluator = build(args.checkpoint, device_name, mode,
                                     config.execution.candidate_buckets)
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
                    "h2d_seconds": statistics.fmean(sample["h2d_seconds"] for sample in samples),
                    "forward_seconds": statistics.fmean(sample["forward_seconds"] for sample in samples),
                    "full_seconds": statistics.fmean(sample["full_seconds"] for sample in samples),
                    "forward_positions_per_second": statistics.fmean(
                        sample["forward_positions_per_second"] for sample in samples),
                    "full_positions_per_second": statistics.fmean(
                        sample["full_positions_per_second"] for sample in samples),
                    "compile": replay_evaluator.stats.to_dict()["compile"],
                }
                print(key, json.dumps({name: round(reports["replay"][key][name], 5) for name in
                                       ("mean_rows", "h2d_seconds", "forward_seconds", "full_seconds",
                                        "forward_positions_per_second", "full_positions_per_second")}))
    reports["recorded_rows_total"] = int(rows_total)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
