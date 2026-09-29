"""Transfer microbenchmark: CPU pack, real H2D, forward, real D2H.

Synthesises a native-batch-shaped payload (same field layout and sizes as a real
policy batch) and measures the costs one live round pays:

* CPU packing of the field views into the final float32 transport layout,
* host-to-device copy of that packed buffer from pageable and from pinned host
  memory into pre-allocated CUDA destinations,
* a pure forward on the resident device buffer,
* device-to-host return of the fused logits/value output into pinned memory.

Every destination is a real CUDA tensor; pageable and pinned rows share the
same pre-allocated destination shape, so allocation cost is reported separately
instead of being folded into one side.  Copies are timed with CUDA events (or a
completion synchronisation) and the D2H result is checked content-wise.  Bytes
are counted from the packed float32 layout actually transferred, so boolean
masks are charged as the float32 values that are really moved.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

FIELD_SHAPES = {
    "global_features": (148,),
    "market_t1": (4, 29), "market_t1_mask": (4,),
    "market_t2": (4, 29), "market_t2_mask": (4,),
    "market_t3": (4, 29), "market_t3_mask": (4,),
    "self_reserved": (3, 29), "self_reserved_mask": (3,),
    "opponent_reserved": (3, 29), "opponent_reserved_mask": (3,),
    "nobles": (3, 17), "nobles_mask": (3,),
    "pending_card": (29,), "pending_card_mask": (),
}
MASK_FIELDS = tuple(name for name in FIELD_SHAPES if name.endswith("_mask"))


def make_batch(rows: int, width: int) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(20260929)
    for name, shape in FIELD_SHAPES.items():
        if name in MASK_FIELDS:
            fields[name] = np.ones((rows, *shape), dtype=np.bool_)
        else:
            fields[name] = rng.standard_normal((rows, *shape)).astype(np.float32)
    fields["candidate_features"] = rng.standard_normal((rows, width, 82)).astype(np.float32)
    fields["candidate_mask"] = np.ones((rows, width), dtype=np.bool_)
    return fields


def packed_bytes(fields: dict[str, np.ndarray]) -> int:
    """Bytes of the packed float32 transport layout (masks included as float32)."""

    return sum(4 * int(np.prod(value.shape, dtype=np.int64)) for value in fields.values())


def pack_concat(fields: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([
        np.ascontiguousarray(value).reshape(-1).astype(np.float32, copy=False)
        for value in fields.values()
    ])


def pack_into(fields: dict[str, np.ndarray], target: torch.Tensor) -> None:
    offset = 0
    for value in fields.values():
        flat = np.ascontiguousarray(value).reshape(-1)
        count = flat.size
        target[offset:offset + count] = torch.from_numpy(flat.astype(np.float32, copy=False))
        offset += count


def timed(function, repeats: int, *, sync: bool) -> list[float]:
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        if sync:
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return durations


def timed_events(function, repeats: int) -> list[float]:
    """Event-measured device time, immune to host submission latency."""

    durations = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        function()
        end_event.record()
        end_event.synchronize()
        durations.append(start_event.elapsed_time(end_event) / 1000.0)
    return durations


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="64,256,512")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the transfer benchmark")
    device = torch.device("cuda")
    report: dict[str, object] = {"device": torch.cuda.get_device_name(), "width": args.width, "rows": {}}
    for rows in (int(part) for part in args.rows.split(",")):
        fields = make_batch(rows, args.width)
        size = packed_bytes(fields)
        concatenated = pack_concat(fields)
        pageable_source = torch.from_numpy(concatenated)
        pinned_source = torch.empty_like(pageable_source, pin_memory=True)
        pinned_source.copy_(pageable_source)
        pageable_destination = torch.empty_like(pageable_source, device=device)
        pinned_destination = torch.empty_like(pageable_source, device=device)
        assert pageable_source.device.type == "cpu" and pageable_source.is_contiguous()
        assert not pageable_source.is_pinned()
        assert pinned_source.is_pinned() and pinned_source.is_contiguous()
        assert pinned_source.dtype == pageable_source.dtype == torch.float32
        assert pageable_destination.device.type == "cuda" and pinned_destination.device.type == "cuda"
        assert pageable_destination.dtype == torch.float32 and pinned_destination.is_contiguous()

        # pre-allocated host destinations for the packed buffers
        pinned_target = torch.empty_like(pageable_source, pin_memory=True)

        # fused output: logits (rows, width) plus one value column per row
        fused_device = torch.arange(rows * (args.width + 1), dtype=torch.float32,
                                    device=device).view(rows, args.width + 1)
        fused_host = torch.empty((rows, args.width + 1), dtype=torch.float32, pin_memory=True)
        split_logits = torch.arange(rows * args.width, dtype=torch.float32,
                                    device=device).view(rows, args.width)
        split_values = torch.arange(rows, dtype=torch.float32, device=device)
        logits_host = torch.empty((rows, args.width), dtype=torch.float32, pin_memory=True)
        values_host = torch.empty(rows, dtype=torch.float32, pin_memory=True)

        allocate_started = time.perf_counter()
        torch.empty_like(pageable_source, device=device)
        torch.empty_like(pageable_source, pin_memory=True)
        allocation_seconds = time.perf_counter() - allocate_started

        for _ in range(3):
            pageable_destination.copy_(pageable_source)
            pinned_destination.copy_(pinned_source, non_blocking=True)
            fused_host.copy_(fused_device, non_blocking=True)
            torch.cuda.synchronize()

        concat_seconds = median(timed(lambda: pack_concat(fields), args.repeats, sync=False))
        copy_into_seconds = median(timed(lambda: pack_into(fields, pinned_target), args.repeats, sync=False))
        pageable_h2d = median(timed(lambda: pageable_destination.copy_(pageable_source, non_blocking=False),
                                    args.repeats, sync=False))
        pinned_h2d_submit = median(timed(lambda: pinned_destination.copy_(pinned_source, non_blocking=True),
                                         args.repeats, sync=False))
        torch.cuda.synchronize()
        pinned_h2d_complete = median(timed(lambda: pinned_destination.copy_(pinned_source, non_blocking=True),
                                           args.repeats, sync=True))
        pinned_h2d_device = median(timed_events(
            lambda: pinned_destination.copy_(pinned_source, non_blocking=True), args.repeats))
        forward_seconds = median(timed(
            lambda: torch.relu(pinned_destination).sum(), args.repeats, sync=True))
        fused_d2h_submit = median(timed(lambda: fused_host[:rows].copy_(fused_device, non_blocking=True),
                                        args.repeats, sync=False))
        torch.cuda.synchronize()
        fused_d2h_complete = median(timed(lambda: fused_host[:rows].copy_(fused_device, non_blocking=True),
                                          args.repeats, sync=True))
        fused_d2h_device = median(timed_events(
            lambda: fused_host[:rows].copy_(fused_device, non_blocking=True), args.repeats))
        split_d2h_complete = median(timed(
            lambda: (logits_host.copy_(split_logits, non_blocking=True),
                     values_host.copy_(split_values, non_blocking=True)), args.repeats, sync=True))

        fused_host[:rows].copy_(fused_device, non_blocking=True)
        torch.cuda.synchronize()
        expected = np.arange(rows * (args.width + 1), dtype=np.float32).reshape(rows, args.width + 1)
        content_ok = bool(np.array_equal(fused_host[:rows].numpy(), expected))

        result = {
            "rows": rows,
            "packed_bytes": size,
            "packed_mib": size / 1024 / 1024,
            "allocation_seconds": allocation_seconds,
            "concat_seconds": concat_seconds,
            "copy_into_pinned_seconds": copy_into_seconds,
            "pageable_h2d_seconds": pageable_h2d,
            "pinned_h2d_submit_seconds": pinned_h2d_submit,
            "pinned_h2d_complete_seconds": pinned_h2d_complete,
            "pinned_h2d_device_seconds": pinned_h2d_device,
            "h2d_speedup_complete": pageable_h2d / pinned_h2d_complete if pinned_h2d_complete else 0.0,
            "pageable_bandwidth_gbps": size / pageable_h2d / 1e9 if pageable_h2d else 0.0,
            "pinned_bandwidth_gbps": size / pinned_h2d_complete / 1e9 if pinned_h2d_complete else 0.0,
            "forward_seconds": forward_seconds,
            "fused_d2h_submit_seconds": fused_d2h_submit,
            "fused_d2h_complete_seconds": fused_d2h_complete,
            "fused_d2h_device_seconds": fused_d2h_device,
            "split_d2h_complete_seconds": split_d2h_complete,
            "fused_d2h_bytes": 4 * rows * (args.width + 1),
            "split_d2h_bytes": 4 * rows * (args.width + 1),
            "content_ok": content_ok,
        }
        report["rows"][str(rows)] = result
        assert content_ok, "D2H returned wrong contents"
        print(json.dumps({key: (round(value, 8) if isinstance(value, float) else value)
                          for key, value in result.items()}, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
