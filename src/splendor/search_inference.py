"""Frozen FP32 search inference: eager, projected and compiled tensor paths."""

from __future__ import annotations

import copy
import importlib
import math
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional

from .actor_critic import (
    ActorCritic, ActorProjection, TorchDecisionBatch, TorchStateBatch,
    tensorize_batch, tensorize_state,
)
from .contracts import DecisionRequest
from .features import (
    FEATURE_SCHEMA_VERSION, EncodedDecision, EncodedState, collate_encoded_decisions,
    collate_encoded_states, encode_decision, encode_decision_state,
)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Frozen inference execution options shared by search and collection.

    ``mode`` selects the eager reference path or the bucketed torch.compile
    path; ``actor_projection`` reuses the actor's state projection across
    candidates.  Both are pure execution switches: the feature schema, model
    weights and search semantics do not change.
    """

    mode: str = "eager"
    actor_projection: bool = False
    candidate_buckets: tuple[int, ...] = (16, 32, 48)
    compile_mode: str | None = None
    compile_failure_policy: str = "fallback"
    native_transport: str = "reused"
    batch_buckets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("eager", "compiled"):
            raise ValueError("inference mode must be eager or compiled")
        if type(self.actor_projection) is not bool:
            raise ValueError("actor_projection must be bool")
        if not self.candidate_buckets or any(
            type(bucket) is not int or bucket <= 0 for bucket in self.candidate_buckets
        ):
            raise ValueError("candidate_buckets must be positive integers")
        if tuple(sorted(set(self.candidate_buckets))) != tuple(self.candidate_buckets):
            raise ValueError("candidate_buckets must be sorted and unique")
        if self.compile_mode is not None and not isinstance(self.compile_mode, str):
            raise ValueError("compile_mode must be a string or None")
        if self.compile_failure_policy not in ("fallback", "fail_fast"):
            raise ValueError("compile_failure_policy must be fallback or fail_fast")
        if self.native_transport not in ("reused", "packed"):
            raise ValueError("native_transport must be reused or packed")
        if self.batch_buckets and (self.mode != "compiled" or
                any(type(size) is not int or size <= 0 for size in self.batch_buckets) or
                tuple(sorted(set(self.batch_buckets))) != tuple(self.batch_buckets)):
            raise ValueError("batch_buckets require compiled mode and sorted positive sizes")
        if self.compile_mode == "native-cudagraph" and (self.mode != "compiled" or
                not self.batch_buckets or self.native_transport != "reused"):
            raise ValueError("native-cudagraph requires compiled static batches and reused transport")

    def to_dict(self) -> dict[str, object]:
        result = {
            "mode": self.mode,
            "actor_projection": self.actor_projection,
            "candidate_buckets": list(self.candidate_buckets),
            "compile_mode": self.compile_mode,
            "compile_failure_policy": self.compile_failure_policy,
            "native_transport": self.native_transport,
        }
        if self.batch_buckets:
            result["batch_buckets"] = list(self.batch_buckets)
        return result


_POLICY_FIELDS: tuple[str, ...] = (
    "global_features",
    "market_t1", "market_t1_mask",
    "market_t2", "market_t2_mask",
    "market_t3", "market_t3_mask",
    "self_reserved", "self_reserved_mask",
    "opponent_reserved", "opponent_reserved_mask",
    "nobles", "nobles_mask",
    "pending_card", "pending_card_mask",
    "candidate_features", "candidate_mask",
)
_STATE_FIELDS: tuple[str, ...] = _POLICY_FIELDS[:-2]


def _dynamo_stats() -> dict[str, int]:
    """Best-effort torch.compile counters; empty when dynamo is unavailable."""

    try:
        from torch._dynamo.utils import counters
    except Exception:  # pragma: no cover - only when dynamo is missing
        return {}
    return {
        str(name): int(value)
        for name, value in counters.get("stats", {}).items()
        if isinstance(value, (int, float))
    }


@dataclass(slots=True)
class CompileStats:
    compile_seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    fallbacks: int = 0
    failure_fallbacks: int = 0
    failures: list[dict[str, object]] = field(default_factory=list)
    disabled: dict[str, str] = field(default_factory=dict)
    dynamo_baseline: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        current = _dynamo_stats()
        reached = sorted(set(self.calls) | set(self.compile_seconds) | set(self.disabled))
        return {
            "compile_seconds": dict(sorted(self.compile_seconds.items())),
            "compiled_calls": dict(sorted(self.calls.items())),
            "eager_fallbacks": self.fallbacks,
            "compile_failure_fallbacks": self.failure_fallbacks,
            "compile_failures": [dict(item) for item in self.failures],
            "disabled_graphs": dict(sorted(self.disabled.items())),
            "effective_mode": {name: ("eager" if name in self.disabled else "compiled") for name in reached},
            "dynamo_delta": {
                name: value - self.dynamo_baseline.get(name, 0)
                for name, value in sorted(current.items())
                if value != self.dynamo_baseline.get(name, 0)
            },
        }


def _compile_failure_types() -> tuple[type[BaseException], ...]:
    """Recognised compiler/backend failures that may fall back to eager.

    Deliberately narrower than ``RuntimeError``: CUDA OOM, invalid data and
    model errors share that base class and must stay visible instead of being
    mistaken for a compile failure.
    """

    types: list[type[BaseException]] = []
    for module_name, class_name in (("torch._dynamo.exc", "TorchDynamoException"),
                                    ("torch._dynamo.exc", "FailOnRecompileLimitHit"),
                                    ("torch._inductor.exc", "InductorError")):
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover - only when dynamo/inductor are absent
            continue
        candidate = getattr(module, class_name, None)
        if isinstance(candidate, type) and issubclass(candidate, Exception):
            types.append(candidate)
    return tuple(types)


class CompiledForward:
    """Shape-bucketed compiled forward for the frozen search model.

    One graph per candidate width bucket (batch dimension dynamic by default,
    or padded to explicit static batch buckets); widths
    beyond the widest bucket fall back to eager rather than compiling an
    unbounded number of graphs.  A recognised compiler or backend failure while
    constructing a graph or on its first real call disables that graph and
    serves it eagerly from then on; ``fail_fast`` re-raises instead.
    """

    def __init__(
        self,
        model: ActorCritic,
        *,
        projection: ActorProjection | None,
        buckets: Sequence[int],
        compile_mode: str | None,
        failure_policy: str = "fallback",
        stats_sink: "InferenceStats | None" = None,
        batch_buckets: Sequence[int] = (),
    ) -> None:
        if failure_policy not in ("fallback", "fail_fast"):
            raise ValueError("compile failure policy must be fallback or fail_fast")
        self.model = model
        self.projection = projection
        self.buckets = tuple(sorted(set(buckets)))
        self.batch_buckets = tuple(batch_buckets)
        self.compile_mode = compile_mode
        self.failure_policy = failure_policy
        self.failure_types = _compile_failure_types()
        self.stats = CompileStats(dynamo_baseline=_dynamo_stats())
        self._sink = stats_sink
        self._graphs: dict[str, Callable[..., object]] = {}

    def _sync(self) -> None:
        if self._sink is not None:
            self._sink.compile = self.stats.to_dict()

    # -- bucketing ---------------------------------------------------------

    def width_for(self, candidates: int) -> int | None:
        for bucket in self.buckets:
            if candidates <= bucket:
                return bucket
        return None

    def batch_for(self, rows: int) -> int:
        if not self.batch_buckets:
            return rows
        for size in self.batch_buckets:
            if rows <= size:
                return size
        raise ValueError(f"batch of {rows} exceeds batch_buckets={self.batch_buckets}")

    def _pad_batch(self, batch: TorchDecisionBatch | TorchStateBatch):
        rows = batch.global_features.shape[0]
        padded = self.batch_for(rows)
        if rows == padded:
            return batch
        return type(batch)(feature_schema_version=batch.feature_schema_version, **{
            name: torch.cat((getattr(batch, name), getattr(batch, name).new_zeros(
                (padded - rows, *getattr(batch, name).shape[1:]))), dim=0)
            for name in batch.__dataclass_fields__ if name != "feature_schema_version"
        })

    def _graph_name(self, name: str, rows: int) -> str:
        return f"{name}:b{rows}" if self.batch_buckets else name

    @staticmethod
    def _pad_width(batch: TorchDecisionBatch, width: int) -> TorchDecisionBatch:
        current = batch.candidate_features.shape[1]
        if current >= width:
            return batch
        padding = width - current
        return TorchDecisionBatch(
            feature_schema_version=batch.feature_schema_version,
            **{
                name: getattr(batch, name)
                for name in batch.__dataclass_fields__
                if name not in ("feature_schema_version", "candidate_features", "candidate_mask")
            },
            candidate_features=torch_functional.pad(batch.candidate_features, (0, 0, 0, padding)),
            candidate_mask=torch_functional.pad(batch.candidate_mask, (0, padding)),
        )

    # -- entry points ------------------------------------------------------

    def policy_value(self, batch: TorchDecisionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.width_for(batch.candidate_features.shape[1])
        if width is None:
            self.stats.fallbacks += 1
            return self.model._forward_prepared(batch, projection=self.projection)
        rows = batch.global_features.shape[0]
        prepared = self._pad_batch(self._pad_width(batch, width))
        logits, values = self._dispatch(self._graph_name(f"policy_value:{width}", prepared.global_features.shape[0]),
                                       width, self._policy_value_impl, self._tensors(prepared, _POLICY_FIELDS))
        return logits[:rows], values[:rows]

    def policy_only(self, batch: TorchDecisionBatch) -> torch.Tensor:
        width = self.width_for(batch.candidate_features.shape[1])
        if width is None:
            self.stats.fallbacks += 1
            return self.model._policy_logits_prepared(batch, projection=self.projection)
        rows = batch.global_features.shape[0]
        prepared = self._pad_batch(self._pad_width(batch, width))
        logits = self._dispatch(self._graph_name(f"policy_only:{width}", prepared.global_features.shape[0]),
                               width, self._policy_only_impl, self._tensors(prepared, _POLICY_FIELDS))
        return logits[:rows]

    def value_only(self, batch: TorchStateBatch) -> torch.Tensor:
        rows = batch.global_features.shape[0]
        prepared = self._pad_batch(batch)
        values = self._dispatch(self._graph_name("value_only", prepared.global_features.shape[0]),
                                None, self._value_only_impl, self._tensors(prepared, _STATE_FIELDS))
        return values[:rows]

    # -- graph bodies ------------------------------------------------------

    def _policy_value_impl(self, width: int, *tensors) -> tuple[torch.Tensor, torch.Tensor]:
        prepared = TorchDecisionBatch(
            feature_schema_version="",
            **dict(zip(_POLICY_FIELDS, tensors)),
        )
        state = self.model.encode_state(prepared)
        logits = (
            self.model._actor_logits(prepared, state) if self.projection is None
            else self.model._actor_logits_projected(prepared, state, self.projection)
        )
        return logits, self.model.critic(state).squeeze(-1)

    def _policy_only_impl(self, width: int, *tensors) -> torch.Tensor:
        prepared = TorchDecisionBatch(feature_schema_version="", **dict(zip(_POLICY_FIELDS, tensors)))
        state = self.model.encode_state(prepared)
        return (
            self.model._actor_logits(prepared, state) if self.projection is None
            else self.model._actor_logits_projected(prepared, state, self.projection)
        )

    def _value_only_impl(self, width: int | None, *tensors) -> torch.Tensor:
        prepared = TorchStateBatch(feature_schema_version="", **dict(zip(_STATE_FIELDS, tensors)))
        return self.model.critic(self.model.encode_state(prepared)).squeeze(-1)

    def _tensors(self, batch: object, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        values = tuple(getattr(batch, name) for name in names)
        if not self.batch_buckets:
            for tensor in values:
                torch._dynamo.mark_dynamic(tensor, 0)
        return values

    def _dispatch(self, name: str, width: int | None, implementation: Callable[..., object],
                  tensors: tuple[torch.Tensor, ...]):
        if name in self.stats.disabled:
            self.stats.failure_fallbacks += 1
            return implementation(width, *tensors)
        graph = self._graphs.get(name)
        if graph is None:
            graph = self._compile(name, width, implementation)
            if graph is not None:
                self._graphs[name] = graph
        if graph is None:
            self.stats.failure_fallbacks += 1
            return implementation(width, *tensors)
        return graph(*tensors)

    def _record_failure(self, name: str, stage: str, error: BaseException) -> None:
        self.stats.disabled[name] = stage
        self.stats.failures.append({
            "graph": name,
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
        })
        self._sync()

    def _compile(self, name: str, width: int | None,
                 implementation: Callable[..., object]) -> Callable[..., object] | None:
        def body(*tensors):
            return implementation(width, *tensors)

        # Dynamo caches guards per code object. Compiling functools.partial
        # puts all models, entry points and widths into external_utils.inner's
        # shared cache, which survives collection and repeated evaluations.
        # Give this graph its own code object; ordinary closures share theirs.
        body.__code__ = body.__code__.replace()
        try:
            options = {"dynamic": False} if self.batch_buckets else {}
            mode = "default" if self.compile_mode == "native-cudagraph" else self.compile_mode
            graph = torch.compile(body, mode=mode, **options)
        except self.failure_types as error:
            self._record_failure(name, "construct", error)
            if self.failure_policy == "fail_fast":
                raise
            return None

        def timed(*tensors):
            started = time.perf_counter()
            try:
                # Torch otherwise silently executes eager code when its guard
                # cache limit is reached. Honor the explicit fail-fast contract.
                compile_options = {}
                if self.failure_policy == "fail_fast":
                    compile_options["fail_on_recompile_limit_hit"] = True
                context = (torch._dynamo.config.patch(**compile_options)
                           if compile_options else nullcontext())
                with context:
                    result = graph(*tensors)
            except self.failure_types as error:
                self._graphs.pop(name, None)
                self._record_failure(name, "execute", error)
                if self.failure_policy == "fail_fast":
                    raise
                return body(*tensors)
            if name not in self.stats.compile_seconds:
                self.stats.compile_seconds[name] = time.perf_counter() - started
                self._sync()
            self.stats.calls[name] = self.stats.calls.get(name, 0) + 1
            return result

        return timed


@dataclass(slots=True)
class InferenceStats:
    batches: int = 0
    positions: int = 0
    feature_seconds: float = 0.0
    forward_seconds: float = 0.0
    h2d_seconds: float = 0.0
    d2h_seconds: float = 0.0
    wait_seconds: float = 0.0
    batch_sizes: Counter[int] = field(default_factory=Counter)
    config: dict[str, object] = field(default_factory=dict)
    compile: dict[str, object] = field(default_factory=dict)
    native_graph_captures: int = 0
    native_graph_replays: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "physical_model_batches": self.batches,
            "logical_model_positions": self.positions,
            "feature_seconds": self.feature_seconds,
            "forward_seconds": self.forward_seconds,
            "h2d_seconds": self.h2d_seconds,
            "d2h_seconds": self.d2h_seconds,
            "wait_seconds": self.wait_seconds,
            "batch_size_histogram": {str(k): v for k, v in sorted(self.batch_sizes.items())},
            "mean_batch_size": self.positions / self.batches if self.batches else 0.0,
            "inference": self.config,
            "compile": self.compile,
            **({"native_graph_captures": self.native_graph_captures,
                "native_graph_replays": self.native_graph_replays}
               if self.native_graph_captures else {}),
        }


class NativeBatchRowError(ValueError):
    """One native batch row failed output validation; carries its row index."""

    def __init__(self, row: int, detail: str) -> None:
        super().__init__(f"batch row {row}: {detail}")
        self.row = row


def _first_nonfinite_row(logits: np.ndarray, values: np.ndarray) -> int:
    bad_values = np.flatnonzero(~np.isfinite(values))
    if bad_values.size:
        return int(bad_values[0])
    if logits.ndim == 1:
        bad_logits = np.flatnonzero(~np.isfinite(logits))
    else:
        bad_logits = np.flatnonzero(~np.isfinite(logits).all(axis=tuple(range(1, logits.ndim))))
    return int(bad_logits[0]) if bad_logits.size else 0


def _packed_torch_batch(batch: object, device: torch.device, *, state_only: bool) -> TorchDecisionBatch | TorchStateBatch:
    """Transfer one contiguous FP32 buffer, then view fields on device."""
    names = tuple(name for name in batch.__dataclass_fields__ if name != "feature_schema_version")
    arrays = {name: getattr(batch, name) for name in names}
    shapes = {name: value.shape for name, value in arrays.items()}
    lengths = {name: value.size for name, value in arrays.items()}
    packed = np.concatenate([arrays[name].reshape(-1).astype(np.float32, copy=False) for name in names])
    flat = torch.from_numpy(packed).to(device)
    offset = 0
    values = {}
    for name in names:
        value = flat.narrow(0, offset, lengths[name]).view(shapes[name])
        values[name] = value.bool() if arrays[name].dtype == np.bool_ else value
        offset += lengths[name]
    cls = TorchStateBatch if state_only else TorchDecisionBatch
    return cls(feature_schema_version=batch.feature_schema_version, **values)


@dataclass(slots=True)
class NativeForwardHandle:
    """One launched native forward; finishing it consumes the rows back."""

    rows: int
    width: int
    state_only: bool
    _consume: Callable[[], None]
    _event: object | None = None
    _stats: InferenceStats | None = None

    def done(self) -> bool:
        return self._event is None or bool(self._event.query())  # type: ignore[attr-defined]

    def discard(self) -> None:
        """Wait for in-flight copies without submitting a failed batch."""
        if self._event is not None:
            self._event.synchronize()  # type: ignore[attr-defined]

    def finish(self) -> None:
        if self._event is not None:
            started = time.perf_counter()
            self._event.synchronize()  # type: ignore[attr-defined]
            if self._stats is not None:
                self._stats.wait_seconds += time.perf_counter() - started
        self._consume()


class NativeArrayTransfer:
    """CPU native path: the search buffer already holds final FP32 host views."""

    def __init__(self, evaluator: object) -> None:
        self.evaluator = evaluator

    def begin_policy_value(self, buffer) -> NativeForwardHandle:
        logits, values = self.evaluator.policy_value_native(buffer.arrays())  # type: ignore[attr-defined]
        return NativeForwardHandle(buffer.row_count(), buffer.padded_width(), False,
                                   lambda: buffer.submit(logits, values))

    def begin_value_only(self, buffer) -> NativeForwardHandle:
        values = self.evaluator.value_only_native(buffer.arrays())  # type: ignore[attr-defined]
        return NativeForwardHandle(buffer.row_count(), 0, True,
                                   lambda: buffer.submit(None, values))


class NativeArrayPackedTransfer:
    """Pre-reuse transport: per batch ``concatenate`` plus fresh device tensors.

    Kept as the measured control for the buffer-reuse comparison; production
    collection uses :class:`NativePackedTransfer`.
    """

    def __init__(self, evaluator: "FrozenSearchInference") -> None:
        self.evaluator = evaluator

    def begin_policy_value(self, buffer) -> NativeForwardHandle:
        prepared_started = time.perf_counter()
        prepared = self.evaluator._tensor_batch_native(buffer.arrays(), state_only=False)
        self.evaluator.stats.feature_seconds += time.perf_counter() - prepared_started
        forward_started = time.perf_counter()
        with torch.inference_mode():
            logits, values = self.evaluator._forward(prepared)
            forward = time.perf_counter() - forward_started
            readback_started = time.perf_counter()
            logits_array = logits.detach().cpu().numpy()
            value_array = values.detach().cpu().numpy()
        self.evaluator._record_native(int(value_array.shape[0]), h2d=0.0, forward=forward,
                                      d2h=time.perf_counter() - readback_started)
        return NativeForwardHandle(buffer.row_count(), buffer.padded_width(), False,
                                   lambda: buffer.submit(logits_array, value_array))

    def begin_value_only(self, buffer) -> NativeForwardHandle:
        prepared_started = time.perf_counter()
        prepared = self.evaluator._tensor_batch_native(buffer.arrays(), state_only=True)
        self.evaluator.stats.feature_seconds += time.perf_counter() - prepared_started
        forward_started = time.perf_counter()
        with torch.inference_mode():
            output = self.evaluator._forward_value(prepared)
            forward = time.perf_counter() - forward_started
            readback_started = time.perf_counter()
            values = output.detach().cpu().numpy()
        self.evaluator._record_native(int(values.shape[0]), h2d=0.0, forward=forward,
                                      d2h=time.perf_counter() - readback_started)
        return NativeForwardHandle(buffer.row_count(), 0, True,
                                   lambda: buffer.submit(None, values))


class NativePackedTransfer:
    """Pinned staging and reusable device buffers for one native batch shape.

    Native encoding writes straight into the pinned host blob, one H2D per blob
    moves the used prefix into reused device buffers, and logits/value return in
    one fused D2H into a pinned host buffer.  The owning slot never starts a new
    forward before its previous handle has been consumed, so the pinned staging
    and the device buffers are never overwritten while a copy is in flight.
    """

    def __init__(self, evaluator: "FrozenSearchInference", *, state_only: bool, capacity: int,
                 width: int, float_stride: int, mask_stride: int) -> None:
        self.evaluator = evaluator
        self.state_only = state_only
        self.capacity = capacity
        self.width = 0 if state_only else width
        self.float_stride = float_stride
        self.mask_stride = mask_stride
        device = evaluator.device
        self.float_host = torch.empty(capacity * float_stride, dtype=torch.float32, pin_memory=True)
        self.mask_host = torch.empty(capacity * mask_stride, dtype=torch.bool, pin_memory=True)
        self.float_staging = self.float_host.numpy()
        self.mask_staging = self.mask_host.numpy()
        self.float_dev = torch.empty(capacity * float_stride, dtype=torch.float32, device=device)
        self.mask_dev = torch.empty(capacity * mask_stride, dtype=torch.bool, device=device)
        self.values_dev = torch.empty(capacity, dtype=torch.float32, device=device)
        self.values_host = torch.empty(capacity, dtype=torch.float32, pin_memory=True)
        self.values_numpy = self.values_host.numpy()
        if state_only:
            self.out_dev = None
            self.out_host = None
            self.out_numpy = None
        else:
            self.out_dev = torch.empty((capacity, width + 1), dtype=torch.float32, device=device)
            self.out_host = torch.empty((capacity, width + 1), dtype=torch.float32, pin_memory=True)
            self.out_numpy = self.out_host.numpy()
        self.event = torch.cuda.Event()
        # The blob views, field offsets and strides of one native batch never
        # change; only the number of filled rows does.  Cache the layout once and
        # reuse the device views per row count instead of rebuilding them.
        self._layout: Mapping[str, object] | None = None
        self._views: dict[int, object] = {}
        self._graphs: dict[int, object] = {}

    def _direct_graph(self, prepared, rows: int) -> bool:
        if (self.evaluator.config.compile_mode != "native-cudagraph" or
                (not self.state_only and self.width > self.evaluator.config.candidate_buckets[-1])):
            return False
        graph = self._graphs.get(rows)
        if graph is None:
            def forward_into_output():
                if self.state_only:
                    self.values_dev[:rows].copy_(self.evaluator._forward_value(prepared))
                else:
                    logits, values = self.evaluator._forward(prepared)
                    self.out_dev[:rows, :self.width].copy_(logits)
                    self.out_dev[:rows, self.width].copy_(values)
            # Compile outside capture. Each transfer owns the device addresses,
            # output and graph pool; another lane never replays these buffers.
            for _ in range(2):
                forward_into_output()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward_into_output()
            self._graphs[rows] = graph
            self.evaluator.stats.native_graph_captures += 1
        graph.replay()
        self.evaluator.stats.native_graph_replays += 1
        return True

    def _layout_for(self, buffer) -> Mapping[str, object]:
        layout = self._layout
        if layout is None:
            layout = buffer.packed()
            self._layout = layout
        return layout

    def _prepared(self, layout: Mapping[str, object], rows: int):
        prepared = self._views.get(rows)
        if prepared is None:
            prepared = self._device_batch(layout, rows)
            if len(self._views) >= 8:
                self._views.clear()
            self._views[rows] = prepared
        return prepared

    def _device_batch(self, packed: Mapping[str, object], rows: int):
        values: dict[str, torch.Tensor] = {}
        # NativeBatch stores complete rows.  Preserve that stride for every field.
        float_rows = self.float_dev.view(self.capacity, self.float_stride)[:rows]
        mask_rows = self.mask_dev.view(self.capacity, self.mask_stride)[:rows]
        for name, offset, count, shape in packed["float_fields"]:  # type: ignore[union-attr]
            view = float_rows[:, offset:offset + count]
            values[name] = view.view(rows, *shape) if shape else view.view(rows)
        for name, offset, count, shape in packed["mask_fields"]:  # type: ignore[union-attr]
            view = mask_rows[:, offset:offset + count]
            values[name] = view.view(rows, *shape) if shape else view.view(rows)
        cls = TorchStateBatch if self.state_only else TorchDecisionBatch
        return cls(feature_schema_version=FEATURE_SCHEMA_VERSION, **values)

    def _stage_inputs(self, packed: Mapping[str, object], rows: int) -> float:
        started = time.perf_counter()
        padded = self.evaluator.compiled.batch_for(rows) if self.evaluator.compiled is not None else rows
        if padded != rows:
            self.float_host[rows * self.float_stride:padded * self.float_stride].zero_()
            self.mask_host[rows * self.mask_stride:padded * self.mask_stride].zero_()
        self.float_dev[:padded * self.float_stride].copy_(
            self.float_host[:padded * self.float_stride], non_blocking=True)
        self.mask_dev[:padded * self.mask_stride].copy_(
            self.mask_host[:padded * self.mask_stride], non_blocking=True)
        return time.perf_counter() - started

    def begin_policy_value(self, buffer) -> NativeForwardHandle:
        layout = self._layout_for(buffer)
        rows = buffer.row_count()
        h2d = self._stage_inputs(layout, rows)
        padded = self.evaluator.compiled.batch_for(rows) if self.evaluator.compiled is not None else rows
        prepared = self._prepared(layout, padded)
        started = time.perf_counter()
        with torch.inference_mode():
            direct = self._direct_graph(prepared, padded)
            if not direct:
                logits, values = self.evaluator._forward(prepared)
            forward = time.perf_counter() - started
            started = time.perf_counter()
            if not direct:
                self.out_dev[:rows, :self.width].copy_(logits[:rows])
                self.out_dev[:rows, self.width].copy_(values[:rows])
            self.out_host[:rows].copy_(self.out_dev[:rows], non_blocking=True)
        d2h = time.perf_counter() - started
        self.event.record()
        fused = self.out_numpy[:rows]
        stats = self.evaluator.stats
        record = self.evaluator._record_native

        def consume() -> None:
            record(rows, h2d=h2d, forward=forward, d2h=d2h)
            buffer.submit_fused(fused)

        return NativeForwardHandle(rows, self.width, False, consume, self.event, stats)

    def begin_value_only(self, buffer) -> NativeForwardHandle:
        layout = self._layout_for(buffer)
        rows = buffer.row_count()
        h2d = self._stage_inputs(layout, rows)
        padded = self.evaluator.compiled.batch_for(rows) if self.evaluator.compiled is not None else rows
        prepared = self._prepared(layout, padded)
        started = time.perf_counter()
        with torch.inference_mode():
            direct = self._direct_graph(prepared, padded)
            if not direct:
                values = self.evaluator._forward_value(prepared)
            forward = time.perf_counter() - started
            started = time.perf_counter()
            if not direct:
                self.values_dev[:rows].copy_(values[:rows])
            self.values_host[:rows].copy_(self.values_dev[:rows], non_blocking=True)
        d2h = time.perf_counter() - started
        self.event.record()
        result = self.values_numpy[:rows]
        stats = self.evaluator.stats
        record = self.evaluator._record_native

        def consume() -> None:
            record(rows, h2d=h2d, forward=forward, d2h=d2h)
            buffer.submit(None, result)

        return NativeForwardHandle(rows, 0, True, consume, self.event, stats)


class FixedValueSearchInference:
    """Instant constant policy/value double for throughput ceilings.

    Returns uniform logits and value 0 without touching torch, while the C++
    search, native batch transport, encoding and scheduling still run exactly
    as in production.  The tree shape differs from a real run because priors are
    uniform instead of network-shaped, so this measures the search/transport
    ceiling rather than equivalence.
    """

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = None
        self.config = InferenceConfig(mode="eager", actor_projection=False)
        self.stats = InferenceStats(config={"mode": "fixed-value"})

    def warmup_native(self, widths: Sequence[int], *, rows: int = 2) -> float:
        return 0.0

    def new_native_transfer(self, *, state_only: bool, capacity: int, width: int,
                            float_stride: int, mask_stride: int) -> NativeArrayTransfer:
        return NativeArrayTransfer(self)

    def _record(self, size: int) -> None:
        self.stats.batches += 1
        self.stats.positions += size
        self.stats.batch_sizes[size] += 1

    def policy_value_native(self, arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        rows = int(arrays["global_features"].shape[0])
        width = int(arrays["candidate_features"].shape[1])
        self._record(rows)
        return (np.zeros((rows, width), dtype=np.float32),
                np.zeros(rows, dtype=np.float32))

    def value_only_native(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        rows = int(arrays["global_features"].shape[0])
        self._record(rows)
        return np.zeros(rows, dtype=np.float32)

    def policy_value_encoded(self, encoded: Sequence[EncodedDecision]) -> tuple[tuple[tuple[float, ...], float], ...]:
        self._record(len(encoded))
        return tuple((tuple(0.0 for _ in range(item.candidate_features.shape[0])), 0.0)
                     for item in encoded)

    def value_only_encoded(self, encoded: Sequence[EncodedState]) -> tuple[float, ...]:
        self._record(len(encoded))
        return (0.0,) * len(encoded)


class FrozenSearchInference:
    """Own an eval-mode model; learner-owned training models are copied once."""

    def __init__(
        self,
        model: ActorCritic,
        *,
        device: torch.device | str = "cpu",
        config: InferenceConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA search inference requested but CUDA is unavailable")
        self.config = config or InferenceConfig()
        if self.config.compile_mode == "native-cudagraph" and self.device.type != "cuda":
            raise ValueError("native-cudagraph requires a CUDA device")
        self.model = (copy.deepcopy(model) if model.training else model).to(self.device).eval()
        self.projection = (
            self.model.build_actor_projection() if self.config.actor_projection else None
        )
        self.stats = InferenceStats(config=self.config.to_dict())
        self.compiled = (
            CompiledForward(self.model, projection=self.projection,
                            buckets=self.config.candidate_buckets,
                            compile_mode=self.config.compile_mode,
                            failure_policy=self.config.compile_failure_policy,
                            stats_sink=self.stats, batch_buckets=self.config.batch_buckets)
            if self.config.mode == "compiled" else None
        )
        if self.compiled is not None:
            self.stats.compile = self.compiled.stats.to_dict()

    @staticmethod
    def _synthetic_state(rows: int, device: torch.device) -> TorchStateBatch:
        def zeros(*shape):
            return torch.zeros((rows, *shape), dtype=torch.float32, device=device)

        def masks(*shape):
            return torch.ones((rows, *shape), dtype=torch.bool, device=device)

        return TorchStateBatch(
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            global_features=zeros(148),
            market_t1=zeros(4, 29), market_t1_mask=masks(4),
            market_t2=zeros(4, 29), market_t2_mask=masks(4),
            market_t3=zeros(4, 29), market_t3_mask=masks(4),
            self_reserved=zeros(3, 29), self_reserved_mask=masks(3),
            opponent_reserved=zeros(3, 29), opponent_reserved_mask=masks(3),
            nobles=zeros(3, 17), nobles_mask=masks(3),
            pending_card=zeros(29), pending_card_mask=masks(),
        )

    @staticmethod
    def _synthetic_decisions(rows: int, width: int, device: torch.device) -> TorchDecisionBatch:
        def zeros(*shape):
            return torch.zeros((rows, *shape), dtype=torch.float32, device=device)

        def masks(*shape):
            return torch.ones((rows, *shape), dtype=torch.bool, device=device)

        return TorchDecisionBatch(
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            global_features=zeros(148),
            market_t1=zeros(4, 29), market_t1_mask=masks(4),
            market_t2=zeros(4, 29), market_t2_mask=masks(4),
            market_t3=zeros(4, 29), market_t3_mask=masks(4),
            self_reserved=zeros(3, 29), self_reserved_mask=masks(3),
            opponent_reserved=zeros(3, 29), opponent_reserved_mask=masks(3),
            nobles=zeros(3, 17), nobles_mask=masks(3),
            pending_card=zeros(29), pending_card_mask=masks(),
            candidate_features=zeros(width, 82), candidate_mask=masks(width),
        )

    def warmup_native(self, widths: Sequence[int], *, rows: int = 2) -> float:
        """Compile the configured policy buckets and the value-only graph.

        Runs before the measurement window so cold compilation is accounted
        separately from the steady window; widths beyond the last bucket stay on
        the eager path and need no warm-up.
        """

        if self.compiled is None:
            return 0.0
        started = time.perf_counter()
        for size in self.config.batch_buckets or (rows,):
            # Static CUDA warm-up must use the same row-strided native views as
            # collection; contiguous synthetic inputs would compile other guards.
            if (self.config.batch_buckets and self.device.type == "cuda" and
                    self.config.native_transport == "reused"):
                from .native_policy import _default_catalog, _extension, _native_catalog
                catalog = _native_catalog(_default_catalog())
                for state_only, width in [(False, w) for w in widths] + [(True, 0)]:
                    probe = _extension().NativeBatch(catalog, state_only, 1, width or 1)
                    transfer = self.new_native_transfer(
                        state_only=state_only, capacity=size, width=width,
                        float_stride=probe.float_stride(), mask_stride=probe.mask_stride())
                    transfer.float_dev.zero_()
                    transfer.mask_dev.fill_(True)
                    batch = transfer._prepared(probe.packed(), size)
                    with torch.inference_mode():
                        for _ in range(3):  # compile, capture, then replay
                            if state_only:
                                self._forward_value(batch)
                            else:
                                self._forward(batch)
                continue
            for width in widths:
                batch = self._synthetic_decisions(size, width, self.device)
                with torch.inference_mode():
                    self._forward(batch)
            state = self._synthetic_state(size, self.device)
            with torch.inference_mode():
                self._forward_value(state)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.stats.compile = self.compiled.stats.to_dict()
        return time.perf_counter() - started

    def new_native_transfer(self, *, state_only: bool, capacity: int, width: int,
                            float_stride: int, mask_stride: int):
        """Reusable staging for one native batch shape on this evaluator's device."""

        if self.device.type == "cuda":
            if self.compiled is not None:
                capacity = self.compiled.batch_for(capacity)
            if self.config.native_transport == "packed":
                return NativeArrayPackedTransfer(self)
            return NativePackedTransfer(self, state_only=state_only, capacity=capacity, width=width,
                                        float_stride=float_stride, mask_stride=mask_stride)
        return NativeArrayTransfer(self)

    def _record_native(self, rows: int, *, h2d: float, forward: float, d2h: float) -> None:
        self.stats.batches += 1
        self.stats.positions += rows
        self.stats.batch_sizes[rows] += 1
        self.stats.h2d_seconds += h2d
        self.stats.forward_seconds += forward
        self.stats.d2h_seconds += d2h
        if self.compiled is not None:
            self.stats.compile = self.compiled.stats.to_dict()

    def _tensor_batch_native(
        self, arrays: Mapping[str, np.ndarray], *, state_only: bool
    ) -> TorchDecisionBatch | TorchStateBatch:
        """Wrap one native batch buffer without a per-leaf Python object."""

        names = _STATE_FIELDS if state_only else _POLICY_FIELDS
        cls = TorchStateBatch if state_only else TorchDecisionBatch
        if arrays["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("native batch feature schema mismatch")
        if self.device.type != "cuda":
            return cls(
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                **{name: torch.as_tensor(arrays[name]) for name in names},
            )
        packed = np.concatenate([
            np.ascontiguousarray(arrays[name]).reshape(-1).astype(np.float32, copy=False)
            for name in names
        ])
        shapes = {name: arrays[name].shape for name in names}
        lengths = {name: int(arrays[name].size) for name in names}
        flat = torch.from_numpy(packed).to(self.device)
        values: dict[str, torch.Tensor] = {}
        offset = 0
        for name in names:
            view = flat.narrow(0, offset, lengths[name]).view(shapes[name])
            values[name] = view if arrays[name].dtype != np.bool_ else view.bool()
            offset += lengths[name]
        return cls(feature_schema_version=FEATURE_SCHEMA_VERSION, **values)

    def policy_value_native(self, arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        started = time.perf_counter()
        prepared = self._tensor_batch_native(arrays, state_only=False)
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            logits, values = self._forward(prepared)
            logits_array = logits.detach().cpu().numpy()
            value_array = values.detach().cpu().numpy()
        self._record(int(value_array.shape[0]), started)
        if not np.all(np.isfinite(logits_array)) or not np.all(np.isfinite(value_array)):
            raise NativeBatchRowError(_first_nonfinite_row(logits_array, value_array),
                                      "nonfinite native batch policy/value output")
        return logits_array, value_array

    def value_only_native(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        started = time.perf_counter()
        prepared = self._tensor_batch_native(arrays, state_only=True)
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            values = self._forward_value(prepared).detach().cpu().numpy()
        self._record(int(values.shape[0]), started)
        if not np.all(np.isfinite(values)):
            raise NativeBatchRowError(int(np.flatnonzero(~np.isfinite(values))[0]),
                                      "nonfinite native batch value output")
        return values

    def _tensor_batch(self, batch: object, *, state_only: bool) -> TorchDecisionBatch | TorchStateBatch:
        if self.device.type == "cuda":
            return _packed_torch_batch(batch, self.device, state_only=state_only)
        return tensorize_state(batch, device=self.device) if state_only else tensorize_batch(batch, device=self.device)

    def _record(self, size: int, started: float) -> None:
        self.stats.batches += 1
        self.stats.positions += size
        self.stats.batch_sizes[size] += 1
        self.stats.forward_seconds += time.perf_counter() - started
        if self.compiled is not None:
            self.stats.compile = self.compiled.stats.to_dict()

    def _forward(self, prepared: TorchDecisionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compiled is not None:
            return self.compiled.policy_value(prepared)
        return self.model._forward_prepared(prepared, projection=self.projection)

    def _forward_policy(self, prepared: TorchDecisionBatch) -> torch.Tensor:
        if self.compiled is not None:
            return self.compiled.policy_only(prepared)
        return self.model._policy_logits_prepared(prepared, projection=self.projection)

    def _forward_value(self, prepared: TorchDecisionBatch | TorchStateBatch) -> torch.Tensor:
        if self.compiled is not None and isinstance(prepared, TorchStateBatch):
            return self.compiled.value_only(prepared)
        return self.model.critic(self.model.encode_state(prepared)).squeeze(-1)

    def policy_value(
        self, requests: Sequence[DecisionRequest], *, encoded: Sequence[EncodedDecision] | None = None,
    ) -> tuple[tuple[tuple[float, ...], float], ...]:
        if not requests:
            return ()
        started = time.perf_counter()
        values = tuple(encoded) if encoded is not None else tuple(map(encode_decision, requests))
        batch = collate_encoded_decisions(values)
        prepared = self._tensor_batch(batch, state_only=False)
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            logits, critics = self._forward(prepared)
            joined = torch.cat((logits, critics.unsqueeze(1)), dim=1).detach().cpu().numpy()
        self._record(len(requests), started)
        result = []
        for request, row in zip(requests, joined):
            logits_row = row[:len(request.legal_actions)]
            value = float(row[-1])
            if not np.all(np.isfinite(logits_row)) or not math.isfinite(value):
                raise ValueError("nonfinite search policy/value output")
            result.append((tuple(float(x) for x in logits_row), value))
        return tuple(result)

    def policy_value_encoded(self, encoded: Sequence[EncodedDecision]) -> tuple[tuple[tuple[float, ...], float], ...]:
        if not encoded:
            return ()
        started = time.perf_counter()
        batch = collate_encoded_decisions(encoded)
        prepared = self._tensor_batch(batch, state_only=False)
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            logits, critics = self._forward(prepared)
            joined = torch.cat((logits, critics.unsqueeze(1)), dim=1).detach().cpu().numpy()
        self._record(len(encoded), started)
        return tuple(
            (tuple(float(x) for x in row[:len(item.candidate_actions)]), float(row[-1]))
            for item, row in zip(encoded, joined)
        )

    def value_only_encoded(self, encoded: Sequence[EncodedState]) -> tuple[float, ...]:
        if not encoded:
            return ()
        started = time.perf_counter()
        batch = collate_encoded_states(encoded)
        prepared = self._tensor_batch(batch, state_only=True)
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            values = self._forward_value(prepared).detach().cpu().numpy()
        self._record(len(encoded), started)
        return tuple(float(x) for x in values)

    def policy_only(self, requests: Sequence[DecisionRequest]) -> tuple[tuple[float, ...], ...]:
        if not requests:
            return ()
        started = time.perf_counter()
        batch = collate_encoded_decisions(tuple(map(encode_decision, requests)))
        prepared = self._tensor_batch(batch, state_only=False)
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            rows = self._forward_policy(prepared).detach().cpu().numpy()
        self._record(len(requests), started)
        result = tuple(tuple(float(x) for x in row[:len(request.legal_actions)]) for request, row in zip(requests, rows))
        if any(not math.isfinite(x) for row in result for x in row):
            raise ValueError("nonfinite search policy output")
        return result

    def value_only(self, requests: Sequence[DecisionRequest]) -> tuple[float, ...]:
        if not requests:
            return ()
        started = time.perf_counter()
        batch = collate_encoded_states(tuple(map(encode_decision_state, requests)))
        prepared = self._tensor_batch(batch, state_only=True)
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            values = self._forward_value(prepared).detach().cpu().numpy()
        self._record(len(requests), started)
        result = tuple(float(x) for x in values)
        if any(not math.isfinite(x) for x in result):
            raise ValueError("nonfinite search value output")
        return result
