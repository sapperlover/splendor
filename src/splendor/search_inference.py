"""Frozen FP32 search inference with shared scalar and batch paths."""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .actor_critic import ActorCritic, TorchDecisionBatch, TorchStateBatch, tensorize_batch, tensorize_state
from .contracts import DecisionRequest
from .features import (
    EncodedDecision, collate_encoded_decisions, collate_encoded_states,
    encode_decision, encode_decision_state,
)


@dataclass(slots=True)
class InferenceStats:
    batches: int = 0
    positions: int = 0
    feature_seconds: float = 0.0
    forward_seconds: float = 0.0
    batch_sizes: Counter[int] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        return {
            "physical_model_batches": self.batches,
            "logical_model_positions": self.positions,
            "feature_seconds": self.feature_seconds,
            "forward_seconds": self.forward_seconds,
            "batch_size_histogram": {str(k): v for k, v in sorted(self.batch_sizes.items())},
            "mean_batch_size": self.positions / self.batches if self.batches else 0.0,
        }


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


class FrozenSearchInference:
    """Own an eval-mode model; learner-owned training models are copied once."""

    def __init__(self, model: ActorCritic, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA search inference requested but CUDA is unavailable")
        self.model = (copy.deepcopy(model) if model.training else model).to(self.device).eval()
        self.stats = InferenceStats()

    def _tensor_batch(self, batch: object, *, state_only: bool) -> TorchDecisionBatch | TorchStateBatch:
        if self.device.type == "cuda":
            return _packed_torch_batch(batch, self.device, state_only=state_only)
        return tensorize_state(batch, device=self.device) if state_only else tensorize_batch(batch, device=self.device)

    def _record(self, size: int, started: float) -> None:
        self.stats.batches += 1
        self.stats.positions += size
        self.stats.batch_sizes[size] += 1
        self.stats.forward_seconds += time.perf_counter() - started

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
            logits, critics = self.model._forward_prepared(prepared)
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
            logits, critics = self.model._forward_prepared(prepared)
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
            values = self.model.critic(self.model.encode_state(prepared)).squeeze(-1).detach().cpu().numpy()
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
            rows = self.model._policy_logits_prepared(prepared).detach().cpu().numpy()
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
            values = self.model.critic(self.model.encode_state(prepared)).squeeze(-1).detach().cpu().numpy()
        self._record(len(requests), started)
        result = tuple(float(x) for x in values)
        if any(not math.isfinite(x) for x in result):
            raise ValueError("nonfinite search value output")
        return result
