"""FP32 batched inference for several independent D6 or MCTS searches.

Each search keeps its own engines, caches and statistics; this runner only
groups the pending network evaluations of independent searches into shared
FP32 batches.  Search semantics (traversal order, pruning, tie-breaks, root
targets) are unchanged, so chosen actions can differ from the scalar batch=1
path only where the network output is within normal FP32 batch-shape noise.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .actions import Action
from .contracts import DecisionRequest
from .mcts_policy import MCTSSearchResult, PUCTSearchPolicy
from .search_inference import FrozenSearchInference
from .search_policy import PolicyValueSearchPolicy, PolicyValueSearchSession


@dataclass(slots=True)
class BatchRunStats:
    """Physical inference work performed by one batched search run."""

    policy_batches: int = 0
    value_batches: int = 0
    policy_positions: int = 0
    value_positions: int = 0
    deduplicated_positions: int = 0
    batch_sizes: Counter[int] = field(default_factory=Counter)
    forward_seconds: float = 0.0

    @property
    def physical_batches(self) -> int:
        return self.policy_batches + self.value_batches

    @property
    def physical_positions(self) -> int:
        return self.policy_positions + self.value_positions

    def to_dict(self) -> dict[str, object]:
        batches = self.physical_batches
        positions = self.physical_positions
        return {
            "policy_batches": self.policy_batches,
            "value_batches": self.value_batches,
            "physical_batches": batches,
            "policy_positions": self.policy_positions,
            "value_positions": self.value_positions,
            "physical_positions": positions,
            "deduplicated_positions": self.deduplicated_positions,
            "mean_batch_size": positions / batches if batches else 0.0,
            "batch_size_histogram": {
                str(size): count for size, count in sorted(self.batch_sizes.items())
            },
            "forward_seconds": self.forward_seconds,
        }


def _chunks(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


class BatchedSearchRunner:
    """Drive independent D6 or MCTS searches with shared batched FP32 inference."""

    def __init__(
        self,
        policy_or_evaluator: PolicyValueSearchPolicy | PUCTSearchPolicy | FrozenSearchInference,
        *,
        max_batch_size: int = 32,
        dedupe: bool = True,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if isinstance(policy_or_evaluator, FrozenSearchInference):
            evaluator = policy_or_evaluator
        else:
            evaluator = policy_or_evaluator.evaluator
        self.evaluator = evaluator
        self.max_batch_size = max_batch_size
        self.dedupe = dedupe
        self.stats = BatchRunStats()
        self.last_sessions: tuple[PolicyValueSearchSession, ...] = ()

    def _limit(self, max_batch_size: int | None) -> int:
        limit = self.max_batch_size if max_batch_size is None else max_batch_size
        if limit <= 0:
            raise ValueError("max_batch_size must be positive")
        return limit

    def _check_policy(self, policy) -> None:
        if policy.evaluator is not self.evaluator:
            raise ValueError("policy does not share the runner's evaluator")

    # -- D6 search ---------------------------------------------------------

    def choose_many(
        self,
        policy: PolicyValueSearchPolicy,
        requests: Sequence[DecisionRequest],
        *,
        max_batch_size: int | None = None,
        pool_size: int | None = None,
    ) -> list[Action]:
        """Choose one action per request, batching the searches together.

        ``pool_size`` limits how many searches run concurrently; extra requests
        wait in a queue so batch occupancy stays high while shorter searches
        finish (defaults to ``2 * max_batch_size``).
        """

        self._check_policy(policy)
        limit = self._limit(max_batch_size)
        pool = limit * 2 if pool_size is None else pool_size
        if pool <= 0:
            raise ValueError("pool_size must be positive")
        self.last_sessions = tuple(
            self._run_d6_pool(policy, requests, limit, pool)
        )
        return [session.chosen for session in self.last_sessions]

    def _run_d6_pool(
        self,
        policy: PolicyValueSearchPolicy,
        requests: Sequence[DecisionRequest],
        limit: int,
        pool: int,
    ) -> list[PolicyValueSearchSession]:
        queue = list(enumerate(requests))
        active: dict[int, PolicyValueSearchSession] = {}
        finished: dict[int, PolicyValueSearchSession] = {}

        def fill() -> None:
            while len(active) < pool and queue:
                index, request = queue.pop(0)
                active[index] = PolicyValueSearchSession(
                    policy.clone_for_session(), request
                )

        fill()
        while active:
            unresolved = []
            for session in active.values():
                if session.done:
                    continue
                if not session.lookup():
                    unresolved.append(session)
            if unresolved:
                policy_group = [s for s in unresolved if s.pending.kind == "policy"]
                value_group = [s for s in unresolved if s.pending.kind == "value"]
                for chunk in _chunks(policy_group, limit):
                    self._evaluate_d6_chunk(chunk, "policy")
                for chunk in _chunks(value_group, limit):
                    self._evaluate_d6_chunk(chunk, "value")
            for index in [index for index, session in active.items() if session.done]:
                finished[index] = active.pop(index)
            fill()
        return [finished[index] for index in range(len(requests))]

    def _evaluate_d6_chunk(
        self, sessions: Sequence[PolicyValueSearchSession], kind: str,
    ) -> None:
        keys: list[object] = []
        for index, session in enumerate(sessions):
            evaluation = session.pending
            if evaluation is None or evaluation.kind != kind:
                raise RuntimeError("search session is not waiting for this evaluation")
            key = (
                evaluation.request if kind == "policy"
                else evaluation.request.observation
            )
            keys.append(key if self.dedupe else (key, index))
        first_index: dict[object, int] = {}
        for index, key in enumerate(keys):
            first_index.setdefault(key, index)
        order = list(first_index.values())
        batch = tuple(sessions[index].pending.request for index in order)
        started = time.perf_counter()
        if kind == "policy":
            rows = self.evaluator.policy_only(batch)
        else:
            rows = self.evaluator.value_only(batch)
        elapsed = time.perf_counter() - started
        size = len(batch)
        if kind == "policy":
            self.stats.policy_batches += 1
            self.stats.policy_positions += size
        else:
            self.stats.value_batches += 1
            self.stats.value_positions += size
        self.stats.batch_sizes[size] += 1
        self.stats.forward_seconds += elapsed
        self.stats.deduplicated_positions += len(sessions) - size
        results = {keys[index]: rows[position] for position, index in enumerate(order)}
        for session, key in zip(sessions, keys):
            session.complete(results[key])

    # -- MCTS search -------------------------------------------------------

    def search_many(
        self,
        policy: PUCTSearchPolicy,
        requests: Sequence[DecisionRequest],
        *,
        add_root_noise: bool = False,
        temperature: float = 0.0,
        sampling_seed: int | None = None,
        max_batch_size: int | None = None,
    ) -> list[MCTSSearchResult]:
        """Run one MCTS job per request, batching their pending evaluations."""

        self._check_policy(policy)
        limit = self._limit(max_batch_size)
        jobs = [
            policy.start_job(
                request,
                add_root_noise=add_root_noise,
                temperature=temperature,
                sampling_seed=sampling_seed,
            )
            for request in requests
        ]
        while True:
            active = [job for job in jobs if not job.done]
            if not active:
                return [job.result for job in jobs]
            progressed = False
            for kind in ("policy_value", "value_only"):
                items = [
                    (job, item)
                    for job in active
                    for item in job.pending_requests()
                    if item.kind == kind
                ]
                for chunk in _chunks(items, limit):
                    self._evaluate_mcts_chunk(policy, kind, chunk)
                    progressed = True
            if not progressed:
                raise RuntimeError("MCTS jobs are blocked with no pending requests")

    def _evaluate_mcts_chunk(self, policy: PUCTSearchPolicy, kind: str, chunk) -> None:
        requests = tuple(item.request for _, item in chunk)
        encoded = tuple(item.encoded for _, item in chunk)
        encoded_arg = None if any(value is None for value in encoded) else encoded
        started = time.perf_counter()
        before_features = self.evaluator.stats.feature_seconds
        before_forward = self.evaluator.stats.forward_seconds
        if kind == "policy_value":
            rows = self.evaluator.policy_value(requests, encoded=encoded_arg)
            self.stats.policy_batches += 1
            self.stats.policy_positions += len(requests)
        else:
            rows = self.evaluator.value_only(requests)
            self.stats.value_batches += 1
            self.stats.value_positions += len(requests)
        elapsed = time.perf_counter() - started
        self.stats.batch_sizes[len(requests)] += 1
        self.stats.forward_seconds += elapsed
        policy.stats.feature_seconds += self.evaluator.stats.feature_seconds - before_features
        policy.stats.inference_seconds += self.evaluator.stats.forward_seconds - before_forward
        policy.stats.model_batches += 1
        for (job, item), row in zip(chunk, rows):
            if kind == "policy_value":
                logits, value = row
                job.submit(item.request_id, logits, value, prior_is_normalized=False)
            else:
                job.submit(item.request_id, None, row)
