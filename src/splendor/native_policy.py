"""Python control and model adapter for the C++ independent-world PUCT engine."""

from __future__ import annotations

import time
from functools import cache
from dataclasses import dataclass
from itertools import count
from typing import Any

from .catalog import Catalog, load_default_catalog
from .contracts import DecisionRequest
from .features import EncodedDecision, EncodedState
from .mcts_policy import MCTSConfig, MCTSSearchResult, MCTSStats
from .search_policy import _request_public_word, _seed_from_public_word


def _extension():
    try:
        from . import _native_search
    except ImportError as error:
        raise RuntimeError(
            "Native search extension is unavailable. Install it with "
            "`conda run -n splendor python -m pip install -e . --no-deps`."
        ) from error
    return _native_search


@cache
def _default_catalog() -> Catalog:
    return load_default_catalog()


_CATALOGS: dict[str, object] = {}

def _native_catalog(catalog: Catalog):
    try:
        return _CATALOGS[catalog.data_fingerprint]
    except KeyError:
        native = _extension().Catalog(catalog)
        _CATALOGS[catalog.data_fingerprint] = native
        return native


@dataclass(frozen=True, slots=True)
class NativePendingEvaluation:
    request_id: str
    kind: str
    encoded: EncodedDecision | EncodedState
    world_index: int
    model_version: str
    request: None = None


_JOB_IDS = count()


class NativeSearchJob:
    def __init__(
        self, policy: NativePUCTSearchPolicy, request: DecisionRequest, *,
        add_root_noise: bool, temperature: float, sampling_seed: int | None,
        root_encoded: EncodedDecision | None = None,
    ) -> None:
        self.policy = policy
        self.request = request
        self.started = time.perf_counter()
        self.job_id = next(_JOB_IDS)
        self.sequence = 0
        self.sampling_seed = policy.config.seed if sampling_seed is None else sampling_seed
        self.public_word = _request_public_word(request) if len(request.legal_actions) > 1 else 0
        self._core = _extension().SearchJob(
            policy.native_catalog, request, policy.config, self.public_word,
            self.sampling_seed & ((1 << 64) - 1), add_root_noise, temperature,
        )
        self._pending: dict[str, NativePendingEvaluation] | None = None
        self._pending_dirty = True
        self._finished = False
        policy.stats.decisions += 1
        if len(request.legal_actions) == 1:
            policy.stats.single_candidate_decisions += 1
            self._collect_result()
        else:
            policy.stats.searchable_decisions += 1

    @property
    def done(self) -> bool:
        return self._core.done()

    @property
    def result(self) -> MCTSSearchResult:
        if not self.done:
            raise RuntimeError("search job has not finished")
        return self._result

    def pending_requests(self) -> tuple[NativePendingEvaluation, ...]:
        if self._pending_dirty:
            by_world = {} if self._pending is None else {
                item.world_index: item for item in self._pending.values()
            }
            pending = {}
            for item in self._core.pending():
                world = item["world"]
                if world in by_world:
                    evaluation = by_world[world]
                else:
                    cls = EncodedState if item["kind"] == "value_only" else EncodedDecision
                    encoded = cls(**item["encoded"])
                    request_id = f"native:{self.job_id}:{self.sequence}"
                    self.sequence += 1
                    evaluation = NativePendingEvaluation(
                        request_id, item["kind"], encoded, world, self.policy.model_version,
                    )
                pending[evaluation.request_id] = evaluation
            self._pending = pending
            self._pending_dirty = False
        return tuple(self._pending.values())

    def submit(self, request_id: str, logits: tuple[float, ...] | None,
               value: float, *, prior_is_normalized: bool = False) -> None:
        if self._pending is None or request_id not in self._pending:
            raise ValueError(f"unknown native search request {request_id}")
        if prior_is_normalized:
            raise ValueError("native search expects raw logits")
        world = self._pending[request_id].world_index
        self._core.submit(world, () if logits is None else logits, value)
        if world == -1:
            self._core.initialize_worlds(self.request)
        del self._pending[request_id]
        self._pending_dirty = True
        if self.done:
            self._collect_result()

    def _collect_result(self) -> None:
        if self._finished:
            return
        result = self._core.result()
        self._result = MCTSSearchResult(**result)
        self.policy.last_search_result = self._result
        stats = self._core.stats()
        for name, value in stats.items():
            if name == "native_compute_seconds":
                self.policy.native_compute_seconds += value
            else:
                setattr(self.policy.stats, name, getattr(self.policy.stats, name) + value)
        self.policy.stats.total_seconds += time.perf_counter() - self.started
        if result["searchable"]:
            raw = max(range(len(result["root_network_prior"])),
                      key=lambda i: (result["root_network_prior"][i], -i))
            self.policy.stats.changed_action_count += raw != result["chosen_candidate_index"]
        self._finished = True

    def cancel(self) -> None:
        self._pending = {}

    def _action_from_native(self, raw):
        from .actions import (TakeDistinct, TakePair, ReserveMarket, ReserveDeck,
                              BuyMarket, BuyReserved, Pay, ReturnTokens, ClaimNoble)
        from .model import Color, Tier, TokenVector
        kind, card_or_noble, tier, tokens = raw
        colors = tuple(Color(name) for name in ("white", "blue", "green", "red", "black"))
        if kind == 0:
            return TakeDistinct(tuple(colors[i] for i in range(5) if tokens[i]))
        if kind == 1:
            return TakePair(colors[next(i for i in range(5) if tokens[i])])
        if kind == 2:
            return ReserveMarket(self.policy.catalog.cards[card_or_noble].card_id)
        if kind == 3:
            return ReserveDeck(Tier(tier + 1))
        if kind == 4:
            return BuyMarket(self.policy.catalog.cards[card_or_noble].card_id)
        if kind == 5:
            return BuyReserved(self.policy.catalog.cards[card_or_noble].card_id)
        if kind == 6:
            return Pay(TokenVector(*tokens))
        if kind == 7:
            return ReturnTokens(TokenVector(*tokens))
        return ClaimNoble(self.policy.catalog.nobles[card_or_noble].noble_id)

    def failure_context(self, request_id: str | None = None) -> dict[str, Any]:
        world = (-1 if request_id is None or self._pending is None or request_id not in self._pending
                 else self._pending[request_id].world_index)
        native = self._core.failure_context(world)
        return {
            "algorithm": "determinized-puct-native-v1",
            "search_protocol": "determinized-puct-native-v1",
            "root_request": self.request,
            "sampling_seed": self.sampling_seed,
            "world_index": native["world_index"],
            "determinization_seed": native["determinization_seed"],
            "simulation_index": native["simulation_index"],
            "branch_actions": [self._action_from_native(action) for action in native["branch_actions"]],
            "attempted_action": (None if native["attempted_action"] is None else
                                 self._action_from_native(native["attempted_action"])),
            "request_id": request_id,
        }


class NativePUCTSearchPolicy:
    def __init__(self, model, *, config: MCTSConfig | None = None,
                 device: str = "cpu", catalog: Catalog | None = None,
                 evaluator=None, model_version: str = "in-memory") -> None:
        from .search_inference import FrozenSearchInference
        self.config = config or MCTSConfig()
        self.model_version = model_version
        self.catalog = catalog or _default_catalog()
        self.native_catalog = _native_catalog(self.catalog)
        self.evaluator = evaluator or FrozenSearchInference(model, device=device)
        self.model = self.evaluator.model
        self.device = self.evaluator.device
        self.stats = MCTSStats()
        self.native_compute_seconds = 0.0
        self.last_search_result = None

    def start_job(self, request: DecisionRequest, *, add_root_noise: bool = False,
                  temperature: float = 0.0, sampling_seed: int | None = None,
                  root_encoded: EncodedDecision | None = None) -> NativeSearchJob:
        return NativeSearchJob(self, request, add_root_noise=add_root_noise,
                               temperature=temperature, sampling_seed=sampling_seed,
                               root_encoded=root_encoded)

    def search(self, request: DecisionRequest, *, add_root_noise: bool = False,
               temperature: float = 0.0, sampling_seed: int | None = None) -> MCTSSearchResult:
        job = self.start_job(request, add_root_noise=add_root_noise,
                             temperature=temperature, sampling_seed=sampling_seed)
        while not job.done:
            for item in job.pending_requests():
                if item.kind == "policy_value":
                    logits, value = self.evaluator.policy_value_encoded((item.encoded,))[0]
                    job.submit(item.request_id, logits, value)
                else:
                    value = self.evaluator.value_only_encoded((item.encoded,))[0]
                    job.submit(item.request_id, None, value)
        return job.result

    def choose(self, request: DecisionRequest):
        return request.legal_actions[self.search(request).chosen_candidate_index]
