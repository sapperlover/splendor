"""Bit-exactness guard for the vectorized feature encoder.

``tests/fixtures/feature_encoding_golden.json`` was captured from the scalar
encoder *before* the vectorization rewrite.  Every case replays the same
deterministic game and re-hashes the full encoded decision, so any drift in the
float values (not merely in shape or semantics) fails the suite.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.catalog import load_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_VERSION,
    collate_encoded_decisions,
    collate_encoded_states,
    encode_decision,
    encode_decision_state,
)
from splendor.heuristic_policy import DevelopmentPolicy, ScorePolicy
from splendor.model import DecisionPhase
from splendor.random_policy import RandomPolicy
from splendor.result import TerminalResult


GOLDEN_PATH = Path(__file__).parent / "fixtures" / "feature_encoding_golden.json"
ENCODED_FIELDS = (
    "global_features",
    "market_t1",
    "market_t1_mask",
    "market_t2",
    "market_t2_mask",
    "market_t3",
    "market_t3_mask",
    "self_reserved",
    "self_reserved_mask",
    "opponent_reserved",
    "opponent_reserved_mask",
    "nobles",
    "nobles_mask",
    "pending_card",
    "pending_card_mask",
    "candidate_features",
    "candidate_mask",
)
STATE_FIELDS = ENCODED_FIELDS[:15]

TARGETS = {
    "main_action": 60,
    "payment": 40,
    "return_tokens": 20,
    "choose_noble": 12,
}


def _digest_encoded(encoded) -> str:
    hasher = hashlib.sha256()
    for name in ENCODED_FIELDS:
        array = np.ascontiguousarray(getattr(encoded, name))
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(struct.pack("<I", array.ndim))
        for size in array.shape:
            hasher.update(struct.pack("<I", size))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_catalog():
    # The scalar golden predates the 12-noble rule correction. Preserve its
    # exact original setup instead of replacing the captured feature hashes.
    return load_catalog(Path(__file__).resolve().parents[1] /
                        'artifacts/noble_pool_bga12_20261021/previous_catalog')


def test_golden_fixture_matches_current_schema(golden: dict) -> None:
    assert golden["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert len(golden["cases"]) >= 100
    phases = {case["phase"] for case in golden["cases"]}
    assert {"main_action", "payment", "return_tokens", "choose_noble"} <= phases
    assert {case["pending"] for case in golden["cases"]} == {True, False}


def test_vectorized_encoder_is_bit_identical_to_the_scalar_golden(golden: dict, golden_catalog) -> None:
    expected = {(case["seed"], case["step"]): case for case in golden["cases"]}
    counts = {phase: 0 for phase in TARGETS}
    checked = 0
    mismatches: list[str] = []

    def run(seed: int, policies) -> bool:
        nonlocal checked
        engine = GameEngine.new(seed, first_player=seed % 2, catalog=golden_catalog)
        step = 0
        while step <= 200:
            response = engine.decision()
            if isinstance(response, TerminalResult):
                return False
            phase = response.observation.phase.value
            if counts[phase] < TARGETS[phase]:
                case = expected[(seed, step)]
                assert case["phase"] == phase
                assert case["pending"] == (response.observation.pending is not None)
                assert case["candidates"] == len(response.legal_actions)
                digest = _digest_encoded(encode_decision(response))
                if digest != case["digest"]:
                    mismatches.append(f"seed={seed} step={step} phase={phase}")
                counts[phase] += 1
                checked += 1
            if all(counts[phase] >= TARGETS[phase] for phase in TARGETS):
                return True
            engine.apply(policies[engine.current_player].choose(response))
            step += 1
        return False

    for seed in range(100, 400):
        if run(seed, (ScorePolicy(), DevelopmentPolicy())):
            break
    for seed in range(0, 120):
        if run(seed, (RandomPolicy(seed * 31 + 1), RandomPolicy(seed * 31 + 2))):
            break

    assert not mismatches, f"vectorized encoder drifted from golden: {mismatches[:5]}"
    assert checked == len(golden["cases"])


def test_golden_samples_match_elementwise(golden: dict, golden_catalog) -> None:
    by_key = {(case["seed"], case["step"]) for case in golden["cases"]}
    wanted = {(sample["seed"], sample["step"]) for sample in golden["samples"]}
    checked = 0

    def walk(seed: int, policies) -> None:
        nonlocal checked
        engine = GameEngine.new(seed, first_player=seed % 2, catalog=golden_catalog)
        step = 0
        while step <= 200 and checked < len(wanted):
            response = engine.decision()
            if isinstance(response, TerminalResult):
                return
            if (seed, step) in by_key and (seed, step) in wanted:
                sample = next(
                    item
                    for item in golden["samples"]
                    if item["seed"] == seed and item["step"] == step
                )
                encoded = encode_decision(response)
                assert np.array_equal(
                    np.asarray(encoded.global_features, dtype=np.float32),
                    np.asarray(sample["global_features"], dtype=np.float32),
                )
                assert np.array_equal(
                    np.asarray(encoded.candidate_features, dtype=np.float32)[
                        :2
                    ].ravel(),
                    np.asarray(sample["candidate_features"], dtype=np.float32),
                )
                assert np.array_equal(
                    np.asarray(encoded.pending_card, dtype=np.float32),
                    np.asarray(sample["pending_card"], dtype=np.float32),
                )
                checked += 1
            engine.apply(policies[engine.current_player].choose(response))
            step += 1

    for seed in range(100, 400):
        walk(seed, (ScorePolicy(), DevelopmentPolicy()))
        if checked >= len(wanted):
            break
    assert checked == len(wanted)


def _random_requests(seed: int, limit: int) -> list[DecisionRequest]:
    engine = GameEngine.new(seed, first_player=seed % 2)
    policies = (ScorePolicy(), DevelopmentPolicy())
    requests: list[DecisionRequest] = []
    for _ in range(limit):
        response = engine.decision()
        if isinstance(response, TerminalResult):
            break
        requests.append(response)
        engine.apply(policies[engine.current_player].choose(response))
    return requests


def test_state_only_encoding_is_the_state_part_of_encode_decision() -> None:
    requests = [
        request for seed in range(4) for request in _random_requests(seed, 40)
    ]
    assert len(requests) >= 40
    for request in requests:
        decision = encode_decision(request)
        state = encode_decision_state(request)
        assert state.feature_schema_version == FEATURE_SCHEMA.version
        for name in STATE_FIELDS:
            assert np.array_equal(
                np.asarray(getattr(state, name)),
                np.asarray(getattr(decision, name)),
            ), name


def test_state_only_encoding_keeps_pending_turn_context() -> None:
    payment_requests = [
        request
        for seed in range(8)
        for request in _random_requests(seed, 60)
        if request.observation.phase is DecisionPhase.PAYMENT
    ]
    assert payment_requests, "no payment decision was generated"
    for request in payment_requests:
        state = encode_decision_state(request)
        assert bool(state.pending_card_mask)
        assert state.pending_card[0] == 1.0
        assert state.pending_card[1] == 1.0
        # The pending block at the end of the global vector must stay intact.
        global_names = list(FEATURE_SCHEMA.global_feature_names)
        assert global_names[-9] == "pending_present"
        assert state.global_features[-9] == 1.0


def test_value_from_state_is_bit_identical_to_value_only() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = ActorCritic().eval()
    requests = [
        request for seed in range(3) for request in _random_requests(seed, 25)
    ]
    states = collate_encoded_states(
        tuple(encode_decision_state(request) for request in requests)
    )
    decisions = collate_encoded_decisions(
        tuple(encode_decision(request) for request in requests)
    )
    with torch.inference_mode():
        from_state = model.value_from_state(states)
        from_decisions = model.value_only(decisions)
    assert torch.equal(from_state, from_decisions)
