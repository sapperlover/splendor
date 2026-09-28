"""Search memoization, rules fast path and N>1 engine-mode regressions."""

from __future__ import annotations

import random

import torch

from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.model import TokenVector
from splendor._state import PlayerState
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.rules import legal_payments_for_card, payment_exists
from splendor.search_policy import PolicyValueSearchPolicy, SearchConfig
from splendor.benchmark_search import fixed_requests


def _positions() -> tuple[DecisionRequest, ...]:
    return tuple(request for _, request, _ in fixed_requests())


def test_payment_exists_matches_full_enumeration() -> None:
    catalog = load_default_catalog()
    rng = random.Random(20240928)
    card_ids = [card.card_id for card in catalog.cards]
    for _ in range(2000):
        tokens = TokenVector(
            white=rng.randint(0, 5), blue=rng.randint(0, 5), green=rng.randint(0, 5),
            red=rng.randint(0, 5), black=rng.randint(0, 5), gold=rng.randint(0, 3),
        )
        purchased = tuple(rng.sample(card_ids, rng.randint(0, 8)))
        player = PlayerState(tokens=tokens, purchased_card_ids=purchased)
        card = rng.choice(catalog.cards)
        assert payment_exists(player, card, catalog) == bool(
            legal_payments_for_card(player, card, catalog)
        )


def test_memoized_search_matches_plain_search() -> None:
    model = load_frozen_actor_critic_checkpoint(
        "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt",
        device="cpu",
    )
    torch.set_num_threads(1)
    requests = _positions()
    for request in requests:
        plain = PolicyValueSearchPolicy(
            model,
            config=SearchConfig(depth=3, top_k=3, seed=17, fold_forced_phases=True,
                                memoize=False),
            device="cpu", engine_mode="fast",
        )
        memoized = PolicyValueSearchPolicy(
            model,
            config=SearchConfig(depth=3, top_k=3, seed=17, fold_forced_phases=True,
                                memoize=True),
            device="cpu", engine_mode="fast",
        )
        assert plain.choose(request) == memoized.choose(request)
        assert plain.last_root_values == memoized.last_root_values
        assert plain.last_root_bound_values == memoized.last_root_bound_values
        for name in ("expanded_nodes", "leaf_nodes", "model_positions", "model_batches",
                     "folded_decisions", "pruned_branches"):
            assert getattr(plain.stats, name) == getattr(memoized.stats, name), name
        assert memoized.stats.physical_model_positions <= memoized.stats.model_positions
        assert memoized.stats.cache_hits >= 0


def test_combined_determinizations_use_requested_engine_mode(monkeypatch) -> None:
    import splendor.search_policy as search_policy

    model = load_frozen_actor_critic_checkpoint(
        "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt",
        device="cpu",
    )
    torch.set_num_threads(1)
    modes: list[str] = []
    original = search_policy.determinize_request

    def spy(request, *, seed, catalog=None, failure_dir=None, engine_mode="checked"):
        modes.append(engine_mode)
        return original(request, seed=seed, catalog=catalog, failure_dir=failure_dir,
                        engine_mode=engine_mode)

    monkeypatch.setattr(search_policy, "determinize_request", spy)
    policy = PolicyValueSearchPolicy(
        model,
        config=SearchConfig(depth=2, top_k=2, determinizations=2, seed=3, memoize=False),
        device="cpu", engine_mode="fast",
    )
    policy.choose(_positions()[0])
    assert modes == ["fast", "fast"]
