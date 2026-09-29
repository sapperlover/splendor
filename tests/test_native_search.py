"""Small differential checks at the native migration boundaries."""

import random

import numpy as np
import pytest

from splendor.actor_critic import ActorCritic
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import encode_decision, encode_decision_state
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy
from splendor.search_policy import _request_public_word

native = pytest.importorskip("splendor._native_search")


def _same_features(actual, expected):
    for name in expected.__dataclass_fields__:
        if name in ("feature_schema_version", "candidate_actions"):
            continue
        observed = actual[name]
        reference = getattr(expected, name)
        assert observed.shape == reference.shape, name
        np.testing.assert_allclose(observed, reference, rtol=0, atol=1e-6, err_msg=name)


def _state_snapshot(state, catalog):
    card = {item.card_id: i for i, item in enumerate(catalog.cards)}
    noble = {item.noble_id: i for i, item in enumerate(catalog.nobles)}
    player_rows = []
    for player in state.players:
        player_rows.append({
            "tokens": list(player.tokens),
            "purchased": [card[x] for x in player.purchased_card_ids],
            "nobles": [noble[x] for x in player.noble_ids],
            "reserved": [(card[x.card_id], 0 if x.source.value == "market" else 1)
                         for x in player.reserved_cards],
        })
    return {
        "phase": ("main_action", "payment", "return_tokens", "choose_noble", "terminal").index(state.phase.value),
        "actor": state.current_player,
        "first": state.first_player,
        "supply": list(state.supply),
        "turns": tuple(state.completed_turns),
        "skips": state.consecutive_forced_skips,
        "trigger": -1 if state.final_round_triggered_by is None else state.final_round_triggered_by,
        "winner": -2 if state.result is None else (-1 if len(state.result.winner_ids) == 2 else state.result.winner_ids[0]),
        "decks": [[card[x] for x in getattr(state.decks, field)] for field in ("tier_one", "tier_two", "tier_three")],
        "market": [[card[x] for x in getattr(state.market, field)] for field in ("tier_one", "tier_two", "tier_three")],
        "nobles": [noble[x] for x in state.nobles_in_play],
        "players": player_rows,
        "return_count": 0 if state.pending is None else state.pending.tokens_to_return,
        "eligible": [] if state.pending is None else [noble[x] for x in state.pending.eligible_noble_ids],
    }


@pytest.mark.parametrize("seed", [1, 42, 99])
def test_native_complete_trajectory_matches_python(seed):
    catalog = load_default_catalog()
    engine = GameEngine.new(seed, first_player=seed % 2, catalog=catalog, mode="fast")
    game = native.NativeGame(native.Catalog(catalog), engine._debug_state())
    rng = random.Random(seed ^ 0xBEEF)
    for _ in range(400):
        request = engine.decision()
        actual = dict(game.snapshot())
        expected = _state_snapshot(engine._debug_state(), catalog)
        for key, value in expected.items():
            assert actual[key] == value, (seed, key)
        if not isinstance(request, DecisionRequest):
            assert not game.actions()
            break
        assert game.verify_actions(request)
        _same_features(game.encoded(), encode_decision(request))
        index = rng.randrange(len(request.legal_actions))
        game.apply_index(index)
        engine.apply(request.legal_actions[index])
    else:
        pytest.fail("fixed game did not terminate")


@pytest.mark.parametrize("worlds", [1, 2])
def test_native_search_matches_controlled_python_worlds(worlds):
    request = GameEngine.new(42, first_player=0, mode="fast").decision()
    catalog = load_default_catalog()
    config = MCTSConfig(simulations=17, determinizations=worlds, seed=155, root_noise_fraction=0)
    native_job = native.SearchJob(native.Catalog(catalog), request, config,
                                  _request_public_word(request), 155, False, 0.0)
    python_job = PUCTSearchPolicy(ActorCritic(), config=config).start_job(request)
    while not native_job.done():
        incoming = native_job.pending()
        reference = python_job.pending_requests()
        assert len(incoming) == len(reference)
        for item, other in zip(incoming, reference):
            assert item["kind"] == other.kind
            encoded = encode_decision_state(other.request) if other.kind == "value_only" else encode_decision(other.request)
            _same_features(item["encoded"], encoded)
        for item, other in zip(incoming, reference):
            logits = None if other.kind == "value_only" else (0.0,) * len(other.request.legal_actions)
            native_job.submit(item["world"], () if logits is None else logits, 0.0)
            if item["world"] == -1:
                native_job.initialize_worlds(request)
            python_job.submit(other.request_id, logits, 0.0)
    actual = native_job.result()
    expected = python_job.result
    assert actual["simulation_budgets"] == list(expected.simulation_budgets)
    assert actual["per_world_root_visit_counts"] == [list(row) for row in expected.per_world_root_visit_counts]
    assert actual["chosen_candidate_index"] == expected.chosen_candidate_index


def test_native_failure_branch_replays_after_bad_network_value():
    from splendor.compact_trace import new_trace
    from splendor.native_policy import NativePUCTSearchPolicy
    from splendor.replay import to_primitive
    from splendor.reproduce_failure import reproduce

    catalog = load_default_catalog()
    request = GameEngine.new(42, first_player=0, catalog=catalog, mode="fast").decision()
    policy = NativePUCTSearchPolicy(
        ActorCritic(), config=MCTSConfig(simulations=4, determinizations=1,
                                        seed=155, root_noise_fraction=0), catalog=catalog,
    )
    job = policy.start_job(request)
    root = job.pending_requests()[0]
    job.submit(root.request_id, (0.0,) * len(request.legal_actions), 0.0)
    leaf = job.pending_requests()[0]
    malformed = (float("nan"),) + (0.0,) * (len(leaf.encoded.candidate_actions) - 1)
    with pytest.raises(ValueError, match="nonfinite"):
        job.submit(leaf.request_id, malformed, 0.0)
    assert job.pending_requests()[0].request_id == leaf.request_id
    with pytest.raises(ValueError, match="nonfinite"):
        job.submit(leaf.request_id, (0.0,) * len(leaf.encoded.candidate_actions), float("nan"))
    scene = {
        "trace": new_trace(0, 42, 0, catalog),
        "root_request": to_primitive(request),
        "search_context": to_primitive(job.failure_context(leaf.request_id)),
        "batch_requests": [],
        "error_type": "ValueError",
    }
    rebuilt = reproduce(scene)
    assert rebuilt["world_index"] == 0
    assert rebuilt["branch_actions_replayed"] >= 1


def test_native_negative_seed_matches_supported_seed_domain():
    from splendor.native_policy import NativePUCTSearchPolicy

    request = GameEngine.new(42, first_player=0, mode="fast").decision()
    policy = NativePUCTSearchPolicy(
        ActorCritic(), config=MCTSConfig(simulations=2, determinizations=1,
                                        seed=-1, root_noise_fraction=0),
    )
    job = policy.start_job(request, sampling_seed=-2)
    assert job.pending_requests()[0].model_version == "in-memory"
    while not job.done:
        for pending in job.pending_requests():
            logits = None if pending.kind == "value_only" else (0.0,) * len(pending.encoded.candidate_actions)
            job.submit(pending.request_id, logits, 0.0)
    assert job.result.simulations == 2


@pytest.mark.parametrize("seed", [42, 99])
def test_native_real_game_request_and_terminal_match_python(seed):
    from splendor.compact_trace import replay_trace
    from splendor.match_stepper import MatchStepper
    from splendor.native_match_stepper import NativeMatchStepper

    python = MatchStepper(0, seed, seed % 2, mode="fast")
    native_game = NativeMatchStepper(0, seed, seed % 2, mode="fast")
    rng = random.Random(seed ^ 0xCAFE)
    for _ in range(400):
        assert native_game.current == python.current
        if native_game.terminal is not None:
            assert replay_trace(native_game.trace) == native_game.terminal
            break
        index = rng.randrange(len(native_game.request.legal_actions))
        native_game.apply(index)
        python.apply(index)
    else:
        pytest.fail("native real game did not terminate")
