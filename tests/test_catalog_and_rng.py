from __future__ import annotations

from collections import Counter

from splendor.catalog import Catalog
from splendor.engine import GameEngine
from splendor.replay import REPLAY_SCHEMA_VERSION, RULESET_ID, new_replay
from splendor.rng import DeterministicRng, RNG_ALGORITHM


def test_normative_catalog_loads_with_stable_ids(catalog: Catalog) -> None:
    assert len(catalog.cards) == 90
    assert len(catalog.nobles) == 12
    assert catalog.card("001").card_id == "001"
    assert len(catalog.data_fingerprint) == 64


def test_bga_noble_pool_contains_each_requested_tile_once(catalog: Catalog) -> None:
    requirements = Counter(tuple(n.requirement) for n in catalog.nobles)
    assert requirements[(0, 3, 3, 0, 3)] == 1
    assert requirements[(4, 0, 0, 4, 0)] == 1
    assert tuple(catalog.noble('N11').requirement) == (0, 3, 3, 0, 3)
    assert tuple(catalog.noble('N12').requirement) == (4, 0, 0, 4, 0)
    assert all(n.points == 3 for n in catalog.nobles)
    assert len(requirements) == 12


def test_new_nobles_are_in_the_default_setup_draw(catalog: Catalog) -> None:
    seen = set()
    for seed in range(64):
        state = GameEngine.new(seed, first_player=0, catalog=catalog)._debug_state()
        assert len(state.nobles_in_play) == 3
        assert len(state.nobles_out_of_game) == 9
        assert len(set(state.nobles_in_play + state.nobles_out_of_game)) == 12
        seen.update(state.nobles_in_play)
    assert {'N11', 'N12'} <= seen


def test_rng_snapshot_restores_exact_stream() -> None:
    rng = DeterministicRng(8675309)
    prefix = [rng.randbelow(17) for _ in range(13)]
    snapshot = rng.snapshot()
    expected_tail = [rng.next_u64() for _ in range(20)]
    restored = DeterministicRng.restore(snapshot)

    # This golden vector detects accidental changes to the versioned algorithm.
    assert prefix == [16, 4, 1, 13, 11, 3, 13, 6, 0, 11, 12, 14, 10]
    assert [restored.next_u64() for _ in range(20)] == expected_tail


def test_shuffle_is_deterministic() -> None:
    left = list(range(50))
    right = list(range(50))
    DeterministicRng(42).shuffle(left)
    DeterministicRng(42).shuffle(right)
    assert left == right
    assert left != list(range(50))


def test_replay_header_pins_rng_algorithm(catalog: Catalog) -> None:
    replay = new_replay(catalog, game_seed=42, first_player=1)
    assert replay.header.schema_version == 2
    assert replay.header.schema_version == REPLAY_SCHEMA_VERSION
    assert replay.header.rng_algorithm == RNG_ALGORITHM
    assert replay.header.ruleset_id == "splendor-base-two-player-v2-forced-skip"
    assert replay.header.ruleset_id == RULESET_ID
