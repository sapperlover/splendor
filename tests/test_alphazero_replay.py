from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from splendor import alphazero_replay
from splendor.actor_critic import ActorCritic
from splendor.alphazero_replay import (
    ReplayRun, build_replay_index, load_index_state,
)
from splendor.alphazero_train import LearnerConfig, draw_training_batch
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.mcts_policy import MCTSConfig
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID
from splendor.rng import DeterministicRng
from splendor.selfplay import derive_seed
from splendor.selfplay_data import game_path
from splendor.selfplay_fast import (
    ExecutionConfig, FastSelfPlayConfig, generate as generate_selfplay,
)

CACHE_BYTES = 64 * 1024 * 1024


def _weights_checkpoint(path: Path) -> Path:
    torch.manual_seed(23)
    model = ActorCritic()
    torch.save(
        {
            "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "metadata": {"rules_version": RULESET_ID},
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _mixed_split_seed(games: int, fraction: float = 0.5) -> int:
    for seed in range(2000):
        splits = {
            "validation" if derive_seed(seed, 0, game_id, "split") / (1 << 64) < fraction else "train"
            for game_id in range(games)
        }
        if splits == {"train", "validation"}:
            return seed
    raise AssertionError("no mixed split seed found")


def _selfplay_run(
    root: Path, checkpoint: Path, *, round_index: int, games: int = 3, split_seed: int | None = None
) -> ReplayRun:
    config = FastSelfPlayConfig(
        protocol="mcts-selfplay-v2",
        run_id=f"az-replay-round-{round_index:03d}",
        round_index=round_index,
        checkpoint=str(checkpoint),
        num_games=games,
        seed=31001 + round_index,
        search_config=MCTSConfig(simulations=4, determinizations=1),
        execution=ExecutionConfig(
            backend="batched_cpu", engine_mode="fast", workers=1,
            active_games_per_worker=1, max_batch_size=1,
        ),
        validation_fraction=0.5,
        split_seed=split_seed if split_seed is not None else _mixed_split_seed(games),
    )
    generate_selfplay(config, root)
    return ReplayRun(round_index, config.run_id, root)


@pytest.fixture(scope="module")
def replay_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("az-replay")
    checkpoint = _weights_checkpoint(root / "base.pt")
    previous_threads = torch.get_num_threads()
    try:
        first = _selfplay_run(root / "run-000", checkpoint, round_index=0)
        second = _selfplay_run(root / "run-001", checkpoint, round_index=1)
    finally:
        torch.set_num_threads(previous_threads)
    return checkpoint, first, second


def test_run_identity_and_split_isolation(replay_runs):
    _, first, second = replay_runs
    index = build_replay_index([first, second], max_games=16, cache_bytes=CACHE_BYTES)
    keys = [game.key for game in index.games]
    assert len(keys) == 6 and keys == sorted(keys)
    # game_id 0..2 appears in both runs; identity keeps them apart.
    assert len({key for key in keys}) == 6
    assert {game.game_id for game in index.games} == {0, 1, 2}
    train = index.split_games("train")
    validation = index.split_games("validation")
    assert len(train) > 0 and len(validation) > 0
    assert len(train) + len(validation) == 6
    train_pool = index.position_pool(split="train", round_index=0)
    expected = sum(game.decision_count for game in train if game.round_index == 0)
    assert len(train_pool) == expected
    train_keys = {game.key for game in train}
    rng = DeterministicRng(17)
    for _ in range(25):
        sample, reference = train_pool.sample(rng)
        assert sample.metadata["split"] == "train"
        assert (reference.round_index, reference.game_id) in train_keys
        assert sample.arrays["player_id"][sample.index] in (0, 1)


def test_eviction_only_drops_the_active_index(replay_runs):
    _, first, second = replay_runs
    archived = [
        path
        for run in (first, second)
        for path in sorted((run.root / "games").glob("game-*.npz"))
    ]
    assert len(archived) == 6
    index = build_replay_index([first, second], max_games=2, cache_bytes=CACHE_BYTES)
    assert index.total_games == 2 and index.evicted_games == 4
    assert all(game.round_index == 1 for game in index.games)
    assert all(path.is_file() for path in archived)


def test_current_history_sampling_contract(replay_runs):
    _, first, second = replay_runs
    index = build_replay_index([first, second], max_games=16, cache_bytes=CACHE_BYTES)
    current = index.position_pool(split="train", round_index=1)
    history = index.position_pool(split="train", before_round=1)
    assert len(current) > 0 and len(history) > 0
    config = LearnerConfig(device="cpu", batch_size=8, history_batch_size=4)
    rng = DeterministicRng(5)
    drawn = draw_training_batch(current, history, rng, config)
    assert drawn.current_positions == 4 and drawn.history_positions == 4
    assert all(reference.round_index == 1 for reference in drawn.references[:4])
    assert all(reference.round_index == 0 for reference in drawn.references[4:])
    no_history = draw_training_batch(current, None, rng, config)
    assert no_history.current_positions == 8 and no_history.history_positions == 0
    empty_history = index.position_pool(split="train", before_round=0)
    assert len(empty_history) == 0
    sized = draw_training_batch(current, empty_history, rng, config)
    assert sized.current_positions == 8 and sized.history_positions == 0
    # Uniform-with-replacement over positions reaches several distinct ones.
    seen = {
        (
            reference.round_index,
            reference.game_id,
            reference.decision_index,
        )
        for _ in range(200)
        for _, reference in [current.sample(rng)]
    }
    assert len(seen) > 5


def test_index_state_avoids_reverifying_unchanged_archives(replay_runs, tmp_path, monkeypatch):
    _, first, second = replay_runs
    state_path = tmp_path / "replay_index.json"
    calls: list[int] = []
    original = alphazero_replay.validate_game_arrays

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(alphazero_replay, "validate_game_arrays", counting)
    index = build_replay_index([first, second], max_games=16, cache_bytes=CACHE_BYTES, state_path=state_path)
    assert len(calls) == 6
    state = load_index_state(state_path)
    assert set(state) == {game.key for game in index.games}
    build_replay_index([first, second], max_games=16, cache_bytes=CACHE_BYTES, state_path=state_path)
    assert len(calls) == 6
    touched = second.root / "games" / "game-000000.npz"
    stat = touched.stat()
    os.utime(touched, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    build_replay_index([first, second], max_games=16, cache_bytes=CACHE_BYTES, state_path=state_path)
    assert len(calls) == 7

def test_whole_game_cache_serves_repeats_without_reloading(replay_runs, monkeypatch):
    _, first, second = replay_runs
    index = build_replay_index([first, second], max_games=16, cache_bytes=CACHE_BYTES)
    loads: list[Path] = []
    original = alphazero_replay.load_game

    def counting_load(path):
        loads.append(path)
        return original(path)

    monkeypatch.setattr(alphazero_replay, "load_game", counting_load)
    game = index.games[0]
    arrays = index.cache.sample(game, 0).arrays
    for position in range(min(game.decision_count, 3)):
        sample = index.cache.sample(game, position)
        assert sample.arrays is arrays and sample.index == position
    assert loads == [game.path]
    assert index.cache.entries == 1 and index.cache.bytes > 0


def test_whole_game_cache_evicts_oldest_and_skips_oversized_games(replay_runs):
    _, first, second = replay_runs
    index = build_replay_index([first, second], max_games=16, cache_bytes=CACHE_BYTES)
    games = list(index.games)[:2]
    probe = alphazero_replay._GameCache(max_bytes=1 << 30)
    sizes = []
    for game in games:
        before = probe.bytes
        probe.sample(game, 0)
        sizes.append(probe.bytes - before)
    assert all(size > 0 for size in sizes)
    # Both games fit the budget individually but not together: the oldest goes.
    cache = alphazero_replay._GameCache(max_bytes=max(sizes))
    cache.sample(games[0], 0)
    cache.sample(games[1], 0)
    assert cache.entries == 1 and cache.bytes == sizes[1]
    # A single oversized game is served but never retained.
    oversized = alphazero_replay._GameCache(max_bytes=1)
    assert oversized.sample(games[0], 0).index == 0
    assert oversized.entries == 0 and oversized.bytes == 0


def test_collection_reports_committed_game_progress(tmp_path):
    checkpoint = _weights_checkpoint(tmp_path / "base.pt")
    config = FastSelfPlayConfig(
        protocol="mcts-selfplay-v2",
        run_id="az-progress-round-000",
        round_index=0,
        checkpoint=str(checkpoint),
        num_games=2,
        seed=51001,
        search_config=MCTSConfig(simulations=2, determinizations=1),
        execution=ExecutionConfig(
            backend="batched_cpu", engine_mode="fast", workers=1,
            active_games_per_worker=1, max_batch_size=1,
        ),
        validation_fraction=0.0,
        split_seed=0,
    )
    root = tmp_path / "progress-run"
    seen: list[tuple[int, int]] = []
    summary = generate_selfplay(
        config, root, progress_callback=lambda done, total: seen.append((done, total))
    )
    assert summary["completed_games"] == 2
    assert [done for done, _ in seen] == [1, 2]
    assert {total for _, total in seen} == {2}

    game_path(root, 1).unlink()
    resumed: list[tuple[int, int]] = []
    generate_selfplay(
        config, root, resume=True,
        progress_callback=lambda done, total: resumed.append((done, total)),
    )
    # A resumed session reports absolute progress: one kept game plus the
    # one it re-collected.
    assert resumed == [(2, 2)]
