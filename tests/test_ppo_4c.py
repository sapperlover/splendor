from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor._state import DeckState, GameState, MarketState, PlayerState, ReservedCard
from splendor.actions import Action, BuyReserved, TakeDistinct
from splendor.actor_critic import ActorCritic
from splendor.catalog import Catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import EncodedDecision, collate_encoded_decisions
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, ReservationSource, Tier, TokenVector
from splendor.ppo import (
    PPOConfig,
    PPOTrainer,
    PolicySampler,
    RolloutBatch,
    RolloutCollector,
    RolloutSample,
    clipped_policy_loss,
    compute_gae,
    explained_variance,
    normalize_advantages,
)
from splendor.random_policy import RandomPolicy
from splendor.replay import new_replay
from splendor.rules import legal_actions, player_score
from splendor.training_env import LearningEnvironment


def _encoded_decision(tmp_path: Path, seed: int = 3) -> EncodedDecision:
    environment = LearningEnvironment.new(
        seed,
        first_player=0,
        learning_player=0,
        opponent_policy=RandomPolicy(seed + 100),
        episode_id=f"encoded-{seed}",
        failure_dir=tmp_path,
    )
    return environment.current_decision().encoded


def _sample(
    encoded: EncodedDecision,
    *,
    old_value: float,
    reward: float,
    episode_id: str,
    bootstrap_value: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    old_logprob: float = 0.0,
) -> RolloutSample:
    return RolloutSample(
        encoded=encoded,
        chosen_candidate_index=0,
        old_logprob=old_logprob,
        old_value=old_value,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        episode_id=episode_id,
        learning_player=0,
        bootstrap_value=bootstrap_value,
    )


def _random_environment_factory(tmp_path: Path):
    def factory(index: int) -> LearningEnvironment:
        learning_player = index % 2
        return LearningEnvironment.new(
            1000 + index,
            first_player=learning_player,
            learning_player=learning_player,
            opponent_policy=RandomPolicy(2000 + index),
            episode_id=f"smoke-{index}",
            max_completed_turns=30,
            failure_dir=tmp_path / str(index),
        )

    return factory


class _FirstPolicy:
    def choose(self, request: DecisionRequest) -> Action:
        return request.legal_actions[0]


def _decisive_final_state(
    initial_state: GameState,
    catalog: Catalog,
    learning_player: int,
) -> GameState:
    """Legal final reply: buy the reserved card to win, or take one token and lose."""

    opponent = 1 - learning_player
    learner_cards = ("090", "076", "080", "008")
    opponent_cards = ("078", "082", "086")
    learner_reserved = ("055", "007", "011")
    market_by_tier = {
        Tier.ONE: ("002", "003", "005", "006"),
        Tier.TWO: ("041", "042", "043", "044"),
        Tier.THREE: ("071", "072", "073", "074"),
    }
    located = set(
        learner_cards
        + opponent_cards
        + learner_reserved
        + tuple(card for cards in market_by_tier.values() for card in cards)
    )
    deck_by_tier = {
        tier: tuple(
            card.card_id
            for card in catalog.cards
            if card.tier is tier and card.card_id not in located
        )
        for tier in Tier
    }
    learner_tokens = TokenVector(green=1, red=4, black=4)
    opponent_tokens = TokenVector(white=3, blue=4, green=3)
    players = [PlayerState(TokenVector()), PlayerState(TokenVector())]
    players[learning_player] = PlayerState(
        tokens=learner_tokens,
        purchased_card_ids=learner_cards,
        reserved_cards=tuple(
            ReservedCard(card_id, ReservationSource.DECK)
            for card_id in learner_reserved
        ),
    )
    players[opponent] = PlayerState(
        tokens=opponent_tokens,
        purchased_card_ids=opponent_cards,
    )
    # Both players have had enough prior turns to acquire the authoritative
    # assets above; the first player has just triggered the final round.
    completed_turns = [11, 11]
    completed_turns[opponent] = 12
    state = replace(
        initial_state,
        supply=TokenVector(white=1, gold=5),
        decks=DeckState(
            deck_by_tier[Tier.ONE],
            deck_by_tier[Tier.TWO],
            deck_by_tier[Tier.THREE],
        ),
        market=MarketState(
            market_by_tier[Tier.ONE],
            market_by_tier[Tier.TWO],
            market_by_tier[Tier.THREE],
        ),
        players=(players[0], players[1]),
        first_player=opponent,
        current_player=learning_player,
        completed_turns=(completed_turns[0], completed_turns[1]),
        phase=DecisionPhase.MAIN_ACTION,
        pending=None,
        final_round_triggered_by=opponent,
    )
    assert_game_state(state, catalog)
    assert player_score(state.players[learning_player], catalog) == 14
    assert player_score(state.players[opponent], catalog) == 15
    assert legal_actions(state, catalog) == (
        TakeDistinct((Color.WHITE,)),
        BuyReserved("055"),
    )
    return state


def _decisive_environment_factory(
    state: GameState,
    catalog: Catalog,
    learning_player: int,
    tmp_path: Path,
):
    def factory(index: int) -> LearningEnvironment:
        engine = GameEngine(
            state,
            catalog,
            new_replay(catalog, 9000 + index, state.first_player),
            tmp_path / str(index),
        )
        return LearningEnvironment(
            engine,
            learning_player=learning_player,
            opponent_policy=_FirstPolicy(),
            episode_id=f"decisive-{learning_player}-{index}",
        )

    return factory


def _action_probability(
    model: ActorCritic,
    encoded: EncodedDecision,
    action: Action,
) -> float:
    index = encoded.candidate_actions.index(action)
    with torch.inference_mode():
        distribution, _ = model.distribution(collate_encoded_decisions((encoded,)))
    return float(distribution.probs[0, index].item())


def test_terminal_win_loss_and_draw_gae_are_exact(tmp_path: Path) -> None:
    encoded = _encoded_decision(tmp_path)
    rollout = RolloutBatch(
        (
            _sample(
                encoded,
                old_value=0.25,
                reward=1.0,
                episode_id="win",
                bootstrap_value=99.0,
                terminated=True,
            ),
            _sample(encoded, old_value=-0.2, reward=-1.0, episode_id="loss", terminated=True),
            _sample(encoded, old_value=0.7, reward=0.0, episode_id="draw", terminated=True),
        )
    )

    prepared = compute_gae(rollout, gamma=1.0, gae_lambda=0.95)

    np.testing.assert_allclose(prepared.advantages, (0.75, -0.8, -0.7))
    np.testing.assert_allclose(prepared.returns, (1.0, -1.0, 0.0))


def test_truncation_bootstraps_but_does_not_cross_episode(tmp_path: Path) -> None:
    encoded = _encoded_decision(tmp_path)
    rollout = RolloutBatch(
        (
            _sample(
                encoded,
                old_value=0.25,
                reward=0.0,
                episode_id="truncated",
                bootstrap_value=0.6,
                truncated=True,
            ),
            _sample(
                encoded,
                old_value=0.0,
                reward=1.0,
                episode_id="new-episode",
                terminated=True,
            ),
        )
    )

    prepared = compute_gae(rollout, gamma=1.0, gae_lambda=0.95)

    np.testing.assert_allclose(prepared.advantages, (0.35, 1.0))
    np.testing.assert_allclose(prepared.returns, (0.6, 1.0))


@pytest.mark.parametrize(
    ("gae_lambda", "expected_first_advantage"),
    ((0.95, 0.86), (1.0, 0.9)),
)
def test_configurable_lambda_matches_short_trajectory_math(
    gae_lambda: float,
    expected_first_advantage: float,
    tmp_path: Path,
) -> None:
    encoded = _encoded_decision(tmp_path)
    rollout = RolloutBatch(
        (
            _sample(
                encoded,
                old_value=0.1,
                reward=0.0,
                episode_id="same",
                bootstrap_value=0.2,
            ),
            _sample(
                encoded,
                old_value=0.2,
                reward=1.0,
                episode_id="same",
                terminated=True,
            ),
        )
    )

    prepared = compute_gae(rollout, gamma=1.0, gae_lambda=gae_lambda)

    np.testing.assert_allclose(
        prepared.advantages, (expected_first_advantage, 0.8), rtol=1e-6
    )
    np.testing.assert_allclose(
        prepared.returns, (expected_first_advantage + 0.1, 1.0), rtol=1e-6
    )


def test_rollout_batch_edge_uses_true_successor_value(tmp_path: Path) -> None:
    encoded = _encoded_decision(tmp_path)
    sample = _sample(
        encoded,
        old_value=0.1,
        reward=0.0,
        episode_id="unfinished",
        bootstrap_value=0.4,
    )

    prepared = compute_gae((sample,), gamma=1.0, gae_lambda=0.95)

    np.testing.assert_allclose(prepared.advantages, (0.3,))
    np.testing.assert_allclose(prepared.returns, (0.4,))


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_loss"),
    (
        (1.0, 1.0, -1.0),
        (1.5, 1.0, -1.2),
        (0.5, -1.0, 0.8),
        (1.5, -1.0, 1.5),
    ),
)
def test_clipped_policy_loss_has_correct_direction(
    ratio: float, advantage: float, expected_loss: float
) -> None:
    old = torch.tensor([0.0])
    new = torch.tensor([np.log(ratio)], dtype=torch.float32)
    advantages = torch.tensor([advantage])

    loss = clipped_policy_loss(new, old, advantages, clip_coef=0.2)

    assert float(loss.item()) == pytest.approx(expected_loss, abs=1e-6)


def test_small_batch_normalization_and_explained_variance_are_finite() -> None:
    normalized = normalize_advantages(np.asarray([4.0], dtype=np.float32))
    np.testing.assert_array_equal(normalized, np.asarray([0.0], dtype=np.float32))
    assert np.isfinite(normalized).all()
    assert explained_variance(torch.tensor([1.0]), torch.tensor([0.5])) == 0.0


def test_sampler_reconstructs_logprob_and_does_not_change_policy(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = ActorCritic()
    sampler = PolicySampler(model, seed=22)
    encoded = _encoded_decision(tmp_path)
    parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())

    sample = sampler.sample(encoded)
    distribution, values = model.distribution(collate_encoded_decisions((encoded,)))
    reconstructed = distribution.log_prob(torch.tensor([sample.chosen_candidate_index]))
    seen_indices = {sample.chosen_candidate_index}
    seen_indices.update(sampler.sample(encoded).chosen_candidate_index for _ in range(31))

    assert encoded.candidate_mask[sample.chosen_candidate_index]
    assert len(seen_indices) > 1
    assert reconstructed.item() == pytest.approx(sample.old_logprob, abs=1e-7)
    assert values[0].item() == pytest.approx(sample.old_value, abs=1e-7)
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    )


def test_single_candidate_probability_logprob_and_entropy_exclusion(tmp_path: Path) -> None:
    torch.manual_seed(12)
    model = ActorCritic()
    encoded = _encoded_decision(tmp_path)
    single = replace(
        encoded,
        candidate_features=encoded.candidate_features[:1].copy(),
        candidate_mask=np.ones(1, dtype=np.bool_),
        candidate_actions=encoded.candidate_actions[:1],
    )
    sampler = PolicySampler(model, seed=23)
    sampled = sampler.sample(single)
    rollout = RolloutBatch(
        (
            _sample(
                single,
                old_value=sampled.old_value,
                old_logprob=sampled.old_logprob,
                reward=1.0,
                episode_id="single",
                terminated=True,
            ),
        )
    )
    prepared = compute_gae(rollout, gamma=1.0, gae_lambda=0.95)
    trainer = PPOTrainer(
        model,
        PPOConfig(ppo_epochs=1, minibatch_size=1, entropy_coef=0.5),
    )

    stats = trainer.update(prepared)

    assert sampled.chosen_candidate_index == 0
    assert sampled.old_logprob == pytest.approx(0.0, abs=1e-7)
    assert stats.entropy == 0.0
    assert stats.entropy_active_sample_count == 0
    assert all(
        np.isfinite(value)
        for value in (
            stats.policy_loss,
            stats.value_loss,
            stats.total_loss,
            stats.approx_kl,
            stats.grad_norm,
            stats.explained_variance,
        )
    )


def test_collector_freezes_policy_and_retains_mid_episode_successor(tmp_path: Path) -> None:
    torch.manual_seed(13)
    model = ActorCritic()
    sampler = PolicySampler(model, seed=24)
    collector = RolloutCollector(sampler, _random_environment_factory(tmp_path))
    parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())

    rollout = collector.collect(5)

    assert len(rollout) == 5
    assert collector.active_environment is not None
    assert not rollout.samples[-1].terminated
    assert not rollout.samples[-1].truncated
    assert np.isfinite(rollout.samples[-1].bootstrap_value)
    for current, following in zip(
        rollout.samples, rollout.samples[1:], strict=False
    ):
        if (
            not current.terminated
            and not current.truncated
            and current.episode_id == following.episode_id
        ):
            assert current.bootstrap_value == pytest.approx(
                following.old_value, abs=1e-7
            )
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_end_to_end_smoke_updates_parameters(
    device: str, tmp_path: Path
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not visible in this execution context")
    torch.manual_seed(14)
    model = ActorCritic()
    sampler = PolicySampler(model, seed=25, device=device)
    collector = RolloutCollector(sampler, _random_environment_factory(tmp_path / device))
    rollout = collector.collect(16)
    prepared = compute_gae(rollout, gamma=1.0, gae_lambda=0.95)
    before = tuple(parameter.detach().cpu().clone() for parameter in model.parameters())
    trainer = PPOTrainer(
        model,
        PPOConfig(ppo_epochs=1, minibatch_size=8, rollout_steps=16),
        device=device,
        shuffle_seed=26,
    )

    stats = trainer.update(prepared)

    assert stats.valid_sample_count == 16
    assert stats.minibatches_processed == 2
    assert all(np.isfinite(value) for value in (
        stats.policy_loss,
        stats.value_loss,
        stats.entropy,
        stats.total_loss,
        stats.approx_kl,
        stats.clip_fraction,
        stats.grad_norm,
        stats.explained_variance,
    ))
    assert any(
        not torch.equal(previous, current.detach().cpu())
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_target_kl_stops_epochs_without_discarding_optimizer_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(15)
    model = ActorCritic()
    sampler = PolicySampler(model, seed=27)
    collector = RolloutCollector(sampler, _random_environment_factory(tmp_path))
    collected = collector.collect(8)
    shifted = RolloutBatch(
        tuple(replace(sample, old_logprob=sample.old_logprob + 5.0) for sample in collected.samples)
    )
    prepared = compute_gae(shifted, gamma=1.0, gae_lambda=0.95)
    trainer = PPOTrainer(
        model,
        PPOConfig(ppo_epochs=4, minibatch_size=4, target_kl=0.02),
        shuffle_seed=28,
    )

    stats = trainer.update(prepared)

    assert stats.early_stopped
    assert stats.epochs_completed == 1
    assert stats.minibatches_processed == 1
    assert trainer.optimizer.state


def test_checkpoint_restores_optimizer_rng_and_exact_cpu_continuation(
    tmp_path: Path,
) -> None:
    random_seed = 101
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    import random

    random.seed(random_seed)
    model = ActorCritic()
    sampler = PolicySampler(model, seed=102)
    collector = RolloutCollector(sampler, _random_environment_factory(tmp_path / "data"))
    prepared = compute_gae(
        collector.collect(12), gamma=1.0, gae_lambda=0.95
    )
    config = PPOConfig(ppo_epochs=2, minibatch_size=5, target_kl=None)
    trainer = PPOTrainer(model, config, shuffle_seed=103)
    trainer.update(prepared)
    checkpoint = tmp_path / "training.pt"
    trainer.save_checkpoint(checkpoint, sampler=sampler, metadata={"tag": "resume"})

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = float(torch.rand(()).item())
    sample_decision = prepared.samples[0].encoded
    expected_sample = sampler.sample(sample_decision)
    expected_stats = trainer.update(prepared)
    expected_parameters = tuple(
        parameter.detach().clone() for parameter in trainer.model.parameters()
    )

    restored, restored_sampler, metadata = PPOTrainer.load_checkpoint(checkpoint)
    assert restored_sampler is not None
    assert metadata == {"tag": "resume"}
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert float(torch.rand(()).item()) == expected_torch
    assert restored_sampler.sample(sample_decision) == expected_sample
    restored_stats = restored.update(prepared)

    assert restored_stats == expected_stats
    assert restored.global_learning_decision_step == trainer.global_learning_decision_step
    assert restored.update_count == trainer.update_count
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            expected_parameters, restored.model.parameters(), strict=True
        )
    )


@pytest.mark.parametrize("learning_player", (0, 1))
def test_formal_one_step_win_rises_and_one_step_loss_falls_for_both_seats(
    learning_player: int,
    initial_state: GameState,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    state = _decisive_final_state(initial_state, catalog, learning_player)
    factory = _decisive_environment_factory(
        state, catalog, learning_player, tmp_path / str(learning_player)
    )
    initial_decision = factory(999).current_decision().encoded
    win_action = BuyReserved("055")
    loss_action = TakeDistinct((Color.WHITE,))

    loss_environment = factory(1000)
    loss_decision = loss_environment.current_decision()
    loss_transition = loss_environment.step_learning_action(
        loss_decision.legal_actions.index(loss_action),
        decision_id=loss_decision.decision_id,
    )
    assert loss_transition.terminated and loss_transition.reward == -1.0

    win_environment = factory(1001)
    win_decision = win_environment.current_decision()
    selected = win_environment.step_learning_action(
        win_decision.legal_actions.index(win_action),
        decision_id=win_decision.decision_id,
    )
    assert not selected.terminated and selected.reward == 0.0
    payment = win_environment.current_decision()
    assert len(payment.legal_actions) == 1
    won = win_environment.step_learning_action(
        0, decision_id=payment.decision_id
    )
    assert won.terminated and won.reward == 1.0

    torch.manual_seed(400 + learning_player)
    model = ActorCritic()
    sampler = PolicySampler(model, seed=500 + learning_player)
    collector = RolloutCollector(sampler, factory)
    trainer = PPOTrainer(
        model,
        PPOConfig(
            learning_rate=2e-3,
            entropy_coef=0.0,
            ppo_epochs=4,
            minibatch_size=64,
            target_kl=0.08,
            rollout_steps=128,
        ),
        shuffle_seed=600 + learning_player,
    )
    initial_win = _action_probability(model, initial_decision, win_action)
    initial_loss = _action_probability(model, initial_decision, loss_action)

    for _ in range(6):
        rollout = collector.collect(128)
        prepared = compute_gae(
            rollout,
            gamma=trainer.config.gamma,
            gae_lambda=trainer.config.gae_lambda,
        )
        stats = trainer.update(prepared)
        assert np.isfinite(stats.total_loss)

    final_win = _action_probability(model, initial_decision, win_action)
    final_loss = _action_probability(model, initial_decision, loss_action)

    assert final_win > initial_win + 0.25
    assert final_win > 0.8
    assert final_loss < initial_loss - 0.25
    assert final_loss < 0.2
