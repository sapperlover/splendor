from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_loop import LoopConfig
from splendor.alphazero_replay import ReplayRun, build_replay_index
from splendor.alphazero_train import (
    AlphazeroLearner, LearnerConfig, alphazero_loss_sums, draw_training_batch,
    run_training_round,
)
from splendor.engine import GameEngine
from splendor.evaluation import SerializablePolicySpec, _create_serializable_policy
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID
from splendor.selfplay import derive_seed
from splendor.selfplay_data import collate_selfplay_samples
from splendor.selfplay_fast import (
    ExecutionConfig, FastSelfPlayConfig, generate as generate_selfplay,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_BYTES = 64 * 1024 * 1024


def _weights_checkpoint(path: Path) -> Path:
    torch.manual_seed(11)
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
    root: Path,
    checkpoint: Path,
    *,
    round_index: int = 0,
    games: int = 3,
    simulations: int = 4,
) -> ReplayRun:
    config = FastSelfPlayConfig(
        protocol="mcts-selfplay-v2",
        run_id=f"az-test-round-{round_index:03d}",
        round_index=round_index,
        checkpoint=str(checkpoint),
        num_games=games,
        seed=12001 + round_index,
        search_config=MCTSConfig(simulations=simulations, determinizations=1),
        execution=ExecutionConfig(
            backend="batched_cpu", engine_mode="fast", workers=1,
            active_games_per_worker=1, max_batch_size=1,
        ),
        add_root_noise=True,
        validation_fraction=0.5,
        split_seed=_mixed_split_seed(games),
    )
    generate_selfplay(config, root)
    return ReplayRun(round_index, config.run_id, root)


@pytest.fixture(scope="module")
def az_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("az-data")
    checkpoint = _weights_checkpoint(root / "base.pt")
    previous_threads = torch.get_num_threads()
    try:
        first = _selfplay_run(root / "run-000", checkpoint, round_index=0, games=3)
        second = _selfplay_run(root / "run-001", checkpoint, round_index=1, games=3)
    finally:
        torch.set_num_threads(previous_threads)
    return checkpoint, first, second


def test_losses_match_hand_computed_values_and_ignore_padding():
    candidate_mask = torch.tensor([[True, True, False], [True, True, True]])
    policy_targets = torch.tensor([[0.25, 0.75, 0.0], [0.0, 1.0, 0.0]])
    policy_weights = torch.tensor([1.0, 1.0])
    value_targets = torch.tensor([1.0, -1.0])
    value_weights = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, -0.5])
    logits = torch.tensor([[1.0, 0.0, 99.0], [0.5, 0.5, 0.5]])
    terms = alphazero_loss_sums(
        logits, values,
        candidate_mask=candidate_mask, policy_targets=policy_targets,
        value_targets=value_targets, policy_weights=policy_weights,
        value_weights=value_weights,
    )
    softmax = torch.softmax(torch.tensor([1.0, 0.0]), dim=-1)
    expected_ce = (
        -(0.25 * math.log(float(softmax[0])) + 0.75 * math.log(float(softmax[1])))
        - math.log(1.0 / 3.0)
    ) / 2
    assert float(terms.policy_loss) == pytest.approx(expected_ce, abs=1e-6)
    assert float(terms.value_loss) == pytest.approx(0.25, abs=1e-9)
    padded = logits.clone()
    padded[0, 2] = -99.0
    other = alphazero_loss_sums(
        padded, values,
        candidate_mask=candidate_mask, policy_targets=policy_targets,
        value_targets=value_targets, policy_weights=policy_weights,
        value_weights=value_weights,
    )
    assert torch.equal(terms.policy_loss, other.policy_loss)
    assert torch.equal(terms.value_loss, other.value_loss)
    probability = float(softmax[0])
    row_entropy = -(probability * math.log(probability) + (1 - probability) * math.log(1 - probability))
    expected_entropy = (row_entropy + math.log(3.0)) / 2
    assert float(terms.entropy) == pytest.approx(expected_entropy, abs=1e-6)


def test_single_candidate_row_is_excluded_from_policy_denominator():
    config = LearnerConfig(device="cpu", batch_size=2, history_batch_size=1)
    logits = torch.tensor([[0.3, 50.0], [0.5, 0.5]])
    mask = torch.tensor([[True, False], [True, True]])
    terms = alphazero_loss_sums(
        logits, torch.zeros(2), candidate_mask=mask,
        policy_targets=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        value_targets=torch.tensor([1.0, -1.0]),
        policy_weights=torch.tensor([0.0, 1.0]), value_weights=torch.ones(2),
    )
    # Only the searchable row contributes: CE is exactly log(2).
    assert float(terms.policy_loss) == pytest.approx(math.log(2.0), abs=1e-6)
    assert torch.isfinite(terms.total(config)).all()


def test_all_zero_policy_weight_batch_has_zero_policy_loss():
    config = LearnerConfig(device="cpu", batch_size=2, history_batch_size=1)
    terms = alphazero_loss_sums(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0.5, -0.5]),
        candidate_mask=torch.ones(2, 2, dtype=torch.bool),
        policy_targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        value_targets=torch.tensor([1.0, -1.0]),
        policy_weights=torch.zeros(2), value_weights=torch.ones(2),
    )
    assert float(terms.policy_loss) == 0.0
    assert float(terms.value_loss) == pytest.approx(0.25, abs=1e-9)
    assert float(terms.total(config)) == pytest.approx(0.25, abs=1e-9)


def test_steps_target_formula():
    config = LearnerConfig(device="cpu", batch_size=256)
    assert config.steps_for_round(0) == 0
    assert config.steps_for_round(64) == 1
    assert config.steps_for_round(65) == 2
    assert config.steps_for_round(5000) == math.ceil(4 * 5000 / 256)
    capped = LearnerConfig(device="cpu", batch_size=256, max_steps_per_round=4)
    assert capped.steps_for_round(5000) == 4


def test_zero_value_loss_weight_removes_value_gradient():
    config = LearnerConfig(device="cpu", value_loss_weight=0.0)
    logits = torch.tensor([[0.2, -0.3]], requires_grad=True)
    values = torch.tensor([0.5], requires_grad=True)
    terms = alphazero_loss_sums(
        logits, values, candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        policy_targets=torch.tensor([[1.0, 0.0]]), value_targets=torch.tensor([-1.0]),
        policy_weights=torch.ones(1), value_weights=torch.ones(1),
    )
    assert torch.equal(terms.total(config), terms.policy_loss)
    terms.total(config).backward()
    assert logits.grad.abs().sum() > 0
    assert torch.equal(values.grad, torch.zeros_like(values))
    assert float(terms.value_loss.detach()) == pytest.approx(2.25)


@pytest.mark.parametrize("weight", [-0.25, float("nan"), float("inf"), True])
def test_invalid_value_loss_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="value_loss_weight"):
        LearnerConfig(value_loss_weight=weight)


def test_learner_update_changes_parameters_and_handles_zero_policy_batch(az_runs):
    checkpoint, first, second = az_runs
    index = build_replay_index([first, second], max_games=10, cache_bytes=CACHE_BYTES)
    current = index.position_pool(split="train", round_index=1)
    history = index.position_pool(split="train", before_round=1)
    assert len(current) > 0 and len(history) > 0
    config = LearnerConfig(
        device="cpu", batch_size=4, history_batch_size=2,
        learning_rate=1e-2, checkpoint_every_steps=100,
    )
    learner = AlphazeroLearner.warm_start(
        checkpoint, config, sampling_seed=7, replay_identity=index.identity(),
        parent_checkpoint={"path": str(checkpoint), "sha256": "unit-test"},
        round_index=1, round_steps_target=1,
    )
    before = {name: tensor.clone() for name, tensor in learner.model.state_dict().items()}
    drawn = draw_training_batch(current, history, learner.sampling_rng, config)
    assert drawn.current_positions == 2 and drawn.history_positions == 2
    assert all(reference.round_index == 1 for reference in drawn.references[:2])
    assert all(reference.round_index == 0 for reference in drawn.references[2:])
    batch = collate_selfplay_samples(drawn.samples)
    record = learner.train_step(batch, references=drawn.references)
    assert math.isfinite(record["total_loss"])
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in learner.model.state_dict().items()
    )
    # A batch whose policy weights are all zero still updates the critic.
    zero_batch = collate_selfplay_samples(drawn.samples)
    zero_batch.policy_weights[:] = 0.0
    critic_before = {name: tensor.clone() for name, tensor in learner.model.critic.state_dict().items()}
    record = learner.train_step(zero_batch, references=drawn.references)
    assert record["policy_loss"] == 0.0
    assert math.isfinite(record["value_loss"])
    assert any(
        not torch.equal(critic_before[name], tensor)
        for name, tensor in learner.model.critic.state_dict().items()
    )


def test_checkpoint_roundtrip_matches_continuous_training(az_runs, tmp_path):
    checkpoint, first, second = az_runs
    index = build_replay_index([first, second], max_games=10, cache_bytes=CACHE_BYTES)
    current = index.position_pool(split="train", round_index=1)
    history = index.position_pool(split="train", before_round=1)
    identity = index.identity()
    config = LearnerConfig(
        device="cpu", batch_size=4, history_batch_size=2,
        learning_rate=1e-2, checkpoint_every_steps=100,
    )
    learner = AlphazeroLearner.warm_start(
        checkpoint, config, sampling_seed=99, replay_identity=identity,
        parent_checkpoint={"path": str(checkpoint), "sha256": "unit-test"},
        round_index=1, round_steps_target=5,
    )
    draws = []
    checkpoint_path = tmp_path / "step-000000002.pt"
    for _ in range(5):
        drawn = draw_training_batch(current, history, learner.sampling_rng, config)
        draws.append(drawn)
        learner.train_step(collate_selfplay_samples(drawn.samples), references=drawn.references)
        if learner.round_train_step == 2:
            learner.save(checkpoint_path)
    reference_parameters = {name: tensor.clone() for name, tensor in learner.model.state_dict().items()}

    resumed = AlphazeroLearner.resume(
        checkpoint_path, config, replay_identity=identity, round_index=1, round_steps_target=5,
    )
    assert resumed.round_train_step == 2
    drawn = draw_training_batch(current, history, resumed.sampling_rng, config)
    assert [reference.to_dict() for reference in drawn.references] == [
        reference.to_dict() for reference in draws[2].references
    ]
    for _ in range(3):
        resumed.train_step(
            collate_selfplay_samples(drawn.samples), references=drawn.references
        )
        if resumed.round_train_step < 5:
            drawn = draw_training_batch(current, history, resumed.sampling_rng, config)
    for name, tensor in resumed.model.state_dict().items():
        assert torch.equal(reference_parameters[name], tensor), name
    payload = load_alphazero_checkpoint(checkpoint_path)
    assert payload.round_train_step == 2 and payload.total_train_step == 2


def test_az_checkpoint_loads_in_frozen_search_and_evaluation(az_runs, tmp_path):
    checkpoint, first, _ = az_runs
    index = build_replay_index([first], max_games=10, cache_bytes=CACHE_BYTES)
    current = index.position_pool(split="train", round_index=0)
    config = LearnerConfig(device="cpu", batch_size=4, history_batch_size=2)
    learner = AlphazeroLearner.warm_start(
        checkpoint, config, sampling_seed=5, replay_identity=index.identity(),
        parent_checkpoint={"path": str(checkpoint), "sha256": "unit-test"},
        round_index=0, round_steps_target=1,
    )
    drawn = draw_training_batch(current, None, learner.sampling_rng, config)
    learner.train_step(collate_selfplay_samples(drawn.samples), references=drawn.references)
    az_path = tmp_path / "theta.pt"
    learner.save(az_path, metadata={"unit": True})

    model = load_frozen_actor_critic_checkpoint(az_path, device="cpu")
    payload = load_alphazero_checkpoint(az_path)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, payload.model_state_dict[name])
    assert model.training is False and all(not parameter.requires_grad for parameter in model.parameters())

    request = GameEngine.new(77).decision()
    policy = PUCTSearchPolicy(
        model, config=MCTSConfig(simulations=8, determinizations=1), device="cpu"
    )
    result = policy.search(request)
    assert 0 <= result.chosen_candidate_index < len(request.legal_actions)

    greedy = _create_serializable_policy(
        SerializablePolicySpec.greedy_checkpoint("az-greedy", az_path), seed=0, device="cpu"
    )
    assert greedy.choose(request) in request.legal_actions
    searched = _create_serializable_policy(
        SerializablePolicySpec.mcts_checkpoint(
            "az-mcts", az_path, search_config=asdict(MCTSConfig(simulations=8, determinizations=1))
        ),
        seed=1,
        device="cpu",
    )
    assert searched.choose(request) in request.legal_actions


def test_production_loop_config_freezes_required_values():
    config = LoopConfig.from_dict(json.loads((ROOT / "configs/alphazero_loop.json").read_text(encoding="utf-8")))
    assert config.num_rounds == 5 and config.new_games_per_round == 500
    assert config.replay_max_games == 3000
    search = config.selfplay.search_config
    assert search["simulations"] == 1024 and search["determinizations"] == 1
    assert search["c_puct"] == 1.5 and search["fold_forced_phases"] is True
    execution = config.selfplay.execution
    assert execution["backend"] == "batched_cpu" and execution["engine_mode"] == "fast"
    assert execution["workers"] == 8
    assert execution["active_games_per_worker"] == 4 and execution["max_batch_size"] == 4
    assert config.selfplay.temperature_schedule["opening_completed_turns"] == 10
    assert config.learner.device == "cuda" and config.learner.batch_size == 256
    assert config.learner.history_batch_size == 128
    assert config.evaluation.mcts_vs_parent.pairs == 10
    assert config.evaluation.greedy_vs_initial.pairs == 20
    assert config.evaluation.workers == 8 and config.evaluation.device == "cpu"
    # Each round derives a distinct collection/split/sampling seed from its index.
    for domain in (
        config.seeds.collection_for, config.seeds.split_for, config.seeds.sampling_for,
    ):
        assert len({domain(r) for r in range(config.num_rounds)}) == config.num_rounds

def _train_learner(learner, pool, config, steps: int) -> None:
    for _ in range(steps):
        drawn = draw_training_batch(pool, None, learner.sampling_rng, config)
        learner.train_step(collate_selfplay_samples(drawn.samples), references=drawn.references)


def test_continue_from_carries_optimizer_and_global_step(az_runs, tmp_path):
    checkpoint, first, second = az_runs
    index = build_replay_index([first, second], max_games=10, cache_bytes=CACHE_BYTES)
    current = index.position_pool(split="train", round_index=1)
    history = index.position_pool(split="train", before_round=1)
    assert len(history) > 0
    config = LearnerConfig(
        device="cpu", batch_size=4, history_batch_size=2,
        learning_rate=1e-2, checkpoint_every_steps=100,
    )
    learner = AlphazeroLearner.warm_start(
        checkpoint, config, sampling_seed=17, replay_identity=index.identity(),
        parent_checkpoint={"path": str(checkpoint), "sha256": "unit-test"},
        round_index=0, round_steps_target=2,
    )
    _train_learner(learner, current, config, 2)
    parent_path = tmp_path / "round-000-final.pt"
    learner.save(parent_path)
    parent = load_alphazero_checkpoint(parent_path)
    assert parent.round_index == 0 and parent.total_train_step == 2

    continued = AlphazeroLearner.continue_from(
        parent_path, config, sampling_seed=23, replay_identity=index.identity(),
        parent_checkpoint={"path": str(parent_path), "sha256": "parent"},
        round_index=1, round_steps_target=2,
    )
    assert (continued.round_index, continued.round_train_step) == (1, 0)
    assert continued.total_train_step == 2 and continued.round_steps_target == 2
    for name, tensor in parent.model_state_dict.items():
        assert torch.equal(tensor, continued.model.state_dict()[name])
    restored = continued.optimizer.state_dict()["state"]
    assert restored.keys() == parent.optimizer_state_dict["state"].keys()
    for key, value in parent.optimizer_state_dict["state"].items():
        assert int(restored[key]["step"]) == int(value["step"])
        assert torch.equal(restored[key]["exp_avg"], value["exp_avg"])

    drawn = draw_training_batch(current, history, continued.sampling_rng, config)
    record = continued.train_step(
        collate_selfplay_samples(drawn.samples), references=drawn.references
    )
    assert record["round_train_step"] == 1 and record["total_train_step"] == 3
    # The round's sampling stream follows its own seed, not the parent's state.
    twin = AlphazeroLearner.continue_from(
        parent_path, config, sampling_seed=23, replay_identity=index.identity(),
        parent_checkpoint={"path": str(parent_path), "sha256": "parent"},
        round_index=1, round_steps_target=2,
    )
    twin_drawn = draw_training_batch(current, history, twin.sampling_rng, config)
    assert [reference.to_dict() for reference in twin_drawn.references] == [
        reference.to_dict() for reference in drawn.references
    ]


def test_metrics_file_is_reconciled_after_interrupted_round(az_runs, tmp_path):
    checkpoint, first, second = az_runs
    index = build_replay_index([first, second], max_games=10, cache_bytes=CACHE_BYTES)
    current = index.position_pool(split="train", round_index=1)
    history = index.position_pool(split="train", before_round=1)
    config = LearnerConfig(
        device="cpu", batch_size=4, history_batch_size=2,
        learning_rate=1e-2, checkpoint_every_steps=2,
    )
    learner = AlphazeroLearner.warm_start(
        checkpoint, config, sampling_seed=31, replay_identity=index.identity(),
        parent_checkpoint={"path": str(checkpoint), "sha256": "unit-test"},
        round_index=1, round_steps_target=4,
    )
    _train_learner(learner, current, config, 2)
    checkpoint_path = tmp_path / "step-000000002.pt"
    learner.save(checkpoint_path)
    resumed = AlphazeroLearner.resume(
        checkpoint_path, config, replay_identity=index.identity(),
        round_index=1, round_steps_target=4,
    )
    metrics_path = tmp_path / "training_metrics.jsonl"
    stale = [
        {
            "event": "train_step", "round_index": 1, "round_train_step": step,
            "total_train_step": step, "total_loss": 1.0,
        }
        for step in (1, 2, 3, 4)
    ]
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in stale)
        + '{"event": "train_step", "round_index\n',
        encoding="utf-8",
    )

    result = run_training_round(
        resumed,
        steps_target=4,
        current_pool=current,
        history_pool=history,
        validation_samples=[],
        validation_games=0,
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_path=metrics_path,
    )
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    step_records = [record for record in records if record["event"] == "train_step"]
    # Steps the crashed session ran but never checkpointed are not written
    # twice: the checkpointed 1-2 stay, the re-run 3-4 replace the stale lines.
    assert [record["round_train_step"] for record in step_records] == [1, 2, 3, 4]
    assert [record["total_train_step"] for record in step_records] == [1, 2, 3, 4]
    assert len({record["round_train_step"] for record in step_records}) == 4
    assert sum(record["event"] == "validation" for record in records) == 1
    assert records[-1]["event"] == "validation"
    assert result["steps_completed"] == 4 and result["steps_this_session"] == 2
