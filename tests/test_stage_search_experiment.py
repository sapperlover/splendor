"""Finite checks for the isolated frozen stage-search experiment."""
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from splendor.actor_critic import ActorCritic, ActorCriticConfig
from splendor.engine import GameEngine
from splendor.evaluation import run_fixed_evaluation
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.neural_policy import GreedyActorCriticPolicy
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID

PATH = Path(__file__).resolve().parents[1] / "artifacts/entity_attention_stage_search_20261008/run.py"
SPEC = importlib.util.spec_from_file_location("stage_search_experiment", PATH)
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


@pytest.fixture
def model():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        value = ActorCritic(ActorCriticConfig(card_hidden=8, noble_hidden=8, state_hidden=16,
                                            actor_hidden=16, critic_hidden=16)).eval()
    yield value
    torch.set_num_threads(threads)


def test_stage_boundary_uses_both_public_scores():
    request = GameEngine.new(42).decision()
    for own, opponent, expected in [(9, 9, "early"), (10, 0, "late"), (0, 10, "late"), (15, 15, "late")]:
        observation = replace(request.observation,
            self_public=replace(request.observation.self_public, score=own),
            opponent_public=replace(request.observation.opponent_public, score=opponent))
        assert experiment.stage_for(replace(request, observation=observation), 10) == expected


def test_greedy_skips_tree_and_searched_branch_preserves_native(model):
    request = GameEngine.new(42).decision()
    config = MCTSConfig(simulations=8, determinizations=1, root_noise_fraction=0., seed=123)
    late = experiment.make_stage_policy("late_only", 10)(model, config=config)
    job = late.start_job(request)
    assert job.done
    assert request.legal_actions[job.result.chosen_candidate_index] == GreedyActorCriticPolicy(model).choose(request)
    assert late.stats.simulations == 0 and not late._active_jobs
    assert late.stats.early_greedy_decisions == 1

    early = experiment.make_stage_policy("early_only", 10)(model, config=config)
    actual = early.search(request)
    expected = NativePUCTSearchPolicy(model, config=config).search(request)
    assert actual.root_visit_counts == expected.root_visit_counts
    assert actual.chosen_candidate_index == expected.chosen_candidate_index
    assert early.stats.early_simulations == 8 and early.stats.late_simulations == 0
    assert early.stats.early_tree_depth_sum == early.stats.tree_depth_sum
    assert early.stats.early_terminal_hits == early.stats.terminal_hits


@pytest.mark.parametrize('late', [False, True])
def test_full_search_keeps_native_budget_and_actions_in_both_stages(model, monkeypatch, late):
    monkeypatch.syspath_prepend(str(PATH.parent))
    spec = importlib.util.spec_from_file_location('full_search_experiment', PATH.with_name('full_search.py'))
    full = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full)
    request = GameEngine.new(42).decision()
    if late:
        request = replace(request, observation=replace(request.observation,
            self_public=replace(request.observation.self_public, score=10)))
    config = MCTSConfig(simulations=8, determinizations=1, root_noise_fraction=0., seed=123)
    policy = full.FullSearchPolicy(model, config=config)
    actual = policy.search(request)
    expected = NativePUCTSearchPolicy(model, config=config).search(request)
    assert actual.root_visit_counts == expected.root_visit_counts
    assert actual.chosen_candidate_index == expected.chosen_candidate_index
    assert policy.stats.simulations == 8
    phase = 'late' if late else 'early'
    assert getattr(policy.stats, f'{phase}_simulations') == 8
    assert policy.stats.early_greedy_decisions == policy.stats.late_greedy_decisions == 0
    assert policy.stats.late_entry_count == int(late)


@pytest.mark.parametrize("mode", ["late_only", "early_only"])
def test_checked_batch_resume_and_phase_budgets(tmp_path, model, mode):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "metadata": {"rules_version": RULESET_ID}, "model_config": asdict(model.config),
                "model_state_dict": model.state_dict()}, checkpoint)
    plan = experiment.read_json(PATH.with_name("plan.json"))
    plan.update(checkpoint=str(checkpoint), device="cpu", pairs=2, bootstrap_resamples=100)
    plan["search_config"]["simulations"] = 3
    plan["execution"].update(active_games=2, max_batch_size=2)
    config = experiment.evaluation_config(plan, mode, "finite-test")
    output = tmp_path / f"{mode}.json"
    import splendor.evaluation_batch as batch
    original = batch.NativePUCTSearchPolicy

    def interrupt_after_pair(_):
        raise KeyboardInterrupt

    with experiment.install_stage_policy(mode, 10):
        with pytest.raises(KeyboardInterrupt):
            run_fixed_evaluation(config, output, pair_callback=interrupt_after_pair)
    assert batch.NativePUCTSearchPolicy is original
    sidecar = Path(str(output) + ".games")
    complete = {p: p.read_bytes() for p in sidecar.glob("pair-*.jsonl")
                if len(p.read_text().splitlines()) == 2}
    assert len(complete) == 1
    with experiment.install_stage_policy(mode, 10):
        resumed = run_fixed_evaluation(config, output, resume=True)
    assert resumed.summary["passed"] and resumed.summary["completed_pairs"] == 2
    assert resumed.summary["valid_games"] == 4 and resumed.summary["policy_b_stats"] == {}
    assert all(p.read_bytes() == content for p, content in complete.items())
    stats = resumed.summary["policy_a_stats"]
    wrong_stage = "early" if mode == "late_only" else "late"
    assert stats[f"{wrong_stage}_simulations"] == 0
    assert stats["simulations"] == 3 * stats["searchable_decisions"]
    assert stats["decisions"] == stats["early_decisions"] + stats["late_decisions"]
    assert stats["early_greedy_decisions"] + stats["early_search_decisions"] == stats["early_decisions"]
    assert stats["late_greedy_decisions"] + stats["late_search_decisions"] == stats["late_decisions"]
