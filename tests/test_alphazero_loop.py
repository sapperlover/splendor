from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from splendor import alphazero_loop
from splendor.actor_critic import ActorCritic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_loop import (
    EvaluationPanel, EvaluationSection, LoopConfig, LoopSeeds, SelfPlaySection,
    run_loop,
)
from splendor.alphazero_train import LearnerConfig
from splendor.alphazero_recording import human_event
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.mcts_policy import MCTSConfig
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID
from splendor.selfplay import TemperatureSchedule
from splendor.selfplay_fast import ExecutionConfig


def _weights_checkpoint(path: Path) -> Path:
    torch.manual_seed(41)
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


def _loop_config(checkpoint: Path) -> LoopConfig:
    small_search = asdict(MCTSConfig(simulations=2, determinizations=1))
    return LoopConfig(
        run_id="az-loop-unit",
        num_rounds=2,
        new_games_per_round=1,
        initial_checkpoint=str(checkpoint),
        replay_max_games=8,
        replay_cache_bytes=16 * 1024 * 1024,
        seeds=LoopSeeds(collection=7700000, split=7701000, sampling=7702000),
        selfplay=SelfPlaySection(
            search_config=small_search,
            execution=asdict(
                ExecutionConfig(
                    backend="batched_cpu", engine_mode="fast", workers=1,
                    active_games_per_worker=1, max_batch_size=1,
                )
            ),
            add_root_noise=True,
            temperature_schedule=asdict(TemperatureSchedule()),
            # Every game lands in train, so the round always has positions to train on.
            validation_fraction=0.0,
            max_completed_turns=400,
        ),
        learner=LearnerConfig(
            device="cpu", batch_size=4, history_batch_size=2,
            learning_rate=1e-2, checkpoint_every_steps=2, max_steps_per_round=2,
        ),
        evaluation=EvaluationSection(
            workers=1,
            device="cpu",
            adjudicate_deterministic_policy_cycles=True,
            mcts_vs_parent=EvaluationPanel(
                pairs=1, seed_start=7710000, policy_seed_base=7720000,
                search_config=small_search,
            ),
            greedy_vs_initial=EvaluationPanel(
                pairs=1, seed_start=7711000, policy_seed_base=7721000,
            ),
        ),
    )


@pytest.fixture
def restore_torch_threads():
    previous = torch.get_num_threads()
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("mcts_opponent", ["parent", "initial"])
def test_loop_chains_rounds_and_resumes_to_no_work(
    tmp_path, monkeypatch, capsys, restore_torch_threads, mcts_opponent
):
    checkpoint = _weights_checkpoint(tmp_path / "base.pt")
    config = _loop_config(checkpoint)
    if mcts_opponent == "initial":
        config = replace(config, evaluation=replace(
            config.evaluation, mcts_vs_parent=None,
            mcts_vs_initial=config.evaluation.mcts_vs_parent,
        ))
    panel_name = f"mcts_vs_{mcts_opponent}"
    # Frozen legacy configs retain their fields; fixed-initial configs omit parent.
    serialized = config.to_dict()
    assert f"mcts_vs_{'parent' if mcts_opponent == 'initial' else 'initial'}" not in serialized["evaluation"]
    assert LoopConfig.from_dict(serialized).to_dict() == serialized
    output = tmp_path / "loop"
    collect_calls: list[bool] = []
    generate = alphazero_loop.generate_selfplay

    def counting_generate(config_arg, root, *, resume=False, progress_callback=None):
        collect_calls.append(resume)
        return generate(config_arg, root, resume=resume, progress_callback=progress_callback)

    monkeypatch.setattr(alphazero_loop, "generate_selfplay", counting_generate)
    result = run_loop(config, output, progress_interval_seconds=0.0)
    assert result["loop_complete"] is True
    assert collect_calls == [False, False]
    state = json.loads((output / "loop_state.json").read_text(encoding="utf-8"))
    assert state["completed_rounds"] == [0, 1]
    # Round 1 collected with the checkpoint produced by round 0.
    assert state["rounds"]["1"]["theta_in"]["sha256"] == state["rounds"]["0"]["final_checkpoint"]["sha256"]
    assert state["rounds"]["1"]["theta_in"]["sha256"] != state["initial_checkpoint"]["sha256"]
    # No validation games in this tiny loop: reported as unavailable, not faked.
    assert state["rounds"]["0"]["validation"]["available"] is False
    assert (output / "rounds" / "round-000" / "checkpoints" / "final.pt").is_file()
    assert (output / "rounds" / "round-001" / "checkpoints" / "final.pt").is_file()
    panel_path = output / "rounds" / "round-001" / "evaluation" / f"{panel_name}.json"
    panel_summary = json.loads(panel_path.read_text(encoding="utf-8"))
    expected_opponent = state["initial_checkpoint"] if mcts_opponent == "initial" else state["rounds"]["1"]["theta_in"]
    assert panel_summary["config"]["policy_b"]["checkpoint"] == expected_opponent["path"]
    assert panel_summary["config"]["metadata"][f"theta_{mcts_opponent}"] == expected_opponent["path"]
    assert set(state["rounds"]["1"]["evaluation"]) == {panel_name, "greedy_vs_initial"}

    # The learner continues across rounds: two steps per round, one global
    # counter, and AdamW moments that are four updates old at the end.
    assert state["rounds"]["0"]["final_checkpoint"]["total_train_step"] == 2
    assert state["rounds"]["1"]["final_checkpoint"]["total_train_step"] == 4
    assert state["rounds"]["1"]["theta_in"]["total_train_step"] == 2
    final = load_alphazero_checkpoint(state["rounds"]["1"]["final_checkpoint"]["path"])
    assert {int(value["step"]) for value in final.optimizer_state_dict["state"].values()} == {4}
    metrics = [
        json.loads(line)
        for line in (output / "rounds" / "round-001" / "training_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    step_metrics = [record for record in metrics if record["event"] == "train_step"]
    assert [record["round_train_step"] for record in step_metrics] == [1, 2]
    assert [record["total_train_step"] for record in step_metrics] == [3, 4]

    # Long phases report progress before the phase-complete line.
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    events = [record["event"] for record in records]
    completed = next(record for record in records if record["event"] == "evaluate_complete")
    assert panel_name in human_event(completed)
    for expected in (
        "collect_start", "collect_progress", "collect_complete",
        "train_progress", "train_complete",
        "evaluate_progress", "evaluate_complete", "round_complete",
    ):
        assert expected in events
    accumulator = EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {
        "train/total_loss", "collect/round-000/games_completed", "collect/games_per_second",
        f"eval/{panel_name}_score_rate",
    } <= tags

    resumed = run_loop(config, output, resume=True)
    assert resumed["loop_complete"] is True
    assert collect_calls == [False, False]


@pytest.mark.parametrize("both", [False, True])
def test_loop_requires_one_mcts_opponent(tmp_path, both):
    config = _loop_config(_weights_checkpoint(tmp_path / "base.pt"))
    with pytest.raises(ValueError, match="exactly one"):
        replace(config.evaluation,
                mcts_vs_parent=config.evaluation.mcts_vs_parent if both else None,
                mcts_vs_initial=config.evaluation.mcts_vs_parent if both else None)


def test_loop_resume_does_not_retrain_a_committed_checkpoint(tmp_path, monkeypatch, restore_torch_threads):
    checkpoint = _weights_checkpoint(tmp_path / "base.pt")
    config = _loop_config(checkpoint)
    output = tmp_path / "loop"
    run_loop(config, output)
    state = json.loads((output / "loop_state.json").read_text(encoding="utf-8"))
    # The state file lags behind: the round-1 final checkpoint is already
    # committed, but the state still says "training".
    state["completed_rounds"] = [0]
    state["latest_checkpoint"] = state["rounds"]["0"]["final_checkpoint"]
    state["rounds"]["1"] = {
        "phase": "train",
        "theta_in": state["rounds"]["1"]["theta_in"],
        "collect": {},
    }
    (output / "loop_state.json").write_text(json.dumps(state), encoding="utf-8")

    trainer_calls: list[int] = []
    collect_calls: list[bool] = []
    original_train = alphazero_loop.run_training_round
    original_generate = alphazero_loop.generate_selfplay

    def counting_train(*args, **kwargs):
        trainer_calls.append(1)
        return original_train(*args, **kwargs)

    def counting_generate(config_arg, root, *, resume=False, progress_callback=None):
        collect_calls.append(resume)
        return original_generate(
            config_arg, root, resume=resume, progress_callback=progress_callback
        )

    monkeypatch.setattr(alphazero_loop, "run_training_round", counting_train)
    monkeypatch.setattr(alphazero_loop, "generate_selfplay", counting_generate)
    result = run_loop(config, output, resume=True)
    assert trainer_calls == []
    assert collect_calls == []
    assert result["completed_rounds"] == [0, 1]
    final = json.loads((output / "loop_state.json").read_text(encoding="utf-8"))
    assert final["latest_checkpoint"]["path"].endswith("round-001/checkpoints/final.pt")


def test_loop_rejects_mismatched_configuration(tmp_path, restore_torch_threads):
    checkpoint = _weights_checkpoint(tmp_path / "base.pt")
    config = _loop_config(checkpoint)
    output = tmp_path / "loop"
    run_loop(config, output)
    changed = LoopConfig.from_dict(
        config.to_dict() | {"seeds": {"collection": 1, "split": 2, "sampling": 3}}
    )
    with pytest.raises(ValueError, match="different configuration"):
        run_loop(changed, output, resume=True)
    with pytest.raises(ValueError, match="pass --resume"):
        run_loop(config, output, resume=False)


def test_loop_native_batched_evaluation_is_connected_and_resumable(tmp_path, restore_torch_threads):
    from dataclasses import replace
    checkpoint = _weights_checkpoint(tmp_path / 'base.pt')
    config = _loop_config(checkpoint)
    config = replace(config, num_rounds=1, evaluation=replace(config.evaluation,
        mcts_backend='native', mcts_execution={'backend': 'native_batched',
            'active_games': 2, 'max_batch_size': 2, 'inference_mode': 'eager'}))
    output = tmp_path / 'native-loop'
    assert run_loop(config, output)['loop_complete']
    panel = output / 'rounds/round-000/evaluation/mcts_vs_parent.json'
    result = json.loads(panel.read_text())
    assert result['passed'] and len(result['execution_stats']['models']) == 2
    assert result['config']['execution'] == config.evaluation.mcts_execution
    assert run_loop(config, output, resume=True)['loop_complete']

def test_loop_stops_when_evaluation_panel_fails_engineering_checks(
    tmp_path, monkeypatch, restore_torch_threads
):
    checkpoint = _weights_checkpoint(tmp_path / "base.pt")
    config = _loop_config(checkpoint)
    output = tmp_path / "loop"
    generate = alphazero_loop.generate_selfplay
    collect_calls: list[bool] = []

    def counting_generate(config_arg, root, *, resume=False, progress_callback=None):
        collect_calls.append(resume)
        return generate(config_arg, root, resume=resume, progress_callback=progress_callback)

    panels: list[str] = []

    def failing_evaluation(config_arg, *args, **kwargs):
        panels.append(config_arg.policy_a.name)
        return SimpleNamespace(
            summary={
                "passed": False,
                "completed_pairs": 1,
                "total_pairs": 1,
                "engineering_truncations": 1,
                "exceptions": 0,
                "illegal_actions": 0,
                "invariant_failures": 0,
                "win_rate_score": None,
                "output": str(args[0]),
            }
        )

    monkeypatch.setattr(alphazero_loop, "generate_selfplay", counting_generate)
    monkeypatch.setattr(alphazero_loop, "run_fixed_evaluation", failing_evaluation)
    with pytest.raises(alphazero_loop.EvaluationEngineeringFailure) as error:
        run_loop(config, output)
    assert error.value.panel == "mcts_vs_parent"
    # The failing panel aborts before the second panel and never starts round 1.
    assert panels == ["theta_new_mcts"]
    assert collect_calls == [False]
    assert not (output / "rounds" / "round-001").exists()
    state = json.loads((output / "loop_state.json").read_text(encoding="utf-8"))
    assert state["completed_rounds"] == []
    scene = json.loads(
        (output / "failures" / "round-000-evaluate.json").read_text(encoding="utf-8")
    )
    assert scene["error_type"] == "EvaluationEngineeringFailure"
    assert scene["evaluation"]["panel"] == "mcts_vs_parent"
    assert scene["evaluation"]["details"]["engineering_truncations"] == 1


def test_recovery_without_state_file_finishes_unfinished_evaluation(
    tmp_path, monkeypatch, restore_torch_threads
):
    checkpoint = _weights_checkpoint(tmp_path / "base.pt")
    config = _loop_config(checkpoint)
    output = tmp_path / "loop"
    run_loop(config, output)
    greedy_summary = output / "rounds" / "round-001" / "evaluation" / "greedy_vs_initial.json"
    assert greedy_summary.is_file()
    greedy_summary.unlink()
    (output / "loop_state.json").unlink()

    trainer_calls: list[int] = []
    collect_calls: list[int] = []
    original_train = alphazero_loop.run_training_round
    original_generate = alphazero_loop.generate_selfplay

    def counting_train(*args, **kwargs):
        trainer_calls.append(1)
        return original_train(*args, **kwargs)

    def counting_generate(config_arg, root, *, resume=False, progress_callback=None):
        collect_calls.append(1)
        return original_generate(
            config_arg, root, resume=resume, progress_callback=progress_callback
        )

    monkeypatch.setattr(alphazero_loop, "run_training_round", counting_train)
    monkeypatch.setattr(alphazero_loop, "generate_selfplay", counting_generate)
    result = run_loop(config, output, resume=True)
    # Round 1 was not treated as complete just because its final checkpoint
    # existed: the lost evaluation summary is rebuilt from the pair log while
    # collection and training stay skipped.
    assert result["completed_rounds"] == [0, 1]
    assert greedy_summary.is_file()
    assert collect_calls == [] and trainer_calls == []
    state = json.loads((output / "loop_state.json").read_text(encoding="utf-8"))
    assert state["completed_rounds"] == [0, 1]
    assert state["rounds"]["1"]["session_phases"] == [
        "collect(skipped)", "train(skipped)", "evaluate"
    ]


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag, value, step) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def flush(self) -> None:
        pass


def test_runner_logs_validation_and_collection_scalars(tmp_path):
    config = _loop_config(_weights_checkpoint(tmp_path / "base.pt"))
    runner = alphazero_loop.LoopRunner(config, tmp_path / "loop")
    writer = _RecordingWriter()
    runner.tensorboard = writer
    runner._log_validation(
        {"total_train_step": 7},
        {
            "available": True, "positions": 12, "games": 1,
            "policy_loss": 1.5, "value_loss": 0.25, "policy_entropy": 2.0,
        },
    )
    runner._log_collection_metrics(
        {"total_train_step": 7},
        {
            "completed_games": 4, "samples": 10, "policy_samples": 8, "simulations": 100,
            "wall_seconds": 2.0, "games_per_wall_second": 2.0,
            "policy_samples_per_wall_second": 4.0, "simulations_per_wall_second": 50.0,
        },
    )
    assert ("validation/policy_ce", 1.5, 7) in writer.scalars
    assert ("validation/value_mse", 0.25, 7) in writer.scalars
    assert ("validation/games", 1.0, 7) in writer.scalars
    assert ("collect/games_per_second", 2.0, 7) in writer.scalars
    assert ("collect/simulations_per_second", 50.0, 7) in writer.scalars

    # A round without validation games reports nothing instead of a fake zero,
    # and a resumed collection session without measured wall time only logs
    # the counts it actually knows.
    writer.scalars.clear()
    runner._log_validation({"total_train_step": 7}, {"available": False, "positions": 0})
    runner._log_collection_metrics({"total_train_step": 7}, {"completed_games": 4})
    assert writer.scalars == [("collect/completed_games", 4.0, 7)]


def test_progress_throttle_limits_lines_without_losing_the_first():
    clock = [0.0]
    throttle = alphazero_loop.ProgressThrottle(10.0, clock=lambda: clock[0])
    assert throttle.ready() is True
    clock[0] = 4.0
    assert throttle.ready() is False
    clock[0] = 10.0
    assert throttle.ready() is True
