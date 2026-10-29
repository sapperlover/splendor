from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_alphazero_training import az_runs
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_replay import build_replay_index
from splendor.alphazero_train import LearnerConfig
from splendor.offline_scratch import (ScratchOfflineConfig, initialize_learner, prepare_inputs,
                                      run, select_checkpoint, validation_schedule)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def scratch(tmp_path, az_runs, monkeypatch):
    import splendor.offline_objectives as entry
    monkeypatch.setattr(entry, "source_identity", lambda *_: {"test": True})
    _, first, second = az_runs
    index = build_replay_index([first, second], max_games=6, cache_bytes=64*1024**2)
    records = []
    for game in index.games:
        stat = game.path.stat()
        records.append({**asdict(game), "path": str(game.path), "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "archive_sha256": sha256(game.path)})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in records))
    n = sum(g.decision_count for g in index.games if g.round_index == 1 and g.split == "train")
    config = ScratchOfflineConfig(manifest=str(manifest), manifest_sha256=sha256(manifest),
        model_config={"architecture":"entity_attention_v1","hidden":16,"heads":2,"layers":1,"ffn_hidden":32},
        first_round=2, last_round=2, expected_games_per_round=3, draw_exposures=20.75*4/n,
        cache_mib=64, training_workers=0, validation_positions=4, validation_percent_step=25,
        learner=LearnerConfig(device="cpu",batch_size=4,history_batch_size=2,validation_batch_size=8,
                              learning_rate=2e-5,checkpoint_every_steps=4))
    return config


def test_formal_configuration_and_percentage_schedule():
    config = ScratchOfflineConfig.from_dict(read_json(ROOT / "configs/entity_attention_v1/offline_256x4_aux_r24_r43_20261014.json"))
    previous = read_json(ROOT / "configs/entity_attention_v1/offline_r24_r43_value_auxiliary_20261013.json")
    assert asdict(config.learner) == previous["learner"]
    assert config.value_auxiliary == previous["value_auxiliary"]
    assert config.policy_surprise_weighting is None
    assert config.model_config == read_json(ROOT / "configs/entity_attention_v1/model_256x4.json")
    steps = validation_schedule(config, 68041)
    assert len(steps) == 20 and steps[:2] == [3403,6805] and steps[-1] == 68041
    assert 1000 not in steps
    with pytest.raises(ValueError, match="surprise"):
        replace(config, policy_surprise_weighting={"fraction":.5})
    with pytest.raises(ValueError, match="divisor"):
        replace(config, validation_percent_step=3)


@pytest.mark.parametrize('auxiliary', ['dual', 'none'])
def test_percentage_checkpoints_full_holdout_and_exact_resume(scratch, tmp_path, auxiliary):
    if auxiliary == 'none':
        scratch = replace(scratch, value_auxiliary=None,
                          model_config={**scratch.model_config, 'market_relations': True,
                                        'value_readout': 'global_players_market'})
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    _, _, _, _, plan = prepare_inputs(scratch, full)
    assert plan["steps_target"] == 21 and plan["validation_steps"] == [6,11,16,21]
    run(scratch, full)
    run(scratch, resumed, max_steps_this_session=7)
    partial = [json.loads(x)["step"] for x in (resumed / "validation_curve.jsonl").read_text().splitlines()]
    assert partial == [0,6]  # Saving for recovery is not an extra validation.
    run(scratch, resumed)
    a = load_alphazero_checkpoint(full / "checkpoints/final.pt")
    b = load_alphazero_checkpoint(resumed / "checkpoints/final.pt")
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value,b.model_state_dict[key],rtol=0,atol=0)
    for key, state in a.optimizer_state_dict["state"].items():
        for name,value in state.items():
            torch.testing.assert_close(value,b.optimizer_state_dict["state"][key][name],rtol=0,atol=0)
    assert a.sampling_rng_state == b.sampling_rng_state
    curve = [json.loads(x)["step"] for x in (resumed / "validation_curve.jsonl").read_text().splitlines()]
    assert curve == [0,6,11,16,21]
    assert (resumed / "checkpoints/step-000000008.pt").exists()
    assert not (resumed / "full_validation/step-000000008.json").exists()
    for step in plan["validation_steps"]:
        point = read_json(resumed / "full_validation" / f"step-{step:09d}.json")
        assert point["metrics"]["positions"] == plan["splits"]["validation"]["positions"]
        assert point["checkpoint_sha256"] == sha256(resumed / "checkpoints" / f"step-{step:09d}.pt")
        if auxiliary == 'none':
            assert 'score_loss' not in point['metrics'] and 'remaining_turns_loss' not in point['metrics']
            assert point['metrics']['total_loss'] == pytest.approx(
                point['metrics']['policy_loss'] + point['metrics']['value_loss'])
    selected = select_checkpoint(scratch,resumed)
    assert selected["selected_step"] in plan["validation_steps"]
    with pytest.raises(ValueError,match="identical frozen"):
        prepare_inputs(replace(scratch,initialization_seed=9),resumed)


def test_primary_selection_ignores_auxiliary_loss_and_ties_earlier(scratch,tmp_path):
    run(scratch,tmp_path / "fit")
    for step,kl,mse,total in ((6,.8,.2,-100),(11,.3,.5,100),(16,.2,.6,0),(21,.8,.5,-200)):
        path = tmp_path / "fit/full_validation" / f"step-{step:09d}.json"
        point = read_json(path)
        point["metrics"].update(policy_kl=kl,value_loss=mse,total_loss=total)
        write_json(path,point)
    selected=select_checkpoint(scratch,tmp_path / "fit")
    assert selected["selected_step"] == 11
    path=tmp_path / "fit/full_validation/step-000000011.json"
    point=read_json(path)
    point["checkpoint_sha256"]="changed"
    write_json(path,point)
    with pytest.raises(ValueError,match="identity"):
        select_checkpoint(scratch,tmp_path / "fit")


def test_interruption_preserves_last_committed_progress_without_extra_validation(scratch,tmp_path,monkeypatch):
    import splendor.offline_objectives as entry
    original=entry.run_training_round
    def interrupted(*args,**kwargs):
        kwargs["max_steps_this_session"]=7
        original(*args,**kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(entry,"run_training_round",interrupted)
    output=tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run(scratch,output)
    progress=read_json(output / "progress.json")
    assert progress["status"] == "incomplete" and progress["step"] == 7
    assert Path(progress["checkpoint"]).exists()
    assert [json.loads(x)["step"] for x in (output / "validation_curve.jsonl").read_text().splitlines()] == [0,6]


def test_random_large_initialization_has_fresh_optimizer_and_correct_auxiliary_size():
    config=ScratchOfflineConfig.from_dict(read_json(ROOT / "configs/entity_attention_v1/offline_256x4_aux_r24_r43_20261014.json"))
    config=replace(config,learner=replace(config.learner,device="cpu"))
    learner=initialize_learner(config,{"digest":"test"},10)
    assert learner.model.trainable_parameter_count() == 3761156
    assert learner.model.config.hidden == 256 and len(learner.model.blocks) == 4
    assert learner.model.config.heads == 8 and learner.model.config.ffn_hidden == 1024
    assert not learner.optimizer.state and learner.total_train_step == 0
    assert learner.parent_checkpoint["kind"] == "random_initialization"


def test_new_small_no_aux_config_changes_only_requested_settings():
    old = read_json(ROOT / 'configs/entity_attention_v1/offline_256x4_aux_r24_r43_20261014.json')
    raw = read_json(ROOT / 'configs/entity_attention_v1/offline_128x2_relations_no_aux_r24_r43_20261019.json')
    expected = json.loads(json.dumps(old))
    expected['model_config'] = read_json(ROOT / 'configs/entity_attention_v1/model_relations_128x2.json')
    expected['learner']['learning_rate'] = 1e-4
    expected['value_auxiliary'] = None
    assert raw == expected
    config = ScratchOfflineConfig.from_dict(raw)
    assert config.value_auxiliary is None and config.policy_surprise_weighting is None
    assert config.draw_exposures == 12 and config.validation_percent_step == 5
    assert validation_schedule(config, 68041) == validation_schedule(ScratchOfflineConfig.from_dict(old), 68041)


def test_random_small_plain_initialization_has_no_auxiliary_parameters_or_optimizer_state():
    from splendor.actor_critic import build_actor_critic
    from splendor.entity_attention import EntityAttention

    config = ScratchOfflineConfig.from_dict(read_json(
        ROOT / 'configs/entity_attention_v1/offline_128x2_relations_no_aux_r24_r43_20261019.json'))
    config = replace(config, learner=replace(config.learner, device='cpu'))
    learner = initialize_learner(config, {'digest': 'test'}, 10)
    rng = torch.get_rng_state()
    assert type(learner.model) is EntityAttention
    assert not hasattr(learner.model, 'score_head') and not hasattr(learner.model, 'remaining_turns_head')
    assert learner.model.trainable_parameter_count() == 583682
    assert not learner.optimizer.state and learner.total_train_step == 0
    assert learner.config.learning_rate == 1e-4
    assert learner.parent_checkpoint['auxiliary_seed'] is None
    torch.manual_seed(config.initialization_seed)
    expected = build_actor_critic(config.model_config)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, learner.model.state_dict()[name], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_new_plain_pipeline_report_does_not_require_auxiliary_metrics(tmp_path, monkeypatch):
    path = ROOT / 'artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/run.py'
    spec = importlib.util.spec_from_file_location('plain_scratch_report', path)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    output, report = tmp_path / 'fit', tmp_path / 'report'
    report.mkdir()
    write_json(output / 'progress.json', {'status': 'incomplete', 'step': 6})
    write_json(output / 'full_validation/step-000000006.json', {
        'step': 6, 'metrics': {'policy_kl': .2, 'value_loss': .7}})
    monkeypatch.setattr(pipeline, 'OUTPUT', output)
    monkeypatch.setattr(pipeline, 'ART', report)
    plan = {'steps_target': 21, 'panels': [{'id': 'greedy', 'kind': 'greedy', 'pairs': 1000}]}
    result = pipeline.report(plan)
    assert result['training']['step'] == 6
    text = (report / 'results.md').read_text()
    assert '| 6 | 0.200000 | 0.700000 | 0.900000 |' in text
    assert '分差MAE' not in text and '剩余回合MAE' not in text


@pytest.mark.parametrize('experiment', ['entity_attention_256x4_aux_r24_r43_20261014',
                                       'entity_attention_128x2_relations_no_aux_r24_r43_20261019'])
def test_pipeline_evaluation_configs_use_selected_model_and_equal_search_budgets(scratch,tmp_path,az_runs,monkeypatch,experiment):
    path=ROOT / 'artifacts' / experiment / 'run.py'
    spec=importlib.util.spec_from_file_location("scratch_pipeline",path)
    pipeline=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    _, _, _, _, plan=prepare_inputs(scratch,tmp_path / "fit")
    learner=initialize_learner(scratch,plan["replay_identity"],plan["steps_target"])
    checkpoint=tmp_path / "candidate.pt"
    learner.save(checkpoint)
    opponent=az_runs[0]
    monkeypatch.setattr(pipeline,"OPPONENT",opponent)
    selection={"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),"selected_step":11}
    identity={"opponent":{"sha256":sha256(opponent)},"selection":"test KL+MSE"}
    for panel in ({"id":"greedy","kind":"greedy","pairs":1000,"seed_start":36600000},
                  {"id":"search","kind":"search","pairs":200,"seed_start":36700000}):
        raw=pipeline.evaluation_config(identity,selection,panel)
        assert len(raw["game_seeds"]) == panel["pairs"]
        assert raw["policy_a"]["checkpoint"] == str(checkpoint)
        assert raw["policy_b"]["checkpoint"] == str(opponent)
        if panel["kind"] == "greedy":
            assert raw["device"] == "cpu" and raw["workers"] == 8 and "execution" not in raw
            assert raw["policy_a"]["kind"] == raw["policy_b"]["kind"] == "greedy_checkpoint"
        else:
            assert raw["device"] == "cuda" and raw["execution"]["backend"] == "native_batched"
            for side in ("policy_a","policy_b"):
                assert raw[side]["search_config"]["simulations"] == 1024
                assert raw[side]["search_config"]["determinizations"] == 1
                assert raw[side]["search_config"]["root_noise_fraction"] == 0
