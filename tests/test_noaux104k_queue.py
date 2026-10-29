import importlib.util
from pathlib import Path

from splendor.alphazero_recording import read_json
from splendor.offline_early_stopping import EarlyStoppingOfflineConfig

ROOT = Path(__file__).resolve().parents[1]
NAME = "entity_attention_128x2_noaux_r21_r37_20261026"


def load(path):
    spec = importlib.util.spec_from_file_location("noaux_queue_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_noaux_formal_config_changes_only_auxiliary_settings_and_keeps_r37_panels():
    base = read_json(ROOT / "configs/entity_attention_v1/offline_policy_score_r21_r37_20261026.json")
    base.pop("continuation_auxiliary")
    base.pop("encoder_grad_norm_every_steps")
    raw = read_json(ROOT / "configs/entity_attention_v1/offline_noaux_r21_r37_20261026.json")
    assert raw == base
    cfg = EarlyStoppingOfflineConfig.from_dict(raw)
    assert cfg.value_auxiliary is None and cfg.training_workers == 4
    plan = read_json(ROOT / "artifacts" / NAME / "plan.json")
    reference = read_json(ROOT / "artifacts/entity_attention_128x2_policy_score_r21_r37_20261026/plan.json")
    for field in ("dataset_digest", "fixed_validation_digest", "splits", "steps_target", "validation_steps",
                  "sampling", "opponents", "panels"):
        assert plan[field] == reference[field]
    assert plan["parameters"] == 583682 and plan["continuation_auxiliary"] is None


def test_noaux_queue_waits_for_the_policy_score_chain_and_has_only_the_requested_successor():
    wrapper = load("artifacts/entity_attention_noaux104k_queue_20261026/serial_queue.py")
    queue = wrapper.configured_queue()
    assert queue.CURRENT == "entity_attention_128x2_policy_score_r21_r37_20261026"
    assert queue.JOBS == ((NAME, "scripts/run_attention128_noaux104k_20261026.sh"),)
    assert queue.RUNTIME.name == "entity_attention_noaux104k_queue_20261026"


def test_actual_noaux_supervisor_routes_only_train_and_select_to_the_base_fitter(monkeypatch):
    wrapper = load("artifacts/" + NAME + "/run.py")
    original_loader = wrapper.importlib.util.module_from_spec
    calls = []

    def fake_loader(spec):
        module = original_loader(spec)
        real_exec = spec.loader.exec_module
        def exec_with_recorder(value):
            real_exec(value)
            value.run_child = lambda command, phase, source: calls.append((command, phase, source))
        monkeypatch.setattr(spec.loader, "exec_module", exec_with_recorder)
        return module

    monkeypatch.setattr(wrapper.importlib.util, "module_from_spec", fake_loader)
    entry = wrapper.configured_entry()
    for phase, module in (("train", "splendor.offline_continuation"), ("select", "splendor.offline_continuation"),
                          ("greedy_vs_R37", "splendor.evaluation")):
        entry.run_child(["python", "-u", "-m", module], phase, "frozen-source")
    assert [command[3] for command, _, _ in calls] == ["splendor.offline_early_stopping",
        "splendor.offline_early_stopping", "splendor.evaluation"]
