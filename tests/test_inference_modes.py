"""Frozen inference execution modes: projection reuse and compiled forwards."""

import numpy as np
import torch

from splendor.actor_critic import ActorCritic, ActorCriticConfig
from splendor.benchmark_search import fixed_requests
from splendor.features import encode_decision
from splendor.search_inference import FrozenSearchInference, InferenceConfig


def _requests():
    return [encode_decision(request) for _, request, _ in fixed_requests()]


def _model() -> ActorCritic:
    torch.manual_seed(20260929)
    model = ActorCritic(ActorCriticConfig(state_hidden=32, actor_hidden=16, critic_hidden=16,
                                          card_hidden=8, noble_hidden=4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    return model.eval().requires_grad_(False)


def _evaluator(model: ActorCritic, **config):
    return FrozenSearchInference(model, device="cpu", config=InferenceConfig(**config))


def test_actor_projection_matches_eager_logits_and_values():
    encoded = _requests()
    model = _model()
    reference = _evaluator(model, mode="eager", actor_projection=False)
    projected = _evaluator(model, mode="eager", actor_projection=True)
    eager_rows = reference.policy_value_encoded(encoded)
    projected_rows = projected.policy_value_encoded(encoded)
    for (eager_logits, eager_value), (rows, value) in zip(eager_rows, projected_rows):
        np.testing.assert_allclose(rows, eager_logits, rtol=0, atol=1e-5)
        assert abs(value - eager_value) <= 1e-6
        assert max(range(len(rows)), key=rows.__getitem__) == max(range(len(eager_logits)), key=eager_logits.__getitem__)


def test_projection_keeps_the_critic_bit_identical():
    encoded = _requests()
    model = _model()
    reference = _evaluator(model, mode="eager", actor_projection=False)
    projected = _evaluator(model, mode="eager", actor_projection=True)
    assert reference.policy_value_encoded(encoded)[0][1] == projected.policy_value_encoded(encoded)[0][1]


def test_compiled_forward_matches_eager_with_bucket_padding():
    encoded = _requests()
    model = _model()
    reference = _evaluator(model, mode="eager", actor_projection=False)
    compiled = _evaluator(model, mode="compiled", actor_projection=True, candidate_buckets=(16, 32, 48))
    for item in encoded:
        eager = reference.policy_value_encoded((item,))[0]
        actual = compiled.policy_value_encoded((item,))[0]
        np.testing.assert_allclose(actual[0], eager[0], rtol=0, atol=1e-5)
        assert abs(actual[1] - eager[1]) <= 1e-6
    stats = compiled.stats.to_dict()["compile"]
    assert set(stats["compiled_calls"]) == {"policy_value:16", "policy_value:32"}
    assert stats["eager_fallbacks"] == 0


def test_widths_beyond_the_last_bucket_fall_back_to_eager():
    encoded = _requests()
    model = _model()
    reference = _evaluator(model, mode="eager", actor_projection=True)
    compiled = _evaluator(model, mode="compiled", actor_projection=True, candidate_buckets=(8,))
    for item in encoded:
        eager = reference.policy_value_encoded((item,))[0]
        actual = compiled.policy_value_encoded((item,))[0]
        np.testing.assert_allclose(actual[0], eager[0], rtol=0, atol=1e-5)
    assert compiled.stats.to_dict()["compile"]["eager_fallbacks"] == len(encoded)


def test_inference_config_rejects_unknown_modes():
    import pytest

    with pytest.raises(ValueError):
        InferenceConfig(mode="turbo")
    with pytest.raises(ValueError):
        InferenceConfig(candidate_buckets=(32, 16))


def _broken_compile(stage: str, counter: dict[str, int]):
    """A ``torch.compile`` stand-in that fails like a real backend failure."""

    def compile_stub(function, **kwargs):
        counter["calls"] = counter.get("calls", 0) + 1
        if stage == "construct":
            raise torch._dynamo.exc.Unsupported("injected compile failure")
        if stage == "execute":
            def graph(*tensors):
                counter["graph"] = counter.get("graph", 0) + 1
                raise torch._inductor.exc.InductorError(ValueError("injected inductor failure"), None)
            return graph
        raise AssertionError(stage)

    return compile_stub


def test_compile_construction_failure_falls_back_to_eager(monkeypatch):
    model = _model()
    reference = _evaluator(model, mode="eager", actor_projection=False)
    monkeypatch.setattr(torch, "compile", _broken_compile("construct", {}))
    compiled = _evaluator(model, mode="compiled", actor_projection=True, candidate_buckets=(16, 32))
    for item in _requests():
        eager = reference.policy_value_encoded((item,))[0]
        actual = compiled.policy_value_encoded((item,))[0]
        np.testing.assert_allclose(actual[0], eager[0], rtol=0, atol=1e-5)
        assert abs(actual[1] - eager[1]) <= 1e-6
    stats = compiled.stats.to_dict()["compile"]
    assert stats["compiled_calls"] == {}
    assert stats["compile_failures"] and all(item["stage"] == "construct" for item in stats["compile_failures"])
    assert stats["compile_failures"][0]["error_type"] == "Unsupported"
    assert set(stats["effective_mode"].values()) == {"eager"}
    assert stats["compile_failure_fallbacks"] == len(_requests())


def test_compile_execute_failure_disables_the_graph_without_retry(monkeypatch):
    model = _model()
    reference = _evaluator(model, mode="eager", actor_projection=False)
    counter: dict[str, int] = {}
    monkeypatch.setattr(torch, "compile", _broken_compile("execute", counter))
    compiled = _evaluator(model, mode="compiled", actor_projection=True, candidate_buckets=(16, 32))
    first = compiled.policy_value_encoded(_requests())
    for (eager_logits, eager_value), (logits, value) in zip(reference.policy_value_encoded(_requests()), first):
        np.testing.assert_allclose(logits, eager_logits, rtol=0, atol=1e-5)
        assert abs(value - eager_value) <= 1e-6
    attempts = counter["graph"]
    assert attempts >= 1
    compiled.policy_value_encoded(_requests())
    assert counter["graph"] == attempts, "a disabled graph must not be retried"
    stats = compiled.stats.to_dict()["compile"]
    assert all(item["stage"] == "execute" for item in stats["compile_failures"])
    assert stats["compiled_calls"] == {}
    assert len(stats["disabled_graphs"]) == attempts
    assert set(stats["effective_mode"].values()) == {"eager"}


def test_compile_fail_fast_reraises_the_backend_failure(monkeypatch):
    import pytest

    model = _model()
    monkeypatch.setattr(torch, "compile", _broken_compile("construct", {}))
    compiled = _evaluator(model, mode="compiled", actor_projection=True,
                          compile_failure_policy="fail_fast")
    with pytest.raises(torch._dynamo.exc.Unsupported):
        compiled.policy_value_encoded(_requests()[:1])
    assert compiled.stats.to_dict()["compile"]["compile_failures"][0]["stage"] == "construct"


def test_compile_does_not_swallow_unrelated_runtime_errors(monkeypatch):
    import pytest

    model = _model()

    def oom_compile(function, **kwargs):
        raise torch.cuda.OutOfMemoryError("injected OOM")

    monkeypatch.setattr(torch, "compile", oom_compile)
    compiled = _evaluator(model, mode="compiled", actor_projection=True)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        compiled.policy_value_encoded(_requests()[:1])
    assert compiled.stats.to_dict()["compile"]["compile_failures"] == []


def test_inference_config_rejects_unknown_compile_failure_policy():
    import pytest

    with pytest.raises(ValueError):
        InferenceConfig(compile_failure_policy="ignore")
