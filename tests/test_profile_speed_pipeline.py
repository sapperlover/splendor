"""Diagnostics must preserve method descriptors and work without cProfile."""
import cProfile
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "profile_speed_pipeline", Path(__file__).resolve().parents[1] / "scripts/profile_speed_pipeline.py"
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_disabled_function_profiler_has_no_rows():
    # pstats rejects an empty Profile; light timing must still produce a result.
    assert profile.profile_rows(cProfile.Profile()) == []


def test_timing_wrapper_restores_classmethod_and_inherited_instance_method():
    class Factory:
        @classmethod
        def create(cls):
            return cls()

        def value(self):
            return 7

    descriptor = Factory.__dict__["create"]
    timer = profile.Timings()
    with profile.wrapped(Factory, "create", timer, "create"):
        instance = Factory.create()
    assert Factory.__dict__["create"] is descriptor
    with profile.wrapped(instance, "value", timer, "value"):
        assert instance.value() == 7
    assert "value" not in vars(instance)
    assert instance.value() == 7
    assert timer.calls == {"create": 1, "value": 1}
