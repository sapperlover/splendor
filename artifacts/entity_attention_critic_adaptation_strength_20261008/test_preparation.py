"""Preparation checks only: never launch a tree search or a game."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from splendor.alphazero_checkpoint import model_state_sha256
from splendor.engine import GameEngine

PATH = Path(__file__).with_name('battle.py')
SPEC = importlib.util.spec_from_file_location('critic_adaptation_battle', PATH)
battle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = battle
SPEC.loader.exec_module(battle)


def test_public_phase_boundary_and_opponent_route():
    request = GameEngine.new(42).decision()
    for own, other, phase in ((9, 9, 'early'), (10, 0, 'late'), (0, 10, 'late')):
        o = replace(request.observation, self_public=replace(request.observation.self_public, score=own),
                    opponent_public=replace(request.observation.opponent_public, score=other))
        assert battle.phase_for(replace(request, observation=o)) == phase
    for mode in ('full', 'early', 'late'):
        for phase in ('early', 'late'):
            assert not battle.use_adapted(mode, False, phase)
            assert battle.use_adapted(mode, True, phase) == (mode=='full' or mode==phase)


def test_mixed_root_jobs_route_to_correct_frozen_evaluator_without_search(monkeypatch):
    import splendor.evaluation_batch as batch
    calls = []
    class FakeRunner:
        def __init__(self, evaluator, catalog, **kwargs):
            self.evaluator = evaluator
            self.batch_records = []
            self.rows = self.batches = self.wide_batches = self.root_rows = self.policy_rows = self.value_rows = 0
            self.fill_seconds = self.consume_seconds = 0.
        def run(self, jobs, *, context=None):
            calls.append((self.evaluator.label, tuple(job.label for job in jobs)))
            self.rows += len(jobs)
            self.batches += 1
            self.batch_records = [{'rows': len(jobs)}]
            return len(jobs)
        def failure_report(self, error):
            return {'evaluator': self.evaluator.label}
    base_head, adapted_head = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    with torch.no_grad():
        base_head.weight.fill_(0.)
        adapted_head.weight.fill_(1.)
    base = SimpleNamespace(model=SimpleNamespace(critic=base_head), label='base')
    adapted = SimpleNamespace(model=SimpleNamespace(critic=adapted_head), label='adapted')
    monkeypatch.setattr(batch, 'NativeBatchRunner', FakeRunner)
    with battle.install_routing('early', 'candidate', model_state_sha256(base_head.state_dict())):
        candidate_runner = batch.NativeBatchRunner(adapted, None, capacity=2, buckets=(1,))
        batch.NativeBatchRunner(base, None, capacity=2, buckets=(1,))
        jobs = [SimpleNamespace(label='early-root', use_adapted=True),
                SimpleNamespace(label='late-root', use_adapted=False)]
        assert candidate_runner.run(jobs) == 2
        assert calls == [('adapted', ('early-root',)), ('base', ('late-root',))]
        assert candidate_runner.rows == 2 and candidate_runner.batches == 2
        assert len(candidate_runner.batch_records) == 2
        assert candidate_runner.failure_report(RuntimeError()) == {'evaluator': 'base'}


def test_frozen_schedule_separates_dev_head_and_shares_stage_seeds():
    plan = battle.read_json(PATH.with_name('battle_plan.json'))
    dev = battle.make_config(plan, 50, 'dev', 'full')
    full = battle.make_config(plan, 50, 'head', 'full')
    early = battle.make_config(plan, 50, 'head', 'early')
    late = battle.make_config(plan, 50, 'head', 'late')
    assert len(dev.game_seeds)==100 and len(full.game_seeds)==2000
    assert len(early.game_seeds)==len(late.game_seeds)==1000
    assert not set(dev.game_seeds)&set(full.game_seeds)
    assert early.game_seeds==late.game_seeds==full.game_seeds[:1000]
    assert early.policy_a.checkpoint==full.policy_a.checkpoint==late.policy_a.checkpoint
    assert early.policy_b.checkpoint==full.policy_b.checkpoint==late.policy_b.checkpoint


def test_search_requires_explicit_flag(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [str(PATH)])
    with pytest.raises(SystemExit) as error:
        battle.main()
    assert error.value.code==2
