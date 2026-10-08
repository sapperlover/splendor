import copy
from types import SimpleNamespace

import numpy as np
import pytest

from splendor import iteration_suite as suite
from splendor.hybrid_train import HybridConfig


def formal_plan():
    return suite.read(suite.ROOT / 'configs/entity_attention_v1/iteration_trend_20261008.json')


def test_trend_design_runs_b_then_c_then_d_without_extra_panels():
    plan = formal_plan()
    suite.validate_plan(plan)
    assert [a['id'] for a in plan['arms']] == ['steps50', 'lr_low', 'fresh50']
    assert not any('disk' in key for key in plan['resources'])
    configs = suite.arm_configs(plan)
    assert len(configs) == 3
    assert [r["id"] for r in plan["replicates"]] == ["seed1"]
    assert configs['seed1_steps50']['updates_per_round'] == 50
    assert configs['seed1_lr_low']['updates_per_round'] == 100
    assert configs['seed1_lr_low']['learning_rate_override'] == 2.5e-5
    fresh, small = copy.deepcopy(configs['seed1_fresh50']), copy.deepcopy(configs['seed1_steps50'])
    fresh.pop('run_id'); small.pop('run_id')
    assert fresh['split_training'].pop('policy_recent_rounds') == 1
    assert fresh == small
    for raw in configs.values():
        assert raw['evaluation'] is None and raw['evaluation_every_rounds'] == 0
        assert HybridConfig.from_dict(raw).to_dict() == raw
    jobs = list(suite.evaluation_jobs(plan))
    assert len(jobs) == 15 and sum(j['pairs'] for j in jobs) == 1500
    assert {j['round'] for j in jobs} == {2, 4, 8, 12, 16}
    schedule = list(suite.stage_schedule(plan, jobs))
    for first, second in [('steps50', 'lr_low'), ('lr_low', 'fresh50')]:
        boundary = schedule.index(('train', f'seed1_{second}', 1))
        for rep in plan['replicates']:
            assert schedule.index(('train', f'{rep["id"]}_{first}', 16)) < boundary
            assert schedule.index(('evaluate', f'{rep["id"]}_{first}_r016', None)) < boundary
    manifest = {'plan': plan, 'configs': configs, 'evaluation_template': suite.evaluation_template(plan)}
    for rep in plan['replicates']:
        raw = [suite.evaluation_config(manifest, j) for j in jobs if j['replicate'] == rep['id']]
        assert len({r['seed_start'] for r in raw}) == 1
        assert len({r['policy_seed_base'] for r in raw}) == 1
        assert len({r['policy_a']['search_config']['seed'] for r in raw}) == 1
        assert all(r['policy_b']['checkpoint'] == str(suite.ROOT / plan['start_checkpoint']) for r in raw)
        assert all(not r['policy_b']['inference']['actor_projection'] for r in raw)


def test_bootstrap_keeps_training_replicates_fixed():
    # No game-seed uncertainty within either trained trajectory; varying mixture weights
    # in a pooled bootstrap would spuriously create an interval here.
    result = suite.stratified_interval([np.zeros(400), np.ones(400)], resamples=1000, seed=7)
    assert result['estimate'] == result['lower'] == result['upper'] == .5
    assert result['strata'] == 2 and result['pair_count'] == 800
    with pytest.raises(ValueError, match='nonempty finite'):
        suite.stratified_interval([np.array([np.nan])])


def test_worker_refuses_missing_hard_memory_bound(tmp_path, monkeypatch):
    cg = tmp_path / 'cg'
    cg.mkdir()
    (cg / 'memory.max').write_text('max')
    (cg / 'memory.swap.max').write_text('0')
    monkeypatch.setattr(suite, 'cgroup_path', lambda: cg)
    with pytest.raises(ValueError):
        suite.worker(SimpleNamespace(memory_max=16*suite.GIB))


def test_resource_guard_stops_only_its_owned_process(tmp_path, monkeypatch):
    import subprocess
    import sys
    plan = formal_plan()
    plan['artifact_root'] = str(tmp_path / 'artifacts')
    plan['runtime_root'] = str(tmp_path / 'runtime')
    plan['resources']['poll_seconds'] = .01
    real_popen = subprocess.Popen
    created, stops = [], []
    def popen(command, **kwargs):
        proc = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        created.append(proc)
        return proc
    def run(command, **kwargs):
        stops.append(command)
        return SimpleNamespace(returncode=0)
    readings = iter([10*suite.GIB, 10*suite.GIB, 0])
    monkeypatch.setattr(suite, 'available_memory', lambda: next(readings))
    monkeypatch.setattr(subprocess, 'Popen', popen)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='resource reserve'):
        suite.run_stage(plan, 'train', 'test', round_number=1)
    assert created[0].poll() is not None
    assert all(c[:3] == ['systemctl', '--user', 'stop'] and c[3].startswith('splendor-suite-') for c in stops)
    records = list((tmp_path / 'artifacts/resources').glob('*.json'))
    assert len(records) == 1 and suite.read(records[0])['status'] == 'failed'


def test_parent_killed_without_cleanup_interrupts_its_worker(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    child = tmp_path / 'child.py'
    ready, stopped = tmp_path / 'ready', tmp_path / 'stopped'
    child.write_text(
        'import os, time\nfrom pathlib import Path\n'
        'from splendor.iteration_suite import arm_parent_death\n'
        'arm_parent_death(os.getppid())\n'
        f'Path({str(ready)!r}).write_text(str(os.getpid()))\n'
        'try:\n    time.sleep(60)\n'
        f'except KeyboardInterrupt:\n    Path({str(stopped)!r}).write_text("interrupted")\n')
    parent = subprocess.Popen([sys.executable, '-c',
        f'import subprocess,sys,time; subprocess.Popen([sys.executable,{str(child)!r}]); time.sleep(60)'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not stopped.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert stopped.read_text() == 'interrupted'
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
