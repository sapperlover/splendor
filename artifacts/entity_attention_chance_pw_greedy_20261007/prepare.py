"""Freeze the same-model CPW versus greedy panel; never starts games."""
from pathlib import Path
import shutil

from splendor.alphazero_recording import read_json, write_json, sha256, source_identity
from splendor.evaluation import FixedEvaluationConfig

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime/entity_attention_chance_pw_greedy_20261007'
PREVIOUS = ROOT / 'artifacts/entity_attention_chance_pw_strength_20261007'


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


if __name__ == '__main__':
    assert not (ART / 'manifest.json').exists()
    previous = read_json(PREVIOUS / 'manifest.json')
    config = read_json(PREVIOUS / 'evaluation_config.json')
    config['policy_b'] = {'kind': 'greedy_checkpoint', 'name': 'greedy_16500_same_weights',
                          'checkpoint': config['policy_a']['checkpoint']}
    config['metadata'].update(purpose='CPW k0.25/alpha0.5 S1024 versus same-weight zero-search greedy.',
                              greedy_inference='existing GreedyActorCriticPolicy, eager single-request logits argmax')
    parsed = FixedEvaluationConfig.from_dict(config)
    assert parsed.policy_a.checkpoint == parsed.policy_b.checkpoint
    write_json(ART / 'evaluation_config.json', config)
    smoke = read_json(ART / 'evaluation_config.json')
    smoke['pairs'] = 2
    smoke['policy_a']['search_config']['simulations'] = 16
    smoke['execution'].update(active_games=4, max_batch_size=4)
    smoke['metadata']['purpose'] = 'Four-game mixed CPW/greedy CUDA checked smoke; excluded from strength.'
    write_json(ART / 'smoke_config.json', smoke)
    source = source_identity(ROOT, RUNTIME / 'source')
    extras = [ROOT / 'src/splendor/native_search.cpp', ROOT / 'setup.py',
              *sorted((ROOT / 'data').glob('**/*')),
              *[ROOT / 'tests' / name for name in ('test_speed_optimization.py', 'test_evaluation_3a.py', 'test_native_chance_search.py')]]
    source['additional_files'] = {}
    for path in extras:
        if path.is_file():
            relative = path.relative_to(ROOT)
            source['additional_files'][str(relative)] = sha256(path)
            target = RUNTIME / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    assert source['native_extensions'] == previous['source']['native_extensions']
    native_snapshots = []
    for path in ROOT.glob('src/splendor/_native_search*.so'):
        target = RUNTIME / 'source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        native_snapshots.append(record(target))
    manifest = {'protocol': 'attention-chance-pw-greedy-v1', 'status': 'prepared',
                'client_date': '2026-10-07', 'checkpoint': previous['checkpoint'],
                'chance_widening': {'k': .25, 'alpha': .5}, 'source': source,
                'native_snapshots': native_snapshots,
                'inputs': [record(ART / name) for name in ('evaluation_config.json', 'smoke_config.json', 'run.py', 'analyze.py', 'prepare.py', 'tests.json')]
                          + [record(PREVIOUS / name) for name in ('evaluation_config.json', 'manifest.json')],
                'outputs': [],
                'scope': 'Same16500-step weights and100 game-seed pairs as prior CPW panel; CPW S1024 versus exact eager logits argmax greedy, zero greedy simulations. Mixed native_batched/greedy evaluator support added and checked against serial. No model update or teacher switch.',
                'commands': {}}
    prefix = 'conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u '
    path = str(ART.relative_to(ROOT))
    manifest['commands'] = {
        'prepare': prefix + path + '/prepare.py',
        'smoke': prefix + path + '/run.py --config ' + path + '/smoke_config.json --output ' + path + '/smoke.json',
        'run': prefix + path + '/run.py --config ' + path + '/evaluation_config.json --output ' + path + '/evaluation.json',
        'analyze': prefix + path + '/analyze.py --write',
        'verify': prefix + path + '/analyze.py',
    }
    manifest['commands']['resume'] = manifest['commands']['run'] + ' --resume'
    write_json(ART / 'manifest.json', manifest)
    print({'prepared': True, 'pairs': len(parsed.game_seeds), 'greedy_search': False})
