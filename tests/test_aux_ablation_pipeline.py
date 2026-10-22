"""The new ablation must retain the old budget and intended strength populations."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'artifacts/entity_attention_r24_r43_aux_ablation_20261019/run.py'
spec = importlib.util.spec_from_file_location('aux_ablation_pipeline', PATH)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_only_auxiliary_objective_changes():
    from splendor.offline_objectives import OfflineObjectiveConfig
    original=json.loads(pipeline.BASE_CONFIG.read_text())
    for name,path in pipeline.CONFIGS.items():
        raw=json.loads(path.read_text());OfflineObjectiveConfig.from_dict(raw)
        assert {k:v for k,v in raw.items() if k!='value_auxiliary'}=={k:v for k,v in original.items() if k!='value_auxiliary'}
        assert raw['value_auxiliary'] is None if name=='plain' else raw['value_auxiliary']=={**original['value_auxiliary'],'score_loss_weight':0.0}


@pytest.mark.parametrize('kind,pairs,seed',[('greedy',1000,36200000),('search',200,39100000)])
def test_final_checkpoint_and_common_evaluation_domains(tmp_path, monkeypatch,kind,pairs,seed):
    monkeypatch.setattr(pipeline,'RUNS',tmp_path)
    plan={'initial_checkpoint':{'path':json.loads(pipeline.BASE_CONFIG.read_text())['initial_checkpoint']},
          'selection':'final22681','scope':'test',
          'evaluation':{'greedy':{'pairs':1000,'seed_start':36200000,'policy_seed_base':36210000},
                        'search':{'pairs':200,'seed_start':39100000,'policy_seed_base':39110000}}}
    # Configuration-only validation; these dummy bytes are never loaded as a model.
    configs=[]
    for branch in pipeline.CONFIGS:
        p=tmp_path/branch/'checkpoints/final.pt';p.parent.mkdir(parents=True);p.write_bytes(b'config-only fixture')
        monkeypatch.setattr(pipeline,'identity',lambda path:{'path':str(path),'sha256':pipeline.sha256(path)})
        raw=pipeline.evaluation_config(plan,branch,kind);configs.append(raw)
        assert raw['game_seeds']==list(range(seed,seed+pairs))
        assert raw['policy_a']['checkpoint'].endswith('checkpoints/final.pt')
        assert raw['policy_b']['checkpoint'].endswith('round-033/checkpoints/final.pt')
        if kind=='search':
            for side in ['policy_a','policy_b']:
                assert raw[side]['search_config']['simulations']==1024
                assert raw[side]['search_config']['determinizations']==1
                assert raw[side]['search_config']['root_noise_fraction']==0
        else: assert raw['device']=='cpu' and raw['workers']==8
    assert configs[0]['game_seeds']==configs[1]['game_seeds']
