import json
from pathlib import Path
import time

import numpy as np
import torch

from splendor.compact_trace import decode_request
from splendor.recommend import LocalAdvisor, decision_identity
from splendor.replay import to_primitive

repo=Path('/home/lc/下载/splendor')
out=Path(__file__).resolve().parent
out.mkdir(exist_ok=True)
torch.set_num_threads(1)
data=repo/'artifacts/bga_927500015_20261020/model_decisions.jsonl'
rows=[json.loads(line) for line in data.read_text().splitlines()]
golden=json.loads((repo/'artifacts/bga_927500015_r20_analysis_20261020/greedy.json').read_text())['rows']
checkpoint=repo/'runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt'
started=time.perf_counter()
advisor=LocalAdvisor(checkpoint)
assert advisor.checkpoint_sha256=='fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd'
advice=[]
prior_deltas=[]
value_deltas=[]
for row,reference in zip(rows,golden,strict=True):
 request=decode_request(row['request'])
 before=decision_identity(request)
 result=advisor.recommend(request,top_k=len(request.legal_actions))
 assert result['recommended_candidate_index']==reference['choice'],row['decision_index']
 assert result['recommended_action']==to_primitive(request.legal_actions[reference['choice']])
 actual={r['candidate_index']:r['policy_probability'] for r in result['candidates']}
 prior_deltas.append(float(np.max(np.abs(np.array([actual[i] for i in range(len(actual))])-reference['prior']))))
 value_deltas.append(abs(result['value_actor_raw']-reference['value_actor_raw']))
 assert before==decision_identity(request) and result['submitted_action'] is False
 result['decision_index']=row['decision_index']
 result['bga_move']=row['bga_move']
 advice.append(result)
assert max(prior_deltas)<1e-5,max(prior_deltas)
assert max(value_deltas)<1e-5,max(value_deltas)
(out/'greedy_all_88.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in advice))
search_advisor=LocalAdvisor(checkpoint,mode='mcts',simulations=32,determinizations=1,seed=39700000)
positions=[0,next(i for i,row in enumerate(rows) if row['request']['observation']['phase']=='payment' and len(row['request']['legal_actions'])>1)]
smoke=[]
for i in positions:
 request=decode_request(rows[i]['request'])
 result=search_advisor.recommend(request,top_k=3)
 assert result['simulations']==32
 assert result['recommended_action']==to_primitive(request.legal_actions[result['recommended_candidate_index']])
 assert result['candidates'][0]['candidate_index']==result['recommended_candidate_index']
 result['decision_index']=i
 result['bga_move']=rows[i]['bga_move']
 smoke.append(result)
(out/'mcts_smoke.json').write_text(json.dumps(smoke,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
summary={'status':'passed','checkpoint':str(checkpoint),'checkpoint_sha256':advisor.checkpoint_sha256,
 'greedy_decisions':len(advice),'greedy_choices_agree_with_existing_analysis':True,
 'max_prior_absolute_difference':max(prior_deltas),'max_value_absolute_difference':max(value_deltas),
 'numeric_comparison':'Single-position vs historical batched FP32 inference, absolute tolerance 1e-5.',
 'request_unchanged':True,'submitted_actions':0,'mcts_smoke_positions':positions,
 'mcts_simulations_per_position':32,'seconds':time.perf_counter()-started,
 'scope':'Offline engineering verification only. Live BGA reading and action execution are not implemented.'}
(out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
