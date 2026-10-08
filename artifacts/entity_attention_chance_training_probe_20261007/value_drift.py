"""Post-hoc head/trunk substitutions on the frozen common bank; no updates or games."""
from pathlib import Path
import json

import numpy as np
import torch
from splendor.alphazero_recording import read_json, write_json
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.features import encode_decision, collate_encoded_decisions
from splendor.compact_trace import decode_request
from diagnostics import clustered_ci
from experiment import ART, ARMS, INITIAL, checkpoint, common_inputs, check_source, record

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    check_source()
    rows,labels,_=common_inputs()
    batch=collate_encoded_decisions([encode_decision(decode_request(r['request'])) for r in rows])
    frozen=load_frozen_actor_critic_checkpoint(INITIAL,device='cpu')
    prepared=frozen._prepare(batch)
    z=np.array([labels[r['id']] for r in rows],dtype=np.float64)
    with torch.inference_mode():
        old_state=frozen.encode_state(prepared)
        v0=frozen.critic(old_state).squeeze(-1).numpy().astype(np.float64)
        out={}
        for arm in ARMS:
            training=read_json(ART/f'train_{arm}.json')
            out[arm]={}
            for step in (100,300,500):
                model=load_frozen_actor_critic_checkpoint(checkpoint(arm,step),device='cpu')
                new_state=model.encode_state(prepared)
                values={
                    'old_trunk_old_head':v0,
                    'new_trunk_old_head':frozen.critic(new_state).squeeze(-1).numpy().astype(np.float64),
                    'old_trunk_new_head':model.critic(old_state).squeeze(-1).numpy().astype(np.float64),
                    'new_trunk_new_head':model.critic(new_state).squeeze(-1).numpy().astype(np.float64)}
                measured=next(r for r in training['curve'] if r['step']==step)['common']['per_root']
                np.testing.assert_allclose(values['new_trunk_new_head'],[r['value'] for r in measured],atol=1e-5,rtol=1e-5)
                keys=[{'game_key':r['game_key']} for r in rows]
                summaries={}
                for name,v in values.items():
                    summaries[name]={'value_mse':float(np.mean((v-z)**2)),
                        'clipped_value_mse':float(np.mean((np.clip(v,-1,1)-z)**2)),
                        'outside_payoff_fraction':float(np.mean(np.abs(v)>1)),
                        'mean_abs_prediction_change_from_initial':float(np.mean(np.abs(v-v0))),
                        'mse_change_from_initial':clustered_ci(keys,(v-z)**2-(v0-z)**2)}
                out[arm][str(step)]={'cpu_gpu_value_max_abs_difference':float(np.max(np.abs(values['new_trunk_new_head']-np.array([r['value'] for r in measured])))),
                    'summary':summaries,'per_root':[
                    {'root_id':r['id'],'game_key':r['game_key'],'stage':r['stage'],'value_target':int(z[i]),
                     **{name:float(v[i]) for name,v in values.items()}} for i,r in enumerate(rows)]}
    write_json(ART/'value_drift.json',{
        'protocol':'chance-training-probe-value-drift-v1','device':'cpu','arms':out,
        'scope':'Post-hoc substitutions of attention state encoder and value head on 200 archived roots. Components were not separately retrained. This diagnoses prediction drift, not causal chess-strength effects; MSE effects need not add.',
        'inputs':[record(INITIAL)]+[record(checkpoint(a,s)) for a in ARMS for s in (100,300,500)],
        'script':record(Path(__file__))})
    print(json.dumps({a:{s:{k:v['value_mse'] for k,v in r['summary'].items()} for s,r in x.items()} for a,x in out.items()}))
