import json, time
from pathlib import Path
import numpy as np
import torch
from splendor.alphazero_recording import sha256
from splendor.mcts_policy import MCTSConfig
from splendor.policy_averaging import PolicyAveragingConfig, generate_labels
from splendor.selfplay_fast import ExecutionConfig, FastSelfPlayConfig, generate
from splendor.selfplay_data import load_game

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
root=Path('runtime/entity_attention_avg4_behavior_20261009/cuda_smoke_final')
config=FastSelfPlayConfig(protocol='mcts-selfplay-v2',run_id='m4-cuda-smoke',round_index=0,
 checkpoint='runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1/rounds/round-019/checkpoints/final.pt',
 num_games=2,seed=27000000,search_config=MCTSConfig(simulations=8,determinizations=1),
 execution=ExecutionConfig(backend='native_cuda',engine_mode='fast',batch_interface='native',
 inference_mode='eager',active_games_per_worker=2,max_batch_size=4),validation_fraction=.5,split_seed=27010000)
started=time.perf_counter()
summary=generate(config,root/'selfplay')
hashes={p.name:sha256(p) for p in (root/'selfplay/games').glob('*.npz')}
labels=generate_labels(root/'selfplay',root/'policy_labels',PolicyAveragingConfig(active_roots=4,games_per_block=2),device='cuda')
for p in (root/'selfplay/games').glob('*.npz'):
 arrays,_=load_game(p)
 with np.load(root/'policy_labels/games'/p.name) as z:
  np.testing.assert_array_equal(z['targets'][0].astype(np.float32),arrays['policy_target'])
  np.testing.assert_array_equal(z['target_mean'],z['targets'].mean(axis=0).astype(np.float32))
assert hashes=={p.name:sha256(p) for p in (root/'selfplay/games').glob('*.npz')}
report={'status':'passed','scope':'2 natural CUDA games, S8/N1 finite smoke only; no long training or strength matches',
 'collection':summary,'label_manifest':labels,'raw_archives_unchanged':True,'behavior_is_sample0':True,
 'mean_of_four_verified':True,'seconds':time.perf_counter()-started}
out=Path('artifacts/entity_attention_avg4_behavior_20261009/cuda_smoke.json')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','games':2,'extra_simulations':labels['extra_simulations'],'seconds':report['seconds']}))
