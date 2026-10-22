"""Finite two-worker CUDA collection with the corrected noble catalog."""
from pathlib import Path
import time
import torch

from splendor.alphazero_recording import read_json, write_json
from splendor.mcts_policy import MCTSConfig
from splendor.selfplay_fast import ExecutionConfig, FastSelfPlayConfig, generate, validate_run
from splendor.search_training import SearchBudgetRandomization

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    previous=read_json(OUT/'collection_validation.json')
    config=FastSelfPlayConfig(protocol='mcts-selfplay-v2',run_id='noble12-cuda-pool-smoke',
        round_index=0,checkpoint=str(ROOT/'runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt'),
        num_games=2,seed=previous['games'][0]['base_seed'],
        search_config=MCTSConfig(simulations=4,determinizations=1,seed=41230000,
            forced_playouts=0.,policy_target_pruning=False),
        execution=ExecutionConfig(backend='native_cuda_pool',engine_mode='fast',workers=1,
            batch_interface='native',inference_mode='eager',actor_projection=False,
            candidate_buckets=(16,32,48,64),max_batch_size=4,pipeline_slots=2,
            search_workers=2,total_active_games=2,local_microbatch=2,shared_slots=2,
            shared_candidate_width=128,max_batch_wait_ms=1),
        search_backend='native_chance',validation_fraction=0.,
        budget_randomization=SearchBudgetRandomization(low_simulations=1,low_probability=.75))
    write_json(OUT/'gpu_config.json',config.to_dict())
    destination=ROOT/'runtime/noble_pool_bga12_20261021/collection_cuda_pool'
    start=time.perf_counter()
    summary=generate(config,destination)
    checked=validate_run(destination,replay=True)
    assert summary['completed_games']==checked['completed_games']==2
    assert read_json(destination/'manifest.json')['catalog_fingerprint']==previous['catalog_fingerprint']
    write_json(OUT/'gpu_collection_validation.json',{'scope':'Two-game CUDA engineering verification only; not strength.',
        'gpu':torch.cuda.get_device_name(0),'elapsed_seconds':time.perf_counter()-start,
        'catalog_fingerprint':previous['catalog_fingerprint'],'summary':summary,'checked_replay':checked})
    print('CUDA pool: two complete games and checked replays passed.',flush=True)


if __name__=='__main__':
    main()
