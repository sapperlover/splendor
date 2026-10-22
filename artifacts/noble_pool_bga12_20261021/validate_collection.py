"""Two real-model engineering games with both newly added nobles in play."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from splendor.alphazero_recording import sha256, write_json
from splendor.catalog import load_catalog, load_default_catalog
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import _extension, _native_catalog
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_training import SearchBudgetRandomization
from splendor.selfplay import derive_seed
from splendor.selfplay_fast import ExecutionConfig, FastSelfPlayConfig, generate, validate_run

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime/noble_pool_bga12_20261021'
CHECKPOINT=ROOT/'runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt'
MODEL_SHA='fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd'


def main():
    torch.set_num_threads(1)
    assert sha256(CHECKPOINT)==MODEL_SHA
    catalog=load_default_catalog()
    old=load_catalog(OUT/'previous_catalog')
    assert len(catalog.nobles)==12 and catalog.data_fingerprint!=old.data_fingerprint
    old_engine=GameEngine.new(42,first_player=0,catalog=old,mode='checked')
    old_native=_extension().NativeGame(_native_catalog(old),old_engine._debug_state())
    assert old_native.verify_actions(old_engine.decision())
    model=load_frozen_actor_critic_checkpoint(CHECKPOINT,device='cpu')
    assert sum(p.numel() for p in model.parameters())==583682
    for base_seed in range(41200000,41201000):
        seed=derive_seed(base_seed,0,0,'game')
        state=GameEngine.new(seed,first_player=0,catalog=catalog,mode='fast')._debug_state()
        if {'N11','N12'}<=set(state.nobles_in_play):
            break
    else:
        raise AssertionError('No engineering opening containing both added nobles.')
    rows=[]
    for backend in ('native','native_chance'):
        randomized=backend=='native_chance'
        config=FastSelfPlayConfig(protocol='mcts-selfplay-v2',run_id='noble12-'+backend,
            round_index=0,checkpoint=str(CHECKPOINT),num_games=1,seed=base_seed,
            search_config=MCTSConfig(simulations=4,determinizations=1,seed=41230000,
                forced_playouts=0.,policy_target_pruning=False),
            execution=ExecutionConfig(backend='native_cpu',workers=1,engine_mode='fast',
                batch_interface='native',inference_mode='eager',actor_projection=False,
                candidate_buckets=(16,32,48,64),active_games_per_worker=1,max_batch_size=4),
            search_backend=backend,validation_fraction=0.,
            budget_randomization=SearchBudgetRandomization(low_simulations=1,low_probability=.75) if randomized else None)
        write_json(OUT/(backend+'_config.json'),config.to_dict())
        destination=RUNTIME/('collection_'+backend)
        start=time.perf_counter()
        summary=generate(config,destination)
        verified=validate_run(destination,replay=True)
        assert summary['completed_games']==verified['completed_games']==1
        manifest=json.loads((destination/'manifest.json').read_text())
        assert manifest['catalog_fingerprint']==catalog.data_fingerprint
        from splendor.selfplay_data import load_game
        _,metadata=load_game(destination/'games/game-000000.npz')
        assert metadata['trace']['catalog_fingerprint']==catalog.data_fingerprint
        rows.append({'backend':backend,'data_schema':config.data_schema,'base_seed':base_seed,
            'game_seed':seed,'initial_nobles':state.nobles_in_play,
            'elapsed_seconds':time.perf_counter()-start,'summary':summary,'checked_replay':verified,
            'archive':str((destination/'games/game-000000.npz').relative_to(ROOT)),
            'archive_sha256':sha256(destination/'games/game-000000.npz')})
    assert sha256(CHECKPOINT)==MODEL_SHA
    write_json(OUT/'collection_validation.json',{'scope':'Two CPU engineering games, not a strength evaluation.',
        'old_catalog_fingerprint':old.data_fingerprint,'catalog_fingerprint':catalog.data_fingerprint,
        'old_10_noble_native_catalog_still_loads_explicitly':True,
        'checkpoint_sha256_unchanged':MODEL_SHA,'model_config':asdict(model.config),
        'parameters':583682,'games':rows})
    print(json.dumps({'engineering_games':2,'checked_replays':2,'catalog_fingerprint':catalog.data_fingerprint}),flush=True)


if __name__=='__main__':
    main()
