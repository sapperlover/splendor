"""Sample actual chance actions by public-score stage, preserving reply anchors."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.features import GLOBAL_FEATURE_NAMES
from splendor.model import DecisionPhase
from splendor.replay import _decode_action,to_primitive

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name
POOL=ROOT/'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-009/selfplay'
MODEL12=ROOT/'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt'
SEED=20261029


def main():
    started=time.perf_counter();a.identity_check();RAW.mkdir(parents=True,exist_ok=True)
    catalog=load_default_catalog()
    manifest=read_json(POOL/'manifest.json')
    assert manifest['checkpoint_identity']['sha256']==a.MODEL_SHA
    rng=np.random.default_rng(SEED)
    ids=sorted(int(x) for x in rng.choice(10000,2000,replace=False))
    pools=defaultdict(list);excluded=Counter();metadata_hash={}
    si=GLOBAL_FEATURE_NAMES.index('self_score_scaled');oi=GLOBAL_FEATURE_NAMES.index('opponent_score_scaled')
    for game_id in ids:
        path=POOL/'games'/f'game-{game_id:06d}.npz'
        with np.load(path,allow_pickle=False) as archive:
            raw=archive['metadata_json'].tobytes();meta=json.loads(raw);glob=archive['global_features']
        metadata_hash[game_id]=hashlib.sha256(raw).hexdigest()
        assert meta['status']=='natural_terminal' and meta['checkpoint_sha256']==a.MODEL_SHA
        trace=meta['trace']['actions']
        mains=[i for i,r in enumerate(trace) if r['phase']=='main_action']
        for k,index in enumerate(mains):
            row=trace[index];action=row['action'];kind=action['__type__']
            if kind not in ('BuyMarket','ReserveMarket','ReserveDeck'):continue
            tier=action['tier'] if kind=='ReserveDeck' else catalog.card(action['card_id']).tier.value
            count=round(float(glob[index,GLOBAL_FEATURE_NAMES.index(f'deck_tier_{tier}_scaled')])*{1:40,2:30,3:20}[tier])
            if count<4:excluded['fewer_than4_draws']+=1;continue
            score=round(15*max(float(glob[index,si]),float(glob[index,oi])))
            stage='early' if score<5 else ('mid' if score<10 else 'late')
            mode='blind' if kind=='ReserveDeck' else 'public'
            target=k+1 if mode=='public' else k+2
            if target>=len(mains):excluded[f'{mode}_no_next_anchor']+=1;continue
            if mode=='blind':
                reply=trace[mains[k+1]]
                if reply['player_id']==row['player_id']:
                    excluded['blind_forced_skip_reply']+=1;continue
                if reply['action']['__type__'] in ('BuyMarket','ReserveMarket','ReserveDeck'):
                    excluded['blind_reply_has_extra_draw']+=1;continue
                assert trace[mains[target]]['player_id']==row['player_id']
            pools[(mode,stage)].append(dict(game_id=game_id,index=index,anchor=mains[target],tier=tier,
                                            stage=stage,mode=mode,actor=row['player_id'],max_score=score))
    selected=[]
    for key in [('public',s) for s in ('early','mid','late')]+[('blind',s) for s in ('early','mid','late')]:
        count=300 if key[0]=='public' else 100
        assert len(pools[key])>=count,(key,len(pools[key]))
        chosen=rng.choice(len(pools[key]),count,replace=False)
        selected.extend(pools[key][int(i)] for i in chosen)
    by_game=defaultdict(list)
    for row in selected:by_game[row['game_id']].append(row)
    cases=[]
    for game_id,items in sorted(by_game.items()):
        path=POOL/'games'/f'game-{game_id:06d}.npz'
        with np.load(path,allow_pickle=False) as z:meta=json.loads(z['metadata_json'].tobytes())
        trace=meta['trace']['actions'];engine=GameEngine.new(meta['game_seed'],meta['first_player'],catalog=catalog,mode='fast')
        wanted={r['index']:r for r in items}
        for index,row in enumerate(trace):
            req=engine.decision();action=_decode_action(row['action'])
            assert engine.current_player==row['player_id'] and action in req.legal_actions
            if index in wanted:
                selected_row=wanted[index]
                assert req.observation.phase is DecisionPhase.MAIN_ACTION
                state=engine._debug_state()
                case=dict(**selected_row,id=f"g{game_id:06d}_d{index:03d}",state=to_primitive(state),
                          actions=[r['action'] for r in trace[index:selected_row['anchor']]],
                          target_actor=trace[selected_row['anchor']]['player_id'],
                          metadata_sha256=metadata_hash[game_id],game_seed=meta['game_seed'])
                cases.append(case)
            engine.apply(action)
        assert to_primitive(engine._debug_state().result)==meta['terminal_result']
    assert len(cases)==1200
    write_json(RAW/'cases.json',cases)
    plan=dict(scope='Actual chosen chance actions sampled from2000 random games of online9 self-play. '
              'Conditional on actual other hidden cards; not a belief-marginalized search evaluation.',
              sample_seed=SEED,sampled_games=2000,sampled_game_ids=ids,case_count=len(cases),
              stage='max public score before action: early0-4,mid5-9,late>=10',
              public_anchor='Next actual MAIN_ACTION after settling the same actual own-turn subdecisions.',
              blind_anchor='Next own MAIN_ACTION after the same complete actual opponent reply; replies with additional draws excluded.',
              population_counts={f'{k[0]}_{k[1]}':len(v) for k,v in pools.items()},excluded=dict(excluded),
              source_pool=str(POOL.relative_to(ROOT)),source_manifest_sha256=sha256(POOL/'manifest.json'),
              model9=str(a.MODEL.relative_to(ROOT)),model9_sha256=sha256(a.MODEL),
              model12=str(MODEL12.relative_to(ROOT)),model12_sha256=sha256(MODEL12),
              source_plan=str((a.ART/'plan.json').relative_to(ROOT)),cases=str((RAW/'cases.json').relative_to(ROOT)),
              cases_sha256=sha256(RAW/'cases.json'),script_sha256=sha256(Path(__file__)),
              prespecified_metrics=dict(score_rate='(clip(V,-1,1)+1)/2; uncalibrated estimate',
                    noticeable='range>=5 percentage points',robust='p90-p10>=5pp',
                    quartile_separation='top-bottom quartile means>=5pp',four_clear_layers='every adjacent quartile mean gap>=2pp',
                    ordering_check='online12 versus online9 on identical valid own-turn observations'),
              seconds=time.perf_counter()-started)
    write_json(ART/'plan.json',plan)
    print(json.dumps(dict(cases=len(cases),unique_games=len(by_game),population=plan['population_counts'],excluded=dict(excluded),seconds=plan['seconds'])),flush=True)


if __name__=='__main__':main()
