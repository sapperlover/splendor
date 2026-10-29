"""Audit the first blind root's derived streams, including compiled-native checks."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from splendor import _native_search as native
from splendor.actions import ReserveDeck
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.mcts_policy import MCTSConfig
from splendor.model import Tier
from splendor.replay import _decode_state
from splendor.search_policy import _request_public_word,_request_seed,determinize_request

sys.path.insert(0,str(Path(__file__).parent))
from analyze import ART,RAW,fast

catalog=load_default_catalog()
root=read_json(RAW/'roots.json')[0]
state=_decode_state(root['state'])
request=fast(state,catalog).decision()
word=_request_public_word(request)
reference=determinize_request(request,seed=_request_seed(request,0,0),catalog=catalog,engine_mode='fast')._debug_state()
# No opponent blind reserves at this first root: unknown pools are exactly decks.
assert not request.observation.opponent_public.reserved_cards or all(r.card is not None for r in request.observation.opponent_public.reserved_cards)
lengths=[len(reference.decks.for_tier(tier)) for tier in Tier]
candidates=sorted(root['candidates'])
assert len(candidates)==lengths[2]==15
skip=(lengths[0]-1)+(lengths[1]-1)
G=0x9E3779B97F4A7C15
MASK=(1<<64)-1

def tops(base):
    base=np.asarray(base,dtype=np.uint64)
    derived=np.bitwise_xor(np.bitwise_xor(np.uint64(word),base[:,None]),np.array([((j+1)*G)&MASK for j in range(4)],dtype=np.uint64)[None,:])
    with np.errstate(over='ignore'):
        value=derived+np.uint64(((skip+1)*G)&MASK)
        value=(value^(value>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        value=(value^(value>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        value=value^(value>>np.uint64(31))
    assert np.all(value>=np.uint64((1<<64)%15))
    return (value%np.uint64(15)).astype(np.int16)

native_catalog=native.Catalog(catalog)
blind_index=request.legal_actions.index(ReserveDeck(Tier.THREE))
checked=[]
check_seeds=list(range(128))+[int(x) for x in np.random.default_rng(2026102902).integers(0,1<<63,size=128,dtype=np.uint64)]
for seed in check_seeds:
    config=MCTSConfig(simulations=4,determinizations=4,seed=seed,root_noise_fraction=0.,fold_forced_phases=False,max_simulation_transitions=1)
    job=native.SearchJob(native_catalog,request,config,word,seed,False,0.)
    prior=[-1000.]*len(request.legal_actions);prior[blind_index]=0.
    job.submit(-1,prior,0.)
    job.initialize_worlds(request)
    actual=[]
    for world in range(4):
        leaf=job.debug_leaf_state(world)
        actual.append(catalog.cards[leaf['players'][0]['reserved'][-1][0]].card_id)
    expected=[candidates[int(x)] for x in tops([seed])[0]]
    python=[determinize_request(request,seed=_request_seed(request,seed,j),catalog=catalog,engine_mode='fast')._debug_state().decks.tier_three[-1] for j in range(4)]
    assert actual==expected==python,(seed,actual,expected,python)
    if seed==0:checked.append(dict(seed=seed,derived_seeds=[_request_seed(request,seed,j) for j in range(4)],native_tops=actual))

N=200000
rng=np.random.default_rng(2026102903)
summary=[]
for label,seeds in [('consecutive_base_seeds',np.arange(N,dtype=np.uint64)),('uniform_64bit_base_seeds',rng.integers(0,1<<64,size=N,dtype=np.uint64))]:
    draws=tops(seeds)
    max_counts=np.ones(N,dtype=np.int16)
    for j in range(4):max_counts=np.maximum(max_counts,(draws==draws[:,j,None]).sum(axis=1))
    counts=np.bincount(max_counts,minlength=5)
    target=candidates.index('076')
    specific=np.sum(draws==target,axis=1)
    white=np.isin(draws,[candidates.index(x) for x in ('076','077','078')]).sum(axis=1)
    summary.append(dict(sample=label,groups=N,any_card_at_least3=int(np.sum(max_counts>=3)),specific076_at_least3=int(np.sum(specific>=3)),multiplicity_counts=counts.tolist(),per_world_card_counts=[np.bincount(draws[:,j],minlength=15).tolist() for j in range(4)],white_count_histogram=np.bincount(white,minlength=5).tolist(),pair_same_counts={f'{a}-{b}':int(np.sum(draws[:,a]==draws[:,b])) for a in range(4) for b in range(a+1,4)}))
report=dict(root_event=root['event'],pool_size=15,sorted_pool=candidates,public_word=word,shuffle_pool_lengths=lengths,prior_shuffle_rng_draws=skip,native_python_vectorized_crosscheck_groups=len(check_seeds),seed0=checked[0],theory=dict(assumption='independent uniform worlds, each reshuffled separately; repeated tops allowed',any_card_at_least3=15*(4*(1/15)**3*(14/15)+(1/15)**4),specific076_at_least3=4*(1/15)**3*(14/15)+(1/15)**4,pair_same=1/15),samples=summary,source=dict(script_sha256=sha256(Path(__file__)),native_sha256=sha256(Path(native.__file__)),native_source_sha256=sha256(Path(native.__file__).with_name('native_search.cpp'))))
write_json(ART/'rng_audit.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
