"""Check trace-based counting against stored candidate/state arrays on a fixed sample."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from splendor.features import ACTION_FEATURE_NAMES, ACTION_TYPE_ORDER, CARD_FEATURE_NAMES

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
RUN=ROOT/'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025'
rng=np.random.default_rng(20261029)
type_columns={cls.__name__:ACTION_FEATURE_NAMES.index('type_'+name)
              for cls,name in zip(ACTION_TYPE_ORDER,('take_distinct','take_pair','reserve_market',
                  'reserve_deck','buy_market','buy_reserved','pay','return_tokens','claim_noble'),strict=True)}
known=CARD_FEATURE_NAMES.index('known')
counts=Counter()
identities=[]
for batch in range(13):
    ids=sorted(int(x) for x in rng.choice(10000,10,replace=False))
    identities.append(dict(batch=batch+1,game_ids=ids))
    for game_id in ids:
        path=RUN/f'rounds/round-{batch:03d}/selfplay/games/game-{game_id:06d}.npz'
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(z['metadata_json'].tobytes())
            actions=meta['trace']['actions']
            chosen=z['candidate_offsets'][:-1]+z['chosen_candidate_index']
            features=z['candidate_features'][chosen]
            op=z['opponent_reserved']
            mask=z['opponent_reserved_mask']
            players=z['player_id']
            phases=z['phase']
        hidden=[0,0]
        public=[set(),set()]
        for index,row in enumerate(actions):
            p=row['player_id'];action=row['action'];kind=action['__type__']
            assert players[index]==p and features[index,type_columns[kind]]==1
            counts['chosen_actions_checked']+=1
            if row['phase']!='main_action':
                continue
            assert phases[index]==0
            assert int(np.sum(mask[index] & (op[index,:,known]==0)))==hidden[1-p]
            counts['main_opponent_hidden_counts_checked']+=1
            if kind=='ReserveDeck':
                hidden[p]+=1;counts['blind_actions_checked']+=1
            elif kind=='ReserveMarket':
                public[p].add(action['card_id'])
            elif kind=='BuyReserved':
                if action['card_id'] in public[p]:
                    public[p].remove(action['card_id'])
                else:
                    hidden[p]-=1
                    assert hidden[p]>=0
        counts['games_checked']+=1
report=dict(seed=20261029,numpy_version=np.__version__,sampling='10 games without replacement per complete batch',
            counts=counts,sample=identities,all_checks_passed=True,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ART/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(dict(counts),flush=True)
