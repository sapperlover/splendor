"""Count actually played reservations from completed self-play traces, CPU only."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT/'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025'
MODEL_SHA = '852cb0420e94d77c146bc2330600f48d52f638e4cca96a5c1578e9dfe54940ec'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n')


def fresh():
    return dict(counts=Counter(), tiers=Counter(), game_blind_hist=Counter(),
                player_blind_hist=Counter(), stages={}, games=0, players=0,
                main_facing_hidden=0, blind_bought=0, blind_unbought=0,
                low_main=0, full_main=0, low_blind=0, full_blind=0,
                low_reservations=0, full_reservations=0)


def scan(folder):
    manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    summary=json.loads((folder/'summary.json').read_text())
    expected=manifest['config']['num_games']
    files=sorted((folder/'games').glob('game-*.npz'))
    assert len(files)==expected==summary['completed_games']==10000
    assert summary['failed_games']==summary['missing_games']==0
    stats=fresh()
    stream=hashlib.sha256()
    for game_id,path in enumerate(files):
        assert path.name==f'game-{game_id:06d}.npz'
        with np.load(path,allow_pickle=False) as archive:
            raw=archive['metadata_json'].tobytes()
            metadata=json.loads(raw)
            low=archive['low_budget']
        stream.update(path.name.encode()+b'\0'+raw+b'\0')
        assert metadata['status']=='natural_terminal' and metadata['game_id']==game_id
        assert metadata['semantic_fingerprint']==manifest['semantic_fingerprint']
        assert metadata['checkpoint_sha256']==manifest['checkpoint_identity']['sha256']
        trace=metadata['trace']['actions']
        assert len(trace)==metadata['decision_count']==len(low)
        assert metadata['trace']['terminal_result']==metadata['terminal_result']
        hidden=[0,0]
        public=[set(),set()]
        turns=[0,0]
        blinds=[0,0]
        for index,row in enumerate(trace):
            assert row['decision_index']==index
            if row['phase']!='main_action':
                continue
            p=row['player_id']
            turns[p]+=1
            stats['main_facing_hidden']+=int(hidden[1-p]>0)
            action=row['action']
            kind=action['__type__']
            stats['counts'][kind]+=1
            budget='low' if bool(low[index]) else 'full'
            stats[f'{budget}_main']+=1
            stage='1-10' if turns[p]<=10 else ('11-20' if turns[p]<=20 else '21+')
            sc=stats['stages'].setdefault(stage,Counter())
            sc['main']+=1
            if kind in ('ReserveDeck','ReserveMarket'):
                sc['reservations']+=1
                stats[f'{budget}_reservations']+=1
            if kind=='ReserveDeck':
                stats['tiers'][str(action['tier'])]+=1
                sc['blind']+=1
                stats[f'{budget}_blind']+=1
                hidden[p]+=1
                blinds[p]+=1
            elif kind=='ReserveMarket':
                assert action['card_id'] not in public[p]
                public[p].add(action['card_id'])
            elif kind=='BuyReserved':
                card=action['card_id']
                if card in public[p]:
                    public[p].remove(card)
                else:
                    assert hidden[p]>0
                    hidden[p]-=1
                    stats['blind_bought']+=1
        stats['blind_unbought']+=sum(hidden)
        stats['game_blind_hist'][str(sum(blinds))]+=1
        for n in blinds:
            stats['player_blind_hist'][str(n)]+=1
        stats['games']+=1
        stats['players']+=2
    assert stats['blind_bought']+stats['blind_unbought']==stats['counts']['ReserveDeck']
    assert sum(stats['counts'].values())==stats['low_main']+stats['full_main']
    return dict(path=str(folder.relative_to(ROOT)),manifest_sha256=digest(manifest_path),
                checkpoint=manifest['checkpoint_identity'],semantic_fingerprint=manifest['semantic_fingerprint'],
                metadata_stream_sha256=stream.hexdigest(),config=manifest['config'],stats=stats)


def merge(rows):
    total=fresh()
    for row in rows:
        s=row['stats']
        for key in ('counts','tiers','game_blind_hist','player_blind_hist'):
            total[key].update(s[key])
        for stage,values in s['stages'].items():
            total['stages'].setdefault(stage,Counter()).update(values)
        for key in total.keys()-{'counts','tiers','game_blind_hist','player_blind_hist','stages'}:
            total[key]+=s[key]
    return total


def rates(stats):
    blind=stats['counts']['ReserveDeck']
    main=sum(stats['counts'].values())
    reservations=blind+stats['counts']['ReserveMarket']
    return dict(blind_per_main=blind/main,blind_per_reservation=blind/reservations,
                blind_per_game=blind/stats['games'],
                games_with_blind=1-stats['game_blind_hist'].get('0',0)/stats['games'],
                players_with_blind=1-stats['player_blind_hist'].get('0',0)/stats['players'],
                main_facing_hidden=stats['main_facing_hidden']/main,
                blind_bought_fraction=stats['blind_bought']/blind,
                low_blind_per_main=stats['low_blind']/stats['low_main'],
                full_blind_per_main=stats['full_blind']/stats['full_main'])


def main():
    started=time.perf_counter()
    rows=[]
    # Complete batches only: local round000 through012, 10000 games each.
    # The partial round013 is deliberately excluded.
    for index in range(13):
        row=scan(RUN/f'rounds/round-{index:03d}/selfplay')
        row['collection_batch']=index+1
        row['source_model_online_round']=index
        row['rates']=rates(row['stats'])
        rows.append(row)
        write(ART/'progress.json',dict(completed_batches=len(rows),rows=rows))
        print(json.dumps(dict(batch=index+1,games=row['stats']['games'],rates=row['rates'],
                              elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    aggregate=merge(rows)
    focus=[r for r in rows if r['checkpoint']['sha256']==MODEL_SHA]
    assert len(focus)==1 and focus[0]['source_model_online_round']==9
    report=dict(scope='All actual main actions in thirteen complete 10000-game batches. '
                'Includes self-play root noise and mixed budgets; not noise-free human-play behavior. '
                'Partial round013 excluded.',run=str(RUN.relative_to(ROOT)),
                script_sha256=digest(Path(__file__)),rows=rows,aggregate=dict(stats=aggregate,rates=rates(aggregate)),
                current_human_model_source_round=9,seconds=time.perf_counter()-started)
    write(ART/'summary.json',report)
    print('COMPLETE',json.dumps(report['aggregate']['rates']),flush=True)


if __name__=='__main__':
    main()
