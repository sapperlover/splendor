"""Independent capture, hidden-information and search-result checks."""
import json
from pathlib import Path
import zipfile

import numpy as np

from splendor.catalog import load_default_catalog
from splendor.compact_trace import decode_request
from splendor.observation import build_observation
from splendor.replay import _decode_state
from splendor.rules import legal_actions
from splendor._state import DeckState
from splendor.model import Tier
from dataclasses import replace

from analyze import ART, RAW, ROOT, MODEL, MODEL_SHA, sha, records


def public(capture):
    kinds={'reserveCard','buyCard','coins','revealCard','updateScore'}
    return [(p['move_id'],[(n['type'],n['args']) for n in p['data'] if n['type'] in kinds])
        for p in capture['logs'] if p['channel']=='/table/t939469760']


def main():
    cat=load_default_catalog()
    zero=json.loads((RAW/'capture_zero.json').read_text())
    blue=json.loads((RAW/'capture_blue.json').read_text())
    assert public(zero)==public(blue)
    assert zero['initial']['gamedatas']['market']==blue['initial']['gamedatas']['market']
    assert str(zero['initial']['player_id'])=='96295883' and str(blue['initial']['player_id'])=='100857139'
    mapping=json.loads((RAW/'card_id_mapping.json').read_text())['cards']
    state=_decode_state(json.loads((RAW/'reconstruction.json').read_text())['final_state'])
    final_files=[RAW/'final_zero.json', RAW/'final_blue.json']
    ids=['96295883','100857139']
    for path in final_files:
        snap=json.loads(path.read_text())['snapshot'];gd=snap['gamedatas']
        assert gd['gamestate']['name']=='gameEnd' and snap['move_text']=='63'
        assert list(state.supply)==[int(gd['market']['pool'][c]) for c in 'CSEROG']
        for t in (1,2,3):
            assert set(state.market.for_tier(Tier(t)))=={mapping[str(c['type'])] for c in gd['market'][f'row_{t}']['cards'].values()}
            assert len(state.decks.for_tier(Tier(t)))==int(gd['market'][f'row_{t}']['count'])
        for i,pid in enumerate(ids):
            p=gd['players'][pid]
            assert state.result.scores[i]==int(p['score'])
            assert list(state.players[i].tokens)==[int(p['coins'][c]) for c in 'CSEROG']
            assert set(state.players[i].purchased_card_ids)=={mapping[str(c['type'])] for c in p['cards_played'].values()}
        viewer=ids.index(str(snap['player_id']))
        hidden=gd['cards_hidden']
        own={mapping[str(c['type'])] for c in hidden.values()} if hidden else set()
        assert {r.card_id for r in state.players[viewer].reserved_cards}==own
    rows=records();states=json.loads((RAW/'states.json').read_text())
    assert len(rows)==len(states)==86
    for row,raw in zip(rows,states,strict=True):
        s=_decode_state(raw);req=decode_request(row['request'])
        assert req.observation==build_observation(s,s.current_player,cat)
        assert req.legal_actions==legal_actions(s,cat)
        assert req.legal_actions[row['candidate_index']].__class__.__name__==row['action']['__type__']
        hidden=[(p,i) for p in (0,1) if p!=s.current_player for i,r in enumerate(s.players[p].reserved_cards) if r.source.value=='deck']
        if hidden:
            p,i=hidden[0];reserved=list(s.players[p].reserved_cards)
            tier=cat.card(reserved[i].card_id).tier;deck=list(s.decks.for_tier(tier))
            previous=reserved[i].card_id;reserved[i]=replace(reserved[i],card_id=deck[-1]);deck[-1]=previous
            players=list(s.players);players[p]=replace(players[p],reserved_cards=tuple(reserved))
            decks=[s.decks.for_tier(t) for t in Tier];decks[int(tier)-1]=tuple(deck)
            alt=replace(s,players=tuple(players),decks=DeckState(*decks))
            assert build_observation(alt,s.current_player,cat)==req.observation
            assert legal_actions(alt,cat)==req.legal_actions
    search=json.loads((RAW/'search.json').read_text());greedy=json.loads((RAW/'greedy.json').read_text())
    assert len(search)==71
    for result in search:
        i=result['decision_index'];v=np.asarray(result['visits']);wv=np.asarray(result['per_world_visits'])
        assert v.sum()==4096 and wv.shape==(4,len(v))
        assert np.all(wv.sum(axis=1)==1024)
        np.testing.assert_array_equal(v,wv.sum(axis=0))
        np.testing.assert_allclose(v/4096,(wv/1024).mean(axis=0),atol=1e-14,rtol=0)
        np.testing.assert_allclose(result['prior'],greedy[i]['prior'],atol=2e-6,rtol=0)
        best=max(range(len(v)),key=lambda j:(v[j],result['prior'][j],-j))
        assert result['choice']==best
    assert sha(MODEL)==MODEL_SHA
    plan=json.loads((ART/'plan.json').read_text())
    for name,digest in plan['input_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/plan['source_snapshot']/name)==digest,name
    files=['review_page.json','capture_zero.json','capture_blue.json','final_zero.json','final_blue.json',
        'reconstruction.json','states.json','model_decisions.jsonl','card_id_mapping.json','reconstruction_checks.json',
        'splendor_cards.csv','splendor_nobles.csv','splendor_config.json']
    hashes={name:sha(RAW/name) for name in files}
    (RAW/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(RAW/'replay_backup.zip','w',compression=zipfile.ZIP_DEFLATED) as backup:
        for name in files+['SHA256SUMS.json']:backup.write(RAW/name,name)
    with zipfile.ZipFile(RAW/'replay_backup.zip') as backup:assert backup.testzip() is None
    result=dict(status='passed',both_perspective_public_events_equal=True,both_final_boards_match_engine=True,
        actor_safe_requests=86,opponent_hidden_card_swap_invariance=True,search_roots=71,total_simulations=71*4096,
        four_equal_1024_world_budgets=True,world_normalization=True,prior_agreement=True,source_and_input_sha_verified=True,
        backup=str((RAW/'replay_backup.zip').relative_to(ROOT)),backup_sha256=sha(RAW/'replay_backup.zip'),
        raw_files={name:dict(sha256=digest,bytes=(RAW/name).stat().st_size) for name,digest in hashes.items()})
    (ART/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_files'}),flush=True)


if __name__=='__main__':main()
