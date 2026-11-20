"""Fixed-profile orchestration, recorder boundaries and resumable statistics."""
import asyncio
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from splendor.kirameki_autoplay import (prepare_session, recover_previous, upgrade_connection_sources,
                                        write_summary, _BEFORE_SHARED_CONNECTION)
from splendor.kirameki_recording import CLEANUP_RECORDER_JS, INSTALL_RECORDER_JS, ProgressRecorder
from splendor.kirameki_suite import GROUPS, build_report, commands, completed_games, wilson
from test_kirameki_autoplay import run_node, sample


def test_requested_eight_groups_have_exact_budgets_and_twenty_games(tmp_path):
    args=SimpleNamespace(checkpoint=None,games_per_group=20,output=tmp_path,profile=None,cdp=None)
    rows=commands(args)
    assert [(r['mode'],r['budget'],r['leaf_batch_size']) for r in rows]==[
        ('greedy',0,1),('mcts',128,1),('mcts',1024,1),('mcts',8192,8),('mcts',131072,64),('mcts',8192,1),
        ('mcts',1024,1),('mcts',1024,1)]
    for row in rows:
        cmd=row['command']
        assert cmd[cmd.index('--games')+1]=='20'
        assert cmd[cmd.index('--determinizations')+1]==('4' if row['name']=='native_s256_n4' else '1')
        assert cmd[cmd.index('--search-backend')+1]==('native' if row['name'].startswith('native_') else 'native_chance')
        assert cmd[cmd.index('--root-selection')+1]=='visits'
        assert cmd[cmd.index('--clock')+1]=='unlimited'
        assert '--record-progress' in cmd and '--resume' in cmd and '--fresh-games-only' in cmd
    assert rows[-1]['command'][rows[-1]['command'].index('--simulations-per-world')+1]=='256'


def test_statistics_use_assigned_completed_games_and_keep_draws_separate(tmp_path):
    stage=tmp_path/'greedy';stage.mkdir()
    ids=['casual-1','casual-2','casual-3']
    (stage/'assigned_games.json').write_text(json.dumps(ids))
    for i,outcome in enumerate(('win','draw','loss','win'),1):
        path=stage/'games'/f'casual-{i}';path.mkdir(parents=True)
        result={'game_id':f'casual-{i}','human_seat':i%2,'outcome':outcome,'points':[15,14],
                'moves':['take:WUG','reserve:C3']}
        (path/'result.json').write_text(json.dumps(result))
    assert len(completed_games(stage))==3  # unassigned previous terminal excluded
    summary=build_report(tmp_path);g=summary['groups'][0]
    assert g['completed_games']==3 and g['win_rate']==pytest.approx(1/3) and g['score_rate']==.5
    assert g['win_rate_95ci'][0]<1/3<g['win_rate_95ci'][1]
    assert len((tmp_path/'games.csv').read_text().splitlines())==4
    page=(tmp_path/'report.html').read_text()
    assert 'casual-4' not in page and 'casual-3' in page
    script=page.split('<script>',1)[1].split('</script>',1)[0]
    subprocess.run(['node','--check'],input=script,text=True,capture_output=True,check=True)
    assert len(summary['groups'])==8 and wilson(0,0) is None
    last=summary['groups'][-1]
    assert (last['search_backend'],last['simulations_per_world'],last['determinizations'],last['simulations'])==('native',256,4,1024)


def test_resume_restores_completed_results_and_rejects_changed_search(tmp_path):
    args=SimpleNamespace(output=tmp_path,resume=True,mode='mcts',simulations_per_world=128,
                         determinizations=1,leaf_batch_size=1,seed=0,record_progress=False)
    advisor=SimpleNamespace(checkpoint_sha256='fixed-model')
    assert prepare_session(args,advisor)==(set(),{})
    (tmp_path/'assigned_games.json').write_text('["casual-1"]')
    path=tmp_path/'games'/'casual-1';path.mkdir(parents=True)
    result={'game_id':'casual-1','outcome':'win'}
    (path/'result.json').write_text(json.dumps(result))
    assigned,finished=prepare_session(args,advisor)
    assert assigned=={'casual-1'} and finished=={'casual-1':result}
    assert write_summary(tmp_path,finished)=={'win':1,'draw':0,'loss':0}
    old=(tmp_path/'session.json').read_text();args.simulations_per_world=1024
    with pytest.raises(ValueError,match='simulations_per_world'):prepare_session(args,advisor)
    assert (tmp_path/'session.json').read_text()==old


def test_resume_recovers_purchase_source_only_from_confirmed_matching_prefix(monkeypatch,tmp_path):
    import splendor.kirameki_autoplay as auto
    before=sample();before['moves']=['take:WUG'];now=deepcopy(before);now['moves']+=['buy:C3','noble:N0']
    path=tmp_path/'decisions'/'one';path.mkdir(parents=True)
    (path/'snapshot.json').write_text(json.dumps(before))
    result={'submitted_action':True,'site_usi':'buy:C3','request':{'observation':{'phase':'main_action'}}}
    (path/'recommendation.json').write_text(json.dumps(result))
    monkeypatch.setattr(auto,'request_from_snapshot',lambda s,c:('restored',s['moves']))
    assert recover_previous(tmp_path,now,None)==('restored',['take:WUG'])
    now['moves'][1]='reserve:C3'
    assert recover_previous(tmp_path,now,None) is None


def test_recorder_masks_secrets_and_preserves_original_promise_and_rate_position():
    s=sample();s['perspective']=s['human_seat']=1
    s['state']['players'][0]['reserved_cards']=[61]
    source='const input='+json.dumps(s)+''';
const record={ranked:false,setup:{startedAt:'2026-10-31T00:00:00Z',humanSeat:1,seed:'SECRET_SEED'},moves:['take:WUG'],token:'SECRET_TOKEN'};
const view={state:input.state,moves:[{usi:'take:WUG'},{usi:'reserve:C1'}],human_seat:1,
 mode:'human-vs-ai',level_ids:['bga1000-v5',null],public_reserved_card_ids:[],hidden_reserved_levels:{'0:0':3},seed:'SECRET_SEED'};
let evalCalls=0;const evalPromise=Promise.resolve({p0:.2}),actionPromise=Promise.resolve(view);
const client={evaluate(){evalCalls++;return evalPromise},humanAction(){return actionPromise},aiAction(){return actionPromise}};
const original=client.evaluate;
const fiber={memoizedState:{memoizedState:{current:client},next:null},return:null};
global.document={querySelector:()=>({'__reactFiber$test':fiber})};
global.localStorage={getItem:()=>JSON.stringify(record)};global.window={};
(async()=>{if(!(INSTALL)())throw Error('not installed');
 const same=client.evaluate()===evalPromise;record.moves.push('reserve:C1');
 await client.humanAction(0);await new Promise(r=>setTimeout(r,0));
 const r=window.__splendorGameRecorder;
 (CLEANUP)();
 process.stdout.write(JSON.stringify({same,evalCalls,pending:r.pending,events:r.events,
 restored:client.evaluate===original,removed:window.__splendorGameRecorder===undefined}));
})().catch(e=>{console.error(e);process.exit(1)});
'''.replace('INSTALL',INSTALL_RECORDER_JS).replace('CLEANUP',CLEANUP_RECORDER_JS)
    r=run_node(source)
    assert r['same'] and r['evalCalls']==1 and r['pending']==0 and r['restored'] and r['removed']
    rate=next(e for e in r['events'] if e['kind']=='site_evaluation')
    assert rate['moves']==['take:WUG'] and rate['p0']==.2
    state=next(e['snapshot'] for e in r['events'] if e['kind']=='action_state')
    assert state['state']['players'][0]['reserved_cards']==[-1]
    assert state['moves']==['take:WUG','reserve:C1'] and 'SECRET' not in json.dumps(r)


def test_recorded_site_rate_orientation_and_compressed_snapshots(tmp_path):
    s=sample();s['perspective']=s['human_seat']=1
    recorder=ProgressRecorder(tmp_path,'bga1000-v5')
    data={'snapshot':s,'at_ms':123,'bar':{'p0':.2,'p1':.8},'dropped':0,'events':[
        {'kind':'site_evaluation','game_id':s['game_id'],'human_seat':1,'p0':.2,'moves':['take:WUG']}]}
    recorder.consume(data);recorder.consume({**data,'events':[]})
    path=tmp_path/'games'/s['game_id']
    with gzip.open(path/'observations.jsonl.gz','rt') as stream:rows=[json.loads(l) for l in stream]
    assert len(rows)==1 and rows[0]['snapshot']==s
    rate=json.loads((path/'site_evaluations.jsonl').read_text())
    assert rate['viewer_rate']==.8 and rate['ply']==1 and rate['position_alignment']=='request_moves'
    assert len((path/'displayed_bar.jsonl').read_text().splitlines())==1
    with pytest.raises(RuntimeError,match='缓冲溢出'):recorder.consume({**data,'dropped':1})


def fake_suite_browser(monkeypatch):
    import splendor.kirameki_autoplay as auto
    import splendor.kirameki_live as live
    calls={'connections':0,'closes':0,'policies_closed':0}
    async def close():calls['closes']+=1
    browser=SimpleNamespace(close=close)
    async def connect(*args):calls['connections']+=1;return browser
    def close_policy():calls['policies_closed']+=1
    monkeypatch.setattr(auto,'connect_browser',connect)
    monkeypatch.setattr(live,'create_advisor',lambda *a,**k:SimpleNamespace(search_policy=SimpleNamespace(close=close_policy)))
    return browser,calls


def test_suite_stops_on_first_group_failure_and_disconnects_once(monkeypatch,tmp_path):
    import splendor.kirameki_suite as suite
    import splendor.kirameki_autoplay as auto
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixed checkpoint')
    browser,calls=fake_suite_browser(monkeypatch)
    launched=[]
    async def watch(args,advisor,**kwargs):
        launched.append(args.output.name)
        assert kwargs['browser'] is browser
        raise RuntimeError('工程失败')
    monkeypatch.setattr(auto,'watch_autoplay',watch)
    output=tmp_path/'suite'
    with pytest.raises(RuntimeError,match='工程失败'):
        suite.main(['--checkpoint',str(checkpoint),'--output',str(output),'--games-per-group','1'])
    assert len(launched)==1 and (output/'report.html').is_file()
    assert calls=={'connections':1,'closes':1,'policies_closed':1}
    assert 'RuntimeError: 工程失败' in (output/'greedy'/'console.log').read_text()


def test_suite_reuses_one_connection_across_all_budgets_and_skips_completed_on_resume(monkeypatch,tmp_path):
    import splendor.kirameki_suite as suite
    import splendor.kirameki_autoplay as auto
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixed checkpoint')
    browser,calls=fake_suite_browser(monkeypatch)
    launched=[]
    async def watch(args,advisor,**kwargs):
        assert kwargs['browser'] is browser
        output=args.output;launched.append(output.name)
        (output/'assigned_games.json').write_text('["casual-1"]')
        path=output/'games'/'casual-1';path.mkdir(parents=True)
        (path/'result.json').write_text(json.dumps({'game_id':'casual-1','human_seat':0,'outcome':'win',
                                                  'points':[15,14],'moves':['take:WUG']}))
        print('终局：win')
        kwargs['on_game_finished']()
        assert json.loads((output.parent/'summary.json').read_text())['completed_games']==len(launched)
    monkeypatch.setattr(auto,'watch_autoplay',watch)
    options=['--checkpoint',str(checkpoint),'--output',str(tmp_path/'suite'),'--games-per-group','1']
    suite.main(options)
    assert launched==[r[0] for r in GROUPS]
    suite.main(options)
    assert len(launched)==8
    assert json.loads((tmp_path/'suite'/'summary.json').read_text())['completed_games']==8
    assert calls=={'connections':1,'closes':1,'policies_closed':8}
    for name,*_ in GROUPS:assert '终局：win' in (tmp_path/'suite'/name/'console.log').read_text()


def test_exact_connection_release_upgrades_sources_and_preserves_original_manifest(tmp_path):
    path=tmp_path/'suite.json'
    old={'source_sha256':{**_BEFORE_SHARED_CONNECTION,'native_search.cpp':'unchanged-search'},'games_per_group':20}
    path.write_text(json.dumps(old));new={**old['source_sha256'],'kirameki_suite.py':'current-suite',
                                      'kirameki_autoplay.py':'current-auto','kirameki_live.py':'current-live'}
    assert not upgrade_connection_sources(path,old,{**new,'native_search.cpp':'different-search'})
    assert not upgrade_connection_sources(path,{**old,'source_sha256':{**old['source_sha256'],'kirameki_live.py':'unknown'}},new)
    assert json.loads(path.read_text())==old
    assert upgrade_connection_sources(path,old,new)
    assert json.loads(path.read_text())['source_sha256']==new
    assert json.loads((tmp_path/'suite.before_shared_connection.json').read_text())==old


def test_session_resumes_known_connection_upgrade_but_keeps_old_config(tmp_path):
    args=SimpleNamespace(output=tmp_path,resume=True,mode='mcts',simulations_per_world=128,
                         determinizations=1,leaf_batch_size=1,seed=0,record_progress=False)
    advisor=SimpleNamespace(checkpoint_sha256='fixed-model')
    prepare_session(args,advisor)
    path=tmp_path/'session.json';old=json.loads(path.read_text())
    for name in ('kirameki_live.py','kirameki_autoplay.py'):old['source_sha256'][name]=_BEFORE_SHARED_CONNECTION[name]
    path.write_text(json.dumps(old))
    assert prepare_session(args,advisor)==(set(),{})
    assert json.loads((tmp_path/'session.before_shared_connection.json').read_text())==old
    assert json.loads(path.read_text())['source_sha256']!=old['source_sha256']


def test_plan_connection_upgrade_preserves_finished_games_and_rejects_budget_change(tmp_path):
    from splendor.kirameki_suite import prepare_plan
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixed checkpoint')
    args=SimpleNamespace(checkpoint=checkpoint,games_per_group=20,output=tmp_path,profile=None,cdp=None)
    prepare_plan(args,commands(args))
    path=tmp_path/'suite.json';old=json.loads(path.read_text())
    old['source_sha256'].update(_BEFORE_SHARED_CONNECTION)
    path.write_text(json.dumps(old))
    assigned=tmp_path/'greedy'/'assigned_games.json';assigned.parent.mkdir();assigned.write_text('["casual-1"]')
    result=assigned.parent/'games'/'casual-1'/'result.json';result.parent.mkdir(parents=True)
    result.write_text('{"game_id":"casual-1","outcome":"win"}')
    before=result.read_bytes()
    prepare_plan(args,commands(args))
    assert json.loads((tmp_path/'suite.before_shared_connection.json').read_text())==old
    assert completed_games(assigned.parent)==[json.loads(before)] and result.read_bytes()==before
    current=path.read_bytes();args.games_per_group=21
    with pytest.raises(SystemExit,match='套件配置'):prepare_plan(args,commands(args))
    assert path.read_bytes()==current


@pytest.mark.parametrize('mutation',[None,'budget','search_source'])
def test_only_exact_six_group_plan_can_append_native_world_profiles(tmp_path,mutation):
    from splendor.kirameki_suite import prepare_plan
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixed checkpoint')
    args=SimpleNamespace(checkpoint=checkpoint,games_per_group=20,output=tmp_path,profile=None,cdp=None)
    rows=commands(args);prepare_plan(args,rows)
    manifest=tmp_path/'suite.json';old=json.loads(manifest.read_text())
    old['groups']=old['groups'][:-2]
    old['source_sha256']['kirameki_suite.py']='b31daaf9da14b85738e7644f2f42d9aeb202a26247f377c8a8de63d11aa11d74'
    if mutation=='budget':old['groups'][0]['budget']=99
    if mutation=='search_source':old['source_sha256']['native_search.cpp']='unknown'
    manifest.write_text(json.dumps(old));before=manifest.read_bytes()
    if mutation:
        with pytest.raises(SystemExit,match='套件配置'):prepare_plan(args,rows)
        assert manifest.read_bytes()==before
        assert not (tmp_path/'suite.before_native_worlds.json').exists()
    else:
        prepare_plan(args,rows)
        assert (tmp_path/'suite.before_native_worlds.json').read_bytes()==before
        assert json.loads(manifest.read_text())['groups']==rows
        prepare_plan(args,rows)  # idempotent resume after the upgrade
