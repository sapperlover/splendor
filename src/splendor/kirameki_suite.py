"""Run the requested casual website profiles serially and summarize them."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import csv
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback


# name, mode, simulations per world, within-tree leaf batch, backend, worlds
GROUPS=(('greedy','greedy',1,1,'native_chance',1),('s128_b1','mcts',128,1,'native_chance',1),
        ('s1024_b1','mcts',1024,1,'native_chance',1),('s8192_b8','mcts',8192,8,'native_chance',1),
        ('s131072_b64','mcts',131072,64,'native_chance',1),('s8192_b1','mcts',8192,1,'native_chance',1),
        ('native_s1024_n1','mcts',1024,1,'native',1),('native_s256_n4','mcts',256,1,'native',4))


def commands(args):
    from .kirameki_live import DEFAULT_CHECKPOINT
    checkpoint=args.checkpoint or DEFAULT_CHECKPOINT
    rows=[]
    for name,mode,budget,batch,backend,worlds in GROUPS:
        command=[sys.executable,'-u','-m','splendor.kirameki_autoplay','--checkpoint',str(checkpoint),
                 '--mode',mode,'--simulations-per-world',str(budget),'--leaf-batch-size',str(batch),
                 '--search-backend',backend,'--determinizations',str(worlds),'--search-workers','1',
                 '--device','cuda','--inference-mode','compiled','--compile-mode','native-cudagraph',
                 '--root-selection','visits','--seed','0','--games',str(args.games_per_group),
                 '--record-progress','--clock','unlimited','--resume','--fresh-games-only',
                 '--opponent-level','bga1000-v5','--output',str(args.output/name)]
        if args.profile:command+=['--profile',str(args.profile)]
        if args.cdp:command+=['--cdp',args.cdp]
        rows.append({'name':name,'mode':mode,'budget':0 if mode=='greedy' else budget*worlds,
                     'leaf_batch_size':batch,'command':command})
    return rows


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def completed_games(path):
    assignments=path/'assigned_games.json'
    ids=json.loads(assignments.read_text()) if assignments.exists() else []
    out=[]
    for game in ids:
        file=path/'games'/game/'result.json'
        if file.exists():out.append(json.loads(file.read_text()))
    return out


def wilson(wins,n):
    if not n:return None
    z=1.959963984540054;p=wins/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half),min(1.,center+half)]


def build_report(root):
    groups=[];game_rows=[];details=[]
    for name,mode,budget,batch,backend,worlds in GROUPS:
        path=root/name;games=completed_games(path);counts={k:sum(g['outcome']==k for g in games) for k in ('win','draw','loss')}
        n=len(games);seconds=[]
        for g in games:
            directory=path/'games'/g['game_id'];local=jsonl(directory/'local_evaluations.jsonl')
            site=jsonl(directory/'site_evaluations.jsonl');display=jsonl(directory/'displayed_bar.jsonl')
            times=[r['search_seconds'] for r in local];seconds+=times
            row={'group':name,'game_id':g['game_id'],'human_seat':g['human_seat'],'outcome':g['outcome'],
                 'points_p0':g['points'][0],'points_p1':g['points'][1],'plies':len(g['moves']),
                 'local_decisions':len(local),'site_evaluations':len(site),'search_seconds':sum(times)}
            game_rows.append(row)
            details.append({**row,'moves':g['moves'],
                'local':[[r['ply'],r.get('decision_rate'),r.get('network_rate')] for r in local],
                'site':[[r['ply'],r['viewer_rate']] for r in site],
                'display':[[r['ply'],r['p0'] if g['human_seat']==0 else r['p1']] for r in display]})
        groups.append({'name':name,'mode':mode,'simulations':0 if mode=='greedy' else budget*worlds,'leaf_batch_size':batch,
                       'search_backend':backend,'determinizations':worlds,
                       'simulations_per_world':0 if mode=='greedy' else budget,
                       'selection_rule':'visits' if mode=='mcts' else 'policy_argmax',
                       'completed_games':n,**counts,'win_rate':counts['win']/n if n else None,
                       'score_rate':(counts['win']+.5*counts['draw'])/n if n else None,
                       'win_rate_95ci':wilson(counts['win'],n),
                       'seats':{str(seat):{k:sum(g['human_seat']==seat and g['outcome']==k for g in games)
                                          for k in ('win','draw','loss')} for seat in (0,1)},
                       'mean_recommendation_seconds':sum(seconds)/len(seconds) if seconds else None})
    root.mkdir(parents=True,exist_ok=True)
    summary={'schema':'kirameki-profile-suite-v1','groups':groups,'completed_games':len(game_rows),
             'win_rate_definition':'wins / games','score_rate_definition':'(wins + draws / 2) / games',
             'sampling':'independent website random-seat games, not paired seeds',
             'site_value_source':'existing site evaluate replies; aligned by public move prefix at request'}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['group','game_id','human_seat','outcome','points_p0','points_p1','plies','local_decisions','site_evaluations','search_seconds']
    with (root/'games.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(game_rows)
    table=[]
    for g in groups:
        fmt=lambda x:'—' if x is None else f'{x:.1%}'
        interval='—' if g['win_rate_95ci'] is None else '–'.join(fmt(x) for x in g['win_rate_95ci'])
        table.append(f'<tr><td>{g["name"]}</td><td>{g["completed_games"]}</td><td>{g["win"]}/{g["draw"]}/{g["loss"]}</td>'
                     f'<td>{fmt(g["win_rate"])}</td><td>{fmt(g["score_rate"])}</td><td>{interval}</td></tr>')
    data=json.dumps(details,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page=r'''<!doctype html><meta charset="utf-8"><title>Kirameki 对弈套件</title>
<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 18px;color:#182330}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}select{padding:8px;max-width:100%}svg{width:100%;background:#fafbfc}#moves{white-space:pre-wrap;line-height:1.7}p{color:#52606d}</style>
<h1>Kirameki 对弈套件</h1><p>每局随机先后手、非排位宝神、不限时。胜率与和棋折半得分率分开统计；95%区间为胜率Wilson区间。</p>
<table><tr><th>配置</th><th>已完成</th><th>胜/和/负</th><th>胜率</th><th>得分率</th><th>胜率95%区间</th></tr>TABLE</table>
<h2>逐局走势与棋谱</h2><select id="game"></select><p>蓝：本地推荐动作估值；绿：本地网络V；红：网站胜率条的原始估值。均为本人视角，和棋折半。横轴为已完成网站动作数；支付／归还条件估值在原建议JSON中另存。</p>
<p>网站屏幕显示可能滞后。下图优先使用网站原有evaluate调用回复，并按调用时的公开棋谱定位；若未捕获到则显示灰色的屏幕采样值，其位置对应关系未经确认。曲线差异不等于校准后的胜率差异。</p>
<svg id="chart" viewBox="0 0 1000 320"></svg><p id="info"></p><details><summary>完整网站USI棋谱</summary><div id="moves"></div></details>
<script>const games=DATA;const select=document.getElementById('game');for(let i=0;i<games.length;i++){let g=games[i],o=document.createElement('option');o.value=i;o.textContent=`${g.group} · ${g.game_id} · P${g.human_seat} · ${g.outcome} · ${g.points_p0}:${g.points_p1}`;select.append(o);}
function draw(){const g=games[+select.value];if(!g)return;const svg=document.getElementById('chart'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();const add=(tag,a)=>{let e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);svg.append(e);return e;};
for(let p of[0,.25,.5,.75,1]){let y=290-p*270;add('line',{x1:45,x2:980,y1:y,y2:y,stroke:'#ddd'});add('text',{x:2,y:y+5,'font-size':13}).textContent=`${p*100}%`;}
const max=Math.max(1,g.plies);const series=[['#2469d7',g.local.map(x=>[x[0],x[1]])],['#23934c',g.local.map(x=>[x[0],x[2]])],['#d74745',g.site.length?g.site:g.display]];
for(const[color,rows]of series){const points=rows.filter(x=>x[1]!==null).sort((a,b)=>a[0]-b[0]).map(x=>`${45+935*x[0]/max},${290-270*x[1]}`).join(' ');add('polyline',{points,fill:'none',stroke:color==='#d74745'&&!g.site.length?'#999':color,'stroke-width':2});}
document.getElementById('info').textContent=`${g.plies}条网站动作；${g.local_decisions}次本地决策；${g.site_evaluations}条网站估值；本地推荐累计${g.search_seconds.toFixed(1)}秒。完整安全盘面保存在该局observations.jsonl.gz。`;
document.getElementById('moves').textContent=g.moves.map((m,i)=>`${i+1}. ${m}`).join('\n');}select.onchange=draw;draw();</script>'''
    (root/'report.html').write_text(page.replace('TABLE',''.join(table)).replace('DATA',data),encoding='utf-8')
    return summary


class _ConsoleLog:
    def __init__(self,terminal,log):self.terminal=terminal;self.log=log
    def write(self,text):
        self.terminal.write(text);self.log.write(text);self.log.flush()
        return len(text)
    def flush(self):self.terminal.flush();self.log.flush()
    def __getattr__(self,name):return getattr(self.terminal,name)


async def run_groups(args,rows):
    from .kirameki_autoplay import browser_session, watch_autoplay
    from .kirameki_live import create_advisor, parse_args
    pending=[]
    for row in rows:
        done=len(completed_games(args.output/row['name']))
        if done>=args.games_per_group:
            print(f'{row["name"]}已完成{done}局，跳过。',flush=True)
        else:pending.append(row)
    if not pending:return
    first=parse_args(pending[0]['command'][4:],autoplay=True)
    # Hold the one permission-based CDP session across model warmup, every game,
    # and all profile switches. Per-group observers still clean up their hooks.
    async with browser_session(first) as browser:
        print('全部组共用当前调试连接；切换预算时不重新连接浏览器。',flush=True)
        for row in pending:
            stage=parse_args(row['command'][4:],autoplay=True)
            stage.output.mkdir(exist_ok=True)
            done=len(completed_games(stage.output))
            print(f'启动{row["name"]}：已有{done}/{args.games_per_group}局。',flush=True)
            with (stage.output/'console.log').open('a',encoding='utf-8') as log:
                with redirect_stdout(_ConsoleLog(sys.stdout,log)),redirect_stderr(_ConsoleLog(sys.stderr,log)):
                    advisor=None
                    try:
                        advisor=create_advisor(stage,autoplay=True)
                        await watch_autoplay(stage,advisor,browser=browser,
                                             on_game_finished=lambda:build_report(args.output))
                    except Exception:
                        traceback.print_exc()
                        raise
                    finally:
                        if advisor is not None and advisor.search_policy is not None:advisor.search_policy.close()
                        advisor=None
                        gc.collect()
                        import torch
                        if stage.device.startswith('cuda'):torch.cuda.empty_cache()
            build_report(args.output)
            if len(completed_games(stage.output))!=args.games_per_group:
                raise RuntimeError(f'{row["name"]}未完成指定局数，停止后续组。')


def prepare_plan(args,rows):
    from .kirameki_autoplay import upgrade_connection_sources
    from .kirameki_live import DEFAULT_CHECKPOINT
    checkpoint=(args.checkpoint or DEFAULT_CHECKPOINT).resolve()
    plan={'groups':rows,'games_per_group':args.games_per_group,'checkpoint':str(checkpoint),
          'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('kirameki_suite.py','kirameki_autoplay.py','kirameki_recording.py',
                                        'kirameki_live.py','parallel_recommend.py','native_policy.py',
                                        'native_search.cpp','search_inference.py','recommend.py')}}
    manifest=args.output/'suite.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old!=plan:
            # This requested append changes only the suite entry point. The
            # currently running six-profile process keeps its frozen plan; a
            # later launch may extend that exact plan without changing a game.
            extension=(old.get('groups')==plan['groups'][:-2]
                and old.get('source_sha256',{}).get('kirameki_suite.py')=='b31daaf9da14b85738e7644f2f42d9aeb202a26247f377c8a8de63d11aa11d74'
                and {k:v for k,v in old.items() if k not in ('groups','source_sha256')}==
                    {k:v for k,v in plan.items() if k not in ('groups','source_sha256')}
                and {k:v for k,v in old.get('source_sha256',{}).items() if k!='kirameki_suite.py'}==
                    {k:v for k,v in plan['source_sha256'].items() if k!='kirameki_suite.py'})
            matching={k:v for k,v in old.items() if k!='source_sha256'}=={k:v for k,v in plan.items() if k!='source_sha256'}
            if extension:
                backup=manifest.with_name('suite.before_native_worlds.json')
                with backup.open('x',encoding='utf-8') as stream:stream.write(manifest.read_text())
                print('已在原六组后追加两组独立世界对照；原计划备份：'+str(backup),flush=True)
            elif not matching or not upgrade_connection_sources(manifest,old,plan['source_sha256']):
                raise SystemExit('套件配置与已保存计划不一致，请使用新的输出目录')
    manifest.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description='八组各20局Kirameki非排位自动对弈；串行、可恢复。')
    parser.add_argument('--output',type=Path,default=Path('runtime/kirameki_suite_20261031'))
    parser.add_argument('--games-per-group',type=int,default=20)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--cdp')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args(argv)
    if args.games_per_group<1:parser.error('每组局数必须为正')
    if args.report_only:build_report(args.output);return
    rows=commands(args)
    if args.dry_run:
        print(json.dumps({'games_per_group':args.games_per_group,'groups':rows},ensure_ascii=False,indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'suite.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('该输出目录已有套件进程运行') from None
        prepare_plan(args,rows)
        try:
            asyncio.run(run_groups(args,rows))
        except KeyboardInterrupt:
            print('\n已停止套件，浏览器和棋局保持打开；重跑同一命令恢复。',flush=True)
            return
        finally:build_report(args.output)
    print('全部组均完成。报告：'+str(args.output/'report.html'),flush=True)


if __name__=='__main__':main()
