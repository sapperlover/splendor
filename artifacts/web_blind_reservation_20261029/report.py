"""Verify every saved counterfactual trace and build a compact Chinese report."""
import json
from pathlib import Path
import statistics
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
import analyze as a
from splendor.alphazero_recording import read_json,write_json,sha256
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.model import DecisionPhase
from splendor.replay import _decode_state,_decode_action,state_digest

catalog=load_default_catalog()
a.identity_check()
summary=read_json(a.ART/'summary.json')
rows=[read_json(p) for p in sorted((a.RAW/'branches').glob('*.json'))]
assert len(rows)==41 and all(r['finished'] for r in rows)
checked=0
for row in rows:
 engine=a.fast(_decode_state(row['initial_state']),catalog)
 for step in row['trace']:
  request=engine.decision()
  action=_decode_action(step['action'])
  assert isinstance(request,DecisionRequest)
  assert engine.current_player==step['actor']
  assert request.observation.own_completed_turns+1==step['turn']
  assert action in request.legal_actions
  if step['actor']==1:
   assert step['search']['chosen']==step['action']
   if len(request.legal_actions)>1:assert sum(step['search']['visits'])==4096
  engine.apply(action)
  checked+=1
 engine.decision()
 assert state_digest(engine._debug_state())==state_digest(_decode_state(row['final_state'])),row['id']
 assert row['main_actions']==[{k:s[k] for k in ('turn','action')}|{'draw':s.get('draw')} for s in row['trace'] if s['actor']==1 and s['phase']=='main_action']
first=[r for r in rows if r['number']==1]
assert len(first)==15
bad=[r for r in first if r['group']=='other']
good=[r for r in first if r['group']!='other']
assert len(bad)==12 and all(r['main_actions'][0]['action']=={'__type__':'BuyReserved','card_id':'045'} for r in bad)
values=read_json(a.ART/'first_blind_values.json')
rng=read_json(a.ART/'rng_audit.json')
seeds=read_json(a.ART/'seed_roots.json')
score=lambda v:(min(1,max(-1,v))+1)/2
colors={'white':'白','blue':'蓝','green':'绿','red':'红','black':'黑','gold':'金'}
def card_name(card):
 c=catalog.card(card)
 cost=''.join(f'{getattr(c.cost,x)}{label}' for x,label in colors.items() if x!='gold' and getattr(c.cost,x))
 return f'{cost}·{c.points}分（{colors[c.bonus.value]}加成）'
def action_name(raw):
 kind=raw['__type__']
 if kind in ('BuyMarket','BuyReserved','ReserveMarket'):
  verb={'BuyMarket':'买','BuyReserved':'买已留','ReserveMarket':'明留'}[kind]
  return verb+raw['card_id']
 if kind=='ReserveDeck':return '盲留'+str(raw['tier'])+'级'
 if kind=='TakeDistinct':return '拿'+''.join(colors[x] for x in raw['colors'])
 if kind=='TakePair':return '拿2'+colors[raw['color']]
 return kind
lines=['# 本地人机一局：三级盲留与反事实续演（2026-10-29）','',
'固定policy在线第9轮、S1024N4（每世界1024×4，总4096）、seed0、无根噪声。原局seed100，人类先手，双方15分，人类购牌10张、AI12张，人类获胜。',
'', '## 先说结论','',
'第一次盲留没有抽到三张白色主费用好牌时，12/12分支下一手都先买已有的5白二级045，随后多数转向绿红黑筹码、黑色加成和7黑目标，并未继续照着7白计划走。白色好抽显著提高它自己的value；但四世界采样恰好偏向好抽，影响了抽牌前的选点。后两次实际盲留牌均未买，拿到的黄金与抢终局的节奏比暗牌本身更关键。',
'', '## 模型自己的估计','',
'比较时点：固定盲留后人类的一次实盘应手（拿绿红黑并归还1白），到AI下一次主行动；输入为AI自己的Observation，包含新暗牌。不会在对手回合把AI观测冒充一个训练内的当前行动者观测，也不会把人类视角value取反。下表是clip(value,−1,1)换算的估计得分率，平局计半分，未经真实胜率校准。',
'', '| 情况 | 估计得分率 |','|---|---:|',
 f"| 盲留前 | {values['preblind_score_rate']:.2%} |"]
for card in ('077','076','078'):
 r=next(r for r in first if r['draw']==card)
 lines.append(f"| 抽到{card_name(card)} | {score(r['initial_value']):.2%} |")
lines += [f"| 未中三张好白，12种等权平均 | {statistics.mean(score(r['initial_value']) for r in bad):.2%} |",f"| 三张好白，3种等权平均 | {statistics.mean(score(r['initial_value']) for r in good):.2%} |",f"| 全部15种等权平均 | {statistics.mean(score(r['initial_value']) for r in first):.2%} |",'',
'15种均值与盲留前value非常接近，是这个局面下局部预期一致的证据，不能据此宣称value已全局校准。未中组三张好白的范围为47.15%–67.11%；最高是082（7黑3白、5分），其余11张为47.15%–56.64%。',
'', '## 第一次：所有15种结果','',
'第一次在AI第14回合，三白加成、两白筹码、一金；5白二级045已能买，银行白筹码为0。盲留再拿1金。下表后续使用同模型greedy代理人类（从固定的一次实盘应手之后开始），AI保持S1024N4；每种牌只续演一条确定性分支，最多12个AI回合。这些比分是示例路径结果，不是反事实真实胜率，也不是正式棋力评测。',
'', '| 暗牌 | AI自己的估计 | 前三次主行动 | 本次暗牌买了？ | AI:代理对手／状态 |','|---|---:|---|---|---|']
for r in first:
 end='终局' if r['terminal'] else '满12回合，未终局'
 lines.append(f"| {r['draw']} {card_name(r['draw'])} | {score(r['initial_value']):.2%} | {' → '.join(action_name(m['action']) for m in r['main_actions'][:3])} | {'是' if r['blind_card_bought'] else '否'} | {r['final_scores'][1]}:{r['final_scores'][0]}，{end} |")
lines += ['',
'未命中12种中，074／081／082在续演内买下了最初的暗牌，另外9种没有买；多条未买暗牌的分支仍通过别的目标取分，不能把一次盲留落空等同于必输。首回合改买045的行为在全部12种未命中结果中一致。',
'', '## 第二、三次实际发生了什么','',
'- 第19回合盲留071（3白3蓝5绿3红、3分），没有买。该步黄金从1变2；下一回合买069（5黑、2分、红加成），用2黑2金支付。暗牌落空后，拿金仍立即兑现到了红色路线。',
'- 第23回合盲留081（3白3红6黑、4分），也没有买，随后买035、038、070，凑到白4红4贵族N12，最终15分但因购牌数多而输。',
'- 一个已逐步验证的合法对照：跳过第三次盲留，第23回合直接买035，再买038、070；若人类仍走实盘的两次应手，AI会在第25回合15:10结束。此处人类应手被固定，且AI着法是指定的合法线，不能说这是对所有应对的必胜或网络必然会选的线；它说明第三次盲留确实可能付出关键的一回合代价。',
'', '其他两次的全部14／12种抽牌与至多6个AI回合续演见summary.json（第三次还包含从AI视角仍可能在牌堆中的、人类暗持086；该分支同步交换隐藏身份以保持AI抽牌前信息完全相同）。',
'', '## 随机数与四世界抽样','',
'四个世界分别洗牌，牌顶可以重复，并非从15张里不放回选4张。第一次的原生实际牌顶为076、082、076、076；四个64位seed互不相同，且不使用真实游戏seed100。',
'', '| 事件 | 理论概率 | 连续20万组seed | 随机64位20万组seed |','|---|---:|---:|---:|']
for key,label in [('any_card_at_least3','任意同一牌至少3次'),('specific076_at_least3','指定076至少3次')]:
 lines.append(f"| {label} | {rng['theory'][key]:.4%} | {rng['samples'][0][key]/200000:.4%} | {rng['samples'][1][key]/200000:.4%} |")
lines += ['',
'原生C++、Python重建及独立向量计算在256组seed、共1024个世界上一致；抽样统计未发现明显异常。此检查不能构成对PRNG所有性质的证明，但不支持本次重复由实现错误造成。Fisher–Yates洗牌、无偏randbelow拒绝采样及世界seed派生已核对。',
'',
'第一次原始CPU根搜索：网络prior更偏向买045（35.31%），三级盲留仅10.56%；搜索后盲留1861/4096次访问，买045为1097次。4世界对盲留的访问分别为891、40、874、56，两棵偏好盲留的树占了大多数支持。',
'',
'保持同一个局面只改变世界seed，32组S1024N4搜索得到25次买045、6次盲留、1次拿绿红黑。没有白色好抽的9组、只有一张白色好抽的13组均未盲留；盲留的6组全部至少有两个白色好抽。该组是局部seed敏感性诊断，不把32组小样本当作随机事件的总体发生率。',
'', '## 执行与证据','',
'- 原始3个CPU根搜索均复现了实盘盲留；GPU FP32短核对也复现了这3个选点，但CPU／GPU批处理数值差异可改变更深的续演。',
'- 11条已完成的第三次分支保留CPU结果，其余30条使用GPU FP32、禁用TF32；每条分支记录device。CPU首轮进程被环境SIGTERM，已保留结果；恢复后每轮落盘，不把未完成的旧轨迹拼入新轨迹。',
'- 完成41条分支；逐个动作校验合法性并重建最终完整状态。原始根、逐世界访问和逐步续演位于runtime/web_blind_reservation_20261029/，不进Git；summary.json保存摘要及原件位置。',
'- plan.json冻结输入SHA、checkpoint、规则指纹、初始源码；后续可恢复驱动保存在runtime的drivers/，每次执行另存命令、驱动SHA及设备。rng_audit.json、seed_roots.json、first_blind_values.json、tempo_control.json分别保存随机数、换seed搜索、value及节奏对照。',
'']
(a.ART/'results.md').write_text('\n'.join(lines))
verification=dict(branches=41,trace_actions_checked=checked,all_trace_actions_legal=True,all_final_states_reproduced=True,first_blind_bad_draws=12,all_bad_draws_buy045_next=True,devices=dict(Counter(r.get('device','cpu') for r in rows)),input_replay_sha256=sha256(a.REPLAY),model_sha256=sha256(a.MODEL),summary_sha256=sha256(a.ART/'summary.json'),report_sha256=sha256(a.ART/'results.md'),driver_sha256=sha256(Path(__file__)))
write_json(a.ART/'verification.json',verification)
print(json.dumps(verification,ensure_ascii=False))
