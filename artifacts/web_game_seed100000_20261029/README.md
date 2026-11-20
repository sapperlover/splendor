# 本地seed100000：漏防公开的两步终局

2026-10-29追加[077候选、逐世界补牌及同预算搜索复核](search_reliability.md)：077被搜索236次；保留046的1134次访问中808次来自一个能补出050并反击获胜的假想世界。实际牌序的强制结束证明仍成立，但解释需加入未来补牌采样的影响。相同4096总预算的N16、N32及既有native_chance单局面诊断均选077，不直接作为替换配置或棋力结论。

原局`runtime/web_replays/6c2654de6ed04a1db9b77da7131cf2c3.json`，seed100000、人类先手，76个引擎事件、双方各27回合。终局人类15分11卡、AI13分9卡，人类胜。回放SHA：`902877efa6178f795eed21b88cf429062efdb9e278f246ed7fb5ff38da2ede78`。

结论：决定性的失误发生在AI第25回合。双方11分时，AI保留046（6黑3分），让人类第26回合明保留077（6白3蓝3黑、4分），获得黄金后第27回合买下直接到15。实盘牌序下，人类保留077之后，AI剩余两回合已无合法防守。这个局部失败无需用对手暗牌或抽中三级好牌解释。

## 关键账目

人类第25回合买053之后，已有白2、蓝1、绿1、红4、黑2折扣，11分，筹码白3蓝2红1黑2、黄金0。077一直是公开市场牌：原价白6蓝3黑3，折扣后需白4蓝2黑1。仅缺1白，明保留赠送的1黄金恰好补足；所以“保留077→买077”是已有公开资源足够完成的两步路线。

AI同期11分，折扣白1蓝1绿1红3黑2，筹码绿1黑2金1。第25回合保留046后金增至2；第26回合再保留081后金增至3。081需要白2黑4，现有黑2金3仍差1枚，无法在最后回合买下。最后实际买055只到13分；改买046虽可到14分，也仍然输。

因此第26回合再保留081不是形成败局的第一步：人类已经拿走077，败局已无法挽回。第25回合才是应当优先检查的防守节点。

## 精确终局复核

[endgame.py](endgame.py)从实盘人类第26回合已保留077、即AI第26回合开始的状态，枚举AI第26、27回合全部主行动、支付（含额外用金）、退币和贵族选择；人类第27回合固定购买077并使用实盘合法付款。

- AI第26回合有29种完整行动结果。
- 共1088个完整终局行动结果，全部人类胜。
- AI最高能到14分，例如第26回合拿白蓝绿、第27回合买046，结果人类15:AI14。

这是实际剩余牌序下对这条人类结束线的完整防守枚举，不依赖神经网络估值；不是对所有未见牌序的证明。原件见[forced_finish.json](forced_finish.json)。

## 同模型搜索复现与防守对照

按会话中此前固定的人机模型：policy在线第9轮、checkpoint SHA `852cb0420e94d77c146bc2330600f48d52f638e4cca96a5c1578e9dfe54940ec`，CPU FP32、S1024N4（每世界1024、4世界、总4096）、seed0、无根噪声。冻结源码身份沿用[前次plan](../web_blind_reservation_20261029/plan.json)。回放只保存规则状态／动作，本身不含逐步模型及搜索预算；本次9个关键根均复现实盘选点，支持该诊断配置与实盘选择一致，不把它当作逐步配置审计。

| 时点 | 实盘／复现搜索选择 | 搜索访问与Q |
|---|---|---|
| AI25 | 保留046 | 1134/4096，Q +0.392 |
| AI25候选 | 保留077防守 | 236/4096，Q -0.225 |
| 人类26 | 保留077 | 1865/4096，Q +0.831 |
| AI26 | 保留081 | 540/4096，Q -0.949 |
| 人类27 | 买077 | 4084/4096，Q +0.997 |
| AI27 | 买055 | 788/4096，Q -1.000 |

Q为各世界访问加权的边值，不是校准胜率。搜索到AI第26回合已经能判断严重落后；错在上一手未提前识别人类的公开结束路线，而非到第27回合还有必胜牌却没买。

从AI25开始固定第一步，其后双方均用同模型S1024N4、独立私人`DecisionRequest`搜索应对：

1. 保留046：续演完整复现实盘末段，人类26保留077、27买077，AI27买055，15:13人类胜。
2. 改保留077：人类26买054、27保留046、28买018并获贵族到15；AI26保留020、27买055、28买068也到15，卡数人类12、AI10，AI获胜。

第二条支持“提前阻断077有可行抵抗机会”，不证明AI对任意人类应对必胜。两个指定首步及其后16个续演动作、完整终态重放检查通过，见[verification.json](verification.json)；原始树向量和完整分支在`runtime/web_game_seed100000_20261029/`。

还把AI25预算提高至S4096N4（每世界4096、4世界、总16384）：选择改为保留081（4209访问），保留077升至3829访问，实盘046降至1946访问。预算增加改变了选择、降低了对046的乐观评估，但仍未选077；不能据此承诺简单增加4倍模拟就能修好该问题。见[alternatives.json](alternatives.json)。

## 网络估值误差与运气的范围

双方都没有三级盲留。AI三次一级盲留分别是011（蓝0分）、027（绿0分）、039（红0分），三张都买了：011第5回合，027第14回合，039第23回合。它们参与了AI折扣路线；不能直接把三次一级盲留判为浪费。人类全局没有盲留。到上述终盘双方保留均公开，暗牌分配不是此处遗漏077的解释。

前中期AI自身网络估计大多低于50%；第23回合62.3%、第24回合67.4%、第25回合80.5%、第26回合79.1%。这些是自己行动前的`(clip(V,-1,1)+1)/2`未校准估计得分率。第26回合实际已被强制结束，原始网络仍严重乐观，而搜索Q已接近-1；应明确区分网络value与搜索判断。

人类第26回合原始网络估计也只有35.2%，greedy首选保留081，而正式搜索能改选077；第27回合买077的原始估计也仅64.8%。因此网络对这类“保留拿金、隔一手结束”的速度判断偏差，是本局直接可见的问题。AI25内部没有给这条反制足够可靠的搜索评估，原因与策略先验、叶子估值和有限搜索分配均可能有关；本次未逐树做归因消融，不能精确归咎于某一项。

整盘当然仍有初始牌面、补牌和一级盲留的随机性；本次结论限于决定性终盘失误：077当时可见、资源可核算，AI未挡住其两步结束，而用户选择正确。

## 复现

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_game_seed100000_20261029/analyze.py curve
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_game_seed100000_20261029/analyze.py roots --events 41 44 57 65 69 71 74
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_game_seed100000_20261029/analyze.py roots --events 70 72
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_game_seed100000_20261029/endgame.py
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_game_seed100000_20261029/alternatives.py
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src python artifacts/web_game_seed100000_20261029/verify.py
```
