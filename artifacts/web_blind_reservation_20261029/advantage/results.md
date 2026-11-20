# 首次盲留成功后，优势怎样回落

原局、checkpoint、冻结源码沿用[上级plan](../plan.json)。本次重放了全部77个引擎事件，恢复52个主行动局面；只在玩家自己行动时评估其私人视角。下列百分比为`(clip(V,-1,1)+1)/2`得到的未校准估计得分率，不是精确胜率，也不是必胜证明。双方暗牌不同，不能将一方数值直接取反作为另一方的判断。

结论：AI确实获得了很好的白色兑现机会，随后你也抽中高度适配蓝色引擎的5分牌，并通过折扣、筹币和黄金高效兑现。AI没有及时阻断引擎，之后四世界搜索又始终没采中你的真实暗牌，低估了实际的结束速度。现有证据不支持把整盘归因于第三次盲留漏掉一个无条件必胜。

## 实盘数值与行动

AI每次自己主行动之前的网络估计如下；回合数是每人各自的回合数。

| AI回合 | 自己的估计得分率 | 之前发生了什么／本回合行动 |
|---|---:|---|
| 14 | 58.83% | 盲留前，随后抽到076：7白4分 |
| 15 | 81.32% | 已知自己的076；买白色折扣020 |
| 17 | 87.94% | 买076，把白色引擎变成4分 |
| 18 | 91.37% | 买045：5白2分，随后总分7分 |
| 19 | 82.19% | 你买060：1分绿色折扣；AI第二次盲留拿金 |
| 20 | 79.86% | 你已暗留086；AI买069，总分9分 |
| 21 | 65.21% | 你买051：2分、第4张蓝色折扣；AI拿白绿黑 |
| 22 | 55.57% | 你拿蓝绿黑；AI保留080：7黑4分，拿金 |
| 23 | 56.98% | 你买084到10分；AI第三次盲留 |
| 24 | 38.02% | 你保留083拿金；AI买035 |
| 25 | 58.63% | 你继续筹币；AI买038，短暂回升并非真正逆转 |
| 26 | 0.77% | 你买086到15分；AI最后买070＋贵族也到15，但卡数多而输 |

完整双方曲线见[curve.json](curve.json)。你的自身估计在第19回合只有13.63%，第20回合盲留前23.89%；第21—24回合依次44.50%、59.92%、81.78%、93.78%。AI揭示076后，你的估计曾降到12.16%，因此并非只有AI单方面认为白色兑现后局面不错；但这些仍是同一个模型的估计。

## 你实际如何反超

你原有052的3分，之后060得1分、051得2分、084得4分、086得5分，合计15分，没有依赖贵族。060先把绿色折扣升到1；051把蓝色折扣升到4；084再把绿色折扣升到2。这使最后的086从7蓝3绿变成实际只需3蓝1绿。第24回合保留083拿到第二枚黄金，第25回合拿蓝绿黑，最后以1蓝1绿2金支付086。

所以051并非慢买一张普通2分牌：它同时降低了084与086两张后续大牌的蓝色成本。你第21回合买051、第22回合拿蓝绿黑，均被同配置S1024N4搜索复现，分别获得4066/4096、2484/4096访问；不能把这条路线视为模型不认可的人类乱走。

你的盲留也是关键好运。固定第20回合之前的真实状态、AI暗牌071和随后实盘的买069及支付，只把你抽到的牌与当时真实牌堆顶交换，下一次你自己行动时：

- 抽到086：估计44.50%。
- 其余12张平均24.48%，范围20.99%—29.12%。

这是同一公开局面下的私人信息差；13种情况下AI第20回合收到的`DecisionRequest`完全相同。不能声称仅凭你盲留这一步，它就应当知道你拿到了086。详见[knowledge.json](knowledge.json)。

## AI没有充分演算到真实威胁

第20—23回合，AI四个搜索世界分配给你的隐藏三级牌分别是：

| AI回合 | 四世界中的人类暗牌 | 实际086命中数 |
|---|---|---:|
| 20 | 088、079、081、089 | 0/4 |
| 21 | 078、089、075、085 | 0/4 |
| 22 | 079、087、085、085 | 0/4 |
| 23 | 074、087、081、075 | 0/4 |

当前实现只根据当前可见观察均匀重建未知牌，并未根据整段对手历史行动更新暗牌后验。这些搜索根因此没有直接演算“你持有086”的实际世界。S1024N4虽然总计4096次模拟，仍只有4个根世界，增加每世界模拟次数不会自动补上第5种暗牌。以上四次抽样不作为独立统计试验，也不据此认定随机数故障；原生／Python重建一致性已由之前的[RNG审计](../rng_audit.json)验证。

这说明它对真实私有威胁存在盲区；并不能证明只要采到086就一定选出最优防守。

## 哪一步更值得怀疑

第18回合有一个较早的候选转折：AI实盘买045，搜索1833访问；次选保留060有1727访问。060正是你下一回合购买、并用来支撑绿色折扣链的牌。买060本身也合法。

固定该回合真实状态及牌序，强制不同第一步，然后双方都使用相同模型、S1024N4、seed0、无噪声独立应对，得到：

| 强制AI第一步 | 后续结果（人类:AI） | 胜者 |
|---|---|---|
| AI18买045（实盘动作） | 16:15，第26回合 | 人类 |
| AI18保留060 | 9:15，第26回合 | AI |
| AI18买060 | 9:15，第27回合 | AI |
| AI22保留080（实盘动作） | 15:15，第26回合，卡数10:11 | 人类 |
| AI22提前买035 | 15:13，第26回合 | 人类 |
| AI22买047 | 15:10，第26回合 | 人类 |

这是每个动作一条确定性、双方都会调整应对的诊断续演，不是棋力赛、胜率，也不是最强应对证明。第18回合的控制分支后来没有选择实盘盲留086，而是买069；各分支的后续牌序消费也可能不同。它支持“应更重视阻断060及后续051、不能只兑现自己2分”的判断，但无法把单一动作判定为理论败着，或精确分解每步损失多少真实胜率。

第22回合提前买035的续演中，人类第25回合保留070截断红色终局。和[第三次盲留防守复核](../defense/results.md)一致：到了后段，简单把某一次盲留换成红牌，未必就能挽救；实盘最后双方同为15分，AI多买了卡，最终以10张对12张输掉平分裁决。

## 复核与复现

本次6个实际根搜索全部复现实盘选点；6条新续演的动作合法性、完整终态逐条重建，见[verification.json](verification.json)。原始52局面、完整搜索向量和续演状态在`runtime/web_blind_reservation_20261029/advantage/`，不进Git。紧凑结果见[roots](roots_49_55_57_59_60_61.json)、[continuations.json](continuations.json)。全部CPU、FP32、单torch线程，S1024N4为每世界1024次、4世界、总4096；续演共约159秒。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_blind_reservation_20261029/advantage/analyze.py curve
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_blind_reservation_20261029/advantage/analyze.py roots --events 49 55 57 59 60 61
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_blind_reservation_20261029/advantage/knowledge.py
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_blind_reservation_20261029/advantage/continuations.py
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src python artifacts/web_blind_reservation_20261029/advantage/verify.py
```
