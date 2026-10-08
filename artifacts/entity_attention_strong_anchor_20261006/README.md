# 原第9轮与第50轮：共同PPO锚点补测及最新数据方案

2026-10-06。用户停止第50轮后授权补测和设计。本次完成两个100 pairs/200局面板，共400局全部有效；没有采新自对弈数据、更新模型或启动正式训练。新增采样入口尚未实现。

## 结果与结论

双方均MCTS S1024/N1，对手是PPO seed1的latest、lifetime 5,512,112 checkpoint，并非其best-dev。score按胜=1、和=0.5计分。置信区间以100个换座开局为单位bootstrap，不把200局当作独立样本。

| Attention节点 | 本次是否补测 | 胜/负/和 | 得分率 | 配对95%区间 |
| --- | --- | ---: | ---: | --- |
| 离线5000步、当前theta_5000锚点同权重 | 历史结果 | 82/117/1 | 41.25% | 34.25%～48.25% |
| 完整离线final、12045步 | 历史结果 | 119/81/0 | 59.50% | 52.00%～67.00% |
| 原混合第9轮、9500步 | 是 | 126/71/3 | **63.75%** | 56.75%～70.50% |
| 小步N1分支第50轮、14500步 | 是 | 123/76/1 | **61.75%** | 54.75%～68.75% |

| 同种子配对差值 | 点估计 | 95%区间（百分点） |
| --- | ---: | --- |
| 第50轮−原第9轮 | −2.00个百分点 | −11.75～+7.50 |
| 原第9轮−离线5000步 | +22.50个百分点 | +13.25～+31.75 |
| 第50轮−离线5000步 | +20.50个百分点 | +11.25～+29.75 |
| 原第9轮−完整离线final | +4.25个百分点 | −4.75～+13.25 |
| 第50轮−完整离线final | +2.25个百分点 | −7.50～+12.00 |

这批结果支持原第9轮取得的收益仍大体保留：两个节点在共同PPO面板上都明显高于离线5000步。后续50轮没有显示出可确认的进一步收益；第50轮低2个百分点仍处于测量不确定性内，不能判定它已明显弱于原第9轮。两个节点也没有显示出对完整离线final的可靠优势。此前theta_5000的70%～79%得分，不能换算成对PPO的70%～79%，更不能据此排列所有分支的全局实力。

本报告沿用被多次查看的DEV种子，只有一个训练seed；它不是独立FINAL验收。共同对手上的得分差不是候选之间直接对打的结果，也不证明跨对手风格的传递关系。完整离线节点的两组原结果没有重跑，保留历史版本限制。

## 当前采样与拟合证据

从原始逐步引用账本统计，第31～50轮均为纯在线，最新一轮200局实际占整个batch平均**6.61%**，新train positions平均14986.55个，即时平均曝光0.113次/位置。第50轮3000局活动窗口为分支第36～50轮，均由S1024/N1教师生成；当前不含旧Deep-Sets离线训练配额。

最近10轮的完整current holdout：policy KL有1轮下降、8轮上升，第50轮几乎不变；value MSE平均变化约−0.00848。第50轮policy KL为0.42621554→0.42623154，value MSE为0.76190678→0.75114895。这些是同轮更新对最新教师搜索标签及终局标签的拟合变化，不是棋力评分。绝对训练CE还受目标熵和样本来源影响，不能把它作为单独学习充分的依据。

上述证据让“最新目标被历史抽样稀释”成为值得优先隔离的假设，没有证明3000局过长、过短，也没有排除value经共享trunk干扰policy或搜索标签质量的问题。新方案只改变最新抽样配额，保持窗口、教师S/N、更新/新局数比、学习率和损失权重。

## 下一阶段设计

推荐从第50轮完整checkpoint接续，继承AdamW/RNG和第36～50轮的3000局。选择第50轮便于单独检验最新份额，避免重新引入旧离线日程；并非认为它已证明强于第9轮。

每轮200局、100步、batch256；最新/历史互斥抽样：新分支1～2轮32/224，3～5轮48/208，6轮起64/192，最终最新占25%。FIFO窗口仍为3000局；第一阶段15轮、3000新局、1500更新，从14500到16000步。第5/10/15轮各对同一PPO锚点测200局，和本次第50轮起点做配对比较；每轮看同一最新train探针与完整holdout的前后变化。不设置拒绝更新或回滚。

实施后先用第50轮已有数据和initial.pt做均匀采样/最新25%各100步的静态回放，检查拟合变化，无需新增采集。详细配额、缓存、来源年龄、指标与实施接口见[训练方案](../../docs/guides/entity_attention_fresh_share_zh-CN.md)和[结构化设计](training_plan.json)。设计尚未实现，不能直接用原配置resume获得最新25%。

## 协议、身份与成本

- Game seeds：13500000～13500099，每seed换座两局；policy/search/bootstrap种子域为13510000/13520000/13530000，bootstrap 10000次。
- 双方无根噪声、温度0、c_puct=1.5、forced phases折叠、simulation transition上限512。使用CUDA FP32、native_batched、compiled/reused；attention projection=false，PPO=true。
- 两个新面板的源码SHA-256均为`733eda03590cb4ad312ab8e0a977681e8c0e380c1384d9db53c85633835fc090`，git HEAD为`0a176b1a269a9850f51e7350e475a6c801f82644`，工作树包含已有未提交改动；不能只用HEAD复现。native二进制及C++源身份见recording，源码快照在忽略的runtime下。
- 原第9轮checkpoint SHA-256：`af3b2b64007705cc9adcbb3e8a6ad68239f4dc967c31c515e857b1240479e44a`。第50轮：`1f794e1ecee4e71b6aa5f3b45ce22bac50caf9055021a792b3e63056c1d94216`。PPO：`009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0`。
- 当前theta_5000与历史离线5000步checkpoint的69个模型tensor全部相同，最大绝对差0；文件SHA不同来自元信息。[身份核对](anchor_identity.json)不能被不同文件SHA否定。
- 实际设备为RTX4050 Laptop 6 GiB、i5-13500H，Conda splendor/Python3.12.14/PyTorch2.14.0+cu130。原第9轮评测542.41秒、第50轮533.34秒，总计17.93分钟；不包含进程启动等外部开销，也不是训练阶段的时间保证。
- 每个新面板valid_games=200、completed_pairs=100，非法动作、异常、invariant failure和engineering truncation均为0；冻结输入执行前后相同，正式run配置、状态与checkpoint均未改写。复核记录见[verification.json](verification.json)。

## 原始产物与复核

[第9轮汇总与完整pair结果](evaluations/round9_vs_ppo.json)、[第50轮汇总与完整pair结果](evaluations/round50_vs_ppo.json)、[配对差值及50轮引用统计](analysis.json)、[冻结协议与输入SHA](specification.json)、[第9轮执行身份](recording/round9-1791277087102515043.json)、[第50轮执行身份](recording/round50-1791277800387981067.json)。完整对局sidecar位于`runtime/entity_attention_strong_anchor_20261006/{round9,round50}/games/`，不纳入Git；模型与训练NPZ仍保留在原runs目录，迁移时需另行备份。

从仓库根目录复核统计，无需GPU或重跑对局：

```bash
conda run -n splendor env PYTHONPATH=src python artifacts/entity_attention_strong_anchor_20261006/analyze.py
conda run -n splendor env PYTHONPATH=src python artifacts/entity_attention_strong_anchor_20261006/verify.py
```

仅在明确需要重跑时执行GPU命令；本次两组已完成，以下不是续训命令：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src TORCHINDUCTOR_COMPILE_THREADS=2 python -u artifacts/entity_attention_strong_anchor_20261006/run_evaluation.py --case round9
conda run --no-capture-output -n splendor env PYTHONPATH=src TORCHINDUCTOR_COMPILE_THREADS=2 python -u artifacts/entity_attention_strong_anchor_20261006/run_evaluation.py --case round50
```

本次新增源码仅为该artifact内的评测、统计和复核脚本，没有修改训练实现。记录与方案的valid范围限于已完成400局、原始轨迹复核及设计一致性，不能解释为新采样训练收益已验证。
