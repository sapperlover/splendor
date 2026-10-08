# 冻结教师采集与短训：确定化／随机节点／CPW

2026-10-07。**全部完成：600局正式采集、三组各500步、16个主面板1600局及独立复测400局，完整审计通过。**

**结论：当前S1024、这个起点和这批固定数据下，没有证据支持把训练教师切换为完全随机节点或CPW。** 完全随机节点较会拟合自己的policy目标，但这个变化没有转化为本次棋力收益；CPW100步初筛相对旧数据组高6个百分点，独立100 pair复测却是49.75%对50.00%，配对差值−0.25个百分点、95%区间[−10.00,+9.75]，优势没有复现。500步也未带来一致收益。建议正式教师继续旧native S1024/N1；本次不是算法永远无效的证明，长期历史混合、多训练seed及大S仍未验证。

## 棋力结果

得分为`(胜 + 0.5×和)/有效局数`，不是纯胜率。各主面板50 pair／100局，先后手互换，使用同一DEV游戏与policy种子。**对手为冻结16500步模型**：greedy列对手也是greedy；全部搜索列的对手均为旧确定化native S1024。不是此前的PPO或θ5对手。学生用共同旧搜索比较训练数据效果；共同CPW列是另一种固定推理方式。不能把两列之差直接当作搜索收益。

| 教师数据 | 100步greedy | 100步旧搜索 | 500步greedy | 500步旧搜索 | 500步CPW搜索 |
|---|---:|---:|---:|---:|---:|
| 旧确定化native | 49.0% | 45.0% | 53.5% | 42.0% | 34.5% |
| 完全随机节点chance | 49.0% | 46.0% | 40.0% | 41.5% | 33.0% |
| CPW k0.25/α0.5 | 45.0% | 51.0% | 48.0% | 41.5% | 30.0% |

起点旧native自对照为52.0%，95%区间[41.5%,62.5%]；由于搜索随机流不同，自对照实测不必恰为50%。三组学生的100/500步旧搜索相对这个起点面板的配对差值区间均跨0，不能宣布明确超越起点，也不能仅凭这些小面板认定确定退化。上述16面板共1600局有效、零异常／非法行动／不变量失败／工程截断；各搜索决策完成S1024，统计和预算由逐局审计核对。

关键配对差值，以同一步数、同推理方式的旧数据组为对照：

| 差值 | 点估计（百分点） | 配对95%区间（百分点） |
|---|---:|---:|
| chance 100步旧搜索 − native | +1.0 | [−12.0,+14.0] |
| CPW 100步旧搜索 − native | +6.0 | [−7.0,+19.0] |
| chance 500步旧搜索 − native | −0.5 | [−12.5,+12.5] |
| CPW 500步旧搜索 − native | −0.5 | [−14.5,+13.5] |
| chance 500步greedy − native | −13.5 | [−27.5,+0.5] |
| CPW 500步greedy − native | −5.5 | [−21.5,+10.5] |
| chance 500步CPW搜索 − native | −1.5 | [−14.5,+11.5] |
| CPW 500步CPW搜索 − native | −4.5 | [−17.5,+8.5] |

DEV属于初筛，多个面板被查看且未作多重比较校正。100步是预定主观察点，500步为重复拟合诊断。独立复测固定选择CPW100与native100，仍用共同旧搜索；新100 pair种子16870000–16870099、policy base16880000、bootstrap16890000。选择原因和冻结身份在`specification.json.confirmation_selection`，不会依据500步结果另挑checkpoint。该复测只独立评估游戏种子，没有重采数据或重复训练seed。

| 独立复测：100步、共同旧搜索 | 胜／和／负 | 得分 | pair 95%区间 |
|---|---|---:|---|
| native数据 | 98／4／98 | 50.00% | [42.75%,57.00%] |
| CPW数据 | 98／3／99 | 49.75% | [43.00%,56.75%] |

相同100 pair上的CPW−native为−0.25个百分点，区间[−10.00,+9.75]；不能确认正收益，也不能排除小幅正负效果。两面板共400局全部自然终局，零工程失败。DEV初筛与复测分开保存，不混池宣布改善。

[DEV主曲线](curves.png) · [独立复测图](confirmation.png)。

## 数据是否学会

三组均为177训练局、23整局留出，game ID split一致；各组轨迹和结果随教师而变。以下只能比较同一组的训练前后，不能把不同教师的原始KL／MSE直接当成网络排名。Policy KL是`KL(搜索访问target || 网络合法动作softmax)`，forced单候选的policy权重为0；value MSE使用该组实际终局±1/0标签。

| 数据组 | train KL：0→100→500 | 留出KL：0→100→500 | 留出value MSE：0→100→500 |
|---|---|---|---|
| native | 0.3972→0.3886→0.3635 | 0.3630→0.3765→0.3757 | 0.6309→0.6527→0.6670 |
| chance | 0.3430→0.3175→0.2826 | 0.3309→0.3218→0.3187 | 0.9040→0.8930→0.8699 |
| CPW | 0.3468→0.3244→0.3057 | 0.3245→0.3247→0.3323 | 0.6463→0.6793→0.6892 |

按23个整局cluster bootstrap10000次，native留出KL在100／500步分别增加0.01355／0.01277，区间均完全大于0；符合“train在拟合，holdout没有同步改善”的现象。Chance在300步留出KL下降0.01508，区间[−0.02814,−0.00302]，但100／500步的改善区间均跨0，value改善区间也跨0；CPW的留出改善未确认。Chance500的policy改善主要来自可见最高分0–4阶段；10分以上policy KL反而增加0.01041、value MSE增加0.07507。阶段分项为描述性统计，不是阶段棋力因果实验。

共同留出bank为77局中的200个请求，从不进入本次训练；三类教师在这些相同请求上各重复8次D组搜索的经验均值作为policy诊断目标。原终局标签属于归档教师的实际后续行为，**不是最优value，也不等于学生策略的反事实终局**。因此它衡量共同历史局面的保留／预测变化，不是最新value总体准确率。

| 数据组 | 共同旧teacher均值KL：0→100→500 | 共同value MSE：0→100→500 |
|---|---|---|
| native | 0.2699→0.2854→0.2905 | 0.7363→0.7223→0.8077 |
| chance | 0.2699→0.2907→0.2995 | 0.7363→0.8033→0.7916 |
| CPW | 0.2699→0.2698→0.2748 | 0.7363→0.8086→0.7678 |

同时，chance500对共同chance均值KL由0.29575降至0.26681，CPW500对共同CPW均值KL由0.27574降至0.25274，说明policy确实向相应新教师目标移动。目标拟合更好不自动意味着目标质量更高或棋力更强。

## 分组裁剪、重复量与value漂移

使用现行`train_split_step`：policy更新trunk和policy专用参数；value用`h.detach()`，只更新critic；两组独立clip1.0，没有全模型联合裁剪。每路batch256，两次独立均匀有放回局面抽样；一个AdamW的完整moments、lr/weight_decay1e-4、sampling RNG及Torch RNG均从同一起点恢复。不是只更换clipping而仍让value反传共享trunk。

| 数据组 | policy裁剪步／500 | value裁剪步／500 | 裁剪前平均policy／value范数 |
|---|---:|---:|---|
| native | 500 | 21 | 2.466／0.559 |
| chance | 500 | 16 | 2.619／0.552 |
| CPW | 500 | 21 | 2.532／0.577 |

100步已覆盖约82.5%–82.9%的训练支持集，每个有效policy位置平均抽约1.75–1.76次；500步覆盖约99.96%以上，每个有效位置约8.74–8.84次。它不是在增加新局面，而主要在增加同一批数据的重复拟合。三组500步总训练仅16.4–17.3秒，本次看不到增加到500步带来一致棋力收益。

Detach阻断value梯度破坏policy，但不冻结完整value函数：policy仍改变trunk，critic自身也仍更新。额外做了共同bank上的事后参数块替换，500步MSE如下：

| 分支 | 起点完整value | 新trunk＋旧head | 旧trunk＋新head | 新完整value |
|---|---:|---:|---:|---:|
| native | 0.7363 | 0.7624 | 0.7950 | 0.8077 |
| chance | 0.7363 | 0.7184 | 0.8407 | 0.7916 |
| CPW | 0.7363 | 0.7401 | 0.7851 | 0.7678 |

例如chance的旧trunk＋新head相对起点MSE增加0.10443，整局95%区间[0.05476,0.15698]；新trunk＋旧head没有同样回退。这提示critic更新本身存在历史预测漂移，不能都归因于共享trunk。这里是事后函数替换，没有独立训练这些组合，也没有据此证明棋力变化的因果；MSE效应不能简单相加。搜索会把value裁到[−1,1]，已同时记录裁后MSE，本次共同bank裁后差异很小。CPU替换诊断与GPU原始value最大差约6.47e-6，通过1e-5容差核对。

## 采集成本与深度

| 教师 | 200局采集＋全量重放 | 相对旧方法成本 | 平均逻辑深度 | simulation终局命中比例 |
|---|---:|---:|---:|---:|
| native | 6分52秒 | 1.00× | 4.805 | 6.405% |
| chance | 8分11秒 | 1.191× | 2.442 | 3.090% |
| CPW | 9分11秒 | 1.337× | 3.354 | 4.170% |

这是每组一次完整采集，包含冷启动、编译、写盘、尾部及checked replay，不是多次重复的纯kernel计时，也不是等墙钟棋力实验。CPW增加深度，但仍未达到旧确定化；轨迹不同，不能把深度差独立解释为棋力因果。

三组模型请求总数约1271万／1280万／1282万，接近；记录的native compute累计时间为108／465／695秒、模拟规则推进27／202／334秒。新增开销主要体现在CPU搜索／信息历史及规则处理。并行worker计时会重叠，不能直接加和成整轮墙钟。CPW约73.1%的物理chance draw来自受限支持集复用，实际fresh/reuse计数保存在每根metadata。

按最高可见分数0–4／5–9／10+分层，旧方法终局命中约0.0033%／1.661%／41.524%，chance为0／0.281%／21.529%，CPW为0.00005%／0.424%／27.203%。这补充说明了搜索阶段差异，仍不是相同状态上阶段收益的配对因果测试。

## 执行与复核

起点为`runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1/rounds/round-019/checkpoints/final.pt`，SHA256 `dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4`。三组各采200局S1024/N1，采集期间教师不更新；chance的N1表示共享统计池，不是预设一个固定世界。根噪声0.25、Dirichlet α0.3，前10个combined completed player turns温度1、后续温度0。使用8个CPU搜索worker、共享槽2、固定地址CUDA Graph、owner单缓冲，B[16,32,64,128,200]、C[16,32,48,64]、FP32、TF32关闭。各组2局S16真实GPU smoke和正式200局均通过完整checked replay。

固定训练0/100/300/500步均遍历完整train与holdout。两路均100%本组新train数据，无历史，不是正式policy800／value3000局方案。相同随机种子不等于相同batch，因为轨迹和池大小不同。评测为FP32 eager、无根噪声、行为温度0，搜索预算固定S1024；不声称eager与compiled目标逐位相同。

新增独立`selfplay_fast`的可选`search_backend`、`chance_widening`、`record_search_diagnostics`，接通多worker共享槽采集。默认字段不写入旧配置，实际已完成父run的manifest fingerprint重建一致；NPZ数组契约不变，诊断只增加metadata。现有hybrid配置保持旧教师，本次未启动正式长训、未改C++／native二进制、未提交或推送。

相关回归75项：73通过、2个sandbox中不可用CUDA测试跳过；GPU集成由六局smoke与600局正式采集补充。第一次greedy配置显式写`backend=serial`被现行接口拒绝，在任何该面板对局前改为省略execution，使用现有默认串行入口；配置和脚本修正均保留旧版本／SHA。两次分析汇总修正只处理greedy缺少execution stats和非CPW缺少PW计数。`specification.json`记录修正范围，审计通过AST确认采集／训练函数未变。

## 产物与命令

- `specification.json`：冻结配置、种子、命令、输入／代码SHA、修正记录和复测选择；Git HEAD为`0a176b1a269a9850f51e7350e475a6c801f82644`，执行的是保留的dirty源码快照，不能只凭HEAD复现。
- `analysis.json`、`strength_statistics.json`：主面板汇总、配对差值及前后／步数比较；`confirmation.json`记录独立复测和未复现的初筛差异。
- `dataset_*.json`、`collect_*_result.json`、`smoke_*_result.json`：每局身份、采集计时、完整规则重放与预算。
- `curve_*.jsonl`、`metrics_*.jsonl`、`train_*.json`、`sampling_analysis.json`：全量拟合曲线、梯度／裁剪、RNG恢复及真实覆盖次数。
- `holdout_diagnostics.json`、`holdout_predictions_*.json`、`value_drift.json`：逐局置信区间、逐位置输出、可见分数阶段及事后参数块替换。
- `evaluations/*.json`、`*_recording.json`：完整逐局主面板与复测汇总、模型身份；原始JSONL在相邻被忽略的`.json.games/`。
- `curves.png`、`curves.pdf`：DEV主曲线；`confirmation.png`、`confirmation.pdf`：独立复测。图中误差棒为按pair bootstrap95%区间。
- `tests.json`、`verification.json`：回归和通过的完整审计，覆盖18个面板2000局、数据／源码／模型／种子／预算／抽样账本。

所有采集、训练、棋力和模型诊断均在conda `splendor`。该环境没有Matplotlib，独立JSON绘图使用本机已有base Matplotlib，不加载模型、不重算统计、不修改任何包；解释器及绘图库版本见`plot_environment.json`。

NPZ、checkpoint和二进制抽样账本位于忽略的`runs/entity_attention_chance_training_probe_20261007/`；源码／native快照、dirty.patch、运行与复测日志在忽略的`runtime/entity_attention_chance_training_probe_20261007/`。这些重型原件需另行备份，JSON中的SHA不等于Git内含模型。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_chance_training_probe_20261007/run_suite.py
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_chance_training_probe_20261007/confirm.py
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_chance_training_probe_20261007/verify.py
```

首次prepare命令见specification，已准备目录不要重复prepare。suite跳过已完成阶段，采集和评测可在同一冻结身份下恢复；固定训练只支持一次完整执行，不支持中途恢复，失败时先保留并核对现场，不直接续跑重复账本。算法同时改变随机事件处理、深度、目标均值及轨迹，不能将差异单独归因于目标JS；没有测关闭根噪声的训练或300步棋力。一个起点、一个采集／训练seed、最新单批固定目标、最多500步，不能裁定多轮自迭代／历史窗口／大S的长期效果。此前固定请求的JS稳定性结果仍是独立实验，见[旧目标噪声](../entity_attention_target_noise_20261007/README.md)、[chance](../entity_attention_chance_search_20261007/README.md)、[CPW](../entity_attention_chance_pw_noise_20261007/README.md)；本次没有重复测学生新权重的JS。
