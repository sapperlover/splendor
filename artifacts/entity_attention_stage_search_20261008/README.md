# 只搜后期／只搜前中期：同权重 greedy 对照

2026-10-08。两组阶段实验均已完成500pair／1000局，结果与进一步的全程搜索对照见 [RESULTS.md](RESULTS.md)。全程S1024/N1对同权重greedy的500pair也已完成：814胜8和178负，得分81.8% [79.3%,84.3%]。独立入口为 [run.py](run.py)，当前参数为 [plan_fast.json](plan_fast.json)，正式产物写入独立的 `fast/` 子目录。分阶段策略仅在本实验进程内安装；通用读取器修复了 greedy 的 inference 参数丢失，以便同权重双方共享 evaluator。现有 B/C/D 冻结配置和实验结果保留。原 [plan.json](plan.json)、manifest、配置与源码快照是此前 eager64／8GiB 方案的准备记录，尚无正式对局结果，已由加速方案替代。

## 刚完成的训练节点

B/steps50 的前两轮共新增400局、更新100步，16500→16600步。首次对冻结起点的100 pair／200局为102胜、1和、97负，**得分51.25%，95%配对区间44.5%–58.0%**。逐局JSONL复算与原汇总一致，200局均自然终局。当前没有确认净收益，也不能据两轮判定长期无效。

| 各轮自己的新数据留出集 | Policy KL：更新前→后 | Value MSE：更新前→后 |
| --- | ---: | ---: |
| 第1轮 | 0.41374→0.41448 | 0.83376→0.84917 |
| 第2轮 | 0.40668→0.41319 | 0.85922→0.87707 |

两轮当前留出指标均略变差，是继续关注泛化与value漂移的信号；这是两个不同留出集合上的轮内变化，不把跨轮绝对值当作连续误差曲线。近期混合留出policy KL第1轮下降、第2轮近乎不变，尚无一致改善。可复核记录见 [first_evaluation_review.json](../entity_attention_iteration_trend_seed1_20261008/first_evaluation_review.json)、[原始评测](../entity_attention_iteration_trend_seed1_20261008/evaluations/seed1_steps50_r002.json)。C/D未运行，本节点不用于排序三组。

## 实验问题和干预

判断在当前网络与greedy对手下，搜索用于后期还是前中期产生更大的整局棋力收益。双方均固定为此前搜索深度诊断使用的16500步attention权重，SHA256 `dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4`；不使用新增训练的16600步权重，以免把权重变化混入阶段归因。

**每个真实决策根上，任一方公开分数达到10分即为后期**：`max(self_public.score, opponent_public.score) >= 10`。分数单调不减；规则同时关注对手接近终局，只访问公开Observation，不使用真实暗牌、未来牌序或事后的整局长度。10分是预先固定的诊断阈值，尚未证明最优，也不同于旧诊断按历史局长15%/50%/85%抽样的口径。

| 顺序 | 名称 | 根决策分数<10 | 根决策分数≥10 | 对手 | Pair |
| --- | --- | --- | --- | --- | ---: |
| 1 | late_only，只搜后期 | 同权重greedy | 旧native S1024/N1 | 同权重greedy | 500 |
| 2 | early_only，只搜前中期 | 旧native S1024/N1 | 同权重greedy | 同权重greedy | 500 |

阈值使用双方分数的最大值。每组1000局，总计2000局；只跑这两组，不增加全程搜索、裸网自对照、PPO或训练面板。Greedy为同一冻结网络的合法动作argmax，零搜索；两组共用game seed22000000–22000499，每seed交换先后手，policy base22100000、search seed22200000。无根噪声、温度0、c_puct1.5、forced折叠、transition cap512；FP32，TF32关闭；搜索使用静态批次的 torch.compile + native CUDA Graph，greedy仍为同模型 eager argmax，不改变搜索预算、阶段阈值或种子。

阶段开关仅控制**真实根决策是否启动搜索**；一旦启动，树内仍按原算法正常搜索，可以跨越10分边界并命中终局，不在边界截断树。这保持了S1024/N1语义。因此early_only也可能提前看到终局，其终局命中统计需要一起解释。

## 如何读结果

主结果为两组各自对greedy的得分及95%按pair bootstrap区间，并对同一seed的两组pair得分相减，报告`late_only − early_only`及配对区间，10,000次重采样。比较的是不同整局策略对同一对手的表现，双方轨迹会变化，不是从同一中盘状态开始的续局对照。

- 只搜后期明显高于50%，而只搜前中期接近50%，且两组差值区间大于0：支持收益更集中在该阈值后的搜索。
- 两组都明显高于50%：前中期也有可见贡献；进一步看相对强弱及各组终局命中。
- 差值区间跨0：本面板未区分相对优势，不能宣布等效。点估计、区间和搜索成本均保留。

不加全程搜索面板时，无法计算“后期保留了全程收益的百分之几”；本实验也不单独隔离终局精确回传与其他搜索收益。10分之前即使没有收益，也不代表所有前中期定义、模型和对手都无收益。

同时记录早/晚根决策、greedy/搜索次数、各阶段模拟数/深度和终局命中、首次进入后期的已完成回合数。终局命中包含胜负和与重复访问，不是必胜比例。完成后检查禁止搜索阶段模拟数为0、每个被搜索的多候选根完整用S1024，以及greedy对手没有搜索统计。工程失败明确保留现场并停止，不把失败当成和棋；既有确定性循环裁定单独列出。

## 一条命令连续运行和续跑

从仓库根目录执行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -u artifacts/entity_attention_stage_search_20261008/run.py \
  --config artifacts/entity_attention_stage_search_20261008/plan_fast.json
```

先完整跑late_only，再early_only。单推理owner，最多200个活动对局，GPU批次≤200，静态batch桶为16/32/64/128/200；两组串行且组间释放引用与CUDA缓存，无训练replay缓存或worker池。按用户新要求，入口和命令均不设置RAM/cgroup硬上限。每25 pair输出RSS、CUDA allocated/reserved，并保存最新内存状态到runtime；这些监控不裁剪数据或终止任务。两组正式500pair均已完成，耗时分别约6.08／17.82分钟；收尾RSS分别约1.919／2.412GB，收尾值不是全程峰值。短窗与收尾监控不能证明任意长运行绝无泄漏。

Ctrl+C后等待进程退出，再运行**同一命令**：完整换座pair跳过，未完整的pair按原种子重做。每25 pair保存汇总并输出进度；逐局JSONL及时持久化。修改权重、阈值、种子、预算、pair数量或执行代码时换输出目录，避免混入已完成结果。不要直接用通用`python -m splendor.evaluation`跑派生配置；阶段开关由本目录入口安装。

当前配置、源码身份和环境在`fast/manifest.json`／`fast/configuration_snapshot.json`；源码快照和失败现场在忽略的`runtime/entity_attention_stage_search_20261008/fast/`；原始评测在`fast/evaluations/late_only.json`、`fast/evaluations/early_only.json`，重型可恢复逐局记录在相邻忽略的`*.games/`；汇总在`fast/analysis.json`。实际启动命令与环境另存`fast/invocations/`。

仅准备或重算既有结果，不运行棋力赛：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_stage_search_20261008/run.py --prepare-only
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_stage_search_20261008/run.py --analyze-only
```

## 必要有限验证

4项CPU检查通过，记录见 [finite_tests.xml](finite_tests.xml)：双方公开分数的10分边界；greedy分支零树/零模拟且选招一致；搜索分支逐项复现原native S8；两种策略各两pair的checked批量运行、完整pair中断后续跑及阶段预算。测试使用小模型和S3/S8，不把测试棋力作为正式结论。两组500pair CUDA棋力面板已经完成；初版CPU记录与S1024 CUDA短验证保留用于追溯。


## 加速验证（2026-10-08）

复用已有200个前/中/后期固定局面、16500步冻结模型、S1024/N1及相同采样种子。每方案独立进程运行4次，第一次包含冷启动，后三次取中位数。机器为RTX4050 Laptop 6GiB；只比较执行速度，不作为棋力评测。

| 方案 | 热身后200根中位耗时 | 相对eager64吞吐 | 冷启动窗口 |
| --- | ---: | ---: | ---: |
| eager，批量64 | 9.388秒 | 1.00× | 9.445秒 |
| eager，批量200 | 5.971秒 | 1.57× | 6.210秒 |
| CUDA Graph，批量200 | 5.094秒 | 1.84× | 14.403秒 |

200根选招均一致；FP32批量/编译改变运算顺序，先验微小数值差异与target差异保存在 [speed/analysis.json](speed/analysis.json)，不声称逐bit一致。加速方案后三窗口RSS约1.867GB，变化约24KiB；CUDA allocated保持30.588MB、reserved保持159.384MB，3个已触达Graph、3个native缓冲区均固定。各窗口active jobs为0，上一窗口job弱引用均已释放；释放最终batch记录后当前job也全部释放。基准自身保留4次结果，微量RSS增长不是训练replay增长。静态Graph形状和活动对局数量有界，实际对局会继续触达其他有限形状。

首次B/r002正式评测1075.6秒，两模型eager前向合计770.6秒，平均batch约25–26，因此扩大供给和复用Graph有明确瓶颈依据。固定根测速仅用一个模型且满批，不能直接据此将双模型面板耗时除以1.84，也没有证明200是硬件全局最优。继续加速的下一处候选是greedy跨局批量推理；当前仍逐请求执行，需要另做结果与耗时核验，未混入本次交付。

相关CPU回归14项通过，见 [speed/finite_tests.xml](speed/finite_tests.xml)，覆盖greedy配置序列化/共享模型、native/chance对greedy与serial一致、失败记录，以及两个阶段策略的边界/预算/中断续跑。上述4项初版CPU记录保留用于历史追溯。

两种阶段策略分别用eager／fast各跑8pair（共64局，32组对应终态），诊断种子22400000–22400007，与正式面板种子分离。只用于验证执行链、阶段预算、共享模型与内存；冷编译成本包含在耗时中，小样本得分不用于棋力结论。原始结果、命令和RSS采样见 `speed/{late_only,early_only}_{eager,fast}.json`、相邻`.command.json`、`.memory.json`，必要失败现场在runtime，恢复逐局sidecar按既有约定忽略。

| 真实对局短验证：各8pair／16局 | eager64（含启动） | Graph200（含启动） | 对应终态一致 |
| --- | ---: | ---: | ---: |
| 只搜后期 | 11.855秒 | 18.632秒 | 16/16 |
| 只搜前中期 | 65.904秒 | 26.987秒 | 16/16 |

两侧搜索阶段预算通过，禁止搜索阶段模拟数均为0，对手搜索统计为空且只创建一份共享evaluator。加速版两种短验证RSS分别约1.300GB／1.281GB，触达新形状后后续采样稳定，allocated约32.263MB、reserved约79.692MB。只搜后期小面板因约10.9秒首次编译变慢；只搜前中期含启动仍快2.44×。这两个小面板最多16活动局，不能据此推算active200的完整500pair耗时；固定根热身吞吐与冷启动实际对局分开报告。

可复核测速命令（已生成的输出不覆盖，复测需使用新目录）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 python -u artifacts/entity_attention_stage_search_20261008/benchmark.py --case eager64
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 python -u artifacts/entity_attention_stage_search_20261008/benchmark.py --case eager200
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 python -u artifacts/entity_attention_stage_search_20261008/benchmark.py --case graph200
conda run -n splendor env PYTHONPATH=src python artifacts/entity_attention_stage_search_20261008/analyze_speed.py
```
