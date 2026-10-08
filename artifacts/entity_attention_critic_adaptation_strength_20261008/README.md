# 16600原critic适配与后续搜索棋力检验

已完成原critic额外0/50/200/500步；固定encoder和policy，保留16600模型与完整AdamW状态，沿用上次同一500批次（128近期＋128历史、lr/weight_decay1e-4、critic clip1），CPU FP32。这是同一轨迹的四个前缀，没有重新初始化critic。搜索对战尚未启动。

| 额外步数 | 完整训练MSE | 完整留出MSE | 前中期留出MSE | 后期留出MSE |
|---:|---:|---:|---:|---:|
| 0 | 0.817707 | 0.808967 | 0.872799 | 0.453151 |
| 50 | 0.797382 | 0.789053 | 0.851345 | 0.441821 |
| 200 | 0.795069 | 0.787500 | 0.849277 | 0.443139 |
| 500 | 0.793023 | 0.786814 | 0.849439 | 0.437722 |

完整15代3000局，train2701局/221955位置，留出299局/24568位置；保留原整局划分。阶段按真实根决策前双方最高公开分数是否≥10划分，阶段绝对MSE不能直接当成棋力。完整逐局分子分母在mse/，逐代与近期／历史统计在mse.json。

核验：0步完整模型参数等于16600；0步MSE复现已有完整统计；500步encoder/policy参数及其AdamW状态不变；冻结特征缓存来自同一16600检查点，128维global token缓存已在此前实验核验与完整value前向一致。

训练与恢复命令（只训练和评估MSE，不执行搜索）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u artifacts/entity_attention_critic_adaptation_strength_20261008/train.py --resume
```

诊断checkpoint：runs/entity_attention_critic_adaptation_strength_20261008/checkpoints/step-{000000,000050,000200,000500}.pt。每50步保存可恢复状态，MSE按节点原子保存；完成后resume直接返回。模型仅用于独立诊断，不是正式hybrid恢复入口。

## 当前结论

整体留出0→50步下降0.019913；0→500步下降0.022153，前50步占本次总降幅约90%。200／500步继续有较小的整体收益，但各阶段并非逐节点单调：后期200步略高于50步，前中期500步略高于200步。最新B2的12局留出MSE依次0.877072／0.858084／0.863213／0.859363，也不能据点估计判定最佳棋力节点。这里只确认冻结主干后的适配能恢复value精度，实际搜索收益尚未验证。

留出前中期20831个位置、后期3737个位置；完整数据的两阶段分别208039／38484个位置。训练峰值RSS约0.87 GiB，完成后恢复命令已核验返回`already_complete`。

## 已准备的搜索对战流程，尚未启动

入口：[battle.py](battle.py)；冻结计划：[battle_plan.json](battle_plan.json)；模型与代码身份：[battle_manifest.json](battle_manifest.json)。默认必须明确选择`--prepare-only`、`--analyze-only`或`--run-search`，不会隐式启动对战。**本次只执行了prepare-only和无搜索的准备检查，没有生成任何对战结果。**

1. DEV筛查：50／200／500步分别对0步模型测100 pair，同一100个换座种子，双方全程旧native S1024/N1、无根噪声、温度0。0步作为共同对手。
2. 在看到HEAD之前固定候选：DEV得分最高；完全同分时选完整留出MSE更低者，再同分选更少步。即使DEV未超过50%，也按该规则独立复测，避免只报告正结果。
3. 主确认：选定候选对0步模型，以独立于DEV的2000 pair／4000局检验，主结果为得分及95%按换座pair bootstrap区间，区间是否整体超过50%。固定规模不保证确认微小收益；区间跨50%就报告未确认。
4. 阶段归因：同一候选，仅在前中期根使用适配critic，或仅在后期根使用，每组1000 pair／2000局。两组共享主确认前1000个种子；两侧始终全程搜索S1024/N1。按真实根的公开最高分≥10划分，一次根搜索的整棵树都沿用该根选定的critic，允许跨阶段树内搜索。对手始终0步critic。
5. 报告两个阶段干预的整局得分、主确认前1000 pair与各干预的配对差、后期−前中期配对差，及各阶段模拟数、深度、终局命中、相对prior改招次数和进入后期时的公开分差。阶段的整局干预结果不等于阶段局面的胜率，三组收益不能简单相加。主确认用95%区间；两项阶段得分和三项配对差另报99%区间，对五项次级假设作Bonferroni控制，95%区间同时保留供观察。

计划共4300 pair／8600局。DEV种子23000000–23000099，HEAD23010000–23011999，两组阶段使用HEAD前1000个种子。种子、搜索预算、候选选择规则已在运行前冻结。此前MSE留出用于开发，棋力HEAD以独立对局确认；只有一条训练轨迹，未验证跨训练种子重复性。

**下面是后续启动命令，本次未执行：**从仓库根目录，在CUDA可用的环境中执行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -u artifacts/entity_attention_critic_adaptation_strength_20261008/battle.py --run-search
```

同一命令用于中断续跑：完整换座pair跳过，未完整pair按原种子重做；已完成DEV决定同一候选，HEAD不参与候选选择。每个面板用独立子进程，结束后释放模型、CUDA Graph及native树；一个GPU owner、最多200活动对局、批次≤200，不设置进程RAM硬上限。每25 pair输出RSS、CUDA allocated/reserved并保存汇总。共享原模型evaluator时分模型执行计时存在重叠，不能把该字段相加解释速度或阶段成本。

对战汇总将保存至`battles/`及`battle_analysis.json`，逐局可恢复记录在忽略的相邻`*.games/`，失败现场与内存监控在runtime。分阶段critic路由由本入口安装，不能直接使用通用evaluation入口运行派生配置。

仅准备或重新分析已有结果：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_critic_adaptation_strength_20261008/battle.py --prepare-only
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_critic_adaptation_strength_20261008/battle.py --analyze-only
```

准备验证：[4项无搜索检查](preparation_tests.xml)通过，覆盖公开分数边界、相同根整批路由到适配／原critic的调度接口（伪runner）、DEV/HEAD分离及阶段种子对齐、必须显式指定启动参数。三份DEV配置经现行FixedEvaluationConfig校验通过；训练checkpoint可由现行AZ加载器读取，policy/encoder哈希一致。**真实native/CUDA路由对战和对战中断恢复尚未实跑验证**，本次按用户要求不启动搜索。
