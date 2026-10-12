# 第八轮终点出发：第九轮单变量拟合实验

2026-10-10已完成10/10个新分支，共20,000步训练，耗时约17分41秒。完整分析见[完成结论](analysis.md)、[曲线图](holdout_curves.svg)，数值及区间见[结果表](results.md)。原100／250步KL均比500步更差；较小lr对policy有正面信号，1k池加重value过拟合，batch需区分相同步数与相同抽样量。本次拟合比较不提供棋力或参数最优结论。

指定节点的集中汇总见[100／250／500／1000／2000步完整矩阵](steps_100_2000.md)：11组配置共55个唯一记录，分别列出完整留出和最新代固定Train子集的KL／MSE；[JSON](steps_100_2000.json)保留完整精度、原始评估身份与已有配对区间。运行`conda run --no-capture-output -n splendor python artifacts/entity_attention_r9_ofat_20261010/consolidate.py`可仅用CPU重新整理，不训练、不加载模型。

2026-10-11后续[直接搜索对战](../entity_attention_r9_lr_step_arena_20261011/results.md)：lr2e-5×1000步对原lr1e-4×500步，新开局200 pair，得分53.250%，95% CI [48.500%, 58.000%]。点估计偏正，尚未确认棋力优势；其他新分支未补测棋力。

[验证记录](verification.json)确认68个评测节点、6,528,000条新增抽样引用，以及各分支模型／AdamW／RNG起点一致；lr组样本流与原配置完全相同，batch的等量抽样前缀也完全相同，没有Validation引用进入训练。验证与绘图命令见完成结论。

基准为学习率1e-4、在线池6000局（第4–9轮）、batch256；使用与原第九轮完全相同的第八轮结束模型（38,398步）、AdamW状态、采样RNG及Torch RNG。除指定变量外，权重衰减1e-4、policy/value等权联合训练、梯度裁剪1、训练数据原split与标签保持一致，不新增采集、不升级教师。

| 实验轴 | 单独改变的档位 | 共同评测节点 |
| --- | --- | --- |
| 原配置步数 | 0、100、250、500、1000、2000、5000 | 复用已有checkpoint，统一重新评估 |
| 学习率 | 5e-5、2e-5、1e-5 | 0、100、250、500、1000、2000步 |
| 在线池 | 9000、3000、1000个原始整局 | 同上 |
| batch | 64、128、512、1024 | 同上；B1024另记125步 |

在线池均按最新完整对局取窗口：9k包含第1–9轮，6k为第4–9轮，3k为第7–9轮，1k只含第9轮。每代原1000局包含原Train与Validation，训练仅在Train局面上均匀抽样；29k旧离线池不参与本次抽样。池大小改变也改变教师代际组成及新1000局的抽样占比，所以不能把结果解释为单纯容量效应。

所有配置统一评估第九轮完整固定留出100局：8174个value局面、6688个有效policy局面。KL为KL(原行为搜索标签∥学生)，MSE使用原终局value标签。同时评估第九轮固定2048个Train局面（与原拟合诊断相同），辅助观察训练与留出的变化。指标复用`pool_fitting.evaluate`，逐局分子／分母与按整局配对bootstrap的差值区间保留。

主要比较相同更新步数。batch另以128000条抽样对齐：B64×2000步、B128×1000步、B256×500步、B512×250步、B1024×125步；该补充比较改变步数以对齐数据量，不等同于固定步数比较。学习率不随batch缩放。源码中的`history_batch_size`在显式零历史drawer中不参与计算；B64仅把这个未使用的配置字段限到64以满足数据类校验，实际历史抽样数始终为0。

只训练10个新分支，各到2000步；原配置曲线直接复用已有模型。运行串行，使用一个共享的无损compact缓存，避免每组重复解压全部数据；不设额外进程内存或磁盘上限。checkpoint每50步保存完整优化器和RNG，125步另保存一次，以便精确记录相同抽样量比较。已有组合搜索对战不修改、不停止。

从仓库根目录启动／中断续跑：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u artifacts/entity_attention_r9_ofat_20261010/run.py run
```

Ctrl+C后重跑同一命令，完整节点跳过，从最近已提交checkpoint续训。`--max-steps-this-session N`限定本次新增更新总数，跨分支累计；默认执行全部。`run.py analyze`仅复算统计。

冻结计划及实际配置见[plan.json](plan.json)，原split及池窗口见[dataset.json](dataset.json)，最新代固定Train引用见[latest_train_references.json](latest_train_references.json)。源码／环境／命令保存在`execution.json`。进度与结果见[results.md](results.md)、[summary.json](summary.json)；完整拟合及配对差值在`fitting/`、`comparisons/`。checkpoint、训练逐步日志与抽样账本在忽略的`runs/entity_attention_r9_ofat_20261010/`，源码快照及进程锁在`runtime/entity_attention_r9_ofat_20261010/`，进度同时输出到启动终端。

这是固定DEV、单起点和单采样种子的探索性比较，未作多重比较校正，没有棋力评测。不能把每个配置在这批DEV上挑到的最低点视为已经确认的最优参数。
