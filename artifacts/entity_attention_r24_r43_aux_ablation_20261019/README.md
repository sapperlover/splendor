# R24–43小模型离线：无辅助与仅剩余回合辅助对照

2026-10-19结果更新：两组训练与四组对战均已完成，见[结果](results.md)。plain／turns-only的greedy为53.850%／53.150%；旧双辅助减plain为+0.050pp [−0.900,+1.025]pp，未确认辅助头额外收益。两组搜索得分区间均跨50%。

2026-10-19。用户授权写好脚本后启动，两组串行训练并评测，不持续轮询。起点、数据与训练预算完全沿用[此前小模型双辅助离线实验](../entity_attention_r24_r43_offline_objectives_20261013/README.md)，仅改变辅助目标。原实验及其他进程均保留。

## 两个分支

| 分支 | 网络参数 | 实际辅助头 | 联合训练目标 |
| --- | ---: | --- | --- |
| `plain` | 532,738 | 无 | 原Policy CE＋胜负value MSE |
| `turns_only` | 549,635 | 仅剩余回合 | 原Policy CE＋胜负value MSE＋0.25×剩余回合MSE |

两个分支都训练原policy与胜负value，共享encoder和原全局clip1；这不是policy-only实验。PSW关闭。`turns_only`没有注册分差头参数，不计算或反传分差loss；其回合头初始参数与旧双辅助实验逐位相同。旧双辅助checkpoint省略`predict_score`时仍构造两个头，原参数顺序、默认初始化和损失计算顺序保留。

原R43小模型128维×2层、4头、FFN512，checkpoint SHA `b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。旧权重、AdamW矩与Torch RNG继承；新辅助参数无历史优化器状态。辅助初始化seed36100000、抽样seed34410000、验证seed34420000均不改。

池为绝对R24–R43的20000局，原整局划分：Train17971局／1451525位置，Validation2029局／163595位置；不刷新搜索标签。每组平均曝光4次、**22681步**，lr2e-5、B256、weight decay1e-4、clip1、FP32、8192MiB compact缓存。所有训练位置均匀有放回抽样，两个主任务同batch；原`history_batch_size=128`字段不另分历史池。回合标签除以40，以双方尚未完成的玩家完整行动次数之和为目标，支付／弃牌／选贵族不另外计回合。

每1000步保存checkpoint并评估同一4096固定留出位置；终点完整评估163595留出位置，保存普通CE／KL、胜负MSE以及适用的回合MSE／MAE。记录每步loss、梯度范数、抽样哈希和可恢复账本。训练结束逐步核对22681个抽样哈希与旧双辅助实验完全一致，否则停止，不进入评测。

## 评测与统计

只评测**训练终点final22681**，与原小模型实验一致，不按留出或棋力挑checkpoint。先串行完成两组训练，然后按`plain greedy → plain search → turns_only greedy → turns_only search`执行：

- 每组greedy对原R43：1000pair／2000局，CPU8worker；复用旧双辅助的36200000–36200999开局和policy seed域，可与旧双辅助1000pair结果作共同对手配对差。
- 每组搜索对原R43：200pair／400局，双方native S1024/N1、c_puct1.5、无根噪声、T0、CUDA compiled；共同新开局39100000–39100199。逐局核对实际模拟数与确定化次数。
- pair同开局交换先后手，和棋半分，整pair bootstrap10000次。保留逐局胜和负、循环和局、异常、非法动作、工程截断、模型SHA及配置。
- 汇总`turns_only−plain`的greedy／搜索共同对手得分差，以及旧双辅助分别减两组的greedy差。旧模型不重新训练、不额外运行搜索面板。共同对手差不是候选直接互打；旧greedy开局为已查看DEV复用，单训练seed，未做多重比较校正。

## 运行与产物

原始配置：[无辅助](../../configs/entity_attention_v1/offline_r24_r43_no_aux_20261019.json)、[仅回合](../../configs/entity_attention_v1/offline_r24_r43_turns_only_20261019.json)。冻结清单、旧证据SHA与规模见[plan.json](plan.json)。同一命令恢复已中断队列，已完成评测自动跳过：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u artifacts/entity_attention_r24_r43_aux_ablation_20261019/run.py run
```

训练checkpoint、AdamW、抽样账本、验证与逐步指标位于`runs/entity_attention_r24_r43_aux_ablation_20261019/{plain,turns_only}/`。链路锁、各阶段日志和进程状态位于`runtime/entity_attention_r24_r43_aux_ablation_20261019/`，`live.json`包含监督／当前子进程PID，`launch.json`记录后台启动身份。正常中断会传递给当前子进程，完整checkpoint和已完成评测pair保留。

本目录[summary.json](summary.json)、[results.md](results.md)在阶段边界自动刷新；`panels/`保存对战原始结果，`evaluation_configs/`保存实际配置，`execution/`保存命令、源码快照身份及退出状态。训练与评测各在独立子进程中执行，阶段退出释放其缓存。

## 有限验证

[tests.xml](tests.xml)：51项CPU相关测试通过，包括旧双头兼容、仅回合头的实际缺省与初始化一致性、梯度和总loss、无辅助／仅回合的断点恢复逐位一致、工厂及冻结加载、评测终点和规模配置。

[cpu_preflight.json](cpu_preflight.json)：真实R43与原R24训练对局4样本／分支，核对旧参数／AdamW矩、回合头与旧双辅助初始权重、两组初始policy/value完全一致、单步更新和保存加载、各3个S8原生根通过。[cuda_preflight.json](cuda_preflight.json)中的相同有限验证也已通过，实际调用compiled推理与native搜索，两组各3个S8根；这些均是工程验证，不计入正式训练或棋力结果。

## 2026-10-19冷产物位置

非主要checkpoint、逐步指标及抽样账本已集中到`archive/20261019/runs/`下相同run结构；关键模型与验证／选点文件留在原位置，原文件路径保留相对链接。见[逐文件迁移清单、保留策略及恢复说明](../repository_archive_20261019/README.md)。只读诊断可在恢复挂载后访问；精确续训前须按清单撤回该run的迁移，恢复原位真实账本／日志和权重，避免续训改写冻结归档。本次没有启动或恢复训练。
