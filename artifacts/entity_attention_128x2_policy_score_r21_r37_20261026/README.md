# Attention128：104k池 policy＋得分归属辅助离线训练

2026-10-27完成状态核对：训练、完整留出选点及原两组对战均已完成；192804步早停，选中140221步。原raw R37 greedy1000 pair得分53.475%，双方S1024N1（总1024）500 pair为56.800%，见[原结果](results.md)。同一候选的新版10000 pair greedy／S128N1结果见[大面板](../offline5_vs_r37_10k_20261027/README.md)。下方按日期保留首次准备、停训和恢复过程，不再代表当前进度。

2026-10-26，按用户要求准备独立脚本；正式训练和对战均未启动，不接入此前串行队列。复用已验证的训练→选点→greedy→搜索监督入口，每阶段失败或中断即停止，同一命令恢复。配置只比上次仅policy组的实际18GiB／4 worker配置增加`score_ownership_weight=0.25`，不继承任何训练checkpoint。

```bash
# 默认完整流程：训练、选点、对raw R37的两组评测。
bash scripts/run_attention128_policy_score104k_20261026.sh

# 只训练并选点，结束后不启动棋力评测。
bash scripts/run_attention128_policy_score104k_20261026.sh train

# 仅准备／核对身份与计划，不预热全池、不训练。
bash scripts/run_attention128_policy_score104k_20261026.sh prepare

# 汇总已存在的训练／评测产物。
bash scripts/run_attention128_policy_score104k_20261026.sh status
```

从仓库根目录运行，脚本使用conda `splendor`，限制OMP／MKL／OpenBLAS各1线程，训练子进程从本实验冻结源码启动。`run`为默认动作，`train`只训练与选点；后续使用默认命令会跳过已完成训练，再继续评测。训练中断由现有完整checkpoint恢复，未消费预取batch丢弃后按保存RNG重算。不会自动等待或更改其他实验进程。

## 配置与样本

| 项目 | 冻结设置 |
| --- | --- |
| 数据 | 原R21–R37共104000局，原NPZ／主标签／整局split；共用原104k清单 |
| 训练／留出 | 93537／10463局，7478428／835367决策；正权重主policy 1559755／174056位置 |
| 模型 | 从零attention128×2，4头、FFN512、market relations、global_players_market读取；852498参数 |
| 辅助 | 对手policy权重0.15；双方×一级／二级／三级／贵族未来新增分8路回归权重0.25；短期value关闭 |
| 主任务 | 原policy和终局value权重各1；主policy仅高预算位置，value及得分辅助保留全部阶段，包括支付 |
| 优化 | 新AdamW，lr1e-4、weight decay1e-4、clip1、FP32；policy／value各独立B256 |
| 预算 | 12次value曝光，上限350552步；每1000步及全部5%节点保存完整checkpoint |
| 早停／选点 | 每5%原始完整留出主KL＋终局MSE；连续3次严格高于前一次即停，持平／下降重置，step0不计；到达节点最低联合指标选完整模型，辅助误差不参与 |
| 执行 | training_workers=4共享只读压缩池、18GiB训练arena容量；流式累计、state-only value／辅助传输、批量标量回读 |
| 记录 | 原逐步training_metrics.jsonl、batch参考SHA；不写大抽样账本、重复steps／会话metrics；共享encoder梯度首步及每100步 |

初始化／辅助／抽样／留出seed依次为43100000、36100000、43110000、43120000，全部沿用上一离线组。得分标签定义为`remaining-score-by-player-source-v1`，当前位置到实际终局的新增分，训练标签除以15、8路平均MSE；详见[实现及验证诊断](../../docs/guides/score_ownership_zh-CN.md)。固定子集与完整留出均保留最新的均值／零比例、零／最优常数基线、阶段及前中后期诊断，不改变训练样本。

容量沿用policy104k的18GiB：该池原policy共享arena实测17869150492字节；新增每训练位置16字节标签和每局最多63字节对齐，合计保守上界17994698171字节（约16.76GiB），低于19327352832字节。此为容量核算，未预热整个104k池；arena容量不是进程树总内存上限。独立验证缓存与执行／恢复边界见[离线指南](../../docs/guides/offline_execution_zh-CN.md)。

## 评测与产物

默认选点后先对raw R37 greedy1000pair（CPU8、零搜索），再双方S1024N1搜索500pair（每世界1024次×1世界，总预算1024）。沿用上一离线组的固定对手、全部种子、裁决和12贵族牌池。搜索配置A：CUDA pool、12 worker、1000活跃、B500、local microbatch128、owner双槽、batch buckets `[64,128,256,500]`、candidate buckets `[32,48,64]`、等待5ms、shared slots2、FP32/native-cudagraph；T0、无根噪声、FP/PTP关闭。

| 产物 | 位置 |
| --- | --- |
| 启动脚本 | [run_attention128_policy_score104k_20261026.sh](../../scripts/run_attention128_policy_score104k_20261026.sh) |
| 配置 | [offline_policy_score_r21_r37_20261026.json](../../configs/entity_attention_v1/offline_policy_score_r21_r37_20261026.json) |
| 计划／源码身份 | [plan.json](plan.json)、[source.json](source.json) |
| 执行源码与日志 | `runtime/entity_attention_128x2_policy_score_r21_r37_20261026/source`及同级日志 |
| 正式训练输出 | `runs/entity_attention_128x2_policy_score_r21_r37_20261026/` |
| 选中模型／评测 | 本目录后续`selected_candidate.json`、`evaluation_configs/`、`panels/`、`results.md` |

正式输出目前只有准备阶段的身份、配置、计划与固定验证清单，无训练checkpoint／progress。原104k输入及原实验产物均保持。

## 有限验证

[tests.xml](tests.xml)：实际冻结源码的54项相关回归通过，覆盖得分／对手标签与损失、旧模型兼容、诊断、供数恢复及主KL＋MSE早停。

[cpu_validation.json](cpu_validation.json)：实际冻结训练模块、完整128×2模型，从零新AdamW；4局真实R37轨迹（2训练／2验证），全部336个决策的8路得分标签与checked引擎牌／贵族归属逐项一致。CPU串行4次更新与4 worker中断于2步再恢复至4步，模型／AdamW／采样RNG／Torch RNG／完整验证逐位一致；两辅助有效、支付仍参与、短期value头／标签／指标均不存在。另以重载模型执行3个native S8N1根，总24次模拟，两个辅助头推理调用均为0。工程验证输出只在独立runtime子目录，不是正式训练或棋力结论。

复现工程验证：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=runtime/entity_attention_128x2_policy_score_r21_r37_20261026/source/src:runtime/entity_attention_128x2_policy_score_r21_r37_20261026/source \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python artifacts/entity_attention_128x2_policy_score_r21_r37_20261026/validate.py
```

本次命令与校验SHA见[verification.json](verification.json)。尚无本实验的CUDA训练吞吐或棋力结论。
