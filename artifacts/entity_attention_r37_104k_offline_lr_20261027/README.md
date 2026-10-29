# R37权重起点：104k池的两组无辅助离线训练

2026-10-29后续结果：1e-4组已完成原完整流程，greedy49.95%、S1024N1搜索51.80%；2e-5组训练记录中断于228000步，原自动完整流程尚未结束。本次独立固定当前最佳140221步，对raw R37 greedy1000 pair已完成：**51.100%**，95%区间**[49.000%, 53.175%]**，未确认提升，见[独立评测](../entity_attention_r37_104k_lr2e5_best_vs_r37_greedy_20261029/results.md)。下方“未启动”等保留首次交付时点；本次没有恢复训练。

2026-10-27，按用户要求准备两条独立命令。第二个`2e--5`按`2e-5`解释；正式训练、评测及队列均未启动。基于最近的104k无辅助离线配置，唯一训练变动是模型初始权重改为原R37，以及第二组学习率改为2e-5。

```bash
# lr=1e-4
bash scripts/run_r37_104k_offline_lr1e4_20261027.sh

# lr=2e-5
bash scripts/run_r37_104k_offline_lr2e5_20261027.sh
```

两条命令各自独立运行／恢复，不相互等待，也不接入现有队列。默认均沿用原离线完整流程：训练→完整留出最低主KL＋MSE选点→raw R37 greedy1000pair→双方S1024N1搜索A500pair（每世界1024×1世界，总1024）。这不是上一项五模型10k pair复核，也不自动增加10k评测。

末尾加`train`只训练并选点，不启动棋力评测；`prepare`仅核对／准备，`status`汇总已有原件。例如：

```bash
bash scripts/run_r37_104k_offline_lr1e4_20261027.sh train
bash scripts/run_r37_104k_offline_lr2e5_20261027.sh train
```

## 起点和优化器语义

按“其他所有配置都用离线训练配置”实现为**仅加载R37模型权重，新建AdamW，使用原离线初始化／抽样种子**。不继承R37在线优化器动量、在线抽样RNG和训练步数；本实验从本段step0计数，中断后恢复本实验自己的AdamW和所有RNG。原R37模型、优化器和文件字节均保持。

起点：`runs/entity_attention_r31_10k_p60k_iteration_20261023/rounds/round-005/checkpoints/final.pt`，SHA `611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045`，原累计86423步；基础attention128×2、4头、FFN512、market relations、global_players_market读取，共583682参数。两组全部辅助目标关闭，仅原主policy及终局value按权重1／1训练。

不直接调用旧`offline_finetune`：该入口要求继承在线AdamW／RNG及父learner设置，并按固定步长验证，与本次离线默认的新AdamW和5%验证不符。本目录[train.py](train.py)只替换初始化和身份记录，供数、更新、完整验证、早停、选点与恢复均复用现行公共训练器；[run.py](run.py)复用原104k完整流程监督器。

## 保持的离线配置

| 项目 | 设置 |
| --- | --- |
| 数据 | 原R21–R37共104000局，原NPZ／主标签／整局split／manifest保持，不重新采集或重标注 |
| 训练／留出 | 93537／10463局，7478428／835367决策；正权重主policy1559755／174056位置 |
| 优化 | 新AdamW、weight decay1e-4、clip1、FP32；仅lr分别为1e-4／2e-5 |
| 抽样 | 原policy-high-value-all-v1；主policy高预算位置与value全预算位置各独立B256 |
| 预算／保存 | 12次value曝光，上限350552步；每1000步及全部5%节点保存完整checkpoint |
| 验证／早停 | 固定4096子集及每5%完整原始留出；主KL＋终局MSE连续3次严格上升即停，持平／下降重置；step0不参与早停／选点 |
| 选点 | 全部到达百分比节点内主KL＋MSE最低的完整checkpoint，平局取较早步 |
| 执行 | training_workers=4共享只读供数，18GiB训练容量；流式累计、state-only value传输、批量标量回读；独立验证缓存 |
| 种子 | 原初始化43100000、抽样43110000、固定验证43120000 |
| 记录 | 主逐步training_metrics.jsonl及小型参考SHA，不写大抽样账本或重复会话metrics |

18GiB沿用相同无辅助池配置。同池旧policy完整arena实测17869150492字节约16.64GiB，是无辅助池的保守容量上界；没有预热整个104k池。该容量不是整个进程树的内存上限。[通用执行／恢复边界](../../docs/guides/offline_execution_zh-CN.md)。

两组默认评测继续使用原固定raw R37、原开局／策略／bootstrap种子及12贵族规则：greedy CPU8；搜索配置A为CUDA pool、12 search worker、1000活跃、B500、local microbatch128、owner双槽、batch buckets `[64,128,256,500]`、candidate buckets `[32,48,64]`、等待5ms、shared slots2、FP32/native-cudagraph，T0、无根噪声、FP/PTP关闭。原评测流程与配置未改动。

## 文件与验证

| lr | 配置 | 独立输出／执行身份 |
| --- | --- | --- |
| 1e-4 | [offline_r37_104k_lr1e4_20261027.json](../../configs/entity_attention_v1/offline_r37_104k_lr1e4_20261027.json) | `entity_attention_r37_104k_lr1e4_20261027` |
| 2e-5 | [offline_r37_104k_lr2e5_20261027.json](../../configs/entity_attention_v1/offline_r37_104k_lr2e5_20261027.json) | `entity_attention_r37_104k_lr2e5_20261027` |

每组的`artifacts/<身份>/`保存紧凑plan／source／summary和后续选点身份／报告；`runs/<身份>/`保存实际训练输入身份、checkpoint、主日志及留出；`runtime/<身份>/source`为实际执行冻结源码，运行日志及本次小型CPU验证也在runtime。原始评测结果及sidecar仍按原监督器的本地路径保存，Git范围遵循当前忽略规则。

本次冻结源码patch只覆盖执行源码、配置及本次入口，避免把仓库既有历史原始数据停止跟踪的巨大diff复制进新发布；全部实际执行Python、native、牌池和配置另有逐文件SHA。未修改Git忽略规则或索引，原暂存与历史实验保持。

有限回归覆盖：R37权重逐位加载、带有更新动量的父checkpoint不能泄漏AdamW状态、原离线RNG／本段零步初始化、两组配置逐字段一致性、模型不匹配明确失败、实际监督训练／选点模块路由，以及原主KL＋MSE百分比早停／恢复。初始测试fixture误用有限运行中不会生成的`final.pt`，已改为该2步实际checkpoint，仅测试文件调整，初始结果留在本地runtime。

每组另用4局真实R37数据、完整128×2模型做串行4步与4 worker中断2步再恢复到4步。初始模型与原R37逐位一致、AdamW为空、原离线抽样／Torch RNG一致；训练后模型／AdamW／采样RNG／Torch RNG／完整验证均逐位一致，实际冻结训练CLI与选点CLI通过。工程数据和checkpoint仅在独立runtime子目录，不能当作正式训练或棋力结论。

紧凑验证汇总与本地原始验证路径、SHA及命令见[verification.json](verification.json)。未运行CUDA训练、全104k预热或正式棋力评测。
