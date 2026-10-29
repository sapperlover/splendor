# Attention128：原104k池无辅助目标离线实验

2026-10-27完成状态核对：训练、完整留出选点及原两组对战均已完成；210332步早停，选中122694步。原raw R37 greedy1000 pair得分51.075%，双方S1024N1（总1024）500 pair为57.500%，见[原结果](results.md)。同一候选的新版10000 pair greedy／S128N1结果见[大面板](../offline5_vs_r37_10k_20261027/README.md)。下方按日期保留首次准备、停训和恢复过程，不再代表当前进度。

2026-10-26，用户要求在当前policy＋得分辅助脚本后串行执行。从零初始化基础`entity_attention_v1`，583682参数，新AdamW；不继承前一模型。只训练原主policy与终局value，权重各1；全部短期value、对手policy、得分归属及旧辅助头／派生标签／损失均关闭。

串行入口：

```bash
bash scripts/run_attention128_noaux104k_queue_20261026.sh
```

最初挂入等待队列时状态为waiting，当前整条链已完成；实际进程身份及状态见[队列说明](../entity_attention_noaux104k_queue_20261026/README.md)及其`launch.json`。队列阻塞等待前序监督`chain.lock`，不轮询训练／GPU。仅当前序训练、最低主KL＋MSE选点、raw R37 greedy1000pair和搜索500pair全部成功后，才启动本组的同一完整流程。前序失败／中断不启动，本组失败／中断停止；不会修改或向当前前序进程发送信号。

单组入口（独立运行／恢复）：

```bash
bash scripts/run_attention128_noaux104k_20261026.sh prepare  # 仅核对和准备
bash scripts/run_attention128_noaux104k_20261026.sh          # 训练→选点→greedy→搜索
bash scripts/run_attention128_noaux104k_20261026.sh train    # 仅训练和选点
bash scripts/run_attention128_noaux104k_20261026.sh status   # 汇总已有原件
```

## 配置及数据

相对于[前序实际离线配置](../../configs/entity_attention_v1/offline_policy_score_r21_r37_20261026.json)，仅移除`continuation_auxiliary`和仅用于辅助的`encoder_grad_norm_every_steps`，改用已有`offline_early_stopping`基础模型入口；其余配置逐字段相同。[新配置](../../configs/entity_attention_v1/offline_noaux_r21_r37_20261026.json)。

| 项目 | 保持的设置 |
| --- | --- |
| 数据 | 原R21–R37共104000局、同一manifest、NPZ／主标签／整局split保持 |
| 训练／留出 | 93537／10463局，7478428／835367决策；正权重主policy1559755／174056位置 |
| 基础模型 | attention128×2、4头、FFN512、market relations、global_players_market读取、dropout0 |
| 优化 | lr1e-4、FP32、新AdamW、weight decay1e-4、clip1；policy高预算与value全预算各独立B256 |
| 预算 | 12次value曝光，上限350552步；每1000步及全部5%节点完整checkpoint |
| 验证／选点 | 固定4096子集及每5%原完整留出；主KL＋终局MSE连续3次严格上升即停，持平／下降重置、step0排除；按到达节点最小联合指标选完整模型 |
| 执行 | 4 worker共享只读训练池、18GiB容量、独立验证缓存；流式会话统计、state-only value传输、批量标量回读 |
| 记录 | 逐步training_metrics.jsonl＋小型参考SHA；不写大抽样账本、重复steps／会话metrics |
| 种子 | 初始化43100000、抽样43110000、固定验证43120000；未使用辅助初始化seed |

18GiB保持。无辅助数组后，同池原policy arena实测17869150492字节（约16.64GiB）是保守容量上界，低于19327352832字节；没有预热整个104k池。容量不是进程树总内存限制，详见[离线执行指南](../../docs/guides/offline_execution_zh-CN.md)。

评测完全沿用前序的固定raw R37、开局／策略／bootstrap种子与规则：greedy CPU8／1000pair／零搜索，然后双方S1024N1搜索500pair（每世界1024×1世界，总1024）。搜索配置A：CUDA pool、12 worker、1000活跃、B500、local microbatch128、owner双槽、batch buckets `[64,128,256,500]`、candidate buckets `[32,48,64]`、等待5ms、shared slots2、FP32/native-cudagraph、T0、无根噪声、FP/PTP关闭，12贵族。

## 验证与产物

[plan.json](plan.json)和[source.json](source.json)冻结配置／数据／执行源码／native／牌池及对手身份。正式训练输出为`runs/entity_attention_128x2_noaux_r21_r37_20261026/`；执行源码与日志在同名`runtime/`。本目录后续保存选中模型身份、两组评测及`results.md`，同命令恢复。

[tests.xml](tests.xml)保存相关回归：无辅助正式配置及模块路由、主KL＋MSE早停／恢复、数据抽样、等待监督锁及完整流程成功边界、串行失败停止。初次回归暴露旧队列测试读取真实已完成实验的问题；已只将该测试的ROOT隔离到pytest临时目录，未改任何运行实现或历史产物，初次结果保留[initial_tests.xml](initial_tests.xml)。

[cpu_validation.json](cpu_validation.json)：4局真实R37数据，完整583682参数基础模型，从零新AdamW；串行4次CPU更新与4 worker中断2步再恢复到4步，模型／AdamW／采样RNG／Torch RNG／完整验证逐位一致，验证没有辅助指标。通过新监督适配器实际启动`offline_early_stopping`训练恢复和选点两个子进程，确认没有误用continuation入口；严格重载后3个native S8N1根，共24次模拟。测试输出仅在独立runtime子目录，不是正式训练或棋力结果。

当前前序配置、runner、冻结计划／源码身份及launcher的保护SHA见[parent_preservation.json](parent_preservation.json)。[verification.json](verification.json)记录本次命令、验证与队列启动身份。正式训练只有等待条件满足后才开始，尚无本实验棋力结论。
