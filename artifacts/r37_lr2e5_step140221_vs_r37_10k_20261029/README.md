# R37初值／104k池／lr2e-5：固定140221步对原R37

2026-10-29，按用户最后明确要求，**固定约14万步的140221 checkpoint**，不取228000步最新节点，不自动跟随训练进度或重新选点。正式评测未启动；不恢复训练，不附加现有队列。

```bash
bash scripts/run_r37_lr2e5_step140221_vs_r37_10k_20261029.sh
```

默认严格串行：先对raw R37执行10000pair greedy，全部成功后再执行10000pair双方S128N1。每pair同一开局交换先后手，共20000局／面板；总20000pair／40000局。同一条命令恢复：跳过已完整验证面板，按原种子复用已完成pair、重跑半个pair；保留相邻`.json.games/`。失败或中断即停止后续面板，`chain.lock`拒绝重复并发。

## 固定模型

候选来自`entity_attention_r37_104k_lr2e5_20261027`：R37权重初值、原104k池、新AdamW／原离线RNG、无辅助、lr2e-5。用户指定的文件为：

```text
runs/entity_attention_r37_104k_lr2e5_20261027/checkpoints/step-000140221.pt
SHA256: 42fbd07c413190389ac5f2cab73ef41ff4a10f5402a13e099a37d49aa3cb6caa
```

原完整留出KL为0.19661209349431433、终局MSE为0.7320730541044629，联合0.9286851475987772；在本次核对的13个已完成完整留出节点中最低。训练当前中断于228000步，尚无最终`selection.json`；本评测明确使用用户指定的中途最佳节点，不依赖训练完成或最终选点文件。后续即使续训，也保持140221步。

对手为原R37：

```text
runs/entity_attention_r31_10k_p60k_iteration_20261023/rounds/round-005/checkpoints/final.pt
SHA256: 611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045
```

## 配置与种子

greedy为CPU8 worker、确定性根policy argmax、零搜索。搜索沿用配置A：CUDA pool、12 search worker、1000活跃、B500、local microbatch128、owner双槽、batch buckets `[64,128,256,500]`、candidate buckets `[32,48,64]`、等待5ms、shared slots2、FP32/native-cudagraph、reused传输和fail-fast。双方均为**S128N1，每世界128×1世界＝总128**；c_puct1.5、折叠强制阶段、每模拟最多512状态转移、无根噪声、T0、FP/PTP关闭。

沿用此前[五模型10k评测](../offline5_vs_r37_10k_20261027/README.md)的种子库，方便对照：开局47270000–47279999、policy_seed_base47300000、双方search seed47330000、bootstrap seed47340000。完整12贵族规则，既有400完整回合工程上限及确定性循环判平；和棋半分，以完整pair做10000次bootstrap并报告95%区间。

每100pair输出进度，每1000pair及结束／中断更新全量summary，逐局日志持续落盘。完成时检查双方checkpoint SHA、开局及先后手、完整pair数和逐局实际模拟预算。调用、起止时间及源码身份随会话保存，底层执行统计保留在原始评测结果。

## 产物和验证

- [plan.json](plan.json)：固定候选／对手、来源、顺序、规模及种子身份。
- `configs/01_greedy_step140221_vs_R37.json`及`configs/02_search_step140221_vs_R37.json`：正式面板配置。
- [source.json](source.json)：实际冻结Python／native／牌池／配置，执行源码在同名`runtime/…/source`。代码patch限定源码及本次入口，未复制Git中历史原始数据停止跟踪的巨大diff。
- `panels/<面板>.json`及相邻`.json.games/`：后续本地原始结果与恢复日志；Git只保留紧凑[summary.json](summary.json)和[results.md](results.md)。
- 同名runtime的`<面板>.log`、`<面板>_execution.json`：标准输出及面板会话。整体调用／环境写本目录后续`execution.json`。

`prepare`只核对和准备，`status`只读取当前状态，不写报告或启动评测：

```bash
bash scripts/run_r37_lr2e5_step140221_vs_r37_10k_20261029.sh prepare
bash scripts/run_r37_lr2e5_step140221_vs_r37_10k_20261029.sh status
```

本次15项相关回归通过：固定140221身份、两组规模／顺序／原种子与搜索A、greedy失败不启动搜索、只读status，以及复用执行器的恢复／跳过／实际预算检查。原始JUnit在本地runtime，不纳管。

两个真实checkpoint使用实际冻结代码完成CPU严格加载、greedy根前向与各一个native S128N1根，总256次模拟；没有完整对局、CUDA评测或棋力结论。源训练checkpoint和已有训练记录未写入。原始验证路径、SHA和命令见[verification.json](verification.json)。
