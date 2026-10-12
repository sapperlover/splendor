# 新 P9 / 旧 P4 组合棋力对战

2026-10-10已完成两组各200pair。新P9＋旧V4对P4V4得分58.500%，95% CI为[53.500%,63.625%]；旧P4＋新V9为50.875% [46.125%,55.750%]。已有P9V9为58.750%。P9V4减P9V9的同开局得分差为-0.250个百分点，区间[-6.875,+6.375]；未确认两者有差异，也未证明等效。P4V9减P9V9为-7.875个百分点[-14.500,-1.000]。结果支持policy来源的重要性，不能把双网络组合对战解释成可相加的独立head贡献；完整统计见[结果表](results.md)。

用户指定的两组对战，依次运行：新 P9＋旧 V4、旧 P4＋新 V9，各对旧 P4V4 做200 pair／400局 S1024/N1搜索。新P9来自本次P4起点自迭代第九轮38,898步；旧P4来自大池离线蒸馏34,398步。每个分支保留来源网络的完整encoder；不重新训练，也不拼接共享encoder的head。

沿用已完成第九轮P9V9 vs P4V4评测的200个DEV开局（30900000–30900199）、搜索随机种子及推理配置：native_batched、compiled、active_games=100、max_batch_size=64、S1024/N1、无根噪声。两组串行，每组独立子进程；不设置人为内存上限。已有P9V9对起点58.750%用于同开局配对得分差比较，不重复评测。

固定来源、SHA、配置和开局见[plan.json](plan.json)与`configs/`；实际源码／环境／启动命令见[execution.json](execution.json)。进度与原始逐pair汇总写`panels/`；可恢复逐局sidecar由既有评测器维护，不纳入Git。模型保存在`runs/entity_attention_p4_p9_crossplay_20261010/compositions/`，日志、锁与源码快照在`runtime/entity_attention_p4_p9_crossplay_20261010/`。

从仓库根目录运行；同一命令可中断续跑，已完成pair不重测：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python -u artifacts/entity_attention_p4_p9_crossplay_20261010/run.py run
```

查看进度／请求停止，将末尾`run`换成`status`／`stop`。停止会保留已完成pair；尚未完成的pair续跑时重新评测。最终[结果表](results.md)、[机器汇总](summary.json)自动生成，CI按整pair bootstrap；同时给出两组合分别减P9V9及两组合之间的同开局得分差。得分差比较的是对固定起点的表现，不是两组合直接互打。

本实验是单条训练轨迹、固定对手、复用DEV的探索性分支交换，未作多重比较校正。结果包含来源encoder、policy/value与搜索之间的交互，不能把两种组合的分数变化直接相加成独立head贡献。
