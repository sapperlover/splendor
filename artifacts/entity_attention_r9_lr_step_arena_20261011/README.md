# 第九轮：小学习率1000步对原配置500步

2026-10-11，按用户要求完成一组200 pair／400局直接搜索对战。完整结果见[结果](results.md)，逐局配对数据见[arena.json](arena.json)，机器汇总见[summary.json](summary.json)。

**A取得212胜／2和／186负，得分53.250%，整pair bootstrap 95% CI为[48.500%, 58.000%]；先手得分51.500%，后手55.000%。** 点估计偏正，区间跨50%，尚未确认优于原配置，也不能视为严格等效。400局均有效，实际S1024/N1预算、checkpoint身份及区间复算通过，异常／非法动作／工程截断／策略循环和棋均为0。对战耗时1185.58秒，约19分46秒。

- A：第九轮lr=2e-5训练1000步，全局39,398步，checkpoint SHA `6cfd9edd97c2817274b436c3c4f50c4ba3b71c7bc14b42bd89438c39404da26f`。
- B：原第九轮lr=1e-4训练500步，全局38,898步，checkpoint SHA `ff34660fece12590762da93d031523e4ae09db4fa6af5d2284c4af0dcd57237b`。
- 共同第八轮终点、继承AdamW状态、在线池6k局（第4–9轮）、batch256。A完整留出KL／MSE为0.432231／0.718900，B为0.437501／0.715736，来源见[固定数据汇总](../entity_attention_r9_ofat_20261010/steps_100_2000.md)。

双方完整网络做S1024/N1搜索，无根噪声；`native_batched`、compiled推理、100局并行、max batch64。搜索与执行设置沿用前次有效对战；使用新的开局种子31100000–31100199，每个种子交换先后手，policy seed base31110000、search seed31130000、bootstrap seed31120000。得分为胜率＋半和棋率，95% CI按整pair做10,000次bootstrap。

本次同时改变学习率与步数，比较的是两个完整checkpoint。新开局不复用之前30900000–30900199的棋力DEV；模型仍来自单条训练轨迹，不能用这场比赛独立归因学习率、policy或value，也不能据此确定最优参数。

从仓库根目录启动／中断续跑：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u artifacts/entity_attention_r9_lr_step_arena_20261011/run.py run
```

每局持久化，完整pair在续跑时跳过，未完整pair按相同种子重跑；每10 pair刷新原始汇总。Ctrl+C或以下命令可停止，随后重跑上述命令续测：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_r9_lr_step_arena_20261011/run.py stop
```

`run.py status`刷新进度，`run.py analyze`仅复算已有结果，不启动对战。固定身份与配置见[plan.json](plan.json)、[config.json](config.json)；实际源码／环境／命令保存在`execution.json`，源码快照、日志与PID在忽略的`runtime/entity_attention_r9_lr_step_arena_20261011/`，恢复用逐局sidecar在相邻忽略的`arena.json.games/`。
