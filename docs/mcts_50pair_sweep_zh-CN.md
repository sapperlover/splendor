# MCTS 四组 50 pairs 固定评测

从仓库根目录运行，四组按编号顺序执行；每组 50 个配对 seed，每个 seed 交换先后手各打一局，共 100 局：

```bash
conda run --no-capture-output -n splendor python -u scripts/run_mcts_50pair_sweep.py
```

中断后使用相同配置和输出路径续跑：

```bash
conda run --no-capture-output -n splendor python -u scripts/run_mcts_50pair_sweep.py --resume
```

只检查配置、checkpoint 和命令，不启动对局：

```bash
conda run -n splendor python scripts/run_mcts_50pair_sweep.py --dry-run
```

| 顺序 | A | B |
| --- | --- | --- |
| 1 | PUCT S=256, N=2 | PUCT S=1024, N=1 |
| 2 | PUCT S=1024, N=1 | PUCT S=1024, N=2 |
| 3 | PUCT S=1024, N=2 | PUCT S=2048, N=1 |
| 4 | PUCT S=1024, N=1 | 折叠强制阶段的 alpha-beta depth=6, top_k=5 |

配置在 [`configs/mcts_50pair_sweep/`](../configs/mcts_50pair_sweep/)；各组使用同一个 `local_005000000_lifetime_005512112.pt` checkpoint、棋局 seed 9800000–9800049、policy seed base 9810000、CPU 8 workers、最多 400 完成回合。PUCT 的 `c_puct=1.5`、强制阶段折叠、模拟转移上限 512 一致。评测调用 `choose()`，无根噪声，温度为 0。第 4 组 alpha-beta 使用 N=1、`leaf_value_clip=1.0`。四组均对成对 seed 的得分率做 100000 次 bootstrap 重采样，区间仅用于描述这 50 pairs 的抽样不确定性。

每组完整汇总写入 `artifacts/mcts_50pair_sweep/<编号与对局名>.json`，包含配置、checkpoint 身份、逐局结果、先后手胜平负、有效局数、异常/截断、搜索计数与耗时、成对 bootstrap 区间。其同名 `.json.games/` 是可续跑逐局记录，由 Git 忽略；中断后应保留。发生失败时脚本停止，不开始后续组。结果尚未运行生成。
