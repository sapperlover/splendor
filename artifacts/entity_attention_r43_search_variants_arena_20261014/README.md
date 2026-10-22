# 原 R43：S1024／S4096／随机节点／CPW 对 greedy 串行队列

四组按以下顺序串行执行，各 500 pair／1000 局，共 4000 局；两侧均为原 R43 完整网络（global step 72898），不使用离线候选。

| 顺序 | 搜索方 | 后端／参数 | 对手 |
| --- | --- | --- | --- |
| 1 | S1024 | native，N1 确定化 | R43 greedy |
| 2 | S4096 | native，N1 确定化 | R43 greedy |
| 3 | 随机节点 S1024 | native_chance，不启用扩宽 | R43 greedy |
| 4 | CPW S1024 | native_chance，k=0.25、α=0.5 | R43 greedy |

随机节点与 CPW 使用此前实现的 `NativeChancePUCTSearchPolicy`；N1 表示一个共享统计池，模拟仍重采对手暗牌与物理随机事件。CPW 沿用此前棋力实验的 k0.25/α0.5，不采用配置类默认的 k1。

模型：`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt`，SHA256 `b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。

四组与[上一批 S128／S256／S512](../entity_attention_r43_low_search_budget_arena_20261014/results.md)共用开局 36300000–36300499、policy seed base 36310000、搜索 seed 36330000、bootstrap seed 36320000；跨组差值 bootstrap seed 36340000。同开局交换先后手，和棋半分，整 pair bootstrap 10000 次，95% 区间不做多重比较校正。c_puct1.5、根噪声0、行为温度0、强制阶段折叠、simulation transition cap512；最大完成玩家回合400，保留现行策略循环判和并单独统计。

执行配置沿用上一批：CUDA/native_batched compiled、active_games100、max_batch64、候选桶16/32/48/64、actor_projection=false、单CPU线程；每组一个独立子进程，前一组通过核验后才进入下一组。生产源码、原生扩展或模型身份改变时队列报错停止，避免混用版本。子进程异常会停止队列，原始产物保留，可用相同命令断点续跑。

从仓库根目录启动后台队列（仅启动一次）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python artifacts/entity_attention_r43_search_variants_arena_20261014/launch.py
```

恢复中断后的队列也使用上述启动命令；已完成的组／pair 会跳过。只读复算：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 python artifacts/entity_attention_r43_search_variants_arena_20261014/run.py analyze
```

- [results.md](results.md)、[summary.json](summary.json)：每组结束自动汇总胜／和／负、先后手得分、配对置信区间，以及 S4096−S1024、随机节点−S1024、CPW−S1024、CPW−随机节点的共同 greedy 对手得分差。这些不是搜索之间直接对战的胜率。
- 原始逐局结果 `panels/*.json` 保存实际 simulation、逻辑深度、精确终局命中、transition cap、动作变化与工程错误计数；完成组逐局核验预算、模型 SHA、种子、交换先后手与 CI。
- `plan.json`、`configs/` 冻结配置；`execution.json` 保存源码、native、环境、命令与结束状态；实验索引在每组结束自动更新。
- 后台 PID、总日志与当前组分别位于忽略的 `runtime/entity_attention_r43_search_variants_arena_20261014/{launch.json,queue.log,live.json}`；每组日志为同目录 `<组名>.log`。必要时对 `live.json` 中的队列 pid 发 `SIGINT`，会转发给当前子进程并保留续跑状态。
- 逐局 `.json.games/` sidecar 与 runtime 不纳管；未改生产搜索代码、训练配置或历史实验原件。报告中的 `incomplete` 不等于进程仍在运行，当前状态以 `live.json`、进程与日志为准。
