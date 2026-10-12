# 原第30轮：S1024对greedy、S4096对S1024

串行执行两组，各200 pair／400局；所有侧均使用**原配置第30轮完整网络**，不使用value权重干预分支或双网络组合。

| 顺序 | A | B | 检验目标 |
| --- | --- | --- | --- |
| 1 | 第30轮 S1024/N1 | 第30轮 无搜索greedy | 当前网络加入搜索的收益 |
| 2 | 第30轮 S4096/N1 | 第30轮 S1024/N1 | 增大搜索预算的边际收益 |

模型是主自迭代绝对第30轮、目录`round-020`终点、global 59898：

`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-020/checkpoints/final.pt`

SHA256：`9fc258b211954cf32cd7b688d5d534a502441304ea7cc92a7a35e059a7c91b0f`。来源与配置冻结在[plan.json](plan.json)，本实验不训练模型。

第一组开局种子32900000–32900199，policy种子基数32910000、bootstrap种子32920000、search种子32930000；第二组分别为33000000–33000199、33010000、33020000、33030000。每个开局交换先后手，和棋半分，整pair bootstrap 10000次。A得分分别对应S1024与S4096；两组使用独立开局面板，不计算跨组配对差值。

除搜索次数外沿用已有评测设置：N1、c_puct1.5、无根噪声、折叠强制阶段、单次模拟最多512次transition、最多400个完整回合；native batched、CUDA compiled搜索、active games100、max batch64、candidate buckets16/32/48/64、wait0。greedy直接对合法动作logits取argmax，无树搜索。相同checkpoint和推理配置在每组共享模型所有者；两组分别启动子进程，依次运行。最终自动核对有效局数、checkpoint SHA、两侧实际预算和配对置信区间。

从仓库根目录启动；中断后使用相同命令续跑，已完成pair保留、未完整pair重放，已完成组跳过。SIGINT／SIGTERM转发到当前评测子进程，同实验文件锁防止重复启动。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u artifacts/entity_attention_r30_search_budget_arena_20261011/run.py run
```

每10pair保存评测汇总，每组结束及链路退出时自动更新[结果表](results.md)、[summary.json](summary.json)及机器实验索引。配置在`configs/`，原始逐pair结果在`panels/`，可恢复sidecar在相邻被忽略的`.json.games/`。运行状态、分组日志和源码快照位于`runtime/entity_attention_r30_search_budget_arena_20261011/`，命令、环境、git与源码身份记录在`execution.json`。同一命令可连续完成两组，无需轮询。

结论限定本checkpoint、N1与固定DEV面板；区间跨50%表示当前样本不足以确认优劣，不能证明等强。未校正多重比较。

启动前短验证已通过：同一模型、同一CUDA compiled/native batched链路，各1pair的S8对greedy、S16对S8共4局；检查真实搜索决策、两侧实际预算、有效局数、模型SHA、评分与CI，见[verification.json](verification.json)。短验证使用独立种子33100000／33200000，仅验证链路，不计入正式200pair，也不作为棋力结论。
