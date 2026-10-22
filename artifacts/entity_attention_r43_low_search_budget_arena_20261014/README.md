# 原 R43：S128／S256／S512 对 greedy

按用户指定，各 500 pair／1000 局，共 3000 局。两侧均使用原 R43 完整模型，不使用 PSW 或辅助头离线候选。结果与核验见 [results.md](results.md)，机器汇总见 [summary.json](summary.json)。

三组已完成：S128／S256／S512 对 greedy 得分 68.300%／70.050%／74.000%，各自区间均高于 50%。共同对手下 S512 减 S128 的配对得分差为 +5.700pp [ +1.900, +9.500 ]pp；S256 减 S128 为 +1.750pp [ −1.900, +5.550 ]pp，尚未确认。S512 减 S256 为 +3.950pp [ +0.249, +7.700 ]pp；这些区间未做多重比较校正，且均不代表不同搜索预算直接对战的得分。

3000 局均自然终局，异常、非法行动、工程截断、不变量失败、策略循环判和均为零。[最终核验](verification.json) 确认三组仅预算与标签不同、开局与 policy seed 日程相同、先后手正确交换，代码、原生扩展与模型身份一致。S128 已有明显搜索收益，但 S512 在本面板仍有额外收益；采集后训练效果需另行验证。

- checkpoint：`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt`；绝对轮 43、global step 72898。
- checkpoint SHA256：`b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。
- 三组共用开局种子 36300000–36300499，每 pair 同开局交换先后手。policy seed base 36310000，搜索 seed 36330000，bootstrap seed 36320000，跨组差值 bootstrap seed 36340000。
- 搜索 N1、c_puct=1.5、根噪声 0、按访问数 argmax；强制阶段折叠、模拟转移上限 512。greedy 按模型策略 argmax。
- 沿用 CUDA/native_batched compiled 评测：active_games=100，max_batch_size=64，候选桶 16/32/48/64；串行运行三组。
- 和棋半分，策略循环判和沿用现有评测规则，单独列出。CI 按整 pair bootstrap 10000 次，未作多重比较校正。

从仓库根目录执行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python -u artifacts/entity_attention_r43_low_search_budget_arena_20261014/run.py run
```

`prepare` 冻结配置，`run` 自动断点续跑，`analyze` 重新核验汇总。配置和计划保存在本目录，逐局原始汇总保存在 `panels/`；日志、源码／配置运行快照在忽略的 `runtime/entity_attention_r43_low_search_budget_arena_20261014/`，可恢复逐局 sidecar 为相邻的 `.json.games/`。`execution.json` 保存代码、设备与命令身份。

这些比赛测量低预算搜索对同模型 greedy 的收益；共用对手的跨组得分差不是 S256 对 S128 等直接对战，也不能代替采集后训练收益的验证。没有训练或修改生产代码。
