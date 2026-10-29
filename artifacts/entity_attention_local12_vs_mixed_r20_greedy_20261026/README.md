# 第12轮 final 对原混合预算 R20：1000 pair greedy

2026-10-26按用户指定完成1000 pair／2000局，**第12轮得分70.350%，95%整pair bootstrap区间[68.350%,72.325%]**，1398胜、18和、584负。先手得分71.850%，后手68.850%；CPU 8 worker，完整记录耗时103.10秒，其中对战阶段约102.1秒。和棋各计半分，得分不等于纯胜局比例。

这个面板支持第12轮模型的greedy明显强于原R20。它与[人类棋谱首选吻合度下降](../bga_927500015_local12_analysis_20261026/comparison.md)可以同时成立：单盘人类行动吻合度与模型间对战棋力衡量的对象不同。本次只测双方零搜索的直接对战，没有测两者的搜索对战。

## 冻结模型与评测口径

| 一方 | 权重 | SHA-256 |
| --- | --- | --- |
| A：当前分支第12轮 final | `runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt` | `2a6304e3470cab3dad78ec0a2789ee892892e741f3731cafc2197cde9bd2d45d` |
| B：原混合预算 raw R20 | `runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt` | `fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd` |

两个模型均由用户在对战前指定；R20是之前分析BGA人类棋谱的同一权重。第12轮采用完整final，未根据本次对战重新选点。

- 当前12贵族规则池，指纹`084574b026e41ad6e5c64dfe36d74fb84a7604f6a0b0eb0ab59227c32a7ff17b`；与人类棋谱重放保留的10贵族历史快照分别解释。
- 双方确定性greedy、零搜索，CPU 8 worker，每worker单线程。
- 开局seed为37000000–37000999，每个开局交换先后手；policy seed base为37010000。沿用日常DEV面板，没有另设独立FINAL种子。
- bootstrap以整pair为单位，10000次，seed37020000。
- 单局上限400个完成玩家回合，启用确定性策略循环和棋裁决。
- 2000局全部有效，异常、非法动作、工程截断、不变量错误、强制跳步和策略循环裁决均为0。

## 产物与复核

[config.json](config.json)和[execution_plan.json](execution_plan.json)保存配置、命令、种子、权重身份与对战前选择；[summary.json](summary.json)保存紧凑结果和代码身份，[pair_scores.json](pair_scores.json)保存1000个开局的先后手得分，[verification.json](verification.json)保存核验结果。

重型原始结果位于`runtime/entity_attention_local12_vs_mixed_r20_greedy_20261026/evaluation.json`，逐pair原始记录在相邻`evaluation.json.games/`，源码快照／Git版本／脏工作区补丁／实际命令／环境和资源记录在`recording/evaluation/recording/`。这些运行产物受现有忽略规则覆盖，未修改`.gitignore`。

已用[summarize.py](summarize.py)逐一核对1000份sidecar与汇总中的2000局，复算胜和负及整pair区间，并检查开局种子、交换先后手、policy种子、权重SHA、配置、规则和执行源码身份；单次执行正常完成，核验通过。复核命令：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_local12_vs_mixed_r20_greedy_20261026/summarize.py
```
