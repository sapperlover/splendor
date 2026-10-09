# 多搜索 worker 供给单 GPU：窗口与整轮结果

2026-10-01 核对已有记录，未新增性能实验。所有窗口请求 35 秒、预热 2 秒，FP32、S=1024/N=1、active=500。pool 配置 W=8；baseline 为单进程 `native_cuda`。checkpoint 路径指向同一 PPO latest；窗口 JSON 本身未保存内容 SHA、测量代码版本和硬件，不能用当前机器或 HEAD 补填这些历史字段。

## 预热后的开局短窗口

所核对的 `single_b500` 与 `pool_b500_mb64_s1_w5` 窗口均未完成对局，记录约 2,000 次根决策，平均每局约 4 步。这些窗口主要覆盖开局阶段，不能代表整轮中后盘及尾部收敛阶段的吞吐；预热完成也不意味着对局负载已达到稳态。

| 文件 | B | microbatch | slots/worker | wait ms | sim/s | 平均 batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pool_b256` | 256 | 16 | 2 | 1 | 33,221 | 167.4 |
| `pool_b256_mb64_s1_w5` | 256 | 64 | 1 | 5 | 45,914 | 255.8 |
| `pool_b500` | 500 | 16 | 2 | 1 | 33,515 | 167.2 |
| `pool_b500_mb64_s1` | 500 | 64 | 1 | 1 | 53,340 | 261.7 |
| `pool_b500_mb64_s1_w5` | 500 | 64 | 1 | 5 | 68,220 | 495.2 |
| `pool_b500_mb64_s1_w5_rerun` | 500 | 64 | 1 | 5 | 67,886 | 494.2 |
| `single_b500` | 500 | — | — | 0 | 58,335 | 500.0 |

microbatch 和 slots/worker 两列仅适用于 pool；baseline JSON 中这两个字段是未使用的默认值，不代表单 owner 的执行布局。baseline 使用 `pipeline_slots=1`。

microbatch=16、slots=2 的最初 pool 窗口低于 baseline；microbatch=64、slots=1、wait=5ms、B=500 的两次窗口约 67.9k–68.2k，baseline 58.3k，提升约 16%–17%。不能把“多 worker”本身认作充分条件；微批和合批等待显著影响结果。

## 各一轮完整 500 局

| 路径 | 墙钟秒 | 局/分钟 | 整轮 sim/s |
| --- | ---: | ---: | ---: |
| full_single_b500 | 765.4 | 39.20 | 43,193 |
| full_pool_b500 | 744.1 | 40.32 | 44,436 |

两组均完成 500/500、0 exceptions、0 engineering_truncations，统计完整。pool 为 W=8/A=500/B=500、microbatch=64、slots=1、wait=5ms。pool 744.1s 对 single 765.4s，墙钟缩短约 2.8%；开局短窗口的吞吐提升约 16%–17%，不能直接推广为整轮加速幅度。每档仅一轮，不证明重复稳定收益，不更换生产推荐。

两组 manifest 均记录 revision `638a06e…` 且 dirty=true，模型 SHA 均为 `009d17d2…`。compile_failure_fallbacks 为零，但两组都有约 6,100 次超宽候选 eager 调用；配置中的 compiled 不是全程每次调用均 compiled。GPU 峰值 pool 约 298 MiB、single 约 82 MiB；worker RSS/累计 CPU 与单 owner 口径不同，不用单个最大值伪装总进程峰值。

## 可复核产物

2026-10-02 新增的有限 [分项时间统计](pipeline_profile_zh-CN.md) 核对 gather、等待、实际候选宽度与尾段；历史完整 500 局原件和本报告的原排名保持不变。

[比较摘要](../../artifacts/inference_throughput/multiworker_gpu/comparison_summary.json)、[完整 manifest/summary 副本](../../artifacts/inference_throughput/multiworker_gpu) 与全部七份原窗口保留；来源哈希见 [保全清单](../../artifacts/retention_manifest_2026-10-01.json)。原 `runtime/module_b/` 的 NPZ、日志、运行状态没有移动。[配置与恢复指南](../guides/multiworker_gpu_zh-CN.md) 描述执行语义。
