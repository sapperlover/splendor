# 全量持久化 NPZ 采样盘点（2026-10-12）

依据迁移前实际文件统计，不按日期筛选。逐局列仅计 `games/game-*.npz`，排除追加标签和源码快照；各来源间可能共享种子或轨迹，数字不代表独立有效对局数。完整分类和使用限制见 [归档报告](README.md)。

| 来源（仓库相对路径） | 逐局 NPZ | 全部 NPZ | NPZ 已分配 MiB | 当前43000池 | 旧29000池 | 本次处理 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/entity_attention_r9_online_lr2e5_1000_forever_20261011` | 34,023 | 34,023 | 1511.61 | 34,000 | 0 | 逐局原件保留，544个中间权重归档 |
| `runs/az_theta5_10x1000_seed1` | 10,000 | 10,010 | 434.16 | 0 | 0 | 逐局采样归档，模型与配置保留 |
| `runs/entity_attention_hybrid_s1024_n1_from_round9_seed1` | 10,000 | 10,000 | 463.48 | 0 | 10,000 | 保留，冻结池原始数据 |
| `runs/entity_attention_hybrid_static_graph_seed1` | 9,000 | 9,001 | 423.48 | 0 | 9,000 | 保留，冻结池原始数据 |
| `runs/entity_attention_p4_pool_iteration_20261010` | 9,000 | 9,000 | 404.12 | 9,000 | 0 | 保留，冻结池原始数据 |
| `runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1` | 4,000 | 4,000 | 181.33 | 0 | 4,000 | 保留，冻结池原始数据 |
| `runs/entity_attention_hybrid_s512_n2_from_round9_seed1` | 3,305 | 3,305 | 152.96 | 0 | 0 | 整体归档，旧路径兼容 |
| `runs/entity_attention_10x1000_seed1` | 3,000 | 3,003 | 135.07 | 0 | 3,000 | 保留，冻结池原始数据 |
| `runs/az_loop_seed1` | 2,500 | 2,500 | 109.13 | 0 | 0 | 逐局采样归档，模型与配置保留 |
| `runs/entity_attention_avg4_behavior_20261009` | 2,400 | 485,334 | 2061.60 | 0 | 2,400 | 追加标签归档，2400局行为原件保留 |
| `runtime/end_to_end_speed_2026-10-03` | 2,004 | 2,006 | 89.78 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/native_search` | 1,530 | 1,530 | 68.51 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/module_b` | 1,000 | 1,000 | 44.82 | 0 | 0 | 整体归档，旧路径兼容 |
| `runs/entity_attention_chance_training_probe_20261007` | 606 | 606 | 29.59 | 0 | 200 | Chance/CPW及smoke归档，native 200局保留 |
| `runs/entity_attention_iteration_trend_seed1_20261008` | 400 | 400 | 18.11 | 0 | 400 | 保留，冻结池原始数据 |
| `runs/az_selfplay_seed1_round000` | 300 | 300 | 14.70 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/inference_throughput` | 263 | 263 | 11.09 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/search_accel` | 124 | 124 | 5.56 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_avg4_behavior_20261009` | 22 | 2,246 | 9.69 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/speed_profile_2026-10-02` | 18 | 18 | 0.81 | 0 | 0 | 整体归档，旧路径兼容 |
| `runs/entity_attention_iteration_suite_validation_20261008` | 16 | 16 | 0.65 | 0 | 0 | 整体归档，旧路径兼容 |
| `runs/entity_attention_iteration_suite_validation_v2_20261008` | 16 | 16 | 0.65 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_memory_gpu_smoke_20261006` | 16 | 17 | 0.64 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/az_loop_smoke` | 8 | 8 | 0.33 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_execution_speed_20261005` | 8 | 8 | 0.32 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_self_iteration_smoke` | 8 | 10 | 0.35 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_hybrid_cpu_smoke_20261005` | 4 | 5 | 0.17 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/mcts_step2_smoke` | 4 | 4 | 0.20 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/az_native_cuda_pool_smoke` | 2 | 2 | 0.08 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_p4_pool_iteration_20261010` | 2 | 2 | 0.08 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_r9_online_validation_20261011` | 2 | 2 | 0.09 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_v1_smoke` | 2 | 4 | 0.11 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/gpu_pool_dev_smoke` | 2 | 2 | 0.09 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/gpu_pool_compiled_dev` | 1 | 1 | 0.04 | 0 | 0 | 整体归档，旧路径兼容 |
| `artifacts/entity_attention_chance_pw_noise_20261007` | 0 | 1 | 2.21 | 0 | 0 | 保留，原始诊断证据 |
| `artifacts/entity_attention_chance_search_20261007` | 0 | 1 | 2.21 | 0 | 0 | 保留，原始诊断证据 |
| `artifacts/entity_attention_formal_lr_step_20261007` | 0 | 28 | 6.29 | 0 | 0 | 保留，原始诊断证据 |
| `artifacts/entity_attention_r30_search_alignment_20261011` | 0 | 84 | 4.45 | 0 | 0 | 保留，原始诊断证据 |
| `artifacts/entity_attention_target_noise_20261007` | 0 | 1 | 2.20 | 0 | 0 | 保留，原始诊断证据 |
| `runs/entity_attention_offline` | 0 | 1 | 3.86 | 0 | 0 | 保留 |
| `runtime/action_benchmark_cli_check` | 0 | 3 | 0.08 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_avg4_delta_kl_20261009` | 0 | 522 | 6.29 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_chance_pw_noise_20261007` | 0 | 32 | 2.38 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_chance_search_20261007` | 0 | 32 | 2.38 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_clip_sweep_20261009` | 0 | 102 | 22.91 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_policy_history_20261009` | 0 | 15 | 45.69 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_target_averaging_20261007` | 0 | 28 | 3.67 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/entity_attention_target_noise_20261007` | 0 | 32 | 2.38 | 0 | 0 | 整体归档，旧路径兼容 |
| `runtime/offline_cache_speed_20261004` | 0 | 2 | 0.01 | 0 | 0 | 整体归档，旧路径兼容 |

合计：逐局 NPZ **93,586**，全部 NPZ **579,650**；NPZ 已分配空间 **6.133 GiB**。这不含 checkpoint、日志、JSON sidecar 或目录本身。

本次归档入口包含 **21,563** 份逐局 NPZ。旧路径仍可读取；对移动后的真实位置，依 [manifest.json](manifest.json) 按来源前缀映射至 `archive/20261012/`。
