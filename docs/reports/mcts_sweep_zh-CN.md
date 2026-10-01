# MCTS 50-pair 对照的实际状态

2026-10-01 核对原始汇总，不启动或补跑。配置使用同一 PPO latest checkpoint `009d17d2…`，CPU 8 workers，game seeds 9800000–9800049、policy base=9810000，温度 0、无根噪声。原四组脚本与命令见 [评测指南](../guides/evaluation_zh-CN.md)。

| 对照 | 状态 | 完整 pairs（记录值） | 有效局 | 完整面板得分率 |
| --- | --- | ---: | ---: | ---: |
| `01_s256_n2_vs_s1024_n1` | valid | 50 | 100 | 20.0% |
| `02_s1024_n1_vs_s1024_n2` | valid | 50 | 100 | 56.0% |
| `03_s1024_n2_vs_s2048_n1` | incomplete | 0 | 4 | — |
| `04_s1024_n1_vs_ab_d6_top5` | not_found | — | — | — |
| `05_s1024_n1_vs_greedy` | valid | 50 | 100 | 92.0% |

第 3 组的汇总只保存 4 个有效局，completed_pairs=0、passed=false、没有 bootstrap CI；不能把它的 62.5% 当成 50-pair 结果，也不能由无 exceptions 推断完成。第 4 组没有找到汇总，不能证明从未执行过。第 5 组是额外的 MCTS vs greedy，不在原脚本四组序列里。

完整面板的 bootstrap 与具体配置直接见 [状态索引](../../artifacts/mcts_50pair_sweep/status_index.json) 和引用的原 JSON。结果属于单 checkpoint、同一 seed 面板、小样本配置比较；没有完成整个矩阵，不能宣称全局最优预算。
