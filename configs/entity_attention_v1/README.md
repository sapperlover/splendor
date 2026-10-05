# Entity attention v1 配置

操作和结果说明见 [专题指南](../../docs/guides/entity_attention_offline_zh-CN.md)。所有命令从仓库根目录执行。

| 配置 | 用途与前置条件 |
| --- | --- |
| `offline_attention.json` | 新结构随机初始化，全部十轮原 split，batch256、12,045步、4 GiB LRU；写 `runs/entity_attention_offline` |
| `offline_deep_sets.json` | 旧结构也随机初始化，与新结构同数据、seed、优化及抽样预算；写 `runs/deep_sets_offline` |
| `offline_smoke.json` | 首轮前6局、batch16、8步、验证16、64 MiB LRU；由有限验证脚本使用 |
| `selfplay_smoke.json` | 新 smoke 导出模型、native_cuda_pool、2局、S16/N1、compiled/reused、projection=false |
| `evaluation_smoke.json` | 新 smoke 对旧θ5，native_batched、1 pair、S16/N1；分别配置新false/旧true projection |
| `mcts_vs_theta5.json` | 正式离线 final 对旧θ5，双方S1024/N1、50 pairs；需要先生成新 final |
| `greedy_vs_theta5.json` | 正式新旧裸网 greedy、50 pairs；需要先生成新 final |
| `greedy_vs_ppo_5_5m_100pairs.json` | 新离线final vs PPO lifetime5,512,112，双方greedy、100 pairs/200局 |
| `mcts_s1024n1_vs_ppo_5_5m_100pairs.json` | 同一双方checkpoint，双方MCTS S1024/N1、100 pairs/200局，compiled/reused、新false/旧true projection |
| `offline_step{1000,5000}_greedy_vs_ppo_5_5m_100pairs.json` | 离线训练1000/5000步对PPO lifetime5,512,112，双方greedy，各100 pairs |
| `offline_step{1000,5000}_mcts_s1024n1_vs_ppo_5_5m_100pairs.json` | 相同离线节点与PPO双方S1024/N1，各100 pairs；与final面板复用相同game/policy/search/bootstrap seed |
| `online_3x200.json` | 此前短阶段建议保留；当前正式自迭代以10×1000配置为准 |
| `online_10x1000.json` | 本次正式自迭代：离线final起点，10轮×1000局、同时调度1000局、S1024/N1；每轮对固定起点搜索/greedy各50 pairs；[命令与验证](../../docs/guides/entity_attention_self_iteration_zh-CN.md) |
| `online_smoke.json` | 本次两轮GPU短测：每轮4局、2步、S16/N1，搜索/greedy各1 pair；独立runtime输出 |

配置交付时没有启动正式任务；随后用户已启动新结构离线拟合。棋力与在线结果尚未验收。两个离线配置保持原冻结4 GiB值，实际运行建议加 `--replay-cache-mib 8192`；resume同样可加此执行参数，无须修改配置。`history_batch_size` 在离线入口无作用；离线只有一个覆盖全选定 train positions 的池。新结构不支持 actor projection，混合评测通过每方 `inference` 明确区分。
