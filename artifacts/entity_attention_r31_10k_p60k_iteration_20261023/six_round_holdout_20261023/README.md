# 在线留出：每六轮固定使用此前六轮

后续发布说明：六轮分段规则与本次结果保持；当前入口已切换[记录／checkpoint发布](../recording_checkpoint_20261023/README.md)，停写双份明细并使用500步周期点＋七个固定点。本目录及`source_six_round_holdout_20261023`保留为本次调度交付的历史身份，不再是启动脚本当前执行目录。

按用户最新更正，当前在线分支采用六轮分段调度，替代初版永久使用R26–R31的设置。R20–R31历史补评因冷checkpoint在未挂载的T7，用户已暂缓；此次没有启动该补评，也未恢复训练。

| 评估模型轮 | 本分支局部轮 | 原始完整validation来源 | 状态 |
| --- | --- | --- | --- |
| R32–R37 | 1–6 | R26–R31：2412局，39845 policy／192218 value | 原42节点结果保留、恢复验证复用 |
| R38–R43 | 7–12 | R32–R37：6092局，101425 policy／485932 value | 新面板已冻结，模型轮尚未训练 |
| R44–R49 | 13–18 | R38–R43 | 第13个本地轮次开始时自动冻结 |
| 后续 | 每六轮一段 | 上一段六轮完整validation | 同一规则自动推进 |

段内每轮0／100／250／500／1000／2000／3000步评估同一份原始标签。换段后面板改变，绝对KL/MSE不可直接跨段连成同一收敛曲线。所有采集、训练、greedy、搜索配置维持原协议；不增加接纳、早停或选模逻辑。

新配置字段为`six_round_holdout`，包含初始`manifest`路径＋SHA、`batch_size=512`、`block_rounds=6`；与历史`fixed_holdout`互斥。后续面板来自原`collection_index.json`全部validation，包含活动池淘汰的数据，冻结来源索引和每局归档SHA。只缓存当前段解码数组在CPU，GPU仅接收当前batch。

`fixed_holdout/schedule.json`冻结运行级规则；后续各段记录到`fixed_holdout/blocks/iterations-007-012/`等独立目录。每段有独立`manifest.json`和`rounds/round-NNN/fixed_holdout/step-XXXXXXXXX.json/.npz`。首段继续复用原位置；未改评测器实现和身份，原42项结果及84个原始文件保持字节一致。

恢复按轮次所属段自动选择面板，补齐已保存节点，并核验数据／模型／指标身份。每轮检查Train与当前留出无交集。分段调度和独立补齐入口共用`.iteration.lock`，不能同时写同一run。

当前启动脚本已指向`runtime/entity_attention_r31_10k_p60k_iteration_20261023/source_six_round_holdout_20261023`。它继承静态初版110个冻结文件，只替换driver并新增调度模块，共111个文件；原模型、learner、native、牌池及固定评测器均保留原字节。旧两版源码快照不改写。恢复仍使用：

```bash
bash scripts/run_r31_10k_p60k_iteration_20261023.sh
```

本次仅用`--check`和`--rounds 6`验证真实入口，不启动第7轮采集／训练。相关CPU回归24项通过，包括第6→7／12→13轮边界、全部原始留出、换段后中断恢复、旧结果复用，以及与无诊断连续训练逐参数／采样RNG一致。测试采集器为mock，不是正式训练或棋力实验。

准备脚本：[prepare_six_round_holdout.py](../prepare_six_round_holdout.py)。当前分段补齐命令见[评测指南](../../../docs/guides/evaluation_zh-CN.md#在线全量固定留出)，使用`splendor.six_round_holdout`，旧`splendor.fixed_holdout`仅对应历史静态配置。

[准备与组成](preparation.json) · [下一段完整清单](next_block_manifest.json) · [冻结源码身份](runtime_source.json) · [验证与命令](verification.json) · [24项回归](tests.xml)
