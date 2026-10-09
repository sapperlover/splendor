# 历史设计与准备说明

这里保存仍有复核价值、但已经完成其准备阶段或被后续方案替代的文档。正文保留原时点设置和完成范围；命令不是当前运行推荐。对应实验原始记录仍保留原目录身份。

| 文档 | 适用范围与后续 |
| --- | --- |
| [旧θ5的10×1000准备](next_az_training_zh-CN.md) | 2026-10-04准备期；源run后续轮0–9完成，数据成为离线源；当前入口见[AZ指南](../guides/alphazero_zh-CN.md) |
| [旧10×1000记录字段](next_az_record_fields_zh-CN.md) | 对应上一项冻结配置的分母与记录协议；不能臆补历史缺失字段 |
| [最新单轮份额阶梯设计](entity_attention_fresh_share_zh-CN.md) | 12.5%→18.75%→25%的未实施候选；由[分头窗口](../guides/entity_attention_split_training_zh-CN.md)替代，后续[M4固定历史对照](../../artifacts/entity_attention_policy_history_20261009/README.md)另有实际实验 |

已被替代的四组套件配置见 [configs/archive](../../configs/archive/README.md)。更早的工作包原文仅本地保全于 `runtime/document_archive/2026-10-01/`；这批历史文档仍随 Git 保存，路径迁移可从[整理清单](../../artifacts/repository_cleanup_20261009/manifest.json)追溯。
