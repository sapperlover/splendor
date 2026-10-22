# 第24轮离线候选对第43轮的直接搜索评测

2026-10-19核对：本面板已完成200/200pair、400局，得分51.250% [46.125%,56.375%]，passed=True，未确认超过R43，见[结果](results.md)。后续R43离线链路亦已完成。

2026-10-12。用户指定比较原第24轮＋24388步完整checkpoint与原绝对第43轮完整checkpoint。双方S1024/N1、无根噪声，200pair／400局，互换先后手、和棋半分，10000次整pair bootstrap；使用35800000起的新开局，不复用此前R24离线DEV或确认开局。

候选：`runs/entity_attention_r24_history_pool_20261012/checkpoints/step-000024388.pt`，SHA256 `deccdc296869a1ceff10b0a6394b9349938e7a26073aa7998130d563696eeeb2`。

对手：`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt`，SHA256 `b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。

候选已在此前对原R24的独立500pair中得到54.00% [50.80%,57.10%]；本面板回答是否超过R43，不能由对R24的结果代替。本次只有200pair，跨50%时视为未确认，不能视为等强。

固定配置见[config.json](config.json)，结果见[results.md](results.md)。本面板是[R43历史池离线链路](../entity_attention_r43_history_pool_20261012/README.md)的第一阶段，同一链路命令启动／恢复；完成后自动进行R43离线训练及其评测。本面板结果不用于R43离线训练的节点选择，也不因高低分阻止后续训练。
