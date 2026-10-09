# 10×1000 AZ 训练准备产物

状态：prepared_not_started，正式采集/训练/棋力面板/4vs8均未执行。49项有限CPU回归通过，native CUDA独立smoke未执行；不据此声称长训、4GiB扩容或棋力验收通过。

`preparation_manifest.json`记录真实源码/dirty身份、配置SHA、机器信息和两个起点checkpoint身份；`seeds.json`包含全部10轮派生和面板输入，`smoke_seeds.json`为独立小验证。计划和统计分母见[训练指南](../../docs/archive/next_az_training_zh-CN.md)及[字段清单](../../docs/archive/next_az_record_fields_zh-CN.md)。

正式运行后这里保存每轮dataset/current_before_after/fixed_curve/usage JSON、summary、训练和固定验证CSV；原NPZ、trace、checkpoint、二进制引用/计数、TensorBoard及恢复sidecar放忽略的runs，资源/源码实际副本也在runs。运行前准备清单不包含未生成的结果。

`decision_template.json`只是人工查看记录模板，未做第3/5/10轮判断，也未默认选模/扩样。运行后每次查看追加到decisions.jsonl，保留时间、模型/面板、依据和是否延后评测。
