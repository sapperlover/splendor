# MLP 94695步对第20轮：1000pair greedy

2026-10-21按用户要求完成1000pair／2000局，同一开局seed交换先后手，双方确定性greedy。候选为已停止MLP运行的94695步：七个已完成百分比节点内完整留出KL+MSE最低；不是97000步恢复checkpoint，也未拼接value。对手为第20轮raw attention。

- **MLP得分37.075%，95%整pair bootstrap区间[35.000%,39.175%]**；736胜、11和、1253负。
- 先手得分37.550%，后手36.600%；8个CPU worker，耗时89.78秒。
- 当前12贵族池，seed43200000–43200999，每个交换先后手；bootstrap10000次、seed43220000。
- 异常、非法行动、工程截断、不变量错误均为0；11盘和棋中含1盘确定性策略循环裁决。

该候选的greedy明显弱于R20，与其同留出KL较差一致；单一模型／训练经历对照不能把差距全部归因于MLP结构。此次没有搜索对战，也未恢复MLP训练。

配置见[config.json](config.json)，全部pair及逐局结果见[result.json](result.json)，核验汇总见[summary.json](summary.json)。冻结代码引用、实际命令与执行状态见[execution_plan.json](execution_plan.json)；复用原MLP运行的源码／12贵族快照，模型SHA写入配置和结果。重型逐局sidecar由评测入口保存，不改忽略规则。
