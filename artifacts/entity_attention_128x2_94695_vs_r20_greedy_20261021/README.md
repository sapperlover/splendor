# Attention128 94695步对第20轮：1000pair greedy

2026-10-21按用户要求完成1000pair／2000局，同一开局seed交换先后手，双方确定性greedy。候选在对战前固定为94695步：截至此前讨论的108223步，前八个已完成百分比节点内完整留出KL+MSE最低（0.4299478942＋0.7490424081）。选择整个checkpoint，没有拼接两头，也没有根据对战结果重新选点；训练原定270557步尚未完成。

- **新attention得分54.400%，95%整pair bootstrap区间[52.299%,56.525%]**；1078胜、20和、902负。
- 先手得分55.100%，后手53.700%；8个CPU worker，耗时95.98秒。
- 当前12贵族池，seed43200000–43200999；bootstrap10000次、seed43220000。
- 异常、非法行动、工程截断、不变量错误、强制跳步均为0；20盘和棋中含1盘确定性策略循环裁决。

本面板支持该候选的greedy强于raw R20，说明这次离线拟合已转化为policy棋力收益；没有测搜索，不能据此判断value或搜索棋力。此前[MLP94695对同一R20](../entity_set_mlp_94695_vs_r20_greedy_20261021/README.md)在相同开局seed及规则下得37.075%；这是共同对手比较，不能当作attention与MLP直接对战的胜率，也不能把差距全部归因于网络结构。

本次只运行独立CPU对战，未修改训练配置、模型、账本或进程。核验结束时原训练的执行记录／日志标记`interrupted`，日志在约135600步以`KeyboardInterrupt`退出；最近提交恢复点为135279步（完整留出KL 0.427218、MSE 0.760894）。当时没有发现仍在运行的训练进程；本次未自动恢复，退出信号来源没有据此判定。

完整配置见[config.json](config.json)，原始1000pair及逐局结果见[result.json](result.json)，核验汇总见[summary.json](summary.json)。实际命令、冻结源码身份和执行状态见[execution_plan.json](execution_plan.json)；复用原attention训练的源码及12贵族快照，模型SHA保存在配置和结果。已重新核对checkpoint／配置／规则身份、整pair种子与交换先后手，并复算bootstrap。
