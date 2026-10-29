# 新增购买结算特征方案：已撤回

2026-10-23归档位置更新：本专题关联`runtime/entity_attention_first_round_20261022`中的19个抽样账本／逐步日志已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。

2026-10-23用户要求删除这批新增特征，包括购买后贵族推进、结算摘要和终局标记；终局设计留待用户后续处理。已删除购买模块、actor query分支、模型配置字段，以及专用`offline_first_round`入口、候选配置、启动脚本和依赖该方案的验证入口。原关系attention128恢复**583682参数**，原有`market_relations`和value读取不变；短期进展辅助保持关闭。

本次没有新增终局特征，没有启动正式训练、采集或棋力赛。两份正式候选目录均只有准备产物、没有训练checkpoint，继续保留原位：

- `runs/entity_attention_first_round_r21_r31_20261022/`：原组合候选准备。
- `runs/entity_attention_purchase_only_r21_r31_20261023/`：关闭辅助后的纯购买候选准备。

旧CPU／CUDA短验证模型和结果保留为历史工程证据。它们包含已删除字段和实验参数，应使用对应历史源码快照读取；当前代码不忽略字段或自动转换旧实验权重。

## 当前撤回与验证

- [撤回记录](withdrawal_20261023.json)：删除范围、删除前文件SHA、快照、当前验证及历史原件核对。
- [删除前源码／配置／脚本快照](withdrawn_purchase_sources_20261023.tar.gz)：按原仓库路径保存；同时包含旧README与冗余报告。它是历史材料，不是当前启动入口。
- [回归结果](withdrawal_20261023_tests.xml)：当前原模型与保留组件的有限CPU验证。
- [原模型输出基准](withdrawal_20261023_baseline.npz)：删除前以原关系模型、seed43100000在固定开／中／后3个根生成的policy/value；用于核对删除后逐位一致。

当前没有该方案的可运行候选配置或训练命令。既有无新增特征的离线与在线分支使用各自原命令；本次没有修改这些分支的配置、模型、数据或冻结源码。

## 保留的历史证据

| 阶段 | 当时配方 | 记录 |
| --- | --- | --- |
| 2026-10-22组合候选 | 购买结算关系＋短期进展辅助，670338参数 | [准备](preparation.json)、[验证身份](verification.json)、[134项历史回归](tests.xml)、[CPU短验证](cpu_validation.json)、[CUDA短验证](cuda_validation.json) |
| 2026-10-23先关闭辅助 | 仅购买结算关系，587138参数 | [准备及核验](purchase_only_20261023.json)、[108项历史回归](purchase_only_20261023_tests.xml)、[CPU短验证](cpu_purchase_only_20261023_validation.json) |
| 2026-10-23冗余审查 | 输入复制与派生恒等式分析，164个主行动／737个购买候选 | [报告](purchase_redundancy_20261023.md)、[来源及计数](purchase_redundancy_20261023.json) |

上述原始JSON、XML、NPZ及模型保持原身份。有效工程记录不等于已证明棋力收益；本方案未运行正式训练或对战，撤回是用户的设计决定。
