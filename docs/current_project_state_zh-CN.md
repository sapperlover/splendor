# 当前项目状态

核验日期：2026-10-09。本文是导航快照，详细事实以当前实现、冻结配置和可复核原始产物为准。完成模型训练、完成评测、确认收益是三个不同状态；历史日期和测试数量不代表当前工作树重新验收。

## 当前研究结果

[M4 与固定数据诊断完整结论](reports/entity_attention_m4_diagnostics_conclusions_zh-CN.md) 是近期研究的统一入口。学生可以降低固定训练目标的误差，但稳定泛化与持续棋力收益未确认；最新数据优先、删除历史、较小学习率都还没有成为经过独立确认的最佳日程。

| 实验／分支 | 已完成与限制 | 依据 |
| --- | --- | --- |
| 行为复用 M4 | 12／16轮，2400新局、1200更新、17700步，当前停止；r2／4／8对冻结16500起点为48.5／56.25／53.0%，区间均跨50%；r12仅61完整pair | [运行记录](../artifacts/entity_attention_avg4_behavior_20261009/README.md)、[命令](guides/entity_attention_avg4_iteration_zh-CN.md) |
| 固定历史 A／B／C | 三个起点、4500更新、36诊断节点；A@100精确复现正式模型／AdamW；C改善训练拟合，但最新代和近期留出没有稳定收益；B尚未确认优于A | [实验报告](../artifacts/entity_attention_policy_history_20261009/README.md) |
| 独立重搜索 | 已修正 bootstrap 重复权重和遗漏补充集；未抽中 Train 的768根 fresh ΔKL +0.00415，95%区间[−0.00154,+0.00980]；未确认改善或稳定退化 | [修订与原始证据](../artifacts/entity_attention_avg4_delta_kl_20261009/README.md) |
| 步数／lr／clip | 旧16500单起点固定账本提供较小更新候选；50步初筛点估计较高，最优步数／学习率未确定；高裁剪频率不能证明 AdamW 的学习率作用很弱 | [剂量](../artifacts/entity_attention_step_dose_20261007/README.md)、[lr](../artifacts/entity_attention_formal_lr_step_20261007/README.md)、[clip](../artifacts/entity_attention_clip_sweep_20261009/README.md) |
| critic 适配 | 额外0／50／200／500步完整留出 MSE 0.808967／0.789053／0.787500／0.786814；已完成三个100pair DEV，51.75／46.0／45.75%，均未确认超50%；HEAD及阶段替换确认未完成 | [报告与对战记录](../artifacts/entity_attention_critic_adaptation_strength_20261008/README.md) |
| 冻结 encoder 拟合 | critic-only留出下降；两版16500／16600 encoder新head重拟合500步留出约0.79247／0.79242，差值区间跨0；未测出绝对拟合上限 | [只练critic](../artifacts/entity_attention_critic_only_replay_20261008/README.md)、[重新拟合](../artifacts/entity_attention_encoder_critic_refit_20261008/README.md) |
| 分阶段搜索 | 同16500权重，各500pair对greedy：只搜后期67.0%、只搜前中期69.35%、全程81.8%；阶段两组差值区间跨0，全程明确高于两组 | [报告](../artifacts/entity_attention_stage_search_20261008/README.md) |
| 固定256根诊断 | 独立真实对局、每局单根，前中期／后期各128；动作变化、JSD和排序只衡量行为变化，没有正确动作标签，不能换算棋力 | [指南](guides/decision_probe_zh-CN.md) |
| B→C→D趋势套件 | B仅完成2轮／16600步及100pair首测51.25% [44.5%,58.0%]；C／D未运行，整体未完成 | [流程](guides/entity_attention_iteration_suite_zh-CN.md)、[原件](../artifacts/entity_attention_iteration_trend_seed1_20261008/README.md) |

下一步候选见统一报告末节：先复用已有 fresh 标签评价固定 A／B／C 节点，再做当前 M4 同账本的 lr 对照。这里只记录建议，没有启动下一阶段实验。

## 实现与较早训练

| 能力／历史阶段 | 证据范围 | 入口 |
| --- | --- | --- |
| 双人规则、Observation、数据与学习接口 | Python checked 规则参照；native 为另一个实现，依对应版本回归验证 | [标准规则](splendor_standard_rules_zh-CN.md)、[契约](specs/engine_contract_zh-CN.md)、[学习契约](specs/learning_contract_zh-CN.md) |
| PPO seed1、AZ seed1 | PPO 5M本轮decisions；AZ5×500及θ1–θ5扩展面板完成，DEV best不等于独立FINAL best | [PPO](reports/ppo_5m_zh-CN.md)、[AZ](reports/alphazero_seed1_zh-CN.md) |
| 旧θ5的10×1000源数据 | 轮0–9完成，10000局原split用于离线学习；770870训练／80699留出位置；准备说明已归档，不在本次重验棋力 | [数据统计](../artifacts/entity_attention_v1_20261004/data_summary.json)、[历史准备](archive/next_az_training_zh-CN.md) |
| 实体注意力离线 | 532738参数、12045步及六个PPO面板完成；final搜索59.5%、greedy61.75%，重复DEV | [指南](guides/entity_attention_offline_zh-CN.md) |
| 新模型10×1000、混合训练 | 10×1000曾运行，已有轮0–2完成记录；5000步起点混合run完成9轮／9500步；不同分支不能互称完成 | [自迭代](guides/entity_attention_self_iteration_zh-CN.md)、[混合](guides/entity_attention_hybrid_zh-CN.md) |
| 第九轮N2与N1 fork | N2至少33新轮／12800步记录，曾OOM；N1完成50新轮／14500步后停止；原9／50轮共同PPO为63.75／61.75%，差值区间跨0 | [N2](guides/entity_attention_hybrid_fork_zh-CN.md)、[N1](guides/entity_attention_hybrid_s1024_n1_zh-CN.md)、[补测](../artifacts/entity_attention_strong_anchor_20261006/README.md) |
| 分头窗口、value detach | 20新轮／16500步完成；policy近4轮、value3000局近期50%；长期净收益未确认 | [指南](guides/entity_attention_split_training_zh-CN.md) |
| 搜索教师比较 | 目标稳定性不等于棋力；同S1024随机节点31.25%、CPW37%均弱于旧native；正式教师未切换 | [系列报告](reports/entity_attention_search_teachers_zh-CN.md) |
| 性能与进程管理 | 多worker、CUDA Graph、缓存与退出修复有有限验证；各报告限定形状／机器／版本，未确定全局最快配置 | [性能验收](reports/end_to_end_speed_optimization_zh-CN.md)、[新模型执行](reports/entity_attention_execution_speed_zh-CN.md)、[OOM修复](../artifacts/entity_attention_worker_cleanup_20261005/README.md) |

所有文档与配置见 [导航](index.md)。状态分类及原件身份见 [实验索引](../artifacts/experiment_index.json)；本次路径和结论维护见 [整理记录](../artifacts/repository_cleanup_20261009/README.md)。
