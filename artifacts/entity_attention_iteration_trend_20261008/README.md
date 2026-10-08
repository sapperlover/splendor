# B → C → D 趋势实验交付（历史双种子准备）

后续按用户要求只保留seed1，当前启动配置已调整为[单种子方案](../entity_attention_iteration_trend_seed1_20261008/README.md)。本目录的manifest及原始验证保留为双种子准备阶段记录，未启动正式长实验；下文数量及配置描述适用于旧版本。

2026-10-08。原方案的配置详见本目录manifest；[当前单种子配置](../../configs/entity_attention_v1/iteration_trend_20261008.json)及[流程/启动命令](../../docs/guides/entity_attention_iteration_suite_zh-CN.md)已按最新要求精简。**正式长实验未启动**；此处仅完成环境检查并保存配置。

完整完成 B（200局/50步、lr1e-4、P4）的两条16轮轨迹及评测，再完成 C（200局/100步、lr2.5e-5、P4），最后 D（200局/50步、lr1e-4、P1）。共同冻结16500步起点，value口径相同。第2/4/8/12/16轮对冻结起点统一100pair，共19200新局、6400更新、6000评测局；粗估16–24小时。

本套件取消A、PPO、0轮自测、组间直接对打、显著优胜判定、逐轮引用账本及checkpoint重复哈希审计、全仓源码恢复审计、磁盘容量门槛和磁盘轮询。保留现成训练诊断、共同评测种子的趋势配对差、配置冻结、轻量完成标记、训练器本身的断点机制、4GiB缓存与16GiB cgroup内存上限。

[必要验证](verification.json)：3项针对性测试通过（32.61秒），覆盖BCD顺序、固定训练轨迹内的pair重采样、真实CPU短训练/评测、中断后的完成标记补写和趋势分析补写、已完成套件再次启动不新增工作。原始测试结果见 [focused_tests.xml](focused_tests.xml)。只执行一次新的CUDA `--prepare-only`，实际检查CUDA及内存上限，未进行正式GPU训练或评测。历史版本的CUDA短流程记录保留在原目录，不替代本版本的全程稳定性测量。

`manifest.json`保存当前方案、六条训练配置、冻结起点的评测模板和30个评测任务；`preflight.json`记录当前环境。`code_version.json`记录Git HEAD和入口脚本SHA，实际训练源码身份由训练器现有每次调用记录保存。原始棋力汇总和`analysis.json`在正式运行后生成，区间只描述固定已训练模型的对局不确定性，不证明参数最优。

模型/NPZ在ignored `runs/entity_attention_iteration_trend_20261008/`，日志、完成标记和调度状态在ignored `runtime/entity_attention_iteration_trend_20261008/`，评测sidecar仍按现行忽略规则保存。旧四组配置、manifest和验证原件不删除，只标明已经被替代。
