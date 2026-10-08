# 原第九轮 S1024/N1 小步更新配置验证

2026-10-06。本文保留B[64,256,500]/batch上限500的准备时验证。随后用户已启动正式分支，采集Graph另改为B[16,32,64,128,200]/上限200，见[新验证记录](../entity_attention_graph200_20261006/README.md)；最新启动/恢复命令见[专题指南](../../docs/guides/entity_attention_hybrid_s1024_n1_zh-CN.md)。原准备方案从9500步恢复，每轮200局/100步，在线窗口缩为**3000局**，保留原S1024/N1搜索、learner、采集执行、种子域、固定验证集与theta_5000。用户明确选择保持离线退出日程：第31个新轮次起纯在线，每5个新轮次评测一次。`--forever`继续训练而不设总轮数上限。

准备时没有修改训练源码，没有启动正式分支长训，也没有采集新局、执行正式更新或新增棋力赛。该冻结方案与父run的差异为run_id、每轮采集/更新量、replay_max_games、换算后的mix_schedule与评测频率；相同新局数下的总更新量保持一致。200局或3000局窗口均不是已验证的最优值，现有N2退化记录不能单独证明新局数不足。

## 3000局窗口准备记录

[verification_replay3000.json](verification_replay3000.json)与[冻结配置副本](config_replay3000.json)记录修订后的真实只初始化验证、教师来源和年龄推算。[调用时源码身份](initialization_replay3000_identity.json)与完整忽略产物 `runtime/entity_attention_hybrid_s1024_n1_validation_20261006/initialize_only_replay3000/` 保留对应源码和命令。

真实初始化保持9500步checkpoint字节一致，69个AdamW参数状态仍为9500步；继承原第7～9轮3000局，教师分别为8000/8500/9000步。教师身份同时核对对应采集manifest、轮次快照和原checkpoint哈希。原固定验证与共同theta_5000不变。

固定200局/100步的FIFO计划，在新轮15完全替换继承在线数据，稳态覆盖最近15个采集教师；最旧教师在训练前落后当前采集教师1400步，训练后落后当前模型1500步。继承过渡期间最大训练步数差为1900步。上述年龄是计划的算术推算，不是新训练或棋力结果。

3000局上限只作用于在线池。旧Deep-Sets来源的10000局离线数据来自10个采集教师，其身份另列在验证JSON；按用户选择继续以64/256、32/256配额参与到新轮30，第31轮归零。没有新增教师加权抽样或年龄筛选代码。原9项CPU回归在同一训练源码上已通过，此次仅变更配置并新增真实3000局初始化验证，未重复测试。

## 此前6000局准备记录

[verification.json](verification.json)保留此前6000局配置的哈希、命令、父checkpoint及有限验证结果；当时配置完整保存为[config_replay6000.json](config_replay6000.json)，其哈希与旧记录一致。[initialization_identity.json](initialization_identity.json)保留调用时HEAD、dirty状态、源码哈希、命令及Torch版本。完整源码快照与继承NPZ引用清单位于忽略的 `runtime/entity_attention_hybrid_s1024_n1_validation_20261006/initialize_only/`；真实初始化使用正式CUDA配置但指定 `--online-rounds 0`，没有构造CUDA learner或推理owner。旧记录中的配置路径后来按用户要求修订，以此冻结副本复核旧哈希。

验证结果：

- 初始checkpoint与原第九轮final字节哈希完全一致，9500步；69个AdamW参数状态均为9500步，完整checkpoint保留RNG。
- 继承原第4～9轮6000局，5400 train / 600 validation，共461221个训练局面；没有继承N2分支数据。
- 原离线dataset/固定验证、原在线固定验证、theta_5000哈希一致，初始化引用的父输入未改变。
- 9项既有混合训练回归全部通过，见[原始JUnit结果](regression_tests.xml)，覆盖fork状态接续、实际小规模CPU训练、数据替换、恢复及留出集保留；不是正式S1024/N1 GPU训练验收。

验证JSON另保存已有N2新轮10/20/30的面板和在线池组成摘要及其原始路径/哈希，作为“不能单独归因于新局数”的证据；未修改这些原始训练产物。

从仓库根目录只读复核准备时冻结配置和父数据：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.hybrid_train \
  --config artifacts/entity_attention_hybrid_s1024_n1_20261006/config_replay3000.json \
  --fork-from runs/entity_attention_hybrid_static_graph_seed1 --fork-round 9 \
  --check-config
```

该命令不写新训练目录，不采集、不更新；需要保留原checkpoint、离线源数据与父run的NPZ路径。测试和初始化验证仅支持配置可接续，不证明棋力提升。
