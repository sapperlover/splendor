# 实体汇总残差MLP预设

2026-10-22[Cross64离线冷产物已外置到T7](../../artifacts/repository_archive_20261022/README.md)：初始及6个百分比checkpoint保留原位，大账本与81个普通恢复模型为归档链接；如需恢复该离线训练，先按清单恢复会读写的账本为原位真实文件。原80k训练池、配置及留出指标未迁移。

2026-10-22新增[角色初始化标定及同8万局离线训练](../../artifacts/entity_set_mlp_cross64_role005_offline_80k_20261022/README.md)：三个角色embedding各std0.05，合成RMS实测0.0864，内容编码0.0909；参数量仍906434，乘积及非线性旁路保留。新配置`offline_cross64_role005_80k_lr1e4_20261022.json`与旧离线训练除模型及manifest路径外相同，95项CPU及GPU短验证通过。最初只交付命令，用户后来启动，现中断、已保存81168步，原270557步计划未完成；六节点完整留出KL＋MSE最低54112步，独立对R20 greedy1000pair得43.225% [41.075%,45.325%]，未测搜索。原命令`bash scripts/run_mlp_cross64_role005_offline_20261022.sh`会恢复训练及后续流程。

`model_256x4.json`是模型配置片段，不是训练CLI配置：牌编码64、贵族32、主干256宽×4残差块、actor/critic128、购买关系开启；当前906,434参数。状态主干保留分组pooling及残差MLP；动作读取82维原属性及派生目标关系8维，并通过一层64维／2头cross-attention读取25个未汇总实体。

当前预设显式开启`action_bottleneck=32`、`multiplicative_interaction=true`、`action_cross_hidden=64`和`action_cross_heads=2`。原90→128直连、90→32→128非线性旁路及128维状态×动作乘积保留；cross-attention的64维输出直接投影到128维，作为残差加在原融合后，再经ReLU→128→1。Query读取动作和共享状态，K/V每局面共享；type/owner/tier及mask保留。新增Query与memory各有LayerNorm，没有实体self-attention或额外FFN。

新增分支复用原牌／贵族编码；value-only跳过全部新增投影和attention，value结构未变。仅回合／双辅助分别939,459／972,484参数。

用`build_actor_critic`构建，或嵌入新的`offline_scratch`配置的`model_config`；默认主模型无辅助，训练配置可通过`value_auxiliary`选择none/turns-only/dual。数据schema兼容旧NPZ，不能把旧attention权重直接按MLP结构恢复。

`action_cross_hidden=0`关闭cross-attention；头数须为正整数，非零hidden须能被头数整除。`action_bottleneck=0`与`multiplicative_interaction=false`可分别消融旧增强，本次未证明它们有益或有害，也未改动其初始化。

旧配置省略cross字段时关闭attention（heads默认2），855,682参数的动作头1＋2权重继续严格加载；所有新字段省略时构建初版848,418参数结构。启用当前预设不能直接严格恢复旧结构的权重或AdamW。已冻结的`offline_80k_lr1e4_20261021.json`仍是旧855,682参数配置，不随此模型片段升级。

当前完整预设的[8万局离线训练流程](../../artifacts/entity_set_mlp_256x4_offline_80k_20261021/README.md)已按用户要求停止并保留97000步checkpoint，未完成原270557步预算，不启动对战。`offline_80k_lr1e4_20261021.json`仍保留原配置：80000局、lr1e-4、16GiB缓存、两头各256联合更新、每5%固定留出。入口`bash scripts/run_entity_set_mlp_offline_20261021.sh`会恢复训练及后续对战，当前不执行；不能交给旧共享minibatch入口以免改变抽样口径。

结构、验证及两项旧增强的风险见[cross-attention升级](../../artifacts/entity_set_mlp_256x4_20261021/action_cross_64x2_20261022/README.md)。[动作头1＋2](../../artifacts/entity_set_mlp_256x4_20261021/actor_upgrade_12/README.md)和[初版交付](../../artifacts/entity_set_mlp_256x4_20261021/README.md)保留各自历史结果；std0.05 cross后续训练及独立greedy评测见本页首段，旧std1验证不包含该结果。
