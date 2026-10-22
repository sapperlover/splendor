# Cross64 MLP：角色初始化标定＋同8万局离线配置

后续状态（2026-10-22）：交付后已由用户自行启动；目前执行记录为interrupted，最近保存81168步，六个百分比节点已完成，原270557步计划尚未完成。完整留出KL＋MSE最低在54112步（0.472901／0.772611）；value MSE最低在40584步（0.769011），此后67640／81168步升至0.777131／0.797616。按新授权单独完成[54112步对R20 greedy1000pair](../entity_set_mlp_cross64_54112_vs_r20_greedy_20261022/README.md)：858胜／13和／1129负，得分43.225%，95%配对区间[41.075%,45.325%]，低于R20；没有恢复训练或替代原最终选点，未测搜索。下文“未启动”及准备验证说明均保留为首次交付时的历史状态，当前进度／指标以summary、results和原运行文件为准。

2026-10-22按用户要求完成初始化修改与新实验准备，随后用户明确要求“无需启动，给我命令即可”，因此**正式离线训练未启动**。本次GPU执行仅为短正确性验证，不是正式训练。

当前模型906434参数：实体牌64、贵族32、922→256状态入口、4个256宽残差块、actor／critic128；动作90维直连、32维非线性旁路、乘积γ初值1、64维／2头动作cross-attention全部保留。只调整新建cross模块的type／owner／tier三个embedding：各自从标准正态缩放到std0.05，合成预期RMS为sqrt(3)×0.05≈0.0866。原卡牌编码、乘积、query、key、value及输出投影的初始化均未调整，缩放不额外消耗RNG。加载checkpoint时存储权重覆盖初始化，不会再次缩放旧模型的角色权重。

用同初始化seed43100000、前中后期三个固定真实根验证：内容编码RMS **0.090920**，角色编码RMS **0.086420**，比例 **0.9505**；原角色编码约为内容的19倍。这里仅标定初始化幅度，不证明收敛或棋力改善。

| 项目 | 沿用旧MLP的冻结配置 |
| --- | --- |
| 数据 | 原在线第1–20轮共80000局，绝对数据轮43–62；不加入新N4分支数据、不重标注 |
| Train／留出 | 71989／8011整局；5771871／641723局面；1206369／134304条有效policy标签 |
| 抽样 | 每步独立抽256条高预算正权重policy＋256条全预算value，一次联合反传／全局clip1／AdamW更新 |
| 优化 | 从零初始化及全新AdamW；lr=1e-4恒定、wd=1e-4、两头loss权重各1、FP32；无辅助头、无PSW |
| 缓存 | **16384MiB＝16GiB**的CompactGameCache，不是8GiB |
| 种子 | 初始化43100000、抽样43110000、固定留出43120000 |
| 预算 | 12次value位置曝光，共270557步；对应约57.41次policy标签曝光 |
| 留出／保存 | 每5%在同一固定4096子集及完整留出评价KL／CE／目标熵／MSE等，保存20个百分比节点；每1000步另存恢复checkpoint |
| 选点 | 完整留出policy KL＋胜负MSE最小，平局选较早，选择整个checkpoint |
| 原流程对战 | 选中模型对raw R20：1000pair greedy及500pair S1024N1（总1024），冻结12贵族规则；无根噪声，交换先后手 |

逐字段核对：除新模型配置及独立manifest路径外，训练配置与旧MLP完全相同；完整dataset及固定留出引用也完全相同，见[configuration_validation.json](configuration_validation.json)。数据仍是原10贵族池的历史数值标签，评测采用12贵族池。不能把旧855682参数模型的结果当成本次906434参数模型的结果。

启动／中断后恢复原完整流程（训练→留出选点→对R20的两组对战）：

```bash
bash scripts/run_mlp_cross64_role005_offline_20261022.sh
```

只训练＋留出选点，不自动对战：

```bash
bash scripts/run_mlp_cross64_role005_offline_20261022.sh train
```

只查看已有产物：

```bash
bash scripts/run_mlp_cross64_role005_offline_20261022.sh status
```

配置见[offline_cross64_role005_80k_lr1e4_20261022.json](../../configs/entity_set_mlp_v1/offline_cross64_role005_80k_lr1e4_20261022.json)，预算／来源身份／种子／比较范围见[plan.json](plan.json)。源码、native扩展、配置及12贵族数据快照在`runtime/entity_set_mlp_cross64_role005_offline_80k_20261022/source/`，身份见[source.json](source.json)；执行入口使用冻结源码。启动时复核准备身份，工作区实现变化不会静默替换冻结实验。

模型、抽样账本、恢复点及留出原始指标在`runs/entity_set_mlp_cross64_role005_offline_80k_20261022/`；运行后日志位于同名`runtime/train.log`、`runtime/select.log`及各对战日志，`runtime/live.json`显示阶段。准备时生成的[summary.json](summary.json)及[results.md](results.md)明确为未启动状态；正式执行后会更新[execution.json](execution.json)。目前没有正式训练checkpoint或对战结果。

验证：**95项CPU测试通过**，覆盖实际预设联合更新、两头抽样、精确恢复、置换／padding、旧cross权重不被重新缩放及模型分派；见[tests.xml](tests.xml)。实际GPU短验证通过：有限更新、纯权重／AZ恢复、编译policy／policy-only／state-only、前中后期三个native S8根（含N4及随机节点），零编译失败／回退；见[cuda_validation.json](cuda_validation.json)。短验证产物在`runtime/<实验>/smoke/cuda/`，没有写入正式训练池或checkpoint目录。

准备时本机可用空间约9.55GB；按旧运行账本增长率及新模型恢复点估算，完整训练产物约8.97GB，不含编译及评测，余量较小。T7当时未挂载；本次未清理或迁移其他产物，保存频率未改。
