# 残差MLP动作头：非线性分支＋乘性交互

2026-10-21按用户要求只实现方案1和2。[本版冻结预设](model_256x4_actor12.json)开启两项；pooling、状态残差主干、policy/value目标、数据schema及搜索预算保持原行为。没有加入Set Transformer或动作Cross-Attention，没有启动正式训练或棋力赛。

2026-10-22：当前模型片段已升级为[cross-attention版本](../action_cross_64x2_20261022/README.md)；本文855,682参数和原始JSON／XML保留原版本身份。复现脚本改读本目录冻结配置，不用当前906,434参数预设覆盖旧结果。

## 动作评分

实现：[entity_set_mlp.py](../../../src/splendor/entity_set_mlp.py)。`a`为原动作属性82维＋既有目标关系8维，`z`为共享状态256维：

```text
s128 = Ws · z + b
u128 = Wa · a + Linear32→128(ReLU(Linear90→32(a)))
h128 = ReLU(s128 + u128 + γ128 ⊙ s128 ⊙ u128)
logit = Linear128→1(h128)
```

保留原始动作直连；32维只约束新增分支，不截断直接通道。γ是128个独立可学习系数，初始化为1。两个分支的Linear均使用原PyTorch默认初始化。状态投影每局面只算一次，动作分支仅对候选计算。配置尺寸仍可调整；关闭购买关系时动作输入为82维。

训练、eager及ActorProjection冻结推理共用同一个融合函数。ActorProjection继续缓存原线性状态／动作投影及输出权重；新增分支和γ从冻结模型读取。value-only不执行动作分支。辅助升级保留全部新增参数，none／turns／dual沿用现有目标和AdamW恢复流程。

## 配置与兼容

- `action_bottleneck=32`启用新增非线性分支；0关闭，必须是非负整数。
- `multiplicative_interaction=true`启用逐元素乘积；false关闭，必须是bool。
- 两项独立可配置，完整保存到纯权重与AZ checkpoint的model config。省略字段为0／false，构建原始848,418参数结构；旧权重严格加载，未做静默迁移。
- 当前预设855,682参数，较原版增加7,264：非线性分支7,136＋γ128。仅回合／双辅助分别888,707／921,732参数。逐项统计见`capacity.json`。
- 新预设不能直接严格恢复初版模型的权重或AdamW；本次没有增加跨结构warm start或权重迁移。
- 初版的`capacity.json`、CPU／CUDA验证JSON和101项测试记录保留原样。初版脚本改为读取冻结的`../model_256x4_baseline.json`，与更新后的当前预设分开。

## 有限验证与复现

`tests.xml`：154项通过、1项CUDA条件跳过。相关CPU回归覆盖新旧结构和两个独立开关、动作／实体置换、candidate padding、非线性与乘积行为、真实梯度更新、严格加载、辅助升级和none／turns／dual三步训练的精确恢复。

`cpu_validation.json`由本目录`validate.py`产生：实际855,682参数预设读取既有归档4个样本，做有限policy/value与双辅助更新、权重和AZ checkpoint往返及CPU精确恢复；对开／中／后期三个实际根检查eager与compiled投影policy/value、policy-only及state-only一致性，再运行原生S8N1（总8）、S2N4（总8）及随机节点S8N1（总8）各一个根。新增分支两层权重和γ在实际预设更新中均有非零变化；编译失败策略为fail-fast，实际编译失败／eager回退均为0。仅属工程验证，不形成吞吐或棋力结论。

原始配置、样本索引／SHA、种子、代码身份、环境和命令保存在JSON；临时checkpoint与源码快照位于忽略的`runtime/entity_set_mlp_actor12_20261021/cpu/`。当前环境`torch.cuda.is_available()`为false，本次没有GPU结果；初版GPU结果只适用于初版结构。

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -o addopts= -q tests/test_entity_set_mlp.py tests/test_entity_set_mlp_training.py tests/test_actor_critic_4a.py tests/test_actor_critic_invariance_4a.py tests/test_entity_attention.py tests/test_entity_attention_relations.py tests/test_value_auxiliary.py tests/test_offline_scratch.py tests/test_inference_modes.py::test_actor_projection_matches_eager_logits_and_values tests/test_inference_modes.py::test_projection_keeps_the_critic_bit_identical --junitxml=artifacts/entity_set_mlp_256x4_20261021/actor_upgrade_12/tests.xml
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python artifacts/entity_set_mlp_256x4_20261021/actor_upgrade_12/validate.py --device cpu
```

验证脚本使用的既有归档与root bank路径定义在脚本顶部，跨机器复现需提供对应原始文件；正式训练需单独准备训练CLI配置，本目录只提供模型预设与有限验证入口。
