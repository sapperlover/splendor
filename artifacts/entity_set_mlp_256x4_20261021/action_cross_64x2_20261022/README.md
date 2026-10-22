# 残差MLP：64维／2头动作cross-attention

后续初始化更新（2026-10-22）：当前代码将新建MLP cross的三个角色embedding各缩放到std0.05，见[标定及新离线准备](../../entity_set_mlp_cross64_role005_offline_80k_20261022/README.md)。本页原197项及CPU验证记录对应修改前std1版本，原始JSON／XML不改写；结构和906434参数数量相同，但当前代码不能直接复现其原初始化数值。加载已保存权重不重新缩放。新实验尚未启动正式训练。

2026-10-22按用户要求增加一层动作cross-attention，保留原动作MLP及乘积，并维持三项独立开关。当前[模型预设](../../../configs/entity_set_mlp_v1/model_256x4.json)为906,434参数。未启动正式训练、棋力评测或长消融；本次工程检查不证明policy收敛或棋力改善。

## 网络与调用链

实现：[entity_set_mlp.py](../../../src/splendor/entity_set_mlp.py)。牌64维及贵族32维的原编码结果保留在`EntitySetMLPState(shared,cards,nobles)`，没有再次执行实体编码器；状态仍以原mean/max/count摘要进入256宽／4残差块MLP，输出z256。

动作分支构造25个64维实体：上下文1、双方玩家2、三层市场12、双方保留6、贵族3、pending1。牌64维直接复用；贵族32→64，上下文120→64，双方玩家共用14→64。附加type／owner／tier语义，组内无槽位位置编码；mask排除padding，未知盲保留仍作为有效实体。三个全局／玩家token始终有效，空市场／贵族／保留不会产生全屏蔽attention。

```text
q64 = LN(GELU(Wq_state · z256 + Wq_action · a90 + bq))
K64, V64 = Linear(LN(实体25×64))
c64 = SDPA(q, K, V)                  # 2头，每头32维，dropout=0
s128 = Ws · z256 + bs
u128 = Wa · a90 + Linear32→128(ReLU(Linear90→32(a90)))
h128 = ReLU(s128 + u128 + γ128 ⊙ s128 ⊙ u128 + Linear64→128(c64))
logit = Linear128→1(h128)
```

同一局面的K/V及Query状态半投影各计算一次；attention拼接后直接64→128，没有重复的64→64输出投影、实体self-attention或新增FFN。使用[PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)的有效位置布尔mask与默认每头缩放。Query／memory各有LayerNorm，仅作用于新增分支。

eager、ActorProjection冻结推理、compiled policy/value与policy-only均使用同一融合函数。value-only仅消费原共享z，不执行实体对齐、角色读取、Q/K/V、cross-attention或动作MLP。胜负及辅助头用Sequential子类提取z，参数键仍是原0／2层，不增加包装层前缀。none／turns／dual辅助及混合预算分别抽样联合更新均经过真实入口验证。

## 配置、容量与旧结果

- `action_cross_hidden=64, action_cross_heads=2`开启当前单层分支；hidden=0关闭。heads须为正整数，hidden非零时须能整除heads。旧配置缺省0／2，不静默迁移旧权重。
- 原`action_bottleneck=32`和`multiplicative_interaction=true`保持开启；分别设置0／false可关闭。本次没有更改γ初值1或旧动作分支的初始化。
- 新增50,752参数，总906,434；仅回合／双辅助分别939,459／972,484。分项见`capacity.json`。原855,682／848,418参数模型的配置和权重仍可严格加载；当前新预设不能直接严格恢复旧结构权重或AdamW。
- 当前预设的主要前向MAC与此前估算一致：32候选、policy＋value约2,335,136，旧动作头1＋2为1,549,472，约+50.7%。估算排除非线性、归一化、mask、逐元素乘加、softmax和调度，不是耗时测量；候选宽度按实际padding bucket计算。value-only新增密集乘加为0。
- 原`offline_80k_lr1e4_20261021.json`、两次8万局训练、checkpoint及对战原始结果保持其冻结855,682／583,682参数身份。本次没有恢复或改写它们。初版及动作头1＋2的原始验证JSON／XML保留，后者的复现脚本改读冻结`../actor_upgrade_12/model_256x4_actor12.json`。

## 动作MLP与乘积可能的负作用

以下是结构风险，尚无本项目消融证据证明它们导致policy慢：

- 小动作MLP可能偏重目标自身或动作类型等局部特征，并与cross-attention路径形成冗余；新增容量也可能改变优化。原始90维直连保留，32维瓶颈没有截断完整动作入口。
- 乘积更值得优先单独消融。对激活前融合`f=s+u+γsu`，有`∂f/∂s=1+γu`、`∂f/∂u=1+γs`，另一分支的幅度会放大、抵消或改变梯度方向，再叠加ReLU门控。原s/u没有LayerNorm；新增query/memory归一化不等于归一化了原乘积。
- γ可学习不保证优化自动找到好解，也不能由两边都高频梯度裁剪推断只有乘积有问题。已有MLP和attention差距同时涉及实体读取、实体self-attention、归一化等，未单独归因于这两项。

下一轮若只增加一个消融，建议固定cross-attention及训练配方，仅关闭`multiplicative_interaction`，观察同一早期留出policy KL；之后再独立关闭`action_bottleneck`。本次只提供开关与正确性检查，没有执行这些正式训练，也没有承诺某项一定有益。

## 有限验证

`tests.xml`：197项通过、1项CUDA条件跳过。覆盖不同cross尺寸、头数和三个独立开关；实体／候选置换、padding／空集、固定z下未汇总实体影响logit及梯度、Query依赖状态和动作、K/V及状态Query只投影一次、value-only跳过分支、零cross残差复原旧输出、848,418与855,682旧权重严格加载、投影一致性、辅助升级、混合预算及三种辅助配置的参数／AdamW／RNG精确恢复。

`cpu_validation.json`用实际906,434参数预设，在已有归档4个样本上做有限更新、纯权重／AZ往返、CPU精确恢复与双辅助更新；确认新旧动作分支参数实际变化；核对eager与compiled投影的policy/value、policy-only和state-only，随后对开／中／后期三个真实根分别执行native S8N1（总8）、native S2N4（总8）、随机节点S8N1（总8）。编译采用fail-fast，无失败回退或eager回退；无transition cap。这里只是工程验证。

命令、种子、样本索引与SHA、代码／模型身份和环境保存在JSON；checkpoint及源码快照位于忽略的`runtime/entity_set_mlp_cross64_20261022/cpu/`。本次环境CUDA不可用，没有新GPU结果、吞吐基准或棋力结论。

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -o addopts= -q tests/test_entity_set_mlp.py tests/test_entity_set_mlp_training.py tests/test_offline_mixed_budget.py tests/test_actor_critic_4a.py tests/test_actor_critic_invariance_4a.py tests/test_entity_attention.py tests/test_entity_attention_relations.py tests/test_value_auxiliary.py tests/test_offline_scratch.py tests/test_inference_modes.py::test_actor_projection_matches_eager_logits_and_values tests/test_inference_modes.py::test_projection_keeps_the_critic_bit_identical --junitxml=artifacts/entity_set_mlp_256x4_20261021/action_cross_64x2_20261022/tests.xml
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python artifacts/entity_set_mlp_256x4_20261021/action_cross_64x2_20261022/validate.py --device cpu
```

归档与root bank路径在验证脚本顶部；跨机器复现需提供相同输入。模型片段不是训练CLI配置，正式训练应使用独立新run并明确嵌入此配置。
