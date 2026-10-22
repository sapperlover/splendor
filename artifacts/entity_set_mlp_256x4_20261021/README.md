# 实体汇总＋256宽、4残差块MLP

2026-10-21按用户要求实现新的可训练模型，不启动正式离线训练或棋力赛。先启动的[S1024N1对Greedy 1000pair](../entity_attention_r20_s1024n1_vs_greedy_20261021/README.md)使用独立源码快照，MLP开发不改变其执行代码、模型或规则。此前[S256N4面板](../entity_attention_r20_s256n4_vs_greedy_20261021/results.md)完成1000pair得77.20% [75.45%,79.00%]；是否优于N1由同开局对照另行判断。

2026-10-22当前预设已加入[64维／2头动作cross-attention](action_cross_64x2_20261022/README.md)，906,434参数；本次独立记录197项通过、1项CUDA跳过及CPU编译／原生短验证，尚无新正式训练或棋力结论。下方初版及动作头1＋2结果仅适用于各自冻结配置。

2026-10-21后续已完成[动作头1＋2升级](actor_upgrade_12/README.md)：该版冻结预设启用90→32→128非线性旁路和逐通道状态×动作乘积，共855,682参数。**下文结构表、capacity.json和101项CPU／CUDA结果属于初版848,418参数模型**，本次结果独立保存在`actor_upgrade_12/`；没有用初版GPU记录证明新动作头已通过GPU验证。

## 初版结构与容量（历史基线）

初版冻结预设：[model_256x4_baseline.json](model_256x4_baseline.json)（`validate.py`现显式读取此冻结文件）；当前升级预设见上方。实现：[entity_set_mlp.py](../../src/splendor/entity_set_mlp.py)。

| 部分 | 结构 | 参数 |
| --- | --- | ---: |
| 牌编码 | 29原特征＋8关系 → 64 → 64，ReLU | 6,592 |
| 贵族编码 | 17 → 32 → 32，ReLU | 1,632 |
| 状态主干 | 922 → 256；4个256→256→256残差块 | 762,624 |
| Policy评分 | 状态256＋动作82＋目标牌关系8 → 128 → 1 | 44,545 |
| 胜负Value | 状态256 → 128 → 1 | 33,025 |
| 合计 | 无attention、无归一化、无dropout | **848,418** |

FP32权重本体3,393,672字节（约3.24MiB），不包含激活、梯度或AdamW状态。可选仅回合辅助881,443参数；双辅助914,468参数。实际分项与配置见`capacity.json`。这里“4残差块”是8个256×256线性变换，另有输入层，不等同于4个普通Linear。

状态按市场T1/T2/T3、己方保留、对方保留、贵族六个集合分别取masked mean、masked max及归一化有效数量；再拼全局148维与pending牌64维。922 = 148＋5×(64＋64＋1)＋(32＋32＋1)＋64。各组顺序具有语义，组内牌的顺序不影响输出。空集合三个摘要均为0；padding编码前清零并从汇总排除，对手盲保留仍是有效unknown实体。

每个残差块为`ReLU(h + Linear2(ReLU(Linear1(h))))`，参考本地`ladrilho/packages/alphazero-bot/train/azt/model.py`的残差结构；动作输出仍按当前合法候选评分，不照搬Azul固定180动作空间。

购买关系复用现有`market_purchase_relations`：对手五色普通缺口、对手用金后剩余缺口、己方可买、对手可买。状态侧仅市场与双方明保留追加8维，pending与盲保留的新增通道为0；原本已知的己方盲牌属性保留。全局148维没有删字段，贵族17维及12贵族规则无需改动。

## 动作head有直接旁路

动作原82维已包括目标牌29维、目标贵族17维，以及阶段/类型/来源/层级、拿取/支付/弃币等元信息。该82维**原样直接送入动作MLP，不经过任何集合汇总**；另由该动作的公开目标牌及对手资源计算相同8维购买关系，形成90维动作输入。己方盲牌的已知29维仍可直读，但不追加公开关系；没有目标牌或未知牌时关系为0。目标贵族属性同样直连。

因此`logit(a)=LinearOut(ReLU(Wstate·z + Waction·a90 + bias))`同时看到共享局面和目标自身。状态半投影每局面只算一次；训练、eager及`ActorProjection`冻结推理均支持。目标牌属性不会因其他卡牌汇总而丢失，但其余市场牌之间的交互仍依赖压缩后的z；本架构不声称汇总完全无损。

## 训练目标与兼容性

- 新架构名`entity_set_mlp_v1`，可选`entity_set_mlp_aux_v1`；`build_actor_critic`、纯权重、AZ checkpoint与frozen loader均显式分派、严格加载。
- 原29/17/148/82数据schema不变，8维关系在模型内部派生，既有NPZ可复用。没有修改policy/value标签、loss归一化、权重、搜索预算或PSW。贵族新开局分布的覆盖仍取决于后续数据。
- 默认模型仅policy和胜负value。`add_auxiliary_heads`可加终局分差及剩余回合：共享z的独立小MLP；分差无界，剩余回合softplus，沿用分差÷15、回合÷40及原损失配置。普通搜索前向不执行辅助头。
- `splendor.offline_scratch`接受新的基础预设及原none/turns-only/dual配置，保存和恢复模型、AdamW及抽样状态。联合更新调用链已验证；原attention专用`train_split_step`没有扩展到本模型。
- 原attention、Deep-Sets配置和checkpoint仍按原结构加载。attention权重不能直接严格resume为此MLP；需要新初始化或另行蒸馏。本任务未做迁移训练。

构建示例（读取当前升级预设；初版复现使用上方冻结配置）：

```python
import json
from splendor.actor_critic import build_actor_critic

with open("configs/entity_set_mlp_v1/model_256x4.json") as f:
    model = build_actor_critic(json.load(f))
```

## 有限验证

`tests.xml`：101项CPU测试通过，包含新模型和原attention/关系/辅助/offline入口回归。新测试覆盖实体及动作置换、空集合/padding、state-only、动作属性在固定z时仍直接影响logit及梯度、关系通道、辅助训练与推理隔离、严格加载和投影一致。原始训练入口在none/turns/dual三种配置各做3步，对比连续训练与中断恢复后的参数、AdamW及抽样状态逐位一致。

`cpu_validation.json`与`cuda_validation.json`由`validate.py`产生：实际848,418参数预设在已有归档4个样本上做有限更新，保存/恢复AZ及纯权重，并做一次双辅助更新；开/中/后期3个实际根分别走native N1、native N4、随机节点S8，核对总预算及无transition cap。GPU另检查eager/compiled策略及value、state-only一致，明确使用CUDA Graph和ActorProjection，编译失败fallback为0。临时checkpoint与源码快照位于`runtime/entity_set_mlp_256x4_20261021/{cpu,cuda}/`；不会纳入正式候选。

以下回归命令现包含新增用例；CPU／CUDA脚本显式读取冻结初版配置。当前升级预设的复现命令见上方升级记录。

```bash
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_entity_set_mlp.py tests/test_entity_set_mlp_training.py tests/test_entity_attention.py tests/test_entity_attention_relations.py tests/test_value_auxiliary.py tests/test_offline_scratch.py
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/entity_set_mlp_256x4_20261021/validate.py --device cpu
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python artifacts/entity_set_mlp_256x4_20261021/validate.py --device cuda
```

按25实体、32候选估算主要乘加约133万，原128×2 attention约1327万；这排除了非线性、mask、归约等操作，只是结构估算。验证期间有用户正式GPU任务同时运行，耗时仅说明有限检查完成，不作为吞吐/采集加速或棋力结论。
