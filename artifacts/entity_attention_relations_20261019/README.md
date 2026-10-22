# 公开卡牌购买关系与 value 多实体读取

2026-10-19实现。最初交付范围是模型结构、预设与有限正确性验证，当时没有启动正式训练或棋力评测；同日后续按用户要求启动[新版128×2无辅助头、lr1e-4的2万局从零训练](../entity_attention_128x2_relations_no_aux_r24_r43_20261019/README.md)，正式学习结果以该独立记录为准。代码入口为[entity_attention.py](../../src/splendor/entity_attention.py)，本目录保留工程检查产物；本地检查用checkpoint保存在忽略的`runtime/entity_attention_relations_20261019/`。

## 输入与结构

从原29维卡牌、148维全局输入派生8维购买关系，不修改特征schema、NPZ或C++传输格式，已有历史数据可直接使用。每张已知有效市场卡、双方明保留的已知牌，额外得到对手各色普通筹码缺口5维、对手黄金后未覆盖缺口1维、双方资源是否足以购买各1维。各色顺序为白蓝绿红黑。明保留扩展于同日后续完成。

对手第c色缺口为`max(成本[c]−对手加成[c]−对手普通筹码[c], 0)`，除以7；黄金后缺口为`max(各色缺口之和−黄金数, 0)`，除以14。已归一化的公开整数先还原并round，再判断阈值，避免浮点误差把刚好买得起判为不足。己方两组缺口原卡牌特征已经包含，继续复用。可购买标志仅表示资源足够，不表示当前决策阶段存在对应合法动作。

总缺口可由各色缺口求和，双方资源差可由两组缺口相减，因此不再另外堆叠；可购买标志保留为显式阈值提示。未知卡与padding的新增关系全零。只读取当前观察中公开的成本、加成和筹码，不访问隐藏世界、终局标签或候选列表。市场和双方明保留共用原8→hidden无偏置投影，结果加到对应卡牌token，候选仍以原82维动作query读取共享状态。

明保留按`source_market=1`且`known=1`筛选。己方盲保留原有的私有29维卡牌信息保留，但不追加8维关系；对手盲保留仍为unknown。无论牌属于谁，己方/对手都相对于当前行动者；保留牌的“资源够买”不意味着非持有者能合法购买，归属由owner embedding和合法动作表达。复用历史名称`market_relations`配置与`market_relation_encoder`参数，不新增开关或参数。

状态交互后，value读取`concat(全局token, 己方玩家token, 对手玩家token, 有效市场卡token的均值)`，接`LayerNorm(4H)→Linear(4H,H)→GELU→Linear(H,1)`。空市场均值为零。均值仍仅聚合12个市场槽位；保留牌通过共享attention影响这些表示。聚合排除padding，保持同层市场集合置换不变。胜负value仍为当前行动者视角、无界输出；有辅助头时，分差和剩余回合各自也使用该读取结构，剩余回合输出保留softplus。三个头并行且参数独立。

| 预设 | 注意力 | FFN | 无辅助参数 | 双辅助参数 |
| --- | --- | --- | ---: | ---: |
| [128×2](../../configs/entity_attention_v1/model_relations_128x2.json) | 4 heads | 512 | 583,682 | 717,316 |
| [256×4](../../configs/entity_attention_v1/model_relations_256x4.json) | 8 heads | 1024 | 3,828,226 | 4,357,636 |

两项改动独立配置：`market_relations=true`与`value_readout="global_players_market"`。省略时分别为false和`global`，保持历史结构、参数顺序与初始化行为。新预设显式启用两项，原训练配置与旧checkpoint不自动切换。`EntityAttentionAuxConfig`和`add_auxiliary_heads`保留这两个选项；现有随机初始化离线入口`offline_scratch`接受新预设作为`model_config`。结构交付时未执行长训练；后续新版128训练见上述独立专题。预设只是模型配置片段，不是训练CLI配置。

```python
import json
from splendor.actor_critic import build_actor_critic

with open("configs/entity_attention_v1/model_relations_128x2.json") as source:
    model = build_actor_critic(json.load(source))
```

checkpoint保存完整开关与尺寸，沿用严格加载。旧R43可正常加载；旧权重不能直接严格加载到新的readout，不静默丢弃或扩展参数。固定历史128×2的`entity_attention_policy_value_v1`组合入口仍保持其原适用范围。

## 有限验证与复现

首次仅市场版本的95项相关CPU测试全部通过，见[原JUnit](tests.xml)。[原128×2 CPU结果](128x2_cpu_validation.json)确认真实样本训练/恢复、后续AdamW更新逐位一致、3个原生S8搜索根通过，且R43原输出逐位不变。[原128×2 CUDA](128x2_cuda_validation.json)及[原256×4 CUDA](256x4_cuda_validation.json)均通过真实样本更新、保存加载、eager/compiled/state-only一致及各3个原生S8搜索根；编译fallback为0。原始结果及版本身份保留，不用本次结果覆盖。

明保留扩展的检查产物独立写入`public_reserves/`。新增测试检查双方明牌收到关系投影及梯度、当前行动者视角、己方已知盲牌/对手未知盲牌/padding新增关系为0，以及两种尺寸的参数量不变。`validate_public_reserves.py`从checked引擎依次执行双方各一次明保留、再各一次盲保留，在后3个真实局面检查eager/compiled/state-only及各S8原生搜索；同时核对旧R43输出逐位不变。固定局面seed39240000，搜索seed39240100起，无正式训练或棋力赛。

本次[97项CPU测试](public_reserves/tests.xml)全部通过；[128×2 CPU](public_reserves/128x2_cpu.json)、[128×2 CUDA](public_reserves/128x2_cuda.json)、[256×4 CUDA](public_reserves/256x4_cuda.json)检查均通过，每组各3个S8原生搜索根，CUDA编译fallback为0，旧R43输出逐位一致。两种尺寸参数仍为583,682／3,828,226（无辅助）。

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_entity_attention.py tests/test_entity_attention_relations.py \
  tests/test_value_auxiliary.py tests/test_offline_scratch.py tests/test_alphazero_training.py \
  --junitxml=artifacts/entity_attention_relations_20261019/public_reserves/tests.xml
conda run --no-capture-output -n splendor env PYTHONPATH=src python \
  artifacts/entity_attention_relations_20261019/validate_public_reserves.py --device cpu --preset 128x2
conda run --no-capture-output -n splendor env PYTHONPATH=src python \
  artifacts/entity_attention_relations_20261019/validate_public_reserves.py --device cuda --preset 128x2
conda run --no-capture-output -n splendor env PYTHONPATH=src python \
  artifacts/entity_attention_relations_20261019/validate_public_reserves.py --device cuda --preset 256x4
```

CPU测试覆盖手算缺口与黄金、阈值、unknown/padding、直接读取的梯度、集合/候选置换、state-only、严格保存加载和辅助升级。有限验证用4个真实历史样本做1–2步AZ更新，检查checkpoint、AdamW恢复、frozen/pure加载，并跑开中后期3个S8/N1原生搜索根；CUDA检查eager与compiled一致、无编译fallback。检查seed39200000，抽样39210000，搜索39230000起。原始JSON记录设备、代码身份、配置SHA、样本引用与耗时。

`legacy_reference.npz`是在改模型代码前以原R43记录的3个固定请求CPU FP32输出，身份见`legacy_reference_identity.json`；验证要求新代码加载R43后的输出逐位相同。这些检查只证明工程调用链正确，不证明拟合或棋力改善。
