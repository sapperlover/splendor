# 工作包 4A：PyTorch Deep-Sets Actor–Critic

## 1. 完成范围与输入边界

本工作包新增第一版小型 PyTorch Actor–Critic，只消费工作包 3B 的数值 batch：全局特征、各物件集合及 mask、pending card、候选特征及 candidate mask。模型模块不导入 `GameState`、engine、rules、RNG 或 replay，也不接收 `candidate_actions`、执行 handle、card/noble ID 或隐藏牌信息。

特征 schema 仍为 `splendor-observation-action-v1`，3B 的 29/17/148/82 维契约未修改。规则引擎、Observation、Random/Score/Development 策略和评测行为未修改。本工作包没有 rollout、reward、GAE、PPO、optimizer、自我对弈或训练 checkpoint。

## 2. 最终网络结构

所有宽度集中在 `ActorCriticConfig`，使用 PyTorch `Linear` 的默认初始化。

| 模块 | 输入 | 结构 | 输出 |
|---|---:|---|---:|
| 共享 CardEncoder | 29 | Linear(29,64) → ReLU → Linear(64,64) → ReLU | 64 |
| 共享 NobleEncoder | 17 | Linear(17,32) → ReLU → Linear(32,32) → ReLU | 32 |
| StateEncoder | 570 | Linear(570,256) → ReLU → Linear(256,256) → ReLU | 256 |
| action representation | 82 拆分 | meta 36 + shared card 64 + shared noble 32 | 132 |
| Actor | 388 | Linear(388,128) → ReLU → Linear(128,1) | 每候选 1 个 logit |
| Critic | 256 | Linear(256,128) → ReLU → Linear(128,1) | 每样本 1 个 value |

派生维度为：

- `state_input_dim = 148 + 5 × (64 + 1) + (32 + 1) + 64 = 570`
- `action_representation_dim = 36 + 64 + 32 = 132`
- `actor_input_dim = 256 + 132 = 388`
- 总 trainable parameter count：`302,626`

Critic 只读取当前玩家视角的 state embedding，不读取候选；输出没有 tanh 限幅。Actor 对每个候选独立应用同一评分网络，没有候选位置参数、slot embedding、attention 或固定 252 维动作空间。

## 3. 集合聚合与共享参数

三个 market tier、self reserve、opponent reserve 分别使用同一个 CardEncoder 后做 masked sum；market 除以固定容量 4，reserve 除以固定容量 3，并分别附加归一化有效数量。noble 使用同一个 NobleEncoder，masked sum 除以容量 3并附加数量。各集合没有 flatten 槽位。

pending card 和 action target card 复用同一个 CardEncoder；场上 noble 和 ClaimNoble target 复用同一个 NobleEncoder。无 pending 时 embedding 为精确零。无 card/noble target 的候选也会在共享编码后显式归零，避免 Linear bias 给不存在的目标产生伪信息。对手未知盲保留仍是 `valid=1, known=0` 的真实物件，会正常编码而不是按 padding 排除。

所有 mask 都在进入编码器前先将无效 raw feature 归零，并在编码后再次排除对应结果。因此即使 mask-false padding 被写入巨大数值，也不能通过溢出或 Linear bias 影响有效输出。

## 4. Candidate mask 与前向接口

`tensorize_batch` 将 3B 的 NumPy batch 转成 FP32/bool `TorchDecisionBatch`；该 tensor 类型故意不含 action 对象。`ActorCritic.forward(batch)` 返回：

```text
logits: [B, M]
values: [B]
```

`M` 只等于当前 batch 的最大候选数。`masked_logits` 将 padding logit 设为 `-inf`，`distribution` 仅据此构造 `torch.distributions.Categorical`。测试确认 padding 概率严格为 0、有效概率和为 1、单候选概率严格为 1；任一样本没有候选会明确失败，forced skip 仍完全由引擎处理。

## 5. 保存与加载

`save_weights` 只保存：

- 权重格式版本；
- feature schema version；
- model config；
- `state_dict`。

`load_weights` 会验证格式和 schema，并严格加载 state dict。相同输入在 save/load 前后的 logits 和 values 逐元素一致。这不是训练 checkpoint，不含 optimizer、训练步数或 RNG。

## 6. 测试结果

新增 19 个模型级测试，覆盖：

- CPU 的 batch size 1/多样本、ragged candidate padding、单候选、真实 51 候选和四个决策阶段；
- empty market/reserve/noble、unknown opponent reserve、无 pending 与 PAYMENT pending；
- market、双方 reserve、noble 任意重排后 value 不变，同一语义 action 的 logit 不变；
- 对手盲保留真实身份交换、未知未来牌堆重排后，Actor/Critic 输出一致；
- market/reserve/pending/action target 确实调用同一 CardEncoder，noble set/action target 调用同一 NobleEncoder；
- 同一公开卡在状态和 action target 中得到相同 card embedding，公开属性变化在两条路径上一致生效；
- candidate 及各集合 mask-false padding 写入巨大值后，有效概率、logit 和 value 不变；
- Actor、Critic、StateEncoder、CardEncoder、NobleEncoder 均获得 finite 非零梯度，candidate card target 路径可回传至共享 CardEncoder；
- schema mismatch 明确失败，保存加载输出一致，模型模块没有越过公开数值边界。

完整旧测试、3B 测试和 4A 测试均通过。受工具沙箱设备隔离影响，普通沙箱运行会跳过 CUDA case；同一 `splendor` Conda 环境在非沙箱 GPU 上单独运行 CUDA 测试通过，确认 logits、values 和 masked distribution 都位于 CUDA 且为 finite。

## 7. Forward sanity benchmark

环境：PyTorch 2.14.0+cu130、NVIDIA GeForce RTX 4050 Laptop GPU。输入为 batch size 32、每样本 30 个候选；预热 20 次后统计 100 次纯 tensor forward，不包含 NumPy 编码与 tensorize，结果仅作粗略 sanity check：

| device | 平均耗时 / batch | 约 decisions/s | CUDA 峰值分配 |
|---|---:|---:|---:|
| CPU | 4.481 ms | 7,141.8 | — |
| CUDA | 1.061 ms | 30,172.2 | 12.85 MiB |

模型规模和显存占用符合轻量设计；本阶段没有据此做 `torch.compile`、JIT、AMP 或性能重构。

## 8. 尚未实现

工作包 4B 及后续的 rollout/environment adapter、reward、GAE、PPO loss、optimizer loop、自我对弈、对手池、行为克隆、完整训练 checkpoint 和长训练均未开始。
