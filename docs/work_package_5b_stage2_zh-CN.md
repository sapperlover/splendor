# 工作包 5B Stage 2：multi-env batched collector

## 范围

本阶段实现多个独立逻辑环境的交错采样：每个环境在 CPU 上推进到 learner 决策边界，等待中的 Observation 和 ragged 合法候选由现有 collator 组批，通过一次 Actor–Critic forward 得到动作分布和 value，然后将候选索引分发回原环境。

没有修改 transition、terminal reward、terminated/truncated、GAE 或 PPO 数学。也没有启动 self-play 训练。

## 实现要点

- `BatchedRolloutCollector` 为每个活动环境保留独立的 game/opponent 状态、learning player、episode ID、replay/failure context 和 CPU policy RNG；
- `PolicySampler.sample_batch` 只合并模型 forward，每个环境仍使用自己的 RNG 从各自分布采样；
- 动作仍是当前 `DecisionRequest.legal_actions` 中的候选索引，不重新实现合法性，不改变 candidate ordering/mask/handle 映射；
- successor value 也合并 forward；rollout 返回前按 episode 保持连续，现有的逆序 GAE scan 不跨 episode；
- collector 能保留未完成对局到下个 rollout，也能在测试/调试中 drain 当前对局；
- benchmark 记录 rollout 吞吐、完成对局吞吐、平均 inference batch、环境/编码、GPU inference、GAE、PPO update、CUDA peak 和进程 peak RSS。

## 确定性与语义验证

测试覆盖：

- active envs = 1 时，与 serial collector 的 episode、学习方、动作索引、candidate handle/mask/features、reward、terminal 标志、logprob/value/bootstrap 一致；
- active envs = 4 的固定输入重跑产生相同的完整动作轨迹和最终状态 digest；
- 批量 forward 与逐样本 forward 的 logits/value 在数值容差内一致，candidate mask 和执行 handle 不变；
- 多环境样本的 episode 分组与既有 GAE 边界一致，对局无工程截断，采样不修改学习模型参数。

FP32 在单样本 GEMV 与批量 GEMM 中可能有末位浮点差异，因此不要求不同 batch shape 的 logits 位级一致；验收的是同一 batched 配置的可重现性、数值一致性和动作/规则语义一致性。

## RTX 4050 Laptop GPU 实测

使用 `step_000500005.pt`，每种配置采集 8,192 个 learning decisions，同时执行一次未修改的 GAE 和 PPO update：

| collector | decisions/s | games/s | mean batch | env + encoding | GPU inference | PPO update | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| serial | 186.34 | 3.87 | 1.00 | 15.87 s | 28.09 s | 1.40 s | 1.00× |
| batched-1 | 186.06 | 3.86 | 1.00 | 15.90 s | 28.13 s | 1.24 s | 1.00× |
| batched-8 | 431.98 | 8.86 | 7.92 | 14.80 s | 4.17 s | 1.20 s | 2.32× |
| batched-16 | 480.94 | 9.57 | 15.84 | 14.72 s | 2.32 s | 1.21 s | 2.58× |
| batched-32 | 513.69 | 9.97 | 31.69 | 14.62 s | 1.33 s | 1.23 s | 2.76× |

CUDA peak allocated 约 121–125 MiB；进程 peak RSS 约 1,799–1,806 MiB。active envs = 32 在本次短基准中最快，但 16 到 32 的边际收益已收窄，且主要瓶颈已转向环境/编码。原始结果保存在 `runs/5b_infrastructure/batched_benchmark_step_000500005.json`。

Stage 2 针对性测试为 `5 passed`。
