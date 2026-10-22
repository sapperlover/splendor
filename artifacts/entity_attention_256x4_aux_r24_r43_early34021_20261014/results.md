# 256×4 第 34,021 步（50%）对原 R43：1000pair greedy

状态：1000/1000 pair；passed=True

- 候选：`runs/entity_attention_256x4_aux_r24_r43_20261014/checkpoints/step-000034021.pt`（sha256 `d527d8039ac643902b6b6a12806c84a68e74008103443d1011fdf104e1b9ae9f`）
- 对手：`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt`
- 开局种子 36600000–36600999（与训练驱动的正式 greedy 面板同域，可逐 pair 比较）；双方 greedy、无搜索、交换先后手、和棋半分

**得分：45.450%（899胜/20和/1081负），95% 配对区间 [43.324%, 47.600%]**

| 项 | 值 |
| --- | --- |
| 有效局 | 2000 |
| 异常/非法动作/工程截断 | 0/0/0 |
| 用时（本次调用） | 191.7 秒 |

本面板只回答“50% 节点的裸网是否已超过 R43”，不覆盖搜索棋力与训练终点；结果受单训练 seed 与固定 DEV 开局限制。
