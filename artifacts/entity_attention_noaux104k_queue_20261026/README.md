# Policy＋得分辅助之后的104k无辅助串行任务

2026-10-27核对：前序policy＋得分和后继无辅助的训练、选点、greedy1000 pair及S1024N1搜索500 pair均已完成，队列execution状态为complete。无辅助210332步早停、选中122694步，原两面板51.075%／57.50%；[新版10000 pair面板](../offline5_vs_r37_10k_20261027/README.md)也已完成。下方等待／PID是最初启动记录。

2026-10-26，用户要求新实验跟在当前脚本后执行。本队列只追加一组：

1. 等待`entity_attention_128x2_policy_score_r21_r37_20261026`的训练、选点、R37 greedy1000pair和搜索A500pair全部成功。
2. 启动`entity_attention_128x2_noaux_r21_r37_20261026`，执行相同完整流程。

```bash
bash scripts/run_attention128_noaux104k_queue_20261026.sh
# 仅查看已记录的队列状态，不轮询GPU或训练进度。
bash scripts/run_attention128_noaux104k_queue_20261026.sh status
```

本次已启动并确认等待中，实际Python队列PID为76148，进程身份见[launch.json](launch.json)；已有活动队列时无需重复启动，`queue.lock`拒绝并发队列。重启同一命令会验证前序完整成功，跳过已完成的后继或按其完整checkpoint恢复。

复用原[serial_queue.py](../entity_attention_auxiliary_queue_20261024/serial_queue.py)，这里只更换前驱与唯一后继，原三组队列不变。阻塞等待前驱`chain.lock`，监督锁覆盖训练、选点、两组arena及最终summary／execution提交；锁释放后还严格校验成功会话、训练完成、checkpoint SHA和两组完整pair数／工程检查。前序失败或中断即停止，不因训练子进程退出、仅有final或只完成一组arena而推进。等待期间停止队列不会向用户前序发送信号。

当前原计划／runner／launcher哈希保存在[plan.json](plan.json)。运行状态：`runtime/entity_attention_noaux104k_queue_20261026/state.json`；队列输出：同目录`queue.log`；退出时本目录保存`execution.json`。完整配置与有限验证见[无辅助实验](../entity_attention_128x2_noaux_r21_r37_20261026/README.md)。
