# 第20轮：纯随机节点S128对同模型Greedy

规则范围补充（2026-10-21）：本面板在本次贵族池修正前已完成1000 pair，搜索方得分59.625%，95%区间[57.500%,61.725%]，使用原10张贵族池。原始结果和执行身份保留；新增12张池后不能用当前代码在本目录续写，以下恢复命令只适用于原冻结代码与数据。[旧规范数据快照与新规则](../noble_pool_bga12_20261021/README.md)已单独保存。

2026-10-21，用户要求启动1000 pair正式评测。当前执行状态见`runtime/entity_attention_r20_chance_s128_vs_greedy_20261021/live.json`；完整结果生成于本目录`results.md`、`summary.json`及`evaluation.json`，未完成前不作棋力结论。

- 双方为混合预算持续迭代第20轮原模型，`round-019/checkpoints/final.pt`，SHA `fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`；与刚才BGA分析完全相同，未用EMA。
- A为`native_chance`／S128：每次模拟重采隐藏信息与随机转移，一个共享历史统计池；CPW关闭。N1在此表示一个共享树，不是固定一个隐藏世界。
- B为相同完整模型的Greedy argmax，零搜索。c_puct1.5、无根噪声、温度0、FP/PTP关闭、强制阶段折叠、每次模拟最多512次转移。
- 1000 pair／2000局，每pair使用同一开局种子交换先后手。开局种子41000000–41000999，policy种子基数41010000，bootstrap种子41020000，搜索基种子41030000；按pair bootstrap10000次，和棋半分。
- 执行使用既有`native_batched`入口，支持搜索对Greedy；当前pool入口要求双方都为MCTS，不扩展生产功能。复用采集的1000活跃局、B500、candidate/batch buckets、FP32、native-cudagraph；唯一owner、单槽、协程就绪即发（wait0）。双方共享相同模型与evaluator；Greedy沿用既有eager决策路径。
- 使用checked真实规则runner，保留自然终局、确定性策略循环判和、异常及工程截断分类。单根CUDA smoke只验证工程，不计入正式对战。

结束后脚本自动核验双方checkpoint SHA、1000个完整交换先手pair、逐局S128/N1预算、Greedy零搜索及配对区间，并输出先后手得分、胜和负、异常／非法／截断、循环判和、搜索与GPU执行统计。对Greedy的搜索收益不直接证明更换自迭代教师后的训练收益；本任务不修改在线教师或训练配置。

```bash
# 已由本次任务启动；不要重复启动。
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  artifacts/entity_attention_r20_chance_s128_vs_greedy_20261021/launch.py

# 若进程确实已退出，恢复相同配置、完整pair与sidecar：
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  artifacts/entity_attention_r20_chance_s128_vs_greedy_20261021/run.py run
```

日志：`runtime/entity_attention_r20_chance_s128_vs_greedy_20261021/run.log`。逐局原始记录在`evaluation.json.games/`，汇总每10 pair刷新一次。配置与命令在`config.json`、`plan.json`、`launch.json`、`execution.json`；执行期源码副本、dirty patch和编译cache留在本实验runtime目录。支持按完整pair恢复。
