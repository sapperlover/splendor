# 五个离线best对raw R37：greedy与双方S128N1各10k pair

2026-10-27十组正式评测全部完成：5组greedy及5组双方S128N1各10000 pair，共100000 pair／200000局。五个模型均使用原R21–R37的104k池及原完整留出主KL＋终局MSE最低的完整checkpoint，没有重选点或拼接policy／value。完整区间见[results.md](results.md)，跨训练组汇总见[近期实验完成结果](../../docs/reports/auxiliary_offline_results_20261027_zh-CN.md)。

| 原104k模型 | Greedy，10000 pair | 双方S128N1，10000 pair |
| --- | ---: | ---: |
| 仅短期value | 54.050% | 56.290% |
| value＋policy | 55.390% | 56.975% |
| 仅对手policy | 54.722% | 57.098% |
| 无辅助 | 53.265% | 56.700% |
| policy＋得分 | 55.080% | 57.013% |

Greedy中辅助组点估计略高；S128N1五组集中于56.29%–57.10%，没有明显拉开。无辅助本身也超过R37。下文保留冻结配置和恢复命令。

从仓库根目录运行：

```bash
bash scripts/run_offline5_vs_r37_10k_20261027.sh
```

脚本使用conda `splendor`，OMP／MKL／OpenBLAS各1线程；严格串行完成下表的10组面板。每组10000pair，每pair同一开局seed交换先后手，共20000局；合计100000pair／200000局。

| 顺序 | 模型 | 原选中步 | 模式 | pair |
| ---: | --- | ---: | --- | ---: |
| 1 | value | 140221 | greedy | 10000 |
| 2 | value＋policy | 140221 | greedy | 10000 |
| 3 | policy | 140221 | greedy | 10000 |
| 4 | 无辅助 | 122694 | greedy | 10000 |
| 5 | policy＋得分预测 | 140221 | greedy | 10000 |
| 6 | value | 140221 | 双方S128N1 | 10000 |
| 7 | value＋policy | 140221 | 双方S128N1 | 10000 |
| 8 | policy | 140221 | 双方S128N1 | 10000 |
| 9 | 无辅助 | 122694 | 双方S128N1 | 10000 |
| 10 | policy＋得分预测 | 140221 | 双方S128N1 | 10000 |

value指原H2／H6／H14短期value辅助；这里没有使用44k仅value组。五个候选及raw R37均已逐一核对本地checkpoint SHA、架构和原选点记录。`plan.json`保存完整路径／SHA，`selections/`冻结五组原选点记录。脚本复核所有原始百分比完整留出记录和最低联合指标；不要求其他普通checkpoint仍留本机。

固定对手为原R37：`runs/entity_attention_r31_10k_p60k_iteration_20261023/rounds/round-005/checkpoints/final.pt`，SHA `611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045`。候选SHA及联合留出值：

| 模型 | checkpoint SHA | 原KL＋MSE |
| --- | --- | ---: |
| value | `c10557938107b83b13ba4807bdd8599994e47053e859fbd6518f0fd4019fb41b` | 0.9455691484 |
| value＋policy | `2d01e903dcc3147166242a238d0aade42750fe95b7eabd25fa83fcd35b3d2e73` | 0.9396931246 |
| policy | `e4edeb20e8caa3c1abec0146d3757594dd6e0428bd833badd0cb3915c2299a9f` | 0.9413569281 |
| 无辅助 | `baa67bd0c767550d1e515b9dc7bca7c921c883594309e467633066a23b701e49` | 0.9486445348 |
| policy＋得分预测 | `30eb6083f2bd23b4d3e3d60c869a08420c9a3862c9706fbfd2fdae40964fc7ce` | 0.9402481746 |

## 评测设置与保存

greedy沿用CPU8 worker、根policy确定性argmax、零搜索。搜索沿用配置A：CUDA pool、12 search worker、1000活跃局、B500、local microbatch128、owner `pipeline_slots=2`、batch buckets `[64,128,256,500]`、candidate buckets `[32,48,64]`、等待5ms、shared slots2、FP32/native-cudagraph、reused传输、fail-fast。双方预算均为**每世界128次×1世界＝总128次**，底层`simulations=128`、`determinizations=1`；c_puct1.5、折叠强制阶段、每模拟最多512次状态转移，T0、无根噪声、FP/PTP关闭。每个完整结果均核对逐局实际模拟数和确定化数。

使用当前完整12贵族规则、每局400完整回合工程上限及既有确定性循环判平规则。每组按pair计分和bootstrap，和棋半分；10000次整pair bootstrap给出95%区间。工程`passed`不代表模型棋力提升。

十组共用一套本次新冻结开局：47270000–47279999；policy_seed_base47300000（实际每pair使用base＋2×pair_id及其后一个seed）；双方search seed47330000；bootstrap seed47340000。相同seed支持后续配对比较，不把这次比较称为独立FINAL验收。

每100pair打印进度；每1000pair更新全量summary，结束或正常中断会再汇总。完整pair的逐局JSONL持续落盘，降低全量summary频率不降低pair恢复精度，也避免10k面板每25pair反复扫描全量结果。沿用已有评测器及其恢复协议，不修改原实验或历史面板。

| 产物 | 路径 |
| --- | --- |
| 脚本 | [run_offline5_vs_r37_10k_20261027.sh](../../scripts/run_offline5_vs_r37_10k_20261027.sh) |
| 计划、候选及种子身份 | [plan.json](plan.json)、`selections/` |
| 十组正式配置 | `configs/01_greedy_value_vs_R37.json`至`configs/10_search_policy_score_vs_R37.json` |
| 冻结源码／native／牌池／配置 | `runtime/offline5_vs_r37_10k_20261027/source`，清单见[source.json](source.json) |
| 正式原始结果 | 本目录`panels/<面板>.json`及相邻被忽略的`.json.games/`可恢复逐局日志 |
| 汇总报告 | [results.md](results.md)、[summary.json](summary.json) |
| 运行会话／资源／标准输出 | 同名runtime的`panels/<面板>/recording/`及`<面板>.log`；本目录后续`execution.json`保存整体会话 |

## 恢复和检查

恢复使用同一条默认命令。已完整通过身份、局数、先后手及预算检查的面板跳过；未完成面板自动追加`--resume`，复用已完成pair、按原seed重跑半个pair。保留相邻`.json.games/`目录。任一面板失败或中断即停止后续组，不自动重试错误局，也不在五组greedy全部成功前启动搜索。`chain.lock`拒绝本任务重复并发。

2026-10-27修复：`status`按已落盘的面板结果在内存中汇总并打印，只读访问文件，可在`run`持有`chain.lock`时使用；`summary.json`、`results.md`及其临时文件仅由持锁的`run`／`prepare`更新，消除并发查询与报告写入的冲突。同步已准备发布的runner及冻结身份，十组配置、模型和种子保持原值；修订与验证见[status只读修复记录](status_readonly_fix_20261027.json)。

```bash
# 仅准备／核对checkpoint、配置和源码，不进行对局。
bash scripts/run_offline5_vs_r37_10k_20261027.sh prepare

# 只读汇总并打印已有面板状态，不写报告或启动评测。
bash scripts/run_offline5_vs_r37_10k_20261027.sh status
```

## 本次有限验证

[tests.xml](tests.xml)：19项实际冻结实现的回归通过，包括`status`在运行锁已占用时只读输出且不创建／覆盖报告（已有报告与无报告两种情况）、10组顺序、固定best／共同种子／搜索配置A、任意阶段失败停止、已完成面板跳过、仅有sidecar时恢复、错误搜索预算／先后手拒绝，以及已有评测器部分pair恢复和双模型CPU pool恢复。

[cpu_validation.json](cpu_validation.json)：6个真实checkpoint严格加载；每个在同一个新开局根执行greedy前向及一次native S128N1搜索，总6根／768模拟，所有辅助头的推理调用为0。没有完整棋力对局、CUDA评测或性能结论。回归测试内部使用微型fixture的对局只验证工程行为，正式10组没有启动。

源码、输入及交付SHA和验证命令见[verification.json](verification.json)。当前正式结果目录、执行会话和后台进程均未创建；准备阶段的`chain.lock`已释放。
