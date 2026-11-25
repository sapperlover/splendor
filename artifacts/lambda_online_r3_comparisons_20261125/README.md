# λ在线第三轮实际使用模型的三组对照：全部FPU／S128N1

2026-11-25按用户要求准备三个独立脚本；用户随后明确“三组都开启FPU”。因此三组的**双方均为`fpu_reduction=0.3、c_puct=1.0`**，双方S128N1（每世界128×1世界＝总128），每组2000pair／4000局，总6000pair／12000局。正式评测未启动，现有在线任务保持原样。

```bash
# 第三轮实际使用final vs 原λ161924
bash scripts/run_lambda_r3_used_vs_lambda161924_fpu_s128n1_2000pair_20261125.sh

# 第三轮实际使用final vs 第一轮实际使用final
bash scripts/run_lambda_r3_used_vs_r1_used_fpu_s128n1_2000pair_20261125.sh

# 第三轮实际使用final vs 同一第三轮的3000步节点
bash scripts/run_lambda_r3_used_vs_r3_step3000_fpu_s128n1_2000pair_20261125.sh
```

## “实际使用”模型的核对

分支为`runs/lambda161924_fpu_p60k_iteration_20261124`。第一轮和第三轮的实际使用模型均为各自**本轮10000步final**。分别由`state.json`已完成轮的checkpoint、已执行greedy的候选SHA，以及下一轮采集manifest中的教师身份交叉确认；没有用采集前的旧教师替代本轮输出。

| 名称 | 相对分支路径 | 本段步 | 累计步 | SHA256 |
| --- | --- | ---: | ---: | --- |
| 第三轮实际使用（所有A方） | `rounds/round-002/checkpoints/final.pt` | 10000 | 191924 | `8b6798463c23feb1745faaff0364b9de7a92e27ba825d507063355309e6c1667` |
| 第一轮实际使用 | `rounds/round-000/checkpoints/final.pt` | 10000 | 171924 | `79e601e2084a0b58020db82e66b4ed3271efd1c4f881757b7280a05b43de1e42` |
| 第三轮3000步 | `rounds/round-002/checkpoints/step-000003000.pt` | 3000 | 184924 | `68396a6caa8b6df1d1189e0b025503d777083098f95e1009afca146f9845e077` |

第一轮final已作为第二轮采集教师；第三轮final已作为第四轮采集教师。第三轮final与第三轮3000步的训练round_index、父模型和replay identity相同。

原λ模型固定为`runs/outcome_60k_scalar_lambda_20261120/checkpoints/step-000161924.pt`，SHA `561f055af56719ad7fbfa9779e93955ce139d5651b6a2b9b9dbc7e10b7182b3b`，本段／累计均为161924。四个模型的完整身份和实际使用证据见[plan.json](plan.json)。未来继续训练时仍固定这些节点，不跟随第四轮或其他更晚轮次。

## 统一搜索与配对

三组仅B方模型不同，A方、双方搜索配置、执行参数及种子相同。沿用配置A：CUDA pool、12 search worker、1000活跃局、B500、local microbatch128、owner双槽、batch buckets `[64,128,256,500]`、candidate buckets `[32,48,64]`、等待5ms、shared slots2、FP32/native-cudagraph、reused传输、fail-fast。使用正式轮间评测执行模板，将双方预算设为S128N1，并按用户最终指令把双方搜索统一设为FPU0.3／c_puct1.0。

根噪声0、T0、FP/PTP关闭，强制阶段折叠、每模拟最多512状态转移；完整12贵族规则、400完整回合工程上限及既有确定性循环判平保持。

三组共用新开局55500000–55501999，policy_seed_base55510000，双方search seed55520000，bootstrap seed55530000。每pair同开局换先后手，和棋半分，以完整pair做10000次bootstrap给出95%区间。每100pair输出进度、每500pair及结束／正常中断更新全量summary，逐局JSONL持续落盘。

## 执行和恢复

从仓库根目录运行，自动使用conda `splendor`、`PYTHONPATH=src`及单线程OMP／MKL／OpenBLAS。三个脚本分别执行对应面板，不自动排队或附加其他实验。相同命令恢复：完整组核验后直接返回，部分组自动`--resume`，完整pair跳过、半个pair按原seed重跑；保留相邻`.json.games/`目录。每组独立锁，失败或中断退出。

任一脚本末尾加`prepare`只核对模型、生成配置并校验冻结源码；加`status`只读取状态，不写报告或启动评测。

| 产物 | 位置 |
| --- | --- |
| 固定模型／实际使用证据、规模和种子 | [plan.json](plan.json) |
| 三组正式配置 | 本目录`configs/r3_used_vs_<lambda或r1或r3_step3000>_FPU_S128N1_2000pair.json` |
| 实际执行源码／native／牌池／配置 | `runtime/lambda_online_r3_comparisons_20261125/source`；[source.json](source.json)保存逐文件SHA |
| 本地原始结果及恢复日志 | 同名runtime的`panels/r3_used_vs_<对手>.json`及相邻`.json.games/` |
| 紧凑报告 | 本目录`summary_<对手>.json`及`results_<对手>.md` |
| 会话／日志／资源 | 本目录后续`execution_<对手>.json`及同名runtime逐面板日志／recording |

## 有限验证

29项相关回归通过，覆盖实际使用身份与本段／累计步区分、同轮3000步和final的replay一致性、三组仅B方模型不同、双方FPU配方／S128N1预算／配置A、独立sidecar恢复、只读status，以及FPU原生与Python参照一致性。四个真实checkpoint在冻结代码中严格加载，分别完成一个FPU0.3／c_puct1.0的CPU native S128N1根，共512次模拟；没有完整对局或CUDA性能验证。

常规JUnit与原始CPU检查保存在忽略的runtime；紧凑结果、原始记录路径／SHA及复现命令见[verification.json](verification.json)。本任务不调用在线训练／采集入口，不写入分支原始状态、checkpoint或既有评测结果。
