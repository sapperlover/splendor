# 固定λ161924对同池MC70842／原R37：双方S128N1，各3000pair

2026-11-23按用户要求准备两个独立脚本。模型节点和SHA固定，不跟随更新、不重新选点；正式评测未启动，不启动训练或队列。

```bash
# λ161924 vs 同池MC70842，双方S128N1，3000pair
bash scripts/run_lambda161924_vs_mc70842_s128n1_3000pair_20261123.sh

# λ161924 vs 原R37，双方S128N1，3000pair
bash scripts/run_lambda161924_vs_r37_s128n1_3000pair_20261123.sh
```

从仓库根目录执行；脚本使用conda `splendor`、`PYTHONPATH=src`，OMP／MKL／OpenBLAS各1线程。两条命令各执行一组搜索评测，不自动交接或额外执行greedy。每组3000pair／6000局；每pair同一开局seed交换先后手。总计两组6000pair／12000局。

## 模型身份

| 模型 | 固定文件 | SHA256 |
| --- | --- | --- |
| 当前归档搜索λ=0.8，161924步 | `runs/outcome_60k_scalar_lambda_20261120/checkpoints/step-000161924.pt` | `561f055af56719ad7fbfa9779e93955ce139d5651b6a2b9b9dbc7e10b7182b3b` |
| 同池标量MC，70842步 | `runs/outcome_60k_scalar_mc_20261120/checkpoints/step-000070842.pt` | `37cbd0bf0082e530bf303fe56e027b99af63b9f9a7afea15ccf4d236a9c69cec` |
| 原R37 | `runs/entity_attention_r31_10k_p60k_iteration_20261023/rounds/round-005/checkpoints/final.pt` | `611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045` |

λ和MC均为2026-11-20的近期60k池实验：原本地在线8–13轮、归档round_index87–92，Train53980局／4317943位置，Validation6020局／481516位置。双方训练manifest SHA同为`1014247e20077ea6dc37e5ce18bc913e49d78a87e799e8aef6d90acbe7b6d39a`，两个指定节点的完整验证dataset digest也相同。λ训练已完成202404步并选中161924步；MC当前记录172044步、尚未最终选点，本脚本直接固定用户指定的70842步，不要求MC训练完成。

三个checkpoint均严格加载为583682参数的标量`entity_attention_v1`。本次只读取模型、配置及验证身份，不扫描／预热60k训练池，也不生成λ标签。

## 配置和种子

复用现行配置A及同一评测器：CUDA pool、12 search worker、1000活跃局、B500、local microbatch128、owner双槽、batch buckets `[64,128,256,500]`、candidate buckets `[32,48,64]`、等待5ms、shared slots2、FP32/native-cudagraph、reused传输、fail-fast。

**双方S128N1均为每世界128次×1世界＝总128次**，底层`simulations=128`、`determinizations=1`；c_puct1.5、强制阶段折叠、单次模拟最多512状态转移、T0、根噪声0、FP/PTP关闭。当前12贵族规则、400完整回合工程上限和既有确定性循环判平规则保持。和棋半分，10000次整pair bootstrap给出95%区间；完成时复核双方SHA、实际预算、先后手、完整pair数与种子。

两组共享本次新种子：开局55230000–55232999，policy_seed_base55240000，双方search seed55250000，bootstrap seed55260000。每100pair打印进度，每500pair及结束／正常中断更新全量summary；逐局JSONL持续落盘。

## 恢复与产物

同一默认命令恢复：已完成并通过验证的整组直接返回，部分面板自动`--resume`，完整pair跳过、半个pair按原seed重跑。保留原始结果相邻`.json.games/`目录。两组独立锁，拒绝同组重复并发；失败或中断即退出，不自动重试错误局。

两个脚本末尾均可加：

- `prepare`：核对固定模型、生成配置并校验执行源码；不启动对局。
- `status`：只读已有状态，不写报告或启动评测。

文件位置：

| 内容 | 位置 |
| --- | --- |
| 计划、模型／池身份 | [plan.json](plan.json) |
| 正式配置 | `configs/lambda161924_vs_mc_S128N1_3000pair.json`、`configs/lambda161924_vs_r37_S128N1_3000pair.json` |
| 执行源码／native／牌池 | `runtime/lambda161924_search3000_20261123/source`；清单见[source.json](source.json) |
| 本地原始结果及恢复日志 | 同名runtime的`panels/lambda161924_vs_mc.json`／`lambda161924_vs_r37.json`及相邻`.json.games/` |
| 紧凑报告 | [results_mc.md](results_mc.md)、[results_r37.md](results_r37.md)、`summary_mc.json`、`summary_r37.json` |
| 整体调用记录 | 本目录后续`execution_mc.json`／`execution_r37.json` |
| 日志／会话资源 | 同名runtime的逐面板`.log`及`panels/<面板>/recording/` |

当前工作区的新value实现与模型加载实现已纳入本任务实际冻结源码，后续修改不会覆盖该发布。原训练配置／checkpoint／已有冻结发布均保持。

## 有限验证

16项相关回归通过：固定节点与同池身份、两组3000pair／双方预算／配置A／共同seed、各自sidecar恢复、只读status，以及复用评测机制的预算与先后手检查。三个真实模型严格加载，分别完成标量根前向及一个CPU native S128N1根，合计384次模拟；没有完整棋力对局或CUDA性能验证。常规JUnit与原始CPU检查仅保存在runtime，紧凑校验及命令见[verification.json](verification.json)。
