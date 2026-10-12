# 从5000步接入自迭代的连续混合训练

2026-10-05。入口为 `python -m splendor.hybrid_train`，原正式配置为 `configs/entity_attention_v1/hybrid_static_graph.json`。原run现已完成9轮、累计9500步；第三轮首次评测失败的历史现场与修复保留在下文。第九轮对固定theta_5000的S1024/N1评测为69胜30负1和、得分69.5%，100局均有效，依据见[原始汇总及有限S/N筛选](../../artifacts/entity_attention_search_screen_20261005/README.md)。新增[从第九轮创建N2小轮次分支](entity_attention_hybrid_fork_zh-CN.md)入口，保留模型/优化器/RNG并逐渐替换旧数据；原run冻结配置保持不变。

2026-10-09留出复核见[固定留出与9代KL矩阵](../../artifacts/entity_attention_hybrid_kl_trend_20261009/README.md)：每代固定2048局面，覆盖9代而非全量留出。θ9500相对θ5000在九代均下降；相对各代训练前为八代下降、一代上升，本轮500步则五降四升。第一代固定在线集8500步为已测父run最低点，之后回升；不能将累计收益解释为每轮稳定改善或500步优于100步，也不能由旧目标KL回升认定棋力退化。

2026-10-09新增[离线9500对照](../../artifacts/entity_attention_hybrid_offline9500_20261009/README.md)：从离线 step-9000 确定性重放出 9500 步纯离线模型（重放 10000 与原 checkpoint 逐张量一致），与 hybrid 第3/6/9轮用同批200个种子对 θ5000 测 S1024/N1 棋力。离线9500 67.38%、hybrid 第3/6/9轮 58.88/66.75/74.38%；第6轮与离线9500不可区分（−0.62pp [−7.38,+6.12]），第9轮高出 +7.00pp [+0.62,+13.25]——前6轮的相对收益可被纯离线继续训练解释，第6–9轮存在额外于纯离线的增量。

## 流程和采集执行配置

新模型随机初始化，前5000个 optimizer 更新只使用原 `az_theta5_10x1000_seed1` 十轮10000局的 train split。随后每轮用最新模型采集1000局，再更新500步；权重、AdamW、全局步数和训练采样随机状态跨阶段连续。第5000步的固定对手保存为 `checkpoints/theta_5000.pt`。

原离线池与最近6000个完整在线对局分别索引和缓存，原split保持不变，在线按整局留出10%。在线窗口6000指split前完整局数。原数据不被在线窗口淘汰；第16轮起其训练配额为0。

| 在线轮次（从1计） | 每batch的离线 / 在线样本数 |
| --- | --- |
| 1 | 224 / 32 |
| 2 | 192 / 64 |
| 3 | 160 / 96 |
| 4～5 | 128 / 128 |
| 6～7 | 96 / 160 |
| 8～10 | 64 / 192 |
| 11～15 | 32 / 224 |
| 16及以后 | 0 / 256 |

每轮直接采用最新模型；验证及每3轮一次、50 pairs的固定theta_5000 MCTS对战只用于诊断。没有胜率接纳、回滚或验证误差早停。工程错误仍明确失败。

采集沿用S1024/N1、W8/A1000/B500、microbatch64，使用：

- `execution.shared_slots=2`：每个CPU worker两个共享请求槽。
- `execution.pipeline_slots=1`：owner单缓冲，与共享请求槽独立。
- `execution.inference_mode=compiled`、`compile_mode=native-cudagraph`。
- 固定batch档位 `[64,256,500]`、candidate档位 `[16,32,48,64]`、FP32、`actor_projection=false`。

以上仅作用于采集执行。learner保持batch256、AdamW lr=1e-4、weight_decay=1e-4、policy/value权重1/1、clip1；评测使用现有native_batched路径。固定地址Graph的已有有限验证范围见[执行报告](../reports/entity_attention_execution_speed_zh-CN.md)，不能把短窗口性能或本次CPU验证当成混合训练棋力收益。

## 从头启动

从仓库根目录执行，输出目录需未用于其他run：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_static_graph.json \
  --output runs/entity_attention_hybrid_static_graph_seed1 \
  --forever
```

`--forever`会在完成离线阶段后持续自迭代，第16轮后维持纯在线。通过Ctrl+C结束当前会话，原始数据和已提交checkpoint保留。恢复仍使用同一冻结配置：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_static_graph.json \
  --output runs/entity_attention_hybrid_static_graph_seed1 \
  --resume --forever
```

中断时未提交的更新会从最后checkpoint重做；不能保证保留中断瞬间内存中的最后一个更新。采集通过既有manifest/完整对局提交机制恢复。

### 第三轮末评测失败的修复与恢复（2026-10-05）

以下是首次失败时的现场与当时恢复步骤；现已恢复有效评测并继续到第九轮，不代表当前仍停在6000/6500步。

用户run `runs/entity_attention_hybrid_static_graph_seed1` 的第三轮1000局采集与500步更新已完成，`rounds/round-002/checkpoints/final.pt` 保存累计6500步及连续AdamW状态。首次评测的100局均为 `FailOnRecompileLimitHit`，valid_games=0；原始汇总为 `rounds/round-002/evaluation/mcts_vs_theta5000.json`，不能把其胜率字段当作棋力结果。状态仍记录前两轮完成、最新6000步，是因为该轮尚未通过工程验收，不代表6500步checkpoint丢失。

根因是 `CompiledForward` 对 `functools.partial` 调用 `torch.compile`，不同模型、候选宽度及policy/value入口共用PyTorch `external_utils.inner` 的Dynamo缓存；先前采集的静态图与动态批量评测相互占用缓存，默认8次重编译上限被触发。修复为每个编译图创建独立函数及code object，不再提高共享缓存上限，仍保持 `fail_fast`。

混合入口现在以 `require_scored_pairs=True` 恢复评测：只跳过两局均已有效计分的pair，工程失败或未完成pair按原seed重做。重做前将原pair日志和旧summary保存在 `mcts_vs_theta5000.json.games/retries/<时间戳>/`，已有效pair不重做。无需更改冻结配置，也无需删除或搬走原始结果。

使用上面的 `--resume --forever` 命令会复用第三轮已提交的完整对局，从轮内500步checkpoint恢复，不再训练这500步；重算有限验证并重新评测，评测完成后采用6500步模型采集第四轮。若只希望恢复到第三轮验收完成，用 `--resume --online-rounds 3` 替代 `--resume --forever`。这是工程错误恢复，不引入胜率接纳或回滚。

有限会话可用 `--online-rounds 3` 替代 `--forever`；这是整个run在线轮数的绝对目标。之后 `--resume --online-rounds 16`会继续到第16轮，不额外从头再跑16轮；也可改用 `--resume --forever`。这些会话预算不属于冻结算法配置，继续时不重置日程。

配置检查不训练、不采集：

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_static_graph.json --check-config
```

## 产物和恢复身份

- `config.json`：冻结模型、learner、来源、日程、探索、采集执行和评测参数。
- `hybrid_state.json`：已完成的绝对在线轮号、固定对手和下一轮最新模型身份。
- `offline/`：原离线冻结数据、引用、训练指标、验证曲线和checkpoint。
- `rounds/round-NNN/`：selfplay归档、训练快照、initial/step/final checkpoint、训练指标、训练前后验证和source_usage。
- `fixed_online_dataset.json`、`fixed_online_validation.json`：第一在线轮留出数据的固定跨轮诊断引用，即使离开训练窗口仍保留。
- 每轮 `usage/offline` 与 `usage/online`：显式隔离的旧格式引用账本，来源映射写入header和汇总描述文件，旧新round/game同号不会混淆。不要合并后忽略来源。
- `source/<session>/`：命令、源码摘要、dirty patch等；`round_metrics.jsonl`：完成轮的来源配额、checkpoint与验证。

正式配置的replay缓存总上限为**16 GiB**（17179869184字节）：纯离线阶段单池上限16 GiB，混合阶段离线/在线分别8 GiB，互不共享键空间。均为按需加载的LRU上限，不预分配满额；进程其他内存、批次临时数组、pinned/shared缓冲不计入这个上限。

2026-10-05按用户要求由总4 GiB提高到16 GiB，修改时正式run尚未创建。原离线10000局的ZIP目录中数组大小上界为5.918 GiB；已有attention在线3000局约1.917 GiB，6000局工作集按相同平均局长估算约3.83 GiB。8 GiB/池给当前工作集留出余量；未来局长变化仍以实际cache驻留、hit/miss和加载耗时判断。当前机器主存约31 GiB、MemAvailable约23 GiB，取值是本机运行准备，不代表任意机器适用，也不是混合训练整轮加速实测。

冻结配置中的缓存容量仍保留，已启动后不要直接编辑原配置冒充resume。2026-10-06起hybrid支持独立`--replay-cache-mib`执行覆盖，每会话记录实际总预算；本机当前小轮次分支建议8192 MiB。在线解码缓存随活动窗口淘汰，保留固定验证；离线配额归零时只保留固定离线验证缓存；采集/评测边界释放进程内编译状态。此前16 GiB建议没有解决跨轮缓存滞留，第31轮RAM OOM及修复证据见[专题](../../artifacts/entity_attention_memory_window_20261006/README.md)。checkpoint恢复仍核验冻结配置、训练快照、数据身份、阶段和轮内进度，采样和训练计划不因此改变。

2026-10-06起，hybrid恢复仅允许调整采集`selfplay.execution.max_batch_size`和`batch_buckets`，其他配置仍冻结；已有采集manifest的轮次保留原尺寸，新采集轮采用请求尺寸。现行200局/100步S1024/N1分支及Graph调整命令见[专题](entity_attention_hybrid_s1024_n1_zh-CN.md)。原1000局配置保留B[64,256,500]，未随小轮次预设修改。

## 有限验证与范围

33项相关CPU回归通过（hybrid5、learner10、原loop/replay18），覆盖配额、零配额采样、旧新同号对局隔离、优化器连续、逐参数/状态/随机流一致的中断恢复、最新模型直接接续、继续进入纯在线、固定在线验证集离开训练窗口后仍可读取，以及评估sidecar恢复。测试入口：

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_hybrid_training.py tests/test_alphazero_training.py \
  tests/test_alphazero_loop.py tests/test_alphazero_replay.py
```

独立CPU smoke配置 `hybrid_cpu_smoke.json` 使用原第一轮前6局、离线4步、每轮2新局/2更新、batch8、S2/N1、batched_cpu eager。其在线validation明确设为0，关闭固定在线诊断和棋力评测；不能代替正式配置的GPU验证。实际短测产物位于 `runtime/entity_attention_hybrid_cpu_smoke_20261005`。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_cpu_smoke.json \
  --output runtime/entity_attention_hybrid_cpu_smoke_new \
  --online-rounds 2
```

CPU smoke验证调用链，不比较胜率。以下记录首次三轮时的编译故障验证；当前已完成9轮并取得有效MCTS面板，新分支及最新状态见本页开头。编译缓存隔离、多模型64→50尾批、失败pair保留重做、训练后评测失败恢复等22项当时的相关CPU回归通过：

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_inference_modes.py tests/test_hybrid_training.py \
  tests/test_evaluation_3a.py::test_fixed_evaluation_retries_only_unscored_pairs_and_preserves_failures \
  tests/test_ladder_search.py::test_discard_pair_removes_the_partial_file
```

2026-10-05有限GPU smoke使用真实theta_5000与第三轮6500步checkpoint：先预热单B64的固定地址CUDA Graph，随后不reset Dynamo，用native_batched compiled路径执行2 pairs、每次搜索2模拟。4局均有效、无工程错误，两模型有效模式均compiled、无编译失败或编译失败回退；评测阶段18.415秒（不含此前Graph预热），不能替代正式S1024棋力面板。配置、种子、源码身份、脚本与原始汇总保存于 `runtime/hybrid_eval_recompile_fix_20261005/`，脚本仅为短验证；不操作用户正式run的summary、pair日志或训练进程。
