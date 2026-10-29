# 固定配对评测与结果解释

2026-10-29新增[固定140221步的R37初值／104k／lr2e-5对R37评测](../../artifacts/r37_lr2e5_step140221_vs_r37_10k_20261029/README.md)：用户明确选定约14万步节点，固定SHA、不依赖训练结束或最终selection；先greedy10000pair，再双方S128N1（各总128）10000pair。沿用此前五模型10k种子与配置A，严格串行、完整pair恢复；未启动正式评测。

## Git中的评测交付边界（2026-10-27）

Git保留评测入口、冻结配置、模型／对手SHA、规则指纹、预算、种子范围、完整pair数、胜和负、得分与整pair置信区间、错误／截断计数、耗时及结论。组间比较保留聚合配对差及计算方法。小型精选表和最终图可以纳管。

完整`pairs`、逐局执行明细、逐root结果、诊断NPZ、原始CSV和资源采样默认留在本地或外部归档；新实验优先将原始输出写`runtime/`。历史冻结入口及正在运行的实验可继续写原路径，按忽略规则保存，不为Git整理改变恢复语义。原件仍是复算和恢复的依据，Git摘要登记其SHA、字节数及真实位置，不能把投影摘要传给原始结果读取器。

若结果JSON同时含摘要和明细，另存紧凑摘要，原件保持字节。2026-10-27的[历史产物整理](../../artifacts/repository_git_cleanup_20261027/README.md)提供显式标记省略字段的摘要与原件身份；本次没有重评、改写原始测量或替其补充新结论。当前运行状态以实际产物为准，下方带日期的发布／准备记录保留当时语义。

2026-10-27[五个104k离线best的大规模对R37复核](../../artifacts/offline5_vs_r37_10k_20261027/README.md)十组均已完成，各模型greedy及双方S128N1（总128）各10000 pair，共20万局；原选点、共同新种子、greedy CPU8及搜索配置A保持。得分、区间和与原S1024N1面板的区别见[完成汇总](../reports/auxiliary_offline_results_20261027_zh-CN.md)。

2026-10-09新增[固定256局面的checkpoint决策诊断](decision_probe_zh-CN.md)：独立真实对局、每局一个根节点、前中期／后期各128；每个checkpoint用固定搜索种子记录首选动作、访问分布JSD、高访问动作排序、根policy/value漂移与搜索波动。入口为`python -m splendor.decision_probe evaluate`，支持逐根恢复；用于快速观察变化，不产生胜率或棋力排名。

旧θ5的[历史每轮／阶段评测计划](../archive/next_az_training_zh-CN.md)分开DEV与预留确认输入，源run后续已运行；各面板完成状态以原件为准。`scripts/run_recorded_evaluation.py`复用本入口并记录单面板会话／资源，不调度训练或自动选模。

从仓库根目录执行，使用 Conda `splendor`、`PYTHONPATH=src`。基线评测入口为 `splendor.evaluate`，冻结 checkpoint/search 的统一入口为 `splendor.evaluation`。配置种类包括 `greedy_checkpoint`、`search_checkpoint`（固定深度）和 `mcts_checkpoint`（PUCT）；两侧模型可不同。

## 配对与计分

每个 pair 是一个 game seed 的两局先后手交换；统计得分率为 `(wins + 0.5 * draws) / valid_games`。bootstrap 以 pair 为单位，不将两局当成独立样本。报告必须标明双方 checkpoint 内容身份、动作模式、规则/特征/后端、搜索预算、seed、完整 pairs 和无效局数。

`passed=true` 表示评测工程完成，不表示棋力提升。未完成、异常、工程截断结果不能用于完整面板排名。确定性 policy-cycle 判平属于评测裁决，不是引擎正式终局；只用于固定确定性策略，不自动用于有根噪声/温度采样的采集。DEV 与 FINAL 的区别由预留和选模使用方式决定，单独种子域不自动成为无偏 FINAL 验收。

## 当前采集与评测配置（2026-10-23）

当前在线分支另有[六轮全量固定留出诊断](../../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/fixed_holdout_20261023/results.md)：首段R32–R37使用R26–R31的2412个原始validation对局，39845个有效policy位置、192218个value位置；此后每六轮自动改用此前六轮留出。每轮0／100／250／500／1000／2000／3000步全量前向，与棋力评测独立；保留原2048位置小面板，但它仅轮内固定、跨轮重抽。实现／回补／恢复说明见下节。

按用户要求，**新建日常搜索评测、当前在线分支的后续搜索评测统一采用配置A**。这是已完成500 pair正式对战的可用配置，不宣称全局最优，也不保证每次冷启动都在固定分钟数内完成。历史单槽面板按原配置解释，不回写结果或补跑。

| 场景 | 规模／节奏 | 策略与设备 | 配置入口 |
| --- | --- | --- | --- |
| 在线采集 | 每轮10000个完整对局 | 单教师自对弈；75% S128N1（总128），25% S256N4（总1024）；CUDA pool | [当前在线配置](../../configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json)的`selfplay` |
| 日常greedy | 默认1000 pair／2000局，按需独立执行 | 双方确定性greedy，零搜索，CPU8 worker | [greedy预设](../../configs/entity_attention_v1/evaluation_greedy_r43.json) |
| 日常搜索 | 默认500 pair／1000局起步，按需独立执行 | 双方S1024N1（总1024），GPU执行配置A | [搜索A预设](../../configs/entity_attention_v1/evaluation_search_a_r43.json) |
| 在线greedy | 每个本分支轮次结束后，1000 pair | 本轮final对固定R43，CPU8 worker，零搜索 | 当前在线配置的`greedy_evaluation` |
| 在线搜索 | 本分支第3、6、9……轮结束后，500 pair | 本轮final对固定R43，双方S1024N1（总1024），配置A | 当前在线配置的`search_evaluation` |

本分支局部第1轮对应谱系R32，第6轮对应R37；评测节奏按局部轮号计算。日常预设中的示例候选**固定为R37**，不自动追踪最新checkpoint；两个模板均以原主线R43为示例对手。换候选／对手时先复制到新配置文件，明确修改双方`checkpoint`和`name`，使用新的输出路径。原R43训练于10贵族时期，当前评测使用12贵族；共同对手得分不能直接归因于扩池或换学习率。

### 在线全量固定留出

当前保存周期为500步，独立强制保留七个诊断节点；0／100／250／500／1000／2000／3000步的评测频率保持。两份抽样`.bin`和两份`.steps.jsonl`默认停写，每步训练指标仍保存主日志；[记录与恢复语义](training_artifacts_zh-CN.md)。原`source_six_round_holdout_20261023`保留历史身份，当前发布见[交付记录](../../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/recording_checkpoint_20261023/README.md)。

当前配置为`six_round_holdout`：`manifest`以路径＋SHA指定第一段面板，`batch_size=512`、`block_rounds=6`。按用户更正实现**每六轮固定使用此前六轮**，段内不换数据，换段时自动冻结新面板：

| 评估模型轮 | 本分支局部轮 | 完整留出来源 |
| --- | --- | --- |
| R32–R37 | 1–6 | R26–R31，2412局／39845 policy／192218 value |
| R38–R43 | 7–12 | R32–R37，6092局／101425 policy／485932 value，已冻结 |
| R44–R49 | 13–18 | R38–R43，轮次完成后自动冻结 |
| 后续 | 每六轮一段 | 使用上一段六轮的全部原始validation |

新面板从前六轮`collection_index.json`提取完整原始validation，不依赖当前活动池，包含已淘汰的留出局。面板内保留原标签与权重，policy只统计正权重，value统计全部位置；每轮0／100／250／500／1000／2000／3000步记录CE、KL、MSE、基线、相关性和逐局充分统计量。不参与训练、教师接纳、早停或选模。第一段R32–R37的42项旧结果保持原字节和身份，永久固定R26–R31的初版调度已被本节替代。

第一段继续使用`rounds/round-NNN/fixed_holdout/step-XXXXXXXXX.json/.npz`；后续各段独立保存于`fixed_holdout/blocks/iterations-007-012/`等目录，其`manifest.json`冻结前六轮来源与档案SHA，结果在该段目录的`rounds/round-NNN/fixed_holdout/`。运行级`fixed_holdout/schedule.json`冻结分段规则。各段指标只在同一面板内直接比较；换段后不可把绝对KL/MSE接成同一曲线。

driver在节点回调内评估当前权重，保留model模式、TF32设置及训练RNG；逐轮检查与Train无交集。恢复按各节点所属段补齐，校验面板／checkpoint／评测器／逐局统计SHA，已完成结果复用。只保留当前段解码面板在CPU，当前batch进入GPU；这部分内存独立于16GiB训练回放缓存，后者仍非RSS上限。

启动脚本已改用`runtime/entity_attention_r31_10k_p60k_iteration_20261023/source_recording_checkpoint_20261023`，原`source/`及初版`source_fixed_holdout_20261023/`保持冻结。当前无训练进程，本次未续训；下次用原启动命令即按六轮分段执行。[交付与验证](../../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/six_round_holdout_20261023/README.md)。

只补齐本分支已完成节点、不采集或训练，使用新版分段入口（原`splendor.fixed_holdout`仅对应历史静态配置）：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=runtime/entity_attention_r31_10k_p60k_iteration_20261023/source_recording_checkpoint_20261023/src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m splendor.six_round_holdout \
  --config configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json \
  --output runs/entity_attention_r31_10k_p60k_iteration_20261023
```

该入口与driver共用`.iteration.lock`，不能与同一run的在线进程并行写入，同命令恢复。R32–R37的原始逐局统计和区间见[首段回补报告](../../artifacts/entity_attention_r31_10k_p60k_iteration_20261023/fixed_holdout_20261023/results.md)。R20–R31补评因冷checkpoint在未挂载T7归档内，用户已暂缓，此次未执行。

### GPU执行参数：采集与搜索A

| 字段 | 当前在线采集 | 日常／在线搜索A |
| --- | --- | --- |
| `backend`／推理模式 | `native_cuda_pool`／compiled／native-cudagraph | 相同 |
| `precision`／`actor_projection` | float32／false | 相同 |
| `search_workers` | 12 | 12 |
| `total_active_games` | **2000** | **1000** |
| `max_batch_size`／`local_microbatch` | 500／128 | 500／128 |
| `pipeline_slots` | 2 | **2** |
| `batch_buckets` | [16,32,64,128,192,256,384,500] | **[64,128,256,500]** |
| `candidate_buckets` | [16,32,48,64] | **[32,48,64]** |
| `max_batch_wait_ms` | 5 | 5 |
| `shared_slots`／`shared_candidate_width` | 2／256 | 2／256 |
| `torch_threads`／`native_transport` | 1／reused | 1／reused |
| `compile_failure_policy` | fail_fast | fail_fast |

这里的两槽是owner流水线，独立于每worker共享请求槽`shared_slots=2`。GPU pool配置的外层`workers=1`不代表单搜索worker，实际并行度由`search_workers=12`控制。活跃上限还受待执行局数约束：500 pair最多1000局并发，不能把采集active2000的吞吐直接外推到该评测。纯greedy的CPU8 worker路径不使用这些GPU batch／槽数参数；两侧checkpoint均零次模拟。

采集高预算根噪声比例0.25、Dirichlet alpha0.3；低预算关闭根噪声、不提供policy监督，value使用全部预算样本。前10个完成回合温度1，之后0；FP/PTP关闭。采集保留自身配置，不套用搜索A的活跃数或桶列表。当前在线训练保持60k池、3000更新／轮、lr2e-5、policy/value各B256、16GiB按需compact缓存；后者是缓存容量，不是进程RSS上限。

### 评测协议与种子

正式搜索双方全程S1024N1（每世界1024次、1世界，总1024），`c_puct=1.5`、强制阶段折叠开启、每次模拟最多512次状态转移、根噪声0、温度0、FP/PTP关闭。采集的低预算随机化、噪声和温度采样不进入评测。greedy使用根policy确定性argmax，不读取搜索结果作为动作。

| 项目 | 日常／在线greedy | 日常／在线搜索 |
| --- | --- | --- |
| 默认开局seed | 37000000–37000999 | 37100000–37100499 |
| `policy_seed_base` | 37010000 | 37110000 |
| 两侧`search_config.seed` | 不适用 | 37130000 |
| 整pair bootstrap | 10000次，seed37020000 | 10000次，seed37120000 |
| `max_completed_turns` | 400 | 400 |
| `adjudicate_deterministic_policy_cycles` | true | true |

同一pair使用相同开局seed交换先后手，和棋半分；当前规范目录为12贵族，每局抽3张，保存catalog指纹。上述开局库已被反复查看，属于DEV复核，不自动成为独立FINAL；确认实验需要另行预留种子。用户指定对手、预算、规模或种子时以明确要求为准。

以下命令会启动正式棋力评测；本次配置整理不自动执行。命令从仓库根目录运行，示例候选是R37，输出目录应为新的：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u scripts/run_recorded_evaluation.py \
  --config configs/entity_attention_v1/evaluation_greedy_r43.json \
  --output runtime/daily_evaluation/R37_vs_R43_greedy/evaluation.json

conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u scripts/run_recorded_evaluation.py \
  --config configs/entity_attention_v1/evaluation_search_a_r43.json \
  --output runtime/daily_evaluation/R37_vs_R43_search_A/evaluation.json
```

恢复使用同一配置和输出路径，追加`--resume`，并保留相邻的`.games/`日志。`run_recorded_evaluation.py`记录调用、代码身份及资源；正式报告仍需保存双方checkpoint SHA、原始结果、配置、实际命令与种子。可用`FixedEvaluationConfig.from_dict`只解析模板，不能把执行完整评测命令当成只读检查。

### 执行器与在线生效边界

`execution.backend="native_cuda_pool"`复用`ExecutionConfig`与`multiworker_gpu.collect`。真实对局由checked runner推进，`engine_mode=fast`不替换评测规则检查。单GPU owner加载双方模型；worker不加载模型，不创建CUDA上下文。每个请求携带checkpoint SHA，只与同模型、同policy/value类型及宽度类别的请求合批，transfer/图缓存也按模型隔离；每行只推理当前行动方。

该后端支持native或native_chance MCTS与`greedy_checkpoint`任意配对。2026-10-21新增`NativeGreedyPolicy`通过root-only native作业共享GPU owner：只前向根logit并argmax（并列取最小候选下标），不创建搜索世界、不展开树，模拟/确定化/搜索状态转移计数均为0；可与同模型搜索根合批，统计包含greedy前向。已有`native_batched`仍使用其原eager greedy执行路径。`device=cpu`提供共享布局的显式CPU复核路径；native-cudagraph正式配置使用CUDA。pool的推理配置由`execution`统一确定；`policy_a/b.inference`若显式指定，必须与之相符，冲突会报错。worker并发写pair日志加文件锁；完整pair支持直接恢复，部分pair沿用现有归档后重跑规则。工程与实际模型验证、正式S256N4对greedy面板见[实验记录](../../artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/README.md)。

结果`execution_stats`记录聚合与逐模型推理行数、物理batch/平均batch、编译/图捕获、GPU峰值显存、worker供给、owner等待/整理/流水线重叠、逐局模拟数等。rule/sidecar时间是多worker累计时间，不能直接除以墙钟当成独占占比。双模型分流仍会拆分供给，也多占图缓存显存；仅短验证通过不能宣称达到采集吞吐。

持续迭代允许调整**搜索评测的pair数及推理执行参数**，不改变训练、采集和搜索协议。本次只更新外部源配置中的搜索`execution`；当前driver在下一轮边界重读它，随后新创建的面板使用A。正在进行的轮次已读入的模板、已有`search_config.json`／`greedy_config.json`继续使用各自冻结参数；不热改已准备面板。无需为这次执行参数更新重启进程，恢复仍用原脚本；`runs/.../config.json`、旧配置快照和结果保持历史身份。旧分支源JSON及历史200 pair面板不批量改写。

配置A来自[三组正式搜索评测](../../artifacts/eval_speed_20261023/README.md)中的已完成执行参数。80k候选实测500 pair约33.4分钟；旧42.3分钟面板含约526秒编译，本次对应约17秒，模型／桶／热状态亦不同，不能把墙钟差全部归因于A。GPU温度、时钟与合批等待影响实际耗时；不再以寻找最优配置为当前任务。60秒吞吐窗口用于独立性能筛选时，必须预热实际两个模型、保留相同并发和搜索预算，并单独报告编译与尾段。

## 可执行入口

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.evaluate \
  --player-a score --player-b development --seed-start 0 --seed-count 100
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.evaluation \
  --config configs/mcts_vs_greedy.json --output runtime/mcts_example/evaluation.json
# 相同配置、输出路径追加 --resume；保留相邻 .games/ sidecar
```

四组 sweep 的配置位于 `configs/mcts_50pair_sweep/`，输出位于 `artifacts/mcts_50pair_sweep/`；脚本失败会停止后续组。

```bash
# 仅检查配置、checkpoint 和命令，不开始对局
conda run -n splendor env PYTHONPATH=src python scripts/run_mcts_50pair_sweep.py --dry-run
# 真正继续历史 sweep 时使用 --resume；当前并非四组全部完成
```

实际 sweep 状态见 [结果报告](../reports/mcts_sweep_zh-CN.md)。AZ 扩展面板入口为 `scripts/run_az_generation_evals.py`；默认 θ1–θ4，θ1–θ5 要显式传 `--rounds 1 2 3 4 5`。summary 默认纳管，生成配置和 sidecar 忽略，见 [AZ 指南](alphazero_zh-CN.md)。

新实验使用新输出目录；恢复既有实验必须保持冻结配置和 checkpoint 身份。5C Quick/Formal/Promotion/FINAL 的种子域与晋升阈值见 [PPO 实验协议](../specs/ppo_training_protocol_zh-CN.md)。

2026-10-21起，默认评测使用完整12张贵族池（新增蓝绿黑333、红白44，双人仍抽3张）。固定配对manifest和summary额外保存`catalog_fingerprint`，恢复时必须匹配；即使checkpoint、开局seed和搜索参数相同，也不能把旧10张池的pair与新12张池拼接。此前已完成面板保留原规则身份，使用旧代码与对应数据复现；新规则正式对战另开输出目录。

单 worker 的有限 native 时间归因入口为 [`profile_speed_pipeline.py evaluation`](../../scripts/profile_speed_pipeline.py)，示例、计时边界及既有结果见 [分项报告](../reports/pipeline_profile_zh-CN.md)。它调用同一个固定配对 runner，保留 sidecar 和状态语义；不是新增跨局批量评测，也不能用 1 pair 宣称棋力或完整面板吞吐。

## 跨局 native 批量入口

新配置可增加 `execution`：`backend=native_batched`、`active_games`、`max_batch_size`、`candidate_buckets`、`inference_mode` 和 `actor_projection`；`workers=1` 指唯一推理 owner，`device` 可选 CPU/CUDA。示例为 `configs/speed_2026-10-03/evaluation_optimized.json`。固定θ5/θ4各50 pairs真实对照已完成：CPU eager8 workers为682.17秒，唯一CUDA owner active100/B64为207.21秒，100局终态/结果/长度一致；详见[完整性能专题](../reports/end_to_end_speed_optimization_zh-CN.md)。这是整条执行路径收益，非棋力提升证据。

此路径接受 `native`／`native_chance` 的 `mcts_checkpoint`，以及 `greedy_checkpoint`，可混合对打。greedy沿用`GreedyActorCriticPolicy`，以eager单请求forward的logits argmax行动，零搜索模拟；native批次只推进搜索一侧的作业。相同checkpoint和推理配置复用同一冻结模型。`execution_stats.greedy_seconds`／`greedy_decisions`单独记录裸网开销，搜索inference计数不包含这些greedy请求；greedy的policy搜索stats为空。

2026-10-08修复`greedy_checkpoint` JSON读取丢弃`inference`的问题：两侧显式填写相同推理配置后可共享同一evaluator，greedy选招仍沿用eager。CUDA搜索可在两侧policy的`inference`中填写`mode=compiled`、`compile_mode=native-cudagraph`、`batch_buckets=[16,32,64,128,200]`、`native_transport=reused`、`compile_failure_policy=fail_fast`，并用`execution.active_games=200`、`max_batch_size=200`；这些字段不是新增搜索预算。配置和短窗依据见[分阶段加速方案](../../artifacts/entity_attention_stage_search_20261008/README.md)。其单模型200固定根热身后比eager64快1.84×，含冷编译的小面板可能更慢；不能当作完整双模型棋力赛耗时保证。该入口不设置RAM硬上限，正式旧面板必须保持原执行配置，新的执行方案使用新的输出目录。

每局各自保留 policy seed，按 checkpoint 内容身份和推理配置隔离 evaluator/批次；原 checked runner 以协程推进真实规则、循环裁决、异常和截断，greedy异常也经相同规则runner记录。所有就绪局已经提供当前可用叶请求，因此 `max_batch_wait_ms` 必须为 0，尾批立即执行。恢复仍以完整 pair 为单位，半个 pair 使用原种子重跑；执行配置参与 manifest 身份，不允许与旧 CPU 路径拼接。

AZ 的 `evaluation.mcts_execution` 接通同一路径；需要 `mcts_backend=native`、`workers=1`，不影响旧配置的默认 CPU 多进程入口。两个配置应使用不同运行目录。

## 本地人机网页

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.web_app
# 使用本地已有PPO模型作为固定深度搜索对手
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.web_app \
  --search-checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --search-depth 6 --search-top-k 8 --search-determinizations 1 \
  --search-seed 9200000 --search-leaf-value-clip 1.0 --search-fold-forced-phases
```

打开 `http://127.0.0.1:8765`。模型在CPU运行，checkpoint需要本地提供；网页只接收当前玩家视图和合法动作，规则在服务端执行。

policy辅助在线第9轮的人机对局（本分支第9轮，对应 `round-008`；现有S1024N1搜索评测对raw R37得分62.00%，为该分支最高点估计，尚未证明显著强于第12轮）。下面固定此权重，使用S1024N4，即每世界1024次、4世界、总4096次：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m splendor.web_app --port 8765 \
  --mcts-checkpoint runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-008/checkpoints/final.pt \
  --mcts-label 'policy在线第9轮 · S1024N4' \
  --mcts-simulations 4096 --mcts-determinizations 4 --mcts-seed 0
```

配置了 MCTS 时，网页默认选中该对手。开局可修改 S（总模拟次数）和 N（隐藏局面采样数）；S 在 N 个独立搜索树间分配，须满足正整数且 N≤S。对局右侧的“AI 搜索预算”可随时修改，点击“应用到后续 AI 行动”后从下一次 AI 决策生效，棋盘和模型权重不变。正在进行的搜索需等待完成，不能中途改预算。输入框提供128、256、512、1024、2048、4096等 S 建议，也可自行填写其他正整数。

这里使用实验相同的 native 独立局面 MCTS（默认 `c_puct=1.5`、折叠强制阶段、单次模拟最多512次状态转移，选根访问量最大动作，无根噪声），与上面的固定深度 top-k 搜索入口分别配置。网页推理固定 CPU、PyTorch 单线程，不占用训练 CUDA 显存，可和训练同时运行；提高搜索预算会延长响应时间，也会争用部分 CPU。模型固定加载，不随后台教师升级而切换。原生扩展依赖见[原生后端指南](native_search_zh-CN.md)。

预算 API：创建 `POST /api/games` 时可传 `searchBudget: {simulations: 1024, determinizations: 1}`；已有 MCTS 对局通过 `POST /api/games/<gameId>/search-budget` 传相同对象。返回的 `searchBudget` 与 `opponentLabel` 显示实际生效值；其他对手不接受 MCTS 预算。

上述命令打开 `http://127.0.0.1:8765`，可用 `--port` 指定其他空闲端口。训练进程独立运行。

2026-10-29起，新启动的网页服务在开局、每次用户操作及AI回应完成后、悔棋后，自动将当前分支的完整状态与回放原子覆盖保存至 `runtime/web_replays/<gameId>.json`；终局也保留。每盘一个文件，悔棋后存档随当前分支回退，不额外保留已撤销的分支。用 `--replay-dir <目录>` 可修改保存位置，保存失败会直接报错。该目录已在现有 `runtime/` 忽略范围内，不进Git。回放包含完整状态和暗牌，用于本地复盘；浏览器仍只接收当前玩家可见信息。服务重启后不会自动恢复网页会话，但可通过 `GameEngine.restore(path)` 读取存档，以 `replay_game(engine.replay_record)` 逐步校验。

更早启动的服务仍只持有内存记录，不能通过修改源码让旧实例自动保存。需要保留旧局时先不要重启：使用 [`scripts/export_running_web_game.py`](../../scripts/export_running_web_game.py) 指定旧服务PID、网页 `gameId` 和新输出文件。该工具需要系统允许调试目标进程（通常由用户在终端用 `sudo /usr/bin/python3 scripts/export_running_web_game.py --pid <PID> --game-id <gameId> --output <绝对路径>` 执行）。它短暂附加GDB，排入Python待执行导出任务后立即断开，由原进程自己调用既有存档方法，不改棋盘或模型；旁边的 `.export-status.json` 记录成功与否。已在Python3.12测试服务上验证，导出回放可逐步复现原终局；导出后应再次校验实际文件。当前网页尚无下载或加载存档按钮。

顶部的局面评估条直接用当前对手模型的 critic 评估**你的 Observation**，不额外运行搜索。百分比为 `(clip(value, -1, 1) + 1) / 2`，表示胜计1、平计0.5、负计0的估计得分率，并非经过校准的纯胜率。悬停可查看原始 value；未结束时使用模型值，正式终局改为实际结果。固定启发式对手没有模型评估条。

顶部及终局弹窗提供“悔棋”：撤回上一次你的操作及随后全部 AI 回应；买牌、支付、归还等子步骤逐步撤销。支持连续悔棋，恢复原牌堆／随机状态、合法候选和行动记录，当前搜索预算继续保留。`POST /api/games/<gameId>/undo` 需传当前 `revision`；回退仍递增 revision，过期点击不会重新生效。

支付方案、归还宝石和贵族选择显示在页面顶部临时操作栏；只在相应阶段出现，完成后移除，不预留栏位。主行动继续直接操作市场卡牌、保留牌和宝石堆。
