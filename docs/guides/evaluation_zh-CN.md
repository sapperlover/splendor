# 固定配对评测与结果解释

2026-10-09新增[固定256局面的checkpoint决策诊断](decision_probe_zh-CN.md)：独立真实对局、每局一个根节点、前中期／后期各128；每个checkpoint用固定搜索种子记录首选动作、访问分布JSD、高访问动作排序、根policy/value漂移与搜索波动。入口为`python -m splendor.decision_probe evaluate`，支持逐根恢复；用于快速观察变化，不产生胜率或棋力排名。

旧θ5的[历史每轮／阶段评测计划](../archive/next_az_training_zh-CN.md)分开DEV与预留确认输入，源run后续已运行；各面板完成状态以原件为准。`scripts/run_recorded_evaluation.py`复用本入口并记录单面板会话／资源，不调度训练或自动选模。

从仓库根目录执行，使用 Conda `splendor`、`PYTHONPATH=src`。基线评测入口为 `splendor.evaluate`，冻结 checkpoint/search 的统一入口为 `splendor.evaluation`。配置种类包括 `greedy_checkpoint`、`search_checkpoint`（固定深度）和 `mcts_checkpoint`（PUCT）；两侧模型可不同。

## 配对与计分

每个 pair 是一个 game seed 的两局先后手交换；统计得分率为 `(wins + 0.5 * draws) / valid_games`。bootstrap 以 pair 为单位，不将两局当成独立样本。报告必须标明双方 checkpoint 内容身份、动作模式、规则/特征/后端、搜索预算、seed、完整 pairs 和无效局数。

`passed=true` 表示评测工程完成，不表示棋力提升。未完成、异常、工程截断结果不能用于完整面板排名。确定性 policy-cycle 判平属于评测裁决，不是引擎正式终局；只用于固定确定性策略，不自动用于有根噪声/温度采样的采集。DEV 与 FINAL 的区别由预留和选模使用方式决定，单独种子域不自动成为无偏 FINAL 验收。

## 搜索评测规模与采集执行器（2026-10-20）

新正式面板默认：greedy 1000 pair，含搜索至少500 pair（1000局）；历史已冻结面板保留原规模。搜索双方仍按评测协议全程使用指定预算（当前持续迭代为S1024/N1），无根噪声、温度0、固定种子、交换先手，并保留checked runner的终局、确定性循环判和及工程截断分类。采集的75%低预算、根噪声和温度采样不带入评测。

`execution.backend="native_cuda_pool"`复用采集的`ExecutionConfig`与`multiworker_gpu.collect`：当前12个搜索worker、1000活跃局、GPU最大batch500、local microbatch64、每worker共享slots2、owner的pipeline_slots1、等待5ms、原candidate/batch buckets及native-cudagraph。可以复制`selfplay.execution`后将`pipeline_slots`设为1；采集仍保留2槽。真实对局由checked runner推进，`engine_mode=fast`不替换评测规则检查。单GPU owner加载双方模型；worker不加载模型，不创建CUDA上下文。每个请求携带checkpoint SHA，只与同模型、同policy/value类型及宽度类别的请求合批，transfer/图缓存也按模型隔离；每行只推理当前行动方。

该后端支持native或native_chance MCTS与`greedy_checkpoint`任意配对。2026-10-21新增`NativeGreedyPolicy`通过root-only native作业共享GPU owner：只前向根logit并argmax（并列取最小候选下标），不创建搜索世界、不展开树，模拟/确定化/搜索状态转移计数均为0；可与同模型搜索根合批，统计包含greedy前向。已有`native_batched`仍使用其原eager greedy执行路径。`device=cpu`提供共享布局的显式CPU复核路径；native-cudagraph正式配置使用CUDA。pool的推理配置由`execution`统一确定；`policy_a/b.inference`若显式指定，必须与之相符，冲突会报错。worker并发写pair日志加文件锁；完整pair支持直接恢复，部分pair沿用现有归档后重跑规则。工程与实际模型验证、正式S256N4对greedy面板见[实验记录](../../artifacts/entity_attention_r20_s256n4_vs_greedy_20261021/README.md)。

结果`execution_stats`记录聚合与逐模型推理行数、物理batch/平均batch、编译/图捕获、GPU峰值显存、worker供给、owner等待/整理/流水线重叠、逐局模拟数等。rule/sidecar时间是多worker累计时间，不能直接除以墙钟当成独占占比。双模型分流仍会拆分供给，也多占图缓存显存；仅短验证通过不能宣称达到采集吞吐。

持续迭代允许调整**搜索评测的pair数及推理执行参数**，不改变训练、采集和搜索协议。每轮边界重读源配置；已经准备的面板沿用自己的`*_config.json`，新面板采用新500 pair配置。恢复仍用原脚本，不改顶层初始配置或历史结果，也不自动补跑旧200 pair。

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

实体注意力第24轮的人机对局（绝对轮次24，对应续训目录 `round-014`，全局53898步）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m splendor.web_app --port 8765 \
  --mcts-checkpoint runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-014/checkpoints/final.pt \
  --mcts-label '第24轮 · θ53898' \
  --mcts-simulations 1024 --mcts-determinizations 1
```

配置了 MCTS 时，网页默认选中该对手。开局可修改 S（总模拟次数）和 N（隐藏局面采样数）；S 在 N 个独立搜索树间分配，须满足正整数且 N≤S。对局右侧的“AI 搜索预算”可随时修改，点击“应用到后续 AI 行动”后从下一次 AI 决策生效，棋盘和模型权重不变。正在进行的搜索需等待完成，不能中途改预算。输入框提供128、256、512、1024、2048、4096等 S 建议，也可自行填写其他正整数。

这里使用实验相同的 native 独立局面 MCTS（默认 `c_puct=1.5`、折叠强制阶段、单次模拟最多512次状态转移，选根访问量最大动作，无根噪声），与上面的固定深度 top-k 搜索入口分别配置。网页推理固定 CPU、PyTorch 单线程，不占用训练 CUDA 显存，可和训练同时运行；提高搜索预算会延长响应时间，也会争用部分 CPU。当前轮24模型固定加载，不随后台教师升级而切换。原生扩展依赖见[原生后端指南](native_search_zh-CN.md)。

预算 API：创建 `POST /api/games` 时可传 `searchBudget: {simulations: 1024, determinizations: 1}`；已有 MCTS 对局通过 `POST /api/games/<gameId>/search-budget` 传相同对象。返回的 `searchBudget` 与 `opponentLabel` 显示实际生效值；其他对手不接受 MCTS 预算。

上述第24轮命令打开 `http://127.0.0.1:8765`，可用 `--port` 指定其他空闲端口。网页对局保存在后端内存中，停止对应服务会丢失该服务的对局；训练进程独立运行。

顶部的局面评估条直接用当前对手模型的 critic 评估**你的 Observation**，不额外运行搜索。百分比为 `(clip(value, -1, 1) + 1) / 2`，表示胜计1、平计0.5、负计0的估计得分率，并非经过校准的纯胜率。悬停可查看原始 value；未结束时使用模型值，正式终局改为实际结果。固定启发式对手没有模型评估条。

顶部及终局弹窗提供“悔棋”：撤回上一次你的操作及随后全部 AI 回应；买牌、支付、归还等子步骤逐步撤销。支持连续悔棋，恢复原牌堆／随机状态、合法候选和行动记录，当前搜索预算继续保留。`POST /api/games/<gameId>/undo` 需传当前 `revision`；回退仍递增 revision，过期点击不会重新生效。

支付方案、归还宝石和贵族选择显示在页面顶部临时操作栏；只在相应阶段出现，完成后移除，不预留栏位。主行动继续直接操作市场卡牌、保留牌和宝石堆。
