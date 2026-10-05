# 新 AZ 实验记录字段与统计口径

2026-10-04。类别以本次改动前的能力区分：**已有**=生产直接记录；**汇总**=现有原件可导出，本次离线接通；**新增**=本次补入。下表是正式run的字段约定，准备状态不表示已有正式数值。历史缺字段为unavailable+原因，不用0替代。

简写：`R=runs/az_theta5_10x1000_seed1`、`A=artifacts/az_theta5_10x1000_seed1`、`r=rounds/round-xxx`、`a=round-xxx`。NPZ/checkpoint/引用计数/完整sidecar保留于忽略的runs；必要聚合JSON/CSV与冻结配置/身份保留于artifacts。全部log单位约定为秒、字节、无量纲比率；熵/CE/KL为自然对数nats。字段不存在时不能把None解释成0。

## 身份、配置、恢复

| 类别 | 字段 / 单位 / 分母 | 作用域与频率 | 原件 → 汇总 |
| --- | --- | --- | --- |
| 新增 | session_id、command、start_unix/end_unix、wall_seconds、status、error_type/error | 每次启动/恢复；墙钟不含两次之间暂停 | `R/recording/sessions/<id>.json` → `A/summary.json.sessions` |
| 新增 | recovery_start/end_state、已完成轮/累计train step/各轮phase | invocation前后；旧提交与本次新增可差分 | 同上 → summary/round summaries |
| 新增 | git_head/git_status/dirty_diff_sha256、source_files/source_sha256、native_extensions | invocation一次；实际源码身份 | `R/recording/identities/<id>.json`、`source/<id>/{src,scripts,dirty.patch}` → 准备/会话清单 |
| 新增 | config_sha256、原配置、脚本SHA | 冻结配置一次；script包含在source_files | `R/config.json`、identities、`configs/az_theta5_10x1000/*.json` → `A/preparation_manifest.json` |
| 已有+新增 | 初始/父/final path、sha256、model_sha256、model_config、parameter_count、round_train_step/total_train_step | 起点、每轮边界与恢复；参数量为现架构 | `config/loop_state.json`、`R/r/checkpoints/{final.json,*.pt}` → round summary、准备清单 |
| 已有+新增 | AdamW config/state、sampling_rng_state、learner_start.optimizer_inherited/sampling_seed | checkpoint每100步/末；首轮fresh、跨轮继承、同轮恢复 | checkpoint、training.json、`usage/<id>.json` → round metadata |
| 新增 | python/executable/torch/torch_cuda、driver/GPU UUID/VRAM、CPU/meminfo、threads、性能环境 | invocation一次；运行环境，非当前HEAD臆补历史 | identities.environment → 原件保留/准备清单 |
| 已有+新增 | collection/split/sampling基种子与10轮派生、game/world/behavior输入规则、评测game/policy/bootstrap | 冻结一次；行为逐决策原始引用/trace | config、manifest、NPZ、eval sidecar → `A/seeds.json` |
| 新增 | compile_cache.warm_state、path、diagnostics配置、SPLENDOR性能开关 | invocation一次；cold/warm unknown不强填 | identities/config → manifest；实际compile hit/cost看owner |
| 新增 | recorded_actual_wall_seconds、pause_seconds_known、hard_kill_wall_lower_bound_seconds、phase_seconds | 所有invocation；仅实际结束段有精确总量 | sessions/resources → `A/summary.json` |
| 新增 | recovery_overhead attempted/committed/repeated steps、重复draw/collate/update秒 | 每轮；完整记录的重复/未提交attempt，不是暂停 | `usage/<id>.steps.jsonl`、checkpoint/metrics → summary.rounds.recovery_overhead |

默认eval字段序列化保持旧Python配置身份；bootstrap为0且seed为0时省略新字段。新有bootstrap/diagnostics配置才携带它们，参与冻结。无需移动/重写旧run。实际SHA不同仍明确拒绝resume，不放宽模型/数据身份检查。

## 数据、搜索与完整性

| 类别 | 字段 / 单位 / 分母 | 作用域与频率 | 原件 → 汇总 |
| --- | --- | --- | --- |
| 已有 | completed_games/missing_games/failed_games/exceptions/engineering_truncations及expected num_games | 每轮全量与失败现场；工程错误不当平局 | `R/r/selfplay/{summary,status,manifest}.json`、failures → round_summary.collect |
| 已有 | pack/validate校验、trace、natural terminal reason、非法动作/invariant报错 | 每局提交前；原trace不额外复制到长日志 | NPZ metadata/trace、failures → summary与dataset |
| 汇总 | extra_checked_replay.games_passed/passed/seconds | 训练后，每轮全部新局Python checked replay；额外成本单列 | NPZ.trace → `A/a/dataset.json.extra_checked_replay`；失败阻断并保留failure/session |
| 已有+汇总 | game_id/game_seed/first_player/checkpoint_sha256/path/archive_sha256 | 每局；跨轮主键round/run/game | NPZ metadata → `A/a/dataset.json.games` |
| 已有+汇总 | completed_player_turns、decision_count、policy_samples、value_samples、player_decisions | 每局；合计回合≠决策，value有效权重计数≠policy计数 | NPZ metadata/arrays → dataset.games |
| 汇总 | combined_turns/decisions.mean/p50/p90/p99/max | 当轮新局；分母完整新game数 | 同上 → dataset |
| 汇总 | first_second_wdl、terminal_scores、first_player_score_margin、terminal_reasons | 新完整局；分差=首手终分−后手终分，判平按正式结果 | terminal_result.scores/winner_ids/reason → dataset |
| 已有+汇总 | current_train/history_train/current_validation/history_validation的games/positions/policy_positions/compressed_bytes | 每轮保留replay；validation为整局split | frozen replay_snapshot、NPZ/index → dataset.pools |
| 汇总 | validation_game_fraction、retained_by_round、retained_round_game_shares、evicted_games/淘汰来源 | 每轮；实际val局数/保留总局数，来源局数/总局数 | replay_snapshot与之前各轮归档计数 → dataset；淘汰可与轮原件对照 |
| 汇总 | root_stage_positions、root_candidate_count_histogram、target_entropy_sums、root_target_entropy | 新轮归档根决策；阶段分母policy_weight_sum | candidate_offsets/policy_target/policy_weight/completed_player_turns → dataset |
| 已有 | simulations、logical/physical rows、batches、owner width_rows/compiled/eager/failure/padding、触发原因 | 完整采集会话与阶段；统计字段以真实status为准 | selfplay.status/summary、`R/recording/collection/<round-id>.json` → round统计引用 |
| 已有 | root_encoding/real_rule/search_rule/native_encoding/native_compute/write秒 | worker累计区间，与owner并行 | summary/NPZ.mcts_stats → 原件引用；不加入互斥墙钟和 |
| 新增+已有 | percent50/90/99/100、tail100/10/8/1时间/工作量 | milestone在节流前采集；owner提供细分行/batch，尾区间嵌套 | events、collection snapshot、status.tail_bins → round记录 |

默认不采逐叶候选宽画像；正式owner宽档/物理容量计数与归档根候选直方图均保留。叶级真实候选总量、worker padding/stage rows若依赖`SPLENDOR_SPEED_ROW_STATS`且未开，则unavailable，不能从根决策统计替代。需要该重诊断时另短测开销。

## 学习、使用与验证

| 类别 | 字段 / 单位 / 分母 | 作用域与频率 | 原件 → 汇总 |
| --- | --- | --- | --- |
| 已有 | round/total_train_step、total_loss/policy_loss/value_loss、grad_norm、learning_rate、batch_size、current/history_positions | 每个已完成更新；训练batch，不称验证 | `R/r/training_metrics.jsonl` → `A/learning_curve.csv`、summary |
| 已有+新增 | policy_ce_sum/policy_weight_sum、entropy_sum/target_entropy_sum、value_sq_sum/value_weight_sum | 每个实际batch；policy零权重排除，padding无贡献 | metrics、usage steps → summary.training_batch_metrics，按sum聚合 |
| 新增 | target_entropy、policy_entropy、policy_kl | 同一个batch/同一个policy权重分母；policy_entropy为预测熵 | 同上 → learning_curve.csv；KL=CE−目标熵，不混平均方式 |
| 新增 | policy_positions/value_positions、value_target_sum/value_target_sq_sum、value_zero_mse | 有效权重样本数与加权标签矩；value平方和/value权重 | 同上 → summary与诊断 |
| 汇总 | grad_norm分位数、grad_clipped、clipped_steps/clip_fraction、nonfinite异常 | 已提交步骤；clip步数/有记录步数 | metrics/failure/session → summary；旧flag缺失unavailable |
| 新增 | reference_sha256、little-endian `[step,round,game,decision]`、sequence_sha256 | 每batch/每invocation；checkpoint前fsync | `R/r/usage/<id>.bin/.json`、checkpoint.metadata → `A/a/usage.json` |
| 新增 | committed_counts.npz references/counts、current/history pool_positions/draws/covered/unseen/mean_uses/use_count_histogram | 最终已提交round step；均值draws/全部池positions；覆盖分母全部池positions | `R/r/usage/committed_counts.npz`（含SHA）→ `A/a/usage.json` |
| 新增 | attempted_draws/repeated_or_uncommitted_draws/torn_bytes | 全部会话attempt与已提交去重序列；撕裂尾不当完整batch | 同上 → usage；不能把hash当覆盖计数 |
| 新增 | training_constant.value/value_weight_sum/source/replay_digest | 每轮保留train标签拟合一次；validation标签不参与 | `R/r/diagnostics/training_constant.json` → 两套验证.constant_baseline |
| 新增 | current_validation.set_sha256/references/composition/positions | 当轮全部validation；训练准备时冻结，原件SHA在诊断检查 | `R/r/diagnostics/current_validation.json` → `A/a/current_before_after.json` |
| 新增 | before/after checkpoint SHA、overall/early/middle/late CE/H/KL/value MSE/有效数/原始和与权重 | 同一当轮冻结集合；父权重和final，训练后离线评估 | 对应checkpoint/manifest → current_before_after.json |
| 新增 | fixed_validation.set_sha256/references/composition/selection_seed | 首轮validation最多2048；整run同一集合，不参与train | `R/diagnostics/fixed_validation.json` → `A/fixed_validation.json` |
| 新增 | fixed_curve节点初始/100步/最终、checkpoint/set SHA、每阶段指标 | 每轮训练结束后读取已存checkpoint；按SHA去重final别名 | checkpoint+fixed原件 → `A/a/fixed_curve.json`、`A/fixed_validation_curve.csv` |
| 新增 | value_zero_mse、value_train_constant_mse、空组available/reason | 两套验证；常数均来自train，对同一val标签计算MSE | 两套诊断JSON/CSV；空policy分母null，空组unavailable |

每100步固定集曲线成本属于diagnostics，不混进正式learner更新秒数。固定集引用原件即使被replay淘汰也保留。不同轮当轮验证数据/搜索目标变化，不用它们的CE直接推断吸收能力；看同轮before/after及同一固定集曲线。

## 速度与资源

| 类别 | 字段 / 单位 / 分母 | 作用域与频率 | 原件 → 汇总 |
| --- | --- | --- | --- |
| 新增 | collect/replay_index/validation_prepare/learner_initialize/checkpoint_commit/diagnostics/eval_*/archive_summary秒 | 每次invocation、每轮，互斥阶段 | session.phases → summary.phase_seconds/round.phase_seconds |
| 新增 | draw_load/collate/update_host/checkpoint/validation/remainder秒 | 每实际训练步/会话聚合；主机区间，不是kernel | metrics.seconds、session.details、training.phase_seconds → summary互斥账本 |
| 已有+新增 | games/min、policy_samples/s、simulations/s；pairs/min；steps/s/positions/s | 工作量除完整计时范围；会话新增量与跨恢复累计量分别标明 | 全部sessions和提交量、owner及eval summary → summary；硬kill精确值不可用 |
| 新增 | cache_delta hits/misses/decode_count/decoded_bytes/load_seconds | draw前后共享cache差值；current/history只记一次 | metrics.cache_delta → summary.cache_windows.first32/last32/all |
| 已有+新增 | capacity/cache_bytes/peak_cache_bytes、entries/peak_entries | 每步snapshot；峰值该cache实例范围 | metrics.replay_cache → training/session/summary；4GiB≠RSS |
| 新增 | cache hit_rate /实际窗口step数 | 前32、末32、全round，少于32用实际数；hits/(hits+misses) | 已提交cache_delta正确去重 → cache_windows |
| 已有 | owner输入/合批等待、gather/dispatch、forward/H2D/D2H/完成等主机计时；compiled/eager宽档/失败/填充 | 完整采集会话及现有owner阶段；不跨进程加总成墙钟 | collection status/snapshot → 原件引用 |
| 已有 | eval execution_stats每模型rows/batches/width/mean/compile/native fill/consume/noninfer/infer host | 完整面板，本次invocation实际执行范围 | `R/r/evaluation/*.json`、`recording/evaluation/<round-id>.json` → 原件引用和eval summary |
| 新增 | 进程树CPU百分比/累计存活CPU秒、RSS/PSS、observed/processes | 每5秒；CPU一核100%，退出子进程尾未知 | `R/recording/resources.jsonl` → summary.resources_by_phase |
| 新增 | GPU uuid/utilization/memory.used/temperature/power/clocks（%/MiB/°C/W/MHz） | 每5秒；设备级，含其他程序占用 | resources.gpu_csv → resources_by_phase.gpu_device_samples |
| 新增 | torch allocated/reserved/peak bytes | 同线程低频、当前进程invocation峰值；没有reset/sync | resources.torch_allocator → resources_by_phase |
| 新增 | 每阶段资源采样数/resource_error、warm/cold/吞吐scope | 每阶段/每会话；少量点不代表稳态，采样失败明确记录 | sessions/resources → summary |
| 新增 | 外部锚点完整/分段墙钟、恢复起止、5秒resources | 每个外部panel单独ledger；不计入主loop耗时 | `R/anchors/recording/<panel>/recording/*`、wall_summary.json → anchors分析/原件引用 |

内部phase秒精确结束的会话与硬kill留下的partial阶段单列，暂停只对已知前一次结束计算。整轮汇总按各invocation中属于该轮的已结束phase起止区间相加，未观测尾部明确下界，不能只用round_summary最后一段覆盖旧耗时。轮内phase之外的状态提交/日志等保留remainder。

## 棋力、查看与决策

| 类别 | 字段 / 单位 / 分母 | 作用域与频率 | 原件 → 汇总 |
| --- | --- | --- | --- |
| 已有+新增 | panel_id/purpose/run/round、双方checkpoint_identity完整SHA、搜索/执行配置、game/policy/bootstrap输入 | 每轮两DEV、阶段/结束外部面板；用途分开 | eval `.games/manifest.json`、config/summary → `A/summary.json.evaluation` |
| 已有 | seat/game_seed/policy_seeds、W/D/L、终态/长度/decision数、异常分类 | 每局、每pair，两局交换首手 | `*.games/pair-*.jsonl` → summary总体/first/second |
| 已有 | completed/valid pairs/games、passed、a_wins/draws/a_losses、win_rate_score | 得分率=(W+.5D)/有效完整局数；工程失败不当draw | sidecar → eval summary；failed不能称完整验收 |
| 已有+新增 | paired_bootstrap CI95%、unit/resamples/seed | 每面板，单位一个paired seed，不拆座位 | sidecar+config → summary.paired_bootstrap；loop新增透传两seed字段 |
| 新增 | viewed_at/node/checkpoint_sha256/panels/decision/reason/expanded/late_evaluation | 每次查看或变更建议，人工追加；3/5/10节点分别写判断 | `A/decision_template.json` → `A/decisions.jsonl`，分析给出路径，不自动选模 |

父代、固定θ5、固定PPO、greedy分开趋势；原始配对数据不可只剩均值。跨run“θ5”使用完整SHA/run身份区分。独立extension500使用不同输入，但由先前查看引发的扩样仍存在选择效应，完整查看记录是解释前提。
