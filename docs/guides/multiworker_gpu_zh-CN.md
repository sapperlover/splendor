# 单 GPU owner 的多进程搜索供给

native_cuda_pool 后端使用 spawn 搜索进程和主进程中的唯一冻结 CUDA 模型。搜索进程独占游戏、原生 PUCT 树、特征编码、逐局校验与归档；主进程只处理跨进程合批、推理与结果分发。现有 native_cuda、native_cpu、CPU eager 和旧采集格式继续走原路径。旧 alphazero_loop_cuda.json 未替换。

## 配置语义

| 字段 | native_cuda_pool 含义 |
| --- | --- |
| backend | native_cuda_pool 启用本协议；旧 native_cuda 仍是单进程 |
| workers | 旧采集外层进程数，池化后端必须为 1，绝不复制 CUDA 模型 |
| search_workers W | 请求的搜索进程数；实际为 min(W, A, 本次缺失游戏数) |
| total_active_games A | 本次采集全局同时保有的游戏上限，与 W、B 独立 |
| max_batch_size B | owner 单次物理推理的全局行数上限 |
| local_microbatch | 一个 worker 单槽导出的最大行数 |
| shared_slots | 每个 worker 的共享输入/结果槽数 |
| shared_candidate_width | 每槽预分配的最大候选宽度；超出时显式报容量错误，需在配置中增大 |
| candidate_buckets | 编译候选档位；超出最大档而不超共享容量时使用共享容量档、eager 推理 |
| max_batch_wait_ms | owner 见到首批 READY 后等待其它描述符的有限时间 |

所有容量为正整数。shared_candidate_width 至少覆盖最大编译档位。池化后端要求 batch_interface=native、native_transport=reused 和 FP32。active_games_per_worker 是旧后端字段，在新后端不控制并发；示例中设为 1 以免误读。

任务队列按 game ID 分配，worker 按额度领取；完成后立即补充，不迁移活动中的搜索树。G<A、G<W、A<W 和恢复尾段均裁剪实际启动数及活跃额度。worker 编号、任务领取顺序、物理 batch 不参与随机种子派生。每个 world 的待评估叶仍由原生 SearchJob 的 exported 标记限制为一个。

## 共享槽与复制

主进程在启动前为每个 worker 的每个槽创建三段预分配共享内存：FP32 特征、uint8 mask、FP32 融合结果。worker 在本进程中把 NativeBatch 直接指向共享输入槽，按原有 row-major/stride 格式写入；大数组和搜索树不进 Queue。队列只发送 worker、slot、generation、rows、kind、width、轮次模型版本和完成通知。checkpoint SHA 在 worker 启动时确认一次，批中只核对轮次模型版本；game/world/simulation 到行的映射保留在 worker 的 NativeBatch。

一个正常批的主机/设备数据路径：

1. 原生编码直接写共享输入，无额外 host 副本。
2. owner 按连续行块把共享输入复制一次到复用的 pinned float/mask staging；宽度不同时复制固定字段，补齐候选字段与 mask。共享内存本身不是 pinned。
3. 两个 H2D 分别传 float 与 mask 到复用设备缓冲。现有 transfer 在设备内融合 logits/value，再做一次融合 D2H。
4. owner 按微批行块把 pinned 结果复制一次到相应共享结果槽；worker 用原生批量 submit 消费。

policy/value-only 分别合批。owner 一批不必等待所有 worker，满 B 立即发；请求不足时按 max_batch_wait_ms 到期发。一个微批大于剩余 B 时分段，维护行偏移与完成数，全部完成后才通知 worker。部分描述符移至待处理队列尾部，避免大批长期占先。结果写完再发通知，worker 消费完成才复用槽；generation 检查拒绝旧结果与重复结果。Queue 的锁与信号量承担进程间发布/获取同步。owner 完成 D2H 并复制结果后才释放源请求；下一次推理不会覆盖尚未分发的输出视图。

编译档位之外的宽度统一回退到 shared_candidate_width 的有界缓冲，不为每一个超档宽度无限保留 GPU staging。正式路径不按叶节点序列化 NumPy/DecisionRequest，也不使用 Manager 的高频代理。

## 故障与恢复

主进程是共享段的唯一创建者和 unlink 责任方。worker 成功退出或异常退出时关闭映射；主进程在 finally 中 join/terminate 全部子进程、关闭并 unlink 全部共享段。worker 异常写 failures/worker-*.json，包含去重的活跃游戏 seed、trace、root request、搜索现场及在途 slot/generation；owner 的输出和批量异常写 failures/owner.json，保留受影响描述符。未完成局不归档，已完成局保留。任务 ID 只预先放入全局活跃额度对应的数量，完成一局后补入下一局；提前终止时取消尚未消费消息的队列发送线程退出等待。进程退出与结果等待有有限超时及对端存活检查。`validate_run()` 将 worker/owner 故障报告与旧格式一起汇总，并按 game ID 去重。新一次 --resume 只跳过完整归档游戏，并重新创建所有进程与槽。

每次 generate() 调用加载本轮唯一冻结模型，结束后不再持有 worker 或共享段。AlphaZero 的 learner 接在 collect 返回之后，可使用同一 GPU。

## 统计与已验证范围

status.json 的 gpu_owner 保存请求和实际 W/A、B、微批/槽容量、共享字节数、owner CPU 秒、输入等待（含超时）、合批等待、gather、分发、峰值待处理行、宽度与 padding 分布、发批原因、owner CUDA 内存峰值和实际推理/编译诊断。workers 保存各 worker 的完成 game IDs、模拟数、CPU 秒、等待结果与队列背压；顶层旧统计继续报告本次 session 的 games/s、policy 样本/s、simulations/s。顶层 cpu_peak_rss_kib 是 owner 与 worker 的单进程峰值最大值，不是所有进程 RSS 之和；shared_bytes 是本协议预分配共享段容量；gpu_peak_bytes 是 PyTorch 分配峰值，不含驱动上下文。进程 CPU 秒可相加且可能超过墙钟；gather/forward/H2D/D2H 为不同主机计时口径，不能直接解释为 GPU 利用率。短窗口基准中，owner 的网络位置/批次按 owner 自己的测量秒数计算，模拟速率为各 worker 在各自测量区间的速率之和；完成局数只计窗口内增量。`common_seconds` 仅供观察区间重叠，不作为这些速率的分母。owner 与 worker 的 CPU 秒/工作量分开报告，不把不同窗口的 CPU 秒相加后除以模拟数。

既有回归覆盖共享布局逐字段一致、混合宽度与补齐、真实 GPU 输出数值一致、确定性 owner 下 N>1/value-only 搜索结果一致、W=1 与非整除/尾段配置、slot generation、worker/owner 故障清理、真实 CUDA 完整游戏 checked replay，以及极小 AlphaZero collect→CUDA learner→checkpoint→evaluation。真实 GPU 合批下只要求合理 FP32 容差；受控 owner 的策略和访问数精确一致。稳态窗口和两组完整 500 局采集已有原始结果，见 [GPU pool 报告](../reports/multiworker_gpu_zh-CN.md)。窗口收益不等于整轮收益，现有结果不支持自动替换生产默认。

## 可执行命令

有限分项统计另见 [2026-10-02 报告](../reports/pipeline_profile_zh-CN.md) 和 [`profile_speed_pipeline.py collection`](../../scripts/profile_speed_pipeline.py)。诊断包装保存真实叶宽、worker 微批宽、owner 物理档位及完成时间；开局窗口与小轮次分开报告，worker 累计等待不并入 owner 墙钟。

小规模正式采集及 checked replay：

~~~bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast generate \
  --config configs/native_search/selfplay_cuda_pool_smoke.json --output runtime/selfplay_cuda_pool_smoke
conda run -n splendor env PYTHONPATH=src python -m splendor.selfplay_fast validate \
  --input runtime/selfplay_cuda_pool_smoke --replay
~~~

极小 AlphaZero 闭环：

~~~bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -m splendor.alphazero_loop \
  --config configs/native_search/alphazero_loop_cuda_pool_smoke.json \
  --output runtime/az_native_cuda_pool_smoke
~~~

需要重新测量时可从以下预设开始。--search-workers、--total-active、--batch、--local-microbatch、--shared-slots 和 --shared-candidate-width 可直接覆盖，不用改代码。短窗口命令的 games-per-worker 只为窗口提供足够游戏，不表示必须完成这些局：

~~~bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/throughput_cuda_pool_8x500_b256.json \
  --games-per-worker 80 --warmup-seconds 2 --seconds 35 \
  --output artifacts/inference_throughput/multiworker_gpu/pool_b256.json
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/throughput_cuda_pool_8x500_b500.json \
  --games-per-worker 80 --warmup-seconds 2 --seconds 35 \
  --output artifacts/inference_throughput/multiworker_gpu/pool_b500.json
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/benchmark_collect_throughput.py \
  --config configs/native_search/selfplay_cuda_baseline_500.json \
  --games-per-worker 640 --warmup-seconds 2 --seconds 35 \
  --output artifacts/inference_throughput/multiworker_gpu/single_b500.json
~~~

只有短窗口显示稳定收益时，再分别运行 selfplay_fast generate，以 selfplay_cuda_baseline_500.json 和胜出的 throughput_cuda_pool_8x500_b*.json 采集各 500 局到不同新目录。已有完整采集的配置快照和 summary 已保留到 artifacts；复测应使用新输出目录，不覆盖历史记录。

## 2026-10-03 等待与搬运改动

worker 在当前可供给槽全部提交后报告 `blocked` 及槽 generation。owner 在处理 hello/complete/done/error 后，按当前生产者和实际在途槽判断：满批、真正到 deadline、所有可供给槽到齐、生产者耗尽。结果返回会撤销对应 worker 的 blocked 状态；分段请求保留槽所有权，policy/value 分开，超出已编译上限的请求单独分组，队列保持轮转。单 worker 或不足 8 局仍完成所有对局。

同布局 gather 直接块复制；不同布局只清零候选 padding 后缀。共享容量仍明确限制为 256，超限报错，不截断候选。`configs/speed_2026-10-03/collection_optimized.json` 在原 16/32/48 档之外采用 64 档，保持 W=8/A=500/B=500、micro=64、slots=1、wait=5ms。两组各1,000局生产采集与全量checked replay已通过，主墙钟1714.56→1384.04秒；这是本机每组一次的联合改动结果，详见[完整性能专题](../reports/end_to_end_speed_optimization_zh-CN.md)。

`gpu_owner.batch_trigger_detail` 记录触发原因、行数、是否等待和等待时长，旧 `deadline` 分类不可当作真实超时。设置 `SPLENDOR_SPEED_ROW_STATS=1` 可额外聚合实际叶候选宽度、微批宽度和根对局阶段；它不写逐叶日志。

`supply_series` 以5秒间隔记录全轮剩余局数、active/blocked worker、pending rows/slots；`tail_intervals_exclusive` 的above100、11to100、9to10、2to8、last1互斥。额外汇总的last100/10/8/1区间则嵌套，不能相加。根opening/middle/late按双方合计回合<10、10–49、≥50统计，与全轮剩余局数分开。

`compile_failure_policy=fail_fast` 除后端编译错误也覆盖Dynamo缓存限额的隐式eager退化；不通过增加无限图数量掩盖。共享容量以内而编译档位外的合法宽请求仍有明确eager执行语义，统计单列，和编译失败回退区分。
