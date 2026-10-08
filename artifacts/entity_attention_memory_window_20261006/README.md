# S1024/N1 第31轮 RAM OOM：修复与历史窗口分析

2026-10-06。正式对象为 `runs/entity_attention_hybrid_s1024_n1_from_round9_seed1`。第30新轮已提交、累计12500步；第31轮开始采集后被系统杀死，没有完整落盘新局。没有重新离线训练、改变正式采样算法或执行新的棋力赛。

## 已确认的内存问题与修复

内核在北京时间13:23:00记录 `global_oom`，杀死PID17949；其RSS为22288212 KiB，约21.26 GiB。机器没有swap。原始内核证据的相关消息见 `oom_evidence.json`。共享内存/semaphore警告发生在SIGKILL后，不能把它们当作OOM原因。

原代码跨轮复用 `_GameCache`，只按8 GiB数组预算LRU淘汰，不随3000局活动索引移除旧游戏。在线缓存第30轮为8218局、5820651070字节数组；这些多出的游戏没有参与训练。16 GiB配置只限制两池数组，不限制元数据、编译状态或其他进程内存。报告的owner进程RSS高水位从新轮10的13.02 GiB增至新轮30的20.71 GiB。数组缓存是确定的增长来源，其他内存没有完整归因。

本次修复：

- 每次更新活动索引后，从在线解码缓存移除窗口之外的游戏，保留固定在线验证所需游戏。NPZ原件、训练索引、采样顺序不变。
- 在首个纯在线轮次开始采集之前，离线解码缓存只保留固定离线验证引用的游戏；继续核验冻结离线数据身份。
- 采集和评测结束后调用 `torch.compiler.reset()`、GC与CUDA空闲缓存释放。此时所有该阶段推理已经结束；Inductor磁盘缓存保留。
- hybrid支持 `--replay-cache-mib` 执行预算覆盖，与既有offline入口一致。原run冻结配置不改，每个会话记录实际字节预算。

建议本机恢复时将数组缓存总预算设为8 GiB（两池各4 GiB），编译线程设为2；CPU搜索worker仍为8。编译线程数只控制Inductor编译池。第31轮进入纯在线后，按第30轮归档估计，活动在线+固定在线验证共3095局，NPY数组字节上界约2.04 GiB；固定离线验证涉及938局，约0.56 GiB。这里包含NPY头部，只估计数组工作集，不是总RSS保证，见 `cache_working_set.json`。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src TORCHINDUCTOR_COMPILE_THREADS=2 \
  python -u -m splendor.hybrid_train \
  --config configs/entity_attention_v1/hybrid_fork_s1024_n1.json \
  --output runs/entity_attention_hybrid_s1024_n1_from_round9_seed1 \
  --resume --forever --replay-cache-mib 8192
```

## 验证

21项相关CPU回归通过，见 `regression_tests.xml`，覆盖实际数组回收、跨轮淘汰与固定holdout存续、纯在线采集前释放离线train缓存、编译推理对象回收，以及不同缓存预算恢复后模型/AdamW/RNG逐项一致。

独立GPU短测为4轮×4局×2步、S2/N1，2 search workers、shared_slots=2、owner单缓冲、固定B16；第2/4轮各评测1 pair，共4局均有效。耗时50.04秒，4轮owner RSS高水位为1.629/1.655/1.868/1.868 GiB。每轮CUDA Graph捕获、采集、更新均成功，固定在线验证离开活动窗口后仍可读取；第3轮进入纯在线前离线训练缓存实际被移除。配置、环境、结果分别见 `gpu_smoke_config.json`、`gpu_smoke_result.json`，完整运行产物位于忽略的 `runtime/entity_attention_memory_gpu_smoke_20261006/`。这是调用链短验证，不证明正式3000局长跑RSS已稳定或棋力提高。

正式输出目录的真实CLI执行了 `--resume --online-rounds 30 --replay-cache-mib 8192`，零采集、零更新。state、冻结config、fork manifest、12500步checkpoint、5000步对手及第30/31轮相关manifest/snapshot哈希全部保持不变。新会话记录8 GiB预算，见 `resume_verification.json`。没有启动正式第31轮。

复核：

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_hybrid_training.py tests/test_alphazero_replay.py
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  artifacts/entity_attention_memory_window_20261006/analyze.py
```

`gpu_smoke.py --output <新的独立runtime目录>`用于重做有限GPU验证；`verify_resume.py`限定正式run已经完成30轮的现场，不作为通用恢复入口。

## 窗口太长还是太短

已有实验不能给出因果判定：没有相同起点、教师、更新量及采样规则下的3000/6000窗口训练对照。原1000局/500步/6000局窗口与现200局/100步/3000局窗口同时改变了更新粒度、来源分布与教师数量。

实际账本表明，新轮30的25600次训练抽样中：

| 来源 | 抽样数 | 占整个batch |
| --- | ---: | ---: |
| 最新200局（12400步教师） | 1452 | 5.67% |
| 在线历史（此前14个教师） | 20948 | 81.83% |
| 旧Deep-Sets离线轨迹 | 3200 | 12.50% |

最新200局有14829个train positions，首轮抽到1375个不同局面，覆盖9.27%；最近10轮的新数据平均占batch 5.73%。它们与缓存8218局是不同概念。原第九轮1000局/500步时，最新数据占总抽样约12.69%，见[既有100/500步分析](../entity_attention_100_vs_500_20261005/README.md)。

现3000局窗口保留15个教师，最旧教师在训练前落后1400步；第30轮实际抽样加权平均教师年龄约704步。原6000局窗口只有6个采集教师，但最旧教师落后2500步。缩小局数同时增加了教师代数，不能只用“局数”或“代数”断言目标过旧。

过多历史**可能稀释最新教师目标并牵制更新方向**。当前新数据立即得到的训练份额偏小，值得检验；但数据不呈现清晰的“历史policy越来越好、最新policy越来越差”。第30轮按新轮30/26～29/16～25各抽512个固定train/holdout局面，更新前后拟合变化混合，完整结果在 `analysis.json.cohort_metrics_round30`。其中最新holdout的512位置抽样KL甚至改善，而同一轮全2194个holdout位置KL略恶化，说明小样本会误导判断。不能用这些抽样结果宣布窗口过长、过短或value梯度冲突已被证明。

一批完整观察15轮的数据（新轮16，11000步教师）有15758个train positions，在其窗口生命周期内实际采样23094次，平均1.466次、独立位置覆盖77.32%、最大单位置8次。它不符合“整池刷很多遍”。在局长与配额近似稳定时，均匀采样下一个位置的预期生命周期曝光约为 `每轮在线抽样数 / 每轮新增训练位置数`：窗口增大降低每轮抽中概率，但延长存续时间，两者大致抵消。大窗口主要改变目标年龄、多样性及反馈延迟，并不自动增加每局累计训练量。

建议暂保留3000局，先完成内存修复后的现有训练观察。若要单独检验新目标稀释，优先对照“最新局面固定25%、其余历史”的采样方式：第31轮纯在线后batch256可取最新64、历史192。按第30轮数据量，最新数据首轮曝光约0.43次，比当前0.098次提高约4.4倍，采集量和更新步数均可保持200/100。这是建议的后续实验，**本次没有实现或应用该采样变化**；不得将它称为已经验证的最优配置。

## 修正此前过拟合解释

此前推断的依据是：混合训练误差下降、固定旧分布policy误差上升、搜索棋力在自迭代开始后回退。它可以提示分布漂移、遗忘或过拟合，却不能区分旧标签过时、历史目标冲突、value/policy共享梯度及搜索目标噪声。把它直接称为“对当前局面过拟合”过于确定，应降为待验证假设。

原第九轮500步online平均曝光仅0.208次，100/500步固定对手得分69.0%/69.5%，没有证明500步造成明显棋力下降。当前最新数据占比更低，最近10轮每轮100步后全量current holdout policy KL均升高、平均+0.01191；value有5轮改善/5轮恶化，平均MSE变化+0.00547。当前更明确的事实是：更新没有稳定改善最新教师的留出数据；尚没有隔离最新来源的实际训练样本前后误差，不能据此证明典型的当前来源过拟合。

第30轮score74%，比第25轮高4个百分点；同种子配对差值95%区间为−6～+14个百分点，尚不足以确认棋力增长。离线固定value改善、固定旧policy误差上升也不能独立决定窗口长度。原始输入哈希、代码身份和逐轮统计见 `manifest.json`、`analysis.json`；分析仅CPU FP32、单线程，零optimizer更新、零新采集，单训练seed与固定对手限制保留。
