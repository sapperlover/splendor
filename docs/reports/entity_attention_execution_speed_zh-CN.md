# 新模型固定形状、kernel启动与独立双缓冲

2026-10-05。两项执行优化均已实现并完成有限验证。推荐的可选采集预设为 **shared_slots=2、固定B64/256/500、native-cudagraph、owner单缓冲**。它在最终代码的开局窗口比原配置快31.1%，但其中主要收益来自增加worker请求供给；固定地址Graph在同供给控制组上仅增加约2%。独立双缓冲的额外收益未稳定，不建议默认开启。没有启动优化版完整1000局、棋力赛或训练。

当前任务只改变采集执行。模型、FP32、S1024/N1、根噪声、温度、游戏种子与搜索预算保持一致；learner及轮间评测沿用原设置。已有自迭代run的配置及产物未修改。

## 实验设置与证据

RTX4050 Laptop 6 GiB、i5-13500H、PyTorch2.14.0+cu130、conda splendor。使用`runs/entity_attention_offline/checkpoints/final.pt`的532738参数新模型，W8/A1000/B500、micro64、wait5ms、C16/32/48/64、reused、projection=false、fail_fast。每组3秒采集预热、15秒测量，顺序执行，测量时无用户训练进程。游戏seed13600000、搜索seed13630000；原始配置记录其余种子。编译预热、进程总耗时和窗口吞吐分开。

[完整原始记录与汇总](../../artifacts/entity_attention_execution_speed_20261005/README.md)包含每组配置、owner/worker窗口与计数、命令、设备状态、代码HEAD及源码快照身份。性能依据是独立窗口，不使用最后一次与CPU回归并行的direct微测计时。一次早期static500窗口与CPU测试开始重叠约0.52秒，已排除，原件在runtime保留，失效依据单独记录。

1000局均实际进入8×125局调度，但窗口内没有完成整局，主要为开局且owner宽度为C32。因此不能将以下比例当作完整1000局、尾段或最新在线父代的速度。离线final是本次固定测量checkpoint，不是用训练更新造成的速度变化。

## 最终代码的同条件对照

表中四组为同一最终源码、同seed的第3组窗口，单位为完成模拟/秒。历史重复用于观察波动，不与这组混成“整轮加速”。

| 配置 | 模拟/秒 | 相对原配置 | 相对shared2控制 |
| --- | ---: | ---: | ---: |
| 原配置：shared1、动态B、owner1 | 43,388 | 1.000× | — |
| 只增加shared2、动态B、owner1 | 55,715 | 1.284× | 1.000× |
| shared2＋固定地址Graph，owner1 | 56,873 | 1.311× | 1.021× |
| shared2＋固定地址Graph，owner2 | 56,375 | 1.299× | 1.012× |

共享槽2控制的三个窗口为56.5k、56.3k、55.7k模拟/秒，平均batch约499–500；原shared1为46.0k、41.2k、43.4k，平均batch约475–476。供给更充分、合批等待减少，是主要收益来源。

固定地址Graph的两组为57.5k、56.9k。联合双缓冲为59.3k、56.4k，相对Graph单缓冲分别+3.1%、−0.9%；未证明独立双缓冲有稳定增益。两个短窗口也不足以确认2%左右的Graph收益在完整轮次中成立。

## 两项优化的实现和负对照

`batch_buckets`将真实行补到最小覆盖B档，固定编译B/C形状。native传输复用pinned/device缓冲，只归还真实行；padding不计入搜索或逻辑位置。静态预热使用实际行stride。Dynamo的partial包装共享图缓存，B×C组合超过默认8图时，按配置的`len(B)×(2×len(C)+1)`设置局部有限上限，保持fail-fast。

最初的`reduce-overhead`加B256/512确实使用了CUDA Graph，但两个窗口比各自原配置慢14.3%、5.1%。Profiler发现每批17次额外D2D字段拷贝，且B388补至512增加32%的计算行数。增加B64、把上限改为500后，shared2下仍比对应动态B控制慢4.0%。减少启动调用不能抵消额外搬运、补齐计算及静态kernel成本。

最终新增项目模式`compile_mode="native-cudagraph"`：以torch.compile default生成固定形状kernel，每个transfer捕获自己的固定device输入、前向及设备输出融合，然后直接重放。H2D与真实行D2H在图外，避免额外字段拷贝。捕获异常明确失败；超C档位保持原eager语义。图及缓冲按lane/kind/C/B独立持有。

真实C32/B388的10批Profiler调用统计：

| 模式 | 普通kernel启动调用 | Graph启动 | H2D / D2H / D2D拷贝事件 |
| --- | ---: | ---: | ---: |
| 原动态B | 470 | 0 | 20 / 10 / 0 |
| B256/512＋reduce-overhead | 20 | 10 | 20 / 10 / 170 |
| 固定地址native-cudagraph | 0 | 10 | 20 / 10 / 0 |

这是主机启动调用数量；图内GPU kernel仍然执行。最终控制组的forward主机计时约3.2秒/15秒窗，固定地址Graph降至约0.16秒，完成等待随之增加；不能把计时从forward转移到wait解释成GPU计算变慢或净节省3秒。

`pipeline_slots=2`让owner在当前GPU执行时整理下一批，输入及输出缓冲独立。始终只有一个GPU前向在途；完成当前输出分发后才启动下一批。共享请求维护预留/完成行数，所有分段完成才归还槽。异常先等待在途拷贝，再清理worker和共享段。

独立双缓冲需要足够前置供给。保持shared1时，B500后缀反复变成12行单独小批，平均B降至256，吞吐仅22.1k。shared2下动态B双缓冲为55.0k、55.5k，低于同供给单缓冲控制。它确实把约2秒gather安排在句柄未finish期间，但增加的合批等待、搬运/分发与调度开销抵消收益。固定地址Graph加双缓冲改善了这一结果，额外收益仍未重复稳定。

## 正确性、剩余瓶颈与命令

55项相关CPU回归通过，覆盖旧冻结配置兼容、固定B补齐、有限图上限、split请求、受控结果精确一致和两种owner模式的故障清理。三个GPU推理模式的policy/value及value-only均通过`atol=rtol=1e-5`，覆盖真实请求和B128/388/500、3行尾批。reduce-overhead与固定地址Graph各完成4局S8/N2、双缓冲采集及checked replay，均0失败；固定地址Graph短局实际发生378批在途准备。FP32验证不要求跨batch的搜索选招逐位相同。

固定地址Graph目前将启动开销显著降低，但在shared2控制上整体只小幅提速，剩余主机计时集中在GPU完成等待（含kernel、拷贝与同步）、CPU gather和请求调度；据此推断，进一步减少启动调用的收益空间已较小。原正式run完整轮2的阶段记录中，采集2076.36秒占83.02%，learner36.39秒占1.45%，轮间评测约12.78%；本次没有改变训练内核，不能据此宣称learner加速。

新采集预设[collection_static_graph.json](../../configs/entity_attention_v1/collection_static_graph.json)已完成加载检查。以下1000局命令已准备但未启动，须使用新的输出目录；不能拿新配置直接恢复已有loop冻结配置。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.selfplay_fast generate \
  --config configs/entity_attention_v1/collection_static_graph.json \
  --output runs/entity_attention_static_graph_collection
```

有限对照复测命令（使用新的repeat编号保存新结果）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python \
  artifacts/entity_attention_execution_speed_20261005/run_windows.py \
  --variants baseline baseline_shared2 direct direct_double --repeat 4
```

独立双缓冲开关保留为`execution.pipeline_slots=2`，不写入推荐预设；其他模型/设备、完整轮次及训练收益均未验证。
