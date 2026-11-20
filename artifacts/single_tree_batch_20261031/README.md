# 在线单根随机节点搜索加速

2026-10-31。接入单棵共享树的多叶子批量推理，以及仓库已有的compiled／固定地址CUDA Graph路径。固定policy在线第9轮，权重SHA256 `852cb0420e94d77c146bc2330600f48d52f638e4cca96a5c1578e9dfe54940ec`。这是工程延迟与搜索记账验证，没有正式棋力对战。

## 推荐配置

本机RTX4050 Laptop、CPU1调度线程、CUDA FP32、TF32关闭，推荐单树总65536次、最多64个在途模拟、compiled/native-cudagraph、visits选点、seed0、无根噪声、CPW关闭。启动命令与网页适配边界见[推荐器指南](../../docs/guides/kirameki_advisor_zh-CN.md)。默认旧命令仍使用逐叶／eager；新参数必须显式指定。

64是同一棵树内的在途数，不是64个确定化世界。购买后有选择的支付、拿筹码后有选择的归还仍独立搜索相同预算；本次没有接入子树转根复用、跨回合缓存或自适应预算，也没有修改采集／评测的冻结配置。

## 算法与调用链

- `kirameki_live → LocalAdvisor → BatchedWorldSearch → NativeBatchRunner → SearchJob`完成实际接入；`LocalAdvisor`接受既有`InferenceConfig`。
- 每个模拟按issued ID获得独立隐藏样本和翻牌随机流；不能用尚未增长的completed计数重复播种。
- 选路时加入不计分的临时访问，已完成Q保持原值。原始N/W仅在真实回传时增长；真实访问、隐藏抽样和预算对账，结束后在途模拟／临时访问均归零。
- 相同待评估信息历史节点合并一次网络前向，保留每条路径并分别回传。碰撞数表示合并路径的数量，不是额外展开节点；相同预算下，有效展开节点数可能随批大小变化。
- 批量结果全部回传后再选择下一批；默认batch1保持历史逐叶顺序。传输行以`(world, simulation)`标识，验证全部行后才修改树，保留过期／重复行拒绝及失败模拟定位。
- Graph预热使用之后实际复用的缓冲。普通根输入额外预热精确宽度和padding两种tensor布局，避免首次推荐临时重编译。首次编译／本进程预热与稳定搜索计时分开。

## 同GPU、同预算的正式工程计时

三个真实保存盘面：第3盘R2、第4盘R13、第5盘R24。每组依次独立搜索，同一配置重复两次；下表取预热后的第二次，不含进程启动／编译／Graph预热。全部配置均为**S65536N1，总65536**。

| 叶子在途上限 | 第3盘R2 | 第4盘R13 | 第5盘R24 | 与串行相比的中位加速 |
| --- | ---: | ---: | ---: | ---: |
| 1，GPU compiled串行 | 19.105秒 | 18.839秒 | 18.332秒 | 1× |
| 32 | 3.559秒 | 3.501秒 | 3.138秒 | 5.38× |
| 64 | 3.080秒 | 3.195秒 | 2.822秒 | 6.20× |
| 128 | 2.750秒 | 3.120秒 | 2.682秒 | 6.83× |

三个根的四组首选均一致。64的碰撞比例为8.14%／0.62%／0.46%，128为16.27%／1.29%／0.97%；128仅再缩短少量延迟，推荐64。Torch峰值allocated约32MiB（64）／45MiB（128），不包含CUDA驱动／上下文和全部RSS。

8192次开发筛选中，逐叶GPU编译约2.25–2.36秒，batch32约0.33–0.38秒，batch64约0.29–0.33秒；部分盘面的首选随批大小变化。因此批量不是访问分布的精确等价变换，上表三个大预算根一致不能证明一般棋力无损。小预算可选16／32，避免过长的回传延迟。

CPU逐叶8192次在三个根精确复现原预算矩阵全部访问数；同seed重复、非整批尾部、强制终局、隐藏抽样、碰撞回传、价值视角与传输失败边界由相关回归覆盖。完整推荐入口另验证购买＋条件支付，每段预算单列；最终结果与测试计数见[summary.json](summary.json)。

## 验证与原件

最终相关回归**114项通过，无跳过**；新增根布局预热回归要求真实输入不再产生新编译图。当前实际模型的64行主行动／支付／归还输入另做eager与Graph数值对照，最大logit绝对误差3.22e-6、value误差7.75e-7。完整命令验证购买＋支付各65536次：主行动4.04秒、付款3.67秒、合计7.72秒；本次已有磁盘缓存，启动预热10.5秒单列，根布局预热后两阶段没有新增编译图。仅使用本机保存快照，不连接或操作网站。

原件在本机`runtime/single_tree_batch_20261031/`，尚未归档T7：

- `baseline_source/`、`baseline_identity.json`和`baseline_git.diff`保留本次修改前已有未提交代码。
- `final_source/`及身份清单冻结大预算计时源码／数据／扩展；`delivery_source/`及`delivery_identity.json`另包含最后的普通根布局预热修正。搜索算法与原生扩展一致，预热修正未改变PUCT或网络计算。
- `deep_cuda_graph_b*.json`与`deep_tuning_commands.json`保存完整结果、环境、源文件SHA、权重SHA和执行命令；每份结果核对结束时源码未变。
- `cli_smoke_final/`、`inference_verification.json`、`final_tests.xml`保存完整入口、数值与回归原件。`serial_verification.json`核对串行访问复现。
- 其余screen／v2文件是开发筛选，不与同GPU大预算工程计时混作正式对照。早期测试失败日志保留，交付依据是最终验证。

源码／聚合结果／本报告进入Git，完整逐根矩阵、编译缓存、依赖临时目录和JUnit保留runtime。汇总复算：

```bash
python artifacts/single_tree_batch_20261031/summarize.py
```

测速复现（先恢复本机原始盘面目录，三个根同一配置串行执行）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/single_tree_batch_20261031/benchmark.py --device cuda --batch 64 --inference compiled --compile-mode native-cudagraph --budget 65536 --repeats 2 --output runtime/single_tree_batch_recheck/b64.json
```
