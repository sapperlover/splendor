# 端到端速度优化验收（已完成）

本次需求为 prompt/15。基线实际工作树、native 扩展和 checkpoint 身份见 baseline_identity.json；dirty diff 保留在 baseline_dirty.patch，独立执行副本在 runtime/end_to_end_speed_2026-10-03/baseline_source。

正式运行走 selfplay_fast.generate / run_fixed_evaluation，不用底层窗口代替完整验收。NPZ、sidecar、checkpoint、脚本快照与中断现场保留 runtime/end_to_end_speed_2026-10-03；本目录存聚合原件和 5 秒资源采样。baseline_source 的源码不随着当前修改更新。

短窗口 overhead_off/on 为初次仪器对照：逐行包装使吞吐下降约17%，已弃用。首次off资源线程因缺少psutil未采样（后续直接读/proc，未安装依赖）；首次聚合窗口下降5.6%，相邻复测双方均采样则只差0.039%，不能据初测判断稳定开销。仪器复测见instrumentation.json。优化组另开启内联叶宽计数，单次开/关短测约2.29%成本，正式收益包含它。首次collection_baseline已被本任务中断，所有现场保留，作为排除的重仪器运行；collection_baseline_low才是正式基线。

两组各1000/1000局（额外全量checked replay通过）、各50/50 pairs和100/100局（passed=true）均完成，异常/截断/缺失0；固定第五轮replay各8预热+128测量步完成。新采集数据另通过两步训练与验证。主墙钟采集1714.56→1384.04秒、learner总成本92.25→11.41秒、评测682.17→207.21秒。每组一次，单机结果；不声称新的棋力或多轮AZ性能。

- [完整分析报告](../../docs/reports/end_to_end_speed_optimization_zh-CN.md)：时间占比、宽度/padding、尾段、资源与限制。
- [summary.json](summary.json)：六组正式结果、工作量归一化比较、逐局身份/结果、互斥成本与资源峰值。
- [manifest.json](manifest.json)：原件SHA、配置/脚本/源码身份及完整产物清单。
- [commands.md](commands.md)：冻结快照复现、恢复与独立复测命令。
- baseline_identity/optimized_identity及dirty.patch：HEAD之外的实际源码证据；native扩展SHA相同，原构建命令未留存，未重建。
- learner_equivalence.json：引用/loss/模型/optimizer/抽样RNG一致性，未消费的全局Torch RNG差异如实记录。
- inference_numeric_comparison.json：真实宽30/51输入的三模型CPU/GPU输出核验。
- new_data_learner_smoke_v2.json：新产物两步训练/验证；首次脚本参数不合法在训练前失败，原目录保留。
- learner_cuda_events_short.json：独立4+4步重仪器诊断，CUDA流区间不混入正式排名。
- tests.json及三份XML：真实CUDA相关回归、后续目标/推理/批量AZ闭环回归，重复执行不累计为不同测试。

完整NPZ、checkpoint、sidecar、源码快照与中断现场位于忽略的 `runtime/end_to_end_speed_2026-10-03/`。迁移环境需要另行备份这些本地产物；本目录的聚合不能替代它们。原始测量JSON与旧历史文件保留，未修改.gitignore、未提交或推送。
