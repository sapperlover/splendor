# 早期采集与工程试验归档：可支持的结论

2026-10-12。补充未形成独立科学报告的早期smoke和临时输出说明，避免归档后把工程对局误用为棋力证据。依据已保存manifest／summary／loop_state，不重新运行原实验，不承诺当前源码可原样复现当时版本。实际移动路径见[归档清单](../../artifacts/repository_archive_20261012/manifest.json)。

## 数据格式与早期采集

`runs/az_selfplay_seed1_round000` 是300局S256/N2、`mcts-selfplay-v1` 的单次采集，不是300pair，也不是训练后的棋力面板。原summary完成300/300、failed／exceptions／engineering truncations为0，原划分Train268局／Validation32局。它说明当时采集产生了完整对局档案；本次未重新重放全部动作，不由自对弈胜负推导模型强度。4局 `runtime/mcts_step2_smoke` 同属采集／归档短验证。

读法、提交协议与训练标签含义见[数据指南](../guides/selfplay_data_zh-CN.md)。旧协议与S预算不同，是不直接并入当前attention S1024/N1池的原因；不是已证明数据损坏。

## 仅作工程闭环的 smoke

| 原位置 | 已保存事实 | 可支持与不可支持的结论 |
| --- | --- | --- |
| `runtime/az_loop_smoke` | 两轮各4局，采集summary均完成且无记录异常 | 当时的小规模AZ采集／训练闭环；不能代替五轮正式结果 |
| `runtime/az_native_cuda_pool_smoke` | 一轮2局，采集summary完成且无记录异常 | 原生GPU池接入的小规模闭环；不能保证正式B500/S1024长期内存或吞吐 |
| `runtime/gpu_pool_dev_smoke` | 2/2局，failed／exceptions／工程截断为0 | 开发态GPU池的短链路可运行；没有完整生产速度对照 |
| `runtime/gpu_pool_compiled_dev` | 1/1局，failed／exceptions／工程截断为0，配置允许fallback | compiled入口的一次短测；不能据此声称编译回退为零、所有形状覆盖或棋力提升 |

原始完整字段仍在归档summary里。这里的“无记录异常”限于这些已保存汇总，不是对当前代码的新验收。原AZ正式五轮结论见[正式报告](alphazero_seed1_zh-CN.md)，后续有效吞吐对照见[单owner报告](inference_throughput_zh-CN.md)和[多worker报告](multiworker_gpu_zh-CN.md)。

## 旧 Python 搜索性能试验

`runtime/search_accel` 保存固定局面微基准、CPU/GPU数值比较、Python合批与串行搜索短测及少量采集。不同文件的预算、batch、设备和源码阶段不同，不能将其中最好的时间当成当前native配置的已知最佳值。固定局面吞吐与有限smoke不提供新checkpoint棋力结论。

归档保留原文件用于复核当时的计时／数值路径，本次没有对其做统一重算或重新认证。现行行为与验证入口见[搜索指南](../guides/search_zh-CN.md)，生产配置选择优先使用与当前后端、模型和形状匹配的专题测量。

## 临时局面校准器输出

`runtime/action_benchmark_cli_check/r19_prior/summary.json` 是200局面／77来源对局的prior模式CLI输出；`split_r19_vs_r9/summary.json` 是83个子集局面的S1024/N4比较。两者协议为 `splendor-action-benchmark-v1`，对应模块不在当前源码树中。保留原输出只能证明这些临时诊断曾产生文件，不能证明其校准方法已经正确、跨模型有效或能替代独立棋力对战。

因此本项标为**未核验的历史调试输出，无正式棋力结论**，不把动作分布变化、JSD或校准数值命名为胜率。当前维护的固定局面诊断见[256局面指南](../guides/decision_probe_zh-CN.md)，仍应与真实配对对战分开解释。

## 已有专题覆盖的工程数据

其他runtime来源已经在[归档结论索引](../../artifacts/repository_archive_20261012/conclusions.md)逐一映射到原专题，包括缓存等价性、GPU Graph数值验证、退出清理、迭代框架恢复、目标随机性及critic特征缓存。它们的测试数量、资源峰值和时间均属于原版本及原负载，不因这次归档获得更广的有效性。
