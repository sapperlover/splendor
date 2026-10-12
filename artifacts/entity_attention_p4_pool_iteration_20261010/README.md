# P4大池起点九轮自迭代：完成结果与框架记录

2026-10-10：**正式9轮、同预算4500步纯旧池对照、全量旧池留出评估及最后两组搜索对战均已完成**，北京时间19:59结束。完整分析见[正式报告](../../docs/reports/entity_attention_p4_pool_iteration_20261010_zh-CN.md)，逐轮／逐面板见[结果表](results.md)和[自动汇总](summary.json)。下文框架有限验证是历史交付记录，有限smoke分数不作为正式棋力证据。

主线终点38,898步，双方S1024/N1、每组200pair：对P4起点58.750% [54.000%,63.500%]；对同预算纯旧池对照61.625% [57.000%,66.250%]；对P12＋V4为58.125% [53.375%,62.875%]。Greedy对P4第7～9轮约71%。旧池全量KL/MSE：P4为0.441453/0.742282，主线0.513325/0.757365，对照0.435839/0.762639。支持这次在线方案的棋力收益，未隔离教师换代的独立贡献，未确定最佳步数／lr／配额；单训练seed、固定DEV开局、未作多重比较校正。

报告必要的完成身份、原始输入SHA、整pair得分和整局拟合分子／分母保存在[completed_evidence.json.gz](completed_evidence.json.gz)，矩阵及配对区间见[completed_analysis.json](completed_analysis.json)。[analyze_completed.py](analyze_completed.py)从保留证据复算，不依赖模型；从仓库根目录执行 `conda run --no-capture-output -n splendor env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python artifacts/entity_attention_p4_pool_iteration_20261010/analyze_completed.py --verify`。两张报告图及[绘图脚本](plot_completed.py)同目录保存，命令见正式报告。

完整流程、七类29,000局来源、原split保留、新在线10%整局留出、双池采样与启动命令见[专题指南](../../docs/guides/entity_attention_p4_pool_iteration_zh-CN.md)。配置：[总流程](../../configs/entity_attention_v1/p4_pool_iteration_20261010.json)、[采集／训练](../../configs/entity_attention_v1/hybrid_p4_pool_20261010.json)。

从P4V4＝34,398步开始，每轮1000局／500步，共9轮；每batch旧池／在线配额前6轮依次224／32、186／70、147／109、109／147、70／186、32／224，第7轮起0／256。在线窗口最近6000个完整对局，旧大池停止训练抽样后仍保留用于诊断和离线对照。采集保持B500，batch档位改为[16,32,64,128,192,256,384,500]。教师每轮500步后无条件升级；评测结果不触发停止或回滚。末尾才运行同预算4500步纯离线对照，以及主线对离线对照／P12V4各200pair搜索。

正式运行生成 `summary.json`、`results.md`、`plan.json`。完整拟合统计、引用账本、checkpoint与评测sidecar保留在 `runs/entity_attention_p4_pool_iteration_20261010/`；有限验证现场保留在 `runtime/entity_attention_p4_pool_iteration_20261010/`，交付时不把smoke写到正式输出目录。

初版43项相关回归通过，最后针对代际固定Train子集的3项回归再次通过。CPU验证分段续跑与连续运行的模型／AdamW／RNG逐位一致、整局淘汰、输棋不阻止升级及完成命令幂等；真实RTX4050 GPU smoke完成2局S2采集、1+1步恢复、2步离线对照及四组各1pair评测。这次GPU训练smoke早于上述渐进配额和batch档位修订，不能作为新正式配置的完整运行证据。

本次修订的10项相关回归全部通过，实际调用渐进配额入口、切入纯在线训练，并核对分段／连续运行的模型、AdamW和RNG逐位一致。正式`check`返回9轮配额与新batch档位，见 [check_progressive.json](check_progressive.json)。GPU另外验证全部8个batch档×四个candidate档及value-only，共40种CUDA Graph形态、50次数值检查；无编译失败或eager回退，最大value误差2.55e-6。PyTorch峰值分配约128 MiB、保留约1.54 GiB；该数值不包括全部CUDA上下文／驱动内存，不是正式采集长期峰值。

初版范围与代码身份保留在 [verification.json](verification.json)；本次修订的命令、源码SHA、JUnit及GPU记录见 [progressive_revision.json](progressive_revision.json)、[GPU原始记录](gpu_progressive_batching.json)、[GPU验证脚本](verify_batching.py)。复现GPU验证可从根目录运行 `conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python -u artifacts/entity_attention_p4_pool_iteration_20261010/verify_batching.py`，输出写入runtime。

上述短验证当时没有覆盖正式W8／active1000／B500／S1024长负载，也没有棋力结论。随后正式任务现已完成：9000局、4500主线更新、4500对照更新、14个200pair面板；整个流程约6小时6分钟。在线缓存第7～9轮约1GiB，不能据此声称全进程绝无泄漏或所有机器参数最优；实际资源口径与结论边界见正式报告。
