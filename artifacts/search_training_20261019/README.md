# 搜索训练改进：有限工程验证

2026-10-23归档位置更新：本专题关联`runtime/search_training_20261019`中的1个抽样账本／逐步日志已按用户授权外置至[当日T7归档](../repository_archive_20261023/README.md)，原路径保留相对链接。复算这些原件前需解包；复训前将需写入的文件恢复为原位真实文件。历史指标、正式结果及来源身份不改写。

2026-10-19完成预算随机化、根Forced Playouts、Policy Target Pruning，并接通既有在线采集、数据校验、replay和训练入口。[算法、配置及命令](../../docs/guides/search_training_zh-CN.md)。未启动正式训练，所有短对战仅用于验证工程链路，不支持棋力结论。

## 回归

188项相关测试中183通过、5项CUDA条件测试跳过、0失败。命令见指南；[verification.json](verification.json)记录Git版本、未提交源码SHA、native扩展SHA、教师身份、实际smoke命令及测试规模。扩展已重新构建；当前conda环境缺少构建依赖pybind11，使用安装于 `/tmp/splendor-search-build-deps` 的pybind11 3.1.0完成本地构建，没有更改运行环境依赖。

新增21项测试覆盖预算抽样、整数裁剪参考、Python/native单/双世界差分，以及Python批量、native逐叶和native连续批三条CPU路径：完整采集、checked replay、归档篡改拒绝、原样resume、replay入池、混合和全低预算batch训练。旧采集、搜索、在线循环、联合训练等回归均通过。

## 新版128模型完整入口smoke

[smoke_loop.json](smoke_loop.json)使用新版128×2的37423步模型，CPU native连续批推理：一轮、2局、S32/N1与S4/N1按25%/75%随机、k=2与pruning开启，batch8、lr1e-4，仅更新2步。包含两个1pair的小预算评测入口，均只用于确认流程。

| 核验量 | 结果 |
| --- | --- |
| 正式终局 / 异常 / 工程截断 | 2 / 0 / 0 |
| 总决策 | 165 |
| 低预算决策 | 93 |
| 高预算有效policy样本 | 40 |
| 单候选、零预算决策 | 32 |
| 实际模拟数 | 1652 = 93×4 + 40×32 |
| 实际Forced Playouts | 373 |
| 标签扣除访问数 | 274 |

[collection_summary.json](collection_summary.json)由独立 `validate_run(replay=True)` 重建，包含每步预算、根Q/先验复算标签及checked真实轨迹重放。完整run保存在 `runtime/search_training_20261019/cpu_loop/`，包含原始NPZ、manifest、训练指标、checkpoint、评测sidecars及loop状态；未纳管重型训练产物。

该loop随机抽到的两个训练batch都无有效policy样本，正确执行value更新、保存checkpoint、进入后续评测。另从同一批真实v3数据固定取一份低预算及一份高预算样本，再做混合和全低预算各一步：混合batch policy权重和为1、policy loss>0；全低预算policy权重和和loss均为0，value loss有限。保存后恢复模型参数逐张量精确一致，见[fixed_batch_training.json](fixed_batch_training.json)。这部分checkpoint位于 `runtime/search_training_20261019/fixed_batch_final.pt`。

GPU执行代码复用同一 `_GameSession` / native SearchJob，但本次未实测CUDA多worker采集和编译吞吐；不能把CPU正确性等同于GPU性能收益。完整S1024/S128示例仅准备并通过配置解析，未启动。
