# 复现与恢复

从仓库根目录执行，conda `splendor`，按顺序运行，两组不同时争抢设备。下列checkpoint/NPZ、源码与脚本快照是本机runtime/runs产物，迁移前须备份。实际首次命令、线程和source路径同时记录在各JSON；快照脚本按SHA保存。这里通过有相同内容的冻结脚本复现，基线collection早期脚本为baseline_runner.py，其余用case名。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
speed_root=runtime/end_to_end_speed_2026-10-03
speed_artifacts=artifacts/end_to_end_speed_2026-10-03
speed_configs=configs/speed_2026-10-03
conda run --no-capture-output -n splendor env PYTHONPATH=$speed_root/baseline_source/src python $speed_root/scripts/baseline_runner.py collection --config $speed_configs/collection_baseline.json --run $speed_root/collection_baseline_low --record $speed_artifacts/collection_baseline_low.json --instrument
conda run --no-capture-output -n splendor env PYTHONPATH=$speed_root/optimized_source/src SPLENDOR_SPEED_ROW_STATS=1 python $speed_root/optimized_source/scripts/run_speed_validation.py collection --config $speed_configs/collection_optimized.json --run $speed_root/collection_optimized --record $speed_artifacts/collection_optimized.json --instrument
conda run --no-capture-output -n splendor env PYTHONPATH=$speed_root/baseline_source/src python $speed_root/scripts/evaluation_baseline_runner.py evaluation --config $speed_configs/evaluation_baseline.json --run $speed_root/evaluation_baseline.json --record $speed_artifacts/evaluation_baseline.json --instrument
conda run --no-capture-output -n splendor env PYTHONPATH=$speed_root/optimized_source/src python $speed_root/optimized_source/scripts/run_speed_validation.py evaluation --config $speed_configs/evaluation_optimized.json --run $speed_root/evaluation_optimized.json --record $speed_artifacts/evaluation_optimized.json
conda run --no-capture-output -n splendor env PYTHONPATH=$speed_root/optimized_source/src python $speed_root/optimized_source/scripts/run_speed_validation.py learner --config runs/az_loop_seed1/config.json --run $speed_root/learner_256 --record $speed_artifacts/learner_256.json --cache-mib 256 --warmup-steps 8 --steps 128
conda run --no-capture-output -n splendor env PYTHONPATH=$speed_root/optimized_source/src python $speed_root/optimized_source/scripts/run_speed_validation.py learner --config runs/az_loop_seed1/config.json --run $speed_root/learner_2048 --record $speed_artifacts/learner_2048.json --cache-mib 2048 --warmup-steps 8 --steps 128
```

**上列正式目录均已完成，不应再次覆盖计时记录。** 完整复测先为 `--run` 与 `--record` 选择新的路径；不删除、清理或搬移原目录。采集/评测中断才向原命令追加 `--resume`，保持配置、source、模型与种子不变。harness保留sessions及中断墙钟，collector只补缺失归档，评测只计完整pair，半pair按原种子/两座位重跑。完成但含ERROR的pair不会被静默改为成功；先定位失败现场。

本次正式六组均一次会话完成。多会话恢复后的累计时间需按 `sessions` 中成功主耗时和中断耗时核对，新增工作量与累计工作量分开；最后一次会话主耗时不能除以全部游戏数。当前聚合脚本要求单会话/完整统计，多会话数据应先单独复核，不直接套用本次排行榜。learner诊断不resume，始终从同一θ4起点在新目录独立启动。

新数据smoke使用修正后的外层脚本，生产src仍指向冻结优化快照（历史snapshot脚本的history_batch_size=0参数不适用于此smoke，原失败现场保留）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=$speed_root/optimized_source/src python scripts/run_speed_validation.py learner_smoke --config $speed_configs/collection_optimized.json --collected-run $speed_root/collection_optimized --run $speed_root/new_data_learner_smoke_v2 --record $speed_artifacts/new_data_learner_smoke_v2.json
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/compare_speed_inference.py --input $speed_root/collection_baseline_low --scratch $speed_root/numeric_inputs_v2 --output $speed_artifacts/inference_numeric_comparison.json
conda run --no-capture-output -n splendor env PYTHONPATH=$speed_root/optimized_source/src python $speed_root/optimized_source/scripts/profile_speed_pipeline.py --output $speed_artifacts/learner_cuda_events_short.json --scratch $speed_root/learner_cuda_events_short --cprofile learner --cache-mib 2048 --warmup-steps 4 --steps 4
conda run --no-capture-output -n splendor env PYTHONPATH=src python scripts/summarize_speed_validation.py
```

前三项也是已存在的产物；复测改新路径。最后一项只聚合正式原件和轻量metadata，无训练/采集，可重新生成summary。沙箱内CUDA不可见时通过主机权限运行同样命令，不重装CUDA。无需重建native；扩展与源码/声明的构建flags已记录，历史原构建命令未留存。
