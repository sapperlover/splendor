# 200局采集的Graph尺寸与恢复验证

2026-10-06。调整`configs/entity_attention_v1/hybrid_fork_s1024_n1.json`采集执行：B[64,256,500] → **B[16,32,64,128,200]**，`max_batch_size`500 → **200**。教师S1024/N1、200局/100步、batch256、3000局回放、离线退出日程、共享槽2和owner单缓冲保持原值。

恢复仅放开上述两项字段；已有采集manifest的轮次沿用该轮原尺寸，新轮采用本次请求尺寸。根目录冻结配置、旧manifest和NPZ不重写。会话源码身份记录请求的`collection_execution`及原`frozen_config_sha256`，每轮实际配置以采集manifest为准。原进程不会热加载，操作与resume命令见[指南](../../docs/guides/entity_attention_hybrid_s1024_n1_zh-CN.md)。

## 验证与原始依据

- [配置快照](config_graph200.json)：本次可执行预设，所有非采集批次字段与已有分支冻结配置一致。
- [CPU回归](regression_tests.xml)：26项通过，覆盖模型/AdamW/RNG的中断恢复、旧采集尺寸保留与下一轮切换、算法字段变更拒绝；实际attention模型在五个新批次桶的静态填充输出与eager对照（policy绝对误差≤1e-5、value≤1e-6），无eager fallback。
- [真实checkpoint恢复](resume_verification.json)、[会话源码身份](resume_source_identity.json)：复制此前9500步、3000局只初始化验证目录，在副本上以新预设执行`--resume --online-rounds 0`。模型、theta_5000、状态、fork manifest和旧冻结配置前后SHA-256一致；记录的请求执行为新桶/上限200。未采集、未训练、未评测，没有操作用户正式进程。
- [填充量推算](padding_projection.json)：引用用户正式分支前两轮原始summary并保存哈希和批次直方图。旧Graph采集耗时422.38秒、471.27秒；保持原直方图时，额外填充行/真实行分别由42.17%/42.02%降至5.13%/5.01%。这是算术推算，没有测量新Graph速度。

复核测试命令：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_hybrid_training.py tests/test_inference_modes.py
```

本环境`torch.cuda.is_available()`为false；CPU静态输出测试使用`torch.compile`的eager backend，未执行CUDA Graph捕获或GPU速度实验。用户正式分支检查时至少已完成2新轮/9700步，使用旧进程加载的Graph尺寸；新尺寸的整轮时间须在用户重启后观察。

调试中失败的测试XML保留于忽略目录`runtime/entity_attention_graph200_validation_20261006/`，最终通过结果见本目录回归XML。重型checkpoint与源码快照保留在该目录的`resume_old_config/`，不随仓库纳管；历史实验原件保持原路径。
