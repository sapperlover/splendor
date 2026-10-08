# 分头窗口与detach value实现验证

2026-10-06。正式实现位于`src/splendor/alphazero_train.py`与`hybrid_train.py`，可执行配置为`configs/entity_attention_v1/hybrid_split_policy_value.json`。[完整参数、观察口径、启动和恢复](../../docs/guides/entity_attention_split_training_zh-CN.md)。本交付确认调用链正确，正式长训未启动，未声称棋力改善。

Policy只在近4采集轮train局面上学习；value在3000归档局的近期/更早两个互斥桶各抽一半。各batch256，value在head边界detach；两组独立clip1.0，AdamW参数顺序、moments和RNG原样继承。每轮200局/100步，S1024/N1，保留共享槽2、固定地址Graph与owner单缓冲。每5轮对固定PPO5.5M、双方S1024/N1测100 pairs，用独立projection；原theta5000评测和旧joint配置的恢复保持原语义。

## 有限验证

- `tests.json`：相关40项测试通过，包含旧joint连续训练、缓存淘汰、已有fork与评测恢复，以及新分头池、留出排除、极大value梯度不改变policy梯度/更新、CPU逐张量断点恢复和PPO独立projection/hash校验。CPU测试的NVML警告不影响结果。
- `verification.json`与`verify.py`：只读从父第50新轮/14500步初始化私有fork，继承3000局，完整比较模型、AdamW、Torch CPU/CUDA RNG和采样RNG。原源文件与3000个继承NPZ的SHA均保持不变。
- 在真实继承数据上，每路batch256，连续4个GPU更新与2步＋checkpoint恢复2步的模型、优化器和全部RNG逐张量完全一致。没有采集新数据；两组各4步只属于验证。256 MiB冷LRU下第一组4步5.66秒，其中抽样/解码5.15秒、更新0.44秒；不能外推正式缓存热态整轮速度。
- 新训练checkpoint与固定PPO跑1 pair/2局S8 compiled/native短评测，全部有效；只验证训练导出与独立推理配置，不用于棋力结论。正式200局评测尚未执行。

起点池的近期4轮为采集索引55–58，policy最新轮期望24.13%、value12.07%；真实首轮采集后会变为56–59。教师身份与比例见verification的`split_replay`，正式每轮会重新冻结实际窗口。父模型SHA为`1f794e1ecee4e71b6aa5f3b45ce22bac50caf9055021a792b3e63056c1d94216`，固定PPO SHA为`009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0`。

## 复核

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_hybrid_training.py tests/test_alphazero_training.py \
  tests/test_alphazero_replay.py tests/test_entity_attention.py
```

短GPU验证需要可用CUDA与父run原始数据；使用新的私有目录，避免覆盖本次证据：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src TORCHINDUCTOR_COMPILE_THREADS=2 \
  python artifacts/entity_attention_split_training_20261006/verify.py \
  --output runs/entity_attention_split_training_validation_repeat
```

重跑会刷新本目录的verification，归档前需要同步实验索引中的哈希。原始测试XML放在忽略的`runtime/entity_attention_split_training_20261006/tests.xml`，模型、逐步引用和短评测sidecar放在忽略的`runs/entity_attention_split_training_validation_20261006/`。源码快照与环境身份同时保存于私有run和verification；正式训练入口亦会冻结源码/配置、教师及数据身份。
