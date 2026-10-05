# 离线训练缓存瓶颈与执行参数优化

2026-10-04，原任务为用户正在运行的新结构离线训练。没有停止用户进程、修改冻结配置或启动新的正式训练；有限验证在独立CPU进程中进行。操作见 [指南](../../docs/guides/entity_attention_offline_zh-CN.md)。

`profile.json` 保存第10,105步时最近1000步原始日志汇总：draw/load 266.689秒、collate 7.613秒、update 15.951秒，三者合计290.253秒；占比分别91.881%、2.623%、5.496%。该分母不含周期checkpoint/验证与其余记录开销。1000步miss63,762、hit192,238，约24.9%抽样需要整局NPZ解压。累计decode661,890次、解码数组416.643GB；此为重复解码产出字节，不等于物理磁盘读量。初始4 GiB只能容纳约6800局。

仅读取10000个ZIP中心目录，数组大小上界：train5,749,363,754字节、validation605,496,721字节，总5.918425 GiB，包含NPY头部；没有解码全量数组、重放或重复计算全量归档hash。统计耗时0.864秒；当时MemAvailable18,420,332KiB，约17.57 GiB。8 GiB数组LRU预算足够保留这份工作集，Python对象和进程其他内存另计，仍为按需加载。

实现只增加 `offline_train --replay-cache-mib` 执行参数，可在resume时覆盖cache大小，冻结config仍保持原值，actual容量记录到每步、source和sessions。训练算法、结构、batch、优化器、损失、预算、split和抽样不变。现有运行不会热更新；切换应在checkpoint提交后中断再resume，恢复时重新预热缓存。

`validation.json` 保存两项有限验证：4步连续CPU更新与2+2步resume（后半从1 MiB扩到8 MiB），model、optimizer、Torch RNG逐位一致，采样RNG和引用序列相同，冻结config相同；24局真实train工作集在8 MiB与32 MiB缓存各2048次预热、4096次测量，前者1844次miss、5.803秒，后者0次miss、0.008669秒，两组抽样序列相同。这仅是draw阶段的缩小工作集证明，不能当作真实8 GiB整轮训练倍率。

以当前阶段成本估算，完全消除加载耗时的理论上限约12.3倍；实际仍有抽样、组批、GPU更新、缓存预热、验证和保存开销。真实8 GiB整轮收益尚未测，使用该参数后的实际会话日志才是依据；不声称已测得10倍或更高整轮加速。

复现有限检查（另选不存在的output目录）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python artifacts/offline_training_speed_20261004/validate_cache.py \
  --output runtime/offline_cache_speed_20261004/check
```

正式续训（先由操作者结束原进程，不要同时启动）：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.offline_train \
  --config configs/entity_attention_v1/offline_attention.json \
  --output runs/entity_attention_offline --resume --replay-cache-mib 8192
```

`source_identity.json`、`commands.json` 和验证配置记录此次源码、操作、种子与范围；不改ignore规则，不纳管重型模型。原始临时验证目录在忽略的runtime中。
