# 第43轮历史池离线训练及前置直接对战

2026-10-12。按用户要求串行执行：先测原第24轮＋24388步离线候选对原第43轮，再从原第43轮在完全相同的43000局固定池上离线训练，训练结束后执行原R24实验的评测日程。完成状态以[结果表](results.md)、`progress.json`及`runtime/entity_attention_r43_history_pool_20261012/live.json`为准；准备和短验证不代表正式训练／对战已完成。

## 执行顺序与固定对手

1. [R24＋24388步 vs 原R43](../entity_attention_r24_offline_vs_r43_20261012/README.md)：双方S1024/N1、无根噪声、200pair，独立开局35800000起。这个面板只回答此前离线候选是否优于原R43，不参与新训练节点选择。
2. 从原R43完整checkpoint继承模型、AdamW矩与Torch随机状态，追加48776步全参数联合训练。池采样与诊断种子保持原R24实验不变。
3. 追加1000／6097／12194／24388／48776步各做200pair greedy，固定对手原R43，开局35500000起。
4. 追加6097／12194／24388／48776步各做200pair S1024/N1，固定对手原R43，四节点共用35600000起的DEV开局。
5. 按已完成搜索DEV最高得分选一个节点，并列取较早节点；接触确认结果前冻结`selection.json`，再用35700000起的独立500pair，对原R43确认。即使DEV没有正向信号也完成此流程。

每个训练／评测阶段独立子进程顺序执行，退出后释放缓存和GPU资源。三个R43评测开局库及前置直接对战开局库互不重叠，也不复用R24离线实验的评测开局。双方互换先后手、和棋半分、整pair bootstrap 10000次。独立的是确认开局和搜索随机流，只有一个训练seed；对R43与对R24的绝对得分不能直接相减当作代际差。

## 同一个历史池与相同训练方法

直接引用原[冻结清单](../entity_attention_r24_history_pool_20261012/pool_games.jsonl)，SHA256为`99d7c8460d45206daa195d9bf0994b8935e148233833ec841a6e9d1dab611cfb`，不重新采集、不重新分组、不扩大到第44轮，也不混旧29000局池。

| 来源 | 总局数 | Train局数／局面 | 留出局数／局面 |
| --- | ---: | ---: | ---: |
| P4主线第1–9轮 | 9000 | 8054／654945 | 946／77267 |
| 同一主线续跑第10–43轮 | 34000 | 30529／2466674 | 3471／280355 |
| 合计 | 43000 | 38583／3121619 | 4417／357622 |

数据来源及标签含义见[原R24流程](../entity_attention_r24_history_pool_20261012/README.md#数据从哪里来)。全部是attention完整网络S1024/N1自对弈，使用原单份行为访问分布和自然终局结果，保留原整局Train／validation身份；只有Train参与更新。对R43而言，这些都是已产生的历史数据，不像R24起点那样包含第25–43轮的后续新数据；本实验检验历史重放能否进一步改进R43。

起点：`runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt`，总步数72898，文件SHA256 `b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。该目录的`round-033`是续跑相对编号，实际为绝对第43轮。

完全沿用R24的训练参数：AdamW、lr固定2e-5、batch256、weight decay1e-4、policy/value权重1:1、clip1、8GiB无损字典缓存，所有Train局面均匀有放回抽样；平均追加抽样4次，共48776步。继承的是R43自己的AdamW与Torch随机状态；新池采样seed34410000、固定诊断seed34420000与R24实验相同。训练数据及预算相同不代表两个起点的优化器状态相同。

| 追加步数 | 完整留出 | Greedy对原R43 | S1024/N1对原R43 |
| ---: | --- | --- | --- |
| 0 | 是 | 无自战 | 无自战 |
| 1000 | 是 | 200pair | — |
| 6097 | 是 | 200pair | 200pair |
| 12194 | 是 | 200pair | 200pair |
| 24388 | 是 | 200pair | 200pair |
| 48776 | 是 | 200pair | 200pair |

每1000步与上述节点保存checkpoint、AdamW及RNG，记录逐步CE/KL、value MSE、熵、裁剪、lr、来源计数、引用账本和耗时；每代固定512个Train／512个留出局面做诊断。完整留出覆盖4417局全部357622局面，另保存逐局充分统计、每代及前中期／后期指标（后期为双方最高分达到10）。最优固定子集KL节点若不在日程中，训练结束还会补测其完整留出；这不自动决定棋力选点。Train曲线是固定子集，完整留出用于泛化评估。

本流程直接复用原R24的训练和评测实现，仅使起点轮次、输出目录及训练脚本位置可配置；原R24默认配置和既有评测配置保持一致。共享源码与SHA保存在`shared_source/`及[config.json](config.json)。没有新增内存RSS上限或磁盘门槛。

## 启动、恢复与结果

从仓库根目录执行：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u artifacts/entity_attention_r43_history_pool_20261012/run.py run
```

同一命令恢复整条链路：已完成的前置对战和面板跳过，未完成对战从已保存对局恢复；训练恢复最近完整checkpoint，并按原机制截去未提交日志／引用尾部。SIGINT／SIGTERM转发给当前子进程，训练在当前更新完成后保存。队列和训练分别有锁，防止此实验重复并发写入。`train`仍先完成前置对战，然后只训练；`run`再接后续评测。

查看简要状态：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python artifacts/entity_attention_r43_history_pool_20261012/run.py status
```

正式日志：`runtime/entity_attention_r43_history_pool_20261012/chain.log`。新模型和精确训练账本在`runs/entity_attention_r43_history_pool_20261012/`。前置对战结果在[独立结果表](../entity_attention_r24_offline_vs_r43_20261012/results.md)，新训练拟合及对战在本目录[结果表](results.md)，全部阶段完成后才能判断实际收益。

有限验证见[verification.json](verification.json)：两局临时CPU数据上的4次更新验证R43权重／AdamW继承、轮次元数据、连续2步与1＋1续跑逐位一致、完整留出计数和各评测配置。它不代替正式GPU训练或棋力结果。按前次同机实测，完整链路预计约3–4小时，实际以保存耗时为准。

本实验没有同预算6000局对照，不能单独区分扩池与增加训练量的贡献；DEV最高分有选点偏差，主要收益判断看独立500pair确认，而不是最低KL或最后一步。
