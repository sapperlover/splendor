# 第17–19轮：原轮内留出预选EMA与第19轮直接对战

存储更新（2026-10-21）：第12–19轮168个EMA来源／轮起点及9个EMA导出模型已完整外置至[20261021包](../repository_archive_20261021/README.md)，原路径保留链接。需要重新构建EMA、加载导出模型或重跑本专题时，先按该批说明解压恢复；原始指标、对战结果与源SHA不变。

2026-10-20，用户授权实现并后台启动，现已全部完成。统一选出半衰期1000步：第19轮EMA对同轮raw的greedy为49.70%（1000pair，95%区间[47.75%,51.675%]），双方S1024为48.45%（500pair，[45.40%,51.50%]），未确认棋力收益；17–19轮留出KL和MSE均小幅下降。详见[完整结果](results.md)、[完成产物](complete.json)与[live.json](live.json)，启动身份见[launch.json](launch.json)。

## 冻结协议

- 源分支为`runs/entity_attention_37423_random_budget_p6k_v24k_20261019`，只读原模型及留出归档，不修改正在进行的采集、训练、配置或teacher。
- 从第12轮训练前的48423累计步模型初始化EMA，按第12–19轮160份、间隔50步的checkpoint累计。第17–19轮各保存三组EMA，半衰期为250／500／1000次训练更新；EMA旧权重系数为`2^(-50/H)`，分别约0.870551／0.933033／0.965936。初始化残留权重、全部来源SHA和时间步在清单中记录。
- 使用CPU float64累积、float32导出，平均完整网络，包含共享encoder、policy和value。**这是稀疏端点EMA近似，不是精确逐训练步EMA**。跨轮检查权重／累计步连续，`initial.pt`与`final.pt`别名不重复计数。
- EMA导出采用现有只读模型schema，没有AdamW或RNG，不是可续训checkpoint。正式对照使用原`step-000001000.pt`，其文件SHA与原验证记录一致，并检查它与`final.pt`参数SHA相同。
- 三轮各用各自原有2048条`validation_references.json`，保留顺序和重复抽样；原始标签、policy权重和整局validation划分不变。CPU FP32前向，输出KL／CE／MSE／熵／首选一致率／value偏差、标准差、相关性。各轮raw结果核对原GPU验证记录，容差5e-5；额外与现有learner联合forward损失核对。
- 统一选择**三轮相对raw的Δ(KL+MSE)等权平均最低**的一个半衰期，精确相同时选更短半衰期。每轮不各挑一个参数，也不跨轮挑有利模型。无论预选结果是否优于raw，都继续执行预定直接对战，避免结果依赖的跳过。
- 逐轮EMA−raw的变化区间按原始整局配对bootstrap4000次，seed=`39530000+轮次`。各轮数据有交集，不能把三轮当作独立训练重复。留出用于DEV预选；统计不等于棋力证据。

## 串行正式对战

仅比较**选中半衰期的第19轮EMA vs 第19轮raw**，不是对R43，也不是三个半衰期轮流测棋力后挑最高分。

| 面板 | 规模 | 开局seed | Policy seed base | Bootstrap seed |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 1000pair／2000局 | 39500000起 | 39510000 | 39520000 |
| 双方S1024/N1 | 500pair／1000局 | 39600000起 | 39610000 | 39620000 |

每pair同种子交换先后手。沿用已实现的确定性循环裁决、工程截断排除和完整pair要求。搜索无根噪声、T0；执行参数从准备时的采集配置冻结，12 worker／1000活跃局／B500／native CUDA Graph，owner为单槽。详细值以[冻结采集配置](source_config.json)和生成的`search_config.json`为准。不同阶段串行运行，不中断用户原有进程。

## 运行与恢复

仓库根目录、conda `splendor`：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u artifacts/entity_attention_ema_17_19_20261020/run.py run
```

相同命令恢复：校验冻结脚本、配置和输入；已有EMA和完成留出复用，未完成留出重算。正式对战复用已有逐局sidecar，仅补未提交完整pair。不要与同目录的活动进程同时启动。`prepare`只冻结输入，不开始网络评估或正式对战。

重型EMA权重、原始arena汇总及可恢复逐局sidecar在`runs/entity_attention_ema_17_19_20261020/`；stdout日志在`runtime/entity_attention_ema_17_19_20261020/run.log`。本目录保存冻结计划、来源与模型SHA、逐局充分统计NPZ、预选数据、对战配置、逐pair结果、关键summary和源码身份；在线学生及其AdamW保持原位。

## 有限验证

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q tests/test_checkpoint_ema_probe.py
conda run -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 python \
  artifacts/entity_attention_ema_17_19_20261020/smoke.py
```

7项回归验证半衰期按更新次数计量、跨轮不重复、原留出身份／重复抽样、统一参数选择、只读模型导出和正式对战配置。真实smoke检查第19轮raw与三组EMA各64条原留出样本，对照现有learner指标，并各做3个native S2根；不作为棋力或吞吐结论。详见`tests.xml`、[smoke.json](smoke.json)与`verification.json`。

本实验只检验既有训练轨迹的EMA推理模型；即使结果为正，也不直接证明使用EMA教师继续自迭代会更好。
