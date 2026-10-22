# 256×4 双辅助头：R24–R43 固定池离线训练

入口为 [run.py](run.py)，配置为 [offline_256x4_aux_r24_r43_20261014.json](../../configs/entity_attention_v1/offline_256x4_aux_r24_r43_20261014.json)。训练复用 `splendor.offline_objectives` 的历史池、整局划分、抽样账本和 CompactGameCache；随机初始化与百分比验证由 `splendor.offline_scratch` 提供。

## 运行与续跑

从仓库根目录执行，下列同一命令会从最近已提交 checkpoint 续跑，然后选点、串行执行两组正式对战：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u artifacts/entity_attention_256x4_aux_r24_r43_20261014/run.py run
```

`train` 只训练并选点；`prepare` 只冻结计划；`status` 刷新汇总。正式配置、池和驱动 SHA 已在 [plan.json](plan.json) 冻结，不能改参数后复用同一输出目录。每次执行的代码快照、命令、身份和退出状态见 [execution.json](execution.json) 及 `runtime/entity_attention_256x4_aux_r24_r43_20261014/`。阶段日志追加保存，中断后同一命令恢复；训练从最后一次 checkpoint 提交点恢复，提交后的未保存更新会重新执行。

## 配方与验证日程

- 256维、4层、8 heads、FFN1024、dropout0；带双辅助头共 **3,761,156** 参数。主网络从随机初始化 seed36500000 开始，辅助头 seed36100000；全新 AdamW。原 R43 为对手，尺寸不同的旧权重没有用于初始化。
- 原 R24–R43 的 20,000 局；Train 17,971局／1,451,525局面，整局留出 2,029局／163,595局面。原始访问分布和终局结果保持为监督，未重新标注。固定4096验证局面与原双辅助头实验完全相同。
- Policy Surprise Weighting 关闭；policy／胜负value 权重各1；终局分差和剩余玩家回合两辅助目标各0.25，分别除以15／40。辅助头通过共享 encoder 的梯度帮助主任务，对战仍只使用 policy 和胜负value。
- lr2e-5、B256、weight decay1e-4、clip1、FP32、8GiB compact cache；抽样 seed34410000，验证 seed34420000。单池均匀有放回采样，`history_batch_size` 在本入口不生效。
- 抽样曝光12次，共 **68,041步**，约为原离线实验22681步的3倍。每1000步仅保存恢复 checkpoint；每5%额外保存 checkpoint、测固定4096与完整留出，不再每1000步评估。
- 20个完整验证点：3403、6805、10207、13609、17011、20413、23815、27217、30619、34021、37423、40825、44227、47629、51031、54433、57835、61237、64639、68041。初始化点仅测固定4096。
- 训练完成后，从上述20个完整 checkpoint 中取 **完整留出 policy KL＋胜负value MSE** 最小者，同分取更早步数；辅助loss不参与选点，不拼接不同 checkpoint 的 policy/value。

原 R43 对手 SHA256：`b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14`。greedy 为1000pair／CPU8worker，开局seed36600000–36600999；搜索为双方 native S1024/N1、c_puct1.5、根噪声0、T0，200pair／CUDA compiled，开局seed36700000–36700199。每个pair同种子交换先后手，和棋半分，整pair bootstrap10000次。比赛结果不参与选点。

## 本次交付核验与已有执行状态

[tests.xml](tests.xml)：28项相关CPU测试通过，覆盖百分比保存／完整留出、恢复后权重和优化器逐位一致、完整验证选点、辅助配置、随机初始化及棋力面板配置；测试为小数据工程验证。核对记录见 [delivery_check.json](delivery_check.json)。这次脚本交付没有代为执行或恢复正式实验。

交付收尾时发现已有用户启动的正式任务，并已中断：训练日志最后到44,448步，最近保存点44,227步。完整留出value MSE在30,619步为0.765951，到44,227步回升至0.797854；policy KL同期0.481831→0.458084。当前已完成点中，KL＋MSE最小为37,423步的1.243087，尚非最终选点。原始曲线与验证 JSON 在 `runs/entity_attention_256x4_aux_r24_r43_20261014/`；自动终点棋力面板尚未执行。另行完成的34,021步1000pair greedy见 [中期结果](../entity_attention_256x4_aux_r24_r43_early34021_20261014/results.md)，不能替代最终选点与搜索面板。

首次执行期间，中断前没有写出`progress.json`，旧results／summary因此曾显示prepared_not_started、0步。2026-10-19整理已按最近完整checkpoint44227、原完整留出和runtime中断记录修正[results.md](results.md)／[summary.json](summary.json)的派生状态，原数值未变。代码交付时已修正checkpoint后的进度记录，但本次没有调用训练／status入口或回填训练原始记录。各会话代码身份分别保留；重新恢复须先核对本次归档挂载。

产物位置：训练checkpoint、抽样账本、训练日志、`validation_curve.jsonl`、`full_validation/step-*.json`、`selection.json` 在上述 `runs/`；正式对战配置、逐pair汇总在本目录的 `evaluation_configs/` 与 `panels/`；阶段日志和临时文件在 `runtime/`。训练曲线用于拟合诊断，单个中期节点不能证明整个大模型方案无效。

## 2026-10-19冷产物位置

非主要checkpoint、逐步指标及抽样账本已集中到`archive/20261019/runs/`下相同run结构；关键模型与验证／选点文件留在原位置，原文件路径保留相对链接。见[逐文件迁移清单、保留策略及恢复说明](../repository_archive_20261019/README.md)。只读诊断可在恢复挂载后访问；精确续训前须按清单撤回该run的迁移，恢复原位真实账本／日志和权重，避免续训改写冻结归档。本次没有启动或恢复训练。
