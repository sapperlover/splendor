# CPW k=0.25、α=0.5：同模型100 pair棋力对照

2026-10-07。100 pair／200局全部完成。CPW 73胜、2和、125负，得分37.00%，按pair抽样的95%区间[30.75%,43.25%]。**当前S1024的CPW仍明显弱于旧确定化，不建议直接替换正式教师。**

双方同一16500步冻结权重，仅搜索算法不同；对手是旧native S1024/N1确定化，均无根噪声、行为温度0。对手不是θ5或PPO5.5M。这次没有更新模型。

## 胜负与配对比较

| CPW位置 | 局数 | 胜 | 和 | 负 | 得分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 先手 | 100 | 30 | 1 | 69 | 30.50% |
| 后手 | 100 | 43 | 1 | 56 | 43.50% |
| 合计 | 200 | 73 | 2 | 125 | 37.00% |

每个pair是同一游戏seed交换先后手的两局平均得分，胜1、和0.5、负0；100个pair、10,000次bootstrap、seed16790000。不是把200局当独立样本。

[上次不扩宽随机节点](../entity_attention_chance_strength_20261007/README.md)对同一个旧native对手为31.25%。本次CPW变化+5.75个百分点，按相同100个pair配对重采样的95%差值区间[-3.75,+15.00]个百分点；配对差值区间跨0，不能确认相对不扩宽版本的棋力改善。同seed pair中41提高、31不变、28降低。

这份差值比较是两次对同一对手的固定面板，不是CPW与不扩宽直接对打。同游戏seed并不保证对局经过同一状态，选招变化会改变后续轨迹。结果限定于这一模型、预算和种子面板，不证明所有模型／预算都相同。

## 搜索统计与稳定性

以下为本次完整对局两方各自经过的状态汇总；不同于固定200根的配对诊断，不能独立解释胜负因果。

| 指标 | CPW | 旧确定化 |
| --- | ---: | ---: |
| 可搜索决策 | 6481 | 6566 |
| 总simulation | 6636544 | 6723584 |
| transition cap命中 | 0 | 0 |
| 平均逻辑深度 | 3.139 | 4.805 |
| simulation命中精确终局比例 | 3.6946% | 6.9081% |

逐局预算审计确认每个可搜索决策均S1024；可搜索决策数量差异来自对局动作。逻辑深度不计folded forced动作。此前不扩宽完整对局深度2.454，旧4.808，跨面板的状态分布不同，不能把本次均值变化全归因于扩宽。

[本轮固定请求A/B/C/D](../entity_attention_chance_pw_noise_20261007/README.md)显示：CPW相对不扩宽深度提高约32%，但C的JS0.00538→0.02664、D0.03072→0.04835；D Top-1一致率85.20%→76.43%。相对旧确定化，C/D JS仍分别减少54.6%／43.0%。支持集近似交换了深度和抽样稳定性；棋力按本节独立判断，不能用JS或KL替代。

本次评测墙钟599.06秒（9.98分钟），200局并发共享一个模型owner；上次不扩宽同面板551.37秒。这是两种搜索混合对打的整场耗时，还受对局长度／状态差异影响，不是CPW单独采集吞吐，也不是相同墙钟预算棋力赛。逐job total_seconds为重叠作业时间，不能直接作墙钟比较。

全部逐局状态计数为`{"natural_terminal": 200}`；工程错误项：异常0、非法动作0、不变量失败0、截断0、周期裁定和局0。

## 配置与身份

- 同checkpoint：`runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1/rounds/round-019/checkpoints/final.pt`，SHA256 `dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4`。同一冻结GPU模型实例，未训练。
- A为native_chance、chance_widening k0.25/alpha0.5；B为旧native，不扩宽。其余双方S1024/N1、c1.5、fold forced、cap512、root noise0、温度0。CPW暗牌仍每simulation重采，N1是一个共享统计池。
- 100个游戏seed16770000–16770099，每seed两局交换先手。搜索base16740000与策略seed`16780000 + 2*pair_id + player_index`异或，同pair两局保留同策略seed。与此前不扩宽面板相同。
- RTX4050 Laptop 6GB、FP32 eager、TF32关闭、PyTorch/OMP/MKL线程1、native_batched、active_games200、max_batch_size200、候选桶16/32/48/64、actor_projection关闭、同步单buffer。
- 真实对局使用Python checked规则和实际游戏牌堆；随机节点／CPW只影响模拟。max_completed_turns400，周期裁定入口沿用现行协议，实际使用数量见上文。

## 产物和复核

[evaluation_config.json](evaluation_config.json)、[manifest.json](manifest.json)保存完整可执行配置、命令、模型／源码／native二进制及输入输出SHA；[evaluation.json](evaluation.json)含全部200局、pair、预算、bootstrap和执行统计；[analysis.json](analysis.json)从100个原始JSONL侧文件逐项复算，并保存相对上次面板的配对区间。环境见[environment.json](environment.json)。

没有修改生产源码，沿用此前已通过45项相关回归的CPW实现。本次必要验证为6400次固定请求、实际200局checked对打、逐局seed／席位／预算／状态／完整汇总复算和源码／原生扩展快照SHA审计。正式训练教师未切换，长期采集／训练收益和等墙钟对打未验证。

运行日志、源码和native快照在忽略的`runtime/entity_attention_chance_pw_strength_20261007/`，相邻`evaluation.json.games/`侧文件按现行规则忽略；未改忽略规则。模型／runtime／侧文件需另行备份；summary本身含全部200局结果。旧实验原始记录保持原位。

首次执行命令：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_chance_pw_strength_20261007/run.py \
  --config artifacts/entity_attention_chance_pw_strength_20261007/evaluation_config.json \
  --output artifacts/entity_attention_chance_pw_strength_20261007/evaluation.json
```

已存在summary用`--resume`，完成pair会跳过；不要覆盖或换配置复用目录。只读复核：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_chance_pw_strength_20261007/analyze.py
```

`analyze.py --write`只用于首次写分析；默认只读复算、核对全部输入输出哈希及保存的analysis。
