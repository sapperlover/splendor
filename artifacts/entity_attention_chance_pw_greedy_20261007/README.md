# CPW k=0.25、α=0.5：对同权重greedy的100 pair

2026-10-07。100 pair／200局全部完成：CPW 145胜、1和、54负，纯胜率72.50%，计和局半分的得分率72.75%，配对95%得分区间[66.50%,78.50%]。**当前冻结模型和S1024预算下，CPW明显强于同权重greedy。**

这是搜索收益的直接对照：双方同一16500步attention checkpoint。A用CPW S1024/N1、k0.25/alpha0.5；B使用现有GreedyActorCriticPolicy，直接取合法候选logits argmax、零搜索模拟。CPW无根噪声、温度0。没有重新训练模型。

## 胜负

| CPW位置 | 局数 | 胜 | 和 | 负 | 得分率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 先手 | 100 | 75 | 0 | 25 | 75.00% |
| 后手 | 100 | 70 | 1 | 29 | 70.50% |
| 合计 | 200 | 145 | 1 | 54 | 72.75% |

胜1分、和0.5分、负0分；每个pair以同一游戏seed的先后手两局平均得分为bootstrap单位。100个pair、10,000次重采样、seed16790000，不把200局当独立样本。完整pair得分分布见analysis。结论限定于这一模型、搜索参数及种子面板，不能推广到训练后的泛化收益。

状态计数为`{"natural_terminal": 200}`；异常0、非法动作0、不变量失败0、工程截断0、周期裁定和局0。周期和局是现行协议认可的得分结果，与异常／工程截断区分。

墙钟406.17秒，约6.77分钟。200局并发共享一份冻结模型；native batch只包含CPW搜索，greedy逐请求eager推理。greedy共8098次forward、累计25.57秒，在execution_stats中单独计数；搜索inference统计不含这些裸网请求。不是同墙钟预算对打或独立采集吞吐测量。

## 配置与控制

- checkpoint：`runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1/rounds/round-019/checkpoints/final.pt`，SHA256 `dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4`。双方相同内容身份、共用冻结模型。
- CPW：native_chance、S1024/N1、c1.5、fold forced、transition cap512、root noise0、温度0；未来物理抽牌扩宽，当前对手暗牌仍每simulation重新抽样。greedy不使用value选招，不搜索。
- 游戏seed16770000–16770099，与此前CPW对旧搜索面板相同；每seed交换先手。搜索base16740000与策略seed`16780000 + 2*pair_id + player_index`异或。seed相同不保证更换对手后的状态轨迹相同，不将两场得分相减解释为裸网与旧搜索的直接比较。
- CUDA FP32 eager、TF32关闭、CPU/OMP/MKL线程1、native_batched owner1、active_games200、max_batch_size200、候选桶16/32/48/64、actor_projection关闭。greedy沿用单请求eager路径。实际环境见environment.json。
- 真实游戏走Python checked规则和真实牌堆，max_completed_turns400，保留现行周期裁定；逐局状态见上文。

## 调用链和验证

原native_batched评测只接受双方搜索。为本次直接对照，evaluation.py的策略校验和evaluation_batch.py的协程推进新增greedy支持，沿用已有GreedyActorCriticPolicy，零模拟，不用小预算MCTS替代greedy。新分支把裸网开销独立记录，异常交给现有checked runner。两边搜索的路径和C++算法未改，native二进制SHA与此前CPW版本相同。正式训练配置／教师未切换。

35项相关测试通过：新增4项CPU serial与batch的完整对局结果／终态一致性测试，覆盖native／CPW和greedy在A/B两侧；另1项greedy异常归因测试。包含既有搜索批次、恢复、截断、cycle和chance回归。真实checkpoint的四局S16 GPU smoke全部自然终局，零异常／截断／cycle，排除在200局棋力统计之外。正式200局逐局seed、席位、CPW预算、greedy空搜索stats、模型／source／native哈希及完整summary／bootstrap复算通过。

[tests.json](tests.json)记录测试命令、源码SHA及日志；[smoke_config.json](smoke_config.json)和[smoke.json](smoke.json)保存短验证。旧实验原始产物保持原位。[此前CPW对旧搜索](../entity_attention_chance_pw_strength_20261007/README.md)为37.0%（区间30.75%–43.25%），是另一对手的面板；[JS复测](../entity_attention_chance_pw_noise_20261007/README.md)测的是目标稳定性，均不能替代本次直接对greedy结论。

## 产物与命令

[evaluation_config.json](evaluation_config.json)、[manifest.json](manifest.json)保存完整配置、命令、模型／源码／native快照与输入输出SHA；[evaluation.json](evaluation.json)保存全部200局、pair、搜索预算、bootstrap和执行统计；[analysis.json](analysis.json)从100个原始pair JSONL复算。[run.py](run.py)沿用此前FP32／线程控制和现行fixed评测入口。

日志、源码和native快照在忽略的`runtime/entity_attention_chance_pw_greedy_20261007/`，相邻`*.json.games/`侧文件按现行规则忽略，未修改忽略规则。模型／runtime／侧文件需另行备份，summary本身已保存所有正式对局记录。

首次运行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u artifacts/entity_attention_chance_pw_greedy_20261007/run.py \
  --config artifacts/entity_attention_chance_pw_greedy_20261007/evaluation_config.json \
  --output artifacts/entity_attention_chance_pw_greedy_20261007/evaluation.json
```

已有summary用`--resume`跳过完整pair；不要覆盖或换配置复用目录。只读复核，不执行搜索／训练：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python artifacts/entity_attention_chance_pw_greedy_20261007/analyze.py
```

`prepare.py`仅首次冻结，禁止覆盖已完成目录；`analyze.py --write`只用于首次生成analysis，默认只读复算和核对所有SHA。
