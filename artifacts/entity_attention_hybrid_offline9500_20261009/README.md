# 离线 9500 步重放、固定留出与 200pair 棋力面板（2026-10-09）

回答“第 1–9 轮混合训练的收益里，有多少是纯离线继续训练的贡献”。三部分：确定性重放出离线 9500 步模型（原 run 只按每 1000 步存档）；在该模型与相邻节点上评固定留出 KL / Value MSE；用同一批 200 个种子做 4 个 S1024/N1 对 θ5000 的棋力面板（离线9500、hybrid 第 3/6/9 轮）。

## 1. 离线 9500 步重放（并校验到 10000）

- 方法：复制 `runs/entity_attention_offline/checkpoints/step-000009000.pt` 与冻结的 `config.json`/`dataset.json`/`fixed_validation.json` 到新目录 `runs/entity_attention_offline_9500_replay/`，用 `splendor.offline_train --resume --max-steps-this-session 500` 重放 9000→9500；再续 500 步到 10000 作为可校验点。
- 校验（重放 10000 vs 原始 `step-000010000.pt`）：模型 `model_sha256` 相同、`state_dict` 逐张量 max abs diff **0.0**、优化器 state 逐张量 max abs diff **0.0**、`param_groups` 相同、torch CPU RNG 相同；9500/10000 在固定离线留出上的 KL 与原 `validation_curve.jsonl` 差 ≤4e-8（CPU/GPU 浮点），9500 KL 0.59406、10000 KL 0.59336。
- 产物：`runs/entity_attention_offline_9500_replay/checkpoints/step-000009500.pt`（同时保留 9000/10000 与重放曲线）。

## 2. 固定留出 KL 与 Value MSE（`holdout_eval.json`）

固定集合与 [KL 趋势实验](../entity_attention_hybrid_kl_trend_20261009/README.md) 完全一致：`offline_fixed`（4096 局面，digest `60df109d…`）与 g1–g9（每代 2048 局面）。KL 越低越接近该集合的存档目标；Value MSE 为加权 MSE。

KL(target‖student)：

| checkpoint | offline_fixed | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| θ5000（起点） | 0.61606 | 0.40633 | 0.47493 | 0.54663 | 0.51613 | 0.51437 | 0.54457 | 0.48994 | 0.53614 | 0.59012 |
| 离线 9000 | 0.59682 | 0.45191 | 0.52008 | 0.57636 | 0.54530 | 0.54671 | 0.56155 | 0.50857 | 0.55035 | 0.60666 |
| **离线 9500** | **0.59406** | 0.46560 | 0.53123 | 0.57883 | 0.55049 | 0.54721 | 0.56820 | 0.51132 | 0.55459 | 0.61539 |
| 离线 10000 | 0.59336 | 0.42960 | 0.49414 | 0.56264 | 0.51523 | 0.52154 | 0.54187 | 0.49405 | 0.52866 | 0.58851 |
| hybrid 第 9 轮 | 0.62585 | 0.37325 | 0.42499 | 0.48381 | 0.43313 | 0.43389 | 0.45190 | 0.41982 | 0.44328 | 0.49223 |

Value MSE：

| checkpoint | offline_fixed | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| θ5000 | 0.77333 | 0.78678 | 0.85334 | 0.78475 | 0.88889 | 0.80957 | 0.85678 | 0.80495 | 0.82424 | 0.81057 |
| 离线 9000 | 0.75422 | 0.74373 | 0.81283 | 0.76812 | 0.87837 | 0.78914 | 0.79705 | 0.77463 | 0.78744 | 0.79755 |
| **离线 9500** | **0.74863** | 0.74162 | 0.79350 | **0.76567** | 0.86274 | 0.78641 | **0.78719** | **0.78535** | 0.79093 | 0.79411 |
| 离线 10000 | 0.80299 | 0.75764 | 0.81655 | 0.78678 | 0.90414 | 0.81514 | 0.81919 | 0.82745 | 0.82327 | 0.81056 |
| hybrid 第 9 轮 | 0.76789 | **0.73410** | **0.79322** | 0.77446 | **0.84580** | **0.79432** | 0.77007 | 0.80631 | 0.79131 | 0.79312 |

（加粗=该列最佳。目标熵 1.41–1.71 不等，绝对 KL 不可跨列比较。）

要点：

- 离线 9500 在旧语料 `offline_fixed` 上最好（0.59406 对 θ5000 0.61606、hybrid 第9轮 0.62585），Value MSE 也是全体最低（0.74863）。
- 但**在所有 9 个在线留出上，离线模型都比 θ5000 更差**（如 g1 0.4656 vs 0.4063、g9 0.6154 vs 0.5901），而 hybrid 第 9 轮明显更好（g1 0.3733）。纯离线继续训会在新在线分布上退化；混合训练在这一点上明确更接近新目标。
- Value MSE 上三者差距较小：hybrid 第 9 轮在 g1/g4/g5 最低，离线 9500 在 offline_fixed/g3/g6/g7 最低；离线 10000 的 offline_fixed MSE（0.8030）反而比 9500 差。

## 3. 200pair 棋力面板（`battle_summary.json`）

四个面板同批 200 个种子（game 14900000+，policy 14910000+），双方 S1024/N1、无根噪声、温度 0；对手均为 θ5000（= 离线 step-5000 权重，二者 69 个张量相同）。执行配置与历史 hybrid 面板一致（native_batched、active 100、B64、compiled）。

| 候选 | 得分（200 pair） | 95% 区间 |
| --- | ---: | --- |
| 离线 9500（重放） | **67.38%** | [62.50, 72.25] |
| hybrid 第 3 轮（6500 步） | 58.88% | [54.00, 63.62] |
| hybrid 第 6 轮（8000 步） | 66.75% | [61.88, 71.62] |
| hybrid 第 9 轮（9500 步） | **74.38%** | [70.00, 78.75] |

同种子配对差（正=前者更强）：

| 对比 | 差值（百分点） | 95% 区间 |
| --- | ---: | --- |
| hybrid r9 − 离线 9500 | **+7.00** | [+0.62, +13.25] |
| hybrid r6 − 离线 9500 | −0.62 | [−7.38, +6.12] |
| hybrid r3 − 离线 9500 | −8.50 | [−14.62, −2.50] |
| hybrid r6 − r3 | +7.88 | [+1.25, +14.50] |
| hybrid r9 − r6 | +7.62 | [+1.12, +14.12] |
| hybrid r9 − r3 | +15.50 | [+9.62, +21.75] |

## 4. 结论（对“离线红利”问题的更新）

1. **到第 3/6 轮为止的收益基本可以用“纯离线继续训练”解释**：离线 9500（完全没有自对弈数据）对 θ5000 为 67.38%，与 hybrid 第 6 轮 66.75% 不可区分（−0.62pp），且显著高于第 3 轮 58.88%（+8.5pp）。同一份旧语料从 5000 步继续训到 9500 步（约 2.4 个 epoch）本身就值这条幅度。
2. **第 6→9 轮出现了纯离线解释不了的增量**：hybrid r9 比同一步数的离线 9500 高 7.0pp（区间不跨 0），且 r9 明显高于 r6（+7.6pp）。结合第 2 节，这一段里混合模型在 9 个在线留出上都明显更贴近新目标，而离线模型持续偏离——在线数据在此阶段确实贡献了棋力，而不只是拟合漂移。
3. 因此更准确的表述是：**前 6 轮≈离线继续训练的延续（红利说成立）；第 6–9 轮开始出现自迭代的独立增量（约 +7pp）**。这与“到第六轮还是在吃离线红利”的直觉一致，但不同意“后续完全无收益”。
4. 与另一条锚点（PPO 5.5M）的旧结论（第 9 轮 63.75% vs 离线 12045 步 59.5%，+4.25pp 跨 0）方向一致：第 9 轮的混合模型略优于纯离线，只是当时样本与对手不同、未达显著；本次 θ5000 锚点、200 配对后达到显著。

## 5. 边界与复核

- 单训练 seed；“离线 9500”是同一离线轨迹的重放点，区间只反映对局抽样与配对差，不覆盖训练随机性。
- 棋力结论限定于 θ5000 这一个对手与 S1024/N1、无根噪声设置；跨对手/跨搜索预算的传递未验证。
- 各代在线留出目标由不同代教师生成（目标熵 1.41–1.71），只比较同一列内部的变化。
- 复原命令（重放与评测已在本地完成，以下为可复算入口）：
  ```bash
  # 重放（GPU；从 9000 到 9500，再可选到 10000 校验）
  conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u -m splendor.offline_train --config runs/entity_attention_offline/config.json \
    --output runs/entity_attention_offline_9500_replay --resume --max-steps-this-session 500
  # 固定留出评测（CPU）
  conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    python -u artifacts/entity_attention_hybrid_offline9500_20261009/holds.py --device cpu
  # 200pair 面板（GPU，4 个串行）
  conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u artifacts/entity_attention_hybrid_offline9500_20261009/battles.py run
  conda run -n splendor env PYTHONPATH=src python artifacts/entity_attention_hybrid_offline9500_20261009/battles.py analyze
  ```
- 产物：`holdout_eval.json`、`battle_summary.json`、`battles/`（含逐 pair 汇总与忽略的 `.games` 逐局记录）；重放 run 在 `runs/entity_attention_offline_9500_replay/`。一次失败的执行尝试（native-cudagraph 在评测中反复编译）移至 `runtime/entity_attention_hybrid_offline9500_20261009/battles_graph200_partial/`。
