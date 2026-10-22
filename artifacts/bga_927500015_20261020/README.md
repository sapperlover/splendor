# BGA #927500015：对局与本地模型输入

已完成抓取和重建。VnEpCxmR（先手/seat 0）15:13 战胜 FourDimensional（seat 1）；双方各 28 回合、15 张已购发展卡。

## 直接查看

- `game_process.md`：完整 56 个行动的中文过程、实际支付和逐步比分。
- `table_metadata.json`：玩家、胜负、桌号、配置及采集范围。
- `actions.json`：结构化主行动、支付、贵族及源页面时间。

## 模型输入

- `model_decisions.jsonl`：88 条已通过 `compact_trace.decode_request` 的原生 `DecisionRequest`，每条包含完整 Observation、现有引擎枚举的全部合法动作、实战选择及候选索引、玩家视角胜负标签。
- `model_features.npz`：现有 `features.encode_decision` 生成的全部特征数组。使用 `numpy.load(..., allow_pickle=False)`；每个样本的数组以 `000/global_features`、`000/candidate_features` 等方式命名。标签位于 `labels/`。
- `features_manifest.json`：数组名称、shape、dtype 与标签说明。
- `model_requirements.json`、`model_contracts/`：与当前本地代码核对的输入契约及来源指纹。
- `catalog/`、`card_id_mapping.json`：本地 90 张卡、10 张贵族、规则常量副本及本局 BGA ID 映射。

样本包括 56 个主行动、30 次支付、2 次贵族领取；自动领取也有记录并标记 automatic。本局没有归还筹码决策。玩家实战选择和最终胜负可作行为/结果标签；没有 MCTS 访问分布或教师 value，不能直接当作原项目的自对弈训练分片。

本人盲保留牌从另一位玩家的本人重播视角确认；对手盲牌在 Observation 中始终隐藏。模型输入只由原项目 build_observation / encode_decision 构造；每个决策都验证了改变未揭示牌堆顺序不会改变 Observation 或合法候选。

## 原始记录与核验

- `raw_replay_logs.json`：56 个行动的原始 DOM 文字、筹码图标标记、公开保留目标与时间。
- `raw_board_snapshots.json`：57 份实际采集的棋盘快照（初始 + 56 个行动后）。初始及早期读取方式与后续略有差异，保留原样。
- `blind_reservation_owner_view.json`：行动 13 后的本人私有手牌。
- `recheck_after_move_53.json`：另一视角对行动 53 的独立复核；此时 BGA 步数栏暂为空，以定位目标和完整状态交叉确认。
- `capture_normalization.json`：动画延迟修正及早期未更新加成属性的说明；原始快照未覆盖。
- `normalized_board_snapshots.json`：统一格式及动画完成后的局面，用于重建和核对。
- `validation.json`：所有主行动后的筹码、市场、牌堆数量、玩家资源和分数核验，以及完整 GameEngine 接口重放结果。
- `final_board.png`：终局网页证据。

实际 BGA 洗牌种子与未揭示牌堆顺序无法从重播确认，模型输入也不需要这些数据。`reconstruction.json` 只为规则重放提供一种与全部已见抽牌一致的未揭示牌堆补全；剩余顺序是人为规范化排列，不能称为 BGA 原始洗牌。它含用于复核的未来牌，不能传给模型。未伪造可从本地 seed 重放的标准 replay。

## 重建

`rebuild_dataset.py` 读取同目录原始文件，并调用 `/home/lc/下载/splendor/src` 当前实现重建导出；本次没有修改原仓库或启动训练。复现时需保留对应源码/数据版本（指纹见 model_requirements.json），使用 splendor Conda 环境：

```bash
conda run -n splendor python rebuild_dataset.py
```

`SHA256SUMS.json` 记录本包文件校验值。页面要求会员的被遮挡统计没有绕过；与模型有关的统计已从完整过程重建。源页面的展示时间没有擅自换算时区。
