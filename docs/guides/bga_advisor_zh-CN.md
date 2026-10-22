# BGA 只读招法推荐器

当前阶段（2026-10-21）：固定脚本读取 BGA 双人基础版的实时局面，用本地混合预算第20轮权重给出主行动、支付组合与弃筹码推荐，由用户手动执行。默认原生搜索 S16384/N1。支持已有 `DecisionRequest` 的离线推荐；不提交游戏动作，不自动对弈。

## 实时使用：连接已有 Chrome

先在平时使用的 Chrome 中打开已登录的 BGA 牌桌，然后在项目根目录运行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u -m splendor.bga_live
```

不需要输入桌号，不会打开独立浏览器，不跳转网页，也不复制登录态。只有一张 BGA 牌桌时自动接入；多张时优先当前有焦点/可见的牌桌，无法唯一判断时提示切换标签页，不随机选桌。换桌或刷新后会重新识别并订阅事件。

需要 Chrome 已允许本地远程调试。当前电脑已实测连接成功，使用 Chrome 写出的 `DevToolsActivePort` 中的完整 WebSocket 地址；权限式调试可能没有 `/json/version` 接口，因此不依赖该 HTTP 发现接口。未启用时请自行在 Chrome 的 `chrome://inspect/#remote-debugging` 页面配置；脚本不会替你更改浏览器安全设置，也不会另开浏览器兜底。默认依次检查 Chrome、Edge、Chromium 的本机配置目录；本次只实测 Chrome。也可通过 `--cdp` 指定本地调试地址。

`--url 'https://boardgamearena.com/tableview?table=…'` 仅作为可选的现有牌桌筛选条件，不会导航到该网址。Ctrl-C 只断开脚本和移除它自己的事件监听，不关闭 Chrome、标签页或登录会话。`--once` 输出一次有效估计后断开。

新环境可安装项目可选依赖 `pip install -e '.[bga]'`（Playwright >=1.60），并准备已有原生搜索扩展。本机依赖已安装。

任意轮次均能接入：已知市场保留牌保留身份；无法恢复的保留牌按对应等级盲保留；自己的可见手牌仍完整读取。`history_<桌号>.json` 保存已知公开信息，缺少它也能中途接入，只是少用一部分之前公开过的信息。原有中途接入逻辑保持。

### 实时胜率估计与事件输出

默认每0.25秒读取，并在浏览器内订阅已加载游戏的 `coins`、`buyCard`、`reserveCard`、`revealCard`、`claimNoble` 通知；不提交游戏动作。双方行动和翻牌分别输出，例如：

```text
[23:27:07] 桌934687426 第16步 · 对方购买完成 | 我方胜率估计 60.7% · 对方 39.3%（模型估计，平局折半）
[23:27:09] 桌934687426 第16步 · 翻牌（BGA牌40） | 我方胜率估计 60.7% · 对方 39.3%（模型估计，平局折半）
```

这两行来自实际只读运行。**同一手的动作和翻牌通知分别保留，使用通知处理及动画完成后的稳定局面估计**，所以两行数值可以相同；不会把市场缺一张牌的动画中间状态当成完整局面，也不声称提供“翻牌前”的精确概率。弃筹码阶段允许先评估待归还状态，完成归还后继续更新。浏览器自行缓冲稳定局面，Python 进行搜索期间仍能记录事件。缓冲上限64组，溢出会明确报告；断开连接期间无法保证捕获每条通知。

“我方”始终指当前登录的玩家。本人行动时直接评估本人观察；对手行动时，从我方已知信息采样对方未知保留牌，切换到实际行动者视角评估并取负号还原我方。不会读取对方私有手牌；也不会把同一个 value 在双方视角下原样使用。`--value-samples` 默认4，只在对手存在未知保留牌时使用多次采样。

**这里的胜率估计是 `(clip(V,-1,1)+1)/2`，平局按半分计，严格说是模型期望得分率，并非校准后的真实获胜概率。** 保存原始 value 和计算口径；不把裁剪到100%的估计解释为必胜。对局结束时停止模型估算，实际结果以 BGA 结算为准，包括认输或弃局。

该显示来自value头，独立于搜索预算，不是搜索根Q。主行动、固定购买目标后的支付阶段、翻牌后的局面以及行动方视角包含不同条件，不能直接用相邻显示值之差衡量某一步的收益损失。例：[第20步买061分析](../../artifacts/bga_934713322_move20_061_20261021/README.md)中，主行动为72.06%、选中061后的支付阶段为70.34%、购买与补牌并切换行动方后为54.94%；这些变化同时包含阶段、信息集、翻牌和模型误差。

默认使用混合预算第20轮原始权重、CPU 单线程；主行动、支付和弃筹码建议均使用原生 MCTS **S16384/N1**、seed 0、无根噪声。`--mode greedy` 可关闭搜索；`--simulations`、`--determinizations`、`--top-k`、`--checkpoint` 可调整推荐设置。胜率估计独立于 MCTS，在轮到对手时也会更新。正在运行的旧进程需要停止后重新启动；若命令显式指定旧预算，应删除该参数或改为 `--simulations 16384`。

### 支付组合与弃筹码

- **支付**：在网页选中要买的市场牌或本人保留牌后，脚本进入该目标的支付决策，搜索全部合法的普通筹码／黄金配比。建议显示完整支付总额。网页自动预选尚未扣除真实持有数量；若预选组合与建议不同，需要取消卡牌选择后重新选择。网页的自动支付／保留黄金偏好可能需要由用户自行调整，脚本不会修改它。
- **弃筹码**：拿筹码或保留获得黄金后，网页进入 `playerDiscard` 时给出具体归还组合。已在网页选中的归还筹码视为固定选择，只搜索剩余需归还的部分；输出同时列出已选与本次合计。最后一枚选完后 BGA 会自动提交归还，应先等待建议再选择。
- 脚本通过公开通知记录本手拿筹码／公开保留／盲保留动作，构建原生阶段上下文；没有读取未知未来牌。如果首次接入或刷新时已经处于弃筹码阶段、且缺少本手动作记录，会明确提示该阶段暂不能还原；保持脚本运行，下一次拿筹码或保留后即可推荐弃法。
- 以上阶段同样要求局面稳定，并在搜索结束后重新确认网页未变。输出始终是手动执行建议。

输出位置为项目 `runtime/bga_advisor/`：

- `latest_evaluation.json`：最近一次有效估计及当前登录视角、原始 value、口径与快照哈希。
- `evaluations.jsonl`：逐事件估计日志，动作和翻牌分别一行，含事件时间与对应完整局面哈希。
- `evaluation_snapshot.json`：最近一次估计的可见局面。
- `latest.md`、`latest.json`：当前主行动推荐或等待状态；`current_request.json` 等继续保留。

动画、尚未完成的拿筹码选择及未支持的阶段期间暂停估计和推荐；选中购买目标和弃筹码阶段按上述规则输出。Chrome 关闭后脚本等待重新连接；BGA 网页自身的服务器断线检测仍以页面提示为准，不把 CDP 连通当作 BGA 在线。

### 当前适用范围

- 双人基础版，四个扩展全部关闭；已在中文 BGA 自测桌验证。
- 主行动：拿筹码、公开保留、盲保留、购买市场牌/保留牌；另支持购买目标的支付组合和超限归还组合。具体建议由当前合法候选和模型决定。
- 任意轮次均可首次接入，公共日志不完整也允许继续：已知的市场保留牌保留公开身份，其余保留牌按等级计为盲保留。`history_<桌号>.json` 用于重启后保留已知信息，建议保留；缺少该文件时会重新读取现有日志，其余按盲保留处理。中途接入或丢失历史会使搜索少用一部分此前公开的信息。
- 网页内尚未完成的拿筹码选择期间暂停推荐；卡牌支付与弃筹码按对应原生阶段处理。贵族选择、跳过回合和扩展阶段尚未完成实时接入，脚本会提示手动完成。支付／弃筹码新增路径完成基于保存快照构造局面的有限工程验证，尚未通过操作真实牌桌的完整流程实测；离线推荐器也支持贵族决策。
- 页面断线时以 BGA 连接提示为准；脚本能等待重新连接 Chrome，但没有独立的 BGA 服务器连接健康检测。

### 读取方式与保留文件

脚本使用 Playwright 读取页面已经加载的 Splendor AMD 模块 `modules/spl_game.js`，其中 `gamestorage`、`playerstorage` 随游戏通知更新；`gamedatas` 只提供静态卡表、设置和 BGA 当前阶段。它不靠反复点击“下一步”，不从控制台文本猜局面，不使用私有接口提交招法。

当前玩家自己的保留牌可见；对方只读取公开数量，以及能从公共日志或本地历史恢复身份的市场保留牌，其余按盲保留采样。BGA 客户端会把本人的初始手牌复制进对方的内部 `storedCards` 字段，读取器明确忽略该字段，只读取对方真实的公开数量。

卡牌按等级、加成、分值和费用与本地90张卡精确映射。2026-10-21起，本地规范贵族池统一为12张，包含蓝绿黑333（N11）和红白44（N12），BGA按牌面需求与分值映射到同一稳定ID。直接使用完整规范目录，不再临时生成`BGA_N11/N12`或截成10个定义。无法唯一映射的贵族明确报错，不自动增加规则之外的牌。原模型权重不修改；更新代码和原生扩展后应重启推荐脚本。

`runtime/bga_advisor/` 还保存当前局面、原生请求、卡牌映射、运行时贵族定义和公开保留历史，供排错复核。浏览器登录目录包含敏感会话，不应分享或纳入 Git；整个 `runtime/` 已由既有忽略规则覆盖。

## 模型与离线运行

本次采用混合预算分支第20轮原始模型：

`runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt`

SHA-256：`fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd`。输出包含实际加载路径与 SHA，不随训练自动换权重。

读取旧棋谱首个决策，默认 CPU 单线程：

```bash
conda run -n splendor env PYTHONPATH=src python -m splendor.recommend \
  --request artifacts/bga_927500015_20261020/model_decisions.jsonl \
  --decision-index 0 \
  --checkpoint runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt \
  --mode greedy --top-k 3 \
  --output runtime/bga_advisor/first_decision.json
```

`--decision-index` 为从0开始的文件行号。单局面 JSON 可直接保存 `DecisionRequest`，或含 `request` 的记录；此时不传行号。旧棋谱使用默认目录；实时局面的额外贵族必须连同对应快照/运行时目录一起复核，不能直接拿默认目录代替。

使用原生搜索，把 `--mode greedy` 改为 `--mode mcts --simulations 16384 --determinizations 1 --seed 0`。未知牌只根据当前可见信息采样；不会读取真实未来牌序。需要已有原生扩展，不会自动编译或退回其他实现。

## 输出含义

- `description`、`recommended_action`：中文首选和原生动作；实时输出另含 BGA 卡牌 ID。
- `candidates`：前 K 个候选。greedy 按策略概率排序；MCTS 按访问量、先验和候选顺序排序。
- `policy_probability` 为策略分布，`value_actor_raw` 为原始 value，都不是校准胜率。MCTS 另给访问量和有访问时的 Q。
- `decision_sha256` 绑定完整局面及候选顺序，不能跨局面沿用索引。
- `submitted_action` 始终为 false。模型入口不调用引擎执行动作，浏览器入口只连接已有浏览器、订阅通知、读取及保存结果，不导航、不操作游戏。

## 有限验证

旧棋谱88个greedy首选与既有分析一致；其88个观察重建出的原生请求与全部合法候选完全一致。自测桌 #934625108 验证开局、本人公开保留和对方公开保留后的读取与搜索；在不刷新页面时继续验证了拿筹码、购买、支付扣除、加成和补牌更新；34项相关测试通过。进一步证据见[实时验证记录](../../artifacts/bga_live_advisor_20261020/README.md)。离线历史记录见[离线验证](../../artifacts/bga_read_only_advisor_20261020/README.md)。这些是工程验证，不是棋力评测结论。

2026-10-20 后续修改允许中途接入：以下回归共35项通过，覆盖无日志、部分日志、本人可见手牌、后续补全历史及重启恢复。另用保存的第4轮第8步快照 `tests/fixtures/bga_live_step8_934625108.json`，分别保留和清空公共日志，调用第20轮权重进行 CPU 单线程 S32/N1、seed 0、无根噪声的短搜索；对手保留牌分别识别为3张明保留和3张盲保留，两种情况均完成32次模拟并返回合法建议。该验证读取保存的快照，未操作当前浏览器。

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_bga_live.py tests/test_recommend.py \
  tests/test_search_policy_equivalence.py tests/test_search_policy_hidden_permutation.py \
  -o addopts=''
```


2026-10-20 后续接入已有 Chrome 和实时事件输出：72项相关回归通过，实际桌 #934687426 无 URL 参数自动识别、中途接入、本人/对手视角 value 和购买/保留/拿筹码/翻牌的独立输出已完成只读验证。旧版“启动独立 Chrome”的操作说明由本节替代；测试及原始输出见 `artifacts/bga_existing_chrome_events_20261020/`。该次实测使用当前 Chrome 已登录账号1tude，未点击或提交任何招法。

2026-10-21 支付／弃筹码接入及默认 S16384：有限回归覆盖实际 JavaScript 读取器与事件监听、两种购买来源、普通筹码／黄金支付配比、已选归还、公开／盲保留后弃筹码、对手弃筹码时的 value 视角和默认预算。另从第8步真实快照构造两条明确标注为合成的工程局面，原始第20轮权重分别完成 S16384/N1、seed 0、CPU单线程搜索：支付2个候选，约26.49秒，建议蓝3金1；弃筹码15个候选，约26.92秒，已选绿1后建议再归还绿2。建议均通过 checked 引擎执行验证。未操作真实牌桌，不构成棋力实验。

原始请求、推荐、配置、代码哈希及工程验证脚本保存在 `runtime/bga_advisor/phase_validation_20261021/`，回归结果为其中的 `pytest.xml`。复核命令：

```bash
conda run -n splendor env PYTHONPATH=src python -m pytest -q \
  tests/test_bga_live.py tests/test_bga_watch.py tests/test_recommend.py \
  tests/test_search_policy_equivalence.py tests/test_search_policy_hidden_permutation.py \
  -o addopts='' --junitxml=runtime/bga_advisor/phase_validation_20261021/pytest.xml

conda run --no-capture-output -n splendor env PYTHONPATH=src python -u \
  runtime/bga_advisor/phase_validation_20261021/smoke.py
```
