# L0／L12 policy・value 嫁接：S128N1 对 raw R37

2026-11-04。目的：把在线第12轮（L12）相对离线起点（L0，140221步）在 S128N1 下的棋力变化，拆成 policy 与 value 两部分。只做推理期组合，不训练、不改变任何在线或离线状态。结果见 [results.md](results.md)。

状态：complete，三组各 2000/2000 pair，异常／非法行动／工程截断／不变量失败均为 0，总耗时约 54 分钟。

**结论**（事实）：L12 64.550%、C1 62.550%、C2 61.687%，L0 参照 57.098%。总收益 +7.45pp；单换 policy（C1−L0）+5.45pp，单换 value（C2−L0）+4.59pp，C1−C2 +0.86pp [-1.20, +2.90] 不显著；交互项 −2.59pp [-5.08, -0.10]。在新 policy 上再换新 value +2.00pp [-0.01, +3.98]，在新 value 上再换新 policy +2.86pp [+0.90, +4.83]。

**推断**：在线12轮的收益并非集中于某一分支，policy 与 value 改进幅度相当，且大部分重叠（次可加）——搜索下任一分支变好即可弥补另一分支的相当一部分误差。交互项含非配对 L0，区间上沿贴近 0，只算弱证据。单一训练轨迹、单一对手与预算，未验证能否推广到其他轮次或预算。

## 模型

| 名称 | policy 来自 | value 来自 | checkpoint |
| --- | --- | --- | --- |
| L0（参照，不重跑） | L0 | L0 | `runs/entity_attention_128x2_opponent_policy_r21_r37_20261024/checkpoints/step-000140221.pt` |
| L12 | L12 | L12 | `runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt` |
| C1 | L12 | L0 | `runs/graft_policy_value_l0_l12_20261104/compositions/C1_policyL12_valueL0.pt` |
| C2 | L0 | L12 | `runs/graft_policy_value_l0_l12_20261104/compositions/C2_policyL0_valueL12.pt` |

对手为 raw R37（`runs/entity_attention_r31_10k_p60k_iteration_20261023/rounds/round-005/checkpoints/final.pt`）。SHA 冻结在 [plan.json](plan.json)。

组合模型保留两套完整 encoder，policy 分支给出先验，value 分支给出叶子估值；不移植 head。两者的对手 policy 辅助头随权重保存，推理不使用。在3局第13轮采集 validation 对局的开头／中段／结尾共9个真实局面上，组合输出与各自来源逐值相等（rtol=atol=0），见 [composition_verification.json](composition_verification.json)。

## 配置与参照

- 模板直接取 [L0 的 10k 面板配置](../offline5_vs_r37_10k_20261027/configs/08_search_policy_vs_R37.json)，只替换 policy_a 的 checkpoint、名称、metadata，并截取前 2000 个开局：种子 47270000–47271999，policy 种子基 47300000，搜索种子 47330000，bootstrap 种子 47340000。
- 双方 S128N1（每世界128×1世界＝总128），配置A：CUDA pool、12 search worker、1000活跃局、B500、local128、owner 双槽、B 桶[64,128,256,500]、C 桶[32,48,64]、等待5ms、shared slots2、FP32／native-cudagraph；无根噪声、T0、FP/PTP 关闭、`c_puct`=1.5、12贵族。
- 每组 2000 pair／4000局，同开局交换先后手，严格串行：L12 → C1 → C2。
- L0 参照：同模板 10000 pair，57.0975% [56.420%, 57.783%]，不重跑。

## 执行源码

评测在 `runtime/graft_policy_value_l0_l12_20261104/source/` 中运行：整体复制 L0 面板的冻结源码（`runtime/offline5_vs_r37_10k_20261027/source/`，含当时的原生扩展），只替换 `src/splendor/entity_attention.py`。替换的原因是旧组合类只能装默认配置的 `entity_attention_v1`，而 L0／L12 是带对手 policy 辅助头、开启市场关系的 `entity_attention_continuation_v1`。本次让 `EntityAttentionPolicyValueConfig` 可选携带 `policy_branch`／`value_branch` 配置；省略时仍为原默认分支，旧组合 checkpoint 加载不变。相对参照源码的差异清单与逐文件 SHA 见 [source.json](source.json)。

这样 L12／C1／C2 与 L0 参照共用同一搜索、评测和原生代码，不受 2026-10-31 之后工作区 `native_search.cpp` 未提交改动的影响。

## 统计口径

- 各组得分与95%区间：整 pair bootstrap 10000 次。
- 与 L0 的差（总收益 L12−L0、policy 收益 C1−L0、value 收益 C2−L0、交互项 L12−C1−C2+L0）：候选按整 pair bootstrap，L0 用 10k 面板区间，按非配对正态合成。此前面板中配对只压低 1–5% 方差，非配对损失很小；L0 原始逐 pair 明细已归档至 T7，解包后可改为严格配对。
- 三组候选之间（L12−C1、L12−C2、C1−C2）共用开局，按整 pair 配对 bootstrap，种子 47340001。
- 单一训练轨迹、单一对手和预算，未作多重比较校正。

## 命令

```bash
bash scripts/run_graft_policy_value_l0_l12_20261104.sh prepare   # 生成组合／配置／冻结源码，不对战
bash scripts/run_graft_policy_value_l0_l12_20261104.sh smoke     # C1 两 pair 冒烟，仅写 runtime
bash scripts/run_graft_policy_value_l0_l12_20261104.sh run       # 正式三组；中断后重跑同一命令按完整 pair 续跑
bash scripts/run_graft_policy_value_l0_l12_20261104.sh status
```

`run` 要求冒烟已通过。正式面板写入 `panels/`，逐局 sidecar 在 `panels/*.json.games/`；日志、live 状态与冻结源码在 `runtime/graft_policy_value_l0_l12_20261104/`，执行会话记录在 [execution.json](execution.json)。

## 验证

- `tests/test_policy_value_composition.py` 新增带辅助头、市场关系分支的组合测试：eager 与 compiled 核心输出逐值等于来源；非法分支架构明确报错。连同 entity attention／continuation 相关共 80 项通过。
- 冒烟：C1 两 pair 走完整配置A路径，检查有效局、换先后手、双方实际 S128N1 预算与 checkpoint SHA，结果见 [smoke.json](smoke.json)。
