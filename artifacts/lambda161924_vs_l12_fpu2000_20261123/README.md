# λ161924的最强历史FPU配置／无FPU配置，对原搜索L12

2026-11-23按用户要求准备两个独立脚本。每组2000pair／4000局，双方S128N1（每世界128×1世界＝总128）；两组共4000pair／8000局。节点和SHA固定，正式评测未启动，不启动训练或队列。

```bash
# λ161924使用历史初筛最高配置：FPU0.3、c_puct1.0
bash scripts/run_lambda161924_fpu_vs_l12_s128n1_2000pair_20261123.sh

# λ161924使用原搜索：FPU关闭、c_puct1.5
bash scripts/run_lambda161924_nofpu_vs_l12_s128n1_2000pair_20261123.sh
```

L12两组始终使用原搜索：`fpu_reduction=None`、`c_puct=1.5`。两组λ模型、L12模型、开局／策略种子、预算、执行配置均相同，仅λ侧搜索配方及名称／说明不同：

| 组 | λ侧FPU reduction | λ侧c_puct | L12侧FPU／c_puct |
| --- | ---: | ---: | --- |
| fpu | 0.3 | 1.0 | None／1.5 |
| nofpu | None | 1.5 | None／1.5 |

本次比较的是整套搜索配方，FPU与c_puct同时变化；可用于评估历史最强配方迁移到λ后的表现。`None`才表示关闭FPU、保留未访问Q=0的原搜索，`0.0`仍会启用父Q式FPU，不能用作关闭值。

## 最强配置的依据

[2026-11-04 FPU／c_puct初筛](../search_fpu_cpuct_l12_20261104/README.md)五组均已完整完成2000pair。`04_2b_fpu0.3_c1`点估计最高：L12新搜索对自身原搜索55.750%，95%整pair区间[54.263%,57.250%]。本次核验其原始结果SHA、4000有效局、passed及零异常／非法行动／工程截断／不变量失败，并复用该配方。

这是此前五组初筛的点估计最高者；原实验r=0.3、c_puct1.0与c_puct1.5的差异未显著。原更高预算／全新种子2c确认的状态继续按原记录解释，本次是λ对L12的新对战。

## 固定模型

| 模型 | 文件 | SHA256 |
| --- | --- | --- |
| 当前归档搜索λ，161924步 | `runs/outcome_60k_scalar_lambda_20261120/checkpoints/step-000161924.pt` | `561f055af56719ad7fbfa9779e93955ce139d5651b6a2b9b9dbc7e10b7182b3b` |
| L12：policy辅助在线分支第12轮原完整final | `runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt` | `2a6304e3470cab3dad78ec0a2789ee892892e741f3731cafc2197cde9bd2d45d` |

L12沿用原FPU初筛和policy／value嫁接实验的同一个完整模型，不使用嫁接模型。评测不执行训练专用辅助头。

## 执行、配对与恢复

沿用配置A：CUDA pool、12 search worker、1000活跃局、B500、local microbatch128、owner双槽、batch buckets `[64,128,256,500]`、candidate buckets `[32,48,64]`、等待5ms、shared slots2、FP32/native-cudagraph、reused传输、fail-fast。无根噪声、T0、FP/PTP关闭、强制阶段折叠、每模拟最多512状态转移；完整12贵族规则、400完整回合工程上限及既有确定性循环判平保持。

两组共享本次新开局55300000–55301999，policy_seed_base55310000，双方search seed55320000，bootstrap seed55330000。每pair同一开局交换先后手，和棋半分，10000次整pair bootstrap报告95%区间。每100pair打印进度、每500pair及结束／正常中断刷新summary；逐局JSONL持续落盘。

从仓库根目录运行，自动使用conda `splendor`和单线程OMP／MKL／OpenBLAS。默认命令同路径恢复：完整面板核验后直接返回，部分面板自动`--resume`，跳过完整pair、按原seed重跑半个pair。保留相邻`.json.games/`目录。两个脚本各自独立锁，失败／中断退出；不自动重试错误局或接续其他实验。

末尾加`prepare`只核对模型、配置及冻结源码；加`status`只读取状态，不写报告或启动评测。

## 产物及有限验证

- [plan.json](plan.json)：固定模型、历史最佳配方依据、两组搜索差异及种子。
- `configs/lambda161924_fpu_vs_L12_S128N1_2000pair.json`和`configs/lambda161924_nofpu_vs_L12_S128N1_2000pair.json`：正式配置。
- `runtime/lambda161924_vs_l12_fpu2000_20261123/source`：实际冻结Python、带FPU的native扩展、牌池和配置；[source.json](source.json)逐文件SHA。
- 同名runtime的`panels/lambda161924_fpu_vs_L12.json`／`lambda161924_nofpu_vs_L12.json`及相邻`.json.games/`：本地原始结果与恢复日志；逐面板`.log`和recording目录保存调用／资源。
- 本目录`summary_fpu.json`／`summary_nofpu.json`和[results_fpu.md](results_fpu.md)／[results_nofpu.md](results_nofpu.md)保存紧凑报告；后续`execution_*.json`保存整体会话。

27项相关回归通过，含两组仅变化λ侧搜索配方、关闭值必须None、L12保持原搜索、实际预算／种子／恢复／只读status，以及FPU参数、公式、native和Python参照的一致性。两个真实模型按正式配置分别走两侧CPU native S128N1根，共4根／512模拟，均完成严格加载，L12辅助头调用为0；没有完整棋力对局或CUDA性能验证。原始JUnit和CPU记录仅保存在runtime，紧凑身份与复现命令见[verification.json](verification.json)。
