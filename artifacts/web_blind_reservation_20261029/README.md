# 本地人机对局：三次盲保留诊断

完整结论与15种首抽结果见[results.md](results.md)。所有41条续演已完成，866个动作合法性和完整终态重建校验通过，见[verification.json](verification.json)。不将代理续演计数当作棋力胜率。

固定本地policy在线第9轮、S1024N4（总4096）、seed0、无根噪声，原局seed100。回放与模型SHA见[plan.json](plan.json)。原局和逐步明细保存在`runtime/`，不进Git。

主要产物：

- [summary.json](summary.json)：三次盲留的15／14／12种结果，下一步搜索、后续主行动、暗牌去向及终态。
- [first_blind_values.json](first_blind_values.json)：AI私人视角的15种首抽value，比较时点为固定一次人类实盘应手之后。
- [rng_audit.json](rng_audit.json)：256组原生／Python／独立计算一致性及两批各20万组抽样检查。
- [seed_roots.json](seed_roots.json)：同一首抽前局面，32组世界seed的S1024N4选点敏感性。
- [tempo_control.json](tempo_control.json)：跳过第三次盲留、固定人类实盘应手的一条合法抢终局线；不是对任意人类应对的必胜证明。

可复核命令（均从仓库根目录执行，使用conda `splendor`）：

```bash
# 首次初始化才调用prepare；已有plan/source时不要重新覆盖冻结来源。
conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/web_blind_reservation_20261029/analyze.py prepare

# 已有冻结来源下续跑，逐轮保存；需要CUDA，或改为--device cpu。
# 未完成分支必须沿用其记录的device；已完成的CPU分支可与新GPU分支并存，摘要分别标记。
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/web_blind_reservation_20261029/analyze.py run --device cuda

# 只复核已经完成的原件并重新生成报告，不续演。
conda run --no-capture-output -n splendor env PYTHONPATH=runtime/web_blind_reservation_20261029/source/src python artifacts/web_blind_reservation_20261029/report.py
```

原始3个CPU根选点复现了实盘，GPU FP32探针也复现3个选点；后续30条GPU／11条CPU分支分别记录设备，不宣称浮点路径完全一致。首次CPU执行被环境SIGTERM后，保留11条已完成分支，其余重新续演，未拼接丢失轨迹。初始源码冻结于`runtime/web_blind_reservation_20261029/source/`，恢复驱动另冻结于`drivers/`，`execution_*.json`记录驱动SHA、命令及设备。

后续人类走法由同一checkpoint的greedy代理，不代表真实人类会如此应对。每种首抽只固定交换一张隐藏牌、保持其余牌序；没有对剩余牌序或人类应对做统计平均。原始网络value未校准，换算率为胜计1、平计0.5的估计得分率。
