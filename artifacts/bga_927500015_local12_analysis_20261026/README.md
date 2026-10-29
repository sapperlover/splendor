# 第12轮模型：原人类棋谱的完整搜索矩阵

2026-10-26，用户明确指定以本分支第12轮为基准，重跑原BGA #927500015的搜索矩阵。全部结果写入本目录，旧R20逐步报告及原始搜索文件保持原件。**完整矩阵已完成：1824次根搜索、15876096次模拟，耗时961.81秒（16.03分钟），编译失败及fallback均为0。** 已校验预算、种子、24份逐根原始文件和冻结输入身份，48份旧R20文件SHA保持不变。

权重固定为`runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt`，SHA256 `2a6304e3470cab3dad78ec0a2789ee892892e741f3731cafc2197cde9bd2d45d`。这是本地第12轮final，非最新第13轮。保留原88条当时行动者可见的请求：56次主行动、20次有选择支付共76个可搜索根，另12次强制决策不执行搜索。

| 方法 | 总预算1024 | 总预算16384 |
| --- | --- | --- |
| 确定化PUCT，1世界 | S1024N1 | S16384N1 |
| 确定化PUCT，4世界 | S256N4 | S4096N4 |
| Chance，共享信息集树 | 每根1024模拟 | 每根16384模拟 |
| CPW，k=0.25／α=0.5 | 每根1024模拟 | 每根16384模拟 |

每组3个固定种子39700000、39701000、39702000，合计1824次根搜索、15876096次模拟。原始文件内部标识`native_S1024_N4`／`native_S16384_N4`沿用旧总S命名，分别对应当前口径S256N4／S4096N4；新旧比较表使用当前口径并注明总预算。与旧矩阵一致：c_puct1.5、无根噪声、T0、折叠强制阶段、单模拟最多512次转移，FP/PTP关闭；总预算在N4的4个世界间均分。

当前执行栈显式使用原捕获的10贵族规则目录；三个已选贵族和当时可见信息保持原样。模型输入逐字段与原NPZ相等，86个非自动请求各经2个安全未知世界回建验证，不读取未来牌序、真实对手盲牌或终局结果作为模型／搜索输入。

CUDA FP32/native-cudagraph，TF32关闭，B16/B64、候选桶16/32/48/64，单进程串行矩阵、每批16个根、N4最多64个世界、owner单槽；编译失败直接报错。该小批布局复用旧棋谱矩阵，不套用对战配置A。

原producer和统计脚本复制后，只调整第12轮权重、独立输出路径、捕获规则目录的显式传递及报告标签；不改原矩阵参数和统计算法。旧结果特定的写死文字已从新统计报告中移除。

有限验证已完成：CPU工程smoke为4种方法各5根、每根总16模拟，共20根；另用当前GPU栈对旧R20的16个原S1024N1根重跑，**16个首选和全部访问计数均完全复现**，原世界种子相同。原始结果见`smoke_cpu.json`、`baseline_check.json`。检查时自迭代进程已停止、GPU空闲，没有停止或重启用户的其他进程。

运行／恢复命令相同：

```bash
conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u artifacts/bga_927500015_local12_analysis_20261026/analyze.py run
```

完整搜索结束后，只读生成原口径统计及新旧比较：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python \
  artifacts/bga_927500015_local12_analysis_20261026/summarize.py
conda run --no-capture-output -n splendor env PYTHONPATH=src python \
  artifacts/bga_927500015_local12_analysis_20261026/compare.py
```

`plan.json`冻结模型、输入、源码／native扩展身份、种子、预算与执行设置；`search/`保存逐根／逐种子／逐世界原始访问与Q，`report.html`为逐步候选报告。最终的新旧吻合度比较见`comparison.md`／`comparison.json`，完整双方、支付、低访问敏感性与种子稳定性见`statistics/results.md`。Q为模型搜索意见，不解释为已校准的真实胜率损失；单局吻合变化不等于棋力变化。

## 结果

[新旧逐配置比较](comparison.md)：8组搜索的56步主行动首选吻合，7组下降、1组上升；实际动作平均访问份额同样7组下降、1组上升。前三覆盖6组提高、1组持平、1组下降。第12轮的首选与人类更少一致，但实际走法多仍在候选前列；这不等同于棋力下降。完整逐步报告为[report.html](report.html)，统计为[statistics/results.md](statistics/results.md)，完成校验为[verification.json](verification.json)。

原始`search/` JSON和`report.html`未命中当前Git忽略规则；本次保持本地未提交，不修改`.gitignore`，旧R20报告不改写。
