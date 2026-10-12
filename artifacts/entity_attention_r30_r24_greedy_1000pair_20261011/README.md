# 第30轮 vs 第24轮：1000pair greedy

原配置第30轮完整模型（A，global 59898）直接对第24轮完整模型（B，global 53898）。双方均取合法动作policy logits的argmax，无搜索；value不参与动作选择。不采用value权重干预分支或双网络组合。

两个来源分别为主自迭代的`round-020/checkpoints/final.pt`和`round-014/checkpoints/final.pt`。来源SHA、绝对轮号与完整路径冻结于[plan.json](plan.json)。

新开局种子33300000–33300999，每个种子交换先后手，共1000pair／2000局；policy种子基数33310000、bootstrap种子33320000，整pair重采样10000次，和棋半分，得分与CI均按第30轮报告。不同于已有搜索对战的328开局面板。

沿用已实际运行的greedy评测路径：CPU 8 worker、每worker Torch线程1、最多400个完整玩家回合，保留确定性循环裁和。无新增内存硬上限。每100pair保存进度汇总，逐局sidecar随完成持久化；中断后同一命令续跑，完整pair跳过，未完整pair按原种子重放。同实验文件锁防止重复启动。

从仓库根目录运行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u artifacts/entity_attention_r30_r24_greedy_1000pair_20261011/run.py run
```

结束后自动核对2000有效局、模型身份、双方零搜索、整pair得分与CI，并更新[结果表](results.md)、[summary.json](summary.json)和实验索引。原始结果为[evaluation.json](evaluation.json)，配置为[config.json](config.json)，环境、命令及git／源码身份保存在`execution.json`。日志、源码快照和实时状态位于`runtime/entity_attention_r30_r24_greedy_1000pair_20261011/`，可恢复sidecar为被忽略的相邻`evaluation.json.games/`。

本实验衡量两个policy及各自encoder的greedy整局表现；不能单独确认搜索平台的原因或value是否退化。区间跨50%不代表严格等强。只覆盖这两个checkpoint与本次开局，不覆盖重新训练的随机性。
