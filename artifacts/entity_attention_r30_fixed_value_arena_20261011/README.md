# 固定原第30轮value：三个policy对P4，再测第24轮对第30轮

按以下顺序串行执行，每组200 pair／400局，双方均为S1024/N1：

| 顺序 | A | B |
| --- | --- | --- |
| 1 | value loss权重1训练所得policy＋原配置第30轮value | P4完整网络 |
| 2 | value loss权重0.25训练所得policy＋原配置第30轮value | P4完整网络 |
| 3 | value loss权重0训练所得policy＋原配置第30轮value | P4完整网络 |
| 4 | 第24轮完整网络 | 原配置第30轮完整网络 |

前三组的policy均来自第29轮终点后同数据、同抽样序列训练1000步的终点，各自保留完整encoder、action query与policy head。value统一由**原配置第30轮完整encoder＋value head**计算。使用已有双网络组合机制，两边分别前向，不能将第30轮head接在另一分支encoder上。权重1也使用相同双网络组合并重新对战。

原第30轮为global 59898；第24轮为global 53898；P4为大池蒸馏global 34398。来源路径和SHA冻结在`plan.json`。本实验不训练新参数，不修改主自迭代设置。

前三组共同使用新的200个开局种子32700000–32700199、policy种子基数32710000、bootstrap种子32720000、search种子32730000；第四组分别为32800000–32800199、32810000、32820000、32830000。每个开局交换先后手，和棋计半分，整pair bootstrap 10000次。前三组报告相对权重1及彼此的同面板得分差；第四组直接对战得分按**第24轮**计算。限定单训练轨迹与本次DEV面板，未作多重比较校正。

执行设置沿用已完成搜索评测：`native_batched`、CUDA compiled、active100、max batch64、c_puct1.5、N1、S1024、无根噪声、折叠强制阶段、最多400个完整回合。每组独立子进程，结束即释放其缓存；整组完成后才开始下一组。已完成pair持久化，SIGINT／SIGTERM可中断，重新运行同一命令续跑，已完成组跳过。

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src python -u artifacts/entity_attention_r30_fixed_value_arena_20261011/run.py run
```

每组结束及整链路退出时自动更新[结果表](results.md)、[汇总](summary.json)，无需轮询。原始逐pair结果保存在`panels/`，配置在`configs/`，组合模型在`runs/entity_attention_r30_fixed_value_arena_20261011/compositions/`，运行状态、各组日志及源码快照在`runtime/entity_attention_r30_fixed_value_arena_20261011/`。命令、环境及代码身份写`execution.json`。

必要验证：固定来源参数逐项一致；从原第30轮整局留出取三局各开头／中段／结尾，共9个真实局面，确认三个组合的policy与各自来源逐值一致，value与原第30轮逐值一致，结果见`composition_verification.json`。完成组核对400有效局、模型SHA、实际S1024/N1预算及配对置信区间。
