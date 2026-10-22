# 新128×2：30619步对原R43的greedy补测

用户指定补测本次新128×2无辅助模型的30619步checkpoint。沿用父实验37423步面板的全部1000个开局、policy seed和评测参数：双方纯greedy、CPU8worker、交换先后手共2000局，和棋半分，整pair bootstrap10000次。仅评测，不训练、不执行搜索。

候选原路径已因归档外置而不可读。已从`/media/lc/T7/splendor/archive-20261019.tar.zst`单独提取`archive/20261019/runs/entity_attention_128x2_relations_no_aux_r24_r43_20261019/checkpoints/step-000030619.pt`，放入忽略的`runtime/entity_attention_128x2_relations_no_aux_r24_r43_20261019/greedy_step30619/step-000030619.pt`。文件7094987字节，SHA256为`ee45ff741b82a03cdd18a66718e4648f3cc1725abefaedbf5b4adfa4a294cb77`，与原完整留出记录完全一致。原归档与原run链接未改动。

冻结身份、配置、来源与种子见[plan.json](plan.json)、[evaluation_config.json](evaluation_config.json)。执行命令、代码/设备身份与状态见`execution.json`；进度日志在上述runtime目录的`evaluation.log`。生产入口保存相邻`evaluation.json.games/`逐局sidecar，可恢复。

完成结果见`evaluation.json`、`results.md`、`summary.json`。[analyze.py](analyze.py)核对checkpoint身份、完整pair、无搜索、bootstrap，并与父实验37423步对R43的相同开局成绩计算配对得分差。这是共同对手比较，不是两个候选直接互打；本次是已查看DEV开局上的事后节点补测，未做多重比较校正。

复现时需先恢复上述runtime checkpoint并核对SHA，再从仓库根目录执行：

```bash
conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m splendor.evaluation \
  --config artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/greedy_step30619/evaluation_config.json \
  --output artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/greedy_step30619/evaluation.json \
  --progress-every 100 --summary-every 100 --resume
conda run -n splendor env PYTHONPATH=src python \
  artifacts/entity_attention_128x2_relations_no_aux_r24_r43_20261019/greedy_step30619/analyze.py
```
