# 已有Chrome接入与逐事件模型估计

实际牌桌934687426直接连接已有Chrome，无URL参数；双方拿筹码、购买、保留和翻牌的独立估计见 actual_live_evaluations.jsonl。72项相关回归通过，最终代码与验证边界见 validation.json。

脚本只读，无游戏操作；Ctrl-C不关闭Chrome。当前一局已结束，后台监视器已停止，用户可在自己的终端运行 `PYTHONPATH=src python -u -m splendor.bga_live`，无需桌号。完整说明见 docs/guides/bga_advisor_zh-CN.md。
