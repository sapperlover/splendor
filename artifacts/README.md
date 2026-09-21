# 实验交付物

`artifacts/` 只保存值得随仓库长期留存的实验交付物，例如：

- 最终验收统计；
- 完成报告引用的 CSV / JSON；
- 用于复核结论的 matchup matrix 和分析表。

这个目录默认纳入 Git，不再使用逐文件 `.gitignore` 白名单。

以下内容不应放在 `artifacts/`：日志、可恢复进度、临时 checkpoint、失败现场、
分片运行目录、cache 和其他程序运行状态。这些内容统一放入 `runtime/`，
且 `runtime/` 整体被 Git 忽略。

大型训练运行仍保留在历史上专用的 `runs/` 目录，该目录同样被 Git 忽略。
