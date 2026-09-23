# 实验交付物

`artifacts/` 只保存值得随仓库长期留存的实验交付物，例如：

- 最终验收统计；
- 完成报告引用的 CSV / JSON；
- 用于复核结论的 matchup matrix 和分析表。

这个目录默认纳入 Git，不再使用逐文件 `.gitignore` 白名单。例外是长实验为
了断点恢复生成的 `<summary>.games/` 逐局 sidecar：它们是可再生成的运行状态，
由根目录 `.gitignore` 统一忽略；同名汇总 JSON 仍作为实验交付物跟踪。

除上述与汇总结果紧邻的可恢复 sidecar 外，日志、临时 checkpoint、失败现场、
分片运行目录、cache 和其他程序运行状态仍应统一放入 `runtime/`，且
`runtime/` 整体被 Git 忽略。

大型训练运行仍保留在历史上专用的 `runs/` 目录，该目录同样被 Git 忽略。
