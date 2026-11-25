# 2026-11-24 四类冷产物归档

按用户授权外置93655个原件；原路径使用相对恢复链接。逐文件大小、SHA256、纳秒mtime和包内路径见files.jsonl.gz。包内manifest是打包前计划，最终结果见artifacts/repository_archive_20261124/completion.json。

从仓库根目录恢复本批全部数据：

```bash
tar --zstd -xf "/media/lc/T7/splendor/archive-20261124.tar.zst" -C "/home/lc/下载/splendor"
cd archive/20261124
sha256sum -c SHA256SUMS
```

部分恢复以files.jsonl.gz中的archive_member为成员名。要继续写入日志或模型，先复制回原位真实文件并保留mtime，不要通过链接修改冻结归档。旧104k复训/续训、旧在线恢复或旧回放快照复核需先恢复相应原件。最新lambda在线输入、近期60k、全部固定留出和关键模型保持本机；旧MC/低学习率未完成实验的主日志及恢复点保留。
