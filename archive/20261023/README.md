# 2026-10-23 冷产物归档

原件流式压缩至T7，本机保留清单和原路径相对链接。files.jsonl.gz为逐路径SHA、大小、纳秒mtime、inode与归档路径，不去重、不丢弃双头账本。

从仓库根目录恢复：

```bash
tar --zstd -xf "/media/lc/T7/splendor/archive-20261023.tar.zst" -C "/home/lc/下载/splendor"
cd archive/20261023
sha256sum -c SHA256SUMS
```

旧实验复训前，将需写入的模型／账本按files.jsonl.gz复制回原位真实文件并恢复mtime，避免通过链接修改冻结归档。原80k局、R21–31及未完成采集、当前6万局和所有未选模型留本机。
