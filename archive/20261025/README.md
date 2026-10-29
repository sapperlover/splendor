# 2026-10-25 冷产物归档

原件流式压缩至T7，本机保留清单和原路径相对链接。files.jsonl.gz为逐路径SHA、大小、纳秒mtime、inode与归档路径；保留原字节，不去重。

从仓库根目录恢复：

```bash
tar --zstd -xf "/media/lc/T7/splendor/archive-20261025.tar.zst" -C "/home/lc/下载/splendor"
cd archive/20261025
sha256sum -c SHA256SUMS
```

旧实验复训前，将需写入的模型／账本按files.jsonl.gz复制回原位真实文件并恢复mtime，避免通过链接修改冻结归档。本批原80k局已外置；其离线复训／完整复核须恢复。本机保留R21–37及新在线对局、全部完整留出模型、最优模型、initial/final及当前恢复依赖。
