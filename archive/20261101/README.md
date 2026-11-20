# 2026-11-01 冷产物归档

234543个原件按批准清单流式压缩到T7；本机原路径保留相对恢复链接。files.jsonl.gz记录逐文件SHA256、大小、纳秒mtime及包内路径；不去重、不改原字节。guard与retained清单记录保留文件身份；包内manifest为打包前冻结计划，完成记录见artifacts/repository_archive_20261101/。

从仓库根目录恢复全部本批数据：

```bash
tar --zstd -xf "/media/lc/T7/splendor/archive-20261101.tar.zst" -C "/home/lc/下载/splendor"
cd archive/20261101
sha256sum -c SHA256SUMS
```

部分恢复可将archive/20261101/加在原路径前作为tar成员名；所需逐局sidecar也应一并恢复。要续写的模型或日志，先按files.jsonl.gz复制回原路径的真实文件并恢复mtime，避免通过链接修改冻结归档。本批不含初始64k、活动60k、104k共享池及所有固定留出；现在线与低学习率恢复入口所需文件保留。旧普通checkpoint、旧逐步日志、淘汰训练局或原始评测明细需按需恢复本包；精确范围见scope_summary.json。
