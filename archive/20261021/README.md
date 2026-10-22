# 2026-10-21 checkpoint冷归档

1885个模型来自已核对的809个旧恢复点及1076个历史/EMA模型。161个字节相同副本已链接到指定原件，不重复存入本包。模型字节和mtime保留，原路径留相对链接。不包含训练NPZ、账本或当前attention模型。

外盘位置：`/media/lc/T7/splendor/archive-20261021.tar.zst`。从仓库根目录恢复：

```bash
tar --zstd -xf '/media/lc/T7/splendor/archive-20261021.tar.zst' -C '/home/lc/下载/splendor'
cd archive/20261021
sha256sum -c SHA256SUMS
```

只读分析可使用恢复后的原路径链接。恢复旧训练前，应将对应模型按manifest恢复为原位真实文件，不要通过归档链接写入冻结包。未解压时冷模型链接不可读；本机锚点、MLP保留模型及当前训练不依赖本包。
