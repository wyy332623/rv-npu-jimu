# cosminscn_130p 权重

130p 模型使用手工构造的权重（共 130 个浮点数），而不是训练得到的 checkpoint。
所有权重值都直接硬编码在 DRAM 布局文件中：

```text
adderboard/layout/layout_130p.py
```

项目没有用于保存权重的 `.pt` 文件；布局构建器会在运行时内联计算权重。全部 130 个参数值都是确定性的，并记录在 `adderboard/docs/compatibility.md` 中。
