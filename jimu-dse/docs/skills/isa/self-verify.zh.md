---
name: self-verify
description: 自验证固件正确性和 DRAM 改进效果
license: MIT
---

# Self-Verify 技能

## 修改 `bert_layer.c` 后

### 1. 数值正确性

```bash
# 快速检查（仅 seq6，约 10 秒）
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s --no-header 2>&1 | grep "max_diff"
```

预期输出：

```text
  Q: max_diff=0.000000, mean_diff=0.000000
  K: max_diff=0.000000, mean_diff=0.000000
  ...
```

所有 `max_diff` 都必须小于 0.05。若有任何一项失败，说明本次修改产生了错误的数值结果。

### 2. DRAM 流量

```bash
# 运行完整测试以检查 DRAM
python3 -m pytest tests/integration/test_bert_e2e.py -k seq6 -s 2>&1 | grep -E "DRAM|max_diff|FAILED|PASSED"
```

检查 DRAM 流量行，其中会显示总字节数、`V_RD_DRAM` 操作数等信息。

### 3. 完整回归

```bash
# 全部 4 种配置（dim2 和 dim4 各测试 seq2、seq6）
python3 -m pytest tests/integration/test_bert_e2e.py -v 2>&1 | tail -10
```

4 项测试必须全部通过。

## 常见失败

| 现象 | 可能原因 | 修复方法 |
|------|----------|----------|
| Q/K 的 `max_diff > 0.05` | VRF bank 或偏移量错误 | 检查 VREG_MOVE 的目标地址 |
| Z 的 `max_diff > 0.05` | V.T 转置重新读取了错误的 V 数据 | 检查 V cache 偏移量公式 |
| LN1/LN2 的 `max_diff > 0.05` | LayerNorm 输入错误 | 检查残差加法的数据流 |
| DRAM 未减少 | 仍在调用 `save_row_tiles` | 在 `bert_layer.c` 中搜索 `OP_V_WR_DRAM` |
| 编译错误 | 对 INC 指令使用了 SEND_SI，而不是 SEND_LO | 使用正确的宏 |
