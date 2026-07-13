> 本文件由自动翻译生成，仅供参考；以英文原文为准。

---
名称:自我验证
说明:自我核实固件正确性和DRAM改进
许可证:麻省理工学院
---

# 自我认证技能

## 修改 QQZPROT000XZ 后

### 1. 数值正确性

```bash
# Quick check (seq6 only, ~10s)
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s --no-header 2>&1 | grep "max_diff"
```

预期产出:
```
  Q: max_diff=0.000000, mean_diff=0.000000
  K: max_diff=0.000000, mean_diff=0.000000
  ...
```

所有ZPROT000XZ必须 < 0.05。 如果有任何失败, 修改会产生不正确的数值输出 。

### 2. DRAM 交通系统

```bash
# Run the full test suite to check DRAM
python3 -m pytest tests/integration/test_bert_e2e.py -k seq6 -s 2>&1 | grep -E "DRAM|max_diff|FAILED|PASSED"
```

寻找DRAM交通线——显示总字节,V RD DRAM ops等.

### 3. 完全回归

```bash
# All 4 configs (seq2 + seq6 for dim2 and dim4)
python3 -m pytest tests/integration/test_bert_e2e.py -v 2>&1 | tail -10
```

四者皆须过.

## 常见失败

|症状|可能的原因是|修补|
|---------|-------------|-----|
|XZPROT000XZ 最大值 diff > 0.05|弗朗索瓦银行或冲销错误|检查 VREG MOVE 目标地址|
|Z 最大值 diff > 0.05|V.T 重译读错 V 数据|检查 V 缓存偏移公式|
|XZPROT000XZ 最大值 diff > 0.05|图层输入错误|检查剩余添加数据流|
|未减少DRAM|保存仍然被调用的小牌|在 bert layer.c 中搜索 OP V WR DRAM|
|编译错误|SEND SI 代替 INC 的 SEND LO|使用正确的宏|
