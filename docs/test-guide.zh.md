# 测试指南

## 测试套件

```text
tests/
├── gen_golden_bert.py            ← BERT 编码器层的 NumPy golden reference
├── conftest.py                   ← pytest 配置（`--instrument` 标志）
└── integration/
    └── test_bert_e2e.py          ← BERT 端到端测试：golden → 模拟器
```

## BERT 端到端测试

主测试 `test_bert_e2e_multi_tile` 会在多种配置下验证固件：

| 参数 | 取值 |
|------|------|
| dim（NATIVE_DIM） | 2、4 |
| hidden_size | 4、8 |
| seq_len | 2、6 |
| num_head | 2 |

### 验证轮次

| 轮次 | 后端 | 检查内容 |
|------|------|----------|
| **R0** | NumPy golden | 通过 `gen_golden_bert.py` 验证算法正确性 |
| **R1** | 模拟器 | 指令语义、DRAM 布局、opcode 覆盖率和最终输出对比 |

数值正确性只检查最终输出。中间结果（Q、K、V、残差、LN）不限定存储方式；固件可以根据需要将它们放在 VRF、片上 SRAM 或 DRAM 中，以便进行优化。

## 运行测试

```bash
# 全部集成测试
python3 -m pytest tests/integration/ -v

# 单项配置
python3 -m pytest tests/integration/test_bert_e2e.py -k "seq6" -v

# 启用逐算子诊断（仅打印最终输出对比）
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s
```

## DRAM 统计

每次测试都会打印 DRAM 流量：

```text
  DRAM traffic (float32 elements):
    V_RD_DRAM: 312 ops × 8 el = 2496 el
    V_WR_DRAM: 12 ops × 8 el = 96 el
    M_RD_DRAM: 144 ops × 64 el = 9216 el
    Total: 11808 elements (47232 bytes)
```

## 提交前检查

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

提交前运行该命令，确保没有引入回归。
