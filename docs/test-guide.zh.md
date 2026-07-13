> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# 测试指南

## 测试套件

```
tests/
├── gen_golden_bert.py            ← numpy golden reference for BERT encoder layer
├── conftest.py                   ← pytest config (--instrument flag)
└── integration/
    └── test_bert_e2e.py          ← BERT E2E: golden → emulator
```

## BERT 端到端试验

primary XQ XQ XQ XQ XQ
多个配置 :

|参数|数值|
|-------|--------|
|暗号( NATION  DIM)| 2, 4 |
|隐藏大小| 4, 8 |
|下列语| 2, 6 |
|数字标题| 2 |

### 验证回合

|圆|后端|它检查什么|
|-------|---------|----------------|
|页:1|数字金色|通过 ZPROT000XZ 计算正确性|
|页:1|模拟器|指令语义, DRAM 排版, opcode 覆盖, 最终输出比较|


仅检查**最终产出** 数字正确性。
中间体(Q、K、V、剩余体、LN)没有优化——固件
可能根据需要将其存储在 VRF 缓存、芯片SRAM 或 DRAM 中。

## 运行测试

```bash
# All integration tests
python3 -m pytest tests/integration/ -v

# Single configuration
python3 -m pytest tests/integration/test_bert_e2e.py -k "seq6" -v

# With per-operator diagnostics (prints final output comparison only)
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s
```

## DRAM 状态

每次测试都会打印 DRAM 流量 :

```
  DRAM traffic (float32 elements):
    V_RD_DRAM: 312 ops × 8 el = 2496 el
    V_WR_DRAM: 12 ops × 8 el = 96 el
    M_RD_DRAM: 144 ops × 64 el = 9216 el
    Total: 11808 elements (47232 bytes)
```

## 承诺前钩

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -v
```

在承诺确保不倒退之前运行。
