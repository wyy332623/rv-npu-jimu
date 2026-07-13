> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# 运行 NPU 模型模拟

有三种配置:

|型号|数据路径|模拟类|
|---|---|---|
|** 130p FP32**|科斯敏的手工业|XZPROT000XZ(FP32,无截断)|
|** 140p FP32**|训练 Quen3 140段|XZPROT000XZ(FP32,无截断)|
|** 140页FP16**|训练 Quen3 140段|XZPROT000XZ(FP16 调校)|

所有模拟器运行在x86上——不需要RTL或FPGA.

---

## 快速启动

```bash
cd /home/ubuntu/work/git-rv-npu

# Build the firmware (needed for ISS tests)
make -C firmware TARGET=adder BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24
make -C firmware TARGET=adder_140p BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24

# Run all tests (~10 minutes, 46 tests)
python3 -m pytest adderboard/tests/ -v
```

---

## 测试结构

测试被整合为3个文件,并跨模型进行参数化:

```
adderboard/tests/
├── conftest.py        # Shared fixtures, model registry, test cases
├── test_phase1.py     # Python-driven NPU instructions (FP32, both models)
├── test_phase2.py     # ISS + firmware (both models) + replay cross-check
├── test_fp16.py       # FP16 emulator datapath (140p only)
├── _test_130p_phase1.py  # (internal) 130p build_forward
├── _test_130p_phase2.py  # (internal) 130p ISS helpers
├── _test_140p_phase1.py  # (internal) 140p build_forward
├── _test_140p_phase2.py  # (internal) 140p ISS helpers
└── _test_130p_cross.py   # (internal) replay capture helper
```

QQZPROT000XZ前置文件是内部模块(不能运行测试文件).

---

## 共用测试案例

所有模型都根据同样的6个数字案例进行测试:

|大小写|说明|
|---|---|
| `0-0-0` |三进制|
| `5-5-10` |小点|
| `555-445-1000` |中间|
| `19492-23919-43411` |随机|
| `9999999999-1-10000000000` |10位数外溢(从9位乘车)|
| `1111111111-8888888889-10000000000` |10位数外溢(平衡)|

---

## 测试清单(46项测试)

### XQ ZPROT000XQZ — Python驱动的NPU指令, FP32 (17个测试)

```bash
python3 -m pytest adderboard/tests/test_phase1.py -v
```

|测试|计数|说明|
|---|---|---|
|津巴布韦| 2 |单步对数匹配金色|
|津巴布韦| 12 |自动推论(2个模型×6个案例)|
|津巴布韦| 2 |50 随机自动递归对|
|津巴布韦| 1 |140p的中间值(ctx,atn res)|

### • ZPROT000Z——国际空间站+固件(22次试验)

```bash
python3 -m pytest adderboard/tests/test_phase2.py -v
```

|测试|计数|说明|
|---|---|---|
|津巴布韦| 2 |固件编译|
|津巴布韦| 2 |国际空间站单步对数匹配金色|
|津巴布韦| 12 |国际空间站自动递减(2个模型x6个案例)|
|津巴布韦| 2 |第2阶段指令重播符合第1阶段|

### • ZPROT000XZ——FP16模拟器(7次测试)

```bash
python3 -m pytest adderboard/tests/test_fp16.py -v
```

|测试|计数|说明|
|---|---|---|
|津巴布韦| 1 |FP16 单步对数(最大 diff < 20)|
|津巴布韦| 6 |FP16 自动递减(6个案件)|

---

## 按型号分列

|型号|第一阶段|第2阶段|FP16 电话|共计|
|---|---|---|---|---|
|130页| 9 | 11 | — | 20 |
|140页| 10 | 11 | 7 | 28 |
|** 全体**| | | | **46** |

---

## 源文件

- XZPROT000XZ — 130p 金色参考(纯数字)
- QQZPROT000XQZ — 140p 金色参考(Quen3 架构)
- - 130p DRAM 重量布局
- XZPROT000XZ — 140p DRAM 重量布局
- ZPROT000Z——训练140p重量
- • ZPROT000Z——130p RISC-V固件(单相)
- - 140p RISC-V固件(两相)
- — FP32测试辅助器(共享)
- QQZPROT000XZ – NPU 模拟器,带有 FP16 切换(共享)

## 结构概览

```
                    ┌──────────────┐
                    │  Test script │  (Python, pre-fills DRAM)
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌─────────────┐ ┌───────────┐ ┌───────────────┐
    │ Phase 1     │ │ Phase 2   │ │ Phase 2+ISS   │
    │ (Python)    │ │ (replay)  │ │ (firmware)    │
    └──────┬──────┘ └─────┬─────┘ └──────┬────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────────────────────────────────────────┐
    │            NPU Emulator                       │
    │  NpuFP32 (FP32) or NpuDeviceMini (FP16)       │
    └──────────────────────────────────────────────┘
```
