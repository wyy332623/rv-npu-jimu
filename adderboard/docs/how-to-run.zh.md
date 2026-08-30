# 运行 NPU 模型模拟

项目提供三种配置：

| 模型 | 数据路径 | 模拟器类 |
|------|----------|----------|
| **130p FP32** | cosmin 手工构造权重 | `NpuFP32`（FP32，不截断） |
| **140p FP32** | 训练得到的 Qwen3 140 参数模型 | `NpuFP32`（FP32，不截断） |
| **140p FP16** | 训练得到的 Qwen3 140 参数模型 | `NpuDeviceMini`（FP16 截断） |

全部模拟器运行在 x86 上，不需要 RTL 或 FPGA。

## 快速开始

```bash
cd /home/ubuntu/work/git-rv-npu
make -C firmware TARGET=adder BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24
make -C firmware TARGET=adder_140p BUILD_DIR=build_dim4 NATIVE_DIM=4 SEQ_LEN=24
python3 -m pytest adderboard/tests/ -v
```

完整测试约需 10 分钟，共 46 项。

## 测试结构

```text
adderboard/tests/
├── conftest.py        # 共享 fixture、模型注册表和测试用例
├── test_phase1.py     # Python 驱动的 NPU 指令（两个模型，FP32）
├── test_phase2.py     # ISS + 固件，以及 replay 交叉校验
├── test_fp16.py       # FP16 模拟器数据通路（仅 140p）
└── _test_*.py         # 内部辅助模块
```

以下划线开头的文件是内部模块，不会被 pytest 作为独立测试文件运行。

## 共享测试用例

所有模型都使用相同的 6 个数值用例：

| 用例 | 说明 |
|------|------|
| `0-0-0` | 约束最简单的情况 |
| `5-5-10` | 小规模加法 |
| `555-445-1000` | 中等规模 |
| `19492-23919-43411` | 随机用例 |
| `9999999999-1-10000000000` | 十位数溢出（9 连续进位） |
| `1111111111-8888888889-10000000000` | 十位数溢出（均衡情况） |

## 测试清单

### 阶段 1：Python 驱动的 FP32 NPU 指令

```bash
python3 -m pytest adderboard/tests/test_phase1.py -v
```

包含单步 logit 对比、自回归推理、50 组随机输入和 140p 中间值检查。

### 阶段 2：ISS + 固件

```bash
python3 -m pytest adderboard/tests/test_phase2.py -v
```

包含固件编译、ISS 单步输出、自回归推理，以及阶段 2 指令 replay 与阶段 1 的交叉校验。

### FP16 模拟器

```bash
python3 -m pytest adderboard/tests/test_fp16.py -v
```

包含 FP16 单步 logits 检查（最大差值 < 20）和 6 组自回归用例。

## 各模型测试数

| 模型 | 阶段 1 | 阶段 2 | FP16 | 总计 |
|------|--------|--------|------|------|
| 130p | 9 | 11 | — | 20 |
| 140p | 10 | 11 | 7 | 28 |
| **全部** | | | | **46** |

## 主要源文件

- `adderboard/golden/golden_130p.py`：130p 纯 NumPy golden reference；
- `adderboard/golden/golden_140p.py`：Qwen3 架构的 golden reference；
- `adderboard/layout/layout_130p.py`、`layout_140p.py`：DRAM 权重布局；
- `adderboard/models/140p/s44_targeted_final_fp16.pt`：训练得到的 140p 权重；
- `adderboard/firmware/adder_130p.c`：130p 单阶段 RISC-V 固件；
- `adderboard/firmware/adder_140p.c`：140p 两阶段 RISC-V 固件；
- `emulator/npu_fp32.py`：FP32 测试辅助类；
- `emulator/npu_device_mini.py`：带 FP16 截断的 NPU 模拟器。

