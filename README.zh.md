> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# NPU — 开源版

一个开源的FPGA神经处理单元(NPU)模拟器.

## 建筑

```
RISC-V firmware (.elf) → MiniRV64 ISS → NPU device (Python + C)
```

NPU是一个基于Python的功能仿真器,由RISC-V固件驱动
指令流。 Python 代码处理指令解码, 注册
文件,并控制流量。 ** 所有数字计算** (MV MUL,软马克斯,
层诺姆, GELU, 矢量 QQZPROT000XZ) 授权给一个 C 内核库
通过型号。

NumPy 金色的参考文献( XZPROT000XZ) 由
用于验证模拟器输出的测试,但不属于模拟器本身。

## 快速启动

```bash
# 1. Install dependencies
pip install pytest numpy pyelftools

# 2. Build C kernel library (matrix multiply, GELU, softmax, etc.)
make kernels

# 3. Build RISC-V firmware (bert.elf, BERT encoder layer)
make firmware

# 4. Run all tests
python3 -m pytest tests/ -v
```

预期产出:**所有测试都通过**。

## 运行测试

```bash
# C kernel tests only (fast, 15 tests)
python3 -m pytest tests/unit/ -m c_kernel -v

# Integration: BERT E2E (parameterized)
python3 -m pytest tests/integration/ -v

# Diagnostic: dispatch audit
python3 -m pytest tests/diagnostic/ -k static -v
```

## 项目结构

```
kernels/         C compute library (libnpukernels.so)
firmware/        RISC-V firmware (C, bare-metal ELF → bert.elf)
emulator/
  npu_device_mini.py  NPU device — functional Python emulator
  trace_recorder.py   MMIO instruction trace recorder
iss/
  mini_rv64.py        MiniRV64 ISS (RV64IM, pure Python)
tests/           pytest suite + golden reference generator
  gen_golden_bert.py   numpy golden reference for BERT encoder layer
jimu-dse/        Closed-loop firmware optimization pipeline
```

## 设计文件

|文档|封面|
|----------|--------|
|津巴布韦|NPU 架构, ISA, 注册文件|
|津巴布韦|完整的ISA 参考文献, opcode 表格, 执行模式|
|津巴布韦|RISC-V 固件,驱动 API|
|津巴布韦|测试金字塔、固定装置、CI|
|津巴布韦|工具安装, 构建步骤|

## 许可证

阿帕奇2.0 (英语).
