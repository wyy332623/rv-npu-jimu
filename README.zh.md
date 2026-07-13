# NPU——开源版

一个开源的 FPGA 神经网络处理单元（NPU）功能模拟器。

## 架构

```text
RISC-V 固件（.elf）→ MiniRV64 指令集模拟器 → NPU 设备（Python + C）
```

NPU 是由 RISC-V 固件指令流驱动的 Python 功能模拟器。Python 代码负责指令解码、寄存器文件和控制流程；所有数值计算（MV_MUL、Softmax、LayerNorm、GELU、向量加/减/乘）都通过 ctypes 委托给 C kernel 库（`libnpukernels.so`）。

测试使用独立的 NumPy golden reference 验证模拟器输出；该 reference 不属于模拟器本身。

## 快速开始

```bash
# 1. 安装依赖
pip install pytest numpy pyelftools

# 2. 构建 C kernel 库（矩阵乘、GELU、Softmax 等）
make kernels

# 3. 构建 RISC-V 固件（bert.elf，BERT 编码器层）
make firmware

# 4. 运行全部测试
python3 -m pytest tests/ -v
```

预期结果：**所有测试通过**。

## 运行测试

```bash
# 仅运行 C kernel 测试（较快，共 15 项）
python3 -m pytest tests/unit/ -m c_kernel -v

# BERT 端到端集成测试
python3 -m pytest tests/integration/ -v

# 诊断：指令分发审计
python3 -m pytest tests/diagnostic/ -k static -v
```

## 项目结构

```text
kernels/         C 计算库（libnpukernels.so）
firmware/        RISC-V 固件（C、裸机 ELF → bert.elf）
emulator/
  npu_device_mini.py  NPU 功能模拟器
  trace_recorder.py   MMIO 指令跟踪器
iss/
  mini_rv64.py        MiniRV64 指令集模拟器（纯 Python）
tests/           pytest 测试套件和 golden reference 生成器
  gen_golden_bert.py   BERT 编码器层的 NumPy golden reference
jimu-dse/        固件闭环优化流程
```

## 设计文档

| 文档 | 内容 |
|------|------|
| `docs/architecture.md` | NPU 架构、ISA 和寄存器文件 |
| `docs/specification.md` | 完整 ISA 参考、opcode 表和执行模型 |
| `docs/firmware-guide.md` | RISC-V 固件和驱动 API |
| `docs/test-guide.md` | 测试分层、fixture 和 CI |
| `docs/build-guide.md` | 工具安装和构建步骤 |

## 许可证

Apache 2.0。
