# NPU 架构

## 概览

NPU 是由固件驱动、面向 Transformer 推理的 SIMD 加速器。RISC-V 控制处理器通过 MMIO 向 NPU 发送 32 位指令字，NPU 再由各功能单元执行这些指令。架构针对 Transformer 中常见的矩阵-向量乘、Softmax、LayerNorm、GELU 和残差加法进行了优化。

### 两种指令提交模式

| 模式 | 机制 | 适用场景 |
|------|------|----------|
| **逐指令模式** | 每次写入 `INST_FIFO` 后等待 `STATUS=DONE` | 配置、简单序列和调试 |
| **链式模式** | 连续写入多条指令，通过 `INST_ISSUE` 一次提交，并读取 `CHAIN_STATUS` | MVM、Softmax、残差等计算密集型序列 |

逐指令模式下，每条指令完成后才能开始下一条；链式模式下，指令可以在 pipeline 中连续传递数据，硬件还可以让相互独立的链在不同功能单元上重叠执行（SMC）。

### 设计原则

- **固件驱动**：所有计算都由 RISC-V CPU 通过 MMIO 写操作表达，NPU 没有指令 cache 或独立 sequencer；
- **同步执行**：逐指令模式中一条指令完成后才执行下一条；
- **异步链式执行**：链内指令进入 pipeline，硬件可在不同功能单元间并行调度；
- **数据并行**：各功能单元处理 `NATIVE_DIM` 个元素的向量；
- **Transformer 优化**：ISA 直接支持 attention、FFN 和归一化所需的操作。

## 数据流和功能单元

```text
DRAM → TMM → VRF/MRF → MVU → pipeline → MFU/SLU → VRF → TMM → DRAM
```

### TMM：Tensor Memory Manager

TMM 包含两个独立子单元：

- **VMM**：传输向量，支持 `V_RD_DRAM`、`V_WR_DRAM` 及 INC 变体；
- **MMM**：传输矩阵 tile，支持 `M_RD_DRAM` 和 `M_WR_DRAM`。

### MVU：Matrix-Vector Multiply Unit

MVU 计算 MRF 行向量和输入向量的点积，支持 `MV_MUL_INC` 累加模式。分块矩阵乘的部分和保存在 `MVM_ACC_VRF` 中。数值计算由 `libnpukernels.so` 的 `mv_mul()` 完成，Python 负责操作数准备和调用 C 库。

### MFU：Multifunction Unit

MFU 按能力路由操作：

- MFU 0：基于 LUT 的 GELU；
- MFU 1：向量加减；
- MFU 2：Tanh、加减和乘法。

### SLU：Softmax/LayerNorm Unit

SLU 通过 `V_FUNC` 完成融合的 Softmax 和 LayerNorm：

- Softmax：最大值归约 → `exp(x-max)` → 求和 → 归一化；
- LayerNorm：求和 → 均值 → 方差 → 逆平方根 → 缩放和偏移。

当 `NATIVE_DIM > LANES` 时使用 SerDes 模式，分块累积统计量，再进行最终归一化。

## 寄存器文件

### 向量寄存器文件 VRF

| Bank | ID | 元素数 | 用途 |
|------|----|--------|------|
| `MEM_DRAM` | 0 | 524288 | 外部 DRAM |
| `MEM_MULTIPLY_VRF` | 1 | 64 | 临时乘法结果 |
| `MEM_MATRIX_RF` | 4 | — | 矩阵寄存器文件的 tile |
| `MEM_MVM_INITIAL_VRF` | 5 | 20480 | MVU 输入向量 |
| `MEM_MFU_INITIAL_VRF` | 6 | 4096 | MFU 输入和 VRF cache |
| `MEM_ADDSUB_VRF_0` | 7 | 1024 | MFU 加法操作数 A |
| `MEM_ADDSUB_VRF_1` | 8 | 4096 | MFU 加法操作数 B |
| `MEM_MVM_ACC_VRF` | 13 | 256 | MVM 累加器 |
| `MEM_VEC_TO_MAT_ROW` | 18 | — | 向量到矩阵行缓冲区 |

### 矩阵寄存器文件 MRF

MRF 中驻留一个 `NATIVE_DIM × NATIVE_DIM` 的 tile，通过 `M_RD_DRAM` 从 DRAM 加载，同一时间只能驻留一个 tile。

## Pipeline register

pipeline 是一个向量宽度的隐式数据寄存器，用于在连续指令间传递计算结果：

```text
MV_MUL：pipeline = MRF[row] × 输入向量
VV_ADD：pipeline = vpipe_a + pipeline
SOFTMAX：pipeline = softmax(pipeline)
V_WR：VRF[dst] = pipeline
```

`V_RD` 会将新向量读入 pipeline，并把旧 pipeline 保存到 `vpipe_a`，供 `VV_ADD`、`VV_MUL` 等二元操作使用。`INST_ISSUE` 提交并结束当前链，链内 pipeline 状态不会跨链保留。

## 执行模型

### 逐指令模式

1. 固件向 `INST_FIFO`（0x00）写入 32 位指令；
2. 模拟器解码 opcode 和操作数；
3. 调用对应功能单元，必要时通过 ctypes 调用 C kernel；
4. 指令同步完成，`STATUS`（0x04）变为 `DONE`；
5. 固件确认状态后发送下一条指令。

### 链式模式

1. 固件连续向 `INST_FIFO` 写入多条指令，不在每条指令后轮询 `STATUS`；
2. 指令在链内按顺序执行，pipeline 在线程参数中传递；
3. 固件写入 `INST_ISSUE` 提交链；
4. 固件轮询 `CHAIN_STATUS`（0x0C），直到所有功能单元空闲。

当前 Python 模拟器是单线程、非周期精确模型：没有真实 FIFO、周期计数器、hazard 检测或并行 dispatch。因此链式模式当前主要用于验证指令语义和数据流；真正的硬件可以实现 TMM、MVU、MFU/SLU 之间的流水及独立链并发。

## 链模型和数据依赖

一条常见链遵循：

```text
Load → Compute → Store
```

例如：

| Load | Compute | Store | 用途 |
|------|---------|-------|------|
| `V_RD_DRAM` | `MV_MUL` | `V_WR` | 矩阵-向量乘 |
| `V_RD` | `V_GELU` | `V_WR` | GELU |
| `V_RD` | `V_FUNC(SOFTMAX)` | `V_WR` | 原地 Softmax |
| `M_RD_DRAM` + `V_RD_DRAM` | `MV_MUL_INC` + `VV_ADD` | `V_WR_DRAM` | 分块矩阵乘和 bias |

链之间需要处理 RAW、WAR 和 WAW 依赖。Python 模拟器不执行真正的 scoreboard；真实硬件调度器需要检查共享 VRF、MRF 和 DRAM 目标。

## Python 与 C 的分工

| 组件 | 语言 | 职责 |
|------|------|------|
| `iss/mini_rv64.py` | Python | 执行 RV64IM ELF |
| `emulator/npu_device_mini.py` | Python | MMIO、寄存器文件、指令解码和 DRAM 传输 |
| TMM | Python | DRAM 与 VRF/MRF 之间的数据传输 |
| MVU/MFU/SLU | Python + C | Python 负责控制和操作数，C 负责数值计算 |
| BERT 固件 | C（RV64IM） | 运行在 ISS 上并通过 MMIO 驱动 NPU |

