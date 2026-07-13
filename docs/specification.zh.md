# NPU 规格说明

## 1. 概览

NPU 是由固件驱动、用于 Transformer 推理的 SIMD 加速器。RISC-V 控制处理器发送 32 位 MMIO 指令字，NPU 执行指令并通过状态寄存器报告完成。

### 设计原则

- **固件驱动**：所有计算都由 RISC-V CPU 发出的 32 位 MMIO 写操作表达；
- **同步执行**：逐指令模式下，每条指令完成后才能开始下一条；
- **数据并行**：计算单元处理 `NATIVE_DIM` 个元素，内部使用 FP16，可选 BFP；
- **Transformer 优化**：ISA 直接支持矩阵-向量乘、Softmax、LayerNorm、GELU、残差加法和 attention mask。

### NATIVE_DIM 与 LANES

| 术语 | 范围 | 定义 |
|------|------|------|
| `NATIVE_DIM` | 固件编译期 | 逻辑向量的元素数；VRF 传输、MV_MUL 迭代和 DRAM 步长都使用此值 |
| `LANES` | 硬件参数 | 物理硬件中的并行 FP16 数据通路数，设计时固定 |

`NATIVE_DIM` 必须是 `LANES` 的整数倍。当 `NATIVE_DIM > LANES` 时，SLU 使用 SerDes 模式，将向量分成 `NATIVE_DIM / LANES` 个块，在最终归一化前跨块累积归约统计量。

## 2. 指令集

指令是 32 位字，包含 SI 和 LO 两种格式。

### SI 格式

```text
31..24：opcode
23..16：opd0（8 bit）
15..0 ：opd1（16 bit）
```

`opd0` 可以是寄存器文件、内存目标或子 opcode，`opd1` 可以是索引、立即数或配置值。

### LO 格式

```text
31..24：opcode
23..0 ：24 位地址
```

DRAM 操作（opcode ≥ 20）使用 LO 格式，提供 24 位平面地址。

### Opcode 表

| Dec | 名称 | 格式 | 操作 |
|-----|------|------|------|
| 0 | `S_WR` | SI | 写标量寄存器 |
| 1 | `S_RD` | SI | 读标量寄存器 |
| 2 | `V_RD` | SI | 从 VRF 读取向量到 pipeline |
| 3 | `M_RD` | SI | 从行缓冲区读取矩阵到 MRF |
| 5 | `V_WR` | SI | 将 pipeline 写入 VRF |
| 6 | `M_WR` | SI | 确认矩阵写入 |
| 7 | `MV_MUL` | SI | MRF × pipeline → pipeline，不累加 |
| 8 | `VV_ADD` | SI | `vpipe_a + pipeline` |
| 11 | `VV_MUL` | SI | `vpipe_a × pipeline` |
| 20 | `V_RD_DRAM` | LO | 从 DRAM 读取向量 |
| 21 | `V_WR_DRAM` | LO | 将 pipeline 写入 DRAM |
| 22 | `V_RD_DRAM_INC` | LO | 带地址自动递增的向量读取 |
| 23 | `V_WR_DRAM_INC` | LO | 带地址自动递增的向量写入 |
| 24 | `M_RD_DRAM` | LO | 从 DRAM 加载矩阵 tile |
| 25 | `M_WR_DRAM` | LO | 将 MRF tile 写入 DRAM |
| 27 | `MV_MUL_INC` | SI | 从 `MVM_ACC_VRF` 读取前一部分和并累加 |
| 42 | `V_GELU` | SI | 通过 LUT 执行 GELU |
| 43 | `V_FUNC` | SI | `opd0=0` 为 Softmax，`opd0=1` 为 LayerNorm |
| 44 | `SS_ADD` | SI | 标量加法 |
| 45 | `INST_ISSUE` | SI | 提交当前指令链 |

### 格式判定

解码器根据 opcode 区分格式：opcode ≥ 20 时，完整的低 24 位是 DRAM 地址；其他 opcode 将低 24 位拆成 8 位 `opd0` 和 16 位 `opd1`。

```python
if opcode >= 20:
    addr = inst & 0xFFFFFF
else:
    file_id = (inst >> 16) & 0xFF
    index = inst & 0xFFFF
```

## 3. VRF 和标量寄存器

| 名称 | ID | 用途 |
|------|----|------|
| `MEM_DRAM` | 0 | 外部 DRAM，24 位平面地址 |
| `MEM_MULTIPLY_VRF` | 1 | 临时乘法结果 |
| `MEM_MATRIX_RF` | 4 | 权重矩阵寄存器文件 |
| `MEM_MVM_INITIAL_VRF` | 5 | MVU 输入向量 |
| `MEM_MFU_INITIAL_VRF` | 6 | MFU 输入和 VRF cache |
| `MEM_ADDSUB_VRF_0` | 7 | 加减操作数 A |
| `MEM_ADDSUB_VRF_1` | 8 | 加减操作数 B |
| `MEM_MVM_ACC_VRF` | 13 | 分块矩阵乘累加器 |
| `MEM_VEC_TO_MAT_ROW` | 18 | 向量到矩阵行缓冲区 |

关键标量寄存器包括 `TILE_ROWS=1`、`TILE_COLS=2`、`ITERATIONS=3`、`READ_VECTOR_MASK=15`、`WRITE_VECTOR_MASK=16`、`READ_MATRIX_MASK=17` 和 `PRECISION_MODE=20`。它们分别控制分块传输次数、lane mask、矩阵行 mask 和 FP16/BFP 模式。

## 4. 指令链模型

指令链是一组连续的微操作，通常形成：

```text
Load → Compute → Store
```

典型链包括：

| Load | Compute | Store | 用途 |
|------|---------|-------|------|
| `V_RD_DRAM` | `MV_MUL` | `V_WR` | 矩阵-向量乘 |
| `V_RD` | `V_GELU` | `V_WR` | GELU |
| `V_RD` | `V_FUNC(SOFTMAX)` | `V_WR` | 原地 Softmax |
| `M_RD_DRAM` + `V_RD_DRAM` | `MV_MUL_INC` + `VV_ADD` | `V_WR_DRAM` | 分块矩阵乘与 bias |
| `V_RD` | `VV_ADD` | `V_WR` | 残差加法 |

在当前 Python 模拟器中，每条指令仍同步完成；链的意义是保留 pipeline 数据流并验证未来硬件的提交语义。

### 链内 pipeline

```text
V_RD_DRAM → MV_MUL → V_FUNC(SOFTMAX) → MV_MUL → V_WR
```

上一阶段写入的 pipeline 值由下一阶段读取，不需要中间 VRF 往返。在硬件中，这可以形成 TMM → MVU → SLU → MVU → TMM 的 tensor pipeline。

### 链间重叠和 hazard

独立且没有 RAW、WAR、WAW 冲突的链可以在硬件中并发运行。`CHAIN_STATUS`（0x0C）记录 VMM、MMM、MVU 的 busy 状态。当前 Python 模拟器不实现真实并行、scoreboard 或 hazard 检查；这些由未来硬件调度器负责。

## 5. 计算单元

- **MVU**：执行 `MV_MUL` 和 `MV_MUL_INC`，数值计算委托给 C kernel 的 `mv_mul()`；
- **MFU**：路由 GELU、向量加减乘和 Tanh，调用 `gelu`、`vec_add`、`vec_sub`、`vec_mul`；
- **SLU**：由 `V_FUNC` 触发 Softmax 或 LayerNorm，并在 SerDes 模式下跨块累积统计量；
- **TMM**：由 VMM 和 MMM 负责 DRAM 与 VRF/MRF 的传输。

## 6. 存储系统

DRAM 是 24 位平面地址空间，模拟器默认提供约 512K 个 float32 元素。MRF 只驻留一个 `NATIVE_DIM × NATIVE_DIM` tile；加载新 tile 会覆盖旧 tile。INC 指令使用内部 DRAM 地址寄存器，每次传输后自动递增，适合流式访问。

## 7. 精度

- FP16：各计算单元边界使用 IEEE 754 binary16；
- BFP：可选的 block floating point，每组 `LANES` 共享指数；
- BFP 位宽：向量和矩阵均使用 4 bit 尾数配置。

## 8. 固件接口

| 偏移 | 名称 | 访问 | 用途 |
|------|------|------|------|
| 0x00 | `INST_FIFO` | W | 写入指令字 |
| 0x04 | `STATUS` | R | 0=IDLE，1=BUSY，2=DONE |
| 0x08 | `RESET` | W | 写非零值复位 |
| 0x0C | `CHAIN_STATUS` | R | 各功能单元 busy 位 |
| 0x20 | `HIDDEN_SIZE` | R/W | hidden 维度配置 |
| 0x24 | `SEQ_LEN` | R/W | 序列长度配置 |

逐指令协议：

```c
npu_send_inst(instruction_word);
while (npu_read_reg(NPU_STATUS) != NPU_STATUS_DONE);
```

链式协议则连续调用 `npu_send_inst()`，发送 `OP_INST_ISSUE` 提交链，再轮询 `NPU_CHAIN_STATUS` 直到所有功能单元空闲。

