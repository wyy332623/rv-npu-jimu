# 固件指南

## 概览

固件是为 RV64IM 编译、运行在 MiniRV64 指令集模拟器上的 C 代码。它通过向 NPU 的 MMIO 寄存器接口写入 32 位指令字来编排 NPU 操作。

## 固件结构

BERT 编码器层固件（`firmware/bert/bert_layer.c`）实现一个 Transformer 编码器层：

```text
main()
  ├─ 从 NPU 寄存器读取配置（hidden_size、seq_len）
  ├─ m_init_bias_accumulators()：预加载 bias
  └─ bert_encoder_layer()
        ├─ 阶段 1：计算所有位置的 K、V
        │    ├─ compute_k_all_positions()
        │    └─ compute_v_all_positions()
        └─ 阶段 2：逐位置循环
             ├─ dot_product_attention()
             │    ├─ 计算 Q
             │    ├─ 构造 K.T 的 MRF tile → score → Softmax
             │    ├─ 构造 V.T 的 MRF tile → context
             │    └─ 累加 context
             ├─ Self-output projection + residual + LN1
             ├─ FFN intermediate + GELU
             └─ FFN output + residual + LN2
```

## 主要辅助函数

| 函数 | 用途 |
|------|------|
| `mvm_tiled_q()` | 多 tile 矩阵-向量乘。遍历 tile 行/列，并通过 VV_ADD 累加；输入可来自 DRAM 或 VRF cache。 |
| `mvm_tiled_vrf()` | 类似 `mvm_tiled_q()`，但从 `MFU_INITIAL_VRF` cache 读取输入向量，而不是从 DRAM 读取。 |
| `save_row_tiles()` | 将 ADDSUB_VRF 中的 tile-row 向量按 stride-8 地址写入 DRAM。 |
| `load_and_add_row_tiles()` | 从 DRAM 读取 tile-row 向量，并与当前 ADDSUB_VRF 值相加。 |
| `apply_layernorm()` | 对 ADDSUB_VRF 中的 tile 行执行 LayerNorm：保存到 scratch，加载 gamma/beta，调用 `V_FUNC(SUB_LAYERNORM)`，再恢复结果。 |

## 标量寄存器配置

在进行任何数据传输或计算之前，固件必须通过 `S_WR` 配置相关标量寄存器。这些寄存器控制 tile 尺寸、lane mask 和精度模式：

```c
// 配置多向量传输的 tile 尺寸
SEND_SI(OP_S_WR, REG_TILE_ROWS, num_tiles);      // 每个 tile 的行数
SEND_SI(OP_S_WR, REG_TILE_COLS, num_tiles);      // 每个 tile 的列数
SEND_SI(OP_S_WR, REG_ITERATIONS, seq_len);       // 外层循环次数

// 设置精度模式
SEND_SI(OP_S_WR, REG_PRECISION_MODE, 1);         // 0=FP16，1=BFP

// 多头注意力的 lane mask
SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);    // 启用全部 lane
SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, 0xFF);   // 启用全部 lane
SEND_SI(OP_S_WR, REG_READ_MATRIX_MASK, 0xFF);    // 启用全部 MRF 行
```

标量寄存器会在后续指令之间保持不变，直到被显式修改。固件通常在每个阶段设置一次，并在每个 attention head 循环结束时恢复 mask。

## 编程模式

### 基本的加载-计算-存储

```c
SEND_LO(OP_M_RD_DRAM, tile_addr);    // 加载权重 tile
SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);  // 确认 MRF 写入
SEND_LO(OP_V_RD_DRAM, vec_addr);     // 加载输入向量
SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_MV_MUL, 0, 0);            // 计算
npu_wait_done();                     // 等待完成
```

### 使用 INC 变体的分块矩阵乘

当 `hidden_size > NATIVE_DIM` 时，固件会将权重矩阵拆分为多个 tile，并使用带地址自动递增的 INC 指令：

```c
// 配置 tile 几何尺寸
SEND_SI(OP_S_WR, REG_TILE_ROWS, 2);       // 2 个 tile 行
SEND_SI(OP_S_WR, REG_TILE_COLS, 2);       // 2 个 tile 列
SEND_SI(OP_S_WR, REG_ITERATIONS, 6);      // 6 个位置

// 批量加载：6 × 2 × 2 = 24 个向量，地址自动递增
SEND_LO(OP_V_RD_DRAM_INC, input_base);    // 第一个向量，自动递增
// ... 每个位置的 tile 重复执行 ...
```

INC 变体会迭代 `ITERATIONS × TILE_COLS` 次，每次将 DRAM 地址增加 `opd1` 指定的步长。

## 多头注意力的 mask

每个 head 只读写向量中的对应元素切片：

```c
for (int h = 0; h < heads_per_tile; h++) {
    uint8_t mask = (h == 0) ? 0x03 : 0x0C;   // 元素 [0,1] 与 [2,3]
    SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, mask);
    SEND_LO(OP_V_RD_DRAM, vec_addr);         // 仅加载 mask 选中的 lane
    // ... 计算该 head 的 attention ...
    SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, mask);
    SEND_SI(OP_V_WR, REG_ADDSUB_VRF_0, 0);   // 仅写入 mask 选中的 lane
}
// 为后续操作恢复完整 mask
SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);
SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, 0xFF);
```

## VRF Cache 模式

主要优化手段是将 DRAM 的保存-加载往返替换为写入 `MFU_INITIAL_VRF`（mem 6）的片上拷贝。不要采用：

```c
// 之前：写入 DRAM，然后重新加载
SEND_SI(OP_V_RD, vrf, 0);
SEND_LO(OP_V_WR_DRAM, dram_base + tr * 8);   // 保存到 DRAM
// ... 其他位置 ...
SEND_LO(OP_V_RD_DRAM, dram_base + tr * 8);   // 从 DRAM 重新加载
```

而应采用：

```c
// 之后：缓存到 VRF，从 VRF 读取
SEND_SI(OP_V_RD, vrf, 0);
SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, offset);  // 片上保存
// ... 其他位置 ...
SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, offset);  // 片上读取
```

## 构建

```bash
cd firmware && make
```

构建过程使用通过环境变量传入的 DRAM 布局宏（hidden_size、seq_len、projection base 地址和 LN 偏移量）。测试套件和闭环流程会自动计算这些值。

## 运行

```python
from iss.mini_rv64 import MiniRV64
from emulator.npu_device_mini import NpuDeviceMini

npu = NpuDeviceMini(native_dim=dim)
npu.set_hidden_size(hidden_size)
npu.set_seq_len(seq_len)
# 将输入 tensor 和权重加载到 npu._vrf[MEM_DRAM]
cpu = MiniRV64()
cpu.set_mmio_device(npu)
cpu.load_elf("firmware/build_dim2/bert.elf")
cpu.run(cycles=200000)
```
