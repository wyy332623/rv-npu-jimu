> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# 固件指南

## 概览

Firmware是为在MiniRV64国际空间站运行的RV64IM编译的C代码.
它通过写入32位指令词来协调 NPU 操作
NPU的MMIO寄存器接口.

## 固件结构

BERT 编码器层固件( XZPROT000XZ) 设备
一个变压器编码器层 :

```
main()
  ├── Read config from NPU registers (hidden_size, seq_len)
  ├── m_init_bias_accumulators() — pre-load bias values
  └── bert_encoder_layer()
        ├── Phase 1: Compute K, V for all positions
        │     └── compute_k_all_positions()
        │     └── compute_v_all_positions()
        └── Phase 2: Per-position loop
              ├── dot_product_attention()
              │     ├── Compute Q
              │     ├── Build K.T MRF tile → score → softmax
              │     ├── Build V.T MRF tile → context
              │     └── Accumulate context
              ├── Self-output projection + residual + LN1
              ├── FFN intermediate + GELU
              └── FFN output + residual + LN2
```

## 密钥帮助函数

|函数|目的|
|----------|---------|
|津巴布韦|多瓦矩阵-载体乘法. 通过VV ADD积累。 从 DRAM 或 VRF 缓存读取输入 。|
|津巴布韦|如QQZPROT000XQZ,但读取来自MFU INITIAL VRF缓存而不是DRAM的输入向量.|
|津巴布韦|从 ADDSUB VRF 保存砖排向量到 DRAM, 地址为 stride-8 。|
|津巴布韦|从 DRAM 中装入 tyle- row 向量, 并添加到当前 ADDSUB  VRF 值 。|
|津巴布韦|在 ADDSUB VRF 中将图层Norm 应用到瓦片行中. 保存为抓取,装入 QZPROT000XQZ,调用 V FUNC(SUB LAYERNORM),恢复.|

## Scalar 注册配置

在任何数据传输或计算操作之前,固件必须配置
通过 QZPROT000XXZ 来进行相关的scalar 注册。 这些登记册控制瓦
尺寸、车道遮盖和精确模式:

```c
// Configure tile dimensions for multi-vector transfer
SEND_SI(OP_S_WR, REG_TILE_ROWS, num_tiles);      // rows per tile
SEND_SI(OP_S_WR, REG_TILE_COLS, num_tiles);       // cols per tile
SEND_SI(OP_S_WR, REG_ITERATIONS, seq_len);         // outer loop count

// Set precision mode
SEND_SI(OP_S_WR, REG_PRECISION_MODE, 1);           // 0=FP16, 1=BFP

// Lane masking for multi-head attention
SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);      // enable all lanes
SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, 0xFF);     // enable all lanes
SEND_SI(OP_S_WR, REG_READ_MATRIX_MASK, 0xFF);      // enable all MRF rows
```

在明确更改之前,Scalar登记册一直贯穿于各种指示。
固件一般每期安装一次,然后恢复口罩
每个注意头环的结束。

## 方案拟订模式

### 基本负载计算系统

```c
SEND_LO(OP_M_RD_DRAM, tile_addr);    // load weight tile
SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);  // acknowledge MRF write
SEND_LO(OP_V_RD_DRAM, vec_addr);     // load input vector
SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_MV_MUL, 0, 0);            // compute
npu_wait_done();                      // wait for completion
```

### 配有 INC 变式的平铺 Matmul

对于隐藏的  大小 > NATION  DIM, 固件将权重矩阵分割为
带有自动递增地址的瓷砖和使用 INC 指令:

```c
// Configure tile geometry
SEND_SI(OP_S_WR, REG_TILE_ROWS, 2);       // 2 tile rows
SEND_SI(OP_S_WR, REG_TILE_COLS, 2);       // 2 tile columns
SEND_SI(OP_S_WR, REG_ITERATIONS, 6);      // 6 positions

// Batch load: 6 x 2 x 2 = 24 vectors with auto-increment
SEND_LO(OP_V_RD_DRAM_INC, input_base);    // first vector, auto-inc
// ... repeats for all tiles per position ...
```

INC变体的斜率为 QQZPROT000Z倍,
DRAM地址由QQZPROT000XQZ(INC量)每一次.

### 多头注意力蒙面

每个头 QQZPROT000XQZ 仅是它的元素向量片:

```c
for (int h = 0; h < heads_per_tile; h++) {
    uint8_t mask = (h == 0) ? 0x03 : 0x0C;   // elements [0,1] vs [2,3]
    SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, mask);
    SEND_LO(OP_V_RD_DRAM, vec_addr);          // load only masked lanes
    // ... compute attention for this head ...
    SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, mask);
    SEND_SI(OP_V_WR, REG_ADDSUB_VRF_0, 0);    // write only masked lanes
}
// Restore full mask for subsequent operations
SEND_SI(OP_S_WR, REG_READ_VECTOR_MASK, 0xFF);
SEND_SI(OP_S_WR, REG_WRITE_VECTOR_MASK, 0xFF);
```

## VRF 缓存模式

主优化技术取代 DRAM 保存载荷的绕行
带有芯片副本到 MFU INITIAL VRF (mem 6). 代替:

```c
// Before: save to DRAM, then reload
SEND_SI(OP_V_RD, vrf, 0);
SEND_LO(OP_V_WR_DRAM, dram_base + tr * 8);   // DRAM save
// ... elsewhere ...
SEND_LO(OP_V_RD_DRAM, dram_base + tr * 8);   // DRAM reload
```

使用 :

```c
// After: cache in VRF, read from VRF
SEND_SI(OP_V_RD, vrf, 0);
SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, offset);  // on-chip save
// ... elsewhere ...
SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, offset);  // on-chip read
```

## 大楼

```bash
cd firmware && make
```

构建使用通过环境变量传递的 DRAM 排版宏
(隐藏大小,下列,预测基址,LN冲抵).
测试带和闭路管道自动计算这些

## 运行

```python
from iss.mini_rv64 import MiniRV64
from emulator.npu_device_mini import NpuDeviceMini

npu = NpuDeviceMini(native_dim=dim)
npu.set_hidden_size(hidden_size)
npu.set_seq_len(seq_len)
# Load input tensors and weights into npu._vrf[MEM_DRAM]
cpu = MiniRV64()
cpu.set_mmio_device(npu)
cpu.load_elf("firmware/build_dim2/bert.elf")
cpu.run(cycles=200000)
```
