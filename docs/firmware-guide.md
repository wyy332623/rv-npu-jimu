# Firmware Guide

## Overview

Firmware is C code compiled for RV64IM that runs on the MiniRV64 ISS.
It orchestrates NPU operations by writing 32-bit instruction words to
the NPU's MMIO register interface.

## Firmware Structure

The BERT encoder layer firmware (`firmware/bert/bert_layer.c`) implements
one transformer encoder layer:

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

## Key Helper Functions

| Function | Purpose |
|----------|---------|
| `mvm_tiled_q()` | Multi-tile matrix-vector multiply. Iterates over tile rows/cols, accumulates via VV_ADD. Reads input from DRAM or VRF cache. |
| `mvm_tiled_vrf()` | Like `mvm_tiled_q()` but reads input vector from MFU_INITIAL_VRF cache instead of DRAM. |
| `save_row_tiles()` | Saves tile-row vectors from ADDSUB_VRF to DRAM with stride-8 addressing. |
| `load_and_add_row_tiles()` | Loads tile-row vectors from DRAM and adds to current ADDSUB_VRF values. |
| `apply_layernorm()` | Applies LayerNorm to tile rows in ADDSUB_VRF. Saves to scratch, loads gamma/beta, calls V_FUNC(SUB_LAYERNORM), restores. |

## Scalar Register Configuration

Before any data transfer or compute operation, the firmware must configure
the relevant scalar registers via `S_WR`. These registers control tile
dimensions, lane masking, and precision mode:

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

Scalar registers persist across instructions until explicitly changed.
Firmware typically sets them once per phase and restores masks at the
end of each attention head loop.

## Programming Pattern

### Basic Load-Compute-Store

```c
SEND_LO(OP_M_RD_DRAM, tile_addr);    // load weight tile
SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);  // acknowledge MRF write
SEND_LO(OP_V_RD_DRAM, vec_addr);     // load input vector
SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_MV_MUL, 0, 0);            // compute
npu_wait_done();                      // wait for completion
```

### Tiled Matmul with INC Variants

For hidden_size > NATIVE_DIM, the firmware splits the weight matrix into
tiles and uses INC instructions with auto-incrementing addresses:

```c
// Configure tile geometry
SEND_SI(OP_S_WR, REG_TILE_ROWS, 2);       // 2 tile rows
SEND_SI(OP_S_WR, REG_TILE_COLS, 2);       // 2 tile columns
SEND_SI(OP_S_WR, REG_ITERATIONS, 6);      // 6 positions

// Batch load: 6 x 2 x 2 = 24 vectors with auto-increment
SEND_LO(OP_V_RD_DRAM_INC, input_base);    // first vector, auto-inc
// ... repeats for all tiles per position ...
```

The INC variants iterate `ITERATIONS x TILE_COLS` times, incrementing the
DRAM address by `opd1` (the INC amount) each time.

### Multi-Head Attention Masking

Each head reads/writes only its element slice of the vector:

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

## VRF Cache Pattern

The primary optimization technique replaces DRAM save-load roundtrips
with on-chip copies to MFU_INITIAL_VRF (mem 6). Instead of:

```c
// Before: save to DRAM, then reload
SEND_SI(OP_V_RD, vrf, 0);
SEND_LO(OP_V_WR_DRAM, dram_base + tr * 8);   // DRAM save
// ... elsewhere ...
SEND_LO(OP_V_RD_DRAM, dram_base + tr * 8);   // DRAM reload
```

Use:

```c
// After: cache in VRF, read from VRF
SEND_SI(OP_V_RD, vrf, 0);
SEND_SI(OP_V_WR, MEM_MFU_INITIAL_VRF, offset);  // on-chip save
// ... elsewhere ...
SEND_SI(OP_V_RD, MEM_MFU_INITIAL_VRF, offset);  // on-chip read
```

## Building

```bash
cd firmware && make
```

The build uses DRAM layout macros passed via environment variables
(hidden_size, seq_len, projection base addresses, LN offsets).
The test harness and closed-loop pipeline compute these automatically.

## Running

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
