---
name: vrf-cache
description: Redirect K/V tensor data from DRAM roundtrip to on-chip VRF cache
license: MIT
---

# VRF Cache Skill

## Problem

The firmware computes K=V=Wx+b for each position, saves to DRAM via `save_row_tiles()`, then the attention code reloads from DRAM via `V_RD_DRAM`. This roundtrip is unnecessary when the target VRF has capacity.

## VRF Capacity

| VRF Bank | Mem ID | Size (elements) | Used for |
|----------|--------|-----------------|----------|
| MFU_INITIAL_VRF | 6 | 4096 | GELU activation (temporary) |
| ADDSUB_VRF_0 | 7 | 1024 | Tile row 0 accumulator |
| ADDSUB_VRF_1 | 8 | 4096 | Tile row 1 accumulator + X cache |
| ADDSUB_VRF_2 | 9 | 64 | Second tile row (multi-tile) |
| MVM_INITIAL_VRF | 5 | 20480 | MVM input vector |
| MULTIPLY_VRF | 1 | 64 | Temporary MVM result |

For dim=2, hidden=4, seq_len=6:
- K per position: 4 elements (2 tile rows × 2 dim)
- V per position: 4 elements
- Q per position: 4 elements
- All 6 positions' K+V: 48 elements total
- MFU_INITIAL_VRF capacity: 4096 elements → **0.6% used**

## Transformation

### Step 1: After computing K/V in `mvm_tiled_q()`

After `mvm_tiled_q()` produces K in `ADDSUB_VRF_0/1`, the firmware calls `save_row_tiles()` to write to DRAM. Instead:

1. Insert VREG_MOVE instructions to copy from `ADDSUB_VRF_0/1` to `MFU_INITIAL_VRF` (mem 6) at position-indexed offsets
2. Skip `save_row_tiles()` — the data stays on-chip

The VREG_MOVE pattern:
```c
// Instead of:
save_row_tiles(num_tiles, SAVE_K_BASE + pos * num_tiles * 8,
               MEM_ADDSUB_VRF_0, MEM_ADDSUB_VRF_1);

// Use:
uint32_t cache_offset = pos * num_tiles * NATIVE_DIM;
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_SI(OP_V_WR, 6, cache_offset);  // MFU_INITIAL_VRF[offset]
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
SEND_SI(OP_V_WR, 6, cache_offset + NATIVE_DIM);
```

### Step 2: Load K/V from VRF during attention

In `dot_product_attention()`, the K.T tile build uses:
```c
SEND_LO(OP_V_RD_DRAM, SAVE_K_BASE + p * num_tiles * 8 + tr * 8);
```

Replace with VRF reads:
```c
uint32_t cache_offset = p * num_tiles * NATIVE_DIM + tr * NATIVE_DIM;
SEND_SI(OP_V_RD, 6, cache_offset);  // MFU_INITIAL_VRF[offset]
```

Similarly for V (used in V.T build and V.T re-transpose).

### Step 3: Handle K/V across tile rows

With num_tiles=2, each position has 2 tile rows (tr=0, tr=1). Both must be cached:
```c
uint32_t cache_offset = pos * num_tiles * NATIVE_DIM;
// tr=0 stored at cache_offset
// tr=1 stored at cache_offset + NATIVE_DIM
```

## What NOT to change

- Do NOT modify the Q projection. Q is computed per-position during attention, not pre-computed for all positions. Q should stay in ADDSUB_VRF and be consumed directly.
- Do NOT modify the weight loading (M_RD_DRAM). Weights must come from DRAM.
- Do NOT modify the emulator. Only modify `firmware/bert/bert_layer.c`.
- Do NOT change the numerical computation. The same W×x+b is performed; only the output routing changes.

## Verification

Run the instrumented test to check all operators produce correct output:
```bash
python3 -m pytest tests/integration/test_bert_e2e.py --instrument -k seq6 -s --no-header 2>&1 | grep "max_diff"
```

All values must be < 0.05. Also verify the DRAM reduction:
```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k seq6 -s 2>&1 | grep "DRAM"
```
