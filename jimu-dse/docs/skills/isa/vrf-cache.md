---
name: vrf-cache
version: 1.0.0
category: isa
description: Redirect K/V tensor data from DRAM roundtrip to on-chip VRF cache
license: MIT
---

# VRF Cache Skill

## Problem

The firmware computes `K = Wk*x + bk` and `V = Wv*x + bv` for each position, saves both tensors to DRAM via `save_row_tiles()`, then the attention code reloads them via `V_RD_DRAM`. This roundtrip is unnecessary when the target VRF has capacity.

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

K and V are distinct live tensors and must occupy disjoint cache ranges. Define
the layout once and use the same bases in both producers and consumers:

```c
uint32_t tensor_span = seq_len * num_tiles * NATIVE_DIM;
uint32_t k_cache_base = 0;
uint32_t v_cache_base = k_cache_base + tensor_span;
uint32_t kv_cache_end = v_cache_base + tensor_span;
```

Apply this optimization only when `kv_cache_end` does not exceed the selected
VRF capacity. For dim=2, hidden=4, seq_len=6, K occupies `[0, 24)` and V
occupies `[24, 48)` in `MFU_INITIAL_VRF`.

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
uint32_t tensor_offset = pos * num_tiles * NATIVE_DIM;
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_SI(OP_V_WR, 6, k_cache_base + tensor_offset);
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
SEND_SI(OP_V_WR, 6, k_cache_base + tensor_offset + NATIVE_DIM);
```

The V producer uses the identical position/tile formula with `v_cache_base`,
never `k_cache_base`.

### Step 2: Load K/V from VRF during attention

In `dot_product_attention()`, the K.T tile build uses:
```c
SEND_LO(OP_V_RD_DRAM, SAVE_K_BASE + p * num_tiles * 8 + tr * 8);
```

Replace with VRF reads:
```c
uint32_t tensor_offset = p * num_tiles * NATIVE_DIM + tr * NATIVE_DIM;
SEND_SI(OP_V_RD, 6, k_cache_base + tensor_offset);
```

V reads (used in V.T build and V.T re-transpose) must instead use:

```c
SEND_SI(OP_V_RD, 6, v_cache_base + tensor_offset);
```

### Step 3: Handle K/V across tile rows

With num_tiles=2, each position has 2 tile rows (tr=0, tr=1). Both must be cached:
```c
uint32_t tensor_offset = pos * num_tiles * NATIVE_DIM;
// K tr=0: k_cache_base + tensor_offset
// K tr=1: k_cache_base + tensor_offset + NATIVE_DIM
// V tr=0: v_cache_base + tensor_offset
// V tr=1: v_cache_base + tensor_offset + NATIVE_DIM
```

Do not reuse the K offsets for V. Doing so overwrites K before attention and
can escape an output-only test when a particular stimulus is insensitive to
the substitution.

## What NOT to change

- Do NOT modify the Q projection. Q is computed per-position during attention, not pre-computed for all positions. Q should stay in ADDSUB_VRF and be consumed directly.
- Do NOT modify the weight loading (M_RD_DRAM). Weights must come from DRAM.
- Do NOT modify the emulator. Only modify `firmware/bert/bert_layer.c`.
- Do NOT change the numerical computation. The same W×x+b is performed; only the output routing changes.

## Verification

Run the end-to-end test. It checks every output position, validates K/V values
at their actual storage boundary, and rejects overlapping MFU VRF ranges:
```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k seq6 -s --no-header
```

All values must be < 0.05 and the K/V storage check must pass. Use
`--instrument` when operator-boundary diagnostics are also needed. Verify the
DRAM reduction separately:
```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k seq6 -s 2>&1 | grep "DRAM"
```
