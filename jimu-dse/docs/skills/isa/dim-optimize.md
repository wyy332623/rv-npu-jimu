---
name: dim-optimize
version: 1.0.0
description: Restructure firmware from multi-tile to single-tile projections
---

# Dim-Optimize Skill

## Problem

When `NATIVE_DIM < hidden_size`, each projection requires a `num_tiles × num_tiles`
tiled matmul. For dim=2, hidden=4: `num_tiles=2`, each projection needs 4 `MV_MUL`
instructions. The goal is to restructure the firmware so that **NATIVE_DIM matches
hidden_size**, making each projection a single `M_RD_DRAM` + `MV_MUL` pair.

## Target Configuration

```
NATIVE_DIM = 4, hidden_size = 4, num_head = 2
MAT_SIZE = NATIVE_DIM × NATIVE_DIM = 16
head_size = hidden_size / num_head = 2
num_tiles = 1  (single tile!)
heads_per_tile = NATIVE_DIM / head_size = 2
```

## Transformation Steps

### 1. Simplify the projection functions

The `mvm_tiled_q()` function has a `for tc, for tr` loop that iterates 2×2=4 times.
For single-tile, this collapses to a straight-line sequence:

**Before (dim=2, num_tiles=2, MAT_SIZE=4):**
```c
for (tc = 0; tc < num_tiles; tc++) {
    SEND_LO(OP_V_RD_DRAM, vec_chunk_addr + tc * NATIVE_DIM);
    SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
    SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
    for (tr = 0; tr < num_tiles; tr++) {
        SEND_LO(OP_M_RD_DRAM, mat_dram_base + (tr * num_tiles + tc) * MAT_SIZE);
        SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);
        SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_1, 0);
        SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);
        SEND_SI(OP_MV_MUL, 0, 0);
        // ... accumulate via VV_ADD ...
    }
}
// Bias add per tile-row
for (tr = 0; tr < num_tiles; tr++) {
    SEND_LO(OP_V_RD_DRAM, bias_dram_base + tr * NATIVE_DIM);
}
// Move tr=1 accumulator to VRF_1
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_2, 0);
SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_1, 0);
```

**After (dim=4, num_tiles=1, MAT_SIZE=16):**
```c
// Single load of full input vector
SEND_LO(OP_V_RD_DRAM, input_vec_addr);
SEND_SI(OP_V_WR, MEM_MVM_INITIAL_VRF, 0);
SEND_SI(OP_V_RD, MEM_MVM_INITIAL_VRF, 0);

// Single load of full weight matrix (4×4 = 16 elements)
SEND_LO(OP_M_RD_DRAM, mat_dram_base);
SEND_SI(OP_M_WR, MEM_MATRIX_RF, 0);

// Single MV_MUL — the MRF holds the full 4×4 matrix
SEND_SI(OP_MV_MUL, 0, 0);
// Pipeline now has 4 elements = full hidden_size result

// No accumulation needed — single tile

// Bias add (single tile-row, full bias)
SEND_SI(OP_V_WR, MEM_MVM_ACC_VRF, 0);  // save result
SEND_SI(OP_V_RD, MEM_FILL, 0);          // or load bias from DRAM
SEND_LO(OP_V_RD_DRAM, bias_dram_base);
SEND_SI(OP_V_RD, MEM_MVM_ACC_VRF, 0);
SEND_SI(OP_VV_ADD, 0, 0);
SEND_SI(OP_V_WR, MEM_ADDSUB_VRF_0, 0);  // store to a single tile-row VRF
```

### 2. Unify `mvm_tiled_q` and `mvm_tiled_vrf`

Since both now do the same single-tile operation, merge them into one function.

### 3. Update attention for `heads_per_tile=2`

At dim=4 with head_size=2, one tile row contains **2 heads** packed in one vector:

```
VRF[6] vector: [h0_q0, h0_q1, h1_q0, h1_q1]
                 head 0     head 1
Mask 0x03 → head 0: [1, 1, 0, 0]
Mask 0x0C → head 1: [0, 0, 1, 1]
```

The `dot_product_attention` function needs:
- Outer loop over `tr=0` only (num_tiles=1)
- Inner loop over `h=0..heads_per_tile-1` (h=0, h=1)
- Each head uses a mask to select its head_size=2 slice from the dim=4 vector
- Write context back with `REG_WRITE_VECTOR_MASK`

### 4. Simplify `save_row_tiles` and `load_and_add_row_tiles`

At num_tiles=1, these operate on a single tile-row:
```c
SEND_SI(OP_V_RD, MEM_ADDSUB_VRF_0, 0);
SEND_LO(OP_V_WR_DRAM, dram_base);
```

### 5. Update `apply_layernorm`

At num_tiles=1, the layernorm operates on a single tile-row:
- Single gamma/beta load
- Single VRF save/restore
- No need for tile-1 backup logic

## Self-Verification

After modifying the firmware, validate with the project's testing instructions. You can use the `self-verify` skill or check your instructions for the exact verify and convergence check commands to run.

## Cost Model

| Metric | dim=2 (baseline) | dim=4 (optimized) |
|--------|------------------|-------------------|
| MV_MUL per projection | 4 | **1** |
| M_RD_DRAM per projection | 4 tiles | **1 tile** |
| Total M_RD_DRAM (seq=6) | 144 ops | **36 ops** |
| VV_ADD per projection | 2 (tile accumulation) | **0** |
| VRF_ADDSUB usage | VRF_0, VRF_1, VRF_2 | **VRF_0 only** |
| Attention heads per tile row | 1 | 2 (masked) |
