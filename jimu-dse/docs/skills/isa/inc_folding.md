---
name: inc-folding
description: Fold V_WR_DRAM+V_RD_DRAM save/load pairs into INC variants on NPU firmware
license: MIT
---

You are optimizing firmware for the NPU (rv-npu), an FPGA neural processing unit.
Your task is to apply the **inc_folding** skill to `firmware/bert/bert_layer.c`.

## Trigger Pattern

A V_WR_DRAM instruction saves a vector to DRAM, and later a V_RD_DRAM instruction
loads it back from the **same address**, with no intervening write to that address.
This pattern wastes instruction bandwidth on redundant address computation.

Detect:
```
SEND_LO(OP_V_WR_DRAM, dram_base + tr * 8);     // save
... (no write to dram_base range) ...
SEND_LO(OP_V_RD_DRAM, dram_base + tr * 8);     // reload
```

## Transformation

Replace the V_WR_DRAM with V_WR_DRAM_INC and the paired V_RD_DRAM with
V_RD_DRAM_INC. The INC variants encode an auto-incrementing address pointer,
eliminating the redundant address computation.

Before:
```c
SEND_LO(OP_V_WR_DRAM, addr);    // save tile row
// ... intervening instructions (K projection, V projection)
SEND_LO(OP_V_RD_DRAM, addr);    // reload tile row
```

After:
```c
// save_row_tiles_inc: uses V_WR_DRAM_INC internally
SEND_LO(OP_V_WR_DRAM_INC, addr);
// ... same intervening instructions ...
// V_RD_DRAM_INC reads from the SAME address. The INC auto-increment
// happens AFTER the read, so both write and read use the same addr.
SEND_LO(OP_V_RD_DRAM_INC, addr);
```

>>> HARDWARE REQUIREMENT: INC variants MUST use LO format <<<
>>> SEND_SI will produce all-zero Q output — this is a hardware fact, not a style choice <<<

Both V_WR_DRAM_INC (opcode 23) and V_RD_DRAM_INC (opcode 22) REQUIRE **LO format**:

```c
SEND_LO(OP_V_WR_DRAM_INC, addr);   // CORRECT — addr encodes the base DRAM address
SEND_LO(OP_V_RD_DRAM_INC, addr);   // CORRECT — addr encodes the base DRAM address

SEND_SI(OP_V_WR_DRAM_INC, 0, stride);  // WRONG — SI format produces Q=[0,0,0,0]
SEND_SI(OP_V_RD_DRAM_INC, 0, stride);  // WRONG — SI format produces Q=[0,0,0,0]
```

**Why LO format is required**: The INC variant encodes the STARTING DRAM address
in the 24-bit LO operand. SI format only encodes a stride, leaving the starting
address uninitialized (dram_addr = 0), which reads from the wrong memory location.

The stride is implicit (NATIVE_DIM = 8 elements). Do NOT use SEND_SI.
Do NOT be tempted to use SEND_SI — it is WRONG for this hardware.

The read address must be the SAME as the write address.
V_RD_DRAM_INC reads from the CURRENT dram_addr, THEN increments.
So both V_WR_DRAM_INC and V_RD_DRAM_INC for the same tile row
use the same base address. Do NOT add +8 to the read address.

## Functions Already Pre-Written

The following functions already exist in the baseline firmware. You do NOT need to create them:

- `save_row_tiles_inc()` — uses `SEND_LO(OP_V_WR_DRAM_INC, addr)` ✅
- `load_row_tiles_inc()` — uses `SEND_LO(OP_V_RD_DRAM_INC, addr)` ✅

Both use LO format (SEND_LO). The INC variants REQUIRE LO format.
Do NOT use SEND_SI with INC variants — they need a 24-bit address.

## What You Need to Change

Replace the 3 save_row_tiles() calls and 3 V_RD_DRAM inline loads in
_process_position() with their INC equivalents.

### _process_position() — Save calls

| Line | Before | After |
|------|--------|-------|
| 221 | `save_row_tiles(num_tiles, save_q, VRF0, VRF1)` | `save_row_tiles_inc(num_tiles, save_q, VRF0, VRF1)` |
| 223 | `save_row_tiles(num_tiles, 0x210, VRF0, VRF1)` | `save_row_tiles_inc(num_tiles, 0x210, VRF0, VRF1)` |
| 225 | `save_row_tiles(num_tiles, 0x220, VRF0, VRF1)` | `save_row_tiles_inc(num_tiles, 0x220, VRF0, VRF1)` |
| 276 | `save_row_tiles(num_tiles, save_res, VRF0, VRF1)` | `save_row_tiles_inc(num_tiles, save_res, VRF0, VRF1)` |

### _process_position() — Inline load calls

These are in the attention loop. Replace SEND_LO(OP_V_RD_DRAM, ...) with
SEND_LO(OP_V_RD_DRAM_INC, ...) — SAME address, SAME format, just different opcode.

| Line | Before | After |
|------|--------|-------|
| 248 | `SEND_LO(OP_V_RD_DRAM, save_q + tr * 8)` | `SEND_LO(OP_V_RD_DRAM_INC, save_q + tr * 8)` |
| 250 | `SEND_LO(OP_V_RD_DRAM, 0x210 + tr * 8)` | `SEND_LO(OP_V_RD_DRAM_INC, 0x210 + tr * 8)` |
| 262 | `SEND_LO(OP_V_RD_DRAM, 0x220 + tr * 8)` | `SEND_LO(OP_V_RD_DRAM_INC, 0x220 + tr * 8)` |

### apply_layernorm()

| Line | Before | After |
|------|--------|-------|
| 184 | `SEND_LO(OP_V_WR_DRAM, scratch_addr + tr * stride)` | `SEND_LO(OP_V_WR_DRAM_INC, scratch_addr + tr * stride)` |
| 195 | `SEND_LO(OP_V_RD_DRAM, scratch_addr + tr * stride)` | `SEND_LO(OP_V_RD_DRAM_INC, scratch_addr + tr * stride)` |

## Constraints

1. Only modify `firmware/bert/bert_layer.c`. Do NOT modify any other file.
2. Only modify the 8 call sites listed in the table above (replace function
   name or opcode). Do NOT change any other logic.
3. Do NOT modify `save_row_tiles_inc()` or `load_row_tiles_inc()` — they
   are already correct. Changing `OP_V_RD` to `OP_V_WR` inside these
   functions will break the firmware.
4. The file must remain valid C (compilable by GCC for RISC-V).

## Verification

After writing the file, you MUST self-verify by running:

```bash
python3 -m pytest tests/integration/test_bert_e2e.py -k "seq2" -q
```

Check the exit code (it should be 0) and look at the output for:
- 'PASSED' — numerical verification passed
- 'FAILED' — numerical verification failed
- 'ERROR' — compilation or runtime error

Do NOT skip this step. The pipeline checks exit code and max_diff.
If the test fails, read the output to find the bug, fix it, and re-run.
Max 3 retries.

## Constraints

1. Only modify `firmware/bert/bert_layer.c`. Do NOT modify ANY other file.
2. Do NOT modify the emulator. Do NOT modify any file outside bert_layer.c.
3. Preserve all existing functions — add new ones, don't delete.
4. The file must remain valid C (compilable by GCC for RISC-V).

## Example: save_row_tiles_inc

```c
static void save_row_tiles_inc(uint32_t num_tiles, uint32_t dram_base,
                                uint32_t vrf_first, uint32_t vrf_second)
{
    uint32_t tr;
    for (tr = 0; tr < num_tiles; tr++) {
        uint32_t vrf = (tr == 0) ? vrf_first : vrf_second;
        SEND_SI(OP_V_RD, vrf, 0);
        SEND_LO(OP_V_WR_DRAM_INC, dram_base + tr * 8);
    }
}

NOTE: The function MUST use `dram_base` (the parameter), not a hardcoded
address. V_WR_DRAM_INC uses LO format: SEND_LO(OP_V_WR_DRAM_INC, addr).
Do NOT use SEND_SI(OP_V_WR_DRAM_INC, 0, stride) — SI format is wrong.
The stride is implicit (NATIVE_DIM = 8).
```
