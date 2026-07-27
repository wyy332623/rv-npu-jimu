# NPU Porting Plan — Completed

Both AdderBoard challenge models are ported to the NPU emulator + RISC-V
firmware stack. This document records the final architecture and what was
done at each step.

## Status: DONE ✅

| Step | 130p (hand-crafted) | 140p (trained) |
|---|---|---|
| 1. DRAM layout | `npu_dram_layout.py` | `npu_dram_layout_140p.py` |
| 2. Golden reference | `golden_130p.py` | `golden_140p.py` (pure numpy Qwen3) |
| 3. NPU instruction stream | `test_phase1b_full.py` (9 tests) | `test_140p_phase1.py` (8 tests) |
| 4. RISC-V firmware | `adder_130p.c` (single-phase) | `adder_140p.c` (two-phase) |
| 5. ISS integration | `test_phase2_iss.py` (6 tests) | `test_140p_phase2.py` (5 tests) |
| Bonus: FP16 path | ❌ (not FP16-safe) | `test_140p_fp16.py` (4 tests) |

---

## 130p Architecture (FP32 only)

```
Input: 22-token prompt (a=0..9 LSB-first, b=0..9 LSB-first)
Output: 11 autoregressive steps (sum LSB-first)

Data flow (all on NPU except last 6 LM logits):
  Embedding → PE → Q/K/V → Attention(tiled, 2h×hd2) → c_proj → residual
  → MLP c_fc + ReLU + bias → MLP rank-1 c_proj → residual
  → LM head (4 logits on NPU, 6 via RISC-V) → argmax

Firmware: single-phase, MMIO window for data exchange
Key ops: MV_MUL, VV_ADD, VV_MUL, V_RELU, VV_B_SUB_A, V_EXP, S_RECIP
```

## 140p Architecture (FP32 + FP16)

```
Input: 24-token prompt (a=0..9 LSB-first + 2 separators + b=0..9 LSB-first)
Output: 11 autoregressive steps (sum LSB-first)

ISS pre-fill (Python):
  Embedding → RMSNorm(norm1) → W_q, W_kv → QK norms → RoPE

Phase 1 firmware (NPU):
  Attention (tiled, 1h×hd4) → O=Q^T → residual → write attn_res to DRAM

ISS gap (Python):
  Read attn_res from DRAM → RMSNorm(norm2) → W_gate, W_up → write to S_BASE2

Phase 2 firmware (NPU):
  SiLU(gate) via V_SIGM+VV_MUL → gate×up via VV_MUL → W_down via MV_MUL
  → residual + write last_h to FW_LAST_H

ISS scalar (Python):
  Read last_h → RMSNorm(norm_final) → LM head (embedding.T) → argmax

Firmware: two-phase, phase flag at DRAM[0x1F00] (raw uint32)
Key ops: MV_MUL, VV_ADD, VV_MUL, V_SIGM, VV_B_SUB_A, V_EXP, S_RECIP
```

---

## Key Design Decisions for 140p

### 1. W_q^T Stored Separately (DRAM 0xD00)

MV_MUL computes `MRF @ pipeline`, but the golden reference computes
`ctx @ W_q` for the O projection. Since W_q is not symmetric,
`W_q @ ctx ≠ ctx @ W_q`. Storing `W_q^T` at a separate DRAM address
fixes this: `W_q^T @ ctx = ctx @ W_q`.

### 2. Two-Phase Firmware

ISS cannot precompute norm2/gate/up before Phase 1 runs because it
doesn't know the attention residual. The Python ISS gap reads
`attn_res` from DRAM between phases, computes norm2/gate/up, writes
to `S_BASE2` (0x3000), then launches Phase 2.

### 3. Phase Flag as Raw uint32

The C firmware uses `npu_read_reg()` which interprets DRAM bytes as
raw integers. Writing `1.0f` (bit pattern 0x3F800000) ≠ 1 in C.
Must write raw uint32 bit pattern: 0x00000000 for Phase 1,
0x00000001 for Phase 2.

### 4. SiLU = V_SIGM + VV_MUL

No new hardware needed. Two existing NPU ops compute `sigmoid(x) * x`.
Verified correct via NpuFP32 tests (exact match to numpy silu).

### 5. SRF Initialization Between Queries

SPU_ADD_REDUCE and SPU_MAX_REDUCE are cumulative — values persist
across queries. Fix:
- SRF[0] (max): reset with SPU_ADD_REDUCE(-inf) → forces to -inf
- SRF[1] (sum): zero via `npu_write_reg(NPU_SRF_BASE + 4, 0)`

---

## Emulator Changes Made

| Change | File | Reason |
|---|---|---|
| Added `OP_V_SIGM` handler | `emulator/npu_device_mini.py` | Was silently ignored |
| Added `OP_V_TANH` handler | `emulator/npu_device_mini.py` | Completeness |
| `sigmoid` in ctypes setup | `emulator/npu_device_mini.py` | V_SIGM needs it |
| SRF reset via MMIO | `adderboard/firmware/adder_140p.c` | Cumulative SPU reduces |
| NpuFP32 class | `emulator/npu_fp32.py` | FP32 verification mode |

---

## Bug Fixes During Development

| Bug | Manifestation | Fix |
|---|---|---|
| VV_A_SUB_B / VV_B_SUB_A always added | `scores - max` computed `scores + max` | Fixed `_vv_add_sub()` |
| V_EXP was no-op | `exp()` returned 0 | Added V_EXP to `_v_activation()` |
| SPU reduces overwrote instead of accumulating | Only last tile's max/sum | Made SPU_ADD_REDUCE/SPU_MAX_REDUCE cumulative |
| S_RECIP, S_SQRT were stubs | `inv_sum` always 0 | Implemented `_spu_func()` |
| V_SIGM not implemented | SiLU computed wrong | Added V_SIGM handler |
| SRF[0] not reset between queries (140p) | CTX[1] used wrong max (3.357 vs 3.263) | Use SPU_ADD_REDUCE(-inf) instead of SPU_MAX_REDUCE(-inf) |
| SRF[1] not zeroed between queries (140p) | Incorrect softmax sum (2.196 vs 1.196) | Add MMIO zero between queries |
| Phase flag as float32 (140p) | C comparison `!= 1` failed | Write raw uint32 bit pattern |
| ISS MMIO window too small | SRF window overlapped DRAM | Extended to 0x10000 |
