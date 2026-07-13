# AdderBoard Models on NPU — Compatibility & Status

Summary of which AdderBoard challenge submissions can run on the NPU
emulator + RISC-V firmware stack.

## Current Status

| Model | Params | Architecture | NPU Status | Tests |
|---|---|---|---|---|
| **cosminscn_130p** | 130 | 1L GPT, d=4, 2h, ReLU, sin PE | ✅ Ported, FP32 only | 20 tests |
| **dimopep_140p** | 140 | 1L Qwen3, d=4, 1h, SwiGLU, RMSNorm, RoPE | ✅ Ported, FP32 + FP16 | 28 tests |
| cosminscn_66p | 66 | 1L GPT, d=4, 2h, ReLU, sin PE | Not attempted | — |
| lichengliu03_50p | 50 | 1L GPT, d=4, 2h, ReLU, sin PE, rank-1 | Not attempted | — |
| zcbtrak_6p | 6 | 1L Qwen, d=2, SiLU, RoPE, float64 | Not attempted | — |

Tests are consolidated into 3 parametrized files (49 total, see `docs/how-to-run.md`).
All models share 6 deterministic test cases plus randomized bulk tests per phase.

---

## Ported Models in Detail

### cosminscn_130p (FP32, hand-crafted)

- **Branch**: `explore/closed-loop-fw-optimization-x86`
- **Emulator class**: `NpuFP32` (no FP16 truncation)
- **Architecture**: d=4, 2 heads × head_dim=2, ff_dim=4
- **Data path**: All operations on NPU — attention (tiled VecToMatRow + SPU softmax),
  MLP (ReLU), rank-1 c_proj (MV_MUL + VV_MUL), LM head first 4 logits (NPU),
  last 6 logits (RISC-V fallback)
- **Firmware**: `adderboard/firmware/adder_130p.c` — single-phase, MMIO data exchange
- **FP16 viability**: ❌ Not FP16-safe — 1000-scale carry detection MLP
  produces 99.2% failure at FP16 precision
- **Source files**:
  - `adderboard/golden/golden_130p.py` — golden reference
  - `adderboard/layout/layout_130p.py` — DRAM layout
  - `adderboard/firmware/adder_130p.c` — RISC-V firmware

### dimopep_140p (FP32 + FP16, trained)

- **Branch**: `explore/closed-loop-fw-optimization-x86`
- **Weights**: `adderboard/models/140p/s44_targeted_final_fp16.pt` (140 params, all <74 → FP16-safe)
- **Emulator classes**: `NpuFP32` (FP32) and `NpuDeviceMini` (FP16 truncation)
- **Architecture**: d=4, 1 head × head_dim=4, ff_dim=4
- **Data path**:
  - ISS pre-fill: embedding, RMSNorm, Q/KV projections, QK norms, RoPE
  - Phase 1 firmware: tiled attention (VecToMatRow + SPU softmax) + O=Q^T + residual
  - ISS gap: RMSNorm(norm2), gate/up projections
  - Phase 2 firmware: SiLU (V_SIGM+VV_MUL), gate×up (VV_MUL), W_down (MV_MUL) + residual
  - ISS scalar: RMSNorm(norm_final), LM head (embedding.T), argmax
- **Firmware**: `adderboard/firmware/adder_140p.c` — two-phase (attention then FFN),
  phase flag at DRAM[0x1F00]
- **FP16 accuracy**: ~99% → matches model's trained accuracy at FP16
- **Key design decisions**:
  - W_q^T stored separately at 0xD00 for O projection (MV_MUL needs transposed W)
  - SiLU = V_SIGM + VV_MUL (two existing ops, no new hardware)
  - RMSNorm and LM head use RISC-V scalar fallback
  - Two-phase firmware: ISS computes norm2/gate/up between phases
- **Source files**:
  - `adderboard/golden/golden_140p.py` — golden reference (pure numpy Qwen3)
  - `adderboard/layout/layout_140p.py` — DRAM layout
  - `adderboard/firmware/adder_140p.c` — RISC-V firmware

---

## Forward Pass — Step-by-Step Operator Trace

### 130p (cosminscn_130p) — d=4, 2h×hd2, ff=4, ReLU, sin PE

| Step | Op | What | Dimensions | Weight |
|---|---|---|---|---|
| 1 | V_RD_DRAM + MV_MUL | Factorized embed A·B | 10×1 → 1×4 / token | embed_A (10×1), embed_B (1×4) |
| 1 | VV_ADD | Sinusoidal PE | 4 | PE_table (T×4) |
| 2 | M_RD_DRAM + MV_MUL | QKV projection | 12×4 @ 4 → 12 | c_attn (12×4) |
| 3 | VecToMatRow | K^T tile build (per head) | 2 → 4×4 MRF | — |
| 3 | MV_MUL | Attention scores | 4×4 @ 2 → 4 | — |
| 3 | VV_MUL | Scale ×0.25 | 4 | — |
| 3 | VV_ADD | Causal mask | 4 | — |
| 3 | SPU_MAX_REDUCE | Global max | scalar | — |
| 3 | VV_B_SUB_A + V_EXP | scores − max, exp | 4 | — |
| 3 | SPU_ADD_REDUCE + S_RECIP | Global sum, 1/sum | scalar | — |
| 3 | VV_MUL | prob = exp × inv_sum | 4 | — |
| 3 | VecToMatRow | V^T tile build | 2 → 4×4 MRF | — |
| 3 | MV_MUL | V^T @ prob → ctx | 4×4 @ 4 → 2 | — |
| 4 | M_RD_DRAM + MV_MUL | O projection | 4×4 @ 4 → 4 | c_proj (4×4) |
| 4 | VV_ADD | Residual add | 4 | — |
| 5 | M_RD_DRAM + MV_MUL | MLP input | 4×4 @ 4 → 4 | c_fc (4×4) |
| 5 | VV_ADD + V_RELU | Bias + ReLU | 4 | c_fc_bias (4) |
| 6 | M_RD_DRAM + MV_MUL | Rank-1 dot (MLP out) | 4×1 @ 4 → 1 | c_proj_u (4×1) |
| 6 | VV_MUL | Broadcast × v | 4 | c_proj_v (1×4) |
| 6 | VV_ADD | Residual add | 4 | — |
| 7 | M_RD_DRAM + MV_MUL | Rank-1 dot (LM) | 10×1 @ 4 → 1 | lm_u (10×1) |
| 7 | VV_MUL | Broadcast × v | 4 | lm_v (1×4) |
| 7 | RISC-V scalar | Last 6 logits | 10 total | lm_bias (10) |

**Per forward step**: ~120-150 NPU instructions, 8 structured MV_MUL + tiled attention.

### 140p (dimopep_140p) — d=4, 1h×hd4, ff=4, SwiGLU, RMSNorm, RoPE

| Step | Op | What | Dimensions | Weight |
|---|---|---|---|---|
| 1 | V_RD_DRAM | Embed lookup | 4 / token | embedding (10×4) |
| 2 | RISC-V scalar | RMSNorm (norm1) | 4 | norm1 (4) |
| 3 | M_RD_DRAM + MV_MUL | Q projection | 4×4 @ 4 → 4 | W_q (4×4) |
| 3 | M_RD_DRAM + MV_MUL | KV projection (tied K=V) | 4×4 @ 4 → 4 | W_kv (4×4) |
| 4 | RISC-V scalar | RMSNorm (q_norm) | 4 | q_norm (4) |
| 4 | RISC-V scalar | RMSNorm (k_norm) | 4 | k_norm (4) |
| 5 | VV_MUL + VV_ADD | RoPE apply | 4 / token | cos_table, sin_table (T×2) |
| 6 | VecToMatRow | K^T tile build | 4 → 4×4 MRF | — |
| 6 | MV_MUL | Attention scores | 4×4 @ 4 → 4 | — |
| 6 | VV_MUL | Scale ×0.5 | 4 | — |
| 6 | VV_ADD | Causal mask | 4 | — |
| 6 | SPU_MAX_REDUCE | Global max | scalar | — |
| 6 | VV_B_SUB_A + V_EXP | scores − max, exp | 4 | — |
| 6 | SPU_ADD_REDUCE + S_RECIP | Global sum, 1/sum | scalar | — |
| 6 | VV_MUL | prob = exp × inv_sum | 4 | — |
| 6 | VecToMatRow | V^T tile build | 4 → 4×4 MRF | — |
| 6 | MV_MUL | V^T @ prob → ctx | 4×4 @ 4 → 4 | — |
| 7 | M_RD_DRAM + MV_MUL | O projection (tied Q^T) | 4×4 @ 4 → 4 | W_q^T (4×4, DRAM 0xD00) |
| 7 | VV_ADD | Residual add | 4 | — |
| 8 | RISC-V scalar | RMSNorm (norm2) | 4 | norm2 (4) |
| 9 | MV_MUL | Gate projection | 4×4 @ 4 → 4 | W_gate (4×4) |
| 9 | MV_MUL | Up projection | 4×4 @ 4 → 4 | W_up (4×4) |
| 10 | V_SIGM | sigmoid(gate) | 4 | — |
| 10 | VV_MUL | gate × up | 4 | — |
| 10 | M_RD_DRAM + MV_MUL | Down projection | 4×4 @ 4 → 4 | W_down (4×4) |
| 10 | VV_ADD | Residual add | 4 | — |
| 11 | RISC-V scalar | RMSNorm (norm_final) | 4 | norm_final (4) |
| 12 | RISC-V scalar | LM head (tied embedding) | 4×10 → 10 logits | embedding.T (4×10) |

**Per forward step**: ~80 NPU instructions, 7 MV_MUL + tiled attention.

---

## Weight Tensor Breakdown

### 130p — 10 tensors, 130 params

| Tensor | Shape | Elements |
|---|---|---|
| embed_A | 10×1 | 10 |
| embed_B | 1×4 | 4 |
| c_attn | 12×4 | 48 |
| c_proj | 4×4 | 16 |
| c_fc | 4×4 | 16 |
| c_fc_bias | 4 | 4 |
| c_proj_u | 4×1 | 4 |
| c_proj_v | 1×4 | 4 |
| lm_u | 10×1 | 10 |
| lm_v / lm_bias | 1×4 + 10 | 14 |
| **Total** | | **130** |

### 140p — 11 tensors, 140 params

| Tensor | Shape | Elements |
|---|---|---|
| embedding | 10×4 | 40 |
| norm1 | 4 | 4 |
| norm2 | 4 | 4 |
| norm_final | 4 | 4 |
| W_q | 4×4 | 16 |
| W_kv | 4×4 | 16 |
| q_norm | 4 | 4 |
| k_norm | 4 | 4 |
| W_gate | 4×4 | 16 |
| W_up | 4×4 | 16 |
| W_down | 4×4 | 16 |
| W_q^T (in DRAM) | 4×4 | 16* |
| cos_table + sin_table | T×4 | DRAM |
| **Total** | | **140** |

\* W_q^T is a view of W_q, not an additional weight parameter. Stored separately
in DRAM at 0xD00 to satisfy MV_MUL orientation for O projection.

---

## NPU Instruction Set (Relevant Ops Used)

| Opcode | Name | Used By | Purpose |
|---|---|---|---|
| 7 | MV_MUL | Both | MRF × pipeline → pipeline (matmul) |
| 8 | VV_ADD | Both | Elementwise add |
| 11 | VV_MUL | Both | Elementwise multiply |
| 12 | V_SIGM | 140p | Sigmoid (for SiLU computation) |
| 14 | V_RELU | 130p | ReLU activation |
| 15 | VV_B_SUB_A | Both | scores − max (softmax) |
| 20 | V_RD_DRAM | Both | Vector load from DRAM |
| 21 | V_WR_DRAM | Both | Vector store to DRAM |
| 24 | M_RD_DRAM | Both | Matrix tile load from DRAM |
| 35 | S_RECIP | Both | Reciprocal (1/sum for softmax) |
| 37 | V_EXP | Both | Exp via 256-entry LUT (softmax) |

**Not used** (available but firmware bypasses):
- 13 V_TANH — no model uses tanh
- 42 V_GELU — 140p uses V_SIGM+VV_MUL for SiLU instead
- 43 V_FUNC/SOFTMAX — hardware softmax unused; firmware uses tiled SPU softmax
- 43 V_FUNC/LAYERNORM — not used for RMSNorm; RISC-V scalar fallback

---

## Key Constraints

- **Pipeline width**: 4 floats → 10-class logits need 3 passes or RISC-V fallback
- **NATIVE_DIM**: 4 (matrix tiles load 4×4 blocks)
- **No SiLU opcode**: Emulated via V_SIGM + VV_MUL
- **No RMSNorm opcode**: RISC-V scalar compute
- **SPU_ADD_REDUCE is cumulative**: Must zero SRF[1] between queries (MMIO write)
- **No FP16 overflow**: -1e30 mask values overflow to -inf in FP16 (correct behavior)
- **Phase flag encoding**: DRAM[0x1F00] must use raw uint32 bit pattern, not float32

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Test Script (Python)                      │
│  Loads weights, pre-computes embedding/norms/RoPE, fills DRAM   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Phase 1     │  │   Phase 2     │  │  Phase 2+ISS  │
│ Python driver │  │ Replay stream │  │  RISC-V CPU   │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NPU Emulator                                │
│  NpuFP32 (FP32, no truncation)  or  NpuDeviceMini (FP16)        │
│  ~80-150 instructions per forward step                           │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Python directly sends NPU instructions via _push_instruction().
Phase 2 (replay): Captures firmware's instruction stream, replays in Python.
Phase 2 (ISS): ISS runs compiled RISC-V firmware → MMIO → NPU.
```

## Precision: FP32 vs FP16

| Model | FP32 Logit Diff | FP16 Logit Diff | FP32 Bulk | FP16 Bulk |
|---|---|---|---|---|
| 130p | <0.001 | N/A (not FP16-safe) | 50/50 (100%) | ❌ 99% failure |
| 140p | <0.001 | <0.76 | 50/50 (~98%) | ~90-100% (~99% model accuracy) |

The 140p FP16 error rate matches the model's inherent 99.0% FP16 accuracy
on 10K test. All decisive (boundary-close) digits are hard — the model
learned FP16-safe carry detection via softmax attention, not 1000-scale
MLP thresholds. The 130p cannot run in FP16 at all due to the 1000-scale
carry MLP which saturates FP16 precision.
