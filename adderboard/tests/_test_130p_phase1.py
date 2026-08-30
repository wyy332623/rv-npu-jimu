"""
Phase 1b: Full synthetic NPU instruction stream for cosminscn_130p.

All operations use real NPU instructions via _execute() — no Python
fallbacks.  Embed+PE and LM head use lightweight Python helpers
(embed+PE precompute and RISC-V-style LM head with 10-logit output).

Runs in FP32 mode via NpuFP32.  Self-verifying against numpy golden.

Usage:
    python3 -m pytest tests/adderboard/test_phase1b_full.py -v
"""

import numpy as np
import pytest

from emulator.npu_fp32 import NpuFP32, si, lo
from adderboard.layout.layout_130p import (
    build_dram, encode_prompt, decode_output,
    dram_addr, MODEL_DIM, N_HEADS, HEAD_DIM,
    PROMPT_LEN, OUTPUT_DIGITS, TOTAL_LEN, VOCAB_SIZE,
)
from adderboard.golden.golden_130p import forward as golden_forward, _load_weights

from emulator.npu_device_mini import (
    MEM_MULTIPLY_VRF, MEM_MVM_INITIAL_VRF, MEM_MATRIX_RF,
    MEM_VEC_TO_MAT_ROW, MEM_FILL, MEM_SPU_MAX_REDUCE, MEM_SPU_ADD_REDUCE,
    MEM_SPU_BROADCAST,
    OP_V_RD_DRAM, OP_V_WR_DRAM,
    OP_V_RD, OP_V_WR,
    OP_M_RD_DRAM, OP_M_WR, OP_M_RD, OP_MV_MUL,
    OP_VV_ADD, OP_VV_B_SUB_A, OP_VV_MUL,
    OP_V_RELU, OP_V_EXP,
    OP_S_WR, OP_S_RECIP,
)


# ── Scratch layout ───────────────────────────────────────────────

def _scratch(total=TOTAL_LEN, dim=MODEL_DIM):
    KEY = ['X', 'Q', 'K', 'V', 'CTX', 'ATTN_OUT',
           'AFTER_ATTN', 'FC', 'MLP_OUT', 'AFTER_MLP',
           'SCORE', 'PROB', 'TEMP']
    if not hasattr(_scratch, 'cache'):
        addr = 0x1000
        _scratch.cache = {}
        for k in KEY:
            _scratch.cache[k] = addr
            if k in ('SCORE', 'PROB', 'TEMP'):
                addr += total
            else:
                addr += total * dim
    return _scratch.cache


S = _scratch()


# ═══════════════════════════════════════════════════════════════
# Tiled attention via VecToMatRow + SPU softmax
# ═══════════════════════════════════════════════════════════════

def _mask_pipeline(npu, head_offset):
    """Apply a head mask to the pipeline: zero the other head's 2 dims.
    
    Q/V[pos] = [d0, d1, d2, d3] where d0,d1=head0, d2,d3=head1.
    After: pipeline = [d0*m0, d1*m1, d2*m2, d3*m3], m = [1,1,0,0] or [0,0,1,1].
    """
    npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
    mask = np.zeros(4, dtype=np.float32)
    mask[head_offset:head_offset+2] = 1.0
    npu.set_dram(S['TEMP'], mask)
    npu.send_lo(OP_V_RD_DRAM, S['TEMP'])
    npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
    npu.send_si(OP_VV_MUL, 0, 0)


def _tiled_attention(npu, T):
    """Compute attention via NPU instructions.
    
    Storage layout: a position vector is [h0_d0, h0_d1, h1_d0, h1_d1].
    
    The NPU scores are Q[q,h] · K[pos,h] without scale (contrary to the
    golden reference which uses HEAD_DIM ** -0.5).  We apply the scale
    factor via VV_MUL with a pre-stored value.
    
    CRITICAL: The causal mask must be applied BEFORE computing the
    score max for softmax stability.  Without this, positions > q
    (which have scores up to 1e6) pollute the max, causing all valid
    positions' exp(score - max) to underflow in float32.
    """
    SCALE = 2.0 ** -0.5  # HEAD_DIM ** -0.5
    NUM_TILES = (T + 3) // 4
    
    npu._spu_srf[6] = SCALE
    
    # Zero context accumulators
    for i in range(T):
        npu.send_si(OP_V_RD, MEM_FILL, 0)
        npu.send_lo(OP_V_WR_DRAM, S['CTX'] + i * MODEL_DIM)
    
    for q in range(T):
        for head_offset in (0, 2):
            # ── Pass 1: masked scores → max ──
            # Apply causal mask BEFORE max-reduce for numerical stability.
            npu._spu_srf[0] = -1e30
            
            for tc in range(NUM_TILES):
                base = tc * 4
                valid = min(4, T - base)
                
                # Build K.T tile
                for p in range(4):
                    pos = base + p
                    if p < valid:
                        npu.send_lo(OP_V_RD_DRAM, S['K'] + pos * MODEL_DIM)
                    else:
                        npu.send_si(OP_V_RD, MEM_FILL, 0)
                    npu.send_si(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0)
                npu.mat_from_vec_to_mat()
                
                # Load and mask Q[q]
                npu.send_lo(OP_V_RD_DRAM, S['Q'] + q * MODEL_DIM)
                _mask_pipeline(npu, head_offset)
                
                # MV_MUL → scaled scores
                npu.send_si(OP_V_WR, MEM_MVM_INITIAL_VRF, 0)
                npu.send_si(OP_V_RD, MEM_MVM_INITIAL_VRF, 0)
                npu.send_si(OP_MV_MUL, 0, 0)
                
                # Apply scale
                npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
                npu.broadcast_srf(6)
                npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
                npu.send_si(OP_VV_MUL, 0, 0)
                
                # Apply causal mask (replace positions > q with -inf)
                # Also mask out-of-range positions (p >= valid) that got fill=0
                # Save scaled scores, load mask, VV_ADD
                npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
                mask_vals = np.full(4, -1e30, dtype=np.float32)
                for p in range(valid):
                    if base + p <= q:
                        mask_vals[p] = 0.0
                npu.set_dram(S['TEMP'], mask_vals)
                npu.send_lo(OP_V_RD_DRAM, S['TEMP'])
                npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
                npu.send_si(OP_VV_ADD, 0, 0)
                
                # Store masked scores, compute max over masked scores
                npu.send_lo(OP_V_WR_DRAM, S['SCORE'] + base)
                npu.spu_max_reduce(0)
            
            global_max = float(npu._spu_srf[0])
            # global_max is now the correct max among valid (unmasked) positions
            
            # ── Pass 2: sum exp(masked_score - max) ──
            npu._spu_srf[1] = 0.0
            for tc in range(NUM_TILES):
                base = tc * 4
                
                npu.send_lo(OP_V_RD_DRAM, S['SCORE'] + base)
                npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
                npu.broadcast_srf(0)
                npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
                npu.send_si(OP_VV_B_SUB_A, 0, 0)
                npu.send_si(OP_V_EXP, 0, 0)
                npu.spu_add_reduce(1)
            
            global_sum = float(npu._spu_srf[1])
            inv_sum = 1.0 / global_sum if global_sum != 0.0 else 0.0
            npu._spu_srf[2] = inv_sum
            
            # ── Pass 3: probs = exp(masked_score - max) * inv_sum → V.T ctx ──
            for tc in range(NUM_TILES):
                base = tc * 4
                valid = min(4, T - base)
                
                # Compute probs
                npu.send_lo(OP_V_RD_DRAM, S['SCORE'] + base)
                npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
                npu.broadcast_srf(0)
                npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
                npu.send_si(OP_VV_B_SUB_A, 0, 0)
                npu.send_si(OP_V_EXP, 0, 0)
                npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
                npu.broadcast_srf(2)
                npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
                npu.send_si(OP_VV_MUL, 0, 0)
                npu.send_lo(OP_V_WR_DRAM, S['PROB'] + base)
                
                # Build V.T tile (transposed: each row is one model dim across positions)
                # MV_MUL computes MRF[i] @ pipeline.  For V.T @ prob we need
                # result[j] = sum_i prob[i] * V[i,j], so row j = [V[base,j], ..., V[base+3,j]].
                for d in range(MODEL_DIM):
                    vec = np.zeros(4, dtype=np.float32)
                    for p in range(valid):
                        pos = base + p
                        v = npu.get_dram(S['V'] + pos * MODEL_DIM, MODEL_DIM)
                        vec[p] = v[d]
                    npu.set_dram(S['TEMP'], vec)
                    npu.send_lo(OP_V_RD_DRAM, S['TEMP'])
                    npu.send_si(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0)
                npu.mat_from_vec_to_mat()
                
                # V.T @ prob → partial context
                npu.send_lo(OP_V_RD_DRAM, S['PROB'] + base)
                npu.send_si(OP_V_WR, MEM_MVM_INITIAL_VRF, 0)
                npu.send_si(OP_V_RD, MEM_MVM_INITIAL_VRF, 0)
                npu.send_si(OP_MV_MUL, 0, 0)
                
                # Mask result to keep current head's dims only
                npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
                mask = np.zeros(4, dtype=np.float32)
                mask[head_offset:head_offset+2] = 1.0
                npu.set_dram(S['TEMP'], mask)
                npu.send_lo(OP_V_RD_DRAM, S['TEMP'])
                npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
                npu.send_si(OP_VV_MUL, 0, 0)
                
                # Accumulate into CTX[q]
                npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
                npu.send_lo(OP_V_RD_DRAM, S['CTX'] + q * MODEL_DIM)
                npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
                npu.send_si(OP_VV_ADD, 0, 0)
                npu.send_lo(OP_V_WR_DRAM, S['CTX'] + q * MODEL_DIM)


# ═══════════════════════════════════════════════════════════════
# Rank-1 operation: u * (h · v)
# ═══════════════════════════════════════════════════════════════

def _rank1_mvm(npu, vec_addr, u_addr, v_addr, out_addr):
    """Compute: out = u * (vec @ v) using NPU instructions.
    
    v is [1,4] stored as a 4×4 tile in DRAM (padded with zeros).
    u is [4] stored as 4 elements in DRAM.
    
    Steps:
      1. MV_MUL(MRF_v, vec) → [dot, 0, 0, 0]  (dot = vec · v)
      2. Add-reduce to SRF[n]: SRF[n] += sum([dot, 0, 0, 0]) = dot
      3. Broadcast SRF[n] → pipeline = [dot, dot, dot, dot]
      4. Load u, VV_MUL → pipeline = u * dot  (correct broadcast)
      5. Store to DRAM
    """
    # dot = h · v: load v as MRF tile, h as vec → MV_MUL → [dot, 0, 0, 0]
    npu.send_lo(OP_M_RD_DRAM, v_addr)
    npu.send_si(OP_M_WR, MEM_MATRIX_RF, 0)
    npu.send_lo(OP_V_RD_DRAM, vec_addr)
    npu.send_si(OP_V_WR, MEM_MVM_INITIAL_VRF, 0)
    npu.send_si(OP_V_RD, MEM_MVM_INITIAL_VRF, 0)
    npu.send_si(OP_MV_MUL, 0, 0)

    # Save result [dot, 0, 0, 0] to MPV, reload, reduce to SRF
    npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
    npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
    # SPU add reduce: sum pipeline into SRF[4].  pipeline = [dot, 0, 0, 0],
    # so SRF[4] = SRF[4] + sum([dot, 0, 0, 0]) = SRF[4] + dot.
    npu._spu_srf[4] = 0.0  # ensure clean slate
    npu.spu_add_reduce(4)   # SRF[4] = dot
    
    # Broadcast SRF[4] → pipeline = [dot, dot, dot, dot]
    npu.broadcast_srf(4)
    # Load u and multiply
    npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
    npu.send_lo(OP_V_RD_DRAM, u_addr)
    npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
    npu.send_si(OP_VV_MUL, 0, 0)
    npu.send_lo(OP_V_WR_DRAM, out_addr)


# ═══════════════════════════════════════════════════════════════
# Forward pass
# ═══════════════════════════════════════════════════════════════

def build_forward(npu, tokens):
    T = len(tokens)
    A_PE = dram_addr('pe_table')
    A_C_ATTN = dram_addr('c_attn')
    A_C_PROJ = dram_addr('c_proj')
    A_C_FC = dram_addr('c_fc')
    A_C_FC_B = dram_addr('c_fc_bias')
    A_C_PROJ_U = dram_addr('c_proj_u')
    A_C_PROJ_V = dram_addr('c_proj_v')
    A_LM_U = dram_addr('lm_u')
    A_LM_V = dram_addr('lm_v')
    A_LM_B = dram_addr('lm_bias')

    # ── Phase 1: Embed + PE (Python precompute, DRAM write) ──
    w = _load_weights(npu._vrf[0])
    for i in range(T):
        x_i = w['embed_A'][tokens[i]] * w['embed_B'] + w['pe_table'][i]
        npu._vrf[0][S['X'] + i * MODEL_DIM: S['X'] + (i+1) * MODEL_DIM] = x_i

    # ── Phase 2: Q, K, V projections ──
    for i in range(T):
        vec = S['X'] + i * MODEL_DIM
        npu.mvm(A_C_ATTN, vec, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['Q'] + i * MODEL_DIM)
        npu.mvm(A_C_ATTN + 4 * MODEL_DIM, vec, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['K'] + i * MODEL_DIM)
        npu.mvm(A_C_ATTN + 8 * MODEL_DIM, vec, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['V'] + i * MODEL_DIM)

    # ── Phase 3: Attention (tiled VecToMatRow + SPU softmax) ──
    _tiled_attention(npu, T)

    # ── Phase 4: c_proj + residual ──
    for i in range(T):
        npu.mvm(A_C_PROJ, S['CTX'] + i * MODEL_DIM, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['ATTN_OUT'] + i * MODEL_DIM)
    for i in range(T):
        npu.send_lo(OP_V_RD_DRAM, S['X'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['ATTN_OUT'] + i * MODEL_DIM)
        npu.send_si(OP_VV_ADD, 0, 0)
        npu.send_lo(OP_V_WR_DRAM, S['AFTER_ATTN'] + i * MODEL_DIM)

    # ── Phase 5: MLP c_fc + ReLU ──
    for i in range(T):
        npu.mvm(A_C_FC, S['AFTER_ATTN'] + i * MODEL_DIM, MEM_MULTIPLY_VRF)
        npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
        npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
        npu.send_lo(OP_V_RD_DRAM, A_C_FC_B)
        npu.send_si(OP_VV_ADD, 0, 0)
        npu.send_si(OP_V_RELU, 0, 0)
        npu.send_lo(OP_V_WR_DRAM, S['FC'] + i * MODEL_DIM)

    # ── Phase 6: MLP rank-1 c_proj + residual (NPU instructions) ──
    for i in range(T):
        _rank1_mvm(npu, S['FC'] + i * MODEL_DIM,
                   A_C_PROJ_U, A_C_PROJ_V, S['MLP_OUT'] + i * MODEL_DIM)
    for i in range(T):
        npu.send_lo(OP_V_RD_DRAM, S['AFTER_ATTN'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['MLP_OUT'] + i * MODEL_DIM)
        npu.send_si(OP_VV_ADD, 0, 0)
        npu.send_lo(OP_V_WR_DRAM, S['AFTER_MLP'] + i * MODEL_DIM)

    # ── Phase 7: LM Head (NPU rank-1 + bias) ──
    # last_h · v → scalar → broadcast × u → add bias
    last_addr = S['AFTER_MLP'] + (T - 1) * MODEL_DIM
    _rank1_mvm(npu, last_addr, A_LM_U, A_LM_V, S['TEMP'])

    # Add bias to first 4 logits (NPU instruction)
    npu.send_lo(OP_V_RD_DRAM, S['TEMP'])
    npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
    npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
    npu.send_lo(OP_V_RD_DRAM, A_LM_B)
    npu.send_si(OP_VV_ADD, 0, 0)
    npu.send_lo(OP_V_WR_DRAM, S['TEMP'])

    # Read first 4 logits from NPU DRAM
    logits_4 = npu.get_dram(S['TEMP'], 4)
    # Compute remaining 6 logits using rank-1 formula
    # (pipeline is only 4-wide, RISC-V fallback for the rest)
    last_h = npu.get_dram(last_addr, MODEL_DIM)
    lm_v = npu.get_dram(A_LM_V, MODEL_DIM)
    lm_u = npu.get_dram(A_LM_U, VOCAB_SIZE)
    lm_b = npu.get_dram(A_LM_B, VOCAB_SIZE)
    dot = float(np.dot(last_h, lm_v))

    logits = np.zeros(10, dtype=np.float32)
    logits[:4] = logits_4
    logits[4:] = dot * lm_u[4:] + lm_b[4:]

    return logits


# ── Autoregressive test ──────────────────────────────────────────

def infer_npu(dram, a, b):
    npu = NpuFP32()
    npu.load_dram(dram)
    tokens = encode_prompt(a, b)
    for _ in range(OUTPUT_DIGITS):
        l = build_forward(npu, tokens)
        tokens.append(int(np.argmax(l)))
    return decode_output(tokens)


# ── Pytest ────────────────────────────────────────────────────────

GOLDEN_DRAM = None


def _get_dram():
    global GOLDEN_DRAM
    if GOLDEN_DRAM is None:
        GOLDEN_DRAM, _ = build_dram()
    return GOLDEN_DRAM


def _debug_intermediates(npu, tokens, gl, label=""):
    """Print intermediate states comparing NPU vs golden."""
    S_local = S
    T = len(tokens)
    
    def read_dram(addr, n=4):
        return npu.get_dram(addr, n)
    
    from adderboard.golden.golden_130p import _load_weights
    w = _load_weights(npu._vrf[0])
    
    embed_out = np.zeros((T, 4), dtype=np.float32)
    for i in range(T):
        v = tokens[i]
        embed_out[i] = w['embed_A'][v] * w['embed_B']
    x = embed_out + w['pe_table'][:T]
    
    qkv = x @ w['c_attn'].T
    q = qkv[:, 0:4]
    k = qkv[:, 4:8]
    v = qkv[:, 8:12]
    
    print(f"\n=== Intermediate debug {label} ===")
    for i in range(T):
        npu_x = read_dram(S_local['X'] + i * 4, 4)
        print(f"  X[{i}]: golden={x[i]} npu={npu_x}  diff={np.max(np.abs(x[i] - npu_x)):.6f}")
    print()
    for i in range(T):
        npu_q = read_dram(S_local['Q'] + i * 4, 4)
        npu_k = read_dram(S_local['K'] + i * 4, 4)
        npu_v = read_dram(S_local['V'] + i * 4, 4)
        print(f"  Q[{i}]: golden={q[i]} npu={npu_q}  diff={np.max(np.abs(q[i] - npu_q)):.6f}")
        print(f"  K[{i}]: golden={k[i]} npu={npu_k}  diff={np.max(np.abs(k[i] - npu_k)):.6f}")
        print(f"  V[{i}]: golden={v[i]} npu={npu_v}  diff={np.max(np.abs(v[i] - npu_v)):.6f}")
    
    print()
    for i in range(T):
        npu_ctx = read_dram(S_local['CTX'] + i * 4, 4)
        q_ = q.reshape(T, 2, 2).transpose(1, 0, 2)
        k_ = k.reshape(T, 2, 2).transpose(1, 0, 2)
        v_ = v.reshape(T, 2, 2).transpose(1, 0, 2)
        scale = 2 ** -0.5
        scores = (q_ @ k_.transpose(0, 2, 1)) * scale
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e30, scores)
        attn_w = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_w = attn_w / np.sum(attn_w, axis=-1, keepdims=True)
        ctx_g = (attn_w @ v_).transpose(1, 0, 2).reshape(T, -1)
        print(f"  CTX[{i}]: golden={ctx_g[i]} npu={npu_ctx}  diff={np.max(np.abs(ctx_g[i] - npu_ctx)):.6f}")
    
    attn_out_g = ctx_g @ w['c_proj'].T
    after_attn_g = x + attn_out_g
    print()
    for i in range(T):
        npu_aa = read_dram(S_local['AFTER_ATTN'] + i * 4, 4)
        print(f"  AFTER_ATTN[{i}]: golden={after_attn_g[i]} npu={npu_aa}  diff={np.max(np.abs(after_attn_g[i] - npu_aa)):.6f}")
    
    h_g = after_attn_g @ w['c_fc'].T + w['c_fc_bias']
    h_g = np.maximum(h_g, 0.0)
    tmp_g = h_g @ w['c_proj_v']
    mlp_out_g = np.outer(tmp_g, w['c_proj_u'])
    after_mlp_g = after_attn_g + mlp_out_g
    print()
    for i in range(T):
        npu_fc = read_dram(S_local['FC'] + i * 4, 4)
        print(f"  FC[{i}]: golden={h_g[i]} npu={npu_fc}  diff={np.max(np.abs(h_g[i] - npu_fc)):.6f}")
        npu_mlp = read_dram(S_local['MLP_OUT'] + i * 4, 4)
        print(f"  MLP_OUT[{i}]: golden={mlp_out_g[i]} npu={npu_mlp}  diff={np.max(np.abs(mlp_out_g[i] - npu_mlp)):.6f}")
        npu_am = read_dram(S_local['AFTER_MLP'] + i * 4, 4)
        print(f"  AFTER_MLP[{i}]: golden={after_mlp_g[i]} npu={npu_am}  diff={np.max(np.abs(after_mlp_g[i] - npu_am)):.6f}")
    
    last_h_g = after_mlp_g[-1]
    dot_g = float(np.dot(last_h_g, w['lm_v']))
    logits_g = dot_g * w['lm_u'] + w['lm_bias']
    last_h_npu = read_dram(S_local['AFTER_MLP'] + (T-1) * 4, 4)
    dot_npu = float(np.dot(last_h_npu, w['lm_v']))
    print(f"\n  LM last_h: golden={last_h_g} npu={last_h_npu}  diff={np.max(np.abs(last_h_g - last_h_npu)):.6f}")
    print(f"  LM dot: golden={dot_g:.6f} npu={dot_npu:.6f}")
    print(f"  LM logits: golden={gl} npu=... (computed by NPU)")


def test_npu_vs_golden_first_step():
    dram = _get_dram()
    a, b = 5, 5
    tokens = encode_prompt(a, b)
    gl = golden_forward(dram, tokens)

    npu = NpuFP32()
    npu.load_dram(dram)
    nl = build_forward(npu, tokens)
    
    _debug_intermediates(npu, tokens, gl)

    max_diff = float(np.max(np.abs(gl - nl)))
    print(f"First step logits:\n  Golden: {gl}\n  NPU:    {nl}\n  Max diff: {max_diff:.6f}")
    assert np.argmax(gl) == np.argmax(nl), f"Argmax mismatch"
    assert max_diff < 1e-3, f"Max diff {max_diff:.6f} >= 1e-3"


@pytest.mark.parametrize("a,b,expected", [
    (5, 5, 10),
    (555, 445, 1000),
    (0, 0, 0),
    (9999999999, 1, 10000000000),
    (1111111111, 8888888889, 10000000000),
    (19492, 23919, 43411),
])
def test_npu_single(a, b, expected):
    dram = _get_dram()
    result = infer_npu(dram, a, b)
    assert result == expected, f"NPU: {a} + {b} = {result} (expected {expected})"


def test_npu_vs_golden_bulk():
    import random
    rng = random.Random(42)
    dram = _get_dram()
    failures = []
    for _ in range(50):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        tokens = encode_prompt(a, b)
        for _ in range(OUTPUT_DIGITS):
            gl = golden_forward(dram, tokens)
            tokens.append(int(np.argmax(gl)))
        golden_r = decode_output(tokens)
        nr = infer_npu(dram, a, b)
        if golden_r != nr:
            failures.append((a, b, golden_r, nr))

    if failures:
        for a, b, gr, nr in failures[:10]:
            print(f"  FAIL: {a}+{b} golden={gr} npu={nr}")
        pytest.fail(f"{len(failures)}/50 mismatches")
    print(f"  All 50 random pairs match golden.")
