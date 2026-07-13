"""
Phase 1 (140p): Synthetic NPU instruction stream for trained Qwen3 model.

All operations use real NPU instructions via _execute() in FP32 mode.
Python helpers are limited to:
  - Embedding + RMSNorm (precomputed, DRAM write) → RISC-V fallback
  - RoPE apply (cos/sin from DRAM table, scalar compute) → RISC-V fallback
  - SiLU (V_SIGM + VV_MUL) → NPU
  - LM head (read last_h from DRAM, scalar compute) → RISC-V fallback

Self-verifying against golden_140p.forward().
"""

import numpy as np
import pytest

from emulator.npu_fp32 import NpuFP32, si, lo
from adderboard.layout.layout_140p import (
    build_dram, encode_prompt, decode_output,
    dram_addr, MODEL_DIM, N_HEADS, HEAD_DIM, FF_DIM,
    PROMPT_LEN, OUTPUT_DIGITS, TOTAL_LEN, VOCAB_SIZE,
)
from adderboard.golden.golden_140p import forward as golden_forward, _load_weights, \
    rms_norm, silu, apply_rope_numpy, softmax

from emulator.npu_device_mini import (
    MEM_MULTIPLY_VRF, MEM_MVM_INITIAL_VRF, MEM_MATRIX_RF,
    MEM_VEC_TO_MAT_ROW, MEM_FILL, MEM_SPU_MAX_REDUCE, MEM_SPU_ADD_REDUCE,
    MEM_SPU_BROADCAST,
    OP_V_RD_DRAM, OP_V_WR_DRAM,
    OP_V_RD, OP_V_WR,
    OP_M_RD_DRAM, OP_M_WR, OP_M_RD, OP_MV_MUL,
    OP_VV_ADD, OP_VV_B_SUB_A, OP_VV_MUL,
    OP_V_SIGM, OP_V_EXP,
)


# ── Scratch layout ───────────────────────────────────────────────

def _scratch(total=TOTAL_LEN, dim=MODEL_DIM):
    KEY = ['X', 'H1', 'Q', 'K', 'V', 'CTX', 'ATTN_OUT', 'ATTN_RESIDUAL',
           'H2', 'GATE', 'UP', 'FFN_RESIDUAL', 'LAST_H',
           'SCORE', 'PROB', 'TEMP']
    if not hasattr(_scratch, 'cache'):
        addr = 0x2000
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
# Tiled single-head attention
# ═══════════════════════════════════════════════════════════════

def _tiled_attention_1head(npu, T):
    """Single-head scaled dot-product attention, tiled with VecToMatRow."""
    SCALE = 4.0 ** -0.5
    NUM_TILES = (T + 3) // 4
    npu._spu_srf[6] = SCALE

    for i in range(T):
        npu.send_si(OP_V_RD, MEM_FILL, 0)
        npu.send_lo(OP_V_WR_DRAM, S['CTX'] + i * MODEL_DIM)

    for q in range(T):
        npu._spu_srf[0] = -1e30
        for tc in range(NUM_TILES):
            base = tc * 4
            valid = min(4, T - base)
            for p in range(4):
                pos = base + p
                if p < valid:
                    npu.send_lo(OP_V_RD_DRAM, S['K'] + pos * MODEL_DIM)
                else:
                    npu.send_si(OP_V_RD, MEM_FILL, 0)
                npu.send_si(OP_V_WR, MEM_VEC_TO_MAT_ROW, 0)
            npu.mat_from_vec_to_mat()
            npu.send_lo(OP_V_RD_DRAM, S['Q'] + q * MODEL_DIM)
            npu.send_si(OP_V_WR, MEM_MVM_INITIAL_VRF, 0)
            npu.send_si(OP_V_RD, MEM_MVM_INITIAL_VRF, 0)
            npu.send_si(OP_MV_MUL, 0, 0)
            npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
            npu.broadcast_srf(6)
            npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
            npu.send_si(OP_VV_MUL, 0, 0)
            npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
            mask_vals = np.full(4, -1e30, dtype=np.float32)
            for p in range(valid):
                if base + p <= q:
                    mask_vals[p] = 0.0
            npu.set_dram(S['TEMP'], mask_vals)
            npu.send_lo(OP_V_RD_DRAM, S['TEMP'])
            npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
            npu.send_si(OP_VV_ADD, 0, 0)
            npu.send_lo(OP_V_WR_DRAM, S['SCORE'] + base)
            npu.spu_max_reduce(0)

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

        for tc in range(NUM_TILES):
            base = tc * 4
            valid = min(4, T - base)
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
            npu.send_lo(OP_V_RD_DRAM, S['PROB'] + base)
            npu.send_si(OP_V_WR, MEM_MVM_INITIAL_VRF, 0)
            npu.send_si(OP_V_RD, MEM_MVM_INITIAL_VRF, 0)
            npu.send_si(OP_MV_MUL, 0, 0)
            npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
            npu.send_lo(OP_V_RD_DRAM, S['CTX'] + q * MODEL_DIM)
            npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
            npu.send_si(OP_VV_ADD, 0, 0)
            npu.send_lo(OP_V_WR_DRAM, S['CTX'] + q * MODEL_DIM)


# ═══════════════════════════════════════════════════════════════
# Python helpers (RISC-V fallback equivalents)
# ═══════════════════════════════════════════════════════════════

def _compute_rmsnorm(npu, vec_addr, gamma_addr, out_addr):
    x = npu.get_dram(vec_addr, MODEL_DIM)
    gamma = npu.get_dram(gamma_addr, MODEL_DIM)
    npu.set_dram(out_addr, rms_norm(x, gamma))


def _apply_rope_npu(npu, vec_addr, pos, out_addr):
    ROPE = dram_addr('rope_table')
    cos_sin = npu.get_dram(ROPE + pos * 4, 4)
    x = npu.get_dram(vec_addr, 4)
    rot = np.zeros(4, dtype=np.float32)
    rot[0] = x[0] * cos_sin[0] - x[1] * cos_sin[1]
    rot[1] = x[0] * cos_sin[1] + x[1] * cos_sin[0]
    rot[2] = x[2] * cos_sin[2] - x[3] * cos_sin[3]
    rot[3] = x[2] * cos_sin[3] + x[3] * cos_sin[2]
    npu.set_dram(out_addr, rot)


def _silu_via_sigm(npu, src_addr, out_addr):
    """SiLU via V_SIGM + VV_MUL. Pipeline ops only, no RISC-V fallback.

    Steps:
      1. Load x, save to SRF[4] (we use SPU broadcast for save)
      2. V_SIGM on x → sigmoid(x)
      3. Load original x from DRAM again, VV_MUL with sigmoid
    """
    # Load x
    npu.send_lo(OP_V_RD_DRAM, src_addr)
    # Compute sigmoid
    npu.send_si(OP_V_SIGM, 0, 0)
    # Save sigmoid to MPV
    npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
    # Reload x
    npu.send_lo(OP_V_RD_DRAM, src_addr)
    # VV_MUL: pipe_a = sigmoid (from MPV), pipe = x
    npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
    npu.send_si(OP_VV_MUL, 0, 0)
    npu.send_lo(OP_V_WR_DRAM, out_addr)


# ═══════════════════════════════════════════════════════════════
# Forward pass with NPU instruction stream
# ═══════════════════════════════════════════════════════════════

def build_forward(npu, tokens):
    """One forward step through NPU. Returns logits for last position."""
    T = len(tokens)

    A_EMBED     = dram_addr('embedding')
    A_NORM1     = dram_addr('norm1')
    A_NORM2     = dram_addr('norm2')
    A_NORM_FINAL = dram_addr('norm_final')
    A_W_Q       = dram_addr('W_q')
    A_W_Q_T     = dram_addr('W_q_t')
    A_W_KV      = dram_addr('W_kv')
    A_Q_NORM    = dram_addr('q_norm')
    A_K_NORM    = dram_addr('k_norm')
    A_W_GATE    = dram_addr('W_gate')
    A_W_UP      = dram_addr('W_up')
    A_W_DOWN    = dram_addr('W_down')

    # ── Phase 1: Embedding (precompute, DRAM write) ──
    w = _load_weights(npu._vrf[0])
    for i in range(T):
        x_i = w['embedding'][tokens[i]]
        npu._vrf[0][S['X'] + i * MODEL_DIM:S['X'] + (i+1) * MODEL_DIM] = x_i

    # ── Phase 2: RMSNorm → Q/KV → QK Norms → RoPE ──
    for i in range(T):
        _compute_rmsnorm(npu, S['X'] + i * MODEL_DIM, A_NORM1,
                         S['H1'] + i * MODEL_DIM)
        npu.mvm(A_W_Q, S['H1'] + i * MODEL_DIM, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['Q'] + i * MODEL_DIM)
        npu.mvm(A_W_KV, S['H1'] + i * MODEL_DIM, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['K'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['K'] + i * MODEL_DIM)
        npu.send_lo(OP_V_WR_DRAM, S['V'] + i * MODEL_DIM)

    for i in range(T):
        _compute_rmsnorm(npu, S['Q'] + i * MODEL_DIM, A_Q_NORM,
                         S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_WR_DRAM, S['Q'] + i * MODEL_DIM)
        _compute_rmsnorm(npu, S['K'] + i * MODEL_DIM, A_K_NORM,
                         S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_WR_DRAM, S['K'] + i * MODEL_DIM)

    for i in range(T):
        _apply_rope_npu(npu, S['Q'] + i * MODEL_DIM, i,
                        S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_WR_DRAM, S['Q'] + i * MODEL_DIM)
        _apply_rope_npu(npu, S['K'] + i * MODEL_DIM, i,
                        S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_WR_DRAM, S['K'] + i * MODEL_DIM)

    # ── Phase 3: Attention (tiled, single-head) ──
    _tiled_attention_1head(npu, T)

    # ── Phase 4: O = Q^T (tied) + residual ──
    for i in range(T):
        # O projection: ctx @ W_q = W_q^T @ ctx.
        # NPU MV_MUL does MRF @ vec. Set MRF = W_q^T so MRF @ ctx = W_q^T @ ctx = ctx @ W_q.
        npu.mvm(A_W_Q_T, S['CTX'] + i * MODEL_DIM, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['ATTN_OUT'] + i * MODEL_DIM)
    for i in range(T):
        npu.send_lo(OP_V_RD_DRAM, S['X'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['ATTN_OUT'] + i * MODEL_DIM)
        npu.send_si(OP_VV_ADD, 0, 0)
        npu.send_lo(OP_V_WR_DRAM, S['ATTN_RESIDUAL'] + i * MODEL_DIM)

    # ── Phase 5: FFN (SwiGLU) ──
    for i in range(T):
        _compute_rmsnorm(npu, S['ATTN_RESIDUAL'] + i * MODEL_DIM, A_NORM2,
                         S['H2'] + i * MODEL_DIM)
        npu.mvm(A_W_GATE, S['H2'] + i * MODEL_DIM, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['GATE'] + i * MODEL_DIM)
        npu.mvm(A_W_UP, S['H2'] + i * MODEL_DIM, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['UP'] + i * MODEL_DIM)
        _silu_via_sigm(npu, S['GATE'] + i * MODEL_DIM,
                       S['TEMP'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['TEMP'] + i * MODEL_DIM)
        npu.send_si(OP_V_WR, MEM_MULTIPLY_VRF, 0)
        npu.send_lo(OP_V_RD_DRAM, S['UP'] + i * MODEL_DIM)
        npu.send_si(OP_V_RD, MEM_MULTIPLY_VRF, 0)
        npu.send_si(OP_VV_MUL, 0, 0)
        npu.send_lo(OP_V_WR_DRAM, S['TEMP'] + i * MODEL_DIM)
        npu.mvm(A_W_DOWN, S['TEMP'] + i * MODEL_DIM, MEM_MULTIPLY_VRF)
        npu.send_lo(OP_V_WR_DRAM, S['FFN_RESIDUAL'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['ATTN_RESIDUAL'] + i * MODEL_DIM)
        npu.send_lo(OP_V_RD_DRAM, S['FFN_RESIDUAL'] + i * MODEL_DIM)
        npu.send_si(OP_VV_ADD, 0, 0)
        npu.send_lo(OP_V_WR_DRAM, S['LAST_H'] + i * MODEL_DIM)

    # ── Phase 6: LM Head (RISC-V fallback) ──
    last_addr = S['LAST_H'] + (T - 1) * MODEL_DIM
    _compute_rmsnorm(npu, last_addr, A_NORM_FINAL, S['TEMP'])
    last_h = npu.get_dram(S['TEMP'], MODEL_DIM)
    embedding = npu.get_dram(A_EMBED, VOCAB_SIZE * MODEL_DIM).reshape(VOCAB_SIZE, MODEL_DIM)
    logits = last_h @ embedding.T
    return logits


# ── Autoregressive inference ────────────────────────────────────

def infer_npu(dram, a, b):
    npu = NpuFP32()
    npu.load_dram(dram)
    tokens = encode_prompt(a, b)
    for _ in range(OUTPUT_DIGITS):
        l = build_forward(npu, tokens)
        tokens.append(int(np.argmax(l)))
    return decode_output(tokens)


# ── Pytest ──────────────────────────────────────────────────────

GOLDEN_DRAM_140P = None

def _get_dram():
    global GOLDEN_DRAM_140P
    if GOLDEN_DRAM_140P is None:
        GOLDEN_DRAM_140P, _ = build_dram()
    return GOLDEN_DRAM_140P


@pytest.mark.parametrize("a,b,expected", [
    (5, 5, 10),
    (555, 445, 1000),
    (0, 0, 0),
    (1111111111, 8888888889, 10000000000),
    (19492, 23919, 43411),
])
def test_npu_single(a, b, expected):
    dram = _get_dram()
    result = infer_npu(dram, a, b)
    assert result == expected, f"NPU: {a}+{b}={result} (expected {expected})"


def test_npu_vs_golden_first_step():
    dram = _get_dram()
    a, b = 5, 5
    tokens = encode_prompt(a, b)
    gl = golden_forward(dram, tokens)
    npu = NpuFP32()
    npu.load_dram(dram)
    nl = build_forward(npu, tokens)
    max_diff = float(np.max(np.abs(gl - nl)))
    print(f"First step logits (5+5):\n  Golden: {gl}\n  NPU:    {nl}\n  Max diff: {max_diff:.6f}")
    assert np.argmax(gl) == np.argmax(nl), \
        f"Argmax mismatch: golden={np.argmax(gl)} npu={np.argmax(nl)}"
    assert max_diff < 1e-3, f"Max diff {max_diff:.6f} >= 1e-3"


def test_npu_vs_golden_bulk():
    import random
    rng = random.Random(42)
    dram = _get_dram()
    failures = []
    for _ in range(50):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        tokens_g = encode_prompt(a, b)[:]
        for _ in range(OUTPUT_DIGITS):
            gl = golden_forward(dram, tokens_g)
            tokens_g.append(int(np.argmax(gl)))
        golden_r = decode_output(tokens_g)
        nr = infer_npu(dram, a, b)
        if golden_r != nr:
            failures.append((a, b, golden_r, nr))
    if failures:
        for a, b, gr, nr in failures[:10]:
            print(f"  FAIL: {a}+{b} golden={gr} npu={nr}")
        pytest.fail(f"{len(failures)}/50 mismatches between NPU and golden")
    print(f"  All 50 random pairs match golden.")


def test_npu_vs_golden_intermediates():
    dram = _get_dram()
    a, b = 5, 5
    tokens = encode_prompt(a, b)
    T = len(tokens)
    gl = golden_forward(dram, tokens)

    npu = NpuFP32()
    npu.load_dram(dram)
    nl = build_forward(npu, tokens)
    w = _load_weights(npu._vrf[0])

    def _d(addr):
        return npu.get_dram(addr, 4)

    print(f"\n=== First-step intermediates (5+5, T={T}) ===")
    x_g = np.array([w['embedding'][t] for t in tokens])
    h1_g = rms_norm(x_g, w['norm1'])
    q_g = h1_g @ w['W_q'].T
    kv_g = h1_g @ w['W_kv'].T
    k_g = v_g = kv_g
    q_g = rms_norm(q_g, w['q_norm'])
    k_g = rms_norm(k_g, w['k_norm'])
    q_g = apply_rope_numpy(q_g.reshape(T, 1, 4), w['rope_table'], np.arange(T)).reshape(T, 4)
    k_g = apply_rope_numpy(k_g.reshape(T, 1, 4), w['rope_table'], np.arange(T)).reshape(T, 4)

    for i in range(min(3, T)):
        print(f"  Q[{i}]: g={q_g[i]} npu={_d(S['Q']+i*4)}  diff={max(abs(q_g[i]-_d(S['Q']+i*4))):.6f}")

    q_ = q_g.reshape(1, T, 4)
    k_ = k_g.reshape(1, T, 4)
    v_ = v_g.reshape(1, T, 4)
    scale = 4.0**-0.5
    scores = q_ @ k_.transpose(0, 2, 1) * scale
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -1e30, scores)
    attn_w = softmax(scores, axis=-1)
    ctx_g = (attn_w @ v_).reshape(T, 4)
    for i in range(min(3, T)):
        print(f"  CTX[{i}]: g={ctx_g[i]} npu={_d(S['CTX']+i*4)}  diff={max(abs(ctx_g[i]-_d(S['CTX']+i*4))):.6f}")

    print(f"\n  LOGITS: g={gl} npu={nl}  diff={max(abs(gl-nl)):.6f}")
    print(f"  Argmax: g={np.argmax(gl)} npu={np.argmax(nl)}")
    assert np.max(np.abs(gl - nl)) < 1e-3, "Max diff exceeded"
