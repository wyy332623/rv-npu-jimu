"""
Phase 2 (140p): RISC-V ISS + firmware for trained Qwen3 model.

Runs adder_140p firmware under MiniRV64 ISS with NpuDeviceMini FP32 emulator.
ISS pre-fills:
  - Embedding at S_X
  - RMSNorm(embed) at S_H1
  - Q, K at S_Q, S_K (after norm+RoPE applied by ISS)
  - V = K at S_V
  - Causal mask table + transposed V tiles
  - Scale (0.5) in SRF[6]
  - After attention: norm2(attn_res) at S_H2, gate=W_gate@h2 at S_GATE,
    up=W_up@h2 at S_UP

Firmware does: tiled attention, O projection, residual, SiLU, W_down, residual.
ISS reads last_h from FW_LAST_H, computes norm_final + LM head.

Self-verifying against golden_140p.forward().
"""

import numpy as np
import subprocess
import pytest
from pathlib import Path

from emulator.npu_device_mini import (
    NpuDeviceMini, MEM_DRAM,
    MEM_FILL, MEM_MULTIPLY_VRF, MEM_VEC_TO_MAT_ROW,
    MEM_MVM_INITIAL_VRF, MEM_MATRIX_RF,
    MEM_SPU_MAX_REDUCE, MEM_SPU_ADD_REDUCE, MEM_SPU_BROADCAST,
)

from adderboard.layout.layout_140p import (
    build_dram, encode_prompt, decode_output,
    dram_addr, MODEL_DIM, HEAD_DIM, FF_DIM,
    PROMPT_LEN, OUTPUT_DIGITS, TOTAL_LEN, VOCAB_SIZE, MAX_ROPE_POS,
)
from adderboard.golden.golden_140p import (
    forward as golden_forward, _load_weights, rms_norm, silu,
    apply_rope_numpy, softmax,
)

try:
    from iss.mini_rv64 import MiniRV64
    HAS_ISS = True
except ImportError:
    HAS_ISS = False
    MiniRV64 = None

FW_DIR = Path(__file__).resolve().parent.parent.parent / "firmware"


def _build_firmware(dim, seq_len=24):
    build_dir = f'build_dim{dim}'
    abs_build_dir = FW_DIR / build_dir
    env = {'NATIVE_DIM': str(dim), 'SEQ_LEN': str(seq_len)}
    import os
    full_env = {**os.environ, **env}
    r = subprocess.run(
        ['make', '-C', str(FW_DIR), f'BUILD_DIR={build_dir}',
         'TARGET=adder_140p', 'clean', 'all'],
        capture_output=True, text=True, env=full_env)
    elf = abs_build_dir / 'adder_140p.elf'
    return r, elf


# Scratch addresses (must match firmware)
MAX_SEQ = 35
SCR_BASE = 0x2000
S_X = SCR_BASE
S_Q = S_X + MAX_SEQ * MODEL_DIM          # 0x208C
S_K = S_Q + MAX_SEQ * MODEL_DIM          # 0x2118
S_V = S_K + MAX_SEQ * MODEL_DIM          # 0x21A4
S_CTX = S_V + MAX_SEQ * MODEL_DIM        # 0x2230
S_ATTN_OUT = S_CTX + MAX_SEQ * MODEL_DIM
S_ATTN_RES = S_ATTN_OUT + MAX_SEQ * MODEL_DIM
S_SCORE = S_ATTN_RES + MAX_SEQ * MODEL_DIM
S_PROB = S_SCORE + MAX_SEQ
S_TEMP = S_PROB + MAX_SEQ
S_MASK_TABLE = S_TEMP + MAX_SEQ
S_FLAGS = 0x1F00  # DRAM phase flag: 0=phase1, 1=phase2

# Phase 2 scratch (separate region, filled by ISS between phases)
S_BASE2 = 0x3000
S_H2 = S_BASE2
S_GATE = S_H2 + MAX_SEQ * MODEL_DIM
S_UP = S_GATE + MAX_SEQ * MODEL_DIM
S_FFN_RES = S_UP + MAX_SEQ * MODEL_DIM
S_LAST_H = S_FFN_RES + MAX_SEQ * MODEL_DIM
S_TEMP2 = S_LAST_H + MAX_SEQ * MODEL_DIM

FW_LAST_H = 0x4000


def make_npu(dram_arr):
    npu = NpuDeviceMini(native_dim=4)
    npu._pipeline_round_fp16 = lambda: None
    npu._store_to_ivrf = lambda: None

    def _v_rd_dram(self, opcode, opd0, opd1, full_operand=0):
        addr = full_operand
        dl = self._vrf.get(MEM_DRAM)
        if dl is not None:
            if self._pipeline is not None:
                self._vpipe_a = self._pipeline.copy()
            n = min(self.native_dim, len(dl) - addr) if addr < len(dl) else 0
            self._pipeline = np.zeros(self.native_dim, dtype=np.float32)
            if n > 0:
                self._pipeline[:n] = dl[addr:addr + n]
    npu._v_rd_dram = lambda *a, **kw: _v_rd_dram(npu, *a, **kw)

    def _v_wr_dram(self, opcode, opd0, opd1, full_operand=0):
        addr = full_operand
        dl = self._vrf.get(MEM_DRAM)
        if dl is not None and self._pipeline is not None:
            n = min(self.native_dim, len(self._pipeline),
                    len(dl) - addr) if addr < len(dl) else 0
            if n > 0:
                dl[addr:addr + n] = self._pipeline[:n]
    npu._v_wr_dram = lambda *a, **kw: _v_wr_dram(npu, *a, **kw)

    orig_m_rd = npu._m_rd

    def _m_rd(self, opcode, opd0, opd1, full_operand=0):
        if opd0 == MEM_VEC_TO_MAT_ROW:
            return orig_m_rd(opcode, opd0, opd1, full_operand)
        addr = full_operand
        dl = self._vrf.get(MEM_DRAM)
        if dl is not None and addr < len(dl):
            n = self._regs.get(1, 1) * self.native_dim
            mrf_size = n * n
            if addr + mrf_size <= len(dl):
                mat = dl[addr:addr + mrf_size].reshape(n, n).copy()
                self._mrf[MEM_MATRIX_RF] = mat
    npu._m_rd = lambda *a, **kw: _m_rd(npu, *a, **kw)

    def _v_rd(self, opcode, opd0, opd1):
        mt, ad = opd0, opd1
        if mt == MEM_FILL:
            if self._pipeline is not None:
                self._vpipe_a = self._pipeline.copy()
            val = np.frombuffer(np.uint16([ad]).tobytes(),
                                dtype=np.float16)[0]
            self._pipeline = np.full(self.native_dim, float(val),
                                      dtype=np.float32)
            return
        if mt == MEM_SPU_BROADCAST:
            if self._pipeline is not None:
                self._vpipe_a = self._pipeline.copy()
            val = self._spu_srf[ad] if ad < len(self._spu_srf) else 0.0
            self._pipeline = np.full(self.native_dim, val, dtype=np.float32)
            return
        vrf = self._vrf.get(mt)
        if vrf is not None:
            if self._pipeline is not None:
                self._vpipe_a = self._pipeline.copy()
            n = min(self.native_dim, max(0, len(vrf) - ad))
            self._pipeline = np.zeros(self.native_dim, dtype=np.float32)
            if n > 0:
                self._pipeline[:n] = vrf[ad:ad + n]
    npu._v_rd = lambda *a, **kw: _v_rd(npu, *a, **kw)

    def _v_wr(self, opcode, opd0, opd1):
        mt, ad = opd0, opd1
        if self._pipeline is None:
            return
        if mt == MEM_VEC_TO_MAT_ROW:
            key = 0
            if key not in self._row_buffer:
                self._row_buffer[key] = []
            self._row_buffer[key].append(self._pipeline.copy())
            return
        if mt == MEM_SPU_ADD_REDUCE:
            if ad < len(self._spu_srf):
                self._spu_srf[ad] = (float(np.sum(self._pipeline))
                                      + self._spu_srf[ad])
            return
        if mt == MEM_SPU_MAX_REDUCE:
            if ad < len(self._spu_srf):
                self._spu_srf[ad] = max(float(np.max(self._pipeline)),
                                         self._spu_srf[ad])
            return
        vrf = self._vrf.setdefault(
            mt, np.zeros(self.native_dim * 8, dtype=np.float32))
        n = min(self.native_dim, max(0, len(vrf) - ad))
        if n > 0:
            vrf[ad:ad + n] = self._pipeline[:n]
    npu._v_wr = lambda *a, **kw: _v_wr(npu, *a, **kw)

    npu._vrf[MEM_DRAM][:len(dram_arr)] = dram_arr.copy()
    return npu


def run_forward_iss(dram, tokens):
    T = len(tokens)

    r, elf_path = _build_firmware(4, seq_len=T)
    if r.returncode != 0 or not elf_path.exists():
        raise RuntimeError(f"Firmware build failed:\n{r.stderr[:500]}")

    npu = make_npu(dram)
    w = _load_weights(npu._vrf[MEM_DRAM])

    # ── ISS pre-fill: Embedding + QKV + RoPE ──
    x = np.zeros((T, MODEL_DIM), dtype=np.float32)
    for i in range(T):
        x[i] = w['embedding'][tokens[i]]

    # norm1
    h1 = rms_norm(x, w['norm1'])
    for i in range(T):
        npu._vrf[MEM_DRAM][S_X + i * MODEL_DIM:S_X + (i+1) * MODEL_DIM] = x[i]

    # Q = h1 @ W_q^T, K,V = h1 @ W_kv^T
    q = h1 @ w['W_q'].T
    kv = h1 @ w['W_kv'].T
    k = kv.copy()
    v = kv.copy()

    # QK norms + RoPE
    q = rms_norm(q, w['q_norm'])
    k = rms_norm(k, w['k_norm'])
    positions = np.arange(T, dtype=np.int32)
    q = apply_rope_numpy(q.reshape(T, 1, HEAD_DIM), w['rope_table'], positions).reshape(T, HEAD_DIM)
    k = apply_rope_numpy(k.reshape(T, 1, HEAD_DIM), w['rope_table'], positions).reshape(T, HEAD_DIM)

    for i in range(T):
        npu._vrf[MEM_DRAM][S_Q + i * MODEL_DIM:S_Q + (i+1) * MODEL_DIM] = q[i]
        npu._vrf[MEM_DRAM][S_K + i * MODEL_DIM:S_K + (i+1) * MODEL_DIM] = k[i]
        npu._vrf[MEM_DRAM][S_V + i * MODEL_DIM:S_V + (i+1) * MODEL_DIM] = v[i]

    # ── Causal mask table ──
    nt = (T + 3) // 4
    mt = S_MASK_TABLE
    for qq in range(T):
        for tc in range(nt):
            base = tc * 4
            valid = min(4, T - base)
            mask = np.full(4, -1e30, dtype=np.float32)
            for p in range(valid):
                if base + p <= qq:
                    mask[p] = 0.0
            addr = mt + (qq * nt + tc) * MODEL_DIM
            npu._vrf[MEM_DRAM][addr:addr + MODEL_DIM] = mask

    # ── Transposed V tiles ──
    svt = mt + MAX_SEQ * nt * MODEL_DIM
    for tc in range(nt):
        base = tc * 4
        valid = min(4, T - base)
        vt = np.zeros((MODEL_DIM, MODEL_DIM), dtype=np.float32)
        for p in range(valid):
            pos = base + p
            for d in range(MODEL_DIM):
                vt[d, p] = v[pos, d]
        tile_addr = svt + tc * MODEL_DIM * MODEL_DIM
        for d in range(MODEL_DIM):
            npu._vrf[MEM_DRAM][tile_addr + d * MODEL_DIM:
                                tile_addr + (d+1) * MODEL_DIM] = vt[d]

    # ── Phase 1: Run firmware (phase flag = 0) ──
    npu._vrf[MEM_DRAM][S_FLAGS] = np.frombuffer(np.uint32(0).tobytes(), dtype=np.float32)[0]
    npu._spu_srf[6] = 4.0 ** -0.5
    npu.set_seq_len(T)
    cpu = MiniRV64()
    cpu.set_mmio_device(npu)
    cpu.load_elf(str(elf_path))
    cpu.run(cycles=300000)

    # ── ISS gap: read attn_res, compute norm2/gate/up, write to DRAM ──
    attn_res = np.zeros((T, MODEL_DIM), dtype=np.float32)
    for i in range(T):
        attn_res[i] = npu._vrf[MEM_DRAM][S_ATTN_RES + i * MODEL_DIM:
                                          S_ATTN_RES + (i+1) * MODEL_DIM]

    h2 = rms_norm(attn_res, w['norm2'])
    gate = h2 @ w['W_gate'].T
    up = h2 @ w['W_up'].T

    for i in range(T):
        npu._vrf[MEM_DRAM][S_BASE2 + i * MODEL_DIM:S_BASE2 + (i+1) * MODEL_DIM] = h2[i]
        npu._vrf[MEM_DRAM][S_GATE + i * MODEL_DIM:S_GATE + (i+1) * MODEL_DIM] = gate[i]
        npu._vrf[MEM_DRAM][S_UP + i * MODEL_DIM:S_UP + (i+1) * MODEL_DIM] = up[i]

    # ── Phase 2: Run firmware (phase flag = 1) ──
    npu._vrf[MEM_DRAM][S_FLAGS] = np.frombuffer(np.uint32(1).tobytes(), dtype=np.float32)[0]
    cpu = MiniRV64()
    cpu.set_mmio_device(npu)
    cpu.load_elf(str(elf_path))
    cpu.run(cycles=200000)

    # ── ISS reads last_h, computes LM head ──
    last_h = npu._vrf[MEM_DRAM][FW_LAST_H:FW_LAST_H + MODEL_DIM].copy()
    last_normed = rms_norm(last_h, w['norm_final'])
    logits = last_normed @ w['embedding'].T
    return logits, npu


def infer_iss(dram, a, b):
    tokens = encode_prompt(a, b)[:]
    for step in range(OUTPUT_DIGITS):
        logits, npu = run_forward_iss(dram, tokens)
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
    return decode_output(tokens)


GOLDEN_DRAM_140P = None

def _get_dram():
    global GOLDEN_DRAM_140P
    if GOLDEN_DRAM_140P is None:
        GOLDEN_DRAM_140P, _ = build_dram()
    return GOLDEN_DRAM_140P


@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
def test_iss_build():
    r, elf = _build_firmware(4, seq_len=24)
    assert r.returncode == 0, f"Build failed:\n{r.stderr[:300]}"
    assert elf.exists(), f"ELF not found: {elf}"
    print(f"Firmware built: {elf}")


@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
@pytest.mark.parametrize("a,b,expected", [
    (5, 5, 10),
    (555, 445, 1000),
    (0, 0, 0),
])
def test_iss_single(a, b, expected):
    dram = _get_dram()
    result = infer_iss(dram, a, b)
    assert result == expected, f"ISS: {a} + {b} = {result} (expected {expected})"


@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
def test_iss_vs_golden_first_step():
    dram = _get_dram()
    a, b = 5, 5
    tokens = encode_prompt(a, b)
    gl = golden_forward(dram, tokens)
    nl, npu = run_forward_iss(dram, tokens)
    max_diff = float(np.max(np.abs(gl - nl)))
    print(f"Golden logits: {gl}")
    print(f"ISS logits:    {nl}")
    print(f"Max diff: {max_diff:.6f}")
    assert np.argmax(gl) == np.argmax(nl), (
        f"Argmax mismatch: golden={np.argmax(gl)}, iss={np.argmax(nl)}")
    assert max_diff < 1e-3, f"Logit max diff {max_diff:.6f} >= 1e-3"
