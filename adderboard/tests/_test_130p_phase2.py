"""
Phase 2: Single-phase firmware using MMIO CPU-NPU data exchange.
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

from adderboard.layout.layout_130p import (
    build_dram, encode_prompt, decode_output,
    dram_addr, MODEL_DIM, N_HEADS, HEAD_DIM,
    PROMPT_LEN, OUTPUT_DIGITS, TOTAL_LEN, VOCAB_SIZE,
)

try:
    from iss.mini_rv64 import MiniRV64
    HAS_ISS = True
except ImportError:
    HAS_ISS = False
    MiniRV64 = None

from adderboard.golden.golden_130p import forward as golden_forward

FW_DIR = Path(__file__).resolve().parent.parent.parent / "firmware"

def _build_firmware(dim, seq_len=1):
    build_dir = f'build_dim{dim}'
    abs_build_dir = FW_DIR / build_dir
    env = {'NATIVE_DIM': str(dim), 'SEQ_LEN': str(seq_len)}
    import os
    full_env = {**os.environ, **env}
    r = subprocess.run(
        ['make', '-C', str(FW_DIR), f'BUILD_DIR={build_dir}',
         'TARGET=adder', 'clean', 'all'],
        capture_output=True, text=True, env=full_env)
    elf = abs_build_dir / 'adder.elf'
    return r, elf

# Scratch addresses (must match C firmware)
MAX_SEQ = 33
SCR_BASE = 0x1000
S_X = SCR_BASE
S_Q = S_X + MAX_SEQ * MODEL_DIM
S_K = S_Q + MAX_SEQ * MODEL_DIM
S_V = S_K + MAX_SEQ * MODEL_DIM
S_CTX = S_V + MAX_SEQ * MODEL_DIM
S_ATTN_OUT = S_CTX + MAX_SEQ * MODEL_DIM
S_AFTER_ATTN = S_ATTN_OUT + MAX_SEQ * MODEL_DIM
S_FC = S_AFTER_ATTN + MAX_SEQ * MODEL_DIM
S_MLP_OUT = S_FC + MAX_SEQ * MODEL_DIM
S_AFTER_MLP = S_MLP_OUT + MAX_SEQ * MODEL_DIM
S_SCORE = S_AFTER_MLP + MAX_SEQ * MODEL_DIM
S_PROB = S_SCORE + MAX_SEQ
S_TEMP = S_PROB + MAX_SEQ
S_MASK_H0 = S_TEMP + 0
S_MASK_H1 = S_TEMP + 4
S_MASK_TABLE = S_TEMP + MAX_SEQ
FW_LAST_H = 0x2000


def make_npu(dram_arr):
    npu = NpuDeviceMini(native_dim=4)
    # Disable FP16 rounding for ISS test path
    npu._round_fp16 = lambda arr: arr if arr is not None else None
    npu._store_to_ivrf = lambda p: None
    npu._vrf[MEM_DRAM][:len(dram_arr)] = dram_arr.copy()
    return npu


def run_forward_iss(dram, tokens):
    T = len(tokens)

    r, elf_path = _build_firmware(4, seq_len=T)
    if r.returncode != 0 or not elf_path.exists():
        raise RuntimeError(f"Firmware build failed:\n{r.stderr[:500]}")

    npu = make_npu(dram)

    # Input tokens as [token, 0, 0, 0]
    for i in range(T):
        npu._vrf[MEM_DRAM][S_X + i * MODEL_DIM:
                            S_X + (i + 1) * MODEL_DIM] = \
            np.array([float(tokens[i]), 0.0, 0.0, 0.0], dtype=np.float32)

    # Head masks
    npu._vrf[MEM_DRAM][S_MASK_H0:S_MASK_H0 + MODEL_DIM] = np.array(
        [1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    npu._vrf[MEM_DRAM][S_MASK_H1:S_MASK_H1 + MODEL_DIM] = np.array(
        [0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    # Causal mask table
    nt = (T + 3) // 4
    mt = S_MASK_TABLE
    for q in range(T):
        for tc in range(nt):
            base = tc * 4
            valid = min(4, T - base)
            mask = np.full(4, -1e30, dtype=np.float32)
            for p in range(valid):
                if base + p <= q:
                    mask[p] = 0.0
            addr = mt + (q * nt + tc) * MODEL_DIM
            npu._vrf[MEM_DRAM][addr:addr + MODEL_DIM] = mask

    # Transposed V tiles
    from adderboard.golden.golden_130p import _load_weights
    w = _load_weights(
        dram if dram.dtype == np.float32 else np.frombuffer(dram, dtype=np.float32)
    )
    x = np.zeros((T, MODEL_DIM), dtype=np.float32)
    for i in range(T):
        x[i] = w['embed_A'][tokens[i]] * w['embed_B'] + w['pe_table'][i]
    qkv = x @ w['c_attn'].T
    v = qkv[:, 2 * MODEL_DIM:3 * MODEL_DIM]

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
                                tile_addr + (d + 1) * MODEL_DIM] = vt[d]

    # Scale in SRF[6]
    npu._spu_srf[6] = 2.0 ** -0.5

    npu.set_seq_len(T)
    cpu = MiniRV64()
    cpu.set_mmio_device(npu)
    cpu.load_elf(str(elf_path))
    cpu.run(cycles=500000)

    last_h = npu._vrf[MEM_DRAM][FW_LAST_H:FW_LAST_H + MODEL_DIM].copy()
    lm_v = npu._vrf[MEM_DRAM][dram_addr('lm_v'):dram_addr('lm_v') + MODEL_DIM].copy()
    lm_u = npu._vrf[MEM_DRAM][dram_addr('lm_u'):dram_addr('lm_u') + VOCAB_SIZE].copy()
    lm_b = npu._vrf[MEM_DRAM][dram_addr('lm_bias'):dram_addr('lm_bias') + VOCAB_SIZE].copy()

    proj = float(np.dot(last_h, lm_v))
    logits = proj * lm_u + lm_b
    return logits, npu


def infer_iss(dram, a, b):
    tokens = encode_prompt(a, b)
    for step in range(OUTPUT_DIGITS):
        logits, npu = run_forward_iss(dram, tokens)
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
    return decode_output(tokens)


GOLDEN_DRAM = None

def _get_dram():
    global GOLDEN_DRAM
    if GOLDEN_DRAM is None:
        GOLDEN_DRAM, _ = build_dram()
    return GOLDEN_DRAM


@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
def test_iss_build():
    r, elf = _build_firmware(4, seq_len=22)
    assert r.returncode == 0, f"Build failed:\n{r.stderr[:300]}"
    assert elf.exists(), f"ELF not found: {elf}"
    print(f"Firmware built: {elf}")


@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
@pytest.mark.parametrize("a,b,expected", [
    (5, 5, 10),
    (555, 445, 1000),
    (0, 0, 0),
    (9999999999, 1, 10000000000),
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
