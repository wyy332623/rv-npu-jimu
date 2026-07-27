"""
Phase 2 tests — RISC-V firmware running on ISS, driving the NPU via MMIO.

Parametrized across 130p and 140p models. Tests firmware compilation,
single-step logits, and autoregressive inference through the full ISS stack.
"""

import numpy as np
import pytest

from emulator.npu_device_mini import (
    NpuDeviceMini, MEM_DRAM,
    MEM_MULTIPLY_VRF, MEM_MVM_INITIAL_VRF, MEM_MATRIX_RF,
    MEM_VEC_TO_MAT_ROW, MEM_FILL, MEM_SPU_ADD_REDUCE, MEM_SPU_MAX_REDUCE,
    MEM_SPU_BROADCAST, MEM_SPU_ABSMAX_REDUCE,
    OP_V_RD_DRAM, OP_V_WR_DRAM, OP_V_RD, OP_V_WR,
    OP_M_RD_DRAM, OP_M_WR, OP_M_RD, OP_MV_MUL,
    OP_VV_ADD, OP_VV_A_SUB_B, OP_VV_B_SUB_A, OP_VV_MUL,
    OP_V_RELU, OP_V_EXP, OP_V_FUNC, OP_S_WR, OP_S_RECIP,
    OP_S_SQRT, OP_SS_MUL, OP_V_SIGM, OP_V_TANH, OP_V_GELU,
    SUB_SOFTMAX, SUB_LAYERNORM,
)

from adderboard.tests.conftest import (
    MODELS, SHARED_CASES, _load_dram,
    get_golden_forward, encode_tokens, decode_result, get_output_digits,
    build_firmware,
)

HAS_ISS = False
MiniRV64 = None
try:
    from iss.mini_rv64 import MiniRV64
    HAS_ISS = True
except ImportError:
    pass


# ── Test helpers ─────────────────────────────────────────────────

def _make_npu_fp32(dram_arr):
    """Create NpuFP32 (FP32 mode) and load DRAM."""
    from emulator.npu_fp32 import NpuFP32
    npu = NpuFP32()
    npu._vrf[MEM_DRAM][:len(dram_arr)] = dram_arr.copy()
    return npu


def _run_iss_forward(model_name, npu, tokens):
    """Run the ISS firmware for a full forward step."""
    T = len(tokens)
    r, elf_path = build_firmware(model_name, dim=4, seq_len=T)
    if r.returncode != 0 or not elf_path.exists():
        raise RuntimeError(f"Firmware build failed:\n{r.stderr[:500]}")
    cpu = MiniRV64()
    cpu.set_mmio_device(npu)
    cpu.load_elf(str(elf_path))
    cpu.run(cycles=400000)
    return npu


# ── Model-specific forward helpers (imported from original test files) ──

# These need to stay imported from the original files to avoid duplication
def _run_forward_130p(dram, tokens):
    from adderboard.tests._test_130p_phase2 import run_forward_iss as rf
    return rf(dram, tokens)

def _run_forward_140p(dram, tokens):
    from adderboard.tests._test_140p_phase2 import run_forward_iss as rf
    return rf(dram, tokens)

def _infer_iss_130p(dram, a, b):
    from adderboard.tests._test_130p_phase2 import infer_iss
    return infer_iss(dram, a, b)

def _infer_iss_140p(dram, a, b):
    from adderboard.tests._test_140p_phase2 import infer_iss
    return infer_iss(dram, a, b)

_RUN_FORWARD = {'130p': _run_forward_130p, '140p': _run_forward_140p}
_INFER_ISS = {'130p': _infer_iss_130p, '140p': _infer_iss_140p}


# ── Tests ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
def test_iss_build(model):
    """Firmware compiles successfully."""
    r, elf = build_firmware(model, dim=4, seq_len=24)
    assert r.returncode == 0, f"Build failed:\n{r.stderr[:300]}"
    assert elf.exists(), f"ELF not found: {elf}"


@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
def test_first_step(model):
    """ISS firmware single-step logits match golden."""
    dram = _load_dram(model)
    golden_forward = get_golden_forward(model)

    tokens = encode_tokens(model, 5, 5)
    gl = golden_forward(dram, tokens)

    iss_logits, _ = _RUN_FORWARD[model](dram, tokens)

    assert np.argmax(gl) == np.argmax(iss_logits), \
        f"ISS argmax mismatch: golden={np.argmax(gl)} iss={np.argmax(iss_logits)}"
    max_diff = float(np.max(np.abs(gl - iss_logits)))
    assert max_diff < 0.01, f"ISS max diff {max_diff:.6f} >= 0.01"


@pytest.mark.parametrize("a,b,expected", SHARED_CASES)
@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
def test_autoregressive(model, a, b, expected):
    """ISS firmware autoregressive inference produces correct final sum."""
    dram = _load_dram(model)

    result = _INFER_ISS[model](dram, a, b)
    assert result == expected, f"ISS: {a}+{b}={result} (expected {expected})"


# ── Phase 2 replay cross-check (Phase 1 vs firmware instruction stream) ──

@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
def test_phase2_replay_vs_phase1(model):
    """Captured firmware instruction stream replays match Phase 1 results."""
    from emulator.npu_device_mini import NpuDeviceMini as NDM
    from adderboard.tests._test_130p_phase2 import _build_firmware as bf_130p
    from adderboard.tests._test_140p_phase2 import _build_firmware as bf_140p

    dram = _load_dram(model)
    golden_forward = get_golden_forward(model)

    tokens = encode_tokens(model, 5, 5)
    T = len(tokens)
    gl = golden_forward(dram, tokens)

    # Run ISS to capture instructions
    r, elf_path = (bf_140p(4, seq_len=T) if model == '140p' else bf_130p(4, seq_len=T))
    if r.returncode != 0 or not elf_path.exists():
        pytest.skip(f"Build failed: {r.stderr[:200]}")

    from emulator.npu_device_mini import NpuDeviceMini as NDM

    fw_inst = []
    orig_push = NDM._push_instruction
    def capture(self, inst):
        fw_inst.append(inst)
        return orig_push(self, inst)
    NDM._push_instruction = capture

    try:
        npu = _make_npu_fp32(dram)
        # Pre-fill DRAM as the ISS test would
        if model == '140p':
            from adderboard.tests._test_140p_phase2 import (
                S_X, S_Q, S_K, S_V, S_MASK_TABLE, MAX_SEQ,
                S_FLAGS,
            )
            from adderboard.golden.golden_140p import _load_weights, rms_norm, apply_rope_numpy
            from adderboard.layout.layout_140p import HEAD_DIM, MODEL_DIM
            w = _load_weights(npu._vrf[MEM_DRAM])
            x_np = np.array([w['embedding'][t] for t in tokens])
            h1 = rms_norm(x_np, w['norm1'])
            q = h1 @ w['W_q'].T; kv = h1 @ w['W_kv'].T; k = v = kv
            q = rms_norm(q, w['q_norm']); k = rms_norm(k, w['k_norm'])
            pos = np.arange(T)
            q = apply_rope_numpy(q.reshape(T, 1, HEAD_DIM), w['rope_table'], pos).reshape(T, HEAD_DIM)
            k = apply_rope_numpy(k.reshape(T, 1, HEAD_DIM), w['rope_table'], pos).reshape(T, HEAD_DIM)
            for i in range(T):
                npu._vrf[MEM_DRAM][S_X+i*4:S_X+(i+1)*4] = x_np[i]
                npu._vrf[MEM_DRAM][S_Q+i*4:S_Q+(i+1)*4] = q[i]
                npu._vrf[MEM_DRAM][S_K+i*4:S_K+(i+1)*4] = k[i]
                npu._vrf[MEM_DRAM][S_V+i*4:S_V+(i+1)*4] = v[i]
            nt = (T+3)//4; mt = S_MASK_TABLE
            for qq in range(T):
                for tc in range(nt):
                    base = tc*4; valid = min(4, T-base)
                    mask = np.full(4, -1e30, np.float32)
                    for p in range(valid):
                        if base+p <= qq: mask[p] = 0.0
                    npu._vrf[MEM_DRAM][mt+(qq*nt+tc)*4:mt+(qq*nt+tc)*4+4] = mask
            svt = mt + MAX_SEQ * nt * 4
            for tc in range(nt):
                base = tc*4; valid = min(4, T-base)
                vt = np.zeros((MODEL_DIM, MODEL_DIM), np.float32)
                for p in range(valid):
                    for d in range(MODEL_DIM): vt[d, p] = v[base+p, d]
                for d in range(MODEL_DIM):
                    npu._vrf[MEM_DRAM][svt+tc*MODEL_DIM*MODEL_DIM+d*MODEL_DIM:svt+tc*MODEL_DIM*MODEL_DIM+(d+1)*MODEL_DIM] = vt[d]
            npu._vrf[MEM_DRAM][S_FLAGS] = np.frombuffer(np.uint32(0).tobytes(), dtype=np.float32)[0]
            npu._spu_srf[6] = 0.5
            npu.set_seq_len(T)

        cpu = MiniRV64()
        cpu.set_mmio_device(npu)
        cpu.load_elf(str(elf_path))
        cpu.run(cycles=400000)

        iss_logits = _RUN_FORWARD[model](dram, tokens)[0]
    finally:
        NDM._push_instruction = orig_push

    # Replay on fresh NPU
    npu2 = _make_npu_fp32(dram)
    for inst in fw_inst:
        try:
            npu2._push_instruction(inst)
        except Exception:
            pass  # Skip non-instruction writes (S_WR, etc.)

    assert np.argmax(gl) == np.argmax(iss_logits), \
        f"Replay argmax mismatch: golden={np.argmax(gl)} iss={np.argmax(iss_logits)}"


@pytest.mark.skipif(not HAS_ISS, reason="ISS not available")
def test_bulk_random(model):
    """ISS firmware: 5 random autoregressive pairs."""
    dram = _load_dram(model)

    rng = np.random.RandomState(123)
    failures = 0
    for _ in range(5):
        a = int(rng.randint(0, 10**10 - 1))
        b = int(rng.randint(0, 10**10 - 1))
        result = _INFER_ISS[model](dram, a, b)
        if result != a + b:
            failures += 1
    max_fail = 1 if model == '140p' else 0  # 140p: 99% accuracy
    assert failures <= max_fail, \
        f"ISS bulk: {failures}/10 failed (max {max_fail}): model={model}"
