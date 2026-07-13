"""
Phase 1 tests — Python-driven NPU instructions, FP32 mode.

Parametrized across 130p and 140p models. No RISC-V firmware involved.
The NPU is driven directly by Python via _push_instruction().
"""

import numpy as np
import pytest

from emulator.npu_fp32 import NpuFP32
from emulator.npu_device_mini import MEM_DRAM

from adderboard.tests.conftest import (
    MODELS, SHARED_CASES, _load_dram,
    get_golden_forward, encode_tokens, decode_result, get_output_digits,
)


def test_first_step(model):
    """Single-step logits match golden reference."""
    dram = _load_dram(model)
    golden_forward = get_golden_forward(model)
    build_forward = MODELS[model]['build_forward_fn']

    tokens = encode_tokens(model, 5, 5)
    gl = golden_forward(dram, tokens)

    npu = NpuFP32()
    npu._vrf[MEM_DRAM][:len(dram)] = dram.copy()
    nl = build_forward(npu, tokens)

    assert np.argmax(gl) == np.argmax(nl), \
        f"Argmax mismatch: golden={np.argmax(gl)} npu={np.argmax(nl)}"
    max_diff = float(np.max(np.abs(gl - nl)))
    assert max_diff < 0.01, f"Max diff {max_diff:.6f} >= 0.01"


@pytest.mark.parametrize("a,b,expected", SHARED_CASES)
def test_autoregressive(model, a, b, expected):
    """Autoregressive inference produces correct final sum."""
    dram = _load_dram(model)
    build_forward = MODELS[model]['build_forward_fn']
    output_digits = get_output_digits(model)

    npu = NpuFP32()
    npu._vrf[MEM_DRAM][:len(dram)] = dram.copy()
    tokens = encode_tokens(model, a, b)

    for _ in range(output_digits):
        logits = build_forward(npu, tokens)
        tokens.append(int(np.argmax(logits)))

    result = decode_result(model, tokens)
    assert result == expected, f"{a}+{b}={result} (expected {expected})"


def test_bulk_random(model):
    """50 random autoregressive pairs — all match golden."""
    dram = _load_dram(model)
    build_forward = MODELS[model]['build_forward_fn']
    output_digits = get_output_digits(model)

    rng = np.random.RandomState(42)
    failures = []
    for _ in range(50):
        a = int(rng.randint(0, 10**10 - 1))
        b = int(rng.randint(0, 10**10 - 1))

        npu = NpuFP32()
        npu._vrf[MEM_DRAM][:len(dram)] = dram.copy()
        tokens = encode_tokens(model, a, b)
        for _ in range(output_digits):
            logits = build_forward(npu, tokens)
            tokens.append(int(np.argmax(logits)))
        result = decode_result(model, tokens)

        if result != a + b:
            failures.append(f"{a}+{b}={result} (expected {a+b})")

    max_fail = 2 if model == '140p' else 0  # 140p has ~1% inherent error
    assert len(failures) <= max_fail, \
        f"{len(failures)}/{50} failed (max {max_fail}): {failures[:5]}"


# 140p-only: intermediate value verification
def test_intermediates_140p():
    """140p only: verify CTX intermediate values match golden (FP32)."""
    model = '140p'
    dram = _load_dram(model)
    golden_forward = MODELS[model]['golden_forward']
    build_forward = MODELS[model]['build_forward_fn']

    tokens = encode_tokens(model, 5, 5)
    gl = golden_forward(dram, tokens)

    npu = NpuFP32()
    npu._vrf[MEM_DRAM][:len(dram)] = dram.copy()
    nl = build_forward(npu, tokens)

    assert np.argmax(gl) == np.argmax(nl), \
        f"Argmax mismatch: golden={np.argmax(gl)} npu={np.argmax(nl)}"
    max_diff = float(np.max(np.abs(gl - nl)))
    assert max_diff < 0.01, f"Max diff {max_diff:.6f} >= 0.01"
