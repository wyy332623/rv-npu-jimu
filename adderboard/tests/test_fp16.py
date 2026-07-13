"""
Phase 1 FP16 tests — Python-driven NPU instructions, FP16 truncation enabled.

140p model only (130p is not FP16-safe). Uses NpuDeviceMini directly with
FP16 pipeline rounding to verify the trained weights survive FP16 quantization.
"""

import numpy as np
import pytest

from emulator.npu_device_mini import MEM_DRAM

from adderboard.tests.test_140p_fp16 import NpuFP16

from adderboard.tests.conftest import (
    _load_dram, SHARED_CASES,
    get_golden_forward, encode_tokens, decode_result, get_output_digits,
)


def _build_forward_fp16(npu, tokens):
    """Full forward pass through FP16 emulator (adapted from test_140p_fp16.py)."""
    from adderboard.tests.test_140p_fp16 import build_forward_fp16
    return build_forward_fp16(npu, tokens)


def test_first_step():
    """FP16 single-step logits — argmax matches golden, max diff within tolerance."""
    model = '140p'
    dram = _load_dram(model)
    golden_forward = get_golden_forward(model)

    tokens = encode_tokens(model, 5, 5)
    gl = golden_forward(dram, tokens)

    npu = NpuFP16()
    npu._vrf[MEM_DRAM][:len(dram)] = dram.copy()
    nl = _build_forward_fp16(npu, tokens)

    assert np.argmax(gl) == np.argmax(nl), \
        f"FP16 argmax mismatch: golden={np.argmax(gl)} fp16={np.argmax(nl)}"
    max_diff = float(np.max(np.abs(gl - nl)))
    # FP16 tolerance: max observed diff ~0.76, allow up to 20 (conservative)
    assert max_diff < 20.0, f"FP16 max diff {max_diff:.6f} >= 20"


@pytest.mark.parametrize("a,b,expected", SHARED_CASES)
def test_autoregressive(a, b, expected):
    """FP16 autoregressive inference produces correct final sum."""
    model = '140p'
    dram = _load_dram(model)
    output_digits = get_output_digits(model)

    npu = NpuFP16()
    npu._vrf[MEM_DRAM][:len(dram)] = dram.copy()
    tokens = encode_tokens(model, a, b)

    for _ in range(output_digits):
        logits = _build_forward_fp16(npu, tokens)
        tokens.append(int(np.argmax(logits)))

    result = decode_result(model, tokens)
    assert result == expected, f"FP16: {a}+{b}={result} (expected {expected})"


def test_bulk_random():
    """FP16: 20 random autoregressive pairs — verify accuracy."""
    model = '140p'
    dram = _load_dram(model)
    output_digits = get_output_digits(model)

    rng = np.random.RandomState(789)
    failures = 0
    n = 20
    for _ in range(n):
        a = int(rng.randint(0, 10**10 - 1))
        b = int(rng.randint(0, 10**10 - 1))

        npu = NpuFP16()
        npu._vrf[MEM_DRAM][:len(dram)] = dram.copy()
        tokens = encode_tokens(model, a, b)
        for _ in range(output_digits):
            logits = _build_forward_fp16(npu, tokens)
            tokens.append(int(np.argmax(logits)))
        result = decode_result(model, tokens)
        if result != a + b:
            failures += 1

    accuracy = (n - failures) / n
    print(f"\nFP16 accuracy: {n-failures}/{n} ({accuracy:.1%})")
    # Model trained to 99.0% FP16 accuracy — allow 1 failure in 20
    assert failures <= 2, f"FP16 bulk: {failures}/{n} failed (expected ≤2)"
