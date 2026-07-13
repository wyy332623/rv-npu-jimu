"""
Shared fixtures for adderboard NPU tests.

Provides model factories, DRAM loading, token encoding, and firmware building
parametrized across both models (130p and 140p).
"""

import subprocess
from pathlib import Path
import pytest
import numpy as np

# ── Model registry ───────────────────────────────────────────────

MODELS = {}

def _register(name, golden_forward, build_forward_fn, encode_fn, decode_fn, output_digits, is_fp16_safe):
    MODELS[name] = {
        'golden_forward': golden_forward,
        'build_forward_fn': build_forward_fn,
        'encode_fn': encode_fn,
        'decode_fn': decode_fn,
        'output_digits': output_digits,
        'is_fp16_safe': is_fp16_safe,
    }

# Lazy imports
def _import_130p():
    from adderboard.layout.layout_130p import build_dram, encode_prompt, decode_output, dram_addr, OUTPUT_DIGITS
    from adderboard.golden.golden_130p import forward as golden_forward
    from adderboard.tests._test_130p_phase1 import build_forward
    _register('130p', golden_forward, build_forward, encode_prompt, decode_output, OUTPUT_DIGITS, is_fp16_safe=False)

def _import_140p():
    from adderboard.layout.layout_140p import build_dram, encode_prompt, decode_output, dram_addr, OUTPUT_DIGITS
    from adderboard.golden.golden_140p import forward as golden_forward
    from adderboard.tests._test_140p_phase1 import build_forward
    _register('140p', golden_forward, build_forward, encode_prompt, decode_output, OUTPUT_DIGITS, is_fp16_safe=True)

_import_130p()
_import_140p()

# ── Shared test cases ────────────────────────────────────────────

# Test cases: (a, b, expected_sum)
# Covers: trivial, small, mid, random, two 10-digit overflow edges
SHARED_CASES = [
    pytest.param(0, 0, 0, id='0-0-0'),
    pytest.param(5, 5, 10, id='5-5-10'),
    pytest.param(555, 445, 1000, id='555-445-1000'),
    pytest.param(19492, 23919, 43411, id='19492-23919-43411'),
    pytest.param(9999999999, 1, 10000000000, id='9999999999-1-10000000000'),
    pytest.param(1111111111, 8888888889, 10000000000, id='1111111111-8888888889-10000000000'),
]

# ── DRAM fixture ─────────────────────────────────────────────────

GOLDEN_DRAM_CACHE = {}

def _load_dram(model_name):
    if model_name not in GOLDEN_DRAM_CACHE:
        info = MODELS[model_name]
        if model_name == '130p':
            from adderboard.layout.layout_130p import build_dram
        else:
            from adderboard.layout.layout_140p import build_dram
        dram, _ = build_dram()
        GOLDEN_DRAM_CACHE[model_name] = dram
    return GOLDEN_DRAM_CACHE[model_name]


def get_golden_forward(model_name):
    return MODELS[model_name]['golden_forward']


def encode_tokens(model_name, a, b):
    return MODELS[model_name]['encode_fn'](a, b)


def decode_result(model_name, tokens):
    return MODELS[model_name]['decode_fn'](tokens)


def get_output_digits(model_name):
    return MODELS[model_name]['output_digits']


# ── Firmware build helper ────────────────────────────────────────

FW_DIR = Path(__file__).resolve().parent.parent.parent / "firmware"
FW_TARGETS = {'130p': 'adder', '140p': 'adder_140p'}


def build_firmware(model_name, dim=4, seq_len=24):
    target = FW_TARGETS[model_name]
    build_dir = f'build_dim{dim}'
    abs_build_dir = FW_DIR / build_dir
    import os
    env = {'NATIVE_DIM': str(dim), 'SEQ_LEN': str(seq_len)}
    full_env = {**os.environ, **env}
    r = subprocess.run(
        ['make', '-C', str(FW_DIR), f'BUILD_DIR={build_dir}',
         f'TARGET={target}', 'clean', 'all'],
        capture_output=True, text=True, env=full_env)
    elf = abs_build_dir / f'{target}.elf' if target == 'adder_140p' else abs_build_dir / 'adder.elf'
    return r, elf


# ── Model-parametrized fixture ───────────────────────────────────

def pytest_generate_tests(metafunc):
    """Auto-parametrize 'model' fixture where requested."""
    if 'model' in metafunc.fixturenames:
        metafunc.parametrize('model', list(MODELS.keys()), ids=list(MODELS.keys()))
