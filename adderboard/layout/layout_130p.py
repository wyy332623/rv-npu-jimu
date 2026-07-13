"""
NPU DRAM layout for cosminscn_130p hand-coded weights.

Weights are dumped from the verified PyTorch model. DRAM is a flat
float32 array indexed by element offset.  All addresses are computed
as: next_addr = addr + count, with no gaps.

Weight sizes:
  PE table:    max_seq_len × 4  floats  (variable, up to 33×4=132)
  embed_A:     10 floats
  embed_B:     4 floats
  c_attn:      12 × 4 = 48 floats  ([12, 4] row-major)
  c_proj:      4 × 4 = 16 floats
  c_fc:        4 × 4 = 16 floats
  c_fc_bias:   4 floats
  c_proj_u:    4 floats
  c_proj_v:    4 floats
  lm_u:        10 floats
  lm_v:        4 floats
  lm_bias:     10 floats
  ─────────────────────
  Total:       268 + PE

All weights loaded as-is from PyTorch model (verified 10926/10926 tests).
"""

import numpy as np
import math

VOCAB_SIZE = 10
MODEL_DIM = 4
N_HEADS = 2
HEAD_DIM = 2
FF_DIM = 4
OUTPUT_DIGITS = 11
PROMPT_LEN = 22
TOTAL_LEN = PROMPT_LEN + OUTPUT_DIGITS  # 33


def _compute_addrs(max_seq_len):
    """Compute DRAM addresses. Returns dict of (name, addr, count)."""
    addrs = {}
    addr = 0
    for name, count in [
        ('pe_table', max_seq_len * MODEL_DIM),
    ]:
        addrs[name] = (addr, count)
        addr += count
    for name, count in [
        ('embed_A', VOCAB_SIZE),
        ('embed_B', MODEL_DIM),
        ('c_attn', 12 * MODEL_DIM),
        ('c_proj', MODEL_DIM * MODEL_DIM),
        ('c_fc', FF_DIM * MODEL_DIM),
        ('c_fc_bias', FF_DIM),
        ('c_proj_u', MODEL_DIM),
        ('c_proj_v', 16),  # 4×4 MRF tile load (padded to 16)
        ('lm_u', VOCAB_SIZE),
        ('lm_bias', VOCAB_SIZE),
        ('lm_v', 16),  # 4×4 MRF tile load (padded to 16) in build_dram
    ]:
        addrs[name] = (addr, count)
        addr += count
    addrs['dram_size'] = addr
    addrs['scratch'] = 256  # generous scratch space after weights
    return addrs


# Module-level address cache (populated at first build)
_ADDRS = {}


def dram_addr(name):
    """Get DRAM element offset for a named tensor."""
    if not _ADDRS:
        _ADDRS.update(_compute_addrs(TOTAL_LEN))
    return _ADDRS[name][0]


def dram_size():
    if not _ADDRS:
        _ADDRS.update(_compute_addrs(TOTAL_LEN))
    return _ADDRS['dram_size']


def build_dram(max_seq_len=TOTAL_LEN, pe_scale=1.0):
    """Build DRAM with exact weights from the verified PyTorch model.

    Parameters
    ----------
    max_seq_len : int
        Maximum sequence length (determines PE table size).
    pe_scale : float
        Scale factor for PE table and embedding weights. Used for FP16
        adaptation (Option A): setting pe_scale=PE_SCALE avoids FP16
        overflow in attention scores.

    When pe_scale != 1.0:
      - pe_table and embed_B are scaled down by pe_scale
      - c_fc is scaled up by 1/pe_scale
      - c_fc_bias is NOT scaled (the x@c_fc.T product auto-cancels:
        (s*x) @ (c_fc/s).T = x @ c_fc.T, so bias stays at original)
    This preserves the model's computation while keeping all intermediate
    values within FP16 range.

    Returns
    -------
    dram : np.ndarray[float32, shape=(524288,)]
    meta : dict
    """
    # PE_SCALE constant: 0.0226
    # Max |Q| ~ 11321 (from PE amplitude 100 * |c_attn|).
    # Q@K max ~ 128M. FP16 max = 65504.
    # sqrt(65504) ~ 256 max |Q| => scale by 256 / 11321.
    _PE_SCALE_DEFAULT = 0.0226
    addrs = _compute_addrs(max_seq_len)
    total = addrs['dram_size'] + 256  # + scratch
    dram = np.zeros(max(524288, total), dtype=np.float32)

    # ---- PE table ----
    th = 2.0 * math.pi / 11.0
    pe = np.zeros((max_seq_len, MODEL_DIM), dtype=np.float32)
    positions = np.arange(max_seq_len, dtype=np.float32)
    amplitude = np.where(positions <= 21, 100.0, 1.0)
    pe[:, 1] = amplitude * np.sin(positions * th)
    pe[:, 2] = amplitude * np.cos(positions * th)
    if pe_scale != 1.0:
        pe = pe * pe_scale
    addr_pe, cnt_pe = addrs['pe_table']
    dram[addr_pe:addr_pe + cnt_pe] = pe.flatten()

    # ---- Embedding: A[v]=v, B=[1,0,0,0] ----
    addr, cnt = addrs['embed_A']
    dram[addr:addr + cnt] = np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.float32)

    embed_B = np.array([1, 0, 0, 0], dtype=np.float32)
    if pe_scale != 1.0:
        embed_B = embed_B * pe_scale
    addr, cnt = addrs['embed_B']
    dram[addr:addr + cnt] = embed_B

    # ---- c_attn [12, 4] — PyTorch's W (NOT transposed) ----
    # From build_adder() verification:
    c_attn = np.array([
        [  0.      ,  14.231483, -98.98215 ,   0.      ],
        [  0.      , -98.98215 , -14.231483,   0.      ],
        [  0.      , -41.5415  , -90.963196,   0.      ],
        [  0.      , -90.963196,  41.5415  ,   0.      ],
        [  0.      ,   1.      ,   0.      ,   0.      ],
        [  0.      ,   0.      ,   1.      ,   0.      ],
        [  0.      ,   1.      ,   0.      ,   0.      ],
        [  0.      ,   0.      ,   1.      ,   0.      ],
        [  1.      ,   0.      ,   0.      ,   0.      ],
        [  0.      ,   0.      ,   0.      ,   0.      ],
        [  1.      ,   0.      ,   0.      ,   0.      ],
        [  0.      ,   0.      ,   0.      ,   0.      ],
    ], dtype=np.float32)  # [12, 4]
    addr, cnt = addrs['c_attn']
    dram[addr:addr + cnt] = c_attn.flatten()

    # ---- c_proj [4, 4] ----
    c_proj = np.array([
        [0., 0., 0., 0.],
        [0., 0., 2., 0.],
        [0., 0., 0., 0.],
        [2., 0., 0., 0.],
    ], dtype=np.float32)
    addr, cnt = addrs['c_proj']
    dram[addr:addr + cnt] = c_proj.flatten()

    # ---- c_fc [4, 4] + bias ----
    c_fc = np.array([
        [-100.,  100.,    0.,    0.],
        [-100.,  100.,    0.,    0.],
        [ -10.,   10.,    0., 1000.],
        [ -10.,   10.,    0., 1000.],
    ], dtype=np.float32)
    c_fc_bias = np.array([-50., -150., -9045., -9055.], dtype=np.float32)
    if pe_scale != 1.0:
        inv = 1.0 / pe_scale
        c_fc = c_fc * inv
        # Bias stays at original value because x@c_fc.T product
        # auto-cancels: (s*x) @ (c_fc/s).T = x @ c_fc.T + bias
    addr, cnt = addrs['c_fc']
    dram[addr:addr + cnt] = c_fc.flatten()
    addr, cnt = addrs['c_fc_bias']
    dram[addr:addr + cnt] = c_fc_bias

    # ---- c_proj MLP u, v ----
    c_proj_u = np.array([0., 0., 0., 1.], dtype=np.float32)
    c_proj_v = np.array([0.01, -0.01, -1.0, 1.0], dtype=np.float32)
    addr, cnt = addrs['c_proj_u']
    dram[addr:addr + cnt] = c_proj_u
    addr, cnt = addrs['c_proj_v']
    dram[addr:addr + 4] = c_proj_v
    dram[addr + 4:addr + 16] = 0.0  # zero-pad for 4×4 MRF tile load

    # ---- LM head ----
    lm_u = np.array([0., 2., 4., 6., 8., 10., 12., 14., 16., 18.], dtype=np.float32)
    lm_v = np.array([0., 0., 0., 1.], dtype=np.float32)
    lm_bias = np.array([0., -1., -4., -9., -16., -25., -36., -49., -64., -81.], dtype=np.float32)
    addr, cnt = addrs['lm_u']
    dram[addr:addr + cnt] = lm_u
    addr, cnt = addrs['lm_bias']
    dram[addr:addr + cnt] = lm_bias
    addr, cnt = addrs['lm_v']
    dram[addr:addr + 4] = lm_v
    dram[addr + 4:addr + 16] = 0.0  # zero-pad for 4×4 MRF tile load

    meta = {
        'max_seq_len': max_seq_len,
        'model_dim': MODEL_DIM,
        'n_heads': N_HEADS,
        'head_dim': HEAD_DIM,
        'ff_dim': FF_DIM,
        'vocab_size': VOCAB_SIZE,
        'output_digits': OUTPUT_DIGITS,
        'prompt_len': PROMPT_LEN,
        'dram_weight_bytes': addrs['dram_size'] * 4,
    }
    return dram, meta


def encode_prompt(a, b):
    """Encode a+b into token sequence matching the model's usage.
    
    Original model: f"{a:010d}+{b:010d}=" where '+' and '=' map to 0.
    """
    s = f"{a:010d}+{b:010d}="
    tok = {'+': 0, '=': 0}
    return [int(c) if c.isdigit() else tok[c] for c in s]


def decode_output(tokens):
    """Decode output digits starting at position PROMPT_LEN (22)."""
    return int("".join(str(t) for t in tokens[PROMPT_LEN:])[::-1])
