"""
NPU DRAM layout for trained 140p Qwen3-style model (seed 44, FP16-safe).

Architecture: 1L Qwen3 decoder, d=4, 1h, hd=4, ff=4, RoPE θ=3,
              SwiGLU, RMSNorm, tied K=V, O=Q^T, LM head=embed.T.
Weights:     140 trained parameters (all <74 → FP16-safe).
Weight file: adderboard/models/140p/s44_targeted_final_fp16.pt

Weight sizes (all in float32 after loading):
  embedding:    10×4 = 40 floats
  norm1:        4 floats
  norm2:        4 floats
  norm_final:   4 floats
  W_q:          4×4 = 16 floats
  W_kv:         4×4 = 16 floats
  q_norm:       4 floats
  k_norm:       4 floats
  W_gate:       4×4 = 16 floats
  W_up:         4×4 = 16 floats
  W_down:       4×4 = 16 floats
  RoPE table:   35×2 = 70 floats (cos, sin interleaved per position)
  embed_B:      4 floats (bias/scalar per digit, all zeros)
  ─────────────────────────────────────
  Total:        210 floats (840 bytes)

DRAM is a flat float32 array of 524288 elements (2 MB).
Addresses are 256-float aligned (1024-byte) for simple firmware indexing.
"""

import numpy as np
import math
import os

# ── Architecture constants ──
VOCAB_SIZE = 10
MODEL_DIM = 4
N_HEADS = 1
HEAD_DIM = 4
FF_DIM = 4
OUTPUT_DIGITS = 11
PROMPT_LEN = 24          # [0] + 10 digits(A) rev + [0,0] + 10 digits(B) rev + [0]
TOTAL_LEN = PROMPT_LEN + OUTPUT_DIGITS   # 35

# RoPE constants
ROPE_THETA = 3.0
MAX_ROPE_POS = 35

# Address alignment (each tensor starts at a 256-float boundary)
ALIGN = 256

# ── DRAM address map ──
ADDR_MAP = {
    'embedding':    0x000,   #  10×4  = 40
    'norm1':        0x100,   #  4
    'norm2':        0x200,   #  4
    'norm_final':   0x300,   #  4
    'W_q':          0x400,   #  4×4  = 16
    'W_kv':         0x500,   #  4×4  = 16
    'q_norm':       0x600,   #  4
    'k_norm':       0x700,   #  4
    'W_gate':       0x800,   #  4×4  = 16
    'W_up':         0x900,   #  4×4  = 16
    'W_down':       0xA00,   #  4×4  = 16
    'rope_table':   0xB00,   #  35×4 = 140
    'embed_b':      0xC00,   #  4
    'W_q_t':        0xD00,   #  4×4 transposed = 16  (for O=tied-Q^T, NPU needs transposed MRF)
}

ADDR_SIZES = {
    'embedding':    40,
    'norm1':        4,
    'norm2':        4,
    'norm_final':   4,
    'W_q':          16,
    'W_kv':         16,
    'q_norm':       4,
    'k_norm':       4,
    'W_gate':       16,
    'W_up':         16,
    'W_down':       16,
    'rope_table':   140,
    'embed_b':      4,
    'W_q_t':        16,
}

# Last address + size (for firmware)
ADDR_END = 0xE00


def dram_addr(name):
    """Get DRAM element offset for a named tensor."""
    return ADDR_MAP[name]


def define_addr(name, index):
    """Compute address for firmware: base + index. Index is element offset."""
    return ADDR_MAP[name] + index


def _load_weights(weight_path=None):
    """Load trained weights from checkpoint."""
    if weight_path is None:
        weight_path = os.path.join(os.path.dirname(__file__), '..', 'models', '140p',
                                   's44_targeted_final_fp16.pt')
    import torch
    sd = torch.load(weight_path,
                     weights_only=True, map_location='cpu')
    return {k: v.numpy() for k, v in sd.items()}


def _rope_table_float():
    """Precompute RoPE cos/sin for positions 0..MAX_ROPE_POS-1.

    RoPE rotation: rotate pairs (x[0], x[1]) and (x[2], x[3]) by cos/sin.

    The retrain model uses per-dimension-pair frequencies (2 pairs for d=4):
        freqs = theta ** (-2*i/d) where i ∈ [0, d/2-1]
    For each position pos, each pair i has:
        cos(pos * freqs[i]), sin(pos * freqs[i])
    Stored interleaved: [cos0, sin0, cos1, sin1] per position.
    Total: 35 × 4 = 140 floats.
    """
    d = MODEL_DIM
    freq = []
    for i in range(d // 2):
        f = ROPE_THETA ** (-2 * i / d)
        freq.append(f)

    table = np.zeros((MAX_ROPE_POS, d), dtype=np.float32)
    for pos in range(MAX_ROPE_POS):
        for i, f in enumerate(freq):
            angle = pos * f
            table[pos, 2*i]     = math.cos(angle)
            table[pos, 2*i + 1] = math.sin(angle)

    return table


def build_dram():
    """Build full DRAM array with trained weights.

    Returns
    -------
    dram : np.ndarray[float32, shape=(524288,)]
    meta : dict
    """
    dram = np.zeros(524288, dtype=np.float32)
    weights = _load_weights()

    # ── embedding: [10, 4] row-major ──
    addr = ADDR_MAP['embedding']
    dram[addr:addr + 40] = weights['embedding.weight'].flatten()

    # ── RMSNorm weights ──
    dram[ADDR_MAP['norm1']:ADDR_MAP['norm1'] + 4] = weights['norm1.weight']
    dram[ADDR_MAP['norm2']:ADDR_MAP['norm2'] + 4] = weights['norm2.weight']
    dram[ADDR_MAP['norm_final']:ADDR_MAP['norm_final'] + 4] = \
        weights['norm_final.weight']

    # ── Q projection [4, 4] row-major ──
    W_q = weights['W_q.weight'].flatten()
    dram[ADDR_MAP['W_q']:ADDR_MAP['W_q'] + 16] = W_q

    # ── W_q transposed (for O=tied-Q^T, NPU MV_MUL needs MRF^T * ctx) ──
    W_q_t = weights['W_q.weight'].T.flatten()
    dram[ADDR_MAP['W_q_t']:ADDR_MAP['W_q_t'] + 16] = W_q_t

    # ── KV projection [4, 4] row-major ──
    dram[ADDR_MAP['W_kv']:ADDR_MAP['W_kv'] + 16] = \
        weights['W_kv.weight'].flatten()

    # ── QK norms ──
    dram[ADDR_MAP['q_norm']:ADDR_MAP['q_norm'] + 4] = \
        weights['q_norm.weight']
    dram[ADDR_MAP['k_norm']:ADDR_MAP['k_norm'] + 4] = \
        weights['k_norm.weight']

    # ── SwiGLU weights [4, 4] row-major ──
    dram[ADDR_MAP['W_gate']:ADDR_MAP['W_gate'] + 16] = \
        weights['W_gate.weight'].flatten()
    dram[ADDR_MAP['W_up']:ADDR_MAP['W_up'] + 16] = \
        weights['W_up.weight'].flatten()
    dram[ADDR_MAP['W_down']:ADDR_MAP['W_down'] + 16] = \
        weights['W_down.weight'].flatten()

    # ── RoPE cos/sin table ──
    rope = _rope_table_float()
    dram[ADDR_MAP['rope_table']:ADDR_MAP['rope_table'] + 140] = rope.flatten()

    # ── embed_B: all zeros (no per-digit bias) ──
    dram[ADDR_MAP['embed_b']:ADDR_MAP['embed_b'] + 4] = 0.0

    meta = {
        'model_dim': MODEL_DIM,
        'n_heads': N_HEADS,
        'head_dim': HEAD_DIM,
        'ff_dim': FF_DIM,
        'vocab_size': VOCAB_SIZE,
        'output_digits': OUTPUT_DIGITS,
        'prompt_len': PROMPT_LEN,
        'max_rope_pos': MAX_ROPE_POS,
        'rope_theta': ROPE_THETA,
        'dram_weight_bytes': ADDR_END * 4,
    }
    return dram, meta


def encode_prompt(a, b):
    """Encode a+b into the model's token sequence.

    Format: [0] + 10 LSB-first digits of A + [0,0] + 10 LSB-first digits of B + [0]
    This is 24 tokens (PROMPT_LEN).
    """
    a_digits = [int(c) for c in f"{a:010d}"[::-1]]
    b_digits = [int(c) for c in f"{b:010d}"[::-1]]
    return [0] + a_digits + [0, 0] + b_digits + [0]


def decode_output(tokens):
    """Decode autoregressive output starting at position PROMPT_LEN.

    The model autoregressively generates OUTPUT_DIGITS (11) tokens
    after the 24-token prompt. Output is LSB-first reversed digits.
    """
    out = tokens[PROMPT_LEN:PROMPT_LEN + OUTPUT_DIGITS]
    return int("".join(str(t) for t in out)[::-1])
