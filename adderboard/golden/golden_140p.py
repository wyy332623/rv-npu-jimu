"""
Pure-numpy golden reference for trained 140p Qwen3-style model.

Architecture: 1L Qwen3 decoder, d=4, 1h, hd=4, ff=4, RoPE θ=3,
              SwiGLU, RMSNorm, tied K=V, O=Q^T, LM head=embed.T.
Weights:     140 trained parameters (seed 44, FP16-safe, 99.0% accuracy).
Weight file: adderboard/models/140p/s44_targeted_final_fp16.pt

Self-verifying: assert output == a + b.

Matches the exact forward pass of retrain/retrain.py TinyAdderQwen3.
"""

import numpy as np
import math
from adderboard.layout.layout_140p import (
    dram_addr, ADDR_SIZES, ADDR_MAP,
    VOCAB_SIZE, MODEL_DIM, N_HEADS, HEAD_DIM, FF_DIM,
    PROMPT_LEN, OUTPUT_DIGITS, MAX_ROPE_POS, ROPE_THETA,
    encode_prompt, decode_output,
)


def _load_weights(dram: np.ndarray):
    """Read all weight tensors from DRAM at layout addresses."""
    w = {}

    def _get(name, shape):
        addr = dram_addr(name)
        size = ADDR_SIZES[name]
        return dram[addr:addr + size].copy().reshape(shape)

    w['embedding']  = _get('embedding', (VOCAB_SIZE, MODEL_DIM))
    w['norm1']      = _get('norm1', (MODEL_DIM,))
    w['norm2']      = _get('norm2', (MODEL_DIM,))
    w['norm_final'] = _get('norm_final', (MODEL_DIM,))
    w['W_q']        = _get('W_q', (HEAD_DIM, MODEL_DIM))
    w['W_q_t']      = _get('W_q_t', (MODEL_DIM, HEAD_DIM))  # transposed for NPU O-proj
    w['W_kv']       = _get('W_kv', (HEAD_DIM, MODEL_DIM))
    w['q_norm']     = _get('q_norm', (HEAD_DIM,))
    w['k_norm']     = _get('k_norm', (HEAD_DIM,))
    w['W_gate']     = _get('W_gate', (FF_DIM, MODEL_DIM))
    w['W_up']       = _get('W_up', (FF_DIM, MODEL_DIM))
    w['W_down']     = _get('W_down', (MODEL_DIM, FF_DIM))
    # rope_table: [MAX_ROPE_POS, 4] = [pos, dim_pair] floats
    w['rope_table'] = _get('rope_table', (MAX_ROPE_POS, HEAD_DIM))

    return w


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: x / sqrt(mean(x²) + eps) * gamma."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * gamma


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)."""
    # Stable sigmoid: use tanh formulation
    half = x / 2.0
    sig = 0.5 * (1.0 + np.tanh(half + 0.5 * np.log1p(np.exp(-np.abs(half)))))
    # Simpler: sigmoid(x) = 1 / (1 + exp(-x))
    exp_neg = np.exp(-np.abs(x))
    sig = np.where(x >= 0,
                   1.0 / (1.0 + np.exp(-x)),
                   np.exp(x) / (1.0 + np.exp(x)))
    return x * sig


def apply_rope_numpy(x: np.ndarray, rope_table: np.ndarray, positions: np.ndarray):
    """Apply RoPE rotation to precomputed cos/sin table.

    x: [T, H, head_dim] reshaped to [T, H, head_dim//2, 2]
    rope_table: [max_pos, head_dim] with [cos0, sin0, cos1, sin1] per position
    positions: [T] array of position indices
    """
    T, H, hd = x.shape
    half = hd // 2

    # Get cos/sin for positions: [T, hd] → [T, 1, hd]
    cos_sin = rope_table[positions]  # [T, hd]
    cos_sin = cos_sin.reshape(T, 1, half, 2)  # [T, 1, half, 2]
    cos_vals = cos_sin[..., 0]  # [T, 1, half]
    sin_vals = cos_sin[..., 1]  # [T, 1, half]

    # Reshape x into pairs: [T, H, half, 2]
    x_pairs = x.reshape(T, H, half, 2)
    x0 = x_pairs[..., 0]  # [T, H, half]
    x1 = x_pairs[..., 1]

    out0 = x0 * cos_vals - x1 * sin_vals
    out1 = x0 * sin_vals + x1 * cos_vals

    result = np.stack([out0, out1], axis=-1)  # [T, H, half, 2]
    return result.reshape(T, H, hd)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax (float64 for numerical accuracy)."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x.astype(np.float64) - m.astype(np.float64))
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def forward(dram: np.ndarray, tokens: list):
    """One autoregressive step. Returns logits for the last position."""
    w = _load_weights(dram)
    T = len(tokens)

    # ── Embedding ──
    x = np.zeros((T, MODEL_DIM), dtype=np.float32)
    for i in range(T):
        x[i] = w['embedding'][tokens[i]]

    # ── Attention ──
    h = rms_norm(x, w['norm1'])  # [T, d_model]

    # Q projection: h @ W_q^T → [T, hd]
    q = h @ w['W_q'].T  # [T, hd]
    q = q.reshape(T, N_HEADS, HEAD_DIM)  # [T, 1, 4]

    # KV projection (tied K=V)
    kv = h @ w['W_kv'].T  # [T, hd]
    kv = kv.reshape(T, N_HEADS, HEAD_DIM)  # [T, 1, 4]
    k = kv.copy()
    v = kv.copy()

    # QK norms
    q = rms_norm(q, w['q_norm'])
    k = rms_norm(k, w['k_norm'])

    # RoPE
    positions = np.arange(T, dtype=np.int32)
    q = apply_rope_numpy(q, w['rope_table'], positions)
    k = apply_rope_numpy(k, w['rope_table'], positions)

    # Transpose for attention: [T, H, hd] → [H, T, hd]
    q = q.transpose(1, 0, 2)  # [1, T, 4]
    k = k.transpose(1, 0, 2)
    v = v.transpose(1, 0, 2)

    # Attention scores
    scale = HEAD_DIM ** -0.5
    scores = (q @ k.transpose(0, 2, 1)) * scale  # [1, T, T]

    # Causal mask
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -1e30, scores)

    attn_w = softmax(scores, axis=-1)
    ctx = (attn_w @ v).transpose(1, 0, 2).reshape(T, HEAD_DIM)  # [T, hd]

    # O = Q^T (tied): ctx @ W_q → [T, d_model]
    # W_q is [hd, d_model], so output = ctx @ W_q (not W_q^T, since W_q row-major)
    attn_out = ctx @ w['W_q']  # [T, hd] @ [hd, d_model] = [T, d_model]

    # Residual
    x = x + attn_out

    # ── FFN (SwiGLU) ──
    h2 = rms_norm(x, w['norm2'])

    # SwiGLU: silu(W_gate(h)) * W_up(h)
    gate = h2 @ w['W_gate'].T  # [T, ff_dim]
    up = h2 @ w['W_up'].T      # [T, ff_dim]
    ffn = silu(gate) * up       # [T, ff_dim]
    ffn = ffn @ w['W_down'].T   # [T, d_model]

    x = x + ffn

    # ── Output head ──
    last_h = x[-1]                     # [d_model]
    last_h = rms_norm(last_h, w['norm_final'])
    # LM head = embedding.T (tied)
    logits = last_h @ w['embedding'].T  # [d_model] @ [d_model, vocab] = [vocab]

    return logits


def infer(dram: np.ndarray, a: int, b: int) -> int:
    """Run full autoregressive inference. Self-verifying."""
    tokens = encode_prompt(a, b)[:]  # copy

    for step in range(OUTPUT_DIGITS):
        logits = forward(dram, tokens)
        next_token = int(np.argmax(logits))
        tokens.append(next_token)

    result = decode_output(tokens)
    expected = a + b
    assert result == expected, (
        f"FAIL: {a} + {b} = {result} (expected {expected})\n"
        f"  output tokens: {tokens[PROMPT_LEN:]}\n"
        f"  logits: {logits}")

    return result
