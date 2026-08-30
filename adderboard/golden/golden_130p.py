"""
Pure-numpy golden reference for cosminscn_130p.

Matches the exact arithmetic of the verified PyTorch model (10926/10926 tests).
Self-verifying: asserts output == a + b.

Weights are read from DRAM layout at addresses from npu_dram_layout.
"""

import numpy as np
from adderboard.layout.layout_130p import (
    build_dram, encode_prompt, decode_output, dram_addr,
    VOCAB_SIZE, MODEL_DIM, N_HEADS, HEAD_DIM, FF_DIM, PROMPT_LEN, OUTPUT_DIGITS,
)


def _load_weights(dram: np.ndarray):
    """Read named weight tensors from DRAM at layout addresses."""
    # Define all weights with their addresses and shapes
    # Format: (name, addr, count, shape)
    _layout = [
        ('pe_table',  'pe_table',  None,      None),  # special: variable length
        ('embed_A',   'embed_A',   VOCAB_SIZE, (VOCAB_SIZE,)),
        ('embed_B',   'embed_B',   MODEL_DIM,  (MODEL_DIM,)),
        ('c_attn',    'c_attn',    12 * MODEL_DIM, (12, MODEL_DIM)),
        ('c_proj',    'c_proj',    MODEL_DIM * MODEL_DIM, (MODEL_DIM, MODEL_DIM)),
        ('c_fc',      'c_fc',      FF_DIM * MODEL_DIM, (FF_DIM, MODEL_DIM)),
        ('c_fc_bias', 'c_fc_bias', FF_DIM,     (FF_DIM,)),
        ('c_proj_u',  'c_proj_u',  MODEL_DIM,  (MODEL_DIM,)),
        ('c_proj_v',  'c_proj_v',  MODEL_DIM,  (MODEL_DIM,)),
        ('lm_u',      'lm_u',      VOCAB_SIZE, (VOCAB_SIZE,)),
        ('lm_v',      'lm_v',      MODEL_DIM,  (MODEL_DIM,)),
        ('lm_bias',   'lm_bias',   VOCAB_SIZE, (VOCAB_SIZE,)),
    ]

    w = {}
    for name, key, count, shape in _layout:
        if name == 'pe_table':
            # Find PE table length by scanning for zeros
            paddr = dram_addr('pe_table')
            max_pe = (len(dram) - paddr) // MODEL_DIM
            for n in range(min(max_pe, 200), 0, -1):
                chunk = dram[paddr + (n-1) * MODEL_DIM:paddr + n * MODEL_DIM]
                if np.any(np.abs(chunk) > 1e-10):
                    break
            addr, cnt = paddr, n * MODEL_DIM
            w['pe_table'] = dram[addr:addr + cnt].copy().reshape(n, MODEL_DIM)
        else:
            addr = dram_addr(key)
            arr = dram[addr:addr + count].copy()
            w[name] = arr.reshape(shape)
    return w


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax matching NPU SLU behavior (float64 sum)."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x.astype(np.float64) - m.astype(np.float64))
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def forward(dram: np.ndarray, tokens: list):
    """One forward step. Returns logits for the last position.
    
    Matches the verified PyTorch model's forward() exactly.
    """
    w = _load_weights(dram)
    T = len(tokens)

    # Phase 1: Embedding + PE
    embed_out = np.zeros((T, MODEL_DIM), dtype=np.float32)
    for i in range(T):
        v = tokens[i]
        embed_out[i] = w['embed_A'][v] * w['embed_B']
    x = embed_out + w['pe_table'][:T]

    # Phase 2: Attention
    # c_attn(x) = x @ W.T  where W is [12, 4]
    qkv = x @ w['c_attn'].T  # [T, 12]

    q = qkv[:, 0:4]
    k = qkv[:, 4:8]
    v = qkv[:, 8:12]

    q = q.reshape(T, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = k.reshape(T, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    v = v.reshape(T, N_HEADS, HEAD_DIM).transpose(1, 0, 2)

    scale = HEAD_DIM ** -0.5
    scores = (q @ k.transpose(0, 2, 1)) * scale
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -1e30, scores)

    attn_w = softmax(scores, axis=-1)
    context = (attn_w @ v).transpose(1, 0, 2).reshape(T, -1)

    # c_proj output projection (sparse)
    attn_out = context @ w['c_proj'].T  # [T, 4] @ [4, 4].T
    x = x + attn_out

    # Phase 3: MLP
    h = x @ w['c_fc'].T + w['c_fc_bias']  # [T, 4] @ [4, 4].T
    h = np.maximum(h, 0.0)

    # Rank1: tmp = h · v, out = tmp · u
    # v: [4], u: [4]; h: [T, 4]
    tmp = h @ w['c_proj_v']  # [T] (dot with v)
    mlp_out = np.outer(tmp, w['c_proj_u'])  # [T, 4]
    x = x + mlp_out

    # Phase 4: LM Head (Rank1 + bias)
    last_h = x[-1, :]  # [4]
    proj = np.dot(last_h, w['lm_v'])   # scalar
    logits = proj * w['lm_u'] + w['lm_bias']
    return logits


def infer(dram: np.ndarray, a: int, b: int) -> int:
    """Run full autoregressive inference. Self-verifying."""
    tokens = encode_prompt(a, b)

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
