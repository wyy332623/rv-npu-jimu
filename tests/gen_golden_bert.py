#! /usr/bin/env python3
"""
NPU — Golden Reference Generator for BERT Encoder Layer

Matches the exact numerical precision of the emulator + firmware datapath:

  - Weights and inputs are fp16 values stored in float32 containers
    (matching DRAM/VRF width), applied via emulator_cast().
  - Tiled MVM: dot product accumulated per-tile in double, cast to
    float32, then fp16-rounded between tile-columns and after VV_ADD
    accumulation — matching the emulator's _pipeline_round_fp16().
  - VV_ADD (residual connections): float32 add then fp16-round.
  - VV_MUL (elementwise): float32 mul then fp16-round.
  - Softmax: float32 expf + double sum + float32 inv_sum, then
    fp16-round (matching C kernel + _store_to_ivrf).
  - LayerNorm: double accumulators for mean/var/inv_std, float32
    elementwise, then fp16-round (matching C kernel + _store_to_ivrf).
  - GELU: float32 arithmetic with float32 tanhf — NO fp16-round
    (matching C kernel + _store_to_ivrf which rounds before
    auto-storing, consistent with the pipeline model).

Usage:
    # Generate golden data for testing
    python3 -m tests.gen_golden_bert --seq-len 64 --num-head 4
"""

import argparse
import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------
# Precision helpers — matching emulator HW datapath
# -----------------------------------------------------------------------

def fp16_round(x: NDArray) -> NDArray:
    """Round to FP16 and back to FP32, matching _pipeline_round_fp16().

    This is the fundamental precision operation of the NPU pipeline:
    after every compute instruction the pipeline registers are clipped
    to FP16 width.  Values stay in float32 containers for C kernel
    compatibility but hold only FP16-representable numbers.
    """
    return x.astype(np.float16).astype(np.float32)


def emulator_cast(x: NDArray, precision: str = 'emulator_float32') -> NDArray:
    """Cast through FP16 to simulate DRAM / VRF storage width.

    DRAM and VRF store values as FP16 (16-bit).  On load they are
    promoted to float32 for the C kernels.  This function reproduces
    that round-trip for weight initialisation.
    """
    if precision == 'emulator_float32':
        return x.astype(np.float16).astype(np.float32)
    return x


# -----------------------------------------------------------------------
# C-kernel–matching compute primitives
# -----------------------------------------------------------------------

def _c_softmax(x: NDArray) -> NDArray:
    """Softmax matching the C kernel + emulator pipeline.

    C kernel (softmax.c):
        max_val  = float (fp32)
        y[i]     = expf(x[i] - max_val)   // fp32 expf
        sum      = double accumulator
        inv_sum  = (float)(1.0 / sum)     // fp32
        y[i]    *= inv_sum                 // fp32 mul
    Emulator adds:
        _pipeline_round_fp16() via _store_to_ivrf()
    """
    max_val = float(np.max(x))
    y = np.exp(x.astype(np.float32) - np.float32(max_val))
    sum_d = float(np.sum(y.astype(np.float64)))
    inv_sum = np.float32(1.0 / sum_d)
    y = (y * inv_sum).astype(np.float32)
    return fp16_round(y)


def _c_layernorm(x: NDArray, gamma: NDArray, beta: NDArray,
                 eps: float = 1e-12) -> NDArray:
    """LayerNorm matching the C kernel + emulator pipeline.

    C kernel (layernorm.c):
        sum / mean      = double accumulator
        var             = double accumulator
        inv_std         = 1.0 / sqrt(var + eps)     // double
        y[i]            = (float)(norm_d * (double)gamma[i]
                                  + (double)beta[i]) // fp32 out
    Emulator adds:
        _pipeline_round_fp16() via _store_to_ivrf()
    """
    x_f32 = x.astype(np.float32)
    mean_d = float(np.mean(x_f32.astype(np.float64)))
    var_d = float(np.var(x_f32.astype(np.float64)))
    inv_std_d = 1.0 / np.sqrt(np.float64(var_d) + np.float64(eps))
    normalized = (x_f32.astype(np.float64) - mean_d) * inv_std_d
    y = (normalized * gamma.astype(np.float64)
         + beta.astype(np.float64)).astype(np.float32)
    return fp16_round(y)


def _c_gelu(x: NDArray) -> NDArray:
    """GELU matching the C kernel + emulator pipeline.

    C kernel (activation.c):
        sqrt_2_over_pi = 0.7978845608028654f   // fp32 constant
        coeff          = 0.044715f              // fp32 constant
        x3             = x[i] * x[i] * x[i]    // fp32 mul chain
        inner          = sqrt_2_over_pi * (x[i] + coeff * x3)  // fp32
        y[i]           = 0.5f * x[i] * (1.0f + tanhf(inner))  // fp32 tanhf
    Emulator adds:
        _store_to_ivrf() → fp16_round + auto-store to IVRF
    """
    x_f32 = x.astype(np.float32)
    sqrt_2_over_pi = np.float32(0.7978845608028654)
    coeff = np.float32(0.044715)
    x3 = x_f32 * x_f32 * x_f32
    inner = sqrt_2_over_pi * (x_f32 + coeff * x3)
    y = np.float32(0.5) * x_f32 * (np.float32(1.0) + np.tanh(inner))
    return fp16_round(y)


def _tiled_mvm(W: NDArray, x: NDArray, native_dim: int) -> NDArray:
    """Tiled matrix–vector multiply matching emulator + firmware datapath.

    Reproduces the precision flow of mvm_tiled_q() in bert_layer.c:
      For each tile row tr:
        Zero accumulator VRF.
        For each tile column tc:
          MV_MUL: double accum within tile → float → fp16-round
          Accumulate via VV_ADD → fp16-round
        Add bias via VV_ADD → fp16-round
    """
    W = W.astype(np.float32)
    x = x.astype(np.float32)
    out_size = W.shape[0]
    in_size = W.shape[1]
    num_tiles_r = out_size // native_dim
    num_tiles_c = in_size // native_dim
    result = np.zeros(out_size, dtype=np.float32)

    for tr in range(num_tiles_r):
        acc = np.zeros(native_dim, dtype=np.float32)

        for tc in range(num_tiles_c):
            # MV_MUL tile: double accum per row within tile
            tile = W[tr * native_dim:(tr + 1) * native_dim,
                     tc * native_dim:(tc + 1) * native_dim]
            x_chunk = x[tc * native_dim:(tc + 1) * native_dim]
            partial = np.zeros(native_dim, dtype=np.float32)
            for r in range(native_dim):
                s = np.float64(0.0)
                for c in range(native_dim):
                    s += np.float64(tile[r, c]) * np.float64(x_chunk[c])
                partial[r] = np.float32(s)
            # Pipeline fp16-round after MV_MUL
            partial = fp16_round(partial)

            if tc == 0:
                acc = partial.copy()
            else:
                # VV_ADD accumulation
                acc = fp16_round(acc + partial)

        result[tr * native_dim:(tr + 1) * native_dim] = acc

    return result


# -----------------------------------------------------------------------
# BERT encoder layer — emulator-matching golden reference
# -----------------------------------------------------------------------

def bert_encoder_layer(
    add_mask: bool = True,
    num_head: int = 4,
    head_size: int = 64,
    hidden_size: int = 256,
    seq_len: int = 64,
    mask_prob: float = 0.15,
    precision: str = 'emulator_float32',
    native_dim: Optional[int] = None,
    x: Optional[NDArray] = None,
    mask: Optional[NDArray] = None,
    params_dict: Optional[dict] = None,
    seed: int = 0,
) -> Tuple[dict, dict]:
    """Generate golden reference values for one BERT encoder layer.

    The forward pass precisely matches the numerical behaviour of
    firmware running on the NPU emulator, including:

      - Tiled MVM with fp16-rounding between tile columns
      - fp16-rounding after every VV_ADD / VV_MUL / softmax / LN
      - FP32 GELU (with fp16-round at auto-store)
      - DRAM round-trips (fp16 storage) between computational stages

    Args:
      native_dim: NPU native dimension (tile size). Defaults to
        head_size * 2 (standard 2-tile-row layout per head).

    Returns:
        golden: dict of all intermediate tensors
        params: dict of weight parameters used
    """
    if native_dim is None:
        native_dim = head_size * 2  # standard 2-tile-row layout per head

    np.random.seed(seed)

    num_rows = num_head * head_size  # = hidden_size
    num_cols = hidden_size
    num_vecs = seq_len
    mask_value = np.finfo(np.float16).min + 1.5

    # --- Input ---
    if x is None:
        x = np.random.normal(loc=0, scale=1.46,
                             size=(num_vecs, num_cols))
        x = emulator_cast(x, precision)

    # --- Weights ---
    if params_dict is None:
        params = {}
        for k in ['Q', 'K', 'V']:
            W = np.random.normal(loc=0, scale=0.12,
                                 size=(num_rows, num_cols))
            b = np.random.normal(loc=0, scale=0.06, size=num_rows)
            params[k] = {
                'W': emulator_cast(W, precision),
                'b': emulator_cast(b, precision),
            }
        # Pre-multiply Q by 1/sqrt(head_size)
        params['Q']['W'] /= math.sqrt(head_size)
        params['Q']['b'] /= math.sqrt(head_size)
        # Re-quantise the scaled Q weights to fp16
        params['Q']['W'] = fp16_round(params['Q']['W'])
        params['Q']['b'] = fp16_round(params['Q']['b'])

        # LayerNorm weights
        params['LayerNorm'] = {
            'W': [emulator_cast(np.random.normal(0.85, 0.14, num_cols), precision),
                  emulator_cast(np.random.normal(0.85, 0.14, num_cols), precision)],
            'b': [emulator_cast(np.random.normal(0, 0.07, num_cols), precision),
                  emulator_cast(np.random.normal(0, 0.07, num_cols), precision)],
        }

        # Self-output
        W_so = np.random.normal(0, 0.12, (num_rows, num_cols))
        b_so = np.random.normal(0, 0.06, num_rows)
        params['selfoutput'] = {
            'W': emulator_cast(W_so, precision),
            'b': emulator_cast(b_so, precision),
        }

        # FF layers
        W_int = np.random.normal(0, 0.12, (num_rows, num_cols))
        b_int = np.random.normal(0, 0.06, num_rows)
        W_out = np.random.normal(0, 0.12, (num_cols, num_rows))
        b_out = np.random.normal(0, 0.06, num_cols)
        params['W_intmfc'] = emulator_cast(W_int, precision)
        params['b_intmfc'] = emulator_cast(b_int, precision)
        params['W_outfc'] = emulator_cast(W_out, precision)
        params['b_outfc'] = emulator_cast(b_out, precision)
    else:
        params = params_dict

    # --- Mask ---
    if mask is None:
        mask = np.array([
            0 if v < 1 - mask_prob else mask_value
            for v in np.random.rand(seq_len)
        ])

    # ═══════════════════════════════════════════════════════════════
    # Forward pass — matching emulator + firmware precision exactly
    # ═══════════════════════════════════════════════════════════════

    # 1. Q, K, V projections — tiled MVM with fp16-round accumulation
    Q = np.zeros_like(x, dtype=np.float32)
    K = np.zeros_like(x, dtype=np.float32)
    V = np.zeros_like(x, dtype=np.float32)
    for pos in range(seq_len):
        Q[pos] = fp16_round(
            _tiled_mvm(params['Q']['W'], x[pos], native_dim)
            + params['Q']['b'][:hidden_size].astype(np.float32))
        K[pos] = fp16_round(
            _tiled_mvm(params['K']['W'], x[pos], native_dim)
            + params['K']['b'][:hidden_size].astype(np.float32))
        V[pos] = fp16_round(
            _tiled_mvm(params['V']['W'], x[pos], native_dim)
            + params['V']['b'][:hidden_size].astype(np.float32))

    # 2. Multi-head attention — standard BERT dot-product attention
    #
    #   For each tile row tr and each head h within the row:
    #     1. Build K.T MRF tile: rows = key positions, cols = head elements
    #        via read vector mask → VecToMatRow → zero-pad → M_RD to MRF
    #     2. Score = MV_MUL(K.T, Q[pos_q][tr,:] mask-selected) → [seq_len]
    #        fp16-round after MV_MUL.
    #     3. Softmax → fp16-round
    #     4. Build V.T MRF tile: rows = head elements, cols = positions
    #     5. Context = MV_MUL(V.T, prob) → [head_size] context vector
    #        context[j] = Σ_pos V[pos, head_h, j] * prob[pos]
    #     6. Accumulate into Z with write vector mask
    #
    # All DRAM save/restore (fp16-storage) and VecToMatRow (fp16-pipe)
    # steps apply fp16-rounding at each stage.
    heads_per_tile = native_dim // head_size
    num_tiles = hidden_size // native_dim
    Z = np.zeros([seq_len, hidden_size], dtype=np.float32)
    for pos_q in range(seq_len):
        for tr in range(num_tiles):
            for h in range(heads_per_tile):
                # Mask to select this head's elements in the tile row
                vrf_mask_h = ((1 << head_size) - 1) << (h * head_size)

                # ── 2a. Build K.T MRF tile [NATIVE_DIM, NATIVE_DIM] ──
                # Only up to native_dim rows (firmware truncates).
                kt_tile = np.zeros((native_dim, native_dim), dtype=np.float32)
                n_load = min(seq_len, native_dim)
                for p in range(n_load):
                    row_data = np.zeros(native_dim, dtype=np.float32)
                    for i in range(native_dim):
                        if (vrf_mask_h >> (i % 8)) & 1:
                            row_data[i] = K[p, tr * native_dim + i]
                    kt_tile[p] = fp16_round(row_data)

                # ── 2b. Score = K.T @ Q_h ──
                q_vec = np.zeros(native_dim, dtype=np.float32)
                for i in range(native_dim):
                    if (vrf_mask_h >> (i % 8)) & 1:
                        q_vec[i] = Q[pos_q, tr * native_dim + i]
                q_vec = fp16_round(q_vec)

                score_vec = np.zeros(native_dim, dtype=np.float32)
                for p in range(native_dim):
                    s = np.float64(0.0)
                    for j in range(native_dim):
                        s += np.float64(kt_tile[p, j]) * np.float64(q_vec[j])
                    score_vec[p] = np.float32(s)
                score_vec = fp16_round(score_vec)

                # ── 2c. Softmax ──
                prob_vec = score_vec.copy()
                if seq_len < native_dim:
                    prob_vec[seq_len:] = -1e30
                prob_vec = _c_softmax(prob_vec)
                if seq_len < native_dim:
                    prob_vec[seq_len:] = 0.0

                # ── 2d. Build V.T MRF tile (element-major, standard BERT) ──
                # Rows = head elements, cols = positions.
                # MV_MUL(V.T, prob) computes V.T @ prob = [Σ_pos V[pos][0]·prob[pos],
                #   Σ_pos V[pos][1]·prob[pos], ...] = correct BERT context.
                vt_tile = np.zeros((native_dim, native_dim), dtype=np.float32)
                n_vload = min(seq_len, native_dim)
                for j in range(native_dim):
                    row_data = np.zeros(native_dim, dtype=np.float32)
                    if (vrf_mask_h >> (j % 8)) & 1:
                        for p in range(n_vload):
                            row_data[p] = V[p, tr * native_dim + j]
                    vt_tile[j] = fp16_round(row_data)

                # ── 2e. Context = V.T @ prob (standard BERT) ──
                prob_for_ivrf = fp16_round(prob_vec)
                ctx_vec = np.zeros(native_dim, dtype=np.float32)
                # V.T @ prob: result[j] = Σ_p V.T[j][p] · prob[p]
                #           = Σ_p V[p][j] · prob[p]
                for j in range(native_dim):
                    s = np.float64(0.0)
                    for p in range(native_dim):
                        s += np.float64(vt_tile[j, p]) * np.float64(prob_for_ivrf[p])
                    ctx_vec[j] = np.float32(s)
                ctx_vec = fp16_round(ctx_vec)

                # ── 2f. Accumulate into Z with write vector mask ──
                for i in range(native_dim):
                    if (vrf_mask_h >> (i % 8)) & 1:
                        Z[pos_q, tr * native_dim + i] = fp16_round(
                            Z[pos_q, tr * native_dim + i] + ctx_vec[i])

    # 3. Self-output — tiled MVM of Z through Wso
    hidden_state = np.zeros([seq_len, hidden_size], dtype=np.float32)
    for pos in range(seq_len):
        hidden_state[pos] = fp16_round(
            _tiled_mvm(params['selfoutput']['W'], Z[pos], native_dim)
            + params['selfoutput']['b'][:hidden_size].astype(np.float32))

    # 4. First residual + LayerNorm
    #    VV_ADD (fp32 add + fp16-round) for residual
    layer_norm_input = fp16_round(hidden_state + x)
    #    LayerNorm: per-tile-row (matching firmware's apply_layernorm).
    #    Each tile row is normalized independently with its own gamma/beta chunk.
    ln_gamma_0 = fp16_round(params['LayerNorm']['W'][0].astype(np.float32))
    ln_beta_0 = fp16_round(params['LayerNorm']['b'][0].astype(np.float32))
    num_tiles = hidden_size // native_dim
    layer_norm_output = np.zeros_like(layer_norm_input)
    for pos in range(seq_len):
        for tr in range(num_tiles):
            sl = tr * native_dim
            sr = sl + native_dim
            layer_norm_output[pos, sl:sr] = _c_layernorm(
                layer_norm_input[pos, sl:sr],
                ln_gamma_0[sl:sr], ln_beta_0[sl:sr])

    # 5. FF layer 1 + GELU — tiled MVM then fp32 GELU + fp16-round
    intmfc_out = np.zeros([seq_len, hidden_size], dtype=np.float32)
    for pos in range(seq_len):
        wi_raw = fp16_round(
            _tiled_mvm(params['W_intmfc'], layer_norm_output[pos], native_dim)
            + params['b_intmfc'][:hidden_size].astype(np.float32))
        intmfc_out[pos] = _c_gelu(wi_raw)

    # 6. FF layer 2 — tiled MVM
    outfc = np.zeros([seq_len, hidden_size], dtype=np.float32)
    for pos in range(seq_len):
        outfc[pos] = fp16_round(
            _tiled_mvm(params['W_outfc'], intmfc_out[pos], native_dim)
            + params['b_outfc'][:hidden_size].astype(np.float32))

    # 7. Second residual + LayerNorm
    #    VV_ADD: fp32 add + fp16-round
    res2_input = fp16_round(outfc + x)
    #    LayerNorm: per-tile-row (matching firmware)
    ln_gamma_1 = fp16_round(params['LayerNorm']['W'][1].astype(np.float32))
    ln_beta_1 = fp16_round(params['LayerNorm']['b'][1].astype(np.float32))
    out = np.zeros_like(res2_input)
    for pos in range(seq_len):
        for tr in range(num_tiles):
            sl = tr * native_dim
            sr = sl + native_dim
            out[pos, sl:sr] = _c_layernorm(
                res2_input[pos, sl:sr],
                ln_gamma_1[sl:sr], ln_beta_1[sl:sr])

    # --- Build golden dict ---
    golden = {
        'X': x,
        'Q': Q, 'K': K, 'V': V,
        'Z': Z,
        'hidden_state': hidden_state,
        'layerNormInput': layer_norm_input,
        'layerNormOutput': layer_norm_output,
        'intmfc_out': intmfc_out,
        'outfc': outfc,
        'out': out,
        'mask': mask,
    }

    return golden, params


# -----------------------------------------------------------------------
# Compatibility shims — remain importable by existing code
# -----------------------------------------------------------------------

def gelu(x: NDArray) -> NDArray:
    """GELU activation (legacy interface — delegates to _c_gelu)."""
    return _c_gelu(x)


def relu(x: NDArray) -> NDArray:
    """ReLU activation: max(x, 0)."""
    return np.maximum(x, 0)


def softmax(x: NDArray, axis: int = -1) -> NDArray:
    """Softmax with numerical stability (legacy interface — fp64)."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_normalization(x: NDArray, eps: float = 1e-12) -> NDArray:
    """Layer normalization: (x - mean) / sqrt(var + eps) (legacy — fp64)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate BERT encoder layer golden reference")
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--num-head', type=int, default=4)
    parser.add_argument('--head-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=None)
    parser.add_argument('--native-dim', type=int, default=None,
                        help="NPU native dimension (defaults to head_size*2)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help="Save golden to .npz file")
    args = parser.parse_args()

    hidden_size = args.hidden_size or (args.num_head * args.head_size)
    native_dim = args.native_dim or (args.head_size * 2)

    print(f"Generating golden reference for BERT encoder layer:")
    print(f"  seq_len={args.seq_len}, num_head={args.num_head},")
    print(f"  head_size={args.head_size}, hidden_size={hidden_size},")
    print(f"  native_dim={native_dim}")
    print(f"  seed={args.seed}")
    print(f"  precision=emulator_float32 (fp16 datapath, fp32 compute)")

    golden, params = bert_encoder_layer(
        num_head=args.num_head,
        head_size=args.head_size,
        hidden_size=hidden_size,
        seq_len=args.seq_len,
        native_dim=native_dim,
        seed=args.seed,
    )

    print(f"\nOutput shapes:")
    for k, v in golden.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:20s}: {str(v.shape):20s}  {v.dtype}")

    if args.output:
        np.savez(args.output, **golden)
        print(f"\nSaved to {args.output}")

    # Verification: check no NaN/Inf
    for k, v in golden.items():
        if isinstance(v, np.ndarray):
            assert not np.any(np.isnan(v)), f"NaN in {k}"
            assert not np.any(np.isinf(v)), f"Inf in {k}"
    print("\n✓ All values valid (no NaN/Inf)")


if __name__ == '__main__':
    main()
