/* NPU — BERT Encoder Layer (End-to-End)
 *
 * HLS seed: This file is the algorithmic specification for future
 * HLS synthesis.  It is decomposed into the same primitive ops
 * (mv_mul, gelu, softmax, layernorm, vec_add, vec_mul) used by
 * the NPU hardware, and validated by the progressive test chain:
 *
 *   Round 0:    numpy golden reference
 *   Round 0.5:  this C implementation (HLS seed) ← you are here
 *   Round 1:    firmware + ISS-driven emulation
 *   Rounds 2-3: Amaranth HDL simulation
 *
 * All rounds must produce identical output on identical inputs.
 */

#include "npu_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int bert_encoder_layer(
    const float* input,        /* [seq_len, hidden_size] */
    int seq_len,
    int hidden_size,
    int num_head,

    const float* Wq, const float* bq,
    const float* Wk, const float* bk,
    const float* Wv, const float* bv,
    const float* W_selfout, const float* b_selfout,
    const float* W_intmfc, const float* b_intmfc,
    const float* W_outfc, const float* b_outfc,
    const float* ln1_gamma, const float* ln1_beta,
    const float* ln2_gamma, const float* ln2_beta,

    float* output               /* [seq_len, hidden_size] */
)
{
    int head_size = hidden_size / num_head;

    /* Allocate temporary buffers (4x: Z, K_full, V_full, attn_out) */
    int buf_size = 4 * seq_len * hidden_size;
    float* temp = (float*)malloc(buf_size * sizeof(float));
    float* q_tmp = (float*)malloc(hidden_size * sizeof(float));
    float* k_tmp = (float*)malloc(hidden_size * sizeof(float));
    float* v_tmp = (float*)malloc(hidden_size * sizeof(float));
    float* attn_score = (float*)malloc(seq_len * sizeof(float));
    float* attn_prob = (float*)malloc(seq_len * sizeof(float));

    if (!temp || !q_tmp || !k_tmp || !v_tmp || !attn_score || !attn_prob) {
        free(temp); free(q_tmp); free(k_tmp); free(v_tmp);
        free(attn_score); free(attn_prob);
        return -1;
    }

    float* Z = temp;                          /* Q storage */
    float* K_full = temp + seq_len * hidden_size;  /* K storage */
    float* V_full = temp + 2 * seq_len * hidden_size;  /* V storage */
    float* attn_out = temp + 3 * seq_len * hidden_size;  /* Attention output (separate from K!) */

    /* --- Q projection --- */
    for (int t = 0; t < seq_len; t++) {
        mv_mul(Wq, input + t * hidden_size, Z + t * hidden_size,
               hidden_size, hidden_size, 0);
        vec_add(Z + t * hidden_size, bq, Z + t * hidden_size, hidden_size);
    }

    /* --- K projection --- */
    for (int t = 0; t < seq_len; t++) {
        mv_mul(Wk, input + t * hidden_size, K_full + t * hidden_size,
               hidden_size, hidden_size, 0);
        vec_add(K_full + t * hidden_size, bk, K_full + t * hidden_size, hidden_size);
    }

    /* --- V projection --- */
    for (int t = 0; t < seq_len; t++) {
        mv_mul(Wv, input + t * hidden_size, V_full + t * hidden_size,
               hidden_size, hidden_size, 0);
        vec_add(V_full + t * hidden_size, bv, V_full + t * hidden_size, hidden_size);
    }

    /* --- Multi-head attention --- */
    memset(attn_out, 0, seq_len * hidden_size * sizeof(float));
    /* Note: attn_out is a SEPARATE buffer from K_full. K_full must be
       preserved because the attention score computation reads from it. */

    for (int h = 0; h < num_head; h++) {
        int head_offset = h * head_size;

        for (int t = 0; t < seq_len; t++) {
            /* Compute attention score for query t against all keys */
            for (int s = 0; s < seq_len; s++) {
                float score = 0;
                for (int d = 0; d < head_size; d++) {
                    score += Z[t * hidden_size + head_offset + d] *
                             K_full[s * hidden_size + head_offset + d];
                }
                attn_score[s] = score;
            }

            /* Softmax over keys */
            softmax(attn_score, attn_prob, seq_len);

            /* Weighted sum of values */
            for (int d = 0; d < head_size; d++) {
                float ctx = 0;
                for (int s = 0; s < seq_len; s++) {
                    ctx += attn_prob[s] *
                           V_full[s * hidden_size + head_offset + d];
                }
                attn_out[t * hidden_size + head_offset + d] = ctx;
            }
        }
    }

    /* --- Self-output: attn_out @ W_selfout^T + b_selfout --- */
    /* Reuse Z buffer for self_output */
    float* self_out = Z;
    for (int t = 0; t < seq_len; t++) {
        mv_mul(W_selfout, attn_out + t * hidden_size,
               self_out + t * hidden_size, hidden_size, hidden_size, 0);
        vec_add(self_out + t * hidden_size, b_selfout,
                self_out + t * hidden_size, hidden_size);
    }

    /* --- Residual + LayerNorm 1 --- */
    /* Reuse V_full buffer for ln output */
    float* ln1_out = V_full;
    for (int t = 0; t < seq_len; t++) {
        vec_add(self_out + t * hidden_size, input + t * hidden_size,
                ln1_out + t * hidden_size, hidden_size);
        layernorm(ln1_out + t * hidden_size, ln1_gamma, ln1_beta,
                  ln1_out + t * hidden_size, hidden_size, 1e-12f);
    }

    /* --- FFN Layer 1: GELU(W_intmfc @ x + b_intmfc) --- */
    float* ff1_out = self_out;  /* reuse Z buffer */
    for (int t = 0; t < seq_len; t++) {
        mv_mul(W_intmfc, ln1_out + t * hidden_size,
               ff1_out + t * hidden_size, hidden_size, hidden_size, 0);
        vec_add(ff1_out + t * hidden_size, b_intmfc,
                ff1_out + t * hidden_size, hidden_size);
        gelu(ff1_out + t * hidden_size, ff1_out + t * hidden_size, hidden_size);
    }

    /* --- FFN Layer 2: W_outfc @ x + b_outfc --- */
    float* ff2_out = ln1_out;  /* reuse V_full buffer */
    for (int t = 0; t < seq_len; t++) {
        mv_mul(W_outfc, ff1_out + t * hidden_size,
               ff2_out + t * hidden_size, hidden_size, hidden_size, 0);
        vec_add(ff2_out + t * hidden_size, b_outfc,
                ff2_out + t * hidden_size, hidden_size);
    }

    /* --- Residual + LayerNorm 2 --- */
    for (int t = 0; t < seq_len; t++) {
        vec_add(ff2_out + t * hidden_size, input + t * hidden_size,
                output + t * hidden_size, hidden_size);
        layernorm(output + t * hidden_size, ln2_gamma, ln2_beta,
                  output + t * hidden_size, hidden_size, 1e-12f);
    }

    free(temp); free(q_tmp); free(k_tmp); free(v_tmp);
    free(attn_score); free(attn_prob);
    return 0;
}
