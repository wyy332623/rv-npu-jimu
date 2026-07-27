/* NPU — C Compute Kernel Library

   Pure C compute kernels with no external dependencies.
   Used by:
     - PySpike MMIO device (via ctypes) for fast emulation
     - Test infrastructure (for kernel-level verification)
     - HLS seed: bert_encoder_layer() serves as the algorithmic
       specification for future HLS synthesis (FPGA / ASIC).
       It is decomposed into the same primitive ops as the NPU
       hardware and validated by the progressive test chain:
         numpy golden → C reference → firmware → Amaranth HDL

   All functions operate on flat float arrays in row-major order.
*/

#ifndef NPU_KERNELS_H
#define NPU_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Block Floating Point (BFP) conversion
 * ----------------------------------------------------------------------- */

/* Convert FP32 vector to BFP format.
 *   input:  length FP32 values
 *   output: length BFP mantissas (packed into int32_t)
 *   exponents: per-block exponents
 *   length: number of elements
 *   elems_per_exp: number of elements sharing one exponent
 *   frac_bits: number of fractional bits in mantissa
 */
void float_to_bfp(const float* input, int32_t* output, int32_t* exponents,
                  int length, int elems_per_exp, int frac_bits);

/* Convert BFP to FP32 vector. */
void bfp_to_float(const int32_t* input, const int32_t* exponents,
                  float* output, int length, int elems_per_exp, int frac_bits);

/* -----------------------------------------------------------------------
 * HDL-format BFP (matching hdl/bfp/bfp.py)
 * ----------------------------------------------------------------------- */

/* Encode FP16 vector to HDL-format BFP.
 *   input:  lanes FP16 values (as uint16_t[])
 *   bfp_out: per-lane 6-bit BFP mantissa (uint8_t[lanes])
 *   bfp_exp_out: shared 5-bit exponent
 *   lanes: number of elements (1..32)
 *
 * Matching FloatToBfp in hdl/bfp/bfp.py with bfp_data_width=6,
 * bfp_exp_width=5.  Used by integration test (precision_mode=1).
 */
void float16_bfp_encode(const uint16_t* input, uint8_t* bfp_out,
                         int* bfp_exp_out, int lanes);

/* Decode HDL-format BFP to FP16 vector.
 *   bfp_in: per-lane 6-bit BFP mantissa (uint8_t[lanes])
 *   bfp_exp: shared 5-bit exponent
 *   output: lanes FP16 values (as uint16_t[])
 *   lanes: number of elements (1..32)
 *
 * Matching BfpToFloat in hdl/bfp/bfp.py.
 */
void float16_bfp_decode(const uint8_t* bfp_in, int bfp_exp,
                          uint16_t* output, int lanes);

/* -----------------------------------------------------------------------
 * Matrix-Vector Multiply
 * ----------------------------------------------------------------------- */

/* y = A * x
 *   A: rows × cols matrix (row-major)
 *   x: cols-length vector
 *   y: rows-length result
 *   accum: accumulate into y if non-zero
 */
void mv_mul(const float* A, const float* x, float* y,
            int rows, int cols, int accum);

/* BFP-precise matrix-vector multiply.
 * Internal computation uses BFP accumulation matching NPU behavior.
 */
void mv_mul_bfp(const int32_t* A_bfp, const int32_t* A_exp,
                const int32_t* x_bfp, const int32_t* x_exp,
                int32_t* y_bfp, int32_t* y_exp,
                int rows, int cols, int elems_per_exp, int frac_bits);

/* -----------------------------------------------------------------------
 * Activation Functions
 * ----------------------------------------------------------------------- */

/* GELU: y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3))) */
void gelu(const float* x, float* y, int n);

/* ReLU: y[i] = max(x[i], 0) */
void relu(const float* x, float* y, int n);

/* Sigmoid: y[i] = 1 / (1 + exp(-x[i])) */
void sigmoid(const float* x, float* y, int n);

/* Tanh: y[i] = tanh(x[i]) */
void tanh_vec(const float* x, float* y, int n);

/* -----------------------------------------------------------------------
 * Softmax
 * ----------------------------------------------------------------------- */

/* Softmax: y[i] = exp(x[i]) / sum(exp(x))
 *   Computes max subtraction for numerical stability.
 */
void softmax(const float* x, float* y, int n);

/* Softmax with 2D mask (for attention). */
void softmax_masked(const float* x, const float* mask,
                    float* y, int rows, int cols);

/* -----------------------------------------------------------------------
 * Layer Normalization
 * ----------------------------------------------------------------------- */

/* y = gamma * (x - mean) / sqrt(var + eps) + beta */
void layernorm(const float* x, const float* gamma, const float* beta,
               float* y, int n, float eps);

/* -----------------------------------------------------------------------
 * Element-wise Operations
 * ----------------------------------------------------------------------- */

void vec_add(const float* a, const float* b, float* y, int n);
void vec_sub(const float* a, const float* b, float* y, int n);
void vec_mul(const float* a, const float* b, float* y, int n);
void vec_max(const float* a, const float* b, float* y, int n);
void vec_scale(float* x, float s, int n);

/* -----------------------------------------------------------------------
 * BERT Encoder Layer (end-to-end)
 * ----------------------------------------------------------------------- */

/* Run one BERT encoder layer.
 * All pointers reference flat arrays in weight-major order.
 * Returns 0 on success, non-zero on error.
 *
 * See docs/architecture.md for BERT layer structure.
 */
int bert_encoder_layer(
    /* Input */
    const float* input,        /* [seq_len, hidden_size] */
    int seq_len,
    int hidden_size,
    int num_head,

    /* Weights (packed) */
    const float* Wq, const float* bq,
    const float* Wk, const float* bk,
    const float* Wv, const float* bv,
    const float* W_selfout, const float* b_selfout,
    const float* W_intmfc, const float* b_intmfc,
    const float* W_outfc, const float* b_outfc,
    const float* ln1_gamma, const float* ln1_beta,
    const float* ln2_gamma, const float* ln2_beta,

    /* Output */
    float* output               /* [seq_len, hidden_size] */
);

#ifdef __cplusplus
}
#endif

#endif /* NPU_KERNELS_H */
