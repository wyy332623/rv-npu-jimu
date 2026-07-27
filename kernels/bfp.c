/* NPU — Block Floating Point (BFP) Conversion Kernels
 *
 * Two BFP formats are supported:
 *
 * 1. Block BFP (original): FP32 block-floating-point with grouped exponents.
 *    Used for training/inference reference.  Kept for backward compatibility.
 *
 * 2. HDL BFP: FP16 per-element with shared exponent, matching the NPU
 *    hardware (hdl/bfp/bfp.py).  6-bit mantissa {sign, 1, fraction[0:4]}
 *    + 5-bit shared exponent.  Used in integration test (precision_mode=1).
 */

#include "npu_kernels.h"
#include <math.h>
#include <string.h>

void float_to_bfp(const float* input, int32_t* output, int32_t* exponents,
                  int length, int elems_per_exp, int frac_bits)
{
    int num_blocks = (length + elems_per_exp - 1) / elems_per_exp;
    /* Signed mantissa range: [-(2^frac_bits - 1), (2^frac_bits - 1)]
       e.g. frac_bits=6 → range [-63, 63] */
    int max_mantissa = (1 << frac_bits) - 1;

    for (int b = 0; b < num_blocks; b++) {
        int start = b * elems_per_exp;
        int end = start + elems_per_exp;
        if (end > length) end = length;

        /* Find max absolute value in block */
        float max_abs = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = fabsf(input[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        /* Compute shared exponent such that max_mantissa * 2^exp ≈ max_abs
           2^exp = max_abs / max_mantissa
           exp = log2(max_abs / max_mantissa) = log2(max_abs) - log2(max_mantissa) */
        int exp = 0;
        if (max_abs > 0.0f) {
            /* exp must be large enough that max_mantissa * 2^exp >= max_abs
               i.e. exp >= log2(max_abs / max_mantissa)
               Use ceil to ensure the mantissa can represent the max value. */
            exp = (int)ceilf(log2f(max_abs / (float)max_mantissa));
        }
        exponents[b] = exp;

        /* Compute scale factor: size of one BFP unit in float */
        float unit = ldexpf(1.0f, exp);  /* 2^exp */

        /* Quantize: mantissa = round(value / unit) */
        for (int i = start; i < end; i++) {
            float normalized = input[i] / unit;
            /* Round to nearest integer */
            int32_t mantissa = (int32_t)(normalized + (normalized >= 0 ? 0.5f : -0.5f));
            /* Clamp to signed range */
            if (mantissa > max_mantissa) mantissa = max_mantissa;
            if (mantissa < -max_mantissa) mantissa = -max_mantissa;
            output[i] = mantissa;
        }
    }
}

void bfp_to_float(const int32_t* input, const int32_t* exponents,
                  float* output, int length, int elems_per_exp, int frac_bits)
{
    int num_blocks = (length + elems_per_exp - 1) / elems_per_exp;

    for (int b = 0; b < num_blocks; b++) {
        int start = b * elems_per_exp;
        int end = start + elems_per_exp;
        if (end > length) end = length;

        float unit = ldexpf(1.0f, exponents[b]);  /* 2^exp */

        for (int i = start; i < end; i++) {
            output[i] = (float)input[i] * unit;
        }
    }
}

/* ────────────────────────────────────────────────────────────────
 * HDL-format BFP (matching hdl/bfp/bfp.py)
 *
 *   Input:  lanes FP16 values (packed as uint16_t[])
 *   Output: bfp_data[0..lanes-1]  — 6-bit mantissa per lane
 *           bfp_exp               — shared 5-bit exponent
 *
 *   BFP 6-bit format per lane:
 *     bit [5] = sign
 *     bit [4] = implied leading 1 (0 for subnormals/zero)
 *     bits [3:0] = fraction
 *
 *   Decoded FP16 ≈ (sign ? -1 : 1) * 2^(bfp_exp) * (1 + fraction/16)
 *   (simplified; see BfpToFloat for exact reconstruction)
 * ──────────────────────────────────────────────────────────────── */

/* FP16 helper: extract fields */
static inline int fp16_sign(uint16_t v) { return (v >> 15) & 1; }
static inline int fp16_exp(uint16_t v)  { return (v >> 10) & 0x1F; }
static inline int fp16_frac(uint16_t v) { return v & 0x3FF; }

/* Build SLF (Sign + Leading + Fraction) matching HDL:
 *   slf = {sign(1), implied_one(1), fraction[9:0]} = 12 bits
 *   Cat(f[0:10], leading, s) in Amaranth
 *   → bit 11 = sign, bit 10 = implied_one, bits [9:0] = fraction
 *   (Cat puts fraction at LSB, leading at bit 10, sign at bit 11)
 */
static inline int fp16_to_slf(uint16_t v) {
    int s = fp16_sign(v);
    int e = fp16_exp(v);
    int f = fp16_frac(v);
    int leading = (e != 0) ? 1 : 0;
    // Pack: bits [9:0] = fraction, bit 10 = leading, bit 11 = sign
    return (s << 11) | (leading << 10) | f;
}

/* Arithmetic right shift of an SLF (12-bit, sign-extended) */
static inline int slf_arshift(int slf, int shift) {
    if (shift >= 12) return (slf < 0) ? -1 : 0;
    if (shift == 0) return slf;
    // Sign extend: bit 11 is sign
    int sign = (slf >> 11) & 1;
    int mask = (1 << (12 - shift)) - 1;
    int shifted = (slf >> shift) & mask;
    if (sign) shifted |= (~mask) & ((1 << 12) - 1);
    return shifted;
}

/* Convert FP16 vector to HDL-format BFP.
 *   input: lanes FP16 values (as uint16_t)
 *   bfp_out: per-lane 6-bit BFP mantissa (stored as uint8_t[lanes])
 *   bfp_exp_out: shared 5-bit exponent
 *   lanes: number of elements (must match HDL FloatToBfp.lanes)
 */
void float16_bfp_encode(const uint16_t* input, uint8_t* bfp_out,
                         int* bfp_exp_out, int lanes) {
    /* S1: compute SLF and find max exponent */
    int slf[32];  /* max 32 lanes */
    int max_exp = 0;
    for (int i = 0; i < lanes && i < 32; i++) {
        slf[i] = fp16_to_slf(input[i]);
        int e = fp16_exp(input[i]);
        if (e > max_exp) max_exp = e;
    }

    /* S2: barrel shift + truncate to 6 bits */
    for (int i = 0; i < lanes && i < 32; i++) {
        int e = fp16_exp(input[i]);
        int shift = max_exp - e;
        int shifted = slf_arshift(slf[i], shift);
        /* Take top 7 bits of 12-bit SLF: bits [5:12) */
        int rounding_val = (shifted >> 5) & 0x7F;  /* top 7 bits */
        /* Truncate to 6 bits: remove MSB of rounding_val */
        int trunc_bits = rounding_val & 0x3F;
        bfp_out[i] = (uint8_t)trunc_bits;
    }

    /* Shared exponent = max_exp - 15 (FP16 bias) */
    *bfp_exp_out = max_exp - 15;
}

/* Convert HDL-format BFP to FP16.
 *   bfp_in: per-lane 6-bit BFP mantissa (stored as uint8_t[lanes])
 *   bfp_exp: shared 5-bit exponent
 *   output: lanes FP16 values (as uint16_t)
 *   lanes: number of elements (must match HDL BfpToFloat.lanes)
 */
void float16_bfp_decode(const uint8_t* bfp_in, int bfp_exp,
                          uint16_t* output, int lanes) {
    for (int i = 0; i < lanes && i < 32; i++) {
        uint8_t bv = bfp_in[i];
        if (bv == 0) {
            output[i] = 0;
            continue;
        }
        int sign = (bv >> 5) & 1;        /* bit 5 = sign */
        int mant = bv & 0x1F;             /* bits [0:5) = mantissa */
        int implied_one = (mant >> 4) & 1; /* bit 4 of mant = implied 1 */
        int fraction = mant & 0x0F;        /* bits [0:4) = fraction */

        /* FP16 mantissa = fraction << 6  (10 - (5-1) = 6) */
        int fp16_mant = fraction << 6;
        /* FP16 exponent = bfp_exp + 15 */
        int fp16_exp_val = bfp_exp + 15;

        /* Clamp exponent to valid range */
        if (fp16_exp_val < 0) fp16_exp_val = 0;
        if (fp16_exp_val > 31) fp16_exp_val = 31;

        /* Pack FP16: {sign, exp[4:0], mant[9:0]} */
        output[i] = (uint16_t)((sign << 15) | (fp16_exp_val << 10) | fp16_mant);
    }
}


/* ────────────────────────────────────────────────────────────────
 * Original block BFP (kept for backward compatibility)
 * ──────────────────────────────────────────────────────────────── */

void mv_mul_bfp(const int32_t* A_bfp, const int32_t* A_exp,
                const int32_t* x_bfp, const int32_t* x_exp,
                int32_t* y_bfp, int32_t* y_exp,
                int rows, int cols, int elems_per_exp, int frac_bits)
{
    for (int r = 0; r < rows; r++) {
        double sum = 0.0;

        for (int c = 0; c < cols; c++) {
            int a_block = c / elems_per_exp;
            int x_block = c / elems_per_exp;
            int a_e = A_exp ? A_exp[a_block] : 0;
            int x_e = x_exp ? x_exp[x_block] : 0;
            double val = (double)A_bfp[r * cols + c] * (double)x_bfp[c];
            /* Account for both exponents: result = (A_m * 2^a_e) * (x_m * 2^x_e)
                                          = (A_m * x_m) * 2^(a_e + x_e) */
            sum += val * ldexp(1.0, a_e + x_e);
        }

        /* Convert back to BFP */
        float f_result = (float)sum;
        if (f_result == 0.0f) {
            y_bfp[r] = 0;
            y_exp[r] = 0;
            continue;
        }

        int max_mantissa = (1 << frac_bits) - 1;
        float abs_val = fabsf(f_result);
        int exp = (int)ceilf(log2f(abs_val / (float)max_mantissa));
        y_exp[r] = exp;

        float unit = ldexpf(1.0f, exp);
        float normalized = f_result / unit;
        y_bfp[r] = (int32_t)(normalized + (normalized >= 0 ? 0.5f : -0.5f));
    }
}
