/* NPU — Element-wise Operation Kernels */

#include "npu_kernels.h"
#include <math.h>

void vec_add(const float* a, const float* b, float* y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void vec_sub(const float* a, const float* b, float* y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = a[i] - b[i];
    }
}

void vec_mul(const float* a, const float* b, float* y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

void vec_max(const float* a, const float* b, float* y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
}

void vec_scale(float* x, float s, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] *= s;
    }
}

void softmax_masked(const float* x, const float* mask,
                    float* y, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const float* row_in = x + r * cols;
        float* row_out = y + r * cols;

        /* Find max */
        float max_val = row_in[0];
        for (int c = 1; c < cols; c++) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        /* Compute exp(x - max) with mask */
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            float val = row_in[c] - max_val;
            if (mask) val += mask[c];
            row_out[c] = expf(val);
            sum += (double)row_out[c];
        }

        /* Normalize */
        float inv_sum = (float)(1.0 / sum);
        for (int c = 0; c < cols; c++) {
            row_out[c] *= inv_sum;
        }
    }
}
