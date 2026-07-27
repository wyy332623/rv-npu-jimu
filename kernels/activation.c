/* NPU — Activation Function Kernels */

#include "npu_kernels.h"
#include <math.h>

void gelu(const float* x, float* y, int n)
{
    /* GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    const float sqrt_2_over_pi = 0.7978845608028654f;  /* sqrt(2.0 / M_PI) */
    const float coeff = 0.044715f;

    for (int i = 0; i < n; i++) {
        float x3 = x[i] * x[i] * x[i];
        float inner = sqrt_2_over_pi * (x[i] + coeff * x3);
        y[i] = 0.5f * x[i] * (1.0f + tanhf(inner));
    }
}

void relu(const float* x, float* y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void sigmoid(const float* x, float* y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

void tanh_vec(const float* x, float* y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = tanhf(x[i]);
    }
}
