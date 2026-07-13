/* NPU — Layer Normalization Kernel */

#include "npu_kernels.h"
#include <math.h>

void layernorm(const float* x, const float* gamma, const float* beta,
               float* y, int n, float eps)
{
    /* Compute mean */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)x[i];
    }
    double mean = sum / n;

    /* Compute variance */
    double var_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)x[i] - mean;
        var_sum += diff * diff;
    }
    double variance = var_sum / n;

    /* Normalize */
    double inv_std = 1.0 / sqrt(variance + (double)eps);
    for (int i = 0; i < n; i++) {
        double normalized = ((double)x[i] - mean) * inv_std;
        y[i] = (float)(normalized * (double)gamma[i] + (double)beta[i]);
    }
}
