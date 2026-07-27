/* NPU — Softmax Kernel */

#include "npu_kernels.h"
#include <math.h>

void softmax(const float* x, float* y, int n)
{
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp(x - max) and sum */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        y[i] = expf(x[i] - max_val);
        sum += (double)y[i];
    }

    /* Normalize */
    float inv_sum = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) {
        y[i] *= inv_sum;
    }
}
