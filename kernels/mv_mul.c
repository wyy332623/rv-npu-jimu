/* NPU — Matrix-Vector Multiply Kernel */

#include "npu_kernels.h"
#include <math.h>

void mv_mul(const float* A, const float* x, float* y,
            int rows, int cols, int accum)
{
    for (int r = 0; r < rows; r++) {
        double sum = accum ? (double)y[r] : 0.0;
        for (int c = 0; c < cols; c++) {
            sum += (double)A[r * cols + c] * (double)x[c];
        }
        y[r] = (float)sum;
    }
}
