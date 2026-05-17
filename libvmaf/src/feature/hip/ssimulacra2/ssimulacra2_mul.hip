/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP kernel for the ssimulacra2 elementwise plane multiply.
 *  Direct port of the CUDA twin at
 *  `libvmaf/src/feature/cuda/ssimulacra2/ssimulacra2_mul.cu`.
 *
 *  out[i] = a[i] * b[i] for all (x, y, channel) samples.
 *  FMA-free by definition (pure multiply, no accumulation).
 */

#include <hip/hip_runtime.h>

extern "C" {

__global__ void ssimulacra2_mul3(const float *__restrict__ a, const float *__restrict__ b,
                                 float *__restrict__ out, unsigned width, unsigned height,
                                 unsigned plane_count, unsigned plane_stride)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const unsigned base = y * width + x;
    for (unsigned c = 0u; c < plane_count; ++c) {
        const unsigned idx = base + c * plane_stride;
        out[idx] = a[idx] * b[idx];
    }
}

} /* extern "C" */
