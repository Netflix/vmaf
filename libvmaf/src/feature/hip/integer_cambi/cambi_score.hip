/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP compute kernels for the CAMBI banding-detection feature extractor.
 *  Direct port of `libvmaf/src/feature/cuda/integer_cambi/cambi_score.cu`
 *  (T3-15 / ADR-0360) to the HIP backend.
 *
 *  HIP adaptation notes vs. CUDA twin:
 *  - `.cu` → `.hip`; `cuda_helper.cuh` / `integer_cambi_cuda.h` includes
 *    replaced by `<hip/hip_runtime.h>` (self-contained).
 *  - `__global__`, `blockIdx`, `threadIdx`, `blockDim`, `__device__`,
 *    `__forceinline__`, `extern "C"` are identical in HIP.
 *  - No warp-shuffle or shared memory is used in these kernels, so there
 *    is no HIP warp-size (64 on GCN/RDNA vs. 32 on CUDA) adaptation.
 *  - All three kernels are integer-only (uint16_t), so the score is
 *    bit-exact with both the CUDA twin and the CPU reference.
 *
 *  Kernel inventory (mirrors CUDA twin exactly):
 *    cambi_spatial_mask_kernel  — 7×7 flat derivative mask.
 *    cambi_decimate_kernel      — 2× stride-2 subsample.
 *    cambi_filter_mode_kernel   — separable 3-tap mode filter (H + V).
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/* ------------------------------------------------------------------
 * Kernel 1: Spatial mask
 *
 * Input:  `image`  — flat uint16 array, stride_words columns per row.
 * Output: `mask`   — flat uint16 array, same layout; 1 = edge, 0 = flat.
 *
 * Algorithm (bit-exact with cambi.c::get_spatial_mask_for_index):
 *   For each pixel (x,y):
 *     zero_deriv = (image[y][x] == image[y][x+1]) && (image[y][x] == image[y+1][x])
 *   Compute 7×7 box sum of zero_deriv around each pixel.
 *   mask[y][x] = (box_sum > mask_index) ? 1 : 0.
 * ------------------------------------------------------------------ */
extern "C" {

__global__ void cambi_spatial_mask_kernel(const uint16_t *image, uint16_t *mask, unsigned width,
                                          unsigned height, unsigned stride_words,
                                          unsigned mask_index)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= (int)width || y >= (int)height)
        return;

    /* 7×7 SAT — each thread computes its own window independently.
     * 49 global reads per thread; warp-parallel at 64-lane GCN/RDNA. */
    const int HALF = 3; /* (MASK_FILTER_SIZE=7) >> 1 */
    unsigned box_sum = 0u;
    for (int dy = -HALF; dy <= HALF; dy++) {
        int ry = y + dy;
        if (ry < 0)
            ry = 0;
        if (ry >= (int)height)
            ry = (int)height - 1;
        for (int dx = -HALF; dx <= HALF; dx++) {
            int rx = x + dx;
            if (rx < 0)
                rx = 0;
            if (rx >= (int)width)
                rx = (int)width - 1;
            const uint16_t p = image[(size_t)ry * stride_words + (unsigned)rx];
            const uint16_t r =
                image[(size_t)ry * stride_words +
                      (unsigned)(rx == (int)width - 1 ? rx : rx + 1)];
            const uint16_t b =
                image[(size_t)(ry == (int)height - 1 ? ry : ry + 1) * stride_words +
                      (unsigned)rx];
            const int eq_right = (rx == (int)width - 1) || (p == r);
            const int eq_below = (ry == (int)height - 1) || (p == b);
            box_sum += (unsigned)(eq_right && eq_below);
        }
    }
    mask[(size_t)y * stride_words + (unsigned)x] = (uint16_t)(box_sum > mask_index ? 1u : 0u);
}

/* ------------------------------------------------------------------
 * Kernel 2: 2× decimate
 *
 * Output pixel (x,y) samples input pixel (2x, 2y). Matches
 * cambi.c::decimate (strict even-pixel subsample, no filtering).
 * ------------------------------------------------------------------ */
__global__ void cambi_decimate_kernel(const uint16_t *src, uint16_t *dst, unsigned out_width,
                                      unsigned out_height, unsigned src_stride_words,
                                      unsigned dst_stride_words)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height)
        return;

    dst[(size_t)y * dst_stride_words + x] = src[(size_t)(y * 2u) * src_stride_words + x * 2u];
}

/* ------------------------------------------------------------------
 * Kernel 3: Separable 3-tap mode filter
 *
 * axis=0 → horizontal pass; axis=1 → vertical pass.
 * mode3(a,b,c): value appearing at least twice, or min of all three
 * when all distinct. Matches cambi.c::mode3 / cambi.c::filter_mode.
 * ------------------------------------------------------------------ */
__device__ static __forceinline__ uint16_t mode3_dev(uint16_t a, uint16_t b, uint16_t c)
{
    if (a == b || a == c)
        return a;
    if (b == c)
        return b;
    return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

__global__ void cambi_filter_mode_kernel(const uint16_t *in, uint16_t *out, unsigned width,
                                         unsigned height, unsigned stride_words, int axis)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= (int)width || y >= (int)height)
        return;

    uint16_t a, b, c;
    if (axis == 0) {
        /* Horizontal: neighbours in x. */
        const int xl = (x > 0) ? x - 1 : 0;
        const int xr = (x < (int)width - 1) ? x + 1 : (int)width - 1;
        a = in[(size_t)y * stride_words + (unsigned)xl];
        b = in[(size_t)y * stride_words + (unsigned)x];
        c = in[(size_t)y * stride_words + (unsigned)xr];
    } else {
        /* Vertical: neighbours in y. */
        const int yu = (y > 0) ? y - 1 : 0;
        const int yd = (y < (int)height - 1) ? y + 1 : (int)height - 1;
        a = in[(size_t)(unsigned)yu * stride_words + (unsigned)x];
        b = in[(size_t)(unsigned)y * stride_words + (unsigned)x];
        c = in[(size_t)(unsigned)yd * stride_words + (unsigned)x];
    }
    out[(size_t)(unsigned)y * stride_words + (unsigned)x] = mode3_dev(a, b, c);
}

} /* extern "C" */
