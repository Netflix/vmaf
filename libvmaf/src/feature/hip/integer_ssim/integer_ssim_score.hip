/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP compute kernels for the integer_ssim feature extractor — eleventh
 *  kernel-template consumer. Direct port of the CUDA twin at
 *  `libvmaf/src/feature/cuda/integer_ssim/ssim_score.cu`, adapted for HIP:
 *
 *    1. calculate_integer_ssim_hip_horiz_{8,16}bpc — horizontal 11-tap
 *       separable Gaussian over ref / cmp / ref^2 / cmp^2 / ref*cmp.
 *       Output: five intermediate float buffers, each (W-10) x H.
 *
 *    2. calculate_integer_ssim_hip_vert_combine — vertical 11-tap on the
 *       five intermediates + per-pixel SSIM combine + per-block float
 *       partial sum (tree reduce in shared memory). Output: one float
 *       per block in `partials`.
 *
 *  Host accumulates partials in double, divides by (W-10)*(H-10).
 *
 *  HIP adaptations from CUDA:
 *  - Warp size is 64 on GCN/RDNA. Shared-memory array for warp partial
 *    sums in Pass 2 is sized for 64-lane warps:
 *    BLOCK_SIZE / WARP_SIZE = 128 / 64 = 2 slots.
 *  - VmafCudaBuffer / VmafPicture replaced by raw pointer parameters.
 *  - `__shfl_down` without mask (HIP convention).
 *  - Pictures arrive as CPU VmafPictures; luma planes are copied HtoD
 *    by the host submit() path before kernel launch (T7-10b posture).
 *
 *  v1 constraint: scale=1 only — same as float_ssim_hip / ssim_vulkan /
 *  ssim_cuda. The host init() rejects scale != 1 with -EINVAL before
 *  dispatching.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

#define ISSIM_BX         16
#define ISSIM_BY         8
#define ISSIM_K          11
#define ISSIM_BLOCK_SIZE (ISSIM_BX * ISSIM_BY)

/* HIP warp size is 64 on GCN/RDNA. */
#define ISSIM_WARP_SIZE       64
#define ISSIM_WARPS_PER_BLOCK (ISSIM_BLOCK_SIZE / ISSIM_WARP_SIZE)

/* 11-tap Gaussian kernel — same coefficients as the CUDA twin and
 * float_ssim_hip (ssim_score.hip). */
__device__ static const float ISSIM_G[ISSIM_K] = {
    0.001028f, 0.007599f, 0.036001f, 0.109361f, 0.213006f, 0.266012f,
    0.213006f, 0.109361f, 0.036001f, 0.007599f, 0.001028f,
};

__device__ __forceinline__ float issim_warp_reduce(float v)
{
    for (int off = ISSIM_WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}

extern "C" {

/**
 * Pass 1 (8-bpc): horizontal 11-tap Gaussian.
 * ref/dis are uint8_t row-major, ref_stride/dis_stride bytes per row.
 * Writes h_ref_mu, h_cmp_mu, h_ref_sq, h_cmp_sq, h_refcmp (each
 * w_horiz x h_horiz floats, stride = w_horiz).
 */
__global__ void calculate_integer_ssim_hip_horiz_8bpc(const uint8_t *__restrict__ ref,
                                                       ptrdiff_t ref_stride,
                                                       const uint8_t *__restrict__ cmp,
                                                       ptrdiff_t cmp_stride,
                                                       float *__restrict__ h_ref_mu,
                                                       float *__restrict__ h_cmp_mu,
                                                       float *__restrict__ h_ref_sq,
                                                       float *__restrict__ h_cmp_sq,
                                                       float *__restrict__ h_refcmp,
                                                       unsigned w_horiz,
                                                       unsigned h_horiz)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w_horiz || y >= h_horiz)
        return;

    float ref_mu_h = 0.0f, cmp_mu_h = 0.0f;
    float ref_sq_h = 0.0f, cmp_sq_h = 0.0f, refcmp_h = 0.0f;

    for (int u = 0; u < ISSIM_K; u++) {
        const unsigned src_x = x + (unsigned)u;
        const float r = (float)ref[(ptrdiff_t)y * ref_stride + src_x];
        const float c = (float)cmp[(ptrdiff_t)y * cmp_stride + src_x];
        const float w = ISSIM_G[u];
        ref_mu_h += w * r;
        cmp_mu_h += w * c;
        ref_sq_h += w * (r * r);
        cmp_sq_h += w * (c * c);
        refcmp_h += w * (r * c);
    }

    const unsigned dst_idx = y * w_horiz + x;
    h_ref_mu[dst_idx] = ref_mu_h;
    h_cmp_mu[dst_idx] = cmp_mu_h;
    h_ref_sq[dst_idx] = ref_sq_h;
    h_cmp_sq[dst_idx] = cmp_sq_h;
    h_refcmp[dst_idx] = refcmp_h;
}

/**
 * Pass 1 (16-bpc): horizontal 11-tap Gaussian.
 * Same as 8bpc but pixels are uint16_t (10/12/16bpc), normalised by
 * scaler (4 / 16 / 256) to the [0,255] float range the CUDA twin uses.
 */
__global__ void calculate_integer_ssim_hip_horiz_16bpc(const uint8_t *__restrict__ ref,
                                                        ptrdiff_t ref_stride,
                                                        const uint8_t *__restrict__ cmp,
                                                        ptrdiff_t cmp_stride,
                                                        float *__restrict__ h_ref_mu,
                                                        float *__restrict__ h_cmp_mu,
                                                        float *__restrict__ h_ref_sq,
                                                        float *__restrict__ h_cmp_sq,
                                                        float *__restrict__ h_refcmp,
                                                        unsigned w_horiz,
                                                        unsigned h_horiz,
                                                        unsigned bpc)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w_horiz || y >= h_horiz)
        return;

    float scaler = 1.0f;
    if (bpc == 10u)
        scaler = 4.0f;
    else if (bpc == 12u)
        scaler = 16.0f;
    else if (bpc == 16u)
        scaler = 256.0f;
    const float inv_sc = 1.0f / scaler;

    float ref_mu_h = 0.0f, cmp_mu_h = 0.0f;
    float ref_sq_h = 0.0f, cmp_sq_h = 0.0f, refcmp_h = 0.0f;

    for (int u = 0; u < ISSIM_K; u++) {
        const unsigned src_x = x + (unsigned)u;
        const float r = (float)reinterpret_cast<const uint16_t *>(
            ref + (ptrdiff_t)y * ref_stride)[src_x] * inv_sc;
        const float c = (float)reinterpret_cast<const uint16_t *>(
            cmp + (ptrdiff_t)y * cmp_stride)[src_x] * inv_sc;
        const float w = ISSIM_G[u];
        ref_mu_h += w * r;
        cmp_mu_h += w * c;
        ref_sq_h += w * (r * r);
        cmp_sq_h += w * (c * c);
        refcmp_h += w * (r * c);
    }

    const unsigned dst_idx = y * w_horiz + x;
    h_ref_mu[dst_idx] = ref_mu_h;
    h_cmp_mu[dst_idx] = cmp_mu_h;
    h_ref_sq[dst_idx] = ref_sq_h;
    h_cmp_sq[dst_idx] = cmp_sq_h;
    h_refcmp[dst_idx] = refcmp_h;
}

/**
 * Pass 2: vertical 11-tap Gaussian + per-pixel SSIM combine +
 * per-block float partial sum. Same algorithm as the CUDA twin.
 * Grid sized over (W-10) x (H-10) = w_final x h_final.
 * `partials` receives one float per block (block_idx = blockIdx.y *
 * gridDim.x + blockIdx.x).
 */
__global__ void
calculate_integer_ssim_hip_vert_combine(const float *__restrict__ h_ref_mu,
                                        const float *__restrict__ h_cmp_mu,
                                        const float *__restrict__ h_ref_sq,
                                        const float *__restrict__ h_cmp_sq,
                                        const float *__restrict__ h_refcmp,
                                        float *__restrict__ partials,
                                        unsigned w_horiz,
                                        unsigned w_final,
                                        unsigned h_final,
                                        float c1,
                                        float c2)
{
    const unsigned x   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y   = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned lid = threadIdx.y * blockDim.x + threadIdx.x;

    float my_ssim = 0.0f;
    if (x < w_final && y < h_final) {
        float ref_mu = 0.0f, cmp_mu = 0.0f;
        float ref_sq = 0.0f, cmp_sq = 0.0f, refcmp = 0.0f;
        for (int v = 0; v < ISSIM_K; v++) {
            const unsigned src_y  = y + (unsigned)v;
            const unsigned src_idx = src_y * w_horiz + x;
            const float w = ISSIM_G[v];
            ref_mu += w * h_ref_mu[src_idx];
            cmp_mu += w * h_cmp_mu[src_idx];
            ref_sq += w * h_ref_sq[src_idx];
            cmp_sq += w * h_cmp_sq[src_idx];
            refcmp += w * h_refcmp[src_idx];
        }
        const float ref_var = ref_sq - ref_mu * ref_mu;
        const float cmp_var = cmp_sq - cmp_mu * cmp_mu;
        const float covar   = refcmp - ref_mu * cmp_mu;
        const float mu_xy   = ref_mu * cmp_mu;
        const float num = (2.0f * mu_xy + c1) * (2.0f * covar + c2);
        const float den = (ref_mu * ref_mu + cmp_mu * cmp_mu + c1)
                        * (ref_var + cmp_var + c2);
        my_ssim = num / den;
    }

    /* Per-block tree reduction into shared warp partials. */
    __shared__ float s_warp_sums[ISSIM_WARPS_PER_BLOCK];
    const float warp_sum = issim_warp_reduce(my_ssim);
    const unsigned lane    = lid % ISSIM_WARP_SIZE;
    const unsigned warp_id = lid / ISSIM_WARP_SIZE;
    if (lane == 0)
        s_warp_sums[warp_id] = warp_sum;
    __syncthreads();

    if (lid == 0) {
        float block_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < ISSIM_WARPS_PER_BLOCK; i++)
            block_sum += s_warp_sums[i];
        const unsigned block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        partials[block_idx] = block_sum;
    }
}

} /* extern "C" */
