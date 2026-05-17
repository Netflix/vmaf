/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP compute kernels for the float_vif feature extractor.
 *  Direct port of the CUDA twin at
 *  `libvmaf/src/feature/cuda/float_vif/float_vif_score.cu`.
 *
 *  Two operating modes:
 *    float_vif_compute:  read ref+dis at this scale, apply separable
 *      V→H filter inline (5 outputs: mu1, mu2, ref², dis², ref·dis),
 *      run vif_stat per pixel, reduce (num, den) per block.
 *    float_vif_decimate: apply this scale's filter at the previous
 *      scale's full dimensions, sample at (2*gx, 2*gy) for output.
 *      CPU's VIF_OPT_HANDLE_BORDERS branch — mirror padding for taps
 *      near the edge.
 *
 *  Mirror padding: reflect-102 (both axes):
 *    idx < 0  →  -idx
 *    idx >= sup  →  2*sup - idx - 2
 *
 *  Per scale: 4 compute + 3 decimate launches. Filter coefficients
 *  constant-folded per scale.
 *
 *  HIP adaptation from CUDA:
 *  - Warp size is 64 on GCN/RDNA (vs 32 in CUDA). The per-block warp
 *    partial array is sized FVIF_BX * FVIF_BY / FVIF_WARP_SIZE = 4 slots.
 *  - `__shfl_down` used without mask (HIP convention).
 *  - Entry-point names match the CUDA twin so the host Module API lookup
 *    in `float_vif_hip.c` resolves correctly without a name change.
 *
 *  Precision contract: places=3 (same as CUDA twin per ADR-0192).
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

#define FVIF_BX            16
#define FVIF_BY            16
#define FVIF_MAX_FW        17
#define FVIF_MAX_HFW        8

/* HIP warp size is 64 on GCN/RDNA. */
#define FVIF_WARP_SIZE        64
#define FVIF_WARPS_PER_BLOCK  (FVIF_BX * FVIF_BY / FVIF_WARP_SIZE)

__device__ static const float FVIF_COEFF_S0[FVIF_MAX_FW] = {
    0.00745626912f, 0.0142655009f, 0.0250313189f, 0.0402820669f, 0.0594526194f, 0.0804751068f,
    0.0999041125f,  0.113746084f,  0.118773937f,  0.113746084f,  0.0999041125f, 0.0804751068f,
    0.0594526194f,  0.0402820669f, 0.0250313189f, 0.0142655009f, 0.00745626912f};
__device__ static const float FVIF_COEFF_S1[FVIF_MAX_FW] = {
    0.0189780835f, 0.0558981746f, 0.120920904f,  0.192116052f, 0.224173605f, 0.192116052f,
    0.120920904f,  0.0558981746f, 0.0189780835f, 0.0f,         0.0f,         0.0f,
    0.0f,          0.0f,          0.0f,           0.0f,         0.0f};
__device__ static const float FVIF_COEFF_S2[FVIF_MAX_FW] = {
    0.054488685f, 0.244201347f, 0.402619958f, 0.244201347f, 0.054488685f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f,         0.0f,         0.0f,         0.0f,         0.0f,         0.0f, 0.0f, 0.0f};
__device__ static const float FVIF_COEFF_S3[FVIF_MAX_FW] = {
    0.166378498f, 0.667243004f, 0.166378498f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f,         0.0f,         0.0f,         0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

__device__ __forceinline__ const float *fvif_coeffs(int scale)
{
    if (scale == 0)
        return FVIF_COEFF_S0;
    if (scale == 1)
        return FVIF_COEFF_S1;
    if (scale == 2)
        return FVIF_COEFF_S2;
    return FVIF_COEFF_S3;
}

__device__ __forceinline__ int fvif_fw(int scale)
{
    const int fw[4] = {17, 9, 5, 3};
    return fw[scale];
}

__device__ __forceinline__ int fvif_mirror_v(int idx, int sup)
{
    if (idx < 0)
        return -idx;
    if (idx >= sup)
        return 2 * sup - idx - 2;
    return idx;
}

__device__ __forceinline__ int fvif_mirror_h(int idx, int sup)
{
    if (idx < 0)
        return -idx;
    if (idx >= sup)
        return 2 * sup - idx - 2;
    return idx;
}

/* HIP warp-level float reduction (64-lane). */
__device__ __forceinline__ float fvif_warp_reduce(float v)
{
    for (int off = FVIF_WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}

/* Read raw pixel (uint8/16) and convert to float. */
__device__ __forceinline__ float fvif_read_raw(const uint8_t *plane, int stride_bytes, int y,
                                               int x, unsigned bpc)
{
    if (bpc <= 8u) {
        return (float)plane[y * stride_bytes + x] - 128.0f;
    }
    const uint16_t v = reinterpret_cast<const uint16_t *>(plane + y * stride_bytes)[x];
    float scaler = 1.0f;
    if (bpc == 10u)
        scaler = 4.0f;
    else if (bpc == 12u)
        scaler = 16.0f;
    else if (bpc == 16u)
        scaler = 256.0f;
    return (float)v / scaler - 128.0f;
}

extern "C" {

/**
 * Compute kernel — reads ref+dis at `scale`, applies separable V→H filter,
 * computes vif_stat per pixel, reduces (num, den) per block.
 *
 * At scale=0, ref_raw/dis_raw are the source pixel planes (uint8/16).
 * At scale>0, ref_f/dis_f are float buffers at this scale's dimensions.
 */
__global__ void float_vif_compute(int scale, const uint8_t *ref_raw, const uint8_t *dis_raw,
                                  ptrdiff_t raw_stride, const float *ref_f, const float *dis_f,
                                  ptrdiff_t f_stride_floats, float *num_partials,
                                  float *den_partials, unsigned width, unsigned height,
                                  unsigned bpc, unsigned grid_x_count)
{
    const int fw  = fvif_fw(scale);
    const int hfw = fw / 2;
    const float *coeff = fvif_coeffs(scale);

    /* MAX_TILE_W = FVIF_BX + 2*FVIF_MAX_HFW = 32. */
    const int MAX_TILE_W = FVIF_BX + 2 * FVIF_MAX_HFW;
    const int tile_w     = FVIF_BX + 2 * hfw;
    const int tile_h     = FVIF_BY + 2 * hfw;

    __shared__ float s_ref[32 * 32];
    __shared__ float s_dis[32 * 32];
    __shared__ float s_v_mu1[FVIF_BY * 32];
    __shared__ float s_v_mu2[FVIF_BY * 32];
    __shared__ float s_v_xx [FVIF_BY * 32];
    __shared__ float s_v_yy [FVIF_BY * 32];
    __shared__ float s_v_xy [FVIF_BY * 32];

    const int gx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy  = blockIdx.y * blockDim.y + threadIdx.y;
    const int lx  = threadIdx.x;
    const int ly  = threadIdx.y;
    const unsigned lid  = ly * FVIF_BX + lx;
    const bool valid    = (gx < (int)width && gy < (int)height);

    const int tile_oy    = blockIdx.y * FVIF_BY - hfw;
    const int tile_ox    = blockIdx.x * FVIF_BX - hfw;
    const int tile_elems = tile_h * tile_w;
    const bool is_raw    = (scale == 0);

    /* Phase 1: tile load with mirror padding. */
    for (int i = lid; i < tile_elems; i += FVIF_BX * FVIF_BY) {
        const int tr = i / tile_w;
        const int tc = i - tr * tile_w;
        const int py = fvif_mirror_v(tile_oy + tr, (int)height);
        const int px = fvif_mirror_h(tile_ox + tc, (int)width);
        float r, d;
        if (is_raw) {
            r = fvif_read_raw(ref_raw, (int)raw_stride, py, px, bpc);
            d = fvif_read_raw(dis_raw, (int)raw_stride, py, px, bpc);
        } else {
            r = ref_f[(size_t)py * f_stride_floats + px];
            d = dis_f[(size_t)py * f_stride_floats + px];
        }
        s_ref[tr * MAX_TILE_W + tc] = r;
        s_dis[tr * MAX_TILE_W + tc] = d;
    }
    __syncthreads();

    /* Phase 2: vertical filter for WG_Y output rows × tile_w cols. */
    const int vert_total = FVIF_BY * tile_w;
    for (int i = lid; i < vert_total; i += FVIF_BX * FVIF_BY) {
        const int r = i / tile_w;
        const int c = i - r * tile_w;
        float a_mu1 = 0.0f, a_mu2 = 0.0f, a_xx = 0.0f, a_yy = 0.0f, a_xy = 0.0f;
        for (int k = 0; k < fw; k++) {
            const float c_k   = coeff[k];
            const float ref_v = s_ref[(r + k) * MAX_TILE_W + c];
            const float dis_v = s_dis[(r + k) * MAX_TILE_W + c];
            a_mu1 += c_k * ref_v;
            a_mu2 += c_k * dis_v;
            a_xx  += c_k * (ref_v * ref_v);
            a_yy  += c_k * (dis_v * dis_v);
            a_xy  += c_k * (ref_v * dis_v);
        }
        s_v_mu1[r * MAX_TILE_W + c] = a_mu1;
        s_v_mu2[r * MAX_TILE_W + c] = a_mu2;
        s_v_xx [r * MAX_TILE_W + c] = a_xx;
        s_v_yy [r * MAX_TILE_W + c] = a_yy;
        s_v_xy [r * MAX_TILE_W + c] = a_xy;
    }
    __syncthreads();

    /* Phase 3: per-thread horizontal filter + vif_stat. */
    float my_num = 0.0f, my_den = 0.0f;
    if (valid) {
        float mu1 = 0.0f, mu2 = 0.0f, xx = 0.0f, yy = 0.0f, xy = 0.0f;
        for (int k = 0; k < fw; k++) {
            const float c_k = coeff[k];
            mu1 += c_k * s_v_mu1[ly * MAX_TILE_W + (lx + k)];
            mu2 += c_k * s_v_mu2[ly * MAX_TILE_W + (lx + k)];
            xx  += c_k * s_v_xx [ly * MAX_TILE_W + (lx + k)];
            yy  += c_k * s_v_yy [ly * MAX_TILE_W + (lx + k)];
            xy  += c_k * s_v_xy [ly * MAX_TILE_W + (lx + k)];
        }
        const float eps           = 1.0e-10f;
        const float vif_sigma_nsq = 2.0f;
        const float vif_egl       = 100.0f;
        const float sigma_max_inv = (2.0f * 2.0f) / (255.0f * 255.0f);

        float sigma1_sq = xx - mu1 * mu1;
        float sigma2_sq = yy - mu2 * mu2;
        float sigma12   = xy - mu1 * mu2;
        sigma1_sq = (sigma1_sq < 0.0f) ? 0.0f : sigma1_sq;
        sigma2_sq = (sigma2_sq < 0.0f) ? 0.0f : sigma2_sq;
        float g     = sigma12 / (sigma1_sq + eps);
        float sv_sq = sigma2_sq - g * sigma12;
        if (sigma1_sq < eps) {
            g         = 0.0f;
            sv_sq     = sigma2_sq;
            sigma1_sq = 0.0f;
        }
        if (sigma2_sq < eps) {
            g     = 0.0f;
            sv_sq = 0.0f;
        }
        if (g < 0.0f) {
            sv_sq = sigma2_sq;
            g     = 0.0f;
        }
        sv_sq = (sv_sq < eps) ? eps : sv_sq;
        g     = (g > vif_egl) ? vif_egl : g;

        float num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + vif_sigma_nsq));
        float den_val = log2f(1.0f + sigma1_sq / vif_sigma_nsq);
        if (sigma12 < 0.0f)
            num_val = 0.0f;
        if (sigma1_sq < vif_sigma_nsq) {
            num_val = 1.0f - sigma2_sq * sigma_max_inv;
            den_val = 1.0f;
        }
        my_num = num_val;
        my_den = den_val;
    }

    /* Phase 4: warp + cross-warp reduction. */
    __shared__ float s_num_warps[FVIF_WARPS_PER_BLOCK];
    __shared__ float s_den_warps[FVIF_WARPS_PER_BLOCK];
    const float wn          = fvif_warp_reduce(my_num);
    const float wd          = fvif_warp_reduce(my_den);
    const unsigned lane     = lid % FVIF_WARP_SIZE;
    const unsigned warp_id  = lid / FVIF_WARP_SIZE;
    if (lane == 0) {
        s_num_warps[warp_id] = wn;
        s_den_warps[warp_id] = wd;
    }
    __syncthreads();
    if (lid == 0) {
        float total_num = 0.0f, total_den = 0.0f;
        for (int i = 0; i < FVIF_WARPS_PER_BLOCK; i++) {
            total_num += s_num_warps[i];
            total_den += s_den_warps[i];
        }
        const unsigned wg_idx = blockIdx.y * grid_x_count + blockIdx.x;
        num_partials[wg_idx]  = total_num;
        den_partials[wg_idx]  = total_den;
    }
}

/**
 * Decimate kernel — applies `scale`'s filter at the previous scale's
 * dimensions, samples at (2*gx, 2*gy) for output.
 * Output dimensions: in_w / 2, in_h / 2.
 */
__global__ void float_vif_decimate(int scale, const uint8_t *ref_raw, const uint8_t *dis_raw,
                                   ptrdiff_t raw_stride, const float *ref_f, const float *dis_f,
                                   ptrdiff_t f_stride_floats, float *ref_out, float *dis_out,
                                   ptrdiff_t out_stride_floats, unsigned out_w, unsigned out_h,
                                   unsigned in_w, unsigned in_h, unsigned bpc)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= (int)out_w || gy >= (int)out_h)
        return;

    const int fw   = fvif_fw(scale);
    const int hfw  = fw / 2;
    const float *coeff = fvif_coeffs(scale);
    const int in_x = 2 * gx;
    const int in_y = 2 * gy;
    const bool is_raw = (scale == 1);

    /* V-inner / H-outer ordering matches CPU vif_filter1d_s. */
    float acc_ref = 0.0f, acc_dis = 0.0f;
    for (int kj = 0; kj < fw; kj++) {
        const float c_j = coeff[kj];
        const int px    = fvif_mirror_h(in_x - hfw + kj, (int)in_w);
        float v_ref = 0.0f, v_dis = 0.0f;
        for (int ki = 0; ki < fw; ki++) {
            const float c_i = coeff[ki];
            const int py    = fvif_mirror_v(in_y - hfw + ki, (int)in_h);
            float r, d;
            if (is_raw) {
                r = fvif_read_raw(ref_raw, (int)raw_stride, py, px, bpc);
                d = fvif_read_raw(dis_raw, (int)raw_stride, py, px, bpc);
            } else {
                r = ref_f[(size_t)py * f_stride_floats + px];
                d = dis_f[(size_t)py * f_stride_floats + px];
            }
            v_ref += c_i * r;
            v_dis += c_i * d;
        }
        acc_ref += c_j * v_ref;
        acc_dis += c_j * v_dis;
    }

    ref_out[(size_t)gy * out_stride_floats + gx] = acc_ref;
    dis_out[(size_t)gy * out_stride_floats + gx] = acc_dis;
}

} /* extern "C" */
