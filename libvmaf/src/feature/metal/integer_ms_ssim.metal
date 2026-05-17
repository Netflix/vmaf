/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for integer_ms_ssim (Metal port of
 *  integer_ms_ssim_cuda.c).
 *
 *  Algorithm: 5-level Laplacian pyramid + per-scale SSIM with Wang 2003
 *  product combine.  Three kernel functions mirror the three CUDA kernels
 *  in cuda/integer_ms_ssim/ms_ssim_score.cu:
 *
 *  1. `ms_ssim_decimate` — 9-tap 9/7 biorthogonal LPF + 2× downsample
 *     (float input at level i → float output at level i+1).
 *
 *  2. `ms_ssim_horiz` — horizontal 11-tap separable Gaussian over 5 SSIM
 *     statistics operating on the float pyramid level.
 *
 *  3. `ms_ssim_vert_lcs` — vertical 11-tap + per-pixel L/C/S compute +
 *     per-WG float partial reduction for L, C, and S.
 *
 *  Host normalises raw uint pixels to float [0, 255] before uploading
 *  pyramid level 0.  The decimate kernel builds levels 1-4.  The host
 *  accumulates per-scale partials (double) and applies Wang weights.
 *
 *  Wang weights (matches ms_ssim.c / CUDA twin):
 *    alphas = {0, 0, 0, 0, 0.1333}
 *    betas  = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333}
 *    gammas = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333}
 *
 *  Constants: C1 = (0.01·255)² = 6.5025
 *             C2 = (0.03·255)² = 58.5225
 *             C3 = C2 / 2      = 29.26125
 *
 *  Min-dim guard: 11 << 4 = 176 (enforced on the host, matches ADR-0153).
 *
 *  Buffer bindings for `ms_ssim_decimate`:
 *   [[buffer(0)]] src      — float * (w_in × h_in, row-major)
 *   [[buffer(1)]] dst      — float * (w_out × h_out)
 *   [[buffer(2)]] dims     — uint4  (.x=w_in, .y=h_in, .z=w_out, .w=h_out)
 *
 *  Buffer bindings for `ms_ssim_horiz`:
 *   [[buffer(0)]] ref_pyr  — float * (w × h pyramid level)
 *   [[buffer(1)]] cmp_pyr  — float * (w × h)
 *   [[buffer(2)]] hbuf     — float * (5 × w_h × h, output of horiz pass)
 *   [[buffer(3)]] params   — uint4  (.x=w, .y=h, .z=w_h, .w=0)
 *
 *  Buffer bindings for `ms_ssim_vert_lcs`:
 *   [[buffer(0)]] hbuf     — float * (5 × w_h × h from horiz pass)
 *   [[buffer(1)]] l_par    — float * (grid_w × grid_h luminance partials)
 *   [[buffer(2)]] c_par    — float * (same, contrast partials)
 *   [[buffer(3)]] s_par    — float * (same, structure partials)
 *   [[buffer(4)]] params   — uint4  (.x=w_h, .y=h, .z=w_f, .w=h_f)
 *   [[buffer(5)]] consts   — float4 (.x=c1, .y=c2, .z=c3, .w=0)
 *   [[buffer(6)]] grid_dim — uint2  (.x=grid_w, .y=grid_h)
 *
 *  Threadgroup size: 16 × 8 for all three kernels.
 */

#include <metal_stdlib>
using namespace metal;

/* 9-tap 9/7 biorthogonal LPF (same coefficients as decimate_filter_9 in
 * libvmaf/src/feature/ms_ssim_decimate.c and the CUDA decimate kernel). */
constant float DEC[9] = {
     0.02807382f, -0.06071049f, -0.07686502f,
     0.41472545f,  0.99272049f,  0.41472545f,
    -0.07686502f, -0.06071049f,  0.02807382f,
};

/* 11-tap Gaussian (same as float_ssim.metal / CUDA ms_ssim kernels). */
constant float G[11] = {
    0.001028f, 0.007599f, 0.036001f, 0.109361f, 0.213006f,
    0.266012f,
    0.213006f, 0.109361f, 0.036001f, 0.007599f, 0.001028f,
};

#define K        11
#define DEC_HALF  4  /* (9 - 1) / 2 */

/* ------------------------------------------------------------------ */
/*  Kernel 1: bilinear-LPF downsample (decimate)                       */
/* ------------------------------------------------------------------ */
kernel void ms_ssim_decimate(
    const device float  *src    [[buffer(0)]],
    device       float  *dst    [[buffer(1)]],
    constant     uint4  &dims   [[buffer(2)]],
    uint2  gid [[thread_position_in_grid]])
{
    const uint w_in  = dims.x;
    const uint h_in  = dims.y;
    const uint w_out = dims.z;
    const uint h_out = dims.w;

    const uint x_out = gid.x;
    const uint y_out = gid.y;
    if (x_out >= w_out || y_out >= h_out) { return; }

    /* Map output pixel → input pixel centre (2× subsample). */
    const int x_in_c = (int)(x_out * 2);
    const int y_in_c = (int)(y_out * 2);

    /* Separable 9-tap: horizontal pass first, then vertical. */
    /* Two-pass separable: accumulate horizontal across a 9-tap window. */
    float row_acc[9];
    for (int v = 0; v < 9; ++v) {
        const int yi = y_in_c + v - DEC_HALF;
        const int yi_c = (yi < 0) ? 0 : ((yi >= (int)h_in) ? (int)(h_in - 1u) : yi);
        float h_sum = 0.0f;
        for (int u = 0; u < 9; ++u) {
            const int xi = x_in_c + u - DEC_HALF;
            const int xi_c = (xi < 0) ? 0 : ((xi >= (int)w_in) ? (int)(w_in - 1u) : xi);
            h_sum += DEC[u] * src[(uint)yi_c * w_in + (uint)xi_c];
        }
        row_acc[v] = h_sum;
    }
    float out_val = 0.0f;
    for (int v = 0; v < 9; ++v) {
        out_val += DEC[v] * row_acc[v];
    }
    dst[y_out * w_out + x_out] = out_val;
}

/* ------------------------------------------------------------------ */
/*  Kernel 2: horizontal 11-tap Gaussian → 5 SSIM stat planes          */
/* ------------------------------------------------------------------ */
kernel void ms_ssim_horiz(
    const device float  *ref_pyr [[buffer(0)]],
    const device float  *cmp_pyr [[buffer(1)]],
    device       float  *hbuf    [[buffer(2)]],
    constant     uint4  &params  [[buffer(3)]],
    uint2  gid [[thread_position_in_grid]])
{
    const uint W   = params.x;   /* pyramid-level width  */
    const uint H   = params.y;   /* pyramid-level height */
    const uint w_h = params.z;   /* W - 10               */

    const uint x = gid.x;
    const uint y = gid.y;
    if (x >= w_h || y >= H) { return; }

    float mu_r = 0.0f, mu_d = 0.0f;
    float sq_r = 0.0f, sq_d = 0.0f, rd = 0.0f;

    for (int u = 0; u < K; ++u) {
        const uint src_idx = y * W + x + (uint)u;
        const float r = ref_pyr[src_idx];
        const float d = cmp_pyr[src_idx];
        const float w = G[u];
        mu_r += w * r;
        mu_d += w * d;
        sq_r += w * (r * r);
        sq_d += w * (d * d);
        rd   += w * (r * d);
    }

    const uint plane_size = w_h * H;
    const uint dst_idx    = y * w_h + x;
    hbuf[0u * plane_size + dst_idx] = mu_r;
    hbuf[1u * plane_size + dst_idx] = mu_d;
    hbuf[2u * plane_size + dst_idx] = sq_r;
    hbuf[3u * plane_size + dst_idx] = sq_d;
    hbuf[4u * plane_size + dst_idx] = rd;
}

/* ------------------------------------------------------------------ */
/*  Kernel 3: vertical 11-tap + per-pixel L/C/S + per-WG reduction     */
/* ------------------------------------------------------------------ */
kernel void ms_ssim_vert_lcs(
    const device float   *hbuf    [[buffer(0)]],
    device       float   *l_par   [[buffer(1)]],
    device       float   *c_par   [[buffer(2)]],
    device       float   *s_par   [[buffer(3)]],
    constant     uint4   &params  [[buffer(4)]],
    constant     float4  &consts  [[buffer(5)]],
    constant     uint2   &grid_dim[[buffer(6)]],
    uint2  gid        [[thread_position_in_grid]],
    uint2  bid        [[threadgroup_position_in_grid]],
    uint   lid        [[thread_index_in_threadgroup]],
    uint   simd_lane  [[thread_index_in_simdgroup]],
    uint   simd_id    [[simdgroup_index_in_threadgroup]],
    uint   simd_count [[simdgroups_per_threadgroup]])
{
    const uint w_h = params.x;   /* horiz-valid width (W - 10) */
    const uint H   = params.y;   /* pyramid-level height       */
    const uint w_f = params.z;   /* final valid width  = w_h   */
    const uint h_f = params.w;   /* final valid height = H - 10*/

    const float c1 = consts.x;
    const float c2 = consts.y;
    const float c3 = consts.z;

    const uint x = gid.x;
    const uint y = gid.y;

    float my_l = 0.0f, my_c = 0.0f, my_s = 0.0f;

    if (x < w_f && y < h_f) {
        const uint plane_size = w_h * H;

        float mu_r = 0.0f, mu_d = 0.0f;
        float sq_r = 0.0f, sq_d = 0.0f, rd = 0.0f;

        for (int v = 0; v < K; ++v) {
            const uint src_y   = y + (uint)v;
            const uint src_idx = src_y * w_h + x;
            const float w = G[v];
            mu_r += w * hbuf[0u * plane_size + src_idx];
            mu_d += w * hbuf[1u * plane_size + src_idx];
            sq_r += w * hbuf[2u * plane_size + src_idx];
            sq_d += w * hbuf[3u * plane_size + src_idx];
            rd   += w * hbuf[4u * plane_size + src_idx];
        }

        const float var_r  = sq_r - mu_r * mu_r;
        const float var_d  = sq_d - mu_d * mu_d;
        const float cov_rd = rd   - mu_r * mu_d;
        const float sig_r  = sqrt(max(var_r, 0.0f));
        const float sig_d  = sqrt(max(var_d, 0.0f));

        /* Wang 2003 L/C/S decomposition:
         *   l = (2·μr·μd + C1) / (μr² + μd² + C1)
         *   c = (2·σr·σd + C2) / (σr² + σd² + C2)
         *   s = (σrd + C3) / (σr·σd + C3)              */
        my_l = (2.0f * mu_r * mu_d + c1) /
               (mu_r * mu_r + mu_d * mu_d + c1);
        my_c = (2.0f * sig_r * sig_d + c2) /
               (var_r + var_d + c2);
        my_s = (cov_rd + c3) / (sig_r * sig_d + c3);
    }

    /* Per-WG float partial: simd_sum → shared sum → partial buffer. */
    threadgroup float sg_l[8], sg_c[8], sg_s[8];

    const float lane_l = simd_sum(my_l);
    const float lane_c = simd_sum(my_c);
    const float lane_s = simd_sum(my_s);

    if (simd_lane == 0) {
        sg_l[simd_id] = lane_l;
        sg_c[simd_id] = lane_c;
        sg_s[simd_id] = lane_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float gs_l = 0.0f, gs_c = 0.0f, gs_s = 0.0f;
        for (uint i = 0; i < simd_count; ++i) {
            gs_l += sg_l[i];
            gs_c += sg_c[i];
            gs_s += sg_s[i];
        }
        const uint par_idx = bid.y * grid_dim.x + bid.x;
        l_par[par_idx] = gs_l;
        c_par[par_idx] = gs_c;
        s_par[par_idx] = gs_s;
    }
}
