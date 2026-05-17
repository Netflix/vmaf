/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for float_ms_ssim (T8-2a / ADR-0435).
 *  Port of `libvmaf/src/feature/cuda/integer_ms_ssim/ms_ssim_score.cu`
 *  to MSL — same 5-scale pyramid, same Wang weights, same formulas.
 *
 *  Three kernel functions, dispatched in order by
 *  `integer_ms_ssim_metal.mm`:
 *
 *  1. `ms_ssim_decimate` — 9-tap 9/7 biorthogonal separable LPF +
 *     2× downsample with period-2n mirror boundary (one call per
 *     inter-scale transition × 2 sides = 8 dispatches for 5 levels).
 *     Input/output: float buffers (scale 0 is picture_copy output
 *     from host). Grid: ceil(w_out/16) × ceil(h_out/16).
 *
 *  2. `ms_ssim_horiz` — horizontal 11-tap separable Gaussian over
 *     the five SSIM statistics on a float input pyramid level. Output
 *     width = W - 10, same height. Grid: ceil(w_h/16) × ceil(H/8).
 *
 *  3. `ms_ssim_vert_lcs` — vertical 11-tap on the horiz-pass
 *     intermediates + per-pixel L/C/S formulas + per-threadgroup
 *     float partial sums × 3. Grid: ceil(w_f/16) × ceil(h_f/8).
 *     Host reduces partials per scale and applies Wang weights.
 *
 *  Buffer bindings documented per kernel below. All float buffers
 *  are packed float arrays (row-major, stride = width * sizeof(float)).
 *
 *  Numeric design notes:
 *   - σ² clamped to ≥ 0 before sqrt (matches `fmaxf` in the CUDA twin
 *     and `MAX(0, ...)` in iqa/ssim_tools.c::ssim_variance_scalar).
 *   - Covariance sign-fix for the S component: when σ_xy_geom ≤ 0
 *     and covariance < 0, clamp covariance to 0 (matches CUDA twin).
 *   - Wang weights (g_alphas / g_betas / g_gammas) are applied
 *     host-side in the .mm after all 5 scales have been read back.
 *
 *  Min-dim guard (ADR-0153): 11 × 2^4 = 176 — enforced in init() of
 *  the .mm host.
 */

#include <metal_stdlib>
using namespace metal;

/* 11-tap Gaussian weights (same as float_ssim.metal and CUDA twin). */
constant float G[11] = {
    0.001028f, 0.007599f, 0.036001f, 0.109361f, 0.213006f,
    0.266012f,
    0.213006f, 0.109361f, 0.036001f, 0.007599f, 0.001028f,
};

/* 9-tap biorthogonal 9/7 LPF coefficients (decimation kernel). */
constant float LPF[9] = {
    0.026727f, -0.016828f, -0.078201f, 0.266846f, 0.602914f,
    0.266846f, -0.078201f, -0.016828f, 0.026727f,
};

/* Period-2n mirror — matches CUDA twin's mirror_idx() and the C
 * reference ms_ssim_decimate_mirror(). Needed by the LPF kernel for
 * sub-kernel-radius boundaries. */
inline int ms_mirror(int idx, int n)
{
    int period = 2 * n;
    int r      = idx % period;
    if (r < 0)   { r += period; }
    if (r >= n)  { r = period - r - 1; }
    return r;
}

/* ------------------------------------------------------------------ */
/*  Kernel 1: ms_ssim_decimate                                         */
/*                                                                      */
/*  Buffer bindings:                                                    */
/*   [[buffer(0)]] src     — const float * input plane (w_in × h_in)   */
/*   [[buffer(1)]] dst     — float * output plane (w_out × h_out)      */
/*   [[buffer(2)]] dims    — uint4 (.x=w_in, .y=h_in,                  */
/*                                   .z=w_out, .w=h_out)               */
/*  Grid: ceil(w_out/16) × ceil(h_out/16),  threads 16×16.            */
/* ------------------------------------------------------------------ */
kernel void ms_ssim_decimate(
    const device float  *src   [[buffer(0)]],
    device       float  *dst   [[buffer(1)]],
    constant     uint4  &dims  [[buffer(2)]],
    uint2  gid [[thread_position_in_grid]])
{
    const uint w_in  = dims.x;
    const uint h_in  = dims.y;
    const uint w_out = dims.z;
    const uint h_out = dims.w;

    const uint x_out = gid.x;
    const uint y_out = gid.y;
    if (x_out >= w_out || y_out >= h_out) { return; }

    const int x_src = (int)x_out * 2;
    const int y_src = (int)y_out * 2;
    const int lpf_half = 4; /* (LPF_LEN - 1) / 2 */

    float acc = 0.0f;
    for (int kv = 0; kv < 9; ++kv) {
        const int yi = ms_mirror(y_src + kv - lpf_half, (int)h_in);
        float row_acc = 0.0f;
        for (int ku = 0; ku < 9; ++ku) {
            const int xi = ms_mirror(x_src + ku - lpf_half, (int)w_in);
            row_acc += src[(uint)yi * w_in + (uint)xi] * LPF[ku];
        }
        acc += row_acc * LPF[kv];
    }
    dst[y_out * w_out + x_out] = acc;
}

/* ------------------------------------------------------------------ */
/*  Kernel 2: ms_ssim_horiz                                            */
/*                                                                      */
/*  Horizontal 11-tap Gaussian convolution over float ref and cmp      */
/*  planes. "Valid" convolution (no padding): output x-range is        */
/*  [0, w_h) where w_h = width - 10. Five output planes in hbuf.      */
/*                                                                      */
/*  Buffer bindings:                                                    */
/*   [[buffer(0)]] ref_in  — const float * reference (width × height)  */
/*   [[buffer(1)]] cmp_in  — const float * distorted (width × height)  */
/*   [[buffer(2)]] hbuf    — float * 5-plane intermediate (w_h × H ×5) */
/*   [[buffer(3)]] params  — uint4 (.x=width, .y=height,               */
/*                                   .z=w_h, .w=0)                     */
/*  Grid: ceil(w_h/16) × ceil(height/8),  threads 16×8.               */
/* ------------------------------------------------------------------ */
kernel void ms_ssim_horiz(
    const device float  *ref_in [[buffer(0)]],
    const device float  *cmp_in [[buffer(1)]],
    device       float  *hbuf   [[buffer(2)]],
    constant     uint4  &params [[buffer(3)]],
    uint2  gid [[thread_position_in_grid]])
{
    const uint width  = params.x;
    const uint height = params.y;
    const uint w_h    = params.z;

    const uint x = gid.x;
    const uint y = gid.y;
    if (x >= w_h || y >= height) { return; }

    float mu_r = 0.0f, mu_c = 0.0f;
    float sq_r = 0.0f, sq_c = 0.0f, rc = 0.0f;

    for (int u = 0; u < 11; ++u) {
        const uint src_idx = y * width + x + (uint)u;
        const float r = ref_in[src_idx];
        const float c = cmp_in[src_idx];
        const float w = G[u];
        mu_r += w * r;
        mu_c += w * c;
        sq_r += w * (r * r);
        sq_c += w * (c * c);
        rc   += w * (r * c);
    }

    /* Five planes, each w_h × height floats. */
    const uint plane_stride = w_h * height;
    const uint dst_idx      = y * w_h + x;
    hbuf[0u * plane_stride + dst_idx] = mu_r;
    hbuf[1u * plane_stride + dst_idx] = mu_c;
    hbuf[2u * plane_stride + dst_idx] = sq_r;
    hbuf[3u * plane_stride + dst_idx] = sq_c;
    hbuf[4u * plane_stride + dst_idx] = rc;
}

/* ------------------------------------------------------------------ */
/*  Kernel 3: ms_ssim_vert_lcs                                         */
/*                                                                      */
/*  Vertical 11-tap Gaussian on the horizontal-pass intermediates,     */
/*  per-pixel L/C/S values, and per-threadgroup float partial sums     */
/*  (l, c, s) written to three separate partials buffers.             */
/*                                                                      */
/*  Buffer bindings:                                                    */
/*   [[buffer(0)]] hbuf        — const float * (5 × w_h × H from K2)  */
/*   [[buffer(1)]] l_partials  — float * per-WG luminance partial      */
/*   [[buffer(2)]] c_partials  — float * per-WG contrast partial       */
/*   [[buffer(3)]] s_partials  — float * per-WG structure partial      */
/*   [[buffer(4)]] params      — uint4 (.x=w_h, .y=height,             */
/*                                       .z=w_f, .w=h_f)               */
/*                  where w_f = w_h - 10, h_f = height - 10           */
/*   [[buffer(5)]] consts      — float4 (.x=c1, .y=c2, .z=c3, .w=0)  */
/*   [[buffer(6)]] grid_dim    — uint2 (.x=grid_w, .y=grid_h)         */
/*  Grid: ceil(w_f/16) × ceil(h_f/8),  threads 16×8.                  */
/* ------------------------------------------------------------------ */
kernel void ms_ssim_vert_lcs(
    const device float   *hbuf       [[buffer(0)]],
    device       float   *l_partials [[buffer(1)]],
    device       float   *c_partials [[buffer(2)]],
    device       float   *s_partials [[buffer(3)]],
    constant     uint4   &params     [[buffer(4)]],
    constant     float4  &consts     [[buffer(5)]],
    constant     uint2   &grid_dim   [[buffer(6)]],
    uint2  gid        [[thread_position_in_grid]],
    uint2  bid        [[threadgroup_position_in_grid]],
    uint   lid        [[thread_index_in_threadgroup]],
    uint   simd_lane  [[thread_index_in_simdgroup]],
    uint   simd_id    [[simdgroup_index_in_threadgroup]],
    uint   simd_count [[simdgroups_per_threadgroup]])
{
    const uint w_h    = params.x;  /* horiz-pass output width */
    const uint height = params.y;  /* horiz-pass output height = original H */
    const uint w_f    = params.z;  /* valid final width  = w_h - 10 */
    const uint h_f    = params.w;  /* valid final height = height - 10 */

    const float c1 = consts.x;
    const float c2 = consts.y;
    const float c3 = consts.z;

    const uint x = gid.x;
    const uint y = gid.y;

    const uint plane_stride = w_h * height;

    float my_l = 0.0f, my_c = 0.0f, my_s = 0.0f;
    if (x < w_f && y < h_f) {
        float mu_r = 0.0f, mu_c = 0.0f;
        float sq_r = 0.0f, sq_c = 0.0f, rc = 0.0f;

        for (int v = 0; v < 11; ++v) {
            const uint src_idx = (y + (uint)v) * w_h + x;
            const float w = G[v];
            mu_r += w * hbuf[0u * plane_stride + src_idx];
            mu_c += w * hbuf[1u * plane_stride + src_idx];
            sq_r += w * hbuf[2u * plane_stride + src_idx];
            sq_c += w * hbuf[3u * plane_stride + src_idx];
            rc   += w * hbuf[4u * plane_stride + src_idx];
        }

        /* σ² clamped to ≥ 0 — matches CUDA twin fmaxf and iqa reference. */
        const float ref_var  = max(sq_r - mu_r * mu_r, 0.0f);
        const float cmp_var  = max(sq_c - mu_c * mu_c, 0.0f);
        const float covar    = rc - mu_r * mu_c;
        const float sigma_geom = sqrt(ref_var * cmp_var);

        /* Covariance sign-fix: when σ_geom ≤ 0 and covariance < 0,
         * clamp to 0 — matches CUDA twin (ADR-0153 / CUDA comment). */
        const float clamped_covar =
            (covar < 0.0f && sigma_geom <= 0.0f) ? 0.0f : covar;

        my_l = (2.0f * mu_r * mu_c + c1) / (mu_r * mu_r + mu_c * mu_c + c1);
        my_c = (2.0f * sigma_geom   + c2) / (ref_var + cmp_var + c2);
        my_s = (clamped_covar       + c3) / (sigma_geom + c3);
    }

    /* Three-output per-WG reduction: SIMD-group → threadgroup partials
     * → one float per WG per output. No atomics (same rationale as
     * motion_v2_metal: MSL does not provide 64-bit atomic_fetch_add on
     * Apple Silicon). */
    threadgroup float sg_l[8];
    threadgroup float sg_c[8];
    threadgroup float sg_s[8];

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
        float gl = 0.0f, gc = 0.0f, gs_val = 0.0f;
        for (uint i = 0; i < simd_count; ++i) {
            gl     += sg_l[i];
            gc     += sg_c[i];
            gs_val += sg_s[i];
        }
        const uint block_idx = bid.y * grid_dim.x + bid.x;
        l_partials[block_idx] = gl;
        c_partials[block_idx] = gc;
        s_partials[block_idx] = gs_val;
    }
}
