/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for integer_ssim (T8-1k).
 *  Direct port of the CUDA twin in feature/cuda/integer_ssim_cuda.c /
 *  feature/cuda/integer_ssim/ssim_score.cu.
 *
 *  Algorithm: separable 11-tap Gaussian convolution on five SSIM
 *  statistics (mu_r, mu_d, sq_r, sq_d, rd), then per-pixel SSIM
 *  combine using the standard formula:
 *
 *    SSIM(x,y) = (2·mu_r·mu_d + C1)·(2·sigma_rd + C2)
 *              / ((mu_r² + mu_d² + C1)·(sigma_r² + sigma_d² + C2))
 *
 *  where sigma² = E[x²] − mu², sigma_rd = E[rd] − mu_r·mu_d.
 *  Constants: C1 = (0.01·255)², C2 = (0.03·255)².
 *
 *  Input pixels are normalised to [0, 255.xxx] in the kernel:
 *    8bpc  → identity (float cast of uint8)
 *    10bpc → divide by 4.0
 *    12bpc → divide by 16.0
 *    16bpc → divide by 256.0
 *
 *  "Valid" convolution — no mirror padding:
 *    horiz output:  w_h × H   where w_h = W - 10
 *    vert output:   w_h × h_v where h_v = H - 10
 *  SSIM is averaged over w_h × h_v pixels.
 *
 *  Two kernel functions dispatched in order:
 *
 *  1. `integer_ssim_horiz_{8,16}bpc` — horizontal 11-tap on ref and dis
 *     → 5 intermediate float planes in hbuf (stride = w_h).
 *     Grid: ceil(w_h / 16) × ceil(H / 8)  [16×8 threadgroup]
 *
 *  2. `integer_ssim_vert_combine` — vertical 11-tap on hbuf → per-pixel
 *     SSIM → per-WG float partial.
 *     Grid: ceil(w_h / 16) × ceil(h_v / 8)
 *
 *  Buffer bindings for `integer_ssim_horiz_8bpc`:
 *   [[buffer(0)]] ref    — const uchar * (full W×H, byte-addressed)
 *   [[buffer(1)]] dis    — const uchar *
 *   [[buffer(2)]] hbuf   — float * (5 × w_h × H interleaved planes)
 *   [[buffer(3)]] params — uint4 (.x=W, .y=H, .z=w_h, .w=ref_stride)
 *
 *  Buffer bindings for `integer_ssim_horiz_16bpc`:
 *   [[buffer(0)]] ref    — const uchar * (uint16, stride-addressed)
 *   [[buffer(1)]] dis    — const uchar *
 *   [[buffer(2)]] hbuf   — float *
 *   [[buffer(3)]] params — uint4 (.x=W, .y=H, .z=w_h, .w=ref_stride in bytes)
 *   [[buffer(4)]] bpcbuf — uint  (bpc, for scaler)
 *
 *  Buffer bindings for `integer_ssim_vert_combine`:
 *   [[buffer(0)]] hbuf     — float * (5 × w_h × H from pass 0)
 *   [[buffer(1)]] partials — float * (grid_w × grid_h per-WG partial)
 *   [[buffer(2)]] params   — uint4 (.x=W, .y=H, .z=w_h, .w=h_v)
 *   [[buffer(3)]] consts   — float2 (.x=c1, .y=c2)
 *   [[buffer(4)]] grid_dim — uint2 (.x=grid_w, .y=grid_h)
 */

#include <metal_stdlib>
using namespace metal;

constant float G[11] = {
    0.001028f, 0.007599f, 0.036001f, 0.109361f, 0.213006f,
    0.266012f,
    0.213006f, 0.109361f, 0.036001f, 0.007599f, 0.001028f
};

#define K       11
#define K_HALF  5

/* ------------------------------------------------------------------ */
/*  Pass 0 (8 bpc): horizontal convolution from raw uint8 pixels       */
/* ------------------------------------------------------------------ */
kernel void integer_ssim_horiz_8bpc(
    const device uchar  *ref    [[buffer(0)]],
    const device uchar  *dis    [[buffer(1)]],
    device       float  *hbuf   [[buffer(2)]],
    constant     uint4  &params [[buffer(3)]],
    uint2  gid [[thread_position_in_grid]])
{
    const uint W        = params.x;
    const uint H        = params.y;
    const uint w_h      = params.z;
    const uint ref_str  = params.w;  /* ref byte stride (== dis byte stride for 8bpc) */

    (void)W;

    const uint x = gid.x;
    const uint y = gid.y;
    if (x >= w_h || y >= H) { return; }

    float mu_r = 0.0f, mu_d = 0.0f;
    float sq_r = 0.0f, sq_d = 0.0f, rd = 0.0f;

    for (int u = 0; u < K; ++u) {
        const uint src_x = x + (uint)u;
        const float r = (float)ref[y * ref_str + src_x];
        const float d = (float)dis[y * ref_str + src_x];
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
/*  Pass 0 (16 bpc): horizontal convolution from raw uint16 pixels     */
/* ------------------------------------------------------------------ */
kernel void integer_ssim_horiz_16bpc(
    const device uchar  *ref    [[buffer(0)]],
    const device uchar  *dis    [[buffer(1)]],
    device       float  *hbuf   [[buffer(2)]],
    constant     uint4  &params [[buffer(3)]],
    constant     uint   &bpcbuf [[buffer(4)]],
    uint2  gid [[thread_position_in_grid]])
{
    const uint W        = params.x;
    const uint H        = params.y;
    const uint w_h      = params.z;
    const uint ref_str  = params.w;  /* byte stride */

    (void)W;

    const uint x = gid.x;
    const uint y = gid.y;
    if (x >= w_h || y >= H) { return; }

    /* Scaler maps raw pixel → [0, 255.xxx] (matches cuda/integer_ssim_cuda.c). */
    float scaler = 1.0f;
    if (bpcbuf == 10u)       { scaler = 4.0f; }
    else if (bpcbuf == 12u)  { scaler = 16.0f; }
    else if (bpcbuf == 16u)  { scaler = 256.0f; }
    const float inv = 1.0f / scaler;

    float mu_r = 0.0f, mu_d = 0.0f;
    float sq_r = 0.0f, sq_d = 0.0f, rd = 0.0f;

    for (int u = 0; u < K; ++u) {
        const uint src_x = x + (uint)u;
        const device ushort *ref_row = (const device ushort *)(ref + y * ref_str);
        const device ushort *dis_row = (const device ushort *)(dis + y * ref_str);
        const float r = (float)ref_row[src_x] * inv;
        const float d = (float)dis_row[src_x] * inv;
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
/*  Pass 1: vertical convolution + per-pixel SSIM + per-WG reduction   */
/* ------------------------------------------------------------------ */
kernel void integer_ssim_vert_combine(
    const device float   *hbuf     [[buffer(0)]],
    device       float   *partials [[buffer(1)]],
    constant     uint4   &params   [[buffer(2)]],
    constant     float2  &consts   [[buffer(3)]],
    constant     uint2   &grid_dim [[buffer(4)]],
    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const uint w_h = params.z;  /* W - 10 */
    const uint h_v = params.w;  /* H - 10 */

    const float c1 = consts.x;
    const float c2 = consts.y;

    const uint x = gid.x;
    const uint y = gid.y;

    float my_ssim = 0.0f;
    if (x < w_h && y < h_v) {
        const uint plane_size = w_h * params.y;  /* w_h × H */

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
        const float num = (2.0f * mu_r * mu_d + c1) * (2.0f * cov_rd + c2);
        const float den = (mu_r * mu_r + mu_d * mu_d + c1) *
                          (var_r + var_d + c2);
        my_ssim = num / den;
    }

    /* Per-WG float partial — simd_sum reduction (ADR-0421 AGENTS.md invariant). */
    threadgroup float sg_sums[8];
    const float lane_sum = simd_sum(my_ssim);
    if (simd_lane == 0) { sg_sums[simd_id] = lane_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float gs = 0.0f;
        for (uint i = 0; i < simd_count; ++i) { gs += sg_sums[i]; }
        partials[bid.y * grid_dim.x + bid.x] = gs;
    }
}
