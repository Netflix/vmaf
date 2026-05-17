/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for float_vif (port of cuda/float_vif/float_vif_score.cu).
 *
 *  Two kernel functions:
 *
 *  1. `float_vif_compute` — at scale s, read ref+dis (raw at scale 0, float
 *     at scales 1-3), apply the scale-s separable V→H Gaussian filter inline,
 *     compute per-pixel vif_stat, reduce (num, den) per threadgroup, write to
 *     num_partials / den_partials at wg_idx = bid.y * grid_x + bid.x.
 *
 *  2. `float_vif_decimate` — for scale s, apply the scale-(s) filter at the
 *     *previous* scale's full dimensions, sample at (2*gx, 2*gy) for output.
 *     Mirrors VIF_OPT_HANDLE_BORDERS CPU branch. Output: in_w/2 × in_h/2.
 *
 *  Mirror padding conventions (must match the CUDA twin):
 *    vertical:   clamp to [0, h-1] symmetric — idx < 0 → -idx;
 *                                               idx >= h → 2*(h-1)-idx
 *    horizontal: same formula (matches convolution_edge_s AVX2 path)
 *
 *  Filter coefficient tables match CUDA float_vif_score.cu.
 *
 *  Buffer bindings for float_vif_compute:
 *   [[buffer(0)]] ref_raw   — const uchar* (scale 0 only; ignored at scale>0)
 *   [[buffer(1)]] dis_raw   — const uchar* (scale 0 only)
 *   [[buffer(2)]] ref_f     — const float* (scale > 0; row stride = scale_w)
 *   [[buffer(3)]] dis_f     — const float*
 *   [[buffer(4)]] num_parts — float* (per-WG partial numerator)
 *   [[buffer(5)]] den_parts — float* (per-WG partial denominator)
 *   [[buffer(6)]] params    — FvifComputeParams
 *
 *  Buffer bindings for float_vif_decimate:
 *   [[buffer(0)]] ref_raw   — const uchar* (scale 1 only; 0 at scale>1)
 *   [[buffer(1)]] dis_raw   — const uchar*
 *   [[buffer(2)]] ref_f     — const float* (scale > 1)
 *   [[buffer(3)]] dis_f     — const float*
 *   [[buffer(4)]] ref_out   — float* (out_w × out_h)
 *   [[buffer(5)]] dis_out   — float*
 *   [[buffer(6)]] params    — FvifDecimateParams
 */

#include <metal_stdlib>
using namespace metal;

/* ------------------------------------------------------------------ */
/*  Filter tables — must match CUDA float_vif_score.cu FVIF_COEFF_*   */
/* ------------------------------------------------------------------ */
constant float FVIF_COEFF_S0[17] = {
    0.00745626912f, 0.0142655009f, 0.0250313189f, 0.0402820669f,
    0.0594526194f,  0.0804751068f, 0.0999041125f, 0.113746084f,
    0.118773937f,   0.113746084f,  0.0999041125f, 0.0804751068f,
    0.0594526194f,  0.0402820669f, 0.0250313189f, 0.0142655009f,
    0.00745626912f
};
constant float FVIF_COEFF_S1[17] = {
    0.0189780835f, 0.0558981746f, 0.120920904f, 0.192116052f,
    0.224173605f,  0.192116052f,  0.120920904f, 0.0558981746f,
    0.0189780835f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};
constant float FVIF_COEFF_S2[17] = {
    0.054488685f, 0.244201347f, 0.402619958f, 0.244201347f, 0.054488685f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};
constant float FVIF_COEFF_S3[17] = {
    0.166378498f, 0.667243004f, 0.166378498f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

/* ------------------------------------------------------------------ */
/*  Parameter structs                                                   */
/* ------------------------------------------------------------------ */
struct FvifComputeParams {
    uint  width;
    uint  height;
    uint  scale;       /* 0-3 */
    uint  bpc;
    uint  grid_x;      /* threadgroups along X — used to compute wg_idx */
    uint  f_stride;    /* stride of ref_f / dis_f in floats (= scale_w) */
    float vif_sigma_nsq;
    float vif_enhn_gain_limit;
    uint  pad0;
    uint  pad1;
};

struct FvifDecimateParams {
    uint out_w;
    uint out_h;
    uint in_w;
    uint in_h;
    uint scale;        /* 1-3 */
    uint bpc;
    uint in_f_stride;  /* stride of ref_f/dis_f in floats (prev scale width) */
    uint out_stride;   /* stride of ref_out/dis_out in floats (= out_w) */
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */
static inline int fvif_mirror(int idx, int sup)
{
    if (idx < 0)    { return -idx; }
    if (idx >= sup) { return 2 * (sup - 1) - idx; }
    return idx;
}

static inline uint fvif_fw(uint scale)
{
    if (scale == 0u) { return 17u; }
    if (scale == 1u) { return 9u; }
    if (scale == 2u) { return 5u; }
    return 3u;
}

static inline float fvif_coeff(uint scale, uint k)
{
    if (scale == 0u) { return FVIF_COEFF_S0[k]; }
    if (scale == 1u) { return FVIF_COEFF_S1[k]; }
    if (scale == 2u) { return FVIF_COEFF_S2[k]; }
    return FVIF_COEFF_S3[k];
}

static inline float fvif_read_raw(const device uchar *plane, uint stride_bytes,
                                   int y, int x, uint bpc)
{
    float v = (float)plane[(uint)y * stride_bytes + (uint)x];
    if (bpc <= 8u) { return v - 128.0f; }
    /* 16bpc packed as uchar pairs — reinterpret as ushort. */
    const device ushort *p16 = (const device ushort *)plane;
    v = (float)p16[(uint)y * (stride_bytes / 2u) + (uint)x];
    float scaler = 1.0f;
    if (bpc == 10u)      { scaler = 4.0f; }
    else if (bpc == 12u) { scaler = 16.0f; }
    else                 { scaler = 256.0f; }
    return v / scaler - 128.0f;
}

/* ------------------------------------------------------------------ */
/*  Compute kernel                                                      */
/*  Threadgroup size 16×16 = 256 threads.                              */
/*  Tile loaded into threadgroup memory: (16 + 2*hfw)² elements.       */
/* ------------------------------------------------------------------ */

/* Worst-case hfw = 8 (scale 0).  Tile side = 16 + 2*8 = 32. */
#define FVIF_BX  16
#define FVIF_BY  16
#define FVIF_MAX_TILE 32   /* FVIF_BX + 2*8 */

kernel void float_vif_compute(
    const device uchar   *ref_raw   [[buffer(0)]],
    const device uchar   *dis_raw   [[buffer(1)]],
    const device float   *ref_f     [[buffer(2)]],
    const device float   *dis_f     [[buffer(3)]],
    device       float   *num_parts [[buffer(4)]],
    device       float   *den_parts [[buffer(5)]],
    constant FvifComputeParams &p   [[buffer(6)]],
    uint2  gid    [[thread_position_in_grid]],
    uint2  bid    [[threadgroup_position_in_grid]],
    uint2  lid2   [[thread_position_in_threadgroup]],
    uint2  tgsize [[threads_per_threadgroup]])
{
    const uint lx  = lid2.x;
    const uint ly  = lid2.y;
    const uint lid = ly * FVIF_BX + lx;

    const uint fw  = fvif_fw(p.scale);
    const uint hfw = fw / 2u;

    const uint tile_w = FVIF_BX + 2u * hfw;
    const uint tile_h = FVIF_BY + 2u * hfw;
    const int  tile_ox = (int)(bid.x * FVIF_BX) - (int)hfw;
    const int  tile_oy = (int)(bid.y * FVIF_BY) - (int)hfw;

    /* Tile storage: worst-case FVIF_MAX_TILE × FVIF_MAX_TILE = 1024 elems.
     * Threadgroup arrays are allocated for worst case at compile time. */
    threadgroup float s_ref[FVIF_MAX_TILE * FVIF_MAX_TILE];
    threadgroup float s_dis[FVIF_MAX_TILE * FVIF_MAX_TILE];
    /* Vertically-filtered intermediates: FVIF_BY output rows × tile_w cols. */
    threadgroup float s_v_mu1[FVIF_BY * FVIF_MAX_TILE];
    threadgroup float s_v_mu2[FVIF_BY * FVIF_MAX_TILE];
    threadgroup float s_v_xx [FVIF_BY * FVIF_MAX_TILE];
    threadgroup float s_v_yy [FVIF_BY * FVIF_MAX_TILE];
    threadgroup float s_v_xy [FVIF_BY * FVIF_MAX_TILE];

    const uint tile_elems = tile_w * tile_h;
    const uint tg_total   = FVIF_BX * FVIF_BY;
    const bool is_raw     = (p.scale == 0u);
    const uint raw_stride = (p.bpc <= 8u) ? p.width : (p.width * 2u);

    /* Phase 1: cooperative tile load with mirror padding. */
    for (uint i = lid; i < tile_elems; i += tg_total) {
        const uint tr = i / tile_w;
        const uint tc = i - tr * tile_w;
        const int  py = fvif_mirror((int)tile_oy + (int)tr, (int)p.height);
        const int  px = fvif_mirror((int)tile_ox + (int)tc, (int)p.width);
        float r, d;
        if (is_raw) {
            r = fvif_read_raw(ref_raw, raw_stride, py, px, p.bpc);
            d = fvif_read_raw(dis_raw, raw_stride, py, px, p.bpc);
        } else {
            r = ref_f[(uint)py * p.f_stride + (uint)px];
            d = dis_f[(uint)py * p.f_stride + (uint)px];
        }
        s_ref[tr * FVIF_MAX_TILE + tc] = r;
        s_dis[tr * FVIF_MAX_TILE + tc] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Phase 2: vertical filter for FVIF_BY output rows × tile_w cols. */
    const uint vert_total = FVIF_BY * tile_w;
    for (uint i = lid; i < vert_total; i += tg_total) {
        const uint r = i / tile_w;
        const uint c = i - r * tile_w;
        float a_mu1 = 0.0f, a_mu2 = 0.0f;
        float a_xx  = 0.0f, a_yy  = 0.0f, a_xy = 0.0f;
        for (uint k = 0u; k < fw; k++) {
            const float ck   = fvif_coeff(p.scale, k);
            const float rv   = s_ref[(r + k) * FVIF_MAX_TILE + c];
            const float dv   = s_dis[(r + k) * FVIF_MAX_TILE + c];
            a_mu1 += ck * rv;
            a_mu2 += ck * dv;
            a_xx  += ck * (rv * rv);
            a_yy  += ck * (dv * dv);
            a_xy  += ck * (rv * dv);
        }
        s_v_mu1[r * FVIF_MAX_TILE + c] = a_mu1;
        s_v_mu2[r * FVIF_MAX_TILE + c] = a_mu2;
        s_v_xx [r * FVIF_MAX_TILE + c] = a_xx;
        s_v_yy [r * FVIF_MAX_TILE + c] = a_yy;
        s_v_xy [r * FVIF_MAX_TILE + c] = a_xy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Phase 3: horizontal filter + vif_stat per pixel. */
    float my_num = 0.0f;
    float my_den = 0.0f;
    const bool valid = (gid.x < p.width && gid.y < p.height);
    if (valid) {
        float mu1 = 0.0f, mu2 = 0.0f;
        float xx  = 0.0f, yy  = 0.0f, xy = 0.0f;
        for (uint k = 0u; k < fw; k++) {
            const float ck = fvif_coeff(p.scale, k);
            mu1 += ck * s_v_mu1[ly * FVIF_MAX_TILE + (lx + k)];
            mu2 += ck * s_v_mu2[ly * FVIF_MAX_TILE + (lx + k)];
            xx  += ck * s_v_xx [ly * FVIF_MAX_TILE + (lx + k)];
            yy  += ck * s_v_yy [ly * FVIF_MAX_TILE + (lx + k)];
            xy  += ck * s_v_xy [ly * FVIF_MAX_TILE + (lx + k)];
        }

        const float eps          = 1.0e-10f;
        const float sigma_max_inv = (2.0f * 2.0f) / (255.0f * 255.0f);
        const float vif_nsq       = p.vif_sigma_nsq;
        const float vif_egl       = p.vif_enhn_gain_limit;

        float sigma1_sq = xx - mu1 * mu1;
        float sigma2_sq = yy - mu2 * mu2;
        float sigma12   = xy - mu1 * mu2;
        sigma1_sq = max(sigma1_sq, 0.0f);
        sigma2_sq = max(sigma2_sq, 0.0f);

        float g     = sigma12 / (sigma1_sq + eps);
        float sv_sq = sigma2_sq - g * sigma12;

        if (sigma1_sq < eps) {
            g       = 0.0f;
            sv_sq   = sigma2_sq;
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
        sv_sq = max(sv_sq, eps);
        g     = min(g, vif_egl);

        float num_val = log2(1.0f + (g * g * sigma1_sq) / (sv_sq + vif_nsq));
        float den_val = log2(1.0f + sigma1_sq / vif_nsq);

        if (sigma12 < 0.0f) {
            num_val = 0.0f;
        }
        if (sigma1_sq < vif_nsq) {
            num_val = 1.0f - sigma2_sq * sigma_max_inv;
            den_val = 1.0f;
        }
        my_num = num_val;
        my_den = den_val;
    }

    /* Phase 4: threadgroup reduction via simd_sum + threadgroup array. */
    const uint simd_count = tgsize.x * tgsize.y / 32u;
    threadgroup float tg_num[8];  /* up to 8 simd groups for 256-thread WG */
    threadgroup float tg_den[8];

    const uint simd_lane = lid % 32u;
    const uint simd_id   = lid / 32u;

    const float lane_num = simd_sum(my_num);
    const float lane_den = simd_sum(my_den);

    if (simd_lane == 0u) {
        tg_num[simd_id] = lane_num;
        tg_den[simd_id] = lane_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        float tot_num = 0.0f, tot_den = 0.0f;
        for (uint i = 0u; i < simd_count; i++) {
            tot_num += tg_num[i];
            tot_den += tg_den[i];
        }
        const uint wg_idx = bid.y * p.grid_x + bid.x;
        num_parts[wg_idx] = tot_num;
        den_parts[wg_idx] = tot_den;
    }
}

/* ------------------------------------------------------------------ */
/*  Decimate kernel                                                     */
/*  One thread per output pixel (gx < out_w, gy < out_h).             */
/*  Reads from the previous scale's buffer, samples at (2*gx, 2*gy).  */
/* ------------------------------------------------------------------ */
kernel void float_vif_decimate(
    const device uchar   *ref_raw  [[buffer(0)]],
    const device uchar   *dis_raw  [[buffer(1)]],
    const device float   *ref_f    [[buffer(2)]],
    const device float   *dis_f    [[buffer(3)]],
    device       float   *ref_out  [[buffer(4)]],
    device       float   *dis_out  [[buffer(5)]],
    constant FvifDecimateParams &p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.out_w || gid.y >= p.out_h) { return; }

    const uint fw  = fvif_fw(p.scale);
    const uint hfw = fw / 2u;
    const int  in_x = (int)(2u * gid.x);
    const int  in_y = (int)(2u * gid.y);
    const bool is_raw = (p.scale == 1u);
    const uint raw_stride = (p.bpc <= 8u) ? p.in_w : (p.in_w * 2u);

    /* V-inner / H-outer ordering matches CPU vif_filter1d_s. */
    float acc_ref = 0.0f, acc_dis = 0.0f;
    for (uint kj = 0u; kj < fw; kj++) {
        const float c_j = fvif_coeff(p.scale, kj);
        const int px = fvif_mirror(in_x - (int)hfw + (int)kj, (int)p.in_w);
        float v_ref = 0.0f, v_dis = 0.0f;
        for (uint ki = 0u; ki < fw; ki++) {
            const float c_i = fvif_coeff(p.scale, ki);
            const int py = fvif_mirror(in_y - (int)hfw + (int)ki, (int)p.in_h);
            float r, d;
            if (is_raw) {
                r = fvif_read_raw(ref_raw, raw_stride, py, px, p.bpc);
                d = fvif_read_raw(dis_raw, raw_stride, py, px, p.bpc);
            } else {
                r = ref_f[(uint)py * p.in_f_stride + (uint)px];
                d = dis_f[(uint)py * p.in_f_stride + (uint)px];
            }
            v_ref += c_i * r;
            v_dis += c_i * d;
        }
        acc_ref += c_j * v_ref;
        acc_dis += c_j * v_dis;
    }

    ref_out[gid.y * p.out_stride + gid.x] = acc_ref;
    dis_out[gid.y * p.out_stride + gid.x] = acc_dis;
}
