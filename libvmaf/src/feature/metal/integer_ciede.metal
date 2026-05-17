/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernel for the ciede2000 feature extractor.
 *  Ported from libvmaf/src/feature/cuda/integer_ciede/ciede_score.cu
 *  (T7-23 / ADR-0182). Mirrors the CUDA and Vulkan twins; float
 *  precision, places=4 empirical floor on real hardware (ADR-0187).
 *
 *  Algorithm: each pixel pair is converted YUV -> BT.709 RGB ->
 *  XYZ -> L*a*b*; CIEDE2000 dE is computed per-pixel. Per-threadgroup
 *  sums are written to a float partial buffer; host accumulates in
 *  double and applies 45 - 20*log10(mean_dE).
 *
 *  Subsampling: kernel grid at luma resolution. Chroma read at
 *  (x >> ss_hor, y >> ss_ver), nearest-neighbour — matches CPU.
 *
 *  Buffer bindings (same for 8bpc and 16bpc variants):
 *   [[buffer(0)]] ref_y   — const uchar *  (row-major, byte-addressed)
 *   [[buffer(1)]] ref_u   — const uchar *
 *   [[buffer(2)]] ref_v   — const uchar *
 *   [[buffer(3)]] dis_y   — const uchar *
 *   [[buffer(4)]] dis_u   — const uchar *
 *   [[buffer(5)]] dis_v   — const uchar *
 *   [[buffer(6)]] partials — float * (grid_w x grid_h per-WG float sums)
 *   [[buffer(7)]] params  — uint4 { width, height, bpc, ss_hor | (ss_ver<<16) }
 *   [[buffer(8)]] strides — uint4 { ref_y_stride, ref_uv_stride,
 *                                    dis_y_stride, dis_uv_stride }
 */

#include <metal_stdlib>
using namespace metal;

/* ------------------------------------------------------------------ */
/*  Math helpers (ported from ciede_score.cu)                           */
/* ------------------------------------------------------------------ */

static inline float srgb_to_linear(float c)
{
    if (c > (10.0f / 255.0f)) {
        const float A = 0.055f;
        const float D = 1.0f / 1.055f;
        return powr(fmax(0.0f, (c + A) * D), 2.4f);
    }
    return c / 12.92f;
}

static inline float xyz_to_lab_map(float t)
{
    if (t > 0.008856f)
        return pow(t, 1.0f / 3.0f);
    return 7.787f * t + (16.0f / 116.0f);
}

static inline void yuv_to_lab(float y_lim, float u_lim, float v_lim, uint bpc,
                               thread float &L, thread float &A, thread float &B)
{
    float scale = 1.0f;
    if (bpc == 10u) scale = 4.0f;
    else if (bpc == 12u) scale = 16.0f;
    else if (bpc == 16u) scale = 256.0f;

    float y = (y_lim - 16.0f * scale) * (1.0f / (219.0f * scale));
    float u = (u_lim - 128.0f * scale) * (1.0f / (224.0f * scale));
    float v = (v_lim - 128.0f * scale) * (1.0f / (224.0f * scale));

    float r = y + 1.28033f * v;
    float g = y - 0.21482f * u - 0.38059f * v;
    float b = y + 2.12798f * u;

    r = srgb_to_linear(r);
    g = srgb_to_linear(g);
    b = srgb_to_linear(b);

    float x  = r * 0.4124564390896921f  + g * 0.357576077643909f   + b * 0.18043748326639894f;
    float yy = r * 0.21267285140562248f + g * 0.715152155287818f   + b * 0.07217499330655958f;
    float z  = r * 0.019333895582329317f + g * 0.119192025881303f  + b * 0.9503040785363677f;

    x  *= 1.0f / 0.95047f;
    z  *= 1.0f / 1.08883f;

    float lx = xyz_to_lab_map(x);
    float ly = xyz_to_lab_map(yy);
    float lz = xyz_to_lab_map(z);

    L = 116.0f * ly - 16.0f;
    A = 500.0f * (lx - ly);
    B = 200.0f * (ly - lz);
}

static inline float get_h_prime(float b, float a)
{
    if (b == 0.0f && a == 0.0f)
        return 0.0f;
    float h = atan2(b, a);
    if (h < 0.0f)
        h += 6.283185307179586f;
    return h * (180.0f / M_PI_F);
}

static inline float get_delta_h_prime(float c1, float c2, float h1, float h2)
{
    if (c1 * c2 == 0.0f)
        return 0.0f;
    float diff = h2 - h1;
    if (fabs(diff) <= 180.0f)
        return diff * (M_PI_F / 180.0f);
    if (diff > 180.0f)
        return (diff - 360.0f) * (M_PI_F / 180.0f);
    return (diff + 360.0f) * (M_PI_F / 180.0f);
}

static inline float get_upcase_h_bar_prime(float h1, float h2)
{
    float diff = fabs(h1 - h2);
    if (diff > 180.0f)
        return ((h1 + h2 + 360.0f) / 2.0f) * (M_PI_F / 180.0f);
    return ((h1 + h2) / 2.0f) * (M_PI_F / 180.0f);
}

static inline float get_upcase_t(float h_bar)
{
    return 1.0f
        - 0.17f * cos(h_bar - M_PI_F / 6.0f)
        + 0.24f * cos(2.0f * h_bar)
        + 0.32f * cos(3.0f * h_bar + M_PI_F / 30.0f)
        - 0.20f * cos(4.0f * h_bar - 63.0f * M_PI_F / 180.0f);
}

static inline float get_r_sub_t(float c_bar, float h_bar)
{
    float exp_arg = -powr((h_bar * (180.0f / M_PI_F) - 275.0f) / 25.0f, 2.0f);
    float c7      = powr(c_bar, 7.0f);
    float r_c     = 2.0f * sqrt(c7 / (c7 + powr(25.0f, 7.0f)));
    return -sin(60.0f * M_PI_F / 180.0f * exp(exp_arg)) * r_c;
}

static inline float ciede2000(float l1, float a1, float b1,
                               float l2, float a2, float b2)
{
    const float k_l = 0.65f;
    const float k_c = 1.0f;
    const float k_h = 4.0f;

    float dl_p  = l2 - l1;
    float l_bar = 0.5f * (l1 + l2);
    float c1    = sqrt(a1 * a1 + b1 * b1);
    float c2    = sqrt(a2 * a2 + b2 * b2);
    float c_bar = 0.5f * (c1 + c2);

    float c_bar_7  = powr(c_bar, 7.0f);
    float g_factor = 1.0f - sqrt(c_bar_7 / (c_bar_7 + powr(25.0f, 7.0f)));

    float a1_p = a1 + 0.5f * a1 * g_factor;
    float a2_p = a2 + 0.5f * a2 * g_factor;
    float c1_p = sqrt(a1_p * a1_p + b1 * b1);
    float c2_p = sqrt(a2_p * a2_p + b2 * b2);
    float c_bar_p = 0.5f * (c1_p + c2_p);
    float dc_p    = c2_p - c1_p;

    float dl2 = (l_bar - 50.0f) * (l_bar - 50.0f);
    float s_l  = 1.0f + (0.015f * dl2) / sqrt(20.0f + dl2);
    float s_c  = 1.0f + 0.045f * c_bar_p;

    float h1_p = get_h_prime(b1, a1_p);
    float h2_p = get_h_prime(b2, a2_p);
    float dh_p = get_delta_h_prime(c1, c2, h1_p, h2_p);
    float dH_p = 2.0f * sqrt(c1_p * c2_p) * sin(dh_p / 2.0f);

    float H_bar_p = get_upcase_h_bar_prime(h1_p, h2_p);
    float t_term  = get_upcase_t(H_bar_p);
    float s_h     = 1.0f + 0.015f * c_bar_p * t_term;
    float r_t     = get_r_sub_t(c_bar_p, H_bar_p);

    float lightness = dl_p / (k_l * s_l);
    float chroma    = dc_p / (k_c * s_c);
    float hue       = dH_p / (k_h * s_h);

    return sqrt(lightness * lightness + chroma * chroma + hue * hue
                + r_t * chroma * hue);
}

/* ------------------------------------------------------------------ */
/*  8 bpc kernel                                                        */
/* ------------------------------------------------------------------ */
kernel void integer_ciede_kernel_8bpc(
    const device uchar  *ref_y    [[buffer(0)]],
    const device uchar  *ref_u    [[buffer(1)]],
    const device uchar  *ref_v    [[buffer(2)]],
    const device uchar  *dis_y    [[buffer(3)]],
    const device uchar  *dis_u    [[buffer(4)]],
    const device uchar  *dis_v    [[buffer(5)]],
    device       float  *partials [[buffer(6)]],
    constant     uint4  &params   [[buffer(7)]],  /* width, height, bpc, ss_packed */
    constant     uint4  &strides  [[buffer(8)]],  /* ry_st, ruv_st, dy_st, duv_st */
    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint2  grid_groups [[threadgroups_per_grid]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const uint width   = params.x;
    const uint height  = params.y;
    const uint bpc     = params.z;
    const uint ss_hor  = params.w & 0xFFFFu;
    const uint ss_ver  = params.w >> 16u;

    float my_de = 0.0f;
    if (gid.x < width && gid.y < height) {
        const uint cx = ss_hor ? (gid.x >> 1u) : gid.x;
        const uint cy = ss_ver ? (gid.y >> 1u) : gid.y;

        const float ry = (float)ref_y[gid.y * strides.x + gid.x];
        const float ru = (float)ref_u[cy * strides.y + cx];
        const float rv = (float)ref_v[cy * strides.y + cx];
        const float dy = (float)dis_y[gid.y * strides.z + gid.x];
        const float du = (float)dis_u[cy * strides.w + cx];
        const float dv = (float)dis_v[cy * strides.w + cx];

        float l1, a1, b1, l2, a2, b2;
        yuv_to_lab(ry, ru, rv, bpc, l1, a1, b1);
        yuv_to_lab(dy, du, dv, bpc, l2, a2, b2);
        my_de = ciede2000(l1, a1, b1, l2, a2, b2);
    }

    /* Per-WG partial sum using simd_sum, written as float (matches CUDA). */
    threadgroup float sg[8];
    const float lane_sum = simd_sum(my_de);
    if (simd_lane == 0)
        sg[simd_id] = lane_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < simd_count; ++i)
            block_sum += sg[i];
        const uint idx = bid.y * grid_groups.x + bid.x;
        partials[idx] = block_sum;
    }
}

/* ------------------------------------------------------------------ */
/*  16 bpc kernel                                                       */
/* ------------------------------------------------------------------ */
kernel void integer_ciede_kernel_16bpc(
    const device uchar  *ref_y    [[buffer(0)]],
    const device uchar  *ref_u    [[buffer(1)]],
    const device uchar  *ref_v    [[buffer(2)]],
    const device uchar  *dis_y    [[buffer(3)]],
    const device uchar  *dis_u    [[buffer(4)]],
    const device uchar  *dis_v    [[buffer(5)]],
    device       float  *partials [[buffer(6)]],
    constant     uint4  &params   [[buffer(7)]],
    constant     uint4  &strides  [[buffer(8)]],
    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint2  grid_groups [[threadgroups_per_grid]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const uint width   = params.x;
    const uint height  = params.y;
    const uint bpc     = params.z;
    const uint ss_hor  = params.w & 0xFFFFu;
    const uint ss_ver  = params.w >> 16u;

    float my_de = 0.0f;
    if (gid.x < width && gid.y < height) {
        const uint cx = ss_hor ? (gid.x >> 1u) : gid.x;
        const uint cy = ss_ver ? (gid.y >> 1u) : gid.y;

        const device ushort *ry_row = (const device ushort *)(ref_y + gid.y * strides.x);
        const device ushort *ru_row = (const device ushort *)(ref_u + cy * strides.y);
        const device ushort *rv_row = (const device ushort *)(ref_v + cy * strides.y);
        const device ushort *dy_row = (const device ushort *)(dis_y + gid.y * strides.z);
        const device ushort *du_row = (const device ushort *)(dis_u + cy * strides.w);
        const device ushort *dv_row = (const device ushort *)(dis_v + cy * strides.w);

        const float ry = (float)ry_row[gid.x];
        const float ru = (float)ru_row[cx];
        const float rv = (float)rv_row[cx];
        const float dy = (float)dy_row[gid.x];
        const float du = (float)du_row[cx];
        const float dv = (float)dv_row[cx];

        float l1, a1, b1, l2, a2, b2;
        yuv_to_lab(ry, ru, rv, bpc, l1, a1, b1);
        yuv_to_lab(dy, du, dv, bpc, l2, a2, b2);
        my_de = ciede2000(l1, a1, b1, l2, a2, b2);
    }

    threadgroup float sg[8];
    const float lane_sum = simd_sum(my_de);
    if (simd_lane == 0)
        sg[simd_id] = lane_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < simd_count; ++i)
            block_sum += sg[i];
        const uint idx = bid.y * grid_groups.x + bid.x;
        partials[idx] = block_sum;
    }
}
