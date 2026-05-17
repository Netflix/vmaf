/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute shaders for the CAMBI banding-detection feature extractor
 *  (Strategy II hybrid — mirrors the CUDA twin in
 *  `libvmaf/src/feature/cuda/integer_cambi/cambi_score.cu`).
 *
 *  Three kernels:
 *
 *    cambi_spatial_mask_kernel
 *      - Bit-exact port of cambi_score.cu::cambi_spatial_mask_kernel.
 *      - One thread per output pixel; 7×7 naive box-sum of zero_deriv.
 *      - mask[y][x] = (box_sum > mask_index) ? 1 : 0.
 *      - Buffer bindings match integer_cambi_metal.mm dispatch.
 *
 *    cambi_decimate_kernel
 *      - Strict 2× stride-2 subsample (matches cambi.c::decimate).
 *      - Output pixel (x,y) = input pixel (2x, 2y). No filtering.
 *
 *    cambi_filter_mode_kernel
 *      - Separable 3-tap mode filter.
 *      - axis=0 → horizontal pass, axis=1 → vertical pass.
 *      - mode3(a,b,c): if two are equal, that value; else min of three.
 *      - Matches cambi_score.cu::cambi_filter_mode_kernel exactly.
 *
 *  All buffers are flat uint16 arrays (ushort in MSL). Stride is in
 *  ushort words (same as CUDA stride_words).
 *
 *  Precision target: ULP=0 vs CPU scalar (places=4 cross-backend gate,
 *  per ADR-0214). All operations are integer — bit-exact by construction.
 */

#include <metal_stdlib>
using namespace metal;

/* ------------------------------------------------------------------ */
/* Kernel 1: Spatial mask                                              */
/* ------------------------------------------------------------------ */

/*
 * Buffer bindings (host: integer_cambi_metal.mm):
 *   [[buffer(0)]] image       — const device ushort*, flat uint16 luma
 *   [[buffer(1)]] mask        — device ushort*, output mask
 *   [[buffer(2)]] params      — uint4: (width, height, stride_words, mask_index)
 */
kernel void cambi_spatial_mask_kernel(
    const device ushort   *image   [[buffer(0)]],
    device       ushort   *mask    [[buffer(1)]],
    constant     uint4    &params  [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int width        = (int)params.x;
    const int height       = (int)params.y;
    const int stride_words = (int)params.z;
    const uint mask_index  = params.w;

    const int x = (int)gid.x;
    const int y = (int)gid.y;
    if (x >= width || y >= height) {
        return;
    }

    /* 7×7 box-sum of zero_deriv around (x,y).
     * zero_deriv[ry][rx] = (image[ry][rx] == image[ry][rx+1])
     *                   && (image[ry][rx] == image[ry+1][rx]).
     * Border pixels treat out-of-bounds as "equal" — matches CUDA. */
    constexpr int HALF = 3; /* (MASK_FILTER_SIZE=7) / 2 */
    uint box_sum = 0u;

    for (int dy = -HALF; dy <= HALF; dy++) {
        int ry = y + dy;
        if (ry < 0) { ry = 0; }
        if (ry >= height) { ry = height - 1; }

        for (int dx = -HALF; dx <= HALF; dx++) {
            int rx = x + dx;
            if (rx < 0) { rx = 0; }
            if (rx >= width) { rx = width - 1; }

            const ushort p = image[(uint)(ry * stride_words + rx)];

            const int rx_r = (rx == width - 1) ? rx : rx + 1;
            const ushort r = image[(uint)(ry * stride_words + rx_r)];

            const int ry_b = (ry == height - 1) ? ry : ry + 1;
            const ushort b = image[(uint)(ry_b * stride_words + rx)];

            const bool eq_right = (rx == width - 1) || (p == r);
            const bool eq_below = (ry == height - 1) || (p == b);
            box_sum += (eq_right && eq_below) ? 1u : 0u;
        }
    }

    mask[(uint)(y * stride_words + x)] = (box_sum > mask_index) ? (ushort)1u : (ushort)0u;
}

/* ------------------------------------------------------------------ */
/* Kernel 2: 2× decimate                                               */
/* ------------------------------------------------------------------ */

/*
 * Buffer bindings (host: integer_cambi_metal.mm):
 *   [[buffer(0)]] src            — const device ushort*
 *   [[buffer(1)]] dst            — device ushort*
 *   [[buffer(2)]] stride_params  — uint4: (out_width, out_height,
 *                                          src_stride_words, dst_stride_words)
 */
kernel void cambi_decimate_kernel(
    const device ushort   *src           [[buffer(0)]],
    device       ushort   *dst           [[buffer(1)]],
    constant     uint4    &stride_params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint out_width        = stride_params.x;
    const uint out_height       = stride_params.y;
    const uint src_stride_words = stride_params.z;
    const uint dst_stride_words = stride_params.w;

    const uint x = gid.x;
    const uint y = gid.y;
    if (x >= out_width || y >= out_height) {
        return;
    }

    /* Strict stride-2 subsample — matches cambi.c::decimate. */
    dst[y * dst_stride_words + x] = src[(y * 2u) * src_stride_words + x * 2u];
}

/* ------------------------------------------------------------------ */
/* Kernel 3: Separable 3-tap mode filter                               */
/* ------------------------------------------------------------------ */

/*
 * mode3(a, b, c): value appearing at least twice, or min if all distinct.
 * Matches cambi_score.cu::mode3_dev exactly.
 */
static inline ushort mode3_msl(ushort a, ushort b, ushort c)
{
    if (a == b || a == c) { return a; }
    if (b == c)           { return b; }
    /* All distinct — return minimum. */
    if (a < b) {
        return (a < c) ? a : c;
    } else {
        return (b < c) ? b : c;
    }
}

/*
 * Buffer bindings (host: integer_cambi_metal.mm):
 *   [[buffer(0)]] in        — const device ushort*
 *   [[buffer(1)]] out       — device ushort*
 *   [[buffer(2)]] params    — uint4: (width, height, stride_words, axis)
 *                             axis=0 → horizontal, axis=1 → vertical
 */
kernel void cambi_filter_mode_kernel(
    const device ushort   *in      [[buffer(0)]],
    device       ushort   *out     [[buffer(1)]],
    constant     uint4    &params  [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int width        = (int)params.x;
    const int height       = (int)params.y;
    const int stride_words = (int)params.z;
    const uint axis        = params.w;

    const int x = (int)gid.x;
    const int y = (int)gid.y;
    if (x >= width || y >= height) {
        return;
    }

    ushort a, b, c;
    if (axis == 0u) {
        /* Horizontal: neighbours in x. */
        const int xl = (x > 0) ? x - 1 : 0;
        const int xr = (x < width - 1) ? x + 1 : width - 1;
        a = in[(uint)(y * stride_words + xl)];
        b = in[(uint)(y * stride_words + x)];
        c = in[(uint)(y * stride_words + xr)];
    } else {
        /* Vertical: neighbours in y. */
        const int yu = (y > 0) ? y - 1 : 0;
        const int yd = (y < height - 1) ? y + 1 : height - 1;
        a = in[(uint)(yu * stride_words + x)];
        b = in[(uint)(y  * stride_words + x)];
        c = in[(uint)(yd * stride_words + x)];
    }
    out[(uint)(y * stride_words + x)] = mode3_msl(a, b, c);
}
