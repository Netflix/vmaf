/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernels for ssimulacra2 (T8-1k / ADR-0421).
 *
 *  Pipeline mirrors the CUDA port (ADR-0201 / ADR-0192):
 *    1. Host (Obj-C++): YUV → linear RGB via sRGB EOTF LUT.
 *    2. Host: linear RGB → XYB (verbatim port — cbrt() divergence
 *       from GPU cbrt would cascade through the pyramid; see
 *       ADR-0201 §Precision investigation).
 *    3. GPU (this file):
 *       a. ssimulacra2_mul3  — elementwise A*B over 3-plane buffer.
 *       b. ssimulacra2_blur_h / ssimulacra2_blur_v — separable IIR blur.
 *    4. Host: SSIM map + EdgeDiff combine in double precision.
 *    5. Host: 108-weight pool via the libjxl polynomial.
 *
 *  IIR bit-exact strategy (mirrors CUDA --fmad=false): Metal MSL
 *  does not have a per-kernel fmad disable flag, but floating-point
 *  operations in MSL are IEEE-754 compliant by default (no
 *  compiler-controlled FMA fusion for correctness-critical paths).
 *  The IIR loop writes temporaries identically to the CUDA kernel so
 *  the order of operations matches:
 *    ns = n2 * sum;  dp = d1 * prev1;  t = ns - dp;  o = t - prev2;
 *  This avoids the FMA path that caused ~1e-3 drift on CUDA without
 *  --fmad=false (see ssimulacra2_blur.cu header).
 *
 *  Precision contract: places=4 (max_abs_diff <= 5e-5) vs CPU on the
 *  Netflix golden pair, following ADR-0192 / ADR-0201.
 *
 *  Buffer bindings — ssimulacra2_mul3:
 *   [[buffer(0)]] a          — const float * (3-plane, plane_stride floats each)
 *   [[buffer(1)]] b          — const float *
 *   [[buffer(2)]] out        — float *
 *   [[buffer(3)]] params     — uint4 (.x=width, .y=height, .z=plane_stride, .w=0)
 *
 *  Buffer bindings — ssimulacra2_blur_h:
 *   [[buffer(0)]] in_buf     — const float * (full 3-plane buffer)
 *   [[buffer(1)]] out_buf    — float *
 *   [[buffer(2)]] rg         — IirParams (n2[3], d1[3], radius, _pad)
 *   [[buffer(3)]] dims       — uint4 (.x=width, .y=height, .z=in_off, .w=out_off)
 *
 *  Buffer bindings — ssimulacra2_blur_v:
 *   Same as blur_h but the thread index maps to column, not row.
 */

#include <metal_stdlib>
using namespace metal;

/* ------------------------------------------------------------------ */
/*  IIR parameter block (matches host Ssimu2StateMetal layout).        */
/* ------------------------------------------------------------------ */
struct IirParams {
    float n2[3];
    float d1[3];
    int   radius;
    int   _pad;
};

/* ------------------------------------------------------------------ */
/*  Kernel 1: elementwise multiply A * B → out (3-plane buffer).       */
/* ------------------------------------------------------------------ */
kernel void ssimulacra2_mul3(
    const device float  *a      [[buffer(0)]],
    const device float  *b      [[buffer(1)]],
    device       float  *out    [[buffer(2)]],
    constant     uint4  &params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint width        = params.x;
    const uint height       = params.y;
    const uint plane_stride = params.z;

    const uint x = gid.x;
    const uint y = gid.y;
    if (x >= width || y >= height) { return; }

    const uint base = y * width + x;
    for (uint c = 0u; c < 3u; ++c) {
        const uint idx = base + c * plane_stride;
        out[idx] = a[idx] * b[idx];
    }
}

/* ------------------------------------------------------------------ */
/*  Kernel 2: IIR horizontal blur — one thread per row.                */
/*  Mirrors ssimulacra2_blur_h.cu exactly (loop structure + FP ops).   */
/* ------------------------------------------------------------------ */
kernel void ssimulacra2_blur_h(
    const device float     *in_buf  [[buffer(0)]],
    device       float     *out_buf [[buffer(1)]],
    constant     IirParams &rg      [[buffer(2)]],
    constant     uint4     &dims    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint width   = dims.x;
    const uint height  = dims.y;
    const uint in_off  = dims.z;
    const uint out_off = dims.w;

    const uint row = gid;
    if (row >= height) { return; }

    const int xsize = (int)width;
    const int N     = rg.radius;

    float prev1_0 = 0.0f, prev1_1 = 0.0f, prev1_2 = 0.0f;
    float prev2_0 = 0.0f, prev2_1 = 0.0f, prev2_2 = 0.0f;

    const uint in_base  = in_off  + row * width;
    const uint out_base = out_off + row * width;

    for (int n = -N + 1; n < xsize; ++n) {
        const int left  = n - N - 1;
        const int right = n + N - 1;
        const float lv  = (left  >= 0)     ? in_buf[in_base + (uint)left]  : 0.0f;
        const float rv  = (right < xsize)  ? in_buf[in_base + (uint)right] : 0.0f;
        const float sum = lv + rv;

        /* Explicit temporaries — avoids FMA fusion, matches CUDA ordering. */
        const float ns0 = rg.n2[0] * sum;
        const float dp0 = rg.d1[0] * prev1_0;
        const float t0  = ns0 - dp0;
        const float o0  = t0  - prev2_0;

        const float ns1 = rg.n2[1] * sum;
        const float dp1 = rg.d1[1] * prev1_1;
        const float t1  = ns1 - dp1;
        const float o1  = t1  - prev2_1;

        const float ns2 = rg.n2[2] * sum;
        const float dp2 = rg.d1[2] * prev1_2;
        const float t2  = ns2 - dp2;
        const float o2  = t2  - prev2_2;

        prev2_0 = prev1_0;  prev2_1 = prev1_1;  prev2_2 = prev1_2;
        prev1_0 = o0;       prev1_1 = o1;       prev1_2 = o2;

        if (n >= 0) {
            out_buf[out_base + (uint)n] = o0 + o1 + o2;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Kernel 3: IIR vertical blur — one thread per column.               */
/*  Mirrors ssimulacra2_blur_v.cu exactly.                             */
/* ------------------------------------------------------------------ */
kernel void ssimulacra2_blur_v(
    const device float     *in_buf  [[buffer(0)]],
    device       float     *out_buf [[buffer(1)]],
    constant     IirParams &rg      [[buffer(2)]],
    constant     uint4     &dims    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint width   = dims.x;
    const uint height  = dims.y;
    const uint in_off  = dims.z;
    const uint out_off = dims.w;

    const uint col = gid;
    if (col >= width) { return; }

    const int ysize = (int)height;
    const int N     = rg.radius;

    float prev1_0 = 0.0f, prev1_1 = 0.0f, prev1_2 = 0.0f;
    float prev2_0 = 0.0f, prev2_1 = 0.0f, prev2_2 = 0.0f;

    for (int n = -N + 1; n < ysize; ++n) {
        const int left  = n - N - 1;
        const int right = n + N - 1;
        const float lv  = (left  >= 0)     ? in_buf[in_off  + (uint)left  * width + col] : 0.0f;
        const float rv  = (right < ysize)  ? in_buf[in_off  + (uint)right * width + col] : 0.0f;
        const float sum = lv + rv;

        const float ns0 = rg.n2[0] * sum;
        const float dp0 = rg.d1[0] * prev1_0;
        const float t0  = ns0 - dp0;
        const float o0  = t0  - prev2_0;

        const float ns1 = rg.n2[1] * sum;
        const float dp1 = rg.d1[1] * prev1_1;
        const float t1  = ns1 - dp1;
        const float o1  = t1  - prev2_1;

        const float ns2 = rg.n2[2] * sum;
        const float dp2 = rg.d1[2] * prev1_2;
        const float t2  = ns2 - dp2;
        const float o2  = t2  - prev2_2;

        prev2_0 = prev1_0;  prev2_1 = prev1_1;  prev2_2 = prev1_2;
        prev1_0 = o0;       prev1_1 = o1;       prev1_2 = o2;

        if (n >= 0) {
            out_buf[out_off + (uint)n * width + col] = o0 + o1 + o2;
        }
    }
}
