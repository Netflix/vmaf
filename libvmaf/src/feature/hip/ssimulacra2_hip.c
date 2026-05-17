/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ssimulacra2 feature extractor on the HIP backend.
 *  Ninth consumer of the HIP feature surface; direct port of
 *  `libvmaf/src/feature/cuda/ssimulacra2_cuda.c`.
 *
 *  Pipeline mirrors the CUDA twin (per ADR-0201):
 *    1. Host: YUV → linear RGB (deterministic LUT sRGB EOTF).
 *    2. Per-scale (up to 6, breaks early when min-dim < 8):
 *       a. Host: linear RGB → XYB (float cbrt via host libm for precision).
 *       b. GPU: 3 elementwise 3-plane multiplies (ref², dis², ref·dis).
 *       c. GPU: 5 separable IIR blurs (s11, s22, s12, mu1, mu2).
 *       d. Host: per-pixel SSIM + EdgeDiff combine in double precision.
 *       e. Host: 2×2 box downsample for the next scale.
 *    3. Host: pool 108 weighted norms via the libjxl polynomial.
 *
 *  HIP adaptation from CUDA:
 *  - `hipModuleLoadData` / `hipModuleGetFunction` / `hipModuleLaunchKernel`
 *    instead of `cuModuleLoadData` / `cuModuleGetFunction` / `cuLaunchKernel`.
 *  - Device buffers allocated via `hipMalloc` (raw void *) instead of
 *    `vmaf_cuda_buffer_alloc` (VmafCudaBuffer * wrapper).
 *  - Pinned host buffers via `hipHostMalloc` / `hipHostFree`.
 *  - 2D async copies use `hipMemcpy2DAsync` instead of `cuMemcpy2DAsync`.
 *  - Stream: `hipStreamCreateWithFlags` / `hipStreamDestroy`.
 *  - ENOSYS scaffold when `HAVE_HIPCC` is not defined.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "mem.h"

#include "feature/ssimulacra2_math.h"
#include "picture.h"
#include "ssimulacra2_hip.h"

#define SS2H_NUM_SCALES 6
#define SS2H_BLUR_BLOCK 64
#define SS2H_MUL_BX 16
#define SS2H_MUL_BY 8
#define SS2H_SIGMA 1.5
#define SS2H_PI 3.14159265358979323846

enum yuv_matrix_h {
    SS2H_MATRIX_BT709_LIMITED = 0,
    SS2H_MATRIX_BT601_LIMITED = 1,
    SS2H_MATRIX_BT709_FULL = 2,
    SS2H_MATRIX_BT601_FULL = 3,
};

/* libjxl 108 pooling weights — bit-identical to ssimulacra2.c::kWeights. */
static const double g_weights[108] = {
    0.0,
    0.0007376606707406586,
    0.0,
    0.0,
    0.0007793481682867309,
    0.0,
    0.0,
    0.0004371155730107379,
    0.0,
    1.1041726426657346,
    0.00066284834129271,
    0.00015231632783718752,
    0.0,
    0.0016406437456599754,
    0.0,
    1.8422455520539298,
    11.441172603757666,
    0.0,
    0.0007989109436015163,
    0.000176816438078653,
    0.0,
    1.8787594979546387,
    10.94906990605142,
    0.0,
    0.0007289346991508072,
    0.9677937080626833,
    0.0,
    0.00014003424285435884,
    0.9981766977854967,
    0.00031949755934435053,
    0.0004550992113792063,
    0.0,
    0.0,
    0.0013648766163243398,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466890328078848,
    0.0,
    17.445833984131262,
    0.0006235601634041466,
    0.0,
    0.0,
    6.683678146179332,
    0.00037724407979611296,
    1.027889937768264,
    225.20515300849274,
    0.0,
    0.0,
    19.213238186143016,
    0.0011401524586618361,
    0.001237755635509985,
    176.39317598450694,
    0.0,
    0.0,
    24.43300999870476,
    0.28520802612117757,
    0.0004485436923833408,
    0.0,
    0.0,
    0.0,
    34.77906344483772,
    44.835625328877896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0008680556573291698,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0005313191874358747,
    0.0,
    0.00016533814161379112,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0004179171803251336,
    0.0017290828234722833,
    0.0,
    0.0020827005846636437,
    0.0,
    0.0,
    8.826982764996862,
    23.19243343998926,
    0.0,
    95.1080498811086,
    0.9863978034400682,
    0.9834382792465353,
    0.0012286405048278493,
    171.2667255897307,
    0.9807858872435379,
    0.0,
    0.0,
    0.0,
    0.0005130064588990679,
    0.0,
    0.00010854057858411537,
};

typedef struct Ssimu2StateHip {
    /* Options. */
    int yuv_matrix;

    /* Geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned scale_w[SS2H_NUM_SCALES];
    unsigned scale_h[SS2H_NUM_SCALES];

    /* Recursive Gaussian coefficients (sigma=1.5). */
    float rg_n2[3];
    float rg_d1[3];
    int rg_radius;

    /* HIP module + kernel handles. */
    hipModule_t module_blur;
    hipModule_t module_mul;
    hipFunction_t func_blur_h;
    hipFunction_t func_blur_v;
    hipFunction_t func_mul3;
    hipStream_t str;

    /* Device buffers (raw hipMalloc pointers, 3-plane contiguous). */
    void *d_ref_xyb;
    void *d_dis_xyb;
    void *d_mul_buf;
    void *d_blur_scratch;
    void *d_mu1;
    void *d_mu2;
    void *d_s11;
    void *d_s22;
    void *d_s12;

    /* Pinned host buffers (hipHostMalloc). */
    float *h_ref_lin;
    float *h_dis_lin;
    float *h_ref_lin_ds;
    float *h_dis_lin_ds;
    float *h_ref_xyb;
    float *h_dis_xyb;
    float *h_mu1;
    float *h_mu2;
    float *h_s11;
    float *h_s22;
    float *h_s12;

    /* Pinned raw-YUV scratch for D2H copy of picture planes. */
    void *h_ref_raw[3];
    void *h_dis_raw[3];
    size_t raw_plane_bytes[3];
    unsigned plane_w[3];
    unsigned plane_h[3];
    ptrdiff_t plane_row_bytes[3];
} Ssimu2StateHip;

static const VmafOption options[] = {
    {
        .name = "yuv_matrix",
        .help = "YUV→RGB matrix: 0=bt709_limited (default), 1=bt601_limited, "
                "2=bt709_full, 3=bt601_full",
        .offset = offsetof(Ssimu2StateHip, yuv_matrix),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = SS2H_MATRIX_BT709_LIMITED,
        .min = 0,
        .max = 3,
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* Recursive Gaussian setup (verbatim port of ss2c_setup_gaussian).   */
/* Splitting would break the line-for-line scalar-diff audit trail     */
/* (ADR-0141 §2 upstream-parity load-bearing invariant).              */
/* NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
/* ------------------------------------------------------------------ */
static void ss2h_setup_gaussian(Ssimu2StateHip *s, double sigma)
{
    const double radius = round(3.2795 * sigma + 0.2546);
    const double pi_div_2r = SS2H_PI / (2.0 * radius);
    const double omega[3] = {pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r};

    const double p1 = +1.0 / tan(0.5 * omega[0]);
    const double p3 = -1.0 / tan(0.5 * omega[1]);
    const double p5 = +1.0 / tan(0.5 * omega[2]);
    const double r1 = +p1 * p1 / sin(omega[0]);
    const double r3 = -p3 * p3 / sin(omega[1]);
    const double r5 = +p5 * p5 / sin(omega[2]);

    const double neg_half_sigma2 = -0.5 * sigma * sigma;
    const double recip_r = 1.0 / radius;
    double rho[3];
    for (int i = 0; i < 3; i++)
        rho[i] = exp(neg_half_sigma2 * omega[i] * omega[i]) * recip_r;

    const double D13 = p1 * r3 - r1 * p3;
    const double D35 = p3 * r5 - r3 * p5;
    const double D51 = p5 * r1 - r5 * p1;
    const double recip_d13 = 1.0 / D13;
    const double zeta_15 = D35 * recip_d13;
    const double zeta_35 = D51 * recip_d13;

    const double A[3][3] = {{p1, p3, p5}, {r1, r3, r5}, {zeta_15, zeta_35, 1.0}};
    const double gamma[3] = {1.0, radius * radius - sigma * sigma,
                             zeta_15 * rho[0] + zeta_35 * rho[1] + rho[2]};
    const double det_A = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                         A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                         A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    const double inv_det = 1.0 / det_A;

    double beta[3];
    for (int col = 0; col < 3; col++) {
        double M[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                M[i][j] = A[i][j];
        for (int i = 0; i < 3; i++)
            M[i][col] = gamma[i];
        beta[col] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                     M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                     M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) *
                    inv_det;
    }

    s->rg_radius = (int)radius;
    for (int i = 0; i < 3; i++) {
        s->rg_n2[i] = (float)(-beta[i] * cos(omega[i] * (radius + 1.0)));
        s->rg_d1[i] = (float)(-2.0 * cos(omega[i]));
    }
}

/* ------------------------------------------------------------------ */
/* Host-side helpers (shared with CUDA twin, same logic).             */
/* ------------------------------------------------------------------ */

static inline float ss2h_clampf(float v, float lo, float hi)
{
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

static inline float ss2h_read_plane(const VmafPicture *pic, int plane, int x, int y)
{
    unsigned pw = pic->w[plane];
    unsigned ph = pic->h[plane];
    unsigned lw = pic->w[0];
    unsigned lh = pic->h[0];
    int sx = (pw == lw)     ? x :
             (pw * 2 == lw) ? (x >> 1) :
                              (int)((int64_t)x * (int64_t)pw / (int64_t)lw);
    int sy = (ph == lh)     ? y :
             (ph * 2 == lh) ? (y >> 1) :
                              (int)((int64_t)y * (int64_t)ph / (int64_t)lh);
    if (sx < 0)
        sx = 0;
    if (sy < 0)
        sy = 0;
    if ((unsigned)sx >= pw)
        sx = (int)pw - 1;
    if ((unsigned)sy >= ph)
        sy = (int)ph - 1;
    if (pic->bpc > 8) {
        const uint16_t *row =
            (const uint16_t *)((const uint8_t *)pic->data[plane] + (size_t)sy * pic->stride[plane]);
        return (float)row[sx];
    }
    const uint8_t *row = (const uint8_t *)pic->data[plane] + (size_t)sy * pic->stride[plane];
    return (float)row[sx];
}

/* Verbatim port of ssimulacra2.c::picture_to_linear_rgb.
 * Splitting would break the line-for-line scalar-diff audit trail
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5 sweep
 * closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2h_picture_to_linear_rgb(const Ssimu2StateHip *s, const VmafPicture *pic, float *out)
{
    const unsigned w = s->width;
    const unsigned h = s->height;
    const size_t plane_sz = (size_t)w * (size_t)h;
    float *rp = out;
    float *gp = out + plane_sz;
    float *bp = out + 2 * plane_sz;

    const float peak = (float)((1u << s->bpc) - 1u);
    const float inv_peak = 1.0f / peak;

    float kr;
    float kg;
    float kb;
    int limited = 1;
    switch (s->yuv_matrix) {
    case SS2H_MATRIX_BT709_FULL:
        limited = 0;
        // fallthrough
    case SS2H_MATRIX_BT709_LIMITED:
        kr = 0.2126f;
        kg = 0.7152f;
        kb = 0.0722f;
        break;
    case SS2H_MATRIX_BT601_FULL:
        limited = 0;
        // fallthrough
    case SS2H_MATRIX_BT601_LIMITED:
    default:
        kr = 0.299f;
        kg = 0.587f;
        kb = 0.114f;
        break;
    }
    const float cr_r = 2.0f * (1.0f - kr);
    const float cb_b = 2.0f * (1.0f - kb);
    const float cb_g = -(2.0f * kb * (1.0f - kb)) / kg;
    const float cr_g = -(2.0f * kr * (1.0f - kr)) / kg;
    const float y_scale = limited ? (255.0f / 219.0f) : 1.0f;
    const float c_scale = limited ? (255.0f / 224.0f) : 1.0f;
    const float y_off = limited ? (16.0f / 255.0f) : 0.0f;
    const float c_off = 0.5f;

    for (unsigned y = 0; y < h; y++) {
        for (unsigned x = 0; x < w; x++) {
            float Y = ss2h_read_plane(pic, 0, (int)x, (int)y) * inv_peak;
            float U = ss2h_read_plane(pic, 1, (int)x, (int)y) * inv_peak;
            float V = ss2h_read_plane(pic, 2, (int)x, (int)y) * inv_peak;
            float Yn = (Y - y_off) * y_scale;
            float Un = (U - c_off) * c_scale;
            float Vn = (V - c_off) * c_scale;
            float R = Yn + cr_r * Vn;
            float G = Yn + cb_g * Un + cr_g * Vn;
            float B = Yn + cb_b * Un;
            R = ss2h_clampf(R, 0.0f, 1.0f);
            G = ss2h_clampf(G, 0.0f, 1.0f);
            B = ss2h_clampf(B, 0.0f, 1.0f);
            const size_t idx = (size_t)y * w + x;
            rp[idx] = vmaf_ss2_srgb_eotf(R);
            gp[idx] = vmaf_ss2_srgb_eotf(G);
            bp[idx] = vmaf_ss2_srgb_eotf(B);
        }
    }
}

static void ss2h_host_linear_rgb_to_xyb(const float *lin, float *xyb, unsigned w, unsigned h,
                                        size_t plane_stride)
{
    const float kM00 = 0.30f;
    const float kM02 = 0.078f;
    const float kM10 = 0.23f;
    const float kM12 = 0.078f;
    const float kM20 = 0.24342268924547819f;
    const float kM21 = 0.20476744424496821f;
    const float kOpsinBias = 0.0037930732552754493f;

    const float m01 = 1.0f - kM00 - kM02;
    const float m11 = 1.0f - kM10 - kM12;
    const float m22 = 1.0f - kM20 - kM21;
    const float cbrt_bias = vmaf_ss2_cbrtf(kOpsinBias);

    const float *rp = lin;
    const float *gp = lin + plane_stride;
    const float *bp = lin + 2u * plane_stride;
    float *xp = xyb;
    float *yp = xyb + plane_stride;
    float *bxp = xyb + 2u * plane_stride;

    const size_t scale_pixels = (size_t)w * (size_t)h;
    for (size_t i = 0; i < scale_pixels; i++) {
        float r = rp[i];
        float g = gp[i];
        float b = bp[i];
        float l = kM00 * r + m01 * g + kM02 * b + kOpsinBias;
        float m = kM10 * r + m11 * g + kM12 * b + kOpsinBias;
        float sv = kM20 * r + kM21 * g + m22 * b + kOpsinBias;
        if (l < 0.0f)
            l = 0.0f;
        if (m < 0.0f)
            m = 0.0f;
        if (sv < 0.0f)
            sv = 0.0f;
        float L = vmaf_ss2_cbrtf(l) - cbrt_bias;
        float M = vmaf_ss2_cbrtf(m) - cbrt_bias;
        float S = vmaf_ss2_cbrtf(sv) - cbrt_bias;
        float X = 0.5f * (L - M);
        float Y = 0.5f * (L + M);
        float B = S;
        B = (B - Y) + 0.55f;
        X = X * 14.0f + 0.42f;
        Y = Y + 0.01f;
        xp[i] = X;
        yp[i] = Y;
        bxp[i] = B;
    }
}

static void ss2h_downsample_2x2(const float *in, unsigned iw, unsigned ih, float *out, unsigned ow,
                                unsigned oh, size_t plane_stride)
{
    for (int c = 0; c < 3; c++) {
        const float *ip = in + (size_t)c * plane_stride;
        float *op = out + (size_t)c * plane_stride;
        for (unsigned oy = 0; oy < oh; oy++) {
            for (unsigned ox = 0; ox < ow; ox++) {
                float sum = 0.0f;
                for (unsigned dy = 0; dy < 2; dy++) {
                    for (unsigned dx = 0; dx < 2; dx++) {
                        unsigned ix = ox * 2 + dx;
                        unsigned iy = oy * 2 + dy;
                        if (ix >= iw)
                            ix = iw - 1;
                        if (iy >= ih)
                            iy = ih - 1;
                        sum += ip[(size_t)iy * iw + ix];
                    }
                }
                op[(size_t)oy * ow + ox] = sum * 0.25f;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* HIP error → errno                                                  */
/* ------------------------------------------------------------------ */

static int ss2h_hip_rc(hipError_t rc)
{
    if (rc == hipSuccess)
        return 0;
    switch (rc) {
    case hipErrorInvalidValue:
    case hipErrorInvalidHandle:
        return -EINVAL;
    case hipErrorOutOfMemory:
        return -ENOMEM;
    case hipErrorNoDevice:
    case hipErrorInvalidDevice:
        return -ENODEV;
    case hipErrorNotSupported:
        return -ENOSYS;
    default:
        return -EIO;
    }
}

/* ------------------------------------------------------------------ */
/* HAVE_HIPCC: lifecycle + per-frame kernels                          */
/* ------------------------------------------------------------------ */

#ifdef HAVE_HIPCC

static int ss2h_alloc_device(Ssimu2StateHip *s)
{
    const size_t three_plane_bytes = 3u * (size_t)s->width * (size_t)s->height * sizeof(float);
    hipError_t hip_rc;
#define SS2H_ALLOC_DEV(f)                                                                          \
    do {                                                                                           \
        hip_rc = hipMalloc(&s->f, three_plane_bytes);                                              \
        if (hip_rc != hipSuccess)                                                                  \
            return ss2h_hip_rc(hip_rc);                                                            \
    } while (0)
    SS2H_ALLOC_DEV(d_ref_xyb);
    SS2H_ALLOC_DEV(d_dis_xyb);
    SS2H_ALLOC_DEV(d_mul_buf);
    SS2H_ALLOC_DEV(d_blur_scratch);
    SS2H_ALLOC_DEV(d_mu1);
    SS2H_ALLOC_DEV(d_mu2);
    SS2H_ALLOC_DEV(d_s11);
    SS2H_ALLOC_DEV(d_s22);
    SS2H_ALLOC_DEV(d_s12);
#undef SS2H_ALLOC_DEV
    return 0;
}

static int ss2h_alloc_pinned(Ssimu2StateHip *s)
{
    const size_t three_plane_bytes = 3u * (size_t)s->width * (size_t)s->height * sizeof(float);
    const size_t worst_plane_bytes = (size_t)s->width * (size_t)s->height * 2u;
    hipError_t hip_rc;
#define SS2H_ALLOC_PIN(f)                                                                          \
    do {                                                                                           \
        hip_rc = hipHostMalloc((void **)&s->f, three_plane_bytes, 0);                              \
        if (hip_rc != hipSuccess)                                                                  \
            return ss2h_hip_rc(hip_rc);                                                            \
    } while (0)
    SS2H_ALLOC_PIN(h_ref_lin);
    SS2H_ALLOC_PIN(h_dis_lin);
    SS2H_ALLOC_PIN(h_ref_lin_ds);
    SS2H_ALLOC_PIN(h_dis_lin_ds);
    SS2H_ALLOC_PIN(h_ref_xyb);
    SS2H_ALLOC_PIN(h_dis_xyb);
    SS2H_ALLOC_PIN(h_mu1);
    SS2H_ALLOC_PIN(h_mu2);
    SS2H_ALLOC_PIN(h_s11);
    SS2H_ALLOC_PIN(h_s22);
    SS2H_ALLOC_PIN(h_s12);
#undef SS2H_ALLOC_PIN
    for (int p = 0; p < 3; p++) {
        hip_rc = hipHostMalloc(&s->h_ref_raw[p], worst_plane_bytes, 0);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
        s->raw_plane_bytes[p] = worst_plane_bytes;
        hip_rc = hipHostMalloc(&s->h_dis_raw[p], worst_plane_bytes, 0);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
    }
    return 0;
}

static int ss2h_launch_mul3(Ssimu2StateHip *s, void *a, void *b, void *out, unsigned scale)
{
    unsigned cw = s->scale_w[scale];
    unsigned ch = s->scale_h[scale];
    unsigned plane_count = 3u;
    unsigned plane_stride = s->width * s->height;
    unsigned gx = (cw + SS2H_MUL_BX - 1u) / SS2H_MUL_BX;
    unsigned gy = (ch + SS2H_MUL_BY - 1u) / SS2H_MUL_BY;
    void *args[] = {&a, &b, &out, &cw, &ch, &plane_count, &plane_stride};
    hipError_t hip_rc = hipModuleLaunchKernel(s->func_mul3, gx, gy, 1, SS2H_MUL_BX, SS2H_MUL_BY, 1,
                                              0, s->str, args, NULL);
    return ss2h_hip_rc(hip_rc);
}

static int ss2h_launch_blur_pass(Ssimu2StateHip *s, hipFunction_t func, void *in_buf, void *out_buf,
                                 unsigned cw, unsigned ch, unsigned in_off, unsigned out_off,
                                 unsigned lines)
{
    float n2_0 = s->rg_n2[0];
    float n2_1 = s->rg_n2[1];
    float n2_2 = s->rg_n2[2];
    float d1_0 = s->rg_d1[0];
    float d1_1 = s->rg_d1[1];
    float d1_2 = s->rg_d1[2];
    int radius = s->rg_radius;
    void *args[] = {&in_buf, &out_buf, &cw,   &ch,     &n2_0,   &n2_1,   &n2_2,
                    &d1_0,   &d1_1,    &d1_2, &radius, &in_off, &out_off};
    unsigned grid = (lines + SS2H_BLUR_BLOCK - 1u) / SS2H_BLUR_BLOCK;
    hipError_t hip_rc =
        hipModuleLaunchKernel(func, grid, 1, 1, SS2H_BLUR_BLOCK, 1, 1, 0, s->str, args, NULL);
    return ss2h_hip_rc(hip_rc);
}

static int ss2h_blur_3plane(Ssimu2StateHip *s, void *in_buf, void *out_buf, unsigned scale)
{
    const unsigned cw = s->scale_w[scale];
    const unsigned ch = s->scale_h[scale];
    const unsigned full_plane = s->width * s->height;
    void *scratch = s->d_blur_scratch;
    int err = 0;
    for (int c = 0; c < 3; c++) {
        const unsigned off = (unsigned)c * full_plane;
        err = ss2h_launch_blur_pass(s, s->func_blur_h, in_buf, scratch, cw, ch, off, off, ch);
        if (err)
            return err;
        err = ss2h_launch_blur_pass(s, s->func_blur_v, scratch, out_buf, cw, ch, off, off, cw);
        if (err)
            return err;
    }
    return 0;
}

/* Per-pixel SSIM + EdgeDiff combine in double precision.
 * Splitting would break the line-for-line scalar-diff audit trail
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5 sweep
 * closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2h_host_combine(const Ssimu2StateHip *s, int scale, double avg_ssim[6],
                              double avg_ed[12])
{
    const unsigned cw = s->scale_w[scale];
    const unsigned ch = s->scale_h[scale];
    const size_t full_plane = (size_t)s->width * (size_t)s->height;
    const size_t scale_pixels = (size_t)cw * (size_t)ch;
    const double inv_pixels = 1.0 / (double)scale_pixels;
    for (int c = 0; c < 3; c++) {
        const float *m1 = s->h_mu1 + (size_t)c * full_plane;
        const float *m2 = s->h_mu2 + (size_t)c * full_plane;
        const float *s11p = s->h_s11 + (size_t)c * full_plane;
        const float *s22p = s->h_s22 + (size_t)c * full_plane;
        const float *s12p = s->h_s12 + (size_t)c * full_plane;
        const float *r1 = s->h_ref_xyb + (size_t)c * full_plane;
        const float *r2 = s->h_dis_xyb + (size_t)c * full_plane;
        double sum_l1 = 0.0;
        double sum_l4 = 0.0;
        double e_art = 0.0;
        double e_art4 = 0.0;
        double e_det = 0.0;
        double e_det4 = 0.0;
        for (size_t i = 0; i < scale_pixels; i++) {
            const float u1 = m1[i];
            const float u2 = m2[i];
            const float u11 = u1 * u1;
            const float u22 = u2 * u2;
            const float u12 = u1 * u2;
            const float num_m = 1.0f - (u1 - u2) * (u1 - u2);
            const float num_s = 2.0f * (s12p[i] - u12) + 0.0009f;
            const float denom_s = (s11p[i] - u11) + (s22p[i] - u22) + 0.0009f;
            double d = 1.0 - ((double)num_m * (double)num_s / (double)denom_s);
            if (d < 0.0)
                d = 0.0;
            sum_l1 += d;
            const double d2 = d * d;
            sum_l4 += d2 * d2;
            const double ed1 = fabs((double)r1[i] - (double)u1);
            const double ed2 = fabs((double)r2[i] - (double)u2);
            const double dd = (1.0 + ed2) / (1.0 + ed1) - 1.0;
            const double art = dd > 0.0 ? dd : 0.0;
            const double det = dd < 0.0 ? -dd : 0.0;
            e_art += art;
            const double a2 = art * art;
            e_art4 += a2 * a2;
            e_det += det;
            const double d2e = det * det;
            e_det4 += d2e * d2e;
        }
        avg_ssim[c * 2 + 0] = inv_pixels * sum_l1;
        avg_ssim[c * 2 + 1] = sqrt(sqrt(inv_pixels * sum_l4));
        avg_ed[c * 4 + 0] = inv_pixels * e_art;
        avg_ed[c * 4 + 1] = sqrt(sqrt(inv_pixels * e_art4));
        avg_ed[c * 4 + 2] = inv_pixels * e_det;
        avg_ed[c * 4 + 3] = sqrt(sqrt(inv_pixels * e_det4));
    }
}

static double ss2h_pool_score(const double avg_ssim[6][6], const double avg_ed[6][12],
                              int num_scales)
{
    double ssim = 0.0;
    size_t i = 0;
    for (int c = 0; c < 3; c++) {
        for (int scale = 0; scale < 6; scale++) {
            for (int n = 0; n < 2; n++) {
                double s_term = scale < num_scales ? avg_ssim[scale][c * 2 + n] : 0.0;
                double r_term = scale < num_scales ? avg_ed[scale][c * 4 + n] : 0.0;
                double b_term = scale < num_scales ? avg_ed[scale][c * 4 + n + 2] : 0.0;
                ssim += g_weights[i++] * fabs(s_term);
                ssim += g_weights[i++] * fabs(r_term);
                ssim += g_weights[i++] * fabs(b_term);
            }
        }
    }
    ssim *= 0.9562382616834844;
    ssim = 2.326765642916932 * ssim - 0.020884521182843837 * ssim * ssim +
           6.248496625763138e-05 * ssim * ssim * ssim;
    if (ssim > 0.0) {
        ssim = 100.0 - 10.0 * pow(ssim, 0.6276336467831387);
    } else {
        ssim = 100.0;
    }
    return ssim;
}

/* Per-scale GPU work: 3 mul + 5 blur.
 * Splitting would obscure the dispatch sequence required for parity
 * audit (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5
 * sweep closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static int ss2h_run_scale_gpu(Ssimu2StateHip *s, int scale)
{
    const size_t plane_full_bytes = (size_t)s->width * (size_t)s->height * sizeof(float);
    const size_t scale_pixels = (size_t)s->scale_w[scale] * (size_t)s->scale_h[scale];
    const size_t scale_bytes_per_plane = scale_pixels * sizeof(float);
    void *ref_xyb = s->d_ref_xyb;
    void *dis_xyb = s->d_dis_xyb;
    void *mul_buf = s->d_mul_buf;
    void *mu1 = s->d_mu1;
    void *mu2 = s->d_mu2;
    void *ds11 = s->d_s11;
    void *ds22 = s->d_s22;
    void *ds12 = s->d_s12;

    /* Upload XYB buffers per plane (valid sub-region only). */
    for (int c = 0; c < 3; c++) {
        const size_t plane_off_bytes = (size_t)c * plane_full_bytes;
        hipError_t hip_rc;
        hip_rc = hipMemcpyAsync((uint8_t *)ref_xyb + plane_off_bytes,
                                (const uint8_t *)s->h_ref_xyb + plane_off_bytes,
                                scale_bytes_per_plane, hipMemcpyHostToDevice, s->str);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
        hip_rc = hipMemcpyAsync((uint8_t *)dis_xyb + plane_off_bytes,
                                (const uint8_t *)s->h_dis_xyb + plane_off_bytes,
                                scale_bytes_per_plane, hipMemcpyHostToDevice, s->str);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
    }

    int err = 0;

    err = ss2h_launch_mul3(s, ref_xyb, ref_xyb, mul_buf, (unsigned)scale);
    if (err)
        return err;
    err = ss2h_blur_3plane(s, mul_buf, ds11, scale);
    if (err)
        return err;

    err = ss2h_launch_mul3(s, dis_xyb, dis_xyb, mul_buf, (unsigned)scale);
    if (err)
        return err;
    err = ss2h_blur_3plane(s, mul_buf, ds22, scale);
    if (err)
        return err;

    err = ss2h_launch_mul3(s, ref_xyb, dis_xyb, mul_buf, (unsigned)scale);
    if (err)
        return err;
    err = ss2h_blur_3plane(s, mul_buf, ds12, scale);
    if (err)
        return err;

    err = ss2h_blur_3plane(s, ref_xyb, mu1, scale);
    if (err)
        return err;
    err = ss2h_blur_3plane(s, dis_xyb, mu2, scale);
    if (err)
        return err;

    /* Download blurred buffers (valid sub-region only). */
    for (int c = 0; c < 3; c++) {
        const size_t plane_off_bytes = (size_t)c * plane_full_bytes;
        hipError_t hip_rc;
        hip_rc =
            hipMemcpyAsync((uint8_t *)s->h_mu1 + plane_off_bytes, (uint8_t *)mu1 + plane_off_bytes,
                           scale_bytes_per_plane, hipMemcpyDeviceToHost, s->str);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
        hip_rc =
            hipMemcpyAsync((uint8_t *)s->h_mu2 + plane_off_bytes, (uint8_t *)mu2 + plane_off_bytes,
                           scale_bytes_per_plane, hipMemcpyDeviceToHost, s->str);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
        hip_rc =
            hipMemcpyAsync((uint8_t *)s->h_s11 + plane_off_bytes, (uint8_t *)ds11 + plane_off_bytes,
                           scale_bytes_per_plane, hipMemcpyDeviceToHost, s->str);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
        hip_rc =
            hipMemcpyAsync((uint8_t *)s->h_s22 + plane_off_bytes, (uint8_t *)ds22 + plane_off_bytes,
                           scale_bytes_per_plane, hipMemcpyDeviceToHost, s->str);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
        hip_rc =
            hipMemcpyAsync((uint8_t *)s->h_s12 + plane_off_bytes, (uint8_t *)ds12 + plane_off_bytes,
                           scale_bytes_per_plane, hipMemcpyDeviceToHost, s->str);
        if (hip_rc != hipSuccess)
            return ss2h_hip_rc(hip_rc);
    }
    hipError_t hip_rc = hipStreamSynchronize(s->str);
    return ss2h_hip_rc(hip_rc);
}

#endif /* HAVE_HIPCC */

/* ------------------------------------------------------------------ */
/* init / extract / close                                             */
/* ------------------------------------------------------------------ */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
#ifndef HAVE_HIPCC
    (void)fex;
    (void)bpc;
    (void)w;
    (void)h;
    return -ENOSYS;
#else
    Ssimu2StateHip *s = fex->priv;

    if (w < 8u || h < 8u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "ssimulacra2_hip: input %ux%u below 8x8 lower bound\n", w,
                 h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    ss2h_setup_gaussian(s, SS2H_SIGMA);

    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < SS2H_NUM_SCALES; i++) {
        s->scale_w[i] = (s->scale_w[i - 1] + 1) / 2;
        s->scale_h[i] = (s->scale_h[i - 1] + 1) / 2;
    }

    hipError_t hip_rc;
    hip_rc = hipStreamCreateWithFlags(&s->str, hipStreamNonBlocking);
    if (hip_rc != hipSuccess)
        return ss2h_hip_rc(hip_rc);

    hip_rc = hipModuleLoadData(&s->module_blur, ssimulacra2_blur_hsaco);
    if (hip_rc != hipSuccess)
        goto fail_stream;
    hip_rc = hipModuleLoadData(&s->module_mul, ssimulacra2_mul_hsaco);
    if (hip_rc != hipSuccess)
        goto fail_mod_blur;
    hip_rc = hipModuleGetFunction(&s->func_blur_h, s->module_blur, "ssimulacra2_blur_h");
    if (hip_rc != hipSuccess)
        goto fail_mod_mul;
    hip_rc = hipModuleGetFunction(&s->func_blur_v, s->module_blur, "ssimulacra2_blur_v");
    if (hip_rc != hipSuccess)
        goto fail_mod_mul;
    hip_rc = hipModuleGetFunction(&s->func_mul3, s->module_mul, "ssimulacra2_mul3");
    if (hip_rc != hipSuccess)
        goto fail_mod_mul;

    int ret = ss2h_alloc_device(s);
    if (ret)
        goto fail_mod_mul;
    ret = ss2h_alloc_pinned(s);
    if (ret)
        goto fail_mod_mul;
    return 0;

fail_mod_mul:
    (void)hipModuleUnload(s->module_mul);
    s->module_mul = NULL;
fail_mod_blur:
    (void)hipModuleUnload(s->module_blur);
    s->module_blur = NULL;
fail_stream:
    (void)hipStreamDestroy(s->str);
    s->str = NULL;
    return ss2h_hip_rc(hip_rc);
#endif
}

/* Per-scale orchestration mirrors the CUDA extract loop.
 * Splitting would obscure the dispatch ordering required for parity
 * audit (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5
 * sweep closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static int extract_fex_hip(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
#ifndef HAVE_HIPCC
    (void)fex;
    (void)ref_pic;
    (void)dist_pic;
    (void)index;
    (void)feature_collector;
    return -ENOSYS;
#else
    Ssimu2StateHip *s = fex->priv;

    /* D2H copy raw YUV planes into pinned host scratch, then build
     * synthetic host-side VmafPictures for ss2h_picture_to_linear_rgb.
     * Pictures on HIP arrive as CPU VmafPictures (HIP flag not set),
     * so data[] is a host pointer — no D2H needed. Clone the pointers. */
    VmafPicture host_ref = *ref_pic;
    VmafPicture host_dis = *dist_pic;

    /* Stage 1: host YUV → linear RGB. */
    ss2h_picture_to_linear_rgb(s, &host_ref, s->h_ref_lin);
    ss2h_picture_to_linear_rgb(s, &host_dis, s->h_dis_lin);

    /* Stage 2: per-scale loop. */
    double avg_ssim[6][6] = {{0}};
    double avg_ed[6][12] = {{0}};
    int completed = 0;
    unsigned cw = s->width;
    unsigned ch = s->height;
    const size_t plane_full = (size_t)s->width * (size_t)s->height;

    for (int scale = 0; scale < SS2H_NUM_SCALES; scale++) {
        if (cw < 8u || ch < 8u)
            break;

        ss2h_host_linear_rgb_to_xyb(s->h_ref_lin, s->h_ref_xyb, cw, ch, plane_full);
        ss2h_host_linear_rgb_to_xyb(s->h_dis_lin, s->h_dis_xyb, cw, ch, plane_full);

        int err = ss2h_run_scale_gpu(s, scale);
        if (err)
            return err;

        ss2h_host_combine(s, scale, avg_ssim[scale], avg_ed[scale]);
        completed++;

        if (scale + 1 < SS2H_NUM_SCALES) {
            const unsigned nw = (cw + 1) / 2;
            const unsigned nh = (ch + 1) / 2;
            ss2h_downsample_2x2(s->h_ref_lin, cw, ch, s->h_ref_lin_ds, nw, nh, plane_full);
            for (int c = 0; c < 3; c++)
                memcpy(s->h_ref_lin + (size_t)c * plane_full,
                       s->h_ref_lin_ds + (size_t)c * plane_full,
                       (size_t)nw * (size_t)nh * sizeof(float));
            ss2h_downsample_2x2(s->h_dis_lin, cw, ch, s->h_dis_lin_ds, nw, nh, plane_full);
            for (int c = 0; c < 3; c++)
                memcpy(s->h_dis_lin + (size_t)c * plane_full,
                       s->h_dis_lin_ds + (size_t)c * plane_full,
                       (size_t)nw * (size_t)nh * sizeof(float));
            cw = nw;
            ch = nh;
        }
    }

    const double score = ss2h_pool_score(avg_ssim, avg_ed, completed);
    return vmaf_feature_collector_append(feature_collector, "ssimulacra2", score, index);
#endif
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    Ssimu2StateHip *s = fex->priv;
    if (!s)
        return 0;
#ifdef HAVE_HIPCC
    if (s->str)
        (void)hipStreamSynchronize(s->str);
    if (s->module_blur)
        (void)hipModuleUnload(s->module_blur);
    if (s->module_mul)
        (void)hipModuleUnload(s->module_mul);
    if (s->str)
        (void)hipStreamDestroy(s->str);

#define SS2H_FREE_DEV(f)                                                                           \
    do {                                                                                           \
        if (s->f) {                                                                                \
            (void)hipFree(s->f);                                                                   \
            s->f = NULL;                                                                           \
        }                                                                                          \
    } while (0)
    SS2H_FREE_DEV(d_ref_xyb);
    SS2H_FREE_DEV(d_dis_xyb);
    SS2H_FREE_DEV(d_mul_buf);
    SS2H_FREE_DEV(d_blur_scratch);
    SS2H_FREE_DEV(d_mu1);
    SS2H_FREE_DEV(d_mu2);
    SS2H_FREE_DEV(d_s11);
    SS2H_FREE_DEV(d_s22);
    SS2H_FREE_DEV(d_s12);
#undef SS2H_FREE_DEV

#define SS2H_FREE_PIN(f)                                                                           \
    do {                                                                                           \
        if (s->f) {                                                                                \
            (void)hipHostFree(s->f);                                                               \
            s->f = NULL;                                                                           \
        }                                                                                          \
    } while (0)
    SS2H_FREE_PIN(h_ref_lin);
    SS2H_FREE_PIN(h_dis_lin);
    SS2H_FREE_PIN(h_ref_lin_ds);
    SS2H_FREE_PIN(h_dis_lin_ds);
    SS2H_FREE_PIN(h_ref_xyb);
    SS2H_FREE_PIN(h_dis_xyb);
    SS2H_FREE_PIN(h_mu1);
    SS2H_FREE_PIN(h_mu2);
    SS2H_FREE_PIN(h_s11);
    SS2H_FREE_PIN(h_s22);
    SS2H_FREE_PIN(h_s12);
#undef SS2H_FREE_PIN

    for (int p = 0; p < 3; p++) {
        if (s->h_ref_raw[p]) {
            (void)hipHostFree(s->h_ref_raw[p]);
            s->h_ref_raw[p] = NULL;
        }
        if (s->h_dis_raw[p]) {
            (void)hipHostFree(s->h_dis_raw[p]);
            s->h_dis_raw[p] = NULL;
        }
    }
#endif
    return 0;
}

static const char *provided_features[] = {"ssimulacra2", NULL};

VmafFeatureExtractor vmaf_fex_ssimulacra2_hip = {
    .name = "ssimulacra2_hip",
    .init = init_fex_hip,
    .extract = extract_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(Ssimu2StateHip),
    .flags = VMAF_FEATURE_EXTRACTOR_HIP,
    .provided_features = provided_features,
};
