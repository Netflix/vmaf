/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ssimulacra2 feature extractor on the Metal backend (T8-1k / ADR-0421).
 *
 *  Pipeline (mirrors CUDA port — ADR-0201 / ADR-0192):
 *    1. Host: YUV → linear RGB via deterministic sRGB EOTF LUT.
 *    2. Host: linear RGB → XYB (verbatim port; GPU cbrt() diverges —
 *       see ADR-0201 §Precision investigation).
 *    3. Per-scale GPU:
 *       a. ssimulacra2_mul3 (ref², dis², ref·dis).
 *       b. ssimulacra2_blur_h + ssimulacra2_blur_v (IIR, sigma=1.5).
 *       c. blur ref → mu1; blur dis → mu2.
 *    4. Host: SSIM map + EdgeDiff combine in double precision.
 *    5. Host: pool 108 weighted norms → libjxl polynomial → score.
 *
 *  Precision contract: places=4 (max_abs_diff <= 5e-5) vs CPU on
 *  the Netflix golden pair (ADR-0192 / ADR-0201 / ADR-0214).
 *
 *  The IIR FP ordering in ssimulacra2.metal avoids FMA fusion by
 *  splitting ns/dp/t/o into separate temporaries (mirrors the CUDA
 *  --fmad=false strategy; Metal MSL does not need a per-kernel flag
 *  because IEEE-754 semantics apply by default).
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" {
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"
#include "log.h"

#include "../../metal/common.h"
#include "../../metal/kernel_template.h"
}

#include "feature/ssimulacra2_math.h"

extern "C" {
extern const unsigned char libvmaf_metallib_start[] __asm("section$start$__TEXT$__metallib");
extern const unsigned char libvmaf_metallib_end[]   __asm("section$end$__TEXT$__metallib");
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define SS2M_NUM_SCALES 6
#define SS2M_MUL_TG_X   16u
#define SS2M_MUL_TG_Y   8u
#define SS2M_BLUR_TG     64u
#define SS2M_SIGMA       1.5
#define SS2M_PI          3.14159265358979323846

enum yuv_matrix_m {
    SS2M_MATRIX_BT709_LIMITED = 0,
    SS2M_MATRIX_BT601_LIMITED = 1,
    SS2M_MATRIX_BT709_FULL    = 2,
    SS2M_MATRIX_BT601_FULL    = 3,
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

/* ------------------------------------------------------------------ */
/*  IIR parameter block — must match ssimulacra2.metal::IirParams.     */
/* ------------------------------------------------------------------ */
typedef struct IirParamsMetal {
    float n2[3];
    float d1[3];
    int   radius;
    int   _pad;
} IirParamsMetal;

/* ------------------------------------------------------------------ */
/*  State                                                              */
/* ------------------------------------------------------------------ */
typedef struct Ssimu2StateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext *ctx;

    /* Pipeline state objects for 3 kernels. */
    void *pso_mul3;    /* ssimulacra2_mul3 */
    void *pso_blur_h;  /* ssimulacra2_blur_h */
    void *pso_blur_v;  /* ssimulacra2_blur_v */

    /* Options + geometry. */
    int      yuv_matrix;
    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned scale_w[SS2M_NUM_SCALES];
    unsigned scale_h[SS2M_NUM_SCALES];

    /* Recursive Gaussian IIR coefficients (sigma = 1.5). */
    IirParamsMetal iir;

    /* GPU-resident 3-plane float buffers (size = 3 * width * height * sizeof(float)).
     * Plane c at byte offset c * width * height * sizeof(float). */
    void *d_ref_xyb;    /* id<MTLBuffer> */
    void *d_dis_xyb;
    void *d_mul_buf;
    void *d_blur_scratch;
    void *d_mu1;
    void *d_mu2;
    void *d_s11;
    void *d_s22;
    void *d_s12;

    /* Shared (CPU-accessible) staging buffers. */
    void *h_ref_lin;      /* float[3 * W * H] — linear RGB (host-computed) */
    void *h_dis_lin;
    void *h_ref_lin_ds;   /* downsample scratch */
    void *h_dis_lin_ds;
    void *h_ref_xyb;      /* float[3 * W * H] — XYB (host-computed) */
    void *h_dis_xyb;
    void *h_mu1;          /* readback targets */
    void *h_mu2;
    void *h_s11;
    void *h_s22;
    void *h_s12;
} Ssimu2StateMetal;

static const VmafOption options[] = {
    {
        .name          = "yuv_matrix",
        .help          = "YUV->RGB matrix: 0=bt709_limited (default), 1=bt601_limited, "
                         "2=bt709_full, 3=bt601_full",
        .offset        = offsetof(Ssimu2StateMetal, yuv_matrix),
        .type          = VMAF_OPT_TYPE_INT,
        .default_val.i = SS2M_MATRIX_BT709_LIMITED,
        .min           = 0,
        .max           = 3,
    },
    {0},
};

/* ------------------------------------------------------------------ */
/*  IIR coefficient setup — verbatim port of ssimulacra2_cuda.c.
 *  Splitting would break the scalar-diff audit trail (ADR-0141 §2).
 *  NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
/* ------------------------------------------------------------------ */
static void ss2m_setup_gaussian(Ssimu2StateMetal *s, double sigma)
{
    const double radius     = round(3.2795 * sigma + 0.2546);
    const double pi_div_2r  = SS2M_PI / (2.0 * radius);
    const double omega[3]   = {pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r};

    const double p1 = +1.0 / tan(0.5 * omega[0]);
    const double p3 = -1.0 / tan(0.5 * omega[1]);
    const double p5 = +1.0 / tan(0.5 * omega[2]);
    const double r1 = +p1 * p1 / sin(omega[0]);
    const double r3 = -p3 * p3 / sin(omega[1]);
    const double r5 = +p5 * p5 / sin(omega[2]);

    const double neg_half_sigma2 = -0.5 * sigma * sigma;
    const double recip_r         = 1.0 / radius;
    double rho[3];
    for (int i = 0; i < 3; i++) {
        rho[i] = exp(neg_half_sigma2 * omega[i] * omega[i]) * recip_r;
    }

    const double D13     = p1 * r3 - r1 * p3;
    const double D35     = p3 * r5 - r3 * p5;
    const double D51     = p5 * r1 - r5 * p1;
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
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M[i][j] = A[i][j];
            }
        }
        for (int i = 0; i < 3; i++) {
            M[i][col] = gamma[i];
        }
        beta[col] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                     M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                     M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) *
                    inv_det;
    }

    s->iir.radius = (int)radius;
    for (int i = 0; i < 3; i++) {
        s->iir.n2[i] = (float)(-beta[i] * cos(omega[i] * (radius + 1.0)));
        s->iir.d1[i] = (float)(-2.0 * cos(omega[i]));
    }
    s->iir._pad = 0;
}

/* ------------------------------------------------------------------ */
/*  Host YUV → linear RGB + linear RGB → XYB + 2x2 downsample.
 *  Verbatim ports of the CUDA extractor's host helper functions;
 *  splitting would break the scalar-diff audit trail (ADR-0141 §2).  */
/* ------------------------------------------------------------------ */

static inline float ss2m_clampf(float v, float lo, float hi)
{
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

static inline float ss2m_read_plane(const VmafPicture *pic, int plane, int x, int y)
{
    const unsigned pw = pic->w[plane];
    const unsigned ph = pic->h[plane];
    const unsigned lw = pic->w[0];
    const unsigned lh = pic->h[0];
    int sx = (pw == lw)       ? x :
             (pw * 2u == lw)  ? (x >> 1) :
                                 (int)((int64_t)x * (int64_t)pw / (int64_t)lw);
    int sy = (ph == lh)       ? y :
             (ph * 2u == lh)  ? (y >> 1) :
                                 (int)((int64_t)y * (int64_t)ph / (int64_t)lh);
    if (sx < 0) { sx = 0; }
    if (sy < 0) { sy = 0; }
    if ((unsigned)sx >= pw) { sx = (int)pw - 1; }
    if ((unsigned)sy >= ph) { sy = (int)ph - 1; }
    if (pic->bpc > 8u) {
        const uint16_t *row =
            (const uint16_t *)((const uint8_t *)pic->data[plane] +
                                (size_t)sy * (size_t)pic->stride[plane]);
        return (float)row[sx];
    }
    const uint8_t *row =
        (const uint8_t *)pic->data[plane] + (size_t)sy * (size_t)pic->stride[plane];
    return (float)row[sx];
}

/* Verbatim port of ssimulacra2_cuda.c::ss2c_picture_to_linear_rgb.
 * Splitting would break the line-for-line scalar-diff audit trail
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5 sweep
 * closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2m_picture_to_linear_rgb(const Ssimu2StateMetal *s, const VmafPicture *pic,
                                       float *out)
{
    const unsigned w        = s->width;
    const unsigned h        = s->height;
    const size_t   plane_sz = (size_t)w * (size_t)h;
    float *rp = out;
    float *gp = out + plane_sz;
    float *bp = out + 2u * plane_sz;

    const float peak     = (float)((1u << s->bpc) - 1u);
    const float inv_peak = 1.0f / peak;

    float kr;
    float kg;
    float kb;
    int   limited = 1;
    switch (s->yuv_matrix) {
    case SS2M_MATRIX_BT709_FULL:
        limited = 0;
        /* fallthrough */
    case SS2M_MATRIX_BT709_LIMITED:
        kr = 0.2126f;  kg = 0.7152f;  kb = 0.0722f;
        break;
    case SS2M_MATRIX_BT601_FULL:
        limited = 0;
        /* fallthrough */
    case SS2M_MATRIX_BT601_LIMITED:
        /* fallthrough */
    default:
        kr = 0.299f;   kg = 0.587f;   kb = 0.114f;
        break;
    }
    const float cr_r   = 2.0f * (1.0f - kr);
    const float cb_b   = 2.0f * (1.0f - kb);
    const float cb_g   = -(2.0f * kb * (1.0f - kb)) / kg;
    const float cr_g   = -(2.0f * kr * (1.0f - kr)) / kg;
    const float y_scale = limited ? (255.0f / 219.0f) : 1.0f;
    const float c_scale = limited ? (255.0f / 224.0f) : 1.0f;
    const float y_off   = limited ? (16.0f / 255.0f)  : 0.0f;
    const float c_off   = 0.5f;

    for (unsigned y = 0u; y < h; y++) {
        for (unsigned x = 0u; x < w; x++) {
            float Y  = ss2m_read_plane(pic, 0, (int)x, (int)y) * inv_peak;
            float U  = ss2m_read_plane(pic, 1, (int)x, (int)y) * inv_peak;
            float V  = ss2m_read_plane(pic, 2, (int)x, (int)y) * inv_peak;
            float Yn = (Y - y_off) * y_scale;
            float Un = (U - c_off) * c_scale;
            float Vn = (V - c_off) * c_scale;
            float R  = Yn + cr_r * Vn;
            float G  = Yn + cb_g * Un + cr_g * Vn;
            float B  = Yn + cb_b * Un;
            R = ss2m_clampf(R, 0.0f, 1.0f);
            G = ss2m_clampf(G, 0.0f, 1.0f);
            B = ss2m_clampf(B, 0.0f, 1.0f);
            const size_t idx = (size_t)y * w + x;
            rp[idx] = vmaf_ss2_srgb_eotf(R);
            gp[idx] = vmaf_ss2_srgb_eotf(G);
            bp[idx] = vmaf_ss2_srgb_eotf(B);
        }
    }
}

/* Host linear-RGB → XYB. Verbatim port of ssimulacra2_cuda.c host helper.
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2m_host_linear_rgb_to_xyb(const float *lin, float *xyb, unsigned w, unsigned h,
                                        size_t plane_stride)
{
    const float kM00         = 0.30f;
    const float kM02         = 0.078f;
    const float kM10         = 0.23f;
    const float kM12         = 0.078f;
    const float kM20         = 0.24342268924547819f;
    const float kM21         = 0.20476744424496821f;
    const float kOpsinBias   = 0.0037930732552754493f;
    const float m01          = 1.0f - kM00 - kM02;
    const float m11          = 1.0f - kM10 - kM12;
    const float m22          = 1.0f - kM20 - kM21;
    const float cbrt_bias    = vmaf_ss2_cbrtf(kOpsinBias);

    const float *rp  = lin;
    const float *gp  = lin  + plane_stride;
    const float *bp  = lin  + 2u * plane_stride;
    float       *xp  = xyb;
    float       *yp  = xyb  + plane_stride;
    float       *bxp = xyb  + 2u * plane_stride;

    const size_t scale_pixels = (size_t)w * (size_t)h;
    for (size_t i = 0u; i < scale_pixels; i++) {
        float r = rp[i];
        float g = gp[i];
        float b = bp[i];
        float l = kM00 * r + m01 * g + kM02 * b + kOpsinBias;
        float m = kM10 * r + m11 * g + kM12 * b + kOpsinBias;
        float sv = kM20 * r + kM21 * g + m22 * b + kOpsinBias;
        if (l  < 0.0f) { l  = 0.0f; }
        if (m  < 0.0f) { m  = 0.0f; }
        if (sv < 0.0f) { sv = 0.0f; }
        float L = vmaf_ss2_cbrtf(l)  - cbrt_bias;
        float M = vmaf_ss2_cbrtf(m)  - cbrt_bias;
        float S = vmaf_ss2_cbrtf(sv) - cbrt_bias;
        float X = 0.5f * (L - M);
        float Y = 0.5f * (L + M);
        float B = S;
        B  = (B - Y) + 0.55f;
        X  = X * 14.0f + 0.42f;
        Y  = Y + 0.01f;
        xp[i]  = X;
        yp[i]  = Y;
        bxp[i] = B;
    }
}

static void ss2m_downsample_2x2(const float *in, unsigned iw, unsigned ih, float *out,
                                unsigned ow, unsigned oh, size_t plane_stride)
{
    for (int c = 0; c < 3; c++) {
        const float *ip = in  + (size_t)c * plane_stride;
        float       *op = out + (size_t)c * plane_stride;
        for (unsigned oy = 0u; oy < oh; oy++) {
            for (unsigned ox = 0u; ox < ow; ox++) {
                float sum = 0.0f;
                for (unsigned dy = 0u; dy < 2u; dy++) {
                    for (unsigned dx = 0u; dx < 2u; dx++) {
                        unsigned ix = ox * 2u + dx;
                        unsigned iy = oy * 2u + dy;
                        if (ix >= iw) { ix = iw - 1u; }
                        if (iy >= ih) { iy = ih - 1u; }
                        sum += ip[(size_t)iy * iw + ix];
                    }
                }
                op[(size_t)oy * ow + ox] = sum * 0.25f;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Per-pixel SSIM + EdgeDiff combine in double precision.
 *  Verbatim port of ssimulacra2_cuda.c::ss2c_host_combine.
 *  Splitting would break the scalar-diff audit trail (ADR-0141 §2 +
 *  ADR-0278).
 *  NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
/* ------------------------------------------------------------------ */
static void ss2m_host_combine(const Ssimu2StateMetal *s, int scale, double avg_ssim[6],
                              double avg_ed[12])
{
    const unsigned cw          = s->scale_w[scale];
    const unsigned ch          = s->scale_h[scale];
    const size_t   full_plane  = (size_t)s->width * (size_t)s->height;
    const size_t   scale_pixels = (size_t)cw * (size_t)ch;
    const double   inv_pixels  = 1.0 / (double)scale_pixels;

    const float *h_ref_xyb   = (const float *)s->h_ref_xyb;
    const float *h_dis_xyb   = (const float *)s->h_dis_xyb;
    const float *h_mu1       = (const float *)s->h_mu1;
    const float *h_mu2       = (const float *)s->h_mu2;
    const float *h_s11       = (const float *)s->h_s11;
    const float *h_s22       = (const float *)s->h_s22;
    const float *h_s12       = (const float *)s->h_s12;

    for (int c = 0; c < 3; c++) {
        const float *m1   = h_mu1    + (size_t)c * full_plane;
        const float *m2   = h_mu2    + (size_t)c * full_plane;
        const float *s11p = h_s11    + (size_t)c * full_plane;
        const float *s22p = h_s22    + (size_t)c * full_plane;
        const float *s12p = h_s12    + (size_t)c * full_plane;
        const float *r1   = h_ref_xyb + (size_t)c * full_plane;
        const float *r2   = h_dis_xyb + (size_t)c * full_plane;

        double sum_l1  = 0.0;
        double sum_l4  = 0.0;
        double e_art   = 0.0;
        double e_art4  = 0.0;
        double e_det   = 0.0;
        double e_det4  = 0.0;

        for (size_t i = 0u; i < scale_pixels; i++) {
            const float u1    = m1[i];
            const float u2    = m2[i];
            const float u11   = u1 * u1;
            const float u22   = u2 * u2;
            const float u12   = u1 * u2;
            const float num_m = 1.0f - (u1 - u2) * (u1 - u2);
            const float num_s = 2.0f * (s12p[i] - u12) + 0.0009f;
            const float denom_s = (s11p[i] - u11) + (s22p[i] - u22) + 0.0009f;
            double d = 1.0 - ((double)num_m * (double)num_s / (double)denom_s);
            if (d < 0.0) { d = 0.0; }
            sum_l1 += d;
            const double d2 = d * d;
            sum_l4 += d2 * d2;

            const double ed1 = fabs((double)r1[i] - (double)u1);
            const double ed2 = fabs((double)r2[i] - (double)u2);
            const double dd  = (1.0 + ed2) / (1.0 + ed1) - 1.0;
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
        avg_ed[c * 4 + 0]   = inv_pixels * e_art;
        avg_ed[c * 4 + 1]   = sqrt(sqrt(inv_pixels * e_art4));
        avg_ed[c * 4 + 2]   = inv_pixels * e_det;
        avg_ed[c * 4 + 3]   = sqrt(sqrt(inv_pixels * e_det4));
    }
}

/* Mirror of ssimulacra2_cuda.c::ss2c_pool_score. */
static double ss2m_pool_score(const double avg_ssim[6][6], const double avg_ed[6][12],
                              int num_scales)
{
    double ssim = 0.0;
    size_t i    = 0u;
    for (int c = 0; c < 3; c++) {
        for (int scale = 0; scale < 6; scale++) {
            for (int n = 0; n < 2; n++) {
                double s_term = scale < num_scales ? avg_ssim[scale][c * 2 + n]     : 0.0;
                double r_term = scale < num_scales ? avg_ed[scale][c * 4 + n]       : 0.0;
                double b_term = scale < num_scales ? avg_ed[scale][c * 4 + n + 2]   : 0.0;
                ssim += g_weights[i++] * fabs(s_term);
                ssim += g_weights[i++] * fabs(r_term);
                ssim += g_weights[i++] * fabs(b_term);
            }
        }
    }
    ssim *= 0.9562382616834844;
    ssim = 2.326765642916932 * ssim
         - 0.020884521182843837 * ssim * ssim
         + 6.248496625763138e-05 * ssim * ssim * ssim;
    if (ssim > 0.0) {
        ssim = 100.0 - 10.0 * pow(ssim, 0.6276336467831387);
    } else {
        ssim = 100.0;
    }
    return ssim;
}

/* ------------------------------------------------------------------ */
/*  Pipeline build                                                     */
/* ------------------------------------------------------------------ */

static int build_pipelines(Ssimu2StateMetal *s, id<MTLDevice> device)
{
    const size_t blob_size =
        (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_size == 0u) { return -ENODEV; }

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_size,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL) { return -ENOMEM; }

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil) { return -ENODEV; }

    id<MTLFunction> fn_mul3   = [lib newFunctionWithName:@"ssimulacra2_mul3"];
    id<MTLFunction> fn_blur_h = [lib newFunctionWithName:@"ssimulacra2_blur_h"];
    id<MTLFunction> fn_blur_v = [lib newFunctionWithName:@"ssimulacra2_blur_v"];
    if (fn_mul3 == nil || fn_blur_h == nil || fn_blur_v == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_mul3   =
        [device newComputePipelineStateWithFunction:fn_mul3   error:&err];
    id<MTLComputePipelineState> pso_blur_h =
        [device newComputePipelineStateWithFunction:fn_blur_h error:&err];
    id<MTLComputePipelineState> pso_blur_v =
        [device newComputePipelineStateWithFunction:fn_blur_v error:&err];
    if (pso_mul3 == nil || pso_blur_h == nil || pso_blur_v == nil) { return -ENODEV; }

    s->pso_mul3   = (__bridge_retained void *)pso_mul3;
    s->pso_blur_h = (__bridge_retained void *)pso_blur_h;
    s->pso_blur_v = (__bridge_retained void *)pso_blur_v;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Buffer allocation helpers                                          */
/* ------------------------------------------------------------------ */

static id<MTLBuffer> ss2m_alloc_shared(id<MTLDevice> device, size_t size)
{
    return [device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

static int ss2m_alloc_buffers(Ssimu2StateMetal *s, id<MTLDevice> device)
{
    const size_t three_plane_bytes =
        3u * (size_t)s->width * (size_t)s->height * sizeof(float);

#define SS2M_ALLOC_BUF(field)                                             \
    do {                                                                  \
        id<MTLBuffer> _b = ss2m_alloc_shared(device, three_plane_bytes); \
        if (_b == nil) { return -ENOMEM; }                                \
        s->field = (__bridge_retained void *)_b;                          \
    } while (0)

    SS2M_ALLOC_BUF(d_ref_xyb);
    SS2M_ALLOC_BUF(d_dis_xyb);
    SS2M_ALLOC_BUF(d_mul_buf);
    SS2M_ALLOC_BUF(d_blur_scratch);
    SS2M_ALLOC_BUF(d_mu1);
    SS2M_ALLOC_BUF(d_mu2);
    SS2M_ALLOC_BUF(d_s11);
    SS2M_ALLOC_BUF(d_s22);
    SS2M_ALLOC_BUF(d_s12);
    /* Staging buffers for host-computed XYB + readback targets share the
     * same Shared MTLBuffer: the CPU writes them before the GPU reads, or
     * the GPU writes them before the CPU reads — never concurrent. */
    SS2M_ALLOC_BUF(h_ref_lin);
    SS2M_ALLOC_BUF(h_dis_lin);
    SS2M_ALLOC_BUF(h_ref_lin_ds);
    SS2M_ALLOC_BUF(h_dis_lin_ds);
    SS2M_ALLOC_BUF(h_ref_xyb);
    SS2M_ALLOC_BUF(h_dis_xyb);
    SS2M_ALLOC_BUF(h_mu1);
    SS2M_ALLOC_BUF(h_mu2);
    SS2M_ALLOC_BUF(h_s11);
    SS2M_ALLOC_BUF(h_s22);
    SS2M_ALLOC_BUF(h_s12);
#undef SS2M_ALLOC_BUF
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    Ssimu2StateMetal *s = (Ssimu2StateMetal *)fex->priv;

    if (w < 8u || h < 8u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssimulacra2_metal: input %ux%u below 8x8 lower bound\n", w, h);
        return -EINVAL;
    }

    s->width  = w;
    s->height = h;
    s->bpc    = bpc;
    ss2m_setup_gaussian(s, SS2M_SIGMA);

    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < SS2M_NUM_SCALES; i++) {
        s->scale_w[i] = (s->scale_w[i - 1] + 1u) / 2u;
        s->scale_h[i] = (s->scale_h[i - 1] + 1u) / 2u;
    }

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_lc; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        err = ss2m_alloc_buffers(s, device);
        if (err != 0) { goto fail_lc; }

        err = build_pipelines(s, device);
        if (err != 0) { goto fail_bufs; }
    }
    return 0;

fail_bufs:
#define SS2M_REL_BUF(field)                                                        \
    do {                                                                            \
        if (s->field) {                                                             \
            (void)(__bridge_transfer id<MTLBuffer>)s->field;                       \
            s->field = NULL;                                                        \
        }                                                                           \
    } while (0)
    SS2M_REL_BUF(d_ref_xyb);  SS2M_REL_BUF(d_dis_xyb);
    SS2M_REL_BUF(d_mul_buf);  SS2M_REL_BUF(d_blur_scratch);
    SS2M_REL_BUF(d_mu1);      SS2M_REL_BUF(d_mu2);
    SS2M_REL_BUF(d_s11);      SS2M_REL_BUF(d_s22); SS2M_REL_BUF(d_s12);
    SS2M_REL_BUF(h_ref_lin);  SS2M_REL_BUF(h_dis_lin);
    SS2M_REL_BUF(h_ref_lin_ds); SS2M_REL_BUF(h_dis_lin_ds);
    SS2M_REL_BUF(h_ref_xyb);  SS2M_REL_BUF(h_dis_xyb);
    SS2M_REL_BUF(h_mu1);      SS2M_REL_BUF(h_mu2);
    SS2M_REL_BUF(h_s11);      SS2M_REL_BUF(h_s22); SS2M_REL_BUF(h_s12);
#undef SS2M_REL_BUF
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

/* ------------------------------------------------------------------ */
/*  Per-scale GPU dispatch                                             */
/* ------------------------------------------------------------------ */

/* Dispatch ssimulacra2_mul3 on the encoder at the current scale. */
static void ss2m_encode_mul3(Ssimu2StateMetal *s,
                              id<MTLComputeCommandEncoder> enc,
                              id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> out,
                              unsigned scale)
{
    const unsigned cw           = s->scale_w[scale];
    const unsigned ch           = s->scale_h[scale];
    const unsigned plane_stride = s->width * s->height;

    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)s->pso_mul3];
    [enc setBuffer:a   offset:0 atIndex:0];
    [enc setBuffer:b   offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    uint32_t params[4] = {(uint32_t)cw, (uint32_t)ch, (uint32_t)plane_stride, 0u};
    [enc setBytes:params length:sizeof(params) atIndex:3];

    const MTLSize tg   = MTLSizeMake(SS2M_MUL_TG_X, SS2M_MUL_TG_Y, 1);
    const MTLSize grid = MTLSizeMake((cw + SS2M_MUL_TG_X - 1u) / SS2M_MUL_TG_X,
                                     (ch + SS2M_MUL_TG_Y - 1u) / SS2M_MUL_TG_Y,
                                     1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}

/* Dispatch one IIR blur pass (H or V) over all 3 channels of in_buf → out_buf. */
static void ss2m_encode_blur_pass(Ssimu2StateMetal *s,
                                   id<MTLComputeCommandEncoder> enc,
                                   id<MTLComputePipelineState> pso,
                                   id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
                                   unsigned cw, unsigned ch,
                                   unsigned in_off, unsigned out_off,
                                   unsigned lines)
{
    [enc setComputePipelineState:pso];
    [enc setBuffer:in_buf  offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    [enc setBytes:&s->iir length:sizeof(s->iir) atIndex:2];
    uint32_t dims[4] = {(uint32_t)cw, (uint32_t)ch, (uint32_t)in_off, (uint32_t)out_off};
    [enc setBytes:dims length:sizeof(dims) atIndex:3];

    const MTLSize tg   = MTLSizeMake(SS2M_BLUR_TG, 1, 1);
    const MTLSize grid = MTLSizeMake((lines + SS2M_BLUR_TG - 1u) / SS2M_BLUR_TG, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}

/* Submit all 3 channels of IIR blur for in_buf → out_buf (H then V), at the given scale.
 * Each H + V pair is submitted as a separate command buffer so the encoder sequence is
 * not interleaved; the caller waits for the returned command buffer to complete. */
static id<MTLCommandBuffer> ss2m_submit_blur_3plane(Ssimu2StateMetal *s,
                                                     id<MTLCommandQueue> queue,
                                                     id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
                                                     id<MTLBuffer> scratch_buf,
                                                     unsigned scale)
{
    const unsigned cw          = s->scale_w[scale];
    const unsigned ch          = s->scale_h[scale];
    const unsigned full_plane  = s->width * s->height;

    id<MTLComputePipelineState> pso_h =
        (__bridge id<MTLComputePipelineState>)s->pso_blur_h;
    id<MTLComputePipelineState> pso_v =
        (__bridge id<MTLComputePipelineState>)s->pso_blur_v;

    id<MTLCommandBuffer> last_cmd = nil;
    for (int c = 0; c < 3; c++) {
        const unsigned off = (unsigned)c * full_plane;
        /* H pass: in_buf channel c → scratch_buf channel c. */
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            ss2m_encode_blur_pass(s, enc, pso_h, in_buf, scratch_buf, cw, ch, off, off, ch);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        /* V pass: scratch_buf channel c → out_buf channel c. */
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            ss2m_encode_blur_pass(s, enc, pso_v, scratch_buf, out_buf, cw, ch, off, off, cw);
            [enc endEncoding];
            [cmd commit];
            last_cmd = cmd;
        }
    }
    return last_cmd;
}

/* ------------------------------------------------------------------ */
/*  Per-frame extract                                                  */
/* ------------------------------------------------------------------ */

/* Per-scale orchestration mirrors the CPU extract loop step-by-step;
 * splitting would obscure the dispatch ordering required for parity
 * audit (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5
 * sweep closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static int extract_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                              VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                              VmafPicture *dist_pic_90, unsigned index,
                              VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    Ssimu2StateMetal *s = (Ssimu2StateMetal *)fex->priv;

    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (qh == NULL) { return -ENODEV; }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)qh;

    /* Metal Shared buffers expose unified CPU+GPU virtual address.
     * Get host-side float pointers from buffer contents. */
    id<MTLBuffer> buf_ref_lin    = (__bridge id<MTLBuffer>)s->h_ref_lin;
    id<MTLBuffer> buf_dis_lin    = (__bridge id<MTLBuffer>)s->h_dis_lin;
    id<MTLBuffer> buf_ref_lin_ds = (__bridge id<MTLBuffer>)s->h_ref_lin_ds;
    id<MTLBuffer> buf_dis_lin_ds = (__bridge id<MTLBuffer>)s->h_dis_lin_ds;
    id<MTLBuffer> buf_ref_xyb    = (__bridge id<MTLBuffer>)s->h_ref_xyb;
    id<MTLBuffer> buf_dis_xyb    = (__bridge id<MTLBuffer>)s->h_dis_xyb;
    id<MTLBuffer> buf_mul        = (__bridge id<MTLBuffer>)s->d_mul_buf;
    id<MTLBuffer> buf_scratch    = (__bridge id<MTLBuffer>)s->d_blur_scratch;
    id<MTLBuffer> buf_mu1        = (__bridge id<MTLBuffer>)s->d_mu1;
    id<MTLBuffer> buf_mu2        = (__bridge id<MTLBuffer>)s->d_mu2;
    id<MTLBuffer> buf_s11        = (__bridge id<MTLBuffer>)s->d_s11;
    id<MTLBuffer> buf_s22        = (__bridge id<MTLBuffer>)s->d_s22;
    id<MTLBuffer> buf_s12        = (__bridge id<MTLBuffer>)s->d_s12;

    float *h_ref_lin    = (float *)[buf_ref_lin    contents];
    float *h_dis_lin    = (float *)[buf_dis_lin    contents];
    float *h_ref_lin_ds = (float *)[buf_ref_lin_ds contents];
    float *h_dis_lin_ds = (float *)[buf_dis_lin_ds contents];
    float *h_ref_xyb    = (float *)[buf_ref_xyb    contents];
    float *h_dis_xyb    = (float *)[buf_dis_xyb    contents];

    /* Stage 1: host YUV → linear RGB. */
    ss2m_picture_to_linear_rgb(s, ref_pic,  h_ref_lin);
    ss2m_picture_to_linear_rgb(s, dist_pic, h_dis_lin);

    double avg_ssim[6][6] = {{0}};
    double avg_ed[6][12]  = {{0}};
    int    completed      = 0;
    unsigned cw = s->width;
    unsigned ch = s->height;
    const size_t plane_full = (size_t)s->width * (size_t)s->height;

    for (int scale = 0; scale < SS2M_NUM_SCALES; scale++) {
        if (cw < 8u || ch < 8u) { break; }

        /* Stage 2: host linear RGB → XYB. */
        ss2m_host_linear_rgb_to_xyb(h_ref_lin, h_ref_xyb, cw, ch, plane_full);
        ss2m_host_linear_rgb_to_xyb(h_dis_lin, h_dis_xyb, cw, ch, plane_full);

        /* Stage 3: GPU — mul3 then blur for each of the 5 SSIM statistics.
         *   buf_ref_xyb and buf_dis_xyb are Shared MTLBuffers: CPU has written
         *   them above; GPU reads them without any explicit copy. */

        /* s11 = blur(ref * ref). */
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            ss2m_encode_mul3(s, enc, buf_ref_xyb, buf_ref_xyb, buf_mul, (unsigned)scale);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        [[ss2m_submit_blur_3plane(s, queue, buf_mul, buf_s11, buf_scratch, (unsigned)scale)
          retain] waitUntilCompleted];

        /* s22 = blur(dis * dis). */
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            ss2m_encode_mul3(s, enc, buf_dis_xyb, buf_dis_xyb, buf_mul, (unsigned)scale);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        [[ss2m_submit_blur_3plane(s, queue, buf_mul, buf_s22, buf_scratch, (unsigned)scale)
          retain] waitUntilCompleted];

        /* s12 = blur(ref * dis). */
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            ss2m_encode_mul3(s, enc, buf_ref_xyb, buf_dis_xyb, buf_mul, (unsigned)scale);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        [[ss2m_submit_blur_3plane(s, queue, buf_mul, buf_s12, buf_scratch, (unsigned)scale)
          retain] waitUntilCompleted];

        /* mu1 = blur(ref). */
        [[ss2m_submit_blur_3plane(s, queue, buf_ref_xyb, buf_mu1, buf_scratch, (unsigned)scale)
          retain] waitUntilCompleted];

        /* mu2 = blur(dis). */
        [[ss2m_submit_blur_3plane(s, queue, buf_dis_xyb, buf_mu2, buf_scratch, (unsigned)scale)
          retain] waitUntilCompleted];

        /* Stage 4: host combine — Shared buffers already CPU-visible after GPU completes.
         * Provide raw float* pointers from the d_mu*/d_s* buffers to ss2m_host_combine
         * by temporarily writing them to the state's h_ void* slots. The state struct
         * holds void* (MTLBuffer handles) in h_mu1 etc.; here we need float* for the
         * combine. Use local variables and pass through a temporary overlay struct. */
        {
            Ssimu2StateMetal tmp = *s;
            tmp.h_ref_xyb = h_ref_xyb;
            tmp.h_dis_xyb = h_dis_xyb;
            tmp.h_mu1     = (float *)[buf_mu1 contents];
            tmp.h_mu2     = (float *)[buf_mu2 contents];
            tmp.h_s11     = (float *)[buf_s11 contents];
            tmp.h_s22     = (float *)[buf_s22 contents];
            tmp.h_s12     = (float *)[buf_s12 contents];
            ss2m_host_combine(&tmp, scale, avg_ssim[scale], avg_ed[scale]);
        }
        completed++;

        if (scale + 1 < SS2M_NUM_SCALES) {
            const unsigned nw = (cw + 1u) / 2u;
            const unsigned nh = (ch + 1u) / 2u;
            ss2m_downsample_2x2(h_ref_lin, cw, ch, h_ref_lin_ds, nw, nh, plane_full);
            for (int c = 0; c < 3; c++) {
                memmove(h_ref_lin + (size_t)c * plane_full,
                        h_ref_lin_ds + (size_t)c * plane_full,
                        (size_t)nw * (size_t)nh * sizeof(float));
            }
            ss2m_downsample_2x2(h_dis_lin, cw, ch, h_dis_lin_ds, nw, nh, plane_full);
            for (int c = 0; c < 3; c++) {
                memmove(h_dis_lin + (size_t)c * plane_full,
                        h_dis_lin_ds + (size_t)c * plane_full,
                        (size_t)nw * (size_t)nh * sizeof(float));
            }
            cw = nw;
            ch = nh;
        }
    }

    const double score = ss2m_pool_score(avg_ssim, avg_ed, completed);
    return vmaf_feature_collector_append(feature_collector, "ssimulacra2", score, index);
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    Ssimu2StateMetal *s = (Ssimu2StateMetal *)fex->priv;
    if (!s) { return 0; }

    if (s->pso_mul3) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_mul3;
        s->pso_mul3 = NULL;
    }
    if (s->pso_blur_h) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_blur_h;
        s->pso_blur_h = NULL;
    }
    if (s->pso_blur_v) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_blur_v;
        s->pso_blur_v = NULL;
    }

#define SS2M_REL_BUF(field)                                                        \
    do {                                                                            \
        if (s->field) {                                                             \
            (void)(__bridge_transfer id<MTLBuffer>)s->field;                       \
            s->field = NULL;                                                        \
        }                                                                           \
    } while (0)
    SS2M_REL_BUF(d_ref_xyb);  SS2M_REL_BUF(d_dis_xyb);
    SS2M_REL_BUF(d_mul_buf);  SS2M_REL_BUF(d_blur_scratch);
    SS2M_REL_BUF(d_mu1);      SS2M_REL_BUF(d_mu2);
    SS2M_REL_BUF(d_s11);      SS2M_REL_BUF(d_s22); SS2M_REL_BUF(d_s12);
    SS2M_REL_BUF(h_ref_lin);  SS2M_REL_BUF(h_dis_lin);
    SS2M_REL_BUF(h_ref_lin_ds); SS2M_REL_BUF(h_dis_lin_ds);
    SS2M_REL_BUF(h_ref_xyb);  SS2M_REL_BUF(h_dis_xyb);
    SS2M_REL_BUF(h_mu1);      SS2M_REL_BUF(h_mu2);
    SS2M_REL_BUF(h_s11);      SS2M_REL_BUF(h_s22); SS2M_REL_BUF(h_s12);
#undef SS2M_REL_BUF

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

static const char *provided_features[] = {"ssimulacra2", NULL};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_ssimulacra2_metal = {
    .name              = "ssimulacra2_metal",
    .init              = init_fex_metal,
    .extract           = extract_fex_metal,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(Ssimu2StateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 10,   /* 3*mul3 + 5*blur (3 planes * 2 passes) */
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
