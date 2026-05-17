/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ssimulacra2 feature kernel on the CUDA backend (T7-23 / GPU
 *  long-tail batch 3 part 7b — ADR-0192 / ADR-0204). CUDA twin of
 *  ssimulacra2_vulkan (PR #156 / ADR-0201).
 *
 *  Pipeline (per ADR-0201 — same shape as the Vulkan port):
 *    1. Host: YUV → linear RGB on full-res frame (deterministic LUT
 *       sRGB EOTF, ADR-0164).
 *    2. Per-scale (up to 6, breaks early when min-dim < 8):
 *       a. Host: linear RGB → XYB (verbatim port of CPU
 *          `linear_rgb_to_xyb`). Computed on host because the GPU's
 *          float `cbrt` differs from libm by 42 ULP at the bit
 *          level — that drift cascades through the IIR + 108-weight
 *          pool to a ~1.5e-2 pooled-score drift on the Vulkan port,
 *          and the same fix carries over to CUDA / SYCL by
 *          construction. See ADR-0201 §Precision investigation.
 *       b. GPU: 3 elementwise 3-plane multiplies (ref², dis², ref·dis)
 *          via `ssimulacra2_mul3`.
 *       c. GPU: 5 separable IIR blurs (s11, s22, s12, mu1, mu2) via
 *          `ssimulacra2_blur_h` + `ssimulacra2_blur_v`. One thread
 *          per row (H pass) / per column (V pass).
 *       d. Host: per-pixel SSIM + EdgeDiff combine in double precision
 *          (verbatim ports of `ssim_map` + `edge_diff_map`). Reads
 *          from host-mapped pinned buffers; the ssim_map's
 *          `1 - num_m * num_s / denom_s` requires `(double)`
 *          promotion at the divide site, which is why we keep the
 *          combine on the CPU rather than running it as a GPU
 *          reduction.
 *       e. Host: 2×2 box downsample of the linear-RGB pyramid for
 *          the next scale.
 *    3. Host: pool 108 weighted norms via the libjxl polynomial.
 *       Emit `ssimulacra2`.
 *
 *  Precision contract (per ADR-0192 + ADR-0201): places=4
 *  (max_abs_diff ≤ 5e-5) on the Netflix CPU vs CUDA pair. The
 *  ssimulacra2_blur fatbin is compiled with --fmad=false (the
 *  `cuda_cu_extra_flags` map in libvmaf/src/meson.build); without
 *  it, NVCC fuses `n2*sum - d1*prev1` into FMAs and the IIR
 *  pole-tracking error compounds through the 6-scale pyramid.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "mem.h"

#include "cuda/ssimulacra2_cuda.h"
#include "cuda_helper.cuh"
#include "feature/ssimulacra2_math.h"
#include "picture.h"
#include "picture_cuda.h"

#define SS2C_NUM_SCALES 6
#define SS2C_BLUR_BLOCK 64
#define SS2C_TILE 32 /* Transpose tile dimension; mirrors SS2C_TILE in ssimulacra2_blur.cu. */
#define SS2C_MUL_BX 16
#define SS2C_MUL_BY 8
#define SS2C_SIGMA 1.5
#define SS2C_PI 3.14159265358979323846

enum yuv_matrix_c {
    SS2C_MATRIX_BT709_LIMITED = 0,
    SS2C_MATRIX_BT601_LIMITED = 1,
    SS2C_MATRIX_BT709_FULL = 2,
    SS2C_MATRIX_BT601_FULL = 3,
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

typedef struct Ssimu2StateCuda {
    /* Options. */
    int yuv_matrix;

    /* Geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned scale_w[SS2C_NUM_SCALES];
    unsigned scale_h[SS2C_NUM_SCALES];

    /* Recursive Gaussian coefficients (sigma=1.5). */
    float rg_n2[3];
    float rg_d1[3];
    int rg_radius;

    /* CUDA module + kernel handles. */
    CUmodule module_blur;
    CUmodule module_mul;
    CUfunction func_blur_h;    /* single-channel H pass (retained; unused after ADR-0456) */
    CUfunction func_blur_v;    /* single-channel V pass (retained; unused after ADR-0456) */
    CUfunction func_blur_h3;   /* fused 3-channel H pass (ADR-0456 Change 1) */
    CUfunction func_transpose; /* row→col-major transpose (ADR-0456 Change 2) */
    CUfunction
        func_blur_v3_transposed; /* fused 3-channel V pass on col-major input (ADR-0456 Changes 1+2) */
    CUfunction func_mul3;
    CUstream str;

    /* Device buffers. All 3-plane buffers are contiguous (X | Y | B
     * with planes at full-resolution stride, kept constant across
     * pyramid scales for layout consistency). */
    VmafCudaBuffer *d_ref_lin;
    VmafCudaBuffer *d_dis_lin;
    VmafCudaBuffer *d_ref_xyb;
    VmafCudaBuffer *d_dis_xyb;
    VmafCudaBuffer *d_mul_buf;
    VmafCudaBuffer *d_blur_scratch;
    /* Column-major transpose scratch for V-pass coalescing (ADR-0456).
     * Same size as d_blur_scratch — 3 × full-plane floats. */
    VmafCudaBuffer *d_transpose_buf;
    VmafCudaBuffer *d_mu1;
    VmafCudaBuffer *d_mu2;
    VmafCudaBuffer *d_s11;
    VmafCudaBuffer *d_s22;
    VmafCudaBuffer *d_s12;

    /* Pinned host buffers for upload + readback. */
    float *h_ref_lin;
    float *h_dis_lin;
    /* Per-scale 2x2-downsample scratch — pre-allocated once (was a
     * per-scale `malloc(3 * plane_full * sizeof(float))` in the hot
     * path; on 1080p that is 24 MB / scale × up to 5 scales / frame,
     * cheap on warm allocators but a real `mmap`/`brk` cost on
     * memory-pressured systems). Reused for both ref and dis on every
     * scale; the previous pyramid level is consumed before the next
     * is written so a single buffer suffices. */
    float *h_ref_lin_ds;
    float *h_dis_lin_ds;
    float *h_ref_xyb;
    float *h_dis_xyb;
    float *h_mu1;
    float *h_mu2;
    float *h_s11;
    float *h_s22;
    float *h_s12;

    /* Pinned raw-YUV scratch + per-plane row strides used by the
     * synthetic host VmafPicture fed into `ss2c_picture_to_linear_rgb`.
     * The picture_cuda backend hands us VmafPictures whose
     * `data[]` is a CUdeviceptr — direct host reads would segfault.
     * We D2H-copy each plane into these pinned buffers and rebuild
     * a host-side VmafPicture mirror for the YUV→linear-RGB pass. */
    void *h_ref_raw[3];
    void *h_dis_raw[3];
    size_t raw_plane_bytes[3];
    unsigned plane_w[3];
    unsigned plane_h[3];
    ptrdiff_t plane_row_bytes[3];
} Ssimu2StateCuda;

static const VmafOption options[] = {
    {
        .name = "yuv_matrix",
        .help = "YUV→RGB matrix: 0=bt709_limited (default), 1=bt601_limited, "
                "2=bt709_full, 3=bt601_full",
        .offset = offsetof(Ssimu2StateCuda, yuv_matrix),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = SS2C_MATRIX_BT709_LIMITED,
        .min = 0,
        .max = 3,
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* Recursive Gaussian coefficient setup — bit-identical port of
 * create_recursive_gaussian in ssimulacra2.c.                        */
/* ------------------------------------------------------------------ */

/* Verbatim port of the libjxl Charalampidis 2016 derivation; per
 * ADR-0141 carve-out, splitting the linear-system solve would
 * obscure the scalar-diff audit trail.
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2c_setup_gaussian(Ssimu2StateCuda *s, double sigma)
{
    const double radius = round(3.2795 * sigma + 0.2546);
    const double pi_div_2r = SS2C_PI / (2.0 * radius);
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
/* Host-side YUV → linear-RGB + linear-RGB → XYB + 2×2 downsample.
 * All match-for-match ports of the corresponding ssimulacra2.c
 * scalar paths so host outputs bit-match the CPU extractor.
 * ------------------------------------------------------------------ */

static inline float ss2c_clampf(float v, float lo, float hi)
{
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

static inline float ss2c_read_plane(const VmafPicture *pic, int plane, int x, int y)
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

/* Verbatim port of ssimulacra2.c::picture_to_linear_rgb. Splitting
 * would break the line-for-line scalar-diff audit trail
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5
 * sweep closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2c_picture_to_linear_rgb(const Ssimu2StateCuda *s, const VmafPicture *pic, float *out)
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
    case SS2C_MATRIX_BT709_FULL:
        limited = 0;
        // fallthrough
    case SS2C_MATRIX_BT709_LIMITED:
        kr = 0.2126f;
        kg = 0.7152f;
        kb = 0.0722f;
        break;
    case SS2C_MATRIX_BT601_FULL:
        limited = 0;
        // fallthrough
    case SS2C_MATRIX_BT601_LIMITED:
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
            float Y = ss2c_read_plane(pic, 0, (int)x, (int)y) * inv_peak;
            float U = ss2c_read_plane(pic, 1, (int)x, (int)y) * inv_peak;
            float V = ss2c_read_plane(pic, 2, (int)x, (int)y) * inv_peak;
            float Yn = (Y - y_off) * y_scale;
            float Un = (U - c_off) * c_scale;
            float Vn = (V - c_off) * c_scale;
            float R = Yn + cr_r * Vn;
            float G = Yn + cb_g * Un + cr_g * Vn;
            float B = Yn + cb_b * Un;
            R = ss2c_clampf(R, 0.0f, 1.0f);
            G = ss2c_clampf(G, 0.0f, 1.0f);
            B = ss2c_clampf(B, 0.0f, 1.0f);
            const size_t idx = (size_t)y * w + x;
            rp[idx] = vmaf_ss2_srgb_eotf(R);
            gp[idx] = vmaf_ss2_srgb_eotf(G);
            bp[idx] = vmaf_ss2_srgb_eotf(B);
        }
    }
}

/* Host linear-RGB → XYB. Verbatim port of
 * ssimulacra2.c::linear_rgb_to_xyb. The 3-plane output stride is
 * `plane_stride` (= full_w * full_h, kept constant across scales). */
static void ss2c_host_linear_rgb_to_xyb(const float *lin, float *xyb, unsigned w, unsigned h,
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
        float s = kM20 * r + kM21 * g + m22 * b + kOpsinBias;
        if (l < 0.0f)
            l = 0.0f;
        if (m < 0.0f)
            m = 0.0f;
        if (s < 0.0f)
            s = 0.0f;
        float L = vmaf_ss2_cbrtf(l) - cbrt_bias;
        float M = vmaf_ss2_cbrtf(m) - cbrt_bias;
        float S = vmaf_ss2_cbrtf(s) - cbrt_bias;
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

static void ss2c_downsample_2x2(const float *in, unsigned iw, unsigned ih, float *out, unsigned ow,
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
/* Lifecycle                                                          */
/* ------------------------------------------------------------------ */

static int ss2c_alloc_buffers(VmafFeatureExtractor *fex, Ssimu2StateCuda *s)
{
    const size_t three_plane_bytes = 3u * (size_t)s->width * (size_t)s->height * sizeof(float);
    /* Worst-case raw plane bytes per channel (luma full-res, 16-bit). The
     * actual sub-sampled chroma dims come from the VmafPicture in
     * `extract_fex_cuda`; we lazily reuse the pinned buffer if its size
     * fits. Allocate to the upper bound so 4:4:4 / 16-bit also fits. */
    const size_t worst_plane_bytes = (size_t)s->width * (size_t)s->height * 2u;
    int ret = 0;
    for (int p = 0; p < 3; p++) {
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->h_ref_raw[p], worst_plane_bytes);
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->h_dis_raw[p], worst_plane_bytes);
        s->raw_plane_bytes[p] = worst_plane_bytes;
    }
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_ref_lin, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_dis_lin, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_ref_xyb, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_dis_xyb, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_mul_buf, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_blur_scratch, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_transpose_buf, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_mu1, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_mu2, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_s11, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_s22, three_plane_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_s12, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_ref_lin, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_dis_lin, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_ref_lin_ds, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_dis_lin_ds, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_ref_xyb, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_dis_xyb, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_mu1, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_mu2, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_s11, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_s22, three_plane_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_s12, three_plane_bytes);
    return ret ? -ENOMEM : 0;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    Ssimu2StateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    if (w < 8u || h < 8u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "ssimulacra2_cuda: input %ux%u below 8x8 lower bound\n", w,
                 h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    ss2c_setup_gaussian(s, SS2C_SIGMA);

    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < SS2C_NUM_SCALES; i++) {
        s->scale_w[i] = (s->scale_w[i - 1] + 1) / 2;
        s->scale_h[i] = (s->scale_h[i - 1] + 1) / 2;
    }

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&s->module_blur, ssimulacra2_blur_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&s->module_mul, ssimulacra2_mul_ptx), fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_blur_h, s->module_blur, "ssimulacra2_blur_h"), fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_blur_v, s->module_blur, "ssimulacra2_blur_v"), fail);
    /* ADR-0456: fused 3-channel H + transpose + fused 3-channel V. */
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_blur_h3, s->module_blur, "ssimulacra2_blur_h3"), fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_transpose, s->module_blur, "ssimulacra2_transpose"),
        fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_blur_v3_transposed, s->module_blur,
                                        "ssimulacra2_blur_v3_transposed"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_mul3, s->module_mul, "ssimulacra2_mul3"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    int ret = ss2c_alloc_buffers(fex, s);
    if (ret)
        return ret;
    return 0;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    return _cuda_err;
}

/* ------------------------------------------------------------------ */
/* Per-frame extract                                                  */
/* ------------------------------------------------------------------ */

static int ss2c_launch_mul3(Ssimu2StateCuda *s, CudaFunctions *cu_f, CUdeviceptr a, CUdeviceptr b,
                            CUdeviceptr out, unsigned scale)
{
    unsigned cw = s->scale_w[scale];
    unsigned ch = s->scale_h[scale];
    unsigned plane_count = 3u;
    unsigned plane_stride = s->width * s->height;
    unsigned gx = (cw + SS2C_MUL_BX - 1u) / SS2C_MUL_BX;
    unsigned gy = (ch + SS2C_MUL_BY - 1u) / SS2C_MUL_BY;
    void *args[] = {&a, &b, &out, &cw, &ch, &plane_count, &plane_stride};
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_mul3, gx, gy, 1, SS2C_MUL_BX, SS2C_MUL_BY, 1, 0,
                                           s->str, args, NULL));
    return 0;
}

/* ADR-0456 Change 1+2: fused 3-channel blur with V-pass coalescing.
 *
 * Per blur call this issues exactly 3 kernel launches:
 *   1. ssimulacra2_blur_h3        (gridDim.z=3 fuses all 3 H passes)
 *   2. ssimulacra2_transpose      (gridDim.z=3 converts H output to col-major)
 *   3. ssimulacra2_blur_v3_transposed (gridDim.z=3 fuses all 3 V passes, coalesced reads)
 *
 * The old dispatch issued 6 launches per blur (2 per channel × 3).
 * With 5 blurs per scale × 6 scales = 30 blurs/frame, the reduction
 * is from 180 to 90 kernel launches per frame.
 *
 * V-pass coalescing: the transpose converts H-pass output from
 * row-major to column-major so the V-pass IIR reads consecutive
 * addresses for each column scan. See ssimulacra2_blur.cu for the
 * detailed analysis. */

/* Fused 3-channel H pass — gridDim = (ceil(ch/BLOCK), 1, 3). */
static int ss2c_launch_blur_h3(Ssimu2StateCuda *s, CudaFunctions *cu_f, CUdeviceptr in_buf,
                               CUdeviceptr out_buf, unsigned cw, unsigned ch)
{
    float n2_0 = s->rg_n2[0];
    float n2_1 = s->rg_n2[1];
    float n2_2 = s->rg_n2[2];
    float d1_0 = s->rg_d1[0];
    float d1_1 = s->rg_d1[1];
    float d1_2 = s->rg_d1[2];
    int radius = s->rg_radius;
    unsigned plane_stride = s->width * s->height;
    void *args[] = {&in_buf, &out_buf, &cw,   &ch,   &n2_0,   &n2_1,
                    &n2_2,   &d1_0,    &d1_1, &d1_2, &radius, &plane_stride};
    unsigned grid_x = (ch + SS2C_BLUR_BLOCK - 1u) / SS2C_BLUR_BLOCK;
    /* gridDim.z = 3: one z-slice per XYB channel. */
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_blur_h3, grid_x, 1, 3u, SS2C_BLUR_BLOCK, 1, 1, 0,
                                           s->str, args, NULL));
    return 0;
}

/* Transpose: row-major → col-major, gridDim.z=3.
 * Tile shape: (SS2C_TILE, SS2C_TILE, 1) per z-slice. */
static int ss2c_launch_transpose(Ssimu2StateCuda *s, CudaFunctions *cu_f, CUdeviceptr in_buf,
                                 CUdeviceptr out_buf, unsigned cw, unsigned ch)
{
    unsigned plane_stride = s->width * s->height;
    void *args[] = {&in_buf, &out_buf, &cw, &ch, &plane_stride};
    unsigned gx = (cw + SS2C_TILE - 1u) / SS2C_TILE;
    unsigned gy = (ch + SS2C_TILE - 1u) / SS2C_TILE;
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_transpose, gx, gy, 3u, SS2C_TILE, SS2C_TILE, 1,
                                           0, s->str, args, NULL));
    return 0;
}

/* Fused 3-channel V pass on col-major input — gridDim = (ceil(cw/BLOCK), 1, 3). */
static int ss2c_launch_blur_v3_transposed(Ssimu2StateCuda *s, CudaFunctions *cu_f,
                                          CUdeviceptr in_transposed, CUdeviceptr out_buf,
                                          unsigned cw, unsigned ch)
{
    float n2_0 = s->rg_n2[0];
    float n2_1 = s->rg_n2[1];
    float n2_2 = s->rg_n2[2];
    float d1_0 = s->rg_d1[0];
    float d1_1 = s->rg_d1[1];
    float d1_2 = s->rg_d1[2];
    int radius = s->rg_radius;
    unsigned plane_stride = s->width * s->height;
    void *args[] = {&in_transposed, &out_buf, &cw,   &ch,   &n2_0,   &n2_1,
                    &n2_2,          &d1_0,    &d1_1, &d1_2, &radius, &plane_stride};
    unsigned grid_x = (cw + SS2C_BLUR_BLOCK - 1u) / SS2C_BLUR_BLOCK;
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_blur_v3_transposed, grid_x, 1, 3u,
                                           SS2C_BLUR_BLOCK, 1, 1, 0, s->str, args, NULL));
    return 0;
}

static int ss2c_blur_3plane(Ssimu2StateCuda *s, CudaFunctions *cu_f, CUdeviceptr in_buf,
                            CUdeviceptr out_buf, unsigned scale)
{
    const unsigned cw = s->scale_w[scale];
    const unsigned ch = s->scale_h[scale];
    CUdeviceptr scratch = (CUdeviceptr)s->d_blur_scratch->data;
    CUdeviceptr transpose_buf = (CUdeviceptr)s->d_transpose_buf->data;
    int err = 0;

    /* H pass: fused 3-channel. Output → scratch (row-major). */
    err = ss2c_launch_blur_h3(s, cu_f, in_buf, scratch, cw, ch);
    if (err)
        return err;
    /* Transpose: scratch (row-major) → transpose_buf (col-major). */
    err = ss2c_launch_transpose(s, cu_f, scratch, transpose_buf, cw, ch);
    if (err)
        return err;
    /* V pass: fused 3-channel on col-major input → out_buf (row-major). */
    err = ss2c_launch_blur_v3_transposed(s, cu_f, transpose_buf, out_buf, cw, ch);
    if (err)
        return err;
    return 0;
}

/* Per-pixel SSIM + EdgeDiff combine in double precision. Mirrors
 * libvmaf/src/feature/ssimulacra2.c::ssim_map + ::edge_diff_map.
 * Splitting would break the line-for-line scalar-diff audit trail
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5 sweep
 * closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2c_host_combine(const Ssimu2StateCuda *s, int scale, double avg_ssim[6],
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

/* Mirror of ssimulacra2.c::pool_score. */
static double ss2c_pool_score(const double avg_ssim[6][6], const double avg_ed[6][12],
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

/* Per-scale GPU work: 3 mul + 5 blur. After this returns the host
 * mu/sigma buffers are populated and the host combine can run.
 * Body mirrors the CPU dispatch ordering step-by-step; splitting
 * would obscure the dispatch sequence required for parity audit
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5 sweep
 * closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static int ss2c_run_scale_gpu(Ssimu2StateCuda *s, CudaFunctions *cu_f, int scale)
{
    const size_t plane_full_pixels = (size_t)s->width * (size_t)s->height;
    const size_t plane_full_bytes = plane_full_pixels * sizeof(float);
    /* Only `scale_w * scale_h` pixels per plane carry valid data — the
     * rest of each plane's `plane_full_pixels` reservation is garbage
     * (host pre-pass) or untouched (GPU mu/s11/s22/s12 outputs). The
     * device-side allocations stay full-size so `plane_stride` offsets
     * remain valid in the kernels; only the `cuMemcpyHtoDAsync` /
     * `cuMemcpyDtoHAsync` byte counts shrink to the valid sub-region.
     * At scale 2 of 1080p that is 518 KB / plane vs the previous 8 MB
     * full-plane transfer per copy (≈15× PCIe traffic reduction). */
    const size_t scale_pixels = (size_t)s->scale_w[scale] * (size_t)s->scale_h[scale];
    const size_t scale_bytes_per_plane = scale_pixels * sizeof(float);
    int err = 0;

    CUdeviceptr ref_xyb = (CUdeviceptr)s->d_ref_xyb->data;
    CUdeviceptr dis_xyb = (CUdeviceptr)s->d_dis_xyb->data;
    CUdeviceptr mul_buf = (CUdeviceptr)s->d_mul_buf->data;
    CUdeviceptr mu1 = (CUdeviceptr)s->d_mu1->data;
    CUdeviceptr mu2 = (CUdeviceptr)s->d_mu2->data;
    CUdeviceptr ds11 = (CUdeviceptr)s->d_s11->data;
    CUdeviceptr ds22 = (CUdeviceptr)s->d_s22->data;
    CUdeviceptr ds12 = (CUdeviceptr)s->d_s12->data;

    /* Upload XYB buffers (host computed) — per-plane, only the valid
     * sub-region. */
    for (size_t c = 0; c < 3u; c++) {
        const size_t plane_off_bytes = c * plane_full_bytes;
        CHECK_CUDA_RETURN(cu_f, cuMemcpyHtoDAsync(ref_xyb + plane_off_bytes,
                                                  (const uint8_t *)s->h_ref_xyb + plane_off_bytes,
                                                  scale_bytes_per_plane, s->str));
        CHECK_CUDA_RETURN(cu_f, cuMemcpyHtoDAsync(dis_xyb + plane_off_bytes,
                                                  (const uint8_t *)s->h_dis_xyb + plane_off_bytes,
                                                  scale_bytes_per_plane, s->str));
    }

    /* 1) ref² → mul → blur into s11. */
    err = ss2c_launch_mul3(s, cu_f, ref_xyb, ref_xyb, mul_buf, (unsigned)scale);
    if (err)
        return err;
    err = ss2c_blur_3plane(s, cu_f, mul_buf, ds11, scale);
    if (err)
        return err;

    /* 2) dis² → mul → blur into s22. */
    err = ss2c_launch_mul3(s, cu_f, dis_xyb, dis_xyb, mul_buf, (unsigned)scale);
    if (err)
        return err;
    err = ss2c_blur_3plane(s, cu_f, mul_buf, ds22, scale);
    if (err)
        return err;

    /* 3) ref·dis → mul → blur into s12. */
    err = ss2c_launch_mul3(s, cu_f, ref_xyb, dis_xyb, mul_buf, (unsigned)scale);
    if (err)
        return err;
    err = ss2c_blur_3plane(s, cu_f, mul_buf, ds12, scale);
    if (err)
        return err;

    /* 4) blur ref_xyb → mu1. */
    err = ss2c_blur_3plane(s, cu_f, ref_xyb, mu1, scale);
    if (err)
        return err;
    /* 5) blur dis_xyb → mu2. */
    err = ss2c_blur_3plane(s, cu_f, dis_xyb, mu2, scale);
    if (err)
        return err;

    /* Download blurred buffers — per-plane, only the valid sub-region.
     * Same rationale as the H2D loop above: kernels only populate the
     * first `scale_w * scale_h` floats of each plane; the unused tail
     * of `plane_full_pixels` is never read by the host combine. */
    for (size_t c = 0; c < 3u; c++) {
        const size_t plane_off_bytes = c * plane_full_bytes;
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync((uint8_t *)s->h_mu1 + plane_off_bytes,
                                            mu1 + plane_off_bytes, scale_bytes_per_plane, s->str));
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync((uint8_t *)s->h_mu2 + plane_off_bytes,
                                            mu2 + plane_off_bytes, scale_bytes_per_plane, s->str));
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync((uint8_t *)s->h_s11 + plane_off_bytes,
                                            ds11 + plane_off_bytes, scale_bytes_per_plane, s->str));
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync((uint8_t *)s->h_s22 + plane_off_bytes,
                                            ds22 + plane_off_bytes, scale_bytes_per_plane, s->str));
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync((uint8_t *)s->h_s12 + plane_off_bytes,
                                            ds12 + plane_off_bytes, scale_bytes_per_plane, s->str));
    }
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));
    return 0;
}

/* Per-scale orchestration mirrors the CPU extract loop step-by-step;
 * splitting would obscure the dispatch ordering required for parity
 * audit (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5
 * sweep closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static int extract_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    Ssimu2StateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), out);
    ctx_pushed = 1;

    /* Stage 0: D2H-copy raw YUV planes into pinned host scratch.
     * picture_cuda hands us a VmafPicture whose `data[]` is a
     * CUdeviceptr; direct host reads via `ss2c_read_plane` would
     * segfault. Build a synthetic host-side VmafPicture mirror that
     * matches the layout so `ss2c_picture_to_linear_rgb` is unchanged.
     *
     * The CPU sibling (ssimulacra2.c) takes a host VmafPicture
     * directly — this D2H is the GPU-extractor cost we pay for not
     * routing the YUV→linear-RGB pass through a kernel. Per ADR-0201,
     * keeping the pre-pass on host is what unlocks places=4. */
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    VmafPicture host_ref = *ref_pic;
    VmafPicture host_dis = *dist_pic;
    for (int p = 0; p < 3; p++) {
        const unsigned pw = ref_pic->w[p];
        const unsigned ph = ref_pic->h[p];
        const ptrdiff_t row_bytes = (ptrdiff_t)pw * (ptrdiff_t)bpp;
        s->plane_w[p] = pw;
        s->plane_h[p] = ph;
        s->plane_row_bytes[p] = row_bytes;

        CUDA_MEMCPY2D cpy = {0};
        cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.srcDevice = (CUdeviceptr)ref_pic->data[p];
        cpy.srcPitch = ref_pic->stride[p];
        cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
        cpy.dstHost = s->h_ref_raw[p];
        cpy.dstPitch = (size_t)row_bytes;
        cpy.WidthInBytes = (size_t)row_bytes;
        cpy.Height = ph;
        CHECK_CUDA_GOTO(cu_f, cuMemcpy2DAsync(&cpy, s->str), out);
        cpy.srcDevice = (CUdeviceptr)dist_pic->data[p];
        cpy.srcPitch = dist_pic->stride[p];
        cpy.dstHost = s->h_dis_raw[p];
        CHECK_CUDA_GOTO(cu_f, cuMemcpy2DAsync(&cpy, s->str), out);

        host_ref.data[p] = s->h_ref_raw[p];
        host_ref.stride[p] = row_bytes;
        host_dis.data[p] = s->h_dis_raw[p];
        host_dis.stride[p] = row_bytes;
    }
    /* Block until the 6 D2H copies are visible to the host. */
    CHECK_CUDA_GOTO(cu_f, cuStreamSynchronize(s->str), out);

    /* Stage 1: host YUV → linear RGB on the pinned host buffers. */
    ss2c_picture_to_linear_rgb(s, &host_ref, s->h_ref_lin);
    ss2c_picture_to_linear_rgb(s, &host_dis, s->h_dis_lin);

    /* Stage 2: per-scale loop. */
    double avg_ssim[6][6] = {{0}};
    double avg_ed[6][12] = {{0}};
    int completed = 0;
    unsigned cw = s->width;
    unsigned ch = s->height;
    const size_t plane_full = (size_t)s->width * (size_t)s->height;

    for (int scale = 0; scale < SS2C_NUM_SCALES; scale++) {
        if (cw < 8u || ch < 8u)
            break;

        ss2c_host_linear_rgb_to_xyb(s->h_ref_lin, s->h_ref_xyb, cw, ch, plane_full);
        ss2c_host_linear_rgb_to_xyb(s->h_dis_lin, s->h_dis_xyb, cw, ch, plane_full);

        int err = ss2c_run_scale_gpu(s, cu_f, scale);
        if (err) {
            _cuda_err = err;
            goto out;
        }

        ss2c_host_combine(s, scale, avg_ssim[scale], avg_ed[scale]);
        completed++;

        if (scale + 1 < SS2C_NUM_SCALES) {
            const unsigned nw = (cw + 1) / 2;
            const unsigned nh = (ch + 1) / 2;
            /* Use the pinned scratch pre-allocated by `ss2c_alloc_buffers`
             * — the previous per-scale `malloc(3 * plane_full *
             * sizeof(float))` cost a fresh `mmap`/`brk` pair per scale
             * under memory pressure (24 MB × up to 5 scales / frame at
             * 1080p). The scratch is reused for ref then dis on every
             * scale: ref is downsampled and copied back before dis is
             * touched, so a single buffer per side is sufficient. */
            ss2c_downsample_2x2(s->h_ref_lin, cw, ch, s->h_ref_lin_ds, nw, nh, plane_full);
            for (int c = 0; c < 3; c++)
                memcpy(s->h_ref_lin + (size_t)c * plane_full,
                       s->h_ref_lin_ds + (size_t)c * plane_full,
                       (size_t)nw * (size_t)nh * sizeof(float));
            ss2c_downsample_2x2(s->h_dis_lin, cw, ch, s->h_dis_lin_ds, nw, nh, plane_full);
            for (int c = 0; c < 3; c++)
                memcpy(s->h_dis_lin + (size_t)c * plane_full,
                       s->h_dis_lin_ds + (size_t)c * plane_full,
                       (size_t)nw * (size_t)nh * sizeof(float));
            cw = nw;
            ch = nh;
        }
    }

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), out);
    ctx_pushed = 0;

    const double score = ss2c_pool_score(avg_ssim, avg_ed, completed);
    return vmaf_feature_collector_append(feature_collector, "ssimulacra2", score, index);

out:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
    return _cuda_err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    Ssimu2StateCuda *s = fex->priv;
    if (!s)
        return 0;
    CudaFunctions *cu_f = fex->cu_state ? fex->cu_state->f : NULL;
    int ret = 0;

    if (cu_f && s->str) {
        (void)cu_f->cuStreamSynchronize(s->str);
        /* Unload the two PTX modules loaded by `init_fex_cuda` —
         * `cuModuleLoadData` allocates ~200-500 KB of GPU-resident
         * module backing store per module, none of which is reclaimed
         * by `cuStreamDestroy` or `cuCtxDestroy` on a primary context.
         * Skipping these calls leaks the modules every `vmaf_close()`
         * cycle (caught by `compute-sanitizer --tool memcheck` on a
         * 100-iteration init/extract/close loop). Guarded by null
         * checks so partial-init failure paths are still safe. */
        if (s->module_blur)
            (void)cu_f->cuModuleUnload(s->module_blur);
        if (s->module_mul)
            (void)cu_f->cuModuleUnload(s->module_mul);
        (void)cu_f->cuStreamDestroy(s->str);
    }

#define SS2C_FREE_DEV(b)                                                                           \
    do {                                                                                           \
        if (s->b) {                                                                                \
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->b);                                     \
            free(s->b);                                                                            \
        }                                                                                          \
    } while (0)
    SS2C_FREE_DEV(d_ref_lin);
    SS2C_FREE_DEV(d_dis_lin);
    SS2C_FREE_DEV(d_ref_xyb);
    SS2C_FREE_DEV(d_dis_xyb);
    SS2C_FREE_DEV(d_mul_buf);
    SS2C_FREE_DEV(d_blur_scratch);
    SS2C_FREE_DEV(d_transpose_buf);
    SS2C_FREE_DEV(d_mu1);
    SS2C_FREE_DEV(d_mu2);
    SS2C_FREE_DEV(d_s11);
    SS2C_FREE_DEV(d_s22);
    SS2C_FREE_DEV(d_s12);
#undef SS2C_FREE_DEV

#define SS2C_FREE_HOST(p)                                                                          \
    do {                                                                                           \
        if (s->p) {                                                                                \
            ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->p);                                \
            s->p = NULL;                                                                           \
        }                                                                                          \
    } while (0)
    SS2C_FREE_HOST(h_ref_lin);
    SS2C_FREE_HOST(h_dis_lin);
    SS2C_FREE_HOST(h_ref_lin_ds);
    SS2C_FREE_HOST(h_dis_lin_ds);
    SS2C_FREE_HOST(h_ref_xyb);
    SS2C_FREE_HOST(h_dis_xyb);
    SS2C_FREE_HOST(h_mu1);
    SS2C_FREE_HOST(h_mu2);
    SS2C_FREE_HOST(h_s11);
    SS2C_FREE_HOST(h_s22);
    SS2C_FREE_HOST(h_s12);
#undef SS2C_FREE_HOST

    for (int p = 0; p < 3; p++) {
        if (s->h_ref_raw[p]) {
            ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->h_ref_raw[p]);
            s->h_ref_raw[p] = NULL;
        }
        if (s->h_dis_raw[p]) {
            ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->h_dis_raw[p]);
            s->h_dis_raw[p] = NULL;
        }
    }
    return ret;
}

static const char *provided_features[] = {"ssimulacra2", NULL};

VmafFeatureExtractor vmaf_fex_ssimulacra2_cuda = {
    .name = "ssimulacra2_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(Ssimu2StateCuda),
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    .provided_features = provided_features,
};
