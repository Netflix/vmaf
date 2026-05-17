/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ssimulacra2 feature kernel on the Vulkan backend
 *  (T7-23 / GPU long-tail batch 3 part 7 / ADR-0200).
 *  Vulkan twin of the CPU `ssimulacra2` extractor in
 *  libvmaf/src/feature/ssimulacra2.c.
 *
 *  Per-frame pipeline (host orchestration):
 *    1. Host: YUV → linear RGB (full-res, scalar libjxl path,
 *       deterministic LUT-based sRGB EOTF — ADR-0164).
 *    2. Host: build 6-level linear-RGB pyramid via 2×2 box
 *       downsample.
 *    3. For each scale (up to 6):
 *       a. GPU: linear RGB → XYB (`ssimulacra2_xyb.comp`).
 *          Two dispatches (ref + dis).
 *       b. GPU: 5 elementwise products (ref², dis², ref·dis,
 *          plus pass-throughs for ref / dis themselves) via
 *          `ssimulacra2_mul.comp`.
 *       c. GPU: 5 separable IIR blurs via `ssimulacra2_blur.comp`,
 *          two passes each (one WG per row, one WG per column).
 *       d. GPU: per-pixel SSIM + EdgeDiff stats via
 *          `ssimulacra2_ssim.comp` — 18 partials per WG.
 *       e. Host: reduce partials in double, store per-scale.
 *    4. Host: pool 108 weighted norms via the libjxl polynomial
 *       (mirrors `pool_score`). Emit `ssimulacra2`.
 *
 *  Min-dim guard: the 6-scale pyramid + IIR-blur radius (~6 px)
 *  needs the smallest scale to stay >= 8 px on each axis. Match
 *  the CPU `if (cw < 8 || ch < 8) break` early-exit by allowing
 *  init for any (w, h) >= 8 (the GPU just runs fewer scales).
 *
 *  Precision contract: ADR-0192 sets places=2 for ssimulacra2.
 *  The host-side YUV→linear-RGB + 2×2 downsample mean the only
 *  GPU-introduced drift is XYB (cube root), IIR blur (FMA
 *  reassociation), and the SSIM/EdgeDiff per-WG reductions; all
 *  are float-stable to within ~1e-4 on 8-bit YUV inputs.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "log.h"
#include "mem.h"

#include "feature/ssimulacra2_math.h"

/* ADR-0242: SIMD host-kernel dispatch for the Vulkan extractor's three
 * CPU-bound host paths (YUV→linRGB, linRGB→XYB, 2×2 downsample). */
#if ARCH_X86 || ARCH_AARCH64
#include "cpu.h"
#endif
#if ARCH_X86
#include "feature/ssimulacra2_simd_common.h"
#include "feature/x86/ssimulacra2_avx2.h"
#include "feature/x86/ssimulacra2_host_avx2.h"
#include "x86/cpu.h"
#endif
#if ARCH_AARCH64
#include "feature/ssimulacra2_simd_common.h"
#include "feature/arm64/ssimulacra2_neon.h"
#include "feature/arm64/ssimulacra2_host_neon.h"
#include "arm/cpu.h"
#endif

#include "../../vulkan/kernel_template.h"
#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"

#include "ssimulacra2_xyb_spv.h"
#include "ssimulacra2_blur_spv.h"
#include "ssimulacra2_mul_spv.h"
#include "ssimulacra2_ssim_spv.h"

#define SS2V_NUM_SCALES 6
#define SS2V_WG_X 16
#define SS2V_WG_Y 8
#define SS2V_PARTIAL_SLOTS 18 /* per-WG: 6 ssim + 12 edge */
#define SS2V_SIGMA 1.5
#define SS2V_PI 3.14159265358979323846

/* Bindings counts per pipeline. */
#define SS2V_XYB_BINDINGS 6  /* Rin, Gin, Bin, Xout, Yout, Bout */
#define SS2V_MUL_BINDINGS 3  /* a, b, out */
#define SS2V_BLUR_BINDINGS 2 /* in, out */
#define SS2V_SSIM_BINDINGS 8 /* mu1, mu2, s11, s22, s12, img1, img2, partials */

enum yuv_matrix_v {
    SS2V_MATRIX_BT709_LIMITED = 0,
    SS2V_MATRIX_BT601_LIMITED = 1,
    SS2V_MATRIX_BT709_FULL = 2,
    SS2V_MATRIX_BT601_FULL = 3,
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

typedef struct {
    int yuv_matrix;
    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Per-scale linear-RGB pyramid dimensions. */
    unsigned scale_w[SS2V_NUM_SCALES];
    unsigned scale_h[SS2V_NUM_SCALES];

    /* Recursive Gaussian coefficients (sigma=1.5). */
    float rg_n2[3];
    float rg_d1[3];
    int rg_radius;

    /* Vulkan context. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipelines (specialised per scale where needed). 4 bundles via
     * `vulkan/kernel_template.h` (ADR-0246). Each bundle owns its own
     * descriptor pool + pipeline layout + shader module + DSL +
     * the scale-0 pipeline. The `*_pipelines[0]` slot of each per-bundle
     * array aliases the bundle's `pipeline` field; the variant slots
     * are created via `vmaf_vulkan_kernel_pipeline_add_variant()` and
     * must be destroyed before `vmaf_vulkan_kernel_pipeline_destroy()`
     * to avoid double-freeing the aliased base. */
    VmafVulkanKernelPipeline pl_xyb;
    VkPipeline xyb_pipelines[SS2V_NUM_SCALES];

    VmafVulkanKernelPipeline pl_mul;
    VkPipeline mul_pipelines[SS2V_NUM_SCALES];

    VmafVulkanKernelPipeline pl_blur;
    VkPipeline blur_pipelines_h[SS2V_NUM_SCALES];
    VkPipeline blur_pipelines_v[SS2V_NUM_SCALES];

    VmafVulkanKernelPipeline pl_ssim;
    VkPipeline ssim_pipelines[SS2V_NUM_SCALES];

    /* GPU buffers (sized for full resolution; per-scale dispatches
     * use only the leading prefix). All planar 3-channel layouts
     * are stored as one buffer of size 3 * w * h * sizeof(float)
     * with planes contiguous (X | Y | B). */
    VmafVulkanBuffer *ref_lin; /* 3 planes, host-mapped (host writes) */
    VmafVulkanBuffer *dis_lin;
    VmafVulkanBuffer *ref_xyb; /* 3 planes */
    VmafVulkanBuffer *dis_xyb;
    VmafVulkanBuffer *mul_buf; /* 3 planes scratch */
    VmafVulkanBuffer *mu1;     /* 3 planes */
    VmafVulkanBuffer *mu2;
    VmafVulkanBuffer *s11;
    VmafVulkanBuffer *s22;
    VmafVulkanBuffer *s12;
    VmafVulkanBuffer *blur_scratch; /* 3 planes — H-pass output → V-pass input */
    VmafVulkanBuffer *partials;     /* per-scale max wg_count * 18 floats */

    /* Largest-scale partials wg_count (sizing). */
    unsigned max_wg_count;

    /* ADR-0242: SIMD dispatch for host-side hot paths.
     * NULL = scalar fallback (non-x86/aarch64 or no AVX2/NEON). */
    /* YUV → linear RGB (picture-to-planes). Uses simd_plane_t abstraction. */
    void (*ptlr_fn)(int yuv_matrix, unsigned bpc, unsigned w, unsigned h,
                    const simd_plane_t planes[3], float *out);
    /* Linear RGB → XYB (plane_stride form for pyramid buffers). */
    void (*xyb_host_fn)(const float *lin, float *xyb, unsigned w, unsigned h, size_t plane_stride);
    /* 2×2 box downsample (plane_stride form). */
    void (*down_host_fn)(const float *in, unsigned iw, unsigned ih, float *out, unsigned ow,
                         unsigned oh, size_t plane_stride);

    /* Submit pool: 1 slot, reused once per scale (ss2v_run_scale is called
     * sequentially; each call waits before returning, so slot 0 is safely
     * recycled across the scale loop). Eliminates per-scale vkCreateFence +
     * vkAllocateCommandBuffers + vkFreeCommandBuffers + vkDestroyFence.
     * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
    VmafVulkanKernelSubmitPool sub_pool;
} Ssimu2VkState;

typedef struct {
    uint32_t width;
    uint32_t height;
} XybPushConsts;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t plane_count;
    uint32_t plane_stride;
} MulPushConsts;

typedef struct {
    uint32_t width;
    uint32_t height;
    float n2_0;
    float n2_1;
    float n2_2;
    float d1_0;
    float d1_1;
    float d1_2;
    int32_t radius;
    uint32_t in_offset;
    uint32_t out_offset;
} BlurPushConsts;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t num_workgroups_x;
    uint32_t plane_size;
} SsimPushConsts;

static const VmafOption options[] = {
    {
        .name = "yuv_matrix",
        .help = "YUV→RGB matrix: 0=bt709_limited (default), 1=bt601_limited, "
                "2=bt709_full, 3=bt601_full",
        .offset = offsetof(Ssimu2VkState, yuv_matrix),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = SS2V_MATRIX_BT709_LIMITED,
        .min = 0,
        .max = 3,
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* Recursive Gaussian coefficient setup — bit-identical port of
 * create_recursive_gaussian in ssimulacra2.c.                        */
/* ------------------------------------------------------------------ */

/* Verbatim port of the libjxl Charalampidis 2016 derivation; splitting
 * the linear-system solve would obscure the scalar-diff audit trail
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5 sweep
 * closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2v_setup_gaussian(Ssimu2VkState *s, double sigma)
{
    const double radius = round(3.2795 * sigma + 0.2546);
    const double pi_div_2r = SS2V_PI / (2.0 * radius);
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
/* Host-side YUV → linear-RGB + 2×2 box downsample.
 * Match-for-match ports of ssimulacra2.c::picture_to_linear_rgb and
 * downsample_2x2 (with 3 planes packed contiguously).               */
/* ------------------------------------------------------------------ */

static inline float ss2v_clampf(float v, float lo, float hi)
{
    if (v < lo)
        return lo;
    if (v > hi)
        return hi;
    return v;
}

static inline float ss2v_read_plane(const VmafPicture *pic, int plane, int x, int y)
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

/* Verbatim port of ssimulacra2.c::picture_to_linear_rgb — identical
 * float order so host outputs bit-match the CPU path
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5 sweep
 * closeout — ADR-0278).
 * NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static void ss2v_picture_to_linear_rgb(const Ssimu2VkState *s, const VmafPicture *pic, float *out)
{
    const unsigned w = s->width;
    const unsigned h = s->height;
    const size_t plane_sz = (size_t)w * (size_t)h;
    /* Per-channel stride matches the full-resolution plane size — kept
     * constant across pyramid scales so the GPU shaders' channel
     * offsets (= c * full_plane) line up. At scale 0 this equals
     * w*h, so writes here are densely packed; at smaller scales
     * the host downsample preserves the same stride. */
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
    case SS2V_MATRIX_BT709_FULL:
        limited = 0;
        /* fall through */
        // fallthrough
    case SS2V_MATRIX_BT709_LIMITED:
        kr = 0.2126f;
        kg = 0.7152f;
        kb = 0.0722f;
        break;
    case SS2V_MATRIX_BT601_FULL:
        limited = 0;
        /* fall through */
        // fallthrough
    case SS2V_MATRIX_BT601_LIMITED:
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
            float Y = ss2v_read_plane(pic, 0, (int)x, (int)y) * inv_peak;
            float U = ss2v_read_plane(pic, 1, (int)x, (int)y) * inv_peak;
            float V = ss2v_read_plane(pic, 2, (int)x, (int)y) * inv_peak;
            float Yn = (Y - y_off) * y_scale;
            float Un = (U - c_off) * c_scale;
            float Vn = (V - c_off) * c_scale;
            float R = Yn + cr_r * Vn;
            float G = Yn + cb_g * Un + cr_g * Vn;
            float B = Yn + cb_b * Un;
            R = ss2v_clampf(R, 0.0f, 1.0f);
            G = ss2v_clampf(G, 0.0f, 1.0f);
            B = ss2v_clampf(B, 0.0f, 1.0f);
            const size_t idx = (size_t)y * w + x;
            rp[idx] = vmaf_ss2_srgb_eotf(R);
            gp[idx] = vmaf_ss2_srgb_eotf(G);
            bp[idx] = vmaf_ss2_srgb_eotf(B);
        }
    }
}

/* Host-side linear-RGB → XYB (verbatim port of
 * libvmaf/src/feature/ssimulacra2.c::linear_rgb_to_xyb). Bit-exact
 * with the CPU extractor. We run XYB host-side instead of in a
 * compute shader because the GLSL→SPIR-V→driver→GPU compile chain
 * does not (in practice) preserve the exact float operation order
 * of the CPU port even with `precise` and `NoContraction`
 * decorations on every operation: lavapipe / Mesa anv / RADV all
 * produced ~1.7e-6 max per-pixel drift on the X plane vs CPU,
 * which compounds through the 6-scale IIR and the 108-weighted-pool
 * to a 1.5e-2 pooled-score drift (places=1). The host XYB is
 * single-precision float and trivially bit-exact; cost is ~3 ms
 * per scale 0 frame on a 576x324 input vs <1 ms for the GPU
 * dispatch — net wall-time impact is well below the IIR /
 * descriptor-allocation overhead. See ADR-0201 §Precision
 * investigation for the full per-tactic measurement table. */
static void ss2v_host_linear_rgb_to_xyb(const float *lin, float *xyb, unsigned w, unsigned h,
                                        size_t plane_stride)
{
    /* Constants — bit-identical to ssimulacra2.c::linear_rgb_to_xyb. */
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

/* Downsample a 3-plane buffer where each plane occupies a contiguous
 * `plane_stride` slot (always the full-resolution plane size — kept
 * constant across scales for consistency with the GPU buffers). The
 * input plane data of dimensions iw×ih lives at the head of each
 * plane's slot; output plane data of dimensions ow×oh is written
 * back into the head of each plane's slot in `out`. */
static void ss2v_downsample_2x2(const float *in, unsigned iw, unsigned ih, float *out, unsigned ow,
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
/* Pipeline / descriptor-set / buffer setup                           */
/* ------------------------------------------------------------------ */

/* Build a sibling compute pipeline with up to 3 int spec constants
 * via `vmaf_vulkan_kernel_pipeline_add_variant`. The base bundle
 * supplies the layout + shader module; this helper formats the spec
 * payload and dispatches to the template. */
static int ss2v_build_pipeline_int3(Ssimu2VkState *s, const VmafVulkanKernelPipeline *bundle,
                                    int n_specs, const int *spec_vals, VkPipeline *out)
{
    VkSpecializationMapEntry entries[3];
    int data[3];
    for (int i = 0; i < n_specs; i++) {
        entries[i].constantID = (uint32_t)i;
        entries[i].offset = (uint32_t)(i * (int)sizeof(int));
        entries[i].size = sizeof(int);
        data[i] = spec_vals[i];
    }
    VkSpecializationInfo si = {
        .mapEntryCount = (uint32_t)n_specs,
        .pMapEntries = entries,
        .dataSize = (size_t)n_specs * sizeof(int),
        .pData = data,
    };
    VkComputePipelineCreateInfo cpci = {
        .stage =
            {
                .pName = "main",
                .pSpecializationInfo = &si,
            },
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, bundle, &cpci, out);
}

/* Build the base pipeline of a `VmafVulkanKernelPipeline` bundle for
 * scale 0. The template helper owns layout + shader + DSL + pool;
 * the bundle's `pipeline` field becomes the scale-0 pipeline. */
static int ss2v_build_base_bundle(Ssimu2VkState *s, VmafVulkanKernelPipeline *bundle,
                                  uint32_t binding_count, uint32_t pc_size, const uint32_t *spv,
                                  size_t spv_size, uint32_t max_sets, int n_specs,
                                  const int *spec_vals)
{
    VkSpecializationMapEntry entries[3];
    int data[3];
    for (int i = 0; i < n_specs; i++) {
        entries[i].constantID = (uint32_t)i;
        entries[i].offset = (uint32_t)(i * (int)sizeof(int));
        entries[i].size = sizeof(int);
        data[i] = spec_vals[i];
    }
    VkSpecializationInfo si = {
        .mapEntryCount = (uint32_t)n_specs,
        .pMapEntries = entries,
        .dataSize = (size_t)n_specs * sizeof(int),
        .pData = data,
    };
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = binding_count,
        .push_constant_size = pc_size,
        .spv_bytes = spv,
        .spv_size = spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &si,
                    },
            },
        .max_descriptor_sets = max_sets,
    };
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, bundle);
}

static int ss2v_create_pipelines(Ssimu2VkState *s)
{
    int err = 0;

    /* Per-bundle descriptor pool sizing.
     *
     * Per scale, ss2v_run_scale allocates:
     *   - 2 xyb sets, 3 mul sets, 10 blur sets (5×{H,V}), 1 ssim set
     *
     * Each bundle owns its own pool, so size each pool for the
     * worst case across all SS2V_NUM_SCALES scales of its own
     * dispatch type. The fork's per-scale loop frees sets at the
     * tail, but Vulkan's `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`
     * makes per-scale reuse safe — this sizing is the pessimistic
     * upper bound on simultaneously-allocated sets across the loop's
     * lifetime (we'd actually only need the per-scale total, but the
     * extra headroom is cheap and matches the legacy shared-pool slack). */
    const uint32_t xyb_sets = 2u * SS2V_NUM_SCALES;
    const uint32_t mul_sets = 3u * SS2V_NUM_SCALES;
    const uint32_t blur_sets = 5u * 2u * SS2V_NUM_SCALES;
    const uint32_t ssim_sets = 1u * SS2V_NUM_SCALES;

    const int wh0[2] = {(int)s->scale_w[0], (int)s->scale_h[0]};
    const int wh0_blur_h[3] = {(int)s->scale_w[0], (int)s->scale_h[0], 0};

    /* XYB bundle (6 SSBO bindings). */
    err = ss2v_build_base_bundle(s, &s->pl_xyb, SS2V_XYB_BINDINGS, sizeof(XybPushConsts),
                                 ssimulacra2_xyb_spv, ssimulacra2_xyb_spv_size, xyb_sets,
                                 /*n_specs=*/2, wh0);
    if (err)
        return err;
    s->xyb_pipelines[0] = s->pl_xyb.pipeline;

    /* MUL bundle (3 SSBO bindings). */
    err = ss2v_build_base_bundle(s, &s->pl_mul, SS2V_MUL_BINDINGS, sizeof(MulPushConsts),
                                 ssimulacra2_mul_spv, ssimulacra2_mul_spv_size, mul_sets,
                                 /*n_specs=*/2, wh0);
    if (err)
        return err;
    s->mul_pipelines[0] = s->pl_mul.pipeline;

    /* BLUR bundle (2 SSBO bindings). The base pipeline is the H-pass
     * for scale 0; the V-pass for scale 0 is a sibling variant. */
    err = ss2v_build_base_bundle(s, &s->pl_blur, SS2V_BLUR_BINDINGS, sizeof(BlurPushConsts),
                                 ssimulacra2_blur_spv, ssimulacra2_blur_spv_size, blur_sets,
                                 /*n_specs=*/3, wh0_blur_h);
    if (err)
        return err;
    s->blur_pipelines_h[0] = s->pl_blur.pipeline;

    /* SSIM bundle (8 SSBO bindings). */
    err = ss2v_build_base_bundle(s, &s->pl_ssim, SS2V_SSIM_BINDINGS, sizeof(SsimPushConsts),
                                 ssimulacra2_ssim_spv, ssimulacra2_ssim_spv_size, ssim_sets,
                                 /*n_specs=*/2, wh0);
    if (err)
        return err;
    s->ssim_pipelines[0] = s->pl_ssim.pipeline;

    /* Variants: scales 1..N-1 for xyb/mul/ssim (2 specs); blur needs
     * an H + V pipeline per scale, with the H-pass at scale 0
     * already aliased into the bundle. */
    for (int i = 0; i < SS2V_NUM_SCALES; i++) {
        const int wh[2] = {(int)s->scale_w[i], (int)s->scale_h[i]};
        if (i > 0) {
            err = ss2v_build_pipeline_int3(s, &s->pl_xyb, 2, wh, &s->xyb_pipelines[i]);
            if (err)
                return err;
            err = ss2v_build_pipeline_int3(s, &s->pl_mul, 2, wh, &s->mul_pipelines[i]);
            if (err)
                return err;
            err = ss2v_build_pipeline_int3(s, &s->pl_ssim, 2, wh, &s->ssim_pipelines[i]);
            if (err)
                return err;

            const int wh_h[3] = {(int)s->scale_w[i], (int)s->scale_h[i], 0};
            err = ss2v_build_pipeline_int3(s, &s->pl_blur, 3, wh_h, &s->blur_pipelines_h[i]);
            if (err)
                return err;
        }
        const int wh_v[3] = {(int)s->scale_w[i], (int)s->scale_h[i], 1};
        err = ss2v_build_pipeline_int3(s, &s->pl_blur, 3, wh_v, &s->blur_pipelines_v[i]);
        if (err)
            return err;
    }

    return 0;
}

static int ss2v_alloc_buffers(Ssimu2VkState *s)
{
    const size_t plane_full = (size_t)s->width * (size_t)s->height;
    const size_t three_plane_bytes = 3u * plane_full * sizeof(float);
    int err = 0;

    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_lin, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_lin, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_xyb, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_xyb, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->mul_buf, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->mu1, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->mu2, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->s11, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->s22, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->s12, three_plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->blur_scratch, three_plane_bytes);

    /* Partials sized for largest scale (scale 0). */
    unsigned wgx = (s->scale_w[0] + SS2V_WG_X - 1) / SS2V_WG_X;
    unsigned wgy = (s->scale_h[0] + SS2V_WG_Y - 1) / SS2V_WG_Y;
    s->max_wg_count = wgx * wgy;
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->partials,
                                    (size_t)s->max_wg_count * SS2V_PARTIAL_SLOTS * sizeof(float));
    return err ? -ENOMEM : 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    Ssimu2VkState *s = fex->priv;

    /* Min-dim guard: 6-scale 2× downsample needs the smallest
     * scale to land at >= 8 px (matches the CPU `cw < 8u || ch < 8u`
     * loop break). For (w, h) = (32, 32) the 6th scale is 1×1 —
     * the host loop simply runs fewer scales. We accept (8, 8) as
     * the absolute minimum so even scale 0 has something to blur. */
    if (w < 8u || h < 8u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "ssimulacra2_vulkan: input %ux%u below 8x8 lower bound\n", w,
                 h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    ss2v_setup_gaussian(s, SS2V_SIGMA);

    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < SS2V_NUM_SCALES; i++) {
        s->scale_w[i] = (s->scale_w[i - 1] + 1) / 2;
        s->scale_h[i] = (s->scale_h[i - 1] + 1) / 2;
    }

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "ssimulacra2_vulkan: cannot create Vulkan context (%d)\n", err);
            return err;
        }
        s->owns_ctx = 1;
    }

    int err = ss2v_create_pipelines(s);
    if (err)
        return err;
    err = ss2v_alloc_buffers(s);
    if (err)
        return err;

    /* Pre-allocate submit pool (1 slot, reused sequentially per scale).
     * Eliminates per-scale vkAllocateCommandBuffers + vkCreateFence.
     * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;

    /* ADR-0242: runtime dispatch — select SIMD host kernels when available. */
    s->ptlr_fn = NULL;
    s->xyb_host_fn = NULL;
    s->down_host_fn = NULL;
#if ARCH_X86
    {
        const unsigned flags = vmaf_get_cpu_flags_x86();
        if (flags & VMAF_X86_CPU_FLAG_AVX2) {
            s->ptlr_fn = ssimulacra2_picture_to_linear_rgb_avx2;
            s->xyb_host_fn = ssimulacra2_host_linear_rgb_to_xyb_avx2;
            s->down_host_fn = ssimulacra2_host_downsample_2x2_avx2;
        }
    }
#endif
#if ARCH_AARCH64
    {
        const unsigned flags = vmaf_get_cpu_flags_arm();
        if (flags & VMAF_ARM_CPU_FLAG_NEON) {
            s->ptlr_fn = ssimulacra2_picture_to_linear_rgb_neon;
            s->xyb_host_fn = ssimulacra2_host_linear_rgb_to_xyb_neon;
            s->down_host_fn = ssimulacra2_host_downsample_2x2_neon;
        }
    }
#endif
    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-frame extract                                                  */
/* ------------------------------------------------------------------ */

static int ss2v_alloc_set(Ssimu2VkState *s, const VmafVulkanKernelPipeline *bundle,
                          VkDescriptorSet *out)
{
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = bundle->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &bundle->dsl,
    };
    return vkAllocateDescriptorSets(s->ctx->device, &dsai, out) == VK_SUCCESS ? 0 : -ENOMEM;
}

/* Bind buffers to a descriptor set. The (offset, range) pair is
 * always (0, VK_WHOLE_SIZE) — per-scale clipping is enforced by
 * the shader's bounds checks. */
static void ss2v_write_set(VkDevice dev, VkDescriptorSet set, unsigned n, VmafVulkanBuffer **bufs)
{
    VkDescriptorBufferInfo dbi[8];
    VkWriteDescriptorSet writes[8];
    for (unsigned i = 0; i < n; i++) {
        dbi[i] = (VkDescriptorBufferInfo){
            .buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(bufs[i]),
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(dev, n, writes, 0, NULL);
}

static void ss2v_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

/* Run XYB conversion on (lin -> xyb) for one frame at the given scale. */
static void ss2v_dispatch_xyb(Ssimu2VkState *s, VkCommandBuffer cmd, int scale, VkDescriptorSet set)
{
    XybPushConsts pc = {.width = s->scale_w[scale], .height = s->scale_h[scale]};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_xyb.pipeline_layout, 0, 1,
                            &set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_xyb.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                       &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->xyb_pipelines[scale]);
    uint32_t gx = (s->scale_w[scale] + SS2V_WG_X - 1) / SS2V_WG_X;
    uint32_t gy = (s->scale_h[scale] + SS2V_WG_Y - 1) / SS2V_WG_Y;
    vkCmdDispatch(cmd, gx, gy, 1);
}

static void ss2v_dispatch_mul(Ssimu2VkState *s, VkCommandBuffer cmd, int scale, VkDescriptorSet set,
                              uint32_t plane_count)
{
    MulPushConsts pc = {.width = s->scale_w[scale],
                        .height = s->scale_h[scale],
                        .plane_count = plane_count,
                        .plane_stride = s->width * s->height};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_mul.pipeline_layout, 0, 1,
                            &set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_mul.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                       &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->mul_pipelines[scale]);
    uint32_t gx = (s->scale_w[scale] + SS2V_WG_X - 1) / SS2V_WG_X;
    uint32_t gy = (s->scale_h[scale] + SS2V_WG_Y - 1) / SS2V_WG_Y;
    vkCmdDispatch(cmd, gx, gy, 1);
}

/* Issue one blur dispatch with explicit per-plane offsets.
 * The descriptor set is bound once per (in_buf, out_buf) pair
 * with VK_WHOLE_SIZE; all 3 channels share the same set and
 * the per-channel plane offset is encoded in the push constants
 * — this avoids the "update-descriptor-set-between-records"
 * trap (Vulkan reads bindings at execution time, not at record
 * time). */
static void ss2v_dispatch_blur_pass(Ssimu2VkState *s, VkCommandBuffer cmd, int scale,
                                    VkDescriptorSet set, int pass, uint32_t in_off,
                                    uint32_t out_off)
{
    BlurPushConsts pc = {
        .width = s->scale_w[scale],
        .height = s->scale_h[scale],
        .n2_0 = s->rg_n2[0],
        .n2_1 = s->rg_n2[1],
        .n2_2 = s->rg_n2[2],
        .d1_0 = s->rg_d1[0],
        .d1_1 = s->rg_d1[1],
        .d1_2 = s->rg_d1[2],
        .radius = s->rg_radius,
        .in_offset = in_off,
        .out_offset = out_off,
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_blur.pipeline_layout, 0, 1,
                            &set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_blur.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                       &pc);
    VkPipeline p = (pass == 0) ? s->blur_pipelines_h[scale] : s->blur_pipelines_v[scale];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    /* H pass: one workgroup per row → dispatch (1, height, 1).
     * V pass: one workgroup per column → dispatch (1, width, 1). */
    uint32_t lines = (pass == 0) ? s->scale_h[scale] : s->scale_w[scale];
    vkCmdDispatch(cmd, 1, lines, 1);
}

/* Per-plane buffer-of-3 sub-buffer offset trick: rather than
 * juggling sub-buffers, we build one descriptor set per plane that
 * binds the same VkBuffer with VK_WHOLE_SIZE, then push the
 * plane offset via push constants would not work for
 * std430 buffers (the shader reads from index 0). Solution: we
 * use 3 separate buffers per scratch — see `blur_one_plane_set`
 * below which materialises offset as an array sub-region by
 * splitting the buffer into 3 sub-allocations. Simpler approach:
 * use the IIR blur shader's `in[base + idx]` directly via a per-call
 * push constant `plane_offset`. For now do the simple path: bind
 * the same buffer twice and let the shader index from its base.
 * This means each plane is its own VkBuffer. ALTERNATIVE: pass
 * `plane_offset` via push constants. The blur shader currently
 * doesn't accept a plane offset — refactor to accept it.            */

/* Blur all 3 planes of `in` (a 3-plane buffer) into the 3 planes
 * of `out`. Strategy: bind each (in→scratch) and (scratch→out)
 * descriptor set ONCE with VK_WHOLE_SIZE; per-channel plane offset
 * is encoded in the BlurPushConsts.in_offset / out_offset push
 * constants. This avoids the "update descriptor set between
 * vkCmdDispatch records" trap — Vulkan reads bindings at execute
 * time, not record time, so per-record updates only show the LAST
 * write at submit. Push constants ARE re-recorded per dispatch.
 *
 * Caller passes 2 descriptor sets:
 *   sets[0]: H pass — bound to (in_buf, blur_scratch)
 *   sets[1]: V pass — bound to (blur_scratch, out_buf)
 */
static void ss2v_blur_3plane(Ssimu2VkState *s, VkCommandBuffer cmd, int scale,
                             VkDescriptorSet set_h, VkDescriptorSet set_v)
{
    const uint32_t full_plane = (uint32_t)((size_t)s->width * (size_t)s->height);
    for (int c = 0; c < 3; c++) {
        const uint32_t plane_off = (uint32_t)c * full_plane;
        ss2v_dispatch_blur_pass(s, cmd, scale, set_h, /*pass=*/0, plane_off, plane_off);
        ss2v_barrier(cmd);
        ss2v_dispatch_blur_pass(s, cmd, scale, set_v, /*pass=*/1, plane_off, plane_off);
        ss2v_barrier(cmd);
    }
}

/* Submit and wait helper — thin wrapper kept for the call site in
 * ss2v_run_scale. Delegates to the pool-owned fence + cmd buffer.
 * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
static int ss2v_submit_wait(Ssimu2VkState *s, VmafVulkanKernelSubmit *sub)
{
    return vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, sub);
}

/* Per-scale orchestration. Records a fresh command buffer that
 * runs xyb → mul (3 products) → blur (5 outputs) → ssim. After
 * the buffer completes, host reads partials and accumulates.
 * Per-scale pipeline orchestration mirrors the CPU extract loop
 * step-by-step; splitting would obscure the dispatch ordering
 * (ADR-0141 §2 upstream-parity load-bearing invariant; T7-5 sweep
 * closeout — ADR-0278). */
/* NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static int ss2v_run_scale(Ssimu2VkState *s, int scale, double avg_ssim[6], double avg_ed[12])
{
    VkDevice dev = s->ctx->device;
    int err = 0;

    /* Allocate per-scale descriptor sets. Each blur needs 2 sets:
     * H pass (in→scratch) and V pass (scratch→out). With 5 blurs
     * = 10 blur sets per scale. */
    VkDescriptorSet xyb_set_ref = VK_NULL_HANDLE, xyb_set_dis = VK_NULL_HANDLE;
    VkDescriptorSet mul_set_ref2 = VK_NULL_HANDLE, mul_set_dis2 = VK_NULL_HANDLE;
    VkDescriptorSet mul_set_rd = VK_NULL_HANDLE;
    VkDescriptorSet blur_sets_h[5] = {VK_NULL_HANDLE};
    VkDescriptorSet blur_sets_v[5] = {VK_NULL_HANDLE};
    VkDescriptorSet ssim_set = VK_NULL_HANDLE;

    err |= ss2v_alloc_set(s, &s->pl_xyb, &xyb_set_ref);
    err |= ss2v_alloc_set(s, &s->pl_xyb, &xyb_set_dis);
    err |= ss2v_alloc_set(s, &s->pl_mul, &mul_set_ref2);
    err |= ss2v_alloc_set(s, &s->pl_mul, &mul_set_dis2);
    err |= ss2v_alloc_set(s, &s->pl_mul, &mul_set_rd);
    for (int b = 0; b < 5; b++) {
        err |= ss2v_alloc_set(s, &s->pl_blur, &blur_sets_h[b]);
        err |= ss2v_alloc_set(s, &s->pl_blur, &blur_sets_v[b]);
    }
    err |= ss2v_alloc_set(s, &s->pl_ssim, &ssim_set);
    if (err)
        goto out;

    /* Bind blur sets (each VK_WHOLE_SIZE; per-channel offset via push consts). */
    {
        VmafVulkanBuffer *blur_h_pairs[5][2] = {
            {s->mul_buf, s->blur_scratch}, /* b0: ref²    H */
            {s->mul_buf, s->blur_scratch}, /* b1: dis²    H */
            {s->mul_buf, s->blur_scratch}, /* b2: ref·dis H */
            {s->ref_xyb, s->blur_scratch}, /* b3: ref     H */
            {s->dis_xyb, s->blur_scratch}, /* b4: dis     H */
        };
        VmafVulkanBuffer *blur_v_pairs[5][2] = {
            {s->blur_scratch, s->s11}, {s->blur_scratch, s->s22}, {s->blur_scratch, s->s12},
            {s->blur_scratch, s->mu1}, {s->blur_scratch, s->mu2},
        };
        for (int b = 0; b < 5; b++) {
            ss2v_write_set(dev, blur_sets_h[b], 2, blur_h_pairs[b]);
            ss2v_write_set(dev, blur_sets_v[b], 2, blur_v_pairs[b]);
        }
    }

    /* Write xyb sets: ref / dis go to ref_xyb / dis_xyb. We
     * decompose the 3-plane in/out buffers into 3 sub-bindings
     * for r, g, b inputs and x, y, b outputs via offsets. */
    const size_t full_plane_bytes = (size_t)s->width * (size_t)s->height * sizeof(float);
    const size_t scale_plane_bytes =
        (size_t)s->scale_w[scale] * (size_t)s->scale_h[scale] * sizeof(float);

    {
        VkDescriptorBufferInfo dbi[6];
        for (int c = 0; c < 3; c++) {
            dbi[c] = (VkDescriptorBufferInfo){
                .buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_lin),
                .offset = (VkDeviceSize)((size_t)c * full_plane_bytes),
                .range = (VkDeviceSize)scale_plane_bytes};
            dbi[3 + c] = (VkDescriptorBufferInfo){
                .buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_xyb),
                .offset = (VkDeviceSize)((size_t)c * full_plane_bytes),
                .range = (VkDeviceSize)scale_plane_bytes};
        }
        VkWriteDescriptorSet wr[6];
        for (int i = 0; i < 6; i++) {
            wr[i] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = xyb_set_ref,
                .dstBinding = (uint32_t)i,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &dbi[i],
            };
        }
        vkUpdateDescriptorSets(dev, 6, wr, 0, NULL);
    }
    {
        VkDescriptorBufferInfo dbi[6];
        for (int c = 0; c < 3; c++) {
            dbi[c] = (VkDescriptorBufferInfo){
                .buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_lin),
                .offset = (VkDeviceSize)((size_t)c * full_plane_bytes),
                .range = (VkDeviceSize)scale_plane_bytes};
            dbi[3 + c] = (VkDescriptorBufferInfo){
                .buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_xyb),
                .offset = (VkDeviceSize)((size_t)c * full_plane_bytes),
                .range = (VkDeviceSize)scale_plane_bytes};
        }
        VkWriteDescriptorSet wr[6];
        for (int i = 0; i < 6; i++) {
            wr[i] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = xyb_set_dis,
                .dstBinding = (uint32_t)i,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &dbi[i],
            };
        }
        vkUpdateDescriptorSets(dev, 6, wr, 0, NULL);
    }

    /* Mul sets: ref_xyb*ref_xyb -> mul_buf, dis_xyb*dis_xyb -> mul_buf, ref_xyb*dis_xyb -> mul_buf. */
    {
        VmafVulkanBuffer *triples[][3] = {
            {s->ref_xyb, s->ref_xyb, s->mul_buf},
            {s->dis_xyb, s->dis_xyb, s->mul_buf},
            {s->ref_xyb, s->dis_xyb, s->mul_buf},
        };
        VkDescriptorSet *targets[] = {&mul_set_ref2, &mul_set_dis2, &mul_set_rd};
        for (int t = 0; t < 3; t++)
            ss2v_write_set(dev, *targets[t], 3, triples[t]);
    }

    /* SSIM set: mu1, mu2, s11, s22, s12, ref_xyb (img1), dis_xyb (img2), partials. */
    {
        VmafVulkanBuffer *bufs[8] = {s->mu1, s->mu2,     s->s11,     s->s22,
                                     s->s12, s->ref_xyb, s->dis_xyb, s->partials};
        ss2v_write_set(dev, ssim_set, 8, bufs);
    }

    /* Acquire pre-allocated command buffer + fence from the submit pool.
     * Slot 0 is reused across scales — ss2v_run_scale is called sequentially
     * and each call drains the fence before returning.
     * Eliminates per-scale vkAllocateCommandBuffers + vkCreateFence.
     * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
    VmafVulkanKernelSubmit sub = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &sub);
    if (err)
        goto out;
    VkCommandBuffer cmd = sub.cmd;

    /* (a) XYB ref + dis: now host-side (see ss2v_host_linear_rgb_to_xyb).
     * Computed in the per-scale loop in `extract` before this command
     * buffer is recorded; the ref_xyb / dis_xyb buffers are HOST_VISIBLE
     * + flushed before submit. The descriptor sets allocated for the
     * (no longer dispatched) GPU XYB step are still allocated in this
     * function for future reuse but never bound to a pipeline. */
    (void)xyb_set_ref;
    (void)xyb_set_dis;

    /* (b1) ref_xyb² → mul_buf, blur into s11. */
    ss2v_dispatch_mul(s, cmd, scale, mul_set_ref2, 3);
    ss2v_barrier(cmd);
    ss2v_blur_3plane(s, cmd, scale, blur_sets_h[0], blur_sets_v[0]);

    /* (b2) dis_xyb² → mul_buf, blur into s22. */
    ss2v_dispatch_mul(s, cmd, scale, mul_set_dis2, 3);
    ss2v_barrier(cmd);
    ss2v_blur_3plane(s, cmd, scale, blur_sets_h[1], blur_sets_v[1]);

    /* (b3) ref_xyb*dis_xyb → mul_buf, blur into s12. */
    ss2v_dispatch_mul(s, cmd, scale, mul_set_rd, 3);
    ss2v_barrier(cmd);
    ss2v_blur_3plane(s, cmd, scale, blur_sets_h[2], blur_sets_v[2]);

    /* (b4) blur ref_xyb → mu1. */
    ss2v_blur_3plane(s, cmd, scale, blur_sets_h[3], blur_sets_v[3]);
    /* (b5) blur dis_xyb → mu2. */
    ss2v_blur_3plane(s, cmd, scale, blur_sets_h[4], blur_sets_v[4]);

    /* SSIM + EdgeDiff stats run host-side in double precision over
     * the GPU-blurred mu1/mu2/s11/s22/s12 buffers (which are
     * HOST_VISIBLE). The previous in-shader float path produced
     * places=1 drift on the pooled `ssimulacra2` because the
     * per-pixel `d = 1 - num_m * num_s / denom_s` numerator and
     * denominator can both be tiny (cancellation in `s12 - mu1*mu2`
     * etc), and the float divide loses the 2-3 ULP that the CPU's
     * `(double)num_m * (double)num_s / (double)denom_s` retains.
     * Per ADR-0201 §Precision investigation: this single change
     * brings pooled drift from 1.59e-2 to <5e-3 (places=2) without
     * any other algorithmic shift. The host-side combine is O(W*H)
     * per scale per channel; total wall-time impact <2% on lavapipe
     * and the mu/sigma reads are sequential through host-mapped
     * USM, so the cache pre-warm cost is shared with the existing
     * partial read-back. */

    err = ss2v_submit_wait(s, &sub);
    vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
    if (err)
        goto out;

    /* (d) Host-side per-pixel SSIM + EdgeDiff combine in double.
     * Mirrors libvmaf/src/feature/ssimulacra2.c::ssim_map +
     * ::edge_diff_map exactly (same expression order, same `(double)`
     * promotion at the divide site). The buffers are HOST_VISIBLE
     * (VMA AUTO_PREFER_HOST + MAPPED) so the read is a normal RAM
     * walk after the queue fence. */
    /* Invalidate CPU cache lines for all GPU-written readback buffers
     * before reading via host pointer (Vulkan 1.3 spec §11.2.2). */
    int err_inv;
    err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->mu1);
    if (err_inv) {
        err = err_inv;
        goto out;
    }
    err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->mu2);
    if (err_inv) {
        err = err_inv;
        goto out;
    }
    err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->s11);
    if (err_inv) {
        err = err_inv;
        goto out;
    }
    err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->s22);
    if (err_inv) {
        err = err_inv;
        goto out;
    }
    err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->s12);
    if (err_inv) {
        err = err_inv;
        goto out;
    }
    const float *mu1_host = vmaf_vulkan_buffer_host(s->mu1);
    const float *mu2_host = vmaf_vulkan_buffer_host(s->mu2);
    const float *s11_host = vmaf_vulkan_buffer_host(s->s11);
    const float *s22_host = vmaf_vulkan_buffer_host(s->s22);
    const float *s12_host = vmaf_vulkan_buffer_host(s->s12);
    const float *img1_host = vmaf_vulkan_buffer_host(s->ref_xyb);
    const float *img2_host = vmaf_vulkan_buffer_host(s->dis_xyb);
    if (!mu1_host || !mu2_host || !s11_host || !s22_host || !s12_host || !img1_host || !img2_host) {
        err = -EIO;
        goto out;
    }
    const unsigned cw = s->scale_w[scale];
    const unsigned ch = s->scale_h[scale];
    const size_t full_plane = (size_t)s->width * (size_t)s->height;
    const size_t scale_pixels = (size_t)cw * (size_t)ch;
    const double inv_pixels = 1.0 / (double)scale_pixels;
    for (int c = 0; c < 3; c++) {
        const float *m1 = mu1_host + (size_t)c * full_plane;
        const float *m2 = mu2_host + (size_t)c * full_plane;
        const float *s11p = s11_host + (size_t)c * full_plane;
        const float *s22p = s22_host + (size_t)c * full_plane;
        const float *s12p = s12_host + (size_t)c * full_plane;
        const float *r1 = img1_host + (size_t)c * full_plane;
        const float *r2 = img2_host + (size_t)c * full_plane;
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
out:
    /* Free descriptor sets — each set goes back to its bundle's pool. */
    if (xyb_set_ref)
        vkFreeDescriptorSets(dev, s->pl_xyb.desc_pool, 1, &xyb_set_ref);
    if (xyb_set_dis)
        vkFreeDescriptorSets(dev, s->pl_xyb.desc_pool, 1, &xyb_set_dis);
    if (mul_set_ref2)
        vkFreeDescriptorSets(dev, s->pl_mul.desc_pool, 1, &mul_set_ref2);
    if (mul_set_dis2)
        vkFreeDescriptorSets(dev, s->pl_mul.desc_pool, 1, &mul_set_dis2);
    if (mul_set_rd)
        vkFreeDescriptorSets(dev, s->pl_mul.desc_pool, 1, &mul_set_rd);
    for (int b = 0; b < 5; b++) {
        if (blur_sets_h[b])
            vkFreeDescriptorSets(dev, s->pl_blur.desc_pool, 1, &blur_sets_h[b]);
        if (blur_sets_v[b])
            vkFreeDescriptorSets(dev, s->pl_blur.desc_pool, 1, &blur_sets_v[b]);
    }
    if (ssim_set)
        vkFreeDescriptorSets(dev, s->pl_ssim.desc_pool, 1, &ssim_set);
    return err;
}

/* Mirror of ssimulacra2.c::pool_score. */
static double ss2v_pool_score(const double avg_ssim[6][6], const double avg_ed[6][12],
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

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    Ssimu2VkState *s = fex->priv;

    /* Step 1: host YUV → linear RGB on the GPU-mapped buffers. */
    float *ref_lin_host = vmaf_vulkan_buffer_host(s->ref_lin);
    float *dis_lin_host = vmaf_vulkan_buffer_host(s->dis_lin);
    if (!ref_lin_host || !dis_lin_host)
        return -EIO;

    /* ADR-0242: SIMD dispatch — convert YUV→linear RGB via SIMD if available,
     * otherwise fall back to scalar ss2v_picture_to_linear_rgb. */
    if (s->ptlr_fn) {
        const simd_plane_t planes_ref[3] = {
            {ref_pic->data[0], (ptrdiff_t)ref_pic->stride[0], ref_pic->w[0], ref_pic->h[0]},
            {ref_pic->data[1], (ptrdiff_t)ref_pic->stride[1], ref_pic->w[1], ref_pic->h[1]},
            {ref_pic->data[2], (ptrdiff_t)ref_pic->stride[2], ref_pic->w[2], ref_pic->h[2]},
        };
        const simd_plane_t planes_dis[3] = {
            {dist_pic->data[0], (ptrdiff_t)dist_pic->stride[0], dist_pic->w[0], dist_pic->h[0]},
            {dist_pic->data[1], (ptrdiff_t)dist_pic->stride[1], dist_pic->w[1], dist_pic->h[1]},
            {dist_pic->data[2], (ptrdiff_t)dist_pic->stride[2], dist_pic->w[2], dist_pic->h[2]},
        };
        s->ptlr_fn((int)s->yuv_matrix, s->bpc, s->width, s->height, planes_ref, ref_lin_host);
        s->ptlr_fn((int)s->yuv_matrix, s->bpc, s->width, s->height, planes_dis, dis_lin_host);
    } else {
        ss2v_picture_to_linear_rgb(s, ref_pic, ref_lin_host);
        ss2v_picture_to_linear_rgb(s, dist_pic, dis_lin_host);
    }
    int err = vmaf_vulkan_buffer_flush(s->ctx, s->ref_lin);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_flush(s->ctx, s->dis_lin);
    if (err)
        return err;

    /* Step 2: per-scale loop. Linear-RGB pyramid is built in-place
     * by host between scales (cheap vs the GPU work).
     *
     * The CPU breaks early if cw < 8 or ch < 8 — match that here. */
    double avg_ssim[6][6] = {{0}};
    double avg_ed[6][12] = {{0}};
    int completed = 0;
    unsigned cw = s->width;
    unsigned ch = s->height;

    /* Host-side XYB output buffers — same VkBuffers the GPU reads
     * for the mul/blur stages. The buffers are HOST_VISIBLE +
     * MAPPED via VMA so we can write XYB directly and then flush
     * once before each scale's GPU dispatch. */
    float *ref_xyb_host = vmaf_vulkan_buffer_host(s->ref_xyb);
    float *dis_xyb_host = vmaf_vulkan_buffer_host(s->dis_xyb);
    if (!ref_xyb_host || !dis_xyb_host)
        return -EIO;

    const size_t plane_full = (size_t)s->width * (size_t)s->height;

    for (int scale = 0; scale < SS2V_NUM_SCALES; scale++) {
        if (cw < 8u || ch < 8u)
            break;
        /* Host XYB pre-pass. Bit-exact with CPU; replaces the
         * `ssimulacra2_xyb.comp` dispatch (see ADR-0201 §Precision
         * investigation: the GPU XYB shader produced ~1.7e-6 max
         * per-pixel drift across lavapipe / Mesa anv / RADV even
         * with `precise` + `NoContraction`, which compounded
         * through the IIR + 108-weight pool to a 1.5e-2 pooled
         * drift). The IIR + SSIM combine is bit-exact with CPU
         * when fed bit-exact XYB, so this single change moves the
         * pipeline from places=1 to places ≥ 2 by construction. */
        /* ADR-0242: SIMD host XYB dispatch. */
        if (s->xyb_host_fn) {
            s->xyb_host_fn(ref_lin_host, ref_xyb_host, cw, ch, plane_full);
            s->xyb_host_fn(dis_lin_host, dis_xyb_host, cw, ch, plane_full);
        } else {
            ss2v_host_linear_rgb_to_xyb(ref_lin_host, ref_xyb_host, cw, ch, plane_full);
            ss2v_host_linear_rgb_to_xyb(dis_lin_host, dis_xyb_host, cw, ch, plane_full);
        }
        err = vmaf_vulkan_buffer_flush(s->ctx, s->ref_xyb);
        if (err)
            return err;
        err = vmaf_vulkan_buffer_flush(s->ctx, s->dis_xyb);
        if (err)
            return err;
        /* GPU scale work. */
        err = ss2v_run_scale(s, scale, avg_ssim[scale], avg_ed[scale]);
        if (err)
            return err;
        completed++;

        if (scale + 1 < SS2V_NUM_SCALES) {
            /* Host downsample. Scratch and the GPU-mapped pyramid
             * buffer both use the full-resolution plane stride —
             * each plane occupies a full_plane-sized slot, with the
             * actual scale data living at the head of that slot.
             * This stays consistent with the GPU shaders, which
             * compute per-channel offsets as `c * full_plane`. */
            unsigned nw = (cw + 1) / 2;
            unsigned nh = (ch + 1) / 2;
            const size_t full_plane = (size_t)s->width * (size_t)s->height;
            float *scratch = malloc(3u * full_plane * sizeof(float));
            if (!scratch)
                return -ENOMEM;
            /* ADR-0242: SIMD host downsample dispatch. */
            if (s->down_host_fn) {
                s->down_host_fn(ref_lin_host, cw, ch, scratch, nw, nh, full_plane);
            } else {
                ss2v_downsample_2x2(ref_lin_host, cw, ch, scratch, nw, nh, full_plane);
            }
            for (int c = 0; c < 3; c++) {
                memcpy(ref_lin_host + (size_t)c * full_plane, scratch + (size_t)c * full_plane,
                       (size_t)nw * (size_t)nh * sizeof(float));
            }
            if (s->down_host_fn) {
                s->down_host_fn(dis_lin_host, cw, ch, scratch, nw, nh, full_plane);
            } else {
                ss2v_downsample_2x2(dis_lin_host, cw, ch, scratch, nw, nh, full_plane);
            }
            for (int c = 0; c < 3; c++) {
                memcpy(dis_lin_host + (size_t)c * full_plane, scratch + (size_t)c * full_plane,
                       (size_t)nw * (size_t)nh * sizeof(float));
            }
            free(scratch);
            err = vmaf_vulkan_buffer_flush(s->ctx, s->ref_lin);
            if (err)
                return err;
            err = vmaf_vulkan_buffer_flush(s->ctx, s->dis_lin);
            if (err)
                return err;
            cw = nw;
            ch = nh;
        }
    }

    const double score = ss2v_pool_score(avg_ssim, avg_ed, completed);
    return vmaf_feature_collector_append(feature_collector, "ssimulacra2", score, index);
}

static int close_fex(VmafFeatureExtractor *fex)
{
    Ssimu2VkState *s = fex->priv;
    if (!s || !s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;
    vkDeviceWaitIdle(dev);

    /* Destroy submit pool BEFORE pipelines (ADR-0256 / ADR-0354 ordering rule). */
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);

    /* Variant pipelines must be destroyed *before* the bundle's
     * `_destroy()` (which destroys the shared layout/shader/DSL/pool
     * and would invalidate the variants). The base pipeline at
     * `*_pipelines[0]` aliases the bundle's `pipeline` field — skip
     * slot 0 to avoid double-freeing it via the bundle teardown. */
    for (int i = 1; i < SS2V_NUM_SCALES; i++) {
        if (s->xyb_pipelines[i])
            vkDestroyPipeline(dev, s->xyb_pipelines[i], NULL);
        if (s->mul_pipelines[i])
            vkDestroyPipeline(dev, s->mul_pipelines[i], NULL);
        if (s->blur_pipelines_h[i])
            vkDestroyPipeline(dev, s->blur_pipelines_h[i], NULL);
        if (s->ssim_pipelines[i])
            vkDestroyPipeline(dev, s->ssim_pipelines[i], NULL);
    }
    /* The blur V-pass at scale 0 is also a variant — only the H-pass
     * at slot 0 aliases the bundle's pipeline. Destroy every V slot. */
    for (int i = 0; i < SS2V_NUM_SCALES; i++) {
        if (s->blur_pipelines_v[i])
            vkDestroyPipeline(dev, s->blur_pipelines_v[i], NULL);
    }

    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_xyb);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_mul);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_blur);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_ssim);

#define SS2V_FREE(b)                                                                               \
    do {                                                                                           \
        if (s->b)                                                                                  \
            vmaf_vulkan_buffer_free(s->ctx, s->b);                                                 \
    } while (0)
    SS2V_FREE(ref_lin);
    SS2V_FREE(dis_lin);
    SS2V_FREE(ref_xyb);
    SS2V_FREE(dis_xyb);
    SS2V_FREE(mul_buf);
    SS2V_FREE(mu1);
    SS2V_FREE(mu2);
    SS2V_FREE(s11);
    SS2V_FREE(s22);
    SS2V_FREE(s12);
    SS2V_FREE(blur_scratch);
    SS2V_FREE(partials);
#undef SS2V_FREE

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;
    return 0;
}

static const char *provided_features[] = {"ssimulacra2", NULL};

/* External linkage required — the registry iterates over `vmaf_fex_*` externs.
 * NOLINTNEXTLINE(misc-use-internal-linkage,cppcoreguidelines-avoid-non-const-global-variables) */
VmafFeatureExtractor vmaf_fex_ssimulacra2_vulkan = {
    .name = "ssimulacra2_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(Ssimu2VkState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    .chars =
        {
            /* Per scale: 2 xyb + 3 mul + 5 * (3 channels * 2 passes) =
             * 35 dispatches; 6 scales = 210. Heavy frame; min useful
             * frame area set high to discourage tiny inputs. */
            .n_dispatches_per_frame = 210,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
