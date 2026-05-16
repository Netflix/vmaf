/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CAMBI banding-detection feature extractor on the SYCL backend
 *  (T3-15 / ADR-0371). SYCL twin of integer_cambi_cuda.c (ADR-0360)
 *  and cambi_vulkan.c (ADR-0210).
 *
 *  Strategy II hybrid — identical to the CUDA twin (ADR-0360):
 *
 *    GPU stages (three SYCL kernels):
 *      - launch_spatial_mask  : derivative + 7×7 box sum + threshold.
 *        Produces a uint16 mask buffer (0 = flat, 1 = edge).
 *        Bit-exact port of cambi_spatial_mask_kernel.
 *      - launch_decimate      : strict 2× stride-2 subsample.
 *        Bit-exact port of cambi_decimate_kernel.
 *      - launch_filter_mode   : separable 3-tap mode filter (H + V).
 *        Bit-exact port of cambi_filter_mode_kernel.
 *
 *    Host CPU stages (exact CPU code via cambi_internal.h wrappers):
 *      - vmaf_cambi_preprocessing: decimate/upcast to 10-bit.
 *      - vmaf_cambi_calculate_c_values: sliding-histogram c-value pass.
 *      - vmaf_cambi_spatial_pooling: top-K pooling → per-scale score.
 *      - vmaf_cambi_weight_scores_per_scale: inner-product scale weights.
 *
 *  Per-frame flow (synchronous per-scale loop, same as CUDA v1):
 *    1. Host preprocessing (CPU): resize/upcast dist_pic → pics[0].
 *    2. H2D upload of pics[0] luma plane → d_image (USM device).
 *    3. GPU launch_spatial_mask over d_image → d_mask.
 *    4. For scale = 0 .. NUM_SCALES-1:
 *         a. (scale > 0) GPU launch_decimate d_image → d_tmp, swap;
 *            GPU launch_decimate d_mask  → d_tmp, swap.
 *         b. GPU launch_filter_mode H: d_image → d_tmp.
 *         c. GPU launch_filter_mode V: d_tmp   → d_image.
 *         d. q.wait() to drain; D2H memcpy → pics[0], pics[1].
 *         e. Host vmaf_cambi_calculate_c_values + vmaf_cambi_spatial_pooling.
 *    5. Host vmaf_cambi_weight_scores_per_scale → final score.
 *    6. Store score; collect() emits "Cambi_feature_cambi_score".
 *
 *  Precision contract: `places=4` (ULP=0 on emitted score). All GPU
 *  stages use integer arithmetic only. The host residual runs the exact
 *  CPU code path from cambi_internal.h, so the emitted score is
 *  bit-for-bit identical to `vmaf_fex_cambi` and the CUDA twin.
 *
 *  SYCL specifics:
 *    - Buffers are USM device pointers (uint16_t *) allocated via
 *      vmaf_sycl_malloc_device / vmaf_sycl_malloc_host.
 *    - Kernels submitted with q.submit([=](sycl::handler &) { ... }).
 *    - q.wait() used between GPU and CPU stages (synchronous v1 posture
 *      matching the CUDA twin; see ADR-0360 §v1 simplification note).
 *    - Does NOT use vmaf_sycl_graph_register because CAMBI's host
 *      residual is non-trivial and the per-scale CPU work serialises
 *      frames already. Same reasoning as the CUDA twin.
 *    - Supports both Intel oneAPI (icpx -fsycl) and AdaptiveCpp
 *      (acpp --acpp-targets=...) per ADR-0335. The strict-FP contract
 *      is honoured by the meson.build sycl_strict_fp_args mechanism
 *      (-fp-model=precise for icpx, -ffp-contract=off for acpp).
 *      Since all arithmetic in the SYCL kernels is integer-only,
 *      strict-FP has no effect here; the flag is inherited from the
 *      common feature build recipe.
 *
 *  Kernel mapping from CUDA → SYCL:
 *    cambi_spatial_mask_kernel → launch_spatial_mask (anonymous ns)
 *    cambi_decimate_kernel     → launch_decimate     (anonymous ns)
 *    cambi_filter_mode_kernel  → launch_filter_mode  (anonymous ns)
 */

#include <sycl/sycl.hpp>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "luminance_tools.h"
#include "picture.h"
#include "sycl/common.h"
#include "feature/cambi_internal.h"
}

/* ------------------------------------------------------------------ */
/* Constants (mirroring integer_cambi_cuda.c). */
/* ------------------------------------------------------------------ */
namespace
{

static constexpr int CAMBI_SYCL_NUM_SCALES = 5;
static constexpr int CAMBI_SYCL_MIN_WIDTH_HEIGHT = 216;
static constexpr unsigned CAMBI_SYCL_MASK_FILTER_SIZE = 7U;
static constexpr double CAMBI_SYCL_DEFAULT_MAX_VAL = 1000.0;
static constexpr int CAMBI_SYCL_DEFAULT_WINDOW_SIZE = 65;
static constexpr double CAMBI_SYCL_DEFAULT_TOPK = 0.6;
static constexpr double CAMBI_SYCL_DEFAULT_TVI = 0.019;
static constexpr double CAMBI_SYCL_DEFAULT_VLT = 0.0;
static constexpr int CAMBI_SYCL_DEFAULT_MAX_LOG_CONTRAST = 2;
/* `default_val.s` in `VmafOption` is declared `char *` (not `const char *`);
 * use a `char[]` so the array decays to `char *` without a const cast.
 * Mirrors the CUDA twin `CAMBI_CUDA_DEFAULT_EOTF` which uses a `#define`
 * macro for the same reason. */
static char CAMBI_SYCL_DEFAULT_EOTF[] = "bt1886";

/* Work-group tile size. */
static constexpr size_t WG_X = 16;
static constexpr size_t WG_Y = 16;

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */
struct CambiStateSycl {
    VmafSyclState *sycl_state;

    /* USM device buffers (flat uint16 arrays). */
    uint16_t *d_image;
    uint16_t *d_mask;
    uint16_t *d_tmp;

    /* USM host staging buffers for D2H. */
    uint16_t *h_image;
    uint16_t *h_mask;

    /* Host VmafPicture pair for the CPU residual. */
    VmafPicture pics[2]; /* [0] = image, [1] = mask */

    /* Host scratch buffers for the CPU residual. */
    VmafCambiHostBuffers buffers;

    /* Callbacks (scalar; mirrors CUDA twin). */
    VmafCambiRangeUpdater inc_range_callback;
    VmafCambiRangeUpdater dec_range_callback;
    VmafCambiDerivativeCalculator derivative_callback;

    /* Configuration options. */
    int enc_width;
    int enc_height;
    int enc_bitdepth;
    int max_log_contrast;
    int window_size;
    double topk;
    double cambi_topk;
    double tvi_threshold;
    double cambi_max_val;
    double cambi_vis_lum_threshold;
    char *eotf;
    char *cambi_eotf;

    /* Resolved per-frame geometry. */
    unsigned src_width;
    unsigned src_height;
    unsigned src_bpc;
    unsigned proc_width;
    unsigned proc_height;

    uint16_t adjusted_window;
    uint16_t vlt_luma;

    /* Pre-computed per-scale score storage. */
    double score; /* final weighted score stored by submit, emitted by collect */

    bool has_pending;
    unsigned pending_index;

    VmafDictionary *feature_name_dict;
};

/* ------------------------------------------------------------------ */
/* Helpers (mirrors integer_cambi_cuda.c's static helpers). */
/* ------------------------------------------------------------------ */
static uint16_t cambi_sycl_adjust_window(int window_size, unsigned w, unsigned h)
{
    unsigned adjusted = (unsigned)(window_size) * (w + h) / 375u;
    adjusted >>= 4;
    if (adjusted < 1u)
        adjusted = 1u;
    if ((adjusted & 1u) == 0u)
        adjusted++;
    return (uint16_t)adjusted;
}

static uint16_t cambi_sycl_ceil_log2(uint32_t num)
{
    if (num == 0u)
        return 0u;
    uint32_t tmp = num - 1u;
    uint16_t shift = 0;
    while (tmp > 0u) {
        tmp >>= 1;
        shift++;
    }
    return shift;
}

static uint16_t cambi_sycl_get_mask_index(unsigned w, unsigned h, unsigned filter_size)
{
    uint32_t shifted_wh = (w >> 6) * (h >> 6);
    return (uint16_t)((filter_size * filter_size + 3u * (cambi_sycl_ceil_log2(shifted_wh) - 11u) -
                       1u) >>
                      1u);
}

static int cambi_sycl_init_tvi(CambiStateSycl *s)
{
    VmafLumaRange luma_range;
    int err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err)
        return err;

    const char *effective_eotf;
    if (s->cambi_eotf && strcmp(s->cambi_eotf, CAMBI_SYCL_DEFAULT_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = (s->eotf != NULL) ? s->eotf : CAMBI_SYCL_DEFAULT_EOTF;
    }

    VmafEOTF eotf;
    err = vmaf_luminance_init_eotf(&eotf, effective_eotf);
    if (err)
        return err;

    const int num_diffs = 1 << s->max_log_contrast;
    for (int d = 0; d < num_diffs; d++) {
        const int diff = (int)s->buffers.diffs_to_consider[d];
        int lo = 0;
        int hi = (1 << 10) - 1 - diff;
        int found = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            double sample_lum = vmaf_luminance_get_luminance(mid, luma_range, eotf);
            double diff_lum =
                vmaf_luminance_get_luminance(mid + diff, luma_range, eotf) - sample_lum;
            if (diff_lum < s->tvi_threshold * sample_lum) {
                found = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (found < 0)
            found = 0;
        s->buffers.tvi_for_diff[d] = (uint16_t)(found + num_diffs);
    }

    int vlt = 0;
    for (int v = 0; v < (1 << 10); v++) {
        double L = vmaf_luminance_get_luminance(v, luma_range, eotf);
        if (L < s->cambi_vis_lum_threshold)
            vlt = v;
    }
    s->vlt_luma = (uint16_t)vlt;
    return 0;
}

/* ------------------------------------------------------------------ */
/* SYCL kernel 1: Spatial mask                                         */
/* Port of cambi_spatial_mask_kernel from cambi_score.cu.              */
/* ------------------------------------------------------------------ */
static void launch_spatial_mask(sycl::queue &q, const uint16_t *image, uint16_t *mask,
                                unsigned width, unsigned height, unsigned stride_words,
                                unsigned mask_index)
{
    const size_t global_x = ((size_t)width + WG_X - 1u) / WG_X * WG_X;
    const size_t global_y = ((size_t)height + WG_Y - 1u) / WG_Y * WG_Y;
    sycl::nd_range<2> ndr{sycl::range<2>{global_y, global_x}, sycl::range<2>{WG_Y, WG_X}};

    const unsigned e_w = width;
    const unsigned e_h = height;
    const unsigned e_stride = stride_words;
    const unsigned e_mask_index = mask_index;
    const uint16_t *e_image = image;
    uint16_t *e_mask = mask;

    q.submit([=](sycl::handler &h) {
        h.parallel_for(ndr, [=](sycl::nd_item<2> it) {
            const int x = (int)it.get_global_id(1);
            const int y = (int)it.get_global_id(0);
            if (x >= (int)e_w || y >= (int)e_h)
                return;

            /* 7×7 box sum of zero_deriv field — mirrors the CUDA kernel
             * strategy: each thread reads its own 7×7 window (49 global
             * reads) independently. Bit-exact with cambi.c's SAT path. */
            static constexpr int HALF = 3;
            unsigned box_sum = 0u;
            for (int dy = -HALF; dy <= HALF; dy++) {
                int ry = y + dy;
                if (ry < 0)
                    ry = 0;
                if (ry >= (int)e_h)
                    ry = (int)e_h - 1;
                for (int dx = -HALF; dx <= HALF; dx++) {
                    int rx = x + dx;
                    if (rx < 0)
                        rx = 0;
                    if (rx >= (int)e_w)
                        rx = (int)e_w - 1;
                    const uint16_t p = e_image[(size_t)ry * e_stride + (unsigned)rx];
                    const int rx_right = (rx == (int)e_w - 1) ? rx : rx + 1;
                    const int ry_below = (ry == (int)e_h - 1) ? ry : ry + 1;
                    const uint16_t r = e_image[(size_t)ry * e_stride + (unsigned)rx_right];
                    const uint16_t b =
                        e_image[(size_t)(unsigned)ry_below * e_stride + (unsigned)rx];
                    const int eq_right = (rx == (int)e_w - 1) || (p == r);
                    const int eq_below = (ry == (int)e_h - 1) || (p == b);
                    box_sum += (unsigned)(eq_right && eq_below);
                }
            }
            e_mask[(size_t)(unsigned)y * e_stride + (unsigned)x] =
                (uint16_t)(box_sum > e_mask_index ? 1u : 0u);
        });
    });
}

/* ------------------------------------------------------------------ */
/* SYCL kernel 2: 2× decimate                                          */
/* Port of cambi_decimate_kernel from cambi_score.cu.                  */
/* ------------------------------------------------------------------ */
static void launch_decimate(sycl::queue &q, const uint16_t *src, uint16_t *dst, unsigned out_w,
                            unsigned out_h, unsigned src_stride_words, unsigned dst_stride_words)
{
    const size_t global_x = ((size_t)out_w + WG_X - 1u) / WG_X * WG_X;
    const size_t global_y = ((size_t)out_h + WG_Y - 1u) / WG_Y * WG_Y;
    sycl::nd_range<2> ndr{sycl::range<2>{global_y, global_x}, sycl::range<2>{WG_Y, WG_X}};

    const unsigned e_out_w = out_w;
    const unsigned e_out_h = out_h;
    const unsigned e_src_stride = src_stride_words;
    const unsigned e_dst_stride = dst_stride_words;
    const uint16_t *e_src = src;
    uint16_t *e_dst = dst;

    q.submit([=](sycl::handler &h) {
        h.parallel_for(ndr, [=](sycl::nd_item<2> it) {
            const unsigned x = (unsigned)it.get_global_id(1);
            const unsigned y = (unsigned)it.get_global_id(0);
            if (x >= e_out_w || y >= e_out_h)
                return;
            /* Strict stride-2 subsample — bit-exact with cambi.c::decimate. */
            e_dst[(size_t)y * e_dst_stride + x] = e_src[(size_t)(y * 2u) * e_src_stride + x * 2u];
        });
    });
}

/* ------------------------------------------------------------------ */
/* SYCL kernel 3: Separable 3-tap mode filter                          */
/* Port of cambi_filter_mode_kernel from cambi_score.cu.               */
/* axis=0 → horizontal, axis=1 → vertical.                             */
/* ------------------------------------------------------------------ */
static void launch_filter_mode(sycl::queue &q, const uint16_t *in, uint16_t *out, unsigned width,
                               unsigned height, unsigned stride_words, int axis)
{
    const size_t global_x = ((size_t)width + WG_X - 1u) / WG_X * WG_X;
    const size_t global_y = ((size_t)height + WG_Y - 1u) / WG_Y * WG_Y;
    sycl::nd_range<2> ndr{sycl::range<2>{global_y, global_x}, sycl::range<2>{WG_Y, WG_X}};

    const unsigned e_w = width;
    const unsigned e_h = height;
    const unsigned e_stride = stride_words;
    const int e_axis = axis;
    const uint16_t *e_in = in;
    uint16_t *e_out = out;

    q.submit([=](sycl::handler &h) {
        h.parallel_for(ndr, [=](sycl::nd_item<2> it) {
            const int x = (int)it.get_global_id(1);
            const int y = (int)it.get_global_id(0);
            if (x >= (int)e_w || y >= (int)e_h)
                return;

            uint16_t a, b, c;
            if (e_axis == 0) {
                /* Horizontal: neighbours in x. */
                const int xl = (x > 0) ? x - 1 : 0;
                const int xr = (x < (int)e_w - 1) ? x + 1 : (int)e_w - 1;
                a = e_in[(size_t)(unsigned)y * e_stride + (unsigned)xl];
                b = e_in[(size_t)(unsigned)y * e_stride + (unsigned)x];
                c = e_in[(size_t)(unsigned)y * e_stride + (unsigned)xr];
            } else {
                /* Vertical: neighbours in y. */
                const int yu = (y > 0) ? y - 1 : 0;
                const int yd = (y < (int)e_h - 1) ? y + 1 : (int)e_h - 1;
                a = e_in[(size_t)(unsigned)yu * e_stride + (unsigned)x];
                b = e_in[(size_t)(unsigned)y * e_stride + (unsigned)x];
                c = e_in[(size_t)(unsigned)yd * e_stride + (unsigned)x];
            }

            /* mode3: two equal → that value; all distinct → min.
             * Bit-exact with mode3_dev from cambi_score.cu. */
            uint16_t result;
            if (a == b || a == c) {
                result = a;
            } else if (b == c) {
                result = b;
            } else {
                result = (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
            }
            e_out[(size_t)(unsigned)y * e_stride + (unsigned)x] = result;
        });
    });
}

} /* anonymous namespace */

/* ------------------------------------------------------------------ */
/* Options (mirrors integer_cambi_cuda.c). */
/* ------------------------------------------------------------------ */
extern "C" {

static const VmafOption options_cambi_sycl[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped",
        .offset = offsetof(CambiStateSycl, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_SYCL_DEFAULT_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiStateSycl, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 180,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encw",
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiStateSycl, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 150,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ench",
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(CambiStateSycl, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encbd",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 65 corresponds to ~1 degree at 4k",
        .offset = offsetof(CambiStateSycl, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_SYCL_DEFAULT_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiStateSycl, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_SYCL_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiStateSycl, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_SYCL_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibility threshold: delta-L < tvi_threshold * L_mean",
        .offset = offsetof(CambiStateSycl, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_SYCL_DEFAULT_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which banding is assumed invisible",
        .offset = offsetof(CambiStateSycl, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_SYCL_DEFAULT_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum log contrast (0 to 5, default 2)",
        .offset = offsetof(CambiStateSycl, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_SYCL_DEFAULT_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "eotf",
        .help = "EOTF for visibility-threshold conversion (bt1886 / pq)",
        .offset = offsetof(CambiStateSycl, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_SYCL_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "EOTF override for cambi (defaults to eotf)",
        .offset = offsetof(CambiStateSycl, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_SYCL_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* init_fex_sycl                                                        */
/* ------------------------------------------------------------------ */
static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    auto *s = static_cast<CambiStateSycl *>(fex->priv);

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi_sycl: no SYCL state\n");
        return -EINVAL;
    }
    s->sycl_state = fex->sycl_state;

    /* Resolve enc geometry (mirrors cambi.c / integer_cambi_cuda.c). */
    if (s->enc_bitdepth == 0)
        s->enc_bitdepth = (int)bpc;
    if (s->enc_width == 0 || s->enc_height == 0) {
        s->enc_width = (int)w;
        s->enc_height = (int)h;
    }
    if ((unsigned)s->enc_height > h || (unsigned)s->enc_width > w) {
        s->enc_width = (int)w;
        s->enc_height = (int)h;
    }
    if (s->enc_width < CAMBI_SYCL_MIN_WIDTH_HEIGHT && s->enc_height < CAMBI_SYCL_MIN_WIDTH_HEIGHT) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi_sycl: encoded resolution %dx%d below minimum %d\n",
                 s->enc_width, s->enc_height, CAMBI_SYCL_MIN_WIDTH_HEIGHT);
        return -EINVAL;
    }

    s->src_width = w;
    s->src_height = h;
    s->src_bpc = bpc;
    s->proc_width = (unsigned)s->enc_width;
    s->proc_height = (unsigned)s->enc_height;
    s->adjusted_window = cambi_sycl_adjust_window(s->window_size, s->proc_width, s->proc_height);

    const size_t buf_elements = (size_t)s->proc_width * s->proc_height;
    const size_t buf_bytes = buf_elements * sizeof(uint16_t);

    /* USM device buffers. */
    s->d_image = static_cast<uint16_t *>(vmaf_sycl_malloc_device(s->sycl_state, buf_bytes));
    s->d_mask = static_cast<uint16_t *>(vmaf_sycl_malloc_device(s->sycl_state, buf_bytes));
    s->d_tmp = static_cast<uint16_t *>(vmaf_sycl_malloc_device(s->sycl_state, buf_bytes));
    /* USM host staging for D2H. */
    s->h_image = static_cast<uint16_t *>(vmaf_sycl_malloc_host(s->sycl_state, buf_bytes));
    s->h_mask = static_cast<uint16_t *>(vmaf_sycl_malloc_host(s->sycl_state, buf_bytes));

    if (!s->d_image || !s->d_mask || !s->d_tmp || !s->h_image || !s->h_mask) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi_sycl: USM allocation failed\n");
        return -ENOMEM;
    }

    /* Host VmafPictures for the CPU residual. */
    int err =
        vmaf_picture_alloc(&s->pics[0], VMAF_PIX_FMT_YUV400P, 10, s->proc_width, s->proc_height);
    if (err)
        goto free_ref;
    err = vmaf_picture_alloc(&s->pics[1], VMAF_PIX_FMT_YUV400P, 10, s->proc_width, s->proc_height);
    if (err)
        goto free_ref;

    /* Host scratch buffers (mirrors integer_cambi_cuda.c::init). */
    {
        const int num_diffs = 1 << s->max_log_contrast;
        s->buffers.diffs_to_consider =
            static_cast<uint16_t *>(malloc(sizeof(uint16_t) * (size_t)num_diffs));
        s->buffers.diff_weights = static_cast<int *>(malloc(sizeof(int) * (size_t)num_diffs));
        s->buffers.all_diffs =
            static_cast<int *>(malloc(sizeof(int) * (size_t)(2 * num_diffs + 1)));
        if (!s->buffers.diffs_to_consider || !s->buffers.diff_weights || !s->buffers.all_diffs) {
            err = -ENOMEM;
            goto free_ref;
        }

        static const int contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                                 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
        for (int d = 0; d < num_diffs; d++) {
            s->buffers.diffs_to_consider[d] = (uint16_t)(d + 1);
            s->buffers.diff_weights[d] = contrast_weights[d];
        }
        for (int d = -num_diffs; d <= num_diffs; d++)
            s->buffers.all_diffs[d + num_diffs] = d;

        s->buffers.tvi_for_diff =
            static_cast<uint16_t *>(malloc(sizeof(uint16_t) * (size_t)num_diffs));
        if (!s->buffers.tvi_for_diff) {
            err = -ENOMEM;
            goto free_ref;
        }

        err = cambi_sycl_init_tvi(s);
        if (err)
            goto free_ref;

        s->buffers.c_values =
            static_cast<float *>(malloc(sizeof(float) * s->proc_width * s->proc_height));
        if (!s->buffers.c_values) {
            err = -ENOMEM;
            goto free_ref;
        }

        const uint16_t num_bins =
            (uint16_t)(1024u +
                       (unsigned)(s->buffers.all_diffs[2 * num_diffs] - s->buffers.all_diffs[0]));
        s->buffers.c_values_histograms =
            static_cast<uint16_t *>(malloc(sizeof(uint16_t) * s->proc_width * (size_t)num_bins));
        if (!s->buffers.c_values_histograms) {
            err = -ENOMEM;
            goto free_ref;
        }

        const int pad_size = (int)(CAMBI_SYCL_MASK_FILTER_SIZE / 2u);
        const int dp_width = (int)s->proc_width + 2 * pad_size + 1;
        const int dp_height = 2 * pad_size + 2;
        s->buffers.mask_dp = static_cast<uint32_t *>(
            malloc(sizeof(uint32_t) * (size_t)dp_width * (size_t)dp_height));
        if (!s->buffers.mask_dp) {
            err = -ENOMEM;
            goto free_ref;
        }

        s->buffers.filter_mode_buffer =
            static_cast<uint16_t *>(malloc(sizeof(uint16_t) * 3u * s->proc_width));
        s->buffers.derivative_buffer =
            static_cast<uint16_t *>(malloc(sizeof(uint16_t) * s->proc_width));
        if (!s->buffers.filter_mode_buffer || !s->buffers.derivative_buffer) {
            err = -ENOMEM;
            goto free_ref;
        }
    }

    vmaf_cambi_default_callbacks(&s->inc_range_callback, &s->dec_range_callback,
                                 &s->derivative_callback);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict) {
        err = -ENOMEM;
        goto free_ref;
    }

    s->has_pending = false;
    return 0;

free_ref:
    (void)vmaf_picture_unref(&s->pics[0]);
    (void)vmaf_picture_unref(&s->pics[1]);
    if (s->d_image)
        vmaf_sycl_free(s->sycl_state, s->d_image);
    if (s->d_mask)
        vmaf_sycl_free(s->sycl_state, s->d_mask);
    if (s->d_tmp)
        vmaf_sycl_free(s->sycl_state, s->d_tmp);
    if (s->h_image)
        vmaf_sycl_free(s->sycl_state, s->h_image);
    if (s->h_mask)
        vmaf_sycl_free(s->sycl_state, s->h_mask);
    free(s->buffers.diffs_to_consider);
    free(s->buffers.diff_weights);
    free(s->buffers.all_diffs);
    free(s->buffers.tvi_for_diff);
    free(s->buffers.c_values);
    free(s->buffers.c_values_histograms);
    free(s->buffers.mask_dp);
    free(s->buffers.filter_mode_buffer);
    free(s->buffers.derivative_buffer);
    if (s->feature_name_dict)
        (void)vmaf_dictionary_free(&s->feature_name_dict);
    return (err != 0) ? err : -ENOMEM;
}

/* ------------------------------------------------------------------ */
/* submit_fex_sycl                                                      */
/*                                                                      */
/* Synchronous per-scale loop (matches CUDA v1 posture). GPU work and  */
/* CPU residual both run in submit(); collect() only emits the score.   */
/* ------------------------------------------------------------------ */
static int submit_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    auto *s = static_cast<CambiStateSycl *>(fex->priv);
    auto *qptr = static_cast<sycl::queue *>(vmaf_sycl_get_queue_ptr(s->sycl_state));
    if (!qptr) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi_sycl: null queue pointer\n");
        return -EINVAL;
    }
    sycl::queue &q = *qptr;

    /* Step 1: host preprocessing → pics[0] (10-bit luma, proc_w × proc_h). */
    int err = vmaf_cambi_preprocessing(dist_pic, &s->pics[0], (int)s->proc_width,
                                       (int)s->proc_height, s->enc_bitdepth);
    if (err)
        return err;

    /* Step 2: H2D upload pics[0].data[0] → d_image (stride-aware). */
    {
        const ptrdiff_t src_stride_bytes = s->pics[0].stride[0];
        const uint8_t *src = static_cast<const uint8_t *>(s->pics[0].data[0]);
        const size_t row_bytes = s->proc_width * sizeof(uint16_t);
        for (unsigned row = 0; row < s->proc_height; row++) {
            const uint8_t *src_row = src + (size_t)row * (size_t)src_stride_bytes;
            uint16_t *dst_row = s->d_image + (size_t)row * s->proc_width;
            /* Synchronous row memcpy via USM (d_image is device USM). */
            q.memcpy(dst_row, src_row, row_bytes);
        }
        q.wait();
    }

    /* Step 3: GPU spatial mask at full scale. */
    {
        const unsigned mask_index = (unsigned)cambi_sycl_get_mask_index(
            s->proc_width, s->proc_height, CAMBI_SYCL_MASK_FILTER_SIZE);
        launch_spatial_mask(q, s->d_image, s->d_mask, s->proc_width, s->proc_height, s->proc_width,
                            mask_index);
        q.wait();
    }

    /* Step 4: per-scale loop. */
    unsigned scaled_w = s->proc_width;
    unsigned scaled_h = s->proc_height;
    const int num_diffs = 1 << s->max_log_contrast;
    double scores_per_scale[CAMBI_SYCL_NUM_SCALES] = {0.0, 0.0, 0.0, 0.0, 0.0};
    const double topk = (s->topk != CAMBI_SYCL_DEFAULT_TOPK) ? s->topk : s->cambi_topk;

    /* d_image / d_mask pointers are swapped each scale; track via locals. */
    uint16_t *cur_image = s->d_image;
    uint16_t *cur_mask = s->d_mask;
    uint16_t *cur_tmp = s->d_tmp;

    for (int scale = 0; scale < CAMBI_SYCL_NUM_SCALES; scale++) {
        if (scale > 0) {
            /* GPU decimate cur_image → cur_tmp. */
            const unsigned new_w = (scaled_w + 1u) >> 1;
            const unsigned new_h = (scaled_h + 1u) >> 1;
            launch_decimate(q, cur_image, cur_tmp, new_w, new_h, scaled_w, new_w);
            q.wait();
            {
                uint16_t *t = cur_image;
                cur_image = cur_tmp;
                cur_tmp = t;
            }
            /* GPU decimate cur_mask → cur_tmp. */
            launch_decimate(q, cur_mask, cur_tmp, new_w, new_h, scaled_w, new_w);
            q.wait();
            {
                uint16_t *t = cur_mask;
                cur_mask = cur_tmp;
                cur_tmp = t;
            }
            scaled_w = new_w;
            scaled_h = new_h;
        }

        /* GPU filter_mode H: cur_image → cur_tmp. */
        launch_filter_mode(q, cur_image, cur_tmp, scaled_w, scaled_h, scaled_w, 0);
        q.wait();
        /* GPU filter_mode V: cur_tmp → cur_image. */
        launch_filter_mode(q, cur_tmp, cur_image, scaled_w, scaled_h, scaled_w, 1);
        q.wait();

        /* D2H: cur_image → h_image, cur_mask → h_mask. */
        {
            const size_t row_bytes = scaled_w * sizeof(uint16_t);
            for (unsigned row = 0; row < scaled_h; row++) {
                q.memcpy(s->h_image + (size_t)row * scaled_w, cur_image + (size_t)row * scaled_w,
                         row_bytes);
                q.memcpy(s->h_mask + (size_t)row * scaled_w, cur_mask + (size_t)row * scaled_w,
                         row_bytes);
            }
            q.wait();
        }

        /* Copy h_image / h_mask → pics[0] / pics[1] (stride-aware). */
        {
            const ptrdiff_t pic_stride_bytes = s->pics[0].stride[0];
            uint8_t *dst0 = static_cast<uint8_t *>(s->pics[0].data[0]);
            uint8_t *dst1 = static_cast<uint8_t *>(s->pics[1].data[0]);
            const size_t row_bytes = scaled_w * sizeof(uint16_t);
            for (unsigned row = 0; row < scaled_h; row++) {
                (void)memcpy(dst0 + (size_t)row * (size_t)pic_stride_bytes,
                             s->h_image + (size_t)row * scaled_w, row_bytes);
                (void)memcpy(dst1 + (size_t)row * (size_t)pic_stride_bytes,
                             s->h_mask + (size_t)row * scaled_w, row_bytes);
            }
        }

        /* CPU residual: calculate_c_values + spatial pooling. */
        vmaf_cambi_calculate_c_values(&s->pics[0], &s->pics[1], s->buffers.c_values,
                                      s->buffers.c_values_histograms, s->adjusted_window,
                                      (uint16_t)num_diffs, s->buffers.tvi_for_diff, s->vlt_luma,
                                      s->buffers.diff_weights, s->buffers.all_diffs, (int)scaled_w,
                                      (int)scaled_h, s->inc_range_callback, s->dec_range_callback);

        scores_per_scale[scale] =
            vmaf_cambi_spatial_pooling(s->buffers.c_values, topk, scaled_w, scaled_h);
    }

    /* Final score. */
    const uint16_t pixels_in_window = vmaf_cambi_get_pixels_in_window(s->adjusted_window);
    double score = vmaf_cambi_weight_scores_per_scale(scores_per_scale, pixels_in_window);
    if (score > s->cambi_max_val)
        score = s->cambi_max_val;
    if (score < 0.0)
        score = 0.0;

    s->score = score;
    s->pending_index = index;
    s->has_pending = true;
    return 0;
}

/* ------------------------------------------------------------------ */
/* collect_fex_sycl — emit the pre-computed score. */
/* ------------------------------------------------------------------ */
static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<CambiStateSycl *>(fex->priv);
    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "Cambi_feature_cambi_score", s->score, index);
}

/* ------------------------------------------------------------------ */
/* close_fex_sycl */
/* ------------------------------------------------------------------ */
static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<CambiStateSycl *>(fex->priv);
    if (s->sycl_state) {
        if (s->d_image)
            vmaf_sycl_free(s->sycl_state, s->d_image);
        if (s->d_mask)
            vmaf_sycl_free(s->sycl_state, s->d_mask);
        if (s->d_tmp)
            vmaf_sycl_free(s->sycl_state, s->d_tmp);
        if (s->h_image)
            vmaf_sycl_free(s->sycl_state, s->h_image);
        if (s->h_mask)
            vmaf_sycl_free(s->sycl_state, s->h_mask);
    }
    (void)vmaf_picture_unref(&s->pics[0]);
    (void)vmaf_picture_unref(&s->pics[1]);
    free(s->buffers.c_values);
    free(s->buffers.c_values_histograms);
    free(s->buffers.mask_dp);
    free(s->buffers.filter_mode_buffer);
    free(s->buffers.derivative_buffer);
    free(s->buffers.diffs_to_consider);
    free(s->buffers.diff_weights);
    free(s->buffers.all_diffs);
    free(s->buffers.tvi_for_diff);
    if (s->feature_name_dict)
        (void)vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features_cambi_sycl[] = {"Cambi_feature_cambi_score", NULL};

extern "C" VmafFeatureExtractor vmaf_fex_cambi_sycl = {
    .name = "cambi_sycl",
    .init = init_fex_sycl,
    .extract = NULL,
    .flush = NULL,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options_cambi_sycl,
    .priv_size = sizeof(CambiStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features_cambi_sycl,
    /* 15 GPU dispatches/frame (5 scales × 3 kernels: mask + filter_H + filter_V).
     * dispatch_hint = DIRECT (matches CUDA twin and Vulkan twin): the per-frame
     * CPU residual (calculate_c_values) serialises frames already.
     * is_reduction_only = false: the GPU phases are not pure reductions. */
    .chars =
        {
            .n_dispatches_per_frame = 15,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_DIRECT,
        },
};

} /* extern "C" */
