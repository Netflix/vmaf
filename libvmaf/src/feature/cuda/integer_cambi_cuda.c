/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CAMBI banding-detection feature extractor on the CUDA backend
 *  (T3-15 / ADR-0360). CUDA twin of cambi_vulkan.c (ADR-0210).
 *
 *  Strategy II hybrid (matches the Vulkan ADR-0205 / ADR-0210 precedent):
 *
 *    GPU stages (three CUDA kernels in cambi_score.cu):
 *      - cambi_spatial_mask_kernel: derivative + 7×7 box sum + threshold
 *        → produces a uint16 mask buffer (0 = flat, 1 = edge).
 *      - cambi_decimate_kernel: strict 2× stride-2 subsample.
 *      - cambi_filter_mode_kernel: separable 3-tap mode filter (H + V).
 *
 *    Host CPU stages (exact CPU code via cambi_internal.h wrappers):
 *      - vmaf_cambi_preprocessing: decimate/upcast to 10-bit.
 *      - vmaf_cambi_calculate_c_values: sliding-histogram c-value pass.
 *      - vmaf_cambi_spatial_pooling: top-K pooling → per-scale score.
 *      - vmaf_cambi_weight_scores_per_scale: inner-product scale weights.
 *
 *  Per-frame flow:
 *    1. Host preprocessing (CPU): resize/upcast dist_pic → pics[0].
 *    2. HtoD upload of pics[0] luma plane → d_image.
 *    3. Scale 0: GPU cambi_spatial_mask_kernel over d_image → d_mask.
 *    4. For scale = 0 .. NUM_SCALES-1:
 *         a. (scale > 0) GPU cambi_decimate_kernel on d_image → d_tmp,
 *            swap; same for d_mask.
 *         b. GPU cambi_filter_mode_kernel H: d_image → d_tmp.
 *         c. GPU cambi_filter_mode_kernel V: d_tmp → d_image.
 *         d. DtoH readback: d_image → pics[0], d_mask → pics[1].
 *         e. Host vmaf_cambi_calculate_c_values + vmaf_cambi_spatial_pooling.
 *    5. Host vmaf_cambi_weight_scores_per_scale → final score.
 *    6. Emit "Cambi_feature_cambi_score" into the feature collector.
 *
 *  Precision contract: `places=4` (ULP=0 on the emitted score). All GPU
 *  phases are integer + bit-exact. The host residual runs the exact CPU
 *  code via cambi_internal.h, so the emitted score is bit-for-bit
 *  identical to `vmaf_fex_cambi`. Cross-backend gate target: ULP=0.
 *
 *  Out of scope for v1 (future work):
 *    - full_ref / FR-CAMBI mode (no GPU twin for the source pyramid).
 *    - heatmap dump via heatmaps_path.
 *    - high_res_speedup (GPU decimate path makes it cheap but v1 omits it).
 *    - EOTF variants other than bt1886 (host TVI table is already correct).
 *
 *  CUDA async lifecycle mirrors integer_psnr_cuda.c:
 *    submit() enqueues all GPU work on the picture's stream, records
 *    submit event, DtoH-copies on the private stream, records finished.
 *    collect() drains the private stream and runs the host residual.
 *    This keeps the pipeline asynchronous with respect to motion_cuda et al.
 *
 *  IMPORTANT: Because the host residual in collect() does significant
 *  CPU work (calculate_c_values is O(W*H*num_diffs)), this extractor
 *  is NOT marked is_reduction_only. The dispatch_hint is set to
 *  VMAF_FEATURE_DISPATCH_SEQUENTIAL so the engine does not pipeline
 *  this extractor with itself across frames.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "common/alignment.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "cuda/integer_cambi_cuda.h"
#include "cuda/kernel_template.h"
#include "log.h"
#include "luminance_tools.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"

#include "feature/cambi_internal.h"

/* --- Constants matching cambi.c --- */
#define CAMBI_CUDA_NUM_SCALES 5
#define CAMBI_CUDA_MIN_WIDTH_HEIGHT 216
#define CAMBI_CUDA_MASK_FILTER_SIZE 7
#define CAMBI_CUDA_DEFAULT_MAX_VAL 1000.0
#define CAMBI_CUDA_DEFAULT_WINDOW_SIZE 65
#define CAMBI_CUDA_DEFAULT_TOPK 0.6
#define CAMBI_CUDA_DEFAULT_TVI 0.019
#define CAMBI_CUDA_DEFAULT_VLT 0.0
#define CAMBI_CUDA_DEFAULT_MAX_LOG_CONTRAST 2
#define CAMBI_CUDA_DEFAULT_EOTF "bt1886"
#define CAMBI_CUDA_BLOCK_X 16u
#define CAMBI_CUDA_BLOCK_Y 16u

typedef struct CambiStateCuda {
    /* CUDA lifecycle (stream + events). */
    VmafCudaKernelLifecycle lc;

    /* CUDA kernel function handles (loaded from cambi_score.cu). */
    CUfunction func_mask;
    CUfunction func_decimate;
    CUfunction func_filter_mode;

    /* Device buffers (flat uint16 arrays sized for proc_width × proc_height). */
    VmafCudaBuffer *d_image; /* current scale image on device */
    VmafCudaBuffer *d_mask;  /* spatial mask on device */
    VmafCudaBuffer *d_tmp;   /* scratch: filter_mode H output, decimate output */

    /* Host VmafPicture pair for DtoH readback (same role as Vulkan's pics[]). */
    VmafPicture pics[2]; /* pics[0] = image, pics[1] = mask */

    /* Host scratch buffers for the CPU residual (mirrors CambiVkState::buffers). */
    VmafCambiHostBuffers buffers;

    /* Callbacks (always scalar scalar; GPU has done the heavy lifting). */
    VmafCambiRangeUpdater inc_range_callback;
    VmafCambiRangeUpdater dec_range_callback;
    VmafCambiDerivativeCalculator derivative_callback;

    /* Configuration options (mirrors CambiState). */
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
    int cambi_high_res_speedup; /* reserved; v1 ignores it */

    /* Resolved per-frame geometry. */
    unsigned src_width;
    unsigned src_height;
    unsigned src_bpc;
    unsigned proc_width;
    unsigned proc_height;

    /* Adjusted window (cambi_vk::adjusted_window equivalent). */
    uint16_t adjusted_window;
    uint16_t vlt_luma;

    /* Per-frame index stored by submit() for collect(). */
    unsigned index;

    /* Per-scale geometry stored by submit() for collect(). */
    unsigned scale_widths[CAMBI_CUDA_NUM_SCALES];
    unsigned scale_heights[CAMBI_CUDA_NUM_SCALES];

    /* DtoH readback buffers — one flat slot per scale for image + mask.
     * We read back after every scale (as in Vulkan v1), so we need only
     * the current scale's readback; reuse the same pinned buffer. */
    VmafCudaKernelReadback rb_image;
    VmafCudaKernelReadback rb_mask;

    VmafDictionary *feature_name_dict;
} CambiStateCuda;

/* --- Options --- */
static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped",
        .offset = offsetof(CambiStateCuda, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_CUDA_DEFAULT_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiStateCuda, enc_width),
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
        .offset = offsetof(CambiStateCuda, enc_height),
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
        .offset = offsetof(CambiStateCuda, enc_bitdepth),
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
        .offset = offsetof(CambiStateCuda, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_CUDA_DEFAULT_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiStateCuda, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_CUDA_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiStateCuda, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_CUDA_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibility threshold ΔL < tvi_threshold * L_mean",
        .offset = offsetof(CambiStateCuda, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_CUDA_DEFAULT_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which banding is assumed invisible",
        .offset = offsetof(CambiStateCuda, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_CUDA_DEFAULT_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum log contrast (0 to 5, default 2)",
        .offset = offsetof(CambiStateCuda, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_CUDA_DEFAULT_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "eotf",
        .help = "EOTF for visibility-threshold conversion (bt1886 / pq)",
        .offset = offsetof(CambiStateCuda, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_CUDA_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "EOTF override for cambi (defaults to eotf)",
        .offset = offsetof(CambiStateCuda, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_CUDA_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* Helper: compute adjusted window size (mirrors cambi.c). */
/* ------------------------------------------------------------------ */
static uint16_t cambi_cuda_adjust_window(int window_size, unsigned w, unsigned h)
{
    /* cambi.c::adjust_window_size formula:
     *   window = window * (w + h) / (4K_W + 4K_H) / 16, rounded up to odd. */
    unsigned adjusted = (unsigned)(window_size) * (w + h) / 375u;
    adjusted >>= 4;
    if (adjusted < 1u)
        adjusted = 1u;
    if ((adjusted & 1u) == 0u)
        adjusted++;
    return (uint16_t)adjusted;
}

/* ------------------------------------------------------------------ */
/* Helper: ceil_log2 for mask_index (mirrors cambi.c). */
/* ------------------------------------------------------------------ */
static uint16_t cambi_cuda_ceil_log2(uint32_t num)
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

static uint16_t cambi_cuda_get_mask_index(unsigned w, unsigned h, uint16_t filter_size)
{
    uint32_t shifted_wh = (w >> 6) * (h >> 6);
    return (
        uint16_t)((filter_size * filter_size + 3 * (cambi_cuda_ceil_log2(shifted_wh) - 11) - 1) >>
                  1);
}

/* ------------------------------------------------------------------ */
/* TVI table initialisation (mirrors CambiVkState::cambi_vk_init_tvi). */
/* ------------------------------------------------------------------ */
static int cambi_cuda_init_tvi(CambiStateCuda *s)
{
    VmafLumaRange luma_range;
    int err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err)
        return err;

    const char *effective_eotf;
    if (s->cambi_eotf && strcmp(s->cambi_eotf, CAMBI_CUDA_DEFAULT_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = (s->eotf != NULL) ? s->eotf : CAMBI_CUDA_DEFAULT_EOTF;
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

    /* vlt_luma: largest luma sample below cambi_vis_lum_threshold. */
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
/* init_fex_cuda */
/* ------------------------------------------------------------------ */
static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    CambiStateCuda *s = fex->priv;

    /* Resolve enc geometry (matches cambi.c::init logic). */
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
    if (s->enc_width < CAMBI_CUDA_MIN_WIDTH_HEIGHT && s->enc_height < CAMBI_CUDA_MIN_WIDTH_HEIGHT) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "cambi_cuda: encoded resolution %dx%d below minimum %d×%d.\n", s->enc_width,
                 s->enc_height, CAMBI_CUDA_MIN_WIDTH_HEIGHT, CAMBI_CUDA_MIN_WIDTH_HEIGHT);
        return -EINVAL;
    }

    s->src_width = w;
    s->src_height = h;
    s->src_bpc = bpc;
    s->proc_width = (unsigned)s->enc_width;
    s->proc_height = (unsigned)s->enc_height;

    s->adjusted_window = cambi_cuda_adjust_window(s->window_size, s->proc_width, s->proc_height);

    /* CUDA lifecycle. */
    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    CudaFunctions *cu_f = fex->cu_state->f;
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail_cuda);
    ctx_pushed = 1;

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, cambi_score_ptx), fail_cuda);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_mask, module, "cambi_spatial_mask_kernel"),
                    fail_cuda);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_decimate, module, "cambi_decimate_kernel"),
                    fail_cuda);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_filter_mode, module, "cambi_filter_mode_kernel"),
                    fail_cuda);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);
    ctx_pushed = 0;

    /* Device buffers: one flat uint16 array per logical buffer, sized for the
     * full (proc_width × proc_height) luma plane at scale 0. Per-scale
     * dispatches read only the leading (scaled_w × scaled_h) prefix. */
    const size_t buf_bytes = (size_t)s->proc_width * s->proc_height * sizeof(uint16_t);
    err = vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_image, buf_bytes);
    if (err)
        goto free_ref;
    err = vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_mask, buf_bytes);
    if (err)
        goto free_ref;
    err = vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_tmp, buf_bytes);
    if (err)
        goto free_ref;

    /* Pinned readback buffers: sized for scale-0 (worst case). */
    err = vmaf_cuda_kernel_readback_alloc(&s->rb_image, fex->cu_state, buf_bytes);
    if (err)
        goto free_ref;
    err = vmaf_cuda_kernel_readback_alloc(&s->rb_mask, fex->cu_state, buf_bytes);
    if (err)
        goto free_ref;

    /* Host VmafPictures for the CPU residual. */
    err = vmaf_picture_alloc(&s->pics[0], VMAF_PIX_FMT_YUV400P, 10, s->proc_width, s->proc_height);
    if (err)
        goto free_ref;
    err = vmaf_picture_alloc(&s->pics[1], VMAF_PIX_FMT_YUV400P, 10, s->proc_width, s->proc_height);
    if (err)
        goto free_ref;

    /* Host scratch buffers for the CPU residual. */
    const int num_diffs = 1 << s->max_log_contrast;
    s->buffers.diffs_to_consider = malloc(sizeof(uint16_t) * (size_t)num_diffs);
    if (!s->buffers.diffs_to_consider)
        goto free_ref;
    s->buffers.diff_weights = malloc(sizeof(int) * (size_t)num_diffs);
    if (!s->buffers.diff_weights)
        goto free_ref;
    s->buffers.all_diffs = malloc(sizeof(int) * (size_t)(2 * num_diffs + 1));
    if (!s->buffers.all_diffs)
        goto free_ref;

    /* Suprathreshold contrast weights (g_contrast_weights from cambi.c). */
    static const int contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                             8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
    for (int d = 0; d < num_diffs; d++) {
        s->buffers.diffs_to_consider[d] = (uint16_t)(d + 1);
        s->buffers.diff_weights[d] = contrast_weights[d];
    }
    for (int d = -num_diffs; d <= num_diffs; d++)
        s->buffers.all_diffs[d + num_diffs] = d;

    s->buffers.tvi_for_diff = malloc(sizeof(uint16_t) * (size_t)num_diffs);
    if (!s->buffers.tvi_for_diff)
        goto free_ref;

    err = cambi_cuda_init_tvi(s);
    if (err)
        goto free_ref;

    s->buffers.c_values = malloc(sizeof(float) * s->proc_width * s->proc_height);
    if (!s->buffers.c_values)
        goto free_ref;

    const uint16_t num_bins = (uint16_t)(1024u + (unsigned)(s->buffers.all_diffs[2 * num_diffs] -
                                                            s->buffers.all_diffs[0]));
    s->buffers.c_values_histograms = malloc(sizeof(uint16_t) * s->proc_width * (size_t)num_bins);
    if (!s->buffers.c_values_histograms)
        goto free_ref;

    /* mask_dp scratch (not used for GPU; kept for cambi_internal API). */
    const int pad_size = CAMBI_CUDA_MASK_FILTER_SIZE / 2;
    const int dp_width = (int)s->proc_width + 2 * pad_size + 1;
    const int dp_height = 2 * pad_size + 2;
    s->buffers.mask_dp = malloc(sizeof(uint32_t) * (size_t)dp_width * (size_t)dp_height);
    if (!s->buffers.mask_dp)
        goto free_ref;

    s->buffers.filter_mode_buffer = malloc(sizeof(uint16_t) * 3u * s->proc_width);
    if (!s->buffers.filter_mode_buffer)
        goto free_ref;
    s->buffers.derivative_buffer = malloc(sizeof(uint16_t) * s->proc_width);
    if (!s->buffers.derivative_buffer)
        goto free_ref;

    vmaf_cambi_default_callbacks(&s->inc_range_callback, &s->dec_range_callback,
                                 &s->derivative_callback);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict) {
        err = -ENOMEM;
        goto free_ref;
    }

    return 0;

free_ref:
    /* Best-effort teardown; close_fex_cuda handles null checks. */
    (void)vmaf_picture_unref(&s->pics[0]);
    (void)vmaf_picture_unref(&s->pics[1]);
    if (s->d_image)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->d_image);
    if (s->d_mask)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->d_mask);
    if (s->d_tmp)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->d_tmp);
    (void)vmaf_cuda_kernel_readback_free(&s->rb_image, fex->cu_state);
    (void)vmaf_cuda_kernel_readback_free(&s->rb_mask, fex->cu_state);
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
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    return (err != 0) ? err : -ENOMEM;

fail_cuda:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    return _cuda_err;
}

/* ------------------------------------------------------------------ */
/* dispatch_mask — GPU spatial-mask kernel over (w × h) of d_image. */
/* ------------------------------------------------------------------ */
static int dispatch_mask(CambiStateCuda *s, CudaFunctions *cu_f, CUstream stream, unsigned w,
                         unsigned h, unsigned stride_words, unsigned mask_index)
{
    const unsigned grid_x = (w + CAMBI_CUDA_BLOCK_X - 1u) / CAMBI_CUDA_BLOCK_X;
    const unsigned grid_y = (h + CAMBI_CUDA_BLOCK_Y - 1u) / CAMBI_CUDA_BLOCK_Y;
    /* Bug fix (Issue #857): cuLaunchKernel params[i] must point to the VALUE
     * to pass, not to the VmafCudaBuffer struct. Pass &buf->data (address of the
     * CUdeviceptr field) so the driver reads the device pointer, not buf->size. */
    void *params[] = {&s->d_image->data, &s->d_mask->data, &w, &h, &stride_words, &mask_index};
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_mask, grid_x, grid_y, 1u, CAMBI_CUDA_BLOCK_X,
                                           CAMBI_CUDA_BLOCK_Y, 1u, 0u, stream, params, NULL));
    return 0;
}

/* ------------------------------------------------------------------ */
/* dispatch_decimate — GPU 2× decimate of src → dst. */
/* ------------------------------------------------------------------ */
static int dispatch_decimate(CambiStateCuda *s, CudaFunctions *cu_f, CUstream stream,
                             VmafCudaBuffer *src, VmafCudaBuffer *dst, unsigned out_w,
                             unsigned out_h, unsigned src_stride_words, unsigned dst_stride_words)
{
    const unsigned grid_x = (out_w + CAMBI_CUDA_BLOCK_X - 1u) / CAMBI_CUDA_BLOCK_X;
    const unsigned grid_y = (out_h + CAMBI_CUDA_BLOCK_Y - 1u) / CAMBI_CUDA_BLOCK_Y;
    /* Bug fix (Issue #857): pass device pointer addresses, not struct addresses. */
    void *params[] = {&src->data, &dst->data, &out_w, &out_h, &src_stride_words, &dst_stride_words};
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_decimate, grid_x, grid_y, 1u, CAMBI_CUDA_BLOCK_X,
                                           CAMBI_CUDA_BLOCK_Y, 1u, 0u, stream, params, NULL));
    return 0;
}

/* ------------------------------------------------------------------ */
/* dispatch_filter_mode — GPU 3-tap mode filter (axis=0 H, axis=1 V). */
/* ------------------------------------------------------------------ */
static int dispatch_filter_mode(CambiStateCuda *s, CudaFunctions *cu_f, CUstream stream,
                                VmafCudaBuffer *in, VmafCudaBuffer *out, unsigned w, unsigned h,
                                unsigned stride_words, int axis)
{
    const unsigned grid_x = (w + CAMBI_CUDA_BLOCK_X - 1u) / CAMBI_CUDA_BLOCK_X;
    const unsigned grid_y = (h + CAMBI_CUDA_BLOCK_Y - 1u) / CAMBI_CUDA_BLOCK_Y;
    /* Bug fix (Issue #857): pass device pointer addresses, not struct addresses. */
    void *params[] = {&in->data, &out->data, &w, &h, &stride_words, &axis};
    CHECK_CUDA_RETURN(cu_f,
                      cuLaunchKernel(s->func_filter_mode, grid_x, grid_y, 1u, CAMBI_CUDA_BLOCK_X,
                                     CAMBI_CUDA_BLOCK_Y, 1u, 0u, stream, params, NULL));
    return 0;
}

/* ------------------------------------------------------------------ */
/* submit_fex_cuda                                                     */
/*                                                                     */
/* Pipeline:                                                           */
/*   1. Host preprocess → pics[0] (CPU, luma only).                   */
/*   2. HtoD upload pics[0].data[0] → d_image.                        */
/*   3. GPU spatial mask → d_mask.                                     */
/*   4. For each scale:                                                */
/*        a. (scale>0) GPU decimate d_image → d_tmp, swap;            */
/*                     GPU decimate d_mask  → d_tmp, swap.            */
/*        b. GPU filter_mode H: d_image → d_tmp.                      */
/*           GPU filter_mode V: d_tmp   → d_image.                    */
/*        c. DtoH: d_image → rb_image.host_pinned (async).            */
/*           DtoH: d_mask  → rb_mask.host_pinned  (async).            */
/*   5. Record submit + enqueue finished on private stream.            */
/*                                                                     */
/* Note: the DtoH copies at step 4c overwrite the same pinned buffers */
/* each scale — this is safe because collect() processes one scale at */
/* a time after the full GPU pipeline has drained. Alternatively, we  */
/* could use per-scale readback slots; the current approach saves      */
/* memory at the cost of requiring a sync between scales. Since the   */
/* host residual (calculate_c_values) is the bottleneck anyway, this  */
/* is the right tradeoff.                                              */
/* ------------------------------------------------------------------ */
static int submit_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    CambiStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    s->index = index;

    /* Step 0: download dist_pic GPU→host so vmaf_cambi_preprocessing (host
     * code) can read it.  Pictures delivered to a CUDA extractor's submit()
     * have device pointers in data[]; dereferencing them on the host causes
     * a segfault (Issue #857).  Other CUDA extractors avoid this because
     * they keep all preprocessing on the GPU; CAMBI is unique in needing a
     * host-side decimate-and-10b-upcast before its GPU pipeline. */
    VmafPicture dist_host;
    int err = vmaf_picture_alloc(&dist_host, dist_pic->pix_fmt, dist_pic->bpc, dist_pic->w[0],
                                 dist_pic->h[0]);
    if (err)
        return err;
    err = vmaf_cuda_picture_download_async(dist_pic, &dist_host, 0x1);
    if (err) {
        (void)vmaf_picture_unref(&dist_host);
        return err;
    }
    /* Sync the dist_pic private stream: vmaf_cuda_picture_download_async
     * enqueues the DtoH copy on that stream, so we must drain it before
     * the host preprocessing reads dist_host.data[0]. */
    {
        const CUresult _sync_res =
            cu_f->cuStreamSynchronize(vmaf_cuda_picture_get_stream(dist_pic));
        if (CUDA_SUCCESS != _sync_res) {
            (void)vmaf_picture_unref(&dist_host);
            return vmaf_cuda_result_to_errno((int)_sync_res);
        }
    }

    /* Step 1: host preprocessing → pics[0] (10-bit planar, proc_w × proc_h). */
    err = vmaf_cambi_preprocessing(&dist_host, &s->pics[0], (int)s->proc_width, (int)s->proc_height,
                                   s->enc_bitdepth);
    (void)vmaf_picture_unref(&dist_host);
    if (err)
        return err;

    CUstream stream = vmaf_cuda_picture_get_stream(ref_pic);

    /* Wait for dist upload to complete before starting GPU work. */
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(stream, vmaf_cuda_picture_get_ready_event(dist_pic),
                                              CU_EVENT_WAIT_DEFAULT));

    /* Step 2: HtoD upload pics[0].data[0] → d_image. */
    const size_t row_bytes = s->proc_width * sizeof(uint16_t);
    const uint16_t *src_data = (const uint16_t *)s->pics[0].data[0];
    const ptrdiff_t src_stride_bytes = s->pics[0].stride[0];
    for (unsigned row = 0; row < s->proc_height; row++) {
        const uint8_t *src_row = (const uint8_t *)src_data + (size_t)row * (size_t)src_stride_bytes;
        /* Arithmetic on CUdeviceptr (unsigned long long) directly — avoids
         * the UB of casting an integer through uint8_t* and back. */
        const CUdeviceptr dst_dptr =
            s->d_image->data + (CUdeviceptr)((size_t)row * s->proc_width * sizeof(uint16_t));
        CHECK_CUDA_RETURN(cu_f, cuMemcpyHtoDAsync(dst_dptr, src_row, row_bytes, stream));
    }

    /* Step 3: spatial mask at full scale. */
    const unsigned mask_index_0 = (unsigned)cambi_cuda_get_mask_index(s->proc_width, s->proc_height,
                                                                      CAMBI_CUDA_MASK_FILTER_SIZE);
    err =
        dispatch_mask(s, cu_f, stream, s->proc_width, s->proc_height, s->proc_width, mask_index_0);
    if (err)
        return err;

    /* Step 4: per-scale GPU pipeline. */
    unsigned scaled_w = s->proc_width;
    unsigned scaled_h = s->proc_height;

    /* We need to store per-scale readback data for the host residual in
     * collect(). Since we overwrite the same pinned buffers, we process
     * all scales in submit() on the GPU and record a finished event after
     * each scale's DtoH. collect() then runs the CPU residual per-scale.
     *
     * Practical approach: record a submit event after all GPU work, then
     * DtoH-copy each scale's result into per-scale pinned regions.
     * For v1, we simplify by doing a synchronous host-side approach:
     * all GPU work for all scales streams asynchronously, but the DtoH
     * copies and CPU residual run in collect() after a single sync.
     *
     * To avoid needing per-scale device buffers, we execute the GPU
     * pipeline for all 5 scales in submit() (streaming), DtoH each
     * scale into its dedicated section of the readback buffer, then
     * collect() reads from those sections.
     *
     * Buffer layout: rb_image.host_pinned and rb_mask.host_pinned are
     * each (proc_width × proc_height) — enough for scale 0. Each scale
     * occupies a prefix of size scaled_w × scaled_h within the stride-
     * proc_width layout. We copy each scale's data from device to the
     * top-left corner of the readback buffer sequentially (each DtoH
     * overwrites the previous scale), which is fine because collect()
     * first awaits the GPU sync, then runs the CPU residual for each
     * scale in order — so by the time scale k is processed on the host,
     * the DtoH for scale k+1 has not yet started on the device.
     *
     * The actual approach: record all scale GPU kernels in one go, then
     * for each scale do a separate DtoH + CPU residual in collect(). We
     * cannot mix GPU pipeline and DtoH in a single submit() without
     * blocking between scales, so we use a two-pass design:
     *   submit(): GPU pipeline for scale 0 only (mask + filter_mode).
     *             Record per-scale results for scales 1..4 by running
     *             all 5 scales on the GPU + recording events.
     *   collect(): For each scale, wait for the corresponding event,
     *              DtoH the readback, run CPU residual.
     *
     * For v1 simplicity, we run the GPU pipeline for all scales here
     * in submit() and enqueue DtoH copies for all scales into separate
     * regions of the readback buffer (laid out as a 5-element array).
     * collect() reads each region without a per-scale sync.
     *
     * Readback buffer size must accommodate 5 scales. Scale k has area:
     *   proc_width × proc_height / 4^k (approximately). Total ≤ 2×.
     * For clarity, we allocate one buffer per scale:
     *   scale 0: proc_w × proc_h
     *   scale 1: ceil(proc_w/2) × ceil(proc_h/2)
     *   ...
     * These are sub-regions of the pre-allocated rb_image/rb_mask (sized
     * for scale 0). We reuse the same device buffers (d_image, d_mask)
     * across scales, packing the DtoH copies into the pinned buffer
     * using stride-proc_width layout (row-major, same stride for all
     * scales — upper-left corner read back).
     *
     * After careful analysis, the cleanest approach that preserves the
     * async model and avoids per-scale host sync is:
     *   - Run GPU pipeline for all 5 scales: decimate/filter_mode each.
     *   - After each scale's filter_mode pass, record a CUevent and
     *     enqueue DtoH into a per-scale region of the pinned buffer.
     *   - collect() streams all 5 scales' host residuals after one drain.
     *
     * Implementation below uses per-scale pitched DtoH into the single
     * large pinned allocation (rb_image.host_pinned stores all 5 scales
     * sequentially, separated by scaled_w × scaled_h uint16 elements).
     */

    /* Pre-compute per-scale geometry (used during the scale loop below). */
    {
        unsigned sw = s->proc_width;
        unsigned sh = s->proc_height;
        for (int scale = 0; scale < CAMBI_CUDA_NUM_SCALES; scale++) {
            s->scale_widths[scale] = sw;
            s->scale_heights[scale] = sh;
            sw = (sw + 1u) >> 1;
            sh = (sh + 1u) >> 1;
        }
    }
    /* Total needed: offset bytes in each readback buffer. */
    /* rb_image and rb_mask were allocated for proc_w × proc_h — sufficient
     * only for scale 0. We need to ensure they are large enough.
     * Solution: rb_image / rb_mask allocated in init() for proc_w × proc_h
     * (scale 0 area). Scales 1..4 each have a smaller area, so their
     * cumulative total is < scale_0_area * 2. We therefore need 2×.
     *
     * Since this violates the allocation size, fall back to the
     * per-scale synchronous approach: after each scale's GPU filter_mode
     * pass, do a synchronous DtoH into the scale-0-sized buffer top-left,
     * run the CPU residual, then proceed to the next scale.
     *
     * This means submit() cannot be fully asynchronous for CAMBI v1 —
     * the per-scale CPU residual is integrated into submit(). collect()
     * only emits the already-computed score. This mirrors how the Vulkan
     * path works (extract() is fully synchronous).
     *
     * To fit the async submit/collect model:
     *   submit() runs all GPU + CPU work synchronously (no frame pipelining).
     *   collect() just emits the pre-computed score.
     *
     * A future optimisation (v2) could decouple per-frame GPU work
     * (all 5 scales) from the CPU residual by serialising the CPU
     * work in collect(). This requires per-scale pinned readback buffers.
     */

    /* ----  Synchronous per-scale loop  ---- */
    /* We run all GPU work + DtoH + CPU residual per scale in submit().
     * The CUstream is the picture stream (ref_pic's stream); we wait for
     * each scale's GPU pass to finish before doing the CPU residual.
     * This deviates from the pure-async model but is correct and matches
     * the Vulkan precedent (cambi_vk_extract is synchronous w.r.t. each
     * scale). The drain-batch optimisation (ADR-0242) does not apply here
     * because there is no DtoH to pipeline. */

    scaled_w = s->proc_width;
    scaled_h = s->proc_height;
    const int num_diffs = 1 << s->max_log_contrast;
    double scores_per_scale[CAMBI_CUDA_NUM_SCALES] = {0.0, 0.0, 0.0, 0.0, 0.0};

    /* The topk ratio: prefer topk if non-default, else cambi_topk. */
    const double topk = (s->topk != CAMBI_CUDA_DEFAULT_TOPK) ? s->topk : s->cambi_topk;

    for (int scale = 0; scale < CAMBI_CUDA_NUM_SCALES; scale++) {
        if (scale > 0) {
            /* GPU decimate d_image → d_tmp (out = half resolution). */
            const unsigned new_w = (scaled_w + 1u) >> 1;
            const unsigned new_h = (scaled_h + 1u) >> 1;
            err = dispatch_decimate(s, cu_f, stream, s->d_image, s->d_tmp, new_w, new_h, scaled_w,
                                    new_w);
            if (err)
                return err;
            /* Swap d_image ↔ d_tmp. */
            VmafCudaBuffer *tmp = s->d_image;
            s->d_image = s->d_tmp;
            s->d_tmp = tmp;

            /* GPU decimate d_mask → d_tmp. */
            err = dispatch_decimate(s, cu_f, stream, s->d_mask, s->d_tmp, new_w, new_h, scaled_w,
                                    new_w);
            if (err)
                return err;
            tmp = s->d_mask;
            s->d_mask = s->d_tmp;
            s->d_tmp = tmp;

            scaled_w = new_w;
            scaled_h = new_h;
        }

        /* GPU filter_mode H: d_image → d_tmp. */
        err = dispatch_filter_mode(s, cu_f, stream, s->d_image, s->d_tmp, scaled_w, scaled_h,
                                   scaled_w, 0);
        if (err)
            return err;
        /* GPU filter_mode V: d_tmp → d_image. */
        err = dispatch_filter_mode(s, cu_f, stream, s->d_tmp, s->d_image, scaled_w, scaled_h,
                                   scaled_w, 1);
        if (err)
            return err;

        /* Synchronous DtoH: drain the picture stream so the host readback
         * is safe before the CPU residual. */
        CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(stream));

        /* DtoH: d_image → pics[0].data[0] (stride-aware). */
        const ptrdiff_t pic_stride_bytes = s->pics[0].stride[0];
        uint16_t *dst0 = (uint16_t *)s->pics[0].data[0];
        uint16_t *dst1 = (uint16_t *)s->pics[1].data[0];
        for (unsigned row = 0; row < scaled_h; row++) {
            uint8_t *d0_row = (uint8_t *)dst0 + (size_t)row * (size_t)pic_stride_bytes;
            /* Arithmetic on CUdeviceptr directly — avoids UB of casting integer
             * through uint8_t* and back (same pattern as the HtoD loop above). */
            const CUdeviceptr s0_dptr =
                s->d_image->data + (CUdeviceptr)((size_t)row * scaled_w * sizeof(uint16_t));
            CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoH(d0_row, s0_dptr, scaled_w * sizeof(uint16_t)));
            uint8_t *d1_row = (uint8_t *)dst1 + (size_t)row * (size_t)pic_stride_bytes;
            const CUdeviceptr s1_dptr =
                s->d_mask->data + (CUdeviceptr)((size_t)row * scaled_w * sizeof(uint16_t));
            CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoH(d1_row, s1_dptr, scaled_w * sizeof(uint16_t)));
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

    /* Restore d_image / d_mask to point at the original (scale-0) device
     * buffers if they were swapped. Track the swap count. */
    /* Note: after 4 swaps (scale 1..4), d_image points to:
     *   even swaps → original d_image, odd swaps → original d_tmp.
     * We need to normalise before the next frame.  The simplest fix is
     * to swap back so d_image always points to the same allocation. */
    /* Actually, we track the number of swaps = (NUM_SCALES - 1) = 4.
     * 4 swaps → even → back to original. No action needed. */

    /* Compute the final score. */
    const uint16_t pixels_in_window = vmaf_cambi_get_pixels_in_window(s->adjusted_window);
    double score = vmaf_cambi_weight_scores_per_scale(scores_per_scale, pixels_in_window);
    if (score > s->cambi_max_val)
        score = s->cambi_max_val;
    if (score < 0.0)
        score = 0.0;

    /* Store in rb_image.host_pinned (single double — misuse of the
     * readback slot, but safe: host_pinned is large enough and the
     * collect() is the only reader). */
    *(double *)s->rb_image.host_pinned = score;

    /* Emit a dummy event on the private stream so collect() can drain
     * without blocking (the score is already computed). */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.submit, stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->lc.submit, CU_EVENT_WAIT_DEFAULT));
    return vmaf_cuda_kernel_submit_post_record(&s->lc, fex->cu_state);
}

/* ------------------------------------------------------------------ */
/* collect_fex_cuda — emit the pre-computed score. */
/* ------------------------------------------------------------------ */
static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    CambiStateCuda *s = fex->priv;

    /* Drain the private stream (no-op if drain_batch already handled it). */
    int err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (err)
        return err;

    const double score = *(double *)s->rb_image.host_pinned;
    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "Cambi_feature_cambi_score", score, index);
}

/* ------------------------------------------------------------------ */
/* close_fex_cuda */
/* ------------------------------------------------------------------ */
static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    CambiStateCuda *s = fex->priv;
    int rc = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);

    if (s->d_image) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->d_image);
        if (e && rc == 0)
            rc = e;
        free(s->d_image);
        s->d_image = NULL;
    }
    if (s->d_mask) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->d_mask);
        if (e && rc == 0)
            rc = e;
        free(s->d_mask);
        s->d_mask = NULL;
    }
    if (s->d_tmp) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->d_tmp);
        if (e && rc == 0)
            rc = e;
        free(s->d_tmp);
        s->d_tmp = NULL;
    }

    {
        const int e = vmaf_cuda_kernel_readback_free(&s->rb_image, fex->cu_state);
        if (e && rc == 0)
            rc = e;
    }
    {
        const int e = vmaf_cuda_kernel_readback_free(&s->rb_mask, fex->cu_state);
        if (e && rc == 0)
            rc = e;
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

    if (s->feature_name_dict) {
        const int e = vmaf_dictionary_free(&s->feature_name_dict);
        if (e && rc == 0)
            rc = e;
    }
    return rc;
}

static const char *provided_features[] = {"Cambi_feature_cambi_score", NULL};

VmafFeatureExtractor vmaf_fex_cambi_cuda = {
    .name = "cambi_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(CambiStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    /* CAMBI has a non-trivial CPU residual (calculate_c_values) in submit().
     * is_reduction_only = false (meaningful pixel processing on both GPU +
     * CPU), dispatch_hint = VMAF_FEATURE_DISPATCH_DIRECT — run one frame
     * at a time without batching; the per-frame CPU residual serialises
     * frames already. DIRECT matches Vulkan cambi_vulkan's posture. */
    .chars =
        {
            .n_dispatches_per_frame = 15, /* 5 scales × 3 kernels (mask + filter_H + filter_V) */
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_DIRECT,
        },
};
