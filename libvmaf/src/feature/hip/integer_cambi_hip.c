/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CAMBI banding-detection feature extractor on the HIP backend.
 *  Direct port of `libvmaf/src/feature/cuda/integer_cambi_cuda.c`
 *  (T3-15 / ADR-0360) to the HIP backend.
 *
 *  Strategy II hybrid (matches the CUDA twin):
 *    GPU stages (three HIP kernels in cambi_score.hip):
 *      - cambi_spatial_mask_kernel: 7×7 box derivative + threshold.
 *      - cambi_decimate_kernel: strict 2× stride-2 subsample.
 *      - cambi_filter_mode_kernel: separable 3-tap mode filter (H + V).
 *
 *    Host CPU stages (via cambi_internal.h wrappers — bit-exact):
 *      - vmaf_cambi_preprocessing: decimate/upcast to 10-bit.
 *      - vmaf_cambi_calculate_c_values: sliding-histogram c-value pass.
 *      - vmaf_cambi_spatial_pooling: top-K pooling → per-scale score.
 *      - vmaf_cambi_weight_scores_per_scale: inner-product scale weights.
 *
 *  HIP adaptation notes vs. CUDA twin:
 *    - `CUmodule` / `cuModuleLoadData` → `hipModule_t` / `hipModuleLoadData`.
 *    - `CUfunction` / `cuModuleGetFunction` → `hipFunction_t` / `hipModuleGetFunction`.
 *    - `cuLaunchKernel` → `hipModuleLaunchKernel`.
 *    - `CUdeviceptr` / `cuMemcpyHtoDAsync` / `cuMemcpyDtoH` →
 *      `hipDeviceptr_t` / `hipMemcpyHtoDAsync` / `hipMemcpyDtoH`.
 *    - `cuStreamSynchronize` → `hipStreamSynchronize`.
 *    - `cuCtxPushCurrent` / `cuCtxPopCurrent` — not needed; HIP uses the
 *      default device context selected by `hipSetDevice`.
 *    - `CuStreamWaitEvent` → `hipStreamWaitEvent`.
 *    - `cuEventRecord` → `hipEventRecord`.
 *    - `VmafCudaKernelLifecycle` / `VmafCudaBuffer` / `VmafCudaKernelReadback`
 *      → `VmafHipKernelLifecycle` / dedicated `hipDeviceptr_t` device
 *      allocations via `hipMalloc` / `hipFree` / host pinned via
 *      `VmafHipKernelReadback`.
 *    - `vmaf_cuda_*` → `vmaf_hip_*` equivalents.
 *
 *  Precision contract: `places=4` (ULP=0 on the emitted score). GPU
 *  phases are integer + bit-exact. The host residual runs the exact CPU
 *  code via cambi_internal.h. Cross-backend gate target: ULP=0.
 *
 *  Lifecycle: mirrors the CUDA twin's synchronous-per-scale approach.
 *  submit() runs all GPU + CPU work per scale synchronously; collect()
 *  emits the pre-computed score. This is correct and matches the Vulkan
 *  precedent (cambi_vulkan::extract is also synchronous per-scale).
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
#include "luminance_tools.h"
#include "mem.h"
#include "picture.h"

/* HIP common.h must come before <hip/hip_runtime_api.h>: common.h provides
 * the typedef stubs (hipError_t = int) used in the non-HAVE_HIPCC path.
 * Including the real HIP header first causes a type-redefinition error. */
#include "../../hip/common.h"
#include "../../hip/kernel_template.h"
#include "integer_cambi_hip.h"

#ifdef HAVE_HIPCC
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#endif /* HAVE_HIPCC */

#include "feature/cambi_internal.h"

/* --- Constants matching cambi.c --- */
#define CAMBI_HIP_NUM_SCALES 5
#define CAMBI_HIP_MIN_WIDTH_HEIGHT 216
#define CAMBI_HIP_MASK_FILTER_SIZE 7
#define CAMBI_HIP_DEFAULT_MAX_VAL 1000.0
#define CAMBI_HIP_DEFAULT_WINDOW_SIZE 65
#define CAMBI_HIP_DEFAULT_TOPK 0.6
#define CAMBI_HIP_DEFAULT_TVI 0.019
#define CAMBI_HIP_DEFAULT_VLT 0.0
#define CAMBI_HIP_DEFAULT_MAX_LOG_CONTRAST 2
#define CAMBI_HIP_DEFAULT_EOTF "bt1886"
#define CAMBI_HIP_BLOCK_X 16u
#define CAMBI_HIP_BLOCK_Y 16u

/* ------------------------------------------------------------------ */
/* Private state                                                       */
/* ------------------------------------------------------------------ */
typedef struct CambiStateHip {
    /* HIP lifecycle (stream + events). */
    VmafHipKernelLifecycle lc;
    VmafHipContext *ctx;

#ifdef HAVE_HIPCC
    /* HIP module + kernel function handles (require real HIP types). */
    hipModule_t module;
    hipFunction_t func_mask;
    hipFunction_t func_decimate;
    hipFunction_t func_filter_mode;

    /* Device buffers (flat uint16 arrays, proc_width × proc_height). */
    hipDeviceptr_t d_image; /* current scale image on device */
    hipDeviceptr_t d_mask;  /* spatial mask on device */
    hipDeviceptr_t d_tmp;   /* scratch: filter_mode H output, decimate output */
    size_t d_buf_bytes;     /* allocation size of each device buffer (scale 0) */
#endif                      /* HAVE_HIPCC */

    /* Host VmafPicture pair for CPU residual. */
    VmafPicture pics[2]; /* pics[0] = image, pics[1] = mask */

    /* Pinned host readback: just a double for the pre-computed score. */
    VmafHipKernelReadback rb_score;

    /* Host scratch buffers for the CPU residual. */
    VmafCambiHostBuffers buffers;

    /* Callbacks (scalar; GPU has done the heavy lifting). */
    VmafCambiRangeUpdater inc_range_callback;
    VmafCambiRangeUpdater dec_range_callback;
    VmafCambiDerivativeCalculator derivative_callback;

    /* Configuration options (mirrors CambiStateCuda). */
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

    /* Adjusted window and vlt_luma threshold. */
    uint16_t adjusted_window;
    uint16_t vlt_luma;

    /* Per-frame index stored by submit() for collect(). */
    unsigned index;

    VmafDictionary *feature_name_dict;
} CambiStateHip;

/* --- Options --- */
static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped",
        .offset = offsetof(CambiStateHip, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_HIP_DEFAULT_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiStateHip, enc_width),
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
        .offset = offsetof(CambiStateHip, enc_height),
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
        .offset = offsetof(CambiStateHip, enc_bitdepth),
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
        .offset = offsetof(CambiStateHip, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_HIP_DEFAULT_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiStateHip, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_HIP_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiStateHip, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_HIP_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibility threshold delta-L < tvi_threshold * L_mean",
        .offset = offsetof(CambiStateHip, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_HIP_DEFAULT_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which banding is assumed invisible",
        .offset = offsetof(CambiStateHip, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_HIP_DEFAULT_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum log contrast (0 to 5, default 2)",
        .offset = offsetof(CambiStateHip, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_HIP_DEFAULT_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "eotf",
        .help = "EOTF for visibility-threshold conversion (bt1886 / pq)",
        .offset = offsetof(CambiStateHip, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_HIP_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "EOTF override for cambi (defaults to eotf)",
        .offset = offsetof(CambiStateHip, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_HIP_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* Helper: compute adjusted window size (mirrors cambi.c). */
/* ------------------------------------------------------------------ */
static uint16_t cambi_hip_adjust_window(int window_size, unsigned w, unsigned h)
{
    unsigned adjusted = (unsigned)(window_size) * (w + h) / 375u;
    adjusted >>= 4;
    if (adjusted < 1u)
        adjusted = 1u;
    if ((adjusted & 1u) == 0u)
        adjusted++;
    return (uint16_t)adjusted;
}

#ifdef HAVE_HIPCC

/* ------------------------------------------------------------------ */
/* HIP error → errno translation (only needed with real HIP runtime). */
/* ------------------------------------------------------------------ */
static int hip_err(hipError_t rc)
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
/* Helper: ceil_log2 for mask_index (mirrors cambi.c). */
/* ------------------------------------------------------------------ */
static uint16_t cambi_hip_ceil_log2(uint32_t num)
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

static uint16_t cambi_hip_get_mask_index(unsigned w, unsigned h, uint16_t filter_size)
{
    uint32_t shifted_wh = (w >> 6) * (h >> 6);
    return (
        uint16_t)((filter_size * filter_size + 3 * (cambi_hip_ceil_log2(shifted_wh) - 11) - 1) >>
                  1);
}

/* ------------------------------------------------------------------ */
/* TVI table initialisation. */
/* ------------------------------------------------------------------ */
static int cambi_hip_init_tvi(CambiStateHip *s)
{
    VmafLumaRange luma_range;
    int err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err)
        return err;

    const char *effective_eotf;
    if (s->cambi_eotf && strcmp(s->cambi_eotf, CAMBI_HIP_DEFAULT_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = (s->eotf != NULL) ? s->eotf : CAMBI_HIP_DEFAULT_EOTF;
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
/* Module load helper (extracted to keep init under 60 lines). */
/* ------------------------------------------------------------------ */
#ifdef HAVE_HIPCC
static int cambi_hip_module_load(CambiStateHip *s)
{
    hipError_t hip_rc = hipModuleLoadData(&s->module, cambi_score_hsaco);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);

    hip_rc = hipModuleGetFunction(&s->func_mask, s->module, "cambi_spatial_mask_kernel");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return hip_err(hip_rc);
    }
    hip_rc = hipModuleGetFunction(&s->func_decimate, s->module, "cambi_decimate_kernel");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return hip_err(hip_rc);
    }
    hip_rc = hipModuleGetFunction(&s->func_filter_mode, s->module, "cambi_filter_mode_kernel");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return hip_err(hip_rc);
    }
    return 0;
}
#endif /* HAVE_HIPCC */

/* ------------------------------------------------------------------ */
/* Device buffer free helper. */
/* ------------------------------------------------------------------ */
static void cambi_hip_free_device_buffers(CambiStateHip *s)
{
#ifdef HAVE_HIPCC
    if (s->d_image) {
        (void)hipFree((void *)s->d_image);
        s->d_image = (hipDeviceptr_t)0;
    }
    if (s->d_mask) {
        (void)hipFree((void *)s->d_mask);
        s->d_mask = (hipDeviceptr_t)0;
    }
    if (s->d_tmp) {
        (void)hipFree((void *)s->d_tmp);
        s->d_tmp = (hipDeviceptr_t)0;
    }
#else
    (void)s;
#endif /* HAVE_HIPCC */
}

/* ------------------------------------------------------------------ */
/* Kernel dispatch helpers. */
/* ------------------------------------------------------------------ */
#ifdef HAVE_HIPCC

static int dispatch_mask_hip(CambiStateHip *s, hipStream_t stream, unsigned w, unsigned h,
                             unsigned stride_words, unsigned mask_index)
{
    const unsigned grid_x = (w + CAMBI_HIP_BLOCK_X - 1u) / CAMBI_HIP_BLOCK_X;
    const unsigned grid_y = (h + CAMBI_HIP_BLOCK_Y - 1u) / CAMBI_HIP_BLOCK_Y;
    void *params[] = {&s->d_image, &s->d_mask, &w, &h, &stride_words, &mask_index};
    hipError_t rc = hipModuleLaunchKernel(s->func_mask, grid_x, grid_y, 1u, CAMBI_HIP_BLOCK_X,
                                          CAMBI_HIP_BLOCK_Y, 1u, 0u, stream, params, NULL);
    return hip_err(rc);
}

static int dispatch_decimate_hip(CambiStateHip *s, hipStream_t stream, hipDeviceptr_t src,
                                 hipDeviceptr_t dst, unsigned out_w, unsigned out_h,
                                 unsigned src_stride_words, unsigned dst_stride_words)
{
    const unsigned grid_x = (out_w + CAMBI_HIP_BLOCK_X - 1u) / CAMBI_HIP_BLOCK_X;
    const unsigned grid_y = (out_h + CAMBI_HIP_BLOCK_Y - 1u) / CAMBI_HIP_BLOCK_Y;
    void *params[] = {&src, &dst, &out_w, &out_h, &src_stride_words, &dst_stride_words};
    hipError_t rc = hipModuleLaunchKernel(s->func_decimate, grid_x, grid_y, 1u, CAMBI_HIP_BLOCK_X,
                                          CAMBI_HIP_BLOCK_Y, 1u, 0u, stream, params, NULL);
    return hip_err(rc);
}

static int dispatch_filter_mode_hip(CambiStateHip *s, hipStream_t stream, hipDeviceptr_t in,
                                    hipDeviceptr_t out, unsigned w, unsigned h,
                                    unsigned stride_words, int axis)
{
    const unsigned grid_x = (w + CAMBI_HIP_BLOCK_X - 1u) / CAMBI_HIP_BLOCK_X;
    const unsigned grid_y = (h + CAMBI_HIP_BLOCK_Y - 1u) / CAMBI_HIP_BLOCK_Y;
    void *params[] = {&in, &out, &w, &h, &stride_words, &axis};
    hipError_t rc =
        hipModuleLaunchKernel(s->func_filter_mode, grid_x, grid_y, 1u, CAMBI_HIP_BLOCK_X,
                              CAMBI_HIP_BLOCK_Y, 1u, 0u, stream, params, NULL);
    return hip_err(rc);
}

#endif /* HAVE_HIPCC */

/* ------------------------------------------------------------------ */
/* init_fex_hip                                                        */
/* ------------------------------------------------------------------ */
static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    CambiStateHip *s = fex->priv;

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
    if (s->enc_width < CAMBI_HIP_MIN_WIDTH_HEIGHT && s->enc_height < CAMBI_HIP_MIN_WIDTH_HEIGHT) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi_hip: encoded resolution %dx%d below minimum %d×%d.\n",
                 s->enc_width, s->enc_height, CAMBI_HIP_MIN_WIDTH_HEIGHT,
                 CAMBI_HIP_MIN_WIDTH_HEIGHT);
        return -EINVAL;
    }

    s->src_width = w;
    s->src_height = h;
    s->src_bpc = bpc;
    s->proc_width = (unsigned)s->enc_width;
    s->proc_height = (unsigned)s->enc_height;
    s->adjusted_window = cambi_hip_adjust_window(s->window_size, s->proc_width, s->proc_height);

#ifndef HAVE_HIPCC
    (void)s;
    return -ENOSYS;
#else
    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err)
        return err;

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err)
        goto fail_after_ctx;

    /* Pinned readback: one double for the pre-computed score. */
    err = vmaf_hip_kernel_readback_alloc(&s->rb_score, s->ctx, sizeof(double));
    if (err)
        goto fail_after_lc;

    /* Load HSACO and resolve kernel functions. */
    err = cambi_hip_module_load(s);
    if (err)
        goto fail_after_rb;

    /* Device buffers: flat uint16, proc_w × proc_h (scale-0 size). */
    s->d_buf_bytes = (size_t)s->proc_width * s->proc_height * sizeof(uint16_t);
    hipError_t hip_rc = hipMalloc((void **)&s->d_image, s->d_buf_bytes);
    if (hip_rc != hipSuccess) {
        err = hip_err(hip_rc);
        goto fail_after_module;
    }
    hip_rc = hipMalloc((void **)&s->d_mask, s->d_buf_bytes);
    if (hip_rc != hipSuccess) {
        err = hip_err(hip_rc);
        goto fail_after_module;
    }
    hip_rc = hipMalloc((void **)&s->d_tmp, s->d_buf_bytes);
    if (hip_rc != hipSuccess) {
        err = hip_err(hip_rc);
        goto fail_after_module;
    }

    /* Host VmafPictures for the CPU residual. */
    err = vmaf_picture_alloc(&s->pics[0], VMAF_PIX_FMT_YUV400P, 10, s->proc_width, s->proc_height);
    if (err)
        goto fail_after_module;
    err = vmaf_picture_alloc(&s->pics[1], VMAF_PIX_FMT_YUV400P, 10, s->proc_width, s->proc_height);
    if (err)
        goto fail_after_module;

    /* Host scratch buffers for the CPU residual. */
    const int num_diffs = 1 << s->max_log_contrast;
    s->buffers.diffs_to_consider = malloc(sizeof(uint16_t) * (size_t)num_diffs);
    if (!s->buffers.diffs_to_consider) {
        err = -ENOMEM;
        goto fail_after_module;
    }
    s->buffers.diff_weights = malloc(sizeof(int) * (size_t)num_diffs);
    if (!s->buffers.diff_weights) {
        err = -ENOMEM;
        goto fail_after_module;
    }
    s->buffers.all_diffs = malloc(sizeof(int) * (size_t)(2 * num_diffs + 1));
    if (!s->buffers.all_diffs) {
        err = -ENOMEM;
        goto fail_after_module;
    }

    static const int contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                             8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
    for (int d = 0; d < num_diffs; d++) {
        s->buffers.diffs_to_consider[d] = (uint16_t)(d + 1);
        s->buffers.diff_weights[d] = contrast_weights[d];
    }
    for (int d = -num_diffs; d <= num_diffs; d++)
        s->buffers.all_diffs[d + num_diffs] = d;

    s->buffers.tvi_for_diff = malloc(sizeof(uint16_t) * (size_t)num_diffs);
    if (!s->buffers.tvi_for_diff) {
        err = -ENOMEM;
        goto fail_after_module;
    }

    err = cambi_hip_init_tvi(s);
    if (err)
        goto fail_after_module;

    s->buffers.c_values = malloc(sizeof(float) * s->proc_width * s->proc_height);
    if (!s->buffers.c_values) {
        err = -ENOMEM;
        goto fail_after_module;
    }

    const uint16_t num_bins = (uint16_t)(1024u + (unsigned)(s->buffers.all_diffs[2 * num_diffs] -
                                                            s->buffers.all_diffs[0]));
    s->buffers.c_values_histograms = malloc(sizeof(uint16_t) * s->proc_width * (size_t)num_bins);
    if (!s->buffers.c_values_histograms) {
        err = -ENOMEM;
        goto fail_after_module;
    }

    const int pad_size = CAMBI_HIP_MASK_FILTER_SIZE / 2;
    const int dp_width = (int)s->proc_width + 2 * pad_size + 1;
    const int dp_height = 2 * pad_size + 2;
    s->buffers.mask_dp = malloc(sizeof(uint32_t) * (size_t)dp_width * (size_t)dp_height);
    if (!s->buffers.mask_dp) {
        err = -ENOMEM;
        goto fail_after_module;
    }
    s->buffers.filter_mode_buffer = malloc(sizeof(uint16_t) * 3u * s->proc_width);
    if (!s->buffers.filter_mode_buffer) {
        err = -ENOMEM;
        goto fail_after_module;
    }
    s->buffers.derivative_buffer = malloc(sizeof(uint16_t) * s->proc_width);
    if (!s->buffers.derivative_buffer) {
        err = -ENOMEM;
        goto fail_after_module;
    }

    vmaf_cambi_default_callbacks(&s->inc_range_callback, &s->dec_range_callback,
                                 &s->derivative_callback);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict) {
        err = -ENOMEM;
        goto fail_after_module;
    }
    return 0;

fail_after_module:
    if (s->module) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
    cambi_hip_free_device_buffers(s);
    (void)vmaf_picture_unref(&s->pics[0]);
    (void)vmaf_picture_unref(&s->pics[1]);
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
fail_after_rb:
    (void)vmaf_hip_kernel_readback_free(&s->rb_score, s->ctx);
fail_after_lc:
    (void)vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
fail_after_ctx:
    vmaf_hip_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
#endif /* HAVE_HIPCC */
}

/* ------------------------------------------------------------------ */
/* submit_fex_hip                                                      */
/* ------------------------------------------------------------------ */
static int submit_fex_hip(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                          VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;

#ifndef HAVE_HIPCC
    (void)fex;
    (void)dist_pic;
    (void)index;
    return -ENOSYS;
#else
    CambiStateHip *s = fex->priv;
    s->index = index;

    /* Host preprocessing: decimate/upcast dist_pic → pics[0] (10-bit). */
    int err = vmaf_cambi_preprocessing(dist_pic, &s->pics[0], (int)s->proc_width,
                                       (int)s->proc_height, s->enc_bitdepth);
    if (err)
        return err;

    const hipStream_t stream = (hipStream_t)s->lc.str;

    /* HtoD upload pics[0].data[0] → d_image. */
    const size_t row_bytes = s->proc_width * sizeof(uint16_t);
    const uint16_t *src_data = (const uint16_t *)s->pics[0].data[0];
    const ptrdiff_t src_stride_bytes = s->pics[0].stride[0];
    for (unsigned row = 0; row < s->proc_height; row++) {
        const uint8_t *src_row = (const uint8_t *)src_data + (size_t)row * (size_t)src_stride_bytes;
        const hipDeviceptr_t dst_dptr =
            s->d_image + (hipDeviceptr_t)((size_t)row * s->proc_width * sizeof(uint16_t));
        hipError_t hip_rc =
            hipMemcpyHtoDAsync(dst_dptr, (void *)(uintptr_t)src_row, row_bytes, stream);
        if (hip_rc != hipSuccess)
            return hip_err(hip_rc);
    }

    /* GPU spatial mask at full scale. */
    const unsigned mask_index_0 = (unsigned)cambi_hip_get_mask_index(s->proc_width, s->proc_height,
                                                                     CAMBI_HIP_MASK_FILTER_SIZE);
    err = dispatch_mask_hip(s, stream, s->proc_width, s->proc_height, s->proc_width, mask_index_0);
    if (err)
        return err;

    /* Per-scale GPU pipeline + CPU residual. */
    unsigned scaled_w = s->proc_width;
    unsigned scaled_h = s->proc_height;
    const int num_diffs = 1 << s->max_log_contrast;
    double scores_per_scale[CAMBI_HIP_NUM_SCALES] = {0.0, 0.0, 0.0, 0.0, 0.0};
    const double topk = (s->topk != CAMBI_HIP_DEFAULT_TOPK) ? s->topk : s->cambi_topk;

    /* Track d_image / d_mask as locals to allow swapping across scales. */
    hipDeviceptr_t d_img = s->d_image;
    hipDeviceptr_t d_msk = s->d_mask;
    hipDeviceptr_t d_tmp = s->d_tmp;

    for (int scale = 0; scale < CAMBI_HIP_NUM_SCALES; scale++) {
        if (scale > 0) {
            const unsigned new_w = (scaled_w + 1u) >> 1;
            const unsigned new_h = (scaled_h + 1u) >> 1;

            /* GPU decimate d_img → d_tmp. */
            err = dispatch_decimate_hip(s, stream, d_img, d_tmp, new_w, new_h, scaled_w, new_w);
            if (err)
                return err;
            hipDeviceptr_t t = d_img;
            d_img = d_tmp;
            d_tmp = t;

            /* GPU decimate d_msk → d_tmp. */
            err = dispatch_decimate_hip(s, stream, d_msk, d_tmp, new_w, new_h, scaled_w, new_w);
            if (err)
                return err;
            t = d_msk;
            d_msk = d_tmp;
            d_tmp = t;

            scaled_w = new_w;
            scaled_h = new_h;
        }

        /* GPU filter_mode H: d_img → d_tmp. */
        err = dispatch_filter_mode_hip(s, stream, d_img, d_tmp, scaled_w, scaled_h, scaled_w, 0);
        if (err)
            return err;
        /* GPU filter_mode V: d_tmp → d_img. */
        err = dispatch_filter_mode_hip(s, stream, d_tmp, d_img, scaled_w, scaled_h, scaled_w, 1);
        if (err)
            return err;

        /* Sync stream before host DtoH readback. */
        hipError_t hip_rc = hipStreamSynchronize(stream);
        if (hip_rc != hipSuccess)
            return hip_err(hip_rc);

        /* DtoH: d_img / d_msk → pics[0] / pics[1]. */
        const ptrdiff_t pic_stride_bytes = s->pics[0].stride[0];
        uint16_t *dst0 = (uint16_t *)s->pics[0].data[0];
        uint16_t *dst1 = (uint16_t *)s->pics[1].data[0];
        for (unsigned row = 0; row < scaled_h; row++) {
            uint8_t *d0_row = (uint8_t *)dst0 + (size_t)row * (size_t)pic_stride_bytes;
            const hipDeviceptr_t s0_dptr =
                d_img + (hipDeviceptr_t)((size_t)row * scaled_w * sizeof(uint16_t));
            hip_rc = hipMemcpyDtoH(d0_row, s0_dptr, scaled_w * sizeof(uint16_t));
            if (hip_rc != hipSuccess)
                return hip_err(hip_rc);

            uint8_t *d1_row = (uint8_t *)dst1 + (size_t)row * (size_t)pic_stride_bytes;
            const hipDeviceptr_t s1_dptr =
                d_msk + (hipDeviceptr_t)((size_t)row * scaled_w * sizeof(uint16_t));
            hip_rc = hipMemcpyDtoH(d1_row, s1_dptr, scaled_w * sizeof(uint16_t));
            if (hip_rc != hipSuccess)
                return hip_err(hip_rc);
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

    /* Compute final score. */
    const uint16_t pixels_in_window = vmaf_cambi_get_pixels_in_window(s->adjusted_window);
    double score = vmaf_cambi_weight_scores_per_scale(scores_per_scale, pixels_in_window);
    if (score > s->cambi_max_val)
        score = s->cambi_max_val;
    if (score < 0.0)
        score = 0.0;

    /* Stash score in the pinned readback buffer for collect(). */
    *(double *)s->rb_score.host_pinned = score;

    /* Record submit event on stream so collect() can wait. */
    hipError_t hip_rc = hipEventRecord((hipEvent_t)s->lc.submit, stream);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);
    hip_rc = hipStreamWaitEvent(stream, (hipEvent_t)s->lc.submit, 0u);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);
    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
#endif /* HAVE_HIPCC */
}

/* ------------------------------------------------------------------ */
/* collect_fex_hip — emit the pre-computed score. */
/* ------------------------------------------------------------------ */
static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
#ifndef HAVE_HIPCC
    (void)fex;
    (void)index;
    (void)feature_collector;
    return -ENOSYS;
#else
    CambiStateHip *s = fex->priv;

    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err)
        return err;

    const double score = *(double *)s->rb_score.host_pinned;
    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "Cambi_feature_cambi_score", score, index);
#endif /* HAVE_HIPCC */
}

/* ------------------------------------------------------------------ */
/* close_fex_hip */
/* ------------------------------------------------------------------ */
static int close_fex_hip(VmafFeatureExtractor *fex)
{
    CambiStateHip *s = fex->priv;
    int rc = 0;

#ifdef HAVE_HIPCC
    if (s->module) {
        int e = hip_err(hipModuleUnload(s->module));
        s->module = NULL;
        if (rc == 0)
            rc = e;
    }
    cambi_hip_free_device_buffers(s);

    int e = vmaf_hip_kernel_readback_free(&s->rb_score, s->ctx);
    if (rc == 0)
        rc = e;
    e = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
    if (rc == 0)
        rc = e;
#endif /* HAVE_HIPCC */

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
        int e = vmaf_dictionary_free(&s->feature_name_dict);
        if (rc == 0)
            rc = e;
    }
#ifdef HAVE_HIPCC
    if (s->ctx) {
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
#endif /* HAVE_HIPCC */
    return rc;
}

static const char *provided_features[] = {"Cambi_feature_cambi_score", NULL};

/* Load-bearing: declared `extern` in feature_extractor.c's
 * `feature_extractor_list[]` under `#if HAVE_HIP`. Making this static
 * would unlink the extractor from the registry. Same pattern as every
 * HIP extractor (see vmaf_fex_float_psnr_hip). */
// NOLINTNEXTLINE(misc-use-internal-linkage) -- ADR-0254: registry linkage invariant
VmafFeatureExtractor vmaf_fex_cambi_hip = {
    .name = "cambi_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(CambiStateHip),
    .provided_features = provided_features,
    /* VMAF_FEATURE_EXTRACTOR_HIP flag cleared: pictures arrive as CPU
     * VmafPictures (same posture as all other HIP consumers; ADR-0254). */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 15, /* 5 scales × 3 kernels */
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_DIRECT,
        },
};
