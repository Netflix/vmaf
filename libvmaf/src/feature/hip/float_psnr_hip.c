/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_psnr feature extractor on the HIP backend — first real kernel
 *  (T7-10b / ADR-0254). Second kernel-template consumer.
 *  enable_chroma option ported from CUDA twin (ADR-0469).
 *
 *  Mirrors `libvmaf/src/feature/cuda/float_psnr_cuda.c` call-graph-for-
 *  call-graph. The device-side kernel lives in
 *  `libvmaf/src/feature/hip/float_psnr/float_psnr_score.hip` and is
 *  loaded at init() time via the HIP module API (`hipModuleLoadData` +
 *  `hipModuleGetFunction`), the direct analog of CUDA's
 *  `cuModuleLoadData` + `cuModuleGetFunction` used by the twin.
 *
 *  When `enable_hipcc=false` (e.g. a CI agent without ROCm), `HAVE_HIPCC`
 *  is undefined and `init()` returns -ENOSYS — same scaffold contract as
 *  the pre-runtime posture (registered, runtime not ready).
 *
 *  Algorithm (mirrors CUDA twin):
 *    - Per-pixel float (ref - dis)^2 reduction, one partial per block.
 *    - Host accumulates partials in double:
 *        noise = sum / (w * h)
 *        score = 10 * log10(peak^2 / max(noise, 1e-10)), clamped.
 *
 *  Bit-exactness posture: float arithmetic + host double log10, no SIMD
 *  or FMA. Per ADR-0138/0139, scores may differ by 1-2 ULP vs CUDA in
 *  the partial accumulation but are identical at the host log10 step.
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"
#include "float_psnr_hip.h"

/* ------------------------------------------------------------------ */
/* HIP-to-errno translation                                            */
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
/* Private state                                                       */
/* ------------------------------------------------------------------ */

#define FPSNR_BX 16
#define FPSNR_BY 16

typedef struct FloatPsnrStateHip {
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    /* Device-side staging buffers (luma planes, ref + dis). */
    void *ref_in;
    void *dis_in;
    VmafHipContext *ctx;
    /* HIP module + per-bpc kernel function handles. */
    hipModule_t module;
    hipFunction_t funcbpc8;
    hipFunction_t funcbpc16;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    unsigned wg_count;
    double peak;
    double psnr_max;
    /* `enable_chroma` option: when false, only luma is computed.
     * Default true mirrors CPU float_psnr.c — see ADR-0469. */
    bool enable_chroma;
    VmafDictionary *feature_name_dict;
} FloatPsnrStateHip;

static const VmafOption options[] = {{
                                         .name = "enable_chroma",
                                         .help = "enable calculation for chroma channels",
                                         .offset = offsetof(FloatPsnrStateHip, enable_chroma),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = true,
                                     },
                                     {0}};

/* Bit-depth → peak / clamp table (mirrors the CUDA twin). Extracted to
 * keep init_fex_hip under the 60-line readability-function-size limit. */
static int float_psnr_hip_resolve_peak_clamp(FloatPsnrStateHip *s, unsigned bpc)
{
    if (bpc == 8u) {
        s->peak = 255.0;
        s->psnr_max = 60.0;
    } else if (bpc == 10u) {
        s->peak = 255.75;
        s->psnr_max = 72.0;
    } else if (bpc == 12u) {
        s->peak = 255.9375;
        s->psnr_max = 84.0;
    } else if (bpc == 16u) {
        s->peak = 255.99609375;
        s->psnr_max = 108.0;
    } else {
        return -EINVAL;
    }
    return 0;
}

#ifdef HAVE_HIPCC
/*
 * Load the HSACO fat binary, resolve both kernel function handles, and
 * allocate luma-plane staging buffers. Extracted to keep init_fex_hip
 * under the 60-line readability-function-size limit.
 *
 * On failure: `s->module`, `s->ref_in`, `s->dis_in` are left NULL /
 * unset; caller unwinds via fail_after_rb.
 */
static int float_psnr_hip_module_load(FloatPsnrStateHip *s, unsigned bpc, unsigned w, unsigned h)
{
    hipError_t hip_rc = hipModuleLoadData(&s->module, float_psnr_score_hsaco);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);

    hip_rc = hipModuleGetFunction(&s->funcbpc8, s->module, "float_psnr_kernel_8bpc");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return hip_err(hip_rc);
    }
    hip_rc = hipModuleGetFunction(&s->funcbpc16, s->module, "float_psnr_kernel_16bpc");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return hip_err(hip_rc);
    }

    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    const size_t plane_bytes = (size_t)w * h * bpp;
    hip_rc = hipMalloc(&s->ref_in, plane_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return hip_err(hip_rc);
    }
    hip_rc = hipMalloc(&s->dis_in, plane_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return hip_err(hip_rc);
    }
    return 0;
}

/*
 * Per-frame submit body: zero accumulator, HtoD copies, kernel launch,
 * submit-event record, DtoH copy. Extracted to keep submit_fex_hip
 * under the 60-line readability-function-size limit.
 */
static int float_psnr_hip_launch(FloatPsnrStateHip *s, VmafPicture *ref_pic, VmafPicture *dist_pic)
{
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t plane_pitch = (ptrdiff_t)(s->frame_w * bpp);
    const hipStream_t str = (hipStream_t)s->lc.str;

    hipError_t hip_rc = hipMemsetAsync(s->rb.device, 0, (size_t)s->wg_count * sizeof(float), str);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);

    hip_rc = hipMemcpy2DAsync(s->ref_in, (size_t)plane_pitch, ref_pic->data[0],
                              (size_t)ref_pic->stride[0], (size_t)plane_pitch, (size_t)s->frame_h,
                              hipMemcpyHostToDevice, str);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);

    hip_rc = hipMemcpy2DAsync(s->dis_in, (size_t)plane_pitch, dist_pic->data[0],
                              (size_t)dist_pic->stride[0], (size_t)plane_pitch, (size_t)s->frame_h,
                              hipMemcpyHostToDevice, str);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);

    const unsigned gx = (s->frame_w + FPSNR_BX - 1u) / FPSNR_BX;
    const unsigned gy = (s->frame_h + FPSNR_BY - 1u) / FPSNR_BY;
    if (s->bpc == 8u) {
        void *args[] = {&s->ref_in,           &s->dis_in,   (void *)&plane_pitch,
                        (void *)&plane_pitch, s->rb.device, (void *)&s->frame_w,
                        (void *)&s->frame_h};
        hip_rc = hipModuleLaunchKernel(s->funcbpc8, gx, gy, 1, FPSNR_BX, FPSNR_BY, 1, 0, str, args,
                                       NULL);
    } else {
        void *args[] = {&s->ref_in,           &s->dis_in,     (void *)&plane_pitch,
                        (void *)&plane_pitch, s->rb.device,   (void *)&s->frame_w,
                        (void *)&s->frame_h,  (void *)&s->bpc};
        hip_rc = hipModuleLaunchKernel(s->funcbpc16, gx, gy, 1, FPSNR_BX, FPSNR_BY, 1, 0, str, args,
                                       NULL);
    }
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);

    /* Record submit event, DtoH copy of partials, record finished event. */
    hip_rc = hipEventRecord((hipEvent_t)s->lc.submit, str);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);

    hip_rc = hipMemcpyAsync(s->rb.host_pinned, s->rb.device, (size_t)s->wg_count * sizeof(float),
                            hipMemcpyDeviceToHost, str);
    if (hip_rc != hipSuccess)
        return hip_err(hip_rc);

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
}
#endif /* HAVE_HIPCC */

/* Release module + staging buffers. Safe to call with NULL handles.
 * Extracted so init_fex_hip's error paths stay under 60 lines. */
static void float_psnr_hip_module_free(FloatPsnrStateHip *s)
{
#ifdef HAVE_HIPCC
    if (s->dis_in != NULL) {
        (void)hipFree(s->dis_in);
        s->dis_in = NULL;
    }
    if (s->ref_in != NULL) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
    }
    if (s->module != NULL) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
#else
    (void)s;
#endif /* HAVE_HIPCC */
}

/* ------------------------------------------------------------------ */
/* init / close                                                        */
/* ------------------------------------------------------------------ */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    FloatPsnrStateHip *s = fex->priv;
    /* float_psnr operates on luma only; chroma planes are not used.
     * `enable_chroma` is wired here so callers can set it (ADR-0469)
     * and the option is not silently dropped. Suppress unused-param. */
    (void)pix_fmt;

    int err = float_psnr_hip_resolve_peak_clamp(s, bpc);
    if (err != 0)
        return err;

    s->bpc = bpc;
    s->frame_w = w;
    s->frame_h = h;
    s->wg_count = ((w + FPSNR_BX - 1u) / FPSNR_BX) * ((h + FPSNR_BY - 1u) / FPSNR_BY);

    err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_after_ctx;

    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx, (size_t)s->wg_count * sizeof(float));
    if (err != 0)
        goto fail_after_lc;

#ifdef HAVE_HIPCC
    err = float_psnr_hip_module_load(s, bpc, w, h);
#else
    err = -ENOSYS;
#endif
    if (err != 0)
        goto fail_after_rb;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
        float_psnr_hip_module_free(s);
        goto fail_after_rb;
    }
    return 0;

fail_after_rb:
    (void)vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
fail_after_lc:
    (void)vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
fail_after_ctx:
    vmaf_hip_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    FloatPsnrStateHip *s = fex->priv;
    int rc = 0;

#ifdef HAVE_HIPCC
    if (s->dis_in != NULL) {
        int e = hip_err(hipFree(s->dis_in));
        s->dis_in = NULL;
        if (rc == 0)
            rc = e;
    }
    if (s->ref_in != NULL) {
        int e = hip_err(hipFree(s->ref_in));
        s->ref_in = NULL;
        if (rc == 0)
            rc = e;
    }
    if (s->module != NULL) {
        int e = hip_err(hipModuleUnload(s->module));
        s->module = NULL;
        if (rc == 0)
            rc = e;
    }
#endif /* HAVE_HIPCC */

    int e = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
    if (rc == 0)
        rc = e;
    e = vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
    if (rc == 0)
        rc = e;
    if (s->feature_name_dict != NULL) {
        e = vmaf_dictionary_free(&s->feature_name_dict);
        if (rc == 0)
            rc = e;
    }
    if (s->ctx != NULL) {
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

/* ------------------------------------------------------------------ */
/* submit / collect                                                    */
/* ------------------------------------------------------------------ */

static int submit_fex_hip(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                          VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    (void)index;

#ifndef HAVE_HIPCC
    (void)fex;
    (void)ref_pic;
    (void)dist_pic;
    return -ENOSYS;
#else
    FloatPsnrStateHip *s = fex->priv;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    /* VMAF_FEATURE_EXTRACTOR_HIP flag is not yet set (T7-10b posture),
     * so pictures arrive as CPU VmafPictures. float_psnr_hip_launch()
     * copies the luma planes host->device, launches the kernel, copies
     * partials device->host, and records the finished event. */
    return float_psnr_hip_launch(s, ref_pic, dist_pic);
#endif /* HAVE_HIPCC */
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
#ifndef HAVE_HIPCC
    (void)fex;
    (void)index;
    (void)feature_collector;
    return -ENOSYS;
#else
    FloatPsnrStateHip *s = fex->priv;

    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

    const float *partials = (const float *)s->rb.host_pinned;
    double total = 0.0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total += (double)partials[i];

    const double n_pix = (double)s->frame_w * (double)s->frame_h;
    const double noise = total / n_pix;
    const double eps = 1e-10;
    const double max_noise = noise > eps ? noise : eps;
    double score = 10.0 * log10(s->peak * s->peak / max_noise);
    if (score > s->psnr_max)
        score = s->psnr_max;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "float_psnr", score, index);
#endif /* HAVE_HIPCC */
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"float_psnr", NULL};

/* Load-bearing: declared `extern` in feature_extractor.c's
 * `feature_extractor_list[]` under `#if HAVE_HIP`. Making this static
 * would unlink the extractor from the registry. Same pattern as every
 * CUDA / SYCL / Vulkan extractor (see vmaf_fex_float_psnr_cuda). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_psnr_hip = {
    .name = "float_psnr_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(FloatPsnrStateHip),
    .provided_features = provided_features,
    /* VMAF_FEATURE_EXTRACTOR_HIP flag cleared until picture buffer-type
     * plumbing lands (T7-10c). Until then pictures arrive as CPU
     * VmafPictures and float_psnr_hip_launch() does explicit HtoD copies.
     * Same posture as all other HIP consumers (ADR-0241 / ADR-0254). */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
