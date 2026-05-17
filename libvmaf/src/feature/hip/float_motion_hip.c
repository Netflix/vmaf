/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_motion feature extractor on the HIP backend — seventh
 *  consumer of `libvmaf/src/hip/kernel_template.h` (T7-10b
 *  follow-up / ADR-0273).  Real kernel promotion: T7-10b batch-2 /
 *  ADR-0373.
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/float_motion_cuda.c`
 *  call-graph-for-call-graph: same private-state struct shape, same
 *  init/submit/collect/close lifecycle, same template helper
 *  invocations, same `flush()` host-only post-processing tail, and
 *  the same `motion_force_zero` short-circuit posture.
 *
 *  Temporal design: a `blur[2]` ping-pong of device float arrays holds
 *  the Gaussian-blurred current/previous frames. `ref_in` is a device
 *  buffer for the raw Y-plane copy (used by the kernel to produce
 *  `cur_blur`). On each submit, `compute_sad` is 0 for the first frame
 *  and 1 afterwards. The per-block float SAD partials land in `rb.device`
 *  (sized `wg_count * sizeof(float)`). The host accumulates them in
 *  double, divides by `w*h`, and emits `VMAF_feature_motion_score` at
 *  `index` and `VMAF_feature_motion2_score = min(prev, cur)` at
 *  `index - 1`. The tail motion2 is emitted in `flush()`.
 *
 *  When `HAVE_HIPCC` is defined (enable_hipcc=true at configure time),
 *  the real HIP Module API path is active. Without it the scaffold
 *  posture is preserved: every lifecycle helper returns -ENOSYS.
 *
 *  HIP adaptation notes vs CUDA twin:
 *  - Warp size 64 on GCN/RDNA; the kernel already accounts for this
 *    (FM_WARP_SIZE=64, FM_WARPS_PER_BLOCK=4 for a 16x16 WG).
 *  - Kernel args are raw pointers (no VmafCudaBuffer indirection).
 *  - HtoD copy uses hipMemcpy2DAsync with hipMemcpyHostToDevice because
 *    pictures arrive as CPU VmafPictures (VMAF_FEATURE_EXTRACTOR_HIP
 *    flag not yet set — same posture as all other HIP consumers).
 *  - `blur[0]` and `blur[1]` are plain hipMalloc device buffers (float,
 *    w*h), analogous to the CUDA twin's `VmafCudaBuffer *blur[2]`.
 */

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"
#include "log.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"

#ifdef HAVE_HIPCC
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include "float_motion_hip.h"
#endif /* HAVE_HIPCC */

/* Block dimensions mirror the HIP kernel's FM_BX / FM_BY. */
#define FMH_BX 16u
#define FMH_BY 16u

typedef struct FloatMotionStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and
     * the (device per-WG SAD float partials, pinned host readback
     * slot) pair are managed by `hip/kernel_template.h`. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;

#ifdef HAVE_HIPCC
    /* HSACO module + per-bpc kernel function handles. */
    hipModule_t module;
    hipFunction_t funcbpc8;
    hipFunction_t funcbpc16;
    /* Device-only raw Y-plane staging buffer (HtoD copy each frame). */
    void *ref_in;
    /* Gaussian-blurred frame ping-pong (float, w*h pixels each). */
    void *blur[2];
#endif /* HAVE_HIPCC */

    int cur_blur;
    unsigned wg_count;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    double prev_motion_score;
    double motion_fps_weight;
    bool debug;
    bool motion_force_zero;

    VmafDictionary *feature_name_dict;
} FloatMotionStateHip;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(FloatMotionStateHip, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .help = "force motion score to zero",
        .offset = offsetof(FloatMotionStateHip, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(FloatMotionStateHip, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0},
};

static int extract_force_zero(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                              VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                              VmafPicture *dist_pic_90, unsigned index,
                              VmafFeatureCollector *feature_collector)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    FloatMotionStateHip *s = fex->priv;

    int err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_motion2_score", 0.0, index);
    if (s->debug && err == 0) {
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_motion_score", 0.0, index);
    }
    return err;
}

/* Extracted from init: motion_force_zero short-circuit. */
static int init_force_zero_hip(VmafFeatureExtractor *fex, FloatMotionStateHip *s)
{
    fex->extract = extract_force_zero;
    fex->submit = NULL;
    fex->collect = NULL;
    fex->flush = NULL;
    fex->close = NULL;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        return -ENOMEM;
    }
    return 0;
}

#ifdef HAVE_HIPCC
/* Translate a HIP error code to a negative errno. */
static int fm_hip_rc(hipError_t rc)
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

/* Load the HSACO module and look up the two per-bpc kernel entry points.
 * Called once from init() when HAVE_HIPCC is defined. */
static int fm_hip_module_load(FloatMotionStateHip *s)
{
    hipError_t rc = hipModuleLoadData(&s->module, float_motion_score_hsaco);
    if (rc != hipSuccess)
        return fm_hip_rc(rc);

    rc = hipModuleGetFunction(&s->funcbpc8, s->module, "float_motion_hip_kernel_8bpc");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return fm_hip_rc(rc);
    }
    rc = hipModuleGetFunction(&s->funcbpc16, s->module, "float_motion_hip_kernel_16bpc");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return fm_hip_rc(rc);
    }
    return 0;
}

/* HtoD copy ref luma plane, launch the motion kernel, record events,
 * enqueue DtoH copy of per-block SAD partials. Extracted to keep
 * submit_fex_hip under the 60-line function-size limit.
 *
 * `compute_sad`: 0 for the first frame (no previous blur — partials will
 * all be 0.0 by kernel contract), 1 for subsequent frames. */
static int fm_hip_launch(FloatMotionStateHip *s, VmafPicture *ref_pic, unsigned compute_sad)
{
    const hipStream_t str = (hipStream_t)s->lc.str;
    const hipStream_t pstr = (hipStream_t)0; /* no VmafPicture stream handle yet */

    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t plane_pitch = (ptrdiff_t)(s->frame_w * bpp);

    /* HtoD copy of ref luma plane into tightly-pitched staging buffer. */
    hipError_t rc = hipMemcpy2DAsync(s->ref_in, (size_t)plane_pitch, ref_pic->data[0],
                                     (size_t)ref_pic->stride[0], (size_t)plane_pitch,
                                     (size_t)s->frame_h, hipMemcpyHostToDevice, pstr);
    if (rc != hipSuccess)
        return fm_hip_rc(rc);

    const unsigned gx = (s->frame_w + FMH_BX - 1u) / FMH_BX;
    const unsigned gy = (s->frame_h + FMH_BY - 1u) / FMH_BY;

    const uint8_t *ref_dev = (const uint8_t *)s->ref_in;
    float *cur_blur = (float *)s->blur[s->cur_blur];
    const float *prev_blur = (const float *)s->blur[1 - s->cur_blur];
    float *partials_dev = (float *)s->rb.device;
    unsigned w = s->frame_w;
    unsigned h = s->frame_h;

    hipFunction_t func;
    void *args8[] = {
        (void *)&ref_dev,      (void *)&plane_pitch, (void *)&cur_blur, (void *)&prev_blur,
        (void *)&partials_dev, (void *)&w,           (void *)&h,        (void *)&compute_sad,
    };
    unsigned bpc = s->bpc;
    void *args16[] = {
        (void *)&ref_dev,   (void *)&plane_pitch,  (void *)&cur_blur,
        (void *)&prev_blur, (void *)&partials_dev, (void *)&w,
        (void *)&h,         (void *)&bpc,          (void *)&compute_sad,
    };

    if (s->bpc == 8u) {
        func = s->funcbpc8;
        rc = hipModuleLaunchKernel(func, gx, gy, 1, FMH_BX, FMH_BY, 1, 0, pstr, args8, NULL);
    } else {
        func = s->funcbpc16;
        rc = hipModuleLaunchKernel(func, gx, gy, 1, FMH_BX, FMH_BY, 1, 0, pstr, args16, NULL);
    }
    if (rc != hipSuccess)
        return fm_hip_rc(rc);

    /* Record submit event on picture stream, wait on private stream,
     * DtoH copy of SAD partials, then record finished event. */
    rc = hipEventRecord((hipEvent_t)s->lc.submit, pstr);
    if (rc != hipSuccess)
        return fm_hip_rc(rc);
    rc = hipStreamWaitEvent(str, (hipEvent_t)s->lc.submit, 0);
    if (rc != hipSuccess)
        return fm_hip_rc(rc);
    rc = hipMemcpyAsync(s->rb.host_pinned, s->rb.device, (size_t)s->wg_count * sizeof(float),
                        hipMemcpyDeviceToHost, str);
    if (rc != hipSuccess)
        return fm_hip_rc(rc);

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
}

/* Allocate ref_in staging buffer and blur[0/1] ping-pong.  On failure,
 * any partially-allocated buffers are freed and NULL-ed; caller unwinds
 * via fail_after_module.  Extracted to keep init_fex_hip under the
 * 60-line readability-function-size limit. */
static int fm_hip_bufs_alloc(FloatMotionStateHip *s, unsigned w, unsigned h, unsigned bpc)
{
    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    const size_t plane_bytes = (size_t)w * h * bpp;
    const size_t blur_bytes = (size_t)w * h * sizeof(float);

    hipError_t rc = hipMalloc(&s->ref_in, plane_bytes);
    if (rc != hipSuccess)
        return -ENOMEM;

    rc = hipMalloc(&s->blur[0], blur_bytes);
    if (rc != hipSuccess) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
        return -ENOMEM;
    }
    rc = hipMalloc(&s->blur[1], blur_bytes);
    if (rc != hipSuccess) {
        (void)hipFree(s->blur[0]);
        s->blur[0] = NULL;
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
        return -ENOMEM;
    }
    return 0;
}

/* Release module + device buffers.  Safe to call with NULL handles.
 * Extracted so init_fex_hip error paths stay under 60 lines. */
static void fm_hip_bufs_free(FloatMotionStateHip *s)
{
    if (s->blur[1] != NULL) {
        (void)hipFree(s->blur[1]);
        s->blur[1] = NULL;
    }
    if (s->blur[0] != NULL) {
        (void)hipFree(s->blur[0]);
        s->blur[0] = NULL;
    }
    if (s->ref_in != NULL) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
    }
    if (s->module != NULL) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
}
#endif /* HAVE_HIPCC */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatMotionStateHip *s = fex->priv;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;
    s->index = 0;
    s->prev_motion_score = 0.0;
    s->cur_blur = 0;

    /* The 5-tap HIP float kernel uses reflect-101 mirror padding; fm_mirror()
     * returns 2*sup - idx - 2, which is negative when sup < 3.  Refuse
     * smaller frames up front.  Minimum: filter_width/2 + 1 = 3. */
    if (h < 3u || w < 3u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "float_motion_hip: frame %ux%u is below the 5-tap filter minimum 3x3; "
                 "refusing to avoid out-of-bounds mirror reads on device\n",
                 w, h);
        return -EINVAL;
    }

    const unsigned gx = (w + FMH_BX - 1u) / FMH_BX;
    const unsigned gy = (h + FMH_BY - 1u) / FMH_BY;
    s->wg_count = gx * gy;

    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_after_ctx;

    if (s->motion_force_zero) {
        err = init_force_zero_hip(fex, s);
        if (err != 0)
            goto fail_after_lc;
        return 0;
    }

    /* Readback pair: device per-WG float SAD partials + pinned host slot. */
    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx, (size_t)s->wg_count * sizeof(float));
    if (err != 0)
        goto fail_after_lc;

#ifdef HAVE_HIPCC
    err = fm_hip_module_load(s);
    if (err != 0)
        goto fail_after_rb;

    /* Staging buffer (ref_in) and blurred-frame ping-pong (blur[0/1]).
     * fm_hip_bufs_alloc frees partial allocations on failure. */
    err = fm_hip_bufs_alloc(s, w, h, bpc);
    if (err != 0)
        goto fail_after_module;
#endif /* HAVE_HIPCC */

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
#ifdef HAVE_HIPCC
        fm_hip_bufs_free(s); /* also unloads module via fm_hip_bufs_free */
        goto fail_after_rb;
#else
        goto fail_after_rb;
#endif
    }

    return 0;

#ifdef HAVE_HIPCC
fail_after_module:
    if (s->module != NULL) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
#endif /* HAVE_HIPCC */
fail_after_rb:
    (void)vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
fail_after_lc:
    (void)vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
fail_after_ctx:
    vmaf_hip_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static int submit_fex_hip(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                          VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)dist_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    FloatMotionStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

#ifdef HAVE_HIPCC
    /* First frame has no previous blurred frame — kernel writes cur_blur
     * but computes no SAD (compute_sad=0, partials all 0.0 by contract). */
    const unsigned compute_sad = (index > 0u) ? 1u : 0u;
    return fm_hip_launch(s, ref_pic, compute_sad);
#else
    /* Scaffold posture: surface -ENOSYS via the pre-launch helper so the
     * feature engine sees "runtime not ready". */
    int err = vmaf_hip_kernel_submit_pre_launch(&s->lc, s->ctx, &s->rb,
                                                /* picture_stream */ 0,
                                                /* dist_ready_event */ 0);
    if (err != 0)
        return err;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    FloatMotionStateHip *s = fex->priv;

    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

#ifdef HAVE_HIPCC
    /* Accumulate per-block float SAD partials in double and compute the
     * per-frame motion score. Mirrors the CUDA twin's cross-block
     * reduction precision posture. */
    const float *partials = (const float *)s->rb.host_pinned;
    double total_sad = 0.0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total_sad += (double)partials[i];

    const double n_pixels = (double)s->frame_w * (double)s->frame_h;
    const double motion_score = total_sad / n_pixels;

    /* Advance blur ping-pong. */
    s->cur_blur = 1 - s->cur_blur;

    if (index == 0u) {
        /* First frame: no previous, emit 0 for both scores. The CUDA
         * twin defers motion2 for the first frame to the next collect;
         * here we match that behaviour by emitting 0 directly. */
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_motion2_score", 0.0, index);
        if (s->debug && err == 0) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_feature_motion_score", 0.0, index);
        }
        s->prev_motion_score = 0.0;
        return err;
    }

    if (index == 1u) {
        /* Second frame: emit motion_score only (debug); skip motion2 at
         * index=0 — it was already written by the index=0 branch above.
         * Mirrors the CUDA twin's `index == 1` guard in collect_fex_cuda. */
        if (s->debug) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_feature_motion_score", motion_score,
                                                          index);
        }
        s->prev_motion_score = motion_score;
        return err;
    }

    /* index >= 2: emit motion2_score = min(prev, cur) at index-1, then
     * (debug) motion_score at current index. Same order as CUDA twin.
     * Apply fps weight to both operands before the min so the weight
     * scales the motion2 output; identity when motion_fps_weight = 1.0. */
    const double w_cur = motion_score * s->motion_fps_weight;
    const double w_prev = s->prev_motion_score * s->motion_fps_weight;
    const double motion2 = (w_cur < w_prev) ? w_cur : w_prev;
    err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "VMAF_feature_motion2_score", motion2, index - 1u);
    if (s->debug && err == 0) {
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_motion_score", motion_score,
                                                      index);
    }
    s->prev_motion_score = motion_score;
    return err;
#else
    (void)feature_collector;
    (void)index;
    /* Advance ping-pong even in scaffold so state stays consistent. */
    s->cur_blur = 1 - s->cur_blur;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int flush_fex_hip(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
#ifndef HAVE_HIPCC
    (void)fex;
    (void)feature_collector;
    /* Scaffold: no scores collected; return 1 ("done") to avoid an
     * infinite flush loop in the feature engine. */
    return 1;
#else
    FloatMotionStateHip *s = fex->priv;

    if (s->index == 0u) {
        /* Zero or one frame processed — no tail motion2 to emit. */
        return 1;
    }

    /* Emit the tail motion2 = prev_motion_score * fps_weight at the last
     * frame index.  Mirrors the CUDA twin's flush_fex_cuda shape exactly;
     * identity when motion_fps_weight = 1.0. */
    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "VMAF_feature_motion2_score",
        s->prev_motion_score * s->motion_fps_weight, s->index);
    return (err != 0) ? err : 1;
#endif /* HAVE_HIPCC */
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    FloatMotionStateHip *s = fex->priv;

    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);

#ifdef HAVE_HIPCC
    /* fm_hip_bufs_free also unloads the module; best-effort only.
     * No separate error surface here — mirrors the CUDA twin pattern
     * of treating module/buffer teardown as best-effort in close(). */
    fm_hip_bufs_free(s);
#endif /* HAVE_HIPCC */

    int err = vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
    if (err != 0 && rc == 0)
        rc = err;
    if (s->feature_name_dict != NULL) {
        err = vmaf_dictionary_free(&s->feature_name_dict);
        if (err != 0 && rc == 0)
            rc = err;
    }
    if (s->ctx != NULL) {
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {"VMAF_feature_motion_score", "VMAF_feature_motion2_score",
                                          NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_float_motion_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_float_motion_cuda` in
 * `libvmaf/src/feature/cuda/float_motion_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_motion_hip = {
    .name = "float_motion_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .flush = flush_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(FloatMotionStateHip),
    .provided_features = provided_features,
    /* TEMPORAL flag is mandatory: float_motion needs the previous-frame
     * blur carry, so the feature engine must drive collect before the
     * next submit. Mirrors the CUDA twin verbatim.
     *
     * VMAF_FEATURE_EXTRACTOR_HIP flag is intentionally absent until
     * picture buffer-type plumbing lands (T7-10c). Until then pictures
     * arrive as CPU VmafPictures and fm_hip_launch() does explicit
     * HtoD copies. Same posture as all prior HIP consumers. */
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
