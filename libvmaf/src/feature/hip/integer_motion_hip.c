/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_motion feature extractor on the HIP backend.
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/integer_motion_cuda.c`
 *  call-graph-for-call-graph. When `HAVE_HIPCC` is defined the real HIP
 *  Module API path is active: module load, ping-pong uint16 blurred-frame
 *  buffers (`blur[2]`) and a uint64 SAD accumulator via `hipMalloc`,
 *  per-frame HtoD copy + `hipModuleLaunchKernel`, and a host-side
 *  flush() computing motion2/motion3 scores.
 *  Without `HAVE_HIPCC` the scaffold posture is preserved (-ENOSYS).
 *
 *  Provided features (mirrors CUDA twin):
 *    VMAF_integer_feature_motion_score   (debug only)
 *    VMAF_integer_feature_motion2_score
 *    VMAF_integer_feature_motion3_score
 *
 *  Temporal design: `blur[2]` ping-pong of uint16 device buffers holds
 *  the Gaussian-blurred current/previous frames. The kernel writes the
 *  current blurred frame into `blur[index%2]` and diffs against
 *  `blur[(index+1)%2]`. The raw Y-plane is transferred HtoD directly
 *  from the CPU VmafPicture (VMAF_FEATURE_EXTRACTOR_HIP flag not set yet
 *  — same posture as all other HIP consumers, ADR-0241).
 *
 *  HIP adaptation notes vs CUDA twin:
 *  - Warp size 64 on GCN/RDNA; warp reduction in kernel uses `__shfl_down`.
 *  - Kernel args are raw pointers (no VmafCudaBuffer indirection).
 *  - `blur[0]` and `blur[1]` are plain hipMalloc device buffers (uint16,
 *    w*h bytes each), analogous to the CUDA twin's `VmafCudaBuffer *blur[2]`.
 *  - SAD accumulator is a plain uint64_t hipMalloc device buffer.
 *  - motion3 post-processing and motion_blend helpers are host-only scalar
 *    work, identical to the CUDA twin.
 *
 *  Kernel entry points (loaded from embedded HSACO `motion_score_hsaco`):
 *    calculate_motion_score_kernel_8bpc
 *    calculate_motion_score_kernel_16bpc
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"
#include "log.h"
#include "motion_blend_tools.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"
#include "integer_motion_hip.h"

#ifdef HAVE_HIPCC
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#endif /* HAVE_HIPCC */

/* Default upper clamp on motion/motion2/motion3 — mirrors
 * MOTION_CUDA_DEFAULT_MAX_VAL in integer_motion_cuda.c / ADR-0219. */
#define MOTION_HIP_DEFAULT_MAX_VAL (10000.0)

/* Block dimensions mirror the HIP kernel's MS_BLOCK_X / MS_BLOCK_Y. */
#define MSH_BX 16u
#define MSH_BY 16u

typedef struct MotionStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device uint64 SAD accumulator, pinned host readback slot) pair
     * are managed by `hip/kernel_template.h`. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;

#ifdef HAVE_HIPCC
    hipModule_t module;
    hipFunction_t funcbpc8;
    hipFunction_t funcbpc16;
    /* Ping-pong of Gaussian-blurred Y planes on device (uint16, w*h each).
     * blur[index%2] is written by the current kernel dispatch;
     * blur[(index+1)%2] is the previous frame's blurred plane. */
    void *blur[2];
    /* Raw ref Y-plane staging buffer (HtoD copy each frame, uint8 or uint16). */
    void *ref_in;
#endif /* HAVE_HIPCC */

    size_t plane_bytes;   /* bytes for one Y plane (bpc-aware) */
    size_t blurred_bytes; /* bytes for one uint16 blurred plane (w*h*2) */
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    double score;                /* most recent motion_score (raw, normalized) */
    double prev_motion3_blended; /* for motion3 moving-average carry */
    unsigned frame_index;        /* count of frames processed */

    bool debug;
    bool motion_force_zero;
    bool motion_five_frame_window;
    bool motion_moving_average;
    double motion_blend_factor;
    double motion_blend_offset;
    double motion_fps_weight;
    double motion_max_val;

    VmafDictionary *feature_name_dict;
} MotionStateHip;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionStateHip, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .help = "forcing motion score to zero",
        .offset = offsetof(MotionStateHip, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_factor",
        .alias = "mbf",
        .help = "blend motion score given an offset",
        .offset = offsetof(MotionStateHip, motion_blend_factor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_offset",
        .alias = "mbo",
        .help = "blend motion score starting from this offset",
        .offset = offsetof(MotionStateHip, motion_blend_offset),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 40.0,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(MotionStateHip, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_max_val",
        .alias = "mmxv",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(MotionStateHip, motion_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = MOTION_HIP_DEFAULT_MAX_VAL,
        .min = 0.0,
        .max = 10000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_five_frame_window",
        .alias = "mffw",
        .help = "use five-frame temporal window (NOT YET SUPPORTED on HIP — deferred)",
        .offset = offsetof(MotionStateHip, motion_five_frame_window),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_moving_average",
        .alias = "mma",
        .help = "use moving average for motion3 scores after first frame",
        .offset = offsetof(MotionStateHip, motion_moving_average),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}};

/* ------------------------------------------------------------------ */
/* motion3 host post-processing — mirrors integer_motion_cuda.c.      */
/* ------------------------------------------------------------------ */
static double motion3_postprocess_hip(MotionStateHip *s, double score2)
{
    const double weighted = score2 * s->motion_fps_weight;
    const double blended = motion_blend(weighted, s->motion_blend_factor, s->motion_blend_offset);
    const double clipped = blended < s->motion_max_val ? blended : s->motion_max_val;
    const double prev_una = s->prev_motion3_blended;
    s->prev_motion3_blended = clipped;
    /* frame_index is pre-incremented in collect() before this runs.
     * Guard matches the CUDA twin: frame_index > 2 corresponds to
     * CPU's index > 1 (cuda-reviewer 2026-05-09). */
    if (s->motion_moving_average && s->frame_index > 2u) {
        return (clipped + prev_una) / 2.0;
    }
    return clipped;
}

static double normalize_and_scale_sad(uint64_t sad, unsigned w, unsigned h)
{
    return (double)(sad / 256.) / ((double)w * (double)h);
}

/* Idempotent append — suppresses duplicate-write warning when flush's
 * collect already wrote the same (feature, index) pair. */
static int append_if_unwritten(VmafFeatureCollector *fc, const char *feature, double value,
                               unsigned index)
{
    double existing;
    if (vmaf_feature_collector_get_score(fc, feature, &existing, index) == 0)
        return 0;
    return vmaf_feature_collector_append(fc, feature, value, index);
}

static int extract_force_zero(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                              VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                              VmafPicture *dist_pic_90, unsigned index,
                              VmafFeatureCollector *feature_collector)
{
    MotionStateHip *s = fex->priv;
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;

    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "VMAF_integer_feature_motion2_score", 0., index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_motion3_score", 0., index);
    if (!s->debug)
        return err;
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_motion_score", 0., index);
    return err;
}

#ifdef HAVE_HIPCC

/* Translate a HIP error code to a negative errno. */
static int msh_rc(hipError_t rc)
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

/* Load HSACO module and resolve both kernel entry points. */
static int msh_module_load(MotionStateHip *s)
{
    hipError_t rc = hipModuleLoadData(&s->module, motion_score_hsaco);
    if (rc != hipSuccess)
        return msh_rc(rc);

    rc = hipModuleGetFunction(&s->funcbpc8, s->module, "calculate_motion_score_kernel_8bpc");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return msh_rc(rc);
    }
    rc = hipModuleGetFunction(&s->funcbpc16, s->module, "calculate_motion_score_kernel_16bpc");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return msh_rc(rc);
    }
    return 0;
}

/* Allocate blurred ping-pong buffers + raw Y-plane staging buffer.
 * On partial failure, frees whatever was allocated. */
static int msh_bufs_alloc(MotionStateHip *s)
{
    hipError_t rc = hipMalloc(&s->blur[0], s->blurred_bytes);
    if (rc != hipSuccess)
        return -ENOMEM;
    rc = hipMalloc(&s->blur[1], s->blurred_bytes);
    if (rc != hipSuccess) {
        (void)hipFree(s->blur[0]);
        s->blur[0] = NULL;
        return -ENOMEM;
    }
    rc = hipMalloc(&s->ref_in, s->plane_bytes);
    if (rc != hipSuccess) {
        (void)hipFree(s->blur[1]);
        s->blur[1] = NULL;
        (void)hipFree(s->blur[0]);
        s->blur[0] = NULL;
        return -ENOMEM;
    }
    return 0;
}

/* Free all device buffers and unload the module. Safe with NULL. */
static void msh_bufs_free(MotionStateHip *s)
{
    if (s->ref_in != NULL) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
    }
    if (s->blur[1] != NULL) {
        (void)hipFree(s->blur[1]);
        s->blur[1] = NULL;
    }
    if (s->blur[0] != NULL) {
        (void)hipFree(s->blur[0]);
        s->blur[0] = NULL;
    }
    if (s->module != NULL) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
}

/* Per-frame kernel dispatch: HtoD copy, memset SAD, kernel launch,
 * DtoH enqueue. Extracted to keep submit_fex_hip under 60 lines. */
static int msh_launch(MotionStateHip *s, VmafPicture *ref_pic, unsigned index)
{
    hipStream_t str = (hipStream_t)s->lc.str;

    const unsigned cur_idx = index % 2u;
    const unsigned prev_idx = (index + 1u) % 2u;
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t src_pitch = (ptrdiff_t)(s->frame_w * bpp);
    /* Blurred buffer stride in bytes: w * sizeof(uint16_t). */
    const ptrdiff_t blurred_stride = (ptrdiff_t)(s->frame_w * sizeof(uint16_t));

    /* HtoD copy of current ref Y plane into staging buffer. */
    hipError_t rc =
        hipMemcpy2DAsync(s->ref_in, (size_t)src_pitch, ref_pic->data[0], (size_t)ref_pic->stride[0],
                         (size_t)src_pitch, (size_t)s->frame_h, hipMemcpyHostToDevice, str);
    if (rc != hipSuccess)
        return msh_rc(rc);

    /* Frame 0: kernel writes blur[0] but no SAD (no prev frame).
     * Still need to launch so blur[0] is populated for frame 1. */
    rc = hipMemsetAsync(s->rb.device, 0, sizeof(uint64_t), str);
    if (rc != hipSuccess)
        return msh_rc(rc);

    const unsigned gx = (s->frame_w + MSH_BX - 1u) / MSH_BX;
    const unsigned gy = (s->frame_h + MSH_BY - 1u) / MSH_BY;

    uint8_t *src_dev = (uint8_t *)s->ref_in;
    uint16_t *cur_blurred_dev = (uint16_t *)s->blur[cur_idx];
    uint16_t *prev_blurred_dev = (uint16_t *)s->blur[prev_idx];
    uint64_t *sad_dev = (uint64_t *)s->rb.device;
    unsigned w = s->frame_w;
    unsigned h = s->frame_h;
    unsigned bpc = s->bpc;

    if (s->bpc <= 8u) {
        void *args[] = {
            (void *)&src_dev,
            (void *)&cur_blurred_dev,
            (void *)&prev_blurred_dev,
            (void *)&sad_dev,
            (void *)&w,
            (void *)&h,
            (void *)&src_pitch,
            (void *)&blurred_stride,
        };
        rc = hipModuleLaunchKernel(s->funcbpc8, gx, gy, 1, MSH_BX, MSH_BY, 1, 0, str, args, NULL);
    } else {
        void *args[] = {
            (void *)&src_dev,
            (void *)&cur_blurred_dev,
            (void *)&prev_blurred_dev,
            (void *)&sad_dev,
            (void *)&w,
            (void *)&h,
            (void *)&src_pitch,
            (void *)&blurred_stride,
            (void *)&bpc,
        };
        rc = hipModuleLaunchKernel(s->funcbpc16, gx, gy, 1, MSH_BX, MSH_BY, 1, 0, str, args, NULL);
    }
    if (rc != hipSuccess)
        return msh_rc(rc);

    rc = hipEventRecord((hipEvent_t)s->lc.submit, str);
    if (rc != hipSuccess)
        return msh_rc(rc);

    if (index == 0u) {
        /* Frame 0: SAD accumulator is meaningless; skip readback. */
        return 0;
    }

    rc = hipMemcpyAsync(s->rb.host_pinned, s->rb.device, sizeof(uint64_t), hipMemcpyDeviceToHost,
                        str);
    if (rc != hipSuccess)
        return msh_rc(rc);

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
}

#endif /* HAVE_HIPCC */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    MotionStateHip *s = fex->priv;

    /* Reject 5-frame window: same as CUDA twin (ADR-0219). */
    if (s->motion_five_frame_window) {
        return -ENOTSUP;
    }

    /* The 5-tap kernel uses reflect-101 mirror; 2*sup - idx - 2 is
     * negative when sup < 3. Refuse smaller frames. */
    if (h < 3u || w < 3u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "motion_hip: frame %ux%u is below the 5-tap filter minimum 3x3; "
                 "refusing to avoid out-of-bounds mirror reads on device\n",
                 w, h);
        return -EINVAL;
    }

    s->frame_w = w;
    s->frame_h = h;
    s->bpc = bpc;
    s->plane_bytes = (size_t)w * h * (bpc <= 8u ? 1u : 2u);
    s->blurred_bytes = (size_t)w * h * sizeof(uint16_t);
    s->score = 0.0;
    s->frame_index = 0;
    s->prev_motion3_blended = 0.0;

    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_after_ctx;

    /* Readback pair: single uint64_t SAD accumulator + pinned host slot. */
    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx, sizeof(uint64_t));
    if (err != 0)
        goto fail_after_lc;

#ifdef HAVE_HIPCC
    err = msh_module_load(s);
    if (err != 0)
        goto fail_after_rb;

    err = msh_bufs_alloc(s);
    if (err != 0)
        goto fail_after_module;
#endif /* HAVE_HIPCC */

    if (s->motion_force_zero) {
        fex->extract = extract_force_zero;
        fex->submit = NULL;
        fex->collect = NULL;
        fex->flush = NULL;
        fex->close = NULL;
        return 0;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
#ifdef HAVE_HIPCC
        msh_bufs_free(s);
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
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    MotionStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

#ifdef HAVE_HIPCC
    return msh_launch(s, ref_pic, index);
#else
    return -ENOSYS;
#endif
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    MotionStateHip *s = fex->priv;

    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

#ifndef HAVE_HIPCC
    (void)feature_collector;
    (void)index;
    return -ENOSYS;
#else
    if (index == 0u) {
        int e = vmaf_feature_collector_append(feature_collector,
                                              "VMAF_integer_feature_motion2_score", 0., 0);
        if (s->debug) {
            e |= vmaf_feature_collector_append(feature_collector,
                                               "VMAF_integer_feature_motion_score", 0., 0);
        }
        s->frame_index++;
        return e;
    }

    double score_prev = s->score;
    const uint64_t *sad_host = (const uint64_t *)s->rb.host_pinned;
    s->score = normalize_and_scale_sad(*sad_host, s->frame_w, s->frame_h);
    s->frame_index++;

    int e = 0;
    if (s->debug) {
        e |= vmaf_feature_collector_append(feature_collector, "VMAF_integer_feature_motion_score",
                                           s->score, index);
    }

    /* Mirror integer_motion_cuda.c collect logic exactly. */
    if (index == 1u) {
        const double score_clipped = s->score * s->motion_fps_weight < s->motion_max_val ?
                                         s->score * s->motion_fps_weight :
                                         s->motion_max_val;
        const double motion3_score = motion3_postprocess_hip(s, score_clipped);
        e |= vmaf_feature_collector_append(feature_collector, "VMAF_integer_feature_motion3_score",
                                           motion3_score, index - 1u);
    }

    if (index > 1u) {
        const double motion2_raw = score_prev < s->score ? score_prev : s->score;
        const double motion2_clipped = motion2_raw * s->motion_fps_weight < s->motion_max_val ?
                                           motion2_raw * s->motion_fps_weight :
                                           s->motion_max_val;
        e |= vmaf_feature_collector_append(feature_collector, "VMAF_integer_feature_motion2_score",
                                           motion2_clipped, index - 1u);
        const double motion3_score = motion3_postprocess_hip(s, motion2_clipped);
        e |= vmaf_feature_collector_append(feature_collector, "VMAF_integer_feature_motion3_score",
                                           motion3_score, index - 1u);
    }

    return e;
#endif /* HAVE_HIPCC */
}

static int flush_fex_hip(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
#ifndef HAVE_HIPCC
    (void)fex;
    (void)feature_collector;
    return 1;
#else
    MotionStateHip *s = fex->priv;
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

    if (s->index > 0u) {
        const double last_motion2 = s->score * s->motion_fps_weight < s->motion_max_val ?
                                        s->score * s->motion_fps_weight :
                                        s->motion_max_val;
        err = append_if_unwritten(feature_collector, "VMAF_integer_feature_motion2_score",
                                  last_motion2, s->index);
        if (err >= 0) {
            const double motion3_score = motion3_postprocess_hip(s, last_motion2);
            const int e3 = append_if_unwritten(
                feature_collector, "VMAF_integer_feature_motion3_score", motion3_score, s->index);
            if (e3 < 0)
                err = e3;
        }
    }

    return (err < 0) ? err : 1;
#endif /* HAVE_HIPCC */
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    MotionStateHip *s = fex->priv;

    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);

#ifdef HAVE_HIPCC
    /* msh_bufs_free also unloads the module. */
    msh_bufs_free(s);
#endif

    const int err_rb = vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
    if (err_rb != 0 && rc == 0)
        rc = err_rb;

    if (s->feature_name_dict != NULL) {
        const int err_dict = vmaf_dictionary_free(&s->feature_name_dict);
        if (err_dict != 0 && rc == 0)
            rc = err_dict;
    }
    if (s->ctx != NULL) {
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {"VMAF_integer_feature_motion_score",
                                          "VMAF_integer_feature_motion2_score",
                                          "VMAF_integer_feature_motion3_score", NULL};

/*
 * Load-bearing: registered via `extern VmafFeatureExtractor
 * vmaf_fex_integer_motion_hip;` in `feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern as every CUDA / SYCL / Vulkan / Metal feature extractor
 * (e.g. `vmaf_fex_integer_motion_cuda` in
 * `feature/cuda/integer_motion_cuda.c`).
 */
// NOLINTNEXTLINE(misc-use-internal-linkage) -- ADR-0278 registry pattern
VmafFeatureExtractor vmaf_fex_integer_motion_hip = {
    .name = "motion_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .flush = flush_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(MotionStateHip),
    .provided_features = provided_features,
    /* TEMPORAL flag is mandatory: motion needs the previous-frame
     * carry, so the feature engine drives collect before the next submit.
     * Mirrors the CUDA twin verbatim.
     *
     * No VMAF_FEATURE_EXTRACTOR_HIP flag yet — pictures arrive as CPU
     * VmafPictures and msh_launch() does explicit HtoD copies. Same
     * posture as all prior HIP consumers (ADR-0241 through ADR-0377). */
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
