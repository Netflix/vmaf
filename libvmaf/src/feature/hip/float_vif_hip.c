/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_vif feature extractor on the HIP backend — ninth
 *  kernel-template consumer (T7-10b batch-5 / ADR-0379).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/float_vif_cuda.c`
 *  call-graph-for-call-graph: same private-state struct shape, same
 *  init/submit/collect/close lifecycle, same provided_features list,
 *  and the same `vif_kernelscale == 1.0` validation gate.
 *
 *  Per-frame flow: 4 compute + 3 decimate kernel launches, all on a
 *  single stream so they serialise naturally.  Per-block (num, den)
 *  float partials are copied to pinned host memory.  The host
 *  accumulates them in double.
 *
 *  When `HAVE_HIPCC` is defined (enable_hipcc=true at configure time),
 *  the real HIP Module API path is active.  Without it the scaffold
 *  posture is preserved: every lifecycle helper returns -ENOSYS.
 *
 *  HIP adaptation notes vs CUDA twin:
 *  - Warp size 64 on GCN/RDNA; the kernel accounts for this via
 *    FVIF_WARP_SIZE=64 and FVIF_WARPS_PER_BLOCK=4 for a 16x16 WG.
 *  - Kernel args are raw pointers (no VmafCudaBuffer indirection).
 *  - HtoD copy uses hipMemcpy2DAsync with hipMemcpyHostToDevice because
 *    pictures arrive as CPU VmafPictures (VMAF_FEATURE_EXTRACTOR_HIP
 *    flag not yet set — same posture as all other HIP consumers).
 *  - Device-only buffers (ref_raw, dis_raw, ref_buf[], dis_buf[],
 *    num_partials[], den_partials[]) are plain hipMalloc allocations.
 *  - Pinned host readback for (num, den) partials uses hipHostMalloc,
 *    mirroring the CUDA twin's vmaf_cuda_buffer_host_alloc pattern.
 */

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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
#include "float_vif_hip.h"
#endif /* HAVE_HIPCC */

#define FVIF_BX 16u
#define FVIF_BY 16u

typedef struct FloatVifStateHip {
    VmafHipKernelLifecycle lc;
    VmafHipContext *ctx;

    bool debug;
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    double vif_sigma_nsq;

#ifdef HAVE_HIPCC
    hipModule_t module;
    hipFunction_t func_compute;
    hipFunction_t func_decimate;

    /* Staging buffers (raw pixel planes, HtoD each frame). */
    void *ref_raw;
    void *dis_raw;
    /* Intermediate float buffers — ping-pong across scales 1-3. */
    void *ref_buf[2];
    void *dis_buf[2];
    /* Per-block (num, den) partial sums — one slot per workgroup per scale. */
    void *num_partials[4];
    void *den_partials[4];
    /* Pinned host readback for partials. */
    float *num_host[4];
    float *den_host[4];
#endif /* HAVE_HIPCC */

    unsigned wg_count[4];
    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned scale_w[4];
    unsigned scale_h[4];

    VmafDictionary *feature_name_dict;
} FloatVifStateHip;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(FloatVifStateHip, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif (>= 1.0)",
        .offset = offsetof(FloatVifStateHip, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 100.0,
        .min = 1.0,
        .max = 100.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_kernelscale",
        .help = "scaling factor for the gaussian kernel",
        .offset = offsetof(FloatVifStateHip, vif_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_sigma_nsq",
        .alias = "snsq",
        .help = "neural noise variance",
        .offset = offsetof(FloatVifStateHip, vif_sigma_nsq),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 2.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0},
};

static void compute_per_scale_dims(FloatVifStateHip *s)
{
    s->scale_w[0] = s->width;
    s->scale_h[0] = s->height;
    for (int i = 1; i < 4; i++) {
        s->scale_w[i] = s->scale_w[i - 1] / 2u;
        s->scale_h[i] = s->scale_h[i - 1] / 2u;
    }
}

#ifdef HAVE_HIPCC
/* Translate a HIP error code to a negative errno. */
static int fvif_hip_rc(hipError_t rc)
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

/* Load the HSACO module and look up the two kernel entry points. */
static int fvif_hip_module_load(FloatVifStateHip *s)
{
    hipError_t rc = hipModuleLoadData(&s->module, float_vif_score_hsaco);
    if (rc != hipSuccess)
        return fvif_hip_rc(rc);

    rc = hipModuleGetFunction(&s->func_compute, s->module, "float_vif_compute");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return fvif_hip_rc(rc);
    }
    rc = hipModuleGetFunction(&s->func_decimate, s->module, "float_vif_decimate");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return fvif_hip_rc(rc);
    }
    return 0;
}

/* Allocate all device + pinned-host buffers.  On failure, any partially-
 * allocated buffers are freed and NULL-ed; caller unwinds via
 * fail_after_module.  Extracted to keep init_fex_hip readable. */
static int fvif_hip_bufs_alloc(FloatVifStateHip *s)
{
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const size_t raw_bytes = (size_t)s->width * s->height * bpp;

    if (hipMalloc(&s->ref_raw, raw_bytes) != hipSuccess)
        return -ENOMEM;
    if (hipMalloc(&s->dis_raw, raw_bytes) != hipSuccess) {
        (void)hipFree(s->ref_raw);
        s->ref_raw = NULL;
        return -ENOMEM;
    }

    /* Intermediate float buffers: scale_w[1]*scale_h[1] floats each.
     * Ping-pong: even scales use ref_buf[0]/dis_buf[0], odd use [1]. */
    const size_t fbytes = (size_t)s->scale_w[1] * s->scale_h[1] * sizeof(float);
    for (int i = 0; i < 2; i++) {
        if (hipMalloc(&s->ref_buf[i], fbytes) != hipSuccess ||
            hipMalloc(&s->dis_buf[i], fbytes) != hipSuccess) {
            for (int j = i; j >= 0; j--) {
                if (s->ref_buf[j] != NULL) {
                    (void)hipFree(s->ref_buf[j]);
                    s->ref_buf[j] = NULL;
                }
                if (s->dis_buf[j] != NULL) {
                    (void)hipFree(s->dis_buf[j]);
                    s->dis_buf[j] = NULL;
                }
            }
            (void)hipFree(s->dis_raw);
            s->dis_raw = NULL;
            (void)hipFree(s->ref_raw);
            s->ref_raw = NULL;
            return -ENOMEM;
        }
    }

    /* Per-scale (num, den) partials + pinned host readback. */
    for (int i = 0; i < 4; i++) {
        const size_t pbytes = (size_t)s->wg_count[i] * sizeof(float);
        bool ok = (hipMalloc(&s->num_partials[i], pbytes) == hipSuccess) &&
                  (hipMalloc(&s->den_partials[i], pbytes) == hipSuccess) &&
                  (hipHostMalloc(&s->num_host[i], pbytes, 0) == hipSuccess) &&
                  (hipHostMalloc(&s->den_host[i], pbytes, 0) == hipSuccess);
        if (!ok) {
            /* Free this scale's partially-allocated slots. */
            if (s->den_host[i] != NULL) {
                (void)hipHostFree(s->den_host[i]);
                s->den_host[i] = NULL;
            }
            if (s->num_host[i] != NULL) {
                (void)hipHostFree(s->num_host[i]);
                s->num_host[i] = NULL;
            }
            if (s->den_partials[i] != NULL) {
                (void)hipFree(s->den_partials[i]);
                s->den_partials[i] = NULL;
            }
            if (s->num_partials[i] != NULL) {
                (void)hipFree(s->num_partials[i]);
                s->num_partials[i] = NULL;
            }
            /* Free prior scales. */
            for (int j = i - 1; j >= 0; j--) {
                (void)hipHostFree(s->den_host[j]);
                s->den_host[j] = NULL;
                (void)hipHostFree(s->num_host[j]);
                s->num_host[j] = NULL;
                (void)hipFree(s->den_partials[j]);
                s->den_partials[j] = NULL;
                (void)hipFree(s->num_partials[j]);
                s->num_partials[j] = NULL;
            }
            for (int k = 1; k >= 0; k--) {
                (void)hipFree(s->dis_buf[k]);
                s->dis_buf[k] = NULL;
                (void)hipFree(s->ref_buf[k]);
                s->ref_buf[k] = NULL;
            }
            (void)hipFree(s->dis_raw);
            s->dis_raw = NULL;
            (void)hipFree(s->ref_raw);
            s->ref_raw = NULL;
            return -ENOMEM;
        }
    }
    return 0;
}

/* Release all device buffers + pinned host slabs + HSACO module.
 * Safe to call with NULL handles. */
static void fvif_hip_bufs_free(FloatVifStateHip *s)
{
    for (int i = 3; i >= 0; i--) {
        if (s->den_host[i] != NULL) {
            (void)hipHostFree(s->den_host[i]);
            s->den_host[i] = NULL;
        }
        if (s->num_host[i] != NULL) {
            (void)hipHostFree(s->num_host[i]);
            s->num_host[i] = NULL;
        }
        if (s->den_partials[i] != NULL) {
            (void)hipFree(s->den_partials[i]);
            s->den_partials[i] = NULL;
        }
        if (s->num_partials[i] != NULL) {
            (void)hipFree(s->num_partials[i]);
            s->num_partials[i] = NULL;
        }
    }
    for (int i = 1; i >= 0; i--) {
        if (s->dis_buf[i] != NULL) {
            (void)hipFree(s->dis_buf[i]);
            s->dis_buf[i] = NULL;
        }
        if (s->ref_buf[i] != NULL) {
            (void)hipFree(s->ref_buf[i]);
            s->ref_buf[i] = NULL;
        }
    }
    if (s->dis_raw != NULL) {
        (void)hipFree(s->dis_raw);
        s->dis_raw = NULL;
    }
    if (s->ref_raw != NULL) {
        (void)hipFree(s->ref_raw);
        s->ref_raw = NULL;
    }
    if (s->module != NULL) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
}

/* Reset partial buffers for one scale, then launch float_vif_compute.
 * Extracted to keep submit_fex_hip under the 60-line limit. */
static int fvif_launch_compute(FloatVifStateHip *s, hipStream_t str, int scale, void *ref_raw_d,
                               void *dis_raw_d, ptrdiff_t raw_stride, void *ref_f_d, void *dis_f_d)
{
    const size_t pbytes = (size_t)s->wg_count[scale] * sizeof(float);
    hipError_t rc = hipMemsetAsync(s->num_partials[scale], 0, pbytes, str);
    if (rc != hipSuccess)
        return fvif_hip_rc(rc);
    rc = hipMemsetAsync(s->den_partials[scale], 0, pbytes, str);
    if (rc != hipSuccess)
        return fvif_hip_rc(rc);

    unsigned w = s->scale_w[scale];
    unsigned h = s->scale_h[scale];
    ptrdiff_t f_stride = (ptrdiff_t)s->scale_w[scale];
    ptrdiff_t nf_stride = 0;
    unsigned gx = (w + FVIF_BX - 1u) / FVIF_BX;
    unsigned gy = (h + FVIF_BY - 1u) / FVIF_BY;
    void *num_d = s->num_partials[scale];
    void *den_d = s->den_partials[scale];
    const bool is_raw = (scale == 0);
    const float *ref_fp = is_raw ? NULL : (const float *)ref_f_d;
    const float *dis_fp = is_raw ? NULL : (const float *)dis_f_d;
    ptrdiff_t used_stride = is_raw ? nf_stride : f_stride;

    void *args[] = {
        (void *)&scale,  (void *)&ref_raw_d, (void *)&dis_raw_d,   (void *)&raw_stride,
        (void *)&ref_fp, (void *)&dis_fp,    (void *)&used_stride, (void *)&num_d,
        (void *)&den_d,  (void *)&w,         (void *)&h,           (void *)&s->bpc,
        (void *)&gx,
    };
    rc = hipModuleLaunchKernel(s->func_compute, gx, gy, 1, FVIF_BX, FVIF_BY, 1, 0, str, args, NULL);
    return fvif_hip_rc(rc);
}

/* Launch float_vif_decimate at `next_scale` then compute.
 * Extracted to keep submit_fex_hip readable. */
static int fvif_launch_decimate_and_compute(FloatVifStateHip *s, hipStream_t str, int next_scale,
                                            void *ref_raw_d, void *dis_raw_d, ptrdiff_t raw_stride)
{
    const int dst_idx = (next_scale - 1) % 2;
    const bool prev_raw = (next_scale == 1);
    void *ref_in = prev_raw ? ref_raw_d : (dst_idx == 0 ? s->ref_buf[1] : s->ref_buf[0]);
    void *dis_in = prev_raw ? dis_raw_d : (dst_idx == 0 ? s->dis_buf[1] : s->dis_buf[0]);
    void *ref_out = (dst_idx == 0) ? s->ref_buf[0] : s->ref_buf[1];
    void *dis_out = (dst_idx == 0) ? s->dis_buf[0] : s->dis_buf[1];
    ptrdiff_t in_f = prev_raw ? 0 : (ptrdiff_t)s->scale_w[next_scale - 1];
    ptrdiff_t out_f = (ptrdiff_t)s->scale_w[next_scale];
    unsigned out_w = s->scale_w[next_scale];
    unsigned out_h = s->scale_h[next_scale];
    unsigned in_w = s->scale_w[next_scale - 1];
    unsigned in_h = s->scale_h[next_scale - 1];
    unsigned dgx = (out_w + FVIF_BX - 1u) / FVIF_BX;
    unsigned dgy = (out_h + FVIF_BY - 1u) / FVIF_BY;

    void *dargs[] = {
        (void *)&next_scale, (void *)&ref_raw_d, (void *)&dis_raw_d, (void *)&raw_stride,
        (void *)&ref_in,     (void *)&dis_in,    (void *)&in_f,      (void *)&ref_out,
        (void *)&dis_out,    (void *)&out_f,     (void *)&out_w,     (void *)&out_h,
        (void *)&in_w,       (void *)&in_h,      (void *)&s->bpc,
    };
    hipError_t rc = hipModuleLaunchKernel(s->func_decimate, dgx, dgy, 1, FVIF_BX, FVIF_BY, 1, 0,
                                          str, dargs, NULL);
    if (rc != hipSuccess)
        return fvif_hip_rc(rc);

    return fvif_launch_compute(s, str, next_scale, ref_raw_d, dis_raw_d, raw_stride, ref_out,
                               dis_out);
}
#endif /* HAVE_HIPCC */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatVifStateHip *s = fex->priv;

    /* Only kernelscale=1.0 is implemented (mirrors CUDA twin). */
    if (s->vif_kernelscale != 1.0)
        return -EINVAL;

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    compute_per_scale_dims(s);

    for (int i = 0; i < 4; i++) {
        const unsigned gx = (s->scale_w[i] + FVIF_BX - 1u) / FVIF_BX;
        const unsigned gy = (s->scale_h[i] + FVIF_BY - 1u) / FVIF_BY;
        s->wg_count[i] = gx * gy;
    }

    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_after_ctx;

#ifdef HAVE_HIPCC
    err = fvif_hip_module_load(s);
    if (err != 0)
        goto fail_after_lc;

    err = fvif_hip_bufs_alloc(s);
    if (err != 0)
        goto fail_after_module;
#endif /* HAVE_HIPCC */

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
#ifdef HAVE_HIPCC
        fvif_hip_bufs_free(s);
        goto fail_after_lc;
#else
        goto fail_after_lc;
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
    (void)dist_pic_90;
    (void)index;
    FloatVifStateHip *s = fex->priv;

#ifdef HAVE_HIPCC
    const hipStream_t pic_stream = (hipStream_t)0;
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t raw_stride = (ptrdiff_t)(s->width * bpp);

    /* HtoD copy of ref and dist luma planes into tightly-pitched staging. */
    hipError_t rc = hipMemcpy2DAsync(s->ref_raw, (size_t)raw_stride, ref_pic->data[0],
                                     (size_t)ref_pic->stride[0], (size_t)raw_stride,
                                     (size_t)s->height, hipMemcpyHostToDevice, pic_stream);
    if (rc != hipSuccess)
        return fvif_hip_rc(rc);
    rc = hipMemcpy2DAsync(s->dis_raw, (size_t)raw_stride, dist_pic->data[0],
                          (size_t)dist_pic->stride[0], (size_t)raw_stride, (size_t)s->height,
                          hipMemcpyHostToDevice, pic_stream);
    if (rc != hipSuccess)
        return fvif_hip_rc(rc);

    /* Launch: compute at scale 0, then decimate + compute for scales 1-3. */
    int err = fvif_launch_compute(s, pic_stream, 0, s->ref_raw, s->dis_raw, raw_stride, NULL, NULL);
    if (err != 0)
        return err;

    for (int ns = 1; ns < 4; ns++) {
        err =
            fvif_launch_decimate_and_compute(s, pic_stream, ns, s->ref_raw, s->dis_raw, raw_stride);
        if (err != 0)
            return err;
    }

    /* Bridge to private stream: record submit event, wait, DtoH partials. */
    const hipStream_t str = (hipStream_t)s->lc.str;
    rc = hipEventRecord((hipEvent_t)s->lc.submit, pic_stream);
    if (rc != hipSuccess)
        return fvif_hip_rc(rc);
    rc = hipStreamWaitEvent(str, (hipEvent_t)s->lc.submit, 0);
    if (rc != hipSuccess)
        return fvif_hip_rc(rc);

    for (int i = 0; i < 4; i++) {
        const size_t pbytes = (size_t)s->wg_count[i] * sizeof(float);
        rc = hipMemcpyAsync(s->num_host[i], s->num_partials[i], pbytes, hipMemcpyDeviceToHost, str);
        if (rc != hipSuccess)
            return fvif_hip_rc(rc);
        rc = hipMemcpyAsync(s->den_host[i], s->den_partials[i], pbytes, hipMemcpyDeviceToHost, str);
        if (rc != hipSuccess)
            return fvif_hip_rc(rc);
    }

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
#else
    /* Scaffold posture: surface -ENOSYS. */
    (void)ref_pic;
    (void)dist_pic;
    int err = vmaf_hip_kernel_submit_pre_launch(&s->lc, s->ctx, NULL, 0, 0);
    if (err != 0)
        return err;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    FloatVifStateHip *s = fex->priv;

    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

#ifdef HAVE_HIPCC
    double scores[8];
    for (int i = 0; i < 4; i++) {
        double n = 0.0, d = 0.0;
        for (unsigned j = 0; j < s->wg_count[i]; j++) {
            n += (double)s->num_host[i][j];
            d += (double)s->den_host[i][j];
        }
        scores[2 * i + 0] = n;
        scores[2 * i + 1] = d;
    }

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale0_score",
                                                   scores[0] / scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale1_score",
                                                   scores[2] / scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale2_score",
                                                   scores[4] / scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_feature_vif_scale3_score",
                                                   scores[6] / scores[7], index);

    if (s->debug && !err) {
        double score_num = scores[0] + scores[2] + scores[4] + scores[6];
        double score_den = scores[1] + scores[3] + scores[5] + scores[7];
        double score = (score_den == 0.0) ? 1.0 : score_num / score_den;
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif", score, index);
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif_num", score_num, index);
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "vif_den", score_den, index);
        const char *names[8] = {
            "vif_num_scale0", "vif_den_scale0", "vif_num_scale1", "vif_den_scale1",
            "vif_num_scale2", "vif_den_scale2", "vif_num_scale3", "vif_den_scale3",
        };
        for (int i = 0; i < 8; i++) {
            err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                           names[i], scores[i], index);
        }
    }

    return err;
#else
    (void)feature_collector;
    (void)index;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    FloatVifStateHip *s = fex->priv;

    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);

#ifdef HAVE_HIPCC
    fvif_hip_bufs_free(s);
#endif /* HAVE_HIPCC */

    if (s->feature_name_dict != NULL) {
        int err = vmaf_dictionary_free(&s->feature_name_dict);
        if (err != 0 && rc == 0)
            rc = err;
    }
    if (s->ctx != NULL) {
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {
    "VMAF_feature_vif_scale0_score",
    "VMAF_feature_vif_scale1_score",
    "VMAF_feature_vif_scale2_score",
    "VMAF_feature_vif_scale3_score",
    "vif",
    "vif_num",
    "vif_den",
    "vif_num_scale0",
    "vif_den_scale0",
    "vif_num_scale1",
    "vif_den_scale1",
    "vif_num_scale2",
    "vif_den_scale2",
    "vif_num_scale3",
    "vif_den_scale3",
    NULL,
};

/* Load-bearing: registered via extern in feature_extractor.c's
 * feature_extractor_list[].  Making this static would unlink the
 * extractor from the registry — same rule as every other HIP consumer
 * (e.g. vmaf_fex_float_motion_hip). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_vif_hip = {
    .name = "float_vif_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(FloatVifStateHip),
    .provided_features = provided_features,
    /* No TEMPORAL flag: VIF is stateless across frames.
     * VMAF_FEATURE_EXTRACTOR_HIP is intentionally absent until picture
     * buffer-type plumbing lands (T7-10c); pictures arrive as CPU
     * VmafPictures and submit does explicit HtoD copies. */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
