/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ciede2000 feature extractor on the HIP backend — canonical TU,
 *  named `integer_ciede_hip.c` to match the CUDA twin convention
 *  (`integer_ciede_cuda.c`). T7-10b follow-up / ADR-0259; real
 *  kernel promotion: T7-10b batch-4 / ADR-0377.
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/integer_ciede_cuda.c`
 *  call-graph-for-call-graph. When `HAVE_HIPCC` is defined the real HIP
 *  Module API path is active: `hipModuleLoadData` + `hipModuleGetFunction`
 *  + per-frame HtoD copies of all 6 YUV planes + `hipModuleLaunchKernel`.
 *  Without `HAVE_HIPCC` the scaffold posture is preserved.
 *
 *  The ciede kernel writes one float per block (no atomic accumulator),
 *  so the template's memset pre-launch is intentionally bypassed here —
 *  same decision as the CUDA twin's inlined pre-launch wait (ADR-0259).
 *
 *  Bit-exactness: float per-pixel arithmetic + host double log10, no SIMD
 *  or FMA. Per ADR-0138/0139, 1-2 ULP differences from CUDA in the partial
 *  accumulation step are permissible; the host log10 step is identical.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"
#include "ciede_hip.h"

#ifdef HAVE_HIPCC
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#endif /* HAVE_HIPCC */

typedef struct CiedeStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device per-block float partials, pinned host readback slot)
     * pair are managed by `hip/kernel_template.h` (T7-10b third
     * consumer / ADR-0259). */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;
    unsigned partials_capacity;
    unsigned partials_count;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    unsigned ss_hor;
    unsigned ss_ver;

#ifdef HAVE_HIPCC
    hipModule_t module;
    hipFunction_t funcbpc8;
    hipFunction_t funcbpc16;
    /* Staging device buffers for all 6 YUV planes (ref + dis, Y/U/V).
     * Chroma planes are sized at chroma width/height which may be
     * half of luma when subsampled. */
    void *ref_y;
    void *ref_u;
    void *ref_v;
    void *dis_y;
    void *dis_u;
    void *dis_v;
    /* Dimensions of the chroma plane staging buffers. */
    unsigned chroma_w;
    unsigned chroma_h;
#endif /* HAVE_HIPCC */

    VmafDictionary *feature_name_dict;
} CiedeStateHip;

/* Mirrors the CUDA twin's 16x16 workgroup tile. */
#define CIEDE_HIP_BX 16
#define CIEDE_HIP_BY 16

static const VmafOption options[] = {{0}};

#ifdef HAVE_HIPCC
/* Translate a HIP error code to a negative errno. */
static int ciede_hip_rc(hipError_t rc)
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

/* Load HSACO module and resolve the two kernel entry points. */
static int ciede_hip_module_load(CiedeStateHip *s)
{
    hipError_t rc = hipModuleLoadData(&s->module, ciede_score_hsaco);
    if (rc != hipSuccess)
        return ciede_hip_rc(rc);

    rc = hipModuleGetFunction(&s->funcbpc8, s->module, "calculate_ciede_kernel_8bpc");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return ciede_hip_rc(rc);
    }
    rc = hipModuleGetFunction(&s->funcbpc16, s->module, "calculate_ciede_kernel_16bpc");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return ciede_hip_rc(rc);
    }
    return 0;
}

/* Allocate the 6 YUV staging device buffers. On partial failure, already-
 * allocated buffers are freed and NULLed before returning an error. */
static int ciede_hip_bufs_alloc(CiedeStateHip *s, unsigned w, unsigned h, unsigned bpc,
                                unsigned ss_hor, unsigned ss_ver)
{
    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    const size_t luma_bytes = (size_t)w * h * bpp;
    s->chroma_w = ss_hor ? (w >> 1) : w;
    s->chroma_h = ss_ver ? (h >> 1) : h;
    const size_t chroma_bytes = (size_t)s->chroma_w * s->chroma_h * bpp;

    void **bufs[6] = {&s->ref_y, &s->ref_u, &s->ref_v, &s->dis_y, &s->dis_u, &s->dis_v};
    size_t sizes[6] = {luma_bytes, chroma_bytes, chroma_bytes,
                       luma_bytes, chroma_bytes, chroma_bytes};
    for (int i = 0; i < 6; i++) {
        hipError_t rc = hipMalloc(bufs[i], sizes[i]);
        if (rc != hipSuccess) {
            for (int j = i - 1; j >= 0; j--) {
                (void)hipFree(*bufs[j]);
                *bufs[j] = NULL;
            }
            return -ENOMEM;
        }
    }
    return 0;
}

/* Free all 6 YUV staging device buffers and unload the module. Safe to
 * call with NULL handles. */
static void ciede_hip_bufs_free(CiedeStateHip *s)
{
    void **bufs[6] = {&s->dis_v, &s->dis_u, &s->dis_y, &s->ref_v, &s->ref_u, &s->ref_y};
    for (int i = 0; i < 6; i++) {
        if (*bufs[i] != NULL) {
            (void)hipFree(*bufs[i]);
            *bufs[i] = NULL;
        }
    }
    if (s->module != NULL) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
}

/* HtoD copy one plane (packed-pitch staging). */
static int ciede_hip_copy_plane(void *dst, const uint8_t *src, ptrdiff_t src_stride, unsigned pw,
                                unsigned ph, unsigned bpc, hipStream_t str)
{
    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    const size_t plane_pitch = (size_t)pw * bpp;
    hipError_t rc = hipMemcpy2DAsync(dst, plane_pitch, src, (size_t)src_stride, plane_pitch,
                                     (size_t)ph, hipMemcpyHostToDevice, str);
    return (rc == hipSuccess) ? 0 : -EIO;
}

/* Launch the appropriate bpc kernel. Extracted to keep submit under 60 lines. */
static int ciede_hip_launch(CiedeStateHip *s, hipStream_t str)
{
    const unsigned gx = (s->frame_w + CIEDE_HIP_BX - 1u) / CIEDE_HIP_BX;
    const unsigned gy = (s->frame_h + CIEDE_HIP_BY - 1u) / CIEDE_HIP_BY;
    float *partials_dev = (float *)s->rb.device;
    unsigned w = s->frame_w;
    unsigned h = s->frame_h;
    unsigned bpc = s->bpc;
    unsigned ss_hor = s->ss_hor;
    unsigned ss_ver = s->ss_ver;

    /* Strides are tightly packed in the staging buffers. */
    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    ptrdiff_t luma_stride = (ptrdiff_t)(s->frame_w * bpp);
    ptrdiff_t chroma_stride = (ptrdiff_t)(s->chroma_w * bpp);
    uint8_t *ry = (uint8_t *)s->ref_y;
    uint8_t *ru = (uint8_t *)s->ref_u;
    uint8_t *rv = (uint8_t *)s->ref_v;
    uint8_t *dy = (uint8_t *)s->dis_y;
    uint8_t *du = (uint8_t *)s->dis_u;
    uint8_t *dv = (uint8_t *)s->dis_v;

    hipFunction_t func = (bpc == 8u) ? s->funcbpc8 : s->funcbpc16;
    void *args[] = {(void *)&ry,
                    (void *)&luma_stride,
                    (void *)&ru,
                    (void *)&chroma_stride,
                    (void *)&rv,
                    (void *)&chroma_stride,
                    (void *)&dy,
                    (void *)&luma_stride,
                    (void *)&du,
                    (void *)&chroma_stride,
                    (void *)&dv,
                    (void *)&chroma_stride,
                    (void *)&partials_dev,
                    (void *)&w,
                    (void *)&h,
                    (void *)&bpc,
                    (void *)&ss_hor,
                    (void *)&ss_ver};
    hipError_t rc =
        hipModuleLaunchKernel(func, gx, gy, 1, CIEDE_HIP_BX, CIEDE_HIP_BY, 1, 0, str, args, NULL);
    return (rc == hipSuccess) ? 0 : ciede_hip_rc(rc);
}

/* Submit: HtoD copies of all 6 YUV planes, kernel launch, event/DtoH. */
static int ciede_hip_do_submit(CiedeStateHip *s, VmafPicture *ref_pic, VmafPicture *dist_pic)
{
    hipStream_t str = (hipStream_t)s->lc.str;

    int err = ciede_hip_copy_plane(s->ref_y, ref_pic->data[0], ref_pic->stride[0], s->frame_w,
                                   s->frame_h, s->bpc, str);
    if (err)
        return err;
    err = ciede_hip_copy_plane(s->ref_u, ref_pic->data[1], ref_pic->stride[1], s->chroma_w,
                               s->chroma_h, s->bpc, str);
    if (err)
        return err;
    err = ciede_hip_copy_plane(s->ref_v, ref_pic->data[2], ref_pic->stride[2], s->chroma_w,
                               s->chroma_h, s->bpc, str);
    if (err)
        return err;
    err = ciede_hip_copy_plane(s->dis_y, dist_pic->data[0], dist_pic->stride[0], s->frame_w,
                               s->frame_h, s->bpc, str);
    if (err)
        return err;
    err = ciede_hip_copy_plane(s->dis_u, dist_pic->data[1], dist_pic->stride[1], s->chroma_w,
                               s->chroma_h, s->bpc, str);
    if (err)
        return err;
    err = ciede_hip_copy_plane(s->dis_v, dist_pic->data[2], dist_pic->stride[2], s->chroma_w,
                               s->chroma_h, s->bpc, str);
    if (err)
        return err;

    err = ciede_hip_launch(s, str);
    if (err)
        return err;

    hipError_t rc = hipEventRecord((hipEvent_t)s->lc.submit, str);
    if (rc != hipSuccess)
        return ciede_hip_rc(rc);

    rc = hipMemcpyAsync(s->rb.host_pinned, s->rb.device, (size_t)s->partials_count * sizeof(float),
                        hipMemcpyDeviceToHost, str);
    if (rc != hipSuccess)
        return ciede_hip_rc(rc);

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
}
#endif /* HAVE_HIPCC */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        return -EINVAL;
    }
    CiedeStateHip *s = fex->priv;

    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0) {
        return err;
    }

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) {
        goto fail_after_ctx;
    }

    s->bpc = bpc;
    s->ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P) ? 1u : 0u;
    s->ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P) ? 1u : 0u;

    const unsigned grid_x = (w + (CIEDE_HIP_BX - 1u)) / CIEDE_HIP_BX;
    const unsigned grid_y = (h + (CIEDE_HIP_BY - 1u)) / CIEDE_HIP_BY;
    s->partials_capacity = grid_x * grid_y;

    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx,
                                         (size_t)s->partials_capacity * sizeof(float));
    if (err != 0) {
        goto fail_after_lc;
    }

#ifdef HAVE_HIPCC
    err = ciede_hip_module_load(s);
    if (err != 0) {
        goto fail_after_rb;
    }

    err = ciede_hip_bufs_alloc(s, w, h, bpc, s->ss_hor, s->ss_ver);
    if (err != 0) {
        goto fail_after_module;
    }
#endif /* HAVE_HIPCC */

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
#ifdef HAVE_HIPCC
        ciede_hip_bufs_free(s);
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
    (void)dist_pic_90;
    CiedeStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    const unsigned grid_x = (s->frame_w + (CIEDE_HIP_BX - 1u)) / CIEDE_HIP_BX;
    const unsigned grid_y = (s->frame_h + (CIEDE_HIP_BY - 1u)) / CIEDE_HIP_BY;
    s->partials_count = grid_x * grid_y;

#ifdef HAVE_HIPCC
    return ciede_hip_do_submit(s, ref_pic, dist_pic);
#else
    (void)dist_pic;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    CiedeStateHip *s = fex->priv;

    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0) {
        return err;
    }

#ifdef HAVE_HIPCC
    /* Per-block partials -> host accumulation in double. Same precision
     * argument as ciede_vulkan (ADR-0187): per-block sums fit in float7
     * precision; cross-block reduction across thousands of partials needs
     * double to retain places=4. */
    const float *partials_host = s->rb.host_pinned;
    double total = 0.0;
    for (unsigned i = 0; i < s->partials_count; i++)
        total += (double)partials_host[i];
    const double n_pixels = (double)s->frame_w * (double)s->frame_h;
    const double mean_de = total / n_pixels;
    const double score = 45.0 - 20.0 * log10(mean_de);

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "ciede2000", score, index);
#else
    (void)feature_collector;
    (void)index;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    CiedeStateHip *s = fex->priv;

    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);

#ifdef HAVE_HIPCC
    /* ciede_hip_bufs_free also unloads the module; best-effort only. */
    ciede_hip_bufs_free(s);
#endif /* HAVE_HIPCC */

    int err = vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
    if (err != 0 && rc == 0) {
        rc = err;
    }
    if (s->feature_name_dict != NULL) {
        err = vmaf_dictionary_free(&s->feature_name_dict);
        if (err != 0 && rc == 0) {
            rc = err;
        }
    }
    if (s->ctx != NULL) {
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {"ciede2000", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_ciede_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_ciede_cuda` in
 * `libvmaf/src/feature/cuda/integer_ciede_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_ciede_hip = {
    .name = "ciede_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(CiedeStateHip),
    .provided_features = provided_features,
    /* Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with T7-10c.
     * Until then, pictures arrive as CPU VmafPictures and the submit
     * path does explicit HtoD copies. Same posture as all prior HIP
     * consumers (ADR-0241, ADR-0254, ADR-0372, ADR-0373). */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
