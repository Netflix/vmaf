/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature extractor on the HIP backend — first consumer of
 *  `libvmaf/src/hip/kernel_template.h` (T7-10 / ADR-0241).
 *  Real kernel promotion: T7-10b batch-1 / ADR-0372.
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/integer_psnr_cuda.c`
 *  call-graph-for-call-graph. When `HAVE_HIPCC` is defined (i.e.,
 *  `enable_hipcc=true` at configure time), the `init`, `submit`, and
 *  `collect` functions use real HIP Module API calls:
 *    `hipModuleLoadData` + `hipModuleGetFunction` + `hipModuleLaunchKernel`
 *  following the canonical pattern established by PR #612 / ADR-0254
 *  (`float_psnr_hip`).
 *
 *  Without `HAVE_HIPCC` (CPU-only builds, `enable_hip=true` but
 *  `enable_hipcc=false`), the scaffold posture is preserved: every
 *  lifecycle helper returns -ENOSYS, so the feature engine reports
 *  "runtime not ready" rather than crashing.
 *
 *  Algorithm (v1 luma-only, mirrors CPU `integer_psnr.c`):
 *      sse = sum_{i,j} (ref[i,j] - dis[i,j])^2;
 *      mse = sse / (w * h);
 *      psnr_y = MIN(10 * log10(peak^2 / max(mse, 1e-16)), psnr_max)
 *  psnr_max = (6 * bpc) + 12  (CPU `integer_psnr.c::init` min_sse==0 branch).
 *
 *  Chroma extension (psnr_cb / psnr_cr) is a follow-up per ADR-0372
 *  Consequences — mirrors the CUDA twin's T3-15(a) extension.
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
#include "integer_psnr_hip.h"

#ifdef HAVE_HIPCC
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

/* HSACO fat binary embedded by xxd -i during the meson hipcc pipeline
 * (ADR-0372 / `hip_hsaco_sources` meson block). The symbol is defined
 * by the auto-generated `psnr_score_hsaco.c` custom_target output. */
extern const unsigned char psnr_score_hsaco[];
extern const unsigned int psnr_score_hsaco_len;
#endif /* HAVE_HIPCC */

typedef struct PsnrStateHip {
    /* Lifecycle (private stream + submit/finished event pair) and the
     * (device SSE accumulator, pinned host readback slot) pair are
     * managed by `hip/kernel_template.h` (T7-10 first consumer /
     * ADR-0241). The struct shape mirrors the CUDA twin's
     * `PsnrStateCuda` — same fields in the same order, modulo the
     * `*_hip` -> `*_cuda` type names. */
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb;
    VmafHipContext *ctx;
    unsigned index;
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    uint32_t peak;
    double psnr_max_y;
    /* HIP module handle and per-bpc kernel function handles.
     * Both are zero-initialised in the scaffold path (no HAVE_HIPCC). */
#ifdef HAVE_HIPCC
    hipModule_t module;
    hipFunction_t funcbpc8;
    hipFunction_t funcbpc16;
    /* Per-frame staging buffers: ref and dis luma planes copied from
     * the VmafPicture device pointers into tightly-packed (width *
     * bytes_per_pixel)-pitch device buffers before the kernel launch.
     * Sized at init() time. */
    void *ref_in;
    void *dis_in;
#endif /* HAVE_HIPCC */
    double min_sse;
    VmafDictionary *feature_name_dict;
} PsnrStateHip;

static const VmafOption options[] = {
    {
        .name = "min_sse",
        .help = "minimum SSE floor; non-zero overrides the psnr_max ceiling",
        .offset = offsetof(PsnrStateHip, min_sse),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.0,
        .min = 0.0,
        .max = 1e16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0},
};

#define PSNR_HIP_BX 16
#define PSNR_HIP_BY 16

#ifdef HAVE_HIPCC
/* Translate a HIP error code to a negative errno. Mirrors
 * `hip_rc_to_errno` in `kernel_template.c`. */
static int psnr_hip_rc(hipError_t rc)
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

/* Load the HSACO module and look up the two kernel entry points.
 * Called once from init() with HAVE_HIPCC. */
static int psnr_hip_module_load(PsnrStateHip *s)
{
    hipError_t rc = hipModuleLoadData(&s->module, psnr_score_hsaco);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);

    rc = hipModuleGetFunction(&s->funcbpc8, s->module, "calculate_psnr_hip_kernel_8bpc");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return psnr_hip_rc(rc);
    }
    rc = hipModuleGetFunction(&s->funcbpc16, s->module, "calculate_psnr_hip_kernel_16bpc");
    if (rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return psnr_hip_rc(rc);
    }
    return 0;
}

/* Launch the appropriate bpc kernel on `pic_stream`, then record the
 * submit event, wait on the private stream, DtoH copy the accumulator,
 * and record the finished event. */
static int psnr_hip_launch(PsnrStateHip *s, uintptr_t pic_stream)
{
    hipStream_t str = (hipStream_t)s->lc.str;
    hipStream_t pstr = (hipStream_t)pic_stream;

    /* Zero the uint64 SSE accumulator before the kernel touches it.
     * Uses the picture stream so the memset is sequenced before the
     * kernel launch on the same stream. */
    hipError_t rc = hipMemsetAsync(s->rb.device, 0, sizeof(uint64_t), pstr);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);

    const ptrdiff_t plane_pitch = (ptrdiff_t)(s->frame_w * (s->bpc <= 8u ? 1u : 2u));
    const unsigned gx = (s->frame_w + PSNR_HIP_BX - 1u) / PSNR_HIP_BX;
    const unsigned gy = (s->frame_h + PSNR_HIP_BY - 1u) / PSNR_HIP_BY;

    unsigned long long *sse_dev = (unsigned long long *)s->rb.device;
    unsigned w = s->frame_w;
    unsigned h = s->frame_h;
    const uint8_t *ref_dev = (const uint8_t *)s->ref_in;
    const uint8_t *dis_dev = (const uint8_t *)s->dis_in;

    hipFunction_t func = (s->bpc == 8u) ? s->funcbpc8 : s->funcbpc16;

    /* hipModuleLaunchKernel argument pack: pass pointers-to-args in
     * the same order as the kernel's parameter list.
     * calculate_psnr_hip_kernel_{8,16}bpc(ref, dis, ref_stride,
     *   dis_stride, sse, width, height). */
    void *args[] = {
        (void *)&ref_dev, (void *)&dis_dev, (void *)&plane_pitch, (void *)&plane_pitch,
        (void *)&sse_dev, (void *)&w,       (void *)&h,
    };
    rc = hipModuleLaunchKernel(func, gx, gy, 1, PSNR_HIP_BX, PSNR_HIP_BY, 1, 0, pstr, args, NULL);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);

    /* Record submit event on the picture stream, then wait for it on
     * the private readback stream before the DtoH copy. */
    rc = hipEventRecord((hipEvent_t)s->lc.submit, pstr);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);
    rc = hipStreamWaitEvent(str, (hipEvent_t)s->lc.submit, 0);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);

    rc = hipMemcpyAsync(s->rb.host_pinned, s->rb.device, sizeof(uint64_t), hipMemcpyDeviceToHost,
                        str);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
}
#endif /* HAVE_HIPCC */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    PsnrStateHip *s = fex->priv;

    /* Allocate a HIP context — the scaffold's `vmaf_hip_context_new`
     * succeeds today (calloc + struct init); the runtime PR swaps in
     * `hipSetDevice` + handle creation. */
    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    /* Stream + event pair via the template. With HAVE_HIPCC this
     * returns 0 (real HIP calls); without it, -ENOSYS surfaces to the
     * caller as "runtime not ready". */
    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_after_ctx;

    s->bpc = bpc;
    s->frame_w = w;
    s->frame_h = h;
    s->peak = (1u << bpc) - 1u;
    /* psnr_max formula: mirrors CPU integer_psnr.c::init (lines 128-138).
     * When min_sse > 0 the user overrides the ceiling to match the MSE
     * floor; otherwise use the (6*bpc)+12 default. */
    if (s->min_sse != 0.0) {
        const double mse = s->min_sse / ((double)w * (double)h);
        s->psnr_max_y = ceil(10.0 * log10((double)s->peak * (double)s->peak / mse));
    } else {
        s->psnr_max_y = (double)(6u * bpc) + 12.0;
    }

    /* Readback pair (device uint64 SSE accumulator + pinned host slot). */
    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx, sizeof(uint64_t));
    if (err != 0)
        goto fail_after_lc;

#ifdef HAVE_HIPCC
    /* Load HSACO module and look up the two kernel entry points. */
    err = psnr_hip_module_load(s);
    if (err != 0)
        goto fail_after_rb;

    /* Allocate tightly-pitched device staging buffers for ref + dis. */
    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    const size_t plane_bytes = (size_t)w * h * bpp;
    hipError_t rc = hipMalloc(&s->ref_in, plane_bytes);
    if (rc != hipSuccess) {
        err = -ENOMEM;
        goto fail_after_module;
    }
    rc = hipMalloc(&s->dis_in, plane_bytes);
    if (rc != hipSuccess) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
        err = -ENOMEM;
        goto fail_after_module;
    }
#endif /* HAVE_HIPCC */

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
#ifdef HAVE_HIPCC
        goto fail_after_bufs;
#else
        goto fail_after_rb;
#endif
    }

    return 0;

#ifdef HAVE_HIPCC
fail_after_bufs:
    if (s->dis_in != NULL) {
        (void)hipFree(s->dis_in);
        s->dis_in = NULL;
    }
    if (s->ref_in != NULL) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
    }
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
    PsnrStateHip *s = fex->priv;

    s->index = index;
    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

#ifdef HAVE_HIPCC
    /* HtoD copy of ref and dis luma planes into the tightly-pitched
     * staging buffers. The picture stream is passed to the kernel
     * launch and readback sequence so all work is properly ordered. */
    const ptrdiff_t plane_pitch = (ptrdiff_t)(s->frame_w * (s->bpc <= 8u ? 1u : 2u));
    /* Use picture_stream = 0 for the cross-stream wait (no VmafPicture
     * stream handle is exposed in the HIP scaffold yet — mirrors the
     * scaffold posture in float_psnr_hip.c and the CUDA twin's
     * `vmaf_cuda_picture_get_stream` wrapper). */
    const uintptr_t pic_stream_handle = 0;

    /* HtoD copy ref luma via 2D memcpy to handle arbitrary source stride. */
    hipError_t rc =
        hipMemcpy2DAsync(s->ref_in, (size_t)plane_pitch, ref_pic->data[0],
                         (size_t)ref_pic->stride[0], (size_t)plane_pitch, (size_t)s->frame_h,
                         hipMemcpyHostToDevice, (hipStream_t)pic_stream_handle);
    if (rc != hipSuccess)
        return -EIO;

    rc = hipMemcpy2DAsync(s->dis_in, (size_t)plane_pitch, dist_pic->data[0],
                          (size_t)dist_pic->stride[0], (size_t)plane_pitch, (size_t)s->frame_h,
                          hipMemcpyHostToDevice, (hipStream_t)pic_stream_handle);
    if (rc != hipSuccess)
        return -EIO;

    return psnr_hip_launch(s, pic_stream_handle);
#else
    /* Scaffold posture: the pre-launch helper returns -ENOSYS, which
     * surfaces as "runtime not ready" without any device work being
     * scheduled. Once HAVE_HIPCC is defined and the module is loaded,
     * the path above handles the real dispatch. */
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
    PsnrStateHip *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffer is
     * safe to read. Mirrors the CUDA twin. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

#ifdef HAVE_HIPCC
    /* Read the uint64 SSE from the pinned host buffer. */
    const double sse = (double)*(const uint64_t *)s->rb.host_pinned;
    const double n_pixels = (double)s->frame_w * (double)s->frame_h;
    const double mse = sse / n_pixels;
    /* Match CPU integer_psnr.c::extract: clamp at psnr_max via
     * MIN(10*log10(peak^2 / max(mse, 1e-16)), psnr_max). The 1e-16
     * floor guards against sse == 0 (identical frames). */
    const double peak_sq = (double)s->peak * (double)s->peak;
    const double mse_clamped = (mse > 1e-16) ? mse : 1e-16;
    double psnr = 10.0 * log10(peak_sq / mse_clamped);
    if (psnr > s->psnr_max_y)
        psnr = s->psnr_max_y;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "psnr_y", psnr, index);
#else
    (void)feature_collector;
    (void)index;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    PsnrStateHip *s = fex->priv;

    /* Lifecycle teardown via the template (sync → destroy stream →
     * destroy events). Best-effort error aggregation matches the
     * CUDA twin's close path. */
    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
    int err = vmaf_hip_kernel_readback_free(&s->rb, s->ctx);
    if (err != 0 && rc == 0)
        rc = err;

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
        hipError_t hip_err = hipModuleUnload(s->module);
        if (hip_err != hipSuccess && rc == 0)
            rc = -EIO;
        s->module = NULL;
    }
#endif /* HAVE_HIPCC */

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

static const char *provided_features[] = {"psnr_y", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_psnr_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_psnr_cuda` in
 * `libvmaf/src/feature/cuda/integer_psnr_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_psnr_hip = {
    .name = "psnr_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(PsnrStateHip),
    .provided_features = provided_features,
    /* Intentionally no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the
     * picture buffer-type plumbing for HIP lands with the runtime
     * PR (T7-10b). Until then the consumer registers as a
     * "CPU-flagged" extractor whose `init()` returns -ENOSYS on
     * non-ROCm builds, so any caller asking for `psnr_hip` gets a
     * clean "runtime not ready" surface. The flag bit is reserved in
     * `feature_extractor.h` so the runtime PR can adopt it
     * without an enum reshuffle. */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
