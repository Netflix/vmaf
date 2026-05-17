/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature extractor on the HIP backend — first consumer of
 *  `libvmaf/src/hip/kernel_template.h` (T7-10 / ADR-0241).
 *  Real kernel promotion: T7-10b batch-1 / ADR-0372.
 *  Chroma extension (psnr_cb / psnr_cr): ADR-0471.
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
 *  Algorithm (mirrors CPU `integer_psnr.c`):
 *      sse_p = sum_{i,j} (ref_p[i,j] - dis_p[i,j])^2;   (per plane p)
 *      mse_p = sse_p / (w_p * h_p);
 *      psnr_p = MIN(10 * log10(peak^2 / max(mse_p, 1e-16)), psnr_max)
 *  psnr_max = (6 * bpc) + 12  (CPU `integer_psnr.c::init` min_sse==0 branch).
 *
 *  enable_chroma (default true): when false or pix_fmt == YUV400P, only
 *  the luma plane is dispatched (mirrors ADR-0453 CPU/CUDA/SYCL/Vulkan pattern).
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
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

#define PSNR_NUM_PLANES 3U

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
    /* Lifecycle (private stream + submit/finished event pair) shared
     * across all plane dispatches — they execute sequentially.
     * Mirrors `PsnrStateCuda` layout (ADR-0371 / T3-15(a)). */
    VmafHipKernelLifecycle lc;
    /* Per-plane readback slots: one (device SSE, pinned host) pair per
     * active plane.  Mirrors the CUDA twin's rb[PSNR_NUM_PLANES]. */
    VmafHipKernelReadback rb[PSNR_NUM_PLANES];
    VmafHipContext *ctx;
    unsigned index;
    /* Per-plane geometry (luma = [0], Cb = [1], Cr = [2]). */
    unsigned width[PSNR_NUM_PLANES];
    unsigned height[PSNR_NUM_PLANES];
    unsigned bpc;
    uint32_t peak;
    /* psnr_max[p] = (6 * bpc) + 12 for all p (CPU integer_psnr.c
     * min_sse==0 branch). Array layout enables per-plane override. */
    double psnr_max[PSNR_NUM_PLANES];
    /* `enable_chroma` option: when false, only luma is dispatched.
     * Default true mirrors CPU integer_psnr.c — see ADR-0453/0471. */
    bool enable_chroma;
    /* Number of active planes (1 for YUV400 or enable_chroma=false,
     * 3 otherwise). */
    unsigned n_planes;
    /* HIP module handle and per-bpc kernel function handles.
     * Both are zero-initialised in the scaffold path (no HAVE_HIPCC). */
#ifdef HAVE_HIPCC
    hipModule_t module;
    hipFunction_t funcbpc8;
    hipFunction_t funcbpc16;
    /* Per-plane staging buffers: ref and dis for each active plane,
     * copied from VmafPicture data pointers at submit time.
     * Sized at init() for the maximum geometry (luma at [0],
     * chroma at [1]/[2]).  NULL when n_planes < 3. */
    void *ref_in[PSNR_NUM_PLANES];
    void *dis_in[PSNR_NUM_PLANES];
#endif /* HAVE_HIPCC */
    VmafDictionary *feature_name_dict;
} PsnrStateHip;

static const VmafOption options[] = {{0}};

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

/* Launch the PSNR kernel for a single plane on `pic_stream`.
 * `plane` selects ref_in[plane]/dis_in[plane] and rb[plane].device.
 * The accumulator is zeroed on the picture stream before the kernel. */
static int psnr_hip_launch_plane(PsnrStateHip *s, unsigned plane, uintptr_t pic_stream)
{
    hipStream_t pstr = (hipStream_t)pic_stream;

    hipError_t rc = hipMemsetAsync(s->rb[plane].device, 0, sizeof(uint64_t), pstr);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);

    const unsigned pw = s->width[plane];
    const unsigned ph = s->height[plane];
    const ptrdiff_t stride = (ptrdiff_t)(pw * (s->bpc <= 8u ? 1u : 2u));
    const unsigned gx = (pw + PSNR_HIP_BX - 1u) / PSNR_HIP_BX;
    const unsigned gy = (ph + PSNR_HIP_BY - 1u) / PSNR_HIP_BY;

    unsigned long long *sse_dev = (unsigned long long *)s->rb[plane].device;
    const uint8_t *ref_dev = (const uint8_t *)s->ref_in[plane];
    const uint8_t *dis_dev = (const uint8_t *)s->dis_in[plane];

    hipFunction_t func = (s->bpc == 8u) ? s->funcbpc8 : s->funcbpc16;

    /* hipModuleLaunchKernel argument pack:
     * calculate_psnr_hip_kernel_{8,16}bpc(ref, dis, ref_stride,
     *   dis_stride, sse, width, height). */
    void *args[] = {
        (void *)&ref_dev, (void *)&dis_dev, (void *)&stride, (void *)&stride,
        (void *)&sse_dev, (void *)&pw,      (void *)&ph,
    };
    rc = hipModuleLaunchKernel(func, gx, gy, 1, PSNR_HIP_BX, PSNR_HIP_BY, 1, 0, pstr, args, NULL);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);
    return 0;
}

/* Launch all active planes, then record submit → readback → finished. */
static int psnr_hip_launch(PsnrStateHip *s, uintptr_t pic_stream)
{
    hipStream_t str = (hipStream_t)s->lc.str;
    hipStream_t pstr = (hipStream_t)pic_stream;

    for (unsigned p = 0; p < s->n_planes; p++) {
        const int err = psnr_hip_launch_plane(s, p, pic_stream);
        if (err != 0)
            return err;
    }

    /* Record submit on the picture stream; wait on the private readback
     * stream; DtoH copy all active plane accumulators; record finished. */
    hipError_t rc = hipEventRecord((hipEvent_t)s->lc.submit, pstr);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);
    rc = hipStreamWaitEvent(str, (hipEvent_t)s->lc.submit, 0);
    if (rc != hipSuccess)
        return psnr_hip_rc(rc);

    for (unsigned p = 0; p < s->n_planes; p++) {
        rc = hipMemcpyAsync(s->rb[p].host_pinned, s->rb[p].device, sizeof(uint64_t),
                            hipMemcpyDeviceToHost, str);
        if (rc != hipSuccess)
            return psnr_hip_rc(rc);
    }

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
}
#endif /* HAVE_HIPCC */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    PsnrStateHip *s = fex->priv;

    /* Per-plane geometry — mirrors CPU integer_psnr.c::init and CUDA twin
     * (ADR-0453/0471). YUV400 has no chroma planes. */
    s->width[0] = w;
    s->height[0] = h;
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    } else {
        s->n_planes = PSNR_NUM_PLANES;
        const int ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P);
        const int ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P);
        /* Ceiling division — mirrors picture.c fix (Research-0094). */
        const unsigned cw = (w + (unsigned)ss_hor) >> ss_hor;
        const unsigned ch = (h + (unsigned)ss_ver) >> ss_ver;
        s->width[1] = s->width[2] = cw;
        s->height[1] = s->height[2] = ch;
    }
    /* enable_chroma guard (ADR-0453/0471): luma-only when the option is
     * false on a non-YUV400 source. */
    if (!s->enable_chroma && s->n_planes > 1U) {
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    }

    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_after_ctx;

    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;
    /* psnr_max formula mirrors CPU integer_psnr.c::init min_sse==0 branch. */
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++)
        s->psnr_max[p] = (double)(6u * bpc) + 12.0;

    /* Per-plane readback pairs (device uint64 SSE accumulator + pinned host
     * slot). Allocate only for active planes to avoid wasting pinned memory. */
    for (unsigned p = 0; p < s->n_planes; p++) {
        err = vmaf_hip_kernel_readback_alloc(&s->rb[p], s->ctx, sizeof(uint64_t));
        if (err != 0)
            goto fail_after_rb;
    }

#ifdef HAVE_HIPCC
    /* Load HSACO module and look up the two kernel entry points. */
    err = psnr_hip_module_load(s);
    if (err != 0)
        goto fail_after_rb;

    /* Allocate tightly-pitched device staging buffers for each active plane. */
    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const size_t plane_bytes = (size_t)s->width[p] * s->height[p] * bpp;
        hipError_t rc = hipMalloc(&s->ref_in[p], plane_bytes);
        if (rc != hipSuccess) {
            err = -ENOMEM;
            goto fail_after_bufs;
        }
        rc = hipMalloc(&s->dis_in[p], plane_bytes);
        if (rc != hipSuccess) {
            (void)hipFree(s->ref_in[p]);
            s->ref_in[p] = NULL;
            err = -ENOMEM;
            goto fail_after_bufs;
        }
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
    for (unsigned p = 0; p < s->n_planes; p++) {
        if (s->dis_in[p] != NULL) {
            (void)hipFree(s->dis_in[p]);
            s->dis_in[p] = NULL;
        }
        if (s->ref_in[p] != NULL) {
            (void)hipFree(s->ref_in[p]);
            s->ref_in[p] = NULL;
        }
    }
    if (s->module != NULL) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
#endif /* HAVE_HIPCC */
fail_after_rb:
    for (unsigned p = 0; p < s->n_planes; p++)
        (void)vmaf_hip_kernel_readback_free(&s->rb[p], s->ctx);
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

#ifdef HAVE_HIPCC
    /* Use picture_stream = 0 (HIP scaffold: no per-picture stream handle
     * yet — mirrors float_psnr_hip.c and the CUDA twin pre-runtime). */
    const uintptr_t pic_stream_handle = 0;
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const size_t plane_pitch = (size_t)s->width[0] * bpp;
    const unsigned frame_h = s->height[0];

    /* HtoD copy ref luma via 2D memcpy to handle arbitrary source stride. */
    hipError_t rc = hipMemcpy2DAsync(s->ref_in[0], plane_pitch, ref_pic->data[0],
                                     (size_t)ref_pic->stride[0], plane_pitch, (size_t)frame_h,
                                     hipMemcpyHostToDevice, (hipStream_t)pic_stream_handle);
    if (rc != hipSuccess)
        return -EIO;

    rc = hipMemcpy2DAsync(s->dis_in[0], plane_pitch, dist_pic->data[0], (size_t)dist_pic->stride[0],
                          plane_pitch, (size_t)frame_h, hipMemcpyHostToDevice,
                          (hipStream_t)pic_stream_handle);
    if (rc != hipSuccess)
        return -EIO;

    return psnr_hip_launch(s, pic_stream_handle);
#else
    /* Scaffold posture: returns -ENOSYS ("runtime not ready"). */
    (void)ref_pic;
    (void)dist_pic;
    int err = vmaf_hip_kernel_submit_pre_launch(&s->lc, s->ctx, &s->rb[0],
                                                /* picture_stream */ 0,
                                                /* dist_ready_event */ 0);
    if (err != 0)
        return err;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

/* psnr_name[p] matches CPU integer_psnr.c and the CUDA twin. */
static const char *const psnr_name[PSNR_NUM_PLANES] = {"psnr_y", "psnr_cb", "psnr_cr"};

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    PsnrStateHip *s = fex->priv;

    /* Drain the private readback stream so host pinned buffers are safe
     * to read. One drain covers all active plane accumulators. */
    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

#ifdef HAVE_HIPCC
    /* Match CPU integer_psnr.c::extract per plane. */
    const double peak_sq = (double)s->peak * (double)s->peak;
    int rc = 0;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const double sse = (double)*(const uint64_t *)s->rb[p].host_pinned;
        const double n_pixels = (double)s->width[p] * (double)s->height[p];
        const double mse = sse / n_pixels;
        /* Clamp at psnr_max[p]; 1e-16 floor guards sse==0. */
        const double mse_clamped = (mse > 1e-16) ? mse : 1e-16;
        double psnr = 10.0 * log10(peak_sq / mse_clamped);
        if (psnr > s->psnr_max[p])
            psnr = s->psnr_max[p];
        const int e = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, psnr_name[p], psnr, index);
        if (e != 0 && rc == 0)
            rc = e;
    }
    return rc;
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
    for (unsigned p = 0; p < s->n_planes; p++) {
        const int e = vmaf_hip_kernel_readback_free(&s->rb[p], s->ctx);
        if (e != 0 && rc == 0)
            rc = e;
    }

#ifdef HAVE_HIPCC
    for (unsigned p = 0; p < s->n_planes; p++) {
        if (s->dis_in[p] != NULL) {
            (void)hipFree(s->dis_in[p]);
            s->dis_in[p] = NULL;
        }
        if (s->ref_in[p] != NULL) {
            (void)hipFree(s->ref_in[p]);
            s->ref_in[p] = NULL;
        }
    }
    if (s->module != NULL) {
        hipError_t hip_err = hipModuleUnload(s->module);
        if (hip_err != hipSuccess && rc == 0)
            rc = -EIO;
        s->module = NULL;
    }
#endif /* HAVE_HIPCC */

    if (s->feature_name_dict != NULL) {
        const int e = vmaf_dictionary_free(&s->feature_name_dict);
        if (e != 0 && rc == 0)
            rc = e;
    }
    if (s->ctx != NULL) {
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

/* Provided features — full luma + chroma, matching the CUDA twin (ADR-0471).
 * YUV400 and enable_chroma=false sources clamp n_planes to 1 at init time
 * so chroma is never dispatched at runtime, but the static list still claims
 * psnr_cb/psnr_cr so the dispatcher routes requests through psnr_hip. */
static const char *provided_features[] = {"psnr_y", "psnr_cb", "psnr_cr", NULL};

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
            /* 3 dispatches/frame (one per plane) when enable_chroma=true.
             * Matches the CUDA twin (ADR-0471). */
            .n_dispatches_per_frame = 3,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
