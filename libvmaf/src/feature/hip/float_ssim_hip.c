/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ssim feature extractor on the HIP backend — eighth consumer
 *  of `libvmaf/src/hip/kernel_template.h` (T7-10b batch-3 / ADR-0374).
 *
 *  Mirrors `libvmaf/src/feature/cuda/integer_ssim_cuda.c` call-graph-for-
 *  call-graph. Two-pass design mirrors the GLSL Vulkan shader and the
 *  CUDA twin:
 *    Pass 1 — horizontal 11-tap separable Gaussian over ref / cmp /
 *             ref^2 / cmp^2 / ref*cmp into five intermediate float
 *             device buffers, grid sized over (W-10) x H.
 *    Pass 2 — vertical 11-tap + per-pixel SSIM combine + per-block
 *             float partial sum, grid sized over (W-10) x (H-10).
 *  Host accumulates partials in double, divides by (W-10)*(H-10) and
 *  emits `float_ssim`.
 *
 *  HIP adaptation from CUDA:
 *  - `hipModuleLoadData` / `hipModuleGetFunction` / `hipModuleLaunchKernel`
 *    instead of `cuModuleLoadData` / `cuModuleGetFunction` / `cuLaunchKernel`.
 *  - Five intermediate float buffers allocated via `hipMalloc` (raw device
 *    pointers) instead of `vmaf_cuda_buffer_alloc` (which carries VmafCudaBuffer
 *    wrapper + `free(wrapper)` dance). The HIP scaffold has no equivalent
 *    buffer-wrapper helper.
 *  - Pictures arrive as CPU VmafPictures (VMAF_FEATURE_EXTRACTOR_HIP flag
 *    cleared, T7-10b posture). Luma planes are copied HtoD via
 *    `hipMemcpy2DAsync` on the private readback stream.
 *  - Pass 1 → Pass 2 ordering: both launches on the same HIP stream, so
 *    the implicit stream order guarantees Pass 1 writes are visible to
 *    Pass 2 reads — same happens-before as the CUDA twin on the CUDA stream.
 *
 *  When `enable_hipcc=false` (e.g. a CI agent without ROCm), `HAVE_HIPCC`
 *  is undefined and `init()` returns -ENOSYS — same scaffold contract as
 *  the pre-runtime posture (registered, runtime not ready).
 *
 *  v1: scale=1 only — same constraint as ssim_vulkan / ssim_cuda. The
 *  auto-detect path rejects scale>1 with -EINVAL at init.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"
#include "log.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"
#include "float_ssim_hip.h"

/* ------------------------------------------------------------------ */
/* Block geometry constants (match ssim_score.hip)                     */
/* ------------------------------------------------------------------ */

#define SSIM_HIP_BLOCK_X 16u
#define SSIM_HIP_BLOCK_Y 8u
#define SSIM_HIP_K 11u

/* ------------------------------------------------------------------ */
/* HIP-to-errno translation                                            */
/* ------------------------------------------------------------------ */

static int ssim_hip_rc(hipError_t rc)
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

typedef struct SsimStateHip {
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb; /* device: per-block float partials;
                                  * host_pinned: readback slot */
    VmafHipContext *ctx;

    int scale_override;

    /* Five intermediate float device buffers for the horiz pass output.
     * Sized (w_horiz * h_horiz * sizeof(float)). Allocated via hipMalloc,
     * freed via hipFree. The CUDA twin carries VmafCudaBuffer * wrappers;
     * the HIP scaffold has no equivalent, so raw void * is used. */
    void *d_ref_mu;
    void *d_cmp_mu;
    void *d_ref_sq;
    void *d_cmp_sq;
    void *d_refcmp;

    /* Staging buffers: CPU luma planes → device (HtoD). One each for
     * ref and cmp (luma-only, no chroma). Sized frame_w * frame_h * bpp.
     * Also allocated via hipMalloc. */
    void *ref_in;
    void *cmp_in;

    /* HIP module + per-bpc horiz kernel + vert-combine kernel handles. */
    hipModule_t module;
    hipFunction_t func_horiz_8;
    hipFunction_t func_horiz_16;
    hipFunction_t func_vert;

    unsigned partials_capacity;
    unsigned partials_count;

    unsigned width;
    unsigned height;
    unsigned w_horiz;
    unsigned h_horiz;
    unsigned w_final;
    unsigned h_final;
    unsigned bpc;
    float c1;
    float c2;

    unsigned index;
    VmafDictionary *feature_name_dict;
} SsimStateHip;

static const VmafOption options[] = {
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling). "
                "v1: GPU path requires scale=1; auto-detect rejects scale>1 with -EINVAL.",
        .offset = offsetof(SsimStateHip, scale_override),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* Dimension helpers                                                   */
/* ------------------------------------------------------------------ */

static int ssim_hip_round_to_int(float x)
{
    return (int)(x + (x < 0.0f ? -0.5f : 0.5f));
}

static int ssim_hip_min_int(int a, int b)
{
    return a < b ? a : b;
}

static int ssim_hip_compute_scale(unsigned w, unsigned h, int override_val)
{
    if (override_val > 0)
        return override_val;
    int scaled = ssim_hip_round_to_int((float)ssim_hip_min_int((int)w, (int)h) / 256.0f);
    return scaled < 1 ? 1 : scaled;
}

/* Extracted to keep init_fex_hip under the 60-line readability-function-size
 * limit. Mirrors validate logic from the CUDA twin. */
static int ssim_hip_validate_dims(const SsimStateHip *s, unsigned w, unsigned h)
{
    int scale = ssim_hip_compute_scale(w, h, s->scale_override);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_hip: v1 supports scale=1 only "
                 "(auto-detected scale=%d at %ux%u). "
                 "Pin --feature float_ssim_hip:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }
    if (w < SSIM_HIP_K || h < SSIM_HIP_K) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_hip: input %ux%u smaller than 11x11 Gaussian footprint.\n", w, h);
        return -EINVAL;
    }
    return 0;
}

/* Populate geometry + SSIM constant fields. Extracted to keep init_fex_hip
 * under the 60-line readability-function-size limit. */
static void ssim_hip_init_dims(SsimStateHip *s, unsigned w, unsigned h, unsigned bpc)
{
    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->w_horiz = w - (SSIM_HIP_K - 1u);
    s->h_horiz = h;
    s->w_final = w - (SSIM_HIP_K - 1u);
    s->h_final = h - (SSIM_HIP_K - 1u);

    /* SSIM stability constants: L = 255.0, K1 = 0.01, K2 = 0.03.
     * The CUDA twin pins L = 255 (float), independent of bpc, so the
     * cross-backend numeric gate has nothing fork-specific to track. */
    const float L = 255.0f;
    const float K1 = 0.01f;
    const float K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);

    const unsigned grid_x = (s->w_final + SSIM_HIP_BLOCK_X - 1u) / SSIM_HIP_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_HIP_BLOCK_Y - 1u) / SSIM_HIP_BLOCK_Y;
    s->partials_capacity = grid_x * grid_y;
}

/* ------------------------------------------------------------------ */
/* HAVE_HIPCC helpers                                                  */
/* ------------------------------------------------------------------ */

#ifdef HAVE_HIPCC
/*
 * Load the HSACO fat binary, resolve the three kernel function handles.
 * On failure returns a negative errno; caller unwinds via fail_after_rb.
 */
static int ssim_hip_module_load(SsimStateHip *s)
{
    hipError_t hip_rc = hipModuleLoadData(&s->module, ssim_score_hsaco);
    if (hip_rc != hipSuccess)
        return ssim_hip_rc(hip_rc);

    hip_rc = hipModuleGetFunction(&s->func_horiz_8, s->module, "calculate_ssim_hip_horiz_8bpc");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return ssim_hip_rc(hip_rc);
    }
    hip_rc = hipModuleGetFunction(&s->func_horiz_16, s->module, "calculate_ssim_hip_horiz_16bpc");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return ssim_hip_rc(hip_rc);
    }
    hip_rc = hipModuleGetFunction(&s->func_vert, s->module, "calculate_ssim_hip_vert_combine");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return ssim_hip_rc(hip_rc);
    }
    return 0;
}

/*
 * Allocate five intermediate float device buffers + two luma staging
 * buffers. Extracted to keep init_fex_hip under the 60-line
 * readability-function-size limit.
 *
 * On failure: partially allocated buffers are freed and all pointers
 * are left NULL; caller unwinds via fail_after_module.
 */
static int ssim_hip_bufs_alloc(SsimStateHip *s)
{
    const size_t horiz_bytes = (size_t)s->w_horiz * s->h_horiz * sizeof(float);
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const size_t stage_bytes = (size_t)s->width * s->height * bpp;

    hipError_t hip_rc;
    hip_rc = hipMalloc(&s->d_ref_mu, horiz_bytes);
    if (hip_rc != hipSuccess)
        return ssim_hip_rc(hip_rc);
    hip_rc = hipMalloc(&s->d_cmp_mu, horiz_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return ssim_hip_rc(hip_rc);
    }
    hip_rc = hipMalloc(&s->d_ref_sq, horiz_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->d_cmp_mu);
        s->d_cmp_mu = NULL;
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return ssim_hip_rc(hip_rc);
    }
    hip_rc = hipMalloc(&s->d_cmp_sq, horiz_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->d_ref_sq);
        s->d_ref_sq = NULL;
        (void)hipFree(s->d_cmp_mu);
        s->d_cmp_mu = NULL;
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return ssim_hip_rc(hip_rc);
    }
    hip_rc = hipMalloc(&s->d_refcmp, horiz_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->d_cmp_sq);
        s->d_cmp_sq = NULL;
        (void)hipFree(s->d_ref_sq);
        s->d_ref_sq = NULL;
        (void)hipFree(s->d_cmp_mu);
        s->d_cmp_mu = NULL;
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return ssim_hip_rc(hip_rc);
    }
    hip_rc = hipMalloc(&s->ref_in, stage_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->d_refcmp);
        s->d_refcmp = NULL;
        (void)hipFree(s->d_cmp_sq);
        s->d_cmp_sq = NULL;
        (void)hipFree(s->d_ref_sq);
        s->d_ref_sq = NULL;
        (void)hipFree(s->d_cmp_mu);
        s->d_cmp_mu = NULL;
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return ssim_hip_rc(hip_rc);
    }
    hip_rc = hipMalloc(&s->cmp_in, stage_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
        (void)hipFree(s->d_refcmp);
        s->d_refcmp = NULL;
        (void)hipFree(s->d_cmp_sq);
        s->d_cmp_sq = NULL;
        (void)hipFree(s->d_ref_sq);
        s->d_ref_sq = NULL;
        (void)hipFree(s->d_cmp_mu);
        s->d_cmp_mu = NULL;
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return ssim_hip_rc(hip_rc);
    }
    return 0;
}

/* Free all seven device buffers. Safe to call with NULL pointers. */
static void ssim_hip_bufs_free(SsimStateHip *s)
{
    if (s->cmp_in != NULL) {
        (void)hipFree(s->cmp_in);
        s->cmp_in = NULL;
    }
    if (s->ref_in != NULL) {
        (void)hipFree(s->ref_in);
        s->ref_in = NULL;
    }
    if (s->d_refcmp != NULL) {
        (void)hipFree(s->d_refcmp);
        s->d_refcmp = NULL;
    }
    if (s->d_cmp_sq != NULL) {
        (void)hipFree(s->d_cmp_sq);
        s->d_cmp_sq = NULL;
    }
    if (s->d_ref_sq != NULL) {
        (void)hipFree(s->d_ref_sq);
        s->d_ref_sq = NULL;
    }
    if (s->d_cmp_mu != NULL) {
        (void)hipFree(s->d_cmp_mu);
        s->d_cmp_mu = NULL;
    }
    if (s->d_ref_mu != NULL) {
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
    }
}

/*
 * Pass 1 — horizontal 11-tap Gaussian kernel launch.
 * Grid sized over w_horiz x h_horiz. Block 16x8.
 * Writes five intermediate float buffers on `str`.
 */
static int ssim_hip_launch_horiz(SsimStateHip *s, hipStream_t str)
{
    const unsigned grid_horiz_x = (s->w_horiz + SSIM_HIP_BLOCK_X - 1u) / SSIM_HIP_BLOCK_X;
    const unsigned grid_horiz_y = (s->h_horiz + SSIM_HIP_BLOCK_Y - 1u) / SSIM_HIP_BLOCK_Y;

    const ptrdiff_t ref_stride = (ptrdiff_t)(s->width * ((s->bpc <= 8u) ? 1u : 2u));
    /* Horiz kernel takes raw uint8* for both bpc variants. */
    hipError_t hip_rc;
    if (s->bpc == 8u) {
        void *args[] = {
            &s->ref_in,   (void *)&ref_stride, &s->cmp_in,   (void *)&ref_stride,
            &s->d_ref_mu, &s->d_cmp_mu,        &s->d_ref_sq, &s->d_cmp_sq,
            &s->d_refcmp, &s->w_horiz,         &s->h_horiz,
        };
        hip_rc = hipModuleLaunchKernel(s->func_horiz_8, grid_horiz_x, grid_horiz_y, 1u,
                                       SSIM_HIP_BLOCK_X, SSIM_HIP_BLOCK_Y, 1u, 0, str, args, NULL);
    } else {
        void *args[] = {
            &s->ref_in,   (void *)&ref_stride, &s->cmp_in,   (void *)&ref_stride,
            &s->d_ref_mu, &s->d_cmp_mu,        &s->d_ref_sq, &s->d_cmp_sq,
            &s->d_refcmp, &s->w_horiz,         &s->h_horiz,  &s->bpc,
        };
        hip_rc = hipModuleLaunchKernel(s->func_horiz_16, grid_horiz_x, grid_horiz_y, 1u,
                                       SSIM_HIP_BLOCK_X, SSIM_HIP_BLOCK_Y, 1u, 0, str, args, NULL);
    }
    return ssim_hip_rc(hip_rc);
}

/*
 * Pass 2 — vertical 11-tap + SSIM combine + per-block partial sum,
 * followed by DtoH readback and finished-event record.
 * Grid sized over w_final x h_final. Block 16x8.
 * Both passes run on the same stream, so implicit ordering holds.
 */
static int ssim_hip_launch_vert_readback(SsimStateHip *s, hipStream_t str)
{
    const unsigned grid_x = (s->w_final + SSIM_HIP_BLOCK_X - 1u) / SSIM_HIP_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_HIP_BLOCK_Y - 1u) / SSIM_HIP_BLOCK_Y;

    void *args2[] = {
        &s->d_ref_mu, &s->d_cmp_mu, &s->d_ref_sq, &s->d_cmp_sq, &s->d_refcmp, &s->rb.device,
        &s->w_horiz,  &s->w_final,  &s->h_final,  &s->c1,       &s->c2,
    };
    hipError_t hip_rc = hipModuleLaunchKernel(s->func_vert, grid_x, grid_y, 1u, SSIM_HIP_BLOCK_X,
                                              SSIM_HIP_BLOCK_Y, 1u, 0, str, args2, NULL);
    if (hip_rc != hipSuccess)
        return ssim_hip_rc(hip_rc);

    /* Record submit event on the picture stream, then DtoH copy on the
     * private readback stream (same pattern as float_psnr_hip.c). */
    hip_rc = hipEventRecord((hipEvent_t)s->lc.submit, str);
    if (hip_rc != hipSuccess)
        return ssim_hip_rc(hip_rc);

    const size_t copy_bytes = (size_t)s->partials_count * sizeof(float);
    hip_rc =
        hipMemcpyAsync(s->rb.host_pinned, s->rb.device, copy_bytes, hipMemcpyDeviceToHost, str);
    if (hip_rc != hipSuccess)
        return ssim_hip_rc(hip_rc);

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
}
#endif /* HAVE_HIPCC */

/* ------------------------------------------------------------------ */
/* init / close                                                        */
/* ------------------------------------------------------------------ */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    SsimStateHip *s = fex->priv;

    int err = ssim_hip_validate_dims(s, w, h);
    if (err != 0)
        return err;

    ssim_hip_init_dims(s, w, h, bpc);

    err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_after_ctx;

    err = vmaf_hip_kernel_readback_alloc(&s->rb, s->ctx,
                                         (size_t)s->partials_capacity * sizeof(float));
    if (err != 0)
        goto fail_after_lc;

#ifdef HAVE_HIPCC
    err = ssim_hip_module_load(s);
    if (err != 0)
        goto fail_after_rb;
    err = ssim_hip_bufs_alloc(s);
    if (err != 0)
        goto fail_after_module;
#else
    err = -ENOSYS;
    if (err != 0)
        goto fail_after_rb;
#endif

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
#ifdef HAVE_HIPCC
        ssim_hip_bufs_free(s);
        goto fail_after_module;
#else
        goto fail_after_rb;
#endif
    }
    return 0;

#ifdef HAVE_HIPCC
fail_after_module:
    (void)hipModuleUnload(s->module);
    s->module = NULL;
#endif
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
    SsimStateHip *s = fex->priv;
    int rc = 0;

#ifdef HAVE_HIPCC
    ssim_hip_bufs_free(s);
    if (s->module != NULL) {
        int e = ssim_hip_rc(hipModuleUnload(s->module));
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

#ifndef HAVE_HIPCC
    (void)fex;
    (void)ref_pic;
    (void)dist_pic;
    (void)index;
    return -ENOSYS;
#else
    SsimStateHip *s = fex->priv;
    s->index = index;
    const unsigned grid_x = (s->w_final + SSIM_HIP_BLOCK_X - 1u) / SSIM_HIP_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_HIP_BLOCK_Y - 1u) / SSIM_HIP_BLOCK_Y;
    s->partials_count = grid_x * grid_y;

    /* Copy both luma planes HtoD on the private stream, then dispatch
     * Pass 1 (horiz Gaussian) and Pass 2 (vert + SSIM combine) on the
     * same stream. VMAF_FEATURE_EXTRACTOR_HIP is not set (T7-10b
     * posture), so pictures arrive as CPU VmafPictures. */
    const hipStream_t str = (hipStream_t)s->lc.str;
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t row_w = (ptrdiff_t)(s->width * bpp);

    hipError_t hip_rc =
        hipMemcpy2DAsync(s->ref_in, (size_t)row_w, ref_pic->data[0], (size_t)ref_pic->stride[0],
                         (size_t)row_w, (size_t)s->height, hipMemcpyHostToDevice, str);
    if (hip_rc != hipSuccess)
        return ssim_hip_rc(hip_rc);

    hip_rc =
        hipMemcpy2DAsync(s->cmp_in, (size_t)row_w, dist_pic->data[0], (size_t)dist_pic->stride[0],
                         (size_t)row_w, (size_t)s->height, hipMemcpyHostToDevice, str);
    if (hip_rc != hipSuccess)
        return ssim_hip_rc(hip_rc);

    int err = ssim_hip_launch_horiz(s, str);
    if (err != 0)
        return err;

    return ssim_hip_launch_vert_readback(s, str);
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
    SsimStateHip *s = fex->priv;

    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

    /* Accumulate per-block float partials in double, then divide by the
     * effective pixel count. Mirrors the CUDA twin's collect path. */
    const float *partials = (const float *)s->rb.host_pinned;
    double total = 0.0;
    for (unsigned i = 0; i < s->partials_count; i++)
        total += (double)partials[i];
    const double n_pixels = (double)s->w_final * (double)s->h_final;
    const double score = total / n_pixels;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "float_ssim", score, index);
#endif /* HAVE_HIPCC */
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"float_ssim", NULL};

/* Load-bearing: the feature extractor is registered via
 * `extern VmafFeatureExtractor vmaf_fex_float_ssim_hip;` in
 * `libvmaf/src/feature/feature_extractor.c`'s
 * `feature_extractor_list[]`. Making this static would unlink the
 * extractor from the registry and fail every name lookup. Same
 * pattern every CUDA / SYCL / Vulkan feature extractor uses (see
 * e.g. `vmaf_fex_float_ssim_cuda` in
 * `libvmaf/src/feature/cuda/integer_ssim_cuda.c`). */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_ssim_hip = {
    .name = "float_ssim_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(SsimStateHip),
    .provided_features = provided_features,
    /* VMAF_FEATURE_EXTRACTOR_HIP flag cleared until picture buffer-type
     * plumbing lands (T7-10c). Until then pictures arrive as CPU
     * VmafPictures and submit() does explicit HtoD copies.
     * Same posture as all other HIP consumers (ADR-0241 / ADR-0254 /
     * ADR-0259 / ADR-0260 / ADR-0266 / ADR-0267 / ADR-0273). */
    .flags = 0,
    /* 2 dispatches/frame (horiz + vert+combine). The horiz intermediate
     * buffers are filled per-frame — not a pure reduction, so
     * is_reduction_only = false. */
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
