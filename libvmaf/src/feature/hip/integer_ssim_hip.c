/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_ssim feature extractor on the HIP backend — eleventh
 *  kernel-template consumer.
 *
 *  Mirrors `libvmaf/src/feature/hip/float_ssim_hip.c` call-graph-for-
 *  call-graph. Two-pass design:
 *    Pass 1 — horizontal 11-tap separable Gaussian over ref / cmp /
 *             ref^2 / cmp^2 / ref*cmp into five intermediate float
 *             device buffers, grid sized over (W-10) x H.
 *    Pass 2 — vertical 11-tap + per-pixel SSIM combine + per-block
 *             float partial sum, grid sized over (W-10) x (H-10).
 *  Host accumulates partials in double, divides by (W-10)*(H-10) and
 *  emits `integer_ssim`.
 *
 *  The emitted feature name is `integer_ssim` (matching the CPU extractor
 *  in `libvmaf/src/feature/integer_ssim.c`).
 *
 *  HIP adaptation from float_ssim_hip:
 *  - Kernel symbol names prefixed `calculate_integer_ssim_hip_*` to
 *    avoid collision with the existing `calculate_ssim_hip_*` symbols
 *    in `float_ssim/ssim_score.hip`.
 *  - Emits `integer_ssim` instead of `float_ssim`.
 *  - HSACO blob symbol is `integer_ssim_score_hsaco` (distinct from
 *    `ssim_score_hsaco`).
 *
 *  When `enable_hipcc=false` (e.g. a CI agent without ROCm), `HAVE_HIPCC`
 *  is undefined and `init()` returns -ENOSYS — same scaffold contract as
 *  the pre-runtime posture.
 *
 *  v1: scale=1 only — same constraint as float_ssim_hip / ssim_vulkan /
 *  ssim_cuda. Auto-detect path rejects scale>1 with -EINVAL at init.
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
#include "integer_ssim_hip.h"

/* ------------------------------------------------------------------ */
/* Block geometry constants (match integer_ssim_score.hip)             */
/* ------------------------------------------------------------------ */

#define ISSIM_HIP_BLOCK_X 16u
#define ISSIM_HIP_BLOCK_Y 8u
#define ISSIM_HIP_K 11u

/* ------------------------------------------------------------------ */
/* HIP-to-errno translation                                            */
/* ------------------------------------------------------------------ */

static int issim_hip_rc(hipError_t rc)
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

typedef struct IssimStateHip {
    VmafHipKernelLifecycle lc;
    VmafHipKernelReadback rb; /* device: per-block float partials;
                                * host_pinned: readback slot */
    VmafHipContext *ctx;

    int scale_override;

    /* Five intermediate float device buffers for the horiz pass output.
     * Sized (w_horiz * h_horiz * sizeof(float)). Allocated via hipMalloc,
     * freed via hipFree. Raw void * — same pattern as float_ssim_hip. */
    void *d_ref_mu;
    void *d_cmp_mu;
    void *d_ref_sq;
    void *d_cmp_sq;
    void *d_refcmp;

    /* Staging buffers: CPU luma planes → device (HtoD). One each for
     * ref and cmp (luma-only, no chroma). Sized frame_w * frame_h * bpp. */
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
} IssimStateHip;

static const VmafOption options[] = {
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling). "
                "v1: GPU path requires scale=1; auto-detect rejects scale>1 with -EINVAL.",
        .offset = offsetof(IssimStateHip, scale_override),
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

static int issim_hip_round_to_int(float x)
{
    return (int)(x + (x < 0.0f ? -0.5f : 0.5f));
}

static int issim_hip_min_int(int a, int b)
{
    return a < b ? a : b;
}

static int issim_hip_compute_scale(unsigned w, unsigned h, int override_val)
{
    if (override_val > 0)
        return override_val;
    int scaled = issim_hip_round_to_int((float)issim_hip_min_int((int)w, (int)h) / 256.0f);
    return scaled < 1 ? 1 : scaled;
}

static int issim_hip_validate_dims(const IssimStateHip *s, unsigned w, unsigned h)
{
    int scale = issim_hip_compute_scale(w, h, s->scale_override);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_ssim_hip: v1 supports scale=1 only "
                 "(auto-detected scale=%d at %ux%u). "
                 "Pin --feature integer_ssim_hip:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }
    if (w < ISSIM_HIP_K || h < ISSIM_HIP_K) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_ssim_hip: input %ux%u smaller than 11x11 "
                 "Gaussian footprint.\n",
                 w, h);
        return -EINVAL;
    }
    return 0;
}

static void issim_hip_init_dims(IssimStateHip *s, unsigned w, unsigned h, unsigned bpc)
{
    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->w_horiz = w - (ISSIM_HIP_K - 1u);
    s->h_horiz = h;
    s->w_final = w - (ISSIM_HIP_K - 1u);
    s->h_final = h - (ISSIM_HIP_K - 1u);

    /* SSIM stability constants: L=255, K1=0.01, K2=0.03.
     * Matches the float_ssim_hip / ssim_cuda pin (L=255.0 regardless
     * of bpc) so the cross-backend numeric gate has nothing fork-
     * specific to track. */
    const float L = 255.0f;
    const float K1 = 0.01f;
    const float K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);

    const unsigned grid_x = (s->w_final + ISSIM_HIP_BLOCK_X - 1u) / ISSIM_HIP_BLOCK_X;
    const unsigned grid_y = (s->h_final + ISSIM_HIP_BLOCK_Y - 1u) / ISSIM_HIP_BLOCK_Y;
    s->partials_capacity = grid_x * grid_y;
}

/* ------------------------------------------------------------------ */
/* HAVE_HIPCC helpers                                                  */
/* ------------------------------------------------------------------ */

#ifdef HAVE_HIPCC

static int issim_hip_module_load(IssimStateHip *s)
{
    hipError_t hip_rc = hipModuleLoadData(&s->module, integer_ssim_score_hsaco);
    if (hip_rc != hipSuccess)
        return issim_hip_rc(hip_rc);

    hip_rc =
        hipModuleGetFunction(&s->func_horiz_8, s->module, "calculate_integer_ssim_hip_horiz_8bpc");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return issim_hip_rc(hip_rc);
    }
    hip_rc = hipModuleGetFunction(&s->func_horiz_16, s->module,
                                  "calculate_integer_ssim_hip_horiz_16bpc");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return issim_hip_rc(hip_rc);
    }
    hip_rc =
        hipModuleGetFunction(&s->func_vert, s->module, "calculate_integer_ssim_hip_vert_combine");
    if (hip_rc != hipSuccess) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
        return issim_hip_rc(hip_rc);
    }
    return 0;
}

static int issim_hip_bufs_alloc(IssimStateHip *s)
{
    const size_t horiz_bytes = (size_t)s->w_horiz * s->h_horiz * sizeof(float);
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const size_t stage_bytes = (size_t)s->width * s->height * bpp;

    hipError_t hip_rc;
    hip_rc = hipMalloc(&s->d_ref_mu, horiz_bytes);
    if (hip_rc != hipSuccess)
        return issim_hip_rc(hip_rc);
    hip_rc = hipMalloc(&s->d_cmp_mu, horiz_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return issim_hip_rc(hip_rc);
    }
    hip_rc = hipMalloc(&s->d_ref_sq, horiz_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->d_cmp_mu);
        s->d_cmp_mu = NULL;
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return issim_hip_rc(hip_rc);
    }
    hip_rc = hipMalloc(&s->d_cmp_sq, horiz_bytes);
    if (hip_rc != hipSuccess) {
        (void)hipFree(s->d_ref_sq);
        s->d_ref_sq = NULL;
        (void)hipFree(s->d_cmp_mu);
        s->d_cmp_mu = NULL;
        (void)hipFree(s->d_ref_mu);
        s->d_ref_mu = NULL;
        return issim_hip_rc(hip_rc);
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
        return issim_hip_rc(hip_rc);
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
        return issim_hip_rc(hip_rc);
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
        return issim_hip_rc(hip_rc);
    }
    return 0;
}

static void issim_hip_bufs_free(IssimStateHip *s)
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

static int issim_hip_launch_horiz(IssimStateHip *s, hipStream_t str)
{
    const unsigned grid_horiz_x = (s->w_horiz + ISSIM_HIP_BLOCK_X - 1u) / ISSIM_HIP_BLOCK_X;
    const unsigned grid_horiz_y = (s->h_horiz + ISSIM_HIP_BLOCK_Y - 1u) / ISSIM_HIP_BLOCK_Y;

    const ptrdiff_t ref_stride = (ptrdiff_t)(s->width * ((s->bpc <= 8u) ? 1u : 2u));
    hipError_t hip_rc;
    if (s->bpc == 8u) {
        void *args[] = {
            &s->ref_in,   (void *)&ref_stride, &s->cmp_in,   (void *)&ref_stride,
            &s->d_ref_mu, &s->d_cmp_mu,        &s->d_ref_sq, &s->d_cmp_sq,
            &s->d_refcmp, &s->w_horiz,         &s->h_horiz,
        };
        hip_rc =
            hipModuleLaunchKernel(s->func_horiz_8, grid_horiz_x, grid_horiz_y, 1u,
                                  ISSIM_HIP_BLOCK_X, ISSIM_HIP_BLOCK_Y, 1u, 0, str, args, NULL);
    } else {
        void *args[] = {
            &s->ref_in,   (void *)&ref_stride, &s->cmp_in,   (void *)&ref_stride,
            &s->d_ref_mu, &s->d_cmp_mu,        &s->d_ref_sq, &s->d_cmp_sq,
            &s->d_refcmp, &s->w_horiz,         &s->h_horiz,  &s->bpc,
        };
        hip_rc =
            hipModuleLaunchKernel(s->func_horiz_16, grid_horiz_x, grid_horiz_y, 1u,
                                  ISSIM_HIP_BLOCK_X, ISSIM_HIP_BLOCK_Y, 1u, 0, str, args, NULL);
    }
    return issim_hip_rc(hip_rc);
}

static int issim_hip_launch_vert_readback(IssimStateHip *s, hipStream_t str)
{
    const unsigned grid_x = (s->w_final + ISSIM_HIP_BLOCK_X - 1u) / ISSIM_HIP_BLOCK_X;
    const unsigned grid_y = (s->h_final + ISSIM_HIP_BLOCK_Y - 1u) / ISSIM_HIP_BLOCK_Y;

    void *args2[] = {
        &s->d_ref_mu, &s->d_cmp_mu, &s->d_ref_sq, &s->d_cmp_sq, &s->d_refcmp, &s->rb.device,
        &s->w_horiz,  &s->w_final,  &s->h_final,  &s->c1,       &s->c2,
    };
    hipError_t hip_rc = hipModuleLaunchKernel(s->func_vert, grid_x, grid_y, 1u, ISSIM_HIP_BLOCK_X,
                                              ISSIM_HIP_BLOCK_Y, 1u, 0, str, args2, NULL);
    if (hip_rc != hipSuccess)
        return issim_hip_rc(hip_rc);

    hip_rc = hipEventRecord((hipEvent_t)s->lc.submit, str);
    if (hip_rc != hipSuccess)
        return issim_hip_rc(hip_rc);

    const size_t copy_bytes = (size_t)s->partials_count * sizeof(float);
    hip_rc =
        hipMemcpyAsync(s->rb.host_pinned, s->rb.device, copy_bytes, hipMemcpyDeviceToHost, str);
    if (hip_rc != hipSuccess)
        return issim_hip_rc(hip_rc);

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
    IssimStateHip *s = fex->priv;

    int err = issim_hip_validate_dims(s, w, h);
    if (err != 0)
        return err;

    issim_hip_init_dims(s, w, h, bpc);

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
    err = issim_hip_module_load(s);
    if (err != 0)
        goto fail_after_rb;
    err = issim_hip_bufs_alloc(s);
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
        issim_hip_bufs_free(s);
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
    IssimStateHip *s = fex->priv;
    int rc = 0;

#ifdef HAVE_HIPCC
    issim_hip_bufs_free(s);
    if (s->module != NULL) {
        int e = issim_hip_rc(hipModuleUnload(s->module));
        s->module = NULL;
        if (rc == 0)
            rc = e;
    }
#endif

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
    IssimStateHip *s = fex->priv;
    s->index = index;
    const unsigned grid_x = (s->w_final + ISSIM_HIP_BLOCK_X - 1u) / ISSIM_HIP_BLOCK_X;
    const unsigned grid_y = (s->h_final + ISSIM_HIP_BLOCK_Y - 1u) / ISSIM_HIP_BLOCK_Y;
    s->partials_count = grid_x * grid_y;

    const hipStream_t str = (hipStream_t)s->lc.str;
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t row_w = (ptrdiff_t)(s->width * bpp);

    hipError_t hip_rc =
        hipMemcpy2DAsync(s->ref_in, (size_t)row_w, ref_pic->data[0], (size_t)ref_pic->stride[0],
                         (size_t)row_w, (size_t)s->height, hipMemcpyHostToDevice, str);
    if (hip_rc != hipSuccess)
        return issim_hip_rc(hip_rc);

    hip_rc =
        hipMemcpy2DAsync(s->cmp_in, (size_t)row_w, dist_pic->data[0], (size_t)dist_pic->stride[0],
                         (size_t)row_w, (size_t)s->height, hipMemcpyHostToDevice, str);
    if (hip_rc != hipSuccess)
        return issim_hip_rc(hip_rc);

    int err = issim_hip_launch_horiz(s, str);
    if (err != 0)
        return err;

    return issim_hip_launch_vert_readback(s, str);
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
    IssimStateHip *s = fex->priv;

    int err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (err != 0)
        return err;

    const float *partials = (const float *)s->rb.host_pinned;
    double total = 0.0;
    for (unsigned i = 0; i < s->partials_count; i++)
        total += (double)partials[i];
    const double n_pixels = (double)s->w_final * (double)s->h_final;
    const double score = total / n_pixels;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_ssim", score, index);
#endif /* HAVE_HIPCC */
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"integer_ssim", NULL};

/* Load-bearing: registered via `extern VmafFeatureExtractor
 * vmaf_fex_integer_ssim_hip;` in feature_extractor.c. Same pattern as
 * every other HIP / CUDA / SYCL / Vulkan extractor. */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_ssim_hip = {
    .name = "integer_ssim_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(IssimStateHip),
    .provided_features = provided_features,
    /* VMAF_FEATURE_EXTRACTOR_HIP flag cleared (T7-10b posture): pictures
     * arrive as CPU VmafPictures; submit() does explicit HtoD copies.
     * Same posture as all other HIP consumers. */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
