/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ssim feature extractor on the CUDA backend
 *  (T7-23 / ADR-0188 / ADR-0189, GPU long-tail batch 2 part 1b).
 *  CUDA twin of ssim_vulkan (PR #139). Two-pass design mirrors
 *  the GLSL shader: horizontal 11-tap separable Gaussian over
 *  ref / cmp / ref² / cmp² / ref·cmp into 5 intermediate float
 *  buffers, then vertical 11-tap + per-pixel SSIM combine +
 *  per-block float partial sums. Host accumulates partials in
 *  `double`, divides by (W-10)·(H-10) and emits `float_ssim`.
 *
 *  Mirrors the psnr_cuda submit/collect scaffolding and the
 *  ciede_cuda per-block-partials precision pattern.
 *
 *  v1: scale=1 only — same constraint as ssim_vulkan. Auto-
 *  decimation is rejected at init with -EINVAL.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common.h"
#include "common/alignment.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "cuda/integer_ssim_cuda.h"
#include "cuda/kernel_template.h"
#include "log.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"

#define SSIM_BLOCK_X 16
#define SSIM_BLOCK_Y 8
#define SSIM_K 11

typedef struct SsimStateCuda {
    /* Stream + event pair owned by `cuda/kernel_template.h` lifecycle
     * (ADR-0246). */
    VmafCudaKernelLifecycle lc;
    /* Per-block float partials: device + pinned host. Owned by the
     * template's readback bundle. */
    VmafCudaKernelReadback rb;

    CUfunction func_horiz_8;
    CUfunction func_horiz_16;
    CUfunction func_vert;
    int scale_override;

    /* 5 intermediate float buffers — kept outside the template's
     * readback bundle since the bundle models a single device+host
     * pair, not a 5-buffer pyramid. */
    VmafCudaBuffer *h_ref_mu;
    VmafCudaBuffer *h_cmp_mu;
    VmafCudaBuffer *h_ref_sq;
    VmafCudaBuffer *h_cmp_sq;
    VmafCudaBuffer *h_refcmp;
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
} SsimStateCuda;

static int round_to_int(float x)
{
    return (int)(x + (x < 0.0f ? -0.5f : 0.5f));
}
static int min_int(int a, int b)
{
    return a < b ? a : b;
}

static int compute_scale(unsigned w, unsigned h, int override)
{
    if (override > 0)
        return override;
    int scaled = round_to_int((float)min_int((int)w, (int)h) / 256.0f);
    return scaled < 1 ? 1 : scaled;
}

static const VmafOption options[] = {
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling). "
                "v1: GPU path requires scale=1; auto-detect rejects scale>1 with -EINVAL.",
        .offset = offsetof(SsimStateCuda, scale_override),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    {0},
};

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    SsimStateCuda *s = fex->priv;

    int scale = compute_scale(w, h, s->scale_override);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_cuda: v1 supports scale=1 only (auto-detected scale=%d at %ux%u). "
                 "Pin --feature float_ssim_cuda:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }
    if (w < SSIM_K || h < SSIM_K) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ssim_cuda: input %ux%u smaller than 11x11 Gaussian footprint.\n", w, h);
        return -EINVAL;
    }

    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    CudaFunctions *cu_f = fex->cu_state->f;
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, ssim_score_ptx), fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_horiz_8, module, "calculate_ssim_horiz_8bpc"), fail);
    CHECK_CUDA_GOTO(
        cu_f, cuModuleGetFunction(&s->func_horiz_16, module, "calculate_ssim_horiz_16bpc"), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_vert, module, "calculate_ssim_vert_combine"),
                    fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->w_horiz = w - (SSIM_K - 1);
    s->h_horiz = h;
    s->w_final = w - (SSIM_K - 1);
    s->h_final = h - (SSIM_K - 1);
    const float L = 255.0f, K1 = 0.01f, K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);

    const unsigned grid_x = (s->w_final + SSIM_BLOCK_X - 1) / SSIM_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_BLOCK_Y - 1) / SSIM_BLOCK_Y;
    s->partials_capacity = grid_x * grid_y;
    const size_t horiz_bytes = (size_t)s->w_horiz * s->h_horiz * sizeof(float);
    const size_t partials_bytes = (size_t)s->partials_capacity * sizeof(float);

    int ret = 0;
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_ref_mu, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_cmp_mu, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_ref_sq, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_cmp_sq, horiz_bytes);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_refcmp, horiz_bytes);
    if (ret)
        goto free_ref;

    ret = vmaf_cuda_kernel_readback_alloc(&s->rb, fex->cu_state, partials_bytes);
    if (ret)
        goto free_ref;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict) {
        ret = -ENOMEM;
        goto free_ref;
    }
    return 0;

free_ref:
    if (s->h_ref_mu)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_mu);
    if (s->h_cmp_mu)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_mu);
    if (s->h_ref_sq)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_sq);
    if (s->h_cmp_sq)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_sq);
    if (s->h_refcmp)
        (void)vmaf_cuda_buffer_free(fex->cu_state, s->h_refcmp);
    (void)vmaf_cuda_kernel_readback_free(&s->rb, fex->cu_state);
    (void)vmaf_dictionary_free(&s->feature_name_dict);
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    return ret;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    return _cuda_err;
}

static int submit_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    SsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->index = index;
    const unsigned grid_x = (s->w_final + SSIM_BLOCK_X - 1) / SSIM_BLOCK_X;
    const unsigned grid_y = (s->h_final + SSIM_BLOCK_Y - 1) / SSIM_BLOCK_Y;
    s->partials_count = grid_x * grid_y;

    /* Sync ref-side stream against dist's ready event (matches
     * psnr_cuda's pattern). */
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(vmaf_cuda_picture_get_stream(ref_pic),
                                              vmaf_cuda_picture_get_ready_event(dist_pic),
                                              CU_EVENT_WAIT_DEFAULT));

    /* Pass 1 — horizontal. Grid sized over (W-10) × H. */
    const unsigned grid_horiz_x = (s->w_horiz + SSIM_BLOCK_X - 1) / SSIM_BLOCK_X;
    const unsigned grid_horiz_y = (s->h_horiz + SSIM_BLOCK_Y - 1) / SSIM_BLOCK_Y;
    CUstream stream = vmaf_cuda_picture_get_stream(ref_pic);
    if (s->bpc == 8) {
        unsigned width = s->width;
        void *params[] = {
            (void *)ref_pic,     (void *)dist_pic,
            (void *)s->h_ref_mu, (void *)s->h_cmp_mu,
            (void *)s->h_ref_sq, (void *)s->h_cmp_sq,
            (void *)s->h_refcmp, &s->w_horiz,
            &s->h_horiz,         &width,
        };
        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_horiz_8, grid_horiz_x, grid_horiz_y, 1,
                                         SSIM_BLOCK_X, SSIM_BLOCK_Y, 1, 0, stream, params, NULL));
    } else {
        unsigned bpc = s->bpc;
        unsigned width = s->width;
        void *params[] = {
            (void *)ref_pic,
            (void *)dist_pic,
            (void *)s->h_ref_mu,
            (void *)s->h_cmp_mu,
            (void *)s->h_ref_sq,
            (void *)s->h_cmp_sq,
            (void *)s->h_refcmp,
            &s->w_horiz,
            &s->h_horiz,
            &bpc,
            &width,
        };
        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_horiz_16, grid_horiz_x, grid_horiz_y, 1,
                                         SSIM_BLOCK_X, SSIM_BLOCK_Y, 1, 0, stream, params, NULL));
    }

    /* Pass 2 — vertical + SSIM combine. Grid sized over
     * (W-10) × (H-10). The horiz pass writes happen-before
     * the vert pass reads on the same stream — implicit
     * stream ordering, no extra event needed. */
    void *params2[] = {
        (void *)s->h_ref_mu,
        (void *)s->h_cmp_mu,
        (void *)s->h_ref_sq,
        (void *)s->h_cmp_sq,
        (void *)s->h_refcmp,
        (void *)s->rb.device,
        &s->w_horiz,
        &s->w_final,
        &s->h_final,
        &s->c1,
        &s->c2,
    };
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_vert, grid_x, grid_y, 1, SSIM_BLOCK_X,
                                           SSIM_BLOCK_Y, 1, 0, stream, params2, NULL));

    /* DtoH copy of the partials on our private stream. */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.submit, stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->lc.submit, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA_RETURN(cu_f,
                      cuMemcpyDtoHAsync(s->rb.host_pinned, (CUdeviceptr)s->rb.device->data,
                                        (size_t)s->partials_count * sizeof(float), s->lc.str));
    return vmaf_cuda_kernel_submit_post_record(&s->lc, fex->cu_state);
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    SsimStateCuda *s = fex->priv;

    int sync_err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (sync_err)
        return sync_err;

    const float *partials_host = s->rb.host_pinned;
    double total = 0.0;
    for (unsigned i = 0; i < s->partials_count; i++)
        total += (double)partials_host[i];
    const double n_pixels = (double)s->w_final * (double)s->h_final;
    const double score = total / n_pixels;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "float_ssim", score, index);
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    SsimStateCuda *s = fex->priv;

    int rc = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);

    if (s->h_ref_mu) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_mu);
        free(s->h_ref_mu);
        if (rc == 0)
            rc = e;
    }
    if (s->h_cmp_mu) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_mu);
        free(s->h_cmp_mu);
        if (rc == 0)
            rc = e;
    }
    if (s->h_ref_sq) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_sq);
        free(s->h_ref_sq);
        if (rc == 0)
            rc = e;
    }
    if (s->h_cmp_sq) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_sq);
        free(s->h_cmp_sq);
        if (rc == 0)
            rc = e;
    }
    if (s->h_refcmp) {
        const int e = vmaf_cuda_buffer_free(fex->cu_state, s->h_refcmp);
        free(s->h_refcmp);
        if (rc == 0)
            rc = e;
    }
    const int rb_rc = vmaf_cuda_kernel_readback_free(&s->rb, fex->cu_state);
    if (rc == 0)
        rc = rb_rc;
    const int dict_rc = vmaf_dictionary_free(&s->feature_name_dict);
    if (rc == 0)
        rc = dict_rc;
    return rc;
}

static const char *provided_features[] = {"float_ssim", NULL};

VmafFeatureExtractor vmaf_fex_float_ssim_cuda = {
    .name = "float_ssim_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(SsimStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    .chars =
        {
            .n_dispatches_per_frame = 2,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
