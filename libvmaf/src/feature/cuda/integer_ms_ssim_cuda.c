/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ms_ssim feature extractor on the CUDA backend
 *  (T7-23 / ADR-0188 / ADR-0190, GPU long-tail batch 2 part 2b).
 *  CUDA twin of ms_ssim_vulkan (PR #141).
 *
 *  5-level pyramid + 3-output SSIM per scale + host-side Wang
 *  product combine. Three CUDA kernels (see ms_ssim_score.cu):
 *
 *    1. ms_ssim_decimate — 9-tap 9/7 biorthogonal LPF + 2×
 *       downsample (mirrors ms_ssim_decimate.c byte-for-byte).
 *    2. ms_ssim_horiz — horizontal 11-tap separable Gaussian
 *       over 5 SSIM stats (operates on float input — pyramid
 *       levels are already float).
 *    3. ms_ssim_vert_lcs — vertical 11-tap + per-pixel l/c/s +
 *       per-block float partials × 3.
 *
 *  picture_copy normalisation runs on the host (uint sample →
 *  float in [0, 255]), uploaded to the pyramid level 0 buffer.
 *  CUDA decimate kernels build levels 1-4. Per-scale SSIM
 *  compute reads levels and writes into shared intermediate +
 *  per-scale partials buffers; host accumulates partials in
 *  `double` per scale and applies the Wang weights for the final
 *  product combine.
 *
 *  Min-dim guard: 11 << 4 = 176 (matches ADR-0153).
 *
 *  enable_lcs (T7-35 / ADR-0243): when set, emits the 15 extra
 *  per-scale metrics float_ms_ssim_{l,c,s}_scale{0..4}. The
 *  vert_lcs kernel already produces the per-scale L/C/S means
 *  (it's where the "_lcs" in its name comes from); gating the
 *  feature_collector_append calls leaves the default-path
 *  output bit-identical to the pre-T7-35 binary.
 *
 *  Engine-scope fence batching (T-GPU-OPT-2 / ADR-0271): all 5
 *  scales' horiz + vert_lcs launches and DtoH partial readbacks
 *  are enqueued in submit() onto the lifecycle's private stream
 *  (s->lc.str). Same-stream ordering serialises kernels and
 *  copies in dependency order without per-scale syncs, and the
 *  partials buffers are now allocated per-scale to avoid the
 *  cross-scale aliasing that previously forced a host-blocking
 *  cuStreamSynchronize after each scale. The final
 *  cuEventRecord(s->lc.finished, s->lc.str) opts the lifecycle
 *  into the engine's drain batch (drain_batch.h) so the
 *  per-frame readbacks coalesce with the rest of the CUDA
 *  feature stack — collect() then becomes a host-side reduction
 *  only.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common.h"
#include "common/alignment.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "cuda/drain_batch.h"
#include "cuda/integer_ms_ssim_cuda.h"
#include "cuda/kernel_template.h"
#include "log.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "picture_copy.h"
#include "cuda_helper.cuh"

#define MS_SSIM_SCALES 5
#define MS_SSIM_GAUSSIAN_LEN 11
#define MS_SSIM_K 11
#define MS_SSIM_BLOCK_X 16
#define MS_SSIM_BLOCK_Y 8

static const float g_alphas[MS_SSIM_SCALES] = {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f};
static const float g_betas[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};
static const float g_gammas[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};

typedef struct MsSsimStateCuda {
    /* Stream + event pair owned by `cuda/kernel_template.h` lifecycle
     * (ADR-0246). Multi-buffer pyramid state stays outside the
     * template's single-pair readback bundle. */
    VmafCudaKernelLifecycle lc;
    CUfunction func_decimate;
    CUfunction func_horiz;
    CUfunction func_vert_lcs;

    unsigned width;
    unsigned height;
    unsigned bpc;

    unsigned scale_w[MS_SSIM_SCALES];
    unsigned scale_h[MS_SSIM_SCALES];
    unsigned scale_w_horiz[MS_SSIM_SCALES];
    unsigned scale_h_horiz[MS_SSIM_SCALES];
    unsigned scale_w_final[MS_SSIM_SCALES];
    unsigned scale_h_final[MS_SSIM_SCALES];
    unsigned scale_grid_x[MS_SSIM_SCALES];
    unsigned scale_grid_y[MS_SSIM_SCALES];
    unsigned scale_block_count[MS_SSIM_SCALES];

    float c1;
    float c2;
    float c3;

    /* Pyramid: 5 levels × ref + cmp, all float. */
    VmafCudaBuffer *pyramid_ref[MS_SSIM_SCALES];
    VmafCudaBuffer *pyramid_cmp[MS_SSIM_SCALES];

    /* Pinned host buffer for picture_copy → upload at scale 0. */
    float *h_ref;
    float *h_cmp;

    /* SSIM intermediates sized for scale 0 (largest). */
    VmafCudaBuffer *h_ref_mu;
    VmafCudaBuffer *h_cmp_mu;
    VmafCudaBuffer *h_ref_sq;
    VmafCudaBuffer *h_cmp_sq;
    VmafCudaBuffer *h_refcmp;

    /* Per-scale partials buffers (T-GPU-OPT-2 / ADR-0271).
     * Previously a single buffer reused across scales; that aliasing
     * forced a per-scale cuStreamSynchronize before the host could
     * walk the partials. Allocating per scale lets all 5 scales'
     * horiz + vert_lcs launches and DtoH copies queue back-to-back
     * on s->lc.str so the readbacks coalesce with the engine's
     * drain batch. */
    VmafCudaBuffer *l_partials[MS_SSIM_SCALES];
    VmafCudaBuffer *c_partials[MS_SSIM_SCALES];
    VmafCudaBuffer *s_partials[MS_SSIM_SCALES];
    /* Pinned host partials for DtoH (per scale; safe to read after
     * the lifecycle's finished event has been waited on, either
     * via the engine's drain_batch or the legacy per-stream sync). */
    float *h_l_partials[MS_SSIM_SCALES];
    float *h_c_partials[MS_SSIM_SCALES];
    float *h_s_partials[MS_SSIM_SCALES];

    unsigned index;
    VmafDictionary *feature_name_dict;

    bool enable_lcs;    /* T7-35 / ADR-0243: emit per-scale L/C/S triples. */
    bool enable_chroma; /* mirrors SSIM CUDA PR #950; MS-SSIM is luma-only
                         * by construction, so n_planes is clamped to 1
                         * unless a future extension lifts that. */
    unsigned n_planes;  /* active plane count: always 1 once clamped. */
} MsSsimStateCuda;

static const VmafOption options[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(MsSsimStateCuda, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_chroma",
        .help = "enable MS-SSIM calculation for chroma channels (Cb and Cr); "
                "currently MS-SSIM is luma-only, so this flag is accepted but "
                "always clamps n_planes to 1 until a chroma extension lands",
        .offset = offsetof(MsSsimStateCuda, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {0},
};

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    MsSsimStateCuda *s = fex->priv;

    /* Clamp n_planes to 1: MS-SSIM is luma-only by construction (Wang et al.
     * 2003 pyramid is applied to the luma plane only).  Mirror the SSIM CUDA
     * PR #950 pattern so the option is accepted without error for callers that
     * set enable_chroma=true for consistency with sister extractors; the guard
     * also covers YUV400P where chroma planes are absent. */
    if (pix_fmt == VMAF_PIX_FMT_YUV400P || !s->enable_chroma)
        s->n_planes = 1U;
    else
        s->n_planes = 1U; /* reserved: MS-SSIM chroma extension not yet impl. */

    /* ADR-0153 minimum resolution check. */
    const unsigned min_dim = (unsigned)MS_SSIM_GAUSSIAN_LEN << (MS_SSIM_SCALES - 1);
    if (w < min_dim || h < min_dim) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ms_ssim_cuda: input %ux%u too small; %d-level %d-tap MS-SSIM pyramid needs"
                 " >= %ux%u (Netflix#1414 / ADR-0153)\n",
                 w, h, MS_SSIM_SCALES, MS_SSIM_GAUSSIAN_LEN, min_dim, min_dim);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;

    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < MS_SSIM_SCALES; i++) {
        s->scale_w[i] = (s->scale_w[i - 1] / 2) + (s->scale_w[i - 1] & 1);
        s->scale_h[i] = (s->scale_h[i - 1] / 2) + (s->scale_h[i - 1] & 1);
    }
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        s->scale_w_horiz[i] = s->scale_w[i] - (MS_SSIM_K - 1);
        s->scale_h_horiz[i] = s->scale_h[i];
        s->scale_w_final[i] = s->scale_w[i] - (MS_SSIM_K - 1);
        s->scale_h_final[i] = s->scale_h[i] - (MS_SSIM_K - 1);
        s->scale_grid_x[i] =
            (s->scale_w_final[i] + (unsigned)MS_SSIM_BLOCK_X - 1) / (unsigned)MS_SSIM_BLOCK_X;
        s->scale_grid_y[i] =
            (s->scale_h_final[i] + (unsigned)MS_SSIM_BLOCK_Y - 1) / (unsigned)MS_SSIM_BLOCK_Y;
        s->scale_block_count[i] = s->scale_grid_x[i] * s->scale_grid_y[i];
    }

    const float L = 255.0f, K1 = 0.01f, K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);
    s->c3 = s->c2 * 0.5f;

    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    CudaFunctions *cu_f = fex->cu_state->f;
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, ms_ssim_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_decimate, module, "ms_ssim_decimate"), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_horiz, module, "ms_ssim_horiz"), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_vert_lcs, module, "ms_ssim_vert_lcs"), fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    /* Allocate device buffers. */
    int ret = 0;
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        const size_t plane_bytes = (size_t)s->scale_w[i] * s->scale_h[i] * sizeof(float);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->pyramid_ref[i], plane_bytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->pyramid_cmp[i], plane_bytes);
    }
    const size_t horiz_bytes_max =
        (size_t)s->scale_w_horiz[0] * s->scale_h_horiz[0] * sizeof(float);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_ref_mu, horiz_bytes_max);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_cmp_mu, horiz_bytes_max);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_ref_sq, horiz_bytes_max);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_cmp_sq, horiz_bytes_max);
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->h_refcmp, horiz_bytes_max);

    /* Per-scale partials buffers (T-GPU-OPT-2 / ADR-0271). Each scale
     * sized to its own block_count so the device DtoH copies are
     * exactly one float per work block; no aliasing across scales. */
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        const size_t scale_bytes = (size_t)s->scale_block_count[i] * sizeof(float);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->l_partials[i], scale_bytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->c_partials[i], scale_bytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->s_partials[i], scale_bytes);
    }

    /* Pinned host buffers — input planes (scale 0) + per-scale
     * partials triples. Pinned host memory is the precondition for
     * cuMemcpyDtoHAsync to actually run async with the device side
     * (pageable host buffers force the driver to stage internally and
     * defeat the drain-batch coalesce). */
    const size_t input_bytes = (size_t)w * h * sizeof(float);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_ref, input_bytes);
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_cmp, input_bytes);
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        const size_t scale_bytes = (size_t)s->scale_block_count[i] * sizeof(float);
        ret |=
            vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_l_partials[i], scale_bytes);
        ret |=
            vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_c_partials[i], scale_bytes);
        ret |=
            vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_s_partials[i], scale_bytes);
    }

    if (ret)
        return -ENOMEM;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;

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
    MsSsimStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->index = index;

    /* picture_copy host-side (matches CPU float_ms_ssim's flow):
     * normalise uint sample → float in [0, 255]. The pic argument
     * here is the device-side VmafPicture from the CUDA picture
     * pipeline, with its data[0] pointing at pitched device
     * memory (cuMemAllocPitch — stride[0] ≥ width*bpc). Use
     * cuMemcpy2DAsync to honour the device pitch on D2H and
     * produce a contiguous host buffer that picture_copy can
     * walk with stride = width*bpc. */
    const unsigned bpc_bytes = (s->bpc <= 8 ? 1u : 2u);
    const size_t input_bytes_uint = (size_t)s->width * s->height * bpc_bytes;
    void *tmp_uint = NULL;
    int ret = vmaf_cuda_buffer_host_alloc(fex->cu_state, &tmp_uint, input_bytes_uint);
    if (ret)
        return -ENOMEM;
    CUstream stream = vmaf_cuda_picture_get_stream(ref_pic);
    CUDA_MEMCPY2D m_ref = {0};
    m_ref.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m_ref.srcDevice = (CUdeviceptr)ref_pic->data[0];
    m_ref.srcPitch = (size_t)ref_pic->stride[0];
    m_ref.dstMemoryType = CU_MEMORYTYPE_HOST;
    m_ref.dstHost = tmp_uint;
    m_ref.dstPitch = (size_t)s->width * bpc_bytes;
    m_ref.WidthInBytes = (size_t)s->width * bpc_bytes;
    m_ref.Height = s->height;
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&m_ref, stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(stream));
    /* Build a fake VmafPicture wrapper for picture_copy. */
    VmafPicture host_pic_ref = {
        .pix_fmt = ref_pic->pix_fmt,
        .bpc = ref_pic->bpc,
        .w = {ref_pic->w[0], 0, 0},
        .h = {ref_pic->h[0], 0, 0},
        .stride = {(ptrdiff_t)((size_t)s->width * bpc_bytes), 0, 0},
        .data = {tmp_uint, NULL, NULL},
    };
    picture_copy(s->h_ref, (ptrdiff_t)((size_t)s->width * sizeof(float)), &host_pic_ref, 0,
                 ref_pic->bpc, 0);

    CUDA_MEMCPY2D m_cmp = m_ref;
    m_cmp.srcDevice = (CUdeviceptr)dist_pic->data[0];
    m_cmp.srcPitch = (size_t)dist_pic->stride[0];
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&m_cmp, stream));
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(stream));
    VmafPicture host_pic_cmp = host_pic_ref;
    host_pic_cmp.data[0] = tmp_uint;
    picture_copy(s->h_cmp, (ptrdiff_t)((size_t)s->width * sizeof(float)), &host_pic_cmp, 0,
                 dist_pic->bpc, 0);

    /* Upload normalised float planes to pyramid level 0. */
    const size_t input_bytes_float = (size_t)s->width * s->height * sizeof(float);
    CHECK_CUDA_RETURN(cu_f, cuMemcpyHtoDAsync((CUdeviceptr)s->pyramid_ref[0]->data, s->h_ref,
                                              input_bytes_float, s->lc.str));
    CHECK_CUDA_RETURN(cu_f, cuMemcpyHtoDAsync((CUdeviceptr)s->pyramid_cmp[0]->data, s->h_cmp,
                                              input_bytes_float, s->lc.str));
    (void)vmaf_cuda_buffer_host_free(fex->cu_state, tmp_uint);

    /* Build pyramid scales 1..4 via decimate kernel × ref + cmp. */
    for (int i = 0; i < MS_SSIM_SCALES - 1; i++) {
        const unsigned w_in = s->scale_w[i];
        const unsigned h_in = s->scale_h[i];
        const unsigned w_out = s->scale_w[i + 1];
        const unsigned h_out = s->scale_h[i + 1];
        const unsigned grid_x = (w_out + MS_SSIM_BLOCK_X - 1) / MS_SSIM_BLOCK_X;
        const unsigned grid_y = (h_out + MS_SSIM_BLOCK_Y - 1) / MS_SSIM_BLOCK_Y;
        for (int side = 0; side < 2; side++) {
            VmafCudaBuffer *src = (side == 0) ? s->pyramid_ref[i] : s->pyramid_cmp[i];
            VmafCudaBuffer *dst = (side == 0) ? s->pyramid_ref[i + 1] : s->pyramid_cmp[i + 1];
            void *params[] = {
                (void *)src,   (void *)dst,    (void *)&w_in,
                (void *)&h_in, (void *)&w_out, (void *)&h_out,
            };
            CHECK_CUDA_RETURN(cu_f,
                              cuLaunchKernel(s->func_decimate, grid_x, grid_y, 1, MS_SSIM_BLOCK_X,
                                             MS_SSIM_BLOCK_Y, 1, 0, s->lc.str, params, NULL));
        }
    }

    /* Per-scale SSIM compute + readback (T-GPU-OPT-2 / ADR-0271).
     * All 5 scales' horiz + vert_lcs launches and DtoH copies are
     * enqueued back-to-back on s->lc.str. CUDA serialises kernels +
     * copies on the same stream in submission order, so the shared
     * SSIM intermediates (h_ref_mu / h_cmp_mu / ...) are safely
     * read-after-write across scales without explicit sync; the
     * partials buffers are now per-scale (see init) so the DtoH
     * copies don't alias either. */
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        const unsigned width = s->scale_w[i];
        const unsigned w_horiz = s->scale_w_horiz[i];
        const unsigned h_horiz = s->scale_h_horiz[i];
        const unsigned w_final = s->scale_w_final[i];
        const unsigned h_final = s->scale_h_final[i];
        const unsigned horiz_grid_x = (w_horiz + MS_SSIM_BLOCK_X - 1) / MS_SSIM_BLOCK_X;
        const unsigned horiz_grid_y = (h_horiz + MS_SSIM_BLOCK_Y - 1) / MS_SSIM_BLOCK_Y;

        void *horiz_params[] = {
            (void *)s->pyramid_ref[i], (void *)s->pyramid_cmp[i],
            (void *)s->h_ref_mu,       (void *)s->h_cmp_mu,
            (void *)s->h_ref_sq,       (void *)s->h_cmp_sq,
            (void *)s->h_refcmp,       (void *)&width,
            (void *)&w_horiz,          (void *)&h_horiz,
        };
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_horiz, horiz_grid_x, horiz_grid_y, 1,
                                               MS_SSIM_BLOCK_X, MS_SSIM_BLOCK_Y, 1, 0, s->lc.str,
                                               horiz_params, NULL));

        void *vert_params[] = {
            (void *)s->h_ref_mu,      (void *)s->h_cmp_mu,      (void *)s->h_ref_sq,
            (void *)s->h_cmp_sq,      (void *)s->h_refcmp,      (void *)s->l_partials[i],
            (void *)s->c_partials[i], (void *)s->s_partials[i], (void *)&w_horiz,
            (void *)&w_final,         (void *)&h_final,         (void *)&s->c1,
            (void *)&s->c2,           (void *)&s->c3,
        };
        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_vert_lcs, s->scale_grid_x[i], s->scale_grid_y[i],
                                         1, MS_SSIM_BLOCK_X, MS_SSIM_BLOCK_Y, 1, 0, s->lc.str,
                                         vert_params, NULL));

        const size_t partials_bytes = (size_t)s->scale_block_count[i] * sizeof(float);
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync(s->h_l_partials[i], (CUdeviceptr)s->l_partials[i]->data,
                                            partials_bytes, s->lc.str));
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync(s->h_c_partials[i], (CUdeviceptr)s->c_partials[i]->data,
                                            partials_bytes, s->lc.str));
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync(s->h_s_partials[i], (CUdeviceptr)s->s_partials[i]->data,
                                            partials_bytes, s->lc.str));
    }

    /* Fence the final readback and opt the lifecycle into the
     * engine's drain batch. drain_batch_register is best-effort:
     * when no batch is open or the cap is hit, lc.drained stays
     * false and collect() falls back to the legacy per-stream
     * cuStreamSynchronize via vmaf_cuda_kernel_collect_wait. */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.finished, s->lc.str));
    (void)vmaf_cuda_drain_batch_register(&s->lc);
    return 0;
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    MsSsimStateCuda *s = fex->priv;

    /* Wait for all 5 scales' DtoH copies to land. Fast path
     * (T-GPU-OPT-2 / ADR-0271): when the engine has already drained
     * lc.finished as part of its batched flush, this is a no-op
     * (lc.drained is true → reset and return). Otherwise falls back
     * to cuStreamSynchronize(lc.str). */
    int wait_err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (wait_err)
        return wait_err;

    double l_means[MS_SSIM_SCALES] = {0}, c_means[MS_SSIM_SCALES] = {0},
           s_means[MS_SSIM_SCALES] = {0};
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        const unsigned w_final = s->scale_w_final[i];
        const unsigned h_final = s->scale_h_final[i];
        double total_l = 0.0, total_c = 0.0, total_s = 0.0;
        for (unsigned j = 0; j < s->scale_block_count[i]; j++) {
            total_l += (double)s->h_l_partials[i][j];
            total_c += (double)s->h_c_partials[i][j];
            total_s += (double)s->h_s_partials[i][j];
        }
        const double n_pixels = (double)w_final * (double)h_final;
        l_means[i] = total_l / n_pixels;
        c_means[i] = total_c / n_pixels;
        s_means[i] = total_s / n_pixels;
    }

    double msssim = 1.0;
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        msssim *= pow(l_means[i], (double)g_alphas[i]) * pow(c_means[i], (double)g_betas[i]) *
                  pow(fabs(s_means[i]), (double)g_gammas[i]);
    }

    int err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "float_ms_ssim", msssim, index);
    if (s->enable_lcs) {
        static const char *const l_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_l_scale0", "float_ms_ssim_l_scale1", "float_ms_ssim_l_scale2",
            "float_ms_ssim_l_scale3", "float_ms_ssim_l_scale4",
        };
        static const char *const c_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_c_scale0", "float_ms_ssim_c_scale1", "float_ms_ssim_c_scale2",
            "float_ms_ssim_c_scale3", "float_ms_ssim_c_scale4",
        };
        static const char *const s_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_s_scale0", "float_ms_ssim_s_scale1", "float_ms_ssim_s_scale2",
            "float_ms_ssim_s_scale3", "float_ms_ssim_s_scale4",
        };
        for (int i = 0; i < MS_SSIM_SCALES; i++) {
            err |= vmaf_feature_collector_append(feature_collector, l_names[i], l_means[i], index);
            err |= vmaf_feature_collector_append(feature_collector, c_names[i], c_means[i], index);
            err |= vmaf_feature_collector_append(feature_collector, s_names[i], s_means[i], index);
        }
    }
    return err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    MsSsimStateCuda *s = fex->priv;
    int ret = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        if (s->pyramid_ref[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->pyramid_ref[i]);
            free(s->pyramid_ref[i]);
        }
        if (s->pyramid_cmp[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->pyramid_cmp[i]);
            free(s->pyramid_cmp[i]);
        }
    }
    if (s->h_ref_mu) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_mu);
        free(s->h_ref_mu);
    }
    if (s->h_cmp_mu) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_mu);
        free(s->h_cmp_mu);
    }
    if (s->h_ref_sq) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_ref_sq);
        free(s->h_ref_sq);
    }
    if (s->h_cmp_sq) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_cmp_sq);
        free(s->h_cmp_sq);
    }
    if (s->h_refcmp) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->h_refcmp);
        free(s->h_refcmp);
    }
    /* Per-scale partials triples (T-GPU-OPT-2 / ADR-0271). */
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        if (s->l_partials[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->l_partials[i]);
            free(s->l_partials[i]);
            s->l_partials[i] = NULL;
        }
        if (s->c_partials[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->c_partials[i]);
            free(s->c_partials[i]);
            s->c_partials[i] = NULL;
        }
        if (s->s_partials[i]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->s_partials[i]);
            free(s->s_partials[i]);
            s->s_partials[i] = NULL;
        }
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {"float_ms_ssim", NULL};

VmafFeatureExtractor vmaf_fex_float_ms_ssim_cuda = {
    .name = "float_ms_ssim_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(MsSsimStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    .chars =
        {
            .n_dispatches_per_frame = 18,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
