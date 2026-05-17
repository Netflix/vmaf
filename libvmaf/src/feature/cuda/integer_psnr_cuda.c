/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature extractor on the CUDA backend (T7-23 / ADR-0182,
 *  GPU long-tail batch 1b; chroma extension T3-15(a) first port,
 *  2026-05-09 — see research digest
 *  `docs/research/0090-t3-15-gpu-coverage-long-tail-2026-05-09.md`
 *  and the Vulkan precedent in
 *  [ADR-0216](../../docs/adr/0216-vulkan-chroma-psnr.md)).
 *
 *  Per-pixel squared-error reduction → host-side log10 → score.
 *  Mirrors the Vulkan psnr_vulkan.c host scaffolding (chroma-extended
 *  in PR #204) but uses CUDA's async submit/collect model (parallel
 *  with motion_cuda.c). One dispatch per plane (Y, Cb, Cr); the same
 *  `calculate_psnr_kernel_{8,16}bpc` entry point is invoked three
 *  times per frame against per-plane (w, h) pairs and a `plane`
 *  argument that selects `ref.data[plane]` / `dis.data[plane]`.
 *  Chroma buffers are sized per the active subsampling
 *  (4:2:0 → w/2 × h/2, 4:2:2 → w/2 × h, 4:4:4 → w × h); the kernel
 *  is plane-agnostic and reads its plane index out of the new
 *  argument.
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_psnr.c::extract):
 *      sse = sum_{i,j} (ref[i,j] - dis[i,j])^2;        (per channel)
 *      mse = sse / (w_p * h_p);
 *      psnr = (sse <= 0)
 *             ? psnr_max[p]
 *             : MIN(10 * log10(peak * peak / mse), psnr_max[p]);
 *  Bit-exactness contract: int64 SSE accumulation → places=4 vs CPU.
 *
 *  4:0:0 (YUV400) handling: chroma planes are absent, so only the
 *  luma plane is dispatched and only `psnr_y` is emitted. This
 *  matches CPU integer_psnr.c::init's `enable_chroma = false`
 *  branch.
 *
 *  Reference consumer of `cuda/kernel_template.h` (ADR-0246) — the
 *  per-frame async lifecycle (private stream + submit/finished event
 *  pair) and the (device, pinned-host) readback pair are dispensed
 *  by the template instead of being open-coded here. Multi-plane
 *  PSNR is the exact use case the template's docstring (lines
 *  115-117) calls out: "For metrics with multi-word accumulators
 *  (multi-plane PSNR, ssimulacra2 multi-band), allocate one
 *  VmafCudaKernelReadback per slot."
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
#include "cuda/integer_psnr_cuda.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "cuda_helper.cuh"
#include "kernel_template.h"

#define PSNR_NUM_PLANES 3U

typedef struct PsnrStateCuda {
    /* Lifecycle (private stream + submit/finished event pair) is shared
     * across all three plane dispatches — they execute on the same
     * stream sequentially. */
    VmafCudaKernelLifecycle lc;
    /* Per-plane readback slots (one (device SSE accumulator, pinned host)
     * pair per plane). The template's docstring explicitly authorises
     * multi-readback layouts for multi-plane reductions. */
    VmafCudaKernelReadback rb[PSNR_NUM_PLANES];
    CUfunction funcbpc8;
    CUfunction funcbpc16;
    unsigned index;
    unsigned width[PSNR_NUM_PLANES];
    unsigned height[PSNR_NUM_PLANES];
    unsigned bpc;
    uint32_t peak;
    /* `enable_chroma` option: when false, only luma is dispatched.
     * Default true mirrors CPU integer_psnr.c — see ADR-0453. */
    bool enable_chroma;
    /* Number of active planes (1 for YUV400, 3 otherwise). */
    unsigned n_planes;
    /* Per-plane psnr_max — `(6 * bpc) + 12` in the default branch
     * (CPU integer_psnr.c::init's `min_sse == 0.0` path). The array
     * layout leaves `min_sse`-driven per-plane formulas a one-line
     * change away. */
    double psnr_max[PSNR_NUM_PLANES];
    VmafDictionary *feature_name_dict;
} PsnrStateCuda;

static const VmafOption options[] = {{
                                         .name = "enable_chroma",
                                         .help = "enable calculation for chroma channels",
                                         .offset = offsetof(PsnrStateCuda, enable_chroma),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = true,
                                     },
                                     {0}};

static int psnr_cuda_dispatch(const VmafPicture *ref, const VmafPicture *dis, VmafCudaBuffer *sse,
                              unsigned width, unsigned height, unsigned plane, unsigned bpc,
                              CUfunction funcbpc8, CUfunction funcbpc16, CudaFunctions *cu_f,
                              CUstream stream)
{
    const int block_dim_x = 16;
    const int block_dim_y = 16;
    const int grid_dim_x = DIV_ROUND_UP(width, block_dim_x);
    const int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

    void *kernelParams[] = {(void *)ref, (void *)dis, (void *)sse, &width, &height, &plane};
    CUfunction func = (bpc == 8) ? funcbpc8 : funcbpc16;
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(func, grid_dim_x, grid_dim_y, 1, block_dim_x,
                                           block_dim_y, 1, 0, stream, kernelParams, NULL));
    return 0;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    /* Per-plane geometry derived from pix_fmt. CPU reference:
     * libvmaf/src/feature/integer_psnr.c::init computes the same
     * (ss_hor, ss_ver) split. YUV400 has chroma absent, so n_planes = 1. */
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
    /* Mirror CPU integer_psnr.c::init's enable_chroma guard (ADR-0453):
     * when the caller passes enable_chroma=false, skip chroma dispatches
     * identically to the YUV400 path above. YUV400 already forces
     * n_planes=1, so this only activates for 4:2:0/4:2:2/4:4:4. */
    if (!s->enable_chroma && s->n_planes > 1U) {
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    }

    /* Stream + event pair via the template — replaces the
     * cuCtxPushCurrent → cuStreamCreateWithPriority → cuEventCreate ×2
     * → cuCtxPopCurrent block every CUDA feature kernel hand-rolled. */
    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    /* Module load + function lookups stay per-feature (each metric
     * has its own .ptx blob and entry-point names). */
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, psnr_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc8, module, "calculate_psnr_kernel_8bpc"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->funcbpc16, module, "calculate_psnr_kernel_16bpc"),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;
    /* Match CPU integer_psnr.c::init's psnr_max default branch
     * (`min_sse == 0.0`): psnr_max[p] = (6 * bpc) + 12. */
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++)
        s->psnr_max[p] = (double)(6U * bpc) + 12.0;

    /* Per-plane readback pairs (device SSE accumulator + pinned host
     * slot) via the template. One pair per plane — matches the
     * template's documented multi-plane PSNR pattern. */
    for (unsigned p = 0; p < s->n_planes; p++) {
        err = vmaf_cuda_kernel_readback_alloc(&s->rb[p], fex->cu_state, sizeof(uint64_t));
        if (err)
            goto free_ref;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_ref;

    return 0;

free_ref:
    for (unsigned p = 0; p < s->n_planes; p++)
        (void)vmaf_cuda_kernel_readback_free(&s->rb[p], fex->cu_state);
    if (s->feature_name_dict) {
        (void)vmaf_dictionary_free(&s->feature_name_dict);
    }
    (void)vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    return -ENOMEM;

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
    PsnrStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    s->index = index;

    /* Pre-launch boilerplate is shared across all three plane
     * dispatches — zero each plane's device accumulator and wait
     * once on the dist-side ready event (the picture-stream wait is
     * a property of the picture, not the per-plane dispatch). The
     * template's `submit_pre_launch` does both; we call it once for
     * plane 0 and then zero the remaining planes' accumulators
     * directly on the same private stream. */
    int err = vmaf_cuda_kernel_submit_pre_launch(&s->lc, fex->cu_state, &s->rb[0],
                                                 vmaf_cuda_picture_get_stream(ref_pic),
                                                 vmaf_cuda_picture_get_ready_event(dist_pic));
    if (err)
        return err;
    for (unsigned p = 1; p < s->n_planes; p++) {
        CHECK_CUDA_RETURN(cu_f,
                          cuMemsetD8Async(s->rb[p].device->data, 0, s->rb[p].bytes, s->lc.str));
    }

    /* One dispatch per active plane against per-plane (w, h). All
     * three execute on the picture stream so the per-frame ordering
     * with motion_cuda.c et al. is preserved. */
    for (unsigned p = 0; p < s->n_planes; p++) {
        err = psnr_cuda_dispatch(ref_pic, dist_pic, s->rb[p].device, ref_pic->w[p], ref_pic->h[p],
                                 p, s->bpc, s->funcbpc8, s->funcbpc16, cu_f,
                                 vmaf_cuda_picture_get_stream(ref_pic));
        if (err)
            return err;
    }

    /* Post-launch readback: record submit on the picture stream, wait
     * for it on the private readback stream, DtoH copy each plane's
     * accumulator + record `finished`. The template documents this
     * exact sequence in its docstring; left inline for clarity since
     * the kernel launch + ref_pic stream are inherently per-feature. */
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->lc.submit, vmaf_cuda_picture_get_stream(ref_pic)));
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->lc.submit, CU_EVENT_WAIT_DEFAULT));
    for (unsigned p = 0; p < s->n_planes; p++) {
        CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(s->rb[p].host_pinned,
                                                  (CUdeviceptr)s->rb[p].device->data,
                                                  s->rb[p].bytes, s->lc.str));
    }
    return vmaf_cuda_kernel_submit_post_record(&s->lc, fex->cu_state);
}

/* psnr_name[p] — same array as the CPU path
 * (libvmaf/src/feature/integer_psnr.c::psnr_name). */
static const char *const psnr_name[PSNR_NUM_PLANES] = {"psnr_y", "psnr_cb", "psnr_cr"};

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    PsnrStateCuda *s = fex->priv;

    /* Drain the private readback stream so the host pinned buffers are
     * safe to read. One drain covers all three plane accumulators. */
    int err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (err)
        return err;

    int rc = 0;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const double sse = (double)*(uint64_t *)s->rb[p].host_pinned;
        const double n_pixels = (double)s->width[p] * (double)s->height[p];
        const double mse = sse / n_pixels;
        /* Match CPU integer_psnr.c::extract — clamp at psnr_max[p]
         * via MIN(10*log10(peak^2 / max(mse, 1e-16)), psnr_max[p]).
         * The 1e-16 floor guards against sse == 0 (trivially identical
         * frames); the CPU path uses the same constant. */
        const double peak_sq = (double)s->peak * (double)s->peak;
        const double mse_clamped = (mse > 1e-16) ? mse : 1e-16;
        double psnr = 10.0 * log10(peak_sq / mse_clamped);
        if (psnr > s->psnr_max[p])
            psnr = s->psnr_max[p];

        const int e = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, psnr_name[p], psnr, index);
        if (e && rc == 0)
            rc = e;
    }
    return rc;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    PsnrStateCuda *s = fex->priv;

    /* Lifecycle teardown via the template (sync → destroy stream →
     * destroy events). Best-effort error aggregation matches the
     * old hand-rolled CHECK_CUDA_GOTO chain. */
    int rc = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);
    for (unsigned p = 0; p < s->n_planes; p++) {
        const int err = vmaf_cuda_kernel_readback_free(&s->rb[p], fex->cu_state);
        if (err && rc == 0)
            rc = err;
    }
    const int err = vmaf_dictionary_free(&s->feature_name_dict);
    if (err && rc == 0)
        rc = err;
    return rc;
}

/* Provided features — full luma + chroma per the chroma extension
 * (T3-15(a) first port, 2026-05-09; mirrors Vulkan ADR-0216). For
 * YUV400 sources `init` clamps `n_planes` to 1 and chroma dispatches
 * are skipped at runtime, but the static list still claims chroma so
 * the dispatcher routes `psnr_cb` / `psnr_cr` requests through the
 * CUDA twin. */
static const char *provided_features[] = {"psnr_y", "psnr_cb", "psnr_cr", NULL};

VmafFeatureExtractor vmaf_fex_psnr_cuda = {
    .name = "psnr_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(PsnrStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    /* 3 dispatches/frame (one per plane), reduction-dominated; AUTO +
     * 1080p area matches motion's profile (see ADR-0181 / ADR-0182).
     * Three small dispatches are still well under the threshold where
     * batching pays off vs. AUTO scheduling. */
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
