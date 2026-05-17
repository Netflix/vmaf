/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  psnr_hvs feature extractor on the CUDA backend
 *  (T7-23 / ADR-0188 / ADR-0191, GPU long-tail batch 2 part 3b).
 *  CUDA twin of psnr_hvs_vulkan (PR #143).
 *
 *  Per-plane single-dispatch design — one CUDA block per output
 *  8×8 image block (step=7), 64 threads per block. Cooperative
 *  load + thread-0-serial reductions matching CPU's exact i,j
 *  summation order (same precision strategy as psnr_hvs.comp).
 *  Picture_copy host-side normalises uint sample → float in
 *  [0, 255] before D2H roundtrip with `cuMemcpy2DAsync` to honour
 *  the device pitch (matches the ms_ssim_cuda fix in PR #142).
 *
 *  3 dispatches per frame (Y, Cb, Cr). Combined `psnr_hvs =
 *  0.8·Y + 0.1·(Cb + Cr)` on the host.
 *
 *  Rejects YUV400P (no chroma) and `bpc > 12` (matches CPU).
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
#include "cuda/integer_psnr_hvs_cuda.h"
#include "cuda/kernel_template.h"
#include "log.h"
#include "mem.h"
#include "picture.h"
#include "picture_cuda.h"
#include "picture_copy.h"
#include "cuda_helper.cuh"

#define PSNR_HVS_BLOCK 8
#define PSNR_HVS_STEP 7
#define PSNR_HVS_NUM_PLANES 3
#define PSNR_HVS_BLOCK_DIM 8

typedef struct PsnrHvsStateCuda {
    /* Stream + event pair owned by `cuda/kernel_template.h` lifecycle
     * (ADR-0221). Multi-plane buffer state stays outside the
     * template's single-pair readback bundle. */
    VmafCudaKernelLifecycle lc;
    CUfunction func_psnr_hvs;

    /* Dedicated H2D upload stream + completion event (T-GPU-OPT-2).
     * H2Ds run on `upload_str` so DMA can overlap kernel launches on
     * `lc.str`; `upload_done` is recorded after the last per-frame H2D
     * and `lc.str` waits on it before the first kernel launch.  This
     * decouples upload latency from kernel-launch serial order on a
     * single non-blocking stream. */
    CUstream upload_str;
    CUevent upload_done;

    unsigned width[PSNR_HVS_NUM_PLANES];
    unsigned height[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_x[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_y[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks[PSNR_HVS_NUM_PLANES];
    unsigned bpc;
    int32_t samplemax_sq;

    /* Per-plane ref / dist float buffers (picture_copy normalised). */
    VmafCudaBuffer *d_ref[PSNR_HVS_NUM_PLANES];
    VmafCudaBuffer *d_dist[PSNR_HVS_NUM_PLANES];
    /* Per-plane block partials. */
    VmafCudaBuffer *d_partials[PSNR_HVS_NUM_PLANES];

    /* Pinned host staging. */
    float *h_ref[PSNR_HVS_NUM_PLANES];
    float *h_dist[PSNR_HVS_NUM_PLANES];
    float *h_partials[PSNR_HVS_NUM_PLANES];

    /* Persistent pinned uint8/uint16 staging for D2H readback of pic
     * planes (T-GPU-OPT-3).  Sized once in init() to
     * width × height × bpc_bytes per plane; reused every frame.  The
     * old per-frame `vmaf_cuda_buffer_host_alloc` round-trip cost ~5 µs
     * × 6 planes/frame; the persistent variant amortises that to 0. */
    void *h_uint_ref[PSNR_HVS_NUM_PLANES];
    void *h_uint_dist[PSNR_HVS_NUM_PLANES];

    /* `enable_chroma` option: when false, only luma (plane 0) is dispatched.
     * Default false mirrors the psnr_hvs_cuda conservative posture — callers
     * that need the full 0.8*Y + 0.1*(Cb+Cr) combined score must opt in.
     * YUV400P sources force n_planes=1 regardless of this flag. */
    bool enable_chroma;
    /* Number of active planes after init-time clamping (1 or 3). */
    unsigned n_planes;

    unsigned index;
    VmafDictionary *feature_name_dict;
} PsnrHvsStateCuda;

static const VmafOption options[] = {
    {
        .name = "enable_chroma",
        .help = "enable calculation for chroma channels",
        .offset = offsetof(PsnrHvsStateCuda, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {0},
};

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    PsnrHvsStateCuda *s = fex->priv;

    if (bpc > 12) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_cuda: invalid bitdepth (%u); bpc must be <= 12\n",
                 bpc);
        return -EINVAL;
    }
    if (w < (unsigned)PSNR_HVS_BLOCK || h < (unsigned)PSNR_HVS_BLOCK) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_cuda: input %ux%u smaller than 8×8 block\n", w, h);
        return -EINVAL;
    }

    s->bpc = bpc;
    const int32_t samplemax = (1 << bpc) - 1;
    s->samplemax_sq = samplemax * samplemax;

    s->width[0] = w;
    s->height[0] = h;
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        /* YUV400P: luma only regardless of enable_chroma. */
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    } else {
        switch (pix_fmt) {
        case VMAF_PIX_FMT_YUV420P:
            /* Ceiling division — mirrors picture.c fix (Research-0094). */
            s->width[1] = s->width[2] = (w + 1u) >> 1;
            s->height[1] = s->height[2] = (h + 1u) >> 1;
            break;
        case VMAF_PIX_FMT_YUV422P:
            /* Ceiling division for horizontal subsampling only. */
            s->width[1] = s->width[2] = (w + 1u) >> 1;
            s->height[1] = s->height[2] = h;
            break;
        case VMAF_PIX_FMT_YUV444P:
            s->width[1] = s->width[2] = w;
            s->height[1] = s->height[2] = h;
            break;
        default:
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_hvs_cuda: unsupported pix_fmt\n");
            return -EINVAL;
        }
        s->n_planes = PSNR_HVS_NUM_PLANES;
        /* Mirror integer_psnr_cuda.c::init's enable_chroma guard (ADR-0453):
         * clamp to luma-only when the caller opts out of chroma. */
        if (!s->enable_chroma) {
            s->n_planes = 1U;
            s->width[1] = s->width[2] = 0U;
            s->height[1] = s->height[2] = 0U;
        }
    }

    for (unsigned p = 0; p < s->n_planes; p++) {
        if (s->width[p] < (unsigned)PSNR_HVS_BLOCK || s->height[p] < (unsigned)PSNR_HVS_BLOCK) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "psnr_hvs_cuda: plane %u dims %ux%u smaller than 8x8 block\n", p, s->width[p],
                     s->height[p]);
            return -EINVAL;
        }
        s->num_blocks_x[p] = (s->width[p] - PSNR_HVS_BLOCK) / PSNR_HVS_STEP + 1;
        s->num_blocks_y[p] = (s->height[p] - PSNR_HVS_BLOCK) / PSNR_HVS_STEP + 1;
        s->num_blocks[p] = s->num_blocks_x[p] * s->num_blocks_y[p];
    }

    int err = vmaf_cuda_kernel_lifecycle_init(&s->lc, fex->cu_state);
    if (err)
        return err;

    CudaFunctions *cu_f = fex->cu_state->f;
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;

    /* Dedicated H2D upload stream + cross-stream completion event
     * (T-GPU-OPT-2). See struct comment. */
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&s->upload_str, CU_STREAM_NON_BLOCKING, 0),
                    fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->upload_done, CU_EVENT_DISABLE_TIMING), fail);

    CUmodule module;
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&module, psnr_hvs_score_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleGetFunction(&s->func_psnr_hvs, module, "psnr_hvs"), fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    const unsigned bpc_bytes = (s->bpc <= 8 ? 1u : 2u);
    int ret = 0;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const size_t plane_bytes = (size_t)s->width[p] * s->height[p] * sizeof(float);
        const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);
        const size_t uint_bytes = (size_t)s->width[p] * s->height[p] * bpc_bytes;
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_ref[p], plane_bytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_dist[p], plane_bytes);
        ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->d_partials[p], partials_bytes);
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_ref[p], plane_bytes);
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_dist[p], plane_bytes);
        ret |=
            vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&s->h_partials[p], partials_bytes);
        /* T-GPU-OPT-3: persistent pinned uint8/uint16 staging for D2H. */
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->h_uint_ref[p], uint_bytes);
        ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->h_uint_dist[p], uint_bytes);
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

/* T-GPU-OPT-2/3: picture_copy-style upload split into three steps.
 *
 *   1. issue_d2h_plane: queue cuMemcpy2DAsync from device pic plane →
 *      the persistent pinned uint staging buffer on the picture
 *      stream.  Non-blocking — issued for all 6 (ref, dist) × plane
 *      copies up front so the DMA engine can pipeline them.
 *
 *   2. convert_plane: after a single host-side sync on each pic
 *      stream, normalise the uint8/uint16 staging into the persistent
 *      pinned float buffer (h_ref / h_dist).  Arithmetic matches
 *      libvmaf/src/picture_copy.c exactly so the score path stays
 *      bit-equivalent to the previous implementation.
 *
 *   3. issue_h2d_plane: queue cuMemcpyHtoDAsync from the persistent
 *      pinned float buffer → device on the dedicated upload stream.
 *      After all 6 H2Ds the caller records s->upload_done; the kernel
 *      stream s->lc.str then cuStreamWaitEvent's on it before
 *      launching, allowing kernel work for plane N to overlap H2D for
 *      plane N+1 on the upload stream's DMA engine. */
static int issue_d2h_plane(PsnrHvsStateCuda *s, VmafFeatureExtractor *fex, VmafPicture *pic,
                           void *h_uint, int plane)
{
    CudaFunctions *cu_f = fex->cu_state->f;
    const unsigned bpc_bytes = (s->bpc <= 8 ? 1u : 2u);
    CUstream stream = vmaf_cuda_picture_get_stream(pic);

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = (CUdeviceptr)pic->data[plane];
    m.srcPitch = (size_t)pic->stride[plane];
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstHost = h_uint;
    m.dstPitch = (size_t)s->width[plane] * bpc_bytes;
    m.WidthInBytes = (size_t)s->width[plane] * bpc_bytes;
    m.Height = s->height[plane];
    CHECK_CUDA_RETURN(cu_f, cuMemcpy2DAsync(&m, stream));
    return 0;
}

static void convert_plane(const PsnrHvsStateCuda *s, const VmafPicture *pic, const void *h_uint,
                          float *h_float, int plane)
{
    if (pic->bpc <= 8) {
        const uint8_t *src = (const uint8_t *)h_uint;
        for (unsigned y = 0; y < s->height[plane]; y++) {
            for (unsigned x = 0; x < s->width[plane]; x++) {
                h_float[y * s->width[plane] + x] = (float)src[y * s->width[plane] + x];
            }
        }
    } else {
        const float scaler = (pic->bpc == 10) ? 4.0f :
                             (pic->bpc == 12) ? 16.0f :
                             (pic->bpc == 16) ? 256.0f :
                                                1.0f;
        const uint16_t *src = (const uint16_t *)h_uint;
        for (unsigned y = 0; y < s->height[plane]; y++) {
            for (unsigned x = 0; x < s->width[plane]; x++) {
                h_float[y * s->width[plane] + x] = (float)src[y * s->width[plane] + x] / scaler;
            }
        }
    }
}

static int issue_h2d_plane(PsnrHvsStateCuda *s, VmafFeatureExtractor *fex, const float *h_float,
                           VmafCudaBuffer *dst_buf, int plane)
{
    CudaFunctions *cu_f = fex->cu_state->f;
    const size_t plane_bytes = (size_t)s->width[plane] * s->height[plane] * sizeof(float);
    CHECK_CUDA_RETURN(
        cu_f, cuMemcpyHtoDAsync((CUdeviceptr)dst_buf->data, h_float, plane_bytes, s->upload_str));
    return 0;
}

/* Run the full three-phase upload of ref+dist planes:
 *   1) queue all 6 D2H on the per-pic streams,
 *   2) drain the pic streams once each, normalise uint→float,
 *   3) queue all 6 H2D on the dedicated upload stream and record
 *      the cross-stream completion event s->upload_done.
 *
 * Caller must `cuStreamWaitEvent(s->lc.str, s->upload_done, ...)`
 * before launching any kernel that reads s->d_ref / s->d_dist. */
static int upload_frame(PsnrHvsStateCuda *s, VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                        VmafPicture *dist_pic)
{
    CudaFunctions *cu_f = fex->cu_state->f;

    /* Phase 1: queue all D2H copies up front (no per-plane sync). */
    for (unsigned p = 0; p < s->n_planes; p++) {
        int err = issue_d2h_plane(s, fex, ref_pic, s->h_uint_ref[p], p);
        if (err)
            return err;
        err = issue_d2h_plane(s, fex, dist_pic, s->h_uint_dist[p], p);
        if (err)
            return err;
    }

    /* Phase 2: drain the two pic streams once each, then convert.
     * Per-pic streams may be the same handle when the pool reuses
     * one — the second sync is then a no-op fast path inside CUDA. */
    CUstream ref_stream = vmaf_cuda_picture_get_stream(ref_pic);
    CUstream dist_stream = vmaf_cuda_picture_get_stream(dist_pic);
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(ref_stream));
    if (dist_stream != ref_stream) {
        CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(dist_stream));
    }
    for (unsigned p = 0; p < s->n_planes; p++) {
        convert_plane(s, ref_pic, s->h_uint_ref[p], s->h_ref[p], p);
        convert_plane(s, dist_pic, s->h_uint_dist[p], s->h_dist[p], p);
    }

    /* Phase 3: queue all H2Ds on the dedicated upload stream, then
     * record the cross-stream event. */
    for (unsigned p = 0; p < s->n_planes; p++) {
        int err = issue_h2d_plane(s, fex, s->h_ref[p], s->d_ref[p], p);
        if (err)
            return err;
        err = issue_h2d_plane(s, fex, s->h_dist[p], s->d_dist[p], p);
        if (err)
            return err;
    }
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->upload_done, s->upload_str));
    return 0;
}

/* Launch one kernel per plane on the kernel stream s->lc.str. The
 * caller is responsible for the cuStreamWaitEvent on s->upload_done
 * before invoking this helper. */
static int launch_plane_kernels(PsnrHvsStateCuda *s, VmafFeatureExtractor *fex)
{
    CudaFunctions *cu_f = fex->cu_state->f;
    for (unsigned p = 0; p < s->n_planes; p++) {
        int plane_arg = (int)p;
        int bpc_arg = (int)s->bpc;
        unsigned width = s->width[p];
        unsigned height = s->height[p];
        unsigned nbx = s->num_blocks_x[p];
        unsigned nby = s->num_blocks_y[p];
        void *params[] = {
            (void *)s->d_ref[p], (void *)s->d_dist[p], (void *)s->d_partials[p],
            (void *)&width,      (void *)&height,      (void *)&nbx,
            (void *)&nby,        (void *)&plane_arg,   (void *)&bpc_arg,
        };
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_psnr_hvs, nbx, nby, 1, PSNR_HVS_BLOCK_DIM,
                                               PSNR_HVS_BLOCK_DIM, 1, 0, s->lc.str, params, NULL));
    }
    return 0;
}

static int enqueue_partials_readback(PsnrHvsStateCuda *s, VmafFeatureExtractor *fex)
{
    CudaFunctions *cu_f = fex->cu_state->f;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);
        CHECK_CUDA_RETURN(cu_f,
                          cuMemcpyDtoHAsync(s->h_partials[p], (CUdeviceptr)s->d_partials[p]->data,
                                            partials_bytes, s->lc.str));
    }
    return 0;
}

static int submit_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    PsnrHvsStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    s->index = index;

    int err = upload_frame(s, fex, ref_pic, dist_pic);
    if (err)
        return err;

    /* Kernel stream waits on the upload event, then launches all 3
     * plane kernels.  The kernel handles all 3 planes via the
     * CSF_TABLES[plane] lookup; PLANE + BPC are runtime args. */
    CHECK_CUDA_RETURN(cu_f, cuStreamWaitEvent(s->lc.str, s->upload_done, CU_EVENT_WAIT_DEFAULT));
    err = launch_plane_kernels(s, fex);
    if (err)
        return err;

    err = enqueue_partials_readback(s, fex);
    if (err)
        return err;

    return vmaf_cuda_kernel_submit_post_record(&s->lc, fex->cu_state);
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    PsnrHvsStateCuda *s = fex->priv;

    int wait_err = vmaf_cuda_kernel_collect_wait(&s->lc, fex->cu_state);
    if (wait_err)
        return wait_err;

    /* Per-plane reduction matching CPU's float `ret` register
     * semantics (see psnr_hvs_vulkan.c for the rationale). */
    double plane_score[PSNR_HVS_NUM_PLANES] = {0.0, 0.0, 0.0};
    for (unsigned p = 0; p < s->n_planes; p++) {
        float ret = 0.0f;
        for (unsigned i = 0; i < s->num_blocks[p]; i++)
            ret += s->h_partials[p][i];
        const int pixels = (int)(s->num_blocks[p] * 64u);
        ret /= (float)pixels;
        ret /= (float)s->samplemax_sq;
        plane_score[p] = (double)ret;
    }

    int err = 0;
    static const char *plane_features[PSNR_HVS_NUM_PLANES] = {"psnr_hvs_y", "psnr_hvs_cb",
                                                              "psnr_hvs_cr"};
    for (unsigned p = 0; p < s->n_planes; p++) {
        const double db = 10.0 * (-1.0 * log10(plane_score[p]));
        err |= vmaf_feature_collector_append(feature_collector, plane_features[p], db, index);
    }
    /* Combined score: when chroma is disabled emit luma dB only. */
    const double combined = (s->n_planes == 1U) ?
                                plane_score[0] :
                                0.8 * plane_score[0] + 0.1 * (plane_score[1] + plane_score[2]);
    const double db_combined = 10.0 * (-1.0 * log10(combined));
    err |= vmaf_feature_collector_append(feature_collector, "psnr_hvs", db_combined, index);
    return err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    PsnrHvsStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    int ret = vmaf_cuda_kernel_lifecycle_close(&s->lc, fex->cu_state);

    /* T-GPU-OPT-2: tear down dedicated upload stream + event.
     * Drain first so any in-flight H2D completes before the pinned
     * staging it sources is freed below. */
    if (s->upload_str != NULL) {
        const CUresult sync_res = cu_f->cuStreamSynchronize(s->upload_str);
        if (sync_res != CUDA_SUCCESS && ret == 0)
            ret = vmaf_cuda_result_to_errno((int)sync_res);
        const CUresult destroy_res = cu_f->cuStreamDestroy(s->upload_str);
        if (destroy_res != CUDA_SUCCESS && ret == 0)
            ret = vmaf_cuda_result_to_errno((int)destroy_res);
        s->upload_str = NULL;
    }
    if (s->upload_done != NULL) {
        const CUresult e = cu_f->cuEventDestroy(s->upload_done);
        if (e != CUDA_SUCCESS && ret == 0)
            ret = vmaf_cuda_result_to_errno((int)e);
        s->upload_done = NULL;
    }

    for (unsigned p = 0; p < s->n_planes; p++) {
        if (s->d_ref[p]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->d_ref[p]);
            free(s->d_ref[p]);
        }
        if (s->d_dist[p]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->d_dist[p]);
            free(s->d_dist[p]);
        }
        if (s->d_partials[p]) {
            ret |= vmaf_cuda_buffer_free(fex->cu_state, s->d_partials[p]);
            free(s->d_partials[p]);
        }
        if (s->h_ref[p])
            (void)vmaf_cuda_buffer_host_free(fex->cu_state, s->h_ref[p]);
        if (s->h_dist[p])
            (void)vmaf_cuda_buffer_host_free(fex->cu_state, s->h_dist[p]);
        if (s->h_partials[p])
            (void)vmaf_cuda_buffer_host_free(fex->cu_state, s->h_partials[p]);
        /* T-GPU-OPT-3: persistent uint staging buffers. */
        if (s->h_uint_ref[p])
            (void)vmaf_cuda_buffer_host_free(fex->cu_state, s->h_uint_ref[p]);
        if (s->h_uint_dist[p])
            (void)vmaf_cuda_buffer_host_free(fex->cu_state, s->h_uint_dist[p]);
    }
    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static const char *provided_features[] = {"psnr_hvs_y", "psnr_hvs_cb", "psnr_hvs_cr", "psnr_hvs",
                                          NULL};

VmafFeatureExtractor vmaf_fex_psnr_hvs_cuda = {
    .name = "psnr_hvs_cuda",
    .init = init_fex_cuda,
    .submit = submit_fex_cuda,
    .collect = collect_fex_cuda,
    .close = close_fex_cuda,
    .options = options,
    .priv_size = sizeof(PsnrHvsStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
