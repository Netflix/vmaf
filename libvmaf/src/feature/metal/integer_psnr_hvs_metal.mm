/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  psnr_hvs feature extractor on the Metal backend.
 *  Port of `libvmaf/src/feature/cuda/integer_psnr_hvs_cuda.c` — same
 *  per-plane single-dispatch design, same host accumulation logic.
 *
 *  3 kernel dispatches per frame (Y, Cb, Cr).  Combined score:
 *  `psnr_hvs = 10·log10(1 / (0.8·Y_mse + 0.1·(Cb_mse + Cr_mse)))`.
 *
 *  Host normalises uint samples → float in [0, 255] (same scaler
 *  as picture_copy.c and the CUDA twin).  Buffers are MTLStorageModeShared
 *  (Apple unified memory — no explicit D2H copy needed).
 *
 *  Rejects YUV400P (no chroma) and bpc > 12 (matches CPU and CUDA).
 *
 *  Metallib resolution: same embedded-blob pattern as every other Metal
 *  feature extractor — reads the __TEXT,__metallib section compiled via
 *  xcrun from integer_psnr_hvs.metal.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" {
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"

#include "../../metal/common.h"
#include "../../metal/kernel_template.h"
}

extern "C" {
extern const unsigned char libvmaf_metallib_start[] __asm("section$start$__TEXT$__metallib");
extern const unsigned char libvmaf_metallib_end[]   __asm("section$end$__TEXT$__metallib");
}

#define PSNR_HVS_BLOCK      8
#define PSNR_HVS_STEP       7
#define PSNR_HVS_NUM_PLANES 3

typedef struct PsnrHvsStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext        *ctx;

    void *pso_psnr_hvs;

    unsigned width[PSNR_HVS_NUM_PLANES];
    unsigned height[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_x[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks_y[PSNR_HVS_NUM_PLANES];
    unsigned num_blocks[PSNR_HVS_NUM_PLANES];

    unsigned bpc;
    float    scaler;
    int32_t  samplemax_sq;

    void *buf_ref[PSNR_HVS_NUM_PLANES];
    void *buf_dist[PSNR_HVS_NUM_PLANES];
    void *buf_partials[PSNR_HVS_NUM_PLANES];

    unsigned index;
    VmafDictionary *feature_name_dict;
} PsnrHvsStateMetal;

static const VmafOption options[] = {{0}};

static int build_pipeline(PsnrHvsStateMetal *s, id<MTLDevice> device)
{
    const size_t blob_size = (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_size == 0u) { return -ENODEV; }

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_size,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL) { return -ENOMEM; }

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil) { return -ENODEV; }

    id<MTLFunction> fn = [lib newFunctionWithName:@"psnr_hvs"];
    if (fn == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil) { return -ENODEV; }

    s->pso_psnr_hvs = (__bridge_retained void *)pso;
    return 0;
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    PsnrHvsStateMetal *s = (PsnrHvsStateMetal *)fex->priv;

    if (bpc > 12u) { return -EINVAL; }
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) { return -EINVAL; }
    if (w < (unsigned)PSNR_HVS_BLOCK || h < (unsigned)PSNR_HVS_BLOCK) { return -EINVAL; }

    s->bpc = bpc;
    if (bpc <= 8u)       { s->scaler = 1.0f; }
    else if (bpc == 10u) { s->scaler = 4.0f; }
    else if (bpc == 12u) { s->scaler = 16.0f; }
    else                 { s->scaler = 256.0f; }

    const int32_t samplemax = (int32_t)((1u << bpc) - 1u);
    s->samplemax_sq = samplemax * samplemax;

    s->width[0]  = w;
    s->height[0] = h;
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        s->width[1]  = s->width[2]  = (w + 1u) >> 1;
        s->height[1] = s->height[2] = (h + 1u) >> 1;
        break;
    case VMAF_PIX_FMT_YUV422P:
        s->width[1]  = s->width[2]  = (w + 1u) >> 1;
        s->height[1] = s->height[2] = h;
        break;
    case VMAF_PIX_FMT_YUV444P:
        s->width[1]  = s->width[2]  = w;
        s->height[1] = s->height[2] = h;
        break;
    default:
        return -EINVAL;
    }

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; ++p) {
        if (s->width[p]  < (unsigned)PSNR_HVS_BLOCK ||
            s->height[p] < (unsigned)PSNR_HVS_BLOCK) { return -EINVAL; }
        s->num_blocks_x[p] = (s->width[p]  - (unsigned)PSNR_HVS_BLOCK) /
                              (unsigned)PSNR_HVS_STEP + 1u;
        s->num_blocks_y[p] = (s->height[p] - (unsigned)PSNR_HVS_BLOCK) /
                              (unsigned)PSNR_HVS_STEP + 1u;
        s->num_blocks[p]   = s->num_blocks_x[p] * s->num_blocks_y[p];
    }

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_lc; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        err = build_pipeline(s, device);
        if (err != 0) { goto fail_lc; }

        for (int p = 0; p < PSNR_HVS_NUM_PLANES; ++p) {
            const size_t plane_bytes    = (size_t)s->width[p] * s->height[p] * sizeof(float);
            const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);

            id<MTLBuffer> br = [device newBufferWithLength:plane_bytes
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> bd = [device newBufferWithLength:plane_bytes
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> bp = [device newBufferWithLength:partials_bytes
                                                   options:MTLResourceStorageModeShared];
            if (br == nil || bd == nil || bp == nil) { err = -ENOMEM; goto fail_pso; }

            s->buf_ref[p]      = (__bridge_retained void *)br;
            s->buf_dist[p]     = (__bridge_retained void *)bd;
            s->buf_partials[p] = (__bridge_retained void *)bp;
        }
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    if (s->pso_psnr_hvs) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_psnr_hvs;
        s->pso_psnr_hvs = NULL;
    }
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; ++p) {
        if (s->buf_ref[p]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->buf_ref[p];
            s->buf_ref[p] = NULL;
        }
        if (s->buf_dist[p]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->buf_dist[p];
            s->buf_dist[p] = NULL;
        }
        if (s->buf_partials[p]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->buf_partials[p];
            s->buf_partials[p] = NULL;
        }
    }
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static void fill_float_plane(VmafPicture *pic, int plane, id<MTLBuffer> dst,
                             unsigned w, unsigned h, float inv_scaler, unsigned bpc)
{
    float *out = (float *)[dst contents];
    if (bpc <= 8u) {
        for (unsigned y = 0; y < h; ++y) {
            const uint8_t *row = (const uint8_t *)pic->data[plane] +
                                 (size_t)y * pic->stride[plane];
            for (unsigned x = 0; x < w; ++x) {
                out[y * w + x] = (float)row[x];
            }
        }
    } else {
        for (unsigned y = 0; y < h; ++y) {
            const uint16_t *row =
                (const uint16_t *)((const uint8_t *)pic->data[plane] +
                                   (size_t)y * pic->stride[plane]);
            for (unsigned x = 0; x < w; ++x) {
                out[y * w + x] = (float)row[x] * inv_scaler;
            }
        }
    }
}

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    PsnrHvsStateMetal *s = (PsnrHvsStateMetal *)fex->priv;
    s->index = index;

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLCommandQueue>  queue = (__bridge id<MTLCommandQueue>)qh;

    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)s->pso_psnr_hvs;

    const float inv = 1.0f / s->scaler;

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; ++p) {
        fill_float_plane(ref_pic,  p,
                         (__bridge id<MTLBuffer>)s->buf_ref[p],
                         s->width[p], s->height[p], inv, s->bpc);
        fill_float_plane(dist_pic, p,
                         (__bridge id<MTLBuffer>)s->buf_dist[p],
                         s->width[p], s->height[p], inv, s->bpc);
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    for (int p = 0; p < PSNR_HVS_NUM_PLANES; ++p) {
        id<MTLBuffer> ref_buf  = (__bridge id<MTLBuffer>)s->buf_ref[p];
        id<MTLBuffer> dist_buf = (__bridge id<MTLBuffer>)s->buf_dist[p];
        id<MTLBuffer> part_buf = (__bridge id<MTLBuffer>)s->buf_partials[p];

        const uint32_t params[4] = {
            (uint32_t)s->width[p],
            (uint32_t)s->height[p],
            (uint32_t)s->num_blocks_x[p],
            (uint32_t)s->num_blocks_y[p],
        };
        const uint32_t bpc_plane[2] = {
            (uint32_t)s->bpc,
            (uint32_t)p,
        };

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:ref_buf  offset:0 atIndex:0];
        [enc setBuffer:dist_buf offset:0 atIndex:1];
        [enc setBuffer:part_buf offset:0 atIndex:2];
        [enc setBytes:params    length:sizeof(params)    atIndex:3];
        [enc setBytes:bpc_plane length:sizeof(bpc_plane) atIndex:4];

        MTLSize tg   = MTLSizeMake(8, 8, 1);
        MTLSize grid = MTLSizeMake(s->num_blocks_x[p], s->num_blocks_y[p], 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    PsnrHvsStateMetal *s = (PsnrHvsStateMetal *)fex->priv;

    double plane_score[PSNR_HVS_NUM_PLANES];
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; ++p) {
        const float *parts =
            (const float *)[(__bridge id<MTLBuffer>)s->buf_partials[p] contents];
        float ret = 0.f;
        for (unsigned i = 0; i < s->num_blocks[p]; ++i) {
            ret += parts[i];
        }
        const int pixels = (int)(s->num_blocks[p] * 64u);
        ret /= (float)pixels;
        ret /= (float)s->samplemax_sq;
        plane_score[p] = (double)ret;
    }

    static const char *const plane_names[PSNR_HVS_NUM_PLANES] = {
        "psnr_hvs_y", "psnr_hvs_cb", "psnr_hvs_cr",
    };
    int err = 0;
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; ++p) {
        const double db = 10.0 * (-1.0 * log10(plane_score[p]));
        err |= vmaf_feature_collector_append(feature_collector, plane_names[p], db, index);
    }
    const double combined = 0.8 * plane_score[0] + 0.1 * (plane_score[1] + plane_score[2]);
    const double db_combined = 10.0 * (-1.0 * log10(combined));
    err |= vmaf_feature_collector_append(feature_collector, "psnr_hvs", db_combined, index);
    return err;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    PsnrHvsStateMetal *s = (PsnrHvsStateMetal *)fex->priv;

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_psnr_hvs) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_psnr_hvs;
        s->pso_psnr_hvs = NULL;
    }
    for (int p = 0; p < PSNR_HVS_NUM_PLANES; ++p) {
        if (s->buf_ref[p]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->buf_ref[p];
            s->buf_ref[p] = NULL;
        }
        if (s->buf_dist[p]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->buf_dist[p];
            s->buf_dist[p] = NULL;
        }
        if (s->buf_partials[p]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->buf_partials[p];
            s->buf_partials[p] = NULL;
        }
    }
    if (s->feature_name_dict) {
        int err = vmaf_dictionary_free(&s->feature_name_dict);
        if (err != 0 && rc == 0) { rc = err; }
    }
    if (s->ctx) {
        vmaf_metal_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {
    "psnr_hvs_y", "psnr_hvs_cb", "psnr_hvs_cr", "psnr_hvs", NULL,
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_psnr_hvs_metal = {
    .name              = "psnr_hvs_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(PsnrHvsStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = PSNR_HVS_NUM_PLANES,
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
