/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_ssim feature extractor on the Metal backend (T8-1k).
 *  Port of the CUDA twin libvmaf/src/feature/cuda/integer_ssim_cuda.c.
 *
 *  Two-pass dispatch from raw integer pixels:
 *    Pass 0: integer_ssim_horiz_{8,16}bpc — horizontal 11-tap Gaussian on
 *            raw luma pixels (normalised to [0,255] inline) → 5 float planes.
 *    Pass 1: integer_ssim_vert_combine    — vertical 11-tap + SSIM combine
 *            + per-WG float partial.
 *
 *  Host sums partials in double, divides by (W-10)·(H-10).
 *  Feature name: "float_ssim" (matches the CUDA twin's output name).
 *
 *  Normalisation scalers (raw → [0, 255.xxx]):
 *    8bpc  → 1.0 (identity)
 *    10bpc → 1/4.0
 *    12bpc → 1/16.0
 *    16bpc → 1/256.0
 *
 *  v1: scale=1 only (no auto-decimation) — mirrors the CUDA twin constraint.
 *  Requests for scale != 1 are rejected at init with -EINVAL.
 *
 *  c1 = (0.01*255)² = 6.5025,  c2 = (0.03*255)² = 58.5225.
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

typedef struct IntegerSsimStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb;        /* float ssim_sum partials, grid_w × grid_h */
    VmafMetalContext *ctx;
    void *pso_horiz_8bpc;            /* integer_ssim_horiz_8bpc */
    void *pso_horiz_16bpc;           /* integer_ssim_horiz_16bpc */
    void *pso_vert;                  /* integer_ssim_vert_combine */
    void *hbuf_buf;                  /* intermediate 5-plane float buffer */

    float    c1;
    float    c2;
    size_t   hbuf_size;
    size_t   partials_count;
    unsigned frame_w;
    unsigned frame_h;
    unsigned w_h;         /* W - 10 */
    unsigned h_v;         /* H - 10 */
    unsigned bpc;
    int      scale_override;

    VmafDictionary *feature_name_dict;
} IntegerSsimStateMetal;

static const VmafOption options[] = {
    {
        .name        = "scale",
        .help        = "decimation scale factor (0=auto, 1=no downscaling). "
                       "v1: Metal path requires scale=1.",
        .offset      = offsetof(IntegerSsimStateMetal, scale_override),
        .type        = VMAF_OPT_TYPE_INT,
        .default_val = {.i = 0},
        .min         = 0,
        .max         = 10,
    },
    {0},
};

static int build_pipelines(IntegerSsimStateMetal *s, id<MTLDevice> device)
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

    id<MTLFunction> fn_h8  = [lib newFunctionWithName:@"integer_ssim_horiz_8bpc"];
    id<MTLFunction> fn_h16 = [lib newFunctionWithName:@"integer_ssim_horiz_16bpc"];
    id<MTLFunction> fn_v   = [lib newFunctionWithName:@"integer_ssim_vert_combine"];
    if (fn_h8 == nil || fn_h16 == nil || fn_v == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_h8  = [device newComputePipelineStateWithFunction:fn_h8  error:&err];
    id<MTLComputePipelineState> pso_h16 = [device newComputePipelineStateWithFunction:fn_h16 error:&err];
    id<MTLComputePipelineState> pso_v   = [device newComputePipelineStateWithFunction:fn_v   error:&err];
    if (pso_h8 == nil || pso_h16 == nil || pso_v == nil) { return -ENODEV; }

    s->pso_horiz_8bpc  = (__bridge_retained void *)pso_h8;
    s->pso_horiz_16bpc = (__bridge_retained void *)pso_h16;
    s->pso_vert        = (__bridge_retained void *)pso_v;
    return 0;
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    IntegerSsimStateMetal *s = (IntegerSsimStateMetal *)fex->priv;

    /* v1: scale=1 only, matching the CUDA twin. */
    if (s->scale_override != 0 && s->scale_override != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_ssim_metal: v1 supports scale=1 only "
                 "(requested scale=%d).\n",
                 s->scale_override);
        return -EINVAL;
    }

    s->frame_w = w;
    s->frame_h = h;
    s->bpc     = bpc;
    s->w_h     = (w >= 10u) ? (w - 10u) : 0u;
    s->h_v     = (h >= 10u) ? (h - 10u) : 0u;
    s->c1      = 6.5025f;    /* (0.01 * 255)^2 */
    s->c2      = 58.5225f;   /* (0.03 * 255)^2 */

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        const size_t grid_w = (s->w_h + 15u) / 16u;
        const size_t grid_h = (s->h_v + 15u) / 16u;
        s->partials_count   = grid_w * grid_h;
        err = vmaf_metal_kernel_buffer_alloc(&s->rb, s->ctx,
                                             s->partials_count * sizeof(float));
    }
    if (err != 0) { goto fail_lc; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_rb; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        /* hbuf: 5 planes × w_h × H floats. */
        s->hbuf_size = 5u * (size_t)s->w_h * (size_t)h * sizeof(float);
        id<MTLBuffer> hb = [device newBufferWithLength:s->hbuf_size
                                               options:MTLResourceStorageModeShared];
        if (hb == nil) { err = -ENOMEM; goto fail_rb; }
        s->hbuf_buf = (__bridge_retained void *)hb;

        err = build_pipelines(s, device);
    }
    if (err != 0) { goto fail_hbuf; }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    if (s->pso_vert)        { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert;        s->pso_vert        = NULL; }
    if (s->pso_horiz_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz_16bpc; s->pso_horiz_16bpc = NULL; }
    if (s->pso_horiz_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz_8bpc;  s->pso_horiz_8bpc  = NULL; }
fail_hbuf:
    if (s->hbuf_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->hbuf_buf; s->hbuf_buf = NULL; }
fail_rb:
    (void)vmaf_metal_kernel_buffer_free(&s->rb, s->ctx);
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    IntegerSsimStateMetal *s = (IntegerSsimStateMetal *)fex->priv;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    s->w_h     = (s->frame_w >= 10u) ? (s->frame_w - 10u) : 0u;
    s->h_v     = (s->frame_h >= 10u) ? (s->frame_h - 10u) : 0u;

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>       device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue>  queue = (__bridge id<MTLCommandQueue>)qh;
    id<MTLBuffer>    hbuf_buf  = (__bridge id<MTLBuffer>)s->hbuf_buf;
    id<MTLBuffer>    par_buf   = (__bridge id<MTLBuffer>)(void *)s->rb.buffer;

    /* Upload luma plane as raw bytes — kernel normalises to [0,255]. */
    const size_t px_bytes     = (s->bpc <= 8u) ? 1u : 2u;
    const size_t row_bytes    = (size_t)s->frame_w * px_bytes;
    const size_t plane_bytes  = row_bytes * s->frame_h;

    id<MTLBuffer> ref_buf = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_buf = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    if (ref_buf == nil || dis_buf == nil) { return -ENOMEM; }

    {
        uint8_t *rd = (uint8_t *)[ref_buf contents];
        uint8_t *dd = (uint8_t *)[dis_buf contents];
        for (unsigned y = 0u; y < s->frame_h; ++y) {
            memcpy(rd + y * row_bytes,
                   (const uint8_t *)ref_pic->data[0] + (size_t)y * ref_pic->stride[0],
                   row_bytes);
            memcpy(dd + y * row_bytes,
                   (const uint8_t *)dist_pic->data[0] + (size_t)y * dist_pic->stride[0],
                   row_bytes);
        }
    }

    const size_t grid_w_h = (s->w_h + 15u) / 16u;
    const size_t grid_h_h = (s->frame_h + 7u) / 8u;
    const size_t grid_w_v = (s->w_h + 15u) / 16u;
    const size_t grid_h_v = (s->h_v + 7u) / 8u;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    /* Zero the partials buffer before accumulation. */
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit fillBuffer:par_buf range:NSMakeRange(0, s->partials_count * sizeof(float)) value:0];
    [blit endEncoding];

    /* Pass 0: horizontal convolution. */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (s->bpc <= 8u) {
            [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)s->pso_horiz_8bpc];
            [enc setBuffer:ref_buf  offset:0 atIndex:0];
            [enc setBuffer:dis_buf  offset:0 atIndex:1];
            [enc setBuffer:hbuf_buf offset:0 atIndex:2];
            uint32_t params[4] = {(uint32_t)s->frame_w, (uint32_t)s->frame_h,
                                  (uint32_t)s->w_h, (uint32_t)row_bytes};
            [enc setBytes:params length:sizeof(params) atIndex:3];
        } else {
            [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)s->pso_horiz_16bpc];
            [enc setBuffer:ref_buf  offset:0 atIndex:0];
            [enc setBuffer:dis_buf  offset:0 atIndex:1];
            [enc setBuffer:hbuf_buf offset:0 atIndex:2];
            uint32_t params[4] = {(uint32_t)s->frame_w, (uint32_t)s->frame_h,
                                  (uint32_t)s->w_h, (uint32_t)row_bytes};
            [enc setBytes:params length:sizeof(params) atIndex:3];
            uint32_t bpc_val = (uint32_t)s->bpc;
            [enc setBytes:&bpc_val length:sizeof(bpc_val) atIndex:4];
        }
        MTLSize tg   = MTLSizeMake(16, 8, 1);
        MTLSize grid = MTLSizeMake(grid_w_h, grid_h_h, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* Pass 1: vertical convolution + SSIM combine. */
    {
        const size_t p_grid_w = (s->w_h + 15u) / 16u;
        const size_t p_grid_h = (s->h_v + 7u) / 8u;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)s->pso_vert];
        [enc setBuffer:hbuf_buf offset:0 atIndex:0];
        [enc setBuffer:par_buf  offset:0 atIndex:1];
        uint32_t params[4] = {(uint32_t)s->frame_w, (uint32_t)s->frame_h,
                              (uint32_t)s->w_h, (uint32_t)s->h_v};
        [enc setBytes:params length:sizeof(params) atIndex:2];
        float consts[2] = {s->c1, s->c2};
        [enc setBytes:consts length:sizeof(consts) atIndex:3];
        uint32_t gdim[2] = {(uint32_t)p_grid_w, (uint32_t)p_grid_h};
        [enc setBytes:gdim length:sizeof(gdim) atIndex:4];
        MTLSize tg   = MTLSizeMake(16, 8, 1);
        MTLSize grid = MTLSizeMake(p_grid_w, p_grid_h, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    (void)grid_w_v;
    (void)grid_h_v;

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    IntegerSsimStateMetal *s = (IntegerSsimStateMetal *)fex->priv;

    const float *parts = (const float *)s->rb.host_view;
    double ssim_sum = 0.0;
    if (parts != NULL) {
        for (size_t i = 0u; i < s->partials_count; ++i) {
            ssim_sum += (double)parts[i];
        }
    }
    const double n_valid = (double)s->w_h * (double)s->h_v;
    const double ssim    = (n_valid > 0.0) ? (ssim_sum / n_valid) : 1.0;

    return vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_ssim", ssim, index);
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    IntegerSsimStateMetal *s = (IntegerSsimStateMetal *)fex->priv;
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_vert)        { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert;        s->pso_vert        = NULL; }
    if (s->pso_horiz_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz_16bpc; s->pso_horiz_16bpc = NULL; }
    if (s->pso_horiz_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz_8bpc;  s->pso_horiz_8bpc  = NULL; }
    if (s->hbuf_buf)        { (void)(__bridge_transfer id<MTLBuffer>)s->hbuf_buf;                      s->hbuf_buf        = NULL; }

    int err = vmaf_metal_kernel_buffer_free(&s->rb, s->ctx);
    if (err != 0 && rc == 0) { rc = err; }
    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

static const char *provided_features[] = {"float_ssim", NULL};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)  — ADR-0421: extern "C" linkage required for C-side registration
VmafFeatureExtractor vmaf_fex_integer_ssim_metal = {
    .name              = "integer_ssim_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(IntegerSsimStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 2,
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
