/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ciede2000 feature extractor on the Metal backend.
 *  Ported from libvmaf/src/feature/cuda/integer_ciede_cuda.c
 *  (T7-23 / ADR-0182). Metal twin of ciede_cuda / ciede_vulkan.
 *
 *  Host side: submits one kernel dispatch per frame
 *  (integer_ciede_kernel_{8,16}bpc from integer_ciede.metal). Each
 *  threadgroup writes one float partial sum; host accumulates in double
 *  and applies 45 - 20*log10(mean_dE). Places=4 empirical floor
 *  on real hardware (matches ADR-0187 precision contract).
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

typedef struct CiedeStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer    rb;         /* float partials */
    VmafMetalContext        *ctx;
    void                    *pso_8bpc;
    void                    *pso_16bpc;

    size_t   partials_count;   /* grid_w * grid_h */
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    unsigned ss_hor;
    unsigned ss_ver;

    VmafDictionary *feature_name_dict;
} CiedeStateMetal;

static const VmafOption options[] = {{0}};

static int build_pipelines(CiedeStateMetal *s, id<MTLDevice> device)
{
    const size_t blob_size = (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_size == 0)
        return -ENODEV;

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_size,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL)
        return -ENOMEM;

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil)
        return -ENODEV;

    id<MTLFunction> fn8  = [lib newFunctionWithName:@"integer_ciede_kernel_8bpc"];
    id<MTLFunction> fn16 = [lib newFunctionWithName:@"integer_ciede_kernel_16bpc"];
    if (fn8 == nil || fn16 == nil)
        return -ENODEV;

    id<MTLComputePipelineState> pso8  = [device newComputePipelineStateWithFunction:fn8  error:&err];
    id<MTLComputePipelineState> pso16 = [device newComputePipelineStateWithFunction:fn16 error:&err];
    if (pso8 == nil || pso16 == nil)
        return -ENODEV;

    s->pso_8bpc  = (__bridge_retained void *)pso8;
    s->pso_16bpc = (__bridge_retained void *)pso16;
    return 0;
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;

    CiedeStateMetal *s = (CiedeStateMetal *)fex->priv;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc     = bpc;
    s->ss_hor  = (pix_fmt != VMAF_PIX_FMT_YUV444P) ? 1u : 0u;
    s->ss_ver  = (pix_fmt == VMAF_PIX_FMT_YUV420P)  ? 1u : 0u;

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_ctx;

    {
        const size_t grid_w   = (w + 15u) / 16u;
        const size_t grid_h   = (h + 15u) / 16u;
        s->partials_count     = grid_w * grid_h;
        const size_t par_size = s->partials_count * sizeof(float);
        err = vmaf_metal_kernel_buffer_alloc(&s->rb, s->ctx, par_size);
        if (err != 0)
            goto fail_lc;
    }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_rb; }
        err = build_pipelines(s, (__bridge id<MTLDevice>)dh);
    }
    if (err != 0)
        goto fail_rb;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    if (s->pso_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_8bpc;  s->pso_8bpc  = NULL; }
    if (s->pso_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_16bpc; s->pso_16bpc = NULL; }
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
    CiedeStateMetal *s = (CiedeStateMetal *)fex->priv;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    const unsigned w = s->frame_w;
    const unsigned h = s->frame_h;
    const size_t grid_w = (w + 15u) / 16u;
    const size_t grid_h = (h + 15u) / 16u;
    s->partials_count = grid_w * grid_h;

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL)
        return -ENODEV;

    id<MTLDevice>       device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue>  queue = (__bridge id<MTLCommandQueue>)qh;
    id<MTLComputePipelineState> pso = (s->bpc <= 8u)
        ? (__bridge id<MTLComputePipelineState>)s->pso_8bpc
        : (__bridge id<MTLComputePipelineState>)s->pso_16bpc;

    const size_t bytes_per_px = (s->bpc <= 8u) ? 1u : 2u;

    /* Upload luma plane. */
    const size_t luma_bytes = (size_t)ref_pic->stride[0] * h;
    const size_t chroma_h   = s->ss_ver ? (h + 1u) / 2u : h;
    const size_t chroma_bytes = (size_t)ref_pic->stride[1] * chroma_h;

    id<MTLBuffer> ref_y_buf = [device newBufferWithLength:luma_bytes   options:MTLResourceStorageModeShared];
    id<MTLBuffer> ref_u_buf = [device newBufferWithLength:chroma_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> ref_v_buf = [device newBufferWithLength:chroma_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_y_buf = [device newBufferWithLength:luma_bytes   options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_u_buf = [device newBufferWithLength:chroma_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_v_buf = [device newBufferWithLength:chroma_bytes options:MTLResourceStorageModeShared];
    if (!ref_y_buf || !ref_u_buf || !ref_v_buf || !dis_y_buf || !dis_u_buf || !dis_v_buf)
        return -ENOMEM;

    memcpy([ref_y_buf contents], ref_pic->data[0], luma_bytes);
    memcpy([ref_u_buf contents], ref_pic->data[1], chroma_bytes);
    memcpy([ref_v_buf contents], ref_pic->data[2], chroma_bytes);
    memcpy([dis_y_buf contents], dist_pic->data[0], luma_bytes);
    memcpy([dis_u_buf contents], dist_pic->data[1], chroma_bytes);
    memcpy([dis_v_buf contents], dist_pic->data[2], chroma_bytes);

    id<MTLBuffer> partials_buf = (__bridge id<MTLBuffer>)(void *)s->rb.buffer;
    const size_t par_size = s->partials_count * sizeof(float);

    /* params: width, height, bpc, ss_hor | (ss_ver << 16) */
    uint32_t params[4] = {
        (uint32_t)w,
        (uint32_t)h,
        (uint32_t)s->bpc,
        (uint32_t)(s->ss_hor | (s->ss_ver << 16u))
    };
    /* strides: ref_y, ref_uv, dis_y, dis_uv (in bytes) */
    uint32_t strides_buf[4] = {
        (uint32_t)(ref_pic->stride[0] / bytes_per_px),
        (uint32_t)(ref_pic->stride[1] / bytes_per_px),
        (uint32_t)(dist_pic->stride[0] / bytes_per_px),
        (uint32_t)(dist_pic->stride[1] / bytes_per_px)
    };
    /* Re-encode strides in bytes so the kernel can byte-address. */
    uint32_t strides_bytes[4] = {
        (uint32_t)ref_pic->stride[0],
        (uint32_t)ref_pic->stride[1],
        (uint32_t)dist_pic->stride[0],
        (uint32_t)dist_pic->stride[1]
    };
    (void)strides_buf; /* suppress unused warning; strides_bytes used below */

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil)
        return -ENOMEM;

    /* Zero partials. */
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit fillBuffer:partials_buf range:NSMakeRange(0, par_size) value:0];
    [blit endEncoding];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:ref_y_buf    offset:0 atIndex:0];
    [enc setBuffer:ref_u_buf    offset:0 atIndex:1];
    [enc setBuffer:ref_v_buf    offset:0 atIndex:2];
    [enc setBuffer:dis_y_buf    offset:0 atIndex:3];
    [enc setBuffer:dis_u_buf    offset:0 atIndex:4];
    [enc setBuffer:dis_v_buf    offset:0 atIndex:5];
    [enc setBuffer:partials_buf offset:0 atIndex:6];
    [enc setBytes:params         length:sizeof(params)        atIndex:7];
    [enc setBytes:strides_bytes  length:sizeof(strides_bytes) atIndex:8];

    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(grid_w, grid_h, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    CiedeStateMetal *s = (CiedeStateMetal *)fex->priv;

    const float  *partials = (const float *)s->rb.host_view;
    double total = 0.0;
    for (size_t i = 0; i < s->partials_count; i++)
        total += (double)partials[i];

    const double n_pixels = (double)s->frame_w * (double)s->frame_h;
    const double mean_de  = (n_pixels > 0.0) ? (total / n_pixels) : 0.0;
    const double score    = 45.0 - 20.0 * log10(mean_de);

    return vmaf_feature_collector_append_with_dict(feature_collector,
                                                   s->feature_name_dict,
                                                   "ciede2000", score, index);
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    CiedeStateMetal *s = (CiedeStateMetal *)fex->priv;

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_16bpc; s->pso_16bpc = NULL; }
    if (s->pso_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_8bpc;  s->pso_8bpc  = NULL; }

    int rb_rc = vmaf_metal_kernel_buffer_free(&s->rb, s->ctx);
    if (rc == 0) rc = rb_rc;

    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

static const char *provided_features[] = {"ciede2000", NULL};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_ciede_metal = {
    .name              = "ciede_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(CiedeStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 1,
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
