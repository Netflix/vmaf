/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_moment feature extractor on the Metal backend (T8-1k / ADR-0421).
 *  Dispatches `integer_moment_kernel_{8,16}bpc` from integer_moment.metal.
 *
 *  Uses uint32 hi/lo split per WG (mirrors integer_psnr.metal / PR #1029):
 *  MSL has no atomic_ulong, so each WG writes 8 uint32 values
 *  (lo + hi for each of ref1, dis1, ref2, dis2).
 *  Host reconstructs uint64 sums, divides by (W * H), emits four features.
 *
 *  Buffer layout (4 × grid_w × grid_h uint32 pairs):
 *    sums_lo[idx*4+0], sums_hi[idx*4+0]  — ref 1st moment WG partial
 *    sums_lo[idx*4+1], sums_hi[idx*4+1]  — dis 1st moment WG partial
 *    sums_lo[idx*4+2], sums_hi[idx*4+2]  — ref 2nd moment WG partial
 *    sums_lo[idx*4+3], sums_hi[idx*4+3]  — dis 2nd moment WG partial
 *
 *  Feature names: float_moment_ref1st, float_moment_dis1st,
 *                 float_moment_ref2nd, float_moment_dis2nd.
 */

#include <errno.h>
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

typedef struct IntegerMomentStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb_lo;  /* 4 × grid_w × grid_h uint32 lo halves */
    VmafMetalKernelBuffer rb_hi;  /* 4 × grid_w × grid_h uint32 hi halves */
    VmafMetalContext *ctx;
    void *pso_8bpc;
    void *pso_16bpc;

    size_t plane_bytes;
    size_t partials_count;   /* grid_w × grid_h */
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} IntegerMomentStateMetal;

static const VmafOption options[] = {{0}};

static int build_pipelines(IntegerMomentStateMetal *s, id<MTLDevice> device)
{
    const size_t blob_size = (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_size == 0) { return -ENODEV; }

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_size,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL) { return -ENOMEM; }

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil) { return -ENODEV; }

    id<MTLFunction> fn8  = [lib newFunctionWithName:@"integer_moment_kernel_8bpc"];
    id<MTLFunction> fn16 = [lib newFunctionWithName:@"integer_moment_kernel_16bpc"];
    if (fn8 == nil || fn16 == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso8  = [device newComputePipelineStateWithFunction:fn8  error:&err];
    id<MTLComputePipelineState> pso16 = [device newComputePipelineStateWithFunction:fn16 error:&err];
    if (pso8 == nil || pso16 == nil) { return -ENODEV; }

    s->pso_8bpc  = (__bridge_retained void *)pso8;
    s->pso_16bpc = (__bridge_retained void *)pso16;
    return 0;
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    IntegerMomentStateMetal *s = (IntegerMomentStateMetal *)fex->priv;

    s->frame_w     = w;
    s->frame_h     = h;
    s->bpc         = bpc;
    s->plane_bytes = (size_t)w * h * (bpc <= 8u ? 1u : 2u);

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        const size_t grid_w = (w + 15) / 16;
        const size_t grid_h = (h + 15) / 16;
        s->partials_count   = grid_w * grid_h;
        /* 4 uint32 lo/hi pairs per WG: ref1, dis1, ref2, dis2 */
        const size_t par_size = s->partials_count * 4u * sizeof(uint32_t);
        err = vmaf_metal_kernel_buffer_alloc(&s->rb_lo, s->ctx, par_size);
        if (err != 0) { goto fail_lc; }
        err = vmaf_metal_kernel_buffer_alloc(&s->rb_hi, s->ctx, par_size);
        if (err != 0) {
            (void)vmaf_metal_kernel_buffer_free(&s->rb_lo, s->ctx);
            goto fail_lc;
        }
    }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_rb; }
        err = build_pipelines(s, (__bridge id<MTLDevice>)dh);
    }
    if (err != 0) { goto fail_rb; }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    if (s->pso_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_8bpc;  s->pso_8bpc  = NULL; }
    if (s->pso_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_16bpc; s->pso_16bpc = NULL; }
fail_rb:
    (void)vmaf_metal_kernel_buffer_free(&s->rb_hi, s->ctx);
    (void)vmaf_metal_kernel_buffer_free(&s->rb_lo, s->ctx);
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
    IntegerMomentStateMetal *s = (IntegerMomentStateMetal *)fex->priv;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    const size_t row_bytes = (size_t)s->frame_w * (s->bpc <= 8u ? 1u : 2u);

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>       device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue>  queue = (__bridge id<MTLCommandQueue>)qh;
    id<MTLBuffer>    lo_buf   = (__bridge id<MTLBuffer>)(void *)s->rb_lo.buffer;
    id<MTLBuffer>    hi_buf   = (__bridge id<MTLBuffer>)(void *)s->rb_hi.buffer;
    id<MTLComputePipelineState> pso = (s->bpc <= 8u)
        ? (__bridge id<MTLComputePipelineState>)s->pso_8bpc
        : (__bridge id<MTLComputePipelineState>)s->pso_16bpc;

    id<MTLBuffer> ref_buf = [device newBufferWithLength:s->plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_buf = [device newBufferWithLength:s->plane_bytes options:MTLResourceStorageModeShared];
    if (ref_buf == nil || dis_buf == nil) { return -ENOMEM; }
    {
        uint8_t *rd = (uint8_t *)[ref_buf contents];
        uint8_t *dd = (uint8_t *)[dis_buf contents];
        for (unsigned y = 0; y < s->frame_h; y++) {
            memcpy(rd + y * row_bytes, (uint8_t *)ref_pic->data[0] + y * ref_pic->stride[0], row_bytes);
            memcpy(dd + y * row_bytes, (uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0], row_bytes);
        }
    }

    const size_t par_size = s->partials_count * 4u * sizeof(uint32_t);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit fillBuffer:lo_buf range:NSMakeRange(0, par_size) value:0];
    [blit fillBuffer:hi_buf range:NSMakeRange(0, par_size) value:0];
    [blit endEncoding];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:ref_buf offset:0 atIndex:0];
    [enc setBuffer:dis_buf offset:0 atIndex:1];
    [enc setBuffer:lo_buf  offset:0 atIndex:2];
    [enc setBuffer:hi_buf  offset:0 atIndex:3];
    if (s->bpc <= 8u) {
        uint32_t st[2] = {(uint32_t)row_bytes, (uint32_t)row_bytes};
        [enc setBytes:st length:sizeof(st) atIndex:4];
    } else {
        uint32_t st[4] = {(uint32_t)row_bytes, (uint32_t)row_bytes, (uint32_t)s->bpc, 0};
        [enc setBytes:st length:sizeof(st) atIndex:4];
    }
    uint32_t dim[2] = {(uint32_t)s->frame_w, (uint32_t)s->frame_h};
    [enc setBytes:dim length:sizeof(dim) atIndex:5];

    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((s->frame_w + 15) / 16, (s->frame_h + 15) / 16, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    IntegerMomentStateMetal *s = (IntegerMomentStateMetal *)fex->priv;

    const uint32_t *lo_p = (const uint32_t *)s->rb_lo.host_view;
    const uint32_t *hi_p = (const uint32_t *)s->rb_hi.host_view;

    double sum[4] = {0.0, 0.0, 0.0, 0.0};
    if (lo_p != NULL && hi_p != NULL) {
        for (size_t i = 0; i < s->partials_count; ++i) {
            const size_t base = i * 4u;
            for (int k = 0; k < 4; ++k) {
                const uint64_t wg_val =
                    ((uint64_t)hi_p[base + (size_t)k] << 32u) |
                    (uint64_t)lo_p[base + (size_t)k];
                sum[k] += (double)wg_val;
            }
        }
    }

    const double n_pix = (double)s->frame_w * (double)s->frame_h;
    const double ref1 = (n_pix > 0.0) ? (sum[0] / n_pix) : 0.0;
    const double dis1 = (n_pix > 0.0) ? (sum[1] / n_pix) : 0.0;
    const double ref2 = (n_pix > 0.0) ? (sum[2] / n_pix) : 0.0;
    const double dis2 = (n_pix > 0.0) ? (sum[3] / n_pix) : 0.0;

    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_moment_ref1st", ref1, index);
    if (err != 0) { return err; }
    err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_moment_dis1st", dis1, index);
    if (err != 0) { return err; }
    err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_moment_ref2nd", ref2, index);
    if (err != 0) { return err; }
    return vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_moment_dis2nd", dis2, index);
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    IntegerMomentStateMetal *s = (IntegerMomentStateMetal *)fex->priv;
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_16bpc; s->pso_16bpc = NULL; }
    if (s->pso_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_8bpc;  s->pso_8bpc  = NULL; }

    int err = vmaf_metal_kernel_buffer_free(&s->rb_hi, s->ctx);
    if (err != 0 && rc == 0) { rc = err; }
    err = vmaf_metal_kernel_buffer_free(&s->rb_lo, s->ctx);
    if (err != 0 && rc == 0) { rc = err; }
    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

static const char *provided_features[] = {
    "float_moment_ref1st", "float_moment_dis1st",
    "float_moment_ref2nd", "float_moment_dis2nd",
    NULL
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_moment_metal = {
    .name              = "integer_moment_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(IntegerMomentStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 1,
        .is_reduction_only      = true,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
