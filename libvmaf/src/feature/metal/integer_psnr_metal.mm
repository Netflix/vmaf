/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_psnr feature extractor on the Metal backend (T8-1g / ADR-0421).
 *  Dispatches `integer_psnr_kernel_{8,16}bpc` from integer_psnr.metal
 *  three times (Y, Cb, Cr planes).
 *
 *  Each dispatch emits lo/hi uint32 partial SSE per WG; host reconstructs
 *  uint64 SSE, computes MSE, then PSNR.
 *    peak     = (1 << bpc) - 1
 *    psnr_max = 6 * bpc + 12
 *    psnr     = min(10·log10(peak² / max(mse, 1e-16)), psnr_max)
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

#define PSNR_NUM_PLANES 3

typedef struct IntegerPsnrStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb_lo[PSNR_NUM_PLANES];  /* uint32 lo SSE partials */
    VmafMetalKernelBuffer rb_hi[PSNR_NUM_PLANES];  /* uint32 hi SSE partials */
    VmafMetalContext *ctx;
    void *pso_8bpc;
    void *pso_16bpc;

    uint32_t peak;
    double   psnr_max;
    size_t   partials_count;   /* grid_w × grid_h (for Y plane) */
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} IntegerPsnrStateMetal;

static const char *const psnr_name[PSNR_NUM_PLANES] = {"psnr_y", "psnr_cb", "psnr_cr"};

static const VmafOption options[] = {{0}};

static int build_pipelines(IntegerPsnrStateMetal *s, id<MTLDevice> device)
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

    id<MTLFunction> fn8  = [lib newFunctionWithName:@"integer_psnr_kernel_8bpc"];
    id<MTLFunction> fn16 = [lib newFunctionWithName:@"integer_psnr_kernel_16bpc"];
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
    IntegerPsnrStateMetal *s = (IntegerPsnrStateMetal *)fex->priv;

    s->frame_w  = w;
    s->frame_h  = h;
    s->bpc      = bpc;
    s->peak     = (1u << bpc) - 1u;
    s->psnr_max = (double)(6u * bpc) + 12.0;

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        const size_t grid_w   = (w + 15) / 16;
        const size_t grid_h   = (h + 15) / 16;
        s->partials_count     = grid_w * grid_h;
        const size_t par_size = s->partials_count * sizeof(uint32_t);
        for (int p = 0; p < PSNR_NUM_PLANES; ++p) {
            err = vmaf_metal_kernel_buffer_alloc(&s->rb_lo[p], s->ctx, par_size);
            if (err != 0) {
                for (int q = 0; q < p; ++q) {
                    (void)vmaf_metal_kernel_buffer_free(&s->rb_lo[q], s->ctx);
                    (void)vmaf_metal_kernel_buffer_free(&s->rb_hi[q], s->ctx);
                }
                goto fail_lc;
            }
            err = vmaf_metal_kernel_buffer_alloc(&s->rb_hi[p], s->ctx, par_size);
            if (err != 0) {
                (void)vmaf_metal_kernel_buffer_free(&s->rb_lo[p], s->ctx);
                for (int q = 0; q < p; ++q) {
                    (void)vmaf_metal_kernel_buffer_free(&s->rb_lo[q], s->ctx);
                    (void)vmaf_metal_kernel_buffer_free(&s->rb_hi[q], s->ctx);
                }
                goto fail_lc;
            }
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
    for (int p = 0; p < PSNR_NUM_PLANES; ++p) {
        (void)vmaf_metal_kernel_buffer_free(&s->rb_lo[p], s->ctx);
        (void)vmaf_metal_kernel_buffer_free(&s->rb_hi[p], s->ctx);
    }
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static int dispatch_plane(IntegerPsnrStateMetal *s, id<MTLDevice> device,
                          id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                          VmafPicture *ref_pic, VmafPicture *dis_pic, int plane)
{
    const unsigned pw = ref_pic->w[plane];
    const unsigned ph = ref_pic->h[plane];
    const size_t row_bytes_ref = (size_t)pw * (s->bpc <= 8u ? 1u : 2u);
    const size_t row_bytes_dis = row_bytes_ref;
    const size_t plane_bytes   = row_bytes_ref * ph;

    id<MTLBuffer> ref_buf = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_buf = [device newBufferWithLength:plane_bytes options:MTLResourceStorageModeShared];
    if (ref_buf == nil || dis_buf == nil) { return -ENOMEM; }
    {
        uint8_t *rd = (uint8_t *)[ref_buf contents];
        uint8_t *dd = (uint8_t *)[dis_buf contents];
        for (unsigned y = 0; y < ph; y++) {
            memcpy(rd + y * row_bytes_ref, (uint8_t *)ref_pic->data[plane] + y * ref_pic->stride[plane], row_bytes_ref);
            memcpy(dd + y * row_bytes_dis, (uint8_t *)dis_pic->data[plane]  + y * dis_pic->stride[plane],  row_bytes_dis);
        }
    }

    const size_t grid_w = (pw + 15) / 16;
    const size_t grid_h = (ph + 15) / 16;
    id<MTLBuffer> lo_buf = (__bridge id<MTLBuffer>)(void *)s->rb_lo[plane].buffer;
    id<MTLBuffer> hi_buf = (__bridge id<MTLBuffer>)(void *)s->rb_hi[plane].buffer;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    const size_t par_size = grid_w * grid_h * sizeof(uint32_t);
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
        uint32_t st[2] = {(uint32_t)row_bytes_ref, (uint32_t)row_bytes_dis};
        [enc setBytes:st length:sizeof(st) atIndex:4];
    } else {
        uint32_t st[4] = {(uint32_t)row_bytes_ref, (uint32_t)row_bytes_dis, (uint32_t)s->bpc, 0};
        [enc setBytes:st length:sizeof(st) atIndex:4];
    }
    uint32_t dim[2] = {(uint32_t)pw, (uint32_t)ph};
    [enc setBytes:dim length:sizeof(dim) atIndex:5];

    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(grid_w, grid_h, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    IntegerPsnrStateMetal *s = (IntegerPsnrStateMetal *)fex->priv;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>       device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue>  queue = (__bridge id<MTLCommandQueue>)qh;
    id<MTLComputePipelineState> pso = (s->bpc <= 8u)
        ? (__bridge id<MTLComputePipelineState>)s->pso_8bpc
        : (__bridge id<MTLComputePipelineState>)s->pso_16bpc;

    for (int p = 0; p < PSNR_NUM_PLANES; ++p) {
        int err = dispatch_plane(s, device, queue, pso, ref_pic, dist_pic, p);
        if (err != 0) { return err; }
    }
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    IntegerPsnrStateMetal *s = (IntegerPsnrStateMetal *)fex->priv;
    const double peak_sq = (double)s->peak * (double)s->peak;

    for (int p = 0; p < PSNR_NUM_PLANES; ++p) {
        const uint32_t *lo_p = (const uint32_t *)s->rb_lo[p].host_view;
        const uint32_t *hi_p = (const uint32_t *)s->rb_hi[p].host_view;
        const unsigned pw = (p == 0) ? s->frame_w : (s->frame_w + 1) / 2;
        const unsigned ph = (p == 0) ? s->frame_h : (s->frame_h + 1) / 2;
        const size_t grid_w = (pw + 15) / 16;
        const size_t grid_h = (ph + 15) / 16;
        const size_t cnt    = grid_w * grid_h;

        double sse_d = 0.0;
        if (lo_p != NULL && hi_p != NULL) {
            for (size_t i = 0; i < cnt; ++i) {
                const uint64_t wg_sse =
                    ((uint64_t)hi_p[i] << 32u) | (uint64_t)lo_p[i];
                sse_d += (double)wg_sse;
            }
        }
        const double n_pix = (double)pw * (double)ph;
        const double mse   = (n_pix > 0.0) ? (sse_d / n_pix) : 0.0;
        double psnr = (mse <= 1e-16) ? s->psnr_max
                                     : 10.0 * log10(peak_sq / mse);
        if (psnr > s->psnr_max) { psnr = s->psnr_max; }

        int err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, psnr_name[p], psnr, index);
        if (err != 0) { return err; }
    }
    return 0;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    IntegerPsnrStateMetal *s = (IntegerPsnrStateMetal *)fex->priv;
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_16bpc; s->pso_16bpc = NULL; }
    if (s->pso_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_8bpc;  s->pso_8bpc  = NULL; }

    for (int p = 0; p < PSNR_NUM_PLANES; ++p) {
        int err = vmaf_metal_kernel_buffer_free(&s->rb_hi[p], s->ctx);
        if (err != 0 && rc == 0) { rc = err; }
        err = vmaf_metal_kernel_buffer_free(&s->rb_lo[p], s->ctx);
        if (err != 0 && rc == 0) { rc = err; }
    }
    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

static const char *provided_features[] = {"psnr_y", "psnr_cb", "psnr_cr", NULL};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_psnr_metal = {
    .name              = "integer_psnr_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(IntegerPsnrStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = PSNR_NUM_PLANES,
        .is_reduction_only      = true,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
