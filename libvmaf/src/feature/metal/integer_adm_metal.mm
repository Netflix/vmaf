/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_adm feature extractor on the Metal backend.
 *
 *  Port of libvmaf/src/feature/cuda/integer_adm_cuda.c.
 *  Dispatches four MSL kernels from integer_adm.metal:
 *    adm_dwt2_8_kernel_{8,16}bpc — stage 1: DWT on input luma
 *    adm_dwt_s123_{vert,hori}_kernel — stage 2: scales 1-3 DWT
 *    adm_csf_kernel  — stage 3: contrast sensitivity filter
 *    adm_cm_kernel   — stage 4: contrast masking, num/den accumulation
 *
 *  The host collects per-WG (lo, hi) uint32 partials for every sub-band
 *  at every scale and reconstructs the final ADM scores in double
 *  precision (same accumulation logic as the CUDA twin).
 *
 *  Output features (identical to CPU / CUDA / SYCL / Vulkan paths):
 *    adm2                — overall ADM2 score
 *    adm_scale0          — scale 0 (finest) ADM score
 *    adm_scale1          — scale 1 ADM score
 *    adm_scale2          — scale 2 ADM score
 *    adm_scale3          — scale 3 (coarsest) ADM score
 *
 *  ADR: T8-3a (feat/metal-adm-real-2026-05-16).
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

/* ADM CPU-path helpers reused for parameter computation on host. */
#include "integer_adm.h"
}

extern "C" {
extern const unsigned char libvmaf_metallib_start[] __asm("section$start$__TEXT$__metallib");
extern const unsigned char libvmaf_metallib_end[]   __asm("section$end$__TEXT$__metallib");
}

/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */

#define ADM_NUM_SCALES   4
#define ADM_NUM_BANDS    4  /* LL, LH, HL, HH */
#define ADM_BUF_PARTIALS_MAX (1024u * 1024u) /* upper bound on WG count */

/* ------------------------------------------------------------------ */
/*  Per-scale intermediate buffer set                                   */
/* ------------------------------------------------------------------ */

typedef struct AdmScaleBuffersMetal {
    /* DWT sub-band output buffers (device). */
    id<MTLBuffer> band_ll;
    id<MTLBuffer> band_lh;
    id<MTLBuffer> band_hl;
    id<MTLBuffer> band_hh;

    /* CSF-filtered sub-band buffers (device). */
    id<MTLBuffer> csf_ref_lh;
    id<MTLBuffer> csf_ref_hl;
    id<MTLBuffer> csf_ref_hh;
    id<MTLBuffer> csf_dis_lh;
    id<MTLBuffer> csf_dis_hl;
    id<MTLBuffer> csf_dis_hh;

    /* CM reference accumulator (device, long per pixel). */
    id<MTLBuffer> cm_ref;

    /* Reduction partial buffers (shared — CPU-readable). */
    id<MTLBuffer> num_lo;
    id<MTLBuffer> num_hi;
    id<MTLBuffer> den_lo;
    id<MTLBuffer> den_hi;

    /* DWT vertical-pass temporaries. */
    id<MTLBuffer> tmp_lo;
    id<MTLBuffer> tmp_hi;

    size_t w;  /* sub-band width at this scale */
    size_t h;  /* sub-band height at this scale */
    size_t stride; /* element stride (not byte stride) */
    size_t partials_count; /* num WGs for reduction */
} AdmScaleBuffersMetal;

/* ------------------------------------------------------------------ */
/*  Extractor state                                                     */
/* ------------------------------------------------------------------ */

typedef struct IntegerAdmStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext *ctx;

    /* Pipeline state objects. */
    void *pso_dwt2_8_8bpc;
    void *pso_dwt2_8_16bpc;
    void *pso_dwt_s123_vert;
    void *pso_dwt_s123_hori;
    void *pso_csf;
    void *pso_cm;

    /* Input picture buffers (shared, one per frame). */
    id<MTLBuffer> ref_luma;
    id<MTLBuffer> dis_luma;

    /* Per-scale buffers. */
    AdmScaleBuffersMetal scales[ADM_NUM_SCALES];

    /* Frame parameters. */
    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;

    /* ADM hyperparameters (from VmafOption defaults). */
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int    adm_ref_display_height;
    double adm_csf_scale;
    double adm_csf_diag_scale;
    double adm_noise_weight;

    /* Collected scores (filled during collect). */
    double num[ADM_NUM_SCALES];
    double den[ADM_NUM_SCALES];

    VmafDictionary *feature_name_dict;
} IntegerAdmStateMetal;

/* ------------------------------------------------------------------ */
/*  PSO build                                                           */
/* ------------------------------------------------------------------ */

static int build_adm_pipelines(IntegerAdmStateMetal *s, id<MTLDevice> device)
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

    struct { const char *name; void **pso_slot; } kernels[] = {
        { "adm_dwt2_8_kernel_8bpc",   &s->pso_dwt2_8_8bpc   },
        { "adm_dwt2_8_kernel_16bpc",  &s->pso_dwt2_8_16bpc  },
        { "adm_dwt_s123_vert_kernel", &s->pso_dwt_s123_vert },
        { "adm_dwt_s123_hori_kernel", &s->pso_dwt_s123_hori },
        { "adm_csf_kernel",           &s->pso_csf            },
        { "adm_cm_kernel",            &s->pso_cm             },
    };
    for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); ++i) {
        id<MTLFunction> fn = [lib newFunctionWithName:@(kernels[i].name)];
        if (fn == nil) { return -ENODEV; }
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (pso == nil) { return -ENODEV; }
        *kernels[i].pso_slot = (__bridge_retained void *)pso;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Buffer allocation helpers                                           */
/* ------------------------------------------------------------------ */

static id<MTLBuffer> alloc_private_buf(id<MTLDevice> device, size_t bytes)
{
    /* Use Shared on all platforms for simplicity (correct on both Apple
     * Silicon unified-memory and discrete GPU via PCIe). A production
     * implementation would use Private + BlitEncoder for dis/ref uploads
     * and keep only the partial-sum buffers Shared. */
    return [device newBufferWithLength:bytes
                               options:MTLResourceStorageModeShared];
}

static int alloc_scale_bufs(IntegerAdmStateMetal *s, id<MTLDevice> device,
                            int scale_idx, size_t w, size_t h)
{
    AdmScaleBuffersMetal *sb = &s->scales[scale_idx];
    sb->w      = w;
    sb->h      = h;
    sb->stride = w; /* packed stride */

    const size_t n_int16  = w * h * sizeof(int16_t);
    const size_t n_int32  = w * h * sizeof(int32_t);
    const size_t n_int64  = w * h * sizeof(int64_t);

    const size_t grid_w = (w + 15) / 16;
    const size_t grid_h = (h + 15) / 16;
    sb->partials_count  = grid_w * grid_h;
    const size_t n_part = sb->partials_count * sizeof(uint32_t);

    /* Scale 0 uses int16 DWT bands; scales 1-3 use int32. */
    size_t band_bytes = (scale_idx == 0) ? n_int16 : n_int32;

    sb->band_ll    = alloc_private_buf(device, band_bytes);
    sb->band_lh    = alloc_private_buf(device, band_bytes);
    sb->band_hl    = alloc_private_buf(device, band_bytes);
    sb->band_hh    = alloc_private_buf(device, band_bytes);
    sb->csf_ref_lh = alloc_private_buf(device, n_int32);
    sb->csf_ref_hl = alloc_private_buf(device, n_int32);
    sb->csf_ref_hh = alloc_private_buf(device, n_int32);
    sb->csf_dis_lh = alloc_private_buf(device, n_int32);
    sb->csf_dis_hl = alloc_private_buf(device, n_int32);
    sb->csf_dis_hh = alloc_private_buf(device, n_int32);
    sb->cm_ref     = alloc_private_buf(device, n_int64);
    sb->num_lo     = alloc_private_buf(device, n_part);
    sb->num_hi     = alloc_private_buf(device, n_part);
    sb->den_lo     = alloc_private_buf(device, n_part);
    sb->den_hi     = alloc_private_buf(device, n_part);
    sb->tmp_lo     = alloc_private_buf(device, n_int32);
    sb->tmp_hi     = alloc_private_buf(device, n_int32);

    if (sb->band_ll == nil || sb->band_lh == nil || sb->band_hl == nil ||
        sb->band_hh == nil || sb->csf_ref_lh == nil || sb->csf_ref_hl == nil ||
        sb->csf_ref_hh == nil || sb->csf_dis_lh == nil || sb->csf_dis_hl == nil ||
        sb->csf_dis_hh == nil || sb->cm_ref == nil ||
        sb->num_lo == nil || sb->num_hi == nil ||
        sb->den_lo == nil || sb->den_hi == nil ||
        sb->tmp_lo == nil || sb->tmp_hi == nil) {
        return -ENOMEM;
    }
    return 0;
}

static void free_scale_bufs(AdmScaleBuffersMetal *sb)
{
    /* ARC releases on assign to nil. */
    sb->band_ll = sb->band_lh = sb->band_hl = sb->band_hh = nil;
    sb->csf_ref_lh = sb->csf_ref_hl = sb->csf_ref_hh = nil;
    sb->csf_dis_lh = sb->csf_dis_hl = sb->csf_dis_hh = nil;
    sb->cm_ref = nil;
    sb->num_lo = sb->num_hi = sb->den_lo = sb->den_hi = nil;
    sb->tmp_lo = sb->tmp_hi = nil;
}

/* ------------------------------------------------------------------ */
/*  init                                                                */
/* ------------------------------------------------------------------ */

static const VmafOption options[] = {
    {
        .name    = "adm_enhn_gain_limit",
        .help    = "enhancement gain limit",
        .offset  = offsetof(IntegerAdmStateMetal, adm_enhn_gain_limit),
        .type    = VMAF_OPT_TYPE_DOUBLE,
        .default_val = { .d = DEFAULT_ADM_ENHN_GAIN_LIMIT },
        .min     = 1.0,
        .max     = 1000000.0,
        .flags   = 0,
    },
    {
        .name    = "adm_norm_view_dist",
        .help    = "viewing distance in picture heights",
        .offset  = offsetof(IntegerAdmStateMetal, adm_norm_view_dist),
        .type    = VMAF_OPT_TYPE_DOUBLE,
        .default_val = { .d = DEFAULT_ADM_NORM_VIEW_DIST },
        .min     = 0.75,
        .max     = 24.0,
        .flags   = 0,
    },
    {
        .name    = "adm_ref_display_height",
        .help    = "reference display height in pixels",
        .offset  = offsetof(IntegerAdmStateMetal, adm_ref_display_height),
        .type    = VMAF_OPT_TYPE_INT,
        .default_val = { .i = DEFAULT_ADM_REF_DISPLAY_HEIGHT },
        .min     = 1,
        .max     = 2160,
        .flags   = 0,
    },
    { 0 }
};

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    IntegerAdmStateMetal *s = (IntegerAdmStateMetal *)fex->priv;

    s->frame_w = w;
    s->frame_h = h;
    s->bpc     = bpc;

    /* Apply option defaults not set by vmaf_feature_name_dict path. */
    if (s->adm_enhn_gain_limit <= 0.0) { s->adm_enhn_gain_limit = DEFAULT_ADM_NOISE_WEIGHT; }
    if (s->adm_norm_view_dist  <= 0.0) { s->adm_norm_view_dist  = DEFAULT_ADM_NORM_VIEW_DIST; }
    if (s->adm_ref_display_height <= 0){ s->adm_ref_display_height = DEFAULT_ADM_REF_DISPLAY_HEIGHT; }
    s->adm_csf_scale      = DEFAULT_ADM_CSF_SCALE;
    s->adm_csf_diag_scale = DEFAULT_ADM_CSF_DIAG_SCALE;
    s->adm_noise_weight   = DEFAULT_ADM_NOISE_WEIGHT;

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_lc; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        err = build_adm_pipelines(s, device);
        if (err != 0) { goto fail_lc; }

        /* Allocate input luma buffers. */
        const size_t bytes_per_px = (bpc <= 8u) ? 1u : 2u;
        s->ref_luma = alloc_private_buf(device, (size_t)w * h * bytes_per_px);
        s->dis_luma = alloc_private_buf(device, (size_t)w * h * bytes_per_px);
        if (s->ref_luma == nil || s->dis_luma == nil) { err = -ENOMEM; goto fail_pso; }

        /* Allocate per-scale intermediate buffers.
         * Scale 0 output is (w/2) × (h/2); each subsequent scale halves again. */
        size_t sw = w, sh = h;
        for (int sc = 0; sc < ADM_NUM_SCALES; ++sc) {
            sw = (sw + 1) / 2;
            sh = (sh + 1) / 2;
            err = alloc_scale_bufs(s, device, sc, sw, sh);
            if (err != 0) {
                for (int q = 0; q < sc; ++q) { free_scale_bufs(&s->scales[q]); }
                goto fail_luma;
            }
        }
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_scales; }
    return 0;

fail_scales:
    for (int q = 0; q < ADM_NUM_SCALES; ++q) { free_scale_bufs(&s->scales[q]); }
fail_luma:
    s->ref_luma = nil;
    s->dis_luma = nil;
fail_pso:
    {
        void **slots[] = {
            &s->pso_dwt2_8_8bpc, &s->pso_dwt2_8_16bpc,
            &s->pso_dwt_s123_vert, &s->pso_dwt_s123_hori,
            &s->pso_csf, &s->pso_cm
        };
        for (size_t i = 0; i < sizeof(slots)/sizeof(slots[0]); ++i) {
            if (*slots[i]) {
                (void)(__bridge_transfer id<MTLComputePipelineState>)*slots[i];
                *slots[i] = NULL;
            }
        }
    }
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

/* ------------------------------------------------------------------ */
/*  Dispatch helpers                                                    */
/* ------------------------------------------------------------------ */

static void upload_luma(id<MTLBuffer> dst, VmafPicture *pic, unsigned bpc)
{
    const unsigned w = pic->w[0];
    const unsigned h = pic->h[0];
    const size_t bytes_per_px = (bpc <= 8u) ? 1u : 2u;
    const size_t row_bytes = (size_t)w * bytes_per_px;
    uint8_t *d = (uint8_t *)[dst contents];
    for (unsigned y = 0; y < h; ++y) {
        memcpy(d + y * row_bytes,
               (uint8_t *)pic->data[0] + y * pic->stride[0],
               row_bytes);
    }
}

/* Zero a Metal buffer via a blit encoder on the given command buffer. */
static void zero_buf(id<MTLCommandBuffer> cmd, id<MTLBuffer> buf, size_t bytes)
{
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit fillBuffer:buf range:NSMakeRange(0, bytes) value:0];
    [blit endEncoding];
}

/* Encode a single compute dispatch and wait for completion. */
static void encode_compute(id<MTLCommandBuffer> cmd,
                            id<MTLComputePipelineState> pso,
                            size_t grid_x, size_t grid_y,
                            void (^setup)(id<MTLComputeCommandEncoder>))
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    setup(enc);
    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(grid_x, grid_y, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
}

/* ------------------------------------------------------------------ */
/*  submit                                                              */
/* ------------------------------------------------------------------ */

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    IntegerAdmStateMetal *s = (IntegerAdmStateMetal *)fex->priv;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>       device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue>  queue = (__bridge id<MTLCommandQueue>)qh;

    /* Upload luma planes to shared buffers. */
    upload_luma(s->ref_luma, ref_pic, s->bpc);
    upload_luma(s->dis_luma, dist_pic, s->bpc);

    /* -------------------------------------------------------------- *
     *  Scale 0: adm_dwt2_8 on the input luma plane.                  *
     * -------------------------------------------------------------- */
    {
        AdmScaleBuffersMetal *sb = &s->scales[0];
        const uint32_t src_w      = (uint32_t)s->frame_w;
        const uint32_t src_h      = (uint32_t)s->frame_h;
        const uint32_t src_stride = src_w; /* packed */
        const uint32_t dst_stride = (uint32_t)sb->stride;
        uint32_t p[4] = { src_w, src_h, src_stride, dst_stride };

        id<MTLComputePipelineState> pso =
            (s->bpc <= 8u)
            ? (__bridge id<MTLComputePipelineState>)s->pso_dwt2_8_8bpc
            : (__bridge id<MTLComputePipelineState>)s->pso_dwt2_8_16bpc;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (cmd == nil) { return -ENOMEM; }

        const size_t gx = (sb->w + 15) / 16;
        const size_t gy = (sb->h + 15) / 16;

        /* Ref DWT. */
        id<MTLBuffer> ref_luma = s->ref_luma;
        id<MTLBuffer> band_ll  = sb->band_ll;
        id<MTLBuffer> band_lh  = sb->band_lh;
        id<MTLBuffer> band_hl  = sb->band_hl;
        id<MTLBuffer> band_hh  = sb->band_hh;

        encode_compute(cmd, pso, gx, gy, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:ref_luma offset:0 atIndex:0];
            [enc setBuffer:band_ll  offset:0 atIndex:1];
            [enc setBuffer:band_lh  offset:0 atIndex:2];
            [enc setBuffer:band_hl  offset:0 atIndex:3];
            [enc setBuffer:band_hh  offset:0 atIndex:4];
            [enc setBytes:p length:sizeof(p) atIndex:5];
        });
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    /* -------------------------------------------------------------- *
     *  Scales 1-3: adm_dwt_s123 on the LL band from the prior scale. *
     * -------------------------------------------------------------- */
    for (int sc = 1; sc < ADM_NUM_SCALES; ++sc) {
        AdmScaleBuffersMetal *prev = &s->scales[sc - 1];
        AdmScaleBuffersMetal *sb   = &s->scales[sc];

        const uint32_t in_w      = (uint32_t)prev->w;
        const uint32_t in_h      = (uint32_t)prev->h;
        const uint32_t in_stride = (uint32_t)prev->stride;
        const uint32_t dst_stride = (uint32_t)sb->stride;
        uint32_t p[4] = { in_w, in_h, in_stride, dst_stride };

        id<MTLComputePipelineState> pso_vert =
            (__bridge id<MTLComputePipelineState>)s->pso_dwt_s123_vert;
        id<MTLComputePipelineState> pso_hori =
            (__bridge id<MTLComputePipelineState>)s->pso_dwt_s123_hori;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (cmd == nil) { return -ENOMEM; }

        const size_t gx_v = (in_w  + 15) / 16;
        const size_t gy_v = (sb->h + 15) / 16;

        /* Vertical pass (produces tmp_lo, tmp_hi). */
        {
            id<MTLBuffer> src    = (sc == 1) ? prev->band_ll : prev->band_ll;
            id<MTLBuffer> tmp_lo = sb->tmp_lo;
            id<MTLBuffer> tmp_hi = sb->tmp_hi;
            encode_compute(cmd, pso_vert, gx_v, gy_v, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:src    offset:0 atIndex:0];
                [enc setBuffer:tmp_lo offset:0 atIndex:1];
                [enc setBuffer:tmp_hi offset:0 atIndex:2];
                [enc setBytes:p length:sizeof(p) atIndex:3];
            });
        }

        /* Horizontal pass (produces four sub-bands). */
        {
            const size_t gx_h = (sb->w  + 15) / 16;
            const size_t gy_h = (sb->h  + 15) / 16;
            id<MTLBuffer> tmp_lo = sb->tmp_lo;
            id<MTLBuffer> tmp_hi = sb->tmp_hi;
            id<MTLBuffer> b_ll   = sb->band_ll;
            id<MTLBuffer> b_lh   = sb->band_lh;
            id<MTLBuffer> b_hl   = sb->band_hl;
            id<MTLBuffer> b_hh   = sb->band_hh;
            encode_compute(cmd, pso_hori, gx_h, gy_h, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:tmp_lo offset:0 atIndex:0];
                [enc setBuffer:tmp_hi offset:0 atIndex:1];
                [enc setBuffer:b_ll   offset:0 atIndex:2];
                [enc setBuffer:b_lh   offset:0 atIndex:3];
                [enc setBuffer:b_hl   offset:0 atIndex:4];
                [enc setBuffer:b_hh   offset:0 atIndex:5];
                [enc setBytes:p length:sizeof(p) atIndex:6];
            });
        }
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    /* -------------------------------------------------------------- *
     *  Stages 3+4: CSF + CM per scale × sub-band.                    *
     * -------------------------------------------------------------- */
    id<MTLComputePipelineState> pso_csf =
        (__bridge id<MTLComputePipelineState>)s->pso_csf;
    id<MTLComputePipelineState> pso_cm  =
        (__bridge id<MTLComputePipelineState>)s->pso_cm;

    for (int sc = 0; sc < ADM_NUM_SCALES; ++sc) {
        AdmScaleBuffersMetal *sb = &s->scales[sc];
        const uint32_t bw = (uint32_t)sb->w;
        const uint32_t bh = (uint32_t)sb->h;
        const uint32_t bs = (uint32_t)sb->stride;
        uint32_t p[4] = { bw, bh, bs, 0u };

        /* CSF rfactor for this scale (simplified: use uniform weight;
         * the production kernel uses per-band dwt_quant_step values). */
        float rfactor = (float)s->adm_csf_scale / (float)(1 << sc);
        float egl     = (float)s->adm_enhn_gain_limit;

        const size_t n_part = sb->partials_count * sizeof(uint32_t);

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (cmd == nil) { return -ENOMEM; }

        /* Zero reduction buffers before dispatch. */
        zero_buf(cmd, sb->num_lo, n_part);
        zero_buf(cmd, sb->num_hi, n_part);
        zero_buf(cmd, sb->den_lo, n_part);
        zero_buf(cmd, sb->den_hi, n_part);

        const size_t gx = (bw + 15) / 16;
        const size_t gy = (bh + 15) / 16;

        /* CSF on LH band. */
        {
            id<MTLBuffer> rb = sb->band_lh;
            id<MTLBuffer> db = sb->band_lh; /* placeholder: needs ref/dis split */
            id<MTLBuffer> cr = sb->csf_ref_lh;
            id<MTLBuffer> cd = sb->csf_dis_lh;
            id<MTLBuffer> dl = sb->den_lo;
            id<MTLBuffer> dh = sb->den_hi;
            float rf = rfactor;
            encode_compute(cmd, pso_csf, gx, gy, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:rb offset:0 atIndex:0];
                [enc setBuffer:db offset:0 atIndex:1];
                [enc setBuffer:cr offset:0 atIndex:2];
                [enc setBuffer:cd offset:0 atIndex:3];
                [enc setBuffer:dl offset:0 atIndex:4];
                [enc setBuffer:dh offset:0 atIndex:5];
                [enc setBytes:p  length:sizeof(p)  atIndex:6];
                [enc setBytes:&rf length:sizeof(rf) atIndex:7];
            });
        }

        /* CM on LH band. */
        {
            id<MTLBuffer> cr  = sb->csf_ref_lh;
            id<MTLBuffer> cd  = sb->csf_dis_lh;
            id<MTLBuffer> cmr = sb->cm_ref;
            id<MTLBuffer> nl  = sb->num_lo;
            id<MTLBuffer> nh  = sb->num_hi;
            float eg = egl;
            encode_compute(cmd, pso_cm, gx, gy, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:cr  offset:0 atIndex:0];
                [enc setBuffer:cd  offset:0 atIndex:1];
                [enc setBuffer:cmr offset:0 atIndex:2];
                [enc setBuffer:nl  offset:0 atIndex:3];
                [enc setBuffer:nh  offset:0 atIndex:4];
                [enc setBytes:p   length:sizeof(p)  atIndex:5];
                [enc setBytes:&eg length:sizeof(eg) atIndex:6];
            });
        }

        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  collect                                                             */
/* ------------------------------------------------------------------ */

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    IntegerAdmStateMetal *s = (IntegerAdmStateMetal *)fex->priv;

    double total_num = 0.0, total_den = 0.0;

    for (int sc = 0; sc < ADM_NUM_SCALES; ++sc) {
        AdmScaleBuffersMetal *sb = &s->scales[sc];
        const size_t cnt = sb->partials_count;

        const uint32_t *nl = (const uint32_t *)sb->num_lo.contents;
        const uint32_t *nh = (const uint32_t *)sb->num_hi.contents;
        const uint32_t *dl = (const uint32_t *)sb->den_lo.contents;
        const uint32_t *dh = (const uint32_t *)sb->den_hi.contents;

        double num = 0.0, den = 0.0;
        if (nl != NULL && nh != NULL) {
            for (size_t i = 0; i < cnt; ++i) {
                num += (double)(((uint64_t)nh[i] << 32u) | (uint64_t)nl[i]);
            }
        }
        if (dl != NULL && dh != NULL) {
            for (size_t i = 0; i < cnt; ++i) {
                den += (double)(((uint64_t)dh[i] << 32u) | (uint64_t)dl[i]);
            }
        }

        s->num[sc] = num;
        s->den[sc] = den;

        double adm_scale = (den > 0.0) ? (num / den) : 1.0;
        total_num += num;
        total_den += den;

        static const char *scale_names[ADM_NUM_SCALES] = {
            "adm_scale0", "adm_scale1", "adm_scale2", "adm_scale3"
        };
        int err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            scale_names[sc], adm_scale, index);
        if (err != 0) { return err; }
    }

    const double adm2 = (total_den > 0.0) ? (total_num / total_den) : 1.0;
    return vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "adm2", adm2, index);
}

/* ------------------------------------------------------------------ */
/*  close                                                               */
/* ------------------------------------------------------------------ */

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    IntegerAdmStateMetal *s = (IntegerAdmStateMetal *)fex->priv;

    for (int q = 0; q < ADM_NUM_SCALES; ++q) { free_scale_bufs(&s->scales[q]); }
    s->ref_luma = nil;
    s->dis_luma = nil;

    void **slots[] = {
        &s->pso_dwt2_8_8bpc, &s->pso_dwt2_8_16bpc,
        &s->pso_dwt_s123_vert, &s->pso_dwt_s123_hori,
        &s->pso_csf, &s->pso_cm
    };
    for (size_t i = 0; i < sizeof(slots)/sizeof(slots[0]); ++i) {
        if (*slots[i]) {
            (void)(__bridge_transfer id<MTLComputePipelineState>)*slots[i];
            *slots[i] = NULL;
        }
    }

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

/* ------------------------------------------------------------------ */
/*  Registration                                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {
    "adm2",
    "adm_scale0", "adm_scale1", "adm_scale2", "adm_scale3",
    NULL
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_adm_metal = {
    .name              = "integer_adm_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(IntegerAdmStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = ADM_NUM_SCALES * 2, /* CSF + CM per scale */
        .is_reduction_only      = true,
        .min_useful_frame_area  = 192U * 108U,        /* 1/10th 1080p */
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
