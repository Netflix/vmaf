/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_vif feature extractor on the Metal backend.
 *
 *  Port of libvmaf/src/feature/cuda/float_vif_cuda.c to Metal. The
 *  algorithm is identical to the CUDA twin (4 compute + 3 decimate
 *  dispatches per frame, scale 0-3 pyramid, partial-sum reduction):
 *
 *    init:   allocate raw upload buffers, ping-pong float buffers for
 *            scales 1-3, and per-scale num/den partial arrays.
 *    submit: for scale 0: compute dispatch (raw input).
 *            for scale 1-3: decimate from previous scale → float buf,
 *                           then compute dispatch on that buf.
 *            Synchronise, read back partials.
 *    collect: sum partials per scale, emit 4 vif_scale scores.
 *
 *  Score formula: score[s] = sum(num_partials[s]) / sum(den_partials[s])
 *
 *  Feature names emitted:
 *    VMAF_feature_vif_scale{0,1,2,3}_score
 *    vif, vif_num, vif_den              (debug=true only)
 *    vif_num_scale{0,1,2,3}            (debug=true only)
 *    vif_den_scale{0,1,2,3}            (debug=true only)
 *
 *  Limitations (v1): vif_kernelscale must be 1.0 (matches CUDA twin).
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

#define FVIF_SCALES 4
#define FVIF_BX     16
#define FVIF_BY     16

typedef struct FloatVifStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext *ctx;

    void *pso_compute;     /* __bridge_retained id<MTLComputePipelineState> */
    void *pso_decimate;

    /* Raw upload buffers (host-side row-packed luma plane). */
    void *ref_raw_buf;     /* __bridge_retained id<MTLBuffer> */
    void *dis_raw_buf;

    /* Ping-pong float buffers for scales 1-3 (size = scale_w[1]*scale_h[1]). */
    void *ref_float_buf[2];
    void *dis_float_buf[2];

    /* Per-scale num/den partial arrays. */
    void *num_parts_buf[FVIF_SCALES];
    void *den_parts_buf[FVIF_SCALES];

    /* Host-side partial readback (allocated at init, reused per frame). */
    float *num_host[FVIF_SCALES];
    float *den_host[FVIF_SCALES];
    unsigned wg_count[FVIF_SCALES];

    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned scale_w[FVIF_SCALES];
    unsigned scale_h[FVIF_SCALES];

    double vif_enhn_gain_limit;
    double vif_kernelscale;
    double vif_sigma_nsq;
    bool   debug;

    VmafDictionary *feature_name_dict;
} FloatVifStateMetal;

/* ------------------------------------------------------------------ */
/*  Options                                                             */
/* ------------------------------------------------------------------ */
static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(FloatVifStateMetal, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif (>= 1.0)",
        .offset = offsetof(FloatVifStateMetal, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 100.0,
        .min = 1.0,
        .max = 100.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_kernelscale",
        .help = "scaling factor for the gaussian kernel (1.0 only in v1)",
        .offset = offsetof(FloatVifStateMetal, vif_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_sigma_nsq",
        .alias = "snsq",
        .help = "neural noise variance",
        .offset = offsetof(FloatVifStateMetal, vif_sigma_nsq),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 2.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0},
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */
static void compute_scale_dims(FloatVifStateMetal *s)
{
    s->scale_w[0] = s->width;
    s->scale_h[0] = s->height;
    for (int i = 1; i < FVIF_SCALES; i++) {
        s->scale_w[i] = s->scale_w[i - 1] / 2u;
        s->scale_h[i] = s->scale_h[i - 1] / 2u;
    }
}

static int build_pipelines(FloatVifStateMetal *s, id<MTLDevice> device)
{
    const size_t blob_sz = (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_sz == 0u) { return -ENODEV; }

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_sz,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL) { return -ENOMEM; }

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil) { return -ENODEV; }

    id<MTLFunction> fn_comp = [lib newFunctionWithName:@"float_vif_compute"];
    id<MTLFunction> fn_dec  = [lib newFunctionWithName:@"float_vif_decimate"];
    if (fn_comp == nil || fn_dec == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_c =
        [device newComputePipelineStateWithFunction:fn_comp error:&err];
    id<MTLComputePipelineState> pso_d =
        [device newComputePipelineStateWithFunction:fn_dec  error:&err];
    if (pso_c == nil || pso_d == nil) { return -ENODEV; }

    s->pso_compute  = (__bridge_retained void *)pso_c;
    s->pso_decimate = (__bridge_retained void *)pso_d;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  init                                                                */
/* ------------------------------------------------------------------ */
static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatVifStateMetal *s = (FloatVifStateMetal *)fex->priv;

    if (s->vif_kernelscale != 1.0) { return -EINVAL; }

    s->width  = w;
    s->height = h;
    s->bpc    = bpc;
    compute_scale_dims(s);

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_lc; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        /* Raw upload buffers for luma plane. */
        const size_t bpp       = (bpc <= 8u) ? 1u : 2u;
        const size_t raw_bytes = (size_t)w * h * bpp;
        id<MTLBuffer> rrb = [device newBufferWithLength:raw_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> drb = [device newBufferWithLength:raw_bytes options:MTLResourceStorageModeShared];
        if (rrb == nil || drb == nil) { err = -ENOMEM; goto fail_lc; }
        s->ref_raw_buf = (__bridge_retained void *)rrb;
        s->dis_raw_buf = (__bridge_retained void *)drb;

        /* Ping-pong float buffers sized to scale 1 (largest float scale). */
        const size_t fbytes = (size_t)s->scale_w[1] * s->scale_h[1] * sizeof(float);
        for (int i = 0; i < 2; i++) {
            id<MTLBuffer> rb = [device newBufferWithLength:fbytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> db = [device newBufferWithLength:fbytes options:MTLResourceStorageModeShared];
            if (rb == nil || db == nil) { err = -ENOMEM; goto fail_raw; }
            s->ref_float_buf[i] = (__bridge_retained void *)rb;
            s->dis_float_buf[i] = (__bridge_retained void *)db;
        }

        /* Per-scale partial buffers. */
        for (int i = 0; i < FVIF_SCALES; i++) {
            const unsigned gx = (s->scale_w[i] + FVIF_BX - 1u) / FVIF_BX;
            const unsigned gy = (s->scale_h[i] + FVIF_BY - 1u) / FVIF_BY;
            s->wg_count[i] = gx * gy;
            const size_t pbytes = (size_t)s->wg_count[i] * sizeof(float);

            id<MTLBuffer> nb = [device newBufferWithLength:pbytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> db = [device newBufferWithLength:pbytes options:MTLResourceStorageModeShared];
            if (nb == nil || db == nil) { err = -ENOMEM; goto fail_float; }
            s->num_parts_buf[i] = (__bridge_retained void *)nb;
            s->den_parts_buf[i] = (__bridge_retained void *)db;

            s->num_host[i] = (float *)malloc(pbytes);
            s->den_host[i] = (float *)malloc(pbytes);
            if (s->num_host[i] == NULL || s->den_host[i] == NULL) {
                err = -ENOMEM;
                goto fail_parts;
            }
        }

        err = build_pipelines(s, device);
    }
    if (err != 0) { goto fail_pso; }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    if (s->pso_compute)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_compute;  s->pso_compute  = NULL; }
    if (s->pso_decimate) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_decimate; s->pso_decimate = NULL; }
fail_parts:
    for (int i = 0; i < FVIF_SCALES; i++) {
        free(s->num_host[i]); s->num_host[i] = NULL;
        free(s->den_host[i]); s->den_host[i] = NULL;
        if (s->num_parts_buf[i]) { (void)(__bridge_transfer id<MTLBuffer>)s->num_parts_buf[i]; s->num_parts_buf[i] = NULL; }
        if (s->den_parts_buf[i]) { (void)(__bridge_transfer id<MTLBuffer>)s->den_parts_buf[i]; s->den_parts_buf[i] = NULL; }
    }
fail_float:
    for (int i = 0; i < 2; i++) {
        if (s->ref_float_buf[i]) { (void)(__bridge_transfer id<MTLBuffer>)s->ref_float_buf[i]; s->ref_float_buf[i] = NULL; }
        if (s->dis_float_buf[i]) { (void)(__bridge_transfer id<MTLBuffer>)s->dis_float_buf[i]; s->dis_float_buf[i] = NULL; }
    }
fail_raw:
    if (s->ref_raw_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->ref_raw_buf; s->ref_raw_buf = NULL; }
    if (s->dis_raw_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->dis_raw_buf; s->dis_raw_buf = NULL; }
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

/* ------------------------------------------------------------------ */
/*  Param structs (must match float_vif.metal layout)                  */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t scale;
    uint32_t bpc;
    uint32_t grid_x;
    uint32_t f_stride;
    float    vif_sigma_nsq;
    float    vif_enhn_gain_limit;
    uint32_t pad0;
    uint32_t pad1;
} FvifComputeParams;

typedef struct {
    uint32_t out_w;
    uint32_t out_h;
    uint32_t in_w;
    uint32_t in_h;
    uint32_t scale;
    uint32_t bpc;
    uint32_t in_f_stride;
    uint32_t out_stride;
} FvifDecimateParams;

/* ------------------------------------------------------------------ */
/*  submit                                                              */
/* ------------------------------------------------------------------ */
static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    FloatVifStateMetal *s = (FloatVifStateMetal *)fex->priv;

    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    void *dh = vmaf_metal_context_device_handle(s->ctx);
    if (qh == NULL || dh == NULL) { return -ENODEV; }

    id<MTLCommandQueue> queue  = (__bridge id<MTLCommandQueue>)qh;
    id<MTLCommandBuffer> cmd   = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    /* Upload luma planes (stride-unpacking). */
    const size_t bpp   = (s->bpc <= 8u) ? 1u : 2u;
    const size_t row_b = (size_t)s->width * bpp;

    {
        uint8_t *rd = (uint8_t *)[(__bridge id<MTLBuffer>)s->ref_raw_buf contents];
        uint8_t *dd = (uint8_t *)[(__bridge id<MTLBuffer>)s->dis_raw_buf contents];
        for (unsigned y = 0; y < s->height; y++) {
            memcpy(rd + y * row_b, (uint8_t *)ref_pic->data[0]  + y * ref_pic->stride[0],  row_b);
            memcpy(dd + y * row_b, (uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0], row_b);
        }
    }

    id<MTLComputePipelineState> pso_comp =
        (__bridge id<MTLComputePipelineState>)s->pso_compute;
    id<MTLComputePipelineState> pso_dec  =
        (__bridge id<MTLComputePipelineState>)s->pso_decimate;

    id<MTLBuffer> ref_raw = (__bridge id<MTLBuffer>)s->ref_raw_buf;
    id<MTLBuffer> dis_raw = (__bridge id<MTLBuffer>)s->dis_raw_buf;

    /* Zero all partial buffers via blit. */
    {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        for (int i = 0; i < FVIF_SCALES; i++) {
            id<MTLBuffer> nb = (__bridge id<MTLBuffer>)s->num_parts_buf[i];
            id<MTLBuffer> db = (__bridge id<MTLBuffer>)s->den_parts_buf[i];
            [blit fillBuffer:nb range:NSMakeRange(0, s->wg_count[i] * sizeof(float)) value:0];
            [blit fillBuffer:db range:NSMakeRange(0, s->wg_count[i] * sizeof(float)) value:0];
        }
        [blit endEncoding];
    }

    /* Launch scale 0 compute (reads raw buffers). */
    {
        const unsigned gx = (s->scale_w[0] + FVIF_BX - 1u) / FVIF_BX;
        const unsigned gy = (s->scale_h[0] + FVIF_BY - 1u) / FVIF_BY;
        FvifComputeParams cp = {};
        cp.width              = s->scale_w[0];
        cp.height             = s->scale_h[0];
        cp.scale              = 0u;
        cp.bpc                = s->bpc;
        cp.grid_x             = gx;
        cp.f_stride           = 0u;   /* unused at scale 0 */
        cp.vif_sigma_nsq       = (float)s->vif_sigma_nsq;
        cp.vif_enhn_gain_limit = (float)s->vif_enhn_gain_limit;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_comp];
        [enc setBuffer:ref_raw                                        offset:0 atIndex:0];
        [enc setBuffer:dis_raw                                        offset:0 atIndex:1];
        [enc setBuffer:ref_raw                                        offset:0 atIndex:2]; /* unused */
        [enc setBuffer:dis_raw                                        offset:0 atIndex:3]; /* unused */
        [enc setBuffer:(__bridge id<MTLBuffer>)s->num_parts_buf[0]   offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)s->den_parts_buf[0]   offset:0 atIndex:5];
        [enc setBytes:&cp length:sizeof(cp)                                    atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
              threadsPerThreadgroup:MTLSizeMake(FVIF_BX, FVIF_BY, 1)];
        [enc endEncoding];
    }

    /* Launch scales 1-3: decimate → compute. */
    for (int sc = 1; sc < FVIF_SCALES; sc++) {
        /* dst_idx toggles ping-pong: sc=1→buf[0], sc=2→buf[1], sc=3→buf[0]. */
        const int dst_idx = (sc - 1) % 2;
        const bool prev_is_raw = (sc == 1);

        /* Previous scale float buffers (source for decimate). */
        id<MTLBuffer> prev_ref_f = prev_is_raw ? ref_raw :
            (__bridge id<MTLBuffer>)s->ref_float_buf[(dst_idx == 0) ? 1 : 0];
        id<MTLBuffer> prev_dis_f = prev_is_raw ? dis_raw :
            (__bridge id<MTLBuffer>)s->dis_float_buf[(dst_idx == 0) ? 1 : 0];

        id<MTLBuffer> cur_ref_f = (__bridge id<MTLBuffer>)s->ref_float_buf[dst_idx];
        id<MTLBuffer> cur_dis_f = (__bridge id<MTLBuffer>)s->dis_float_buf[dst_idx];

        const unsigned in_w  = s->scale_w[sc - 1];
        const unsigned in_h  = s->scale_h[sc - 1];
        const unsigned out_w = s->scale_w[sc];
        const unsigned out_h = s->scale_h[sc];

        /* Decimate. */
        {
            FvifDecimateParams dp = {};
            dp.out_w       = out_w;
            dp.out_h       = out_h;
            dp.in_w        = in_w;
            dp.in_h        = in_h;
            dp.scale       = (uint32_t)sc;
            dp.bpc         = s->bpc;
            dp.in_f_stride = prev_is_raw ? 0u : in_w;
            dp.out_stride  = out_w;

            const unsigned dgx = (out_w + FVIF_BX - 1u) / FVIF_BX;
            const unsigned dgy = (out_h + FVIF_BY - 1u) / FVIF_BY;

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_dec];
            [enc setBuffer:ref_raw    offset:0 atIndex:0];
            [enc setBuffer:dis_raw    offset:0 atIndex:1];
            [enc setBuffer:prev_ref_f offset:0 atIndex:2];
            [enc setBuffer:prev_dis_f offset:0 atIndex:3];
            [enc setBuffer:cur_ref_f  offset:0 atIndex:4];
            [enc setBuffer:cur_dis_f  offset:0 atIndex:5];
            [enc setBytes:&dp length:sizeof(dp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(dgx, dgy, 1)
                  threadsPerThreadgroup:MTLSizeMake(FVIF_BX, FVIF_BY, 1)];
            [enc endEncoding];
        }

        /* Compute at this scale on the just-written float buffers. */
        {
            const unsigned gx = (out_w + FVIF_BX - 1u) / FVIF_BX;
            const unsigned gy = (out_h + FVIF_BY - 1u) / FVIF_BY;
            FvifComputeParams cp = {};
            cp.width              = out_w;
            cp.height             = out_h;
            cp.scale              = (uint32_t)sc;
            cp.bpc                = s->bpc;
            cp.grid_x             = gx;
            cp.f_stride           = out_w;
            cp.vif_sigma_nsq       = (float)s->vif_sigma_nsq;
            cp.vif_enhn_gain_limit = (float)s->vif_enhn_gain_limit;

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_comp];
            [enc setBuffer:ref_raw                                         offset:0 atIndex:0];
            [enc setBuffer:dis_raw                                         offset:0 atIndex:1];
            [enc setBuffer:cur_ref_f                                       offset:0 atIndex:2];
            [enc setBuffer:cur_dis_f                                       offset:0 atIndex:3];
            [enc setBuffer:(__bridge id<MTLBuffer>)s->num_parts_buf[sc]   offset:0 atIndex:4];
            [enc setBuffer:(__bridge id<MTLBuffer>)s->den_parts_buf[sc]   offset:0 atIndex:5];
            [enc setBytes:&cp length:sizeof(cp)                                     atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                  threadsPerThreadgroup:MTLSizeMake(FVIF_BX, FVIF_BY, 1)];
            [enc endEncoding];
        }
    }

    [cmd commit];
    [cmd waitUntilCompleted];

    /* Read back partials. */
    for (int i = 0; i < FVIF_SCALES; i++) {
        const float *nsrc = (const float *)[(__bridge id<MTLBuffer>)s->num_parts_buf[i] contents];
        const float *dsrc = (const float *)[(__bridge id<MTLBuffer>)s->den_parts_buf[i] contents];
        memcpy(s->num_host[i], nsrc, s->wg_count[i] * sizeof(float));
        memcpy(s->den_host[i], dsrc, s->wg_count[i] * sizeof(float));
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  collect                                                             */
/* ------------------------------------------------------------------ */
static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    FloatVifStateMetal *s = (FloatVifStateMetal *)fex->priv;

    double scores[8]; /* [num0, den0, num1, den1, ...] */
    for (int i = 0; i < FVIF_SCALES; i++) {
        double n = 0.0, d = 0.0;
        for (unsigned j = 0u; j < s->wg_count[i]; j++) {
            n += (double)s->num_host[i][j];
            d += (double)s->den_host[i][j];
        }
        scores[2 * i + 0] = n;
        scores[2 * i + 1] = d;
    }

    static const char *scale_names[4] = {
        "VMAF_feature_vif_scale0_score",
        "VMAF_feature_vif_scale1_score",
        "VMAF_feature_vif_scale2_score",
        "VMAF_feature_vif_scale3_score",
    };

    int err = 0;
    for (int i = 0; i < FVIF_SCALES; i++) {
        const double val = (scores[2*i+1] != 0.0) ?
                           (scores[2*i] / scores[2*i+1]) : 1.0;
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, scale_names[i], val, index);
    }

    if (s->debug && !err) {
        const double score_num = scores[0] + scores[2] + scores[4] + scores[6];
        const double score_den = scores[1] + scores[3] + scores[5] + scores[7];
        const double score     = (score_den == 0.0) ? 1.0 : (score_num / score_den);
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "vif", score, index);
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "vif_num", score_num, index);
        err |= vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "vif_den", score_den, index);

        static const char *nd_names[8] = {
            "vif_num_scale0", "vif_den_scale0",
            "vif_num_scale1", "vif_den_scale1",
            "vif_num_scale2", "vif_den_scale2",
            "vif_num_scale3", "vif_den_scale3",
        };
        for (int i = 0; i < 8; i++) {
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict, nd_names[i], scores[i], index);
        }
    }
    return err;
}

/* ------------------------------------------------------------------ */
/*  close                                                               */
/* ------------------------------------------------------------------ */
static int close_fex_metal(VmafFeatureExtractor *fex)
{
    FloatVifStateMetal *s = (FloatVifStateMetal *)fex->priv;
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_compute)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_compute;  s->pso_compute  = NULL; }
    if (s->pso_decimate) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_decimate; s->pso_decimate = NULL; }

    for (int i = 0; i < FVIF_SCALES; i++) {
        free(s->num_host[i]); s->num_host[i] = NULL;
        free(s->den_host[i]); s->den_host[i] = NULL;
        if (s->num_parts_buf[i]) { (void)(__bridge_transfer id<MTLBuffer>)s->num_parts_buf[i]; s->num_parts_buf[i] = NULL; }
        if (s->den_parts_buf[i]) { (void)(__bridge_transfer id<MTLBuffer>)s->den_parts_buf[i]; s->den_parts_buf[i] = NULL; }
    }
    for (int i = 0; i < 2; i++) {
        if (s->ref_float_buf[i]) { (void)(__bridge_transfer id<MTLBuffer>)s->ref_float_buf[i]; s->ref_float_buf[i] = NULL; }
        if (s->dis_float_buf[i]) { (void)(__bridge_transfer id<MTLBuffer>)s->dis_float_buf[i]; s->dis_float_buf[i] = NULL; }
    }
    if (s->ref_raw_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->ref_raw_buf; s->ref_raw_buf = NULL; }
    if (s->dis_raw_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->dis_raw_buf; s->dis_raw_buf = NULL; }

    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

/* ------------------------------------------------------------------ */
/*  Feature extractor descriptor                                        */
/* ------------------------------------------------------------------ */
static const char *provided_features[] = {
    "VMAF_feature_vif_scale0_score",
    "VMAF_feature_vif_scale1_score",
    "VMAF_feature_vif_scale2_score",
    "VMAF_feature_vif_scale3_score",
    "vif",
    "vif_num",
    "vif_den",
    "vif_num_scale0",
    "vif_den_scale0",
    "vif_num_scale1",
    "vif_den_scale1",
    "vif_num_scale2",
    "vif_den_scale2",
    "vif_num_scale3",
    "vif_den_scale3",
    NULL
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_vif_metal = {
    .name              = "float_vif_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(FloatVifStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
};
} /* extern "C" */
