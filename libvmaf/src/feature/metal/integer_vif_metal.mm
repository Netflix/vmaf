/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_vif feature extractor on the Metal backend (T8-2b / ADR-0436).
 *  Dispatches `vif_vertical_{8,16}bpc` + `vif_horizontal_accum` from
 *  integer_vif.metal for each of the 4 VIF pyramid scales.
 *
 *  Algorithm mirrors libvmaf/src/feature/cuda/integer_vif_cuda.c:
 *    init:   allocate intermediate ushort buffers (mu1, mu2) per scale,
 *            plus 4 × 7 int64 accumulator blocks.
 *    submit: for each scale 0-3:
 *              vertical pass  → mu1_vert, mu2_vert
 *              horizontal pass + in-kernel accumulation
 *            async copy accum_device → accum_host, wait.
 *    collect: read accum_host, compute num/den per scale, write scores.
 *
 *  Score formula (matches write_scores in integer_vif_cuda.c):
 *    num = num_log/2048 + x2 + (den_non_log - num_non_log/16384) / 65025
 *    den = den_log/2048 - (x + num_x*17) + den_non_log
 *    score[s] = num[s] / den[s]
 *
 *  Feature names emitted:
 *    VMAF_integer_feature_vif_scale{0,1,2,3}_score
 *    integer_vif, integer_vif_num, integer_vif_den  (debug=true only)
 *    integer_vif_{num,den}_scale{0,1,2,3}           (debug=true only)
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
#include "integer_vif.h"

#include "../../metal/common.h"
#include "../../metal/kernel_template.h"
}

extern "C" {
extern const unsigned char libvmaf_metallib_start[] __asm("section$start$__TEXT$__metallib");
extern const unsigned char libvmaf_metallib_end[]   __asm("section$end$__TEXT$__metallib");
}

/* Number of VIF pyramid scales. */
#define VIF_SCALES 4

/* 7 int64 accumulators per scale, each split into 2 int32 for Metal atomic ops. */
#define VIF_ACCUM_INT32_PER_SCALE 14
#define VIF_ACCUM_TOTAL_INT32     (VIF_SCALES * VIF_ACCUM_INT32_PER_SCALE)

typedef struct VifAccumHost {
    int64_t x;
    int64_t x2;
    int64_t num_x;
    int64_t num_log;
    int64_t den_log;
    int64_t num_non_log;
    int64_t den_non_log;
} VifAccumHost;

typedef struct VifVertParams {
    uint32_t w;
    uint32_t h;
    uint32_t in_stride;
    uint32_t out_stride;
    uint32_t scale;
    uint32_t bpc;
    uint32_t is_8bpc;
    uint32_t pad0;
    int32_t  shift_VP;
    int32_t  add_shift_round_VP;
} VifVertParams;

typedef struct VifHorizParams {
    uint32_t w;
    uint32_t h;
    uint32_t stride;
    uint32_t scale;
    uint32_t enhn_gain_limit_q16;
    int32_t  pad0;
    int32_t  pad1;
} VifHorizParams;

typedef struct IntegerVifStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext *ctx;

    void *pso_vert_8bpc;    /* __bridge_retained id<MTLComputePipelineState> */
    void *pso_vert_16bpc;
    void *pso_horiz;

    /* Intermediate per-scale mu1/mu2 buffers (ushort). */
    void *mu1_buf;           /* __bridge_retained id<MTLBuffer> */
    void *mu2_buf;

    /* Accumulator buffer on device (VIF_ACCUM_TOTAL_INT32 × int32). */
    void *accum_buf;         /* __bridge_retained id<MTLBuffer> */

    /* Host-side accumulator readback. */
    VifAccumHost accum_host[VIF_SCALES];

    unsigned frame_w;
    unsigned frame_h;
    unsigned bpc;
    bool     debug;
    bool     vif_skip_scale0;
    double   vif_enhn_gain_limit;

    VmafDictionary *feature_name_dict;
} IntegerVifStateMetal;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "enable additional output",
        .offset = offsetof(IntegerVifStateMetal, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif, must be >= 1.0",
        .offset = offsetof(IntegerVifStateMetal, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_skip_scale0",
        .help = "when set, skip scale 0 calculations",
        .alias = "ssclz",
        .offset = offsetof(IntegerVifStateMetal, vif_skip_scale0),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .min = 0.0,
        .max = 0.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}
};

static int build_pipelines(IntegerVifStateMetal *s, id<MTLDevice> device)
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

    id<MTLFunction> fn_vert8  = [lib newFunctionWithName:@"vif_vertical_8bpc"];
    id<MTLFunction> fn_vert16 = [lib newFunctionWithName:@"vif_vertical_16bpc"];
    id<MTLFunction> fn_horiz  = [lib newFunctionWithName:@"vif_horizontal_accum"];
    if (fn_vert8 == nil || fn_vert16 == nil || fn_horiz == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_v8  = [device newComputePipelineStateWithFunction:fn_vert8  error:&err];
    id<MTLComputePipelineState> pso_v16 = [device newComputePipelineStateWithFunction:fn_vert16 error:&err];
    id<MTLComputePipelineState> pso_h   = [device newComputePipelineStateWithFunction:fn_horiz  error:&err];
    if (pso_v8 == nil || pso_v16 == nil || pso_h == nil) { return -ENODEV; }

    s->pso_vert_8bpc  = (__bridge_retained void *)pso_v8;
    s->pso_vert_16bpc = (__bridge_retained void *)pso_v16;
    s->pso_horiz      = (__bridge_retained void *)pso_h;
    return 0;
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    IntegerVifStateMetal *s = (IntegerVifStateMetal *)fex->priv;
    s->frame_w = w;
    s->frame_h = h;
    s->bpc     = bpc;

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_lc; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        /* Intermediate mu1/mu2: largest scale is scale 0 = w × h ushort. */
        const size_t mu_size = (size_t)w * h * sizeof(uint16_t);
        id<MTLBuffer> mu1b = [device newBufferWithLength:mu_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> mu2b = [device newBufferWithLength:mu_size options:MTLResourceStorageModeShared];
        if (mu1b == nil || mu2b == nil) { err = -ENOMEM; goto fail_lc; }
        s->mu1_buf = (__bridge_retained void *)mu1b;
        s->mu2_buf = (__bridge_retained void *)mu2b;

        /* Accumulator buffer: VIF_ACCUM_TOTAL_INT32 × int32. */
        const size_t accum_size = VIF_ACCUM_TOTAL_INT32 * sizeof(int32_t);
        id<MTLBuffer> ab = [device newBufferWithLength:accum_size options:MTLResourceStorageModeShared];
        if (ab == nil) { err = -ENOMEM; goto fail_bufs; }
        s->accum_buf = (__bridge_retained void *)ab;

        err = build_pipelines(s, device);
    }
    if (err != 0) { goto fail_accum; }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    if (s->pso_vert_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert_8bpc;  s->pso_vert_8bpc  = NULL; }
    if (s->pso_vert_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert_16bpc; s->pso_vert_16bpc = NULL; }
    if (s->pso_horiz)      { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz;      s->pso_horiz      = NULL; }
fail_accum:
    if (s->accum_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->accum_buf; s->accum_buf = NULL; }
fail_bufs:
    if (s->mu2_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->mu2_buf; s->mu2_buf = NULL; }
    if (s->mu1_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->mu1_buf; s->mu1_buf = NULL; }
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

/* Dispatch the vertical + horizontal passes for one scale. */
static int dispatch_scale(IntegerVifStateMetal *s,
                          id<MTLCommandBuffer> cmd,
                          id<MTLBuffer> ref_buf,
                          id<MTLBuffer> dis_buf,
                          unsigned w, unsigned h,
                          unsigned scale)
{
    id<MTLBuffer> mu1b    = (__bridge id<MTLBuffer>)s->mu1_buf;
    id<MTLBuffer> mu2b    = (__bridge id<MTLBuffer>)s->mu2_buf;
    id<MTLBuffer> accum_b = (__bridge id<MTLBuffer>)s->accum_buf;

    id<MTLComputePipelineState> pso_vert = (s->bpc <= 8u && scale == 0u)
        ? (__bridge id<MTLComputePipelineState>)s->pso_vert_8bpc
        : (__bridge id<MTLComputePipelineState>)s->pso_vert_16bpc;
    id<MTLComputePipelineState> pso_h = (__bridge id<MTLComputePipelineState>)s->pso_horiz;

    /* Vertical params. */
    VifVertParams vp = {};
    vp.w          = w;
    vp.h          = h;
    vp.in_stride  = w;
    vp.out_stride = w;
    vp.scale      = scale;
    vp.bpc        = s->bpc;
    vp.is_8bpc    = (s->bpc <= 8u && scale == 0u) ? 1u : 0u;
    if (scale == 0u) {
        vp.shift_VP           = (int32_t)s->bpc;
        vp.add_shift_round_VP = (int32_t)(1 << (s->bpc - 1));
    } else {
        vp.shift_VP           = 16;
        vp.add_shift_round_VP = 32768;
    }

    /* Horizontal params. */
    VifHorizParams hp = {};
    hp.w = w;
    hp.h = h;
    hp.stride = w;
    hp.scale  = scale;
    hp.enhn_gain_limit_q16 = (uint32_t)(s->vif_enhn_gain_limit * 65536.0);

    /* Accumulator buffer offset for this scale (in bytes). */
    const size_t accum_offset = (size_t)scale * VIF_ACCUM_INT32_PER_SCALE * sizeof(int32_t);

    /* ---- Vertical pass ---- */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_vert];
        [enc setBuffer:ref_buf offset:0 atIndex:0];
        [enc setBuffer:dis_buf offset:0 atIndex:1];
        [enc setBuffer:mu1b    offset:0 atIndex:2];
        [enc setBuffer:mu2b    offset:0 atIndex:3];
        [enc setBytes:&vp length:sizeof(vp) atIndex:4];

        MTLSize tg   = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((w + 15) / 16, (h + 15) / 16, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* ---- Horizontal + accumulate pass ---- */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_h];
        [enc setBuffer:mu1b    offset:0           atIndex:0];
        [enc setBuffer:mu2b    offset:0           atIndex:1];
        [enc setBuffer:accum_b offset:accum_offset atIndex:2];
        [enc setBytes:&hp length:sizeof(hp) atIndex:3];

        MTLSize tg   = MTLSizeMake(64, 1, 1);
        MTLSize grid = MTLSizeMake((w + 63) / 64, h, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    return 0;
}

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    IntegerVifStateMetal *s = (IntegerVifStateMetal *)fex->priv;

    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    void *dh = vmaf_metal_context_device_handle(s->ctx);
    if (qh == NULL || dh == NULL) { return -ENODEV; }

    id<MTLDevice>       device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue> queue  = (__bridge id<MTLCommandQueue>)qh;
    id<MTLBuffer>       accumb = (__bridge id<MTLBuffer>)s->accum_buf;

    unsigned w = ref_pic->w[0];
    unsigned h = ref_pic->h[0];

    /* Copy luma plane into a Metal shared buffer (row-packed). */
    const size_t bpp   = (s->bpc <= 8u) ? 1u : 2u;
    const size_t plane = (size_t)w * h * bpp;

    id<MTLBuffer> ref_b = [device newBufferWithLength:plane options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_b = [device newBufferWithLength:plane options:MTLResourceStorageModeShared];
    if (ref_b == nil || dis_b == nil) { return -ENOMEM; }

    {
        uint8_t *rdst = (uint8_t *)[ref_b contents];
        uint8_t *ddst = (uint8_t *)[dis_b contents];
        for (unsigned y = 0; y < h; y++) {
            memcpy(rdst + y * w * bpp, (uint8_t *)ref_pic->data[0]  + y * ref_pic->stride[0],  w * bpp);
            memcpy(ddst + y * w * bpp, (uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0], w * bpp);
        }
    }

    /* Zero the accumulator buffer. */
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit fillBuffer:accumb range:NSMakeRange(0, VIF_ACCUM_TOTAL_INT32 * sizeof(int32_t)) value:0];
        [blit endEncoding];
    }

    /* Process all 4 scales. */
    id<MTLBuffer> cur_ref = ref_b;
    id<MTLBuffer> cur_dis = dis_b;

    for (unsigned scale = 0; scale < VIF_SCALES; scale++) {
        if (scale > 0u) {
            /* Downsample by 2 (simple box subsample matching CUDA path). */
            const unsigned sw = w / 2u;
            const unsigned sh = h / 2u;
            const size_t sbpp = 2u; /* always ushort at scales > 0 */
            const size_t splane = (size_t)sw * sh * sbpp;

            id<MTLBuffer> ref_ds = [device newBufferWithLength:splane options:MTLResourceStorageModeShared];
            id<MTLBuffer> dis_ds = [device newBufferWithLength:splane options:MTLResourceStorageModeShared];
            if (ref_ds == nil || dis_ds == nil) { return -ENOMEM; }

            /* CPU-side subsampling for now (matching CUDA strategy of using
             * half-res pointers from the prior-scale intermediate buffer).
             * A future optimization can use a Metal blit/compute downsample. */
            const uint16_t *rsrc = (const uint16_t *)[cur_ref contents];
            const uint16_t *dsrc = (const uint16_t *)[cur_dis contents];
            uint16_t       *rdst2 = (uint16_t *)[ref_ds contents];
            uint16_t       *ddst2 = (uint16_t *)[dis_ds contents];
            for (unsigned y = 0; y < sh; y++) {
                for (unsigned x = 0; x < sw; x++) {
                    rdst2[y * sw + x] = rsrc[(y * 2u) * w + (x * 2u)];
                    ddst2[y * sw + x] = dsrc[(y * 2u) * w + (x * 2u)];
                }
            }
            cur_ref = ref_ds;
            cur_dis = dis_ds;
            w = sw;
            h = sh;
        }

        int err = dispatch_scale(s, cmd, cur_ref, cur_dis, w, h, scale);
        if (err != 0) { return err; }
    }

    [cmd commit];
    [cmd waitUntilCompleted];

    /* Read back accumulators. */
    const int32_t *raw = (const int32_t *)[accumb contents];
    for (unsigned sc = 0; sc < VIF_SCALES; sc++) {
        const int32_t *slot = raw + sc * VIF_ACCUM_INT32_PER_SCALE;
        /* Reconstruct int64 from split [lo, hi] int32 pairs. */
        s->accum_host[sc].x           = (int64_t)(uint32_t)slot[0]  | ((int64_t)slot[1]  << 32);
        s->accum_host[sc].x2          = (int64_t)(uint32_t)slot[2]  | ((int64_t)slot[3]  << 32);
        s->accum_host[sc].num_x       = (int64_t)(uint32_t)slot[4]  | ((int64_t)slot[5]  << 32);
        s->accum_host[sc].num_log     = (int64_t)(uint32_t)slot[6]  | ((int64_t)slot[7]  << 32);
        s->accum_host[sc].den_log     = (int64_t)(uint32_t)slot[8]  | ((int64_t)slot[9]  << 32);
        s->accum_host[sc].num_non_log = (int64_t)(uint32_t)slot[10] | ((int64_t)slot[11] << 32);
        s->accum_host[sc].den_non_log = (int64_t)(uint32_t)slot[12] | ((int64_t)slot[13] << 32);
    }

    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    IntegerVifStateMetal *s = (IntegerVifStateMetal *)fex->priv;

    /* Score formula mirrors write_scores() in integer_vif_cuda.c. */
    struct { float num; float den; } scale_score[VIF_SCALES];
    for (unsigned sc = 0; sc < VIF_SCALES; sc++) {
        const VifAccumHost *a = &s->accum_host[sc];
        scale_score[sc].num = (float)(
            (double)a->num_log / 2048.0 + (double)a->x2 +
            ((double)a->den_non_log - ((double)a->num_non_log / 16384.0)) / 65025.0);
        scale_score[sc].den = (float)(
            (double)a->den_log / 2048.0 -
            ((double)a->x + (double)a->num_x * 17.0) + (double)a->den_non_log);
    }

    /* When vif_skip_scale0 is set, emit 0.0 for scale-0 score and exclude
     * scale-0 num/den from the combined totals — matching integer_vif.c
     * write_scores() parity and the CUDA/SYCL/Vulkan twins. */
    const double scale0_score =
        s->vif_skip_scale0 ? 0.0
        : (scale_score[0].den != 0.0f) ? (double)scale_score[0].num / (double)scale_score[0].den
        : 0.0;

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "VMAF_integer_feature_vif_scale0_score", scale0_score, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "VMAF_integer_feature_vif_scale1_score",
        (scale_score[1].den != 0.0f) ? scale_score[1].num / scale_score[1].den : 0.0, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "VMAF_integer_feature_vif_scale2_score",
        (scale_score[2].den != 0.0f) ? scale_score[2].num / scale_score[2].den : 0.0, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "VMAF_integer_feature_vif_scale3_score",
        (scale_score[3].den != 0.0f) ? scale_score[3].num / scale_score[3].den : 0.0, index);

    if (!s->debug) { return err; }

    const double score_num = (s->vif_skip_scale0 ? 0.0 : (double)scale_score[0].num) +
                             (double)scale_score[1].num + (double)scale_score[2].num +
                             (double)scale_score[3].num;
    const double score_den = (s->vif_skip_scale0 ? 0.0 : (double)scale_score[0].den) +
                             (double)scale_score[1].den + (double)scale_score[2].den +
                             (double)scale_score[3].den;
    const double score = (score_den == 0.0) ? 1.0 : score_num / score_den;

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "integer_vif", score, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "integer_vif_num", score_num, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "integer_vif_den", score_den, index);

    /* Debug per-scale num/den: emit 0.0 / -1.0 sentinels for scale-0 when
     * vif_skip_scale0 is active, matching integer_vif.c and GPU twins. */
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "integer_vif_num_scale0",
        s->vif_skip_scale0 ? 0.0 : (double)scale_score[0].num, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
        "integer_vif_den_scale0",
        s->vif_skip_scale0 ? -1.0 : (double)scale_score[0].den, index);

    for (unsigned sc = 1; sc < VIF_SCALES; sc++) {
        char name_num[64], name_den[64];
        (void)snprintf(name_num, sizeof(name_num), "integer_vif_num_scale%u", sc);
        (void)snprintf(name_den, sizeof(name_den), "integer_vif_den_scale%u", sc);
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
            name_num, (double)scale_score[sc].num, index);
        err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
            name_den, (double)scale_score[sc].den, index);
    }

    return err;
}

static int flush_fex_metal(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)fex; (void)feature_collector;
    return 1; /* VIF is not a temporal metric; no tail frame needed. */
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    IntegerVifStateMetal *s = (IntegerVifStateMetal *)fex->priv;
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_vert_8bpc)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert_8bpc;  s->pso_vert_8bpc  = NULL; }
    if (s->pso_vert_16bpc) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert_16bpc; s->pso_vert_16bpc = NULL; }
    if (s->pso_horiz)      { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz;      s->pso_horiz      = NULL; }
    if (s->accum_buf)      { (void)(__bridge_transfer id<MTLBuffer>)s->accum_buf;                    s->accum_buf      = NULL; }
    if (s->mu2_buf)        { (void)(__bridge_transfer id<MTLBuffer>)s->mu2_buf;                      s->mu2_buf        = NULL; }
    if (s->mu1_buf)        { (void)(__bridge_transfer id<MTLBuffer>)s->mu1_buf;                      s->mu1_buf        = NULL; }

    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_vif_scale0_score",
    "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score",
    "VMAF_integer_feature_vif_scale3_score",
    "integer_vif",
    "integer_vif_num",
    "integer_vif_den",
    "integer_vif_num_scale0",
    "integer_vif_den_scale0",
    "integer_vif_num_scale1",
    "integer_vif_den_scale1",
    "integer_vif_num_scale2",
    "integer_vif_den_scale2",
    "integer_vif_num_scale3",
    "integer_vif_den_scale3",
    NULL
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_vif_metal = {
    .name              = "integer_vif_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = flush_fex_metal,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(IntegerVifStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
};
} /* extern "C" */
