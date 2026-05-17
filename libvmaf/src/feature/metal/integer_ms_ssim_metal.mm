/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_ms_ssim feature extractor on the Metal backend.
 *
 *  Metal port of libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c.
 *  Mirrors the float_ssim_metal.mm pattern (PR #997 / ADR-0421) but
 *  implements the full 5-level MS-SSIM pyramid with three kernel passes
 *  (ms_ssim_decimate, ms_ssim_horiz, ms_ssim_vert_lcs).
 *
 *  Algorithm summary:
 *   1. Host normalises raw uint pixels → float [0, 255] and uploads as
 *      pyramid level 0 (shared MTLBuffer).
 *   2. ms_ssim_decimate builds levels 1-4 (9-tap 9/7 LPF + 2× subsample).
 *   3. ms_ssim_horiz + ms_ssim_vert_lcs compute per-scale L/C/S partial
 *      sums for each of the 5 pyramid levels.
 *   4. Host accumulates partials (double), applies Wang 2003 weights:
 *        MS-SSIM = Π_i  l_i^α_i · c_i^β_i · |s_i|^γ_i
 *
 *  Wang weights:
 *    alphas = {0, 0, 0, 0, 0.1333}
 *    betas  = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333}
 *    gammas = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333}
 *
 *  Min-dim guard: 11 << 4 = 176 (ADR-0153).
 *
 *  enable_lcs: when true, emits the 15 extra per-scale metrics
 *  float_ms_ssim_{l,c,s}_scale{0..4} (matches CUDA twin).
 *
 *  Feature name: float_ms_ssim (same output key as CPU / CUDA / SYCL twins).
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

#define MS_SSIM_SCALES        5
#define MS_SSIM_GAUSSIAN_LEN  11
#define MS_SSIM_K             11
#define MS_SSIM_BLOCK_X       16
#define MS_SSIM_BLOCK_Y        8

static const float g_alphas[MS_SSIM_SCALES] = {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f};
static const float g_betas[MS_SSIM_SCALES]  = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};
static const float g_gammas[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};

typedef struct IntegerMsSsimStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext        *ctx;

    void *pso_decimate;   /* ms_ssim_decimate  */
    void *pso_horiz;      /* ms_ssim_horiz     */
    void *pso_vert_lcs;   /* ms_ssim_vert_lcs  */

    unsigned width;
    unsigned height;
    unsigned bpc;
    float    scaler;   /* raw → [0, 255] multiplier (8bpc=1, 10bpc=4, ...) */

    /* Per-scale geometry (mirrors MsSsimStateCuda). */
    unsigned scale_w[MS_SSIM_SCALES];
    unsigned scale_h[MS_SSIM_SCALES];
    unsigned scale_w_horiz[MS_SSIM_SCALES];
    unsigned scale_h_horiz[MS_SSIM_SCALES];
    unsigned scale_w_final[MS_SSIM_SCALES];
    unsigned scale_h_final[MS_SSIM_SCALES];
    unsigned scale_grid_x[MS_SSIM_SCALES];
    unsigned scale_grid_y[MS_SSIM_SCALES];
    unsigned scale_block_count[MS_SSIM_SCALES];

    float c1;
    float c2;
    float c3;

    /* Pyramid float buffers (shared-storage MTLBuffer, scale 0 = full). */
    void *pyramid_ref[MS_SSIM_SCALES];  /* id<MTLBuffer>, bridged */
    void *pyramid_cmp[MS_SSIM_SCALES];

    /* Intermediate 5-stat horiz buffer (sized for scale 0). */
    void *hbuf;           /* id<MTLBuffer> */
    size_t hbuf_size;

    /* Per-scale L/C/S partial-sum MTLBuffers. */
    void *l_partials[MS_SSIM_SCALES];   /* id<MTLBuffer> */
    void *c_partials[MS_SSIM_SCALES];
    void *s_partials[MS_SSIM_SCALES];

    bool enable_lcs;

    VmafDictionary *feature_name_dict;
} IntegerMsSsimStateMetal;

static const VmafOption options[] = {
    {
        .name        = "enable_lcs",
        .help        = "enable luminance, contrast and structure intermediate output",
        .offset      = offsetof(IntegerMsSsimStateMetal, enable_lcs),
        .type        = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = false},
    },
    {0},
};

/* ------------------------------------------------------------------ */
/*  Pipeline-state object build                                         */
/* ------------------------------------------------------------------ */

static int build_pipelines(IntegerMsSsimStateMetal *s, id<MTLDevice> device)
{
    const size_t blob_size =
        (size_t)(libvmaf_metallib_end - libvmaf_metallib_start);
    if (blob_size == 0u) { return -ENODEV; }

    dispatch_data_t data = dispatch_data_create(
        libvmaf_metallib_start, blob_size,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (data == NULL) { return -ENOMEM; }

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&err];
    if (lib == nil) { return -ENODEV; }

    id<MTLFunction> fn_dec   = [lib newFunctionWithName:@"ms_ssim_decimate"];
    id<MTLFunction> fn_horiz = [lib newFunctionWithName:@"ms_ssim_horiz"];
    id<MTLFunction> fn_vert  = [lib newFunctionWithName:@"ms_ssim_vert_lcs"];
    if (fn_dec == nil || fn_horiz == nil || fn_vert == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_d =
        [device newComputePipelineStateWithFunction:fn_dec   error:&err];
    id<MTLComputePipelineState> pso_h =
        [device newComputePipelineStateWithFunction:fn_horiz error:&err];
    id<MTLComputePipelineState> pso_v =
        [device newComputePipelineStateWithFunction:fn_vert  error:&err];
    if (pso_d == nil || pso_h == nil || pso_v == nil) { return -ENODEV; }

    s->pso_decimate = (__bridge_retained void *)pso_d;
    s->pso_horiz    = (__bridge_retained void *)pso_h;
    s->pso_vert_lcs = (__bridge_retained void *)pso_v;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  init                                                                */
/* ------------------------------------------------------------------ */

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                           unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    IntegerMsSsimStateMetal *s = (IntegerMsSsimStateMetal *)fex->priv;

    /* ADR-0153 minimum resolution check (matches CUDA twin). */
    const unsigned min_dim =
        (unsigned)MS_SSIM_GAUSSIAN_LEN << (MS_SSIM_SCALES - 1);
    if (w < min_dim || h < min_dim) {
        return -EINVAL;
    }

    s->width  = w;
    s->height = h;
    s->bpc    = bpc;

    if (bpc <= 8u)       { s->scaler = 1.0f; }
    else if (bpc == 10u) { s->scaler = 4.0f; }
    else if (bpc == 12u) { s->scaler = 16.0f; }
    else                 { s->scaler = 256.0f; }

    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < MS_SSIM_SCALES; ++i) {
        s->scale_w[i] = (s->scale_w[i - 1] / 2u) + (s->scale_w[i - 1] & 1u);
        s->scale_h[i] = (s->scale_h[i - 1] / 2u) + (s->scale_h[i - 1] & 1u);
    }
    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        s->scale_w_horiz[i] = s->scale_w[i] - (MS_SSIM_K - 1u);
        s->scale_h_horiz[i] = s->scale_h[i];
        s->scale_w_final[i] = s->scale_w[i] - (MS_SSIM_K - 1u);
        s->scale_h_final[i] = s->scale_h[i] - (MS_SSIM_K - 1u);
        s->scale_grid_x[i]  =
            (s->scale_w_final[i] + (unsigned)MS_SSIM_BLOCK_X - 1u) /
            (unsigned)MS_SSIM_BLOCK_X;
        s->scale_grid_y[i]  =
            (s->scale_h_final[i] + (unsigned)MS_SSIM_BLOCK_Y - 1u) /
            (unsigned)MS_SSIM_BLOCK_Y;
        s->scale_block_count[i] = s->scale_grid_x[i] * s->scale_grid_y[i];
    }

    const float L = 255.0f, K1 = 0.01f, K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);
    s->c3 = s->c2 * 0.5f;

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_lc; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        err = build_pipelines(s, device);
        if (err != 0) { goto fail_lc; }

        /* Pyramid float buffers. */
        for (int i = 0; i < MS_SSIM_SCALES; ++i) {
            const size_t plane_bytes =
                (size_t)s->scale_w[i] * s->scale_h[i] * sizeof(float);
            id<MTLBuffer> br =
                [device newBufferWithLength:plane_bytes
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bc =
                [device newBufferWithLength:plane_bytes
                                    options:MTLResourceStorageModeShared];
            if (br == nil || bc == nil) { err = -ENOMEM; goto fail_pso; }
            s->pyramid_ref[i] = (__bridge_retained void *)br;
            s->pyramid_cmp[i] = (__bridge_retained void *)bc;
        }

        /* Horizontal 5-stat buffer — sized for scale 0 (largest). */
        s->hbuf_size =
            5u * (size_t)s->scale_w_horiz[0] * s->scale_h_horiz[0] *
            sizeof(float);
        {
            id<MTLBuffer> hb =
                [device newBufferWithLength:s->hbuf_size
                                    options:MTLResourceStorageModeShared];
            if (hb == nil) { err = -ENOMEM; goto fail_pyr; }
            s->hbuf = (__bridge_retained void *)hb;
        }

        /* Per-scale L/C/S partial buffers. */
        for (int i = 0; i < MS_SSIM_SCALES; ++i) {
            const size_t par_bytes =
                (size_t)s->scale_block_count[i] * sizeof(float);
            id<MTLBuffer> bl =
                [device newBufferWithLength:par_bytes
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bc2 =
                [device newBufferWithLength:par_bytes
                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bs =
                [device newBufferWithLength:par_bytes
                                    options:MTLResourceStorageModeShared];
            if (bl == nil || bc2 == nil || bs == nil) {
                err = -ENOMEM; goto fail_hbuf;
            }
            s->l_partials[i] = (__bridge_retained void *)bl;
            s->c_partials[i] = (__bridge_retained void *)bc2;
            s->s_partials[i] = (__bridge_retained void *)bs;
        }
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(
            fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_partials; }
    return 0;

fail_partials:
    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        if (s->l_partials[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->l_partials[i];
            s->l_partials[i] = NULL;
        }
        if (s->c_partials[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->c_partials[i];
            s->c_partials[i] = NULL;
        }
        if (s->s_partials[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->s_partials[i];
            s->s_partials[i] = NULL;
        }
    }
fail_hbuf:
    if (s->hbuf) {
        (void)(__bridge_transfer id<MTLBuffer>)s->hbuf;
        s->hbuf = NULL;
    }
fail_pyr:
    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        if (s->pyramid_ref[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->pyramid_ref[i];
            s->pyramid_ref[i] = NULL;
        }
        if (s->pyramid_cmp[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->pyramid_cmp[i];
            s->pyramid_cmp[i] = NULL;
        }
    }
fail_pso:
    if (s->pso_vert_lcs) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert_lcs;
        s->pso_vert_lcs = NULL;
    }
    if (s->pso_horiz) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz;
        s->pso_horiz = NULL;
    }
    if (s->pso_decimate) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_decimate;
        s->pso_decimate = NULL;
    }
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

/* ------------------------------------------------------------------ */
/*  submit                                                              */
/* ------------------------------------------------------------------ */

static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                             VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                             VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90; (void)dist_pic_90; (void)index;
    IntegerMsSsimStateMetal *s = (IntegerMsSsimStateMetal *)fex->priv;

    /* Normalise raw uint pixels → float [0, 255] into pyramid level 0. */
    {
        float *rf = (float *)[(__bridge id<MTLBuffer>)s->pyramid_ref[0] contents];
        float *df = (float *)[(__bridge id<MTLBuffer>)s->pyramid_cmp[0] contents];
        const float inv = 1.0f / s->scaler;

        if (s->bpc <= 8u) {
            for (unsigned y = 0; y < s->height; ++y) {
                const uint8_t *rs =
                    (const uint8_t *)ref_pic->data[0] + y * ref_pic->stride[0];
                const uint8_t *ds =
                    (const uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0];
                for (unsigned x = 0; x < s->width; ++x) {
                    rf[y * s->width + x] = (float)rs[x];
                    df[y * s->width + x] = (float)ds[x];
                }
            }
        } else {
            for (unsigned y = 0; y < s->height; ++y) {
                const uint16_t *rs =
                    (const uint16_t *)((const uint8_t *)ref_pic->data[0] +
                                       y * ref_pic->stride[0]);
                const uint16_t *ds =
                    (const uint16_t *)((const uint8_t *)dist_pic->data[0] +
                                       y * dist_pic->stride[0]);
                for (unsigned x = 0; x < s->width; ++x) {
                    rf[y * s->width + x] = (float)rs[x] * inv;
                    df[y * s->width + x] = (float)ds[x] * inv;
                }
            }
        }
    }

    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (qh == NULL) { return -ENODEV; }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)qh;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    id<MTLComputePipelineState> pso_dec =
        (__bridge id<MTLComputePipelineState>)s->pso_decimate;
    id<MTLComputePipelineState> pso_hor =
        (__bridge id<MTLComputePipelineState>)s->pso_horiz;
    id<MTLComputePipelineState> pso_vrt =
        (__bridge id<MTLComputePipelineState>)s->pso_vert_lcs;

    /* Build pyramid levels 1-4 via decimate. */
    for (int i = 0; i < MS_SSIM_SCALES - 1; ++i) {
        const unsigned w_in  = s->scale_w[i];
        const unsigned h_in  = s->scale_h[i];
        const unsigned w_out = s->scale_w[i + 1];
        const unsigned h_out = s->scale_h[i + 1];
        const unsigned gx = (w_out + (unsigned)MS_SSIM_BLOCK_X - 1u) /
                            (unsigned)MS_SSIM_BLOCK_X;
        const unsigned gy = (h_out + (unsigned)MS_SSIM_BLOCK_Y - 1u) /
                            (unsigned)MS_SSIM_BLOCK_Y;
        const uint32_t dims[4] = {w_in, h_in, w_out, h_out};

        for (int side = 0; side < 2; ++side) {
            id<MTLBuffer> src_buf =
                (__bridge id<MTLBuffer>)(side == 0 ?
                    s->pyramid_ref[i] : s->pyramid_cmp[i]);
            id<MTLBuffer> dst_buf =
                (__bridge id<MTLBuffer>)(side == 0 ?
                    s->pyramid_ref[i + 1] : s->pyramid_cmp[i + 1]);

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_dec];
            [enc setBuffer:src_buf offset:0 atIndex:0];
            [enc setBuffer:dst_buf offset:0 atIndex:1];
            [enc setBytes:dims length:sizeof(dims) atIndex:2];
            MTLSize tg   = MTLSizeMake(MS_SSIM_BLOCK_X, MS_SSIM_BLOCK_Y, 1);
            MTLSize grid = MTLSizeMake(gx, gy, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }

    /* Per-scale SSIM compute. */
    id<MTLBuffer> hbuf_buf = (__bridge id<MTLBuffer>)s->hbuf;

    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        const unsigned w     = s->scale_w[i];
        const unsigned h     = s->scale_h[i];
        const unsigned w_h   = s->scale_w_horiz[i];
        const unsigned h_h   = s->scale_h_horiz[i];
        const unsigned w_f   = s->scale_w_final[i];
        const unsigned h_f   = s->scale_h_final[i];
        const unsigned gx_h  = (w_h + (unsigned)MS_SSIM_BLOCK_X - 1u) /
                               (unsigned)MS_SSIM_BLOCK_X;
        const unsigned gy_h  = (h_h + (unsigned)MS_SSIM_BLOCK_Y - 1u) /
                               (unsigned)MS_SSIM_BLOCK_Y;

        id<MTLBuffer> ref_buf =
            (__bridge id<MTLBuffer>)s->pyramid_ref[i];
        id<MTLBuffer> cmp_buf =
            (__bridge id<MTLBuffer>)s->pyramid_cmp[i];
        id<MTLBuffer> lp_buf =
            (__bridge id<MTLBuffer>)s->l_partials[i];
        id<MTLBuffer> cp_buf =
            (__bridge id<MTLBuffer>)s->c_partials[i];
        id<MTLBuffer> sp_buf =
            (__bridge id<MTLBuffer>)s->s_partials[i];

        /* Zero partials. */
        {
            const size_t par_bytes =
                (size_t)s->scale_block_count[i] * sizeof(float);
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit fillBuffer:lp_buf range:NSMakeRange(0, par_bytes) value:0];
            [blit fillBuffer:cp_buf range:NSMakeRange(0, par_bytes) value:0];
            [blit fillBuffer:sp_buf range:NSMakeRange(0, par_bytes) value:0];
            [blit endEncoding];
        }

        /* Horiz pass. */
        {
            const uint32_t params[4] = {w, h, w_h, 0u};
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_hor];
            [enc setBuffer:ref_buf  offset:0 atIndex:0];
            [enc setBuffer:cmp_buf  offset:0 atIndex:1];
            [enc setBuffer:hbuf_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
            MTLSize tg   = MTLSizeMake(MS_SSIM_BLOCK_X, MS_SSIM_BLOCK_Y, 1);
            MTLSize grid = MTLSizeMake(gx_h, gy_h, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        /* Vert-LCS pass. */
        {
            const unsigned gx_f = s->scale_grid_x[i];
            const unsigned gy_f = s->scale_grid_y[i];
            const uint32_t params[4] = {w_h, h, w_f, h_f};
            const float    consts[4] = {s->c1, s->c2, s->c3, 0.0f};
            const uint32_t gdim[2]   = {gx_f, gy_f};
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_vrt];
            [enc setBuffer:hbuf_buf offset:0 atIndex:0];
            [enc setBuffer:lp_buf   offset:0 atIndex:1];
            [enc setBuffer:cp_buf   offset:0 atIndex:2];
            [enc setBuffer:sp_buf   offset:0 atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
            [enc setBytes:consts length:sizeof(consts) atIndex:5];
            [enc setBytes:gdim   length:sizeof(gdim)   atIndex:6];
            MTLSize tg   = MTLSizeMake(MS_SSIM_BLOCK_X, MS_SSIM_BLOCK_Y, 1);
            MTLSize grid = MTLSizeMake(gx_f, gy_f, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

/* ------------------------------------------------------------------ */
/*  collect                                                             */
/* ------------------------------------------------------------------ */

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                              VmafFeatureCollector *feature_collector)
{
    IntegerMsSsimStateMetal *s = (IntegerMsSsimStateMetal *)fex->priv;

    double l_means[MS_SSIM_SCALES] = {0};
    double c_means[MS_SSIM_SCALES] = {0};
    double s_means[MS_SSIM_SCALES] = {0};

    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        const float *lp =
            (const float *)[(__bridge id<MTLBuffer>)s->l_partials[i] contents];
        const float *cp =
            (const float *)[(__bridge id<MTLBuffer>)s->c_partials[i] contents];
        const float *sp =
            (const float *)[(__bridge id<MTLBuffer>)s->s_partials[i] contents];

        double tl = 0.0, tc = 0.0, ts = 0.0;
        for (unsigned j = 0; j < s->scale_block_count[i]; ++j) {
            tl += (double)lp[j];
            tc += (double)cp[j];
            ts += (double)sp[j];
        }
        const double n = (double)s->scale_w_final[i] *
                         (double)s->scale_h_final[i];
        if (n > 0.0) {
            l_means[i] = tl / n;
            c_means[i] = tc / n;
            s_means[i] = ts / n;
        } else {
            l_means[i] = 1.0;
            c_means[i] = 1.0;
            s_means[i] = 1.0;
        }
    }

    double msssim = 1.0;
    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        msssim *= pow(l_means[i],        (double)g_alphas[i]) *
                  pow(c_means[i],        (double)g_betas[i])  *
                  pow(fabs(s_means[i]),  (double)g_gammas[i]);
    }

    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_ms_ssim", msssim, index);

    if (s->enable_lcs) {
        static const char *const l_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_l_scale0", "float_ms_ssim_l_scale1",
            "float_ms_ssim_l_scale2", "float_ms_ssim_l_scale3",
            "float_ms_ssim_l_scale4",
        };
        static const char *const c_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_c_scale0", "float_ms_ssim_c_scale1",
            "float_ms_ssim_c_scale2", "float_ms_ssim_c_scale3",
            "float_ms_ssim_c_scale4",
        };
        static const char *const s_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_s_scale0", "float_ms_ssim_s_scale1",
            "float_ms_ssim_s_scale2", "float_ms_ssim_s_scale3",
            "float_ms_ssim_s_scale4",
        };
        for (int i = 0; i < MS_SSIM_SCALES; ++i) {
            err |= vmaf_feature_collector_append(
                feature_collector, l_names[i], l_means[i], index);
            err |= vmaf_feature_collector_append(
                feature_collector, c_names[i], c_means[i], index);
            err |= vmaf_feature_collector_append(
                feature_collector, s_names[i], s_means[i], index);
        }
    }
    return err;
}

/* ------------------------------------------------------------------ */
/*  close                                                               */
/* ------------------------------------------------------------------ */

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    IntegerMsSsimStateMetal *s = (IntegerMsSsimStateMetal *)fex->priv;

    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        if (s->l_partials[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->l_partials[i];
            s->l_partials[i] = NULL;
        }
        if (s->c_partials[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->c_partials[i];
            s->c_partials[i] = NULL;
        }
        if (s->s_partials[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->s_partials[i];
            s->s_partials[i] = NULL;
        }
    }
    if (s->hbuf) {
        (void)(__bridge_transfer id<MTLBuffer>)s->hbuf;
        s->hbuf = NULL;
    }
    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        if (s->pyramid_ref[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->pyramid_ref[i];
            s->pyramid_ref[i] = NULL;
        }
        if (s->pyramid_cmp[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->pyramid_cmp[i];
            s->pyramid_cmp[i] = NULL;
        }
    }
    if (s->pso_vert_lcs) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert_lcs;
        s->pso_vert_lcs = NULL;
    }
    if (s->pso_horiz) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz;
        s->pso_horiz = NULL;
    }
    if (s->pso_decimate) {
        (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_decimate;
        s->pso_decimate = NULL;
    }

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
    if (s->feature_name_dict) {
        (void)vmaf_dictionary_free(&s->feature_name_dict);
    }
    if (s->ctx) {
        vmaf_metal_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

/* ------------------------------------------------------------------ */
/*  Registration                                                        */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"float_ms_ssim", NULL};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_ms_ssim_metal = {
    .name              = "integer_ms_ssim_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(IntegerMsSsimStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 18,
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
