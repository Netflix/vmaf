/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ssim feature extractor on the Metal backend (T8-1j / ADR-0421).
 *  Two-pass dispatch: float_ssim_horiz → float_ssim_vert_combine.
 *
 *  Input pixels are normalised to [0, 255.xxx] before the kernel:
 *    scaler: 8bpc → 1.0, 10bpc → 4.0, 12bpc → 16.0, 16bpc → 256.0
 *
 *  Valid convolution with 11-tap Gaussian (K=11), no mirror padding:
 *    w_h = W - 10   (horiz valid output width)
 *    h_v = H - 10   (vert valid output height)
 *
 *  hbuf layout: 5 planes, each w_h × H floats:
 *    plane 0: mu_r,  plane 1: mu_d,  plane 2: sq_r,
 *    plane 3: sq_d,  plane 4: rd
 *
 *  SSIM partials: float ssim_sum per WG, indexed by bid.y*grid_w + bid.x.
 *  LCS  partials: 3 × (grid_w × grid_h) floats [L | C | S]; present only
 *                 when enable_lcs=true (lcs_flags bit-0 == 1).
 *  Host: score = sum(partials) / (w_h * h_v).
 *
 *  c1 = (0.01 * 255)^2 = 6.5025,  c2 = (0.03 * 255)^2 = 58.5225.
 *
 *  Options (matching float_ssim.c CPU parity):
 *    enable_lcs  — emit float_ssim_l, float_ssim_c, float_ssim_s sub-scores.
 *    enable_db   — convert SSIM to dB: score = -10*log10(1 - ssim).
 *    clip_db     — clamp dB output to a finite maximum derived from frame size.
 *    scale       — v1 supports scale=1 only; auto-detect rejects scale>1.
 *
 *  Feature name: float_ssim.
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

typedef struct FloatSsimStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalKernelBuffer rb;        /* float ssim_sum partials, grid_w × grid_h */
    VmafMetalContext *ctx;
    void *pso_horiz;                 /* float_ssim_horiz (pass 0) */
    void *pso_vert;                  /* float_ssim_vert_combine (pass 1) */
    void *hbuf_buf;                  /* intermediate 5-plane buffer */
    void *lcs_buf;                   /* 3 × partials_count floats; NULL unless enable_lcs */

    /* Options (matching float_ssim.c CPU). */
    bool    enable_lcs;
    bool    enable_db;
    bool    clip_db;
    int     scale;          /* 0 = auto-detect; v1: only 1 is supported. */

    float   c1;
    float   c2;
    float   scaler;         /* raw → [0, 255.xxx] multiplier (1/scaler of raw) */
    double  max_db;         /* INFINITY unless clip_db */
    size_t  hbuf_size;
    size_t  partials_count;
    unsigned frame_w;
    unsigned frame_h;
    unsigned w_h;           /* W - 10 */
    unsigned h_v;           /* H - 10 */
    unsigned bpc;

    VmafDictionary *feature_name_dict;
} FloatSsimStateMetal;

/* ------------------------------------------------------------------ */
/* Options                                                              */
/* ------------------------------------------------------------------ */

static const VmafOption options[] = {
    {
        .name        = "enable_lcs",
        .help        = "emit luminance, contrast and structure sub-scores",
        .offset      = offsetof(FloatSsimStateMetal, enable_lcs),
        .type        = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = false},
    },
    {
        .name        = "enable_db",
        .help        = "output SSIM score in dB: -10*log10(1 - ssim)",
        .offset      = offsetof(FloatSsimStateMetal, enable_db),
        .type        = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = false},
    },
    {
        .name        = "clip_db",
        .help        = "clamp dB score to a finite maximum",
        .offset      = offsetof(FloatSsimStateMetal, clip_db),
        .type        = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = false},
    },
    {
        .name        = "scale",
        .help        = "decimation scale factor (0=auto, 1=no downscaling). "
                       "v1: Metal path supports scale=1 only.",
        .offset      = offsetof(FloatSsimStateMetal, scale),
        .type        = VMAF_OPT_TYPE_INT,
        .default_val = {.i = 0},
        .min         = 0,
        .max         = 10,
    },
    {0}
};

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static int ssim_metal_compute_scale(unsigned w, unsigned h, int override_val)
{
    if (override_val > 0) { return override_val; }
    int scaled = (int)((float)(w < h ? w : h) / 256.0f + 0.5f);
    return (scaled < 1) ? 1 : scaled;
}

static double ssim_to_db(double ssim, double max_db)
{
    const double db = -10.0 * log10(1.0 - ssim);
    return (db < max_db) ? db : max_db;
}

static int build_pipelines(FloatSsimStateMetal *s, id<MTLDevice> device)
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

    id<MTLFunction> fn_horiz = [lib newFunctionWithName:@"float_ssim_horiz"];
    id<MTLFunction> fn_vert  = [lib newFunctionWithName:@"float_ssim_vert_combine"];
    if (fn_horiz == nil || fn_vert == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_h = [device newComputePipelineStateWithFunction:fn_horiz error:&err];
    id<MTLComputePipelineState> pso_v = [device newComputePipelineStateWithFunction:fn_vert  error:&err];
    if (pso_h == nil || pso_v == nil) { return -ENODEV; }

    s->pso_horiz = (__bridge_retained void *)pso_h;
    s->pso_vert  = (__bridge_retained void *)pso_v;
    return 0;
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatSsimStateMetal *s = (FloatSsimStateMetal *)fex->priv;

    /* Validate scale: v1 supports scale=1 only. */
    const int scale = ssim_metal_compute_scale(w, h, s->scale);
    if (scale != 1) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "float_ssim_metal: v1 supports scale=1 only "
                 "(auto-detected scale=%d at %ux%u). "
                 "Pin --feature float_ssim_metal:scale=1 if intended.\n",
                 scale, w, h);
        return -EINVAL;
    }
    if (w < 11u || h < 11u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "float_ssim_metal: frame %ux%u smaller than 11×11 Gaussian footprint.\n",
                 w, h);
        return -EINVAL;
    }

    s->frame_w = w;
    s->frame_h = h;
    s->bpc     = bpc;
    s->w_h     = w - 10u;
    s->h_v     = h - 10u;
    s->c1      = 6.5025f;   /* (0.01 * 255)^2 */
    s->c2      = 58.5225f;  /* (0.03 * 255)^2 */

    if (bpc <= 8u)       { s->scaler = 1.0f; }
    else if (bpc == 10u) { s->scaler = 4.0f; }
    else if (bpc == 12u) { s->scaler = 16.0f; }
    else                 { s->scaler = 256.0f; }

    /* Derive max_db used by clip_db. Mirrors float_ssim.c::init. */
    if (s->clip_db) {
        const double peak = (double)((1u << bpc) - 1u);
        const double mse  = 0.5 / ((double)w * (double)h);
        s->max_db = ceil(10.0 * log10(peak * peak / mse));
    } else {
        s->max_db = INFINITY;
    }

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        /* SSIM partials grid is over the vert-combined output: w_h × h_v */
        const size_t grid_w = (s->w_h + 15u) / 16u;
        const size_t grid_h = (s->h_v +  7u) /  8u;
        s->partials_count   = grid_w * grid_h;
        err = vmaf_metal_kernel_buffer_alloc(&s->rb, s->ctx,
                                             s->partials_count * sizeof(float));
    }
    if (err != 0) { goto fail_lc; }

    {
        void *dh = vmaf_metal_context_device_handle(s->ctx);
        if (dh == NULL) { err = -ENODEV; goto fail_rb; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dh;

        /* hbuf: 5 planes × w_h × H floats */
        s->hbuf_size = 5u * (size_t)s->w_h * (size_t)h * sizeof(float);
        id<MTLBuffer> hb = [device newBufferWithLength:s->hbuf_size
                                               options:MTLResourceStorageModeShared];
        if (hb == nil) { err = -ENOMEM; goto fail_rb; }
        s->hbuf_buf = (__bridge_retained void *)hb;

        /* lcs_buf: 3 planes × partials_count floats; only when enable_lcs. */
        if (s->enable_lcs) {
            const size_t lcs_size = 3u * s->partials_count * sizeof(float);
            id<MTLBuffer> lb = [device newBufferWithLength:lcs_size
                                                   options:MTLResourceStorageModeShared];
            if (lb == nil) { err = -ENOMEM; goto fail_hbuf; }
            s->lcs_buf = (__bridge_retained void *)lb;
        }

        err = build_pipelines(s, device);
    }
    if (err != 0) { goto fail_lcs; }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

fail_pso:
    if (s->pso_vert)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert;  s->pso_vert  = NULL; }
    if (s->pso_horiz) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz; s->pso_horiz = NULL; }
fail_lcs:
    if (s->lcs_buf) { (void)(__bridge_transfer id<MTLBuffer>)s->lcs_buf; s->lcs_buf = NULL; }
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
    FloatSsimStateMetal *s = (FloatSsimStateMetal *)fex->priv;

    s->frame_w = ref_pic->w[0];
    s->frame_h = ref_pic->h[0];
    s->w_h     = (s->frame_w >= 10u) ? (s->frame_w - 10u) : 0u;
    s->h_v     = (s->frame_h >= 10u) ? (s->frame_h - 10u) : 0u;

    const size_t row_bytes   = (size_t)s->frame_w * (s->bpc <= 8u ? 1u : 2u);
    const size_t plane_bytes = row_bytes * s->frame_h;

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>       device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue>  queue = (__bridge id<MTLCommandQueue>)qh;
    id<MTLBuffer>    hbuf_buf  = (__bridge id<MTLBuffer>)s->hbuf_buf;
    id<MTLBuffer>    par_buf   = (__bridge id<MTLBuffer>)(void *)s->rb.buffer;

    /* Upload ref and dis as raw pixels; normalise to float in a CPU loop. */
    const size_t float_plane_bytes = (size_t)s->frame_w * s->frame_h * sizeof(float);
    id<MTLBuffer> ref_f = [device newBufferWithLength:float_plane_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dis_f = [device newBufferWithLength:float_plane_bytes options:MTLResourceStorageModeShared];
    if (ref_f == nil || dis_f == nil) { return -ENOMEM; }

    {
        float *rf = (float *)[ref_f contents];
        float *df = (float *)[dis_f contents];
        const float inv = 1.0f / s->scaler;
        if (s->bpc <= 8u) {
            for (unsigned y = 0; y < s->frame_h; ++y) {
                const uint8_t *rs = (const uint8_t *)ref_pic->data[0] + y * ref_pic->stride[0];
                const uint8_t *ds = (const uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0];
                for (unsigned x = 0; x < s->frame_w; ++x) {
                    rf[y * s->frame_w + x] = (float)rs[x];
                    df[y * s->frame_w + x] = (float)ds[x];
                }
            }
        } else {
            for (unsigned y = 0; y < s->frame_h; ++y) {
                const uint16_t *rs = (const uint16_t *)((uint8_t *)ref_pic->data[0] + y * ref_pic->stride[0]);
                const uint16_t *ds = (const uint16_t *)((uint8_t *)dist_pic->data[0] + y * dist_pic->stride[0]);
                for (unsigned x = 0; x < s->frame_w; ++x) {
                    rf[y * s->frame_w + x] = (float)rs[x] * inv;
                    df[y * s->frame_w + x] = (float)ds[x] * inv;
                }
            }
        }
        (void)plane_bytes; /* unused: float conversion above handles luma */
        (void)row_bytes;
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit fillBuffer:par_buf range:NSMakeRange(0, s->partials_count * sizeof(float)) value:0];
        if (s->lcs_buf != NULL) {
            id<MTLBuffer> lcs_b = (__bridge id<MTLBuffer>)s->lcs_buf;
            [blit fillBuffer:lcs_b range:NSMakeRange(0, 3u * s->partials_count * sizeof(float)) value:0];
        }
        [blit endEncoding];
    }

    /* Pass 0: horizontal convolution → hbuf */
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)s->pso_horiz];
        [enc setBuffer:ref_f    offset:0 atIndex:0];
        [enc setBuffer:dis_f    offset:0 atIndex:1];
        [enc setBuffer:hbuf_buf offset:0 atIndex:2];
        uint32_t params[4] = {(uint32_t)s->frame_w, (uint32_t)s->frame_h,
                              (uint32_t)s->w_h, 0};
        [enc setBytes:params length:sizeof(params) atIndex:3];
        MTLSize tg   = MTLSizeMake(16, 8, 1);
        MTLSize grid = MTLSizeMake((s->w_h + 15u) / 16u, (s->frame_h + 7u) / 8u, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* Pass 1: vertical convolution + SSIM combine + optional LCS → partials */
    {
        const size_t grid_w = (s->w_h + 15u) / 16u;
        const size_t grid_h = (s->h_v +  7u) /  8u;

        /* Dummy 4-byte LCS buffer when enable_lcs == false; Metal requires a
         * bound buffer even for [[buffer(5)]] when the kernel declares it.
         * We re-use par_buf's first 4 bytes (read-only by the disabled path). */
        id<MTLBuffer> lcs_b = (s->lcs_buf != NULL)
            ? (__bridge id<MTLBuffer>)s->lcs_buf
            : par_buf;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)s->pso_vert];
        [enc setBuffer:hbuf_buf offset:0 atIndex:0];
        [enc setBuffer:par_buf  offset:0 atIndex:1];
        uint32_t params[4] = {(uint32_t)s->frame_w, (uint32_t)s->frame_h,
                              (uint32_t)s->w_h, (uint32_t)s->h_v};
        [enc setBytes:params length:sizeof(params) atIndex:2];
        float consts[2] = {s->c1, s->c2};
        [enc setBytes:consts length:sizeof(consts) atIndex:3];
        uint32_t gdim[2] = {(uint32_t)grid_w, (uint32_t)grid_h};
        [enc setBytes:gdim length:sizeof(gdim) atIndex:4];
        [enc setBuffer:lcs_b   offset:0 atIndex:5];
        uint32_t lcs_flags = s->enable_lcs ? 1u : 0u;
        [enc setBytes:&lcs_flags length:sizeof(lcs_flags) atIndex:6];
        MTLSize tg   = MTLSizeMake(16, 8, 1);
        MTLSize grid_sz = MTLSizeMake(grid_w, grid_h, 1);
        [enc dispatchThreadgroups:grid_sz threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    FloatSsimStateMetal *s = (FloatSsimStateMetal *)fex->priv;

    const float *parts = (const float *)s->rb.host_view;
    double ssim_sum = 0.0;
    if (parts != NULL) {
        for (size_t i = 0; i < s->partials_count; ++i) {
            ssim_sum += (double)parts[i];
        }
    }
    const double n_valid = (double)s->w_h * (double)s->h_v;
    double ssim = (n_valid > 0.0) ? (ssim_sum / n_valid) : 1.0;

    if (s->enable_db) {
        ssim = ssim_to_db(ssim, s->max_db);
    }

    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_ssim", ssim, index);
    if (err != 0) { return err; }

    if (s->enable_lcs && s->lcs_buf != NULL) {
        const float *lp = (const float *)[(__bridge id<MTLBuffer>)s->lcs_buf contents];
        if (lp != NULL) {
            double l_sum = 0.0, c_sum = 0.0, ss_sum = 0.0;
            for (size_t i = 0; i < s->partials_count; ++i) {
                l_sum  += (double)lp[0 * s->partials_count + i];
                c_sum  += (double)lp[1 * s->partials_count + i];
                ss_sum += (double)lp[2 * s->partials_count + i];
            }
            const double l_score = (n_valid > 0.0) ? (l_sum  / n_valid) : 1.0;
            const double c_score = (n_valid > 0.0) ? (c_sum  / n_valid) : 1.0;
            const double s_score = (n_valid > 0.0) ? (ss_sum / n_valid) : 1.0;
            err  = vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict, "float_ssim_l", l_score, index);
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict, "float_ssim_c", c_score, index);
            err |= vmaf_feature_collector_append_with_dict(
                feature_collector, s->feature_name_dict, "float_ssim_s", s_score, index);
        }
    }

    return err;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    FloatSsimStateMetal *s = (FloatSsimStateMetal *)fex->priv;
    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    if (s->pso_vert)  { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_vert;  s->pso_vert  = NULL; }
    if (s->pso_horiz) { (void)(__bridge_transfer id<MTLComputePipelineState>)s->pso_horiz; s->pso_horiz = NULL; }
    if (s->lcs_buf)   { (void)(__bridge_transfer id<MTLBuffer>)s->lcs_buf;                 s->lcs_buf   = NULL; }
    if (s->hbuf_buf)  { (void)(__bridge_transfer id<MTLBuffer>)s->hbuf_buf;                s->hbuf_buf  = NULL; }

    int err = vmaf_metal_kernel_buffer_free(&s->rb, s->ctx);
    if (err != 0 && rc == 0) { rc = err; }
    if (s->feature_name_dict) { (void)vmaf_dictionary_free(&s->feature_name_dict); }
    if (s->ctx) { vmaf_metal_context_destroy(s->ctx); s->ctx = NULL; }
    return rc;
}

static const char *provided_features[] = {
    "float_ssim", "float_ssim_l", "float_ssim_c", "float_ssim_s", NULL
};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_ssim_metal = {
    .name              = "float_ssim_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(FloatSsimStateMetal),
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
