/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ms_ssim feature extractor on the Metal backend (T8-2a / ADR-0435).
 *  Port of `libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c` — same
 *  5-scale pyramid, same Wang weights, same host accumulation logic.
 *
 *  Algorithm summary:
 *    1. Host normalises ref+cmp YUV → float in [0, 255] (cpu loop, same
 *       as float_ssim_metal.mm).
 *    2. ms_ssim_decimate kernel builds pyramid levels 1–4 (× ref + cmp).
 *    3. Per scale: ms_ssim_horiz → ms_ssim_vert_lcs, both in a single
 *       MTLCommandBuffer. DtoH is implicit (Shared storage on Apple
 *       unified memory — no explicit copy needed).
 *    4. Host reduces per-WG partials × 3 in double precision per scale,
 *       applies Wang weights for the final product combine.
 *
 *  Metallib resolution: same embedded-blob pattern as every other Metal
 *  feature extractor — reads the __TEXT,__metallib section compiled via
 *  xcrun from integer_ms_ssim.metal.
 *
 *  Min-dim guard (ADR-0153): 11 × 2^4 = 176 × 176 enforced in init().
 *
 *  enable_lcs (mirrors CUDA twin T7-35 / ADR-0243): when set, emits the
 *  15 extra per-scale metrics float_ms_ssim_{l,c,s}_scale{0..4}. Default
 *  path output is bit-identical to the pre-T7-35 binary.
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

#define MS_SSIM_SCALES         5
#define MS_SSIM_GAUSSIAN_LEN  11
#define MS_SSIM_K             11
#define MS_SSIM_BLOCK_X       16
#define MS_SSIM_BLOCK_Y        8

static const float g_alphas[MS_SSIM_SCALES] = {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f};
static const float g_betas[MS_SSIM_SCALES]  = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};
static const float g_gammas[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};

typedef struct MsSsimStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext *ctx;

    /* Pipeline states for the three MS-SSIM kernels. */
    void *pso_decimate;
    void *pso_horiz;
    void *pso_vert_lcs;

    /* Pyramid buffers: 5 levels × ref + cmp (float, Shared storage). */
    void *pyramid_ref[MS_SSIM_SCALES];
    void *pyramid_cmp[MS_SSIM_SCALES];

    /* Intermediate 5-plane horiz buffer (w_h × H × 5 floats);
     * sized for scale 0 (largest). */
    void *hbuf;

    /* Per-scale partials buffers: l, c, s (float, Shared storage).
     * Sized to block_count[i] = ceil(w_f[i]/16) × ceil(h_f[i]/8). */
    void *l_partials[MS_SSIM_SCALES];
    void *c_partials[MS_SSIM_SCALES];
    void *s_partials[MS_SSIM_SCALES];

    unsigned width;
    unsigned height;
    unsigned bpc;
    float    scaler;  /* raw → [0,255] multiplier inverse for >8bpc */

    /* Per-scale geometry (mirrors CUDA twin). */
    unsigned scale_w[MS_SSIM_SCALES];
    unsigned scale_h[MS_SSIM_SCALES];
    unsigned scale_w_h[MS_SSIM_SCALES];   /* w_h = scale_w[i] - 10 */
    unsigned scale_w_f[MS_SSIM_SCALES];   /* w_f = scale_w[i] - 10 */
    unsigned scale_h_f[MS_SSIM_SCALES];   /* h_f = scale_h[i] - 10 */
    unsigned scale_grid_w[MS_SSIM_SCALES];
    unsigned scale_grid_h[MS_SSIM_SCALES];
    unsigned scale_block_count[MS_SSIM_SCALES];

    float c1;
    float c2;
    float c3;

    bool enable_lcs;

    unsigned index;
    VmafDictionary *feature_name_dict;
} MsSsimStateMetal;

static const VmafOption options[] = {
    {
        .name        = "enable_lcs",
        .help        = "enable luminance, contrast and structure intermediate output",
        .offset      = offsetof(MsSsimStateMetal, enable_lcs),
        .type        = VMAF_OPT_TYPE_BOOL,
        .default_val = {.b = false},
    },
    {0},
};

static int build_pipelines(MsSsimStateMetal *s, id<MTLDevice> device)
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

    id<MTLFunction> fn_dec  = [lib newFunctionWithName:@"ms_ssim_decimate"];
    id<MTLFunction> fn_hor  = [lib newFunctionWithName:@"ms_ssim_horiz"];
    id<MTLFunction> fn_vlcs = [lib newFunctionWithName:@"ms_ssim_vert_lcs"];
    if (fn_dec == nil || fn_hor == nil || fn_vlcs == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_dec =
        [device newComputePipelineStateWithFunction:fn_dec  error:&err];
    id<MTLComputePipelineState> pso_hor =
        [device newComputePipelineStateWithFunction:fn_hor  error:&err];
    id<MTLComputePipelineState> pso_vlcs =
        [device newComputePipelineStateWithFunction:fn_vlcs error:&err];
    if (pso_dec == nil || pso_hor == nil || pso_vlcs == nil) { return -ENODEV; }

    s->pso_decimate = (__bridge_retained void *)pso_dec;
    s->pso_horiz    = (__bridge_retained void *)pso_hor;
    s->pso_vert_lcs = (__bridge_retained void *)pso_vlcs;
    return 0;
}

static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    MsSsimStateMetal *s = (MsSsimStateMetal *)fex->priv;

    /* ADR-0153 minimum resolution guard. */
    const unsigned min_dim = (unsigned)MS_SSIM_GAUSSIAN_LEN << (MS_SSIM_SCALES - 1u);
    if (w < min_dim || h < min_dim) { return -EINVAL; }

    s->width  = w;
    s->height = h;
    s->bpc    = bpc;

    if (bpc <= 8u)       { s->scaler = 1.0f; }
    else if (bpc == 10u) { s->scaler = 4.0f; }
    else if (bpc == 12u) { s->scaler = 16.0f; }
    else                 { s->scaler = 256.0f; }

    /* Build pyramid geometry (mirrors CUDA twin). */
    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < MS_SSIM_SCALES; ++i) {
        s->scale_w[i] = (s->scale_w[i - 1] / 2u) + (s->scale_w[i - 1] & 1u);
        s->scale_h[i] = (s->scale_h[i - 1] / 2u) + (s->scale_h[i - 1] & 1u);
    }
    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        s->scale_w_h[i] = s->scale_w[i] - (unsigned)(MS_SSIM_K - 1);
        s->scale_w_f[i] = s->scale_w_h[i];  /* valid width after vert pass */
        s->scale_h_f[i] = s->scale_h[i] - (unsigned)(MS_SSIM_K - 1);
        s->scale_grid_w[i] =
            (s->scale_w_f[i] + (unsigned)MS_SSIM_BLOCK_X - 1u) / (unsigned)MS_SSIM_BLOCK_X;
        s->scale_grid_h[i] =
            (s->scale_h_f[i] + (unsigned)MS_SSIM_BLOCK_Y - 1u) / (unsigned)MS_SSIM_BLOCK_Y;
        s->scale_block_count[i] = s->scale_grid_w[i] * s->scale_grid_h[i];
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

        /* Pyramid buffers. */
        for (int i = 0; i < MS_SSIM_SCALES; ++i) {
            const size_t bytes = (size_t)s->scale_w[i] * s->scale_h[i] * sizeof(float);
            id<MTLBuffer> br = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> bc = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
            if (br == nil || bc == nil) { err = -ENOMEM; goto fail_lc; }
            s->pyramid_ref[i] = (__bridge_retained void *)br;
            s->pyramid_cmp[i] = (__bridge_retained void *)bc;
        }

        /* hbuf: 5 planes × w_h[0] × H[0] floats. */
        {
            const size_t hbuf_bytes = 5u * (size_t)s->scale_w_h[0] * s->scale_h[0] * sizeof(float);
            id<MTLBuffer> hb = [device newBufferWithLength:hbuf_bytes
                                                   options:MTLResourceStorageModeShared];
            if (hb == nil) { err = -ENOMEM; goto fail_lc; }
            s->hbuf = (__bridge_retained void *)hb;
        }

        /* Per-scale partials buffers. */
        for (int i = 0; i < MS_SSIM_SCALES; ++i) {
            const size_t pb = (size_t)s->scale_block_count[i] * sizeof(float);
            id<MTLBuffer> bl = [device newBufferWithLength:pb
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> bc2 = [device newBufferWithLength:pb
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> bs = [device newBufferWithLength:pb
                                                   options:MTLResourceStorageModeShared];
            if (bl == nil || bc2 == nil || bs == nil) { err = -ENOMEM; goto fail_lc; }
            s->l_partials[i] = (__bridge_retained void *)bl;
            s->c_partials[i] = (__bridge_retained void *)bc2;
            s->s_partials[i] = (__bridge_retained void *)bs;
        }

        err = build_pipelines(s, device);
    }
    if (err != 0) { goto fail_lc; }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_pso; }
    return 0;

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

/* Normalise ref/cmp picture Y-plane → float in [0, 255]. */
static void fill_float_plane(VmafPicture *pic, id<MTLBuffer> dst, unsigned w, unsigned h,
                             float inv_scaler, unsigned bpc)
{
    float *out = (float *)[dst contents];
    if (bpc <= 8u) {
        for (unsigned y = 0; y < h; ++y) {
            const uint8_t *row = (const uint8_t *)pic->data[0] + (size_t)y * pic->stride[0];
            for (unsigned x = 0; x < w; ++x) {
                out[y * w + x] = (float)row[x];
            }
        }
    } else {
        for (unsigned y = 0; y < h; ++y) {
            const uint16_t *row =
                (const uint16_t *)((const uint8_t *)pic->data[0] + (size_t)y * pic->stride[0]);
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
    MsSsimStateMetal *s = (MsSsimStateMetal *)fex->priv;

    s->index   = index;
    s->width   = ref_pic->w[0];
    s->height  = ref_pic->h[0];

    void *dh = vmaf_metal_context_device_handle(s->ctx);
    void *qh = vmaf_metal_context_queue_handle(s->ctx);
    if (dh == NULL || qh == NULL) { return -ENODEV; }

    id<MTLDevice>       device = (__bridge id<MTLDevice>)dh;
    id<MTLCommandQueue>  queue = (__bridge id<MTLCommandQueue>)qh;

    /* Normalise ref + cmp into pyramid[0] buffers directly. */
    const float inv = 1.0f / s->scaler;
    id<MTLBuffer> pyr_ref0 = (__bridge id<MTLBuffer>)s->pyramid_ref[0];
    id<MTLBuffer> pyr_cmp0 = (__bridge id<MTLBuffer>)s->pyramid_cmp[0];
    fill_float_plane(ref_pic,  pyr_ref0, s->scale_w[0], s->scale_h[0], inv, s->bpc);
    fill_float_plane(dist_pic, pyr_cmp0, s->scale_w[0], s->scale_h[0], inv, s->bpc);

    id<MTLComputePipelineState> pso_dec  =
        (__bridge id<MTLComputePipelineState>)s->pso_decimate;
    id<MTLComputePipelineState> pso_hor  =
        (__bridge id<MTLComputePipelineState>)s->pso_horiz;
    id<MTLComputePipelineState> pso_vlcs =
        (__bridge id<MTLComputePipelineState>)s->pso_vert_lcs;
    id<MTLBuffer> hbuf = (__bridge id<MTLBuffer>)s->hbuf;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { return -ENOMEM; }

    /* Build pyramid scales 1..4 via decimate kernel (× ref + cmp). */
    for (int i = 0; i < MS_SSIM_SCALES - 1; ++i) {
        const uint32_t dims[4] = {
            (uint32_t)s->scale_w[i], (uint32_t)s->scale_h[i],
            (uint32_t)s->scale_w[i + 1], (uint32_t)s->scale_h[i + 1],
        };
        const size_t grid_x = (s->scale_w[i + 1] + 15u) / 16u;
        const size_t grid_y = (s->scale_h[i + 1] + 15u) / 16u;

        for (int side = 0; side < 2; ++side) {
            id<MTLBuffer> src_buf = (side == 0)
                ? (__bridge id<MTLBuffer>)s->pyramid_ref[i]
                : (__bridge id<MTLBuffer>)s->pyramid_cmp[i];
            id<MTLBuffer> dst_buf = (side == 0)
                ? (__bridge id<MTLBuffer>)s->pyramid_ref[i + 1]
                : (__bridge id<MTLBuffer>)s->pyramid_cmp[i + 1];

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_dec];
            [enc setBuffer:src_buf offset:0 atIndex:0];
            [enc setBuffer:dst_buf offset:0 atIndex:1];
            [enc setBytes:dims length:sizeof(dims) atIndex:2];
            MTLSize tg   = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake(grid_x, grid_y, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }

    /* Per-scale SSIM: horiz → vert_lcs. */
    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        id<MTLBuffer> ref_i = (__bridge id<MTLBuffer>)s->pyramid_ref[i];
        id<MTLBuffer> cmp_i = (__bridge id<MTLBuffer>)s->pyramid_cmp[i];
        id<MTLBuffer> lp    = (__bridge id<MTLBuffer>)s->l_partials[i];
        id<MTLBuffer> cp    = (__bridge id<MTLBuffer>)s->c_partials[i];
        id<MTLBuffer> sp    = (__bridge id<MTLBuffer>)s->s_partials[i];

        const uint32_t hor_params[4] = {
            (uint32_t)s->scale_w[i], (uint32_t)s->scale_h[i],
            (uint32_t)s->scale_w_h[i], 0u,
        };
        const size_t hor_gx = (s->scale_w_h[i] + 15u) / 16u;
        const size_t hor_gy = (s->scale_h[i]   +  7u) /  8u;

        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_hor];
            [enc setBuffer:ref_i  offset:0 atIndex:0];
            [enc setBuffer:cmp_i  offset:0 atIndex:1];
            [enc setBuffer:hbuf   offset:0 atIndex:2];
            [enc setBytes:hor_params length:sizeof(hor_params) atIndex:3];
            MTLSize tg   = MTLSizeMake(16, 8, 1);
            MTLSize grid = MTLSizeMake(hor_gx, hor_gy, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        const uint32_t vlcs_params[4] = {
            (uint32_t)s->scale_w_h[i], (uint32_t)s->scale_h[i],
            (uint32_t)s->scale_w_f[i], (uint32_t)s->scale_h_f[i],
        };
        const float vlcs_consts[4] = {s->c1, s->c2, s->c3, 0.0f};
        const uint32_t grid_dim[2] = {
            (uint32_t)s->scale_grid_w[i],
            (uint32_t)s->scale_grid_h[i],
        };

        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_vlcs];
            [enc setBuffer:hbuf offset:0 atIndex:0];
            [enc setBuffer:lp   offset:0 atIndex:1];
            [enc setBuffer:cp   offset:0 atIndex:2];
            [enc setBuffer:sp   offset:0 atIndex:3];
            [enc setBytes:vlcs_params  length:sizeof(vlcs_params)  atIndex:4];
            [enc setBytes:vlcs_consts  length:sizeof(vlcs_consts)  atIndex:5];
            [enc setBytes:grid_dim     length:sizeof(grid_dim)     atIndex:6];
            MTLSize tg   = MTLSizeMake(MS_SSIM_BLOCK_X, MS_SSIM_BLOCK_Y, 1);
            MTLSize grid = MTLSizeMake(s->scale_grid_w[i], s->scale_grid_h[i], 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    return 0;
}

static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    MsSsimStateMetal *s = (MsSsimStateMetal *)fex->priv;

    double l_means[MS_SSIM_SCALES] = {0.0};
    double c_means[MS_SSIM_SCALES] = {0.0};
    double s_means[MS_SSIM_SCALES] = {0.0};

    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        const float *lp = (const float *)[(__bridge id<MTLBuffer>)s->l_partials[i] contents];
        const float *cp = (const float *)[(__bridge id<MTLBuffer>)s->c_partials[i] contents];
        const float *sp = (const float *)[(__bridge id<MTLBuffer>)s->s_partials[i] contents];

        double tl = 0.0, tc = 0.0, ts = 0.0;
        for (unsigned j = 0; j < s->scale_block_count[i]; ++j) {
            tl += (double)lp[j];
            tc += (double)cp[j];
            ts += (double)sp[j];
        }
        const double n = (double)s->scale_w_f[i] * (double)s->scale_h_f[i];
        l_means[i] = (n > 0.0) ? (tl / n) : 0.0;
        c_means[i] = (n > 0.0) ? (tc / n) : 0.0;
        s_means[i] = (n > 0.0) ? (ts / n) : 0.0;
    }

    double msssim = 1.0;
    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        msssim *= pow(l_means[i], (double)g_alphas[i]) *
                  pow(c_means[i], (double)g_betas[i]) *
                  pow(fabs(s_means[i]), (double)g_gammas[i]);
    }

    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "float_ms_ssim", msssim, index);

    if (s->enable_lcs) {
        static const char *const l_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_l_scale0", "float_ms_ssim_l_scale1", "float_ms_ssim_l_scale2",
            "float_ms_ssim_l_scale3", "float_ms_ssim_l_scale4",
        };
        static const char *const c_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_c_scale0", "float_ms_ssim_c_scale1", "float_ms_ssim_c_scale2",
            "float_ms_ssim_c_scale3", "float_ms_ssim_c_scale4",
        };
        static const char *const s_names[MS_SSIM_SCALES] = {
            "float_ms_ssim_s_scale0", "float_ms_ssim_s_scale1", "float_ms_ssim_s_scale2",
            "float_ms_ssim_s_scale3", "float_ms_ssim_s_scale4",
        };
        for (int i = 0; i < MS_SSIM_SCALES; ++i) {
            err |= vmaf_feature_collector_append(feature_collector, l_names[i], l_means[i], index);
            err |= vmaf_feature_collector_append(feature_collector, c_names[i], c_means[i], index);
            err |= vmaf_feature_collector_append(feature_collector, s_names[i], s_means[i], index);
        }
    }
    return err;
}

static int close_fex_metal(VmafFeatureExtractor *fex)
{
    MsSsimStateMetal *s = (MsSsimStateMetal *)fex->priv;

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

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

    for (int i = 0; i < MS_SSIM_SCALES; ++i) {
        if (s->pyramid_ref[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->pyramid_ref[i];
            s->pyramid_ref[i] = NULL;
        }
        if (s->pyramid_cmp[i]) {
            (void)(__bridge_transfer id<MTLBuffer>)s->pyramid_cmp[i];
            s->pyramid_cmp[i] = NULL;
        }
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

static const char *provided_features[] = {"float_ms_ssim", NULL};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_float_ms_ssim_metal = {
    .name              = "float_ms_ssim_metal",
    .init              = init_fex_metal,
    .submit            = submit_fex_metal,
    .collect           = collect_fex_metal,
    .flush             = NULL,
    .close             = close_fex_metal,
    .options           = options,
    .priv_size         = sizeof(MsSsimStateMetal),
    .provided_features = provided_features,
    .flags             = 0,
    .chars = {
        .n_dispatches_per_frame = 3 * MS_SSIM_SCALES + 2 * (MS_SSIM_SCALES - 1),
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_AUTO,
    },
};
} /* extern "C" */
