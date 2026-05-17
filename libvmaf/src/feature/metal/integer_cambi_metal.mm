/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CAMBI banding-detection feature extractor on the Metal backend.
 *
 *  Strategy II hybrid (mirrors the CUDA twin in
 *  `libvmaf/src/feature/cuda/integer_cambi_cuda.c` + ADR-0360):
 *
 *    Metal GPU kernels (integer_cambi.metal):
 *      - cambi_spatial_mask_kernel: 7×7 zero-deriv box-sum + threshold.
 *      - cambi_decimate_kernel: strict 2× stride-2 subsample.
 *      - cambi_filter_mode_kernel: separable 3-tap mode filter (H + V).
 *
 *    Host CPU (via cambi_internal.h — exact same CPU code as CUDA twin):
 *      - vmaf_cambi_preprocessing: resize/upcast to 10-bit.
 *      - vmaf_cambi_calculate_c_values: sliding-histogram c-value pass.
 *      - vmaf_cambi_spatial_pooling: top-K pooling → per-scale score.
 *      - vmaf_cambi_weight_scores_per_scale: inner-product scale weights.
 *
 *  Per-frame flow (same as CUDA; submit is synchronous for v1):
 *    1. Host vmaf_cambi_preprocessing → pics[0].
 *    2. Copy pics[0].data[0] → d_image (MTLBuffer, Shared storage).
 *    3. GPU cambi_spatial_mask_kernel over d_image → d_mask.
 *    4. For scale = 0..4:
 *         a. (scale>0) GPU decimate d_image → d_tmp, swap;
 *                      GPU decimate d_mask  → d_tmp, swap.
 *         b. GPU filter_mode H: d_image → d_tmp.
 *            GPU filter_mode V: d_tmp   → d_image.
 *         c. [cmd commit] + [cmd waitUntilCompleted].
 *         d. Copy d_image → pics[0], d_mask → pics[1].
 *         e. Host vmaf_cambi_calculate_c_values + vmaf_cambi_spatial_pooling.
 *    5. vmaf_cambi_weight_scores_per_scale → final score.
 *    6. Emit "Cambi_feature_cambi_score" in collect().
 *
 *  Precision target: ULP=0 vs CPU scalar (places=4), mirroring the
 *  CUDA twin's precision contract.  Validation pending Apple Silicon run.
 *
 *  macOS-only: compiled only when enable_metal=true in meson.
 *  CPU-only builds produce a -ENOSYS stub (not this file).
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" {
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"
#include "log.h"
#include "luminance_tools.h"
#include "mem.h"

#include "../../metal/common.h"
#include "../../metal/kernel_template.h"
#include "feature/cambi_internal.h"
}

/* --- Constants (match cambi.c / CUDA twin) --- */
#define CAMBI_METAL_NUM_SCALES         5
#define CAMBI_METAL_MIN_WIDTH_HEIGHT   216
#define CAMBI_METAL_MASK_FILTER_SIZE   7
#define CAMBI_METAL_DEFAULT_MAX_VAL    1000.0
#define CAMBI_METAL_DEFAULT_WINDOW_SIZE 65
#define CAMBI_METAL_DEFAULT_TOPK       0.6
#define CAMBI_METAL_DEFAULT_TVI        0.019
#define CAMBI_METAL_DEFAULT_VLT        0.0
#define CAMBI_METAL_DEFAULT_MAX_LOG_CONTRAST 2
#define CAMBI_METAL_DEFAULT_EOTF       "bt1886"
#define CAMBI_METAL_BLOCK_X            16u
#define CAMBI_METAL_BLOCK_Y            16u

/* Linker-defined symbols bracketing the embedded metallib byte range
 * (same embed mechanism as integer_motion_v2_metal.mm). */
extern "C" {
extern const unsigned char libvmaf_metallib_start[] __asm("section$start$__TEXT$__metallib");
extern const unsigned char libvmaf_metallib_end[]   __asm("section$end$__TEXT$__metallib");
}

/* --- MTLBuffer wrapper: shared-storage uint16 plane --- */
typedef struct {
    void   *mtl_buf;   /* bridge-retained id<MTLBuffer> */
    void   *contents;  /* [mtl_buf contents] — directly-mapped host ptr */
    size_t  size;      /* allocation size in bytes */
} CambiMetalBuf;

static int cambi_metal_buf_alloc(CambiMetalBuf *b, id<MTLDevice> device, size_t sz)
{
    id<MTLBuffer> buf = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
    if (buf == nil) {
        return -ENOMEM;
    }
    b->mtl_buf  = (__bridge_retained void *)buf;
    b->contents = [buf contents];
    b->size     = sz;
    return 0;
}

static void cambi_metal_buf_free(CambiMetalBuf *b)
{
    if (b->mtl_buf != NULL) {
        id<MTLBuffer> buf __attribute__((unused)) =
            (__bridge_transfer id<MTLBuffer>)b->mtl_buf;
        b->mtl_buf  = NULL;
        b->contents = NULL;
        b->size     = 0;
    }
}

/* --- Private state --- */
typedef struct CambiStateMetal {
    VmafMetalKernelLifecycle lc;
    VmafMetalContext *ctx;

    /* Pipeline states for the three kernels. */
    void *pso_mask;        /* cambi_spatial_mask_kernel */
    void *pso_decimate;    /* cambi_decimate_kernel */
    void *pso_filter_mode; /* cambi_filter_mode_kernel */

    /* Device buffers (flat uint16 planes, scale-0 size). */
    CambiMetalBuf d_image; /* current scale image */
    CambiMetalBuf d_mask;  /* spatial mask */
    CambiMetalBuf d_tmp;   /* scratch for decimate / filter_mode H */

    /* Host VmafPicture pair for CPU residual. */
    VmafPicture pics[2]; /* pics[0]=image, pics[1]=mask */

    /* Host scratch buffers for the CPU residual. */
    VmafCambiHostBuffers buffers;
    VmafCambiRangeUpdater inc_range_callback;
    VmafCambiRangeUpdater dec_range_callback;
    VmafCambiDerivativeCalculator derivative_callback;

    /* Configuration options. */
    int enc_width;
    int enc_height;
    int enc_bitdepth;
    int max_log_contrast;
    int window_size;
    double topk;
    double cambi_topk;
    double tvi_threshold;
    double cambi_max_val;
    double cambi_vis_lum_threshold;
    char *eotf;
    char *cambi_eotf;

    /* Resolved per-frame geometry. */
    unsigned src_width;
    unsigned src_height;
    unsigned src_bpc;
    unsigned proc_width;
    unsigned proc_height;

    uint16_t adjusted_window;
    uint16_t vlt_luma;

    /* Index + score stored by submit for collect. */
    unsigned index;
    double   last_score;

    VmafDictionary *feature_name_dict;
} CambiStateMetal;

/* --- Options --- */
static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped",
        .offset = offsetof(CambiStateMetal, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_METAL_DEFAULT_MAX_VAL,
        .min = 0.0, .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiStateMetal, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0, .min = 180, .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encw",
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiStateMetal, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0, .min = 150, .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ench",
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(CambiStateMetal, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0, .min = 6, .max = 16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encbd",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 65 corresponds to ~1 degree at 4k",
        .offset = offsetof(CambiStateMetal, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_METAL_DEFAULT_WINDOW_SIZE, .min = 15, .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiStateMetal, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_METAL_DEFAULT_TOPK, .min = 0.0001, .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation (alias)",
        .offset = offsetof(CambiStateMetal, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_METAL_DEFAULT_TOPK, .min = 0.0001, .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibility threshold deltaL < tvi_threshold * L_mean",
        .offset = offsetof(CambiStateMetal, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_METAL_DEFAULT_TVI, .min = 0.0001, .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which banding is assumed invisible",
        .offset = offsetof(CambiStateMetal, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_METAL_DEFAULT_VLT, .min = 0.0, .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum log contrast (0 to 5, default 2)",
        .offset = offsetof(CambiStateMetal, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_METAL_DEFAULT_MAX_LOG_CONTRAST, .min = 0, .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "eotf",
        .help = "EOTF for visibility-threshold conversion (bt1886 / pq)",
        .offset = offsetof(CambiStateMetal, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_METAL_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "EOTF override for cambi (defaults to eotf)",
        .offset = offsetof(CambiStateMetal, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_METAL_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* Helpers (mirrors CUDA twin helper functions)                        */
/* ------------------------------------------------------------------ */
static uint16_t cambi_metal_adjust_window(int window_size, unsigned w, unsigned h)
{
    unsigned adjusted = (unsigned)(window_size) * (w + h) / 375u;
    adjusted >>= 4;
    if (adjusted < 1u) { adjusted = 1u; }
    if ((adjusted & 1u) == 0u) { adjusted++; }
    return (uint16_t)adjusted;
}

static uint16_t cambi_metal_ceil_log2(uint32_t num)
{
    if (num == 0u) { return 0u; }
    uint32_t tmp = num - 1u;
    uint16_t shift = 0;
    while (tmp > 0u) { tmp >>= 1; shift++; }
    return shift;
}

static uint16_t cambi_metal_get_mask_index(unsigned w, unsigned h, uint16_t filter_size)
{
    const uint32_t shifted_wh = (w >> 6) * (h >> 6);
    return (uint16_t)(
        (filter_size * filter_size + 3 * (cambi_metal_ceil_log2(shifted_wh) - 11) - 1) >> 1);
}

static int cambi_metal_init_tvi(CambiStateMetal *s)
{
    VmafLumaRange luma_range;
    int err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err != 0) { return err; }

    const char *effective_eotf;
    if (s->cambi_eotf != NULL &&
        strcmp(s->cambi_eotf, CAMBI_METAL_DEFAULT_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = (s->eotf != NULL) ? s->eotf : CAMBI_METAL_DEFAULT_EOTF;
    }

    VmafEOTF eotf;
    err = vmaf_luminance_init_eotf(&eotf, effective_eotf);
    if (err != 0) { return err; }

    const int num_diffs = 1 << s->max_log_contrast;
    for (int d = 0; d < num_diffs; d++) {
        const int diff = (int)s->buffers.diffs_to_consider[d];
        int lo = 0;
        int hi = (1 << 10) - 1 - diff;
        int found = -1;
        while (lo <= hi) {
            const int mid = (lo + hi) / 2;
            const double sample_lum = vmaf_luminance_get_luminance(mid, luma_range, eotf);
            const double diff_lum =
                vmaf_luminance_get_luminance(mid + diff, luma_range, eotf) - sample_lum;
            if (diff_lum < s->tvi_threshold * sample_lum) {
                found = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (found < 0) { found = 0; }
        s->buffers.tvi_for_diff[d] = (uint16_t)(found + num_diffs);
    }

    int vlt = 0;
    for (int v = 0; v < (1 << 10); v++) {
        const double L = vmaf_luminance_get_luminance(v, luma_range, eotf);
        if (L < s->cambi_vis_lum_threshold) { vlt = v; }
    }
    s->vlt_luma = (uint16_t)vlt;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Pipeline construction                                               */
/* ------------------------------------------------------------------ */
static int build_pipelines(CambiStateMetal *s, id<MTLDevice> device)
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
    if (lib == nil) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "cambi_metal: failed to load metallib: %s\n",
                 err ? [[err localizedDescription] UTF8String] : "unknown");
        return -ENODEV;
    }

    id<MTLFunction> fn_mask        = [lib newFunctionWithName:@"cambi_spatial_mask_kernel"];
    id<MTLFunction> fn_decimate    = [lib newFunctionWithName:@"cambi_decimate_kernel"];
    id<MTLFunction> fn_filter_mode = [lib newFunctionWithName:@"cambi_filter_mode_kernel"];
    if (fn_mask == nil || fn_decimate == nil || fn_filter_mode == nil) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "cambi_metal: one or more kernel functions not found in metallib.\n");
        return -ENODEV;
    }

    NSError *pso_err = nil;
    id<MTLComputePipelineState> pso_mask =
        [device newComputePipelineStateWithFunction:fn_mask error:&pso_err];
    if (pso_mask == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_decimate =
        [device newComputePipelineStateWithFunction:fn_decimate error:&pso_err];
    if (pso_decimate == nil) { return -ENODEV; }

    id<MTLComputePipelineState> pso_filter =
        [device newComputePipelineStateWithFunction:fn_filter_mode error:&pso_err];
    if (pso_filter == nil) { return -ENODEV; }

    s->pso_mask        = (__bridge_retained void *)pso_mask;
    s->pso_decimate    = (__bridge_retained void *)pso_decimate;
    s->pso_filter_mode = (__bridge_retained void *)pso_filter;
    return 0;
}

static void release_pipelines(CambiStateMetal *s)
{
    if (s->pso_mask != NULL) {
        id<MTLComputePipelineState> p __attribute__((unused)) =
            (__bridge_transfer id<MTLComputePipelineState>)s->pso_mask;
        s->pso_mask = NULL;
    }
    if (s->pso_decimate != NULL) {
        id<MTLComputePipelineState> p __attribute__((unused)) =
            (__bridge_transfer id<MTLComputePipelineState>)s->pso_decimate;
        s->pso_decimate = NULL;
    }
    if (s->pso_filter_mode != NULL) {
        id<MTLComputePipelineState> p __attribute__((unused)) =
            (__bridge_transfer id<MTLComputePipelineState>)s->pso_filter_mode;
        s->pso_filter_mode = NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Dispatch helpers                                                    */
/* ------------------------------------------------------------------ */

/* Encode one compute pass for cambi_spatial_mask_kernel.
 * params: uint4(width, height, stride_words, mask_index). */
static void encode_mask(id<MTLCommandBuffer> cmd,
                        id<MTLComputePipelineState> pso,
                        id<MTLBuffer> image_buf,
                        id<MTLBuffer> mask_buf,
                        unsigned width, unsigned height,
                        unsigned stride_words, unsigned mask_index)
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:image_buf offset:0 atIndex:0];
    [enc setBuffer:mask_buf  offset:0 atIndex:1];
    const uint32_t p[4] = {width, height, stride_words, mask_index};
    [enc setBytes:p length:sizeof(p) atIndex:2];
    const MTLSize tg   = MTLSizeMake(CAMBI_METAL_BLOCK_X, CAMBI_METAL_BLOCK_Y, 1);
    const MTLSize grid = MTLSizeMake((width  + CAMBI_METAL_BLOCK_X - 1) / CAMBI_METAL_BLOCK_X,
                                     (height + CAMBI_METAL_BLOCK_Y - 1) / CAMBI_METAL_BLOCK_Y, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
}

/* Encode one compute pass for cambi_decimate_kernel.
 * params: uint4(out_width, out_height, src_stride_words, dst_stride_words). */
static void encode_decimate(id<MTLCommandBuffer> cmd,
                            id<MTLComputePipelineState> pso,
                            id<MTLBuffer> src_buf, id<MTLBuffer> dst_buf,
                            unsigned out_w, unsigned out_h,
                            unsigned src_stride, unsigned dst_stride)
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:src_buf offset:0 atIndex:0];
    [enc setBuffer:dst_buf offset:0 atIndex:1];
    const uint32_t p[4] = {out_w, out_h, src_stride, dst_stride};
    [enc setBytes:p length:sizeof(p) atIndex:2];
    const MTLSize tg   = MTLSizeMake(CAMBI_METAL_BLOCK_X, CAMBI_METAL_BLOCK_Y, 1);
    const MTLSize grid = MTLSizeMake((out_w + CAMBI_METAL_BLOCK_X - 1) / CAMBI_METAL_BLOCK_X,
                                     (out_h + CAMBI_METAL_BLOCK_Y - 1) / CAMBI_METAL_BLOCK_Y, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
}

/* Encode one compute pass for cambi_filter_mode_kernel.
 * params: uint4(width, height, stride_words, axis). */
static void encode_filter_mode(id<MTLCommandBuffer> cmd,
                               id<MTLComputePipelineState> pso,
                               id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
                               unsigned width, unsigned height,
                               unsigned stride_words, unsigned axis)
{
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:in_buf  offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    const uint32_t p[4] = {width, height, stride_words, axis};
    [enc setBytes:p length:sizeof(p) atIndex:2];
    const MTLSize tg   = MTLSizeMake(CAMBI_METAL_BLOCK_X, CAMBI_METAL_BLOCK_Y, 1);
    const MTLSize grid = MTLSizeMake((width  + CAMBI_METAL_BLOCK_X - 1) / CAMBI_METAL_BLOCK_X,
                                     (height + CAMBI_METAL_BLOCK_Y - 1) / CAMBI_METAL_BLOCK_Y, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
}

/* Copy a sub-region of a Shared-storage MTLBuffer into a VmafPicture.
 * src_buf_ptr is [mtl_buf contents], stride_words is the flat uint16 stride. */
static void copy_metal_to_picture(const uint16_t *src_contents, VmafPicture *pic,
                                  unsigned width, unsigned height,
                                  unsigned stride_words)
{
    uint16_t *dst = (uint16_t *)pic->data[0];
    const ptrdiff_t dst_stride_words = (ptrdiff_t)(pic->stride[0] / sizeof(uint16_t));
    for (unsigned row = 0; row < height; row++) {
        const uint16_t *s = src_contents + (size_t)row * stride_words;
        uint16_t *d = dst + (size_t)row * (size_t)dst_stride_words;
        memcpy(d, s, width * sizeof(uint16_t));
    }
}

/* Copy a host VmafPicture luma plane into a Shared-storage MTLBuffer. */
static void copy_picture_to_metal(const VmafPicture *pic, uint16_t *dst_contents,
                                  unsigned width, unsigned height,
                                  unsigned stride_words)
{
    const uint16_t *src = (const uint16_t *)pic->data[0];
    const ptrdiff_t src_stride_words = (ptrdiff_t)(pic->stride[0] / sizeof(uint16_t));
    for (unsigned row = 0; row < height; row++) {
        const uint16_t *s = src + (size_t)row * (size_t)src_stride_words;
        uint16_t *d = dst_contents + (size_t)row * stride_words;
        memcpy(d, s, width * sizeof(uint16_t));
    }
}

/* ------------------------------------------------------------------ */
/* init_fex_metal                                                      */
/* ------------------------------------------------------------------ */
static int init_fex_metal(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                          unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    CambiStateMetal *s = (CambiStateMetal *)fex->priv;

    /* Resolve enc geometry. */
    if (s->enc_bitdepth == 0) { s->enc_bitdepth = (int)bpc; }
    if (s->enc_width == 0 || s->enc_height == 0) {
        s->enc_width  = (int)w;
        s->enc_height = (int)h;
    }
    if ((unsigned)s->enc_height > h || (unsigned)s->enc_width > w) {
        s->enc_width  = (int)w;
        s->enc_height = (int)h;
    }
    if (s->enc_width  < CAMBI_METAL_MIN_WIDTH_HEIGHT &&
        s->enc_height < CAMBI_METAL_MIN_WIDTH_HEIGHT) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "cambi_metal: encoded resolution %dx%d below minimum %d×%d.\n",
                 s->enc_width, s->enc_height,
                 CAMBI_METAL_MIN_WIDTH_HEIGHT, CAMBI_METAL_MIN_WIDTH_HEIGHT);
        return -EINVAL;
    }

    s->src_width    = w;
    s->src_height   = h;
    s->src_bpc      = bpc;
    s->proc_width   = (unsigned)s->enc_width;
    s->proc_height  = (unsigned)s->enc_height;
    s->adjusted_window = cambi_metal_adjust_window(s->window_size,
                                                   s->proc_width, s->proc_height);

    int err = vmaf_metal_context_new(&s->ctx, 0);
    if (err != 0) { return err; }

    err = vmaf_metal_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0) { goto fail_ctx; }

    {
        void *dev_handle = vmaf_metal_context_device_handle(s->ctx);
        if (dev_handle == NULL) { err = -ENODEV; goto fail_lc; }
        id<MTLDevice> device = (__bridge id<MTLDevice>)dev_handle;

        const size_t buf_bytes =
            (size_t)s->proc_width * s->proc_height * sizeof(uint16_t);

        err = cambi_metal_buf_alloc(&s->d_image, device, buf_bytes);
        if (err != 0) { goto fail_lc; }
        err = cambi_metal_buf_alloc(&s->d_mask, device, buf_bytes);
        if (err != 0) { goto fail_image; }
        err = cambi_metal_buf_alloc(&s->d_tmp, device, buf_bytes);
        if (err != 0) { goto fail_mask; }

        err = build_pipelines(s, device);
        if (err != 0) { goto fail_tmp; }
    }

    /* Host VmafPictures for CPU residual. */
    err = vmaf_picture_alloc(&s->pics[0], VMAF_PIX_FMT_YUV400P, 10,
                             s->proc_width, s->proc_height);
    if (err != 0) { goto fail_pso; }
    err = vmaf_picture_alloc(&s->pics[1], VMAF_PIX_FMT_YUV400P, 10,
                             s->proc_width, s->proc_height);
    if (err != 0) { goto fail_pic0; }

    /* Host scratch. */
    {
        const int num_diffs = 1 << s->max_log_contrast;
        s->buffers.diffs_to_consider = (uint16_t *)malloc(
            sizeof(uint16_t) * (size_t)num_diffs);
        if (!s->buffers.diffs_to_consider) { err = -ENOMEM; goto fail_pic1; }
        s->buffers.diff_weights = (int *)malloc(sizeof(int) * (size_t)num_diffs);
        if (!s->buffers.diff_weights) { err = -ENOMEM; goto fail_scratch; }
        s->buffers.all_diffs = (int *)malloc(sizeof(int) * (size_t)(2 * num_diffs + 1));
        if (!s->buffers.all_diffs) { err = -ENOMEM; goto fail_scratch; }

        static const int contrast_weights[32] = {
            1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
            8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
        for (int d = 0; d < num_diffs; d++) {
            s->buffers.diffs_to_consider[d] = (uint16_t)(d + 1);
            s->buffers.diff_weights[d] = contrast_weights[d];
        }
        for (int d = -num_diffs; d <= num_diffs; d++) {
            s->buffers.all_diffs[d + num_diffs] = d;
        }

        s->buffers.tvi_for_diff = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)num_diffs);
        if (!s->buffers.tvi_for_diff) { err = -ENOMEM; goto fail_scratch; }

        err = cambi_metal_init_tvi(s);
        if (err != 0) { goto fail_scratch; }

        s->buffers.c_values = (float *)malloc(
            sizeof(float) * s->proc_width * s->proc_height);
        if (!s->buffers.c_values) { err = -ENOMEM; goto fail_scratch; }

        const uint16_t num_bins = (uint16_t)(
            1024u + (unsigned)(s->buffers.all_diffs[2 * num_diffs] -
                               s->buffers.all_diffs[0]));
        s->buffers.c_values_histograms = (uint16_t *)malloc(
            sizeof(uint16_t) * s->proc_width * (size_t)num_bins);
        if (!s->buffers.c_values_histograms) { err = -ENOMEM; goto fail_scratch; }

        const int pad_size = CAMBI_METAL_MASK_FILTER_SIZE / 2;
        const int dp_width  = (int)s->proc_width + 2 * pad_size + 1;
        const int dp_height = 2 * pad_size + 2;
        s->buffers.mask_dp = (uint32_t *)malloc(
            sizeof(uint32_t) * (size_t)dp_width * (size_t)dp_height);
        if (!s->buffers.mask_dp) { err = -ENOMEM; goto fail_scratch; }

        s->buffers.filter_mode_buffer = (uint16_t *)malloc(
            sizeof(uint16_t) * 3u * s->proc_width);
        if (!s->buffers.filter_mode_buffer) { err = -ENOMEM; goto fail_scratch; }
        s->buffers.derivative_buffer = (uint16_t *)malloc(
            sizeof(uint16_t) * s->proc_width);
        if (!s->buffers.derivative_buffer) { err = -ENOMEM; goto fail_scratch; }
    }

    vmaf_cambi_default_callbacks(&s->inc_range_callback, &s->dec_range_callback,
                                 &s->derivative_callback);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (s->feature_name_dict == NULL) { err = -ENOMEM; goto fail_scratch; }

    return 0;

fail_scratch:
    free(s->buffers.diffs_to_consider);
    free(s->buffers.diff_weights);
    free(s->buffers.all_diffs);
    free(s->buffers.tvi_for_diff);
    free(s->buffers.c_values);
    free(s->buffers.c_values_histograms);
    free(s->buffers.mask_dp);
    free(s->buffers.filter_mode_buffer);
    free(s->buffers.derivative_buffer);
fail_pic1:
    (void)vmaf_picture_unref(&s->pics[1]);
fail_pic0:
    (void)vmaf_picture_unref(&s->pics[0]);
fail_pso:
    release_pipelines(s);
fail_tmp:
    cambi_metal_buf_free(&s->d_tmp);
fail_mask:
    cambi_metal_buf_free(&s->d_mask);
fail_image:
    cambi_metal_buf_free(&s->d_image);
fail_lc:
    (void)vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);
fail_ctx:
    vmaf_metal_context_destroy(s->ctx);
    s->ctx = NULL;
    return (err != 0) ? err : -ENOMEM;
}

/* ------------------------------------------------------------------ */
/* submit_fex_metal — synchronous per-scale GPU + CPU residual         */
/* (mirrors integer_cambi_cuda.c::submit_fex_cuda's per-scale loop)   */
/* ------------------------------------------------------------------ */
static int submit_fex_metal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    CambiStateMetal *s = (CambiStateMetal *)fex->priv;
    s->index = index;

    /* Step 1: host preprocessing → pics[0] (10-bit, proc_w × proc_h). */
    int err = vmaf_cambi_preprocessing(dist_pic, &s->pics[0],
                                       (int)s->proc_width, (int)s->proc_height,
                                       s->enc_bitdepth);
    if (err != 0) { return err; }

    void *dev_handle   = vmaf_metal_context_device_handle(s->ctx);
    void *queue_handle = vmaf_metal_context_queue_handle(s->ctx);
    if (dev_handle == NULL || queue_handle == NULL) { return -ENODEV; }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_handle;

    /* Retrieve bridge-retained pipeline states. */
    id<MTLComputePipelineState> pso_mask =
        (__bridge id<MTLComputePipelineState>)s->pso_mask;
    id<MTLComputePipelineState> pso_decimate =
        (__bridge id<MTLComputePipelineState>)s->pso_decimate;
    id<MTLComputePipelineState> pso_filter =
        (__bridge id<MTLComputePipelineState>)s->pso_filter_mode;

    /* Retrieve MTLBuffers. */
    id<MTLBuffer> image_buf = (__bridge id<MTLBuffer>)s->d_image.mtl_buf;
    id<MTLBuffer> mask_buf  = (__bridge id<MTLBuffer>)s->d_mask.mtl_buf;
    id<MTLBuffer> tmp_buf   = (__bridge id<MTLBuffer>)s->d_tmp.mtl_buf;

    /* Step 2: copy pics[0] luma → d_image (Shared memory — direct copy). */
    copy_picture_to_metal(&s->pics[0],
                          (uint16_t *)s->d_image.contents,
                          s->proc_width, s->proc_height, s->proc_width);

    /* Step 3: spatial mask at full scale. */
    {
        const unsigned mask_index = (unsigned)cambi_metal_get_mask_index(
            s->proc_width, s->proc_height, CAMBI_METAL_MASK_FILTER_SIZE);
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (cmd == nil) { return -ENOMEM; }
        encode_mask(cmd, pso_mask, image_buf, mask_buf,
                    s->proc_width, s->proc_height, s->proc_width, mask_index);
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    /* Step 4: per-scale GPU pipeline + CPU residual. */
    unsigned scaled_w = s->proc_width;
    unsigned scaled_h = s->proc_height;
    const int num_diffs = 1 << s->max_log_contrast;
    double scores_per_scale[CAMBI_METAL_NUM_SCALES] = {0.0, 0.0, 0.0, 0.0, 0.0};
    const double topk =
        (s->topk != CAMBI_METAL_DEFAULT_TOPK) ? s->topk : s->cambi_topk;

    /* Track which MTLBuffers are currently playing "image" and "mask" roles.
     * We swap between d_image and d_tmp for the decimate passes. */
    id<MTLBuffer> cur_image = image_buf;
    id<MTLBuffer> cur_mask  = mask_buf;
    void *cur_image_contents = s->d_image.contents;
    void *cur_mask_contents  = s->d_mask.contents;

    for (int scale = 0; scale < CAMBI_METAL_NUM_SCALES; scale++) {
        if (scale > 0) {
            const unsigned new_w = (scaled_w + 1u) >> 1;
            const unsigned new_h = (scaled_h + 1u) >> 1;

            /* GPU decimate cur_image → tmp_buf. */
            {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                if (cmd == nil) { return -ENOMEM; }
                encode_decimate(cmd, pso_decimate,
                                cur_image, tmp_buf,
                                new_w, new_h, scaled_w, new_w);
                [cmd commit];
                [cmd waitUntilCompleted];
            }
            /* Swap cur_image ↔ tmp. */
            {
                id<MTLBuffer> t = cur_image;
                void *tc = cur_image_contents;
                cur_image = tmp_buf;
                cur_image_contents = s->d_tmp.contents;
                tmp_buf = t;
                s->d_tmp.contents = tc; /* keep contents in sync */
            }

            /* GPU decimate cur_mask → tmp_buf. */
            {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                if (cmd == nil) { return -ENOMEM; }
                encode_decimate(cmd, pso_decimate,
                                cur_mask, tmp_buf,
                                new_w, new_h, scaled_w, new_w);
                [cmd commit];
                [cmd waitUntilCompleted];
            }
            /* Swap cur_mask ↔ tmp. */
            {
                id<MTLBuffer> t = cur_mask;
                void *tc = cur_mask_contents;
                cur_mask = tmp_buf;
                cur_mask_contents = s->d_tmp.contents;
                tmp_buf = t;
                s->d_tmp.contents = tc;
            }

            scaled_w = new_w;
            scaled_h = new_h;
        }

        /* GPU filter_mode H: cur_image → tmp_buf. */
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            if (cmd == nil) { return -ENOMEM; }
            encode_filter_mode(cmd, pso_filter,
                               cur_image, tmp_buf,
                               scaled_w, scaled_h, scaled_w, 0u);
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        /* GPU filter_mode V: tmp_buf → cur_image. */
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            if (cmd == nil) { return -ENOMEM; }
            encode_filter_mode(cmd, pso_filter,
                               tmp_buf, cur_image,
                               scaled_w, scaled_h, scaled_w, 1u);
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        /* Copy GPU results back to host VmafPictures (Shared mem — no DtoH latency). */
        copy_metal_to_picture((const uint16_t *)cur_image_contents,
                              &s->pics[0], scaled_w, scaled_h, scaled_w);
        copy_metal_to_picture((const uint16_t *)cur_mask_contents,
                              &s->pics[1], scaled_w, scaled_h, scaled_w);

        /* CPU residual. */
        vmaf_cambi_calculate_c_values(
            &s->pics[0], &s->pics[1],
            s->buffers.c_values, s->buffers.c_values_histograms,
            s->adjusted_window, (uint16_t)num_diffs,
            s->buffers.tvi_for_diff, s->vlt_luma,
            s->buffers.diff_weights, s->buffers.all_diffs,
            (int)scaled_w, (int)scaled_h,
            s->inc_range_callback, s->dec_range_callback);

        scores_per_scale[scale] =
            vmaf_cambi_spatial_pooling(s->buffers.c_values, topk, scaled_w, scaled_h);
    }

    /* Step 5: compute final score. */
    const uint16_t pixels_in_window =
        vmaf_cambi_get_pixels_in_window(s->adjusted_window);
    double score = vmaf_cambi_weight_scores_per_scale(scores_per_scale, pixels_in_window);
    if (score > s->cambi_max_val) { score = s->cambi_max_val; }
    if (score < 0.0)              { score = 0.0; }

    s->last_score = score;
    return 0;
}

/* ------------------------------------------------------------------ */
/* collect_fex_metal — emit pre-computed score from submit.           */
/* ------------------------------------------------------------------ */
static int collect_fex_metal(VmafFeatureExtractor *fex, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    const CambiStateMetal *s = (const CambiStateMetal *)fex->priv;
    return vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict,
        "Cambi_feature_cambi_score", s->last_score, index);
}

/* ------------------------------------------------------------------ */
/* close_fex_metal                                                     */
/* ------------------------------------------------------------------ */
static int close_fex_metal(VmafFeatureExtractor *fex)
{
    CambiStateMetal *s = (CambiStateMetal *)fex->priv;

    int rc = vmaf_metal_kernel_lifecycle_close(&s->lc, s->ctx);

    release_pipelines(s);
    cambi_metal_buf_free(&s->d_image);
    cambi_metal_buf_free(&s->d_mask);
    cambi_metal_buf_free(&s->d_tmp);

    (void)vmaf_picture_unref(&s->pics[0]);
    (void)vmaf_picture_unref(&s->pics[1]);

    free(s->buffers.c_values);
    free(s->buffers.c_values_histograms);
    free(s->buffers.mask_dp);
    free(s->buffers.filter_mode_buffer);
    free(s->buffers.derivative_buffer);
    free(s->buffers.diffs_to_consider);
    free(s->buffers.diff_weights);
    free(s->buffers.all_diffs);
    free(s->buffers.tvi_for_diff);

    if (s->feature_name_dict != NULL) {
        const int e = vmaf_dictionary_free(&s->feature_name_dict);
        if (e != 0 && rc == 0) { rc = e; }
    }
    if (s->ctx != NULL) {
        vmaf_metal_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {"Cambi_feature_cambi_score", NULL};

extern "C" {
// NOLINTNEXTLINE(misc-use-internal-linkage) — ADR-0360: public fex table entry
VmafFeatureExtractor vmaf_fex_cambi_metal = {
    .name    = "cambi_metal",
    .init    = init_fex_metal,
    .submit  = submit_fex_metal,
    .collect = collect_fex_metal,
    .close   = close_fex_metal,
    .options = options,
    .priv_size = sizeof(CambiStateMetal),
    .provided_features = provided_features,
    .flags   = 0,
    .chars   = {
        .n_dispatches_per_frame = 15, /* 5 scales × 3 kernels */
        .is_reduction_only      = false,
        .min_useful_frame_area  = 1920U * 1080U,
        .dispatch_hint          = VMAF_FEATURE_DISPATCH_DIRECT,
    },
};
} /* extern "C" */
