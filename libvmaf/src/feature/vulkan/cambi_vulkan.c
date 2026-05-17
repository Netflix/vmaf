/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  cambi feature kernel on the Vulkan backend (T7-36 / ADR-0205,
 *  Strategy II hybrid). Vulkan twin of the CPU `cambi` extractor in
 *  libvmaf/src/feature/cambi.c.
 *
 *  Per-frame pipeline (host orchestration):
 *
 *      1. Host: cambi_preprocessing (CPU path) — decimate/upcast to
 *         10-bit, optional anti-dither. Output is a 10-bit planar
 *         VmafPicture sized (enc_width, enc_height).
 *      2. Upload luma plane → image_buf (VmafVulkanBuffer).
 *      3. For scale = 0 .. NUM_SCALES-1:
 *           a. (scale > 0 || high_res_speedup): GPU decimate
 *              image_buf and mask_buf at 2× stride.
 *           b. (scale == 0): GPU derivative + SAT + threshold compare
 *              → mask_buf (the spatial mask is a one-time per-frame
 *              compute; subsequent scales decimate the mask alongside
 *              the image, matching the CPU path).
 *           c. GPU filter_mode (separable horizontal + vertical) over
 *              image_buf.
 *           d. Readback image_buf + mask_buf into the host
 *              VmafPicture pair (`pics[0]`, `pics[1]`).
 *           e. Host calculate_c_values + top-K spatial pooling — the
 *              precision-sensitive sliding-histogram phase that
 *              ADR-0205 §Decision keeps on the host for v1.
 *      4. Host: scale-weighted final score, MIN(score, cambi_max_val).
 *
 *  Precision contract — places=4 (set by ADR-0205 §Precision contract):
 *      Every GPU phase is integer + bit-exact w.r.t. the CPU. The
 *      readback into the host-allocated VmafPicture pair is byte-
 *      identical to what the CPU would have produced in-place. The
 *      host residual then runs the *exact* CPU code path against
 *      these buffers, so the emitted score is bit-identical to
 *      `vmaf_fex_cambi`. Cross-backend gate target: ULP=0.
 *
 *  Out of scope for v1 (see ADR-0205 §"Out of scope"):
 *      - Fully-on-GPU calculate_c_values (Strategy III).
 *      - GPU heatmap dump.
 *      - CUDA + SYCL twins (will follow per ADR-0192 cadence).
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"

#include "feature/cambi_internal.h"

#include "../../vulkan/kernel_template.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/vulkan_internal.h"

#include "cambi_decimate_spv.h"
#include "cambi_derivative_spv.h"
#include "cambi_filter_mode_spv.h"
#include "cambi_mask_dp_spv.h"
#include "cambi_preprocess_spv.h"

#define CAMBI_VK_PIC_BUFFERS 2
#define CAMBI_VK_DEFAULT_MAX_VAL 5.0
#define CAMBI_VK_DEFAULT_WINDOW_SIZE 63
#define CAMBI_VK_DEFAULT_TOPK 0.6
#define CAMBI_VK_DEFAULT_TVI 0.019
#define CAMBI_VK_DEFAULT_VLT 1000.0
#define CAMBI_VK_DEFAULT_MAX_LOG_CONTRAST 2
#define CAMBI_VK_DEFAULT_EOTF "bt1886"
#define CAMBI_VK_MIN_WIDTH_HEIGHT 64

#define CAMBI_VK_WG_X 16
#define CAMBI_VK_WG_Y 16

/* Pipeline IDs — index into the per-pipeline arrays in CambiVkState. */
enum CambiVkPipelineKind {
    CAMBI_PL_PREPROCESS = 0,
    CAMBI_PL_DERIVATIVE,
    CAMBI_PL_FILTER_MODE_H,
    CAMBI_PL_FILTER_MODE_V,
    CAMBI_PL_DECIMATE,
    CAMBI_PL_MASK_SAT_ROW,
    CAMBI_PL_MASK_SAT_COL,
    CAMBI_PL_MASK_THRESHOLD,
    CAMBI_PL_COUNT
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t in_stride_words;
    uint32_t out_stride_words;
} CambiVkPushTrivial;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t in_stride_words;
    uint32_t out_stride_words;
    uint32_t mask_index;
    uint32_t pad_size;
} CambiVkPushMaskDp;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t stride_words;
} CambiVkPushFilterMode;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t stride_words;
    uint32_t deriv_stride_words;
} CambiVkPushDerivative;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t in_stride_words;
    uint32_t out_stride_words;
} CambiVkPushDecimate;

typedef struct CambiVkState {
    /* Configuration (matches cambi.c options). */
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
    int cambi_high_res_speedup;
    int full_ref;        /* unused on Vulkan v1 (no full-ref path) */
    char *heatmaps_path; /* unused on Vulkan v1 */

    /* Resolved per-frame geometry. */
    unsigned src_width;
    unsigned src_height;
    unsigned src_bpc;
    unsigned proc_width;  /* enc_width post init */
    unsigned proc_height; /* enc_height post init */

    /* Window adjusted for actual frame size + anti-banding factor. */
    uint16_t adjusted_window;
    uint16_t vlt_luma;

    /* Vulkan plumbing. Five distinct pipeline shapes (per push-constant
     * struct size), each owning its own DSL + pipeline_layout + shader
     * module + descriptor pool via the kernel_template bundles. The
     * shape itself is `dsl_2bind` everywhere (in + out SSBO), but the
     * template owns one DSL handle per bundle. The first slot of
     * `pipelines[]` per stage aliases the bundle's base
     * `VkPipeline`; remaining slots (per-stage spec-constant variants
     * — FILTER_MODE_V, MASK_SAT_COL, MASK_THRESHOLD) are siblings via
     * `vmaf_vulkan_kernel_pipeline_add_variant()` and must be
     * destroyed *before* the bundle's `_destroy()` (see ADR-0221 +
     * libvmaf/src/vulkan/AGENTS.md "Multi-bundle kernels"). */
    VmafVulkanContext *ctx;
    int owns_ctx;
    VmafVulkanKernelPipeline pl_trivial;     /* CAMBI_PL_PREPROCESS */
    VmafVulkanKernelPipeline pl_derivative;  /* CAMBI_PL_DERIVATIVE */
    VmafVulkanKernelPipeline pl_filter_mode; /* CAMBI_PL_FILTER_MODE_H + variant V */
    VmafVulkanKernelPipeline pl_decimate;    /* CAMBI_PL_DECIMATE */
    VmafVulkanKernelPipeline pl_mask_dp;     /* CAMBI_PL_MASK_SAT_ROW + variants COL + THRESHOLD */
    VkPipeline pipelines[CAMBI_PL_COUNT];

    /* GPU buffers — sized for full-resolution scale 0. Per-scale
     * dispatches consume only the leading prefix. */
    VmafVulkanBuffer *raw_in_buf; /* host upload of source luma */
    VmafVulkanBuffer *image_buf;  /* preprocessed 10-bit image */
    VmafVulkanBuffer *mask_buf;
    VmafVulkanBuffer *deriv_buf;   /* uint16 0/1 per pixel */
    VmafVulkanBuffer *sat_row_buf; /* int32 SAT intermediate */
    VmafVulkanBuffer *sat_col_buf; /* int32 SAT */
    VmafVulkanBuffer *scratch_buf; /* alternate for filter_mode H pass */

    /* Host VmafPictures used as readback targets and as input to the
     * CPU residual (vmaf_cambi_calculate_c_values takes VmafPicture *). */
    VmafPicture pics[CAMBI_VK_PIC_BUFFERS];

    /* Host scratch for the residual. */
    VmafCambiHostBuffers buffers;

    /* Default callbacks (always scalar; the GPU has already done
     * the heavy lifting before we hit this stage). */
    VmafCambiRangeUpdater inc_range_callback;
    VmafCambiRangeUpdater dec_range_callback;
    VmafCambiDerivativeCalculator derivative_callback;

    /* Submit pool: 1 slot, reused for all sequential per-dispatch command buffers.
     * cambi_vk_run_record calls are strictly sequential (each waits before
     * returning), so slot 0 can be safely recycled for each call.
     * Eliminates per-dispatch vkAllocateCommandBuffers / vkCreateFence.
     * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
    VmafVulkanKernelSubmitPool sub_pool;

    VmafDictionary *feature_name_dict;
} CambiVkState;

static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(CambiVkState, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_VK_DEFAULT_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiVkState, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 180,
        .max = 7680,
        .alias = "encw",
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiVkState, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 150,
        .max = 7680,
        .alias = "ench",
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(CambiVkState, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
        .alias = "encbd",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 65 corresponds to ~1 degree at 4k",
        .offset = offsetof(CambiVkState, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_VK_DEFAULT_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiVkState, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_VK_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(CambiVkState, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_VK_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibility threshold ΔL < tvi_threshold * L_mean",
        .offset = offsetof(CambiVkState, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_VK_DEFAULT_TVI,
        .min = 0.0001,
        .max = 1.0,
        .alias = "tvit",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum log contrast (capped at 5)",
        .offset = offsetof(CambiVkState, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = CAMBI_VK_DEFAULT_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .alias = "mlc",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Visibility luminance threshold (cd/m²)",
        .offset = offsetof(CambiVkState, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = CAMBI_VK_DEFAULT_VLT,
        .min = 0.0,
        .max = 10000.0,
        .alias = "cvlt",
    },
    {
        .name = "eotf",
        .help = "EOTF for visibility-threshold conversion (bt1886 / pq)",
        .offset = offsetof(CambiVkState, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_VK_DEFAULT_EOTF,
    },
    {
        .name = "cambi_eotf",
        .help = "EOTF override for cambi (defaults to `eotf`)",
        .offset = offsetof(CambiVkState, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = CAMBI_VK_DEFAULT_EOTF,
        .alias = "ceotf",
    },
    {0},
};

/* ------------------------------------------------------------------ */
/*  Vulkan helpers — kernel_template-backed (ADR-0221).               */
/* ------------------------------------------------------------------ */

/* Build a sibling compute pipeline with up to 4 int32 spec constants
 * via `vmaf_vulkan_kernel_pipeline_add_variant`. The base bundle
 * supplies the layout + shader module; this helper formats the spec
 * payload and dispatches to the template. */
static int cambi_vk_build_variant(CambiVkState *s, const VmafVulkanKernelPipeline *bundle,
                                  int n_specs, const int32_t *spec_vals, VkPipeline *out)
{
    VkSpecializationMapEntry entries[4];
    int32_t data[4];
    for (int i = 0; i < n_specs; i++) {
        entries[i].constantID = (uint32_t)i;
        entries[i].offset = (uint32_t)(i * (int)sizeof(int32_t));
        entries[i].size = sizeof(int32_t);
        data[i] = spec_vals[i];
    }
    VkSpecializationInfo si = {
        .mapEntryCount = (uint32_t)n_specs,
        .pMapEntries = entries,
        .dataSize = (size_t)n_specs * sizeof(int32_t),
        .pData = data,
    };
    VkComputePipelineCreateInfo cpci = {
        .stage =
            {
                .pName = "main",
                .pSpecializationInfo = &si,
            },
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, bundle, &cpci, out);
}

/* Build the base pipeline of a `VmafVulkanKernelPipeline` bundle. The
 * template helper owns layout + shader + DSL + pool; the bundle's
 * `pipeline` field becomes the bundle's base pipeline. */
static int cambi_vk_build_base(CambiVkState *s, VmafVulkanKernelPipeline *bundle, uint32_t pc_size,
                               const uint32_t *spv, size_t spv_size, uint32_t max_sets, int n_specs,
                               const int32_t *spec_vals)
{
    VkSpecializationMapEntry entries[4];
    int32_t data[4];
    for (int i = 0; i < n_specs; i++) {
        entries[i].constantID = (uint32_t)i;
        entries[i].offset = (uint32_t)(i * (int)sizeof(int32_t));
        entries[i].size = sizeof(int32_t);
        data[i] = spec_vals[i];
    }
    VkSpecializationInfo si = {
        .mapEntryCount = (uint32_t)n_specs,
        .pMapEntries = entries,
        .dataSize = (size_t)n_specs * sizeof(int32_t),
        .pData = data,
    };
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 2,
        .push_constant_size = pc_size,
        .spv_bytes = spv,
        .spv_size = spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &si,
                    },
            },
        .max_descriptor_sets = max_sets,
    };
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, bundle);
}

static int cambi_vk_create_pipelines(CambiVkState *s)
{
    int err = 0;

    /* Per-bundle descriptor-pool sizing.
     *
     * Per frame, the cambi orchestration loop runs NUM_SCALES (=5)
     * iterations; each iteration may run:
     *   - decimate × 2 (image + mask), at scale > 0 || high-res-speedup
     *   - filter_mode × 2 (H + V)
     *   - mask_dp once at scale 0 (3 dispatches: row + col + threshold)
     *   - derivative once at scale 0
     *
     * Each bundle owns its own pool, so size each pool for the
     * worst case across all NUM_SCALES of its own dispatch type.
     * `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` (set by
     * the template) lets the per-frame free path recycle, but for
     * v1 we keep the legacy 64-set pessimistic budget split across
     * the bundles in proportion to their per-frame dispatch counts. */
    const uint32_t trivial_sets = 8;      /* preprocess unused on v1; keep small. */
    const uint32_t derivative_sets = 8;   /* once per frame. */
    const uint32_t filter_mode_sets = 32; /* 2 per scale × 5 scales × headroom. */
    const uint32_t decimate_sets = 32;    /* 2 per scale × 5 scales × headroom. */
    const uint32_t mask_dp_sets = 16;     /* 3 once per frame × headroom. */

    const int32_t W = (int32_t)s->proc_width;
    const int32_t H = (int32_t)s->proc_height;

    /* Trivial bundle (preprocess) — push: CambiVkPushTrivial.
     * Spec: W, H, src_bpc. Base pipeline = preprocess. */
    {
        const int32_t spec[3] = {W, H, (int32_t)s->src_bpc};
        err = cambi_vk_build_base(s, &s->pl_trivial, sizeof(CambiVkPushTrivial),
                                  cambi_preprocess_spv, cambi_preprocess_spv_size, trivial_sets,
                                  /*n_specs=*/3, spec);
        if (err)
            return err;
        s->pipelines[CAMBI_PL_PREPROCESS] = s->pl_trivial.pipeline;
    }

    /* Derivative bundle — push: CambiVkPushDerivative.
     * Spec: W, H. Base pipeline = derivative. */
    {
        const int32_t spec[2] = {W, H};
        err = cambi_vk_build_base(s, &s->pl_derivative, sizeof(CambiVkPushDerivative),
                                  cambi_derivative_spv, cambi_derivative_spv_size, derivative_sets,
                                  /*n_specs=*/2, spec);
        if (err)
            return err;
        s->pipelines[CAMBI_PL_DERIVATIVE] = s->pl_derivative.pipeline;
    }

    /* Filter-mode bundle — push: CambiVkPushFilterMode.
     * Spec: W, H, AXIS. Two pipelines from one shader module: H is
     * the bundle's base; V is a sibling variant. */
    {
        const int32_t spec_h[3] = {W, H, 0};
        err =
            cambi_vk_build_base(s, &s->pl_filter_mode, sizeof(CambiVkPushFilterMode),
                                cambi_filter_mode_spv, cambi_filter_mode_spv_size, filter_mode_sets,
                                /*n_specs=*/3, spec_h);
        if (err)
            return err;
        s->pipelines[CAMBI_PL_FILTER_MODE_H] = s->pl_filter_mode.pipeline;
        const int32_t spec_v[3] = {W, H, 1};
        err = cambi_vk_build_variant(s, &s->pl_filter_mode, 3, spec_v,
                                     &s->pipelines[CAMBI_PL_FILTER_MODE_V]);
        if (err)
            return err;
    }

    /* Decimate bundle — push: CambiVkPushDecimate.
     * Spec: W, H. Base pipeline = decimate. The shader bounds-checks
     * `gx >= pc.out_width`, so a single full-size specialisation
     * covers all scales. */
    {
        const int32_t spec[2] = {W, H};
        err = cambi_vk_build_base(s, &s->pl_decimate, sizeof(CambiVkPushDecimate),
                                  cambi_decimate_spv, cambi_decimate_spv_size, decimate_sets,
                                  /*n_specs=*/2, spec);
        if (err)
            return err;
        s->pipelines[CAMBI_PL_DECIMATE] = s->pl_decimate.pipeline;
    }

    /* Mask DP bundle — push: CambiVkPushMaskDp.
     * Spec: W, H, PASS, PAD_SIZE. Three pipelines from one shader
     * module: ROW (PASS=0) is the bundle's base; COL (PASS=1) and
     * THRESHOLD (PASS=2) are sibling variants. */
    {
        const int32_t pad = VMAF_CAMBI_MASK_FILTER_SIZE / 2;
        const int32_t spec_row[4] = {W, H, 0, pad};
        err = cambi_vk_build_base(s, &s->pl_mask_dp, sizeof(CambiVkPushMaskDp), cambi_mask_dp_spv,
                                  cambi_mask_dp_spv_size, mask_dp_sets,
                                  /*n_specs=*/4, spec_row);
        if (err)
            return err;
        s->pipelines[CAMBI_PL_MASK_SAT_ROW] = s->pl_mask_dp.pipeline;
        const int32_t spec_col[4] = {W, H, 1, pad};
        err = cambi_vk_build_variant(s, &s->pl_mask_dp, 4, spec_col,
                                     &s->pipelines[CAMBI_PL_MASK_SAT_COL]);
        if (err)
            return err;
        const int32_t spec_thr[4] = {W, H, 2, pad};
        err = cambi_vk_build_variant(s, &s->pl_mask_dp, 4, spec_thr,
                                     &s->pipelines[CAMBI_PL_MASK_THRESHOLD]);
        if (err)
            return err;
    }
    return 0;
}

static int cambi_vk_alloc_buffers(CambiVkState *s)
{
    const size_t W = s->proc_width;
    const size_t H = s->proc_height;
    /* Source upload buffer: bytes_per_pixel = 1 (8-bit) or 2 (>8-bit).
     * For the GPU read path, both layouts are word-packed; we pad to
     * 32-bit alignment. */
    const size_t src_words_per_row = (s->src_bpc <= 8) ? ((W + 3u) / 4u) : ((W + 1u) / 2u);
    const size_t raw_bytes = src_words_per_row * H * sizeof(uint32_t);
    /* 10-bit packed: 2 px per word. */
    const size_t img_words_per_row = (W + 1u) / 2u;
    const size_t img_bytes = img_words_per_row * H * sizeof(uint32_t);
    /* Mask + derivative: same 16-bit packing as image. */
    const size_t mask_bytes = img_bytes;
    const size_t deriv_bytes = img_bytes;
    /* SAT planes: int32 per pixel. */
    const size_t sat_bytes = W * H * sizeof(int32_t);

    int err = 0;
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->raw_in_buf, raw_bytes ? raw_bytes : 4);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->image_buf, img_bytes ? img_bytes : 4);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->mask_buf, mask_bytes ? mask_bytes : 4);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->deriv_buf, deriv_bytes ? deriv_bytes : 4);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->sat_row_buf, sat_bytes ? sat_bytes : 4);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->sat_col_buf, sat_bytes ? sat_bytes : 4);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->scratch_buf, img_bytes ? img_bytes : 4);
    return err ? -ENOMEM : 0;
}

static int cambi_vk_alloc_host(CambiVkState *s)
{
    const unsigned alloc_w = s->proc_width;
    const unsigned alloc_h = s->proc_height;

    int err = 0;
    for (unsigned i = 0; i < CAMBI_VK_PIC_BUFFERS; i++) {
        err |= vmaf_picture_alloc(&s->pics[i], VMAF_PIX_FMT_YUV400P, 10, alloc_w, alloc_h);
    }
    if (err)
        return err;

    const int num_diffs = 1 << s->max_log_contrast;

    /* diffs_to_consider, diff_weights, all_diffs — mirror cambi.c::set_contrast_arrays. */
    s->buffers.diffs_to_consider = malloc(sizeof(uint16_t) * num_diffs);
    s->buffers.diff_weights = malloc(sizeof(int) * num_diffs);
    s->buffers.all_diffs = malloc(sizeof(int) * (2 * num_diffs + 1));
    if (!s->buffers.diffs_to_consider || !s->buffers.diff_weights || !s->buffers.all_diffs)
        return -ENOMEM;
    /* The CPU file declares g_contrast_weights[32] internally; we re-derive
     * the same series here to avoid a second header export.  diffs are
     * powers of two: 1, 2, 4, ... up to (1 << max_log_contrast). */
    static const int contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                             8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
    for (int d = 0; d < num_diffs; d++) {
        s->buffers.diffs_to_consider[d] = (uint16_t)(d + 1);
        s->buffers.diff_weights[d] = contrast_weights[d];
    }
    for (int d = -num_diffs; d <= num_diffs; d++)
        s->buffers.all_diffs[d + num_diffs] = d;

    /* tvi_for_diff — derived in init(). For the residual we need it
     * shaped exactly like the CPU buffer; allocate and fill below. */
    s->buffers.tvi_for_diff = malloc(sizeof(uint16_t) * num_diffs);
    if (!s->buffers.tvi_for_diff)
        return -ENOMEM;

    s->buffers.c_values = malloc(sizeof(float) * alloc_w * alloc_h);
    if (!s->buffers.c_values)
        return -ENOMEM;

    const uint16_t num_bins =
        (uint16_t)(1024 + (s->buffers.all_diffs[2 * num_diffs] - s->buffers.all_diffs[0]));
    s->buffers.c_values_histograms = malloc(sizeof(uint16_t) * alloc_w * num_bins);
    if (!s->buffers.c_values_histograms)
        return -ENOMEM;

    const int pad_size = VMAF_CAMBI_MASK_FILTER_SIZE / 2;
    const int dp_width = (int)alloc_w + 2 * pad_size + 1;
    const int dp_height = 2 * pad_size + 2;
    s->buffers.mask_dp = malloc(sizeof(uint32_t) * (size_t)dp_width * (size_t)dp_height);
    if (!s->buffers.mask_dp)
        return -ENOMEM;

    s->buffers.filter_mode_buffer = malloc(sizeof(uint16_t) * 3u * alloc_w);
    if (!s->buffers.filter_mode_buffer)
        return -ENOMEM;
    s->buffers.derivative_buffer = malloc(sizeof(uint16_t) * alloc_w);
    if (!s->buffers.derivative_buffer)
        return -ENOMEM;

    return 0;
}

/* tvi_for_diff requires the luminance helpers from the CPU path. Rather
 * than re-export those, we let cambi.c export a small filler.  For v1
 * we replicate the CPU logic on the host using the public luminance
 * subset (vmaf_luminance_init_*). */
#include "luminance_tools.h"

static int cambi_vk_init_tvi(CambiVkState *s)
{
    VmafLumaRange luma_range;
    int err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err)
        return err;
    const char *effective_eotf;
    if (s->cambi_eotf && strcmp(s->cambi_eotf, CAMBI_VK_DEFAULT_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = s->eotf ? s->eotf : CAMBI_VK_DEFAULT_EOTF;
    }
    VmafEOTF eotf;
    err = vmaf_luminance_init_eotf(&eotf, effective_eotf);
    if (err)
        return err;

    const int num_diffs = 1 << s->max_log_contrast;
    /* Match cambi.c get_tvi_for_diff: bisect in [0, 1023] for the
     * largest sample whose tvi_condition holds, then offset by num_diffs
     * (the histogram base). The CPU file makes this static; we replicate
     * it inline here.  */
    for (int d = 0; d < num_diffs; d++) {
        const int diff = (int)s->buffers.diffs_to_consider[d];
        int lo = 0;
        int hi = (1 << 10) - 1 - diff;
        int found = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            double sample_lum = vmaf_luminance_get_luminance(mid, luma_range, eotf);
            double diff_lum =
                vmaf_luminance_get_luminance(mid + diff, luma_range, eotf) - sample_lum;
            if (diff_lum < s->tvi_threshold * sample_lum) {
                found = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (found < 0)
            found = 0;
        s->buffers.tvi_for_diff[d] = (uint16_t)(found + num_diffs);
    }

    /* vlt_luma — largest sample whose luminance is below
     * cambi_vis_lum_threshold. */
    int vlt = 0;
    for (int v = 0; v < (1 << 10); v++) {
        double L = vmaf_luminance_get_luminance(v, luma_range, eotf);
        if (L < s->cambi_vis_lum_threshold)
            vlt = v;
    }
    s->vlt_luma = (uint16_t)vlt;
    return 0;
}

/* Mirror cambi.c::adjust_window_size. */
static void cambi_vk_adjust_window(uint16_t *window_size, unsigned w, unsigned h, int high_res)
{
    /* CPU formula: window = window * (sqrt(w*h) / sqrt(3840*2160)) when
     * frame is below 4K, else passthrough. The high_res_speedup halves
     * the effective window for 1080p+ shortcuts. Replicating the
     * upstream behaviour verbatim. */
    (void)high_res;
    if (w >= 3840 && h >= 2160)
        return;
    double scale = sqrt((double)w * (double)h) / sqrt(3840.0 * 2160.0);
    int adjusted = (int)((double)*window_size * scale + 0.5);
    if (adjusted < 1)
        adjusted = 1;
    if (adjusted % 2 == 0)
        adjusted++;
    *window_size = (uint16_t)adjusted;
}

/* ------------------------------------------------------------------ */
/*  init / extract / close                                            */
/* ------------------------------------------------------------------ */

static int cambi_vk_init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    (void)pix_fmt;
    CambiVkState *s = fex->priv;

    if (s->enc_bitdepth == 0)
        s->enc_bitdepth = (int)bpc;
    if (s->enc_width == 0 || s->enc_height == 0) {
        s->enc_width = (int)w;
        s->enc_height = (int)h;
    }
    if ((unsigned)s->enc_height > h || (unsigned)s->enc_width > w) {
        s->enc_width = (int)w;
        s->enc_height = (int)h;
    }
    if (s->enc_width < CAMBI_VK_MIN_WIDTH_HEIGHT && s->enc_height < CAMBI_VK_MIN_WIDTH_HEIGHT)
        return -EINVAL;

    s->src_width = w;
    s->src_height = h;
    s->src_bpc = bpc;
    s->proc_width = (unsigned)s->enc_width;
    s->proc_height = (unsigned)s->enc_height;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "cambi_vulkan: cannot create Vulkan context (%d)\n",
                     err);
            return err;
        }
        s->owns_ctx = 1;
    }

    int err = cambi_vk_create_pipelines(s);
    if (err)
        return err;
    err = cambi_vk_alloc_buffers(s);
    if (err)
        return err;
    err = cambi_vk_alloc_host(s);
    if (err)
        return err;
    err = cambi_vk_init_tvi(s);
    if (err)
        return err;

    s->adjusted_window = (uint16_t)s->window_size;
    cambi_vk_adjust_window(&s->adjusted_window, s->proc_width, s->proc_height,
                           s->cambi_high_res_speedup);

    vmaf_cambi_default_callbacks(&s->inc_range_callback, &s->dec_range_callback,
                                 &s->derivative_callback);

    /* Pre-allocate submit pool (1 slot, reused sequentially for every
     * cambi_vk_run_record call within a frame). Eliminates per-dispatch
     * vkAllocateCommandBuffers + vkCreateFence overhead.
     * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;
}

static int cambi_vk_alloc_set(CambiVkState *s, const VmafVulkanKernelPipeline *bundle,
                              VkDescriptorSet *out)
{
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = bundle->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &bundle->dsl,
    };
    return vkAllocateDescriptorSets(s->ctx->device, &dsai, out) == VK_SUCCESS ? 0 : -ENOMEM;
}

static void cambi_vk_write_set(CambiVkState *s, VkDescriptorSet set, VmafVulkanBuffer *in_buf,
                               VmafVulkanBuffer *out_buf)
{
    VkDescriptorBufferInfo dbi[2] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(in_buf),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(out_buf),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[2];
    for (int i = 0; i < 2; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 2, writes, 0, NULL);
}

static void cambi_vk_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static uint16_t cambi_vk_ceil_log2(uint32_t num)
{
    if (num == 0)
        return 0;
    uint32_t tmp = num - 1u;
    uint16_t shift = 0;
    while (tmp > 0u) {
        tmp >>= 1;
        shift += 1;
    }
    return shift;
}

static uint16_t cambi_vk_get_mask_index(unsigned w, unsigned h, uint16_t filter_size)
{
    uint32_t shifted_wh = (w >> 6) * (h >> 6);
    return (uint16_t)((filter_size * filter_size + 3 * (cambi_vk_ceil_log2(shifted_wh) - 11) - 1) >>
                      1);
}

/* Submit + wait helper — thin wrapper kept for the call sites in
 * cambi_vk_run_record. Passes through to submit_end_and_wait, which
 * uses the pool's fence rather than creating one per call.
 * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
static int cambi_vk_submit_wait(CambiVkState *s, VmafVulkanKernelSubmit *sub)
{
    return vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, sub);
}

/* Upload source luma plane into raw_in_buf (word-packed). */
static int cambi_vk_upload_luma(CambiVkState *s, VmafPicture *pic)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(s->raw_in_buf);
    if (!dst)
        return -EIO;
    const unsigned W = s->src_width;
    const unsigned H = s->src_height;
    const unsigned bpc = s->src_bpc;
    if (bpc <= 8) {
        const size_t dst_words_per_row = (W + 3u) / 4u;
        const size_t dst_stride = dst_words_per_row * sizeof(uint32_t);
        for (unsigned y = 0; y < H; y++) {
            const uint8_t *src = (const uint8_t *)pic->data[0] + (size_t)y * pic->stride[0];
            uint8_t *drow = dst + (size_t)y * dst_stride;
            memset(drow, 0, dst_stride);
            memcpy(drow, src, W);
        }
    } else {
        const size_t dst_words_per_row = (W + 1u) / 2u;
        const size_t dst_stride = dst_words_per_row * sizeof(uint32_t);
        for (unsigned y = 0; y < H; y++) {
            const uint8_t *src = (const uint8_t *)pic->data[0] + (size_t)y * pic->stride[0];
            uint8_t *drow = dst + (size_t)y * dst_stride;
            memset(drow, 0, dst_stride);
            memcpy(drow, src, (size_t)W * 2u);
        }
    }
    return vmaf_vulkan_buffer_flush(s->ctx, s->raw_in_buf);
}

/* Read back image_buf into pics[0] luma plane (10-bit packed → uint16). */
static int cambi_vk_readback_image(CambiVkState *s, unsigned w, unsigned h)
{
    int err_inv_img = vmaf_vulkan_buffer_invalidate(s->ctx, s->image_buf);
    if (err_inv_img)
        return err_inv_img;
    const uint32_t *src = vmaf_vulkan_buffer_host(s->image_buf);
    uint16_t *dst = (uint16_t *)s->pics[0].data[0];
    const ptrdiff_t dst_stride = s->pics[0].stride[0] >> 1;
    const size_t img_words_per_row = (s->proc_width + 1u) / 2u;
    for (unsigned y = 0; y < h; y++) {
        const uint32_t *srow = src + y * img_words_per_row;
        uint16_t *drow = dst + (size_t)y * dst_stride;
        for (unsigned x = 0; x < w; x++) {
            uint32_t word = srow[x >> 1];
            drow[x] = (uint16_t)((word >> ((x & 1u) * 16u)) & 0xFFFFu);
        }
    }
    return 0;
}

/* Read back mask_buf into pics[1] luma plane. */
static int cambi_vk_readback_mask(CambiVkState *s, unsigned w, unsigned h)
{
    int err_inv_mask = vmaf_vulkan_buffer_invalidate(s->ctx, s->mask_buf);
    if (err_inv_mask)
        return err_inv_mask;
    const uint32_t *src = vmaf_vulkan_buffer_host(s->mask_buf);
    uint16_t *dst = (uint16_t *)s->pics[1].data[0];
    const ptrdiff_t dst_stride = s->pics[1].stride[0] >> 1;
    const size_t mask_words_per_row = (s->proc_width + 1u) / 2u;
    for (unsigned y = 0; y < h; y++) {
        const uint32_t *srow = src + y * mask_words_per_row;
        uint16_t *drow = dst + (size_t)y * dst_stride;
        for (unsigned x = 0; x < w; x++) {
            uint32_t word = srow[x >> 1];
            drow[x] = (uint16_t)((word >> ((x & 1u) * 16u)) & 0xFFFFu);
        }
    }
    return 0;
}

/* Upload pics[0] luma into image_buf (used at scale 0 init).  Useful
 * because cambi_preprocessing on the GPU shader is integer-bit-exact
 * with the CPU path only for the exact-resolution fast path; if the
 * source != enc resolution we let the CPU path do the resize and then
 * upload the post-preprocessed image. v1 ships only the CPU
 * preprocess to avoid the float bilinear-resize coordinate drift; the
 * GPU preprocess shader is a forward-compatible scaffold for the
 * exact-size fast path that a future ADR may switch on. */
static int cambi_vk_upload_image_pic(CambiVkState *s)
{
    uint32_t *dst = vmaf_vulkan_buffer_host(s->image_buf);
    if (!dst)
        return -EIO;
    const uint16_t *src = (const uint16_t *)s->pics[0].data[0];
    const ptrdiff_t src_stride = s->pics[0].stride[0] >> 1;
    const size_t img_words_per_row = (s->proc_width + 1u) / 2u;
    for (unsigned y = 0; y < s->proc_height; y++) {
        const uint16_t *srow = src + (size_t)y * src_stride;
        uint32_t *drow = dst + y * img_words_per_row;
        memset(drow, 0, img_words_per_row * sizeof(uint32_t));
        for (unsigned x = 0; x < s->proc_width; x++) {
            uint32_t shift = (x & 1u) * 16u;
            drow[x >> 1] |= (uint32_t)(srow[x] & 0xFFFFu) << shift;
        }
    }
    return vmaf_vulkan_buffer_flush(s->ctx, s->image_buf);
}

/* Dispatch derivative shader. */
static void cambi_vk_dispatch_derivative(CambiVkState *s, VkCommandBuffer cmd, unsigned w,
                                         unsigned h)
{
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (cambi_vk_alloc_set(s, &s->pl_derivative, &set))
        return;
    cambi_vk_write_set(s, set, s->image_buf, s->deriv_buf);
    CambiVkPushDerivative pc = {
        .width = w,
        .height = h,
        .stride_words = (uint32_t)((s->proc_width + 1u) / 2u),
        .deriv_stride_words = (uint32_t)((s->proc_width + 1u) / 2u),
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_derivative.pipeline_layout,
                            0, 1, &set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_derivative.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[CAMBI_PL_DERIVATIVE]);
    uint32_t gx = (w + CAMBI_VK_WG_X - 1u) / CAMBI_VK_WG_X;
    uint32_t gy = (h + CAMBI_VK_WG_Y - 1u) / CAMBI_VK_WG_Y;
    vkCmdDispatch(cmd, gx, gy, 1);
}

static void cambi_vk_dispatch_mask_dp(CambiVkState *s, VkCommandBuffer cmd, unsigned w, unsigned h)
{
    const uint16_t mask_index = cambi_vk_get_mask_index(w, h, VMAF_CAMBI_MASK_FILTER_SIZE);
    const uint32_t pad = VMAF_CAMBI_MASK_FILTER_SIZE / 2u;

    /* PASS 0 — row SAT: deriv_buf → sat_row_buf. */
    {
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (cambi_vk_alloc_set(s, &s->pl_mask_dp, &set))
            return;
        cambi_vk_write_set(s, set, s->deriv_buf, s->sat_row_buf);
        CambiVkPushMaskDp pc = {
            .width = w,
            .height = h,
            .in_stride_words = (uint32_t)((s->proc_width + 1u) / 2u),
            .out_stride_words = (uint32_t)s->proc_width,
            .mask_index = mask_index,
            .pad_size = pad,
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_mask_dp.pipeline_layout,
                                0, 1, &set, 0, NULL);
        vkCmdPushConstants(cmd, s->pl_mask_dp.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pc), &pc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[CAMBI_PL_MASK_SAT_ROW]);
        vkCmdDispatch(cmd, h, 1, 1);
    }
    cambi_vk_barrier(cmd);

    /* PASS 1 — col SAT: sat_row_buf → sat_col_buf. */
    {
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (cambi_vk_alloc_set(s, &s->pl_mask_dp, &set))
            return;
        cambi_vk_write_set(s, set, s->sat_row_buf, s->sat_col_buf);
        CambiVkPushMaskDp pc = {
            .width = w,
            .height = h,
            .in_stride_words = (uint32_t)s->proc_width,
            .out_stride_words = (uint32_t)s->proc_width,
            .mask_index = mask_index,
            .pad_size = pad,
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_mask_dp.pipeline_layout,
                                0, 1, &set, 0, NULL);
        vkCmdPushConstants(cmd, s->pl_mask_dp.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pc), &pc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[CAMBI_PL_MASK_SAT_COL]);
        vkCmdDispatch(cmd, w, 1, 1);
    }
    cambi_vk_barrier(cmd);

    /* PASS 2 — threshold compare: sat_col_buf → mask_buf. */
    {
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (cambi_vk_alloc_set(s, &s->pl_mask_dp, &set))
            return;
        cambi_vk_write_set(s, set, s->sat_col_buf, s->mask_buf);
        CambiVkPushMaskDp pc = {
            .width = w,
            .height = h,
            .in_stride_words = (uint32_t)s->proc_width,
            .out_stride_words = (uint32_t)((s->proc_width + 1u) / 2u),
            .mask_index = mask_index,
            .pad_size = pad,
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_mask_dp.pipeline_layout,
                                0, 1, &set, 0, NULL);
        vkCmdPushConstants(cmd, s->pl_mask_dp.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pc), &pc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          s->pipelines[CAMBI_PL_MASK_THRESHOLD]);
        uint32_t gx = (w + CAMBI_VK_WG_X - 1u) / CAMBI_VK_WG_X;
        uint32_t gy = (h + CAMBI_VK_WG_Y - 1u) / CAMBI_VK_WG_Y;
        vkCmdDispatch(cmd, gx, gy, 1);
    }
}

/* Decimate one buffer (in-place semantics on host; we use scratch_buf
 * as the output and then memcpy host-side via a readback round-trip
 * for v1 simplicity. Cleaner on-GPU in-place would need a temp; we
 * keep the explicit copy for correctness). */
static void cambi_vk_dispatch_decimate(CambiVkState *s, VkCommandBuffer cmd,
                                       VmafVulkanBuffer *in_buf, VmafVulkanBuffer *out_buf,
                                       unsigned out_w, unsigned out_h, unsigned in_w)
{
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (cambi_vk_alloc_set(s, &s->pl_decimate, &set))
        return;
    cambi_vk_write_set(s, set, in_buf, out_buf);
    CambiVkPushDecimate pc = {
        .width = out_w,
        .height = out_h,
        .in_stride_words = (uint32_t)((in_w + 1u) / 2u),
        .out_stride_words = (uint32_t)((s->proc_width + 1u) / 2u),
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_decimate.pipeline_layout, 0,
                            1, &set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_decimate.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[CAMBI_PL_DECIMATE]);
    uint32_t gx = (out_w + CAMBI_VK_WG_X - 1u) / CAMBI_VK_WG_X;
    uint32_t gy = (out_h + CAMBI_VK_WG_Y - 1u) / CAMBI_VK_WG_Y;
    vkCmdDispatch(cmd, gx, gy, 1);
}

static void cambi_vk_dispatch_filter_mode(CambiVkState *s, VkCommandBuffer cmd,
                                          VmafVulkanBuffer *in_buf, VmafVulkanBuffer *out_buf,
                                          unsigned w, unsigned h, int axis)
{
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (cambi_vk_alloc_set(s, &s->pl_filter_mode, &set))
        return;
    cambi_vk_write_set(s, set, in_buf, out_buf);
    CambiVkPushFilterMode pc = {
        .width = w,
        .height = h,
        .stride_words = (uint32_t)((s->proc_width + 1u) / 2u),
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_filter_mode.pipeline_layout,
                            0, 1, &set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_filter_mode.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      s->pipelines[axis == 0 ? CAMBI_PL_FILTER_MODE_H : CAMBI_PL_FILTER_MODE_V]);
    uint32_t gx = (w + CAMBI_VK_WG_X - 1u) / CAMBI_VK_WG_X;
    uint32_t gy = (h + CAMBI_VK_WG_Y - 1u) / CAMBI_VK_WG_Y;
    vkCmdDispatch(cmd, gx, gy, 1);
}

/* One-shot command-buffer helper: record + submit + wait.
 * Uses slot 0 of the pre-allocated submit pool — each call acquires,
 * records, ends, and waits synchronously, so slot 0 is guaranteed
 * signalled before the next acquire. Eliminates per-call
 * vkAllocateCommandBuffers + vkCreateFence + vkFreeCommandBuffers +
 * vkDestroyFence. (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
static int cambi_vk_run_record(CambiVkState *s, void (*record)(CambiVkState *, VkCommandBuffer))
{
    VmafVulkanKernelSubmit sub = {0};
    int err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &sub);
    if (err)
        return err;
    record(s, sub.cmd);
    err = cambi_vk_submit_wait(s, &sub);
    vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
    return err;
}

/* Recorder closure state — rather than a true closure we stash the
 * required fields on the state struct between record_xxx invocations. */
static struct {
    unsigned w;
    unsigned h;
    unsigned in_w;
    int axis;
    VmafVulkanBuffer *in_buf;
    VmafVulkanBuffer *out_buf;
} g_cambi_vk_rec;

static void cambi_vk_rec_mask_dp(CambiVkState *s, VkCommandBuffer cmd)
{
    cambi_vk_dispatch_derivative(s, cmd, g_cambi_vk_rec.w, g_cambi_vk_rec.h);
    cambi_vk_barrier(cmd);
    cambi_vk_dispatch_mask_dp(s, cmd, g_cambi_vk_rec.w, g_cambi_vk_rec.h);
}

static void cambi_vk_rec_decimate(CambiVkState *s, VkCommandBuffer cmd)
{
    cambi_vk_dispatch_decimate(s, cmd, g_cambi_vk_rec.in_buf, g_cambi_vk_rec.out_buf,
                               g_cambi_vk_rec.w, g_cambi_vk_rec.h, g_cambi_vk_rec.in_w);
}

static void cambi_vk_rec_filter_mode(CambiVkState *s, VkCommandBuffer cmd)
{
    cambi_vk_dispatch_filter_mode(s, cmd, g_cambi_vk_rec.in_buf, g_cambi_vk_rec.out_buf,
                                  g_cambi_vk_rec.w, g_cambi_vk_rec.h, g_cambi_vk_rec.axis);
}

/* Per-frame extract. Mirrors cambi.c::cambi_score and friends but with
 * the GPU servicing the embarrassingly parallel phases.
 * The per-scale loop replicates the CPU extract step-by-step so any
 * reviewer can diff against cambi.c::cambi_score; splitting would
 * obscure that 1:1 correspondence (ADR-0141 §2 upstream-parity
 * load-bearing invariant; T7-5 sweep closeout — ADR-0278). */
/* NOLINTNEXTLINE(readability-function-size,google-readability-function-size) */
static int cambi_vk_extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    CambiVkState *s = fex->priv;
    int err = 0;

    /* Step 1: host-side preprocess (CPU path) into pics[0]. The GPU
     * preprocess shader exists as a forward-compatible scaffold but
     * isn't used for v1: matching the CPU bilinear-resize coordinate
     * arithmetic on integer rounding boundaries needs more validation
     * than v1's scope allows. The post-preprocess image is a 10-bit
     * planar buffer; v1 uploads it into image_buf in step 2. */
    err = vmaf_cambi_preprocessing(dist_pic, &s->pics[0], (int)s->proc_width, (int)s->proc_height,
                                   s->enc_bitdepth);
    if (err)
        return err;

    /* Step 2: upload preprocessed image to GPU. */
    err = cambi_vk_upload_image_pic(s);
    if (err)
        return err;

    /* Step 3: per-scale GPU + host residual loop. */
    double scores_per_scale[VMAF_CAMBI_NUM_SCALES] = {0};
    unsigned scaled_w = s->proc_width;
    unsigned scaled_h = s->proc_height;
    const int num_diffs = 1 << s->max_log_contrast;

    /* Compute the spatial mask once at scale 0 (matches CPU). */
    g_cambi_vk_rec.w = scaled_w;
    g_cambi_vk_rec.h = scaled_h;
    err = cambi_vk_run_record(s, cambi_vk_rec_mask_dp);
    if (err)
        return err;

    for (unsigned scale = 0; scale < VMAF_CAMBI_NUM_SCALES; scale++) {
        if (scale > 0 || s->cambi_high_res_speedup) {
            unsigned new_w = (scaled_w + 1u) >> 1;
            unsigned new_h = (scaled_h + 1u) >> 1;

            /* Decimate image_buf → scratch_buf, then memcpy back. */
            g_cambi_vk_rec.w = new_w;
            g_cambi_vk_rec.h = new_h;
            g_cambi_vk_rec.in_w = scaled_w;
            g_cambi_vk_rec.in_buf = s->image_buf;
            g_cambi_vk_rec.out_buf = s->scratch_buf;
            err = cambi_vk_run_record(s, cambi_vk_rec_decimate);
            if (err)
                return err;
            /* Copy scratch_buf → image_buf (host memcpy through mapped
             * pointers; cheap relative to GPU dispatch). */
            {
                const size_t img_words_per_row = (s->proc_width + 1u) / 2u;
                int err_inv_scratch = vmaf_vulkan_buffer_invalidate(s->ctx, s->scratch_buf);
                if (err_inv_scratch)
                    return err_inv_scratch;
                const uint32_t *src = vmaf_vulkan_buffer_host(s->scratch_buf);
                uint32_t *dst = vmaf_vulkan_buffer_host(s->image_buf);
                for (unsigned y = 0; y < new_h; y++) {
                    memcpy(dst + (size_t)y * img_words_per_row, src + (size_t)y * img_words_per_row,
                           ((new_w + 1u) / 2u) * sizeof(uint32_t));
                }
                vmaf_vulkan_buffer_flush(s->ctx, s->image_buf);
            }

            /* Same for mask_buf → scratch_buf → mask_buf. */
            g_cambi_vk_rec.in_buf = s->mask_buf;
            g_cambi_vk_rec.out_buf = s->scratch_buf;
            err = cambi_vk_run_record(s, cambi_vk_rec_decimate);
            if (err)
                return err;
            {
                const size_t img_words_per_row = (s->proc_width + 1u) / 2u;
                int err_inv_scratch = vmaf_vulkan_buffer_invalidate(s->ctx, s->scratch_buf);
                if (err_inv_scratch)
                    return err_inv_scratch;
                const uint32_t *src = vmaf_vulkan_buffer_host(s->scratch_buf);
                uint32_t *dst = vmaf_vulkan_buffer_host(s->mask_buf);
                for (unsigned y = 0; y < new_h; y++) {
                    memcpy(dst + (size_t)y * img_words_per_row, src + (size_t)y * img_words_per_row,
                           ((new_w + 1u) / 2u) * sizeof(uint32_t));
                }
                vmaf_vulkan_buffer_flush(s->ctx, s->mask_buf);
            }

            scaled_w = new_w;
            scaled_h = new_h;
        }

        /* Filter-mode horizontal: image_buf → scratch_buf. */
        g_cambi_vk_rec.w = scaled_w;
        g_cambi_vk_rec.h = scaled_h;
        g_cambi_vk_rec.in_buf = s->image_buf;
        g_cambi_vk_rec.out_buf = s->scratch_buf;
        g_cambi_vk_rec.axis = 0;
        err = cambi_vk_run_record(s, cambi_vk_rec_filter_mode);
        if (err)
            return err;
        /* Filter-mode vertical: scratch_buf → image_buf. */
        g_cambi_vk_rec.in_buf = s->scratch_buf;
        g_cambi_vk_rec.out_buf = s->image_buf;
        g_cambi_vk_rec.axis = 1;
        err = cambi_vk_run_record(s, cambi_vk_rec_filter_mode);
        if (err)
            return err;

        /* Read back to host VmafPicture pair. */
        err = cambi_vk_readback_image(s, scaled_w, scaled_h);
        if (err)
            return err;
        err = cambi_vk_readback_mask(s, scaled_w, scaled_h);
        if (err)
            return err;

        /* Host residual: calculate_c_values + spatial pooling. */
        vmaf_cambi_calculate_c_values(&s->pics[0], &s->pics[1], s->buffers.c_values,
                                      s->buffers.c_values_histograms, s->adjusted_window,
                                      (uint16_t)num_diffs, s->buffers.tvi_for_diff, s->vlt_luma,
                                      s->buffers.diff_weights, s->buffers.all_diffs, (int)scaled_w,
                                      (int)scaled_h, s->inc_range_callback, s->dec_range_callback);

        scores_per_scale[scale] =
            vmaf_cambi_spatial_pooling(s->buffers.c_values, s->topk, scaled_w, scaled_h);
    }

    uint16_t pixels_in_window = vmaf_cambi_get_pixels_in_window(s->adjusted_window);
    double score = vmaf_cambi_weight_scores_per_scale(scores_per_scale, pixels_in_window);
    if (score > s->cambi_max_val)
        score = s->cambi_max_val;
    if (score < 0.0)
        score = 0.0;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "Cambi_feature_cambi_score", score, index);
}

static int cambi_vk_close(VmafFeatureExtractor *fex)
{
    CambiVkState *s = fex->priv;
    if (!s || !s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;
    vkDeviceWaitIdle(dev);

    /* Destroy submit pool BEFORE pipelines (ADR-0256 / ADR-0354 ordering
     * rule: pool must be torn down before any pipeline + pool resource
     * it references is freed). */
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);

    /* Variant pipelines must be destroyed *before* the bundle's
     * `_destroy()` (which destroys the shared layout/shader/DSL/pool
     * and would invalidate the variants). The base pipeline at the
     * stage's primary slot aliases the bundle's `pipeline` field —
     * skip those slots to avoid double-freeing the aliased base via
     * the bundle teardown. */
    if (s->pipelines[CAMBI_PL_FILTER_MODE_V])
        vkDestroyPipeline(dev, s->pipelines[CAMBI_PL_FILTER_MODE_V], NULL);
    if (s->pipelines[CAMBI_PL_MASK_SAT_COL])
        vkDestroyPipeline(dev, s->pipelines[CAMBI_PL_MASK_SAT_COL], NULL);
    if (s->pipelines[CAMBI_PL_MASK_THRESHOLD])
        vkDestroyPipeline(dev, s->pipelines[CAMBI_PL_MASK_THRESHOLD], NULL);

    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_trivial);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_derivative);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_filter_mode);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_decimate);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_mask_dp);

#define CAMBI_VK_FREE(b)                                                                           \
    do {                                                                                           \
        if (s->b)                                                                                  \
            vmaf_vulkan_buffer_free(s->ctx, s->b);                                                 \
    } while (0)
    CAMBI_VK_FREE(raw_in_buf);
    CAMBI_VK_FREE(image_buf);
    CAMBI_VK_FREE(mask_buf);
    CAMBI_VK_FREE(deriv_buf);
    CAMBI_VK_FREE(sat_row_buf);
    CAMBI_VK_FREE(sat_col_buf);
    CAMBI_VK_FREE(scratch_buf);
#undef CAMBI_VK_FREE

    for (unsigned i = 0; i < CAMBI_VK_PIC_BUFFERS; i++)
        (void)vmaf_picture_unref(&s->pics[i]);

    free(s->buffers.c_values);
    free(s->buffers.mask_dp);
    free(s->buffers.c_values_histograms);
    free(s->buffers.filter_mode_buffer);
    free(s->buffers.diffs_to_consider);
    free(s->buffers.tvi_for_diff);
    free(s->buffers.derivative_buffer);
    free(s->buffers.diff_weights);
    free(s->buffers.all_diffs);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *cambi_vk_provided_features[] = {"Cambi_feature_cambi_score", NULL};

/* External linkage required — the registry iterates over `vmaf_fex_*` externs.
 * NOLINTNEXTLINE(misc-use-internal-linkage,cppcoreguidelines-avoid-non-const-global-variables) */
VmafFeatureExtractor vmaf_fex_cambi_vulkan = {
    .name = "cambi_vulkan",
    .init = cambi_vk_init,
    .extract = cambi_vk_extract,
    .options = options,
    .close = cambi_vk_close,
    .priv_size = sizeof(CambiVkState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = cambi_vk_provided_features,
    .chars =
        {
            /* ~6 dispatches per scale (decimate ×2 + filter ×2 + mask-dp + derivative
             * once at scale 0); 5 scales = ~32 dispatches/frame. Heavy-ish. */
            .n_dispatches_per_frame = 32,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
