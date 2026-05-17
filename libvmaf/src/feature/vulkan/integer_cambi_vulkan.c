/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CAMBI banding-detection feature extractor on the Vulkan backend —
 *  integer variant. Vulkan twin of integer_cambi_cuda.c (ADR-0360).
 *
 *  Strategy II hybrid (mirrors cambi_vulkan.c / ADR-0205):
 *
 *    GPU stages (three compute shaders):
 *      - cambi_spatial_mask.comp: fused derivative + 7x7 box-sum +
 *        threshold in a single pass. Output: uint16 mask buffer (0/1).
 *      - cambi_decimate.comp (reused from legacy cambi_vulkan): strict
 *        2x stride-2 subsample — bit-exact w.r.t. CPU decimate().
 *      - cambi_filter_mode.comp (reused): separable 3-tap mode filter.
 *
 *    Host CPU stages (exact CPU code via cambi_internal.h wrappers):
 *      - vmaf_cambi_preprocessing: decimate/upcast to 10-bit.
 *      - vmaf_cambi_calculate_c_values: sliding-histogram c-value pass.
 *      - vmaf_cambi_spatial_pooling: top-K pooling -> per-scale score.
 *      - vmaf_cambi_weight_scores_per_scale: inner-product scale weights.
 *
 *  Per-frame flow:
 *    1. Host preprocessing (CPU): resize/upcast dist_pic -> pics[0].
 *    2. Upload pics[0] luma plane -> image_buf (host-mapped VmafVulkanBuffer).
 *    3. Scale 0: GPU spatial_mask over image_buf -> mask_buf.
 *    4. For scale = 0 .. NUM_SCALES-1:
 *         a. (scale > 0) GPU decimate image_buf -> scratch_buf, swap;
 *                        GPU decimate mask_buf  -> scratch_buf, swap.
 *         b. GPU filter_mode H: image_buf -> scratch_buf.
 *            GPU filter_mode V: scratch_buf -> image_buf.
 *         c. Readback image_buf -> pics[0], mask_buf -> pics[1].
 *         d. Host vmaf_cambi_calculate_c_values + vmaf_cambi_spatial_pooling.
 *    5. Host vmaf_cambi_weight_scores_per_scale -> final score.
 *    6. Emit "Cambi_feature_cambi_score" into the feature collector.
 *
 *  Precision contract: places=4 (ULP=0 on the emitted score). All GPU
 *  phases are integer + bit-exact. The host residual runs exact CPU code
 *  (cambi_internal.h), so the emitted score is bit-for-bit identical to
 *  vmaf_fex_cambi. Cross-backend gate target: ULP=0.
 *
 *  Key difference from legacy cambi_vulkan.c: the fused spatial-mask
 *  shader collapses derivative + SAT + threshold into one pass, removing
 *  the SAT intermediate buffers and four pipeline shapes.
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
#include "luminance_tools.h"
#include "picture.h"

#include "feature/cambi_internal.h"

#include "../../vulkan/kernel_template.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/vulkan_internal.h"

#include "cambi_decimate_spv.h"
#include "cambi_filter_mode_spv.h"
#include "cambi_spatial_mask_spv.h"

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */
#define ICAMBI_VK_NUM_SCALES 5
#define ICAMBI_VK_MIN_WIDTH_HEIGHT 216
#define ICAMBI_VK_MASK_FILTER_SIZE 7
#define ICAMBI_VK_DEFAULT_MAX_VAL 1000.0
#define ICAMBI_VK_DEFAULT_WINDOW_SIZE 65
#define ICAMBI_VK_DEFAULT_TOPK 0.6
#define ICAMBI_VK_DEFAULT_TVI 0.019
#define ICAMBI_VK_DEFAULT_VLT 0.0
#define ICAMBI_VK_DEFAULT_MAX_LOG_CONTRAST 2
#define ICAMBI_VK_DEFAULT_EOTF "bt1886"
#define ICAMBI_VK_WG_X 16u
#define ICAMBI_VK_WG_Y 16u
/* Descriptor sets budget: 2 per dispatch x 2 (image+mask) x 5 scales x headroom. */
#define ICAMBI_VK_MASK_DESC_SETS 8u
#define ICAMBI_VK_FM_DESC_SETS 48u
#define ICAMBI_VK_DEC_DESC_SETS 48u

/* ------------------------------------------------------------------ */
/* Push-constant structs (must match .comp push_constant blocks)      */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t stride_words;
    uint32_t mask_index;
} ICambiVkPushSpatialMask;

typedef struct {
    uint32_t out_width;
    uint32_t out_height;
    uint32_t in_stride_words;
    uint32_t out_stride_words;
} ICambiVkPushDecimate;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t stride_words;
} ICambiVkPushFilterMode;

/* ------------------------------------------------------------------ */
/* Pipeline IDs                                                        */
/* ------------------------------------------------------------------ */
enum ICambiVkPipelineKind {
    ICAMBI_PL_SPATIAL_MASK = 0,
    ICAMBI_PL_FILTER_MODE_H = 1,
    ICAMBI_PL_FILTER_MODE_V = 2,
    ICAMBI_PL_DECIMATE = 3,
    ICAMBI_PL_COUNT = 4,
};

/* ------------------------------------------------------------------ */
/* State struct                                                        */
/* ------------------------------------------------------------------ */
typedef struct ICambiVkState {
    /* Configuration options (mirrors CambiStateCuda). */
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
    int cambi_high_res_speedup; /* reserved; v1 ignores it */

    /* Resolved per-frame geometry. */
    unsigned src_width;
    unsigned src_height;
    unsigned src_bpc;
    unsigned proc_width;
    unsigned proc_height;

    /* Derived. */
    uint16_t adjusted_window;
    uint16_t vlt_luma;

    /* Vulkan context. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Three pipeline bundles. */
    VmafVulkanKernelPipeline pl_spatial_mask;
    VmafVulkanKernelPipeline pl_filter_mode;
    VmafVulkanKernelPipeline pl_decimate;
    VkPipeline pipelines[ICAMBI_PL_COUNT];

    /* GPU buffers (host-mapped coherent for HtoD/DtoH convenience). */
    VmafVulkanBuffer *image_buf;
    VmafVulkanBuffer *mask_buf;
    VmafVulkanBuffer *scratch_buf;

    /* Submit pool (slot 0 reused sequentially per dispatch). */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Host VmafPictures for the CPU residual readback. */
    VmafPicture pics[2]; /* pics[0]=image, pics[1]=mask */

    /* Host scratch for the CPU residual. */
    VmafCambiHostBuffers buffers;

    /* Default callbacks (always scalar; GPU handles the heavy lifting). */
    VmafCambiRangeUpdater inc_range_callback;
    VmafCambiRangeUpdater dec_range_callback;
    VmafCambiDerivativeCalculator derivative_callback;

    VmafDictionary *feature_name_dict;
} ICambiVkState;

/* ------------------------------------------------------------------ */
/* Options                                                             */
/* ------------------------------------------------------------------ */
static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped",
        .offset = offsetof(ICambiVkState, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = ICAMBI_VK_DEFAULT_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(ICambiVkState, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 180,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encw",
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(ICambiVkState, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 150,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ench",
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(ICambiVkState, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encbd",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 65 corresponds to ~1 degree at 4k",
        .offset = offsetof(ICambiVkState, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = ICAMBI_VK_DEFAULT_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(ICambiVkState, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = ICAMBI_VK_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation",
        .offset = offsetof(ICambiVkState, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = ICAMBI_VK_DEFAULT_TOPK,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibility threshold DeltaL < tvi_threshold * L_mean",
        .offset = offsetof(ICambiVkState, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = ICAMBI_VK_DEFAULT_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which banding is assumed invisible",
        .offset = offsetof(ICambiVkState, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = ICAMBI_VK_DEFAULT_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum log contrast (0 to 5, default 2)",
        .offset = offsetof(ICambiVkState, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = ICAMBI_VK_DEFAULT_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "eotf",
        .help = "EOTF for visibility-threshold conversion (bt1886 / pq)",
        .offset = offsetof(ICambiVkState, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = ICAMBI_VK_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "EOTF override for cambi (defaults to eotf)",
        .offset = offsetof(ICambiVkState, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = ICAMBI_VK_DEFAULT_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {0},
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
static uint16_t icambi_vk_adjust_window(int window_size, unsigned w, unsigned h)
{
    unsigned adjusted = (unsigned)(window_size) * (w + h) / 375u;
    adjusted >>= 4;
    if (adjusted < 1u)
        adjusted = 1u;
    if ((adjusted & 1u) == 0u)
        adjusted++;
    return (uint16_t)adjusted;
}

static uint16_t icambi_vk_ceil_log2(uint32_t num)
{
    if (num == 0u)
        return 0u;
    uint32_t tmp = num - 1u;
    uint16_t shift = 0;
    while (tmp > 0u) {
        tmp >>= 1;
        shift++;
    }
    return shift;
}

static uint16_t icambi_vk_get_mask_index(unsigned w, unsigned h, uint16_t filter_size)
{
    uint32_t shifted_wh = (w >> 6) * (h >> 6);
    return (
        uint16_t)((filter_size * filter_size + 3 * (icambi_vk_ceil_log2(shifted_wh) - 11) - 1) >>
                  1);
}

static int icambi_vk_init_tvi(ICambiVkState *s)
{
    VmafLumaRange luma_range;
    int err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err)
        return err;
    const char *effective_eotf;
    if (s->cambi_eotf && strcmp(s->cambi_eotf, ICAMBI_VK_DEFAULT_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = (s->eotf != NULL) ? s->eotf : ICAMBI_VK_DEFAULT_EOTF;
    }
    VmafEOTF eotf;
    err = vmaf_luminance_init_eotf(&eotf, effective_eotf);
    if (err)
        return err;
    const int num_diffs = 1 << s->max_log_contrast;
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
    int vlt = 0;
    for (int v = 0; v < (1 << 10); v++) {
        double L = vmaf_luminance_get_luminance(v, luma_range, eotf);
        if (L < s->cambi_vis_lum_threshold)
            vlt = v;
    }
    s->vlt_luma = (uint16_t)vlt;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Pipeline build helper (mirrors cambi_vk_build_base).               */
/* ------------------------------------------------------------------ */
static int icambi_vk_build_pipeline(ICambiVkState *s, VmafVulkanKernelPipeline *bundle,
                                    uint32_t pc_size, const uint32_t *spv, size_t spv_size,
                                    uint32_t max_sets, int n_specs, const int32_t *spec_vals)
{
    VkSpecializationMapEntry entries[4];
    int32_t data[4];
    for (int i = 0; i < n_specs; i++) {
        entries[i].constantID = (uint32_t)i;
        entries[i].offset = (uint32_t)((size_t)i * sizeof(int32_t));
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
                        .pSpecializationInfo = (n_specs > 0) ? &si : NULL,
                    },
            },
        .max_descriptor_sets = max_sets,
    };
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, bundle);
}

/* ------------------------------------------------------------------ */
/* Descriptor set alloc + bind (mirrors cambi_vk_alloc_set/write_set) */
/* ------------------------------------------------------------------ */
static int icambi_vk_alloc_set(ICambiVkState *s, const VmafVulkanKernelPipeline *bundle,
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

static void icambi_vk_write_set(ICambiVkState *s, VkDescriptorSet set, VmafVulkanBuffer *in_buf,
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

/* ------------------------------------------------------------------ */
/* Single GPU dispatch (bind pipeline + descriptor set + push + call) */
/* ------------------------------------------------------------------ */
static int icambi_vk_dispatch(ICambiVkState *s, VkCommandBuffer cmd,
                              VmafVulkanKernelPipeline *bundle, VkPipeline pipeline,
                              VmafVulkanBuffer *in_buf, VmafVulkanBuffer *out_buf,
                              const void *pc_data, uint32_t pc_size, uint32_t gx, uint32_t gy)
{
    VkDescriptorSet set = VK_NULL_HANDLE;
    int err = icambi_vk_alloc_set(s, bundle, &set);
    if (err)
        return err;
    icambi_vk_write_set(s, set, in_buf, out_buf);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bundle->pipeline_layout, 0, 1,
                            &set, 0, NULL);
    vkCmdPushConstants(cmd, bundle->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size,
                       pc_data);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdDispatch(cmd, gx, gy, 1);
    return 0;
}

/* Submit the current command buffer and wait (pool slot 0). */
static int icambi_vk_submit_wait(ICambiVkState *s, VmafVulkanKernelSubmit *sub)
{
    return vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, sub);
}

/* ------------------------------------------------------------------ */
/* init                                                                */
/* ------------------------------------------------------------------ */
static int icambi_vk_init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                          unsigned w, unsigned h)
{
    (void)pix_fmt;
    ICambiVkState *s = fex->priv;

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
    if (s->enc_width < ICAMBI_VK_MIN_WIDTH_HEIGHT && s->enc_height < ICAMBI_VK_MIN_WIDTH_HEIGHT) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_cambi_vulkan: encoded resolution %dx%d below minimum %dx%d.\n",
                 s->enc_width, s->enc_height, ICAMBI_VK_MIN_WIDTH_HEIGHT,
                 ICAMBI_VK_MIN_WIDTH_HEIGHT);
        return -EINVAL;
    }
    s->src_width = w;
    s->src_height = h;
    s->src_bpc = bpc;
    s->proc_width = (unsigned)s->enc_width;
    s->proc_height = (unsigned)s->enc_height;
    s->adjusted_window = icambi_vk_adjust_window(s->window_size, s->proc_width, s->proc_height);

    if (fex->vulkan_state) {
        s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, -1);
        if (err)
            return err;
        s->owns_ctx = 1;
    }

    /* spatial_mask pipeline — no spec constants. */
    int err = icambi_vk_build_pipeline(s, &s->pl_spatial_mask, sizeof(ICambiVkPushSpatialMask),
                                       cambi_spatial_mask_spv, sizeof(cambi_spatial_mask_spv),
                                       ICAMBI_VK_MASK_DESC_SETS, 0, NULL);
    if (err)
        goto fail;
    s->pipelines[ICAMBI_PL_SPATIAL_MASK] = s->pl_spatial_mask.pipeline;

    /* filter_mode H pipeline — AXIS spec constant = 0. */
    {
        const int32_t spec_h[3] = {(int32_t)s->proc_width, (int32_t)s->proc_height, 0};
        err = icambi_vk_build_pipeline(s, &s->pl_filter_mode, sizeof(ICambiVkPushFilterMode),
                                       cambi_filter_mode_spv, sizeof(cambi_filter_mode_spv),
                                       ICAMBI_VK_FM_DESC_SETS, 3, spec_h);
        if (err)
            goto fail;
    }
    s->pipelines[ICAMBI_PL_FILTER_MODE_H] = s->pl_filter_mode.pipeline;

    /* filter_mode V variant — AXIS spec constant = 1. */
    {
        const int32_t spec_v[3] = {(int32_t)s->proc_width, (int32_t)s->proc_height, 1};
        VkSpecializationMapEntry entries[3];
        int32_t data[3];
        for (int i = 0; i < 3; i++) {
            entries[i].constantID = (uint32_t)i;
            entries[i].offset = (uint32_t)((size_t)i * sizeof(int32_t));
            entries[i].size = sizeof(int32_t);
            data[i] = spec_v[i];
        }
        VkSpecializationInfo si = {
            .mapEntryCount = 3,
            .pMapEntries = entries,
            .dataSize = 3 * sizeof(int32_t),
            .pData = data,
        };
        VkComputePipelineCreateInfo cpci = {
            .stage =
                {
                    .pName = "main",
                    .pSpecializationInfo = &si,
                },
        };
        err = vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl_filter_mode, &cpci,
                                                      &s->pipelines[ICAMBI_PL_FILTER_MODE_V]);
        if (err)
            goto fail;
    }

    /* decimate pipeline — spec: OUT_WIDTH, OUT_HEIGHT (proc-level; actual dims via push). */
    {
        const int32_t spec_dec[2] = {(int32_t)s->proc_width, (int32_t)s->proc_height};
        err = icambi_vk_build_pipeline(s, &s->pl_decimate, sizeof(ICambiVkPushDecimate),
                                       cambi_decimate_spv, sizeof(cambi_decimate_spv),
                                       ICAMBI_VK_DEC_DESC_SETS, 2, spec_dec);
        if (err)
            goto fail;
    }
    s->pipelines[ICAMBI_PL_DECIMATE] = s->pl_decimate.pipeline;

    /* GPU buffers — host-mapped coherent (alloc_readback gives HtoD + DtoH). */
    const size_t stride_words = ((size_t)s->proc_width + 1u) >> 1;
    const size_t buf_bytes = stride_words * s->proc_height * sizeof(uint32_t);

    err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->image_buf, buf_bytes);
    if (err)
        goto fail;
    err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->mask_buf, buf_bytes);
    if (err)
        goto fail;
    err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->scratch_buf, buf_bytes);
    if (err)
        goto fail;

    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, 1, &s->sub_pool);
    if (err)
        goto fail;

    err = vmaf_picture_alloc(&s->pics[0], VMAF_PIX_FMT_YUV400P, 10, s->proc_width, s->proc_height);
    if (err)
        goto fail;
    err = vmaf_picture_alloc(&s->pics[1], VMAF_PIX_FMT_YUV400P, 10, s->proc_width, s->proc_height);
    if (err)
        goto fail;

    /* Host scratch for the CPU residual (mirrors integer_cambi_cuda.c::init). */
    const int num_diffs = 1 << s->max_log_contrast;
    static const int contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                             8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
    s->buffers.diffs_to_consider = malloc(sizeof(uint16_t) * (size_t)num_diffs);
    if (!s->buffers.diffs_to_consider)
        goto fail;
    s->buffers.diff_weights = malloc(sizeof(int) * (size_t)num_diffs);
    if (!s->buffers.diff_weights)
        goto fail;
    s->buffers.all_diffs = malloc(sizeof(int) * (size_t)(2 * num_diffs + 1));
    if (!s->buffers.all_diffs)
        goto fail;
    for (int d = 0; d < num_diffs; d++) {
        s->buffers.diffs_to_consider[d] = (uint16_t)(d + 1);
        s->buffers.diff_weights[d] = contrast_weights[d];
    }
    for (int d = -num_diffs; d <= num_diffs; d++)
        s->buffers.all_diffs[d + num_diffs] = d;

    s->buffers.tvi_for_diff = malloc(sizeof(uint16_t) * (size_t)num_diffs);
    if (!s->buffers.tvi_for_diff)
        goto fail;
    err = icambi_vk_init_tvi(s);
    if (err)
        goto fail;

    s->buffers.c_values = malloc(sizeof(float) * s->proc_width * s->proc_height);
    if (!s->buffers.c_values)
        goto fail;

    const uint16_t num_bins = (uint16_t)(1024u + (unsigned)(s->buffers.all_diffs[2 * num_diffs] -
                                                            s->buffers.all_diffs[0]));
    s->buffers.c_values_histograms = malloc(sizeof(uint16_t) * s->proc_width * (size_t)num_bins);
    if (!s->buffers.c_values_histograms)
        goto fail;

    const int pad_size = ICAMBI_VK_MASK_FILTER_SIZE / 2;
    const int dp_width = (int)s->proc_width + 2 * pad_size + 1;
    const int dp_height = 2 * pad_size + 2;
    s->buffers.mask_dp = malloc(sizeof(uint32_t) * (size_t)dp_width * (size_t)dp_height);
    if (!s->buffers.mask_dp)
        goto fail;
    s->buffers.filter_mode_buffer = malloc(sizeof(uint16_t) * 3u * s->proc_width);
    if (!s->buffers.filter_mode_buffer)
        goto fail;
    s->buffers.derivative_buffer = malloc(sizeof(uint16_t) * s->proc_width);
    if (!s->buffers.derivative_buffer)
        goto fail;

    vmaf_cambi_default_callbacks(&s->inc_range_callback, &s->dec_range_callback,
                                 &s->derivative_callback);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict) {
        err = -ENOMEM;
        goto fail;
    }
    return 0;

fail:
    (void)vmaf_picture_unref(&s->pics[0]);
    (void)vmaf_picture_unref(&s->pics[1]);
    if (s->image_buf)
        vmaf_vulkan_buffer_free(s->ctx, s->image_buf);
    if (s->mask_buf)
        vmaf_vulkan_buffer_free(s->ctx, s->mask_buf);
    if (s->scratch_buf)
        vmaf_vulkan_buffer_free(s->ctx, s->scratch_buf);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_spatial_mask);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_filter_mode);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_decimate);
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
    free(s->buffers.diffs_to_consider);
    free(s->buffers.diff_weights);
    free(s->buffers.all_diffs);
    free(s->buffers.tvi_for_diff);
    free(s->buffers.c_values);
    free(s->buffers.c_values_histograms);
    free(s->buffers.mask_dp);
    free(s->buffers.filter_mode_buffer);
    free(s->buffers.derivative_buffer);
    if (s->feature_name_dict)
        (void)vmaf_dictionary_free(&s->feature_name_dict);
    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    return (err != 0) ? err : -ENOMEM;
}

/* ------------------------------------------------------------------ */
/* Upload luma plane: pics[0].data[0] -> image_buf (host-mapped)      */
/* ------------------------------------------------------------------ */
static int icambi_vk_upload_image(ICambiVkState *s)
{
    uint32_t *dst = vmaf_vulkan_buffer_host(s->image_buf);
    if (!dst)
        return -EIO;
    const uint16_t *src = (const uint16_t *)s->pics[0].data[0];
    const ptrdiff_t src_stride = s->pics[0].stride[0] >> 1; /* in uint16 units */
    const size_t img_words_per_row = ((size_t)s->proc_width + 1u) >> 1;
    for (unsigned y = 0; y < s->proc_height; y++) {
        const uint16_t *srow = src + (size_t)y * (size_t)src_stride;
        uint32_t *drow = dst + (size_t)y * img_words_per_row;
        memset(drow, 0, img_words_per_row * sizeof(uint32_t));
        for (unsigned x = 0; x < s->proc_width; x++) {
            uint32_t shift = (x & 1u) * 16u;
            drow[x >> 1] |= (uint32_t)(srow[x] & 0xFFFFu) << shift;
        }
    }
    return vmaf_vulkan_buffer_flush(s->ctx, s->image_buf);
}

/* ------------------------------------------------------------------ */
/* Readback one device buffer into a host VmafPicture luma plane.     */
/* ------------------------------------------------------------------ */
static int icambi_vk_readback(ICambiVkState *s, VmafVulkanBuffer *buf, VmafPicture *pic,
                              unsigned scaled_w, unsigned scaled_h)
{
    int err = vmaf_vulkan_buffer_invalidate(s->ctx, buf);
    if (err)
        return err;
    const uint32_t *src = vmaf_vulkan_buffer_host(buf);
    if (!src)
        return -EIO;
    uint16_t *dst = (uint16_t *)pic->data[0];
    const ptrdiff_t dst_stride = pic->stride[0] >> 1;               /* in uint16 units */
    const size_t words_per_row = ((size_t)s->proc_width + 1u) >> 1; /* image_buf stride */
    for (unsigned y = 0; y < scaled_h; y++) {
        const uint32_t *srow = src + (size_t)y * words_per_row;
        uint16_t *drow = dst + (size_t)y * (size_t)dst_stride;
        for (unsigned x = 0; x < scaled_w; x++) {
            uint32_t word = srow[x >> 1];
            drow[x] = (uint16_t)((word >> ((x & 1u) * 16u)) & 0xFFFFu);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* extract (synchronous, mirrors cambi_vk_extract)                    */
/* ------------------------------------------------------------------ */
static int icambi_vk_extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                             VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                             VmafPicture *dist_pic_90, unsigned index,
                             VmafFeatureCollector *feature_collector)
{
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic_90;
    ICambiVkState *s = fex->priv;

    /* Step 1: host preprocessing -> pics[0] (10-bit planar, proc_w x proc_h). */
    int err = vmaf_cambi_preprocessing(dist_pic, &s->pics[0], (int)s->proc_width,
                                       (int)s->proc_height, s->enc_bitdepth);
    if (err)
        return err;

    /* Step 2: upload pics[0] -> image_buf. */
    err = icambi_vk_upload_image(s);
    if (err)
        return err;

    /* Step 3: GPU spatial mask at scale 0 (image_buf -> mask_buf). */
    {
        const unsigned stride_words = ((unsigned)s->proc_width + 1u) >> 1;
        const uint16_t mi =
            icambi_vk_get_mask_index(s->proc_width, s->proc_height, ICAMBI_VK_MASK_FILTER_SIZE);
        const ICambiVkPushSpatialMask pc_sm = {
            .width = s->proc_width,
            .height = s->proc_height,
            .stride_words = stride_words,
            .mask_index = (uint32_t)mi,
        };
        VmafVulkanKernelSubmit sub;
        err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, 0, &sub);
        if (err)
            return err;
        err = icambi_vk_dispatch(s, sub.cmd, &s->pl_spatial_mask,
                                 s->pipelines[ICAMBI_PL_SPATIAL_MASK], s->image_buf, s->mask_buf,
                                 &pc_sm, sizeof(pc_sm),
                                 (s->proc_width + ICAMBI_VK_WG_X - 1u) / ICAMBI_VK_WG_X,
                                 (s->proc_height + ICAMBI_VK_WG_Y - 1u) / ICAMBI_VK_WG_Y);
        if (err) {
            vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
            return err;
        }
        err = icambi_vk_submit_wait(s, &sub);
        if (err)
            return err;
    }

    /* Step 4: per-scale GPU pipeline + CPU residual. */
    const int num_diffs = 1 << s->max_log_contrast;
    const double topk = (s->topk != ICAMBI_VK_DEFAULT_TOPK) ? s->topk : s->cambi_topk;
    double scores_per_scale[ICAMBI_VK_NUM_SCALES] = {0.0, 0.0, 0.0, 0.0, 0.0};
    unsigned scaled_w = s->proc_width;
    unsigned scaled_h = s->proc_height;

    for (int scale = 0; scale < ICAMBI_VK_NUM_SCALES; scale++) {
        if (scale > 0) {
            const unsigned new_w = (scaled_w + 1u) >> 1;
            const unsigned new_h = (scaled_h + 1u) >> 1;
            const unsigned in_sw = (scaled_w + 1u) >> 1;
            const unsigned out_sw = (new_w + 1u) >> 1;

            /* Decimate image_buf -> scratch_buf. */
            {
                const ICambiVkPushDecimate pc_dec = {
                    .out_width = new_w,
                    .out_height = new_h,
                    .in_stride_words = in_sw,
                    .out_stride_words = out_sw,
                };
                VmafVulkanKernelSubmit sub;
                err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, 0, &sub);
                if (err)
                    return err;
                err = icambi_vk_dispatch(s, sub.cmd, &s->pl_decimate,
                                         s->pipelines[ICAMBI_PL_DECIMATE], s->image_buf,
                                         s->scratch_buf, &pc_dec, sizeof(pc_dec),
                                         (new_w + ICAMBI_VK_WG_X - 1u) / ICAMBI_VK_WG_X,
                                         (new_h + ICAMBI_VK_WG_Y - 1u) / ICAMBI_VK_WG_Y);
                if (err) {
                    vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
                    return err;
                }
                err = icambi_vk_submit_wait(s, &sub);
                if (err)
                    return err;
            }
            /* Swap image_buf <-> scratch_buf. */
            {
                VmafVulkanBuffer *tmp = s->image_buf;
                s->image_buf = s->scratch_buf;
                s->scratch_buf = tmp;
            }

            /* Decimate mask_buf -> scratch_buf. */
            {
                const ICambiVkPushDecimate pc_dec = {
                    .out_width = new_w,
                    .out_height = new_h,
                    .in_stride_words = in_sw,
                    .out_stride_words = out_sw,
                };
                VmafVulkanKernelSubmit sub;
                err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, 0, &sub);
                if (err)
                    return err;
                err = icambi_vk_dispatch(s, sub.cmd, &s->pl_decimate,
                                         s->pipelines[ICAMBI_PL_DECIMATE], s->mask_buf,
                                         s->scratch_buf, &pc_dec, sizeof(pc_dec),
                                         (new_w + ICAMBI_VK_WG_X - 1u) / ICAMBI_VK_WG_X,
                                         (new_h + ICAMBI_VK_WG_Y - 1u) / ICAMBI_VK_WG_Y);
                if (err) {
                    vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
                    return err;
                }
                err = icambi_vk_submit_wait(s, &sub);
                if (err)
                    return err;
            }
            /* Swap mask_buf <-> scratch_buf. */
            {
                VmafVulkanBuffer *tmp = s->mask_buf;
                s->mask_buf = s->scratch_buf;
                s->scratch_buf = tmp;
            }

            scaled_w = new_w;
            scaled_h = new_h;
        }

        const unsigned stride_words = (scaled_w + 1u) >> 1;

        /* filter_mode H: image_buf -> scratch_buf. */
        {
            const ICambiVkPushFilterMode pc_fm = {
                .width = scaled_w,
                .height = scaled_h,
                .stride_words = stride_words,
            };
            VmafVulkanKernelSubmit sub;
            err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, 0, &sub);
            if (err)
                return err;
            err = icambi_vk_dispatch(s, sub.cmd, &s->pl_filter_mode,
                                     s->pipelines[ICAMBI_PL_FILTER_MODE_H], s->image_buf,
                                     s->scratch_buf, &pc_fm, sizeof(pc_fm),
                                     (scaled_w + ICAMBI_VK_WG_X - 1u) / ICAMBI_VK_WG_X,
                                     (scaled_h + ICAMBI_VK_WG_Y - 1u) / ICAMBI_VK_WG_Y);
            if (err) {
                vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
                return err;
            }
            err = icambi_vk_submit_wait(s, &sub);
            if (err)
                return err;
        }

        /* filter_mode V: scratch_buf -> image_buf. */
        {
            const ICambiVkPushFilterMode pc_fm = {
                .width = scaled_w,
                .height = scaled_h,
                .stride_words = stride_words,
            };
            VmafVulkanKernelSubmit sub;
            err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, 0, &sub);
            if (err)
                return err;
            err = icambi_vk_dispatch(s, sub.cmd, &s->pl_filter_mode,
                                     s->pipelines[ICAMBI_PL_FILTER_MODE_V], s->scratch_buf,
                                     s->image_buf, &pc_fm, sizeof(pc_fm),
                                     (scaled_w + ICAMBI_VK_WG_X - 1u) / ICAMBI_VK_WG_X,
                                     (scaled_h + ICAMBI_VK_WG_Y - 1u) / ICAMBI_VK_WG_Y);
            if (err) {
                vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
                return err;
            }
            err = icambi_vk_submit_wait(s, &sub);
            if (err)
                return err;
        }

        /* Readback image_buf -> pics[0], mask_buf -> pics[1]. */
        err = icambi_vk_readback(s, s->image_buf, &s->pics[0], scaled_w, scaled_h);
        if (err)
            return err;
        err = icambi_vk_readback(s, s->mask_buf, &s->pics[1], scaled_w, scaled_h);
        if (err)
            return err;

        /* CPU residual. */
        vmaf_cambi_calculate_c_values(&s->pics[0], &s->pics[1], s->buffers.c_values,
                                      s->buffers.c_values_histograms, s->adjusted_window,
                                      (uint16_t)num_diffs, s->buffers.tvi_for_diff, s->vlt_luma,
                                      s->buffers.diff_weights, s->buffers.all_diffs, (int)scaled_w,
                                      (int)scaled_h, s->inc_range_callback, s->dec_range_callback);

        scores_per_scale[scale] =
            vmaf_cambi_spatial_pooling(s->buffers.c_values, topk, scaled_w, scaled_h);
    }

    /* Step 5: final score. */
    const uint16_t pixels_in_window = vmaf_cambi_get_pixels_in_window(s->adjusted_window);
    double score = vmaf_cambi_weight_scores_per_scale(scores_per_scale, pixels_in_window);
    if (score > s->cambi_max_val)
        score = s->cambi_max_val;
    if (score < 0.0)
        score = 0.0;

    return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "Cambi_feature_cambi_score", score, index);
}

/* ------------------------------------------------------------------ */
/* close                                                               */
/* ------------------------------------------------------------------ */
static int icambi_vk_close(VmafFeatureExtractor *fex)
{
    ICambiVkState *s = fex->priv;
    int rc = 0;

    if (s->image_buf) {
        vmaf_vulkan_buffer_free(s->ctx, s->image_buf);
        s->image_buf = NULL;
    }
    if (s->mask_buf) {
        vmaf_vulkan_buffer_free(s->ctx, s->mask_buf);
        s->mask_buf = NULL;
    }
    if (s->scratch_buf) {
        vmaf_vulkan_buffer_free(s->ctx, s->scratch_buf);
        s->scratch_buf = NULL;
    }

    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_spatial_mask);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_filter_mode);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_decimate);
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);

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

    if (s->feature_name_dict) {
        const int e = vmaf_dictionary_free(&s->feature_name_dict);
        if (e && rc == 0)
            rc = e;
    }
    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;
    return rc;
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */
static const char *icambi_vk_provided_features[] = {"Cambi_feature_cambi_score", NULL};

/* External linkage required — the registry iterates over vmaf_fex_* externs.
 * NOLINTNEXTLINE(misc-use-internal-linkage,cppcoreguidelines-avoid-non-const-global-variables) */
VmafFeatureExtractor vmaf_fex_integer_cambi_vulkan = {
    .name = "integer_cambi_vulkan",
    .init = icambi_vk_init,
    .extract = icambi_vk_extract,
    .close = icambi_vk_close,
    .options = options,
    .priv_size = sizeof(ICambiVkState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = icambi_vk_provided_features,
    .chars =
        {
            /* 5 scales x (spatial_mask once + filter_mode H + V) + decimate x2 for scales 1-4
         * = 3 + 4*2 + 5*2 = ~21 dispatches/frame (conservative estimate). */
            .n_dispatches_per_frame = 21,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_DIRECT,
        },
};
