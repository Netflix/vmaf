/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_ms_ssim feature kernel on the Vulkan backend
 *  (T7-23 / ADR-0188 / ADR-0190, GPU long-tail batch 2 part 2a).
 *  Vulkan twin of the active CPU `float_ms_ssim` extractor in
 *  libvmaf/src/feature/float_ms_ssim.c.
 *
 *  5-level pyramid + 3-output SSIM per scale + host-side
 *  Wang product combine. Per-scale work uses two shaders:
 *    - ms_ssim_decimate.comp — 9-tap 9/7 biorthogonal LPF +
 *      2× downsample (skipped for scale 0 which uses the
 *      picture_copy-normalised input directly).
 *    - ms_ssim.comp — same horizontal pass as ssim.comp, but
 *      the vertical pass emits 3 per-WG partial sums (l, c, s).
 *
 *  Host accumulates partials in `double` per scale, divides by
 *  the per-scale (W-10)·(H-10), applies the Wang weights:
 *    MS-SSIM = ∏_i l[i]^α[i] · c[i]^β[i] · s[i]^γ[i]
 *  with α = {0,0,0,0,0.1333}, β=γ = {0.0448, 0.2856, 0.3001,
 *  0.2363, 0.1333} (mirrors ms_ssim.c::g_alphas / g_betas /
 *  g_gammas). Emits `float_ms_ssim`.
 *
 *  Min-dim guard: 11 << 4 = 176 (matches ADR-0153).
 *
 *  enable_lcs (T7-35 / ADR-0243): when set, emits the 15 extra
 *  per-scale metrics float_ms_ssim_{l,c,s}_scale{0..4} (luminance
 *  / contrast / structure components). The kernel already produces
 *  l_means / c_means / s_means per scale — gating only the
 *  feature_collector_append calls keeps default-path output
 *  bit-identical to the pre-T7-35 binary.
 *
 *  Submit-pool migration (T-GPU-OPT-VK-1 / ADR-0353 / PR-B):
 *  Replaced per-frame vmaf_vulkan_kernel_submit_begin + per-frame
 *  alloc_descriptor_set with:
 *    - Two submit pools: sub_pool_decimate (slot 0) + sub_pool_ssim
 *      (MS_SSIM_SCALES slots, one per scale).
 *    - Pre-allocated descriptor sets at init:
 *        dec_sets_ref[MS_SSIM_SCALES-1] / dec_sets_cmp[MS_SSIM_SCALES-1]
 *        ssim_sets[MS_SSIM_SCALES]
 *      All SSBO handles are stable from init() — descriptor writes
 *      happen once. (T-GPU-OPT-VK-4 / ADR-0353.)
 *
 *  Two submit pools are needed because the decimate phase and the
 *  per-scale SSIM phases have different lifetimes and the SSIM
 *  phases interleave with host-side reduction between scales.
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"

#include "../../vulkan/kernel_template.h"
#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"

#include "ms_ssim_decimate_spv.h" /* generated SPIR-V byte array */
#include "ms_ssim_spv.h"
#include "../picture_copy.h"

#define MS_SSIM_SCALES 5
#define MS_SSIM_GAUSSIAN_LEN 11
#define MS_SSIM_K 11
#define MS_SSIM_WG_X 16
#define MS_SSIM_WG_Y 8
#define MS_SSIM_DECIMATE_BINDINGS 2 /* src, dst */
#define MS_SSIM_BINDINGS 10         /* ref, cmp, 5 intermediates, 3 partials */

static const float g_alphas[MS_SSIM_SCALES] = {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f};
static const float g_betas[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};
static const float g_gammas[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};

typedef struct {
    bool enable_lcs; /* Emit per-scale L/C/S triples (T7-35 / ADR-0243). */
    bool enable_db;  /* Currently unused; kept for option parity. */
    bool clip_db;
    double max_db;
    /* `enable_chroma`: when false, only luma is dispatched.
     * Default false mirrors CPU float_ms_ssim.c PR #939 / ssim_vulkan.c PR #956. */
    bool enable_chroma;
    /* Number of active planes (1 for YUV400P or !enable_chroma, 3 otherwise).
     * v1 kernel reads data[0] only; n_planes>1 is reserved for v2. */
    unsigned n_planes;

    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Pyramid dimensions per scale. */
    unsigned scale_w[MS_SSIM_SCALES];
    unsigned scale_h[MS_SSIM_SCALES];
    /* Per-scale "valid" output dims after the 11-tap conv. */
    unsigned scale_w_horiz[MS_SSIM_SCALES];
    unsigned scale_h_horiz[MS_SSIM_SCALES];
    unsigned scale_w_final[MS_SSIM_SCALES];
    unsigned scale_h_final[MS_SSIM_SCALES];
    unsigned scale_wg_count_x[MS_SSIM_SCALES];
    unsigned scale_wg_count_y[MS_SSIM_SCALES];
    unsigned scale_wg_count[MS_SSIM_SCALES];

    float c1;
    float c2;
    float c3;

    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Decimate pipeline. The kernel-template bundle owns the
     * shared dsl + pipeline_layout + shader + descriptor pool plus
     * the scale-0 base pipeline; the remaining (MS_SSIM_SCALES - 2)
     * scales are sibling pipelines created via _add_variant().
     * decimate_pipelines[0] aliases pl_decimate.pipeline. */
    VmafVulkanKernelPipeline pl_decimate;
    VkPipeline decimate_pipelines[MS_SSIM_SCALES - 1];

    /* SSIM pipeline (horizontal + vertical-with-l/c/s) — same
     * 2-bundle pattern. ssim_pipeline_horiz[0] aliases
     * pl_ssim.pipeline (base = scale 0, pass 0); the other
     * MS_SSIM_SCALES * 2 - 1 entries are variants. */
    VmafVulkanKernelPipeline pl_ssim;
    VkPipeline ssim_pipeline_horiz[MS_SSIM_SCALES];
    VkPipeline ssim_pipeline_vert[MS_SSIM_SCALES];

    /* Submit pools (T-GPU-OPT-VK-1 / ADR-0353):
     *   sub_pool_decimate — 1 slot for the single decimate command buffer.
     *   sub_pool_ssim     — MS_SSIM_SCALES slots; slot i holds the SSIM
     *                       command buffer for scale i. Using separate
     *                       slots lets us interleave the wait+readback
     *                       between scales without pool exhaustion. */
    VmafVulkanKernelSubmitPool sub_pool_decimate;
    VmafVulkanKernelSubmitPool sub_pool_ssim;

    /* Pre-allocated descriptor sets (T-GPU-OPT-VK-4 / ADR-0353).
     * All SSBO handles are init-time-stable (pyramid buffers allocated
     * once below). Written once at init(); no per-frame allocation. */
    VkDescriptorSet dec_sets_ref[MS_SSIM_SCALES - 1];
    VkDescriptorSet dec_sets_cmp[MS_SSIM_SCALES - 1];
    VkDescriptorSet ssim_sets[MS_SSIM_SCALES];

    /* Pyramid: 5 ref + 5 cmp float buffers (host-mapped). */
    VmafVulkanBuffer *pyramid_ref[MS_SSIM_SCALES];
    VmafVulkanBuffer *pyramid_cmp[MS_SSIM_SCALES];

    /* SSIM intermediates (5 buffers) sized for the largest
     * scale (scale 0). Reused across scales by binding with
     * VK_WHOLE_SIZE — each scale writes its (w_horiz × h_horiz)
     * region at the buffer start. */
    VmafVulkanBuffer *h_ref_mu;
    VmafVulkanBuffer *h_cmp_mu;
    VmafVulkanBuffer *h_ref_sq;
    VmafVulkanBuffer *h_cmp_sq;
    VmafVulkanBuffer *h_refcmp;

    /* 3 partials buffers sized for scale 0's wg_count. */
    VmafVulkanBuffer *l_partials;
    VmafVulkanBuffer *c_partials;
    VmafVulkanBuffer *s_partials;

    size_t float_stride;

    VmafDictionary *feature_name_dict;
} MsSsimVulkanState;

static const VmafOption options[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(MsSsimVulkanState, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write MS-SSIM values as dB (host-side post-process)",
        .offset = offsetof(MsSsimVulkanState, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(MsSsimVulkanState, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_chroma",
        .help = "enable calculation for chroma channels (mirrors CPU PR #939 / "
                "ssim_vulkan PR #956; v1 kernel defers multi-plane dispatch to v2)",
        .offset = offsetof(MsSsimVulkanState, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {0},
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t w_out;
    uint32_t h_out;
} DecimatePushConsts;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t w_horiz;
    uint32_t h_horiz;
    uint32_t w_final;
    uint32_t h_final;
    uint32_t num_workgroups_x;
    float c1;
    float c2;
    float c3;
} MsSsimPushConsts;

/* ---- pipeline helpers ---- */

struct DecimateSpecData {
    int32_t width;
    int32_t height;
};

static void decimate_fill_spec(struct DecimateSpecData *spec_data,
                               VkSpecializationMapEntry spec_entries[2],
                               VkSpecializationInfo *spec_info, const MsSsimVulkanState *s,
                               int scale_idx)
{
    spec_data->width = (int32_t)s->scale_w[scale_idx];
    spec_data->height = (int32_t)s->scale_h[scale_idx];
    spec_entries[0] = (VkSpecializationMapEntry){
        .constantID = 0,
        .offset = offsetof(struct DecimateSpecData, width),
        .size = sizeof(int32_t),
    };
    spec_entries[1] = (VkSpecializationMapEntry){
        .constantID = 1,
        .offset = offsetof(struct DecimateSpecData, height),
        .size = sizeof(int32_t),
    };
    *spec_info = (VkSpecializationInfo){
        .mapEntryCount = 2,
        .pMapEntries = spec_entries,
        .dataSize = sizeof(*spec_data),
        .pData = spec_data,
    };
}

static int build_decimate_pipeline_for_scale(MsSsimVulkanState *s, int scale_idx,
                                             VkPipeline *out_pipeline)
{
    struct DecimateSpecData spec_data = {0};
    VkSpecializationMapEntry spec_entries[2];
    VkSpecializationInfo spec_info = {0};
    decimate_fill_spec(&spec_data, spec_entries, &spec_info, s, scale_idx);

    VkComputePipelineCreateInfo cpci = {
        .stage =
            {
                .pName = "main",
                .pSpecializationInfo = &spec_info,
            },
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl_decimate, &cpci, out_pipeline);
}

struct SsimSpecData {
    int32_t width;
    int32_t height;
    int32_t pass;
    int32_t subgroup_size;
};

static void ssim_fill_spec(struct SsimSpecData *spec_data, VkSpecializationMapEntry spec_entries[4],
                           VkSpecializationInfo *spec_info, const MsSsimVulkanState *s,
                           int scale_idx, int pass_id)
{
    spec_data->width = (int32_t)s->scale_w[scale_idx];
    spec_data->height = (int32_t)s->scale_h[scale_idx];
    spec_data->pass = pass_id;
    spec_data->subgroup_size = 32;
    spec_entries[0] = (VkSpecializationMapEntry){
        .constantID = 0,
        .offset = offsetof(struct SsimSpecData, width),
        .size = sizeof(int32_t),
    };
    spec_entries[1] = (VkSpecializationMapEntry){
        .constantID = 1,
        .offset = offsetof(struct SsimSpecData, height),
        .size = sizeof(int32_t),
    };
    spec_entries[2] = (VkSpecializationMapEntry){
        .constantID = 2,
        .offset = offsetof(struct SsimSpecData, pass),
        .size = sizeof(int32_t),
    };
    spec_entries[3] = (VkSpecializationMapEntry){
        .constantID = 3,
        .offset = offsetof(struct SsimSpecData, subgroup_size),
        .size = sizeof(int32_t),
    };
    *spec_info = (VkSpecializationInfo){
        .mapEntryCount = 4,
        .pMapEntries = spec_entries,
        .dataSize = sizeof(*spec_data),
        .pData = spec_data,
    };
}

static int build_ssim_pipeline_for_scale(MsSsimVulkanState *s, int scale_idx, int pass_id,
                                         VkPipeline *out_pipeline)
{
    struct SsimSpecData spec_data = {0};
    VkSpecializationMapEntry spec_entries[4];
    VkSpecializationInfo spec_info = {0};
    ssim_fill_spec(&spec_data, spec_entries, &spec_info, s, scale_idx, pass_id);

    VkComputePipelineCreateInfo cpci = {
        .stage =
            {
                .pName = "main",
                .pSpecializationInfo = &spec_info,
            },
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl_ssim, &cpci, out_pipeline);
}

static int create_pipelines(MsSsimVulkanState *s)
{
    /* Decimate bundle: scale-0 spec drives the base pipeline; the
     * remaining (MS_SSIM_SCALES - 2) scales are siblings via
     * _add_variant. decimate_pipelines[0] aliases pl_decimate.pipeline. */
    {
        struct DecimateSpecData spec_data = {0};
        VkSpecializationMapEntry spec_entries[2];
        VkSpecializationInfo spec_info = {0};
        decimate_fill_spec(&spec_data, spec_entries, &spec_info, s, /*scale_idx=*/0);

        const VmafVulkanKernelPipelineDesc desc = {
            .ssbo_binding_count = MS_SSIM_DECIMATE_BINDINGS,
            .push_constant_size = (uint32_t)sizeof(DecimatePushConsts),
            .spv_bytes = ms_ssim_decimate_spv,
            .spv_size = ms_ssim_decimate_spv_size,
            .pipeline_create_info =
                {
                    .stage =
                        {
                            .pName = "main",
                            .pSpecializationInfo = &spec_info,
                        },
                },
            /* (MS_SSIM_SCALES - 1) decimations × 2 (ref + cmp). */
            .max_descriptor_sets = (uint32_t)((MS_SSIM_SCALES - 1) * 2),
        };
        int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl_decimate);
        if (err)
            return err;
        s->decimate_pipelines[0] = s->pl_decimate.pipeline;

        for (int i = 1; i < MS_SSIM_SCALES - 1; i++) {
            err = build_decimate_pipeline_for_scale(s, i, &s->decimate_pipelines[i]);
            if (err)
                return err;
        }
    }

    /* SSIM bundle: scale 0 / pass 0 drives the base pipeline; the
     * other (MS_SSIM_SCALES * 2 - 1) entries are siblings via
     * _add_variant. ssim_pipeline_horiz[0] aliases pl_ssim.pipeline. */
    {
        struct SsimSpecData spec_data = {0};
        VkSpecializationMapEntry spec_entries[4];
        VkSpecializationInfo spec_info = {0};
        ssim_fill_spec(&spec_data, spec_entries, &spec_info, s, /*scale_idx=*/0, /*pass_id=*/0);

        const VmafVulkanKernelPipelineDesc desc = {
            .ssbo_binding_count = MS_SSIM_BINDINGS,
            .push_constant_size = (uint32_t)sizeof(MsSsimPushConsts),
            .spv_bytes = ms_ssim_spv,
            .spv_size = ms_ssim_spv_size,
            .pipeline_create_info =
                {
                    .stage =
                        {
                            .pName = "main",
                            .pSpecializationInfo = &spec_info,
                        },
                },
            /* MS_SSIM_SCALES sets (one per scale). */
            .max_descriptor_sets = (uint32_t)MS_SSIM_SCALES,
        };
        int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl_ssim);
        if (err)
            return err;
        s->ssim_pipeline_horiz[0] = s->pl_ssim.pipeline;

        for (int i = 1; i < MS_SSIM_SCALES; i++) {
            err = build_ssim_pipeline_for_scale(s, i, /*pass=*/0, &s->ssim_pipeline_horiz[i]);
            if (err)
                return err;
        }
        for (int i = 0; i < MS_SSIM_SCALES; i++) {
            err = build_ssim_pipeline_for_scale(s, i, /*pass=*/1, &s->ssim_pipeline_vert[i]);
            if (err)
                return err;
        }
    }

    return 0;
}

static int alloc_buffers(MsSsimVulkanState *s)
{
    int err = 0;
    /* Pyramid: 5 levels, ref + cmp. */
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        size_t plane_bytes = (size_t)s->scale_w[i] * s->scale_h[i] * sizeof(float);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->pyramid_ref[i], plane_bytes);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->pyramid_cmp[i], plane_bytes);
    }

    /* Intermediates sized for scale 0 (largest). */
    size_t horiz_bytes_max = (size_t)s->scale_w_horiz[0] * s->scale_h_horiz[0] * sizeof(float);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_ref_mu, horiz_bytes_max);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_cmp_mu, horiz_bytes_max);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_ref_sq, horiz_bytes_max);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_cmp_sq, horiz_bytes_max);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_refcmp, horiz_bytes_max);

    /* Partials sized for scale 0 wg_count (largest). */
    size_t partials_bytes_max = (size_t)s->scale_wg_count[0] * sizeof(float);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->l_partials, partials_bytes_max);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->c_partials, partials_bytes_max);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->s_partials, partials_bytes_max);
    return err ? -ENOMEM : 0;
}

static void write_decimate_descriptor_set(MsSsimVulkanState *s, VkDescriptorSet set,
                                          VmafVulkanBuffer *src, VmafVulkanBuffer *dst)
{
    VkDescriptorBufferInfo dbi[2] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(src), .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(dst), .offset = 0, .range = VK_WHOLE_SIZE},
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

static void write_ssim_descriptor_set(MsSsimVulkanState *s, VkDescriptorSet set,
                                      VmafVulkanBuffer *ref, VmafVulkanBuffer *cmp)
{
    VkDescriptorBufferInfo dbi[MS_SSIM_BINDINGS] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(ref), .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(cmp), .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_ref_mu),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_cmp_mu),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_ref_sq),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_cmp_sq),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_refcmp),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->l_partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->c_partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->s_partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[MS_SSIM_BINDINGS];
    for (int i = 0; i < MS_SSIM_BINDINGS; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, MS_SSIM_BINDINGS, writes, 0, NULL);
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    MsSsimVulkanState *s = fex->priv;

    /* Derive n_planes from pix_fmt, then clamp if !enable_chroma.
     * Mirrors integer_psnr_cuda.c::init's enable_chroma guard (ADR-0453)
     * and ssim_vulkan.c PR #956. */
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        s->n_planes = 1U;
    } else {
        s->n_planes = s->enable_chroma ? 3U : 1U;
    }

    /* ADR-0153 minimum resolution check: 5-level pyramid +
     * 11-tap kernel needs >= 11 << 4 = 176 in min(w, h). */
    const unsigned min_dim = (unsigned)MS_SSIM_GAUSSIAN_LEN << (MS_SSIM_SCALES - 1);
    if (w < min_dim || h < min_dim) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "ms_ssim_vulkan: input %ux%u too small; %d-level %d-tap MS-SSIM pyramid needs"
                 " >= %ux%u (Netflix#1414 / ADR-0153)\n",
                 w, h, MS_SSIM_SCALES, MS_SSIM_GAUSSIAN_LEN, min_dim, min_dim);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->float_stride = (size_t)w * sizeof(float);

    /* Pyramid dimensions. Each next scale: w' = w/2 + (w&1). */
    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < MS_SSIM_SCALES; i++) {
        s->scale_w[i] = (s->scale_w[i - 1] / 2) + (s->scale_w[i - 1] & 1);
        s->scale_h[i] = (s->scale_h[i - 1] / 2) + (s->scale_h[i - 1] & 1);
    }
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        s->scale_w_horiz[i] = s->scale_w[i] - (MS_SSIM_K - 1);
        s->scale_h_horiz[i] = s->scale_h[i];
        s->scale_w_final[i] = s->scale_w[i] - (MS_SSIM_K - 1);
        s->scale_h_final[i] = s->scale_h[i] - (MS_SSIM_K - 1);
        s->scale_wg_count_x[i] =
            (s->scale_w_final[i] + (unsigned)MS_SSIM_WG_X - 1) / (unsigned)MS_SSIM_WG_X;
        s->scale_wg_count_y[i] =
            (s->scale_h_final[i] + (unsigned)MS_SSIM_WG_Y - 1) / (unsigned)MS_SSIM_WG_Y;
        s->scale_wg_count[i] = s->scale_wg_count_x[i] * s->scale_wg_count_y[i];
    }

    /* SSIM constants; L=255 matches CPU after picture_copy. */
    const float L = 255.0f, K1 = 0.01f, K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);
    s->c3 = s->c2 * 0.5f;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "ms_ssim_vulkan: cannot create Vulkan context (%d)\n",
                     err);
            return err;
        }
        s->owns_ctx = 1;
    }

    int err = create_pipelines(s);
    if (err)
        return err;
    err = alloc_buffers(s);
    if (err)
        return err;

    /* Pre-allocate the two submit pools (T-GPU-OPT-VK-1 / ADR-0353):
     *   sub_pool_decimate — 1 slot for the decimate command buffer.
     *   sub_pool_ssim     — MS_SSIM_SCALES slots (one per scale). */
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool_decimate);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/(uint32_t)MS_SSIM_SCALES,
                                                &s->sub_pool_ssim);
    if (err)
        return err;

    /* Pre-allocate all descriptor sets (T-GPU-OPT-VK-4 / ADR-0353).
     * All SSBO handles are stable from here — write each set once. */
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl_decimate.desc_pool,
                                                   s->pl_decimate.dsl,
                                                   (uint32_t)(MS_SSIM_SCALES - 1), s->dec_sets_ref);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl_decimate.desc_pool,
                                                   s->pl_decimate.dsl,
                                                   (uint32_t)(MS_SSIM_SCALES - 1), s->dec_sets_cmp);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl_ssim.desc_pool, s->pl_ssim.dsl,
                                                   (uint32_t)MS_SSIM_SCALES, s->ssim_sets);
    if (err)
        return err;

    /* Write descriptor sets once — all handles are stable. */
    for (int i = 0; i < MS_SSIM_SCALES - 1; i++) {
        write_decimate_descriptor_set(s, s->dec_sets_ref[i], s->pyramid_ref[i],
                                      s->pyramid_ref[i + 1]);
        write_decimate_descriptor_set(s, s->dec_sets_cmp[i], s->pyramid_cmp[i],
                                      s->pyramid_cmp[i + 1]);
    }
    for (int i = 0; i < MS_SSIM_SCALES; i++)
        write_ssim_descriptor_set(s, s->ssim_sets[i], s->pyramid_ref[i], s->pyramid_cmp[i]);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;
}

static int upload_pic(MsSsimVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic)
{
    float *dst = vmaf_vulkan_buffer_host(dst_buf);
    if (!dst)
        return -EIO;
    picture_copy(dst, (ptrdiff_t)s->float_stride, pic, /*offset=*/0, pic->bpc, /*channel=*/0);
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int run_one_scale(MsSsimVulkanState *s, VkCommandBuffer cmd, int scale_idx,
                         VkDescriptorSet ssim_set)
{
    /* Pass 1: horizontal. Grid sized over scale's (W-10) × H. */
    MsSsimPushConsts pc = {
        .width = s->scale_w[scale_idx],
        .height = s->scale_h[scale_idx],
        .w_horiz = s->scale_w_horiz[scale_idx],
        .h_horiz = s->scale_h_horiz[scale_idx],
        .w_final = s->scale_w_final[scale_idx],
        .h_final = s->scale_h_final[scale_idx],
        .num_workgroups_x = s->scale_wg_count_x[scale_idx],
        .c1 = s->c1,
        .c2 = s->c2,
        .c3 = s->c3,
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_ssim.pipeline_layout, 0, 1,
                            &ssim_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_ssim.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                       &pc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->ssim_pipeline_horiz[scale_idx]);
    const uint32_t gx_h = (s->scale_w_horiz[scale_idx] + MS_SSIM_WG_X - 1) / MS_SSIM_WG_X;
    const uint32_t gy_h = (s->scale_h_horiz[scale_idx] + MS_SSIM_WG_Y - 1) / MS_SSIM_WG_Y;
    vkCmdDispatch(cmd, gx_h, gy_h, 1);

    /* Storage-buffer barrier between horiz and vert. */
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    /* Pass 2: vertical + l/c/s combine. */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->ssim_pipeline_vert[scale_idx]);
    vkCmdDispatch(cmd, s->scale_wg_count_x[scale_idx], s->scale_wg_count_y[scale_idx], 1);
    return 0;
}

static int run_decimate(MsSsimVulkanState *s, VkCommandBuffer cmd, int scale_idx,
                        VkDescriptorSet dec_set)
{
    DecimatePushConsts pc = {
        .width = s->scale_w[scale_idx],
        .height = s->scale_h[scale_idx],
        .w_out = s->scale_w[scale_idx + 1],
        .h_out = s->scale_h[scale_idx + 1],
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_decimate.pipeline_layout, 0,
                            1, &dec_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_decimate.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->decimate_pipelines[scale_idx]);
    const uint32_t gx = (s->scale_w[scale_idx + 1] + MS_SSIM_WG_X - 1) / MS_SSIM_WG_X;
    const uint32_t gy = (s->scale_h[scale_idx + 1] + MS_SSIM_WG_Y - 1) / MS_SSIM_WG_Y;
    vkCmdDispatch(cmd, gx, gy, 1);
    return 0;
}

static double sum_partials_double(const float *p, unsigned n)
{
    double total = 0.0;
    for (unsigned i = 0; i < n; i++)
        total += (double)p[i];
    return total;
}

/* Emit the 15 enable_lcs metrics — 5 scales × {l, c, s}. Naming
 * mirrors the CPU host extractor in float_ms_ssim.c (T7-35 /
 * ADR-0243): float_ms_ssim_{l,c,s}_scale{0..4}. */
static int emit_lcs_metrics(VmafFeatureCollector *fc, unsigned index, const double l_means[5],
                            const double c_means[5], const double s_means[5])
{
    static const char *const l_names[5] = {
        "float_ms_ssim_l_scale0", "float_ms_ssim_l_scale1", "float_ms_ssim_l_scale2",
        "float_ms_ssim_l_scale3", "float_ms_ssim_l_scale4",
    };
    static const char *const c_names[5] = {
        "float_ms_ssim_c_scale0", "float_ms_ssim_c_scale1", "float_ms_ssim_c_scale2",
        "float_ms_ssim_c_scale3", "float_ms_ssim_c_scale4",
    };
    static const char *const s_names[5] = {
        "float_ms_ssim_s_scale0", "float_ms_ssim_s_scale1", "float_ms_ssim_s_scale2",
        "float_ms_ssim_s_scale3", "float_ms_ssim_s_scale4",
    };
    int err = 0;
    for (int i = 0; i < 5; i++) {
        err |= vmaf_feature_collector_append(fc, l_names[i], l_means[i], index);
        err |= vmaf_feature_collector_append(fc, c_names[i], c_means[i], index);
        err |= vmaf_feature_collector_append(fc, s_names[i], s_means[i], index);
    }
    return err;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    MsSsimVulkanState *s = fex->priv;
    int err = 0;

    /* Upload at scale 0 only — decimation builds the higher scales. */
    err = upload_pic(s, s->pyramid_ref[0], ref_pic);
    if (err)
        return err;
    err = upload_pic(s, s->pyramid_cmp[0], dist_pic);
    if (err)
        return err;

    /* All descriptor sets are pre-allocated and written at init();
     * no per-frame vkAllocateDescriptorSets or vkUpdateDescriptorSets
     * needed here. (T-GPU-OPT-VK-4 / ADR-0353.) */

    /* Build pyramid scales 1..4 (decimate scales 0..3) in one command
     * buffer, then submit + wait (T-GPU-OPT-VK-1 / ADR-0353). */
    {
        VmafVulkanKernelSubmit decimate_sub = {0};
        err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool_decimate,
                                                /*pool_slot=*/0, &decimate_sub);
        if (err)
            return err;

        VkMemoryBarrier rw_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };

        for (int i = 0; i < MS_SSIM_SCALES - 1; i++) {
            run_decimate(s, decimate_sub.cmd, i, s->dec_sets_ref[i]);
            run_decimate(s, decimate_sub.cmd, i, s->dec_sets_cmp[i]);
            vkCmdPipelineBarrier(decimate_sub.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &rw_barrier, 0, NULL,
                                 0, NULL);
        }

        err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &decimate_sub);
        vmaf_vulkan_kernel_submit_free(s->ctx, &decimate_sub);
        if (err)
            return err;
    }

    /* Now run SSIM per scale in separate command buffers, reading
     * back l/c/s partials between scales. The pyramid is fully
     * built so each scale's input is ready. Each per-scale submit
     * uses a pre-allocated slot from sub_pool_ssim. */
    double l_means[MS_SSIM_SCALES] = {0};
    double c_means[MS_SSIM_SCALES] = {0};
    double s_means[MS_SSIM_SCALES] = {0};
    for (int i = 0; i < MS_SSIM_SCALES && !err; i++) {
        VmafVulkanKernelSubmit ssim_sub = {0};
        err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool_ssim,
                                                /*pool_slot=*/(uint32_t)i, &ssim_sub);
        if (err)
            break;
        run_one_scale(s, ssim_sub.cmd, i, s->ssim_sets[i]);
        err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &ssim_sub);
        vmaf_vulkan_kernel_submit_free(s->ctx, &ssim_sub);
        if (err)
            break;

        int err_inv;
        err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->l_partials);
        if (err_inv)
            return err_inv;
        err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->c_partials);
        if (err_inv)
            return err_inv;
        err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->s_partials);
        if (err_inv)
            return err_inv;
        const float *lp = vmaf_vulkan_buffer_host(s->l_partials);
        const float *cp = vmaf_vulkan_buffer_host(s->c_partials);
        const float *sp = vmaf_vulkan_buffer_host(s->s_partials);
        const unsigned wgc = s->scale_wg_count[i];
        const double n_pixels = (double)s->scale_w_final[i] * (double)s->scale_h_final[i];
        l_means[i] = sum_partials_double(lp, wgc) / n_pixels;
        c_means[i] = sum_partials_double(cp, wgc) / n_pixels;
        s_means[i] = sum_partials_double(sp, wgc) / n_pixels;
    }
    if (err)
        return err;

    /* Wang combine: MS-SSIM = ∏_i l[i]^α[i] · c[i]^β[i] · s[i]^γ[i]. */
    double msssim = 1.0;
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        msssim *= pow(l_means[i], (double)g_alphas[i]) * pow(c_means[i], (double)g_betas[i]) *
                  pow(fabs(s_means[i]), (double)g_gammas[i]);
    }

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "float_ms_ssim", msssim, index);
    if (s->enable_lcs)
        err |= emit_lcs_metrics(feature_collector, index, l_means, c_means, s_means);

    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    MsSsimVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;

    /* Drain submit pools before the pipelines that own the descriptor
     * pools are torn down (ADR-0353). Descriptor sets allocated via
     * descriptor_sets_alloc are freed implicitly with their desc_pool
     * — do NOT call vkFreeDescriptorSets on any pre_set. */
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool_decimate);
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool_ssim);

    /* Variants must be destroyed before the bundle's _destroy()
     * frees the shared shader/layout. The base pipelines
     * (decimate_pipelines[0] and ssim_pipeline_horiz[0]) alias
     * pl_decimate.pipeline / pl_ssim.pipeline — _destroy() handles
     * those; skip them here to avoid a double-free. */
    for (int i = 1; i < MS_SSIM_SCALES - 1; i++)
        if (s->decimate_pipelines[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, s->decimate_pipelines[i], NULL);
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        if (i != 0 && s->ssim_pipeline_horiz[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, s->ssim_pipeline_horiz[i], NULL);
        if (s->ssim_pipeline_vert[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, s->ssim_pipeline_vert[i], NULL);
    }
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_decimate);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_ssim);

    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        if (s->pyramid_ref[i])
            vmaf_vulkan_buffer_free(s->ctx, s->pyramid_ref[i]);
        if (s->pyramid_cmp[i])
            vmaf_vulkan_buffer_free(s->ctx, s->pyramid_cmp[i]);
    }
    if (s->h_ref_mu)
        vmaf_vulkan_buffer_free(s->ctx, s->h_ref_mu);
    if (s->h_cmp_mu)
        vmaf_vulkan_buffer_free(s->ctx, s->h_cmp_mu);
    if (s->h_ref_sq)
        vmaf_vulkan_buffer_free(s->ctx, s->h_ref_sq);
    if (s->h_cmp_sq)
        vmaf_vulkan_buffer_free(s->ctx, s->h_cmp_sq);
    if (s->h_refcmp)
        vmaf_vulkan_buffer_free(s->ctx, s->h_refcmp);
    if (s->l_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->l_partials);
    if (s->c_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->c_partials);
    if (s->s_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->s_partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {"float_ms_ssim", NULL};

VmafFeatureExtractor vmaf_fex_float_ms_ssim_vulkan = {
    .name = "float_ms_ssim_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(MsSsimVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    .chars =
        {
            /* 4 decimate dispatches × 2 (ref+cmp) + 5 horiz + 5 vert
             * = 18 dispatches/frame at scale 0. Compute-heavy
             * across multiple scales. */
            .n_dispatches_per_frame = 18,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
