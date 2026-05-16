/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  VIF feature kernel on the Vulkan backend (T5-1b-iv-vif).
 *
 *  This is the kernel-correctness PR follow-up to the dispatch-only
 *  pathfinder shipped as PR #116. The placeholder Gaussian-blur
 *  pipeline is gone; the file now registers a full
 *  `VmafFeatureExtractor` named "vif_vulkan" that emits the four
 *  `VMAF_integer_feature_vif_scale0..3_score` outputs identical to
 *  the SYCL / CUDA implementations.
 *
 *  Side-by-side correspondence with
 * libvmaf/src/feature/sycl/integer_vif_sycl.cpp: init_fex_sycl              ->
 * init() enqueue_vif_work_impl      ->  record_dispatch() + per-scale pipeline
 * lookup submit_fex_sycl            ->  extract() (front half) collect_fex_sycl
 * ->  extract() (back half) — Vulkan port runs sync per frame for the first
 * cut; async submit / collect lands in T5-1b-iv-vif-2. close_fex_sycl ->
 * close()
 *
 *  The Vulkan framework does NOT (yet) push a `VmafVulkanContext *`
 *  onto VmafFeatureExtractor the way HAVE_CUDA / HAVE_SYCL do. We
 *  obtain the context lazily on the first `init()` call via a
 *  `vmaf_vulkan_context_get_or_create()` shim that the runtime PR
 *  scaffolding (PR #116) installed in libvmaf/src/vulkan/common.c.
 *  When the framework gains a `vk_state` field in the next runtime
 *  PR (T5-1b-v), this TU swaps over without breaking ABI.
 *
 *  T-GPU-PERF-VK-3 / ADR-0350: two-level GPU reduction.
 *  The per-WG accumulator SSBO (was ~1.2 MB/frame at 1080p) is now
 *  reduced on-GPU by a second compute dispatch (vif_reduce.comp).
 *  The host reads only VIF_ACCUM_FIELDS * sizeof(int64_t) = 56 bytes
 *  per scale (224 bytes for 4 scales) instead of
 *  wg_count * VIF_ACCUM_FIELDS * sizeof(int64_t).  This eliminates the
 *  dominant 59.73% CPU-time bottleneck in reduce_and_emit at 1080p on
 *  discrete GPU (PCIe BAR uncached reads).  See ADR-0350.
 */

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

#include "vif_spv.h"        /* per-WG accumulator kernel */
#include "vif_reduce_spv.h" /* two-level reduction kernel (ADR-0350) */

/* ------------------------------------------------------------------ */
/* Constants — must match vif.comp + integer_vif_sycl.cpp.            */
/* ------------------------------------------------------------------ */

#define VIF_NUM_SCALES 4
#define VIF_ACCUM_FIELDS 7
#define VIF_LOG2_LUT_SIZE 32768
#define VIF_WG_X 32
#define VIF_WG_Y 4

/* Per-scale int64 accumulator struct, identical layout to SYCL. */
struct vif_accums {
    int64_t x;
    int64_t x2;
    int64_t num_x;
    int64_t num_log;
    int64_t den_log;
    int64_t num_non_log;
    int64_t den_non_log;
};

/* ------------------------------------------------------------------ */
/* Per-extractor state.                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Options. */
    bool debug;
    bool vif_skip_scale0;
    double vif_enhn_gain_limit;

    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Vulkan context handle. When the framework imports a shared
     * VmafVulkanState (CLI: --vulkan_device), we borrow its context
     * and `owns_ctx == 0`. Otherwise we lazy-create one and own it. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipelines (`vulkan/kernel_template.h` bundle, ADR-0246).
     * `pl` carries the shared layout / shader / DSL / pool plus the
     * scale=0 pipeline. `scale_variants[0..2]` are sibling pipelines
     * for scales 1, 2, 3 — created via
     * `vmaf_vulkan_kernel_pipeline_add_variant()`, same layout +
     * shader + DSL + pool, different SCALE spec-constant.
     *
     * `pl_reduce` is the two-level reduction pipeline (ADR-0350):
     * one pipeline shared across all 4 scales (no spec-constant
     * variance needed — the push-constant `wg_count` varies per
     * scale at dispatch time). */
    VmafVulkanKernelPipeline pl;
    VkPipeline scale_variants[VIF_NUM_SCALES - 1]; /* scales 1, 2, 3 */
    VmafVulkanKernelPipeline pl_reduce;

    /* Submit-side template (T-GPU-OPT-VK-1 / ADR-0256). */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor sets per scale (T-GPU-OPT-VK-4). */
    VkDescriptorSet pre_sets[VIF_NUM_SCALES];
    /* Pre-allocated descriptor sets for the reducer (one per scale). */
    VkDescriptorSet reduce_sets[VIF_NUM_SCALES];

    /* Per-scale GPU resources. */
    struct {
        unsigned w, h;
        VmafVulkanBuffer *ref_in; /* SCALE 0 holds host-uploaded plane;
                                 SCALE>0 aliases prev rd_ref. */
        VmafVulkanBuffer *dis_in;
        VmafVulkanBuffer *rd_ref_out; /* downsampled output (SCALE<3 only) */
        VmafVulkanBuffer *rd_dis_out;
        VmafVulkanBuffer *accum; /* per-WG int64 accumulator slots */
        /* ADR-0350: tiny output buffer for the GPU reducer.
         * Size: VIF_ACCUM_FIELDS * sizeof(int64_t) = 56 bytes. */
        VmafVulkanBuffer *reduced_accum;
        unsigned wg_count; /* == gx * gy */
    } scale[VIF_NUM_SCALES];

    VmafVulkanBuffer *log2_lut; /* uint32[32768] */

    VmafDictionary *feature_name_dict;
} VifVulkanState;

/* ------------------------------------------------------------------ */
/* Options table.                                                      */
/* ------------------------------------------------------------------ */

static const VmafOption options[] = {{
                                         .name = "debug",
                                         .help = "debug mode: enable additional output",
                                         .offset = offsetof(VifVulkanState, debug),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = true,
                                     },
                                     {
                                         .name = "vif_enhn_gain_limit",
                                         .help = "enhancement gain imposed on VIF, must be >= 1.0, "
                                                 "where 1.0 means the gain is unrestricted",
                                         .offset = offsetof(VifVulkanState, vif_enhn_gain_limit),
                                         .type = VMAF_OPT_TYPE_DOUBLE,
                                         .default_val.d = 100.0,
                                         .min = 1.0,
                                         .max = 100.0,
                                         .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
                                     },
                                     {
                                         .name = "vif_skip_scale0",
                                         .help = "when set, skip scale 0 calculations",
                                         .alias = "ssclz",
                                         .offset = offsetof(VifVulkanState, vif_skip_scale0),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = false,
                                         .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
                                     },
                                     {0}};

/* ------------------------------------------------------------------ */
/* Push constants. Must mirror `Params` in vif.comp.                   */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t src_stride;
    float vif_enhn_gain_limit;
    uint32_t num_workgroups_x;
} VifPushConsts;

/* Push constants for the reduction shader (vif_reduce.comp). */
typedef struct {
    uint32_t wg_count;
} VifReducePushConsts;

/* ------------------------------------------------------------------ */
/* Helper: workgroup count for a given (w,h).                          */
/* ------------------------------------------------------------------ */

static inline void vif_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + VIF_WG_X - 1u) / VIF_WG_X;
    *gy = (h + VIF_WG_Y - 1u) / VIF_WG_Y;
}

/* Number of reduction workgroups needed to reduce wg_count per-WG
 * slots, with 256 threads per reduction workgroup. */
static inline uint32_t vif_reduce_wg_count(unsigned wg_count)
{
    return (wg_count + 255u) / 256u;
}

/* ------------------------------------------------------------------ */
/* Pipeline / descriptor-set layout creation.                          */
/* ------------------------------------------------------------------ */

struct VifSpecData {
    int32_t scale;
    int32_t bpc;
    int32_t subgroup_size;
};

static void vif_fill_spec(struct VifSpecData *spec_data, VkSpecializationMapEntry spec_entries[3],
                          VkSpecializationInfo *spec_info, const VifVulkanState *s, int scale)
{
    spec_data->scale = scale;
    spec_data->bpc = (int32_t)s->bpc;
    spec_data->subgroup_size = 32;
    spec_entries[0] = (VkSpecializationMapEntry){
        .constantID = 0,
        .offset = offsetof(struct VifSpecData, scale),
        .size = sizeof(int32_t),
    };
    spec_entries[1] = (VkSpecializationMapEntry){
        .constantID = 1,
        .offset = offsetof(struct VifSpecData, bpc),
        .size = sizeof(int32_t),
    };
    spec_entries[2] = (VkSpecializationMapEntry){
        .constantID = 2,
        .offset = offsetof(struct VifSpecData, subgroup_size),
        .size = sizeof(int32_t),
    };
    *spec_info = (VkSpecializationInfo){
        .mapEntryCount = 3,
        .pMapEntries = spec_entries,
        .dataSize = sizeof(*spec_data),
        .pData = spec_data,
    };
}

static int build_pipeline_for_scale(VifVulkanState *s, int scale, VkPipeline *out_pipeline)
{
    struct VifSpecData spec_data = {0};
    VkSpecializationMapEntry spec_entries[3];
    VkSpecializationInfo spec_info = {0};
    vif_fill_spec(&spec_data, spec_entries, &spec_info, s, scale);

    VkComputePipelineCreateInfo cpci = {
        .stage =
            {
                .pName = "main",
                .pSpecializationInfo = &spec_info,
            },
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl, &cpci, out_pipeline);
}

static inline VkPipeline vif_scale_pipeline(const VifVulkanState *s, int scale)
{
    return scale == 0 ? s->pl.pipeline : s->scale_variants[scale - 1];
}

static int create_pipelines(VifVulkanState *s)
{
    /* Scale 0 is the base pipeline — the template owns the shared
     * layout / shader / DSL / pool plus the scale=0 pipeline. Pool
     * sized for 4 frames in flight × 4 scales = 16 sets, 6 SSBOs
     * per set. */
    struct VifSpecData spec_data = {0};
    VkSpecializationMapEntry spec_entries[3];
    VkSpecializationInfo spec_info = {0};
    vif_fill_spec(&spec_data, spec_entries, &spec_info, s, /*scale=*/0);

    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 6U,
        .push_constant_size = (uint32_t)sizeof(VifPushConsts),
        .spv_bytes = vif_spv,
        .spv_size = vif_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &spec_info,
                    },
            },
        .max_descriptor_sets = (uint32_t)(VIF_NUM_SCALES * 4),
    };
    int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
    if (err)
        return err;

    /* Scales 1, 2, 3 — same layout/shader/DSL/pool, different SCALE
     * spec-constant. */
    for (int scale = 1; scale < VIF_NUM_SCALES; scale++) {
        err = build_pipeline_for_scale(s, scale, &s->scale_variants[scale - 1]);
        if (err)
            return err;
    }

    /* ADR-0350: two-level reduction pipeline.
     * 2 SSBO bindings: accum_in (per-WG slots), accum_out (56-byte result).
     * Pool sized for 4 scales × 2 frames = 8 descriptor sets. */
    const VmafVulkanKernelPipelineDesc reduce_desc = {
        .ssbo_binding_count = 2U,
        .push_constant_size = (uint32_t)sizeof(VifReducePushConsts),
        .spv_bytes = vif_reduce_spv,
        .spv_size = vif_reduce_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                    },
            },
        .max_descriptor_sets = (uint32_t)(VIF_NUM_SCALES * 2),
    };
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &reduce_desc, &s->pl_reduce);
}

/* ------------------------------------------------------------------ */
/* Allocate per-scale buffers + upload log2 LUT.                       */
/* ------------------------------------------------------------------ */

static int alloc_scale_buffers(VifVulkanState *s)
{
    unsigned w = s->width;
    unsigned h = s->height;
    unsigned bpc = s->bpc;

    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        s->scale[scale].w = w;
        s->scale[scale].h = h;

        size_t in_bytes;
        if (scale == 0) {
            /* Packed source plane. */
            size_t bytes_per_pixel = (bpc <= 8) ? 1 : 2;
            in_bytes = (size_t)w * h * bytes_per_pixel;
            int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->scale[scale].ref_in, in_bytes);
            if (err)
                return err;
            err = vmaf_vulkan_buffer_alloc(s->ctx, &s->scale[scale].dis_in, in_bytes);
            if (err)
                return err;
        }
        /* For SCALE>0, ref_in / dis_in alias the previous scale's
     * rd_ref_out / rd_dis_out (no separate alloc). */

        /* Downsampled buffer for next scale (SCALE<3). uint32 per
     * pixel, lower 16 bits are payload.
     *
     * Buffer size must use ceil(w/2) x ceil(h/2), NOT floor(w/2) x floor(h/2).
     * The VIF fused kernel writes rd data for every even-coordinate pixel:
     *   (gx & 1) == 0 && (gy & 1) == 0
     * so for a height that is odd (e.g. h=81 at scale 2 for 576x324 input),
     * gy ranges over {0, 2, 4, ..., 80}, giving rd_y values {0, 1, ..., 40}
     * — 41 rows, not 40.  Floor division (h/2 = 40) undersizes the buffer
     * by one row, causing a 72-uint32 out-of-bounds write that corrupts the
     * immediately-following per-WG accumulator buffer and produces the
     * massive-negative int64 denominator at scales 2/3 (PR #718 / ADR-0352).
     * Ceiling division ((h+1)/2 = 41) allocates the correct size. */
        if (scale < VIF_NUM_SCALES - 1) {
            size_t rd_bytes = (size_t)((w + 1u) / 2u) * ((h + 1u) / 2u) * sizeof(uint32_t);
            int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->scale[scale].rd_ref_out, rd_bytes);
            if (err)
                return err;
            err = vmaf_vulkan_buffer_alloc(s->ctx, &s->scale[scale].rd_dis_out, rd_bytes);
            if (err)
                return err;
        }

        /* Per-workgroup accumulator. */
        uint32_t gx, gy;
        vif_wg_dims(w, h, &gx, &gy);
        s->scale[scale].wg_count = gx * gy;
        size_t accum_bytes = (size_t)s->scale[scale].wg_count * VIF_ACCUM_FIELDS * sizeof(int64_t);
        if (accum_bytes == 0)
            accum_bytes = sizeof(int64_t);
        int err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->scale[scale].accum, accum_bytes);
        if (err)
            return err;

        /* ADR-0350: tiny reduced-accumulator output buffer —
         * VIF_ACCUM_FIELDS int64_t = 56 bytes per scale. This is the
         * only buffer the host reads after the GPU reduction. It must
         * be host-visible so reduce_and_emit can map it; the VMA
         * HOST_ACCESS_RANDOM_BIT flag enables efficient uncached
         * reads for discrete-GPU BAR2 windows. */
        size_t reduced_bytes = (size_t)VIF_ACCUM_FIELDS * sizeof(int64_t);
        err = vmaf_vulkan_buffer_alloc(s->ctx, &s->scale[scale].reduced_accum, reduced_bytes);
        if (err)
            return err;

        /* Wire SCALE+1's input to SCALE's rd output. */
        if (scale < VIF_NUM_SCALES - 1) {
            s->scale[scale + 1].ref_in = s->scale[scale].rd_ref_out;
            s->scale[scale + 1].dis_in = s->scale[scale].rd_dis_out;
        }

        w /= 2;
        h /= 2;
    }

    /* Log2 LUT — 32768 entries of uint32, identical to SYCL. */
    size_t lut_bytes = VIF_LOG2_LUT_SIZE * sizeof(uint32_t);
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->log2_lut, lut_bytes);
    if (err)
        return err;
    uint32_t *lut = (uint32_t *)vmaf_vulkan_buffer_host(s->log2_lut);
    for (int j = 0; j < VIF_LOG2_LUT_SIZE; j++)
        lut[j] = (uint32_t)lroundf(log2f((float)(j + 32768)) * 2048.0f);
    err = vmaf_vulkan_buffer_flush(s->ctx, s->log2_lut);
    if (err)
        return err;

    return 0;
}

/* ------------------------------------------------------------------ */
/* init() — VmafFeatureExtractor entry point.                          */
/* ------------------------------------------------------------------ */

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    VifVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;

    /* Borrow the framework's imported context when available; fall back
     * to lazy creation for in-process callers that didn't go through
     * vmaf_vulkan_state_init / _import_state. */
    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "vif_vulkan: cannot create Vulkan context (%d)\n", err);
            return err;
        }
        s->owns_ctx = 1;
    }
    int err = 0;

    err = create_pipelines(s);
    if (err)
        return err;

    err = alloc_scale_buffers(s);
    if (err)
        return err;

    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;

    /* Pre-allocate main descriptor sets (VIF per-WG kernel). */
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl,
                                                   (uint32_t)VIF_NUM_SCALES, s->pre_sets);
    if (err)
        return err;

    /* Pre-allocate reducer descriptor sets (one per scale). */
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl_reduce.desc_pool, s->pl_reduce.dsl,
                                                   (uint32_t)VIF_NUM_SCALES, s->reduce_sets);
    if (err)
        return err;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-frame: record + submit + wait + reduce.                         */
/* ------------------------------------------------------------------ */

static int upload_input_plane(VifVulkanState *s, VmafPicture *pic, int which)
{
    VmafVulkanBuffer *buf = (which == 0) ? s->scale[0].ref_in : s->scale[0].dis_in;
    uint8_t *dst = vmaf_vulkan_buffer_host(buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, buf);
}

static int write_descriptor_set(VifVulkanState *s, VkDescriptorSet set, int scale)
{
    /* For SCALE 3 we have no rd outputs; bind log2_lut twice as a
   * harmless dummy (the shader's SCALE==3 path never writes there
   * because FW_RD == 0). */
    VmafVulkanBuffer *rd_ref =
        (scale < VIF_NUM_SCALES - 1) ? s->scale[scale].rd_ref_out : s->log2_lut;
    VmafVulkanBuffer *rd_dis =
        (scale < VIF_NUM_SCALES - 1) ? s->scale[scale].rd_dis_out : s->log2_lut;

    VkDescriptorBufferInfo dbi[6] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->scale[scale].ref_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->scale[scale].dis_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(rd_ref),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(rd_dis),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->log2_lut),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->scale[scale].accum),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[6];
    for (int i = 0; i < 6; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 6, writes, 0, NULL);
    return 0;
}

/* Write the reducer descriptor set: accum_in → per-WG SSBO,
 * accum_out → 56-byte reduced result buffer. Called once at init
 * time (pre-allocated sets updated here). Per ADR-0350 the bindings
 * don't change frame-to-frame so we only call this once per scale. */
static int write_reduce_descriptor_set(VifVulkanState *s, VkDescriptorSet set, int scale)
{
    VkDescriptorBufferInfo dbi[2] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->scale[scale].accum),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->scale[scale].reduced_accum),
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
    return 0;
}

/* ADR-0350: GPU-reduced version of reduce_and_emit.
 * The host now reads only 56 bytes per scale (224 bytes total for 4
 * scales) from the tiny `reduced_accum` buffer, instead of iterating
 * over all per-WG slots. */
static int reduce_and_emit(VifVulkanState *s, unsigned index, VmafFeatureCollector *fc)
{
    struct vif_accums totals[VIF_NUM_SCALES];
    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        /* Invalidate the host cache for the reduced_accum buffer so
         * we see the GPU-written values (required for device-local
         * buffers that are also host-visible; VMA handles the actual
         * vkInvalidateMappedMemoryRanges call internally). */
        (void)vmaf_vulkan_buffer_invalidate(s->ctx, s->scale[scale].reduced_accum);
        const int64_t *r = vmaf_vulkan_buffer_host(s->scale[scale].reduced_accum);
        totals[scale].x = r[0];
        totals[scale].x2 = r[1];
        totals[scale].num_x = r[2];
        totals[scale].num_log = r[3];
        totals[scale].den_log = r[4];
        totals[scale].num_non_log = r[5];
        totals[scale].den_non_log = r[6];
    }

    /* Per-scale VIF score formula — identical to SYCL collect path. */
    double score_num = 0.0;
    double score_den = 0.0;
    double scale_num[VIF_NUM_SCALES];
    double scale_den[VIF_NUM_SCALES];
    for (int i = 0; i < VIF_NUM_SCALES; i++) {
        double num =
            totals[i].num_log / 2048.0 + (double)totals[i].x2 +
            ((double)totals[i].den_non_log - ((double)totals[i].num_non_log / 16384.0) / 65025.0);
        double den = totals[i].den_log / 2048.0 -
                     ((double)totals[i].x + (double)totals[i].num_x * 17.0) +
                     (double)totals[i].den_non_log;
        scale_num[i] = num;
        scale_den[i] = den;
        /* Skip scale 0 contribution when vif_skip_scale0 is set,
         * matching integer_vif.c write_scores() parity. */
        if (i == 0 && s->vif_skip_scale0)
            continue;
        score_num += num;
        score_den += den;
    }

    static const char *const key_names[] = {
        "VMAF_integer_feature_vif_scale0_score",
        "VMAF_integer_feature_vif_scale1_score",
        "VMAF_integer_feature_vif_scale2_score",
        "VMAF_integer_feature_vif_scale3_score",
    };
    for (int i = 0; i < VIF_NUM_SCALES; i++) {
        double score;
        if (i == 0 && s->vif_skip_scale0) {
            score = 0.0;
        } else {
            score = (scale_den[i] > 0.0) ? scale_num[i] / scale_den[i] : 1.0;
        }
        int err = vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, key_names[i],
                                                          score, index);
        if (err)
            return err;
    }

    if (s->debug) {
        double vif = (score_den > 0.0) ? score_num / score_den : 1.0;
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "integer_vif", vif,
                                                index);
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "integer_vif_num",
                                                score_num, index);
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "integer_vif_den",
                                                score_den, index);
        for (int i = 0; i < VIF_NUM_SCALES; i++) {
            double num_val = (i == 0 && s->vif_skip_scale0) ? 0.0f : scale_num[i];
            double den_val = (i == 0 && s->vif_skip_scale0) ? -1.0f : scale_den[i];
            char name[64];
            (void)snprintf(name, sizeof(name), "integer_vif_num_scale%d", i);
            vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, name, num_val, index);
            (void)snprintf(name, sizeof(name), "integer_vif_den_scale%d", i);
            vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, name, den_val, index);
        }
    }
    return 0;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    VifVulkanState *s = fex->priv;
    int err = 0;

    err = upload_input_plane(s, ref_pic, /*which=*/0);
    if (err)
        return err;
    err = upload_input_plane(s, dist_pic, /*which=*/1);
    if (err)
        return err;

    /* Zero the per-WG accumulator buffers (they're host-mapped). */
    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        size_t bytes = (size_t)s->scale[scale].wg_count * VIF_ACCUM_FIELDS * sizeof(int64_t);
        memset(vmaf_vulkan_buffer_host(s->scale[scale].accum), 0, bytes);
        err = vmaf_vulkan_buffer_flush(s->ctx, s->scale[scale].accum);
        if (err)
            return err;

        /* ADR-0350: zero the reduced_accum output before the reduction
         * dispatch. The reduction shader uses atomicAdd, so the output
         * must start at zero each frame. */
        memset(vmaf_vulkan_buffer_host(s->scale[scale].reduced_accum), 0,
               (size_t)VIF_ACCUM_FIELDS * sizeof(int64_t));
        err = vmaf_vulkan_buffer_flush(s->ctx, s->scale[scale].reduced_accum);
        if (err)
            return err;
    }

    /* Pre-allocated descriptor sets — rebind per frame (VK-4). */
    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        write_descriptor_set(s, s->pre_sets[scale], scale);
        /* Reducer descriptor sets are stable (same buffers every
         * frame); only write them on the first frame. Subsequent
         * frames don't need a rebind because the buffer handles
         * haven't changed. */
        if (index == 0u)
            write_reduce_descriptor_set(s, s->reduce_sets[scale], scale);
    }

    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        unsigned cw = s->scale[scale].w;
        unsigned ch = s->scale[scale].h;
        uint32_t gx, gy;
        vif_wg_dims(cw, ch, &gx, &gy);

        size_t bytes_per_pixel;
        if (scale == 0)
            bytes_per_pixel = (s->bpc <= 8) ? 1 : 2;
        else
            bytes_per_pixel = sizeof(uint32_t); /* rd buffer: uint32/pixel */

        VifPushConsts pc = {
            .width = cw,
            .height = ch,
            .src_stride = (uint32_t)(cw * bytes_per_pixel),
            .vif_enhn_gain_limit = (float)s->vif_enhn_gain_limit,
            .num_workgroups_x = gx,
        };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vif_scale_pipeline(s, scale));
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1,
                                &s->pre_sets[scale], 0, NULL);
        vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                           &pc);
        vkCmdDispatch(cmd, gx, gy, 1);

        /* Pipeline barrier between scales: SCALE N's rd output is
     * SCALE N+1's input. Read-after-write. */
        if (scale < VIF_NUM_SCALES - 1) {
            VkMemoryBarrier mb = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        }
    }

    /* ADR-0350: after all per-WG vif.comp dispatches are done, issue a
     * single barrier to flush the per-WG accumulator SSBOs, then
     * dispatch vif_reduce.comp once per scale.  All four reductions
     * share the same command buffer — no vkQueueWaitIdle between them
     * (Vulkan spec §7.1: pipeline barriers are intra-queue ordered).
     *
     * Barrier: SHADER_WRITE from per-WG kernel → SHADER_READ |
     *          SHADER_WRITE for the reducer (the reducer also atomicAdd-
     *          writes to reduced_accum, hence SHADER_WRITE on dstAccess
     *          too).  VK_ACCESS_HOST_READ_BIT on dstAccess is NOT needed
     *          here because the CPU reads after the fence (submit_end_and_wait)
     *          which already provides the CPU-visible ordering guarantee. */
    VkMemoryBarrier reduce_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &reduce_barrier, 0, NULL, 0,
                         NULL);

    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        uint32_t n_reduce_wg = vif_reduce_wg_count(s->scale[scale].wg_count);
        VifReducePushConsts rpc = {
            .wg_count = s->scale[scale].wg_count,
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_reduce.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_reduce.pipeline_layout,
                                0, 1, &s->reduce_sets[scale], 0, NULL);
        vkCmdPushConstants(cmd, s->pl_reduce.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(rpc), &rpc);
        vkCmdDispatch(cmd, n_reduce_wg, 1, 1);
    }

    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    err = reduce_and_emit(s, index, feature_collector);

cleanup:
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

/* ------------------------------------------------------------------ */
/* close()                                                              */
/* ------------------------------------------------------------------ */

static int close_fex(VmafFeatureExtractor *fex)
{
    VifVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;

    vkDeviceWaitIdle(dev);

    /* Destroy the scale 1/2/3 variants first; the base pipeline +
     * shared layout/shader/DSL/pool are owned by the template. */
    for (int i = 0; i < VIF_NUM_SCALES - 1; i++) {
        if (s->scale_variants[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, s->scale_variants[i], NULL);
    }
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_reduce);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    for (int scale = 0; scale < VIF_NUM_SCALES; scale++) {
        if (scale == 0) {
            if (s->scale[scale].ref_in)
                vmaf_vulkan_buffer_free(s->ctx, s->scale[scale].ref_in);
            if (s->scale[scale].dis_in)
                vmaf_vulkan_buffer_free(s->ctx, s->scale[scale].dis_in);
        }
        if (s->scale[scale].rd_ref_out)
            vmaf_vulkan_buffer_free(s->ctx, s->scale[scale].rd_ref_out);
        if (s->scale[scale].rd_dis_out)
            vmaf_vulkan_buffer_free(s->ctx, s->scale[scale].rd_dis_out);
        if (s->scale[scale].accum)
            vmaf_vulkan_buffer_free(s->ctx, s->scale[scale].accum);
        if (s->scale[scale].reduced_accum)
            vmaf_vulkan_buffer_free(s->ctx, s->scale[scale].reduced_accum);
    }
    if (s->log2_lut)
        vmaf_vulkan_buffer_free(s->ctx, s->log2_lut);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Provided features + registration.                                   */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"VMAF_integer_feature_vif_scale0_score",
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
                                          NULL};

VmafFeatureExtractor vmaf_fex_integer_vif_vulkan = {
    .name = "vif_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(VifVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
