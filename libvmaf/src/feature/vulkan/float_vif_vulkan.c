/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_vif feature kernel on the Vulkan backend (T7-23 / batch 3
 *  part 5a — ADR-0192 / ADR-0197). Float twin of integer_vif's
 *  GPU kernels but the algorithm follows CPU `compute_vif`
 *  (libvmaf/src/feature/vif.c) — separable 4-scale pyramid with
 *  decimation between scales.
 *
 *  v1 scope: `vif_kernelscale = 1.0` only (the production default).
 *  Filter widths per scale: {17, 9, 5, 3}.
 *
 *  Per-frame flow (4 scales):
 *     scale 0 compute: read raw ref/dis            → (num0, den0)
 *     scale 1 decimate: read raw → buf_A           (filter+ds by 2)
 *     scale 1 compute:  read buf_A                  → (num1, den1)
 *     scale 2 decimate: read buf_A → buf_B
 *     scale 2 compute:  read buf_B                  → (num2, den2)
 *     scale 3 decimate: read buf_B → buf_A
 *     scale 3 compute:  read buf_A                  → (num3, den3)
 *
 *  Host: per-scale (num, den) double accumulation; final emit:
 *     float_vif_scaleN_score = num[N] / den[N]   (4 scores)
 *     debug:                                       vif, vif_num, vif_den, vif_num_scaleN/vif_den_scaleN
 */

#include <errno.h>
#include <math.h>
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

#include "float_vif_spv.h"

#define FVIF_WG_X 16
#define FVIF_WG_Y 16

/* Default vif_kernelscale=1.0 filter widths per scale. */
static const int FVIF_FW[4] = {17, 9, 5, 3};

typedef struct {
    bool debug;
    double vif_enhn_gain_limit;
    double vif_kernelscale;
    double vif_sigma_nsq;

    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Per-scale dimensions (CPU-mirroring; scale 0 is full, each
     * subsequent scale is (prev - 2*HALF_FW(scale)) / 2). */
    unsigned scale_w[4];
    unsigned scale_h[4];

    VmafVulkanContext *ctx;
    int owns_ctx;

    /* 7 pipelines: pipelines[mode][scale]. mode 0 = compute (4 scales),
     * mode 1 = decimate (3 scales — 1, 2, 3; index 0 unused).
     * `pl` carries the shared layout / shader / DSL / pool plus the
     * (mode=0, scale=0) compute pipeline. The 6 remaining variants
     * (mode 0 scales 1-3, mode 1 scales 1-3) are sibling pipelines
     * created via `vmaf_vulkan_kernel_pipeline_add_variant()`.
     * `pipelines[mode][scale]` is reconstructed at end of
     * `create_pipelines` so the dispatch path stays a clean 2-D
     * lookup. */
    VmafVulkanKernelPipeline pl;
    VkPipeline pipelines[2][4];

    /* Submit-side template (T-GPU-OPT-VK-1 / ADR-0256). */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor sets — 7 passes per frame (T-GPU-OPT-VK-4). */
    VkDescriptorSet pre_sets[7];

    /* Raw input buffers (uint8/16). */
    VmafVulkanBuffer *ref_raw;
    VmafVulkanBuffer *dis_raw;

    /* Two float ping-pong buffers (ref + dis each). buf_A reused for
     * scale 1 input and scale 3 input; buf_B for scale 2 input. */
    VmafVulkanBuffer *ref_buf[2];
    VmafVulkanBuffer *dis_buf[2];

    /* Per-scale (num, den) partials. Each pair sized to the scale's
     * WG count; we use the worst-case (scale 0) WG count for all four
     * to keep allocation trivial. */
    VmafVulkanBuffer *num_partials[4];
    VmafVulkanBuffer *den_partials[4];
    unsigned wg_count[4];

    VmafDictionary *feature_name_dict;
} FloatVifVulkanState;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
    uint32_t in_width;
    uint32_t in_height;
} FloatVifPushConsts;

static const VmafOption options[] = {{.name = "debug",
                                      .help = "debug mode: enable additional output",
                                      .offset = offsetof(FloatVifVulkanState, debug),
                                      .type = VMAF_OPT_TYPE_BOOL,
                                      .default_val.b = false},
                                     {.name = "vif_enhn_gain_limit",
                                      .alias = "egl",
                                      .help = "enhancement gain imposed on vif (>= 1.0)",
                                      .offset = offsetof(FloatVifVulkanState, vif_enhn_gain_limit),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 100.0,
                                      .min = 1.0,
                                      .max = 100.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {.name = "vif_kernelscale",
                                      .help = "scaling factor for the gaussian kernel",
                                      .offset = offsetof(FloatVifVulkanState, vif_kernelscale),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 1.0,
                                      .min = 0.1,
                                      .max = 4.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {.name = "vif_sigma_nsq",
                                      .alias = "snsq",
                                      .help = "neural noise variance",
                                      .offset = offsetof(FloatVifVulkanState, vif_sigma_nsq),
                                      .type = VMAF_OPT_TYPE_DOUBLE,
                                      .default_val.d = 2.0,
                                      .min = 0.0,
                                      .max = 5.0,
                                      .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
                                     {0}};

static inline void wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + FVIF_WG_X - 1u) / FVIF_WG_X;
    *gy = (h + FVIF_WG_Y - 1u) / FVIF_WG_Y;
}

struct FloatVifSpecData {
    int32_t mode;
    int32_t scale;
    int32_t subgroup_size;
};

static void float_vif_fill_spec(struct FloatVifSpecData *spec_data,
                                VkSpecializationMapEntry spec_entries[3],
                                VkSpecializationInfo *spec_info, int mode, int scale)
{
    spec_data->mode = mode;
    spec_data->scale = scale;
    spec_data->subgroup_size = 32;
    spec_entries[0] = (VkSpecializationMapEntry){
        .constantID = 0,
        .offset = offsetof(struct FloatVifSpecData, mode),
        .size = sizeof(int32_t),
    };
    spec_entries[1] = (VkSpecializationMapEntry){
        .constantID = 1,
        .offset = offsetof(struct FloatVifSpecData, scale),
        .size = sizeof(int32_t),
    };
    spec_entries[2] = (VkSpecializationMapEntry){
        .constantID = 2,
        .offset = offsetof(struct FloatVifSpecData, subgroup_size),
        .size = sizeof(int32_t),
    };
    *spec_info = (VkSpecializationInfo){
        .mapEntryCount = 3,
        .pMapEntries = spec_entries,
        .dataSize = sizeof(*spec_data),
        .pData = spec_data,
    };
}

static int build_pipeline_for(FloatVifVulkanState *s, int mode, int scale, VkPipeline *out_pipeline)
{
    struct FloatVifSpecData spec_data = {0};
    VkSpecializationMapEntry spec_entries[3];
    VkSpecializationInfo spec_info = {0};
    float_vif_fill_spec(&spec_data, spec_entries, &spec_info, mode, scale);

    VkComputePipelineCreateInfo cpci = {
        .stage =
            {
                .pName = "main",
                .pSpecializationInfo = &spec_info,
            },
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl, &cpci, out_pipeline);
}

static int create_pipelines(FloatVifVulkanState *s)
{
    /* (mode=0, scale=0) compute is the base — the template owns the
     * shared layout / shader / DSL / pool plus this base pipeline.
     * 8 SSBO bindings (some unused per pipeline; same set keeps
     * descriptor management simple). Up to 7 dispatches/frame → 8
     * descriptor sets. */
    struct FloatVifSpecData spec_data = {0};
    VkSpecializationMapEntry spec_entries[3];
    VkSpecializationInfo spec_info = {0};
    float_vif_fill_spec(&spec_data, spec_entries, &spec_info, /*mode=*/0, /*scale=*/0);

    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 8U,
        .push_constant_size = (uint32_t)sizeof(FloatVifPushConsts),
        .spv_bytes = float_vif_spv,
        .spv_size = float_vif_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &spec_info,
                    },
            },
        .max_descriptor_sets = 8U,
    };
    int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
    if (err)
        return err;
    s->pipelines[0][0] = s->pl.pipeline;

    /* The remaining 6 variants (mode 0 scales 1-3, mode 1 scales
     * 1-3) — same layout, shader, DSL, pool, different
     * mode/scale spec-constants. */
    for (int mode = 0; mode < 2; mode++) {
        for (int scale = 0; scale < 4; scale++) {
            if (mode == 0 && scale == 0)
                continue; /* base */
            if (mode == 1 && scale == 0)
                continue; /* unused slot */
            err = build_pipeline_for(s, mode, scale, &s->pipelines[mode][scale]);
            if (err)
                return err;
        }
    }
    return 0;
}

static void compute_per_scale_dims(FloatVifVulkanState *s)
{
    /* CPU `decimate_to_next_scale` runs with VIF_OPT_HANDLE_BORDERS
     * defined (vif_options.h) → buf_valid_{w,h} = full filtered dims,
     * mu_adj = b->mu (no border crop). vif_dec2_s then plain-subsamples
     * by 2 starting at (0,0). So each scale's output is simply
     * floor(prev_dim / 2) — no hfw_prev shrink. */
    s->scale_w[0] = s->width;
    s->scale_h[0] = s->height;
    for (int i = 1; i < 4; i++) {
        s->scale_w[i] = s->scale_w[i - 1] / 2u;
        s->scale_h[i] = s->scale_h[i - 1] / 2u;
    }
}

/* Forward decl — defined after alloc_set_and_bind below. Called
 * from init() once buffers are allocated. */
static void prebind_descriptor_sets(FloatVifVulkanState *s);

static int alloc_buffers(FloatVifVulkanState *s)
{
    size_t bpp = (s->bpc <= 8) ? 1 : 2;
    size_t raw_bytes = (size_t)s->width * s->height * bpp;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_raw, raw_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_raw, raw_bytes);
    if (err)
        return err;

    /* Float ping-pong: scale 1 input is the largest; allocate both at
     * scale-1 size + slack for scale 2/3 reuse. */
    size_t float_bytes = (size_t)s->scale_w[1] * s->scale_h[1] * sizeof(float);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_buf[0], float_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_buf[0], float_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_buf[1], float_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_buf[1], float_bytes);
    if (err)
        return err;

    /* Per-scale (num, den) partials, sized per scale's WG count. */
    for (int i = 0; i < 4; i++) {
        uint32_t gx = 0, gy = 0;
        wg_dims(s->scale_w[i], s->scale_h[i], &gx, &gy);
        s->wg_count[i] = gx * gy;
        size_t pbytes = (size_t)s->wg_count[i] * sizeof(float);
        if (pbytes == 0)
            pbytes = sizeof(float);
        err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->num_partials[i], pbytes);
        if (err)
            return err;
        err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->den_partials[i], pbytes);
        if (err)
            return err;
    }
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    FloatVifVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;

    if (s->vif_kernelscale != 1.0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "float_vif_vulkan: only vif_kernelscale=1.0 is supported in v1\n");
        return -EINVAL;
    }
    compute_per_scale_dims(s);

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "float_vif_vulkan: cannot create Vulkan context (%d)\n",
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

    /* Pre-alloc 1 cmd buffer + 1 fence (T-GPU-OPT-VK-1) and 7
     * descriptor sets — one per pass — for the per-frame loop
     * (T-GPU-OPT-VK-4). Buffers are init()-time-stable so the
     * descriptor binding is established once and reused across
     * frames. */
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl,
                                                   /*count=*/7, s->pre_sets);
    if (err)
        return err;
    prebind_descriptor_sets(s);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;
}

static int upload_plane(FloatVifVulkanState *s, VmafPicture *pic, VmafVulkanBuffer *buf)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, buf);
}

static void bind_descriptor_set(FloatVifVulkanState *s, VkDescriptorSet set,
                                VmafVulkanBuffer *bufs[8])
{
    VkDescriptorBufferInfo dbi[8];
    VkWriteDescriptorSet writes[8];
    for (int i = 0; i < 8; i++) {
        dbi[i] = (VkDescriptorBufferInfo){
            .buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(bufs[i]),
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 8, writes, 0, NULL);
}

/* Per-pass binding pattern. The 7 passes follow a fixed sequence
 * (scale 0 compute, scale 1 decimate, scale 1 compute, ...). All
 * buffers are init()-time-stable, so the descriptor sets can be
 * bound once at init and reused across every frame. */
static void prebind_descriptor_sets(FloatVifVulkanState *s)
{
    /* Pass 0: scale 0 compute  — bufs[2,3] aliased to ref_buf[0]/dis_buf[0]
     * (unused), bufs[4,5] same. */
    VmafVulkanBuffer *bufs[8];
    /* Pass 0: scale 0 compute. ref_in_idx=-1 dis_in_idx=-1 ref_out_idx=-1 dis_out_idx=-1. */
    bufs[0] = s->ref_raw;
    bufs[1] = s->dis_raw;
    bufs[2] = s->ref_buf[0];
    bufs[3] = s->dis_buf[0];
    bufs[4] = s->ref_buf[0];
    bufs[5] = s->dis_buf[0];
    bufs[6] = s->num_partials[0];
    bufs[7] = s->den_partials[0];
    bind_descriptor_set(s, s->pre_sets[0], bufs);
    /* Pass 1: scale 1 decimate. ref_in_idx=-1 dis_in_idx=-1 ref_out_idx=0 dis_out_idx=0. */
    bufs[2] = s->ref_buf[0];
    bufs[3] = s->dis_buf[0];
    bufs[4] = s->ref_buf[0];
    bufs[5] = s->dis_buf[0];
    bufs[6] = s->num_partials[1];
    bufs[7] = s->den_partials[1];
    bind_descriptor_set(s, s->pre_sets[1], bufs);
    /* Pass 2: scale 1 compute. ref_in_idx=0 dis_in_idx=0 ref_out_idx=-1 dis_out_idx=-1. */
    bufs[2] = s->ref_buf[0];
    bufs[3] = s->dis_buf[0];
    bufs[4] = s->ref_buf[0];
    bufs[5] = s->dis_buf[0];
    bufs[6] = s->num_partials[1];
    bufs[7] = s->den_partials[1];
    bind_descriptor_set(s, s->pre_sets[2], bufs);
    /* Pass 3: scale 2 decimate. ref_in_idx=0 dis_in_idx=0 ref_out_idx=1 dis_out_idx=1. */
    bufs[2] = s->ref_buf[0];
    bufs[3] = s->dis_buf[0];
    bufs[4] = s->ref_buf[1];
    bufs[5] = s->dis_buf[1];
    bufs[6] = s->num_partials[2];
    bufs[7] = s->den_partials[2];
    bind_descriptor_set(s, s->pre_sets[3], bufs);
    /* Pass 4: scale 2 compute. ref_in_idx=1 dis_in_idx=1 ref_out_idx=-1 dis_out_idx=-1. */
    bufs[2] = s->ref_buf[1];
    bufs[3] = s->dis_buf[1];
    bufs[4] = s->ref_buf[0];
    bufs[5] = s->dis_buf[0];
    bufs[6] = s->num_partials[2];
    bufs[7] = s->den_partials[2];
    bind_descriptor_set(s, s->pre_sets[4], bufs);
    /* Pass 5: scale 3 decimate. ref_in_idx=1 dis_in_idx=1 ref_out_idx=0 dis_out_idx=0. */
    bufs[2] = s->ref_buf[1];
    bufs[3] = s->dis_buf[1];
    bufs[4] = s->ref_buf[0];
    bufs[5] = s->dis_buf[0];
    bufs[6] = s->num_partials[3];
    bufs[7] = s->den_partials[3];
    bind_descriptor_set(s, s->pre_sets[5], bufs);
    /* Pass 6: scale 3 compute. ref_in_idx=0 dis_in_idx=0 ref_out_idx=-1 dis_out_idx=-1. */
    bufs[2] = s->ref_buf[0];
    bufs[3] = s->dis_buf[0];
    bufs[4] = s->ref_buf[0];
    bufs[5] = s->dis_buf[0];
    bufs[6] = s->num_partials[3];
    bufs[7] = s->den_partials[3];
    bind_descriptor_set(s, s->pre_sets[6], bufs);
}

static void cmd_storage_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void dispatch_pass(FloatVifVulkanState *s, VkCommandBuffer cmd, int mode, int scale,
                          unsigned out_w, unsigned out_h, unsigned in_w, unsigned in_h,
                          VkDescriptorSet pre_set)
{
    uint32_t gx = 0, gy = 0;
    wg_dims(out_w, out_h, &gx, &gy);
    FloatVifPushConsts pc = {
        .width = out_w,
        .height = out_h,
        .bpc = s->bpc,
        .num_workgroups_x = gx,
        .in_width = in_w,
        .in_height = in_h,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[mode][scale]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1,
                            &pre_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, gx, gy, 1);
    cmd_storage_barrier(cmd);
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    FloatVifVulkanState *s = fex->priv;
    int err = 0;

    err = upload_plane(s, ref_pic, s->ref_raw);
    if (err)
        return err;
    err = upload_plane(s, dist_pic, s->dis_raw);
    if (err)
        return err;

    /* Reset partials. */
    for (int i = 0; i < 4; i++) {
        memset(vmaf_vulkan_buffer_host(s->num_partials[i]), 0,
               (size_t)s->wg_count[i] * sizeof(float));
        memset(vmaf_vulkan_buffer_host(s->den_partials[i]), 0,
               (size_t)s->wg_count[i] * sizeof(float));
        err = vmaf_vulkan_buffer_flush(s->ctx, s->num_partials[i]);
        if (err)
            return err;
        err = vmaf_vulkan_buffer_flush(s->ctx, s->den_partials[i]);
        if (err)
            return err;
    }

    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    /* Scale 0 compute (read raw, write num/den 0). */
    dispatch_pass(s, cmd, /*mode=*/0, /*scale=*/0, s->scale_w[0], s->scale_h[0], s->scale_w[0],
                  s->scale_h[0], s->pre_sets[0]);
    /* Scale 1 decimate (raw → buf[0]). */
    dispatch_pass(s, cmd, /*mode=*/1, /*scale=*/1, s->scale_w[1], s->scale_h[1], s->scale_w[0],
                  s->scale_h[0], s->pre_sets[1]);
    /* Scale 1 compute (read buf[0], write num/den 1). */
    dispatch_pass(s, cmd, /*mode=*/0, /*scale=*/1, s->scale_w[1], s->scale_h[1], s->scale_w[1],
                  s->scale_h[1], s->pre_sets[2]);
    /* Scale 2 decimate (buf[0] → buf[1]). */
    dispatch_pass(s, cmd, /*mode=*/1, /*scale=*/2, s->scale_w[2], s->scale_h[2], s->scale_w[1],
                  s->scale_h[1], s->pre_sets[3]);
    /* Scale 2 compute. */
    dispatch_pass(s, cmd, /*mode=*/0, /*scale=*/2, s->scale_w[2], s->scale_h[2], s->scale_w[2],
                  s->scale_h[2], s->pre_sets[4]);
    /* Scale 3 decimate (buf[1] → buf[0]). */
    dispatch_pass(s, cmd, /*mode=*/1, /*scale=*/3, s->scale_w[3], s->scale_h[3], s->scale_w[2],
                  s->scale_h[2], s->pre_sets[5]);
    /* Scale 3 compute. */
    dispatch_pass(s, cmd, /*mode=*/0, /*scale=*/3, s->scale_w[3], s->scale_h[3], s->scale_w[3],
                  s->scale_h[3], s->pre_sets[6]);

    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    /* Reduce per-scale partials in double, emit ratios + (debug) totals. */
    double scores[8];
    for (int i = 0; i < 4; i++) {
        int err_inv;
        err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->num_partials[i]);
        if (err_inv)
            return err_inv;
        err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->den_partials[i]);
        if (err_inv)
            return err_inv;
        const float *num_slots = vmaf_vulkan_buffer_host(s->num_partials[i]);
        const float *den_slots = vmaf_vulkan_buffer_host(s->den_partials[i]);
        double total_num = 0.0;
        double total_den = 0.0;
        for (unsigned j = 0; j < s->wg_count[i]; j++) {
            total_num += (double)num_slots[j];
            total_den += (double)den_slots[j];
        }
        scores[2 * i + 0] = total_num;
        scores[2 * i + 1] = total_den;
    }

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "VMAF_feature_vif_scale0_score",
                                                  scores[0] / scores[1], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_vif_scale1_score",
                                                      scores[2] / scores[3], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_vif_scale2_score",
                                                      scores[4] / scores[5], index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_feature_vif_scale3_score",
                                                      scores[6] / scores[7], index);

    if (s->debug && !err) {
        double score_num = scores[0] + scores[2] + scores[4] + scores[6];
        double score_den = scores[1] + scores[3] + scores[5] + scores[7];
        double score = score_den == 0.0 ? 1.0 : score_num / score_den;
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "vif", score, index);
        if (!err)
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "vif_num", score_num, index);
        if (!err)
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "vif_den", score_den, index);
        const char *names[8] = {"vif_num_scale0", "vif_den_scale0", "vif_num_scale1",
                                "vif_den_scale1", "vif_num_scale2", "vif_den_scale2",
                                "vif_num_scale3", "vif_den_scale3"};
        for (int i = 0; i < 8 && !err; i++) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          names[i], scores[i], index);
        }
    }

cleanup:
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    FloatVifVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    vkDeviceWaitIdle(s->ctx->device);

    /* Destroy the 6 sibling variants first (all (mode, scale)
     * combinations except the base (0, 0) and the unused (1, 0));
     * the base pipeline + shared layout/shader/DSL/pool are owned
     * by the template. */
    for (int mode = 0; mode < 2; mode++) {
        for (int scale = 0; scale < 4; scale++) {
            if (mode == 0 && scale == 0)
                continue;
            if (s->pipelines[mode][scale] != VK_NULL_HANDLE)
                vkDestroyPipeline(s->ctx->device, s->pipelines[mode][scale], NULL);
        }
    }
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->ref_raw)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_raw);
    if (s->dis_raw)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_raw);
    for (int i = 0; i < 2; i++) {
        if (s->ref_buf[i])
            vmaf_vulkan_buffer_free(s->ctx, s->ref_buf[i]);
        if (s->dis_buf[i])
            vmaf_vulkan_buffer_free(s->ctx, s->dis_buf[i]);
    }
    for (int i = 0; i < 4; i++) {
        if (s->num_partials[i])
            vmaf_vulkan_buffer_free(s->ctx, s->num_partials[i]);
        if (s->den_partials[i])
            vmaf_vulkan_buffer_free(s->ctx, s->den_partials[i]);
    }

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {"VMAF_feature_vif_scale0_score",
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
                                          NULL};

VmafFeatureExtractor vmaf_fex_float_vif_vulkan = {
    .name = "float_vif_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(FloatVifVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
