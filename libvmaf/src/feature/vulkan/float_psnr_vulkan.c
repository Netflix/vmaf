/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_psnr feature kernel on the Vulkan backend (T7-23 / batch 3
 *  part 3a — ADR-0192 / ADR-0195). Single-dispatch GLSL kernel
 *  computes per-pixel `(ref - dis)²`, reduces per-WG float partials,
 *  and the host applies the CPU formula:
 *
 *    score = MIN(10 * log10(peak² / max(noise/(w·h), 1e-10)), psnr_max)
 *
 *  Pattern reference: float_ansnr_vulkan.c (single-dispatch float
 *  partials), but simpler — no convolution, no halo, single output
 *  metric.
 *
 *  Submit-pool migration (T-GPU-OPT-VK-1 / ADR-0353 / PR-B):
 *  Replaced per-frame vkAllocateCommandBuffers + vkCreateFence +
 *  vkAllocateDescriptorSets with:
 *    - vmaf_vulkan_kernel_submit_pool_create / _acquire / _end_and_wait
 *    - vmaf_vulkan_kernel_descriptor_sets_alloc (one set at init)
 *  All 3 SSBO bindings are init-time-stable; no per-frame
 *  vkUpdateDescriptorSets needed. (T-GPU-OPT-VK-4 / ADR-0256.)
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

#include "float_psnr_spv.h"

#define FPSNR_WG_X 16
#define FPSNR_WG_Y 16
#define FPSNR_NUM_BINDINGS 3

typedef struct {
    unsigned width;
    unsigned height;
    unsigned bpc;
    double peak;
    double psnr_max;

    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects (`vulkan/kernel_template.h` bundle, ADR-0246). */
    VmafVulkanKernelPipeline pl;

    /* Per-frame submit pool (T-GPU-OPT-VK-1 / ADR-0353).
     * Single slot: the single dispatch uses one command buffer. */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor set reused across frames.
     * All 3 SSBO bindings are init-time-stable; written once at
     * init() — no per-frame vkUpdateDescriptorSets needed.
     * (T-GPU-OPT-VK-4 / ADR-0353.) */
    VkDescriptorSet pre_set;

    VmafVulkanBuffer *ref_in;
    VmafVulkanBuffer *dis_in;
    VmafVulkanBuffer *partials;
    unsigned wg_count;

    VmafDictionary *feature_name_dict;
} FloatPsnrVulkanState;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} FloatPsnrPushConsts;

static inline void fpsnr_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + FPSNR_WG_X - 1u) / FPSNR_WG_X;
    *gy = (h + FPSNR_WG_Y - 1u) / FPSNR_WG_Y;
}

static int create_pipelines(FloatPsnrVulkanState *s)
{
    struct {
        int32_t width;
        int32_t height;
        int32_t bpc;
        int32_t subgroup_size;
    } spec_data = {(int32_t)s->width, (int32_t)s->height, (int32_t)s->bpc, 32};

    VkSpecializationMapEntry spec_entries[4] = {
        {.constantID = 0,
         .offset = offsetof(__typeof__(spec_data), width),
         .size = sizeof(int32_t)},
        {.constantID = 1,
         .offset = offsetof(__typeof__(spec_data), height),
         .size = sizeof(int32_t)},
        {.constantID = 2, .offset = offsetof(__typeof__(spec_data), bpc), .size = sizeof(int32_t)},
        {.constantID = 3,
         .offset = offsetof(__typeof__(spec_data), subgroup_size),
         .size = sizeof(int32_t)},
    };
    VkSpecializationInfo spec_info = {
        .mapEntryCount = 4,
        .pMapEntries = spec_entries,
        .dataSize = sizeof(spec_data),
        .pData = &spec_data,
    };
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = (uint32_t)FPSNR_NUM_BINDINGS,
        .push_constant_size = (uint32_t)sizeof(FloatPsnrPushConsts),
        .spv_bytes = float_psnr_spv,
        .spv_size = float_psnr_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &spec_info,
                    },
            },
        .max_descriptor_sets = 4U,
    };
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
}

static int alloc_buffers(FloatPsnrVulkanState *s)
{
    size_t bpp = (s->bpc <= 8) ? 1 : 2;
    size_t in_bytes = (size_t)s->width * s->height * bpp;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in, in_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_in, in_bytes);
    if (err)
        return err;

    uint32_t gx = 0;
    uint32_t gy = 0;
    fpsnr_wg_dims(s->width, s->height, &gx, &gy);
    s->wg_count = gx * gy;
    size_t pbytes = (size_t)s->wg_count * sizeof(float);
    if (pbytes == 0)
        pbytes = sizeof(float);
    return vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->partials, pbytes);
}

static int write_descriptor_set(FloatPsnrVulkanState *s, VkDescriptorSet set)
{
    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 3, writes, 0, NULL);
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    FloatPsnrVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;

    if (bpc == 8) {
        s->peak = 255.0;
        s->psnr_max = 60.0;
    } else if (bpc == 10) {
        s->peak = 255.75;
        s->psnr_max = 72.0;
    } else if (bpc == 12) {
        s->peak = 255.9375;
        s->psnr_max = 84.0;
    } else if (bpc == 16) {
        s->peak = 255.99609375;
        s->psnr_max = 108.0;
    } else
        return -EINVAL;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "float_psnr_vulkan: cannot create Vulkan context (%d)\n",
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

    /* Pre-allocate submit pool and the single descriptor set.
     * All 3 SSBO bindings are init-time-stable; write once here
     * and reuse every frame — no per-frame vkUpdateDescriptorSets.
     * (T-GPU-OPT-VK-1 + T-GPU-OPT-VK-4 / ADR-0353.) */
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl,
                                                   /*count=*/1, &s->pre_set);
    if (err)
        return err;
    (void)write_descriptor_set(s, s->pre_set);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;
}

static int upload_plane(FloatPsnrVulkanState *s, VmafPicture *pic, VmafVulkanBuffer *buf)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, buf);
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    FloatPsnrVulkanState *s = fex->priv;
    int err = 0;

    err = upload_plane(s, ref_pic, s->ref_in);
    if (err)
        return err;
    err = upload_plane(s, dist_pic, s->dis_in);
    if (err)
        return err;

    memset(vmaf_vulkan_buffer_host(s->partials), 0, (size_t)s->wg_count * sizeof(float));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->partials);
    if (err)
        return err;

    /* All 3 SSBO bindings are init-time-stable; no per-frame
     * vkUpdateDescriptorSets needed (T-GPU-OPT-VK-4 / ADR-0353). */

    /* Acquire a pre-allocated command buffer + fence from the submit
     * pool. Eliminates per-frame vkAllocateCommandBuffers +
     * vkCreateFence + vkAllocateDescriptorSets.
     * (T-GPU-OPT-VK-1 / ADR-0353.) */
    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    uint32_t gx = 0;
    uint32_t gy = 0;
    fpsnr_wg_dims(s->width, s->height, &gx, &gy);
    FloatPsnrPushConsts pc = {
        .width = s->width,
        .height = s->height,
        .bpc = s->bpc,
        .num_workgroups_x = gx,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1,
                            &s->pre_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, gx, gy, 1);

    /* End recording, submit and wait synchronously. */
    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->partials);
    if (err_inv)
        return err_inv;
    {
        const float *slots = vmaf_vulkan_buffer_host(s->partials);
        double total = 0.0;
        for (unsigned i = 0; i < s->wg_count; i++)
            total += (double)slots[i];
        const double n_pix = (double)s->width * (double)s->height;
        const double noise = total / n_pix;
        const double eps = 1e-10;
        const double max_noise = noise > eps ? noise : eps;
        double score = 10.0 * log10(s->peak * s->peak / max_noise);
        if (score > s->psnr_max)
            score = s->psnr_max;

        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "float_psnr", score, index);
    }

cleanup:
    /* Pool-owned submit: submit_free clears local handles only; pool
     * retains the fence + cmd buffer for reuse next frame. */
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    FloatPsnrVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;

    /* Drain the submit pool before the pipeline (ADR-0353).
     * Descriptor sets pre-allocated via descriptor_sets_alloc are
     * freed implicitly with the pool — do NOT call
     * vkFreeDescriptorSets on pre_set. */
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->ref_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_in);
    if (s->dis_in)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_in);
    if (s->partials)
        vmaf_vulkan_buffer_free(s->ctx, s->partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const VmafOption options[] = {{0}};

static const char *provided_features[] = {"float_psnr", NULL};

VmafFeatureExtractor vmaf_fex_float_psnr_vulkan = {
    .name = "float_psnr_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(FloatPsnrVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    /* Single luma-only dispatch per frame; reduction-dominated.
     * Mirrors the profile of psnr_vulkan at 1080p (ADR-0182 / ADR-0195). */
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
