/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_moment feature kernel on the Vulkan backend (T7-23 /
 *  ADR-0182, GPU long-tail batch 1d).
 *
 *  Single dispatch per frame — kernel reads ref + dis luma planes
 *  and emits four int64 sums (1st and 2nd raw moment for each
 *  plane) into a 4-slot atomic counter. Host divides by w*h and
 *  appends as float metrics, matching the CPU `float_moment`
 *  extractor's output names: `float_moment_{ref,dis}{1st,2nd}`.
 *
 *  Bit-exactness contract: source pixels are integer YUV; the
 *  CPU path computes the equivalent in `double` post-picture_copy
 *  but the input values are integers regardless. int64 sums are
 *  exact within int64's range:
 *    1st moment: 4096*4096 * 65535      = 1.1e12 < 2^63
 *    2nd moment: 4096*4096 * 65535*65535 = 7.2e16 < 2^63
 *
 *  Pattern reference: psnr_vulkan.c (PR #125) — same single-
 *  dispatch + atomic-int64-reduction shape, just 4 atomics
 *  instead of 1.
 *
 *  Submit-pool migration: ADR-0354 (PR-C). Replaces per-frame
 *  vkCreateFence / vkAllocateCommandBuffers / vkAllocateDescriptorSets
 *  with pre-allocated VmafVulkanKernelSubmitPool + pre-allocated
 *  descriptor set written once at init(). All 3 SSBO bindings
 *  (ref_in, dis_in, sums) are init-time-stable.
 *  (T-GPU-OPT-VK-1 / T-GPU-OPT-VK-4 / ADR-0256 / ADR-0354.)
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"

#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"
#include "../../vulkan/kernel_template.h"

#include "moment_spv.h" /* generated SPIR-V byte array */

#define MOMENT_WG_X 16
#define MOMENT_WG_Y 8

typedef struct {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Vulkan context handle. Borrow on imported state, lazy-create otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects (`vulkan/kernel_template.h` bundle, ADR-0246). */
    VmafVulkanKernelPipeline pl;

    /* Submit pool: 1 slot (single dispatch per frame).
     * Pre-allocated in init(); recycled per frame via acquire/end_and_wait.
     * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor set written once at init().
     * All 3 SSBO bindings (ref_in, dis_in, sums) are init-time-stable —
     * no per-frame vkUpdateDescriptorSets needed.
     * (T-GPU-OPT-VK-4 / ADR-0256 / ADR-0354.) */
    VkDescriptorSet pre_set;

    /* Per-channel input buffers (host-mapped). */
    VmafVulkanBuffer *ref_in;
    VmafVulkanBuffer *dis_in;

    /* 4-slot int64 sum buffer: [ref1st, dis1st, ref2nd, dis2nd]. */
    VmafVulkanBuffer *sums;

    VmafDictionary *feature_name_dict;
} MomentVulkanState;

static const VmafOption options[] = {{0}};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} MomentPushConsts;

static int create_pipeline(MomentVulkanState *s)
{
    /* Spec constants pin the (width, height, bpc, subgroup_size) for
     * the shader's shared-memory sizing. Caller supplies the spec-info
     * pointer; the template wires up the pipeline. */
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

    /* `vulkan/kernel_template.h` (ADR-0246) owns the descriptor-set
     * layout (3 SSBO bindings — ref, dis, sums), pipeline layout,
     * shader module, compute pipeline, and descriptor pool sizing. */
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 3U,
        .push_constant_size = (uint32_t)sizeof(MomentPushConsts),
        .spv_bytes = moment_spv,
        .spv_size = moment_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &spec_info,
                    },
            },
        /* 1 pre-allocated set is sufficient: single dispatch, all bindings stable. */
        .max_descriptor_sets = 1U,
    };
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
}

static int alloc_buffers(MomentVulkanState *s)
{
    size_t bytes_per_pixel = (s->bpc <= 8) ? 1 : 2;
    size_t in_bytes = (size_t)s->width * s->height * bytes_per_pixel;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in, in_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_in, in_bytes);
    if (err)
        return err;

    /* 4 int64 slots: ref1st / dis1st / ref2nd / dis2nd. */
    err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->sums, 4 * sizeof(int64_t));
    return err;
}

/* Write the single pre-allocated descriptor set.
 * All 3 bindings are init-time-stable; called once from init().
 * (T-GPU-OPT-VK-4 / ADR-0256 / ADR-0354.) */
static int write_descriptor_set(MomentVulkanState *s, VkDescriptorSet set)
{
    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->sums),
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
    MomentVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "moment_vulkan: cannot create Vulkan context (%d)\n",
                     err);
            return err;
        }
        s->owns_ctx = 1;
    }

    int err = create_pipeline(s);
    if (err)
        return err;

    err = alloc_buffers(s);
    if (err)
        return err;

    /* Pre-allocate submit pool (1 slot — single dispatch per frame).
     * Eliminates per-frame vkAllocateCommandBuffers / vkCreateFence.
     * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;

    /* Pre-allocate 1 descriptor set and write it once.
     * No per-frame vkAllocateDescriptorSets / vkFreeDescriptorSets needed.
     * (T-GPU-OPT-VK-4 / ADR-0256 / ADR-0354.) */
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl,
                                                   /*count=*/1, &s->pre_set);
    if (err)
        return err;
    err = write_descriptor_set(s, s->pre_set);
    if (err)
        return err;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;
}

static int upload_plane(MomentVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(dst_buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    MomentVulkanState *s = fex->priv;
    int err = 0;

    err = upload_plane(s, s->ref_in, ref_pic);
    if (err)
        return err;
    err = upload_plane(s, s->dis_in, dist_pic);
    if (err)
        return err;

    /* Zero the 4-slot sums buffer before each dispatch. */
    memset(vmaf_vulkan_buffer_host(s->sums), 0, 4 * sizeof(int64_t));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->sums);
    if (err)
        return err;

    /* Acquire pre-allocated command buffer + fence from the submit pool.
     * Eliminates per-frame vkAllocateCommandBuffers / vkCreateFence.
     * (T-GPU-OPT-VK-1 / ADR-0256 / ADR-0354.) */
    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    uint32_t gx = (s->width + MOMENT_WG_X - 1) / MOMENT_WG_X;
    uint32_t gy = (s->height + MOMENT_WG_Y - 1) / MOMENT_WG_Y;
    MomentPushConsts pc = {
        .width = s->width,
        .height = s->height,
        .bpc = s->bpc,
        .num_workgroups_x = gx,
    };

    /* Use the pre-allocated descriptor set — no per-frame allocation.
     * (T-GPU-OPT-VK-4 / ADR-0256 / ADR-0354.) */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1,
                            &s->pre_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, gx, gy, 1);

    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    /* Host-side division + score emit. */
    int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->sums);
    if (err_inv)
        return err_inv;
    const int64_t *sums = vmaf_vulkan_buffer_host(s->sums);
    const double n_pixels = (double)s->width * (double)s->height;
    const double ref1 = (double)sums[0] / n_pixels;
    const double dis1 = (double)sums[1] / n_pixels;
    const double ref2 = (double)sums[2] / n_pixels;
    const double dis2 = (double)sums[3] / n_pixels;

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "float_moment_ref1st", ref1, index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "float_moment_dis1st", dis1, index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "float_moment_ref2nd", ref2, index);
    if (!err)
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "float_moment_dis2nd", dis2, index);

cleanup:
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    MomentVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;

    vkDeviceWaitIdle(s->ctx->device);

    /* Destroy submit pool BEFORE pipeline (ADR-0256 / ADR-0354 ordering rule).
     * pre_set is implicitly destroyed with the descriptor pool below. */
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);

    /* `vulkan/kernel_template.h` collapses the vkDeviceWaitIdle +
     * 5×vkDestroy* sweep into one call. */
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->ref_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_in);
    if (s->dis_in)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_in);
    if (s->sums)
        vmaf_vulkan_buffer_free(s->ctx, s->sums);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {
    "float_moment_ref1st",
    "float_moment_dis1st",
    "float_moment_ref2nd",
    "float_moment_dis2nd",
    NULL,
};

VmafFeatureExtractor vmaf_fex_float_moment_vulkan = {
    .name = "float_moment_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(MomentVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    /* 1 dispatch/frame, reduction-dominated. AUTO + 1080p area
     * matches motion's profile (see ADR-0181 / ADR-0182). */
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
