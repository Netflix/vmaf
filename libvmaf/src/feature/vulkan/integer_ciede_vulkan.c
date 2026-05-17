/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ciede2000 feature extractor on the Vulkan backend — integer-path
 *  variant (mirrors integer_ciede_cuda.c).  This file exposes
 *  `vmaf_fex_integer_ciede_vulkan` for symmetry with the CUDA twin
 *  `vmaf_fex_ciede_cuda` (ADR-0182 / ADR-0187).
 *
 *  Implementation delegates to the same GLSL compute shader
 *  (`ciede.comp` → `ciede_spv.h`) and host-side infrastructure used
 *  by `ciede_vulkan.c`.  The only differences are:
 *   - extractor name: "integer_ciede_vulkan" vs "ciede_vulkan"
 *   - symbol:         vmaf_fex_integer_ciede_vulkan
 *  Both emit the same `ciede2000` feature key, so the two
 *  extractors are not registered simultaneously in the same pipeline.
 *
 *  Precision contract: places=2 vs CPU scalar reference (per-pixel
 *  transcendentals on GPU — see ADR-0187).  Per-WG float partials
 *  are reduced in `double` on the host; host applies
 *  `45 - 20*log10(mean_dE)` to match ciede.c output.
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

#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"
#include "../../vulkan/kernel_template.h"

#include "ciede_spv.h" /* generated SPIR-V byte array (shared with ciede_vulkan.c) */

#define INT_CIEDE_WG_X 16
#define INT_CIEDE_WG_Y 8
#define INT_CIEDE_NUM_BINDINGS 7 /* 6 input planes + 1 partial-sums output */

typedef struct {
    unsigned width;
    unsigned height;
    unsigned bpc;
    enum VmafPixelFormat pix_fmt;

    uint32_t wg_count_x;
    uint32_t wg_count_y;
    uint32_t wg_count;

    VmafVulkanContext *ctx;
    int owns_ctx;

    VmafVulkanKernelPipeline pl;
    VmafVulkanKernelSubmitPool sub_pool;
    VkDescriptorSet pre_set;

    VmafVulkanBuffer *ref_y_in;
    VmafVulkanBuffer *ref_u_in;
    VmafVulkanBuffer *ref_v_in;
    VmafVulkanBuffer *dis_y_in;
    VmafVulkanBuffer *dis_u_in;
    VmafVulkanBuffer *dis_v_in;
    VmafVulkanBuffer *partials;

    VmafDictionary *feature_name_dict;
} IntCiedeVulkanState;

static const VmafOption options[] = {{0}};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} IntCiedePushConsts;

static int create_pipeline(IntCiedeVulkanState *s)
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
        .ssbo_binding_count = (uint32_t)INT_CIEDE_NUM_BINDINGS,
        .push_constant_size = (uint32_t)sizeof(IntCiedePushConsts),
        .spv_bytes = ciede_spv,
        .spv_size = ciede_spv_size,
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

static int alloc_buffers(IntCiedeVulkanState *s)
{
    size_t bytes_per_pixel = (s->bpc <= 8) ? 1 : 2;
    size_t plane_bytes = (size_t)s->width * s->height * bytes_per_pixel;

    int err = 0;
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_y_in, plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_u_in, plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_v_in, plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_y_in, plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_u_in, plane_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_v_in, plane_bytes);
    if (err)
        return -ENOMEM;

    size_t partials_bytes = (size_t)s->wg_count * sizeof(float);
    return vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->partials, partials_bytes);
}

static int write_descriptor_set(IntCiedeVulkanState *s, VkDescriptorSet set)
{
    VkDescriptorBufferInfo dbi[INT_CIEDE_NUM_BINDINGS] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_y_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_u_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_v_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_y_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_u_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_v_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[INT_CIEDE_NUM_BINDINGS];
    for (int i = 0; i < INT_CIEDE_NUM_BINDINGS; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, INT_CIEDE_NUM_BINDINGS, writes, 0, NULL);
    return 0;
}

static void upscale_plane_8(unsigned p, const VmafPicture *pic, void *dst, unsigned out_w,
                            unsigned out_h, enum VmafPixelFormat pix_fmt)
{
    const int ss_hor = (p > 0u) && (pix_fmt != VMAF_PIX_FMT_YUV444P);
    const int ss_ver = (p > 0u) && (pix_fmt == VMAF_PIX_FMT_YUV420P);
    const uint8_t *in_buf = pic->data[p];
    uint8_t *out_buf = dst;
    for (unsigned i = 0; i < out_h; i++) {
        for (unsigned j = 0; j < out_w; j++) {
            unsigned in_x = ss_hor ? (j >> 1) : j;
            out_buf[j] = in_buf[in_x];
        }
        unsigned in_row_step = ss_ver ? (i & 1u) : 1u;
        in_buf += in_row_step * pic->stride[p];
        out_buf += out_w;
    }
}

static void upscale_plane_16(unsigned p, const VmafPicture *pic, void *dst, unsigned out_w,
                             unsigned out_h, enum VmafPixelFormat pix_fmt)
{
    const int ss_hor = (p > 0u) && (pix_fmt != VMAF_PIX_FMT_YUV444P);
    const int ss_ver = (p > 0u) && (pix_fmt == VMAF_PIX_FMT_YUV420P);
    const uint16_t *in_buf = pic->data[p];
    uint16_t *out_buf = dst;
    const ptrdiff_t in_stride16 = pic->stride[p] / 2;
    for (unsigned i = 0; i < out_h; i++) {
        for (unsigned j = 0; j < out_w; j++) {
            unsigned in_x = ss_hor ? (j >> 1) : j;
            out_buf[j] = in_buf[in_x];
        }
        unsigned in_row_step = ss_ver ? (i & 1u) : 1u;
        in_buf += in_row_step * in_stride16;
        out_buf += out_w;
    }
}

static int upload_pic(IntCiedeVulkanState *s, VmafVulkanBuffer *y_buf, VmafVulkanBuffer *u_buf,
                      VmafVulkanBuffer *v_buf, VmafPicture *pic)
{
    void *y_dst = vmaf_vulkan_buffer_host(y_buf);
    void *u_dst = vmaf_vulkan_buffer_host(u_buf);
    void *v_dst = vmaf_vulkan_buffer_host(v_buf);
    if (s->bpc <= 8) {
        upscale_plane_8(0, pic, y_dst, s->width, s->height, s->pix_fmt);
        upscale_plane_8(1, pic, u_dst, s->width, s->height, s->pix_fmt);
        upscale_plane_8(2, pic, v_dst, s->width, s->height, s->pix_fmt);
    } else {
        upscale_plane_16(0, pic, y_dst, s->width, s->height, s->pix_fmt);
        upscale_plane_16(1, pic, u_dst, s->width, s->height, s->pix_fmt);
        upscale_plane_16(2, pic, v_dst, s->width, s->height, s->pix_fmt);
    }
    int err = 0;
    err |= vmaf_vulkan_buffer_flush(s->ctx, y_buf);
    err |= vmaf_vulkan_buffer_flush(s->ctx, u_buf);
    err |= vmaf_vulkan_buffer_flush(s->ctx, v_buf);
    return err;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;

    IntCiedeVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->pix_fmt = pix_fmt;
    s->wg_count_x = (w + INT_CIEDE_WG_X - 1u) / INT_CIEDE_WG_X;
    s->wg_count_y = (h + INT_CIEDE_WG_Y - 1u) / INT_CIEDE_WG_Y;
    s->wg_count = s->wg_count_x * s->wg_count_y;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "integer_ciede_vulkan: cannot create Vulkan context (%d)\n", err);
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

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    IntCiedeVulkanState *s = fex->priv;
    int err = 0;

    err = upload_pic(s, s->ref_y_in, s->ref_u_in, s->ref_v_in, ref_pic);
    if (err)
        return err;
    err = upload_pic(s, s->dis_y_in, s->dis_u_in, s->dis_v_in, dist_pic);
    if (err)
        return err;

    memset(vmaf_vulkan_buffer_host(s->partials), 0, (size_t)s->wg_count * sizeof(float));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->partials);
    if (err)
        return err;

    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    IntCiedePushConsts pc = {
        .width = s->width,
        .height = s->height,
        .bpc = s->bpc,
        .num_workgroups_x = s->wg_count_x,
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1,
                            &s->pre_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, s->wg_count_x, s->wg_count_y, 1);

    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    {
        int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->partials);
        if (err_inv) {
            vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
            return err_inv;
        }
        const float *partials = vmaf_vulkan_buffer_host(s->partials);
        double total = 0.0;
        for (unsigned i = 0; i < s->wg_count; i++)
            total += (double)partials[i];
        const double n_pixels = (double)s->width * (double)s->height;
        const double mean_de = total / n_pixels;
        const double score = 45.0 - 20.0 * log10(mean_de);

        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "ciede2000", score, index);
    }

cleanup:
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    IntCiedeVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;

    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->ref_y_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_y_in);
    if (s->ref_u_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_u_in);
    if (s->ref_v_in)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_v_in);
    if (s->dis_y_in)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_y_in);
    if (s->dis_u_in)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_u_in);
    if (s->dis_v_in)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_v_in);
    if (s->partials)
        vmaf_vulkan_buffer_free(s->ctx, s->partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {"ciede2000", NULL};

VmafFeatureExtractor vmaf_fex_integer_ciede_vulkan = {
    .name = "integer_ciede_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(IntCiedeVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
