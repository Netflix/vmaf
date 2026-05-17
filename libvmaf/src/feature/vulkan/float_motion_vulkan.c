/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_motion feature kernel on the Vulkan backend (T7-23 / batch 3
 *  part 4a — ADR-0192 / ADR-0196). Float twin of motion_vulkan: same
 *  ping-pong of blurred ref planes, but the blur is done with FILTER_5_s
 *  (float, sums to 1.0) and SAD is computed in float.
 *
 *  Frame 0: blur into blur[0], emit motion2=0 (and motion=0 if debug).
 *  Frame >= 1: blur into blur[index%2], SAD against blur[(index+1)%2],
 *              emit motion = sad / (w*h). motion2 = min(prev, cur)
 *              emitted at index-1 (delayed-by-one pattern).
 *
 *  Bit-exactness disclaimer: float convolution + per-WG float SAD
 *  reduction + double host accumulation. ADR-0192 nominal contract:
 *  places=3.
 *
 *  Submit-pool migration (T-GPU-OPT-VK-1 / ADR-0353 / PR-B):
 *  Replaced per-frame vkAllocateCommandBuffers + vkCreateFence +
 *  vkAllocateDescriptorSets with:
 *    - vmaf_vulkan_kernel_submit_pool_create / _acquire / _end_and_wait
 *    - vmaf_vulkan_kernel_descriptor_sets_alloc (one set at init)
 *  The set is updated once per frame because the blur ping-pong flips
 *  which buffer is "current" vs "previous" — the buffer handles are
 *  stable from init() onward (mirrors motion_vulkan.c pattern).
 *  (T-GPU-OPT-VK-4 partial / ADR-0353.)
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

#include "float_motion_spv.h"

#define FMOTION_WG_X 32
#define FMOTION_WG_Y 4
#define FMOTION_NUM_BINDINGS 4

typedef struct {
    bool debug;
    bool motion_force_zero;

    unsigned width;
    unsigned height;
    unsigned bpc;

    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects (`vulkan/kernel_template.h` bundle, ADR-0246). */
    VmafVulkanKernelPipeline pl;

    /* Per-frame submit pool (T-GPU-OPT-VK-1 / ADR-0353).
     * Single slot: the single dispatch uses one command buffer. */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor set — updated once per frame because
     * blur[cur/prev] ping-pong flips (buffer handles are stable from
     * init() onward, only the cur/prev assignment changes). */
    VkDescriptorSet pre_set;

    VmafVulkanBuffer *ref_in;
    VmafVulkanBuffer *blur[2];
    int cur_blur;

    VmafVulkanBuffer *sad_partials;
    unsigned wg_count;

    unsigned frame_index;
    double prev_motion_score;
    double motion_fps_weight;

    VmafDictionary *feature_name_dict;
} FloatMotionVulkanState;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t compute_sad;
    uint32_t num_workgroups_x;
} FloatMotionPushConsts;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(FloatMotionVulkanState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "force motion score to zero",
        .offset = offsetof(FloatMotionVulkanState, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(FloatMotionVulkanState, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}};

static inline void fmotion_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + FMOTION_WG_X - 1u) / FMOTION_WG_X;
    *gy = (h + FMOTION_WG_Y - 1u) / FMOTION_WG_Y;
}

static int create_pipelines(FloatMotionVulkanState *s)
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
        .ssbo_binding_count = (uint32_t)FMOTION_NUM_BINDINGS,
        .push_constant_size = (uint32_t)sizeof(FloatMotionPushConsts),
        .spv_bytes = float_motion_spv,
        .spv_size = float_motion_spv_size,
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

static int alloc_buffers(FloatMotionVulkanState *s)
{
    size_t bpp = (s->bpc <= 8) ? 1 : 2;
    size_t in_bytes = (size_t)s->width * s->height * bpp;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in, in_bytes);
    if (err)
        return err;

    /* Blurred plane = float per pixel. */
    size_t blur_bytes = (size_t)s->width * s->height * sizeof(float);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->blur[0], blur_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->blur[1], blur_bytes);
    if (err)
        return err;

    uint32_t gx = 0;
    uint32_t gy = 0;
    fmotion_wg_dims(s->width, s->height, &gx, &gy);
    s->wg_count = gx * gy;
    size_t sad_bytes = (size_t)s->wg_count * sizeof(float);
    if (sad_bytes == 0)
        sad_bytes = sizeof(float);
    return vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->sad_partials, sad_bytes);
}

static int write_descriptor_set(FloatMotionVulkanState *s, VkDescriptorSet set)
{
    int cur = s->cur_blur;
    int prev = 1 - cur;
    VkDescriptorBufferInfo dbi[4] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->blur[cur]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->blur[prev]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->sad_partials),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 4, writes, 0, NULL);
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    FloatMotionVulkanState *s = fex->priv;

    /* The 5-tap Vulkan float_motion shader uses reflect-101 mirror padding;
     * the mirror formula requires dim >= 3 in both axes.  Refuse smaller
     * frames up front to prevent out-of-bounds reads in the shader.
     * Minimum: filter_width/2 + 1 = 3. */
    if (h < 3u || w < 3u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "float_motion_vulkan: frame %ux%u is below the 5-tap filter minimum 3x3; "
                 "refusing to avoid out-of-bounds mirror reads in shader\n",
                 w, h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->frame_index = 0;
    s->prev_motion_score = 0.0;
    s->cur_blur = 0;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "float_motion_vulkan: cannot create Vulkan context (%d)\n", err);
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

    /* Pre-allocate the submit pool and the single persistent descriptor
     * set (T-GPU-OPT-VK-1 + T-GPU-OPT-VK-4 partial / ADR-0353).
     * The set is written per-frame in extract() because blur[cur/prev]
     * flips each frame; the buffer *handles* are stable from here
     * forward. (Mirrors motion_vulkan.c / ADR-0256 pattern.) */
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl,
                                                   /*count=*/1, &s->pre_set);
    if (err)
        return err;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;
}

static int upload_ref(FloatMotionVulkanState *s, VmafPicture *pic)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(s->ref_in);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, s->ref_in);
}

static double reduce_sad_partials(const FloatMotionVulkanState *s)
{
    int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->sad_partials);
    if (err_inv)
        return err_inv;
    const float *slots = vmaf_vulkan_buffer_host(s->sad_partials);
    double total = 0.0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total += (double)slots[i];
    return total / ((double)s->width * s->height);
}

static int extract_force_zero(FloatMotionVulkanState *s, unsigned index, VmafFeatureCollector *fc)
{
    int err = 0;
    if (s->frame_index > 0) {
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                       "VMAF_feature_motion_score", 0.0, index);
    }
    err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                   "VMAF_feature_motion2_score", 0.0, index);
    s->frame_index++;
    return err;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    FloatMotionVulkanState *s = fex->priv;
    int err = 0;

    if (s->motion_force_zero)
        return extract_force_zero(s, index, feature_collector);

    err = upload_ref(s, ref_pic);
    if (err)
        return err;

    memset(vmaf_vulkan_buffer_host(s->sad_partials), 0, (size_t)s->wg_count * sizeof(float));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->sad_partials);
    if (err)
        return err;

    /* Update the pre-allocated descriptor set with the current-frame
     * blur ping-pong binding (cur_blur flips each frame — the handles
     * are stable but which slot is "cur" vs "prev" changes). */
    (void)write_descriptor_set(s, s->pre_set);

    /* Acquire a pre-allocated command buffer + fence from the pool
     * (T-GPU-OPT-VK-1 / ADR-0353). No per-frame allocation. */
    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    uint32_t gx = 0;
    uint32_t gy = 0;
    fmotion_wg_dims(s->width, s->height, &gx, &gy);
    int compute_sad = (s->frame_index > 0) ? 1 : 0;
    FloatMotionPushConsts pc = {
        .width = s->width,
        .height = s->height,
        .bpc = s->bpc,
        .compute_sad = (uint32_t)compute_sad,
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

    {
        double motion_score = 0.0;
        if (s->frame_index > 0)
            motion_score = reduce_sad_partials(s);

        /* Match CPU motion algorithm:
         *   index 0:        write motion2 = 0 (and motion = 0 in debug). No prior frame.
         *   index 1:        emit motion (debug). motion2 not yet computable.
         *   index >= 2:     emit motion2[index-1] = min(prev, cur). emit motion (debug).
         */
        if (s->frame_index == 0) {
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_feature_motion2_score", 0.0, index);
            if (s->debug && !err)
                err = vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict, "VMAF_feature_motion_score", 0.0,
                    index);
        } else if (s->frame_index == 1) {
            if (s->debug)
                err = vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict, "VMAF_feature_motion_score",
                    motion_score, index);
        } else {
            /* Apply fps weight before taking the min — mirrors float_motion.c CPU path.
             * Bit-exact when motion_fps_weight = 1.0 (default). */
            const double w_cur = motion_score * s->motion_fps_weight;
            const double w_prev = s->prev_motion_score * s->motion_fps_weight;
            const double motion2 = (w_cur < w_prev) ? w_cur : w_prev;
            err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                          "VMAF_feature_motion2_score", motion2,
                                                          index - 1);
            if (s->debug && !err)
                err = vmaf_feature_collector_append_with_dict(
                    feature_collector, s->feature_name_dict, "VMAF_feature_motion_score",
                    motion_score, index);
        }

        s->prev_motion_score = motion_score;
    }

    s->cur_blur = 1 - s->cur_blur;
    s->frame_index++;

cleanup:
    /* Pool-owned submit: submit_free is a near-no-op that just clears
     * the local handles; the pool keeps cmd + fence alive for reuse. */
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int flush(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    FloatMotionVulkanState *s = fex->priv;
    int ret = 0;
    if (s->motion_force_zero)
        return 1;

    if (s->frame_index > 1) {
        /* Apply fps weight on the tail motion2 — mirrors the extract path.
         * Bit-exact when motion_fps_weight = 1.0 (default). */
        ret = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "VMAF_feature_motion2_score",
            s->prev_motion_score * s->motion_fps_weight, s->frame_index - 1);
    }
    return (ret < 0) ? ret : !ret;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    FloatMotionVulkanState *s = fex->priv;
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
    if (s->blur[0])
        vmaf_vulkan_buffer_free(s->ctx, s->blur[0]);
    if (s->blur[1])
        vmaf_vulkan_buffer_free(s->ctx, s->blur[1]);
    if (s->sad_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->sad_partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {"VMAF_feature_motion_score", "VMAF_feature_motion2_score",
                                          NULL};

VmafFeatureExtractor vmaf_fex_float_motion_vulkan = {
    .name = "float_motion_vulkan",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(FloatMotionVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
