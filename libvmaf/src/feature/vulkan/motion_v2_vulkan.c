/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  motion_v2 feature kernel on the Vulkan backend (T7-23 / batch 3 part
 *  1a — ADR-0192 / ADR-0193).
 *
 *  Stateless variant of `motion_vulkan`: exploits convolution linearity
 *  (`SAD(blur(prev), blur(cur)) == sum(|blur(prev - cur)|)`) so we can
 *  compute the score in one dispatch over (prev_ref - cur_ref) without
 *  storing blurred frames across `extract` calls. Direct port of
 *  libvmaf/src/feature/integer_motion_v2.c.
 *
 *  Algorithm (mirrors CPU integer_motion_v2):
 *    1. V->H separable 5-tap Gaussian blur of (prev_ref - cur_ref)
 *       (signed; filter sum 65536; round 1<<(bpc-1) >> bpc, then
 *       round 1<<15 >> 16) — implemented in shaders/motion_v2.comp.
 *    2. Sum |blurred_diff| over the plane (per-WG int64 partial,
 *       host-side scalar reduction).
 *    3. motion_v2_sad_score = SAD / 256.0 / (width * height)
 *    4. motion2_v2_score = min(cur, next) emitted in flush(), reading
 *       back the collected sad scores (same shape as CPU's flush).
 *
 *  Pattern reference: motion_vulkan.c (the closest sibling — same
 *  separable filter, same int64 partials, same VkSpecializationInfo
 *  shape). motion_v2 simplifies by dropping the blur ping-pong (no
 *  blur output buffer to keep across frames) but adds a raw-pixel
 *  ping-pong (so we don't re-upload the previous frame each call).
 *
 *  Submit-pool migration (T-GPU-OPT-VK-1 / ADR-0353 / PR-B):
 *  Replaced per-frame vkAllocateCommandBuffers + vkCreateFence +
 *  vkAllocateDescriptorSets with:
 *    - vmaf_vulkan_kernel_submit_pool_create / _acquire / _end_and_wait
 *    - vmaf_vulkan_kernel_descriptor_sets_alloc (one set at init)
 *  The set is updated once per frame because the raw-pixel ping-pong
 *  flips which ref_buf slot is "current" vs "previous" — the buffer
 *  handles are stable from init() onward.
 *  (T-GPU-OPT-VK-4 partial / ADR-0353.)
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

#include "../../vulkan/kernel_template.h"
#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"

#include "motion_v2_spv.h" /* generated SPIR-V byte array */

#define MOTION_V2_WG_X 32
#define MOTION_V2_WG_Y 4
#define MOTION_V2_NUM_BINDINGS 3

typedef struct {
    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Vulkan context handle. Borrow on imported state, lazy-create otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Single pipeline — kernel always computes the same SAD. Frame 0
     * is short-circuited host-side without a dispatch. Owned by
     * `vulkan/kernel_template.h` (ADR-0246). */
    VmafVulkanKernelPipeline pl;

    /* Per-frame submit pool (T-GPU-OPT-VK-1 / ADR-0353).
     * Single slot: the single dispatch uses one command buffer. */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor set — updated once per frame because
     * ref_buf[cur/prev] ping-pong flips (buffer handles are stable
     * from init() onward, only the cur/prev assignment changes). */
    VkDescriptorSet pre_set;

    /* Ping-pong of raw ref planes. ref_buf[cur_ref_idx] = the frame
     * currently being processed; ref_buf[1 - cur_ref_idx] = previous. */
    VmafVulkanBuffer *ref_buf[2];
    int cur_ref_idx;

    /* Per-workgroup int64 SAD partials. */
    VmafVulkanBuffer *sad_partials;
    unsigned wg_count;

    unsigned frame_index;
    double motion_fps_weight;
    bool motion_force_zero;
    bool flushed; /* idempotency guard: flush() is a no-op after first call */

    VmafDictionary *feature_name_dict;
} MotionV2VulkanState;

static const VmafOption options[] = {
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(MotionV2VulkanState, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "force motion score to zero (mirrors CPU integer_motion_v2.c)",
        .offset = offsetof(MotionV2VulkanState, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}};

/* Push constants — must mirror `Params` in motion_v2.comp. */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} MotionV2PushConsts;

static inline void motion_v2_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + MOTION_V2_WG_X - 1u) / MOTION_V2_WG_X;
    *gy = (h + MOTION_V2_WG_Y - 1u) / MOTION_V2_WG_Y;
}

static int create_pipelines(MotionV2VulkanState *s)
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
        .ssbo_binding_count = (uint32_t)MOTION_V2_NUM_BINDINGS,
        .push_constant_size = (uint32_t)sizeof(MotionV2PushConsts),
        .spv_bytes = motion_v2_spv,
        .spv_size = motion_v2_spv_size,
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

static int alloc_buffers(MotionV2VulkanState *s)
{
    size_t bytes_per_pixel = (s->bpc <= 8) ? 1 : 2;
    size_t in_bytes = (size_t)s->width * s->height * bytes_per_pixel;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_buf[0], in_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_buf[1], in_bytes);
    if (err)
        return err;

    uint32_t gx = 0;
    uint32_t gy = 0;
    motion_v2_wg_dims(s->width, s->height, &gx, &gy);
    s->wg_count = gx * gy;
    size_t sad_bytes = (size_t)s->wg_count * sizeof(int64_t);
    if (sad_bytes == 0)
        sad_bytes = sizeof(int64_t);
    err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->sad_partials, sad_bytes);
    if (err)
        return err;

    return 0;
}

static int write_descriptor_set(MotionV2VulkanState *s, VkDescriptorSet set)
{
    int cur = s->cur_ref_idx;
    int prev = 1 - cur;

    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_buf[prev]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_buf[cur]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->sad_partials),
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
    MotionV2VulkanState *s = fex->priv;

    /* The 5-tap Vulkan motion_v2 shader uses reflect-101 mirror padding;
     * the mirror formula requires dim >= 3 in both axes.  Refuse smaller
     * frames up front to prevent out-of-bounds reads in the shader.
     * Minimum: filter_width/2 + 1 = 3. */
    if (h < 3u || w < 3u) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "motion_v2_vulkan: frame %ux%u is below the 5-tap filter minimum 3x3; "
                 "refusing to avoid out-of-bounds mirror reads in shader\n",
                 w, h);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    s->frame_index = 0;
    s->cur_ref_idx = 0;
    s->flushed = false;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "motion_v2_vulkan: cannot create Vulkan context (%d)\n",
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

    /* Pre-allocate the submit pool and the single persistent descriptor
     * set (T-GPU-OPT-VK-1 + T-GPU-OPT-VK-4 partial / ADR-0353).
     * The set is updated per-frame because the raw-pixel ping-pong flips
     * which slot is "cur" vs "prev"; the buffer handles themselves are
     * stable from here forward. */
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

static int upload_ref_plane(MotionV2VulkanState *s, VmafPicture *pic, int slot)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(s->ref_buf[slot]);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, s->ref_buf[slot]);
}

static double reduce_sad_partials(const MotionV2VulkanState *s)
{
    int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->sad_partials);
    if (err_inv)
        return err_inv;
    const int64_t *slots = vmaf_vulkan_buffer_host(s->sad_partials);
    int64_t total = 0;
    for (unsigned i = 0; i < s->wg_count; i++)
        total += slots[i];
    return (double)total / 256.0 / ((double)s->width * s->height);
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    MotionV2VulkanState *s = fex->priv;
    int err = 0;

    /* motion_force_zero: skip GPU work, emit 0 for every frame. */
    if (s->motion_force_zero) {
        s->frame_index++;
        return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "VMAF_integer_feature_motion_v2_sad_score",
                                                       0.0, index);
    }

    /* Frame 0: store the ref pixels (so frame 1 has a "prev"); emit 0. */
    if (s->frame_index == 0) {
        err = upload_ref_plane(s, ref_pic, s->cur_ref_idx);
        if (err)
            return err;
        s->cur_ref_idx = 1 - s->cur_ref_idx;
        s->frame_index++;
        return vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                       "VMAF_integer_feature_motion_v2_sad_score",
                                                       0.0, index);
    }

    /* Frame 1+: upload current ref into the slot we will treat as
     * "current"; previous frame's pixels are still in 1 - cur_ref_idx. */
    err = upload_ref_plane(s, ref_pic, s->cur_ref_idx);
    if (err)
        return err;

    memset(vmaf_vulkan_buffer_host(s->sad_partials), 0, (size_t)s->wg_count * sizeof(int64_t));
    err = vmaf_vulkan_buffer_flush(s->ctx, s->sad_partials);
    if (err)
        return err;

    /* Update the pre-allocated descriptor set with the current-frame
     * ping-pong binding (cur_ref_idx flips each frame — the handles
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
    motion_v2_wg_dims(s->width, s->height, &gx, &gy);
    MotionV2PushConsts pc = {
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

    {
        double sad_score = reduce_sad_partials(s);
        err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                      "VMAF_integer_feature_motion_v2_sad_score",
                                                      sad_score, index);
    }

    s->cur_ref_idx = 1 - s->cur_ref_idx;
    s->frame_index++;

cleanup:
    /* Pool-owned submit: submit_free is a near-no-op that just clears
     * the local handles; the pool keeps cmd + fence alive for reuse. */
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int flush(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    MotionV2VulkanState *s = fex->priv;

    /* Idempotency guard: framework may call flush() more than once on the
     * same instance (threaded dispatch path); only the first call emits. */
    if (s->flushed)
        return 1;
    s->flushed = true;

    unsigned n_frames = 0;
    double dummy;
    while (!vmaf_feature_collector_get_score(
        feature_collector, "VMAF_integer_feature_motion_v2_sad_score", &dummy, n_frames))
        n_frames++;

    if (n_frames < 2)
        return 1;

    for (unsigned i = 0; i < n_frames; i++) {
        double score_cur;
        double score_next;
        vmaf_feature_collector_get_score(feature_collector,
                                         "VMAF_integer_feature_motion_v2_sad_score", &score_cur, i);
        /* Apply fps weight — mirrors CPU integer_motion_v2.c flush logic.
         * Bit-exact when motion_fps_weight = 1.0 (default). */
        score_cur *= s->motion_fps_weight;

        double motion2;
        if (i + 1 < n_frames) {
            vmaf_feature_collector_get_score(
                feature_collector, "VMAF_integer_feature_motion_v2_sad_score", &score_next, i + 1);
            score_next *= s->motion_fps_weight;
            motion2 = score_cur < score_next ? score_cur : score_next;
        } else {
            motion2 = score_cur;
        }

        vmaf_feature_collector_append(feature_collector, "VMAF_integer_feature_motion2_v2_score",
                                      motion2, i);
    }

    return 1;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    MotionV2VulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;

    /* Drain the submit pool before the pipeline (ADR-0353).
     * Descriptor sets pre-allocated via descriptor_sets_alloc are
     * freed implicitly with the pool — do NOT call
     * vkFreeDescriptorSets on pre_set. */
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->ref_buf[0])
        vmaf_vulkan_buffer_free(s->ctx, s->ref_buf[0]);
    if (s->ref_buf[1])
        vmaf_vulkan_buffer_free(s->ctx, s->ref_buf[1]);
    if (s->sad_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->sad_partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features[] = {"VMAF_integer_feature_motion_v2_sad_score",
                                          "VMAF_integer_feature_motion2_v2_score", NULL};

VmafFeatureExtractor vmaf_fex_integer_motion_v2_vulkan = {
    .name = "motion_v2_vulkan",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(MotionV2VulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
