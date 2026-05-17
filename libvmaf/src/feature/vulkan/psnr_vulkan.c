/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature kernel on the Vulkan backend (T7-23 / ADR-0182,
 *  GPU long-tail batch 1; chroma extension T3-15(b) / ADR-0216).
 *
 *  Per-pixel squared-error reduction → host-side log10 → score.
 *  One dispatch per plane (Y, Cb, Cr); the same `psnr.comp`
 *  pipeline is invoked three times per frame against per-plane
 *  buffers and per-plane (width, height) push-constants. Chroma
 *  buffers are sized for the active subsampling
 *  (4:2:0 → w/2 × h/2, 4:2:2 → w/2 × h, 4:4:4 → w × h); the
 *  shader is plane-agnostic and reads its dims out of push
 *  constants.
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_psnr.c::extract):
 *      sse = sum_{i,j} (ref[i,j] - dis[i,j])^2;        (per channel)
 *      mse = sse / (w_p * h_p);
 *      psnr = (sse == 0)
 *             ? psnr_max[p]
 *             : 10 * log10(peak * peak / mse);
 *  Bit-exactness contract: int64 SSE accumulation → places=4 vs CPU.
 *
 *  Pattern reference: libvmaf/src/feature/vulkan/motion_vulkan.c
 *  (single-dispatch + per-WG int64 reduction). PSNR remains the
 *  simplest Vulkan extractor in the matrix — 3 small dispatches/
 *  frame, no temporal state.
 *
 *  4:0:0 (YUV400) handling: chroma planes are absent, so only the
 *  luma plane is dispatched and only `psnr_y` is emitted. This
 *  matches CPU integer_psnr.c::init's `enable_chroma = false`
 *  branch.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
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

#include "psnr_spv.h" /* generated SPIR-V byte array */

#define PSNR_WG_X 16
#define PSNR_WG_Y 8
#define PSNR_NUM_PLANES 3

typedef struct {
    /* Per-plane geometry: [0] luma (full), [1] Cb, [2] Cr (subsampled per pix_fmt). */
    unsigned width[PSNR_NUM_PLANES];
    unsigned height[PSNR_NUM_PLANES];
    unsigned bpc;
    uint32_t peak;
    double psnr_max[PSNR_NUM_PLANES];

    /* `enable_chroma` option: when false, only luma is dispatched.
     * Default true mirrors CPU integer_psnr.c — see ADR-0453. */
    bool enable_chroma;
    /* Number of active planes (1 for YUV400, 3 otherwise). */
    unsigned n_planes;

    /* Vulkan context handle. Borrow on imported state, lazy-create otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects (`vulkan/kernel_template.h` bundle).
     *
     * The shader is plane-agnostic — push constants carry the
     * per-plane width/height/num_workgroups_x. One pipeline suffices
     * for all three dispatches; spec-constants pin the *max* (luma)
     * frame dims for the shared-memory size, runtime guards on
     * push-constant width/height inside the shader.
     *
     * `pl.dsl` / `pl.pipeline_layout` / `pl.shader` / `pl.pipeline` /
     * `pl.desc_pool` are dispensed by the template's
     * `vmaf_vulkan_kernel_pipeline_create()` and torn down via
     * `vmaf_vulkan_kernel_pipeline_destroy()`. */
    VmafVulkanKernelPipeline pl;

    /* Per-frame submit pool (T-GPU-OPT-VK-1 / ADR-0256).
     * Single slot: all n_planes dispatches share one command buffer. */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor sets, one per active plane.
     * All buffer handles (ref_in/dis_in/se_partials) are stable from
     * init() onward; sets are written once at init and reused every
     * frame without any vkUpdateDescriptorSets in the hot path. */
    VkDescriptorSet pre_sets[PSNR_NUM_PLANES];

    /* Per-plane input buffers (host-mapped). */
    VmafVulkanBuffer *ref_in[PSNR_NUM_PLANES];
    VmafVulkanBuffer *dis_in[PSNR_NUM_PLANES];

    /* Per-plane per-workgroup int64 SE partials. */
    VmafVulkanBuffer *se_partials[PSNR_NUM_PLANES];
    unsigned wg_count[PSNR_NUM_PLANES];

    VmafDictionary *feature_name_dict;
} PsnrVulkanState;

static const VmafOption options[] = {{
                                         .name = "enable_chroma",
                                         .help = "enable calculation for chroma channels",
                                         .offset = offsetof(PsnrVulkanState, enable_chroma),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = true,
                                     },
                                     {0}};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} PsnrPushConsts;

static inline void psnr_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + PSNR_WG_X - 1u) / PSNR_WG_X;
    *gy = (h + PSNR_WG_Y - 1u) / PSNR_WG_Y;
}

static int create_pipeline(PsnrVulkanState *s)
{
    /* Spec constants pin the *max* per-plane dims (luma) for the
     * shared-memory size; runtime guards on push-constant width/height
     * inside the shader, so chroma's smaller dispatches reuse the same
     * pipeline safely. */
    struct {
        int32_t width;
        int32_t height;
        int32_t bpc;
        int32_t subgroup_size;
    } spec_data = {(int32_t)s->width[0], (int32_t)s->height[0], (int32_t)s->bpc, 32};

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

    /* `vulkan/kernel_template.h` (ADR-0246) owns the boilerplate:
     * descriptor-set layout (3 SSBO bindings — ref, dis, SE partials),
     * pipeline layout (1 set + 1 push-constant range), shader module
     * from psnr_spv, compute pipeline (caller-supplied spec-constants),
     * and descriptor pool (12 sets × 3 buffers = 36 descriptors —
     * matches the original psnr_vulkan.c heuristic of 4 frames in
     * flight × 3 planes). */
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 3U,
        .push_constant_size = (uint32_t)sizeof(PsnrPushConsts),
        .spv_bytes = psnr_spv,
        .spv_size = psnr_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &spec_info,
                    },
            },
        .max_descriptor_sets = 12U,
    };
    return vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
}

static int alloc_buffers(PsnrVulkanState *s)
{
    const size_t bytes_per_pixel = (s->bpc <= 8) ? 1U : 2U;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const size_t in_bytes = (size_t)s->width[p] * s->height[p] * bytes_per_pixel;
        int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in[p], in_bytes);
        if (err)
            return err;
        err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_in[p], in_bytes);
        if (err)
            return err;

        uint32_t gx = 0;
        uint32_t gy = 0;
        psnr_wg_dims(s->width[p], s->height[p], &gx, &gy);
        s->wg_count[p] = gx * gy;
        size_t se_bytes = (size_t)s->wg_count[p] * sizeof(int64_t);
        if (se_bytes == 0)
            se_bytes = sizeof(int64_t);
        err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->se_partials[p], se_bytes);
        if (err)
            return err;
    }
    return 0;
}

/* Forward declaration — see comment in adm_vulkan.c (same pattern). */
static int write_descriptor_set(PsnrVulkanState *s, VkDescriptorSet set, unsigned plane);

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    PsnrVulkanState *s = fex->priv;
    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;

    /* Per-plane geometry derived from pix_fmt. CPU reference:
     * libvmaf/src/feature/integer_psnr.c::init computes the same
     * (ss_hor, ss_ver) split. YUV400 has chroma absent, so n_planes = 1. */
    s->width[0] = w;
    s->height[0] = h;
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        s->n_planes = 1;
        s->width[1] = s->width[2] = 0;
        s->height[1] = s->height[2] = 0;
    } else {
        s->n_planes = PSNR_NUM_PLANES;
        const int ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P);
        const int ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P);
        const unsigned cw = ss_hor ? (w / 2U) : w;
        const unsigned ch = ss_ver ? (h / 2U) : h;
        s->width[1] = s->width[2] = cw;
        s->height[1] = s->height[2] = ch;
    }
    /* Mirror CPU integer_psnr.c::init's enable_chroma guard (ADR-0453):
     * when the caller passes enable_chroma=false, skip chroma dispatches
     * identically to the YUV400 path above. YUV400 already forces
     * n_planes=1, so this only activates for 4:2:0/4:2:2/4:4:4. */
    if (!s->enable_chroma && s->n_planes > 1U) {
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    }

    /* Match CPU integer_psnr.c::init's psnr_max default branch
     * (`min_sse == 0.0`): psnr_max[p] = (6 * bpc) + 12. The CPU path
     * uses a per-plane vector to leave room for the `min_sse`-driven
     * formula; we replicate the array even though all three entries
     * are identical in the default branch, so a future `min_sse`
     * option flip stays a one-line change. */
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++)
        s->psnr_max[p] = (double)(6U * bpc) + 12.0;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_vulkan: cannot create Vulkan context (%d)\n", err);
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

    /* Pre-allocate submit pool and descriptor sets (T-GPU-OPT-VK-1 +
     * T-GPU-OPT-VK-4 / ADR-0256). All SSBO handles are stable from
     * this point forward, so the sets are fully written here and
     * reused every frame without any per-frame vkUpdateDescriptorSets. */
    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl, s->n_planes,
                                                   s->pre_sets);
    if (err)
        return err;
    for (unsigned p = 0; p < s->n_planes; p++)
        (void)write_descriptor_set(s, s->pre_sets[p], p);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;
}

static int upload_plane(PsnrVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic,
                        unsigned plane)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(dst_buf);
    const uint8_t *src = (const uint8_t *)pic->data[plane];
    const size_t src_stride = (size_t)pic->stride[plane];
    const unsigned w = s->width[plane];
    const unsigned h = s->height[plane];
    const size_t dst_stride = (s->bpc <= 8) ? (size_t)w : (size_t)w * 2U;
    for (unsigned y = 0; y < h; y++)
        memcpy(dst + (size_t)y * dst_stride, src + (size_t)y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int write_descriptor_set(PsnrVulkanState *s, VkDescriptorSet set, unsigned plane)
{
    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_in[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->se_partials[plane]),
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

static double reduce_se_partials(const PsnrVulkanState *s, unsigned plane)
{
    int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->se_partials[plane]);
    if (err_inv)
        return err_inv;
    const int64_t *slots = vmaf_vulkan_buffer_host(s->se_partials[plane]);
    int64_t total = 0;
    for (unsigned i = 0; i < s->wg_count[plane]; i++)
        total += slots[i];
    return (double)total;
}

/* psnr_name[p] — same array as the CPU path
 * (libvmaf/src/feature/integer_psnr.c::psnr_name). */
static const char *const psnr_name[PSNR_NUM_PLANES] = {"psnr_y", "psnr_cb", "psnr_cr"};

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    PsnrVulkanState *s = fex->priv;
    int err = 0;

    /* 1) Host → device upload + zero out partials, per active plane. */
    for (unsigned p = 0; p < s->n_planes; p++) {
        err = upload_plane(s, s->ref_in[p], ref_pic, p);
        if (err)
            return err;
        err = upload_plane(s, s->dis_in[p], dist_pic, p);
        if (err)
            return err;

        memset(vmaf_vulkan_buffer_host(s->se_partials[p]), 0,
               (size_t)s->wg_count[p] * sizeof(int64_t));
        err = vmaf_vulkan_buffer_flush(s->ctx, s->se_partials[p]);
        if (err)
            return err;
    }

    /* 2) Acquire a pre-allocated command buffer + fence from the pool
     * (T-GPU-OPT-VK-1 / ADR-0256). Pre-allocated descriptor sets
     * (written once at init — all SSBO handles are stable) are reused
     * directly; no per-frame vkAllocate/Free on any resource. */
    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline);
    for (unsigned p = 0; p < s->n_planes; p++) {
        uint32_t gx = 0;
        uint32_t gy = 0;
        psnr_wg_dims(s->width[p], s->height[p], &gx, &gy);
        PsnrPushConsts pc = {
            .width = s->width[p],
            .height = s->height[p],
            .bpc = s->bpc,
            .num_workgroups_x = gx,
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1,
                                &s->pre_sets[p], 0, NULL);
        vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                           &pc);
        vkCmdDispatch(cmd, gx, gy, 1);
    }
    /* End recording, submit and wait synchronously. */
    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    /* 3) Host-side reduction + score emit, per active plane.
     * Mirrors integer_psnr.c::psnr emission semantics. */
    const double peak_sq = (double)s->peak * (double)s->peak;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const double sse = reduce_se_partials(s, p);
        const double n_pixels = (double)s->width[p] * (double)s->height[p];
        const double mse = sse / n_pixels;
        double psnr = (sse <= 0.0) ? s->psnr_max[p] : 10.0 * log10(peak_sq / mse);
        if (psnr > s->psnr_max[p])
            psnr = s->psnr_max[p];

        const int e = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, psnr_name[p], psnr, index);
        if (e && !err)
            err = e;
    }

cleanup:
    /* Pool-owned submit: submit_free clears local handles only; pool
     * retains the fence + cmd buffer for reuse next frame. */
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    PsnrVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;

    /* Drain the submit pool before the pipeline (and its descriptor
     * pool) are torn down (ADR-0256 / T-GPU-OPT-VK-1). */
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);

    /* `vulkan/kernel_template.h` (ADR-0246) collapses the
     * vkDeviceWaitIdle + 5x vkDestroy* sweep into one call. The
     * descriptor sets pre-allocated via descriptor_sets_alloc are
     * freed implicitly with the pool — do NOT call vkFreeDescriptorSets
     * on pre_sets. */
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++) {
        if (s->ref_in[p])
            vmaf_vulkan_buffer_free(s->ctx, s->ref_in[p]);
        if (s->dis_in[p])
            vmaf_vulkan_buffer_free(s->ctx, s->dis_in[p]);
        if (s->se_partials[p])
            vmaf_vulkan_buffer_free(s->ctx, s->se_partials[p]);
    }

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

/* Provided features — full luma + chroma per T3-15(b) / ADR-0216.
 * For YUV400 sources `init` clamps `n_planes` to 1 and chroma
 * dispatches are skipped, so only `psnr_y` is emitted at runtime —
 * but the static list still claims chroma so the dispatcher routes
 * `psnr_cb` / `psnr_cr` requests through the Vulkan twin. */
static const char *provided_features[] = {"psnr_y", "psnr_cb", "psnr_cr", NULL};

VmafFeatureExtractor vmaf_fex_psnr_vulkan = {
    .name = "psnr_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(PsnrVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    /* 3 small dispatches/frame (Y + Cb + Cr in one command buffer),
     * reduction-dominated; AUTO + 1080p area matches motion's
     * profile (see ADR-0181 / ADR-0182 / ADR-0216). */
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
