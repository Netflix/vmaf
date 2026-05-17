/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_psnr feature extractor on the Vulkan backend.
 *
 *  This is the explicit integer-named Vulkan PSNR twin, complementing
 *  vmaf_fex_psnr_vulkan (which uses the name "psnr_vulkan" and shares the
 *  psnr.comp SPIR-V module).  The two extractors provide the same integer-
 *  domain SSE algorithm and the same feature names (psnr_y / psnr_cb /
 *  psnr_cr), but vmaf_fex_integer_psnr_vulkan is selectable by the explicit
 *  name "integer_psnr_vulkan" — mirroring the CUDA split:
 *    vmaf_fex_psnr_cuda        → "psnr_cuda"        (integer)
 *    vmaf_fex_integer_psnr_vulkan → "integer_psnr_vulkan" (integer, Vulkan)
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_psnr.c::extract):
 *      sse = sum_{i,j} (ref[i,j] - dis[i,j])^2;        (per channel, int64)
 *      mse = sse / (w_p * h_p);
 *      psnr = (sse <= 0)
 *             ? psnr_max[p]
 *             : MIN(10 * log10(peak^2 / mse), psnr_max[p]);
 *  where peak = (1 << bpc) - 1 and psnr_max = 6 * bpc + 12.
 *
 *  Bit-exactness contract: int64 SSE accumulation → places=4 vs CPU.
 *
 *  Pattern reference: libvmaf/src/feature/vulkan/psnr_vulkan.c.
 *  Shader: shaders/integer_psnr.comp → integer_psnr_spv.h (embedded SPIR-V).
 *
 *  4:0:0 (YUV400) and enable_chroma=false: only luma is dispatched.
 *  This mirrors CPU integer_psnr.c::init's enable_chroma guard (ADR-0453).
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

#include "integer_psnr_spv.h" /* generated SPIR-V byte array */

#define IPSNR_WG_X 16
#define IPSNR_WG_Y 8
#define IPSNR_NUM_PLANES 3

typedef struct {
    /* Per-plane geometry: [0] luma (full), [1] Cb, [2] Cr (subsampled per pix_fmt). */
    unsigned width[IPSNR_NUM_PLANES];
    unsigned height[IPSNR_NUM_PLANES];
    unsigned bpc;
    uint32_t peak;
    double psnr_max[IPSNR_NUM_PLANES];

    /* `enable_chroma` option: when false, only luma is dispatched.
     * Default true mirrors CPU integer_psnr.c — see ADR-0453. */
    bool enable_chroma;
    /* Number of active planes (1 for YUV400 or enable_chroma=false, 3 otherwise). */
    unsigned n_planes;

    /* Vulkan context handle. Borrowed on imported state, lazy-created otherwise. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects (vulkan/kernel_template.h bundle, ADR-0246). */
    VmafVulkanKernelPipeline pl;

    /* Per-frame submit pool (T-GPU-OPT-VK-1 / ADR-0256).
     * Single slot: all n_planes dispatches share one command buffer. */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor sets, one per active plane.
     * All buffer handles (ref_in / dis_in / se_partials) are stable from
     * init() onward; sets are written once at init and reused every frame
     * without any vkUpdateDescriptorSets in the hot path. */
    VkDescriptorSet pre_sets[IPSNR_NUM_PLANES];

    /* Per-plane host-mapped input buffers. */
    VmafVulkanBuffer *ref_in[IPSNR_NUM_PLANES];
    VmafVulkanBuffer *dis_in[IPSNR_NUM_PLANES];

    /* Per-plane per-workgroup int64 SE partials (readback-mapped). */
    VmafVulkanBuffer *se_partials[IPSNR_NUM_PLANES];
    unsigned wg_count[IPSNR_NUM_PLANES];

    VmafDictionary *feature_name_dict;
} IntegerPsnrVulkanState;

static const VmafOption options[] = {{
                                         .name = "enable_chroma",
                                         .help = "enable calculation for chroma channels",
                                         .offset = offsetof(IntegerPsnrVulkanState, enable_chroma),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = true,
                                     },
                                     {0}};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t bpc;
    uint32_t num_workgroups_x;
} IntegerPsnrPushConsts;

static inline void ipsnr_wg_dims(unsigned w, unsigned h, uint32_t *gx, uint32_t *gy)
{
    *gx = (w + IPSNR_WG_X - 1u) / IPSNR_WG_X;
    *gy = (h + IPSNR_WG_Y - 1u) / IPSNR_WG_Y;
}

static int create_pipeline(IntegerPsnrVulkanState *s)
{
    /* Spec constants pin the *max* per-plane dims (luma) for the
     * shared-memory array size; runtime guards on push-constant width/height
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

    /* vulkan/kernel_template.h (ADR-0246) owns the boilerplate:
     * descriptor-set layout (3 SSBO bindings — ref, dis, SE partials),
     * pipeline layout (1 set + 1 push-constant range), shader module from
     * integer_psnr_spv, compute pipeline with caller-supplied spec-constants,
     * and descriptor pool (12 sets × 3 buffers = 36 descriptors). */
    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 3U,
        .push_constant_size = (uint32_t)sizeof(IntegerPsnrPushConsts),
        .spv_bytes = integer_psnr_spv,
        .spv_size = integer_psnr_spv_size,
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

static int alloc_buffers(IntegerPsnrVulkanState *s)
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
        ipsnr_wg_dims(s->width[p], s->height[p], &gx, &gy);
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

/* Forward declaration — same pattern as psnr_vulkan.c / adm_vulkan.c. */
static int write_descriptor_set(IntegerPsnrVulkanState *s, VkDescriptorSet set, unsigned plane);

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    IntegerPsnrVulkanState *s = fex->priv;
    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;

    /* Per-plane geometry derived from pix_fmt. CPU reference:
     * libvmaf/src/feature/integer_psnr.c::init computes the same
     * (ss_hor, ss_ver) split. YUV400 has chroma absent, so n_planes = 1. */
    s->width[0] = w;
    s->height[0] = h;
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    } else {
        s->n_planes = IPSNR_NUM_PLANES;
        const int ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P);
        const int ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P);
        const unsigned cw = ss_hor ? (w / 2U) : w;
        const unsigned ch = ss_ver ? (h / 2U) : h;
        s->width[1] = s->width[2] = cw;
        s->height[1] = s->height[2] = ch;
    }
    /* Mirror CPU integer_psnr.c::init's enable_chroma guard (ADR-0453):
     * when the caller passes enable_chroma=false, skip chroma dispatches. */
    if (!s->enable_chroma && s->n_planes > 1U) {
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    }

    /* Match CPU integer_psnr.c::init's psnr_max default branch
     * (`min_sse == 0.0`): psnr_max[p] = (6 * bpc) + 12. */
    for (unsigned p = 0; p < IPSNR_NUM_PLANES; p++)
        s->psnr_max[p] = (double)(6U * bpc) + 12.0;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "integer_psnr_vulkan: cannot create Vulkan context (%d)\n", err);
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
     * this point forward; the sets are fully written here and reused
     * every frame without any per-frame vkUpdateDescriptorSets. */
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

static int upload_plane(IntegerPsnrVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic,
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

static int write_descriptor_set(IntegerPsnrVulkanState *s, VkDescriptorSet set, unsigned plane)
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

static double reduce_se_partials(const IntegerPsnrVulkanState *s, unsigned plane)
{
    int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->se_partials[plane]);
    if (err_inv)
        return (double)err_inv;
    const int64_t *slots = vmaf_vulkan_buffer_host(s->se_partials[plane]);
    int64_t total = 0;
    for (unsigned i = 0; i < s->wg_count[plane]; i++)
        total += slots[i];
    return (double)total;
}

/* psnr_name[p] — same array as the CPU path
 * (libvmaf/src/feature/integer_psnr.c::psnr_name). */
static const char *const psnr_name[IPSNR_NUM_PLANES] = {"psnr_y", "psnr_cb", "psnr_cr"};

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    IntegerPsnrVulkanState *s = fex->priv;
    int err = 0;

    /* 1) Host → device upload + zero out SE partials, per active plane. */
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
     * (written once at init) are reused directly — no per-frame allocs. */
    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline);
    for (unsigned p = 0; p < s->n_planes; p++) {
        uint32_t gx = 0;
        uint32_t gy = 0;
        ipsnr_wg_dims(s->width[p], s->height[p], &gx, &gy);
        IntegerPsnrPushConsts pc = {
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
     * Mirrors integer_psnr.c::extract PSNR emission semantics. */
    {
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
    }

cleanup:
    /* Pool-owned submit: submit_free clears local handles only; pool
     * retains the fence + cmd buffer for reuse next frame. */
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    IntegerPsnrVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;

    /* Drain the submit pool before the pipeline (and its descriptor pool)
     * are torn down (ADR-0256 / T-GPU-OPT-VK-1). */
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);

    /* vulkan/kernel_template.h (ADR-0246) collapses the teardown sweep.
     * Descriptor sets pre-allocated via descriptor_sets_alloc are freed
     * implicitly with the pool — do NOT call vkFreeDescriptorSets on
     * pre_sets. */
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    for (unsigned p = 0; p < IPSNR_NUM_PLANES; p++) {
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
 * dispatches are skipped at runtime. */
static const char *provided_features[] = {"psnr_y", "psnr_cb", "psnr_cr", NULL};

VmafFeatureExtractor vmaf_fex_integer_psnr_vulkan = {
    .name = "integer_psnr_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(IntegerPsnrVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    /* 3 small dispatches/frame (Y + Cb + Cr in one command buffer),
     * reduction-dominated; mirrors psnr_vulkan.c profile (ADR-0216). */
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
