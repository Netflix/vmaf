/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_psnr_hvs feature extractor on the Vulkan backend.
 *  Port of feature/cuda/integer_psnr_hvs_cuda.c to Vulkan; Vulkan
 *  twin of psnr_hvs_vulkan.c (float path) using raw integer SSBO
 *  upload instead of host-side float pre-normalisation.
 *
 *  Design differences vs psnr_hvs_vulkan.c (float path):
 *    - Input buffers hold packed uint8 or uint16 samples (one uint
 *      element per sample), not float.  The host uploads with
 *      bytes_per_sample = (bpc <= 8 ? 1 : 2); no CPU normalisation.
 *    - The GLSL shader (integer_psnr_hvs.comp) reads raw integers
 *      directly and passes them to the integer DCT — matching the
 *      CPU calc_psnrhvs() input contract.
 *    - Emits "integer_psnr_hvs" (and per-plane siblings) consumed
 *      by the integer-path VMAF model.
 *
 *  Per-plane single-dispatch design (mirrors psnr_hvs_vulkan.c):
 *    - One workgroup per output 8×8 block (step=7 sliding window).
 *    - 64 threads/WG for cooperative load into shared memory.
 *    - Thread 0 performs all per-block arithmetic (serial DCT + mask).
 *    - 3 pipelines (one per plane; plane index baked via spec-constant).
 *
 *  Rejects YUV400P (no chroma) and bpc > 12 (matches CPU + CUDA twin).
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

#include "integer_psnr_hvs_spv.h" /* generated SPIR-V byte array */

#define IPSNR_HVS_WG_X 8
#define IPSNR_HVS_WG_Y 8
#define IPSNR_HVS_BLOCK 8
#define IPSNR_HVS_STEP 7
#define IPSNR_HVS_NUM_PLANES 3
#define IPSNR_HVS_NUM_BINDINGS 3 /* ref, dist, partials */

typedef struct {
    /* Frame geometry per plane. */
    unsigned width[IPSNR_HVS_NUM_PLANES];
    unsigned height[IPSNR_HVS_NUM_PLANES];
    unsigned num_blocks_x[IPSNR_HVS_NUM_PLANES];
    unsigned num_blocks_y[IPSNR_HVS_NUM_PLANES];
    unsigned num_blocks[IPSNR_HVS_NUM_PLANES];
    unsigned bpc;
    int32_t samplemax_sq;

    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipeline objects: base (plane=0) + two chroma variants. */
    VmafVulkanKernelPipeline pl;
    VkPipeline pipeline_chroma_u; /* plane=1 */
    VkPipeline pipeline_chroma_v; /* plane=2 */

    /* Single-slot submit pool (T-GPU-OPT-VK-1 / ADR-0256). */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor sets — one per plane, written once at
     * init() since SSBO handles are stable (VK-6 / perf-audit 2026-05-16). */
    VkDescriptorSet pre_sets[IPSNR_HVS_NUM_PLANES];

    /* Per-plane input buffers (raw integer) and partials readback. */
    VmafVulkanBuffer *ref_in[IPSNR_HVS_NUM_PLANES];
    VmafVulkanBuffer *dist_in[IPSNR_HVS_NUM_PLANES];
    VmafVulkanBuffer *partials[IPSNR_HVS_NUM_PLANES];

    VmafDictionary *feature_name_dict;
} IntPsnrHvsVulkanState;

static const VmafOption options[] = {{0}};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t num_blocks_x;
    uint32_t num_blocks_y;
} IntPsnrHvsPushConsts;

static inline VkPipeline plane_pipeline(const IntPsnrHvsVulkanState *s, int plane)
{
    if (plane == 0)
        return s->pl.pipeline;
    if (plane == 1)
        return s->pipeline_chroma_u;
    return s->pipeline_chroma_v;
}

struct IntPsnrHvsSpecData {
    int32_t bpc;
    int32_t plane;
    int32_t subgroup_size;
};

static void fill_spec(struct IntPsnrHvsSpecData *sd, VkSpecializationMapEntry entries[3],
                      VkSpecializationInfo *info, const IntPsnrHvsVulkanState *s, int plane)
{
    sd->bpc = (int32_t)s->bpc;
    sd->plane = (int32_t)plane;
    sd->subgroup_size = 32;
    entries[0] = (VkSpecializationMapEntry){.constantID = 0,
                                            .offset = offsetof(struct IntPsnrHvsSpecData, bpc),
                                            .size = sizeof(int32_t)};
    entries[1] = (VkSpecializationMapEntry){.constantID = 1,
                                            .offset = offsetof(struct IntPsnrHvsSpecData, plane),
                                            .size = sizeof(int32_t)};
    entries[2] =
        (VkSpecializationMapEntry){.constantID = 2,
                                   .offset = offsetof(struct IntPsnrHvsSpecData, subgroup_size),
                                   .size = sizeof(int32_t)};
    *info = (VkSpecializationInfo){
        .mapEntryCount = 3, .pMapEntries = entries, .dataSize = sizeof(*sd), .pData = sd};
}

static int build_chroma_pipeline(IntPsnrHvsVulkanState *s, int plane, VkPipeline *out)
{
    struct IntPsnrHvsSpecData sd = {0};
    VkSpecializationMapEntry entries[3];
    VkSpecializationInfo info = {0};
    fill_spec(&sd, entries, &info, s, plane);

    VkComputePipelineCreateInfo cpci = {
        .stage = {.pName = "main", .pSpecializationInfo = &info},
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl, &cpci, out);
}

static int create_pipeline(IntPsnrHvsVulkanState *s)
{
    struct IntPsnrHvsSpecData sd = {0};
    VkSpecializationMapEntry entries[3];
    VkSpecializationInfo info = {0};
    fill_spec(&sd, entries, &info, s, 0);

    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = (uint32_t)IPSNR_HVS_NUM_BINDINGS,
        .push_constant_size = (uint32_t)sizeof(IntPsnrHvsPushConsts),
        .spv_bytes = integer_psnr_hvs_spv,
        .spv_size = integer_psnr_hvs_spv_size,
        .pipeline_create_info =
            {
                .stage = {.pName = "main", .pSpecializationInfo = &info},
            },
        .max_descriptor_sets = (uint32_t)(2 * IPSNR_HVS_NUM_PLANES),
    };
    int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
    if (err)
        return err;
    err = build_chroma_pipeline(s, 1, &s->pipeline_chroma_u);
    if (err)
        return err;
    return build_chroma_pipeline(s, 2, &s->pipeline_chroma_v);
}

static int alloc_buffers(IntPsnrHvsVulkanState *s)
{
    /* Each sample is one uint element: 1 byte for 8bpc, 2 bytes for 10/12bpc.
     * The SSBO uses std430 uint[] layout (4 bytes per element), so we
     * allocate width*height*sizeof(uint) regardless of bpc — the shader
     * masks off the relevant bits.  This matches integer_ssim_vulkan.c. */
    int err = 0;
    for (int p = 0; p < IPSNR_HVS_NUM_PLANES; p++) {
        const size_t input_bytes = (size_t)s->width[p] * s->height[p] * sizeof(uint32_t);
        const size_t partials_bytes = (size_t)s->num_blocks[p] * sizeof(float);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_in[p], input_bytes);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->dist_in[p], input_bytes);
        err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->partials[p], partials_bytes);
    }
    return err ? -ENOMEM : 0;
}

/* Forward declaration — write_descriptor_set is called by init() after
 * alloc_buffers() to populate the pre-allocated sets once. */
static int write_descriptor_set(IntPsnrHvsVulkanState *s, VkDescriptorSet set, int plane);

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    IntPsnrHvsVulkanState *s = fex->priv;

    if (bpc > 12) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_psnr_hvs_vulkan: invalid bitdepth (%u); bpc must be <= 12\n", bpc);
        return -EINVAL;
    }
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_psnr_hvs_vulkan: YUV400P unsupported (needs all 3 planes)\n");
        return -EINVAL;
    }
    if (w < (unsigned)IPSNR_HVS_BLOCK || h < (unsigned)IPSNR_HVS_BLOCK) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_psnr_hvs_vulkan: input %ux%u smaller than 8x8 block\n", w, h);
        return -EINVAL;
    }

    s->bpc = bpc;
    const int32_t samplemax = (1 << bpc) - 1;
    s->samplemax_sq = samplemax * samplemax;

    s->width[0] = w;
    s->height[0] = h;
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        s->width[1] = s->width[2] = (w + 1u) >> 1;
        s->height[1] = s->height[2] = (h + 1u) >> 1;
        break;
    case VMAF_PIX_FMT_YUV422P:
        s->width[1] = s->width[2] = (w + 1u) >> 1;
        s->height[1] = s->height[2] = h;
        break;
    case VMAF_PIX_FMT_YUV444P:
        s->width[1] = s->width[2] = w;
        s->height[1] = s->height[2] = h;
        break;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "integer_psnr_hvs_vulkan: unsupported pix_fmt\n");
        return -EINVAL;
    }

    for (int p = 0; p < IPSNR_HVS_NUM_PLANES; p++) {
        if (s->width[p] < (unsigned)IPSNR_HVS_BLOCK || s->height[p] < (unsigned)IPSNR_HVS_BLOCK) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "integer_psnr_hvs_vulkan: plane %d dims %ux%u smaller than 8x8 block\n", p,
                     s->width[p], s->height[p]);
            return -EINVAL;
        }
        s->num_blocks_x[p] = (s->width[p] - IPSNR_HVS_BLOCK) / IPSNR_HVS_STEP + 1;
        s->num_blocks_y[p] = (s->height[p] - IPSNR_HVS_BLOCK) / IPSNR_HVS_STEP + 1;
        s->num_blocks[p] = s->num_blocks_x[p] * s->num_blocks_y[p];
    }

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, -1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "integer_psnr_hvs_vulkan: cannot create Vulkan context (%d)\n", err);
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

    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, 1, &s->sub_pool);
    if (err)
        return err;

    /* Pre-allocate 3 descriptor sets and write them once (VK-6). */
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl,
                                                   (uint32_t)IPSNR_HVS_NUM_PLANES, s->pre_sets);
    if (err)
        return err;
    for (int p = 0; p < IPSNR_HVS_NUM_PLANES; p++)
        (void)write_descriptor_set(s, s->pre_sets[p], p);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;
}

/* Upload one plane's raw integer samples into the host-mapped SSBO.
 * Each uint32 element holds one sample in the low bits (8bpc: bits[7:0],
 * 10/12bpc: bits[15:0]).  This avoids the float normalisation the float
 * path (psnr_hvs_vulkan.c) performs on the CPU, matching the CUDA twin. */
static int upload_plane_int(IntPsnrHvsVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic,
                            int plane)
{
    uint32_t *dst = (uint32_t *)vmaf_vulkan_buffer_host(dst_buf);
    if (!dst)
        return -EIO;

    if (pic->bpc <= 8) {
        const uint8_t *src = (const uint8_t *)pic->data[plane];
        for (unsigned y = 0; y < s->height[plane]; y++) {
            const uint8_t *src_row = src + y * (size_t)pic->stride[plane];
            uint32_t *dst_row = dst + y * s->width[plane];
            for (unsigned x = 0; x < s->width[plane]; x++)
                dst_row[x] = (uint32_t)src_row[x];
        }
    } else {
        const uint16_t *src = (const uint16_t *)pic->data[plane];
        const size_t src_stride_words = (size_t)pic->stride[plane] / sizeof(uint16_t);
        for (unsigned y = 0; y < s->height[plane]; y++) {
            const uint16_t *src_row = src + y * src_stride_words;
            uint32_t *dst_row = dst + y * s->width[plane];
            for (unsigned x = 0; x < s->width[plane]; x++)
                dst_row[x] = (uint32_t)src_row[x];
        }
    }
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int write_descriptor_set(IntPsnrHvsVulkanState *s, VkDescriptorSet set, int plane)
{
    VkDescriptorBufferInfo dbi[IPSNR_HVS_NUM_BINDINGS] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_in[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dist_in[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->partials[plane]),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[IPSNR_HVS_NUM_BINDINGS];
    for (int i = 0; i < IPSNR_HVS_NUM_BINDINGS; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, IPSNR_HVS_NUM_BINDINGS, writes, 0, NULL);
    return 0;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    IntPsnrHvsVulkanState *s = fex->priv;
    int err = 0;

    /* Upload raw integer samples for all three planes. */
    for (int p = 0; p < IPSNR_HVS_NUM_PLANES; p++) {
        err = upload_plane_int(s, s->ref_in[p], ref_pic, p);
        if (err)
            return err;
        err = upload_plane_int(s, s->dist_in[p], dist_pic, p);
        if (err)
            return err;
    }

    /* Acquire pre-allocated command buffer + fence from the submit pool. */
    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, 0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    for (int p = 0; p < IPSNR_HVS_NUM_PLANES; p++) {
        IntPsnrHvsPushConsts pc = {
            .width = s->width[p],
            .height = s->height[p],
            .num_blocks_x = s->num_blocks_x[p],
            .num_blocks_y = s->num_blocks_y[p],
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0, 1,
                                &s->pre_sets[p], 0, NULL);
        vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                           &pc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plane_pipeline(s, p));
        vkCmdDispatch(cmd, s->num_blocks_x[p], s->num_blocks_y[p], 1);
    }

    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    /* Per-plane reduction: sum partials in float (matches CPU float `ret`
     * register semantics — parity with psnr_hvs_vulkan.c collect path). */
    double plane_score[IPSNR_HVS_NUM_PLANES];
    for (int p = 0; p < IPSNR_HVS_NUM_PLANES; p++) {
        int err_inv = vmaf_vulkan_buffer_invalidate(s->ctx, s->partials[p]);
        if (err_inv) {
            err = err_inv;
            goto cleanup;
        }
        const float *partials = (const float *)vmaf_vulkan_buffer_host(s->partials[p]);
        float ret = 0.0f;
        for (unsigned i = 0; i < s->num_blocks[p]; i++)
            ret += partials[i];
        const int pixels = (int)(s->num_blocks[p] * 64u);
        ret /= (float)pixels;
        ret /= (float)s->samplemax_sq;
        plane_score[p] = (double)ret;
    }

    {
        static const char *plane_features[IPSNR_HVS_NUM_PLANES] = {
            "integer_psnr_hvs_y", "integer_psnr_hvs_cb", "integer_psnr_hvs_cr"};
        for (int p = 0; p < IPSNR_HVS_NUM_PLANES; p++) {
            const double db = 10.0 * (-1.0 * log10(plane_score[p]));
            err |= vmaf_feature_collector_append(feature_collector, plane_features[p], db, index);
        }
        const double combined = 0.8 * plane_score[0] + 0.1 * (plane_score[1] + plane_score[2]);
        const double db_combined = 10.0 * (-1.0 * log10(combined));
        err |= vmaf_feature_collector_append(feature_collector, "integer_psnr_hvs", db_combined,
                                             index);
    }

cleanup:
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    IntPsnrHvsVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    vkDeviceWaitIdle(s->ctx->device);

    if (s->pipeline_chroma_u != VK_NULL_HANDLE)
        vkDestroyPipeline(s->ctx->device, s->pipeline_chroma_u, NULL);
    if (s->pipeline_chroma_v != VK_NULL_HANDLE)
        vkDestroyPipeline(s->ctx->device, s->pipeline_chroma_v, NULL);
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    for (int p = 0; p < IPSNR_HVS_NUM_PLANES; p++) {
        if (s->ref_in[p])
            vmaf_vulkan_buffer_free(s->ctx, s->ref_in[p]);
        if (s->dist_in[p])
            vmaf_vulkan_buffer_free(s->ctx, s->dist_in[p]);
        if (s->partials[p])
            vmaf_vulkan_buffer_free(s->ctx, s->partials[p]);
    }

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {"integer_psnr_hvs_y", "integer_psnr_hvs_cb",
                                          "integer_psnr_hvs_cr", "integer_psnr_hvs", NULL};

VmafFeatureExtractor vmaf_fex_integer_psnr_hvs_vulkan = {
    .name = "integer_psnr_hvs_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(IntPsnrHvsVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
