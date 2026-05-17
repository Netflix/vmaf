/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  integer_ms_ssim_vulkan — Vulkan port of CUDA integer_ms_ssim_cuda.c.
 *
 *  Provides "float_ms_ssim" on the Vulkan backend, mirroring
 *  vmaf_fex_float_ms_ssim_cuda (ADR-0188 / ADR-0190, T7-23) but
 *  using the Vulkan compute pipeline instead of CUDA kernels.
 *
 *  Key difference vs ms_ssim_vulkan.c (float_ms_ssim_vulkan):
 *  The horizontal-pass shader (integer_ms_ssim.comp) reads raw integer
 *  pixel samples directly from the VmafPicture's GPU-mapped SSBO and
 *  normalises them to float-in-[0,255] on the GPU, eliminating the
 *  host-side picture_copy() call that ms_ssim_vulkan.c requires.
 *  The vertical-pass and decimation shaders are shared with the float
 *  variant (ms_ssim.comp, ms_ssim_decimate.comp).
 *
 *  Design summary:
 *    - 5-scale MS-SSIM pyramid via ms_ssim_decimate.comp.
 *    - Per-scale horiz + vert dispatch via integer_ms_ssim.comp
 *      (PASS=0 horiz reads uint input; PASS=1 vert reads float
 *      intermediates — identical to ms_ssim.comp PASS=1).
 *    - Host accumulates per-WG partial sums and applies Wang weights.
 *    - enable_lcs flag emits 15 extra per-scale L/C/S features.
 *
 *  Min-dim guard: 176 (11 << 4), ADR-0153.
 *
 *  Bindings for integer_ms_ssim.comp:
 *    0  ref_int      (uint SSBO — raw integer luma)
 *    1  cmp_int      (uint SSBO — raw integer luma)
 *    2  h_ref_mu     (float SSBO — horiz μ ref)
 *    3  h_cmp_mu     (float SSBO — horiz μ cmp)
 *    4  h_ref_sq     (float SSBO — horiz E[ref²])
 *    5  h_cmp_sq     (float SSBO — horiz E[cmp²])
 *    6  h_refcmp     (float SSBO — horiz E[ref·cmp])
 *    7  l_partials   (float SSBO — per-WG l partial sums)
 *    8  c_partials   (float SSBO — per-WG c partial sums)
 *    9  s_partials   (float SSBO — per-WG s partial sums)
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

#include "integer_ms_ssim_spv.h"  /* generated SPIR-V: integer_ms_ssim.comp */
#include "ms_ssim_decimate_spv.h" /* shared with float ms_ssim_vulkan.c */

#define MS_SSIM_SCALES 5
#define MS_SSIM_GAUSSIAN_LEN 11
#define MS_SSIM_K 11
#define MS_SSIM_WG_X 16
#define MS_SSIM_WG_Y 8
/* Decimate: src + dst. */
#define IMS_DECIMATE_BINDINGS 2
/* SSIM: ref_int, cmp_int, 5 float intermediates, 3 partial buffers. */
#define IMS_SSIM_BINDINGS 10

static const float g_alphas[MS_SSIM_SCALES] = {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f};
static const float g_betas[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};
static const float g_gammas[MS_SSIM_SCALES] = {0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f};

typedef struct {
    bool enable_lcs;

    unsigned width;
    unsigned height;
    unsigned bpc;

    /* Pyramid dimensions per scale. */
    unsigned scale_w[MS_SSIM_SCALES];
    unsigned scale_h[MS_SSIM_SCALES];
    unsigned scale_w_horiz[MS_SSIM_SCALES];
    unsigned scale_h_horiz[MS_SSIM_SCALES];
    unsigned scale_w_final[MS_SSIM_SCALES];
    unsigned scale_h_final[MS_SSIM_SCALES];
    unsigned scale_wg_x[MS_SSIM_SCALES];
    unsigned scale_wg_y[MS_SSIM_SCALES];
    unsigned scale_wg_count[MS_SSIM_SCALES];

    float c1;
    float c2;
    float c3;

    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Decimate pipeline (shared spec pattern with ms_ssim_vulkan.c). */
    VmafVulkanKernelPipeline pl_decimate;
    VkPipeline decimate_pipelines[MS_SSIM_SCALES - 1];

    /* SSIM pipeline (integer_ms_ssim.comp). */
    VmafVulkanKernelPipeline pl_ssim;
    VkPipeline ssim_pipeline_horiz[MS_SSIM_SCALES];
    VkPipeline ssim_pipeline_vert[MS_SSIM_SCALES];

    /* Submit pools. */
    VmafVulkanKernelSubmitPool sub_pool_decimate;
    VmafVulkanKernelSubmitPool sub_pool_ssim;

    /* Pre-allocated descriptor sets. */
    VkDescriptorSet dec_sets_ref[MS_SSIM_SCALES - 1];
    VkDescriptorSet dec_sets_cmp[MS_SSIM_SCALES - 1];
    VkDescriptorSet ssim_sets[MS_SSIM_SCALES];

    /* Pyramid buffers: 5 levels × ref + cmp, float (produced by decimate). */
    VmafVulkanBuffer *pyramid_ref[MS_SSIM_SCALES];
    VmafVulkanBuffer *pyramid_cmp[MS_SSIM_SCALES];

    /* Integer-input upload buffers at scale 0 (uint SSBO, host-mapped). */
    VmafVulkanBuffer *int_ref;
    VmafVulkanBuffer *int_cmp;

    /* Five float intermediate buffers (largest scale). */
    VmafVulkanBuffer *h_ref_mu;
    VmafVulkanBuffer *h_cmp_mu;
    VmafVulkanBuffer *h_ref_sq;
    VmafVulkanBuffer *h_cmp_sq;
    VmafVulkanBuffer *h_refcmp;

    /* Three partial-sum readback buffers (largest scale). */
    VmafVulkanBuffer *l_partials;
    VmafVulkanBuffer *c_partials;
    VmafVulkanBuffer *s_partials;

    VmafDictionary *feature_name_dict;
} IntMsSsimVulkanState;

static const VmafOption options[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(IntMsSsimVulkanState, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {0},
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t w_out;
    uint32_t h_out;
} DecimatePushConsts;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t w_horiz;
    uint32_t h_horiz;
    uint32_t w_final;
    uint32_t h_final;
    uint32_t num_wg_x;
    uint32_t stride_pixels;
    float c1;
    float c2;
    float c3;
} IntMsSsimPushConsts;

/* ---- specialisation helpers ---- */

struct DecSpecData {
    int32_t width;
    int32_t height;
};

static void dec_fill_spec(struct DecSpecData *d, VkSpecializationMapEntry e[2],
                          VkSpecializationInfo *si, const IntMsSsimVulkanState *s, int idx)
{
    d->width = (int32_t)s->scale_w[idx];
    d->height = (int32_t)s->scale_h[idx];
    e[0] = (VkSpecializationMapEntry){0, offsetof(struct DecSpecData, width), sizeof(int32_t)};
    e[1] = (VkSpecializationMapEntry){1, offsetof(struct DecSpecData, height), sizeof(int32_t)};
    *si = (VkSpecializationInfo){2, e, sizeof(*d), d};
}

struct SsimSpecData {
    int32_t width;
    int32_t height;
    int32_t pass;
    int32_t subgroup_size;
    int32_t bpc;
};

static void ssim_fill_spec(struct SsimSpecData *d, VkSpecializationMapEntry e[5],
                           VkSpecializationInfo *si, const IntMsSsimVulkanState *s, int idx,
                           int pass)
{
    d->width = (int32_t)s->scale_w[idx];
    d->height = (int32_t)s->scale_h[idx];
    d->pass = pass;
    d->subgroup_size = 32;
    d->bpc = (int32_t)s->bpc;
    e[0] = (VkSpecializationMapEntry){0, offsetof(struct SsimSpecData, width), sizeof(int32_t)};
    e[1] = (VkSpecializationMapEntry){1, offsetof(struct SsimSpecData, height), sizeof(int32_t)};
    e[2] = (VkSpecializationMapEntry){2, offsetof(struct SsimSpecData, pass), sizeof(int32_t)};
    e[3] = (VkSpecializationMapEntry){3, offsetof(struct SsimSpecData, subgroup_size),
                                      sizeof(int32_t)};
    e[4] = (VkSpecializationMapEntry){4, offsetof(struct SsimSpecData, bpc), sizeof(int32_t)};
    *si = (VkSpecializationInfo){5, e, sizeof(*d), d};
}

static int build_dec_variant(IntMsSsimVulkanState *s, int idx, VkPipeline *out)
{
    struct DecSpecData d = {0};
    VkSpecializationMapEntry e[2];
    VkSpecializationInfo si = {0};
    dec_fill_spec(&d, e, &si, s, idx);
    VkComputePipelineCreateInfo cpci = {.stage = {.pName = "main", .pSpecializationInfo = &si}};
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl_decimate, &cpci, out);
}

static int build_ssim_variant(IntMsSsimVulkanState *s, int idx, int pass, VkPipeline *out)
{
    struct SsimSpecData d = {0};
    VkSpecializationMapEntry e[5];
    VkSpecializationInfo si = {0};
    ssim_fill_spec(&d, e, &si, s, idx, pass);
    VkComputePipelineCreateInfo cpci = {.stage = {.pName = "main", .pSpecializationInfo = &si}};
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl_ssim, &cpci, out);
}

static int create_pipelines(IntMsSsimVulkanState *s)
{
    /* Decimate bundle. */
    {
        struct DecSpecData d = {0};
        VkSpecializationMapEntry e[2];
        VkSpecializationInfo si = {0};
        dec_fill_spec(&d, e, &si, s, 0);

        const VmafVulkanKernelPipelineDesc desc = {
            .ssbo_binding_count = IMS_DECIMATE_BINDINGS,
            .push_constant_size = (uint32_t)sizeof(DecimatePushConsts),
            .spv_bytes = ms_ssim_decimate_spv,
            .spv_size = ms_ssim_decimate_spv_size,
            .pipeline_create_info = {.stage = {.pName = "main", .pSpecializationInfo = &si}},
            .max_descriptor_sets = (uint32_t)((MS_SSIM_SCALES - 1) * 2),
        };
        int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl_decimate);
        if (err)
            return err;
        s->decimate_pipelines[0] = s->pl_decimate.pipeline;
        for (int i = 1; i < MS_SSIM_SCALES - 1; i++) {
            err = build_dec_variant(s, i, &s->decimate_pipelines[i]);
            if (err)
                return err;
        }
    }

    /* SSIM bundle (integer_ms_ssim.comp). */
    {
        struct SsimSpecData d = {0};
        VkSpecializationMapEntry e[5];
        VkSpecializationInfo si = {0};
        ssim_fill_spec(&d, e, &si, s, 0, 0);

        const VmafVulkanKernelPipelineDesc desc = {
            .ssbo_binding_count = IMS_SSIM_BINDINGS,
            .push_constant_size = (uint32_t)sizeof(IntMsSsimPushConsts),
            .spv_bytes = integer_ms_ssim_spv,
            .spv_size = integer_ms_ssim_spv_size,
            .pipeline_create_info = {.stage = {.pName = "main", .pSpecializationInfo = &si}},
            .max_descriptor_sets = (uint32_t)MS_SSIM_SCALES,
        };
        int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl_ssim);
        if (err)
            return err;
        s->ssim_pipeline_horiz[0] = s->pl_ssim.pipeline;
        for (int i = 1; i < MS_SSIM_SCALES; i++) {
            err = build_ssim_variant(s, i, 0, &s->ssim_pipeline_horiz[i]);
            if (err)
                return err;
        }
        for (int i = 0; i < MS_SSIM_SCALES; i++) {
            err = build_ssim_variant(s, i, 1, &s->ssim_pipeline_vert[i]);
            if (err)
                return err;
        }
    }
    return 0;
}

static int alloc_buffers(IntMsSsimVulkanState *s)
{
    int err = 0;
    /* Integer input at scale 0 (uint SSBO, host-mapped, largest plane). */
    const unsigned bpc_bytes = (s->bpc <= 8) ? 1u : 2u;
    size_t int_bytes = (size_t)s->scale_w[0] * s->scale_h[0] * bpc_bytes;
    /* Round up to uint32 alignment: each element is one uint. */
    size_t int_buf_bytes = (size_t)s->scale_w[0] * s->scale_h[0] * sizeof(uint32_t);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->int_ref, int_buf_bytes);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->int_cmp, int_buf_bytes);
    (void)int_bytes; /* size recorded; int_buf_bytes is the allocated size */

    /* Pyramid float buffers for scales 0..4. */
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        size_t plane_bytes = (size_t)s->scale_w[i] * s->scale_h[i] * sizeof(float);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->pyramid_ref[i], plane_bytes);
        err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->pyramid_cmp[i], plane_bytes);
    }

    /* Float intermediates sized for scale 0 (largest). */
    size_t horiz_max = (size_t)s->scale_w_horiz[0] * s->scale_h_horiz[0] * sizeof(float);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_ref_mu, horiz_max);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_cmp_mu, horiz_max);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_ref_sq, horiz_max);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_cmp_sq, horiz_max);
    err |= vmaf_vulkan_buffer_alloc(s->ctx, &s->h_refcmp, horiz_max);

    /* Partials sized for scale 0 wg_count (largest). */
    size_t partials_max = (size_t)s->scale_wg_count[0] * sizeof(float);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->l_partials, partials_max);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->c_partials, partials_max);
    err |= vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->s_partials, partials_max);
    return err ? -ENOMEM : 0;
}

static void write_dec_set(IntMsSsimVulkanState *s, VkDescriptorSet set, VmafVulkanBuffer *src,
                          VmafVulkanBuffer *dst)
{
    VkDescriptorBufferInfo dbi[2] = {
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(src), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(dst), 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet wr[2];
    for (int i = 0; i < 2; i++) {
        wr[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, 2, wr, 0, NULL);
}

/* For scale 0 the SSIM descriptor references int_ref / int_cmp;
 * for scales 1..4 it references pyramid_ref[i] / pyramid_cmp[i]
 * (float, produced by the decimate shader). */
static void write_ssim_set(IntMsSsimVulkanState *s, VkDescriptorSet set, VmafVulkanBuffer *ref,
                           VmafVulkanBuffer *cmp)
{
    VkDescriptorBufferInfo dbi[IMS_SSIM_BINDINGS] = {
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(ref), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(cmp), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_ref_mu), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_cmp_mu), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_ref_sq), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_cmp_sq), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(s->h_refcmp), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(s->l_partials), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(s->c_partials), 0, VK_WHOLE_SIZE},
        {(VkBuffer)vmaf_vulkan_buffer_vkhandle(s->s_partials), 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet wr[IMS_SSIM_BINDINGS];
    for (int i = 0; i < IMS_SSIM_BINDINGS; i++) {
        wr[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(s->ctx->device, IMS_SSIM_BINDINGS, wr, 0, NULL);
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    IntMsSsimVulkanState *s = fex->priv;

    const unsigned min_dim = (unsigned)MS_SSIM_GAUSSIAN_LEN << (MS_SSIM_SCALES - 1);
    if (w < min_dim || h < min_dim) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "integer_ms_ssim_vulkan: input %ux%u too small; needs >= %ux%u "
                 "(ADR-0153)\n",
                 w, h, min_dim, min_dim);
        return -EINVAL;
    }

    s->width = w;
    s->height = h;
    s->bpc = bpc;

    s->scale_w[0] = w;
    s->scale_h[0] = h;
    for (int i = 1; i < MS_SSIM_SCALES; i++) {
        s->scale_w[i] = (s->scale_w[i - 1] / 2) + (s->scale_w[i - 1] & 1);
        s->scale_h[i] = (s->scale_h[i - 1] / 2) + (s->scale_h[i - 1] & 1);
    }
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        s->scale_w_horiz[i] = s->scale_w[i] - (MS_SSIM_K - 1);
        s->scale_h_horiz[i] = s->scale_h[i];
        s->scale_w_final[i] = s->scale_w[i] - (MS_SSIM_K - 1);
        s->scale_h_final[i] = s->scale_h[i] - (MS_SSIM_K - 1);
        s->scale_wg_x[i] =
            (s->scale_w_final[i] + (unsigned)MS_SSIM_WG_X - 1) / (unsigned)MS_SSIM_WG_X;
        s->scale_wg_y[i] =
            (s->scale_h_final[i] + (unsigned)MS_SSIM_WG_Y - 1) / (unsigned)MS_SSIM_WG_Y;
        s->scale_wg_count[i] = s->scale_wg_x[i] * s->scale_wg_y[i];
    }

    const float L = 255.0f, K1 = 0.01f, K2 = 0.03f;
    s->c1 = (K1 * L) * (K1 * L);
    s->c2 = (K2 * L) * (K2 * L);
    s->c3 = s->c2 * 0.5f;

    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, -1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "integer_ms_ssim_vulkan: cannot create Vulkan context (%d)\n", err);
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

    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, 1, &s->sub_pool_decimate);
    if (err)
        return err;
    err =
        vmaf_vulkan_kernel_submit_pool_create(s->ctx, (uint32_t)MS_SSIM_SCALES, &s->sub_pool_ssim);
    if (err)
        return err;

    /* Pre-allocate descriptor sets. */
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl_decimate.desc_pool,
                                                   s->pl_decimate.dsl,
                                                   (uint32_t)(MS_SSIM_SCALES - 1), s->dec_sets_ref);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl_decimate.desc_pool,
                                                   s->pl_decimate.dsl,
                                                   (uint32_t)(MS_SSIM_SCALES - 1), s->dec_sets_cmp);
    if (err)
        return err;
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl_ssim.desc_pool, s->pl_ssim.dsl,
                                                   (uint32_t)MS_SSIM_SCALES, s->ssim_sets);
    if (err)
        return err;

    /* Write descriptor sets once — SSBO handles are stable from init(). */
    for (int i = 0; i < MS_SSIM_SCALES - 1; i++) {
        write_dec_set(s, s->dec_sets_ref[i], s->pyramid_ref[i], s->pyramid_ref[i + 1]);
        write_dec_set(s, s->dec_sets_cmp[i], s->pyramid_cmp[i], s->pyramid_cmp[i + 1]);
    }
    /* Scale 0 SSIM set reads integer input. Scales 1..4 read float
     * pyramid buffers (same float layout as ms_ssim_vulkan.c). */
    write_ssim_set(s, s->ssim_sets[0], s->int_ref, s->int_cmp);
    for (int i = 1; i < MS_SSIM_SCALES; i++)
        write_ssim_set(s, s->ssim_sets[i], s->pyramid_ref[i], s->pyramid_cmp[i]);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;
}

/* Upload an integer picture plane into an SSBO (uint32 per sample, host-mapped). */
static int upload_int_pic(IntMsSsimVulkanState *s, VmafVulkanBuffer *dst_buf, VmafPicture *pic)
{
    uint32_t *dst = vmaf_vulkan_buffer_host(dst_buf);
    if (!dst)
        return -EIO;

    const unsigned bpc_bytes = (s->bpc <= 8) ? 1u : 2u;
    const unsigned w = s->width;
    const unsigned h = s->height;

    if (bpc_bytes == 1) {
        const uint8_t *src = (const uint8_t *)pic->data[0];
        const ptrdiff_t src_stride = pic->stride[0];
        for (unsigned row = 0; row < h; row++) {
            for (unsigned col = 0; col < w; col++)
                dst[row * w + col] = (uint32_t)src[row * src_stride + col];
        }
    } else {
        const uint16_t *src = (const uint16_t *)pic->data[0];
        const ptrdiff_t src_stride = pic->stride[0] / 2;
        for (unsigned row = 0; row < h; row++) {
            for (unsigned col = 0; col < w; col++)
                dst[row * w + col] = (uint32_t)src[row * src_stride + col];
        }
    }
    return vmaf_vulkan_buffer_flush(s->ctx, dst_buf);
}

static int run_decimate(IntMsSsimVulkanState *s, VkCommandBuffer cmd, int idx,
                        VkDescriptorSet dec_set)
{
    DecimatePushConsts pc = {
        .width = s->scale_w[idx],
        .height = s->scale_h[idx],
        .w_out = s->scale_w[idx + 1],
        .h_out = s->scale_h[idx + 1],
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_decimate.pipeline_layout, 0,
                            1, &dec_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_decimate.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->decimate_pipelines[idx]);
    const uint32_t gx = (s->scale_w[idx + 1] + MS_SSIM_WG_X - 1) / MS_SSIM_WG_X;
    const uint32_t gy = (s->scale_h[idx + 1] + MS_SSIM_WG_Y - 1) / MS_SSIM_WG_Y;
    vkCmdDispatch(cmd, gx, gy, 1);
    return 0;
}

static int run_ssim_scale(IntMsSsimVulkanState *s, VkCommandBuffer cmd, int idx,
                          VkDescriptorSet ssim_set)
{
    /* For scale 0 the integer horiz pass uses stride = width (uint layout). */
    const uint32_t stride_px = (idx == 0) ? s->scale_w[0] : s->scale_w[idx];

    IntMsSsimPushConsts pc = {
        .width = s->scale_w[idx],
        .height = s->scale_h[idx],
        .w_horiz = s->scale_w_horiz[idx],
        .h_horiz = s->scale_h_horiz[idx],
        .w_final = s->scale_w_final[idx],
        .h_final = s->scale_h_final[idx],
        .num_wg_x = s->scale_wg_x[idx],
        .stride_pixels = stride_px,
        .c1 = s->c1,
        .c2 = s->c2,
        .c3 = s->c3,
    };

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_ssim.pipeline_layout, 0, 1,
                            &ssim_set, 0, NULL);
    vkCmdPushConstants(cmd, s->pl_ssim.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc),
                       &pc);

    /* Pass 0: horizontal. */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->ssim_pipeline_horiz[idx]);
    const uint32_t gx_h = (s->scale_w_horiz[idx] + MS_SSIM_WG_X - 1) / MS_SSIM_WG_X;
    const uint32_t gy_h = (s->scale_h_horiz[idx] + MS_SSIM_WG_Y - 1) / MS_SSIM_WG_Y;
    vkCmdDispatch(cmd, gx_h, gy_h, 1);

    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    /* Pass 1: vertical + l/c/s. */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->ssim_pipeline_vert[idx]);
    vkCmdDispatch(cmd, s->scale_wg_x[idx], s->scale_wg_y[idx], 1);
    return 0;
}

static double sum_partials(const float *p, unsigned n)
{
    double t = 0.0;
    for (unsigned i = 0; i < n; i++)
        t += (double)p[i];
    return t;
}

static int emit_lcs(VmafFeatureCollector *fc, unsigned index, const double l[5], const double c[5],
                    const double sp[5])
{
    static const char *const ln[5] = {
        "float_ms_ssim_l_scale0", "float_ms_ssim_l_scale1", "float_ms_ssim_l_scale2",
        "float_ms_ssim_l_scale3", "float_ms_ssim_l_scale4",
    };
    static const char *const cn[5] = {
        "float_ms_ssim_c_scale0", "float_ms_ssim_c_scale1", "float_ms_ssim_c_scale2",
        "float_ms_ssim_c_scale3", "float_ms_ssim_c_scale4",
    };
    static const char *const sn[5] = {
        "float_ms_ssim_s_scale0", "float_ms_ssim_s_scale1", "float_ms_ssim_s_scale2",
        "float_ms_ssim_s_scale3", "float_ms_ssim_s_scale4",
    };
    int err = 0;
    for (int i = 0; i < 5; i++) {
        err |= vmaf_feature_collector_append(fc, ln[i], l[i], index);
        err |= vmaf_feature_collector_append(fc, cn[i], c[i], index);
        err |= vmaf_feature_collector_append(fc, sn[i], sp[i], index);
    }
    return err;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    IntMsSsimVulkanState *s = fex->priv;
    int err = 0;

    /* Upload raw integer luma planes to the integer input SSBOs. */
    err = upload_int_pic(s, s->int_ref, ref_pic);
    if (err)
        return err;
    err = upload_int_pic(s, s->int_cmp, dist_pic);
    if (err)
        return err;

    /* Build pyramid (scales 1..4) via one decimate command buffer. */
    {
        VmafVulkanKernelSubmit sub = {0};
        err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool_decimate, 0, &sub);
        if (err)
            return err;

        VkMemoryBarrier rw = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        for (int i = 0; i < MS_SSIM_SCALES - 1; i++) {
            run_decimate(s, sub.cmd, i, s->dec_sets_ref[i]);
            run_decimate(s, sub.cmd, i, s->dec_sets_cmp[i]);
            vkCmdPipelineBarrier(sub.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &rw, 0, NULL, 0, NULL);
        }
        err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &sub);
        vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
        if (err)
            return err;
    }

    /* Per-scale SSIM. */
    double l_means[MS_SSIM_SCALES] = {0};
    double c_means[MS_SSIM_SCALES] = {0};
    double s_means[MS_SSIM_SCALES] = {0};

    for (int i = 0; i < MS_SSIM_SCALES && !err; i++) {
        VmafVulkanKernelSubmit sub = {0};
        err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool_ssim, (uint32_t)i, &sub);
        if (err)
            break;
        run_ssim_scale(s, sub.cmd, i, s->ssim_sets[i]);
        err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &sub);
        vmaf_vulkan_kernel_submit_free(s->ctx, &sub);
        if (err)
            break;

        int ei;
        ei = vmaf_vulkan_buffer_invalidate(s->ctx, s->l_partials);
        if (ei)
            return ei;
        ei = vmaf_vulkan_buffer_invalidate(s->ctx, s->c_partials);
        if (ei)
            return ei;
        ei = vmaf_vulkan_buffer_invalidate(s->ctx, s->s_partials);
        if (ei)
            return ei;

        const float *lp = vmaf_vulkan_buffer_host(s->l_partials);
        const float *cp = vmaf_vulkan_buffer_host(s->c_partials);
        const float *sp = vmaf_vulkan_buffer_host(s->s_partials);
        const unsigned wgc = s->scale_wg_count[i];
        const double n_px = (double)s->scale_w_final[i] * (double)s->scale_h_final[i];
        l_means[i] = sum_partials(lp, wgc) / n_px;
        c_means[i] = sum_partials(cp, wgc) / n_px;
        s_means[i] = sum_partials(sp, wgc) / n_px;
    }
    if (err)
        return err;

    double msssim = 1.0;
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        msssim *= pow(l_means[i], (double)g_alphas[i]) * pow(c_means[i], (double)g_betas[i]) *
                  pow(fabs(s_means[i]), (double)g_gammas[i]);
    }

    err = vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                  "float_ms_ssim", msssim, index);
    if (s->enable_lcs)
        err |= emit_lcs(feature_collector, index, l_means, c_means, s_means);
    return err;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    IntMsSsimVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;

    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool_decimate);
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool_ssim);

    for (int i = 1; i < MS_SSIM_SCALES - 1; i++)
        if (s->decimate_pipelines[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, s->decimate_pipelines[i], NULL);
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        if (i != 0 && s->ssim_pipeline_horiz[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, s->ssim_pipeline_horiz[i], NULL);
        if (s->ssim_pipeline_vert[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(dev, s->ssim_pipeline_vert[i], NULL);
    }
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_decimate);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_ssim);

    if (s->int_ref)
        vmaf_vulkan_buffer_free(s->ctx, s->int_ref);
    if (s->int_cmp)
        vmaf_vulkan_buffer_free(s->ctx, s->int_cmp);
    for (int i = 0; i < MS_SSIM_SCALES; i++) {
        if (s->pyramid_ref[i])
            vmaf_vulkan_buffer_free(s->ctx, s->pyramid_ref[i]);
        if (s->pyramid_cmp[i])
            vmaf_vulkan_buffer_free(s->ctx, s->pyramid_cmp[i]);
    }
    if (s->h_ref_mu)
        vmaf_vulkan_buffer_free(s->ctx, s->h_ref_mu);
    if (s->h_cmp_mu)
        vmaf_vulkan_buffer_free(s->ctx, s->h_cmp_mu);
    if (s->h_ref_sq)
        vmaf_vulkan_buffer_free(s->ctx, s->h_ref_sq);
    if (s->h_cmp_sq)
        vmaf_vulkan_buffer_free(s->ctx, s->h_cmp_sq);
    if (s->h_refcmp)
        vmaf_vulkan_buffer_free(s->ctx, s->h_refcmp);
    if (s->l_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->l_partials);
    if (s->c_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->c_partials);
    if (s->s_partials)
        vmaf_vulkan_buffer_free(s->ctx, s->s_partials);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {"float_ms_ssim", NULL};

VmafFeatureExtractor vmaf_fex_integer_ms_ssim_vulkan = {
    .name = "integer_ms_ssim_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(IntMsSsimVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
    .chars =
        {
            .n_dispatches_per_frame = 18,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
