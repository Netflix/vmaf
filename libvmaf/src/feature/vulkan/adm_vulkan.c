/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  ADM (Adaptive Detail Model) feature kernel on the Vulkan backend
 *  (T5-1c-adm). Mirrors the SYCL port in
 *  libvmaf/src/feature/sycl/integer_adm_sycl.cpp and the CPU reference
 *  in libvmaf/src/feature/integer_adm.c.
 *
 *  Per-frame pipeline (per scale, 4 scales total):
 *    Stage 0 — DWT vertical (ref+dis fused, dim_z=2)
 *    Stage 1 — DWT horizontal (ref+dis fused, dim_z=2)
 *    Stage 2 — Decouple + CSF fused
 *    Stage 3 — CSF denominator + Contrast measure fused (1D dispatch
 *              over 3 bands × num_active_rows; per-WG int64 reductions)
 *
 *  Stage 3 produces six int64 partials per workgroup (csf_h/v/d and
 *  cm_h/v/d). The host CPU reduces across WGs, then runs the same
 *  conclude_adm_csf_den / conclude_adm_cm scoring helpers as the SYCL
 *  port to produce per-scale numerator / denominator values.
 *
 *  Pattern reference: libvmaf/src/feature/vulkan/vif_vulkan.c and
 *  motion_vulkan.c — same lazy-or-borrow context, owns_ctx flag,
 *  VkSpecializationInfo-driven pipelines (one per (scale, stage)
 *  pair = 16 pipelines total), host-side reduction.
 *
 *  Scope note: aim_score and adm3_score (CPU-only debug paths in
 *  integer_adm.c lines 3415-3422) are NOT emitted here. The SYCL ADM
 *  port has the same scope; matching it keeps the cross-backend gate
 *  identity-equal between Vulkan and SYCL.
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "feature/adm_options.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"

#include "../../vulkan/kernel_template.h"
#include "../../vulkan/vulkan_common.h"
#include "../../vulkan/picture_vulkan.h"
#include "../../vulkan/vulkan_internal.h"

#include "adm_spv.h"        /* per-WG accumulator kernel */
#include "adm_reduce_spv.h" /* two-level reduction kernel (ADR-0350) */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/* Constants — must match adm.comp + integer_adm_sycl.cpp.            */
/* ------------------------------------------------------------------ */

#define ADM_NUM_SCALES 4
#define ADM_NUM_BANDS 3
#define ADM_NUM_STAGES 4
#define ADM_BORDER_FACTOR 0.1
#define ADM_DIV_LOOKUP_SIZE 65537
#define ADM_ACCUM_SLOTS_PER_WG 6 /* csf_h,csf_v,csf_d, cm_h,cm_v,cm_d */

#define ADM_WG_X 16
#define ADM_WG_Y 16
#define ADM_WG_SIZE (ADM_WG_X * ADM_WG_Y)

/* DB2 wavelet noise model (Watson 1997, Y channel). Used host-side to
 * compute per-(lambda,theta) quantisation steps and from those the
 * per-scale rfactors. */
typedef struct {
    float a, k, f0;
    float g[4];
} DwtModelParams;

static const DwtModelParams dwt_model_Y = {0.495f, 0.466f, 0.401f, {1.501f, 1.0f, 0.534f, 1.0f}};

static const float dwt_basis_amp[6][4] = {
    {0.62171f, 0.67234f, 0.72709f, 0.67234f},     {0.34537f, 0.41317f, 0.49428f, 0.41317f},
    {0.18004f, 0.22727f, 0.28688f, 0.22727f},     {0.091401f, 0.11792f, 0.15214f, 0.11792f},
    {0.045943f, 0.059758f, 0.077727f, 0.059758f}, {0.023013f, 0.030018f, 0.039156f, 0.030018f},
};

static float dwt_quant_step_host(int lambda, int theta, double view_dist, int display_h)
{
    float r = (float)(view_dist * (double)display_h * M_PI / 180.0);
    float temp =
        log10f(powf(2.0f, (float)(lambda + 1)) * dwt_model_Y.f0 * dwt_model_Y.g[theta] / r);
    return 2.0f * dwt_model_Y.a * powf(10.0f, dwt_model_Y.k * temp * temp) /
           dwt_basis_amp[lambda][theta];
}

/* ------------------------------------------------------------------ */
/* Per-extractor state.                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Options. */
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    double adm_csf_scale;
    double adm_csf_diag_scale;
    double adm_noise_weight;

    /* Frame geometry. */
    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned buf_stride; /* aligned int32 elements per band row */

    /* rfactors (host-side scoring uses float; device uses Q21/Q23/Q32 ints). */
    float rfactor[12]; /* 4 scales × 3 bands */
    uint32_t i_rfactor[12];

    /* Vulkan context handle. */
    VmafVulkanContext *ctx;
    int owns_ctx;

    /* Pipelines: one per (stage, scale) combination = 4 stages × 4 scales = 16. */
    /* `pl` carries the shared layout / shader / DSL / pool plus the
     * (stage=0, scale=0) base pipeline. The other 15 entries in
     * `pipelines[stage][scale]` are sibling pipelines created via
     * `vmaf_vulkan_kernel_pipeline_add_variant()`.
     * `pipelines[0][0]` aliases `pl.pipeline` so the dispatch path
     * keeps a clean 2-D lookup. (ADR-0221.) */
    VmafVulkanKernelPipeline pl;
    VkPipeline pipelines[ADM_NUM_STAGES][ADM_NUM_SCALES];
    /* ADR-0350: two-level GPU reduction pipeline + per-scale descriptor sets. */
    VmafVulkanKernelPipeline pl_reduce;
    VkDescriptorSet reduce_sets[ADM_NUM_SCALES];

    /* Submit-side pool (T-GPU-OPT-VK-1 / ADR-0256). One slot suffices:
     * all 16 dispatches are batched into a single command buffer. */
    VmafVulkanKernelSubmitPool sub_pool;

    /* Pre-allocated descriptor sets, one per scale (T-GPU-OPT-VK-4 /
     * ADR-0256). All 9 bindings (including the per-scale accum buffer at
     * slot 7) are init-time-stable; no vkUpdateDescriptorSets is needed
     * in extract(). */
    VkDescriptorSet pre_sets[ADM_NUM_SCALES];

    /* GPU buffers. dwt_tmp[]: int32 plane, sized for the largest
     * (scale 0) frame (cur_w*2 × half_h elements). All scales reuse
     * the same buffer since they fit. */
    VmafVulkanBuffer *src_ref; /* scale-0 host-uploaded source plane */
    VmafVulkanBuffer *src_dis;
    VmafVulkanBuffer *dwt_tmp_ref;
    VmafVulkanBuffer *dwt_tmp_dis;

    /* Band buffers: 4 bands packed into one SSBO each, addressed as
     * band_idx * (buf_stride * half_h) + y*buf_stride + x. */
    VmafVulkanBuffer *ref_band;
    VmafVulkanBuffer *dis_band;
    VmafVulkanBuffer *csf_f; /* 3 bands packed contiguously */

    VmafVulkanBuffer *div_lookup;

    /* Per-WG int64 accumulator buffers, one per scale. */
    VmafVulkanBuffer *accum[ADM_NUM_SCALES];
    /* ADR-0350: tiny GPU-reduced output buffer per scale —
     * ADM_ACCUM_SLOTS_PER_WG int64_t each (48 bytes). */
    VmafVulkanBuffer *reduced_accum[ADM_NUM_SCALES];
    unsigned wg_count[ADM_NUM_SCALES];

    /* Per-scale dimensions cached. */
    unsigned scale_w[ADM_NUM_SCALES];
    unsigned scale_h[ADM_NUM_SCALES];
    unsigned scale_half_w[ADM_NUM_SCALES];
    unsigned scale_half_h[ADM_NUM_SCALES];

    VmafDictionary *feature_name_dict;
} AdmVulkanState;

/* ------------------------------------------------------------------ */
/* Options.                                                            */
/* ------------------------------------------------------------------ */

static const VmafOption options[] = {{
                                         .name = "debug",
                                         .help = "debug mode: enable additional output",
                                         .offset = offsetof(AdmVulkanState, debug),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = true,
                                     },
                                     {
                                         .name = "adm_enhn_gain_limit",
                                         .help = "enhancement gain imposed on ADM, must be >= 1.0",
                                         .offset = offsetof(AdmVulkanState, adm_enhn_gain_limit),
                                         .type = VMAF_OPT_TYPE_DOUBLE,
                                         .default_val.d = 100.0,
                                         .min = 1.0,
                                         .max = 100.0,
                                         .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
                                     },
                                     {
                                         .name = "adm_norm_view_dist",
                                         .help = "normalized viewing distance",
                                         .offset = offsetof(AdmVulkanState, adm_norm_view_dist),
                                         .type = VMAF_OPT_TYPE_DOUBLE,
                                         .default_val.d = 3.0,
                                         .min = 0.75,
                                         .max = 24.0,
                                         .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
                                     },
                                     {
                                         .name = "adm_ref_display_height",
                                         .help = "reference display height in pixels",
                                         .offset = offsetof(AdmVulkanState, adm_ref_display_height),
                                         .type = VMAF_OPT_TYPE_INT,
                                         .default_val.i = 1080,
                                         .min = 480.0,
                                         .max = 4320.0,
                                         .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
                                     },
                                     {
                                         .name = "adm_csf_scale",
                                         .alias = "cs",
                                         .help = "CSF band-scale multiplier for h/v bands "
                                                 "(default 1.0 = no scaling)",
                                         .offset = offsetof(AdmVulkanState, adm_csf_scale),
                                         .type = VMAF_OPT_TYPE_DOUBLE,
                                         .default_val.d = DEFAULT_ADM_CSF_SCALE,
                                         .min = 0.0,
                                         .max = 100.0,
                                         .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
                                     },
                                     {
                                         .name = "adm_csf_diag_scale",
                                         .alias = "cds",
                                         .help = "CSF band-scale multiplier for diagonal bands "
                                                 "(default 1.0 = no scaling)",
                                         .offset = offsetof(AdmVulkanState, adm_csf_diag_scale),
                                         .type = VMAF_OPT_TYPE_DOUBLE,
                                         .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
                                         .min = 0.0,
                                         .max = 100.0,
                                         .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
                                     },
                                     {
                                         .name = "adm_noise_weight",
                                         .alias = "nw",
                                         .help = "noise floor weight for CM numerator "
                                                 "(default 0.03125 = 1/32)",
                                         .offset = offsetof(AdmVulkanState, adm_noise_weight),
                                         .type = VMAF_OPT_TYPE_DOUBLE,
                                         .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
                                         .min = 0.0,
                                         .max = 100.0,
                                         .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
                                     },
                                     {0}};

/* ------------------------------------------------------------------ */
/* Push constants — must mirror `Params` in adm.comp.                  */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t cur_w;
    uint32_t cur_h;
    uint32_t half_w;
    uint32_t half_h;
    uint32_t buf_stride;
    uint32_t in_stride;
    uint32_t v_shift;
    uint32_t v_add;
    uint32_t h_shift;
    uint32_t i_rfactor_h;
    uint32_t i_rfactor_v;
    uint32_t i_rfactor_d;
    uint32_t num_workgroups_x;
    int32_t gain_hi;
    int32_t gain_lo;
    int32_t active_left;
    int32_t active_top;
    int32_t active_right;
    int32_t active_bottom;
    uint32_t csf_shift_sq;
    uint32_t csf_shift_cub;
    uint32_t csf_shift_accum;
    uint32_t cm_shift_inner;
    uint32_t cm_shift_sub_h;
    uint32_t cm_shift_sub_d;
    uint32_t cm_shift_xsq_h;
    uint32_t cm_shift_xsq_d;
    uint32_t cm_shift_xcub_h;
    uint32_t cm_shift_xcub_d;
} AdmPushConsts;

/* ------------------------------------------------------------------ */
/* Q31 gain split helper.                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int32_t gain_hi;
    int32_t gain_lo;
} GainQ31;

static GainQ31 gain_to_q31(double gain_limit)
{
    int64_t q = (int64_t)llround(gain_limit * (double)(1LL << 31));
    GainQ31 g = {(int32_t)(q >> 16), (int32_t)(q & 0xFFFF)};
    return g;
}

/* ------------------------------------------------------------------ */
/* Pipeline / descriptor-set layout creation.                          */
/* ------------------------------------------------------------------ */

struct AdmSpecData {
    int32_t width;
    int32_t height;
    int32_t bpc;
    int32_t scale;
    int32_t stage;
};

static void adm_fill_spec(struct AdmSpecData *spec_data, VkSpecializationMapEntry map[5],
                          VkSpecializationInfo *spec_info, const AdmVulkanState *s, int stage,
                          int scale)
{
    spec_data->width = (int32_t)s->width;
    spec_data->height = (int32_t)s->height;
    spec_data->bpc = (int32_t)s->bpc;
    spec_data->scale = scale;
    spec_data->stage = stage;
    map[0] = (VkSpecializationMapEntry){.constantID = 0, .offset = 0, .size = sizeof(int32_t)};
    map[1] = (VkSpecializationMapEntry){.constantID = 1, .offset = 4, .size = sizeof(int32_t)};
    map[2] = (VkSpecializationMapEntry){.constantID = 2, .offset = 8, .size = sizeof(int32_t)};
    map[3] = (VkSpecializationMapEntry){.constantID = 3, .offset = 12, .size = sizeof(int32_t)};
    map[4] = (VkSpecializationMapEntry){.constantID = 4, .offset = 16, .size = sizeof(int32_t)};
    *spec_info = (VkSpecializationInfo){
        .mapEntryCount = 5,
        .pMapEntries = map,
        .dataSize = sizeof(*spec_data),
        .pData = spec_data,
    };
}

static int build_pipeline_for(AdmVulkanState *s, int stage, int scale, VkPipeline *out_pipeline)
{
    struct AdmSpecData spec_data = {0};
    VkSpecializationMapEntry map[5];
    VkSpecializationInfo spec_info = {0};
    adm_fill_spec(&spec_data, map, &spec_info, s, stage, scale);

    VkComputePipelineCreateInfo cpci = {
        .stage =
            {
                .pName = "main",
                .pSpecializationInfo = &spec_info,
            },
    };
    return vmaf_vulkan_kernel_pipeline_add_variant(s->ctx, &s->pl, &cpci, out_pipeline);
}

static int create_pipelines(AdmVulkanState *s)
{
    /* (stage=0, scale=0) is the base pipeline. The template owns
     * the shared layout / shader / DSL / pool plus this base.
     * 9 SSBOs per set; pool sized for 4 frames × 16 (stage,scale)
     * pairs = 64 sets. */
    struct AdmSpecData spec_data = {0};
    VkSpecializationMapEntry map[5];
    VkSpecializationInfo spec_info = {0};
    adm_fill_spec(&spec_data, map, &spec_info, s, /*stage=*/0, /*scale=*/0);

    const VmafVulkanKernelPipelineDesc desc = {
        .ssbo_binding_count = 9U,
        .push_constant_size = (uint32_t)sizeof(AdmPushConsts),
        .spv_bytes = adm_spv,
        .spv_size = adm_spv_size,
        .pipeline_create_info =
            {
                .stage =
                    {
                        .pName = "main",
                        .pSpecializationInfo = &spec_info,
                    },
            },
        .max_descriptor_sets = (uint32_t)(ADM_NUM_SCALES * ADM_NUM_STAGES * 4),
    };
    int err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &desc, &s->pl);
    if (err)
        return err;
    s->pipelines[0][0] = s->pl.pipeline;

    /* The remaining 15 (stage, scale) variants — same layout,
     * shader, DSL, pool, different stage/scale spec-constants. */
    for (int stage = 0; stage < ADM_NUM_STAGES; stage++) {
        for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
            if (stage == 0 && scale == 0)
                continue;
            err = build_pipeline_for(s, stage, scale, &s->pipelines[stage][scale]);
            if (err)
                return err;
        }
    }
    /* ADR-0350: two-level reduction pipeline.
     * 2 SSBO bindings: accum_in (per-WG slots), accum_out (48-byte result).
     * Pool sized for 4 scales × 2 frames = 8 sets. */
    {
        const VmafVulkanKernelPipelineDesc reduce_desc = {
            .ssbo_binding_count = 2U,
            .push_constant_size = (uint32_t)sizeof(uint32_t),
            .spv_bytes = adm_reduce_spv,
            .spv_size = adm_reduce_spv_size,
            .pipeline_create_info = {.stage = {.pName = "main"}},
            .max_descriptor_sets = (uint32_t)(ADM_NUM_SCALES * 2),
        };
        err = vmaf_vulkan_kernel_pipeline_create(s->ctx, &reduce_desc, &s->pl_reduce);
        if (err)
            return err;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Buffer allocation.                                                  */
/* ------------------------------------------------------------------ */

static int alloc_buffers(AdmVulkanState *s)
{
    unsigned w = s->width;
    unsigned h = s->height;
    unsigned half_w0 = (w + 1) / 2;
    unsigned half_h0 = (h + 1) / 2;
    s->buf_stride = (half_w0 + 3u) & ~3u;

    /* Source planes (scale 0 only). */
    size_t bpp = (s->bpc <= 8) ? 1 : 2;
    size_t src_bytes = (size_t)w * h * bpp;
    int err = vmaf_vulkan_buffer_alloc(s->ctx, &s->src_ref, src_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->src_dis, src_bytes);
    if (err)
        return err;

    /* DWT scratch: max size occurs at scale 0: cur_w*2 × half_h elements. */
    size_t dwt_bytes = (size_t)w * 2 * half_h0 * sizeof(int32_t);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dwt_tmp_ref, dwt_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dwt_tmp_dis, dwt_bytes);
    if (err)
        return err;

    /* Band buffers: 4 bands × buf_stride × half_h, max at scale 0. */
    size_t band_bytes = (size_t)4 * s->buf_stride * half_h0 * sizeof(int32_t);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->ref_band, band_bytes);
    if (err)
        return err;
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->dis_band, band_bytes);
    if (err)
        return err;

    /* csf_f: 3 bands × buf_stride × half_h. */
    size_t csf_bytes = (size_t)3 * s->buf_stride * half_h0 * sizeof(int32_t);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->csf_f, csf_bytes);
    if (err)
        return err;

    /* div_lookup. */
    size_t div_bytes = (size_t)ADM_DIV_LOOKUP_SIZE * sizeof(int32_t);
    err = vmaf_vulkan_buffer_alloc(s->ctx, &s->div_lookup, div_bytes);
    if (err)
        return err;
    int32_t *lut = vmaf_vulkan_buffer_host(s->div_lookup);
    memset(lut, 0, div_bytes);
    static const int32_t Q_factor = 1073741824; /* 2^30 */
    for (int i = 1; i <= 32768; i++) {
        int32_t recip = (int32_t)(Q_factor / i);
        lut[32768 + i] = recip;
        lut[32768 - i] = -recip;
    }
    err = vmaf_vulkan_buffer_flush(s->ctx, s->div_lookup);
    if (err)
        return err;

    /* Per-scale accumulators. The stage-3 launch grid is
     *    num_wg = 3 × num_active_rows
     * each WG writes 6 int64 slots. We allocate the upper bound for
     * each scale based on its dimensions. */
    unsigned cw = w, ch = h;
    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        unsigned hw = (cw + 1) / 2;
        unsigned hh = (ch + 1) / 2;
        s->scale_w[scale] = cw;
        s->scale_h[scale] = ch;
        s->scale_half_w[scale] = hw;
        s->scale_half_h[scale] = hh;

        int top = (int)((double)hh * ADM_BORDER_FACTOR - 0.5);
        int bottom = (int)hh - top;
        if (top < 0)
            top = 0;
        unsigned num_rows = (unsigned)(bottom - top);
        if (num_rows == 0)
            num_rows = 1;
        unsigned wg_count = 3u * num_rows;
        s->wg_count[scale] = wg_count;
        size_t accum_bytes = (size_t)wg_count * ADM_ACCUM_SLOTS_PER_WG * sizeof(int64_t);
        if (accum_bytes == 0)
            accum_bytes = sizeof(int64_t);
        err = vmaf_vulkan_buffer_alloc_readback(s->ctx, &s->accum[scale], accum_bytes);
        if (err)
            return err;

        /* ADR-0350: tiny reduced-accumulator output buffer — 48 bytes per scale. */
        size_t reduced_bytes = (size_t)ADM_ACCUM_SLOTS_PER_WG * sizeof(int64_t);
        err = vmaf_vulkan_buffer_alloc(s->ctx, &s->reduced_accum[scale], reduced_bytes);
        if (err)
            return err;

        cw = hw;
        ch = hh;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* init().                                                             */
/* ------------------------------------------------------------------ */

/* Forward declarations — definitions are below `init()`, but `init()`
 * calls them via the per-scale loop after the descriptor sets are
 * pre-allocated. Without an explicit `static` forward decl the C
 * compiler creates an implicit non-static one at the call site, then
 * fails with "static declaration follows non-static declaration"
 * when it sees the actual definition further down. */
static void write_descriptor_set(AdmVulkanState *s, VkDescriptorSet set);
static void write_accum_binding(AdmVulkanState *s, VkDescriptorSet set, int scale);

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;
    AdmVulkanState *s = fex->priv;
    s->width = w;
    s->height = h;
    s->bpc = bpc;

    /* Compute rfactors (host-side; same logic as SYCL init).
     * adm_csf_scale / adm_csf_diag_scale multiply the CSF sensitivity.
     * Default 1.0 → identical to the pre-PR-731 behaviour. */
    for (unsigned scale = 0; scale < ADM_NUM_SCALES; scale++) {
        float f1 =
            dwt_quant_step_host((int)scale, 1, s->adm_norm_view_dist, s->adm_ref_display_height);
        float f2 =
            dwt_quant_step_host((int)scale, 2, s->adm_norm_view_dist, s->adm_ref_display_height);
        s->rfactor[scale * 3 + 0] = (float)s->adm_csf_scale / f1;
        s->rfactor[scale * 3 + 1] = (float)s->adm_csf_scale / f1;
        s->rfactor[scale * 3 + 2] = (float)s->adm_csf_diag_scale / f2;

        double pow2_32 = pow(2.0, 32);
        double pow2_21 = pow(2.0, 21);
        double pow2_23 = pow(2.0, 23);
        if (scale == 0) {
            /* Hardcoded scale-0 i_rfactor constants are only valid when both
             * adm_csf_scale and adm_csf_diag_scale are exactly 1.0 (default)
             * and norm_view_dist/ref_display_height are at their defaults. */
            double default_check = 3.0 * 1080.0;
            double actual = s->adm_norm_view_dist * (double)s->adm_ref_display_height;
            bool csf_default =
                (fabs(s->adm_csf_scale - 1.0) < 1e-9) && (fabs(s->adm_csf_diag_scale - 1.0) < 1e-9);
            if (fabs(actual - default_check) < 1e-8 && csf_default) {
                s->i_rfactor[0] = 36453;
                s->i_rfactor[1] = 36453;
                s->i_rfactor[2] = 49417;
            } else {
                s->i_rfactor[0] = (uint32_t)((double)s->rfactor[0] * pow2_21);
                s->i_rfactor[1] = (uint32_t)((double)s->rfactor[1] * pow2_21);
                s->i_rfactor[2] = (uint32_t)((double)s->rfactor[2] * pow2_23);
            }
        } else {
            s->i_rfactor[scale * 3 + 0] = (uint32_t)((double)s->rfactor[scale * 3 + 0] * pow2_32);
            s->i_rfactor[scale * 3 + 1] = (uint32_t)((double)s->rfactor[scale * 3 + 1] * pow2_32);
            s->i_rfactor[scale * 3 + 2] = (uint32_t)((double)s->rfactor[scale * 3 + 2] * pow2_32);
        }
    }

    /* Borrow framework's imported context, fall back to lazy create. */
    s->ctx = vmaf_vulkan_state_get_context(fex->vulkan_state);
    if (s->ctx) {
        s->owns_ctx = 0;
    } else {
        int err = vmaf_vulkan_context_new(&s->ctx, /*device_index=*/-1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "adm_vulkan: cannot create Vulkan context (%d)\n", err);
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

    err = vmaf_vulkan_kernel_submit_pool_create(s->ctx, /*slot_count=*/1, &s->sub_pool);
    if (err)
        return err;

    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl.desc_pool, s->pl.dsl,
                                                   (uint32_t)ADM_NUM_SCALES, s->pre_sets);
    if (err)
        return err;

    /* Write all 9 bindings once — all are init-time-stable.
     * write_descriptor_set fills slots 0-6 and 8; write_accum_binding
     * fills slot 7 (one per-scale accum buffer). Neither changes
     * between frames, so no per-frame vkUpdateDescriptorSets is
     * needed in extract(). (T-GPU-OPT-VK-4 / ADR-0256.) */
    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        write_descriptor_set(s, s->pre_sets[scale]);
        write_accum_binding(s, s->pre_sets[scale], scale);
    }

    /* ADR-0350 follow-up (2026-05-09 review-fix): pre-allocate the
     * reducer descriptor sets ONCE here and reuse them every frame.
     * The accum_in / reduced_accum bindings are stable per-scale, so
     * we write them once after the buffers are allocated.  This
     * mirrors the VIF reducer pattern (s->reduce_sets in
     * vif_vulkan.c) and removes the per-frame
     * vkAllocateDescriptorSets call from extract() — over a long
     * sequence the prior per-frame alloc/free pattern fragmented
     * the descriptor pool and could exhaust it on drivers that do
     * not coalesce freed entries.  See PR #561 review comment 2. */
    err = vmaf_vulkan_kernel_descriptor_sets_alloc(s->ctx, s->pl_reduce.desc_pool, s->pl_reduce.dsl,
                                                   (uint32_t)ADM_NUM_SCALES, s->reduce_sets);
    if (err)
        return err;
    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        VkDescriptorBufferInfo rdbi[2] = {
            {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->accum[scale]),
             .offset = 0,
             .range = VK_WHOLE_SIZE},
            {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->reduced_accum[scale]),
             .offset = 0,
             .range = VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet rwrites[2];
        for (int i = 0; i < 2; i++) {
            rwrites[i] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = s->reduce_sets[scale],
                .dstBinding = (uint32_t)i,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &rdbi[i],
            };
        }
        vkUpdateDescriptorSets(s->ctx->device, 2, rwrites, 0, NULL);
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-frame helpers.                                                  */
/* ------------------------------------------------------------------ */

static int upload_plane(AdmVulkanState *s, VmafPicture *pic, VmafVulkanBuffer *buf)
{
    uint8_t *dst = vmaf_vulkan_buffer_host(buf);
    const uint8_t *src = (const uint8_t *)pic->data[0];
    size_t src_stride = pic->stride[0];
    size_t dst_stride = (s->bpc <= 8) ? s->width : (s->width * 2);
    for (unsigned y = 0; y < s->height; y++)
        memcpy(dst + y * dst_stride, src + y * src_stride, dst_stride);
    return vmaf_vulkan_buffer_flush(s->ctx, buf);
}

static void write_descriptor_set(AdmVulkanState *s, VkDescriptorSet set)
{
    VkDescriptorBufferInfo dbi[9] = {
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->src_ref),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dwt_tmp_ref),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dwt_tmp_dis),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->ref_band),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->dis_band),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->csf_f),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->div_lookup),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = VK_NULL_HANDLE, /* accum bound per-scale below */
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->src_dis),
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[9];
    for (int i = 0; i < 9; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    /* Skip slot 7 (accum) for now — written separately per scale. */
    vkUpdateDescriptorSets(s->ctx->device, 9, writes, 0, NULL);
}

static void write_accum_binding(AdmVulkanState *s, VkDescriptorSet set, int scale)
{
    VkDescriptorBufferInfo dbi = {
        .buffer = (VkBuffer)vmaf_vulkan_buffer_vkhandle(s->accum[scale]),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkWriteDescriptorSet w = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = 7,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &dbi,
    };
    vkUpdateDescriptorSets(s->ctx->device, 1, &w, 0, NULL);
}

/* ------------------------------------------------------------------ */
/* Score reduction (mirrors SYCL conclude_adm_csf_den / conclude_adm_cm). */
/* ------------------------------------------------------------------ */

static void conclude_csf_den_host(const uint64_t accum[3], int hh, int hw, int scale,
                                  float view_dist, float display_h, float adm_csf_scale,
                                  float adm_csf_diag_scale, float noise_weight, float *result)
{
    int left = (int)((double)hw * ADM_BORDER_FACTOR - 0.5);
    int top = (int)((double)hh * ADM_BORDER_FACTOR - 0.5);
    int right = hw - left;
    int bottom = hh - top;

    float factor1 = dwt_quant_step_host(scale, 1, view_dist, (int)display_h);
    float factor2 = dwt_quant_step_host(scale, 2, view_dist, (int)display_h);
    /* Apply CSF scale multipliers; default 1.0 → no change. */
    float rfactor[3] = {adm_csf_scale / factor1, adm_csf_scale / factor1,
                        adm_csf_diag_scale / factor2};
    static const uint32_t accum_convert[4] = {18, 32, 27, 23};

    int32_t shift_accum;
    double shift_csf;
    if (scale == 0) {
        shift_accum = (int32_t)ceil(log2((double)((bottom - top) * (right - left))) - 20.0);
        if (shift_accum < 0)
            shift_accum = 0;
        shift_csf = pow(2.0, (double)accum_convert[scale] - (double)shift_accum);
    } else {
        shift_accum = (int32_t)ceil(log2((double)(bottom - top)));
        uint32_t shift_cub = (uint32_t)ceil(log2((double)(right - left)));
        shift_csf =
            pow(2.0, (double)accum_convert[scale] - (double)shift_accum - (double)shift_cub);
    }
    float powf_add = powf((float)((bottom - top) * (right - left)) * noise_weight, 1.0f / 3.0f);
    *result = 0.0f;
    for (int i = 0; i < 3; i++) {
        double csf = (double)accum[i] / shift_csf * pow((double)rfactor[i], 3.0);
        *result += powf((float)csf, 1.0f / 3.0f) + powf_add;
    }
}

static void conclude_cm_host(const int64_t accum[3], int hh, int hw, int scale, float noise_weight,
                             float *result)
{
    int left = (int)((double)hw * ADM_BORDER_FACTOR - 0.5);
    int top = (int)((double)hh * ADM_BORDER_FACTOR - 0.5);
    int right = hw - left;
    int bottom = hh - top;
    uint32_t shift_inner_accum = (uint32_t)ceil(log2((double)hh));
    float powf_add = powf((float)((bottom - top) * (right - left)) * noise_weight, 1.0f / 3.0f);
    *result = 0.0f;
    for (int i = 0; i < 3; i++) {
        float f_accum;
        if (scale == 0) {
            uint32_t shift_xcub[3] = {
                (uint32_t)(ceil(log2((double)hw)) - 4.0),
                (uint32_t)(ceil(log2((double)hw)) - 4.0),
                (uint32_t)(ceil(log2((double)hw)) - 3.0),
            };
            int constant_offset[3] = {52, 52, 57};
            f_accum = (float)((double)accum[i] /
                              pow(2.0, (double)constant_offset[i] - (double)shift_xcub[i] -
                                           (double)shift_inner_accum));
        } else {
            uint32_t shift_cub = (uint32_t)ceil(log2((double)hw));
            float final_shift[3] = {
                powf(2.0f, 45.0f - (float)shift_cub - (float)shift_inner_accum),
                powf(2.0f, 39.0f - (float)shift_cub - (float)shift_inner_accum),
                powf(2.0f, 36.0f - (float)shift_cub - (float)shift_inner_accum),
            };
            f_accum = (float)((double)accum[i] / (double)final_shift[scale - 1]);
        }
        *result += powf(f_accum, 1.0f / 3.0f) + powf_add;
    }
}

/* ------------------------------------------------------------------ */
/* extract().                                                          */
/* ------------------------------------------------------------------ */

static void compute_csf_cm_shifts(unsigned hw, unsigned hh, int scale, int active_w, int active_h,
                                  AdmPushConsts *pc)
{
    if (scale == 0) {
        pc->csf_shift_sq = 0;
        pc->csf_shift_cub = 0;
        int area = active_w * active_h;
        int sa = (int)ceil(log2((double)area) - 20.0);
        pc->csf_shift_accum = sa > 0 ? (uint32_t)sa : 0u;
    } else {
        pc->csf_shift_sq = (scale == 2) ? 30u : 31u;
        pc->csf_shift_cub = (uint32_t)ceil(log2((double)active_w));
        pc->csf_shift_accum = (uint32_t)ceil(log2((double)active_h));
    }
    pc->cm_shift_inner = (uint32_t)ceil(log2((double)hh));
    if (scale == 0) {
        pc->cm_shift_sub_h = 10u;
        pc->cm_shift_sub_d = 12u;
        pc->cm_shift_xsq_h = 29u;
        pc->cm_shift_xsq_d = 30u;
        pc->cm_shift_xcub_h = (uint32_t)(ceil(log2((double)hw)) - 4.0);
        pc->cm_shift_xcub_d = (uint32_t)(ceil(log2((double)hw)) - 3.0);
    } else {
        pc->cm_shift_sub_h = 15u;
        pc->cm_shift_sub_d = 15u;
        pc->cm_shift_xsq_h = 30u;
        pc->cm_shift_xsq_d = 30u;
        pc->cm_shift_xcub_h = (uint32_t)ceil(log2((double)hw));
        pc->cm_shift_xcub_d = (uint32_t)ceil(log2((double)hw));
    }
}

static void issue_pipeline_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static int reduce_and_emit(AdmVulkanState *s, unsigned index, VmafFeatureCollector *fc)
{
    /* Per-scale int64 host-side reduction. The 6 slots per WG layout
     * is csf_h,csf_v,csf_d, cm_h,cm_v,cm_d. */
    int64_t cm_totals[ADM_NUM_SCALES][ADM_NUM_BANDS] = {0};
    int64_t csf_totals[ADM_NUM_SCALES][ADM_NUM_BANDS] = {0};

    /* ADR-0350: read the GPU-reduced 48-byte result per scale instead of
     * iterating over all per-WG slots.  56 → 48 bytes because ADM has 6
     * fields (csf_h/v/d + cm_h/v/d) vs VIF's 7.  Total host read:
     * 4 × 48 = 192 bytes (was 4 × wg_count × 48, ~1.1 MB at 1080p). */
    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        (void)vmaf_vulkan_buffer_invalidate(s->ctx, s->reduced_accum[scale]);
        const int64_t *r = vmaf_vulkan_buffer_host(s->reduced_accum[scale]);
        for (int b = 0; b < ADM_NUM_BANDS; b++) {
            csf_totals[scale][b] = r[b];
            cm_totals[scale][b] = r[3 + b];
        }
    }

    /* Score per scale. */
    double num = 0.0, den = 0.0;
    double scores_num[ADM_NUM_SCALES];
    double scores_den[ADM_NUM_SCALES];

    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        int hw = (int)s->scale_half_w[scale];
        int hh = (int)s->scale_half_h[scale];
        float num_scale = 0.0f;
        float den_scale = 0.0f;
        conclude_cm_host(cm_totals[scale], hh, hw, scale, (float)s->adm_noise_weight, &num_scale);
        conclude_csf_den_host((const uint64_t *)csf_totals[scale], hh, hw, scale,
                              (float)s->adm_norm_view_dist, (float)s->adm_ref_display_height,
                              (float)s->adm_csf_scale, (float)s->adm_csf_diag_scale,
                              (float)s->adm_noise_weight, &den_scale);
        num += num_scale;
        den += den_scale;
        scores_num[scale] = num_scale;
        scores_den[scale] = den_scale;
    }

    int last_w = (int)s->scale_half_w[ADM_NUM_SCALES - 1];
    int last_h = (int)s->scale_half_h[ADM_NUM_SCALES - 1];
    double numden_limit = 1e-10 * ((double)last_w * (double)last_h) / (1920.0 * 1080.0);
    if (num < numden_limit)
        num = 0.0;
    if (den < numden_limit)
        den = 0.0;
    double score = (den == 0.0) ? 1.0 : num / den;

    int err = vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_integer_feature_adm2_score", score, index);

    for (int i = 0; i < ADM_NUM_SCALES && !err; i++) {
        char name[64];
        snprintf(name, sizeof(name), "integer_adm_scale%d", i);
        double scale_score = (scores_den[i] == 0.0) ? 1.0 : scores_num[i] / scores_den[i];
        err = vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, name, scale_score,
                                                      index);
    }

    if (s->debug && !err) {
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "integer_adm", score,
                                                index);
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "integer_adm_num", num,
                                                index);
        vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "integer_adm_den", den,
                                                index);
        for (int i = 0; i < ADM_NUM_SCALES; i++) {
            char name[64];
            snprintf(name, sizeof(name), "integer_adm_num_scale%d", i);
            vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, name, scores_num[i],
                                                    index);
            snprintf(name, sizeof(name), "integer_adm_den_scale%d", i);
            vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, name, scores_den[i],
                                                    index);
        }
    }
    return err;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    AdmVulkanState *s = fex->priv;
    int err = 0;

    err = upload_plane(s, ref_pic, s->src_ref);
    if (err)
        return err;
    err = upload_plane(s, dist_pic, s->src_dis);
    if (err)
        return err;

    /* Zero per-scale accumulator buffers (host-mapped). */
    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        size_t bytes = (size_t)s->wg_count[scale] * ADM_ACCUM_SLOTS_PER_WG * sizeof(int64_t);
        memset(vmaf_vulkan_buffer_host(s->accum[scale]), 0, bytes);
        err = vmaf_vulkan_buffer_flush(s->ctx, s->accum[scale]);
        if (err)
            return err;
        /* ADR-0350: zero the reduced output (atomicAdd accumulates into it). */
        memset(vmaf_vulkan_buffer_host(s->reduced_accum[scale]), 0,
               (size_t)ADM_ACCUM_SLOTS_PER_WG * sizeof(int64_t));
        err = vmaf_vulkan_buffer_flush(s->ctx, s->reduced_accum[scale]);
        if (err)
            return err;
    }

    /* All 9 descriptor bindings are init-time-stable (the accum buffers
     * are allocated once in init() and never reallocated). No per-frame
     * vkUpdateDescriptorSets needed. (T-GPU-OPT-VK-4 / ADR-0256.) */
    /* ADR-0350 follow-up (2026-05-09 review-fix): the reducer
     * descriptor sets are pre-allocated and written once in init().
     * The bindings (accum[scale] -> reduced_accum[scale]) never
     * change at runtime, so per-frame vkAllocateDescriptorSets +
     * vkFreeDescriptorSets is unnecessary and on long sequences was
     * fragmenting the small reducer pool.  We just use s->reduce_sets
     * directly below.  See PR #561 review comment 2. */

    /* Acquire a pre-allocated command buffer + fence from the submit pool.
     * Eliminates per-frame vkAllocateCommandBuffers / vkCreateFence.
     * (T-GPU-OPT-VK-1 / ADR-0256.) */
    VmafVulkanKernelSubmit submit = {0};
    err = vmaf_vulkan_kernel_submit_acquire(s->ctx, &s->sub_pool, /*pool_slot=*/0, &submit);
    if (err)
        return err;
    VkCommandBuffer cmd = submit.cmd;

    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        unsigned cw = s->scale_w[scale];
        unsigned ch = s->scale_h[scale];
        unsigned hw = s->scale_half_w[scale];
        unsigned hh = s->scale_half_h[scale];

        /* DWT shift parameters per scale (matches SYCL dwt_shifts). */
        unsigned v_shift, v_add, h_shift;
        if (scale == 0) {
            v_shift = s->bpc;
            v_add = 1u << (s->bpc - 1u);
            h_shift = 16u;
        } else if (scale == 1) {
            v_shift = 0u;
            v_add = 0u;
            h_shift = 15u;
        } else if (scale == 2) {
            v_shift = 16u;
            v_add = 32768u;
            h_shift = 16u;
        } else {
            v_shift = 16u;
            v_add = 32768u;
            h_shift = 15u;
        }

        int left = (int)((double)hw * ADM_BORDER_FACTOR - 0.5);
        int top = (int)((double)hh * ADM_BORDER_FACTOR - 0.5);
        if (left < 0)
            left = 0;
        if (top < 0)
            top = 0;
        int right = (int)hw - left;
        int bottom = (int)hh - top;
        int active_w = right - left;
        int active_h = bottom - top;

        unsigned in_stride;
        if (scale == 0)
            in_stride = (s->bpc <= 8) ? cw : cw * 2u;
        else
            in_stride = s->buf_stride;

        AdmPushConsts pc = {
            .cur_w = cw,
            .cur_h = ch,
            .half_w = hw,
            .half_h = hh,
            .buf_stride = s->buf_stride,
            .in_stride = in_stride,
            .v_shift = v_shift,
            .v_add = v_add,
            .h_shift = h_shift,
            .i_rfactor_h = s->i_rfactor[scale * 3 + 0],
            .i_rfactor_v = s->i_rfactor[scale * 3 + 1],
            .i_rfactor_d = s->i_rfactor[scale * 3 + 2],
            .num_workgroups_x = 1,
            .active_left = left,
            .active_top = top,
            .active_right = right,
            .active_bottom = bottom,
        };
        GainQ31 gq = gain_to_q31(s->adm_enhn_gain_limit);
        pc.gain_hi = gq.gain_hi;
        pc.gain_lo = gq.gain_lo;
        compute_csf_cm_shifts(hw, hh, scale, active_w, active_h, &pc);

        /* Stage 0: DWT vertical (z=2 fused ref+dis). */
        {
            uint32_t gx = (cw + ADM_WG_X - 1u) / ADM_WG_X;
            uint32_t gy = (hh + ADM_WG_Y - 1u) / ADM_WG_Y;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[0][scale]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl.pipeline_layout, 0,
                                    1, &s->pre_sets[scale], 0, NULL);
            vkCmdPushConstants(cmd, s->pl.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pc), &pc);
            vkCmdDispatch(cmd, gx, gy, 2);
            issue_pipeline_barrier(cmd);
        }

        /* Stage 1: DWT horizontal (z=2 fused ref+dis). */
        {
            uint32_t gx = (hw + ADM_WG_X - 1u) / ADM_WG_X;
            uint32_t gy = (hh + ADM_WG_Y - 1u) / ADM_WG_Y;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[1][scale]);
            vkCmdDispatch(cmd, gx, gy, 2);
            issue_pipeline_barrier(cmd);
        }

        /* Stage 2: Decouple + CSF (writes csf_f). */
        {
            uint32_t gx = (hw + ADM_WG_X - 1u) / ADM_WG_X;
            uint32_t gy = (hh + ADM_WG_Y - 1u) / ADM_WG_Y;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[2][scale]);
            vkCmdDispatch(cmd, gx, gy, 1);
            issue_pipeline_barrier(cmd);
        }

        /* Stage 3: CSF denominator + CM fused. 1D dispatch over
         *   3 bands × num_active_rows. */
        {
            unsigned num_rows = (unsigned)(active_h > 0 ? active_h : 1);
            uint32_t gx = 3u * num_rows;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipelines[3][scale]);
            vkCmdDispatch(cmd, gx, 1u, 1u);
            issue_pipeline_barrier(cmd);
        }
    }

    /* ADR-0350: flush per-WG accum SSBOs, then run GPU reducer per scale.
     * All four reductions are in the same command buffer — no queueWaitIdle. */
    {
        VkMemoryBarrier reduce_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &reduce_barrier, 0, NULL,
                             0, NULL);
        for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
            uint32_t wg_count_scale = (uint32_t)s->wg_count[scale];
            uint32_t n_rg = (wg_count_scale + 255u) / 256u;
            if (n_rg == 0u)
                n_rg = 1u;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_reduce.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    s->pl_reduce.pipeline_layout, 0, 1, &s->reduce_sets[scale], 0,
                                    NULL);
            vkCmdPushConstants(cmd, s->pl_reduce.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(uint32_t), &wg_count_scale);
            vkCmdDispatch(cmd, n_rg, 1u, 1u);
        }
    }

    err = vmaf_vulkan_kernel_submit_end_and_wait(s->ctx, &submit);
    if (err)
        goto cleanup;

    err = reduce_and_emit(s, index, feature_collector);

cleanup:
    vmaf_vulkan_kernel_submit_free(s->ctx, &submit);
    /* ADR-0350 follow-up (2026-05-09 review-fix): s->reduce_sets are
     * pre-allocated at init() time and freed implicitly when the
     * reducer descriptor pool is destroyed in close_fex() —
     * no per-frame free here. */
    return err;
}

/* ------------------------------------------------------------------ */
/* close().                                                            */
/* ------------------------------------------------------------------ */

static int close_fex(VmafFeatureExtractor *fex)
{
    AdmVulkanState *s = fex->priv;
    if (!s->ctx)
        return 0;
    VkDevice dev = s->ctx->device;
    vkDeviceWaitIdle(dev);

    /* Destroy the 15 sibling variants first (every (stage, scale)
     * except the (0, 0) base); the base + shared layout / shader /
     * DSL / pool are owned by the template. (0, 0) is skipped to
     * avoid double-freeing — pipelines[0][0] aliases s->pl.pipeline. */
    for (int stage = 0; stage < ADM_NUM_STAGES; stage++) {
        for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
            if (stage == 0 && scale == 0)
                continue;
            if (s->pipelines[stage][scale] != VK_NULL_HANDLE)
                vkDestroyPipeline(dev, s->pipelines[stage][scale], NULL);
        }
    }
    vmaf_vulkan_kernel_submit_pool_destroy(s->ctx, &s->sub_pool);
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl);

    if (s->src_ref)
        vmaf_vulkan_buffer_free(s->ctx, s->src_ref);
    if (s->src_dis)
        vmaf_vulkan_buffer_free(s->ctx, s->src_dis);
    if (s->dwt_tmp_ref)
        vmaf_vulkan_buffer_free(s->ctx, s->dwt_tmp_ref);
    if (s->dwt_tmp_dis)
        vmaf_vulkan_buffer_free(s->ctx, s->dwt_tmp_dis);
    if (s->ref_band)
        vmaf_vulkan_buffer_free(s->ctx, s->ref_band);
    if (s->dis_band)
        vmaf_vulkan_buffer_free(s->ctx, s->dis_band);
    if (s->csf_f)
        vmaf_vulkan_buffer_free(s->ctx, s->csf_f);
    if (s->div_lookup)
        vmaf_vulkan_buffer_free(s->ctx, s->div_lookup);
    for (int scale = 0; scale < ADM_NUM_SCALES; scale++) {
        if (s->accum[scale])
            vmaf_vulkan_buffer_free(s->ctx, s->accum[scale]);
        if (s->reduced_accum[scale])
            vmaf_vulkan_buffer_free(s->ctx, s->reduced_accum[scale]);
    }
    vmaf_vulkan_kernel_pipeline_destroy(s->ctx, &s->pl_reduce);

    if (s->owns_ctx)
        vmaf_vulkan_context_destroy(s->ctx);
    s->ctx = NULL;
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Provided features + registration.                                   */
/* Matches the SYCL ADM extractor (no aim_score / adm3_score).         */
/* ------------------------------------------------------------------ */

static const char *provided_features[] = {"VMAF_integer_feature_adm2_score",
                                          "integer_adm_scale0",
                                          "integer_adm_scale1",
                                          "integer_adm_scale2",
                                          "integer_adm_scale3",
                                          "integer_adm",
                                          "integer_adm_num",
                                          "integer_adm_den",
                                          "integer_adm_num_scale0",
                                          "integer_adm_den_scale0",
                                          "integer_adm_num_scale1",
                                          "integer_adm_den_scale1",
                                          "integer_adm_num_scale2",
                                          "integer_adm_den_scale2",
                                          "integer_adm_num_scale3",
                                          "integer_adm_den_scale3",
                                          NULL};

VmafFeatureExtractor vmaf_fex_integer_adm_vulkan = {
    .name = "adm_vulkan",
    .init = init,
    .extract = extract,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(AdmVulkanState),
    .flags = VMAF_FEATURE_EXTRACTOR_VULKAN,
    .provided_features = provided_features,
};
