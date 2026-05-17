/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  float_adm feature extractor on the HIP backend — ninth consumer
 *  of `libvmaf/src/hip/kernel_template.h` (T7-10b batch-2 / ADR-0468).
 *
 *  This TU mirrors `libvmaf/src/feature/cuda/float_adm_cuda.c`
 *  call-graph-for-call-graph. When `HAVE_HIPCC` is defined (i.e.,
 *  `enable_hipcc=true` at configure time), the `init`, `submit`, and
 *  `collect` functions use real HIP Module API calls following the
 *  canonical pattern established by PR #612 / ADR-0254.
 *
 *  Without `HAVE_HIPCC` (CPU-only or HIP-scaffold builds), every
 *  lifecycle helper returns -ENOSYS (scaffold posture preserved).
 *
 *  Algorithm: 16-launch (4 stages × 4 scales) DWT+CSF+CM pipeline.
 *  Same four pipeline stages as the CUDA twin, same `-1` mirror form,
 *  same fused stage 3 with cross-band CM threshold.
 *  Host reduction in double precision (places=4 contract).
 *
 *  Key HIP adaptation:
 *  - Module API: `hipModuleLoadData` / `hipModuleGetFunction` /
 *    `hipModuleLaunchKernel` replace CUDA driver-API equivalents.
 *  - Buffer alloc: `hipMalloc` / `hipMemsetAsync` / `hipMemcpyAsync`
 *    replace `vmaf_cuda_buffer_alloc` + `cuMemsetD8Async`.
 *  - Stream/event: `hipStream_t` / `hipEvent_t` from `lc`; same
 *    submit/finished event-fence pattern as every HIP consumer.
 *  - Warp size 64 on GCN/RDNA: shared-memory partial arrays sized
 *    at FADM_WARPS_PER_BLOCK = 4.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "dict.h"
#include "feature/adm_options.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "libvmaf/picture.h"

#include "../../hip/common.h"
#include "../../hip/kernel_template.h"
#include "float_adm_hip.h"

#ifdef HAVE_HIPCC
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

extern const unsigned char float_adm_score_hsaco[];
extern const unsigned int float_adm_score_hsaco_len;
#endif /* HAVE_HIPCC */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FADM_NUM_SCALES 4
#define FADM_NUM_BANDS 3
#define FADM_BX 16
#define FADM_BY 16
#define FADM_BORDER_FACTOR 0.1
#define FADM_ACCUM_SLOTS 6

typedef struct FloatAdmStateHip {
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    int adm_csf_mode;
    double adm_csf_scale;
    double adm_csf_diag_scale;
    double adm_noise_weight;

    unsigned width;
    unsigned height;
    unsigned bpc;
    unsigned buf_stride;

    float rfactor[12];

    VmafHipKernelLifecycle lc;
    VmafHipContext *ctx;

#ifdef HAVE_HIPCC
    hipModule_t module;
    hipFunction_t func_dwt_vert;
    hipFunction_t func_dwt_hori;
    hipFunction_t func_decouple_csf;
    hipFunction_t func_csf_cm;

    void *src_ref;
    void *src_dis;
    void *dwt_tmp_ref;
    void *dwt_tmp_dis;
    void *ref_band[FADM_NUM_SCALES];
    void *dis_band[FADM_NUM_SCALES];
    void *csf_a;
    void *csf_f;
    void *accum[FADM_NUM_SCALES];
    float *accum_host[FADM_NUM_SCALES];
#endif /* HAVE_HIPCC */

    unsigned wg_count[FADM_NUM_SCALES];
    unsigned scale_w[FADM_NUM_SCALES];
    unsigned scale_h[FADM_NUM_SCALES];
    unsigned scale_half_w[FADM_NUM_SCALES];
    unsigned scale_half_h[FADM_NUM_SCALES];

    VmafDictionary *feature_name_dict;
} FloatAdmStateHip;

static const VmafOption options[] = {
    {.name = "debug",
     .help = "debug mode: enable additional output",
     .offset = offsetof(FloatAdmStateHip, debug),
     .type = VMAF_OPT_TYPE_BOOL,
     .default_val.b = false},
    {.name = "adm_enhn_gain_limit",
     .alias = "egl",
     .help = "enhancement gain imposed on adm, must be >= 1.0",
     .offset = offsetof(FloatAdmStateHip, adm_enhn_gain_limit),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = 100.0,
     .min = 1.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_norm_view_dist",
     .alias = "nvd",
     .help = "normalized viewing distance",
     .offset = offsetof(FloatAdmStateHip, adm_norm_view_dist),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = 3.0,
     .min = 0.75,
     .max = 24.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_ref_display_height",
     .alias = "rdf",
     .help = "reference display height in pixels",
     .offset = offsetof(FloatAdmStateHip, adm_ref_display_height),
     .type = VMAF_OPT_TYPE_INT,
     .default_val.i = 1080,
     .min = 1,
     .max = 4320,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_csf_mode",
     .alias = "csf",
     .help = "contrast sensitivity function (mode 0 only on HIP v1)",
     .offset = offsetof(FloatAdmStateHip, adm_csf_mode),
     .type = VMAF_OPT_TYPE_INT,
     .default_val.i = 0,
     .min = 0,
     .max = 9,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_csf_scale",
     .alias = "cs",
     .help = "CSF band-scale multiplier for h/v bands (default 1.0 = no scaling)",
     .offset = offsetof(FloatAdmStateHip, adm_csf_scale),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = DEFAULT_ADM_CSF_SCALE,
     .min = 0.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_csf_diag_scale",
     .alias = "cds",
     .help = "CSF band-scale multiplier for diagonal bands (default 1.0 = no scaling)",
     .offset = offsetof(FloatAdmStateHip, adm_csf_diag_scale),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
     .min = 0.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {.name = "adm_noise_weight",
     .alias = "nw",
     .help = "noise floor weight for CM numerator (default 0.03125 = 1/32)",
     .offset = offsetof(FloatAdmStateHip, adm_noise_weight),
     .type = VMAF_OPT_TYPE_DOUBLE,
     .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
     .min = 0.0,
     .max = 100.0,
     .flags = VMAF_OPT_FLAG_FEATURE_PARAM},
    {0}};

/* DB2/CDF-9-7 wavelet noise model — identical to the CUDA twin. */
static const float fadm_dwt_basis_amp[6][4] = {
    {0.62171f, 0.67234f, 0.72709f, 0.67234f},     {0.34537f, 0.41317f, 0.49428f, 0.41317f},
    {0.18004f, 0.22727f, 0.28688f, 0.22727f},     {0.091401f, 0.11792f, 0.15214f, 0.11792f},
    {0.045943f, 0.059758f, 0.077727f, 0.059758f}, {0.023013f, 0.030018f, 0.039156f, 0.030018f},
};
static const float fadm_dwt_a_Y = 0.495f;
static const float fadm_dwt_k_Y = 0.466f;
static const float fadm_dwt_f0_Y = 0.401f;
static const float fadm_dwt_g_Y[4] = {1.501f, 1.0f, 0.534f, 1.0f};

static float fadm_dwt_quant_step(int lambda, int theta, double view_dist, int display_h)
{
    const float r = (float)(view_dist * (double)display_h * M_PI / 180.0);
    const float temp = (float)log10(pow(2.0, (double)(lambda + 1)) * (double)fadm_dwt_f0_Y *
                                    (double)fadm_dwt_g_Y[theta] / (double)r);
    const float Q = (float)(2.0 * (double)fadm_dwt_a_Y *
                            pow(10.0, (double)fadm_dwt_k_Y * (double)temp * (double)temp) /
                            (double)fadm_dwt_basis_amp[lambda][theta]);
    return Q;
}

static void compute_per_scale_dims(FloatAdmStateHip *s)
{
    unsigned cw = s->width;
    unsigned ch = s->height;
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const unsigned hw = (cw + 1u) / 2u;
        const unsigned hh = (ch + 1u) / 2u;
        s->scale_w[scale] = cw;
        s->scale_h[scale] = ch;
        s->scale_half_w[scale] = hw;
        s->scale_half_h[scale] = hh;
        cw = hw;
        ch = hh;
    }
    s->buf_stride = (s->scale_half_w[0] + 3u) & ~3u;
}

#ifdef HAVE_HIPCC
static int fadm_hip_rc(hipError_t rc)
{
    if (rc == hipSuccess)
        return 0;
    switch (rc) {
    case hipErrorInvalidValue:
    case hipErrorInvalidHandle:
        return -EINVAL;
    case hipErrorOutOfMemory:
        return -ENOMEM;
    case hipErrorNoDevice:
    case hipErrorInvalidDevice:
        return -ENODEV;
    case hipErrorNotSupported:
        return -ENOSYS;
    default:
        return -EIO;
    }
}

static int fadm_hip_module_load(FloatAdmStateHip *s)
{
    hipError_t rc = hipModuleLoadData(&s->module, float_adm_score_hsaco);
    if (rc != hipSuccess)
        return fadm_hip_rc(rc);

    rc = hipModuleGetFunction(&s->func_dwt_vert, s->module, "float_adm_dwt_vert");
    if (rc != hipSuccess)
        goto fail;
    rc = hipModuleGetFunction(&s->func_dwt_hori, s->module, "float_adm_dwt_hori");
    if (rc != hipSuccess)
        goto fail;
    rc = hipModuleGetFunction(&s->func_decouple_csf, s->module, "float_adm_decouple_csf");
    if (rc != hipSuccess)
        goto fail;
    rc = hipModuleGetFunction(&s->func_csf_cm, s->module, "float_adm_csf_cm");
    if (rc != hipSuccess)
        goto fail;
    return 0;

fail:
    (void)hipModuleUnload(s->module);
    s->module = NULL;
    return fadm_hip_rc(rc);
}

static int fadm_hip_launch(FloatAdmStateHip *s, uintptr_t pic_stream_handle)
{
    hipStream_t pstr = (hipStream_t)pic_stream_handle;
    hipStream_t str = (hipStream_t)s->lc.str;

    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t raw_stride = (ptrdiff_t)(s->width * bpp);

    float scaler = 1.0f;
    float pixel_offset = -128.0f;
    if (s->bpc == 10u)
        scaler = 4.0f;
    else if (s->bpc == 12u)
        scaler = 16.0f;
    else if (s->bpc == 16u)
        scaler = 256.0f;

    const uint8_t *ref_raw_d = (const uint8_t *)s->src_ref;
    const uint8_t *dis_raw_d = (const uint8_t *)s->src_dis;
    float *dwt_ref_d = (float *)s->dwt_tmp_ref;
    float *dwt_dis_d = (float *)s->dwt_tmp_dis;
    float *csf_a_d = (float *)s->csf_a;
    float *csf_f_d = (float *)s->csf_f;

    hipError_t rc;

    /* Zero all accumulator buffers — stage 3 writes only 2 out of 6
     * slots per WG; the others must be zero for the host reduction. */
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        rc = hipMemsetAsync(s->accum[scale], 0,
                            (size_t)s->wg_count[scale] * FADM_ACCUM_SLOTS * sizeof(float), pstr);
        if (rc != hipSuccess)
            return fadm_hip_rc(rc);
    }

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const int cur_w = (int)s->scale_w[scale];
        const int cur_h = (int)s->scale_h[scale];
        const int half_w = (int)s->scale_half_w[scale];
        const int half_h = (int)s->scale_half_h[scale];

        const int parent_w = (scale > 0) ? (int)s->scale_w[scale] : 0;
        const int parent_h = (scale > 0) ? (int)s->scale_h[scale] : 0;
        const int parent_half_h = (scale > 0) ? (int)s->scale_half_h[scale - 1] : 0;
        const int parent_buf_stride = (int)s->buf_stride;

        float *ref_band_d = (float *)s->ref_band[scale];
        float *dis_band_d = (float *)s->dis_band[scale];
        const float *parent_ref_band_d = (scale > 0) ? (const float *)s->ref_band[scale - 1] : NULL;
        const float *parent_dis_band_d = (scale > 0) ? (const float *)s->dis_band[scale - 1] : NULL;

        int top = (int)((double)half_h * FADM_BORDER_FACTOR - 0.5);
        int left = (int)((double)half_w * FADM_BORDER_FACTOR - 0.5);
        if (top < 0)
            top = 0;
        if (left < 0)
            left = 0;
        const int bottom = half_h - top;
        const int right = half_w - left;
        const int active_h = bottom - top;

        const float rfactor_h = s->rfactor[scale * 3 + 0];
        const float rfactor_v = s->rfactor[scale * 3 + 1];
        const float rfactor_d = s->rfactor[scale * 3 + 2];
        const float gain_limit = (float)s->adm_enhn_gain_limit;

        /* Stage 0 — DWT vertical (z=2 fuses ref+dis). */
        {
            const unsigned gx = ((unsigned)cur_w + FADM_BX - 1u) / FADM_BX;
            const unsigned gy = ((unsigned)half_h + FADM_BY - 1u) / FADM_BY;
            int scale_arg = scale;
            int half_h_arg = half_h;
            int par_half_h_arg = parent_half_h;
            int par_buf_stride_arg = parent_buf_stride;
            int par_w_arg = parent_w;
            int par_h_arg = parent_h;
            int cur_w_arg = cur_w;
            int cur_h_arg = cur_h;
            unsigned bpc_arg = s->bpc;
            float scaler_arg = scaler;
            float poff_arg = pixel_offset;
            void *args[] = {
                &scale_arg,         &ref_raw_d,         &dis_raw_d,          (void *)&raw_stride,
                &parent_ref_band_d, &parent_dis_band_d, &par_buf_stride_arg, &par_half_h_arg,
                &par_w_arg,         &par_h_arg,         &dwt_ref_d,          &dwt_dis_d,
                &cur_w_arg,         &cur_h_arg,         &half_h_arg,         &bpc_arg,
                &scaler_arg,        &poff_arg};
            rc = hipModuleLaunchKernel(s->func_dwt_vert, gx, gy, 2u, FADM_BX, FADM_BY, 1u, 0u, pstr,
                                       args, NULL);
            if (rc != hipSuccess)
                return fadm_hip_rc(rc);
        }

        /* Stage 1 — DWT horizontal. */
        {
            const unsigned gx = ((unsigned)half_w + FADM_BX - 1u) / FADM_BX;
            const unsigned gy = ((unsigned)half_h + FADM_BY - 1u) / FADM_BY;
            int scale_arg = scale;
            int cur_w_arg = cur_w;
            int half_w_arg = half_w;
            int half_h_arg = half_h;
            int buf_stride_arg = (int)s->buf_stride;
            void *args[] = {&scale_arg, &dwt_ref_d,  &dwt_dis_d,  &ref_band_d,    &dis_band_d,
                            &cur_w_arg, &half_w_arg, &half_h_arg, &buf_stride_arg};
            rc = hipModuleLaunchKernel(s->func_dwt_hori, gx, gy, 2u, FADM_BX, FADM_BY, 1u, 0u, pstr,
                                       args, NULL);
            if (rc != hipSuccess)
                return fadm_hip_rc(rc);
        }

        /* Stage 2 — Decouple + CSF. */
        {
            const unsigned gx = ((unsigned)half_w + FADM_BX - 1u) / FADM_BX;
            const unsigned gy = ((unsigned)half_h + FADM_BY - 1u) / FADM_BY;
            int half_w_arg = half_w;
            int half_h_arg = half_h;
            int buf_stride_arg = (int)s->buf_stride;
            float rfh = rfactor_h;
            float rfv = rfactor_v;
            float rfd = rfactor_d;
            float gl = gain_limit;
            void *args[] = {&ref_band_d, &dis_band_d,     &csf_a_d, &csf_f_d, &half_w_arg,
                            &half_h_arg, &buf_stride_arg, &rfh,     &rfv,     &rfd,
                            &gl};
            rc = hipModuleLaunchKernel(s->func_decouple_csf, gx, gy, 1u, FADM_BX, FADM_BY, 1u, 0u,
                                       pstr, args, NULL);
            if (rc != hipSuccess)
                return fadm_hip_rc(rc);
        }

        /* Stage 3 — CSF denominator + CM fused (1D over 3 bands × active rows). */
        {
            const unsigned num_rows = (unsigned)(active_h > 0 ? active_h : 1);
            const unsigned gx = 3u * num_rows;
            int half_w_arg = half_w;
            int half_h_arg = half_h;
            int buf_stride_arg = (int)s->buf_stride;
            int active_left_arg = left;
            int active_top_arg = top;
            int active_right_arg = right;
            int active_bottom_arg = bottom;
            float rfh = rfactor_h;
            float rfv = rfactor_v;
            float rfd = rfactor_d;
            float gl = gain_limit;
            float *accum_d = (float *)s->accum[scale];
            void *args[] = {&ref_band_d,
                            &dis_band_d,
                            &csf_a_d,
                            &csf_f_d,
                            &accum_d,
                            &half_w_arg,
                            &half_h_arg,
                            &buf_stride_arg,
                            &active_left_arg,
                            &active_top_arg,
                            &active_right_arg,
                            &active_bottom_arg,
                            &rfh,
                            &rfv,
                            &rfd,
                            &gl};
            rc = hipModuleLaunchKernel(s->func_csf_cm, gx, 1u, 1u, FADM_BX, FADM_BY, 1u, 0u, pstr,
                                       args, NULL);
            if (rc != hipSuccess)
                return fadm_hip_rc(rc);
        }
    }

    /* Event fence → secondary stream → D2H copy partials. */
    rc = hipEventRecord((hipEvent_t)s->lc.submit, pstr);
    if (rc != hipSuccess)
        return fadm_hip_rc(rc);
    rc = hipStreamWaitEvent(str, (hipEvent_t)s->lc.submit, 0);
    if (rc != hipSuccess)
        return fadm_hip_rc(rc);

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const size_t n_bytes = (size_t)s->wg_count[scale] * FADM_ACCUM_SLOTS * sizeof(float);
        rc = hipMemcpyAsync(s->accum_host[scale], s->accum[scale], n_bytes, hipMemcpyDeviceToHost,
                            str);
        if (rc != hipSuccess)
            return fadm_hip_rc(rc);
    }

    return vmaf_hip_kernel_submit_post_record(&s->lc, s->ctx);
}
#endif /* HAVE_HIPCC */

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    FloatAdmStateHip *s = fex->priv;

    if (s->adm_csf_mode != 0)
        return -EINVAL;

    s->width = w;
    s->height = h;
    s->bpc = bpc;
    compute_per_scale_dims(s);

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const float f1 =
            fadm_dwt_quant_step(scale, 1, s->adm_norm_view_dist, s->adm_ref_display_height);
        const float f2 =
            fadm_dwt_quant_step(scale, 2, s->adm_norm_view_dist, s->adm_ref_display_height);
        s->rfactor[scale * 3 + 0] = (float)s->adm_csf_scale / f1;
        s->rfactor[scale * 3 + 1] = (float)s->adm_csf_scale / f1;
        s->rfactor[scale * 3 + 2] = (float)s->adm_csf_diag_scale / f2;
    }

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const int hh = (int)s->scale_half_h[scale];
        int top = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
        if (top < 0)
            top = 0;
        const int bottom = hh - top;
        const unsigned num_rows = (bottom > top) ? (unsigned)(bottom - top) : 1u;
        s->wg_count[scale] = 3u * num_rows;
    }

    int err = vmaf_hip_context_new(&s->ctx, 0);
    if (err != 0)
        return err;

    err = vmaf_hip_kernel_lifecycle_init(&s->lc, s->ctx);
    if (err != 0)
        goto fail_after_ctx;

#ifdef HAVE_HIPCC
    err = fadm_hip_module_load(s);
    if (err != 0)
        goto fail_after_lc;

    hipError_t rc;
    const size_t bpp = (bpc <= 8u) ? 1u : 2u;
    const size_t raw_bytes = (size_t)w * h * bpp;
    rc = hipMalloc(&s->src_ref, raw_bytes);
    if (rc != hipSuccess) {
        err = -ENOMEM;
        goto fail_after_module;
    }
    rc = hipMalloc(&s->src_dis, raw_bytes);
    if (rc != hipSuccess) {
        err = -ENOMEM;
        goto fail_after_src_ref;
    }

    const size_t dwt_bytes = (size_t)s->width * 2u * s->scale_half_h[0] * sizeof(float);
    rc = hipMalloc(&s->dwt_tmp_ref, dwt_bytes);
    if (rc != hipSuccess) {
        err = -ENOMEM;
        goto fail_after_src_dis;
    }
    rc = hipMalloc(&s->dwt_tmp_dis, dwt_bytes);
    if (rc != hipSuccess) {
        err = -ENOMEM;
        goto fail_after_dwt_ref;
    }

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const size_t band_bytes =
            (size_t)4u * s->buf_stride * s->scale_half_h[scale] * sizeof(float);
        rc = hipMalloc(&s->ref_band[scale], band_bytes);
        if (rc != hipSuccess) {
            err = -ENOMEM;
            /* Clean up already-allocated band buffers before bailing. */
            for (int j = 0; j < scale; j++) {
                (void)hipFree(s->ref_band[j]);
                s->ref_band[j] = NULL;
                (void)hipFree(s->dis_band[j]);
                s->dis_band[j] = NULL;
            }
            goto fail_after_dwt_dis;
        }
        rc = hipMalloc(&s->dis_band[scale], band_bytes);
        if (rc != hipSuccess) {
            err = -ENOMEM;
            (void)hipFree(s->ref_band[scale]);
            s->ref_band[scale] = NULL;
            for (int j = 0; j < scale; j++) {
                (void)hipFree(s->ref_band[j]);
                s->ref_band[j] = NULL;
                (void)hipFree(s->dis_band[j]);
                s->dis_band[j] = NULL;
            }
            goto fail_after_dwt_dis;
        }
    }

    const size_t csf_bytes =
        (size_t)FADM_NUM_BANDS * s->buf_stride * s->scale_half_h[0] * sizeof(float);
    rc = hipMalloc(&s->csf_a, csf_bytes);
    if (rc != hipSuccess) {
        err = -ENOMEM;
        goto fail_after_bands;
    }
    rc = hipMalloc(&s->csf_f, csf_bytes);
    if (rc != hipSuccess) {
        err = -ENOMEM;
        goto fail_after_csf_a;
    }

    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const size_t accum_bytes = (size_t)s->wg_count[scale] * FADM_ACCUM_SLOTS * sizeof(float);
        rc = hipMalloc(&s->accum[scale], accum_bytes);
        if (rc != hipSuccess) {
            err = -ENOMEM;
            for (int j = 0; j < scale; j++) {
                (void)hipFree(s->accum[j]);
                s->accum[j] = NULL;
                (void)hipHostFree(s->accum_host[j]);
                s->accum_host[j] = NULL;
            }
            goto fail_after_csf_f;
        }
        rc = hipHostMalloc((void **)&s->accum_host[scale], accum_bytes, hipHostMallocDefault);
        if (rc != hipSuccess) {
            err = -ENOMEM;
            (void)hipFree(s->accum[scale]);
            s->accum[scale] = NULL;
            for (int j = 0; j < scale; j++) {
                (void)hipFree(s->accum[j]);
                s->accum[j] = NULL;
                (void)hipHostFree(s->accum_host[j]);
                s->accum_host[j] = NULL;
            }
            goto fail_after_csf_f;
        }
    }
#endif /* HAVE_HIPCC */

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL) {
        err = -ENOMEM;
#ifdef HAVE_HIPCC
        goto fail_after_accum;
#else
        goto fail_after_lc;
#endif
    }
    return 0;

#ifdef HAVE_HIPCC
fail_after_accum:
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->accum_host[scale] != NULL) {
            (void)hipHostFree(s->accum_host[scale]);
            s->accum_host[scale] = NULL;
        }
        if (s->accum[scale] != NULL) {
            (void)hipFree(s->accum[scale]);
            s->accum[scale] = NULL;
        }
    }
fail_after_csf_f:
    if (s->csf_f != NULL) {
        (void)hipFree(s->csf_f);
        s->csf_f = NULL;
    }
fail_after_csf_a:
    if (s->csf_a != NULL) {
        (void)hipFree(s->csf_a);
        s->csf_a = NULL;
    }
fail_after_bands:
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->dis_band[scale] != NULL) {
            (void)hipFree(s->dis_band[scale]);
            s->dis_band[scale] = NULL;
        }
        if (s->ref_band[scale] != NULL) {
            (void)hipFree(s->ref_band[scale]);
            s->ref_band[scale] = NULL;
        }
    }
fail_after_dwt_dis:
    if (s->dwt_tmp_dis != NULL) {
        (void)hipFree(s->dwt_tmp_dis);
        s->dwt_tmp_dis = NULL;
    }
fail_after_dwt_ref:
    if (s->dwt_tmp_ref != NULL) {
        (void)hipFree(s->dwt_tmp_ref);
        s->dwt_tmp_ref = NULL;
    }
fail_after_src_dis:
    if (s->src_dis != NULL) {
        (void)hipFree(s->src_dis);
        s->src_dis = NULL;
    }
fail_after_src_ref:
    if (s->src_ref != NULL) {
        (void)hipFree(s->src_ref);
        s->src_ref = NULL;
    }
fail_after_module:
    if (s->module != NULL) {
        (void)hipModuleUnload(s->module);
        s->module = NULL;
    }
#endif /* HAVE_HIPCC */
fail_after_lc:
    (void)vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);
fail_after_ctx:
    vmaf_hip_context_destroy(s->ctx);
    s->ctx = NULL;
    return err;
}

static int submit_fex_hip(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                          VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    (void)index;
    FloatAdmStateHip *s = fex->priv;

#ifdef HAVE_HIPCC
    const size_t bpp = (s->bpc <= 8u) ? 1u : 2u;
    const ptrdiff_t raw_stride = (ptrdiff_t)(s->width * bpp);
    const uintptr_t pic_stream_handle = 0;
    hipStream_t pstr = (hipStream_t)pic_stream_handle;

    hipError_t rc = hipMemcpy2DAsync(s->src_ref, (size_t)raw_stride, ref_pic->data[0],
                                     (size_t)ref_pic->stride[0], (size_t)raw_stride,
                                     (size_t)s->height, hipMemcpyHostToDevice, pstr);
    if (rc != hipSuccess)
        return -EIO;

    rc = hipMemcpy2DAsync(s->src_dis, (size_t)raw_stride, dist_pic->data[0],
                          (size_t)dist_pic->stride[0], (size_t)raw_stride, (size_t)s->height,
                          hipMemcpyHostToDevice, pstr);
    if (rc != hipSuccess)
        return -EIO;

    return fadm_hip_launch(s, pic_stream_handle);
#else
    (void)dist_pic;
    (void)ref_pic;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index, VmafFeatureCollector *fc)
{
    FloatAdmStateHip *s = fex->priv;

    int sync_err = vmaf_hip_kernel_collect_wait(&s->lc, s->ctx);
    if (sync_err != 0)
        return sync_err;

#ifdef HAVE_HIPCC
    /* Per-scale double accumulation across WGs. */
    double cm_totals[FADM_NUM_SCALES][FADM_NUM_BANDS] = {{0.0}};
    double csf_totals[FADM_NUM_SCALES][FADM_NUM_BANDS] = {{0.0}};
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const float *slots = s->accum_host[scale];
        const unsigned wg_count = s->wg_count[scale];
        for (unsigned wg = 0u; wg < wg_count; wg++) {
            const float *p = slots + (size_t)wg * FADM_ACCUM_SLOTS;
            for (int b = 0; b < FADM_NUM_BANDS; b++) {
                csf_totals[scale][b] += (double)p[b];
                cm_totals[scale][b] += (double)p[3 + b];
            }
        }
    }

    double score_num = 0.0;
    double score_den = 0.0;
    double scores[8];
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        const int hw = (int)s->scale_half_w[scale];
        const int hh = (int)s->scale_half_h[scale];
        int left = (int)((double)hw * FADM_BORDER_FACTOR - 0.5);
        int top = (int)((double)hh * FADM_BORDER_FACTOR - 0.5);
        if (left < 0)
            left = 0;
        if (top < 0)
            top = 0;
        const int right = hw - left;
        const int bottom = hh - top;
        const float area_cbrt = powf(
            (float)((bottom - top) * (right - left)) * (float)s->adm_noise_weight, 1.0f / 3.0f);
        float num_scale = 0.0f;
        float den_scale = 0.0f;
        for (int b = 0; b < FADM_NUM_BANDS; b++) {
            num_scale += powf((float)cm_totals[scale][b], 1.0f / 3.0f) + area_cbrt;
            den_scale += powf((float)csf_totals[scale][b], 1.0f / 3.0f) + area_cbrt;
        }
        scores[2 * scale + 0] = num_scale;
        scores[2 * scale + 1] = den_scale;
        score_num += num_scale;
        score_den += den_scale;
    }

    const int w = (int)s->scale_w[0];
    const int h = (int)s->scale_h[0];
    const double numden_limit = 1e-2 * (double)(w * h) / (1920.0 * 1080.0);
    if (score_num < numden_limit)
        score_num = 0.0;
    if (score_den < numden_limit)
        score_den = 0.0;
    const double score = (score_den == 0.0) ? 1.0 : score_num / score_den;

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                                                   "VMAF_feature_adm2_score", score, index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale0_score", scores[0] / scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale1_score", scores[2] / scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale2_score", scores[4] / scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(
        fc, s->feature_name_dict, "VMAF_feature_adm_scale3_score", scores[6] / scores[7], index);

    if (s->debug && !err) {
        err |=
            vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm", score, index);
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm_num",
                                                       score_num, index);
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, "adm_den",
                                                       score_den, index);
        const char *names[8] = {"adm_num_scale0", "adm_den_scale0", "adm_num_scale1",
                                "adm_den_scale1", "adm_num_scale2", "adm_den_scale2",
                                "adm_num_scale3", "adm_den_scale3"};
        for (int i = 0; i < 8 && !err; i++) {
            err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict, names[i],
                                                           scores[i], index);
        }
    }
    return err;
#else
    (void)fc;
    (void)index;
    return -ENOSYS;
#endif /* HAVE_HIPCC */
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    FloatAdmStateHip *s = fex->priv;

    int rc = vmaf_hip_kernel_lifecycle_close(&s->lc, s->ctx);

#ifdef HAVE_HIPCC
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->accum_host[scale] != NULL) {
            (void)hipHostFree(s->accum_host[scale]);
            s->accum_host[scale] = NULL;
        }
        if (s->accum[scale] != NULL) {
            (void)hipFree(s->accum[scale]);
            s->accum[scale] = NULL;
        }
    }
    if (s->csf_f != NULL) {
        (void)hipFree(s->csf_f);
        s->csf_f = NULL;
    }
    if (s->csf_a != NULL) {
        (void)hipFree(s->csf_a);
        s->csf_a = NULL;
    }
    for (int scale = 0; scale < FADM_NUM_SCALES; scale++) {
        if (s->dis_band[scale] != NULL) {
            (void)hipFree(s->dis_band[scale]);
            s->dis_band[scale] = NULL;
        }
        if (s->ref_band[scale] != NULL) {
            (void)hipFree(s->ref_band[scale]);
            s->ref_band[scale] = NULL;
        }
    }
    if (s->dwt_tmp_dis != NULL) {
        (void)hipFree(s->dwt_tmp_dis);
        s->dwt_tmp_dis = NULL;
    }
    if (s->dwt_tmp_ref != NULL) {
        (void)hipFree(s->dwt_tmp_ref);
        s->dwt_tmp_ref = NULL;
    }
    if (s->src_dis != NULL) {
        (void)hipFree(s->src_dis);
        s->src_dis = NULL;
    }
    if (s->src_ref != NULL) {
        (void)hipFree(s->src_ref);
        s->src_ref = NULL;
    }
    if (s->module != NULL) {
        hipError_t hip_err = hipModuleUnload(s->module);
        if (hip_err != hipSuccess && rc == 0)
            rc = -EIO;
        s->module = NULL;
    }
#endif /* HAVE_HIPCC */

    if (s->feature_name_dict != NULL) {
        int err = vmaf_dictionary_free(&s->feature_name_dict);
        if (err != 0 && rc == 0)
            rc = err;
    }
    if (s->ctx != NULL) {
        vmaf_hip_context_destroy(s->ctx);
        s->ctx = NULL;
    }
    return rc;
}

static const char *provided_features[] = {"VMAF_feature_adm2_score",
                                          "VMAF_feature_adm_scale0_score",
                                          "VMAF_feature_adm_scale1_score",
                                          "VMAF_feature_adm_scale2_score",
                                          "VMAF_feature_adm_scale3_score",
                                          "adm",
                                          "adm_num",
                                          "adm_den",
                                          "adm_num_scale0",
                                          "adm_den_scale0",
                                          "adm_num_scale1",
                                          "adm_den_scale1",
                                          "adm_num_scale2",
                                          "adm_den_scale2",
                                          "adm_num_scale3",
                                          "adm_den_scale3",
                                          NULL};

/* Load-bearing: registered via `extern VmafFeatureExtractor vmaf_fex_float_adm_hip;`
 * in `libvmaf/src/feature/feature_extractor.c`'s `feature_extractor_list[]`.
 * Ninth HIP kernel-template consumer (ADR-0468). Same pattern as
 * every CUDA / SYCL / Vulkan / HIP feature extractor. */
// NOLINTNEXTLINE(misc-use-internal-linkage) — ADR-0468: registration symbol must have external linkage
VmafFeatureExtractor vmaf_fex_float_adm_hip = {
    .name = "float_adm_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .close = close_fex_hip,
    .options = options,
    .priv_size = sizeof(FloatAdmStateHip),
    .provided_features = provided_features,
    /* Same scaffold-posture flags as every other HIP consumer:
     * no VMAF_FEATURE_EXTRACTOR_HIP flag yet — the picture
     * buffer-type plumbing lands with the runtime PR. */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 16,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
