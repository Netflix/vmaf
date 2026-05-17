/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Integer ADM feature extractor — HIP backend.
 *  Ported from libvmaf/src/feature/cuda/integer_adm_cuda.c call-graph-for-call-graph.
 *
 *  Scaffold posture (no HAVE_HIPCC): every lifecycle helper returns -ENOSYS;
 *  the extractor registers under the name "adm_hip" so name-lookup works and
 *  the caller receives the cleaner "runtime not ready" surface instead of
 *  "no such extractor".
 *
 *  With HAVE_HIPCC: four HSACO blobs (adm_dwt2, adm_csf, adm_csf_den,
 *  adm_cm) are loaded via hipModuleLoadData at init() time and the full
 *  4-scale DWT + CSF + CM pipeline runs on device.
 *
 *  Algorithm: identical to the CUDA twin — integer DWT2 (Daubechies-7/9),
 *  contrast sensitivity function masking, and contrast masking numerator /
 *  denominator accumulation across 4 dyadic scales. See also:
 *  Li, Z. et al. (2016). "Toward A Practical Perceptual Video Quality
 *  Metric" — the ADM2 model shipped as "VMAF_integer_feature_adm2_score".
 */

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "integer_adm.h"
#include "libvmaf/picture.h"

#include "hip/integer_adm_hip.h"

#ifdef HAVE_HIPCC
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#endif /* HAVE_HIPCC */

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

#define RES_BUFFER_SIZE (4 * 3 * 2)

/* ------------------------------------------------------------------ */
/* Internal state                                                       */
/* ------------------------------------------------------------------ */

typedef struct AdmStateHip {
    size_t integer_stride;
    AdmBufferHip buf;
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    double adm_csf_scale;
    double adm_csf_diag_scale;
    double adm_noise_weight;
    unsigned submit_w, submit_h; /* stored by submit for collect */

#ifdef HAVE_HIPCC
    hipStream_t str;
    hipEvent_t ref_event, dis_event, finished;
    hipModule_t adm_dwt_module;
    hipModule_t adm_csf_module;
    hipModule_t adm_csf_den_module;
    hipModule_t adm_cm_module;

    /* DWT kernel handles */
    hipFunction_t func_dwt_s123_combined_vert_kernel_0_0_int32_t;
    hipFunction_t func_dwt_s123_combined_vert_kernel_32768_16_int32_t;
    hipFunction_t func_dwt_s123_combined_hori_kernel_16384_15;
    hipFunction_t func_dwt_s123_combined_hori_kernel_32768_16;
    hipFunction_t func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t;
    hipFunction_t func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t;

    /* CSF kernel handles */
    hipFunction_t func_adm_csf_kernel_1_4;
    hipFunction_t func_i4_adm_csf_kernel_1_4;

    /* CSF-den kernel handles */
    hipFunction_t func_adm_csf_den_scale_line_kernel;
    hipFunction_t func_adm_csf_den_s123_line_kernel;

    /* CM kernel handles */
    hipFunction_t func_adm_cm_reduce_line_kernel_4;
    hipFunction_t func_adm_cm_line_kernel_8;
    hipFunction_t func_i4_adm_cm_line_kernel;
#endif /* HAVE_HIPCC */

    VmafDictionary *feature_name_dict;
} AdmStateHip;

/* ------------------------------------------------------------------ */
/* dwt_quant_step — identical to the CUDA twin                         */
/* ------------------------------------------------------------------ */

static inline float dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta,
                                   double adm_norm_view_dist, int adm_ref_display_height)
{
    float r = (float)(adm_norm_view_dist * adm_ref_display_height) * (float)M_PI / 180.0f;
    float temp = log10f(powf(2.0f, (float)(lambda + 1)) * params->f0 * params->g[theta] / r);
    float Q = 2.0f * params->a * powf(10.0f, params->k * temp * temp) /
              dwt_7_9_basis_function_amplitudes[lambda][theta];
    return Q;
}

/* ------------------------------------------------------------------ */
/* Score computation helpers (host-side, same as CUDA twin)            */
/* ------------------------------------------------------------------ */

static void conclude_adm_cm(int64_t *accum, int h, int w, int scale, float noise_weight,
                            float *result)
{
    int left = (int)(w * ADM_BORDER_FACTOR - 0.5);
    int top = (int)(h * ADM_BORDER_FACTOR - 0.5);
    int right = w - left;
    int bottom = h - top;
    const uint32_t shift_inner_accum = (uint32_t)ceil(log2((double)h));

    const uint32_t shift_xcub[3] = {(uint32_t)ceil(log2((double)w) - 4.0),
                                    (uint32_t)ceil(log2((double)w) - 4.0),
                                    (uint32_t)ceil(log2((double)w) - 3.0)};
    int constant_offset[3] = {52, 52, 57};

    uint32_t shift_cub = (uint32_t)ceil(log2((double)w));
    float final_shift[3] = {powf(2.0f, (float)(45 - (int)shift_cub - (int)shift_inner_accum)),
                            powf(2.0f, (float)(39 - (int)shift_cub - (int)shift_inner_accum)),
                            powf(2.0f, (float)(36 - (int)shift_cub - (int)shift_inner_accum))};
    float powf_add = powf((float)((bottom - top) * (right - left)) * noise_weight, 1.0f / 3.0f);

    float f_accum;
    *result = 0;
    for (int i = 0; i < 3; ++i) {
        if (scale == 0) {
            f_accum = (float)(accum[i] / pow(2.0, (double)(constant_offset[i] - (int)shift_xcub[i] -
                                                           (int)shift_inner_accum)));
        } else {
            f_accum = (float)((double)accum[i] / (double)final_shift[scale - 1]);
        }
        *result += powf(f_accum, 1.0f / 3.0f) + powf_add;
    }
}

static void conclude_adm_csf_den(uint64_t *accum, int h, int w, int scale, float *result,
                                 float adm_norm_view_dist, float adm_ref_display_height,
                                 float adm_csf_scale, float adm_csf_diag_scale, float noise_weight)
{
    const int left = (int)(w * ADM_BORDER_FACTOR - 0.5);
    const int top = (int)(h * ADM_BORDER_FACTOR - 0.5);
    const int right = w - left;
    const int bottom = h - top;
    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist,
                                         (int)adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist,
                                         (int)adm_ref_display_height);
    const float rfactor[3] = {adm_csf_scale / factor1, adm_csf_scale / factor1,
                              adm_csf_diag_scale / factor2};
    const uint32_t accum_convert_float[4] = {18, 32, 27, 23};

    int32_t shift_accum;
    double shift_csf;
    if (scale == 0) {
        shift_accum = (int32_t)ceil(log2((double)((bottom - top) * (right - left))) - 20.0);
        shift_accum = shift_accum > 0 ? shift_accum : 0;
        shift_csf = pow(2.0, (double)(accum_convert_float[scale] - (uint32_t)shift_accum));
    } else {
        shift_accum = (int32_t)ceil(log2((double)(bottom - top)));
        const uint32_t shift_cub = (uint32_t)ceil(log2((double)(right - left)));
        shift_csf =
            pow(2.0, (double)(accum_convert_float[scale] - (uint32_t)shift_accum - shift_cub));
    }
    const float powf_add =
        powf((float)((bottom - top) * (right - left)) * noise_weight, 1.0f / 3.0f);

    *result = 0;
    for (int i = 0; i < 3; ++i) {
        const double csf = (double)(accum[i] / shift_csf) * pow((double)rfactor[i], 3.0);
        *result += powf((float)csf, 1.0f / 3.0f) + powf_add;
    }
}

/* ------------------------------------------------------------------ */
/* write_scores — host-side, mirror of CUDA twin                       */
/* ------------------------------------------------------------------ */

typedef struct write_score_parameters_adm_hip {
    VmafFeatureCollector *feature_collector;
    AdmStateHip *s;
    unsigned index, h, w;
} write_score_parameters_adm_hip;

static void write_scores(const write_score_parameters_adm_hip *params)
{
    VmafFeatureCollector *feature_collector = params->feature_collector;
    const AdmStateHip *s = params->s;
    unsigned index = params->index;

    double scores[8];
    double score, score_num, score_den;
    double num = 0;
    double den = 0;

    unsigned w = params->w;
    unsigned h = params->h;

    int64_t *adm_cm = (int64_t *)s->buf.results_host;
    uint64_t *adm_csf = &((uint64_t *)s->buf.results_host)[RES_BUFFER_SIZE / 2];
    float num_scale;
    float den_scale;

    for (unsigned scale = 0; scale < 4; ++scale) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;

        conclude_adm_cm(&adm_cm[scale * 3], (int)h, (int)w, (int)scale, (float)s->adm_noise_weight,
                        &num_scale);
        conclude_adm_csf_den(&adm_csf[scale * 3], (int)h, (int)w, (int)scale, &den_scale,
                             (float)s->adm_norm_view_dist, (float)s->adm_ref_display_height,
                             (float)s->adm_csf_scale, (float)s->adm_csf_diag_scale,
                             (float)s->adm_noise_weight);

        num += num_scale;
        den += den_scale;

        scores[2 * scale + 0] = num_scale;
        scores[2 * scale + 1] = den_scale;
    }

    const double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);
    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;

    if (den == 0.0) {
        score = 1.0;
    } else {
        score = num / den;
    }
    score_num = num;
    score_den = den;

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "VMAF_integer_feature_adm2_score", score, index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                "integer_adm_scale0", scores[0] / scores[1], index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                "integer_adm_scale1", scores[2] / scores[3], index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                "integer_adm_scale2", scores[4] / scores[5], index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                "integer_adm_scale3", scores[6] / scores[7], index);

    if (!s->debug) {
        (void)err;
        return;
    }

    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm", score, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_num", score_num, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_den", score_den, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_num_scale0", scores[0], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_den_scale0", scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_num_scale1", scores[2], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_den_scale1", scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_num_scale2", scores[4], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_den_scale2", scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_num_scale3", scores[6], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector, s->feature_name_dict,
                                                   "integer_adm_den_scale3", scores[7], index);
    (void)err; /* accumulated collector status intentionally discarded; void writer API */
}

/* ------------------------------------------------------------------ */
/* VmafOption table                                                     */
/* ------------------------------------------------------------------ */

static const VmafOption options_hip[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(AdmStateHip, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmStateHip, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmStateHip, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
    },
    {
        .name = "adm_ref_display_height",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmStateHip, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
    },
    {
        .name = "adm_csf_scale",
        .alias = "cs",
        .help = "CSF band-scale multiplier for h/v bands (default 1.0 = no scaling)",
        .offset = offsetof(AdmStateHip, adm_csf_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_SCALE,
        .min = 0.0,
        .max = 100.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_diag_scale",
        .alias = "cds",
        .help = "CSF band-scale multiplier for diagonal bands (default 1.0 = no scaling)",
        .offset = offsetof(AdmStateHip, adm_csf_diag_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
        .min = 0.0,
        .max = 100.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_noise_weight",
        .alias = "nw",
        .help = "noise floor weight for CM numerator (default 0.03125 = 1/32)",
        .offset = offsetof(AdmStateHip, adm_noise_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 100.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}};

/* ================================================================== */
/* HAVE_HIPCC path — real kernel dispatch                              */
/* ================================================================== */

#ifdef HAVE_HIPCC

/* Translate a HIP error to a negative errno. */
static int hip_rc(hipError_t rc)
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

/* Helper: load one HSACO module and get one function. */
static int load_module_fn(hipModule_t *mod, hipFunction_t *fn, const unsigned char *hsaco,
                          const char *fn_name)
{
    hipError_t rc = hipModuleLoadData(mod, (const void *)hsaco);
    if (rc != hipSuccess)
        return hip_rc(rc);
    rc = hipModuleGetFunction(fn, *mod, fn_name);
    if (rc != hipSuccess) {
        (void)hipModuleUnload(*mod);
        *mod = NULL;
        return hip_rc(rc);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Device-dispatch helpers (mirror integer_adm_cuda.c functions)      */
/* ------------------------------------------------------------------ */

#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))

static int dwt2_8_device_hip(AdmStateHip *s, const uint8_t *d_picture, hip_adm_dwt_band_t *d_dst,
                             hip_i4_adm_dwt_band_t i4_dwt_dst, int w, int h, int src_stride,
                             int dst_stride, AdmFixedParametersHip *p, hipStream_t c_stream)
{
    const int rows_per_thread = 4;
    const int vert_out_tile_rows = 8;
    const int vert_out_tile_cols = 128;
    const int horz_out_tile_cols = vert_out_tile_cols / 2 - 2;
    const int horz_out_tile_rows = vert_out_tile_rows;
    int16_t v_shift = 8;
    int32_t v_add_shift = 1 << (v_shift - 1);

    void *args[] = {&d_picture,  d_dst,       &i4_dwt_dst, &w,           &h,
                    &src_stride, &dst_stride, &v_shift,    &v_add_shift, p};
    hipError_t rc = hipModuleLaunchKernel(
        s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t,
        (uint32_t)DIV_ROUND_UP((w + 1) / 2, horz_out_tile_cols),
        (uint32_t)DIV_ROUND_UP((h + 1) / 2, horz_out_tile_rows), 1, (uint32_t)vert_out_tile_cols,
        (uint32_t)(vert_out_tile_rows / rows_per_thread), 1, 0, c_stream, args, NULL);
    return hip_rc(rc);
}

static int dwt2_16_device_hip(AdmStateHip *s, const uint16_t *d_picture, hip_adm_dwt_band_t *d_dst,
                              hip_i4_adm_dwt_band_t i4_dwt_dst, int w, int h, int src_stride,
                              int dst_stride, int inp_size_bits, AdmFixedParametersHip *p,
                              hipStream_t c_stream)
{
    const int rows_per_thread = 4;
    const int vert_out_tile_rows = 8;
    const int vert_out_tile_cols = 128;
    const int horz_out_tile_cols = vert_out_tile_cols / 2 - 2;
    const int horz_out_tile_rows = vert_out_tile_rows;
    int16_t v_shift = (int16_t)inp_size_bits;
    int32_t v_add_shift = 1 << (inp_size_bits - 1);

    void *args[] = {&d_picture,  d_dst,       &i4_dwt_dst, &w,           &h,
                    &src_stride, &dst_stride, &v_shift,    &v_add_shift, p};
    hipError_t rc = hipModuleLaunchKernel(
        s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t,
        (uint32_t)DIV_ROUND_UP((w + 1) / 2, horz_out_tile_cols),
        (uint32_t)DIV_ROUND_UP((h + 1) / 2, horz_out_tile_rows), 1, (uint32_t)vert_out_tile_cols,
        (uint32_t)(vert_out_tile_rows / rows_per_thread), 1, 0, c_stream, args, NULL);
    return hip_rc(rc);
}

static int adm_dwt2_s123_combined_device_hip(AdmStateHip *s, const int32_t *d_i4_scale,
                                             int32_t *tmp_buf, hip_i4_adm_dwt_band_t i4_dwt, int w,
                                             int h, int img_stride, int dst_stride, int scale,
                                             AdmFixedParametersHip *p, hipStream_t cu_stream)
{
    const int BLOCK_Y = (h + 1) / 2;

    void *args_vert[] = {&d_i4_scale, &tmp_buf, &w, &h, &img_stride, p};
    hipError_t rc;
    switch (scale) {
    case 1:
        rc = hipModuleLaunchKernel(s->func_dwt_s123_combined_vert_kernel_0_0_int32_t,
                                   (uint32_t)DIV_ROUND_UP(w, 128), (uint32_t)BLOCK_Y, 1, 128, 1, 1,
                                   0, cu_stream, args_vert, NULL);
        if (rc != hipSuccess)
            return hip_rc(rc);
        break;
    case 2: /* fall-through */
    case 3:
        rc = hipModuleLaunchKernel(s->func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
                                   (uint32_t)DIV_ROUND_UP(w, 128), (uint32_t)BLOCK_Y, 1, 128, 1, 1,
                                   0, cu_stream, args_vert, NULL);
        if (rc != hipSuccess)
            return hip_rc(rc);
        break;
    default:
        break; /* scale 0 handled in caller */
    }

    void *args_hori[] = {&i4_dwt, &tmp_buf, &w, &h, &dst_stride, p};
    switch (scale) {
    case 1:
        rc = hipModuleLaunchKernel(s->func_dwt_s123_combined_hori_kernel_16384_15,
                                   (uint32_t)DIV_ROUND_UP((w + 1) / 2, 128), (uint32_t)BLOCK_Y, 1,
                                   128, 1, 1, 0, cu_stream, args_hori, NULL);
        if (rc != hipSuccess)
            return hip_rc(rc);
        break;
    case 2:
        rc = hipModuleLaunchKernel(s->func_dwt_s123_combined_hori_kernel_32768_16,
                                   (uint32_t)DIV_ROUND_UP((w + 1) / 2, 128), (uint32_t)BLOCK_Y, 1,
                                   128, 1, 1, 0, cu_stream, args_hori, NULL);
        if (rc != hipSuccess)
            return hip_rc(rc);
        break;
    case 3:
        rc = hipModuleLaunchKernel(s->func_dwt_s123_combined_hori_kernel_16384_15,
                                   (uint32_t)DIV_ROUND_UP((w + 1) / 2, 128), (uint32_t)BLOCK_Y, 1,
                                   128, 1, 1, 0, cu_stream, args_hori, NULL);
        if (rc != hipSuccess)
            return hip_rc(rc);
        break;
    default:
        break;
    }
    return 0;
}

static int adm_csf_device_hip(AdmStateHip *s, AdmBufferHip *buf, int w, int h, int stride,
                              AdmFixedParametersHip *p, hipStream_t c_stream)
{
    for (int band = 0; band < 3; ++band)
        assert(((size_t)(buf->csf_f.bands[band]) & 15) == 0);
    assert(stride % 4 == 0);

    int left = (int)(w * (float)(ADM_BORDER_FACTOR)-0.5f - 1.0f);
    int top = (int)(h * (float)(ADM_BORDER_FACTOR)-0.5f - 1.0f);
    int right = w - left + 2;
    int bottom = h - top + 2;

    if (left < 0)
        left = 0;
    if (right > w)
        right = w;
    if (top < 0)
        top = 0;
    if (bottom > h)
        bottom = h;
    left = left & ~3;

    const int cols_per_thread = 4;
    const int rows_per_thread = 1;
    const int BLOCKX = 32, BLOCKY = 4;

    void *args[] = {buf, &top, &bottom, &left, &right, &stride, p};
    hipError_t rc = hipModuleLaunchKernel(
        s->func_adm_csf_kernel_1_4, (uint32_t)DIV_ROUND_UP(right - left, BLOCKX * cols_per_thread),
        (uint32_t)DIV_ROUND_UP(bottom - top, BLOCKY * rows_per_thread), 3, (uint32_t)BLOCKX,
        (uint32_t)BLOCKY, 1, 0, c_stream, args, NULL);
    return hip_rc(rc);
}

static int i4_adm_csf_device_hip(AdmStateHip *s, AdmBufferHip *buf, int scale, int w, int h,
                                 int stride, AdmFixedParametersHip *p, hipStream_t c_stream)
{
    for (int band = 0; band < 3; ++band)
        assert(((size_t)(buf->i4_csf_f.bands[band]) & 15) == 0);
    assert(stride % 4 == 0);

    int left = (int)(w * (float)(ADM_BORDER_FACTOR)-0.5f - 1.0f);
    int top = (int)(h * (float)(ADM_BORDER_FACTOR)-0.5f - 1.0f);
    int right = w - left + 2;
    int bottom = h - top + 2;

    if (left < 0)
        left = 0;
    if (right > w)
        right = w;
    if (top < 0)
        top = 0;
    if (bottom > h)
        bottom = h;
    left = left & ~3;

    const int cols_per_thread = 4;
    const int rows_per_thread = 1;
    const int BLOCKX = 32, BLOCKY = 4;

    void *args[] = {buf, &scale, &top, &bottom, &left, &right, &stride, p};
    hipError_t rc =
        hipModuleLaunchKernel(s->func_i4_adm_csf_kernel_1_4,
                              (uint32_t)DIV_ROUND_UP(right - left, BLOCKX * cols_per_thread),
                              (uint32_t)DIV_ROUND_UP(bottom - top, BLOCKY * rows_per_thread), 3,
                              (uint32_t)BLOCKX, (uint32_t)BLOCKY, 1, 0, c_stream, args, NULL);
    return hip_rc(rc);
}

static int adm_csf_den_s123_device_hip(AdmStateHip *s, AdmBufferHip *buf, int scale, int w, int h,
                                       int src_stride, hipStream_t c_stream)
{
    int left = (int)(w * (float)(ADM_BORDER_FACTOR)-0.5f);
    int top = (int)(h * (float)(ADM_BORDER_FACTOR)-0.5f);
    int right = w - left;
    int bottom = h - top;
    int buffer_stride = right - left;
    int buffer_h = bottom - top;

    const int val_per_thread = 8;
    const int warps_per_cta = 4;
    const int BLOCKX = 32 * warps_per_cta;

    uint32_t shift_sq[3] = {31, 30, 31};
    uint32_t add_shift_sq[3] = {1u << shift_sq[0], 1u << shift_sq[1], 1u << shift_sq[2]};

    void *args[] = {&buf->i4_ref_dwt2,
                    &h,
                    &top,
                    &bottom,
                    &left,
                    &right,
                    &src_stride,
                    &add_shift_sq[scale - 1],
                    &shift_sq[scale - 1],
                    &buf->adm_csf_den[scale]};
    hipError_t rc = hipModuleLaunchKernel(
        s->func_adm_csf_den_s123_line_kernel,
        (uint32_t)DIV_ROUND_UP(buffer_stride, BLOCKX * val_per_thread), (uint32_t)buffer_h, 3,
        (uint32_t)BLOCKX, 1, 1, 0, c_stream, args, NULL);
    return hip_rc(rc);
}

static int adm_csf_den_scale_device_hip(AdmStateHip *s, AdmBufferHip *buf, int w, int h,
                                        int src_stride, hipStream_t c_stream)
{
    int scale = 0;
    int left = (int)(w * (float)(ADM_BORDER_FACTOR)-0.5f);
    int top = (int)(h * (float)(ADM_BORDER_FACTOR)-0.5f);
    int right = w - left;
    int bottom = h - top;
    int buffer_stride = right - left;
    int buffer_h = bottom - top;

    const int val_per_thread = 8;
    const int warps_per_cta = 4;
    const int BLOCKX = 32 * warps_per_cta;

    void *args[] = {&buf->ref_dwt2, &h,     &top,        &bottom,
                    &left,          &right, &src_stride, &buf->adm_csf_den[scale]};
    hipError_t rc = hipModuleLaunchKernel(
        s->func_adm_csf_den_scale_line_kernel,
        (uint32_t)DIV_ROUND_UP(buffer_stride, BLOCKX * val_per_thread), (uint32_t)buffer_h, 3,
        (uint32_t)BLOCKX, 1, 1, 0, c_stream, args, NULL);
    return hip_rc(rc);
}

typedef struct WarpShiftHip {
    uint32_t shift_cub[3];
    uint32_t add_shift_cub[3];
    uint32_t shift_sq[3];
    uint32_t add_shift_sq[3];
} WarpShiftHip;

static int i4_adm_cm_device_hip(AdmStateHip *s, AdmBufferHip *buf, int w, int h, int src_stride,
                                int csf_a_stride, int scale, AdmFixedParametersHip *p,
                                hipStream_t c_stream)
{
    int left = (int)(w * (float)(ADM_BORDER_FACTOR)-0.5f);
    int top = (int)(h * (float)(ADM_BORDER_FACTOR)-0.5f);
    int right = w - left;
    int bottom = h - top;

    int start_col = (left > 1) ? left : ((left <= 0) ? 0 : 1);
    int end_col = (right < (w - 1)) ? right : ((right > (w - 1)) ? w : w - 1);
    int start_row = (top > 1) ? top : ((top <= 0) ? 0 : 1);
    int end_row = (bottom < (h - 1)) ? bottom : ((bottom > (h - 1)) ? h : h - 1);

    int buffer_stride = end_col - start_col;
    int buffer_h = end_row - start_row;

    /* inner CM kernel */
    {
        const int BLOCKX = 128;
        void *args[] = {
            buf,           &h,         &w,        &top,           &bottom,         &left,
            &right,        &start_row, &end_row,  &start_col,     &end_col,        &src_stride,
            &csf_a_stride, &scale,     &buffer_h, &buffer_stride, &buf->tmp_accum, p};
        hipError_t rc = hipModuleLaunchKernel(
            s->func_i4_adm_cm_line_kernel, (uint32_t)DIV_ROUND_UP(buffer_stride, BLOCKX),
            (uint32_t)buffer_h, 3, (uint32_t)BLOCKX, 1, 1, 0, c_stream, args, NULL);
        if (rc != hipSuccess)
            return hip_rc(rc);
    }

    /* reduce kernel */
    {
        const int val_per_thread = 4;
        const int warps_per_cta = 4;
        const int BLOCKX = 32 * warps_per_cta;
        void *args[] = {
            &h, &w, &scale, &buffer_h, &buffer_stride, &buf->tmp_accum, &buf->adm_cm[scale]};
        hipError_t rc = hipModuleLaunchKernel(
            s->func_adm_cm_reduce_line_kernel_4,
            (uint32_t)DIV_ROUND_UP(buffer_stride, BLOCKX * val_per_thread), (uint32_t)buffer_h, 3,
            (uint32_t)BLOCKX, 1, 1, 0, c_stream, args, NULL);
        if (rc != hipSuccess)
            return hip_rc(rc);
    }
    return 0;
}

static int adm_cm_device_hip(AdmStateHip *s, AdmBufferHip *buf, int w, int h, int src_stride,
                             int csf_a_stride, AdmFixedParametersHip *p, hipStream_t c_stream)
{
    int scale = 0;
    int left = (int)(w * (float)(ADM_BORDER_FACTOR)-0.5f);
    int top = (int)(h * (float)(ADM_BORDER_FACTOR)-0.5f);
    int right = w - left;
    int bottom = h - top;

    int start_col = (left > 0) ? left : 0;
    int end_col = (right < w) ? right : w;
    int start_row = (top > 0) ? top : 0;
    int end_row = (bottom < h) ? bottom : h;

    int buffer_stride = end_col - start_col;
    int buffer_h = end_row - start_row;

    const int fixed_shift[3] = {4, 4, 3};
    const int32_t shift_xsq[3] = {29, 29, 30};
    const int32_t add_shift_xsq[3] = {268435456, 268435456, 536870912};

    WarpShiftHip ws;
    for (int band = 0; band < 3; ++band) {
        ws.shift_cub[band] = (uint32_t)ceilf(log2f((float)w));
        ws.shift_cub[band] -= (uint32_t)fixed_shift[band];
        ws.shift_sq[band] = (uint32_t)shift_xsq[band];
        ws.add_shift_sq[band] = (uint32_t)add_shift_xsq[band];
        ws.add_shift_cub[band] = 1u << (ws.shift_cub[band] - 1u);
    }

    uint32_t shift_inner_accum = (uint32_t)ceilf(log2f((float)h));
    uint32_t add_shift_inner_accum = 1u << (shift_inner_accum - 1u);

    /* fused CM + reduce kernel */
    {
        const int rows_per_thread = 8;
        const int BLOCKX = 32, BLOCKY = 4;
        void *args[] = {buf,
                        &h,
                        &w,
                        &top,
                        &bottom,
                        &left,
                        &right,
                        &start_row,
                        &end_row,
                        &start_col,
                        &end_col,
                        &src_stride,
                        &csf_a_stride,
                        &buffer_h,
                        &buffer_stride,
                        &buf->tmp_accum,
                        p,
                        &scale,
                        &buf->adm_cm[scale],
                        &ws,
                        &shift_inner_accum,
                        &add_shift_inner_accum};
        hipError_t rc = hipModuleLaunchKernel(
            s->func_adm_cm_line_kernel_8, (uint32_t)DIV_ROUND_UP(buffer_stride, BLOCKX),
            (uint32_t)DIV_ROUND_UP(buffer_h, BLOCKY * rows_per_thread), 3, (uint32_t)BLOCKX,
            (uint32_t)BLOCKY, 1, 0, c_stream, args, NULL);
        if (rc != hipSuccess)
            return hip_rc(rc);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main per-frame computation                                           */
/* ------------------------------------------------------------------ */

static int integer_compute_adm_hip(AdmStateHip *s, VmafPicture *ref_pic, VmafPicture *dis_pic,
                                   AdmBufferHip *buf, double adm_enhn_gain_limit,
                                   double adm_norm_view_dist, int adm_ref_display_height)
{
    int w = (int)ref_pic->w[0];
    int h = (int)ref_pic->h[0];

    AdmFixedParametersHip p;
    memset(&p, 0, sizeof(p));
    p.dwt2_db2_coeffs_lo[0] = 15826;
    p.dwt2_db2_coeffs_lo[1] = 27411;
    p.dwt2_db2_coeffs_lo[2] = 7345;
    p.dwt2_db2_coeffs_lo[3] = -4240;
    p.dwt2_db2_coeffs_hi[0] = -4240;
    p.dwt2_db2_coeffs_hi[1] = -7345;
    p.dwt2_db2_coeffs_hi[2] = 27411;
    p.dwt2_db2_coeffs_hi[3] = -15826;
    p.dwt2_db2_coeffs_lo_sum = 46342;
    p.dwt2_db2_coeffs_hi_sum = 0;
    p.log2_w = log2f((float)w);
    p.log2_h = log2f((float)h);
    p.adm_ref_display_height = adm_ref_display_height;
    p.adm_norm_view_dist = adm_norm_view_dist;
    p.adm_enhn_gain_limit = adm_enhn_gain_limit;

    const double pow2_32 = (double)(1ULL << 32);
    const double pow2_21 = (double)(1ULL << 21);
    const double pow2_23 = (double)(1ULL << 23);
    for (unsigned scale = 0; scale < 4; ++scale) {
        float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], (int)scale, 1,
                                       adm_norm_view_dist, adm_ref_display_height);
        float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], (int)scale, 2,
                                       adm_norm_view_dist, adm_ref_display_height);
        p.factor1[scale] = factor1;
        p.factor2[scale] = factor2;
        p.rfactor[scale * 3] = 1.0f / factor1;
        p.rfactor[scale * 3 + 1] = 1.0f / factor1;
        p.rfactor[scale * 3 + 2] = 1.0f / factor2;
        if (scale == 0) {
            if (fabs(p.adm_norm_view_dist * p.adm_ref_display_height -
                     DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8) {
                p.i_rfactor[scale * 3] = 36453;
                p.i_rfactor[scale * 3 + 1] = 36453;
                p.i_rfactor[scale * 3 + 2] = 49417;
            } else {
                p.i_rfactor[scale * 3] = (uint32_t)(p.rfactor[scale * 3] * pow2_21);
                p.i_rfactor[scale * 3 + 1] = (uint32_t)(p.rfactor[scale * 3 + 1] * pow2_21);
                p.i_rfactor[scale * 3 + 2] = (uint32_t)(p.rfactor[scale * 3 + 2] * pow2_23);
            }
        } else {
            p.i_rfactor[scale * 3] = (uint32_t)(p.rfactor[scale * 3] * pow2_32);
            p.i_rfactor[scale * 3 + 1] = (uint32_t)(p.rfactor[scale * 3 + 1] * pow2_32);
            p.i_rfactor[scale * 3 + 2] = (uint32_t)(p.rfactor[scale * 3 + 2] * pow2_32);
        }
    }

    /* Zero result accumulator */
    hipError_t hip_err = hipMemsetAsync(buf->tmp_res, 0, sizeof(int64_t) * RES_BUFFER_SIZE, s->str);
    if (hip_err != hipSuccess)
        return hip_rc(hip_err);

    size_t curr_ref_stride;
    size_t curr_dis_stride;
    size_t buf_stride = buf->ind_size_x >> 2; /* bytes → int32 elements */

    int32_t *i4_curr_ref_scale = NULL;
    int32_t *i4_curr_dis_scale = NULL;

    if (ref_pic->bpc == 8) {
        curr_ref_stride = ref_pic->stride[0];
        curr_dis_stride = dis_pic->stride[0];
    } else {
        curr_ref_stride = ref_pic->stride[0] >> 1;
        curr_dis_stride = dis_pic->stride[0] >> 1;
    }

    int err = 0;
    for (unsigned scale = 0; scale < 4; ++scale) {
        if (scale == 0) {
            if (ref_pic->bpc == 8) {
                err = dwt2_8_device_hip(s, (const uint8_t *)ref_pic->data[0], &buf->ref_dwt2,
                                        buf->i4_ref_dwt2, w, h, (int)curr_ref_stride,
                                        (int)buf_stride, &p, /* pic_stream */ 0);
                if (err)
                    return err;
                err = dwt2_8_device_hip(s, (const uint8_t *)dis_pic->data[0], &buf->dis_dwt2,
                                        buf->i4_dis_dwt2, w, h, (int)curr_dis_stride,
                                        (int)buf_stride, &p, /* pic_stream */ 0);
                if (err)
                    return err;
            } else {
                err = dwt2_16_device_hip(s, (const uint16_t *)ref_pic->data[0], &buf->ref_dwt2,
                                         buf->i4_ref_dwt2, w, h, (int)curr_ref_stride,
                                         (int)buf_stride, (int)ref_pic->bpc, &p,
                                         /* pic_stream */ 0);
                if (err)
                    return err;
                err = dwt2_16_device_hip(s, (const uint16_t *)dis_pic->data[0], &buf->dis_dwt2,
                                         buf->i4_dis_dwt2, w, h, (int)curr_dis_stride,
                                         (int)buf_stride, (int)dis_pic->bpc, &p,
                                         /* pic_stream */ 0);
                if (err)
                    return err;
            }

            /* Sync: record per-picture events, wait on the ADM stream */
            hip_err = hipEventRecord(s->ref_event, /* pic stream */ 0);
            if (hip_err != hipSuccess)
                return hip_rc(hip_err);
            hip_err = hipEventRecord(s->dis_event, /* pic stream */ 0);
            if (hip_err != hipSuccess)
                return hip_rc(hip_err);
            hip_err = hipStreamWaitEvent(s->str, s->dis_event, 0);
            if (hip_err != hipSuccess)
                return hip_rc(hip_err);
            hip_err = hipStreamWaitEvent(s->str, s->ref_event, 0);
            if (hip_err != hipSuccess)
                return hip_rc(hip_err);

            w = (w + 1) / 2;
            h = (h + 1) / 2;

            err = adm_csf_den_scale_device_hip(s, buf, w, h, (int)buf_stride, s->str);
            if (err)
                return err;

            err = adm_csf_device_hip(s, buf, w, h, (int)buf_stride, &p, s->str);
            if (err)
                return err;

            err = adm_cm_device_hip(s, buf, w, h, (int)buf_stride, (int)buf_stride, &p, s->str);
            if (err)
                return err;
        } else {
            err = adm_dwt2_s123_combined_device_hip(s, i4_curr_ref_scale, (int32_t *)buf->tmp_ref,
                                                    buf->i4_ref_dwt2, w, h, (int)curr_ref_stride,
                                                    (int)buf_stride, (int)scale, &p, s->str);
            if (err)
                return err;
            err = adm_dwt2_s123_combined_device_hip(s, i4_curr_dis_scale, (int32_t *)buf->tmp_dis,
                                                    buf->i4_dis_dwt2, w, h, (int)curr_dis_stride,
                                                    (int)buf_stride, (int)scale, &p, s->str);
            if (err)
                return err;

            w = (w + 1) / 2;
            h = (h + 1) / 2;

            err = adm_csf_den_s123_device_hip(s, buf, (int)scale, w, h, (int)buf_stride, s->str);
            if (err)
                return err;

            err = i4_adm_csf_device_hip(s, buf, (int)scale, w, h, (int)buf_stride, &p, s->str);
            if (err)
                return err;

            err = i4_adm_cm_device_hip(s, buf, w, h, (int)buf_stride, (int)buf_stride, (int)scale,
                                       &p, s->str);
            if (err)
                return err;
        }

        i4_curr_ref_scale = buf->i4_ref_dwt2.band_a;
        i4_curr_dis_scale = buf->i4_dis_dwt2.band_a;
        curr_ref_stride = buf_stride;
        curr_dis_stride = buf_stride;
    }

    hip_err = hipMemcpyAsync(buf->results_host, buf->tmp_res, sizeof(int64_t) * RES_BUFFER_SIZE,
                             hipMemcpyDeviceToHost, s->str);
    if (hip_err != hipSuccess)
        return hip_rc(hip_err);
    hip_err = hipEventRecord(s->finished, s->str);
    return hip_rc(hip_err);
}

#endif /* HAVE_HIPCC */

/* ================================================================== */
/* init / submit / collect / flush / close                             */
/* ================================================================== */

/* ALIGN_CEIL: round up to the nearest multiple of 64-byte cache line. */
#define ADM_HIP_ALIGN 64
#define ADM_ALIGN_CEIL(x) (((x) + ADM_HIP_ALIGN - 1) & ~(size_t)(ADM_HIP_ALIGN - 1))

static int init_fex_hip(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                        unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    AdmStateHip *s = fex->priv;

    if (s->adm_norm_view_dist * s->adm_ref_display_height <
        DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) {
        return -EINVAL;
    }

#ifndef HAVE_HIPCC
    (void)w;
    (void)h;
    /* Scaffold: no runtime available. */
    return -ENOSYS;
#else
    hipError_t hip_err;

    /* Private stream */
    hip_err = hipStreamCreateWithFlags(&s->str, hipStreamNonBlocking);
    if (hip_err != hipSuccess)
        return hip_rc(hip_err);

    /* Events */
    hip_err = hipEventCreateWithFlags(&s->finished, hipEventDefault);
    if (hip_err != hipSuccess)
        goto fail_stream;
    hip_err = hipEventCreateWithFlags(&s->ref_event, hipEventDefault);
    if (hip_err != hipSuccess)
        goto fail_ev_finished;
    hip_err = hipEventCreateWithFlags(&s->dis_event, hipEventDefault);
    if (hip_err != hipSuccess)
        goto fail_ev_ref;

    /* Load HSACO modules */
    hip_err = hipModuleLoadData(&s->adm_dwt_module, (const void *)adm_dwt2_hsaco);
    if (hip_err != hipSuccess)
        goto fail_ev_dis;
    hip_err = hipModuleLoadData(&s->adm_csf_module, (const void *)adm_csf_hsaco);
    if (hip_err != hipSuccess)
        goto fail_mod_dwt;
    hip_err = hipModuleLoadData(&s->adm_csf_den_module, (const void *)adm_csf_den_hsaco);
    if (hip_err != hipSuccess)
        goto fail_mod_csf;
    hip_err = hipModuleLoadData(&s->adm_cm_module, (const void *)adm_cm_hsaco);
    if (hip_err != hipSuccess)
        goto fail_mod_csf_den;

    /* Kernel function handles */
#define GET_FN(mod, fn_ptr, name)                                                                  \
    do {                                                                                           \
        hip_err = hipModuleGetFunction((fn_ptr), (mod), (name));                                   \
        if (hip_err != hipSuccess)                                                                 \
            goto fail_mod_cm;                                                                      \
    } while (0)

    GET_FN(s->adm_dwt_module, &s->func_dwt_s123_combined_vert_kernel_0_0_int32_t,
           "dwt_s123_combined_vert_kernel_0_0_int32_t");
    GET_FN(s->adm_dwt_module, &s->func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
           "dwt_s123_combined_vert_kernel_32768_16_int32_t");
    GET_FN(s->adm_dwt_module, &s->func_dwt_s123_combined_hori_kernel_16384_15,
           "dwt_s123_combined_hori_kernel_16384_15");
    GET_FN(s->adm_dwt_module, &s->func_dwt_s123_combined_hori_kernel_32768_16,
           "dwt_s123_combined_hori_kernel_32768_16");
    GET_FN(s->adm_dwt_module, &s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t,
           "adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t");
    GET_FN(s->adm_dwt_module, &s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t,
           "adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t");

    GET_FN(s->adm_csf_module, &s->func_adm_csf_kernel_1_4, "adm_csf_kernel_1_4");
    GET_FN(s->adm_csf_module, &s->func_i4_adm_csf_kernel_1_4, "i4_adm_csf_kernel_1_4");

    GET_FN(s->adm_csf_den_module, &s->func_adm_csf_den_scale_line_kernel,
           "adm_csf_den_scale_line_kernel_8_128");
    GET_FN(s->adm_csf_den_module, &s->func_adm_csf_den_s123_line_kernel,
           "adm_csf_den_s123_line_kernel_8_128");

    GET_FN(s->adm_cm_module, &s->func_adm_cm_reduce_line_kernel_4, "adm_cm_reduce_line_kernel_4");
    GET_FN(s->adm_cm_module, &s->func_adm_cm_line_kernel_8, "adm_cm_line_kernel_8");
    GET_FN(s->adm_cm_module, &s->func_i4_adm_cm_line_kernel, "i4_adm_cm_line_kernel");
#undef GET_FN

    /* Buffer allocation — mirrors init_fex_cuda layout exactly */
    s->integer_stride = ADM_ALIGN_CEIL(w * sizeof(int32_t));
    s->buf.ind_size_x = ADM_ALIGN_CEIL(((w + 1) / 2) * sizeof(int32_t));
    s->buf.ind_size_y = ADM_ALIGN_CEIL(((h + 1) / 2) * sizeof(int32_t));
    const size_t buf_sz_one = s->buf.ind_size_x * ((h + 1) / 2);

    hip_err = hipMalloc(&s->buf.data_buf, buf_sz_one * 11 + buf_sz_one / 2 * 11);
    if (hip_err != hipSuccess)
        goto fail_mod_cm;
    hip_err = hipMalloc(&s->buf.tmp_ref, s->integer_stride * 4 * ((h + 1) / 2));
    if (hip_err != hipSuccess)
        goto fail_data_buf;
    hip_err = hipMalloc(&s->buf.tmp_dis, s->integer_stride * 4 * ((h + 1) / 2));
    if (hip_err != hipSuccess)
        goto fail_tmp_ref;
    hip_err = hipMalloc(&s->buf.tmp_accum, sizeof(uint64_t) * 3u * (size_t)w * (size_t)h);
    if (hip_err != hipSuccess)
        goto fail_tmp_dis;
    hip_err = hipMalloc(&s->buf.tmp_accum_h, sizeof(uint64_t) * 3u * (size_t)h);
    if (hip_err != hipSuccess)
        goto fail_tmp_accum;
    hip_err = hipMalloc(&s->buf.tmp_res, sizeof(uint64_t) * RES_BUFFER_SIZE);
    if (hip_err != hipSuccess)
        goto fail_tmp_accum_h;
    hip_err = hipHostMalloc(&s->buf.results_host, sizeof(uint64_t) * RES_BUFFER_SIZE,
                            hipHostMallocDefault);
    if (hip_err != hipSuccess)
        goto fail_tmp_res;

    /* Slice the backing buffer into band pointers — mirrors init_dwt_band_cuda logic */
    {
        uint8_t *top = (uint8_t *)s->buf.data_buf;
        const size_t half = buf_sz_one / 2;

        s->buf.ref_dwt2.band_a = (int16_t *)(top);
        s->buf.ref_dwt2.band_h = (int16_t *)(top + half);
        s->buf.ref_dwt2.band_v = (int16_t *)(top + 2u * half);
        s->buf.ref_dwt2.band_d = (int16_t *)(top + 3u * half);
        top += 4u * half;

        s->buf.dis_dwt2.band_a = (int16_t *)(top);
        s->buf.dis_dwt2.band_h = (int16_t *)(top + half);
        s->buf.dis_dwt2.band_v = (int16_t *)(top + 2u * half);
        s->buf.dis_dwt2.band_d = (int16_t *)(top + 3u * half);
        top += 4u * half;

        /* csf_f: band_a == NULL (hvd only) */
        s->buf.csf_f.band_a = NULL;
        s->buf.csf_f.band_h = (int16_t *)(top);
        s->buf.csf_f.band_v = (int16_t *)(top + half);
        s->buf.csf_f.band_d = (int16_t *)(top + 2u * half);
        top += 3u * half;

        /* i4 bands (full int32 size = buf_sz_one each) */
        s->buf.i4_ref_dwt2.band_a = (int32_t *)(top);
        s->buf.i4_ref_dwt2.band_h = (int32_t *)(top + buf_sz_one);
        s->buf.i4_ref_dwt2.band_v = (int32_t *)(top + 2u * buf_sz_one);
        s->buf.i4_ref_dwt2.band_d = (int32_t *)(top + 3u * buf_sz_one);
        top += 4u * buf_sz_one;

        s->buf.i4_dis_dwt2.band_a = (int32_t *)(top);
        s->buf.i4_dis_dwt2.band_h = (int32_t *)(top + buf_sz_one);
        s->buf.i4_dis_dwt2.band_v = (int32_t *)(top + 2u * buf_sz_one);
        s->buf.i4_dis_dwt2.band_d = (int32_t *)(top + 3u * buf_sz_one);
        top += 4u * buf_sz_one;

        s->buf.i4_csf_f.band_a = NULL;
        s->buf.i4_csf_f.band_h = (int32_t *)(top);
        s->buf.i4_csf_f.band_v = (int32_t *)(top + buf_sz_one);
        s->buf.i4_csf_f.band_d = (int32_t *)(top + 2u * buf_sz_one);
    }

    /* Slice result accumulator */
    {
        const size_t cm_stride = 3u * sizeof(int64_t);
        const size_t csf_stride = 3u * sizeof(uint64_t);
        uint8_t *res = (uint8_t *)s->buf.tmp_res;
        for (int i = 0; i < 4; ++i) {
            s->buf.adm_cm[i] = (int64_t *)(res + (size_t)i * cm_stride);
        }
        res += 4u * cm_stride;
        for (int i = 0; i < 4; ++i) {
            s->buf.adm_csf_den[i] = (uint64_t *)(res + (size_t)i * csf_stride);
        }
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (s->feature_name_dict == NULL)
        goto fail_host;

    return 0;

fail_host:
    (void)hipHostFree(s->buf.results_host);
    s->buf.results_host = NULL;
fail_tmp_res:
    (void)hipFree(s->buf.tmp_res);
    s->buf.tmp_res = NULL;
fail_tmp_accum_h:
    (void)hipFree(s->buf.tmp_accum_h);
    s->buf.tmp_accum_h = NULL;
fail_tmp_accum:
    (void)hipFree(s->buf.tmp_accum);
    s->buf.tmp_accum = NULL;
fail_tmp_dis:
    (void)hipFree(s->buf.tmp_dis);
    s->buf.tmp_dis = NULL;
fail_tmp_ref:
    (void)hipFree(s->buf.tmp_ref);
    s->buf.tmp_ref = NULL;
fail_data_buf:
    (void)hipFree(s->buf.data_buf);
    s->buf.data_buf = NULL;
fail_mod_cm:
    (void)hipModuleUnload(s->adm_cm_module);
    s->adm_cm_module = NULL;
fail_mod_csf_den:
    (void)hipModuleUnload(s->adm_csf_den_module);
    s->adm_csf_den_module = NULL;
fail_mod_csf:
    (void)hipModuleUnload(s->adm_csf_module);
    s->adm_csf_module = NULL;
fail_mod_dwt:
    (void)hipModuleUnload(s->adm_dwt_module);
    s->adm_dwt_module = NULL;
fail_ev_dis:
    (void)hipEventDestroy(s->dis_event);
fail_ev_ref:
    (void)hipEventDestroy(s->ref_event);
fail_ev_finished:
    (void)hipEventDestroy(s->finished);
fail_stream:
    (void)hipStreamDestroy(s->str);
    return hip_rc(hip_err);
#endif /* HAVE_HIPCC */
}

static int submit_fex_hip(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                          VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;
    (void)index;

    AdmStateHip *s = fex->priv;
    s->submit_w = ref_pic->w[0];
    s->submit_h = ref_pic->h[0];

#ifndef HAVE_HIPCC
    return -ENOSYS;
#else
    return integer_compute_adm_hip(s, ref_pic, dist_pic, &s->buf, s->adm_enhn_gain_limit,
                                   s->adm_norm_view_dist, s->adm_ref_display_height);
#endif
}

static int collect_fex_hip(VmafFeatureExtractor *fex, unsigned index,
                           VmafFeatureCollector *feature_collector)
{
    AdmStateHip *s = fex->priv;

#ifndef HAVE_HIPCC
    (void)index;
    (void)feature_collector;
    return -ENOSYS;
#else
    hipError_t hip_err = hipStreamSynchronize(s->str);
    if (hip_err != hipSuccess)
        return hip_rc(hip_err);

    write_score_parameters_adm_hip params = {
        .feature_collector = feature_collector,
        .s = s,
        .index = index,
        .w = s->submit_w,
        .h = s->submit_h,
    };
    write_scores(&params);
    return 0;
#endif
}

static int flush_fex_hip(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    AdmStateHip *s = fex->priv;
#ifndef HAVE_HIPCC
    (void)s;
    return -ENOSYS;
#else
    hipError_t hip_err = hipStreamSynchronize(s->str);
    if (hip_err != hipSuccess)
        return hip_rc(hip_err);
    return 1;
#endif
}

static int close_fex_hip(VmafFeatureExtractor *fex)
{
    AdmStateHip *s = fex->priv;
    int rc = 0;

#ifdef HAVE_HIPCC
    hipError_t hip_err = hipStreamSynchronize(s->str);
    if (hip_err != hipSuccess)
        rc = hip_rc(hip_err);
    hip_err = hipStreamDestroy(s->str);
    if (hip_err != hipSuccess && rc == 0)
        rc = hip_rc(hip_err);
    hip_err = hipEventDestroy(s->finished);
    if (hip_err != hipSuccess && rc == 0)
        rc = hip_rc(hip_err);
    hip_err = hipEventDestroy(s->ref_event);
    if (hip_err != hipSuccess && rc == 0)
        rc = hip_rc(hip_err);
    hip_err = hipEventDestroy(s->dis_event);
    if (hip_err != hipSuccess && rc == 0)
        rc = hip_rc(hip_err);

    if (s->adm_cm_module != NULL) {
        (void)hipModuleUnload(s->adm_cm_module);
        s->adm_cm_module = NULL;
    }
    if (s->adm_csf_den_module != NULL) {
        (void)hipModuleUnload(s->adm_csf_den_module);
        s->adm_csf_den_module = NULL;
    }
    if (s->adm_csf_module != NULL) {
        (void)hipModuleUnload(s->adm_csf_module);
        s->adm_csf_module = NULL;
    }
    if (s->adm_dwt_module != NULL) {
        (void)hipModuleUnload(s->adm_dwt_module);
        s->adm_dwt_module = NULL;
    }
    if (s->buf.results_host != NULL) {
        (void)hipHostFree(s->buf.results_host);
        s->buf.results_host = NULL;
    }
    if (s->buf.tmp_res != NULL) {
        (void)hipFree(s->buf.tmp_res);
        s->buf.tmp_res = NULL;
    }
    if (s->buf.tmp_accum_h != NULL) {
        (void)hipFree(s->buf.tmp_accum_h);
        s->buf.tmp_accum_h = NULL;
    }
    if (s->buf.tmp_accum != NULL) {
        (void)hipFree(s->buf.tmp_accum);
        s->buf.tmp_accum = NULL;
    }
    if (s->buf.tmp_dis != NULL) {
        (void)hipFree(s->buf.tmp_dis);
        s->buf.tmp_dis = NULL;
    }
    if (s->buf.tmp_ref != NULL) {
        (void)hipFree(s->buf.tmp_ref);
        s->buf.tmp_ref = NULL;
    }
    if (s->buf.data_buf != NULL) {
        (void)hipFree(s->buf.data_buf);
        s->buf.data_buf = NULL;
    }
#endif /* HAVE_HIPCC */

    if (s->feature_name_dict != NULL) {
        int err = vmaf_dictionary_free(&s->feature_name_dict);
        if (err != 0 && rc == 0)
            rc = err;
    }
    return rc;
}

/* ------------------------------------------------------------------ */
/* Feature extractor descriptor                                        */
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

/*
 * Registration note: the extractor is declared as non-static so it can be
 * referenced by `extern VmafFeatureExtractor vmaf_fex_integer_adm_hip` in
 * feature_extractor.c's lookup table. Making it static would unlink it from
 * the registry. Same pattern as every other GPU feature extractor in this
 * tree (e.g. `vmaf_fex_integer_adm_cuda` in integer_adm_cuda.c).
 */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_integer_adm_hip = {
    .name = "adm_hip",
    .init = init_fex_hip,
    .submit = submit_fex_hip,
    .collect = collect_fex_hip,
    .flush = flush_fex_hip,
    .close = close_fex_hip,
    .options = options_hip,
    .priv_size = sizeof(AdmStateHip),
    .provided_features = provided_features,
    /*
     * VMAF_FEATURE_EXTRACTOR_HIP flag bit is reserved (mirrors the
     * pattern used by the CUDA twin which uses VMAF_FEATURE_EXTRACTOR_CUDA).
     * The runtime PR (T7-10b) wires in the buffer-type plumbing and flips
     * this flag on. Until then, callers receive -ENOSYS from init() on
     * non-ROCm builds.
     */
    .flags = 0,
    .chars =
        {
            .n_dispatches_per_frame = 1,
            .is_reduction_only = false,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};
