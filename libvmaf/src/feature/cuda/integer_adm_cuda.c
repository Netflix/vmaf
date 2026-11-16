/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include "common.h"
#include <stdio.h>
#include <stdlib.h>

#include "cuda_helper.cuh"
#include "mem.h"

#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "cpu.h"
#include "cuda/integer_adm_cuda.h"
#include "picture_cuda.h"

#include <assert.h>

#include "barten_csf_tools.h"

/* 4 scales of adm_cm, adm_csf_den and aim_num, 3 bands each */
#define RES_BUFFER_SIZE (4 * 3 * 3)

/*
 * The state of the CUDA implementation. Every option of the reference
 * implementation is mirrored here, including its name, alias, type, default
 * value and feature parameter flag, so that the generated feature names are
 * identical.
 */
typedef struct AdmStateCuda {
    size_t integer_stride;
    AdmBufferCuda buf;
    bool debug;
    double adm_csf_scale;
    double adm_csf_diag_scale;
    double adm_dlm_weight;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    int adm_csf_mode;
    double adm_noise_weight;
    bool adm_skip_aim;
    bool adm_skip_scale0;
    double adm_min_val;
    CUstream str, host_stream;
    void* write_score_parameters;
    CUevent ref_event, dis_event, finished;
    VmafDictionary *feature_name_dict;

    // adm_dwt kernels
    CUfunction func_dwt_s123_combined_vert_kernel_0_0_int32_t,
               func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
               func_dwt_s123_combined_hori_kernel_16384_15,
               func_dwt_s123_combined_hori_kernel_32768_16,
               func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t,
               func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t,
               // adm_decouple kernel
               func_adm_decouple_kernel,
               func_adm_decouple_s123_kernel,
               // adm_csf kernel
               func_i4_adm_csf_kernel_1_4,
               func_adm_csf_kernel_1_4,
               // adm_csf_den kernel
               func_adm_csf_den_scale_row_kernel,
               func_adm_csf_den_s123_row_kernel,
               // adm_cm kernel
               func_adm_cm_row_kernel,
               func_i4_adm_cm_row_kernel;

} AdmStateCuda;

/*
 * lambda = 0 (finest scale), 1, 2, 3 (coarsest scale);
 * theta = 0 (ll), 1 (lh - vertical), 2 (hh - diagonal), 3(hl - horizontal).
 */
static inline float dwt_quant_step(const struct dwt_model_params *params,
        int lambda, int theta, double adm_norm_view_dist, int adm_ref_display_height)
{
    // Formula (1), page 1165 - display visual resolution (DVR), in pixels/degree of visual angle. This should be 56.55
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;

    // Formula (9), page 1171
    float temp = log10(pow(2.0,lambda+1)*params->f0*params->g[theta]/r);
    float Q = 2.0*params->a*pow(10.0,params->k*temp*temp)/dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

/*
 * Border of the decouple/csf kernels, identical to the reference
 * implementation (-1 for the filter tap).
 */
static void adm_csf_borders(const int w, const int h, int *top, int *bottom,
        int *left, int *right)
{
    int l = w * ADM_BORDER_FACTOR - 0.5 - 1; // -1 for filter tap
    int t = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int r = w - l + 2; // +2 for filter tap
    int b = h - t + 2;

    if (l < 0) {
        l = 0;
    }
    if (r > w) {
        r = w;
    }
    if (t < 0) {
        t = 0;
    }
    if (b > h) {
        b = h;
    }

    *top = t;
    *bottom = b;
    *left = l;
    *right = r;
}

/*
 * Border of the csf_den/cm kernels, identical to the reference
 * implementation.
 */
static void adm_cm_borders(const int w, const int h, int *top, int *bottom,
        int *left, int *right)
{
    const int l = w * ADM_BORDER_FACTOR - 0.5;
    const int t = h * ADM_BORDER_FACTOR - 0.5;

    *top = t;
    *bottom = h - t;
    *left = l;
    *right = w - l;
}

/*
 * rfactor[3] = { factor1, factor1, factor2 }, identical to the reference
 * implementation for the requested contrast sensitivity function.
 */
static void adm_csf_rfactor(const int scale, const int adm_csf_mode,
        const double adm_norm_view_dist, const int adm_ref_display_height,
        const double adm_csf_scale, const double adm_csf_diag_scale,
        float rfactor[3])
{
    float factor1, factor2;
    if (adm_csf_mode == ADM_CSF_MODE_BARTEN) {
        factor1 = barten_csf(scale, adm_norm_view_dist, adm_ref_display_height,
                DEFAULT_ADM_CSF_LUM, adm_csf_scale);
        factor2 = barten_csf(scale, adm_norm_view_dist, adm_ref_display_height,
                DEFAULT_ADM_CSF_LUM, adm_csf_diag_scale);
    } else if (adm_csf_mode == ADM_CSF_MODE_BARTEN_WATSON_BLEND) {
        factor1 = barten_watson_blend_csf(scale, 0, adm_norm_view_dist, adm_ref_display_height);
        factor2 = barten_watson_blend_csf(scale, 1, adm_norm_view_dist, adm_ref_display_height);
    } else if (adm_csf_mode == ADM_CSF_MODE_BARTEN_WATSON_BLEND_MAE) {
        factor1 = barten_watson_blend_csf_mae(scale, 0, adm_norm_view_dist, adm_ref_display_height);
        factor2 = barten_watson_blend_csf_mae(scale, 1, adm_norm_view_dist, adm_ref_display_height);
    } else {
        factor1 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1,
                adm_norm_view_dist, adm_ref_display_height);
        factor2 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2,
                adm_norm_view_dist, adm_ref_display_height);
    }
    rfactor[0] = factor1;
    rfactor[1] = factor1;
    rfactor[2] = factor2;
}

/*
 * The scale 0 fixed-point rfactor of the reference implementation: Q21 for
 * the first two bands and Q23 for the last one.
 */
static void adm_i_rfactor_scale0(const float rfactor[3],
        const double adm_norm_view_dist, const int adm_ref_display_height,
        const int adm_csf_mode, uint16_t i_rfactor[3])
{
    if (fabs(adm_norm_view_dist * adm_ref_display_height -
                DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8 &&
            adm_csf_mode == ADM_CSF_MODE_WATSON97) {
        i_rfactor[0] = 36453;
        i_rfactor[1] = 36453;
        i_rfactor[2] = 49417;
    } else {
        const double pow2_21 = pow(2, 21);
        const double pow2_23 = pow(2, 23);
        i_rfactor[0] = (uint16_t)(rfactor[0] * pow2_21);
        i_rfactor[1] = (uint16_t)(rfactor[1] * pow2_21);
        i_rfactor[2] = (uint16_t)(rfactor[2] * pow2_23);
    }
}

/*
 * The scale 1..3 fixed-point rfactor of the reference implementation: Q32
 * for all three bands.
 */
static void adm_i_rfactor_s123(const float rfactor[3], uint32_t i_rfactor[3])
{
    const double pow2_32 = pow(2, 32);
    i_rfactor[0] = (uint32_t)(rfactor[0] * pow2_32);
    i_rfactor[1] = (uint32_t)(rfactor[1] * pow2_32);
    i_rfactor[2] = (uint32_t)(rfactor[2] * pow2_32);
}

static void dwt2_8_device(AdmStateCuda *s, const uint8_t *d_picture, cuda_adm_dwt_band_t *d_dst, cuda_i4_adm_dwt_band_t i4_dwt_dst,
        int w, int h, int src_stride, int dst_stride, AdmFixedParametersCuda *p,
        CudaFunctions* cu_f, CUstream c_stream) {
    int rows_per_thread = 4;

    int vert_out_tile_rows = 8;
    int vert_out_tile_cols = 128;

    int horz_out_tile_rows = vert_out_tile_rows;
    int horz_out_tile_cols = vert_out_tile_cols / 2 - 2;

    int16_t v_shift = 8;
    int32_t v_add_shift = 1 << (v_shift - 1);

    void *args[] = {&d_picture, &*d_dst, &i4_dwt_dst, &w, &h, &src_stride, &dst_stride, &v_shift, &v_add_shift, &*p};
    CHECK_CUDA(cu_f, cuLaunchKernel(s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t,
                DIV_ROUND_UP((w + 1) / 2, horz_out_tile_cols), DIV_ROUND_UP((h + 1) / 2, horz_out_tile_rows), 1,
                vert_out_tile_cols, vert_out_tile_rows / rows_per_thread, 1,
                0, c_stream, args, NULL));
}

static void dwt2_16_device(AdmStateCuda *s, const uint16_t *d_picture, cuda_adm_dwt_band_t *d_dst, cuda_i4_adm_dwt_band_t i4_dwt_dst,
        int w, int h, int src_stride, int dst_stride, int inp_size_bits,
        AdmFixedParametersCuda *p, CudaFunctions* cu_f, CUstream c_stream) {
    int rows_per_thread = 4;

    int vert_out_tile_rows = 8;
    int vert_out_tile_cols = 128;

    int horz_out_tile_rows = vert_out_tile_rows;
    int horz_out_tile_cols = vert_out_tile_cols / 2 - 2;

    int16_t v_shift = inp_size_bits;
    int32_t v_add_shift = 1 << (inp_size_bits - 1);

    void *args[] = {&d_picture, &*d_dst, &i4_dwt_dst, &w, &h, &src_stride, &dst_stride, &v_shift, &v_add_shift, &*p};
    CHECK_CUDA(cu_f, cuLaunchKernel(s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t,
                DIV_ROUND_UP((w + 1) / 2, horz_out_tile_cols), DIV_ROUND_UP((h + 1) / 2, horz_out_tile_rows), 1,
                vert_out_tile_cols, vert_out_tile_rows / rows_per_thread, 1,
                0, c_stream, args, NULL));
}

static void adm_dwt2_s123_combined_device(AdmStateCuda *s, const int32_t *d_i4_scale, int32_t *tmp_buf,
        cuda_i4_adm_dwt_band_t i4_dwt, int w, int h, int img_stride, int dst_stride, int scale,
        AdmFixedParametersCuda *p, CudaFunctions* cu_f, CUstream cu_stream) {
    const int BLOCK_Y = (h + 1) / 2;

    void * args_vert[] = {&d_i4_scale, &tmp_buf, &w, &h, &img_stride, &*p};
    switch (scale) {
        case 1:
            CHECK_CUDA(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_vert_kernel_0_0_int32_t,
                        DIV_ROUND_UP(w, 128), BLOCK_Y, 1,
                        128, 1, 1,
                        0, cu_stream, args_vert, NULL));
            break;
        case 2:
            CHECK_CUDA(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
                        DIV_ROUND_UP(w, 128), BLOCK_Y, 1,
                        128, 1, 1,
                        0, cu_stream, args_vert, NULL));
            break;
        case 3:
            CHECK_CUDA(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
                        DIV_ROUND_UP(w, 32), BLOCK_Y, 1,
                        32, 1, 1,
                        0, cu_stream, args_vert, NULL));
            break;
    }

    void * args_hori[] = {&i4_dwt, &tmp_buf, &w, &h, &dst_stride, &*p};
    switch (scale) {
        case 1:
            CHECK_CUDA(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_hori_kernel_16384_15,
                        DIV_ROUND_UP(((w + 1) / 2), 128), BLOCK_Y, 1,
                        128, 1, 1,
                        0, cu_stream, args_hori, NULL));
            break;
        case 2:
            CHECK_CUDA(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_hori_kernel_32768_16,
                        DIV_ROUND_UP(((w + 1) / 2), 128), BLOCK_Y, 1,
                        128, 1, 1,
                        0, cu_stream, args_hori, NULL));
            break;
        case 3:
            CHECK_CUDA(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_hori_kernel_16384_15,
                        DIV_ROUND_UP(((w + 1) / 2), 32), BLOCK_Y, 1,
                        32, 1, 1,
                        0, cu_stream, args_hori, NULL));
            break;
    }
}

static void adm_decouple_device(AdmStateCuda *s, AdmBufferCuda *buf, int w, int h, int stride,
        double adm_enhn_gain_limit, float cos_1deg_sq, CudaFunctions* cu_f, CUstream c_stream) {
    int top, bottom, left, right;
    adm_csf_borders(w, h, &top, &bottom, &left, &right);

    const int BLOCKX = 16, BLOCKY = 8;
    void* args[] = {&*buf, &top, &bottom, &left, &right, &stride, &adm_enhn_gain_limit, &cos_1deg_sq};
    CHECK_CUDA(cu_f, cuLaunchKernel(s->func_adm_decouple_kernel,
                DIV_ROUND_UP(right - left, BLOCKX), DIV_ROUND_UP(bottom - top, BLOCKY), 1,
                BLOCKX, BLOCKY, 1,
                0, c_stream, args, NULL));
}

static void adm_decouple_s123_device(AdmStateCuda *s, AdmBufferCuda *buf, int w, int h, int stride,
        double adm_enhn_gain_limit, float cos_1deg_sq, CudaFunctions* cu_f, CUstream c_stream) {
    int top, bottom, left, right;
    adm_csf_borders(w, h, &top, &bottom, &left, &right);

    const int BLOCKX = 8, BLOCKY = 8;
    void* args[] = {&*buf, &top, &bottom, &left, &right, &stride, &adm_enhn_gain_limit, &cos_1deg_sq};
    CHECK_CUDA(cu_f, cuLaunchKernel(s->func_adm_decouple_s123_kernel,
                DIV_ROUND_UP(right - left, BLOCKX), DIV_ROUND_UP(bottom - top, BLOCKY), 1,
                BLOCKX, BLOCKY, 1,
                0, c_stream, args, NULL));
}

static void adm_csf_device(AdmStateCuda *s, AdmBufferCuda *buf, int scale, int w, int h, int stride,
        AdmCsfKernelParamsCuda *params, int measure_aim, CudaFunctions* cu_f, CUstream c_stream) {
    int top, bottom, left, right;
    adm_csf_borders(w, h, &top, &bottom, &left, &right);

    params->measure_aim = measure_aim;

    const int cols_per_thread = 4;
    const int rows_per_thread = 1;
    const int BLOCKX = 32, BLOCKY = 4;

    if (scale == 0) {
        void* args[] = {&*buf, &top, &bottom, &left, &right, &stride, params};
        CHECK_CUDA(cu_f, cuLaunchKernel(s->func_adm_csf_kernel_1_4,
                    DIV_ROUND_UP(right - left, BLOCKX * cols_per_thread),
                    DIV_ROUND_UP(bottom - top, BLOCKY * rows_per_thread), 3,
                    BLOCKX, BLOCKY, 1,
                    0, c_stream, args, NULL));
    } else {
        void* args[] = {&*buf, &scale, &top, &bottom, &left, &right, &stride, params};
        CHECK_CUDA(cu_f, cuLaunchKernel(s->func_i4_adm_csf_kernel_1_4,
                    DIV_ROUND_UP(right - left, BLOCKX * cols_per_thread),
                    DIV_ROUND_UP(bottom - top, BLOCKY * rows_per_thread), 3,
                    BLOCKX, BLOCKY, 1,
                    0, c_stream, args, NULL));
    }
}

static void adm_csf_den_device(AdmStateCuda *s, AdmBufferCuda *buf, int scale,
        AdmCsfDenKernelParamsCuda *params, CudaFunctions* cu_f, CUstream c_stream) {
    const int BLOCKX = VMAF_CUDA_THREADS_PER_WARP; /* one warp per row */
    const int rows = params->bottom - params->top;

    void* args[] = {&*buf, params, &buf->adm_csf_den[scale]};
    CHECK_CUDA(cu_f, cuLaunchKernel(scale == 0 ? s->func_adm_csf_den_scale_row_kernel
                                               : s->func_adm_csf_den_s123_row_kernel,
                1, rows, 3,
                BLOCKX, 1, 1,
                0, c_stream, args, NULL));
}

static void adm_cm_device(AdmStateCuda *s, AdmBufferCuda *buf, int scale,
        AdmCmKernelParamsCuda *params, int64_t *accum, int measure_aim,
        CudaFunctions* cu_f, CUstream c_stream) {
    const int BLOCKX = VMAF_CUDA_THREADS_PER_WARP; /* one warp per row */

    params->measure_aim = measure_aim;

    void* args[] = {&*buf, params, &accum};
    CHECK_CUDA(cu_f, cuLaunchKernel(scale == 0 ? s->func_adm_cm_row_kernel
                                               : s->func_i4_adm_cm_row_kernel,
                1, params->h, 1,
                BLOCKX, 1, 1,
                0, c_stream, args, NULL));
}

/*
 * Finalization of the adm_cm kernels, identical to the reference
 * implementation.
 */
static void conclude_adm_cm(const int64_t *accum, int h, int w, int scale,
        double adm_noise_weight, float *result) {
    int left = w * ADM_BORDER_FACTOR - 0.5;
    int top = h * ADM_BORDER_FACTOR - 0.5;
    int right = w - left;
    int bottom = h - top;

    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));

    const uint32_t shift_xhcub = (uint32_t)ceil(log2(w) - 4);
    const uint32_t shift_xvcub = (uint32_t)ceil(log2(w) - 4);
    const uint32_t shift_xdcub = (uint32_t)ceil(log2(w) - 3);

    const uint32_t shift_xcub[3] = { shift_xhcub, shift_xvcub, shift_xdcub };
    const uint32_t constant_offset[3] = { 52, 52, 57 };

    const uint32_t shift_cub = (uint32_t)ceil(log2(w));
    const float final_shift[3] = { pow(2, (45 - shift_cub - shift_inner_accum)),
        pow(2, (39 - shift_cub - shift_inner_accum)),
        pow(2, (36 - shift_cub - shift_inner_accum)) };

    const float powf_add = powf((bottom - top) * (right - left) * adm_noise_weight,
            1.0f / 3.0f);

    float result_v = 0;
    for (int i = 0; i < 3; ++i) {
        float f_accum;
        if (scale == 0) {
            f_accum = (float)(accum[i] / pow(2, (constant_offset[i] -
                            shift_xcub[i] - shift_inner_accum)));
        } else {
            f_accum = (float)(accum[i] / final_shift[scale - 1]);
        }
        result_v += powf(f_accum, 1.0f / 3.0f) + powf_add;
    }
    *result = result_v;
}

/*
 * Finalization of the adm_csf_den kernels, identical to the reference
 * implementation.
 */
static void conclude_adm_csf_den(const uint64_t *accum, int h, int w, int scale,
        double adm_norm_view_dist, int adm_ref_display_height, int adm_csf_mode,
        double adm_csf_scale, double adm_csf_diag_scale, double adm_noise_weight,
        float *result) {
    float rfactor[3];
    adm_csf_rfactor(scale, adm_csf_mode, adm_norm_view_dist, adm_ref_display_height,
            adm_csf_scale, adm_csf_diag_scale, rfactor);

    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;

    double shift_csf;
    if (scale == 0) {
        int32_t shift_accum = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
        shift_accum = shift_accum > 0 ? shift_accum : 0;
        shift_csf = pow(2, (18 - shift_accum));
    } else {
        const uint32_t accum_convert_float[3] = { 32, 27, 23 };
        const uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
        const uint32_t shift_accum = (uint32_t)ceil(log2(bottom - top));
        shift_csf = pow(2, (accum_convert_float[scale - 1] - shift_accum - shift_cub));
    }

    const float powf_add = powf((bottom - top) * (right - left) * adm_noise_weight,
            1.0f / 3.0f);

    float result_v = 0;
    for (int i = 0; i < 3; ++i) {
        const double csf = (double)(accum[i] / shift_csf) * pow(rfactor[i], 3);
        result_v += powf(csf, 1.0f / 3.0f) + powf_add;
    }
    *result = result_v;
}

static const VmafOption options_cuda[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(AdmStateCuda, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_csf_scale",
        .alias = "scf",
        .help = "scale coefficient for the horizontal & vertical direction terms of CSF",
        .offset = offsetof(AdmStateCuda, adm_csf_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_diag_scale",
        .alias = "scfd",
        .help = "scale coefficient for the diagonal direction term of CSF",
        .offset = offsetof(AdmStateCuda, adm_csf_diag_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_dlm_weight",
        .alias = "dlmw",
        .help = "linear weighting between DLM and AIM; 1 corresponds to DLM-only",
        .offset = offsetof(AdmStateCuda, adm_dlm_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.5,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmStateCuda, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmStateCuda, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdh",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmStateCuda, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_mode",
        .alias = "csf",
        .help = "contrast sensitivity function",
        .offset = offsetof(AdmStateCuda, adm_csf_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_CSF_MODE,
        .min = 0,
        .max = 3,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_noise_weight",
        .alias = "nw",
        .help = "noise weight",
        .offset = offsetof(AdmStateCuda, adm_noise_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_skip_aim",
        .help = "skip the calculation of AIM",
        .offset = offsetof(AdmStateCuda, adm_skip_aim),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_skip_scale0",
        .alias = "ssz",
        .help = "skip the calculation of scale 0",
        .offset = offsetof(AdmStateCuda, adm_skip_scale0),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_min_val",
        .alias = "min",
        .help = "minimum value allowed; lower values will be clipped to this value",
        .offset = offsetof(AdmStateCuda, adm_min_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_MIN_VAL,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

typedef struct write_score_parameters_adm {
    VmafFeatureCollector *feature_collector;
    AdmStateCuda *s;
    unsigned index, h, w;
} write_score_parameters_adm;

static void write_scores(write_score_parameters_adm* params)
{
    VmafFeatureCollector *feature_collector = params->feature_collector;
    AdmStateCuda *s = params->s;
    unsigned index = params->index;

    unsigned w = params->w;
    unsigned h = params->h;
    const double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);

    int64_t *res_host = (int64_t*)s->buf.results_host;
    int64_t *adm_cm = &res_host[0 * 3 * 4];
    uint64_t *adm_csf_den = (uint64_t*)&res_host[1 * 3 * 4];
    int64_t *aim_num = &res_host[2 * 3 * 4];

    double scores[8];
    double score, score_num, score_den, score_aim = 0.0;

    double num = 0;
    double den = 0;
    double aim_total = 0;

    for (unsigned scale = 0; scale < 4; ++scale) {
        float num_scale = 0.0f;
        float den_scale = 0.0f;
        float aim_num_scale = 0.0f;

        w = (w + 1) / 2;
        h = (h + 1) / 2;

        if (scale == 0 && s->adm_skip_scale0) {
            den_scale = 1e-10f; // avoid divide by zero
        } else {
            conclude_adm_cm(&adm_cm[scale * 3], h, w, scale, s->adm_noise_weight,
                    &num_scale);
            conclude_adm_csf_den(&adm_csf_den[scale * 3], h, w, scale,
                    s->adm_norm_view_dist, s->adm_ref_display_height,
                    s->adm_csf_mode, s->adm_csf_scale, s->adm_csf_diag_scale,
                    s->adm_noise_weight, &den_scale);

            if (!s->adm_skip_aim) {
                conclude_adm_cm(&aim_num[scale * 3], h, w, scale, 0.0,
                        &aim_num_scale);
            }
        }

        num += num_scale;
        den += den_scale;
        aim_total += aim_num_scale;

        scores[2 * scale + 0] = num_scale;
        scores[2 * scale + 1] = den_scale;
    }

    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;

    if (den == 0.0) {
        score = 1.0f;
    }
    else {
        // normalize AIM score by the DLM denominator
        score_aim = aim_total / den;
        score = num / den;
    }
    score_num = num;
    score_den = den;

    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_adm2_score", score,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_aim_score", score_aim,
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "VMAF_integer_feature_adm3_score",
            MAX(score * s->adm_dlm_weight + (1 - score_aim) * (1 - s->adm_dlm_weight), s->adm_min_val),
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_scale0", scores[0] / scores[1],
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_scale1", scores[2] / scores[3],
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_scale2", scores[4] / scores[5],
            index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_scale3", scores[6] / scores[7],
            index);

    if (!s->debug) return;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm", score, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num", score_num, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den", score_den, index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num_scale0", scores[0], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den_scale0", scores[1], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num_scale1", scores[2], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den_scale1", scores[3], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num_scale2", scores[4], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den_scale2", scores[5], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_num_scale3", scores[6], index);

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "integer_adm_den_scale3", scores[7], index);

    return;
}

/*
 * Scale 0 pipeline: decouple, csf_den, csf and cm of the DLM and of the AIM
 * pass, in the exact order of the reference implementation.
 */
static void adm_compute_scale0_cuda(VmafFeatureExtractor *fex, AdmStateCuda *s,
        AdmBufferCuda *buf, int w, int h, int stride, float cos_1deg_sq,
        const uint16_t i_rfactor[3])
{
    CudaFunctions* cu_f = fex->cu_state->f;

    AdmCsfKernelParamsCuda csf_p = { 0 };
    AdmCsfDenKernelParamsCuda den_p = { 0 };
    AdmCmKernelParamsCuda cm_p = { 0 };

    /*
     * Shifts pending from previous stage is 6, hence variables multiplied by
     * i_rfactor[0,1] have to be shifted by 21+6=27 to convert into
     * floating-point, but shifted by 15 to make it Q16; variables multiplied
     * by i_rfactor[2] have to be shifted by 23+6=29, but shifted by 17 to
     * make it Q16.
     */
    static const uint32_t i_shifts[3] = { 15, 15, 17 };
    static const int64_t i_shiftsadd[3] = { 16384, 16384, 65535 };
    for (int i = 0; i < 3; ++i) {
        csf_p.i_rfactor[i] = i_rfactor[i];
        csf_p.shift_dst[i] = i_shifts[i];
        csf_p.add_dst[i] = i_shiftsadd[i];
        csf_p.shift_flt[i] = 12;
        csf_p.add_flt[i] = 2048;
    }

    int top, bottom, left, right;
    adm_cm_borders(w, h, &top, &bottom, &left, &right);

    /*
     * max_value of h^3, v^3, d^3 is 1.205624776 * 10^13, accum_h can hold
     * till 1.844674407 * 10^19; accum_h's maximum is reached when it is
     * 2^20 * max(h^3), therefore the accum is shifted based on width and
     * height subtracted by 20.
     */
    int32_t shift_accum = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
    shift_accum = shift_accum > 0 ? shift_accum : 0;
    den_p.top = top;
    den_p.bottom = bottom;
    den_p.left = left;
    den_p.right = right;
    den_p.src_stride = stride;
    den_p.shift_accum = shift_accum;
    den_p.add_shift_accum = shift_accum > 0 ? (1 << (shift_accum - 1)) : 0;

    cm_p.w = w;
    cm_p.h = h;
    cm_p.top = top;
    cm_p.bottom = bottom;
    cm_p.left = left;
    cm_p.right = right;
    cm_p.start_col = (left > 1) ? left : 1;
    cm_p.end_col = (right < (w - 1)) ? right : (w - 1);
    cm_p.start_row = (top > 1) ? top : 1;
    cm_p.end_row = (bottom < (h - 1)) ? bottom : (h - 1);
    cm_p.src_stride = stride;
    for (int i = 0; i < 3; ++i) {
        cm_p.i_rfactor[i] = i_rfactor[i];
    }
    cm_p.shift_sub[0] = 10;
    cm_p.shift_sub[1] = 10;
    cm_p.shift_sub[2] = 12;
    cm_p.shift_sq[0] = 29;
    cm_p.shift_sq[1] = 29;
    cm_p.shift_sq[2] = 30;
    cm_p.add_shift_sq[0] = 268435456;
    cm_p.add_shift_sq[1] = 268435456;
    cm_p.add_shift_sq[2] = 536870912;
    cm_p.shift_cub[0] = (uint32_t)ceil(log2(w) - 4);
    cm_p.shift_cub[1] = (uint32_t)ceil(log2(w) - 4);
    cm_p.shift_cub[2] = (uint32_t)ceil(log2(w) - 3);
    for (int i = 0; i < 3; ++i) {
        cm_p.add_shift_cub[i] = (uint32_t)pow(2, (cm_p.shift_cub[i] - 1));
    }
    cm_p.shift_inner_accum = (uint32_t)ceil(log2(h));
    cm_p.add_shift_inner_accum = (uint32_t)pow(2, (cm_p.shift_inner_accum - 1));

    // consumes buf->ref_dwt2 , buf->dis_dwt2
    // produces buf->decouple_r , buf->decouple_a
    adm_decouple_device(s, buf, w, h, stride, s->adm_enhn_gain_limit,
            cos_1deg_sq, cu_f, s->str);

    // consumes buf->ref_dwt2
    // produces buf->adm_csf_den[0]
    adm_csf_den_device(s, buf, 0, &den_p, cu_f, s->str);

    // consumes buf->decouple_a
    // produces buf->csf_a , buf->csf_f
    adm_csf_device(s, buf, 0, w, h, stride, &csf_p, 0, cu_f, s->str);

    // consumes buf->decouple_r, buf->csf_a, buf->csf_f
    // produces buf->adm_cm[0]
    adm_cm_device(s, buf, 0, &cm_p, buf->adm_cm[0], 0, cu_f, s->str);

    if (!s->adm_skip_aim) {
        // consumes buf->decouple_r
        // produces buf->csf_a , buf->csf_f
        adm_csf_device(s, buf, 0, w, h, stride, &csf_p, 1, cu_f, s->str);

        // consumes buf->decouple_a, buf->csf_a, buf->csf_f
        // produces buf->aim_num[0]
        adm_cm_device(s, buf, 0, &cm_p, buf->aim_num[0], 1, cu_f, s->str);
    }
}

/*
 * Scale 1..3 pipeline: dwt, decouple, csf_den, csf and cm of the DLM and of
 * the AIM pass, in the exact order of the reference implementation.
 */
static void adm_compute_scale_s123_cuda(VmafFeatureExtractor *fex, AdmStateCuda *s,
        AdmBufferCuda *buf, int scale, int w, int h, int stride,
        float cos_1deg_sq, const float rfactor[3])
{
    CudaFunctions* cu_f = fex->cu_state->f;

    AdmCsfKernelParamsCuda csf_p = { 0 };
    AdmCsfDenKernelParamsCuda den_p = { 0 };
    AdmCmKernelParamsCuda cm_p = { 0 };

    uint32_t i_rfactor[3];
    adm_i_rfactor_s123(rfactor, i_rfactor);

    for (int i = 0; i < 3; ++i) {
        csf_p.i_rfactor[i] = i_rfactor[i];
        csf_p.shift_dst[i] = 28;
        /* the reference implementation stores 1u << 27 / 1u << 31 in int32_t */
        csf_p.add_dst[i] = (int64_t)(int32_t)(1u << 27);
        csf_p.shift_flt[i] = 32;
        csf_p.add_flt[i] = (int64_t)(int32_t)(1u << 31);
    }

    int top, bottom, left, right;
    adm_cm_borders(w, h, &top, &bottom, &left, &right);

    const uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
    const uint32_t shift_accum = (uint32_t)ceil(log2(bottom - top));
    static const uint32_t shift_sq_tab[3] = { 31, 30, 31 };

    den_p.top = top;
    den_p.bottom = bottom;
    den_p.left = left;
    den_p.right = right;
    den_p.src_stride = stride;
    den_p.shift_accum = shift_accum;
    den_p.add_shift_accum = (uint32_t)pow(2, (shift_accum - 1));
    den_p.shift_sq = shift_sq_tab[scale - 1];
    den_p.add_shift_sq = (uint32_t)1u << shift_sq_tab[scale - 1];
    den_p.shift_cub = shift_cub;
    den_p.add_shift_cub = (uint32_t)pow(2, (shift_cub - 1));

    cm_p.w = w;
    cm_p.h = h;
    cm_p.top = top;
    cm_p.bottom = bottom;
    cm_p.left = left;
    cm_p.right = right;
    cm_p.start_col = (left > 1) ? left : 1;
    cm_p.end_col = (right < (w - 1)) ? right : (w - 1);
    cm_p.start_row = (top > 1) ? top : 1;
    cm_p.end_row = (bottom < (h - 1)) ? bottom : (h - 1);
    cm_p.src_stride = stride;
    const uint32_t cm_shift_cub = (uint32_t)ceil(log2(w));
    const uint32_t cm_shift_inner_accum = (uint32_t)ceil(log2(h));
    for (int i = 0; i < 3; ++i) {
        cm_p.i_rfactor[i] = i_rfactor[i];
        cm_p.shift_sub[i] = 0;
        cm_p.shift_sq[i] = 30;
        cm_p.add_shift_sq[i] = 536870912; // 2^29
        cm_p.shift_cub[i] = cm_shift_cub;
        cm_p.add_shift_cub[i] = (uint32_t)pow(2, (cm_shift_cub - 1));
    }
    cm_p.shift_inner_accum = cm_shift_inner_accum;
    cm_p.add_shift_inner_accum = (uint32_t)pow(2, (cm_shift_inner_accum - 1));

    // consumes buf->i4_ref_dwt2 , buf->i4_dis_dwt2
    // produces buf->i4_decouple_r , buf->i4_decouple_a
    adm_decouple_s123_device(s, buf, w, h, stride, s->adm_enhn_gain_limit,
            cos_1deg_sq, cu_f, s->str);

    // consumes buf->i4_ref_dwt2
    // produces buf->adm_csf_den[scale]
    adm_csf_den_device(s, buf, scale, &den_p, cu_f, s->str);

    // consumes buf->i4_decouple_a
    // produces buf->i4_csf_a , buf->i4_csf_f
    adm_csf_device(s, buf, scale, w, h, stride, &csf_p, 0, cu_f, s->str);

    // consumes buf->i4_decouple_r, buf->i4_csf_a, buf->i4_csf_f
    // produces buf->adm_cm[scale]
    adm_cm_device(s, buf, scale, &cm_p, buf->adm_cm[scale], 0, cu_f, s->str);

    if (!s->adm_skip_aim) {
        // consumes buf->i4_decouple_r
        // produces buf->i4_csf_a , buf->i4_csf_f
        adm_csf_device(s, buf, scale, w, h, stride, &csf_p, 1, cu_f, s->str);

        // consumes buf->i4_decouple_a, buf->i4_csf_a, buf->i4_csf_f
        // produces buf->aim_num[scale]
        adm_cm_device(s, buf, scale, &cm_p, buf->aim_num[scale], 1, cu_f, s->str);
    }
}

static void integer_compute_adm_cuda(VmafFeatureExtractor *fex, AdmStateCuda *s,
        VmafPicture *ref_pic, VmafPicture *dis_pic, AdmBufferCuda *buf)
{
    CudaFunctions* cu_f = fex->cu_state->f;
    int w = ref_pic->w[0];
    int h = ref_pic->h[0];

    /*
     * the angle flag is evaluated in double precision by the reference
     * implementation; the float constant is computed on the host with the
     * exact same expression
     */
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);

    AdmFixedParametersCuda p = {
        cos_1deg_sq,
        {15826, 27411, 7345, -4240},
        {-4240, -7345, 27411, -15826},
        46342,
        0,
    };

    CHECK_CUDA(cu_f, cuMemsetD8Async(buf->tmp_res->data, 0,
                sizeof(int64_t) * RES_BUFFER_SIZE, s->str));

    size_t curr_ref_stride;
    size_t curr_dis_stride;
    size_t buf_stride = buf->ind_size_x >> 2;

    int32_t *i4_curr_ref_scale = NULL;
    int32_t *i4_curr_dis_scale = NULL;

    if (ref_pic->bpc == 8) {
        curr_ref_stride = ref_pic->stride[0];
        curr_dis_stride = dis_pic->stride[0];
    }
    else {
        curr_ref_stride = ref_pic->stride[0] >> 1;
        curr_dis_stride = dis_pic->stride[0] >> 1;
    }

    for (unsigned scale = 0; scale < 4; ++scale) {
        float rfactor[3];
        adm_csf_rfactor(scale, s->adm_csf_mode, s->adm_norm_view_dist,
                s->adm_ref_display_height, s->adm_csf_scale,
                s->adm_csf_diag_scale, rfactor);

        if (scale == 0) {
            // run these first dwt kernels on the input image stream to make sure it is consumed afterwards continue
            // consumes reference picture
            // produces buf->ref_dwt2, buf->dis_dwt2
            if (ref_pic->bpc == 8) {
                dwt2_8_device(s, (const uint8_t*)ref_pic->data[0], &buf->ref_dwt2, buf->i4_ref_dwt2, w, h, curr_ref_stride, buf_stride, &p, cu_f, vmaf_cuda_picture_get_stream(ref_pic));

                dwt2_8_device(s, (const uint8_t*)dis_pic->data[0], &buf->dis_dwt2, buf->i4_dis_dwt2, w, h, curr_dis_stride, buf_stride, &p, cu_f, vmaf_cuda_picture_get_stream(dis_pic));
            }
            else {
                dwt2_16_device(s, (const uint16_t*)ref_pic->data[0], &buf->ref_dwt2, buf->i4_ref_dwt2, w, h, curr_ref_stride, buf_stride, ref_pic->bpc, &p, cu_f, vmaf_cuda_picture_get_stream(ref_pic));

                dwt2_16_device(s, (const uint16_t*)dis_pic->data[0], &buf->dis_dwt2, buf->i4_dis_dwt2, w, h, curr_dis_stride, buf_stride, dis_pic->bpc, &p, cu_f, vmaf_cuda_picture_get_stream(dis_pic));
            }
            CHECK_CUDA(cu_f, cuEventRecord(s->ref_event,  vmaf_cuda_picture_get_stream(ref_pic)));
            CHECK_CUDA(cu_f, cuEventRecord(s->dis_event,  vmaf_cuda_picture_get_stream(dis_pic)));

            w = (w + 1) / 2;
            h = (h + 1) / 2;

            // This event ensures the input buffer is consumed
            CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));

            CHECK_CUDA(cu_f, cuStreamWaitEvent(s->str, s->dis_event, CU_EVENT_WAIT_DEFAULT));
            CHECK_CUDA(cu_f, cuStreamWaitEvent(s->str, s->ref_event, CU_EVENT_WAIT_DEFAULT));

            CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

            if (!s->adm_skip_scale0) {
                /*
                 * rfactor is converted to fixed-point for scale0 and stored
                 * in i_rfactor multiplied by 2^21 for rfactor[0,1] and by
                 * 2^23 for rfactor[2].
                 */
                uint16_t i_rfactor[3];
                adm_i_rfactor_scale0(rfactor, s->adm_norm_view_dist,
                        s->adm_ref_display_height, s->adm_csf_mode, i_rfactor);

                adm_compute_scale0_cuda(fex, s, buf, w, h, buf_stride,
                        cos_1deg_sq, i_rfactor);
            }
        }
        else {
            // consumes buf->i4_ref_dwt2.band_a , buf->i4_dis_dwt2.band_a
            // produces buf->i4_ref_dwt2.band_[ahvd] , buf->i4_dis_dwt2.band_[ahvd]
            // uses buf->tmp_ref
            adm_dwt2_s123_combined_device(s, i4_curr_ref_scale, (int32_t*)buf->tmp_ref->data, buf->i4_ref_dwt2, w, h,
                    curr_ref_stride, buf_stride, scale, &p, cu_f, s->str);
            adm_dwt2_s123_combined_device(s, i4_curr_dis_scale, (int32_t*)buf->tmp_dis->data, buf->i4_dis_dwt2, w, h,
                    curr_dis_stride, buf_stride, scale, &p, cu_f, s->str);

            w = (w + 1) / 2;
            h = (h + 1) / 2;

            adm_compute_scale_s123_cuda(fex, s, buf, scale, w, h, buf_stride,
                    cos_1deg_sq, rfactor);
        }

        i4_curr_ref_scale = buf->i4_ref_dwt2.band_a;
        i4_curr_dis_scale = buf->i4_dis_dwt2.band_a;

        curr_ref_stride = buf_stride;
        curr_dis_stride = buf_stride;
    }

    CHECK_CUDA(cu_f, cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(buf->results_host, buf->tmp_res->data, sizeof(int64_t) * RES_BUFFER_SIZE, s->str));
    CHECK_CUDA(cu_f, cuEventRecord(s->finished, s->str));
}

static CUdeviceptr init_dwt_band_cuda(struct VmafCudaState *cu_state,
        struct cuda_adm_dwt_band_t *band,
        CUdeviceptr data_top, size_t stride)
{
    (void)cu_state;
    band->band_a = (int16_t *)data_top; data_top += stride;
    band->band_h = (int16_t *)data_top; data_top += stride;
    band->band_v = (int16_t *)data_top; data_top += stride;
    band->band_d = (int16_t *)data_top; data_top += stride;
    return data_top;
}

static CUdeviceptr init_dwt_band_hvd_cuda(struct VmafCudaState *cu_state,
        struct cuda_adm_dwt_band_t *band,
        CUdeviceptr data_top, size_t stride)
{
    (void)cu_state;
    band->band_a = NULL;
    band->band_h = (int16_t *)data_top; data_top += stride;
    band->band_v = (int16_t *)data_top; data_top += stride;
    band->band_d = (int16_t *)data_top; data_top += stride;
    return data_top;
}

static CUdeviceptr i4_init_dwt_band_cuda(struct VmafCudaState *cu_state,
        struct cuda_i4_adm_dwt_band_t *band,
        CUdeviceptr data_top, size_t stride)
{
    (void)cu_state;
    band->band_a = (int32_t *)data_top; data_top += stride;
    band->band_h = (int32_t *)data_top; data_top += stride;
    band->band_v = (int32_t *)data_top; data_top += stride;
    band->band_d = (int32_t *)data_top; data_top += stride;
    return data_top;
}
static CUdeviceptr i4_init_dwt_band_hvd_cuda(struct VmafCudaState *cu_state,
        struct cuda_i4_adm_dwt_band_t *band,
        CUdeviceptr data_top, size_t stride)
{
    (void)cu_state;
    band->band_a = NULL;
    band->band_h = (int32_t *)data_top; data_top += stride;
    band->band_v = (int32_t *)data_top; data_top += stride;
    band->band_d = (int32_t *)data_top; data_top += stride;
    return data_top;
}

/*
 * adm_cm[4], adm_csf_den[4] and aim_num[4] of RES_BUFFER_SIZE int64_t, three
 * bands each.
 */
static CUdeviceptr init_res_cm_cuda(AdmBufferCuda *buf, CUdeviceptr data_top)
{
    for (int scale = 0; scale < 4; ++scale) {
        buf->adm_cm[scale] = (int64_t *)data_top;
        data_top += 3 * sizeof(int64_t);
    }
    return data_top;
}

static CUdeviceptr init_res_csf_cuda(AdmBufferCuda *buf, CUdeviceptr data_top)
{
    for (int scale = 0; scale < 4; ++scale) {
        buf->adm_csf_den[scale] = (uint64_t *)data_top;
        data_top += 3 * sizeof(uint64_t);
    }
    return data_top;
}

static CUdeviceptr init_res_aim_cuda(AdmBufferCuda *buf, CUdeviceptr data_top)
{
    for (int scale = 0; scale < 4; ++scale) {
        buf->aim_num[scale] = (int64_t *)data_top;
        data_top += 3 * sizeof(int64_t);
    }
    return data_top;
}

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
        unsigned bpc, unsigned w, unsigned h)
{
    AdmStateCuda *s = fex->priv;

    (void) pix_fmt;
    (void) bpc;
    int ret = 0;
    CudaFunctions* cu_f = fex->cu_state->f;
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&s->ref_event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&s->dis_event, CU_EVENT_DEFAULT));


    CUmodule adm_cm_module, adm_csf_den_module, adm_csf_module, adm_decouple_module, adm_dwt_module;


    CHECK_CUDA(cu_f, cuModuleLoadData(&adm_dwt_module, adm_dwt2_ptx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&adm_csf_module, adm_csf_ptx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&adm_decouple_module, adm_decouple_ptx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&adm_csf_den_module, adm_csf_den_ptx));
    CHECK_CUDA(cu_f, cuModuleLoadData(&adm_cm_module, adm_cm_ptx));

    // Get DWT kernel function pointers check adm_dwt2.cu for __global__ templated kernels
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_dwt_s123_combined_vert_kernel_0_0_int32_t,  adm_dwt_module, "dwt_s123_combined_vert_kernel_0_0_int32_t"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_dwt_s123_combined_vert_kernel_32768_16_int32_t, adm_dwt_module, "dwt_s123_combined_vert_kernel_32768_16_int32_t"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_dwt_s123_combined_hori_kernel_16384_15, adm_dwt_module, "dwt_s123_combined_hori_kernel_16384_15"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_dwt_s123_combined_hori_kernel_32768_16, adm_dwt_module, "dwt_s123_combined_hori_kernel_32768_16"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t, adm_dwt_module, "adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t, adm_dwt_module, "adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t"));


    // Get csf kernel function pointers check adm_csf.cu for __global__ templated kernels
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_adm_csf_kernel_1_4, adm_csf_module, "adm_csf_kernel_1_4"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_i4_adm_csf_kernel_1_4, adm_csf_module, "i4_adm_csf_kernel_1_4"));


    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_adm_decouple_kernel, adm_decouple_module, "adm_decouple_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_adm_decouple_s123_kernel, adm_decouple_module, "adm_decouple_s123_kernel"));


    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_adm_csf_den_scale_row_kernel, adm_csf_den_module, "adm_csf_den_scale_row_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_adm_csf_den_s123_row_kernel, adm_csf_den_module, "adm_csf_den_s123_row_kernel"));

    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_adm_cm_row_kernel, adm_cm_module, "adm_cm_row_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&s->func_i4_adm_cm_row_kernel, adm_cm_module, "i4_adm_cm_row_kernel"));


    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    s->integer_stride   = ALIGN_CEIL(w * sizeof(int32_t));
    s->buf.ind_size_x   = ALIGN_CEIL(((w + 1) / 2) * sizeof(int32_t));
    s->buf.ind_size_y   = ALIGN_CEIL(((h + 1) / 2) * sizeof(int32_t));
    size_t buf_sz_one   = s->buf.ind_size_x * ((h + 1) / 2);

    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.data_buf, buf_sz_one * NUM_BUFS_ADM);
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_ref, (s->integer_stride * 4 * ((h + 1) / 2)));
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_dis, (s->integer_stride * 4 * ((h + 1) / 2)));
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_res, sizeof(int64_t) * RES_BUFFER_SIZE);
    if (ret) goto free_ref;
    ret = vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->buf.results_host, sizeof(int64_t) * RES_BUFFER_SIZE);
    if (ret) goto free_ref;

    CUdeviceptr cu_res_top;
    ret = vmaf_cuda_buffer_get_dptr(s->buf.tmp_res, &cu_res_top);
    if (ret) goto free_ref;

    cu_res_top = init_res_cm_cuda(&s->buf, cu_res_top);
    cu_res_top = init_res_csf_cuda(&s->buf, cu_res_top);
    cu_res_top = init_res_aim_cuda(&s->buf, cu_res_top);

    CUdeviceptr cu_data_top;
    vmaf_cuda_buffer_get_dptr(s->buf.data_buf, &cu_data_top);

    cu_data_top = init_dwt_band_cuda(fex->cu_state, &s->buf.ref_dwt2, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_cuda(fex->cu_state, &s->buf.dis_dwt2, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.decouple_r, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.decouple_a, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.csf_a, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.csf_f, cu_data_top, buf_sz_one / 2);

    cu_data_top = i4_init_dwt_band_cuda(fex->cu_state, &s->buf.i4_ref_dwt2, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_cuda(fex->cu_state, &s->buf.i4_dis_dwt2, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_decouple_r, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_decouple_a, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_csf_a, cu_data_top, buf_sz_one);
    cu_data_top = i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_csf_f, cu_data_top, buf_sz_one);

    s->write_score_parameters = malloc(sizeof(write_score_parameters_adm));
    ((write_score_parameters_adm*)s->write_score_parameters)->s = s;


    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto free_ref;

    return 0;

free_ref:
    if (s->buf.data_buf) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.data_buf);
        free(s->buf.data_buf);
    }
    if (s->buf.tmp_ref) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_ref);
        free(s->buf.tmp_ref);
    }
    if (s->buf.tmp_dis) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_dis);
        free(s->buf.tmp_dis);
    }
    if (s->buf.tmp_res) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_res);
        free(s->buf.tmp_res);
    }
    if (s->buf.results_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->buf.results_host);
    }
    vmaf_dictionary_free(&s->feature_name_dict);

    return -ENOMEM;
}

static int extract_fex_cuda(VmafFeatureExtractor *fex,
        VmafPicture *ref_pic, VmafPicture *ref_pic_90,
        VmafPicture *dist_pic, VmafPicture *dist_pic_90,
        unsigned index, VmafFeatureCollector *feature_collector)
{
    (void) ref_pic_90;
    (void) dist_pic_90;

    AdmStateCuda *s = fex->priv;
    CudaFunctions* cu_f = fex->cu_state->f;

    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    // CHECK_CUDA(cu_f, cuEventSynchronize(s->finished));
    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    // current implementation is limited by the 16-bit data pipeline, thus
    // cannot handle an angular frequency smaller than 1080p * 3H
    if (s->adm_norm_view_dist * s->adm_ref_display_height <
            DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) {
        return -EINVAL;
    }

    integer_compute_adm_cuda(fex, s, ref_pic, dist_pic, &s->buf);

    write_score_parameters_adm *data = s->write_score_parameters;
    data->feature_collector = feature_collector;
    data->index = index;
    data->h = ref_pic->h[0];
    data->w = ref_pic->w[0];
    CHECK_CUDA(cu_f, cuStreamWaitEvent(s->host_stream, s->finished, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuLaunchHostFunc(s->host_stream, (CUhostFn*)write_scores, data));
    return 0;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    AdmStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    CHECK_CUDA(cu_f, cuEventDestroy(s->finished));
    CHECK_CUDA(cu_f, cuEventDestroy(s->ref_event));
    CHECK_CUDA(cu_f, cuEventDestroy(s->dis_event));

    int ret = 0;

    if (s->buf.data_buf) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.data_buf);
        free(s->buf.data_buf);
    }
    if (s->buf.tmp_ref) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_ref);
        free(s->buf.tmp_ref);
    }
    if (s->buf.tmp_dis) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_dis);
        free(s->buf.tmp_dis);
    }
    if (s->buf.tmp_res) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_res);
        free(s->buf.tmp_res);
    }
    if (s->buf.results_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->buf.results_host);
    }
    if (s->write_score_parameters)  free(s->write_score_parameters);

    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex,
        VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    AdmStateCuda *s = fex->priv;
    CudaFunctions* cu_f = fex->cu_state->f;
    CHECK_CUDA(cu_f, cuStreamSynchronize(s->str));
    return 1;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_adm2_score", "VMAF_integer_feature_aim_score", "VMAF_integer_feature_adm3_score",
    "integer_adm_scale0", "integer_adm_scale1", "integer_adm_scale2", "integer_adm_scale3",
    "integer_adm", "integer_adm_num", "integer_adm_den",
    "integer_adm_num_scale0", "integer_adm_den_scale0", "integer_adm_num_scale1",
    "integer_adm_den_scale1", "integer_adm_num_scale2", "integer_adm_den_scale2",
    "integer_adm_num_scale3", "integer_adm_den_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_adm_cuda = {
    .name = "adm_cuda",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .flush = flush_fex_cuda,
    .options = options_cuda,
    .close = close_fex_cuda,
    .priv_size = sizeof(AdmStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA
};
