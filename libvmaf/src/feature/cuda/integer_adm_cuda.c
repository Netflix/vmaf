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

#include "cuda_helper.cuh"
#include "mem.h"

#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"

#include "cpu.h"
#include "cuda/integer_adm_cuda.h"
/* DEFAULT_ADM_NOISE_WEIGHT / DEFAULT_ADM_CSF_SCALE / DEFAULT_ADM_CSF_DIAG_SCALE and
 * enum ADM_CSF_MODE are pulled in transitively via cuda/integer_adm_cuda.h →
 * feature/integer_adm.h. No separate adm_options.h include is needed here. */
#include "drain_batch.h"
#include "picture_cuda.h"

#include <assert.h>

#define RES_BUFFER_SIZE (4 * 3 * 2)

typedef struct WarpShift {
    uint32_t shift_cub[3];
    uint32_t add_shift_cub[3];
    uint32_t shift_sq[3];
    uint32_t add_shift_sq[3];
} WarpShift;

typedef struct AdmStateCuda {
    size_t integer_stride;
    AdmBufferCuda buf;
    bool debug;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    int adm_ref_display_height;
    double adm_csf_scale;
    double adm_csf_diag_scale;
    double adm_noise_weight;
    unsigned submit_w, submit_h; // stored by submit for collect
    void (*dwt2_8)(const uint8_t *src, const cuda_adm_dwt_band_t *dst, void *tmp_buf,
                   AdmBufferCuda *buf, int w, int h, int src_stride, int dst_stride,
                   CUstream c_stream);
    CUstream str;
    CUevent ref_event, dis_event, finished;
    /* Engine-scope fence batching opt-in flag (T-GPU-OPT-1, ADR-0242). */
    bool drained;
    VmafDictionary *feature_name_dict;

    // adm_dwt kernels
    CUfunction func_dwt_s123_combined_vert_kernel_0_0_int32_t,
        func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
        func_dwt_s123_combined_hori_kernel_16384_15, func_dwt_s123_combined_hori_kernel_32768_16,
        func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t,
        func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t, // untested
        // adm_csf kernel
        func_i4_adm_csf_kernel_1_4, func_adm_csf_kernel_1_4,
        // adm_csf_den kernel
        func_adm_csf_den_scale_line_kernel, func_adm_csf_den_s123_line_kernel,
        // adm_cm kernel
        func_adm_cm_reduce_line_kernel_4, func_adm_cm_line_kernel_8, func_i4_adm_cm_line_kernel;

} AdmStateCuda;

/*
 * lambda = 0 (finest scale), 1, 2, 3 (coarsest scale);
 * theta = 0 (ll), 1 (lh - vertical), 2 (hh - diagonal), 3(hl - horizontal).
 */
static inline float dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta,
                                   double adm_norm_view_dist, int adm_ref_display_height)
{
    // Formula (1), page 1165 - display visual resolution (DVR), in pixels/degree of visual angle. This should be 56.55
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;

    // Formula (9), page 1171
    float temp = log10(pow(2.0, lambda + 1) * params->f0 * params->g[theta] / r);
    float Q = 2.0 * params->a * pow(10.0, params->k * temp * temp) /
              dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

int dwt2_8_device(AdmStateCuda *s, const uint8_t *d_picture, cuda_adm_dwt_band_t *d_dst,
                  cuda_i4_adm_dwt_band_t i4_dwt_dst, int w, int h, int src_stride, int dst_stride,
                  AdmFixedParametersCuda *p, CudaFunctions *cu_f, CUstream c_stream)
{
    int rows_per_thread = 4;

    int vert_out_tile_rows = 8;
    int vert_out_tile_cols = 128;

    int horz_out_tile_rows = vert_out_tile_rows;
    int horz_out_tile_cols = vert_out_tile_cols / 2 - 2;

    int16_t v_shift = 8;
    int32_t v_add_shift = 1 << (v_shift - 1);

    void *args[] = {&d_picture,  &*d_dst,     &i4_dwt_dst, &w,           &h,
                    &src_stride, &dst_stride, &v_shift,    &v_add_shift, &*p};
    CHECK_CUDA_RETURN(
        cu_f, cuLaunchKernel(s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t,
                             DIV_ROUND_UP((w + 1) / 2, horz_out_tile_cols),
                             DIV_ROUND_UP((h + 1) / 2, horz_out_tile_rows), 1, vert_out_tile_cols,
                             vert_out_tile_rows / rows_per_thread, 1, 0, c_stream, args, NULL));
    return 0;
}

int adm_dwt2_s123_combined_device(AdmStateCuda *s, const int32_t *d_i4_scale, int32_t *tmp_buf,
                                  cuda_i4_adm_dwt_band_t i4_dwt, int w, int h, int img_stride,
                                  int dst_stride, int scale, AdmFixedParametersCuda *p,
                                  CudaFunctions *cu_f, CUstream cu_stream)
{
    const int BLOCK_Y = (h + 1) / 2;

    void *args_vert[] = {&d_i4_scale, &tmp_buf, &w, &h, &img_stride, &*p};
    switch (scale) {
    case 1:
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_vert_kernel_0_0_int32_t,
                                               DIV_ROUND_UP(w, 128), BLOCK_Y, 1, 128, 1, 1, 0,
                                               cu_stream, args_vert, NULL));
        break;
    case 2:
        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
                                         DIV_ROUND_UP(w, 128), BLOCK_Y, 1, 128, 1, 1, 0, cu_stream,
                                         args_vert, NULL));
        break;
    case 3:
        CHECK_CUDA_RETURN(cu_f,
                          cuLaunchKernel(s->func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
                                         DIV_ROUND_UP(w, 128), BLOCK_Y, 1, 128, 1, 1, 0, cu_stream,
                                         args_vert, NULL));
        break;
    default:
        break; /* scale 0 is handled in the caller's if/else branch; no vert kernel for it */
    }

    void *args_hori[] = {&i4_dwt, &tmp_buf, &w, &h, &dst_stride, &*p};
    switch (scale) {
    case 1:
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_hori_kernel_16384_15,
                                               DIV_ROUND_UP(((w + 1) / 2), 128), BLOCK_Y, 1, 128, 1,
                                               1, 0, cu_stream, args_hori, NULL));
        break;
    case 2:
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_hori_kernel_32768_16,
                                               DIV_ROUND_UP(((w + 1) / 2), 128), BLOCK_Y, 1, 128, 1,
                                               1, 0, cu_stream, args_hori, NULL));
        break;
    case 3:
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_dwt_s123_combined_hori_kernel_16384_15,
                                               DIV_ROUND_UP(((w + 1) / 2), 128), BLOCK_Y, 1, 128, 1,
                                               1, 0, cu_stream, args_hori, NULL));
        break;
    default:
        break; /* scale 0 is handled in the caller's if/else branch; no hori kernel for it */
    }
    return 0;
}

int adm_dwt2_16_device(AdmStateCuda *s, const uint16_t *d_picture, cuda_adm_dwt_band_t *d_dst,
                       cuda_i4_adm_dwt_band_t i4_dwt_dst, int w, int h, int src_stride,
                       int dst_stride, int inp_size_bits, AdmFixedParametersCuda *p,
                       CudaFunctions *cu_f, CUstream c_stream)
{
    int rows_per_thread = 4;

    int vert_out_tile_rows = 8;
    int vert_out_tile_cols = 128;

    int horz_out_tile_rows = vert_out_tile_rows;
    int horz_out_tile_cols = vert_out_tile_cols / 2 - 2;

    int16_t v_shift = inp_size_bits;
    int32_t v_add_shift = 1 << (inp_size_bits - 1);

    void *args[] = {&d_picture,  &*d_dst,     &i4_dwt_dst, &w,           &h,
                    &src_stride, &dst_stride, &v_shift,    &v_add_shift, &*p};
    CHECK_CUDA_RETURN(
        cu_f, cuLaunchKernel(s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t,
                             DIV_ROUND_UP((w + 1) / 2, horz_out_tile_cols),
                             DIV_ROUND_UP((h + 1) / 2, horz_out_tile_rows), 1, vert_out_tile_cols,
                             vert_out_tile_rows / rows_per_thread, 1, 0, c_stream, args, NULL));
    return 0;
}

int adm_csf_device(AdmStateCuda *s, AdmBufferCuda *buf, int w, int h, int stride,
                   AdmFixedParametersCuda *p, CudaFunctions *cu_f, CUstream c_stream)
{
    // ensure that csf_f pointers are aligned to 16 bytes for vectorized memory access
    for (int band = 0; band < 3; ++band) {
        assert(((size_t)(buf->csf_f.bands[band]) & 15) == 0);
    }

    // ensure that the stride is a multiple of 4 so that each row starts 16 byte aligned.
    assert(stride % 4 == 0);

    /* The computation of the score is not required for the regions
       which lie outside the frame borders */
    int left = w * (float)(ADM_BORDER_FACTOR)-0.5f - 1; // -1 for filter tap
    int top = h * (float)(ADM_BORDER_FACTOR)-0.5f - 1;
    int right = w - left + 2; // +2 for filter tap
    int bottom = h - top + 2;

    if (left < 0) {
        left = 0;
    }
    if (right > w) {
        right = w;
    }
    if (top < 0) {
        top = 0;
    }
    if (bottom > h) {
        bottom = h;
    }

    // align left side to ensure that all memory accesses start at a multiple of 16 bytes.
    // this will do a little bit more work than originally requested, though the result is unchanged.
    left = left & ~3;

    const int cols_per_thread = 4;
    const int rows_per_thread = 1;
    const int BLOCKX = 32, BLOCKY = 4;

    void *args[] = {&*buf, &top, &bottom, &left, &right, &stride, &*p};
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_adm_csf_kernel_1_4,
                                           DIV_ROUND_UP(right - left, BLOCKX * cols_per_thread),
                                           DIV_ROUND_UP(bottom - top, BLOCKY * rows_per_thread), 3,
                                           BLOCKX, BLOCKY, 1, 0, c_stream, args, NULL));
    return 0;
}

int i4_adm_csf_device(AdmStateCuda *s, AdmBufferCuda *buf, int scale, int w, int h, int stride,
                      AdmFixedParametersCuda *p, CudaFunctions *cu_f, CUstream c_stream)
{
    // ensure that csf_f pointers are aligned to 16 bytes for vectorized memory access
    for (int band = 0; band < 3; ++band) {
        assert(((size_t)(buf->i4_csf_f.bands[band]) & 15) == 0);
    }

    // ensure that the stride is a multiple of 4 so that each row starts 16 byte aligned.
    assert(stride % 4 == 0);

    /* The computation of the score is not required for the regions
       which lie outside the frame borders */
    int left = w * (float)(ADM_BORDER_FACTOR)-0.5f - 1; // -1 for filter tap
    int top = h * (float)(ADM_BORDER_FACTOR)-0.5f - 1;
    int right = w - left + 2; // +2 for filter tap
    int bottom = h - top + 2;

    if (left < 0) {
        left = 0;
    }
    if (right > w) {
        right = w;
    }
    if (top < 0) {
        top = 0;
    }
    if (bottom > h) {
        bottom = h;
    }

    // align left side to ensure that all memory accesses start at a multiple of 16 bytes.
    // this will do a little bit more work than originally requested, though the result is unchanged.
    left = left & ~3;

    const int cols_per_thread = 4;
    const int rows_per_thread = 1;
    const int BLOCKX = 32, BLOCKY = 4;

    void *args[] = {&*buf, &scale, &top, &bottom, &left, &right, &stride, &*p};
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_i4_adm_csf_kernel_1_4,
                                           DIV_ROUND_UP(right - left, BLOCKX * cols_per_thread),
                                           DIV_ROUND_UP(bottom - top, BLOCKY * rows_per_thread), 3,
                                           BLOCKX, BLOCKY, 1, 0, c_stream, args, NULL));
    return 0;
}

int adm_csf_den_s123_device(AdmStateCuda *s, AdmBufferCuda *buf, int scale, int w, int h,
                            int src_stride, double adm_norm_view_dist, int adm_ref_display_height,
                            CudaFunctions *cu_f, CUstream c_stream)
{
    /* The computation of the denominator scales is not required for the regions
     * which lie outside the frame borders
     */

    int left = w * (float)(ADM_BORDER_FACTOR)-0.5f;
    int top = h * (float)(ADM_BORDER_FACTOR)-0.5f;
    int right = w - left;
    int bottom = h - top;

    int buffer_stride = right - left;
    int buffer_h = bottom - top;

    int val_per_thread = 8;
    int warps_per_cta = 4;
    int BLOCKX = VMAF_CUDA_THREADS_PER_WARP * warps_per_cta;

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
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_adm_csf_den_s123_line_kernel,
                                           DIV_ROUND_UP(buffer_stride, BLOCKX * val_per_thread),
                                           buffer_h, 3, BLOCKX, 1, 1, 0, c_stream, args, NULL));
    return 0;
}

int adm_csf_den_scale_device(AdmStateCuda *s, AdmBufferCuda *buf, int w, int h, int src_stride,
                             double adm_norm_view_dist, int adm_ref_display_height,
                             CudaFunctions *cu_f, CUstream c_stream)
{
    /* The computation of the denominator scales is not required for the regions
     * which lie outside the frame borders
     */
    int scale = 0;
    int left = w * (float)(ADM_BORDER_FACTOR)-0.5f;
    int top = h * (float)(ADM_BORDER_FACTOR)-0.5f;
    int right = w - left;
    int bottom = h - top;

    int buffer_stride = right - left;
    int buffer_h = bottom - top;

    int val_per_thread = 8;
    int warps_per_cta = 4;

    const int BLOCKX = VMAF_CUDA_THREADS_PER_WARP * warps_per_cta;

    void *args[] = {&buf->ref_dwt2, &h,     &top,        &bottom,
                    &left,          &right, &src_stride, &buf->adm_csf_den[scale]};
    CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_adm_csf_den_scale_line_kernel,
                                           DIV_ROUND_UP(buffer_stride, BLOCKX * val_per_thread),
                                           buffer_h, 3, BLOCKX, 1, 1, 0, c_stream, args, NULL));
    return 0;
}

int i4_adm_cm_device(AdmStateCuda *s, AdmBufferCuda *buf, int w, int h, int src_stride,
                     int csf_a_stride, int scale, AdmFixedParametersCuda *p, CudaFunctions *cu_f,
                     CUstream c_stream)
{

    int left = w * (float)(ADM_BORDER_FACTOR)-0.5f;
    int top = h * (float)(ADM_BORDER_FACTOR)-0.5f;
    int right = w - left;
    int bottom = h - top;

    int start_col = (left > 1) ? left : ((left <= 0) ? 0 : 1);
    int end_col = (right < (w - 1)) ? right : ((right > (w - 1)) ? w : w - 1);
    int start_row = (top > 1) ? top : ((top <= 0) ? 0 : 1);
    int end_row = (bottom < (h - 1)) ? bottom : ((bottom > (h - 1)) ? h : h - 1);

    int buffer_stride = end_col - start_col;
    int buffer_h = end_row - start_row;

    {
        const int BLOCKX = 128;

        void *args[] = {&*buf,
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
                        &scale,
                        &buffer_h,
                        &buffer_stride,
                        &buf->tmp_accum->data,
                        &*p};
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_i4_adm_cm_line_kernel,
                                               DIV_ROUND_UP(buffer_stride, BLOCKX), buffer_h, 3,
                                               BLOCKX, 1, 1, 0, c_stream, args, NULL));
    }
    {
        const int val_per_thread = 4;
        const int warps_per_cta = 4;
        const int BLOCKX = VMAF_CUDA_THREADS_PER_WARP * warps_per_cta;

        void *args[] = {
            &h, &w, &scale, &buffer_h, &buffer_stride, &buf->tmp_accum->data, &buf->adm_cm[scale]};
        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_adm_cm_reduce_line_kernel_4,
                                               DIV_ROUND_UP(buffer_stride, BLOCKX * val_per_thread),
                                               buffer_h, 3, BLOCKX, 1, 1, 0, c_stream, args, NULL));
    }
    return 0;
}

int adm_cm_device(AdmStateCuda *s, AdmBufferCuda *buf, int w, int h, int src_stride,
                  int csf_a_stride, AdmFixedParametersCuda *p, CudaFunctions *cu_f,
                  CUstream c_stream)
{

    int scale = 0;
    int left = w * (float)(ADM_BORDER_FACTOR)-0.5f;
    int top = h * (float)(ADM_BORDER_FACTOR)-0.5f;
    int right = w - left;
    int bottom = h - top;

    int start_col = MAX(0, left);
    int end_col = MIN(right, w);
    int start_row = MAX(0, top);
    int end_row = MIN(bottom, h);

    int buffer_stride = end_col - start_col;
    int buffer_h = end_row - start_row;

    // precompute warp shift per band
    //const int32_t shift_sub[3] = {10, 10, 12};
    const int fixed_shift[3] = {4, 4, 3};

    // accumulation
    const int32_t shift_xsq[3] = {29, 29, 30};
    const int32_t add_shift_xsq[3] = {268435456, 268435456, 536870912};

    const int NUM_BANDS = 3;
    WarpShift ws;
    for (int band = 0; band < NUM_BANDS; ++band) {
        ws.shift_cub[band] = (uint32_t)(ceil(log2f(w)));
        if (scale == 0) {
            ws.shift_cub[band] -= fixed_shift[band];
            ws.shift_sq[band] = shift_xsq[band];
            ws.add_shift_sq[band] = add_shift_xsq[band];
        } else {
            ws.shift_sq[band] = 30;
            ws.add_shift_sq[band] = (1 << (ws.shift_sq[band] - 1));
        }
        ws.add_shift_cub[band] = 1 << (ws.shift_cub[band] - 1);
    }

    // precompute global shift
    uint32_t shift_inner_accum = (uint32_t)(ceil(log2f(h)));
    uint32_t add_shift_inner_accum = 1 << (shift_inner_accum - 1);

    // fused
    {
        const int rows_per_thread = 8;
        const int BLOCKX = 32, BLOCKY = 4;

        void *args[] = {&*buf,
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
                        &buf->tmp_accum->data,
                        &*p,
                        &scale,
                        &buf->adm_cm[scale],
                        &ws,
                        &shift_inner_accum,
                        &add_shift_inner_accum};

        CHECK_CUDA_RETURN(cu_f, cuLaunchKernel(s->func_adm_cm_line_kernel_8,
                                               DIV_ROUND_UP(buffer_stride, BLOCKX),
                                               DIV_ROUND_UP(buffer_h, BLOCKY * rows_per_thread), 3,
                                               BLOCKX, BLOCKY, 1, 0, c_stream, args, NULL));
    }
    return 0;
}

static void conclude_adm_cm(int64_t *accum, int h, int w, int scale, float noise_weight,
                            float *result)
{
    int left = w * ADM_BORDER_FACTOR - 0.5;
    int top = h * ADM_BORDER_FACTOR - 0.5;
    int right = w - left;
    int bottom = h - top;
    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));

    // scale 0
    const uint32_t shift_xcub[3] = {(uint32_t)ceil(log2(w) - 4), (uint32_t)ceil(log2(w) - 4),
                                    (uint32_t)ceil(log2(w) - 3)};
    int constant_offset[3] = {52, 52, 57};

    // scale 123
    uint32_t shift_cub = (uint32_t)ceil(log2(w));
    float final_shift[3] = {powf(2, (45 - shift_cub - shift_inner_accum)),
                            powf(2, (39 - shift_cub - shift_inner_accum)),
                            powf(2, (36 - shift_cub - shift_inner_accum))};
    float powf_add = powf((float)((bottom - top) * (right - left)) * noise_weight, 1.0f / 3.0f);

    float f_accum;
    *result = 0;
    for (int i = 0; i < 3; ++i) {
        if (scale == 0) {
            f_accum = (float)(accum[i] /
                              pow(2, (constant_offset[i] - shift_xcub[i] - shift_inner_accum)));
        } else {
            f_accum = (float)(accum[i] / final_shift[scale - 1]);
        }
        *result += powf(f_accum, 1.0f / 3.0f) + powf_add;
    }
}

static void conclude_adm_csf_den(uint64_t *accum, int h, int w, int scale, float *result,
                                 float adm_norm_view_dist, float adm_ref_display_height,
                                 float adm_csf_scale, float adm_csf_diag_scale, float noise_weight)
{
    const int left = w * ADM_BORDER_FACTOR - 0.5;
    const int top = h * ADM_BORDER_FACTOR - 0.5;
    const int right = w - left;
    const int bottom = h - top;
    const float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist,
                                         adm_ref_display_height);
    const float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist,
                                         adm_ref_display_height);
    /* Apply CSF scale multipliers: rfactor = scale / quant_step.
     * Default adm_csf_scale = adm_csf_diag_scale = 1.0 → no change. */
    const float rfactor[3] = {adm_csf_scale / factor1, adm_csf_scale / factor1,
                              adm_csf_diag_scale / factor2};
    const uint32_t accum_convert_float[4] = {18, 32, 27, 23};

    int32_t shift_accum;
    double shift_csf;
    if (scale == 0) {
        shift_accum = (int32_t)ceil(log2((bottom - top) * (right - left)) - 20);
        shift_accum = shift_accum > 0 ? shift_accum : 0;
        shift_csf = pow(2, (accum_convert_float[scale] - shift_accum));
    } else {
        shift_accum = (int32_t)ceil(log2(bottom - top));
        const uint32_t shift_cub = (uint32_t)ceil(log2(right - left));
        shift_csf = pow(2, (accum_convert_float[scale] - shift_accum - shift_cub));
    }
    const float powf_add =
        powf((float)((bottom - top) * (right - left)) * noise_weight, 1.0f / 3.0f);

    *result = 0;
    for (int i = 0; i < 3; ++i) {
        const double csf = (double)(accum[i] / shift_csf) * pow(rfactor[i], 3);
        *result += powf(csf, 1.0f / 3.0f) + powf_add;
    }
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
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmStateCuda, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
    },
    {
        .name = "adm_ref_display_height",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmStateCuda, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
    },
    {
        .name = "adm_csf_scale",
        .alias = "cs",
        .help = "CSF band-scale multiplier for h/v bands (default 1.0 = no scaling)",
        .offset = offsetof(AdmStateCuda, adm_csf_scale),
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
        .offset = offsetof(AdmStateCuda, adm_csf_diag_scale),
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
        .offset = offsetof(AdmStateCuda, adm_noise_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 100.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {0}};

typedef struct write_score_parameters_adm {
    VmafFeatureCollector *feature_collector;
    AdmStateCuda *s;
    unsigned index, h, w;
} write_score_parameters_adm;

static void write_scores(write_score_parameters_adm *params)
{
    VmafFeatureCollector *feature_collector = params->feature_collector;
    AdmStateCuda *s = params->s;
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

        conclude_adm_cm(&adm_cm[scale * 3], h, w, scale, (float)s->adm_noise_weight, &num_scale);
        conclude_adm_csf_den(&adm_csf[scale * 3], h, w, scale, &den_scale, s->adm_norm_view_dist,
                             s->adm_ref_display_height, (float)s->adm_csf_scale,
                             (float)s->adm_csf_diag_scale, (float)s->adm_noise_weight);

        num += num_scale;
        den += den_scale;

        scores[2 * scale + 0] = num_scale;
        scores[2 * scale + 1] = den_scale;
    }
    const double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);

    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;

    if (den == 0.0) {
        score = 1.0f;
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

    if (!s->debug)
        return;

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
    (void)err; // accumulated collector status intentionally discarded; void writer API

    return;
}

static int integer_compute_adm_cuda(VmafFeatureExtractor *fex, AdmStateCuda *s,
                                    VmafPicture *ref_pic, VmafPicture *dis_pic, AdmBufferCuda *buf,
                                    double adm_enhn_gain_limit, double adm_norm_view_dist,
                                    int adm_ref_display_height)
{
    CudaFunctions *cu_f = fex->cu_state->f;
    int w = ref_pic->w[0];
    int h = ref_pic->h[0];

    AdmFixedParametersCuda p = {
        .dwt2_db2_coeffs_lo = {15826, 27411, 7345, -4240},
        .dwt2_db2_coeffs_hi = {-4240, -7345, 27411, -15826},
        .dwt2_db2_coeffs_lo_sum = 46342,
        .dwt2_db2_coeffs_hi_sum = 0,
        .log2_w = log2(w),
        .log2_h = log2(h),
        .adm_ref_display_height = adm_ref_display_height,
        .adm_norm_view_dist = adm_norm_view_dist,
        .adm_enhn_gain_limit = adm_enhn_gain_limit,
    };

    const double pow2_32 = (double)(1ULL << 32);
    const double pow2_21 = (double)(1ULL << 21);
    const double pow2_23 = (double)(1ULL << 23);
    for (unsigned scale = 0; scale < 4; ++scale) {
        float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, adm_norm_view_dist,
                                       adm_ref_display_height);
        float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, adm_norm_view_dist,
                                       adm_ref_display_height);
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
    CHECK_CUDA_RETURN(
        cu_f, cuMemsetD8Async(buf->tmp_res->data, 0, sizeof(int64_t) * RES_BUFFER_SIZE, s->str));

    size_t curr_ref_stride;
    size_t curr_dis_stride;
    size_t buf_stride = buf->ind_size_x >> 2;

    int32_t *i4_curr_ref_scale = NULL;
    int32_t *i4_curr_dis_scale = NULL;

    if (ref_pic->bpc == 8) {
        curr_ref_stride = ref_pic->stride[0];
        curr_dis_stride = dis_pic->stride[0];
    } else {
        curr_ref_stride = dis_pic->stride[0] >> 1;
        curr_dis_stride = ref_pic->stride[0] >> 1;
    }

    int err = 0;
    for (unsigned scale = 0; scale < 4; ++scale) {
        if (scale == 0) {
            // run these first dwt kernels on the input iamge stream to make sure it is consumed afterwards continue
            // consumes reference picture
            // produces buf->ref_dwt2, buf->dis_dwt2
            if (ref_pic->bpc == 8) {
                err = dwt2_8_device(s, (const uint8_t *)ref_pic->data[0], &buf->ref_dwt2,
                                    buf->i4_ref_dwt2, w, h, curr_ref_stride, buf_stride, &p, cu_f,
                                    vmaf_cuda_picture_get_stream(ref_pic));
                if (err)
                    return err;

                err = dwt2_8_device(s, (const uint8_t *)dis_pic->data[0], &buf->dis_dwt2,
                                    buf->i4_dis_dwt2, w, h, curr_dis_stride, buf_stride, &p, cu_f,
                                    vmaf_cuda_picture_get_stream(dis_pic));
                if (err)
                    return err;
            } else {
                err = adm_dwt2_16_device(s, (uint16_t *)ref_pic->data[0], &buf->ref_dwt2,
                                         buf->i4_ref_dwt2, w, h, curr_ref_stride, buf_stride,
                                         ref_pic->bpc, &p, cu_f,
                                         vmaf_cuda_picture_get_stream(ref_pic));
                if (err)
                    return err;

                err = adm_dwt2_16_device(s, (uint16_t *)dis_pic->data[0], &buf->dis_dwt2,
                                         buf->i4_dis_dwt2, w, h, curr_dis_stride, buf_stride,
                                         dis_pic->bpc, &p, cu_f,
                                         vmaf_cuda_picture_get_stream(dis_pic));
                if (err)
                    return err;
            }
            CHECK_CUDA_RETURN(cu_f,
                              cuEventRecord(s->ref_event, vmaf_cuda_picture_get_stream(ref_pic)));
            CHECK_CUDA_RETURN(cu_f,
                              cuEventRecord(s->dis_event, vmaf_cuda_picture_get_stream(dis_pic)));

            w = (w + 1) / 2;
            h = (h + 1) / 2;

            /* Push the fex context briefly to sync the local str stream
             * with the per-picture events. Use goto-style unwind so the
             * pop runs even if a wait fails mid-sequence. */
            int _cuda_err = 0;
            int ctx_pushed = 0;
            CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), push_fail);
            ctx_pushed = 1;
            CHECK_CUDA_GOTO(cu_f, cuStreamWaitEvent(s->str, s->dis_event, CU_EVENT_WAIT_DEFAULT),
                            push_fail);
            CHECK_CUDA_GOTO(cu_f, cuStreamWaitEvent(s->str, s->ref_event, CU_EVENT_WAIT_DEFAULT),
                            push_fail);
            CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), after_pop);
            goto push_ok;
        push_fail:
            if (ctx_pushed)
                (void)cu_f->cuCtxPopCurrent(NULL);
        after_pop:
            return _cuda_err;
        push_ok:;

            // consumes buf->ref_dwt2
            // produces buf->adm_csf_den[0]
            err = adm_csf_den_scale_device(s, buf, w, h, buf_stride, adm_norm_view_dist,
                                           adm_ref_display_height, cu_f, s->str);
            if (err)
                return err;

            // consumes buf->ref_dwt2 , buf->dis_dwt2 (inline decouple)
            // produces buf->csf_f
            err = adm_csf_device(s, buf, w, h, buf_stride, &p, cu_f, s->str);
            if (err)
                return err;

            // consumes buf->ref_dwt2, buf->dis_dwt2, buf->csf_f (inline decouple + csf_a)
            // produces buf->adm_cm[0]
            err = adm_cm_device(s, buf, w, h, buf_stride, buf_stride, &p, cu_f, s->str);
            if (err)
                return err;
        } else {
            // consumes buf->i4_ref_dwt2.band_a , buf->i4_dis_dwt2.band_a
            // produces buf->i4_ref_dwt2.band_[ahvd] , buf->i4_dis_dwt2.band_[ahvd]
            // uses buf->tmp_ref
            // CUDA device pointers arrive as CUdeviceptr (unsigned long long) and
            // must be cast to host-visible pointer types to populate kernel-args
            // structs / layout helpers for cuLaunchKernel. The cast is inherent to
            // the CUDA Driver API and cannot be refactored away without changing
            // the public libvmaf-CUDA contract. Per ADR-0141 touched-file rule,
            // upstream-parity exception.
            // NOLINTBEGIN(performance-no-int-to-ptr)
            err = adm_dwt2_s123_combined_device(s, i4_curr_ref_scale, (int32_t *)buf->tmp_ref->data,
                                                buf->i4_ref_dwt2, w, h, curr_ref_stride, buf_stride,
                                                scale, &p, cu_f, s->str);
            if (err)
                return err;
            err = adm_dwt2_s123_combined_device(s, i4_curr_dis_scale, (int32_t *)buf->tmp_dis->data,
                                                buf->i4_dis_dwt2, w, h, curr_dis_stride, buf_stride,
                                                scale, &p, cu_f, s->str);
            if (err)
                return err;

            w = (w + 1) / 2;
            h = (h + 1) / 2;

            // consumes buf->i4_ref_dwt2
            // produces buf->adm_csf_den[1,2,3]
            err = adm_csf_den_s123_device(s, buf, scale, w, h, buf_stride, adm_norm_view_dist,
                                          adm_ref_display_height, cu_f, s->str);
            if (err)
                return err;

            // consumes buf->i4_ref_dwt2 , buf->i4_dis_dwt2 (inline decouple)
            // produces buf->i4_csf_f
            err = i4_adm_csf_device(s, buf, scale, w, h, buf_stride, &p, cu_f, s->str);
            if (err)
                return err;

            // consumes buf->i4_ref_dwt2, buf->i4_dis_dwt2, buf->i4_csf_f (inline decouple + csf_a)
            // produces buf->adm_cm[1,2,3]
            err = i4_adm_cm_device(s, buf, w, h, buf_stride, buf_stride, scale, &p, cu_f, s->str);
            if (err)
                return err;
        }

        i4_curr_ref_scale = buf->i4_ref_dwt2.band_a;
        i4_curr_dis_scale = buf->i4_dis_dwt2.band_a;

        curr_ref_stride = buf_stride;
        curr_dis_stride = buf_stride;
    }
    CHECK_CUDA_RETURN(cu_f, cuMemcpyDtoHAsync(buf->results_host, buf->tmp_res->data,
                                              sizeof(int64_t) * RES_BUFFER_SIZE, s->str));
    CHECK_CUDA_RETURN(cu_f, cuEventRecord(s->finished, s->str));
    /* Engine-scope fence batching opt-in (T-GPU-OPT-1, ADR-0242).
     * Best-effort: registration failure (overflow / no batch open)
     * silently degrades to the per-stream sync in collect(). */
    (void)vmaf_cuda_drain_batch_register_event(s->finished, &s->drained);
    return 0;
}

static CUdeviceptr init_dwt_band_cuda(struct VmafCudaState *cu_state,
                                      struct cuda_adm_dwt_band_t *band, CUdeviceptr data_top,
                                      size_t stride)
{
    (void)cu_state;
    band->band_a = (int16_t *)data_top;
    data_top += stride;
    band->band_h = (int16_t *)data_top;
    data_top += stride;
    band->band_v = (int16_t *)data_top;
    data_top += stride;
    band->band_d = (int16_t *)data_top;
    data_top += stride;
    return data_top;
}

static CUdeviceptr init_dwt_band_hvd_cuda(struct VmafCudaState *cu_state,
                                          struct cuda_adm_dwt_band_t *band, CUdeviceptr data_top,
                                          size_t stride)
{
    (void)cu_state;
    band->band_a = NULL;
    band->band_h = (int16_t *)data_top;
    data_top += stride;
    band->band_v = (int16_t *)data_top;
    data_top += stride;
    band->band_d = (int16_t *)data_top;
    data_top += stride;
    return data_top;
}

static CUdeviceptr i4_init_dwt_band_cuda(struct VmafCudaState *cu_state,
                                         struct cuda_i4_adm_dwt_band_t *band, CUdeviceptr data_top,
                                         size_t stride)
{
    (void)cu_state;
    band->band_a = (int32_t *)data_top;
    data_top += stride;
    band->band_h = (int32_t *)data_top;
    data_top += stride;
    band->band_v = (int32_t *)data_top;
    data_top += stride;
    band->band_d = (int32_t *)data_top;
    data_top += stride;
    return data_top;
}
static CUdeviceptr i4_init_dwt_band_hvd_cuda(struct VmafCudaState *cu_state,
                                             struct cuda_i4_adm_dwt_band_t *band,
                                             CUdeviceptr data_top, size_t stride)
{
    (void)cu_state;
    band->band_a = NULL;
    band->band_h = (int32_t *)data_top;
    data_top += stride;
    band->band_v = (int32_t *)data_top;
    data_top += stride;
    band->band_d = (int32_t *)data_top;
    data_top += stride;
    return data_top;
}

static CUdeviceptr init_res_cm_cuda(struct VmafCudaState *cu_state, int64_t *scale_pointer[],
                                    CUdeviceptr data_top)
{
    (void)cu_state;
    const int stride = 3 * sizeof(int64_t);
    scale_pointer[0] = (int64_t *)data_top;
    data_top += stride;
    scale_pointer[1] = (int64_t *)data_top;
    data_top += stride;
    scale_pointer[2] = (int64_t *)data_top;
    data_top += stride;
    scale_pointer[3] = (int64_t *)data_top;
    data_top += stride;
    return data_top;
}

static CUdeviceptr init_res_csf_cuda(struct VmafCudaState *cu_state, uint64_t *scale_pointer[],
                                     CUdeviceptr data_top)
{
    (void)cu_state;
    const int stride = 3 * sizeof(uint64_t);
    scale_pointer[0] = (uint64_t *)data_top;
    data_top += stride;
    scale_pointer[1] = (uint64_t *)data_top;
    data_top += stride;
    scale_pointer[2] = (uint64_t *)data_top;
    data_top += stride;
    scale_pointer[3] = (uint64_t *)data_top;
    data_top += stride;
    return data_top;
}

static inline CUdeviceptr init_index_cuda(int32_t **index, CUdeviceptr data_top, size_t stride)
{
    index[0] = (int32_t *)data_top;
    data_top += stride;
    index[1] = (int32_t *)data_top;
    data_top += stride;
    index[2] = (int32_t *)data_top;
    data_top += stride;
    index[3] = (int32_t *)data_top;
    data_top += stride;
    return data_top;
}
// NOLINTEND(performance-no-int-to-ptr)

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    AdmStateCuda *s = fex->priv;

    (void)pix_fmt;
    (void)bpc;
    int ret = 0;
    CudaFunctions *cu_f = fex->cu_state->f;
    int _cuda_err = 0;
    int ctx_pushed = 0;
    CHECK_CUDA_GOTO(cu_f, cuCtxPushCurrent(fex->cu_state->ctx), fail);
    ctx_pushed = 1;
    CHECK_CUDA_GOTO(cu_f, cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->finished, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->ref_event, CU_EVENT_DEFAULT), fail);
    CHECK_CUDA_GOTO(cu_f, cuEventCreate(&s->dis_event, CU_EVENT_DEFAULT), fail);

    CUmodule adm_cm_module, adm_csf_den_module, adm_csf_module, adm_dwt_module;

    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&adm_dwt_module, adm_dwt2_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&adm_csf_module, adm_csf_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&adm_csf_den_module, adm_csf_den_ptx), fail);
    CHECK_CUDA_GOTO(cu_f, cuModuleLoadData(&adm_cm_module, adm_cm_ptx), fail);

    // Get DWT kernel function pointers check adm_dwt2.cu for __global__ templated kernels
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_dwt_s123_combined_vert_kernel_0_0_int32_t,
                                        adm_dwt_module,
                                        "dwt_s123_combined_vert_kernel_0_0_int32_t"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_dwt_s123_combined_vert_kernel_32768_16_int32_t,
                                        adm_dwt_module,
                                        "dwt_s123_combined_vert_kernel_32768_16_int32_t"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_dwt_s123_combined_hori_kernel_16384_15,
                                        adm_dwt_module, "dwt_s123_combined_hori_kernel_16384_15"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_dwt_s123_combined_hori_kernel_32768_16,
                                        adm_dwt_module, "dwt_s123_combined_hori_kernel_32768_16"),
                    fail);
    CHECK_CUDA_GOTO(
        cu_f,
        cuModuleGetFunction(&s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t,
                            adm_dwt_module, "adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint8_t"),
        fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(
                        &s->func_adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t,
                        adm_dwt_module, "adm_dwt2_8_vert_hori_kernel_4_16_32768_128_8_uint16_t"),
                    fail);

    // Get csf kernel function pointers check adm_csf.cu for __global__ templated kernels
    CHECK_CUDA_GOTO(
        cu_f,
        cuModuleGetFunction(&s->func_adm_csf_kernel_1_4, adm_csf_module, "adm_csf_kernel_1_4"),
        fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_i4_adm_csf_kernel_1_4, adm_csf_module,
                                        "i4_adm_csf_kernel_1_4"),
                    fail);

    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_adm_csf_den_scale_line_kernel, adm_csf_den_module,
                                        "adm_csf_den_scale_line_kernel_8_128"),
                    fail);
    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_adm_csf_den_s123_line_kernel, adm_csf_den_module,
                                        "adm_csf_den_s123_line_kernel_8_128"),
                    fail);

    CHECK_CUDA_GOTO(cu_f,
                    cuModuleGetFunction(&s->func_adm_cm_reduce_line_kernel_4, adm_cm_module,
                                        "adm_cm_reduce_line_kernel_4"),
                    fail);
    CHECK_CUDA_GOTO(
        cu_f,
        cuModuleGetFunction(&s->func_adm_cm_line_kernel_8, adm_cm_module, "adm_cm_line_kernel_8"),
        fail);
    CHECK_CUDA_GOTO(
        cu_f,
        cuModuleGetFunction(&s->func_i4_adm_cm_line_kernel, adm_cm_module, "i4_adm_cm_line_kernel"),
        fail);

    CHECK_CUDA_GOTO(cu_f, cuCtxPopCurrent(NULL), fail_after_pop);

    // s->dwt2_8 = dwt2_8_device;

    s->integer_stride = ALIGN_CEIL(w * sizeof(int32_t));
    s->buf.ind_size_x = ALIGN_CEIL(((w + 1) / 2) * sizeof(int32_t));
    s->buf.ind_size_y = ALIGN_CEIL(((h + 1) / 2) * sizeof(int32_t));
    size_t buf_sz_one = s->buf.ind_size_x * ((h + 1) / 2);

    /* Buffer layout after decouple/csf_a elimination:
     * Scale 0 (int16): ref_dwt2(4), dis_dwt2(4), csf_f(3) = 11 half-bands = 5.5 buf_sz_one
     * Scale 1-3 (int32): i4_ref_dwt2(4), i4_dis_dwt2(4), i4_csf_f(3) = 11 full-bands = 11 buf_sz_one
     * Total = 16.5 buf_sz_one → allocate 17 (round up) */
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.data_buf,
                                 buf_sz_one * 11 + buf_sz_one / 2 * 11);
    if (ret)
        goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_ref,
                                 (s->integer_stride * 4 * ((h + 1) / 2)));
    if (ret)
        goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_dis,
                                 (s->integer_stride * 4 * ((h + 1) / 2)));
    if (ret)
        goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_accum, sizeof(uint64_t) * 3 * w * h);
    if (ret)
        goto free_ref;
    ret = vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_accum_h, sizeof(uint64_t) * 3 * h);
    if (ret)
        goto free_ref;
    ret =
        vmaf_cuda_buffer_alloc(fex->cu_state, &s->buf.tmp_res, sizeof(uint64_t) * RES_BUFFER_SIZE);
    if (ret)
        goto free_ref;
    ret = vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->buf.results_host,
                                      sizeof(uint64_t) * RES_BUFFER_SIZE);
    if (ret)
        goto free_ref;

    CUdeviceptr cu_res_top;
    ret = vmaf_cuda_buffer_get_dptr(s->buf.tmp_res, &cu_res_top);
    if (ret)
        goto free_ref;

    cu_res_top = init_res_cm_cuda(fex->cu_state, s->buf.adm_cm, cu_res_top);
    cu_res_top = init_res_csf_cuda(fex->cu_state, s->buf.adm_csf_den, cu_res_top);

    CUdeviceptr cu_data_top;
    vmaf_cuda_buffer_get_dptr(s->buf.data_buf, &cu_data_top);

    cu_data_top = init_dwt_band_cuda(fex->cu_state, &s->buf.ref_dwt2, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_cuda(fex->cu_state, &s->buf.dis_dwt2, cu_data_top, buf_sz_one / 2);
    cu_data_top = init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.csf_f, cu_data_top, buf_sz_one / 2);

    cu_data_top =
        i4_init_dwt_band_cuda(fex->cu_state, &s->buf.i4_ref_dwt2, cu_data_top, buf_sz_one);
    cu_data_top =
        i4_init_dwt_band_cuda(fex->cu_state, &s->buf.i4_dis_dwt2, cu_data_top, buf_sz_one);
    cu_data_top =
        i4_init_dwt_band_hvd_cuda(fex->cu_state, &s->buf.i4_csf_f, cu_data_top, buf_sz_one);

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        goto free_ref;

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
    if (s->buf.tmp_accum) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_accum);
        free(s->buf.tmp_accum);
    }
    if (s->buf.tmp_accum_h) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_accum_h);
        free(s->buf.tmp_accum_h);
    }
    if (s->buf.tmp_res) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_res);
        free(s->buf.tmp_res);
    }
    if (s->buf.results_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->buf.results_host);
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    (void)ret; // accumulated cleanup status intentionally discarded on error path

    return -ENOMEM;

fail:
    if (ctx_pushed)
        (void)cu_f->cuCtxPopCurrent(NULL);
fail_after_pop:
    return _cuda_err;
}

static int submit_fex_cuda(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;

    AdmStateCuda *s = fex->priv;

    s->submit_w = ref_pic->w[0];
    s->submit_h = ref_pic->h[0];

    // current implementation is limited by the 16-bit data pipeline, thus
    // cannot handle an angular frequency smaller than 1080p * 3H
    if (s->adm_norm_view_dist * s->adm_ref_display_height <
        DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) {
        return -EINVAL;
    }

    return integer_compute_adm_cuda(fex, s, ref_pic, dist_pic, &s->buf, s->adm_enhn_gain_limit,
                                    s->adm_norm_view_dist, s->adm_ref_display_height);
}

static int collect_fex_cuda(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    AdmStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    if (s->drained) {
        s->drained = false;
    } else {
        CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));
    }

    // Results are now in results_host — compute and write scores
    write_score_parameters_adm params = {
        .feature_collector = feature_collector,
        .s = s,
        .index = index,
        .w = s->submit_w,
        .h = s->submit_h,
    };
    write_scores(&params);
    return 0;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    AdmStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    /* Close path continues even on CUDA errors so every allocated
     * buffer gets freed. Individual CHECK_CUDA_GOTO steps skip forward
     * to the next handle without bailing. */
    int _cuda_err = 0;
    CHECK_CUDA_GOTO(cu_f, cuStreamSynchronize(s->str), after_sync);
after_sync:
    CHECK_CUDA_GOTO(cu_f, cuStreamDestroy(s->str), after_stream);
after_stream:
    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(s->finished), after_ev1);
after_ev1:
    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(s->ref_event), after_ev2);
after_ev2:
    CHECK_CUDA_GOTO(cu_f, cuEventDestroy(s->dis_event), after_ev3);
after_ev3:;

    int ret = _cuda_err;

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
    if (s->buf.tmp_accum) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_accum);
        free(s->buf.tmp_accum);
    }
    if (s->buf.tmp_accum_h) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_accum_h);
        free(s->buf.tmp_accum_h);
    }
    if (s->buf.tmp_res) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->buf.tmp_res);
        free(s->buf.tmp_res);
    }
    if (s->buf.results_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->buf.results_host);
    }

    ret |= vmaf_dictionary_free(&s->feature_name_dict);
    return ret;
}

static int flush_fex_cuda(VmafFeatureExtractor *fex, VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;
    AdmStateCuda *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CHECK_CUDA_RETURN(cu_f, cuStreamSynchronize(s->str));
    return 1;
}

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

VmafFeatureExtractor vmaf_fex_integer_adm_cuda = {.name = "adm_cuda",
                                                  .init = init_fex_cuda,
                                                  .submit = submit_fex_cuda,
                                                  .collect = collect_fex_cuda,
                                                  .flush = flush_fex_cuda,
                                                  .options = options_cuda,
                                                  .close = close_fex_cuda,
                                                  .priv_size = sizeof(AdmStateCuda),
                                                  .provided_features = provided_features,
                                                  .flags = VMAF_FEATURE_EXTRACTOR_CUDA};
