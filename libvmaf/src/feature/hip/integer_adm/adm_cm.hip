/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
 *  Copyright 2026 Lusoris and Claude (Anthropic)
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
 *  HIP port of cuda/integer_adm/adm_cm.cu — identical arithmetic,
 *  CUDA-specific headers and type names replaced with HIP equivalents.
 */

#include <hip/hip_runtime.h>
#include "hip/integer_adm_hip.h"
#include "common.h"

#if defined(COMPARE_FUSED_SPLIT)
#include <iostream>


/* Inline helper: compute csf_a for a single band at a given pixel position
 * for scales 1-3 (i4 path). Returns the csf_a value = rfactor * a_val. */
__device__ __forceinline__ int32_t inline_i4_csf_a(const hip_i4_adm_dwt_band_t *ref,
                                                   const hip_i4_adm_dwt_band_t *dis, int idx,
                                                   int theta, /* theta 0..2 → band h/v/d */
                                                   const uint32_t *rfactor,
                                                   double adm_enhn_gain_limit)
{
    const uint32_t shift_dst = 28;
    const int32_t add_bef_shift_dst = (1u << (shift_dst - 1));

    int32_t oh = ref->band_h[idx];
    int32_t ov = ref->band_v[idx];
    int32_t od = ref->band_d[idx];
    int32_t th = dis->band_h[idx];
    int32_t tv = dis->band_v[idx];
    int32_t td = dis->band_d[idx];

    int angle_flag = decouple_angle_flag_s123(oh, ov, th, tv);
    int32_t r_val = decouple_r_s123(oh, ov, od, th, tv, td, theta, angle_flag, adm_enhn_gain_limit);

    int32_t t_val;
    if (theta == 0)
        t_val = th;
    else if (theta == 1)
        t_val = tv;
    else
        t_val = td;

    int32_t a_val = t_val - r_val;

    /* dst_val = rfactor * a_val — this is csf_a */
    return (int32_t)(((rfactor[theta] * int64_t(a_val)) + add_bef_shift_dst) >> shift_dst);
}

/* Inline helper: compute decouple_r for a single band at a given pixel position
 * for scales 1-3 (i4 path). */
__device__ __forceinline__ int32_t inline_i4_decouple_r(const hip_i4_adm_dwt_band_t *ref,
                                                        const hip_i4_adm_dwt_band_t *dis, int idx,
                                                        int band_idx, /* band_idx 0..2 → h/v/d */
                                                        double adm_enhn_gain_limit)
{
    int32_t oh = ref->band_h[idx];
    int32_t ov = ref->band_v[idx];
    int32_t od = ref->band_d[idx];
    int32_t th = dis->band_h[idx];
    int32_t tv = dis->band_v[idx];
    int32_t td = dis->band_d[idx];

    int angle_flag = decouple_angle_flag_s123(oh, ov, th, tv);
    return decouple_r_s123(oh, ov, od, th, tv, td, band_idx, angle_flag, adm_enhn_gain_limit);
}

/* Inline helper: compute csf_a for scale 0 (int16 path). */
__device__ __forceinline__ int16_t inline_s0_csf_a(const hip_adm_dwt_band_t *ref,
                                                   const hip_adm_dwt_band_t *dis, int idx,
                                                   int theta, /* theta 0..2 → band h/v/d */
                                                   const uint32_t *i_rfactor,
                                                   double adm_enhn_gain_limit)
{
    __constant__ static const uint8_t i_shifts_cm[4] = {0, 15, 15, 17};
    __constant__ static const uint16_t i_shiftsadd_cm[4] = {0, 16384, 16384, 65535};

    int16_t oh = ref->band_h[idx];
    int16_t ov = ref->band_v[idx];
    int16_t od = ref->band_d[idx];
    int16_t th = dis->band_h[idx];
    int16_t tv = dis->band_v[idx];
    int16_t td = dis->band_d[idx];

    int angle_flag = decouple_angle_flag_s0(oh, ov, th, tv);
    int16_t r_val = decouple_r_s0(oh, ov, od, th, tv, td, theta, angle_flag, adm_enhn_gain_limit);

    int16_t t_val;
    if (theta == 0)
        t_val = th;
    else if (theta == 1)
        t_val = tv;
    else
        t_val = td;

    int16_t a_val = t_val - r_val;

    int band = theta + 1; // band index 1..3
    int32_t dst_val = i_rfactor[theta] * (uint32_t)a_val;
    return (dst_val + i_shiftsadd_cm[band]) >> i_shifts_cm[band];
}

/* Inline helper: compute decouple_r for scale 0 (int16 path). */
__device__ __forceinline__ int16_t inline_s0_decouple_r(const hip_adm_dwt_band_t *ref,
                                                        const hip_adm_dwt_band_t *dis, int idx,
                                                        int band_idx, /* band_idx 0..2 → h/v/d */
                                                        double adm_enhn_gain_limit)
{
    int16_t oh = ref->band_h[idx];
    int16_t ov = ref->band_v[idx];
    int16_t od = ref->band_d[idx];
    int16_t th = dis->band_h[idx];
    int16_t tv = dis->band_v[idx];
    int16_t td = dis->band_d[idx];

    int angle_flag = decouple_angle_flag_s0(oh, ov, th, tv);
    return decouple_r_s0(oh, ov, od, th, tv, td, band_idx, angle_flag, adm_enhn_gain_limit);
}

extern "C" {
__global__ void i4_adm_cm_line_kernel(AdmBufferHip buf, int h, int w, int top, int bottom,
                                      int left, int right, int start_row, int end_row,
                                      int start_col, int end_col, int src_stride, int csf_a_stride,
                                      int scale, int buffer_h, int buffer_stride,
                                      int32_t *accum_per_thread, AdmFixedParametersHip params)
{
    const hip_i4_adm_dwt_band_t *ref = &buf.i4_ref_dwt2;
    const hip_i4_adm_dwt_band_t *dis = &buf.i4_dis_dwt2;
    const hip_i4_adm_dwt_band_t *csf_f = &buf.i4_csf_f;
    const int band = blockIdx.z + 1;
    int32_t *const *flt_angles = csf_f->bands + 1;

    const uint32_t *rfactor = &params.i_rfactor[scale * 3];
    const double adm_enhn_gain_limit = params.adm_enhn_gain_limit;

    const uint32_t shift_flt = 32;
    const int32_t add_bef_shift_flt = (1u << (shift_flt - 1));
    const uint32_t shift_dst = 28;
    const int32_t add_bef_shift_dst = (1u << (shift_dst - 1));

    uint32_t shift_cub = __float2uint_ru(__log2f(w));

    const int32_t shift_sub = 0;

    int i = start_row + blockIdx.y;
    int j = start_col + blockIdx.x * blockDim.x + threadIdx.x;

    int32_t accum_thread = 0;
    if (i < end_row && j < end_col) {

        int16_t offset_i[2] = {-1, 1};
        if (i == 0 && top <= 0) {
            offset_i[0] = 1;
        } else if (i == (h - 1) && bottom > (h - 1)) {
            offset_i[1] = 0;
        }

        int16_t offset_j[2] = {-1, 1};
        if (j == 0 && left <= 0) {
            offset_j[0] = 1;
        } else if (j == (w - 1) && right > (w - 1)) {
            offset_j[1] = 0;
        }

        int32_t thr = 0;
        for (int theta = 0; theta < 3; ++theta) {
            int32_t sum = 0;
            /* Inline csf_a at center pixel [i, j] */
            int32_t csf_a_val = inline_i4_csf_a(ref, dis, src_stride * (i + offset_i[0] + 1) + j,
                                                theta, rfactor, adm_enhn_gain_limit);

            int32_t *flt_ptr_line = flt_angles[theta];
            flt_ptr_line += (src_stride * (i + offset_i[0]));
            sum += flt_ptr_line[j + offset_j[0]];
            sum += flt_ptr_line[j];
            sum += flt_ptr_line[j + offset_j[1]];
            flt_ptr_line += src_stride;
            sum += flt_ptr_line[j + offset_j[0]];
            sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs(csf_a_val)) + add_bef_shift_flt) >>
                             shift_flt);
            sum += flt_ptr_line[j + offset_j[1]];
            flt_ptr_line += src_stride * offset_i[1];
            sum += flt_ptr_line[j + offset_j[0]];
            sum += flt_ptr_line[j];
            sum += flt_ptr_line[j + offset_j[1]];
            thr += sum;
        }
        /* Inline decouple_r at pixel [i, j] */
        int32_t r_val =
            inline_i4_decouple_r(ref, dis, i * src_stride + j, band - 1, adm_enhn_gain_limit);
        int32_t x =
            (int32_t)((((int64_t)r_val * rfactor[blockIdx.z]) + add_bef_shift_dst) >> shift_dst);
        x = abs(x) - (thr >> shift_sub);
        accum_thread = x < 0 ? 0 : x;
    }
    if ((blockIdx.x * blockDim.x + threadIdx.x) < buffer_stride)
        accum_per_thread[(blockIdx.z * buffer_h + blockIdx.y) * buffer_stride +
                         blockIdx.x * blockDim.x + threadIdx.x] = accum_thread;
}
}
__constant__ const int32_t shift_sub[3] = {10, 10, 12};
__constant__ const int fixed_shift[3] = {4, 4, 3};

// accumulation
__constant__ const int32_t shift_xsq[3] = {29, 29, 30};
__constant__ const int32_t add_shift_xsq[3] = {268435456, 268435456, 536870912};

// HACK: the 256 byte alignment is required to ensure that the struct is not moved to lmem
struct WarpShift {
    uint32_t shift_cub[3];
    uint32_t add_shift_cub[3];
    uint32_t shift_sq[3];
    uint32_t add_shift_sq[3];
};

template <int rows_per_thread>
__device__ __forceinline__ void
adm_cm_line_kernel(AdmBufferHip buf, int h, int w, int top, int bottom, int left, int right,
                   int start_row, int end_row, int start_col, int end_col, int src_stride,
                   int csf_a_stride, int buffer_h, int buffer_stride, int32_t *accum_per_block,
                   AdmFixedParametersHip params,
                   // reduce
                   int scale, int64_t *accum_global,

                   // shift warp
                   WarpShift ws,
                   // shift global
                   const uint32_t shift_inner_accum, const uint32_t add_shift_inner_accum)
{
    const hip_adm_dwt_band_t *ref = &buf.ref_dwt2;
    const hip_adm_dwt_band_t *dis = &buf.dis_dwt2;
    const hip_adm_dwt_band_t *csf_f = &buf.csf_f;
    const int band = blockIdx.z + 1;
    int16_t *const *flt_angles = csf_f->bands + 1;

    uint32_t *i_rfactor = params.i_rfactor;
    const double adm_enhn_gain_limit = params.adm_enhn_gain_limit;

    int cta_y = (blockDim.y * blockIdx.y + threadIdx.y) * rows_per_thread;
    int cta_x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = start_row + cta_y;
    int x = start_col + cta_x;

    const int total_rows = (3 + rows_per_thread - 1);
    int32_t thr[rows_per_thread] = {0};

    if (y < end_row && x < end_col) {
        int pos_x[3] = {x - 1, x, x + 1};
        pos_x[0] = abs(pos_x[0]);
        pos_x[2] = pos_x[2] - max(0, 2 * (x - w) + 1);

#pragma unroll
        for (int theta = 0; theta < 3; ++theta) {
#pragma unroll
            for (int row = 0; row < total_rows; ++row) {
                int pos_y = y - 1 + row;
                pos_y = abs(pos_y);
                pos_y = pos_y - max(0, 2 * (y - h) + 1);

                /* Inline csf_a at center pixel */
                int16_t csf_a_val = inline_s0_csf_a(ref, dis, pos_y * src_stride + x, theta,
                                                    i_rfactor, adm_enhn_gain_limit);
                int16_t *flt_ptr = flt_angles[theta] + pos_y * src_stride;
                int16_t flt_row[3] = {flt_ptr[pos_x[0]], flt_ptr[pos_x[1]], flt_ptr[pos_x[2]]};

#pragma unroll
                for (int thread_item = 0; thread_item < rows_per_thread; ++thread_item) {
                    int thread_row = row - thread_item;
                    if (thread_row >= 0 && thread_row < 3) {
                        thr[thread_item] += flt_row[0] + flt_row[2];
                        if (thread_row != 1) {
                            thr[thread_item] += flt_row[1];
                        } else {
                            thr[thread_item] +=
                                (int16_t)(((ONE_BY_15 * abs((int32_t)csf_a_val)) + 2048) >> 12);
                            ;
                        }
                    }
                }
            }
        }
    }

    int32_t shift_sub_block = shift_sub[blockIdx.z];
    int32_t accum_thread_reg[rows_per_thread] = {0};
    for (int row = 0; row < rows_per_thread; ++row) {
        int16_t sb = 0;
        if ((y + row) < end_row && x < end_col) {
            /* Inline decouple_r at pixel [y + row, x] */
            sb = inline_s0_decouple_r(ref, dis, (y + row) * src_stride + x, band - 1,
                                      adm_enhn_gain_limit);
        }
        int32_t val = abs(int32_t(i_rfactor[blockIdx.z] * sb)) - (thr[row] << shift_sub_block);
        accum_thread_reg[row] = max(0, val);
    }

    const int band2 = blockIdx.z;
    int64_t accum = 0;

    // the compiler does not assume that parameters are constant, move them to local variables to give the compiler
    // a hint that those values have to be loaded only once from constant memory.
    int32_t add_shift_cub = ws.add_shift_cub[band2];
    int32_t shift_cub = ws.shift_cub[band2];
    int32_t add_shift_sq = ws.add_shift_sq[band2];
    int32_t shift_sq = ws.shift_sq[band2];

    // accumulate per thread
    for (int row = 0; row < rows_per_thread; ++row) {
        int32_t accum_thread = accum_thread_reg[row];
        const int32_t x_sq =
            (int32_t)((((int64_t)accum_thread * accum_thread) + add_shift_sq >> shift_sq));
        accum += (((int64_t)x_sq * accum_thread) + add_shift_cub) >> shift_cub;
    }

    // accumulate warp
    accum = warp_reduce(accum);

    if (threadIdx.x % 32 == 0) {
        accum = (accum + add_shift_inner_accum) >> shift_inner_accum;
        atomicAdd_int64(&accum_global[band2], accum);
    }
}

template <int val_per_thread>
__device__ __forceinline__ void adm_cm_reduce_line_kernel(int h, int w, int scale, int buffer_h,
                                                          int buffer_stride, const int32_t *buffer,
                                                          int64_t *accum)
{
    const int band = blockIdx.z;
    const int line = blockIdx.y;
    const int off = band * (buffer_h * buffer_stride);
    const int b_off = off + line * buffer_stride;

    uint32_t shift_cub = __float2uint_ru(__log2f(w));
    uint32_t add_shift_cub = 1 << (shift_cub - 1);
    int32_t shift_sq = 30;
    int32_t add_shift_sq = 536870912; // 2^29
    if (scale == 0) {
        shift_cub = __float2uint_ru(__log2f(w) - fixed_shift[band]);
        add_shift_cub = 1 << (shift_cub - 1);
        shift_sq = shift_xsq[band];
        add_shift_sq = add_shift_xsq[band];
    }

    int64_t temp_value = 0;
    const int buffer_col = (blockDim.x * blockIdx.x + threadIdx.x) * val_per_thread;
    const int32_t *buffer_loc = buffer + b_off + buffer_col;
    for (int i = 0; i < val_per_thread; ++i) {
        if ((buffer_col + i) < buffer_stride) {
            const int32_t x = buffer_loc[i];
            const int32_t x_sq = (int32_t)((((int64_t)x * x) + add_shift_sq) >> shift_sq);
            temp_value += (((int64_t)x_sq * x) + add_shift_cub) >> shift_cub;
        }
    }
    temp_value = warp_reduce(temp_value);

    if ((threadIdx.x % VMAF_CUDA_THREADS_PER_WARP) == 0) {
        const uint32_t shift_inner_accum = __float2uint_ru(__log2f(h));
        const uint32_t add_shift_inner_accum = 1 << (shift_inner_accum - 1);
        atomicAdd_int64(&accum[band], (temp_value + add_shift_inner_accum) >> shift_inner_accum);
    }
}

#define ADM_CM_REDUCE_LINE(val_per_thread)                                                         \
    __global__ void adm_cm_reduce_line_kernel_##val_per_thread(                                    \
        int h, int w, int scale, int buffer_h, int buffer_stride, const int32_t *buffer,           \
        int64_t *accum)                                                                            \
    {                                                                                              \
        adm_cm_reduce_line_kernel<val_per_thread>(h, w, scale, buffer_h, buffer_stride, buffer,    \
                                                  accum);                                          \
    }

#define ADM_CM_LINE(rows_per_thread)                                                               \
    __global__ void adm_cm_line_kernel_##rows_per_thread(                                          \
        AdmBufferHip buf, int h, int w, int top, int bottom, int left, int right, int start_row,  \
        int end_row, int start_col, int end_col, int src_stride, int csf_a_stride, int buffer_h,   \
        int buffer_stride, int32_t *accum_per_block, AdmFixedParametersHip params, int scale,     \
        int64_t *accum_global, WarpShift ws, const uint32_t shift_inner_accum,                     \
        const uint32_t add_shift_inner_accum)                                                      \
    {                                                                                              \
        adm_cm_line_kernel<rows_per_thread>(                                                       \
            buf, h, w, top, bottom, left, right, start_row, end_row, start_col, end_col,           \
            src_stride, csf_a_stride, buffer_h, buffer_stride, accum_per_block, params, scale,     \
            accum_global, ws, shift_inner_accum, add_shift_inner_accum);                           \
    }

extern "C" {
// 128 = warps_per_thread * val_per_thread = 32 * 4 -- assuming 32 threads per warp, this might change in the future
ADM_CM_REDUCE_LINE(4); // adm_cm_reduce_line_kernel_4
ADM_CM_LINE(8);        // adm_cm_line_kernel_8
}
