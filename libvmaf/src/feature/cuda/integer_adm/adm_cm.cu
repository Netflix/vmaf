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
#include "feature_collector.h"
#include "cuda/integer_adm_cuda.h"
#include "common.h"
#include "cuda_helper.cuh"

#include <algorithm>

//#define COMPARE_FUSED_SPLIT
#if defined(COMPARE_FUSED_SPLIT)
#include <iostream>
#endif

extern "C" {
    __global__ void i4_adm_cm_line_kernel(
            AdmBufferCuda buf, int h, int w, int top, int bottom, int left, int right,
            int start_row, int end_row, int start_col, int end_col, int src_stride,
            int csf_a_stride, int scale, int buffer_h, int buffer_stride,
            int32_t *accum_per_thread, AdmFixedParametersCuda params) {
        const cuda_i4_adm_dwt_band_t *src = &buf.i4_decouple_r;
        const cuda_i4_adm_dwt_band_t *csf_f = &buf.i4_csf_f;
        const cuda_i4_adm_dwt_band_t *csf_a = &buf.i4_csf_a;
        const int band = blockIdx.z + 1;
        int32_t *src_band = src->bands[band];
        int32_t *const *angles = csf_a->bands + 1;
        int32_t *const *flt_angles = csf_f->bands + 1;

        // for ADM: scales goes from 0 to 3 but in noise floor paper, it goes from
        // 1 to 4 (from finest scale to coarsest scale).
        const uint32_t *rfactor = &params.i_rfactor[scale * 3];

        const uint32_t shift_dst = 28;
        const uint32_t shift_flt = 32;
        const int32_t add_bef_shift_dst = (1u << (shift_dst - 1));
        const int32_t add_bef_shift_flt = (1u << (shift_flt - 1));

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
                int32_t src = angles[theta][src_stride * (i + offset_i[0] + 1) + j];
                int32_t *flt_ptr = flt_angles[theta];
                flt_ptr += (src_stride * (i + offset_i[0]));
                sum += flt_ptr[j + offset_j[0]];
                sum += flt_ptr[j];
                sum += flt_ptr[j + offset_j[1]];
                flt_ptr += src_stride;
                sum += flt_ptr[j + offset_j[0]];
                sum += (int32_t)((((int64_t)I4_ONE_BY_15 * abs((int32_t)src)) +
                            add_bef_shift_flt) >>
                        shift_flt);
                sum += flt_ptr[j + offset_j[1]];
                flt_ptr += src_stride * offset_i[1];
                sum += flt_ptr[j + offset_j[0]];
                sum += flt_ptr[j];
                sum += flt_ptr[j + offset_j[1]];
                thr += sum;
            }
            int32_t x = (int32_t)((((int64_t)src_band[i * src_stride + j] *
                            rfactor[blockIdx.z]) +
                        add_bef_shift_dst) >>
                    shift_dst);
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
struct WarpShift
{
    uint32_t shift_cub[3];
    uint32_t add_shift_cub[3];
    uint32_t shift_sq[3];
    uint32_t add_shift_sq[3];
};

template <int rows_per_thread>
__device__ __forceinline__ void adm_cm_line_kernel(AdmBufferCuda buf, int h, int w, int top,
        int bottom, int left, int right,
        int start_row, int end_row, int start_col,
        int end_col, int src_stride,
        int csf_a_stride, int buffer_h,
        int buffer_stride, int32_t *accum_per_block,
        AdmFixedParametersCuda params,
        // reduce
        int scale, int64_t* accum_global,

        // shift warp
        WarpShift ws,
        // shift global
        const uint32_t shift_inner_accum, const uint32_t add_shift_inner_accum
        ) {
    const cuda_adm_dwt_band_t *src = &buf.decouple_r;
    const cuda_adm_dwt_band_t *csf_f = &buf.csf_f;
    const cuda_adm_dwt_band_t *csf_a = &buf.csf_a;
    const int band = blockIdx.z + 1;
    int16_t *src_band = src->bands[band];
    int16_t *const *angles = csf_a->bands + 1;
    int16_t *const *flt_angles = csf_f->bands + 1;

    uint32_t *i_rfactor = params.i_rfactor;

    int cta_y = (blockDim.y * blockIdx.y + threadIdx.y) * rows_per_thread;
    int cta_x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = start_row + cta_y;
    int x = start_col + cta_x;

    const int total_rows = (3 + rows_per_thread - 1);
    int32_t thr[rows_per_thread] = {0};

    if (y < end_row && x < end_col)
    {
        int pos_x[3] = {x - 1, x, x + 1};
        pos_x[0] = abs(pos_x[0]);
        pos_x[2] = pos_x[2] - max(0, 2*(x - w)+1);

#pragma unroll
        for (int theta = 0; theta < 3; ++theta)
        {
            // the goal of the algorithm below is to read each row only once.
            // to achieve this goal iterate over all rows and then sum up the row
            // to each row accumulator.
            // the sum is accumulated through the following pattern
            // flt[0][0] flt[0][1] flt[0][2]
            // flt[1][0] src[1][1] flt[1][2]
            // flt[2][0] flt[2][1] flt[2][2]
            // for each input row we compute the row overlap with the filter
            // and use the correct summation.
            // since all loops are unrolled the compiler eliminates all ifs
            // and generates just the minimal amount of LDG / IADD instructions.
#pragma unroll
            for (int row = 0; row < total_rows;++row)
            {
                int pos_y = y - 1 + row;
                pos_y = abs(pos_y);
                pos_y = pos_y - max(0, 2*(y - h)+1);

                int16_t src = angles[theta][pos_y * src_stride + x];
                int16_t *flt_ptr = flt_angles[theta] + pos_y*src_stride;
                int16_t flt_row[3] = {flt_ptr[pos_x[0]], flt_ptr[pos_x[1]], flt_ptr[pos_x[2]]};

#pragma unroll
                for (int thread_item = 0;thread_item < rows_per_thread;++thread_item)
                {
                    int thread_row = row - thread_item;
                    if (thread_row >= 0 && thread_row < 3)
                    {
                        thr[thread_item] += flt_row[0] + flt_row[2];
                        if (thread_row != 1) {
                            thr[thread_item] += flt_row[1];
                        }
                        else {
                            thr[thread_item] += (int16_t)(((ONE_BY_15 * abs((int32_t)src)) + 2048) >> 12);;
                        }
                    }
                }
            }
        }
    }


    int32_t shift_sub_block = shift_sub[blockIdx.z];
    int32_t accum_thread_reg[rows_per_thread] = {0};
    for (int row = 0;row < rows_per_thread;++row) {
        int32_t sb = 0;
        if ((y + row) < end_row && x < end_col) {
            sb = src_band[(y + row) * src_stride + x];
        }
        sb = abs(int32_t(i_rfactor[blockIdx.z] * sb)) - (thr[row] << shift_sub_block);
        accum_thread_reg[row] = max(0, sb);
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
    for (int row = 0;row < rows_per_thread;++row) {
        int32_t accum_thread = accum_thread_reg[row];
        const int32_t x_sq = (int32_t)((((int64_t)accum_thread * accum_thread) + add_shift_sq >> shift_sq));
        accum += (((int64_t)x_sq * accum_thread) + add_shift_cub) >> shift_cub;
    }

    // accumulate warp
    accum = warp_reduce(accum);

    if (threadIdx.x % 32 == 0)
    {
        accum = (accum + add_shift_inner_accum) >> shift_inner_accum;
        atomicAdd_int64(&accum_global[band2],
                accum);
    }
}

template <int val_per_thread>
__device__ __forceinline__ void adm_cm_reduce_line_kernel(int h, int w, int scale, int buffer_h,
        int buffer_stride,
        const int32_t *buffer,
        int64_t *accum) {
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
            const int32_t x_sq =
                (int32_t)((((int64_t)x * x) + add_shift_sq) >> shift_sq);
            temp_value += (((int64_t)x_sq * x) + add_shift_cub) >> shift_cub;
        }
    }
    temp_value = warp_reduce(temp_value);

    if ((threadIdx.x % VMAF_CUDA_THREADS_PER_WARP) == 0) {
        const uint32_t shift_inner_accum = __float2uint_ru(__log2f(h));
        const uint32_t add_shift_inner_accum = 1 << (shift_inner_accum - 1);
        atomicAdd_int64(&accum[band],
                (temp_value + add_shift_inner_accum) >> shift_inner_accum);
    }
}

#define ADM_CM_REDUCE_LINE(val_per_thread)                           \
    __global__ void adm_cm_reduce_line_kernel_##val_per_thread (  \
            int h, int w, int scale, int buffer_h,                             \
            int buffer_stride, const int32_t *buffer, int64_t *accum)          \
{                                                                              \
    adm_cm_reduce_line_kernel<val_per_thread>(                       \
            h, w, scale, buffer_h, buffer_stride,  buffer, accum);             \
}

#define ADM_CM_LINE(rows_per_thread)                                  \
    __global__ void adm_cm_line_kernel_##rows_per_thread (         \
            AdmBufferCuda buf, int h, int w, int top,                                    \
            int bottom, int left, int right, int start_row, int end_row, int start_col,  \
            int end_col, int src_stride, int csf_a_stride, int buffer_h,                 \
            int buffer_stride, int32_t *accum_per_block, AdmFixedParametersCuda params,  \
            int scale, int64_t* accum_global, WarpShift ws,                              \
            const uint32_t shift_inner_accum, const uint32_t add_shift_inner_accum)      \
{                                                                                        \
    adm_cm_line_kernel<rows_per_thread>(                              \
            buf, h, w, top, bottom, left, right, start_row, end_row, start_col,          \
            end_col, src_stride, csf_a_stride, buffer_h, buffer_stride,                  \
            accum_per_block, params,scale, accum_global,                                 \
            ws, shift_inner_accum, add_shift_inner_accum);                               \
}



extern "C" {
    // 128 = warps_per_thread * val_per_thread = 32 * 4 -- assuming 32 threads per warp, this might change in the future
    ADM_CM_REDUCE_LINE(4);   // adm_cm_reduce_line_kernel_4
    ADM_CM_LINE(8);            // adm_cm_line_kernel_8
}
