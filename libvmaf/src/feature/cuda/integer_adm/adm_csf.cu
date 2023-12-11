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

#include <assert.h>

#include "cuda_helper.cuh"

template <int cols_per_thread>
static __device__ __forceinline__ void copy_vec_4(const int32_t * __restrict__ in, int32_t * __restrict__ out)
{
    //__builtin_assume_aligned(in, 16);
    //__builtin_assume_aligned(out, 16);

    static_assert(cols_per_thread % 4 == 0, "implemented only for a multiple of 4");
#pragma unroll
    for (int col = 0;col <cols_per_thread;col += 4) {
        *reinterpret_cast<uint4*>(out + col) = *reinterpret_cast<const uint4*>(in + col);
    }
}

template <int cols_per_thread>
static __device__ __forceinline__ void copy_vec_4(const int16_t * __restrict__ in, int16_t * __restrict__ out)
{
    // __builtin_assume_aligned(in, 8);
    // __builtin_assume_aligned(out, 8);

    static_assert(cols_per_thread % 4 == 0, "implemented only for a multiple of 4");
#pragma unroll
    for (int col = 0;col <cols_per_thread;col += 4) {
        *reinterpret_cast<ushort4*>(out + col) = *reinterpret_cast<const ushort4*>(in + col);
    }
}

template <int rows_per_thread, int cols_per_thread>
__device__ __forceinline__ void i4_adm_csf_kernel(AdmBufferCuda buf, int scale, int top,
        int bottom, int left, int right, int stride,
        AdmFixedParametersCuda params) {

    int band = blockIdx.z + 1;
    const int32_t *src_ptr = buf.i4_decouple_a.bands[band]; // this is evaluated to a LDC
    int32_t *dst_ptr = buf.i4_csf_a.bands[band];
    int32_t *flt_ptr = buf.i4_csf_f.bands[band];

    int y = top + (blockIdx.y * blockDim.y + threadIdx.y) * rows_per_thread;
    int x = left + (blockIdx.x * blockDim.x + threadIdx.x) * cols_per_thread;

    const uint32_t i_rfactor = params.i_rfactor[scale * 3 + blockIdx.z];
    const uint32_t FIX_ONE_BY_30 = 143165577;
    const uint32_t shift_dst = 28;
    const uint32_t shift_flt = 32;
    const int32_t add_bef_shift_dst = (1u << (shift_dst - 1));
    const int32_t add_bef_shift_flt = (1u << (shift_flt - 1));

    const int offset = y * stride + x;

    if (y < bottom && x < right) {
        __align__(16) int32_t src[cols_per_thread];
        __align__(16) int32_t dst_vec[cols_per_thread];
        __align__(16) int32_t flt_vec[cols_per_thread];

        for (int row = 0;row < rows_per_thread;++row) {
            copy_vec_4<cols_per_thread>(src_ptr + offset + row * stride, src);

            for (int col = 0;col < cols_per_thread;++col) {
                int32_t dst_val = (int32_t)(((i_rfactor * int64_t(src[col])) +
                            add_bef_shift_dst) >>
                        shift_dst);
                dst_vec[col] = dst_val;
                flt_vec[col] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val)) +
                            add_bef_shift_flt) >>
                        shift_flt);
            }
            copy_vec_4<cols_per_thread>(dst_vec, dst_ptr + offset + row * stride);
            copy_vec_4<cols_per_thread>(flt_vec, flt_ptr + offset + row * stride);
        }
    }
}

__constant__ const uint8_t i_shifts[4] = {0, 15, 15, 17};
__constant__ const uint16_t i_shiftsadd[4] = {0, 16384, 16384, 65535};

template <int rows_per_thread, int cols_per_thread>
__device__ __forceinline__ void adm_csf_kernel(AdmBufferCuda buf, int top, int bottom, int left,
        int right, int stride,
        AdmFixedParametersCuda params) {
    const int band = blockIdx.z + 1;

    const int16_t *src_ptr = buf.decouple_a.bands[band]; // this is evaluated to a LDC
    int16_t *dst_ptr = buf.csf_a.bands[band];
    int16_t *flt_ptr = buf.csf_f.bands[band];
    int y = top + (blockIdx.y * blockDim.y + threadIdx.y) * rows_per_thread;
    int x = left + (blockIdx.x * blockDim.x + threadIdx.x) * cols_per_thread;

    const uint32_t i_rfactor = params.i_rfactor[blockIdx.z];
    const uint16_t FIX_ONE_BY_30 = 4369; //(1/30)*2^17

    if (y < bottom && x < right) {
        __align__(8) int16_t src[cols_per_thread];
        __align__(8) int16_t dst_vec[cols_per_thread];
        __align__(8) int16_t flt_vec[cols_per_thread];

        const int offset = y * stride + x;

        for (int row = 0;row < rows_per_thread;++row) {
            copy_vec_4<cols_per_thread>(src_ptr + offset + row * stride, src);

            for (int col = 0;col < cols_per_thread;++col) {
                int32_t dst_val = i_rfactor * (uint32_t)src[col];
                int16_t i16_dst_val = (dst_val + i_shiftsadd[band]) >> i_shifts[band];
                dst_vec[col] = i16_dst_val;
                flt_vec[col] =
                    ((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val)) + 2048) >> 12;
            }
            copy_vec_4<cols_per_thread>(dst_vec, dst_ptr + offset + row * stride);
            copy_vec_4<cols_per_thread>(flt_vec, flt_ptr + offset + row * stride);
        }
    }
}

#define ADM_CSF_KERNEL(rows_per_thread, cols_per_thread)                       \
    __global__ void adm_csf_kernel_##rows_per_thread##_##cols_per_thread (     \
            AdmBufferCuda buf, int top, int bottom, int left,                  \
            int right, int stride,                                             \
            AdmFixedParametersCuda params) {                                   \
        adm_csf_kernel<rows_per_thread, cols_per_thread>(                      \
                buf, top, bottom, left, right, stride,  params);               \
    }

#define I4_ADM_CSF_KERNEL(rows_per_thread, cols_per_thread)                    \
    __global__ void i4_adm_csf_kernel_##rows_per_thread##_##cols_per_thread (  \
            AdmBufferCuda buf, int scale, int top, int bottom, int left,       \
            int right, int stride,                                             \
            AdmFixedParametersCuda params) {                                   \
        i4_adm_csf_kernel<rows_per_thread, cols_per_thread>(                   \
                buf, scale,top, bottom, left, right, stride,  params);         \
    }

extern "C" {
    ADM_CSF_KERNEL(1, 4);     // adm_csf_kernel_1_4
    I4_ADM_CSF_KERNEL(1, 4);  // i4_adm_csf_kernel_1_4
}
