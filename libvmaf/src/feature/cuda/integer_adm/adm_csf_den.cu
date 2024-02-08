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

template <int val_per_thread, int cta_size>
__device__ __forceinline__ void adm_csf_den_scale_line_kernel(const cuda_adm_dwt_band_t src, int h,
        int top, int bottom, int left,
        int right, int src_stride,
        uint64_t *accum) {
    const int band = blockIdx.z + 1;
    // this is evaluated to a 4 STL and one LDL therefore we need the switch
    // const int16_t * src_ptr = src.bands[band] + (top + blockIdx.y) * src_stride;
    int16_t * src_ptr;
    switch (band) {
        case 1:
            src_ptr = src.band_h + (top + blockIdx.y) * src_stride;
            break;
        case 2:
            src_ptr = src.band_v + (top + blockIdx.y) * src_stride;
            break;
        case 3:
            src_ptr = src.band_d + (top + blockIdx.y) * src_stride;
            break;
    }
    int j = left + (cta_size * blockIdx.x) * val_per_thread + threadIdx.x;

    uint64_cu temp_value = 0;
#pragma unroll
    for (int i = 0; i < val_per_thread; ++i) {
        if (j < right) {
            uint16_t t = (uint16_t)abs(src_ptr[j]);
            temp_value += ((uint64_t)t * t) * t;
        }
        j += cta_size;
    }
    temp_value = warp_reduce(temp_value);

    if ((threadIdx.x % VMAF_CUDA_THREADS_PER_WARP) == 0) {
        uint32_t shift_accum = (uint32_t)__float2uint_ru(
                __log2f((bottom - top) * (right - left)) - 20);
        shift_accum = shift_accum > 0 ? shift_accum : 0;
        int32_t add_shift_accum = shift_accum > 0 ? (1 << (shift_accum - 1)) : 0;
        atomicAdd((uint64_cu *)&accum[band - 1],
                (temp_value + add_shift_accum) >> shift_accum);
    }
}

template <int val_per_thread, int cta_size>
__device__ __forceinline__ void adm_csf_den_s123_line_kernel(
        const cuda_i4_adm_dwt_band_t src, int h, int top, int bottom, int left, int right,
        int src_stride, uint32_t add_shift_sq, uint32_t shift_sq, uint64_t *accum) {
    const uint32_t shift_cub = (uint32_t)__float2uint_ru(__log2f((right - left)));
    const uint32_t add_shift_cub = (uint32_t)1 << (shift_cub - 1);

    const int band = blockIdx.z + 1;
    // this is evaluated to a 4 STL and one LDL therefore we need the switch
    // const int32_t *src_ptr = src.bands[band] + (top + blockIdx.y) * src_stride;
    int32_t * src_ptr;
    switch (band) {
        case 1:
            src_ptr = src.band_h + (top + blockIdx.y) * src_stride;
            break;
        case 2:
            src_ptr = src.band_v + (top + blockIdx.y) * src_stride;
            break;
        case 3:
            src_ptr = src.band_d + (top + blockIdx.y) * src_stride;
            break;
    }
    int j = left + (cta_size * blockIdx.x) * val_per_thread + threadIdx.x;

    uint64_cu temp_value = 0;
#pragma unroll
    for (int i = 0; i < val_per_thread; ++i) {
        if (j < right) {
            uint32_t t = (uint32_t)abs(src_ptr[j]);
            temp_value += (((((((uint64_t)t * t) + add_shift_sq) >> shift_sq) * t) +
                        add_shift_cub) >>
                    shift_cub);
        }
        j += cta_size;
    }
    temp_value = warp_reduce(temp_value);

    if ((threadIdx.x % VMAF_CUDA_THREADS_PER_WARP) == 0) {
        uint32_t shift_accum = (uint32_t)__float2uint_ru(__log2f((bottom - top)));
        uint32_t add_shift_accum = (uint32_t)1 << (shift_accum - 1);
        atomicAdd((uint64_cu *)&accum[band - 1],
                (temp_value + add_shift_accum) >> shift_accum);
    }
}
#define ADM_CSF_SCALE_LINE(val_per_thread, cta_size)                                        \
    __global__ void adm_csf_den_scale_line_kernel_##val_per_thread##_##cta_size (           \
            const cuda_adm_dwt_band_t src, int h, int top, int bottom, int left, int right, \
            int src_stride, uint64_t *accum)                                                \
{                                                                                           \
    adm_csf_den_scale_line_kernel<val_per_thread, cta_size>(                                \
            src, h, top, bottom, left, right, src_stride, accum);                           \
}

#define ADM_CSF_DEN_S123_LINE(val_per_thread, cta_size)                                        \
    __global__ void adm_csf_den_s123_line_kernel_##val_per_thread##_##cta_size (               \
            const cuda_i4_adm_dwt_band_t src, int h, int top, int bottom, int left, int right, \
            int src_stride, uint32_t add_shift_sq, uint32_t shift_sq, uint64_t *accum )        \
{                                                                                              \
    adm_csf_den_s123_line_kernel<val_per_thread, cta_size>(                                    \
            src, h, top, bottom, left, right, src_stride, add_shift_sq, shift_sq, accum);      \
}

extern "C" {
    // 128 = VMAF_CUDA_THREADS_PER_WARP * 4 -- we are assuming 32 threads per warp, this might change in the future
    ADM_CSF_SCALE_LINE(8, 128);      // adm_csf_den_scale_line_kernel_8_128
    ADM_CSF_DEN_S123_LINE(8, 128);   // adm_csf_den_s123_line_kernel_8_128
}
