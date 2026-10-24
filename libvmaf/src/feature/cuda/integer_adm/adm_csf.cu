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

/*
 * adm_csf() / i4_adm_csf() of the reference implementation. The value of the
 * csf kernels is only computed for the region inside the frame borders, hence
 * only cells with j < right are stored, exactly like in the reference
 * implementation. Everything else (including the vectorized stores) is
 * identical to the reference implementation.
 */

namespace {

template <int cols_per_thread>
static __device__ __forceinline__ void copy_vec_4(const int32_t * __restrict__ in,
        int32_t * __restrict__ out)
{
    static_assert(cols_per_thread % 4 == 0, "implemented only for a multiple of 4");
#pragma unroll
    for (int col = 0; col < cols_per_thread; col += 4) {
        *reinterpret_cast<uint4*>(out + col) = *reinterpret_cast<const uint4*>(in + col);
    }
}

template <int cols_per_thread>
static __device__ __forceinline__ void copy_vec_4(const int16_t * __restrict__ in,
        int16_t * __restrict__ out)
{
    static_assert(cols_per_thread % 4 == 0, "implemented only for a multiple of 4");
#pragma unroll
    for (int col = 0; col < cols_per_thread; col += 4) {
        *reinterpret_cast<ushort4*>(out + col) = *reinterpret_cast<const ushort4*>(in + col);
    }
}

/*
 * scale 0: i_rfactor is a uint16_t Q15 value, the csf value is the fixed-point
 * product shifted back and the flatness value is derived from the csf value.
 */
template <int rows_per_thread, int cols_per_thread>
__device__ __forceinline__ void adm_csf_kernel(AdmBufferCuda buf, int top,
        int bottom, int left, int right, int stride, AdmCsfKernelParamsCuda p)
{
    const int band = blockIdx.z;
    const int16_t *src_ptr =
        (p.measure_aim ? buf.decouple_r : buf.decouple_a).bands[band + 1];
    int16_t *dst_ptr =
        (p.measure_aim ? buf.csf_f : buf.csf_a).bands[band + 1];
    int16_t *flt_ptr =
        (p.measure_aim ? buf.csf_a : buf.csf_f).bands[band + 1];

    const int32_t i_rfactor = (int32_t)p.i_rfactor[band];
    const int32_t shift_dst = (int32_t)p.shift_dst[band];
    const int32_t add_dst = (int32_t)p.add_dst[band];
    const int32_t shift_flt = (int32_t)p.shift_flt[band];
    const int32_t add_flt = (int32_t)p.add_flt[band];
    const int32_t FIX_ONE_BY_30 = 4369; /* (1/30)*2^17 */

    const int y = top + (blockIdx.y * blockDim.y + threadIdx.y) * rows_per_thread;
    const int x = left + (blockIdx.x * blockDim.x + threadIdx.x) * cols_per_thread;

    if (y >= bottom || x >= right)
        return;

    for (int row = 0; row < rows_per_thread; ++row) {
        const int i = y + row;
        if (i >= bottom)
            break;

        const int offset = i * stride;
        const int remaining = right - x;

        if (remaining >= cols_per_thread && ((x & 3) == 0)) {
            __align__(8) int16_t src[cols_per_thread];
            __align__(8) int16_t dst_vec[cols_per_thread];
            __align__(8) int16_t flt_vec[cols_per_thread];

            copy_vec_4<cols_per_thread>(src_ptr + offset + x, src);

            for (int col = 0; col < cols_per_thread; ++col) {
                const int32_t dst_val = i_rfactor * (int32_t)src[col];
                const int16_t i16_dst_val = (int16_t)((dst_val + add_dst) >> shift_dst);
                dst_vec[col] = i16_dst_val;
                flt_vec[col] = (int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val)) + add_flt) >> shift_flt);
            }

            copy_vec_4<cols_per_thread>(dst_vec, dst_ptr + offset + x);
            copy_vec_4<cols_per_thread>(flt_vec, flt_ptr + offset + x);
        } else {
            for (int col = 0; col < cols_per_thread; ++col) {
                const int j = x + col;
                if (j >= right)
                    break;

                const int32_t dst_val = i_rfactor * (int32_t)src_ptr[offset + j];
                const int16_t i16_dst_val = (int16_t)((dst_val + add_dst) >> shift_dst);
                dst_ptr[offset + j] = i16_dst_val;
                flt_ptr[offset + j] =
                    (int16_t)(((FIX_ONE_BY_30 * abs((int32_t)i16_dst_val)) + add_flt) >> shift_flt);
            }
        }
    }
}

/*
 * scales 1..3: i_rfactor is a uint32_t Q32 value, the csf value and the
 * flatness value are 64 bit fixed-point products shifted back to int32.
 */
template <int rows_per_thread, int cols_per_thread>
__device__ __forceinline__ void i4_adm_csf_kernel(AdmBufferCuda buf, int scale,
        int top, int bottom, int left, int right, int stride,
        AdmCsfKernelParamsCuda p)
{
    const int band = blockIdx.z;
    const int32_t *src_ptr =
        (p.measure_aim ? buf.i4_decouple_r : buf.i4_decouple_a).bands[band + 1];
    int32_t *dst_ptr =
        (p.measure_aim ? buf.i4_csf_f : buf.i4_csf_a).bands[band + 1];
    int32_t *flt_ptr =
        (p.measure_aim ? buf.i4_csf_a : buf.i4_csf_f).bands[band + 1];

    const uint32_t i_rfactor = p.i_rfactor[band];
    const uint32_t shift_dst = p.shift_dst[band];     /* 28 */
    const uint32_t shift_flt = p.shift_flt[band];     /* 32 */
    /* the reference implementation stores 1u << 31 in an int32_t */
    const int64_t add_bef_shift_dst = p.add_dst[band];
    const int64_t add_bef_shift_flt = p.add_flt[band];
    const uint32_t FIX_ONE_BY_30 = 143165577; /* (1/30)*2^32 */

    (void)scale;

    const int y = top + (blockIdx.y * blockDim.y + threadIdx.y) * rows_per_thread;
    const int x = left + (blockIdx.x * blockDim.x + threadIdx.x) * cols_per_thread;

    if (y >= bottom || x >= right)
        return;

    for (int row = 0; row < rows_per_thread; ++row) {
        const int i = y + row;
        if (i >= bottom)
            break;

        const int offset = i * stride;
        const int remaining = right - x;

        if (remaining >= cols_per_thread && ((x & 3) == 0)) {
            __align__(16) int32_t src[cols_per_thread];
            __align__(16) int32_t dst_vec[cols_per_thread];
            __align__(16) int32_t flt_vec[cols_per_thread];

            copy_vec_4<cols_per_thread>(src_ptr + offset + x, src);

            for (int col = 0; col < cols_per_thread; ++col) {
                const int32_t dst_val = (int32_t)((((int64_t)i_rfactor * src[col]) +
                            add_bef_shift_dst) >> shift_dst);
                dst_vec[col] = dst_val;
                flt_vec[col] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val)) +
                            add_bef_shift_flt) >> shift_flt);
            }

            copy_vec_4<cols_per_thread>(dst_vec, dst_ptr + offset + x);
            copy_vec_4<cols_per_thread>(flt_vec, flt_ptr + offset + x);
        } else {
            for (int col = 0; col < cols_per_thread; ++col) {
                const int j = x + col;
                if (j >= right)
                    break;

                const int32_t dst_val = (int32_t)((((int64_t)i_rfactor * src_ptr[offset + j]) +
                            add_bef_shift_dst) >> shift_dst);
                dst_ptr[offset + j] = dst_val;
                flt_ptr[offset + j] = (int32_t)((((int64_t)FIX_ONE_BY_30 * abs(dst_val)) +
                            add_bef_shift_flt) >> shift_flt);
            }
        }
    }
}

} // namespace

#define ADM_CSF_KERNEL(rows_per_thread, cols_per_thread)                       \
    __global__ void adm_csf_kernel_##rows_per_thread##_##cols_per_thread (     \
            AdmBufferCuda buf, int top, int bottom, int left,                  \
            int right, int stride,                                             \
            AdmCsfKernelParamsCuda params) {                                   \
        adm_csf_kernel<rows_per_thread, cols_per_thread>(                      \
                buf, top, bottom, left, right, stride, params);                \
    }

#define I4_ADM_CSF_KERNEL(rows_per_thread, cols_per_thread)                    \
    __global__ void i4_adm_csf_kernel_##rows_per_thread##_##cols_per_thread (  \
            AdmBufferCuda buf, int scale, int top, int bottom, int left,       \
            int right, int stride,                                             \
            AdmCsfKernelParamsCuda params) {                                   \
        i4_adm_csf_kernel<rows_per_thread, cols_per_thread>(                   \
                buf, scale, top, bottom, left, right, stride, params);         \
    }

extern "C" {
    ADM_CSF_KERNEL(1, 4);     // adm_csf_kernel_1_4
    I4_ADM_CSF_KERNEL(1, 4);  // i4_adm_csf_kernel_1_4
}
