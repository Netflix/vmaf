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

/*
 * One warp is assigned to a single output row of a single band. Each lane
 * accumulates its share of the row, the warp reduces it, and lane 0 applies
 * the per row rounding shift exactly once, mirroring the reference
 * implementation:
 *
 *   accum_inner = sum over the row
 *   accum      += (accum_inner + add_shift_accum) >> shift_accum
 *
 * The addition of the row contributions is done with integer atomicAdd, hence
 * the order of the rows does not influence the result.
 */

extern "C" {

__global__ void adm_csf_den_scale_row_kernel(AdmBufferCuda buf, AdmCsfDenKernelParamsCuda p, uint64_t *accum)
{
    const int band = blockIdx.z + 1;
    const int row = p.top + blockIdx.y;

    const int16_t *src_ptr;
    switch (band) {
    case 1:
        src_ptr = buf.ref_dwt2.band_h + (int64_t)row * p.src_stride;
        break;
    case 2:
        src_ptr = buf.ref_dwt2.band_v + (int64_t)row * p.src_stride;
        break;
    default:
        src_ptr = buf.ref_dwt2.band_d + (int64_t)row * p.src_stride;
        break;
    }

    uint64_cu temp_value = 0;
    for (int j = p.left + threadIdx.x; j < p.right; j += blockDim.x) {
        uint32_t t = (uint32_t)abs((int32_t)src_ptr[j]);
        temp_value += ((uint64_t)t * t) * t;
    }

    temp_value = warp_reduce(temp_value);

    if (threadIdx.x == 0) {
        atomicAdd((uint64_cu *)&accum[band - 1],
                (uint64_cu)((temp_value + p.add_shift_accum) >> p.shift_accum));
    }
}

__global__ void adm_csf_den_s123_row_kernel(AdmBufferCuda buf, AdmCsfDenKernelParamsCuda p, uint64_t *accum)
{
    const int band = blockIdx.z + 1;
    const int row = p.top + blockIdx.y;

    const int32_t *src_ptr;
    switch (band) {
    case 1:
        src_ptr = buf.i4_ref_dwt2.band_h + (int64_t)row * p.src_stride;
        break;
    case 2:
        src_ptr = buf.i4_ref_dwt2.band_v + (int64_t)row * p.src_stride;
        break;
    default:
        src_ptr = buf.i4_ref_dwt2.band_d + (int64_t)row * p.src_stride;
        break;
    }

    uint64_cu temp_value = 0;
    for (int j = p.left + threadIdx.x; j < p.right; j += blockDim.x) {
        uint32_t t = (uint32_t)abs(src_ptr[j]);
        temp_value += ((((((uint64_t)t * t) + p.add_shift_sq) >> p.shift_sq) * t) +
                p.add_shift_cub) >> p.shift_cub;
    }

    temp_value = warp_reduce(temp_value);

    if (threadIdx.x == 0) {
        atomicAdd((uint64_cu *)&accum[band - 1],
                (uint64_cu)((temp_value + p.add_shift_accum) >> p.shift_accum));
    }
}

} // extern "C"
