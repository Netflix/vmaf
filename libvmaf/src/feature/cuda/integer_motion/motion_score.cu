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

#include "cuda_helper.cuh"
#include "cuda/integer_motion_cuda.h"

#include "common.h"

__constant__ uint16_t filter_d[5] = { 3571, 16004, 26386, 16004, 3571 };
__constant__ int filter_width_d = sizeof(filter_d) / sizeof(filter_d[0]);
__constant__ int radius = (sizeof(filter_d) / sizeof(filter_d[0])) / 2;

// Device function that mirrors an idx along its valid [0,sup) range
__device__ __forceinline__ int mirror(const int idx, const int sup)
{
    int out = abs(idx);
    return (out < sup) ? out : (sup - (out - sup + 1));
}

extern "C" {

__global__ void calculate_motion_score_kernel_8bpc(const VmafPicture src, VmafCudaBuffer src_blurred,
        const VmafCudaBuffer prev_blurred, VmafCudaBuffer sad,
        unsigned width, unsigned height,
        ptrdiff_t src_stride, ptrdiff_t blurred_stride) {

    constexpr unsigned shift_var_y = 8u;
    constexpr unsigned add_before_shift_y = 128u;
    constexpr unsigned shift_var_x = 16u;
    constexpr unsigned add_before_shift_x = 32768u;

    const int x = blockIdx.x*blockDim.x+threadIdx.x;
    const int y = blockIdx.y*blockDim.y+threadIdx.y;

    uint32_t abs_dist = 0u;
    if ( x < width && y < height) {
        // Blur src
        uint32_t blurred = 0u;
#pragma unroll
        for (int xf=0; xf < filter_width_d; ++xf) {
            uint32_t blurred_y = 0u;
#pragma unroll
            for (int yf=0; yf < filter_width_d; ++yf) {
                blurred_y += filter_d[yf] * reinterpret_cast<const uint8_t*>(src.data[0] + mirror(y-radius+yf, height) * src.stride[0])[mirror(x-radius+xf, width)];
            }
            blurred += filter_d[xf]*((blurred_y + add_before_shift_y) >> shift_var_y);
        }

        blurred = (blurred + add_before_shift_x) >> shift_var_x;
        reinterpret_cast<uint16_t*>(src_blurred.data + y*blurred_stride)[x] = static_cast<uint16_t>(blurred);
        abs_dist = abs(static_cast<int>(blurred)-static_cast<int>(reinterpret_cast<uint16_t*>(prev_blurred.data + y*blurred_stride)[x]));
    }

    // Warp-reduce abs_dist
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 16);
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 8);
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 4);
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 2);
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 1);
    // Let threads in lane zero add warp-reduced abs_dist atomically to global sad
    const int lane = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (lane == 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(sad.data), static_cast<unsigned long long>(abs_dist));
}

__global__ void calculate_motion_score_kernel_16bpc(const VmafPicture src, VmafCudaBuffer src_blurred,
        const VmafCudaBuffer prev_blurred, VmafCudaBuffer sad,
        unsigned width, unsigned height,
        ptrdiff_t src_stride, ptrdiff_t blurred_stride) {

    unsigned shift_var_y = src.bpc;
    unsigned add_before_shift_y = pow(2, (src.bpc - 1));
    constexpr unsigned shift_var_x = 16u;
    constexpr unsigned add_before_shift_x = 32768u;

    const int x = blockIdx.x*blockDim.x+threadIdx.x;
    const int y = blockIdx.y*blockDim.y+threadIdx.y;

    uint32_t abs_dist = 0u;
    if ( x < width && y < height) {
        // Blur src
        uint32_t blurred = 0u;
#pragma unroll
        for (int xf=0; xf < filter_width_d; ++xf) {
            uint32_t blurred_y = 0u;
#pragma unroll
            for (int yf=0; yf < filter_width_d; ++yf) {
                blurred_y += filter_d[yf] * reinterpret_cast<const uint16_t*>(src.data[0] + mirror(y-radius+yf, height) * src.stride[0])[mirror(x-radius+xf, width)];
            }
            blurred += filter_d[xf]*((blurred_y + add_before_shift_y) >> shift_var_y);
        }

        blurred = (blurred + add_before_shift_x) >> shift_var_x;
        reinterpret_cast<uint16_t*>(src_blurred.data + y*blurred_stride)[x] = static_cast<uint16_t>(blurred);
        abs_dist = abs(static_cast<int>(blurred)-static_cast<int>(reinterpret_cast<uint16_t*>(prev_blurred.data + y*blurred_stride)[x]));
    }

    // Warp-reduce abs_dist
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 16);
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 8);
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 4);
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 2);
    abs_dist += __shfl_down_sync(0xffffffff, abs_dist, 1);
    // Let threads in lane zero add warp-reduced abs_dist atomically to global sad
    const int lane = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (lane == 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(sad.data), static_cast<unsigned long long>(abs_dist));
}

}
