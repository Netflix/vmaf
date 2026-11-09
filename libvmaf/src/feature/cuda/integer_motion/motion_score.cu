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

// Device function that mirrors an idx along its valid [0,sup) range.
// Must be identical to the CPU reference implementation in integer_motion.c:
//   idx < 0     -> -idx
//   idx >= sup  -> 2*sup - idx - 2
__device__ __forceinline__ int mirror(const int idx, const int sup)
{
    if (idx < 0) return -idx;
    if (idx >= sup) return 2 * sup - idx - 2;
    return idx;
}

extern "C" {

__global__ void calculate_motion_score_kernel_8bpc(const VmafPicture src, VmafCudaBuffer cur_raw,
        const VmafCudaBuffer prev_raw, VmafCudaBuffer sad,
        unsigned width, unsigned height,
        ptrdiff_t src_stride, ptrdiff_t raw_stride, unsigned bpc) {

    (void)bpc;
    // Mirrors motion_score_pipeline_8() from integer_motion.c: the filter is
    // applied to the *difference* of the two frames (vertical first, then
    // horizontal), y_round = 1 << 7, >> 8, x_round = 1 << 15, >> 16.
    constexpr int32_t y_round = 128;
    constexpr int32_t x_round = 32768;

    const int x = blockIdx.x*blockDim.x+threadIdx.x;
    const int y = blockIdx.y*blockDim.y+threadIdx.y;

    uint32_t abs_dist = 0u;
    if ( x < width && y < height) {
        const uint8_t *cur = reinterpret_cast<const uint8_t*>(src.data[0]);
        const uint16_t *prev = reinterpret_cast<const uint16_t*>(prev_raw.data);
        uint16_t *out = reinterpret_cast<uint16_t*>(cur_raw.data);

        // Save this frame's raw samples for the next invocation (raw_stride is
        // in elements, the raw plane is always 16 bits per sample).
        out[y*raw_stride + x] = static_cast<uint16_t>(cur[y*src_stride + x]);

        // Vertical filter of the difference for every column of the
        // horizontal window.
        int32_t y_row[5];
#pragma unroll
        for (int xf = 0; xf < 5; ++xf) {
            const int col = mirror(x - radius + xf, width);
            int64_t accum = 0;
#pragma unroll
            for (int yf = 0; yf < 5; ++yf) {
                const int row = mirror(y - radius + yf, height);
                const int32_t diff = static_cast<int32_t>(prev[row*raw_stride + col]) -
                                     static_cast<int32_t>(cur[row*src_stride + col]);
                accum += static_cast<int64_t>(filter_d[yf]) * diff;
            }
            y_row[xf] = static_cast<int32_t>((accum + y_round) >> 8);
        }

        int64_t accum = 0;
#pragma unroll
        for (int xf = 0; xf < 5; ++xf)
            accum += static_cast<int64_t>(filter_d[xf]) * y_row[xf];
        const int32_t val = static_cast<int32_t>((accum + x_round) >> 16);
        abs_dist = static_cast<uint32_t>(abs(val));
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

__global__ void calculate_motion_score_kernel_16bpc(const VmafPicture src, VmafCudaBuffer cur_raw,
        const VmafCudaBuffer prev_raw, VmafCudaBuffer sad,
        unsigned width, unsigned height,
        ptrdiff_t src_stride, ptrdiff_t raw_stride, unsigned bpc) {

    // High bit depth path: mirrors motion_score_pipeline_16() from
    // integer_motion.c - the filter is applied to the *difference* of the two
    // frames, y_round = 1 << (bpc - 1), >> bpc, x_round = 1 << 15, >> 16.
    const unsigned shift_y = bpc;
    const int32_t y_round = static_cast<int32_t>(1u << (bpc - 1));
    constexpr int32_t x_round = 32768;
    // src_stride is in bytes, the 16-bit path indexes uint16_t elements
    const ptrdiff_t src_stride_el = src_stride / 2;

    const int x = blockIdx.x*blockDim.x+threadIdx.x;
    const int y = blockIdx.y*blockDim.y+threadIdx.y;

    uint32_t abs_dist = 0u;
    if ( x < width && y < height) {
        const uint16_t *cur = reinterpret_cast<const uint16_t*>(src.data[0]);
        const uint16_t *prev = reinterpret_cast<const uint16_t*>(prev_raw.data);
        uint16_t *out = reinterpret_cast<uint16_t*>(cur_raw.data);

        out[y*raw_stride + x] = cur[y*src_stride_el + x];

        int32_t y_row[5];
#pragma unroll
        for (int xf = 0; xf < 5; ++xf) {
            const int col = mirror(x - radius + xf, width);
            int64_t accum = 0;
#pragma unroll
            for (int yf = 0; yf < 5; ++yf) {
                const int row = mirror(y - radius + yf, height);
                const int32_t diff = static_cast<int32_t>(prev[row*raw_stride + col]) -
                                     static_cast<int32_t>(cur[row*src_stride_el + col]);
                accum += static_cast<int64_t>(filter_d[yf]) * diff;
            }
            y_row[xf] = static_cast<int32_t>((accum + y_round) >> shift_y);
        }

        int64_t accum = 0;
#pragma unroll
        for (int xf = 0; xf < 5; ++xf)
            accum += static_cast<int64_t>(filter_d[xf]) * y_row[xf];
        const int32_t val = static_cast<int32_t>((accum + x_round) >> 16);
        abs_dist = static_cast<uint32_t>(abs(val));
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
