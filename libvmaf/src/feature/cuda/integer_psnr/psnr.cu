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

#include "common.h"

// Block-reduce a per-thread value and let thread 0 issue ONE atomicAdd per
// block: one atomic per warp serializes on the single global accumulator
// (~2.6 ms/frame on 4K yuv444p16), one per block is ~32x fewer
__device__ __forceinline__ void block_reduce_add(uint64_t v,
        unsigned long long *accum)
{
    __shared__ uint64_t warp_sums[8]; // 256 threads = 8 warps

    const int t = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        v += uint64_t(__shfl_down_sync(0xffffffff, uint32_t(v), i)) |
             (uint64_t(__shfl_down_sync(0xffffffff, uint32_t(v >> 32), i)) << 32);
    }
    if ((t % 32) == 0)
        warp_sums[t / 32] = v;
    __syncthreads();
    if (t == 0) {
        uint64_t sum = 0;
#pragma unroll
        for (int i = 0; i < 8; i++)
            sum += warp_sums[i];
        if (sum)
            atomicAdd(accum, static_cast<unsigned long long>(sum));
    }
}

extern "C" {

__global__ void psnr_kernel_8bpc(const VmafPicture ref, const VmafPicture dis,
        VmafCudaBuffer sse, unsigned plane, unsigned width, unsigned height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t sq = 0u;
    if (x < width && y < height) {
        const int r = (reinterpret_cast<const uint8_t*>(ref.data[plane]) +
                y * ref.stride[plane])[x];
        const int d = (reinterpret_cast<const uint8_t*>(dis.data[plane]) +
                y * dis.stride[plane])[x];
        const int e = r - d;
        sq = e * e;
    }

    block_reduce_add(sq,
            reinterpret_cast<unsigned long long*>(sse.data) + plane);
}

__global__ void psnr_kernel_16bpc(const VmafPicture ref, const VmafPicture dis,
        VmafCudaBuffer sse, unsigned plane, unsigned width, unsigned height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    uint64_t sq = 0;
    if (x < width && y < height) {
        const int r = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(ref.data[plane]) +
                y * ref.stride[plane])[x];
        const int d = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(dis.data[plane]) +
                y * dis.stride[plane])[x];
        const int e = r - d;
        sq = static_cast<uint64_t>(static_cast<int64_t>(e) * e);
    }

    block_reduce_add(sq,
            reinterpret_cast<unsigned long long*>(sse.data) + plane);
}

}
