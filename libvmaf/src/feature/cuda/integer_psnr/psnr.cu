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

    // Warp-reduce sq; max per-thread value is 255^2 so a full warp fits u32
    sq += __shfl_down_sync(0xffffffff, sq, 16);
    sq += __shfl_down_sync(0xffffffff, sq, 8);
    sq += __shfl_down_sync(0xffffffff, sq, 4);
    sq += __shfl_down_sync(0xffffffff, sq, 2);
    sq += __shfl_down_sync(0xffffffff, sq, 1);
    // Let threads in lane zero add warp-reduced sq atomically to global sse
    const int lane = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (lane == 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(sse.data) + plane,
                static_cast<unsigned long long>(sq));
}

__global__ void psnr_kernel_16bpc(const VmafPicture ref, const VmafPicture dis,
        VmafCudaBuffer sse, unsigned plane, unsigned width, unsigned height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int64_t sq = 0;
    if (x < width && y < height) {
        const int r = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(ref.data[plane]) +
                y * ref.stride[plane])[x];
        const int d = reinterpret_cast<const uint16_t*>(
                reinterpret_cast<const uint8_t*>(dis.data[plane]) +
                y * dis.stride[plane])[x];
        const int e = r - d;
        sq = static_cast<int64_t>(e) * e;
    }

    // 65535^2 overflows u32 once warp-accumulated, reduce in 64-bit
    sq = warp_reduce(sq);
    const int lane = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (lane == 0)
        atomicAdd_int64(reinterpret_cast<int64_t*>(sse.data) + plane, sq);
}

}
