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
// block: one atomic per warp serializes on the single global accumulator,
// one per block is ~32x fewer
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

// Grid-stride over uchar4 vectors so each warp reads full 128-byte segments
// (cuMemAllocPitch aligns every row start); the <=3 tail pixels of each row
// when the width isn't a multiple of 4 are covered by a scalar per-row loop.
// Integer sums are order-independent, so the score is unchanged.
__global__ void psnr_kernel_8bpc(const VmafPicture ref, const VmafPicture dis,
        VmafCudaBuffer sse, unsigned plane, unsigned width, unsigned height)
{
    const uint8_t *rbase = reinterpret_cast<const uint8_t*>(ref.data[plane]);
    const uint8_t *dbase = reinterpret_cast<const uint8_t*>(dis.data[plane]);
    const long vecs = width / 4;
    const long total = vecs * height;
    const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    const long step = (long)gridDim.x * blockDim.x;

    uint64_t sq = 0;
    for (long i = tid; i < total; i += step) {
        const long y = i / vecs;
        const long x = (i - y * vecs) * 4;
        const uchar4 r = *reinterpret_cast<const uchar4*>(
                rbase + y * ref.stride[plane] + x);
        const uchar4 d = *reinterpret_cast<const uchar4*>(
                dbase + y * dis.stride[plane] + x);
        int e = r.x - d.x; sq += e * e;
        e = r.y - d.y; sq += e * e;
        e = r.z - d.z; sq += e * e;
        e = r.w - d.w; sq += e * e;
    }

    const unsigned tail = width & 3;
    if (tail) {
        for (long y = tid; y < height; y += step) {
            const uint8_t *r = rbase + y * ref.stride[plane];
            const uint8_t *d = dbase + y * dis.stride[plane];
            for (unsigned x = width - tail; x < width; x++) {
                const int e = r[x] - d[x];
                sq += e * e;
            }
        }
    }

    block_reduce_add(sq,
            reinterpret_cast<unsigned long long*>(sse.data) + plane);
}

__global__ void psnr_kernel_16bpc(const VmafPicture ref, const VmafPicture dis,
        VmafCudaBuffer sse, unsigned plane, unsigned width, unsigned height)
{
    const uint8_t *rbase = reinterpret_cast<const uint8_t*>(ref.data[plane]);
    const uint8_t *dbase = reinterpret_cast<const uint8_t*>(dis.data[plane]);
    const long vecs = width / 2;
    const long total = vecs * height;
    const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    const long step = (long)gridDim.x * blockDim.x;

    uint64_t sq = 0;
    for (long i = tid; i < total; i += step) {
        const long y = i / vecs;
        const long x = (i - y * vecs) * 4; // byte offset of the ushort2
        const ushort2 r = *reinterpret_cast<const ushort2*>(
                rbase + y * ref.stride[plane] + x);
        const ushort2 d = *reinterpret_cast<const ushort2*>(
                dbase + y * dis.stride[plane] + x);
        int e = r.x - d.x;
        sq += static_cast<uint64_t>(static_cast<int64_t>(e) * e);
        e = r.y - d.y;
        sq += static_cast<uint64_t>(static_cast<int64_t>(e) * e);
    }

    if (width & 1) {
        const unsigned x = width - 1;
        for (long y = tid; y < height; y += step) {
            const int e = reinterpret_cast<const uint16_t*>(
                        rbase + y * ref.stride[plane])[x] -
                    reinterpret_cast<const uint16_t*>(
                        dbase + y * dis.stride[plane])[x];
            sq += static_cast<uint64_t>(static_cast<int64_t>(e) * e);
        }
    }

    block_reduce_add(sq,
            reinterpret_cast<unsigned long long*>(sse.data) + plane);
}

}
