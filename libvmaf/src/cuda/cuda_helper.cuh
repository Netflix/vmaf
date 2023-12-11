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

#ifndef __CUDA_HELPER_H__
#define __CUDA_HELPER_H__

#include "assert.h"
#include "stdio.h"
#include <cuda.h>
#include <vector_types.h>

#define DIV_ROUND_UP(x, y) (((x) + (y)-1) / (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define CHECK_CUDA(CALL)                                                       \
    do {                                                                       \
        if (CUDA_SUCCESS != CALL) {                                            \
            const char *err_txt;                                               \
            const CUresult err = CALL;                                         \
            cuGetErrorName(err, &err_txt);                                     \
            printf("code: %d; description: %s\n", (int)err, err_txt);          \
            assert(0);                                                         \
        }                                                                      \
    } while (0);

#ifdef __CUDACC__
#include <cstdint>
namespace {

    __forceinline__ __device__ int64_t warp_reduce(int64_t x) {
#pragma unroll
        for (int i = 16; i > 0; i >>= 1) {
            x += int64_t(__shfl_down_sync(0xffffffff, x & 0xffffffff, i)) |
                int64_t(__shfl_down_sync(0xffffffff, x >> 32, i) << 32);
        }
        return x;
    }

    typedef unsigned long long int uint64_cu;
    __forceinline__ __device__ int64_t atomicAdd_int64(int64_t *address,
            int64_t val) {
        return atomicAdd(reinterpret_cast<uint64_cu *>(address),
                static_cast<uint64_cu>(val));
    }
} // namespace
#endif

#endif
