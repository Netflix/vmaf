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

/*
 * Replacement for cuda_runtime.h when the kernels are compiled with
 * clang -nocudainc (enable_nvcc=false). The CUDA toolkit headers do not
 * support a MinGW-w64 host: crt/math_functions.hpp takes a pre-2013-MSVC
 * code path there that conflicts with the CUDA wrappers shipped by clang.
 * Those wrappers are self-contained, so this file includes them the way
 * __clang_cuda_runtime_wrapper.h does, minus the toolkit parts,
 * and adds the kernel-facing definitions that live in toolkit headers (vector
 * types, warp shuffles, atomicAdd). Force-included on the clang path only;
 * nvcc never sees it.
 */
#ifndef VMAF_CUDA_RUNTIME_COMPAT_H
#define VMAF_CUDA_RUNTIME_COMPAT_H

#define __host__        __attribute__((host))
#define __device__      __attribute__((device))
#define __global__      __attribute__((global))
#define __shared__      __attribute__((shared))
#define __constant__    __attribute__((constant))
#define __align__(n)    __attribute__((aligned(n)))
#define __forceinline__ __inline__ __attribute__((always_inline))
/* nvcc always defines this; older libstdc++ (up to 13) checks it to leave
 * out its __float128 code, which the device target cannot compile. */
#define __CUDACC__ 1

/* The __device__ math overloads must be declared before <math.h>: libstdc++
 * has constexpr (hence implicitly __host__ __device__) templates such as
 * pow(int, int) that must call the device versions. */
#include <__clang_cuda_math_forward_declares.h>
#include <limits.h>
#include <math.h>
/* The wrappers select code paths on the toolkit version; anything >= 9.2
 * gives the current ones. Scoped to these includes since
 * ffnvcodec/dynlink_cuda.h skips its typedefs when it sees CUDA_VERSION. */
#define CUDA_VERSION 9020
#include <__clang_cuda_builtin_vars.h>      /* threadIdx, blockIdx, blockDim */
#include <__clang_cuda_libdevice_declares.h>
#include <__clang_cuda_device_functions.h>  /* __clz, __log2f, __float2uint_ru, ... */
#include <__clang_cuda_math.h>              /* abs, min, max, log2f, cosf, pow, ... */
#include <__clang_cuda_cmath.h>             /* integer and std:: overloads */
#undef CUDA_VERSION

#define DEVICE_FN static __device__ __forceinline__

/* The wrappers only have the int overloads, the rest are toolkit-side. */
DEVICE_FN double max(double a, double b) { return fmax(a, b); }
DEVICE_FN double min(double a, double b) { return fmin(a, b); }

struct __align__(4)  short2  { short x, y; };
struct __align__(8)  uint2   { unsigned x, y; };
struct __align__(8)  ushort4 { unsigned short x, y, z, w; };
struct __align__(16) int4    { int x, y, z, w; };
struct __align__(16) uint4   { unsigned x, y, z, w; };
DEVICE_FN short2 make_short2(short x, short y) { short2 r = {x, y}; return r; }
DEVICE_FN int4 make_int4(int x, int y, int z, int w) { int4 r = {x, y, z, w}; return r; }

/* __clang_cuda_intrinsics.h has these too but includes a toolkit header for
 * sm_70+. Same semantics as the toolkit versions. */
DEVICE_FN int __shfl_down_sync(unsigned mask, int v, unsigned delta, int width = 32)
{
    return __nvvm_shfl_sync_down_i32(mask, v, delta, ((32 - width) << 8) | 0x1f);
}
DEVICE_FN unsigned __shfl_down_sync(unsigned mask, unsigned v, unsigned delta, int width = 32)
{
    return (unsigned)__shfl_down_sync(mask, (int)v, delta, width);
}
DEVICE_FN long long __shfl_down_sync(unsigned mask, long long v, unsigned delta, int width = 32)
{
    int lo = (int)(unsigned long long)v;
    int hi = (int)((unsigned long long)v >> 32);
    lo = __shfl_down_sync(mask, lo, delta, width);
    hi = __shfl_down_sync(mask, hi, delta, width);
    return (long long)(((unsigned long long)(unsigned)hi << 32) | (unsigned)lo);
}
DEVICE_FN long __shfl_down_sync(unsigned mask, long v, unsigned delta, int width = 32)
{
    /* int64_t is long on LP64 and long long on LLP64 */
    if (sizeof(long) == sizeof(int))
        return (long)__shfl_down_sync(mask, (int)v, delta, width);
    else
        return (long)__shfl_down_sync(mask, (long long)v, delta, width);
}

DEVICE_FN unsigned long long atomicAdd(unsigned long long *p, unsigned long long v)
{
    return __ullAtomicAdd(p, v);
}

#endif
