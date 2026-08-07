/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

/*
 * Equivalent of filter_mode() in cambi.c: a separable 3-tap mode filter,
 * horizontal then vertical.
 *
 * The CPU version fuses both passes through a 3-row ring buffer and writes
 * back in place with a one-row lag. Two things about that structure are
 * load-bearing and easy to get wrong when unrolling it:
 *
 *  1. mode3() is SYMMETRIC in its arguments (whichever value is duplicated
 *     wins, regardless of position; all-distinct falls through to min3).
 *     That is why the CPU may read its ring in fixed slot order 0,1,2 rather
 *     than temporal order. So the vertical pass is a plain symmetric 3-tap.
 *
 *  2. The vertical pass only runs for output rows 1 .. height-2 (the CPU
 *     guard is `if (i > 1)`, writing row i-1). Rows 0 and height-1 are never
 *     written, so they keep their ORIGINAL values -- NOT the horizontally
 *     filtered ones, since the horizontal pass writes to the ring buffer and
 *     never to the image. Sourcing those two rows from the horizontal result
 *     is the obvious mistake and it is wrong.
 *
 *     A corollary: when height < 3 the vertical pass never runs at all and
 *     the image is returned completely unmodified.
 *
 * Strides are in uint16_t ELEMENTS, not bytes.
 */

__device__ __forceinline__ uint16_t min3_d(uint16_t a, uint16_t b, uint16_t c)
{
    if (a <= b && a <= c) return a;
    if (b <= c) return b;
    return c;
}

__device__ __forceinline__ uint16_t mode3_d(uint16_t a, uint16_t b, uint16_t c)
{
    if (a == b || a == c) return a;
    if (b == c) return b;
    return min3_d(a, b, c);
}

extern "C" {

/* Pass 1: horizontal 3-tap mode. First and last column are copied through,
 * matching the CPU's explicit assignments outside the j loop. */
__global__ void cambi_filter_mode_h_kernel(const uint16_t *src, uint16_t *tmp,
                                           int width, int height,
                                           ptrdiff_t src_stride_elems,
                                           ptrdiff_t tmp_stride_elems)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    const uint16_t *s = src + (ptrdiff_t)row * src_stride_elems;
    uint16_t out;

    if (col == 0 || col == width - 1)
        out = s[col];
    else
        out = mode3_d(s[col - 1], s[col], s[col + 1]);

    tmp[(ptrdiff_t)row * tmp_stride_elems + col] = out;
}

/* Pass 2: vertical 3-tap mode over the horizontal result, for rows
 * 1 .. height-2 only. Rows 0 and height-1 pass through from SRC. */
__global__ void cambi_filter_mode_v_kernel(const uint16_t *src,
                                           const uint16_t *tmp, uint16_t *dst,
                                           int width, int height,
                                           ptrdiff_t src_stride_elems,
                                           ptrdiff_t tmp_stride_elems,
                                           ptrdiff_t dst_stride_elems)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    uint16_t out;

    if (row == 0 || row == height - 1) {
        out = src[(ptrdiff_t)row * src_stride_elems + col];
    } else {
        out = mode3_d(tmp[(ptrdiff_t)(row - 1) * tmp_stride_elems + col],
                      tmp[(ptrdiff_t)row       * tmp_stride_elems + col],
                      tmp[(ptrdiff_t)(row + 1) * tmp_stride_elems + col]);
    }

    dst[(ptrdiff_t)row * dst_stride_elems + col] = out;
}

} // extern "C"
