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
 * Equivalent of decimate() in cambi.c: 2x subsampling, taking the top-left
 * sample of each 2x2 block.
 *
 * NOTE: the CPU version is IN-PLACE and relies on sequential ordering --
 * output (i,j) reads input (2i,2j), which is always at or ahead of the write
 * position, so a serial loop never clobbers a value it still needs. A kernel
 * has no such ordering guarantee, so this writes to a SEPARATE destination.
 *
 * width/height are the OUTPUT (already-halved) dimensions, matching the CPU
 * function's parameters. Strides are in uint16_t ELEMENTS, not bytes.
 */

extern "C" {

__global__ void cambi_decimate_kernel(const uint16_t *src, uint16_t *dst,
                                      int width, int height,
                                      ptrdiff_t src_stride_elems,
                                      ptrdiff_t dst_stride_elems)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    dst[row * dst_stride_elems + col] =
        src[(row << 1) * src_stride_elems + (col << 1)];
}

} // extern "C"
