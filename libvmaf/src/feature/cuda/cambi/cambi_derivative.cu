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
 * Whole-frame equivalent of get_derivative_data_for_row() in cambi.c.
 *
 * A pixel has zero_derivative = 1 iff it equals both its right and its bottom
 * neighbour. Pixels on the last column / last row count as equal (edges are
 * treated as "equal" by the reference implementation).
 *
 * STRIDES ARE IN uint16_t ELEMENTS, NOT BYTES. The caller is responsible for
 * the >> 1 conversion from VmafPicture::stride, exactly as the CPU path does
 * (`ptrdiff_t stride = image->stride[0] >> 1`). Passing a byte stride here
 * silently produces wrong results rather than failing -- see issue #1566 for
 * what that class of bug looks like in practice.
 */

extern "C" {

__global__ void cambi_derivative_kernel(const uint16_t *image,
                                        uint16_t *derivative,
                                        int width, int height,
                                        ptrdiff_t src_stride_elems,
                                        ptrdiff_t dst_stride_elems)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    const uint16_t v = image[row * src_stride_elems + col];

    const bool horizontal_derivative =
        (col == width - 1) || (v == image[row * src_stride_elems + col + 1]);
    const bool vertical_derivative =
        (row == height - 1) || (v == image[(row + 1) * src_stride_elems + col]);

    derivative[row * dst_stride_elems + col] =
        (horizontal_derivative && vertical_derivative) ? 1 : 0;
}

} // extern "C"
