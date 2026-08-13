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
 * Equivalent of get_spatial_mask_for_index() in cambi.c.
 *
 * The CPU version maintains a cyclic (2*pad_size + 2)-row DP matrix and
 * derives each output via four-corner inclusion-exclusion. That structure
 * exists purely to bound host memory -- what it computes is simply:
 *
 *     mask[i][j] = ( sum of zero_derivative over the (2*pad+1)^2 window
 *                    centred on (i,j), clipped to the image ) > mask_index
 *
 * Reproduced here as a direct box sum. With MASK_FILTER_SIZE == 7 that is 49
 * taps per output, which is cheap enough that a summed-area table is not
 * worth its scan. Integer addition is associative, so the direct form is
 * bit-identical to the DP form rather than merely close.
 *
 * Out-of-image taps contribute zero. That matches the CPU, whose DP matrix is
 * memset to 0 and whose left padding columns [0, pad_size] are never written.
 *
 * Input is the derivative map produced by cambi_derivative_kernel. Strides
 * are in uint16_t ELEMENTS, not bytes.
 */

extern "C" {

__global__ void cambi_spatial_mask_kernel(const uint16_t *deriv, uint16_t *mask,
                                          int width, int height,
                                          int pad_size, unsigned int mask_index,
                                          ptrdiff_t deriv_stride_elems,
                                          ptrdiff_t mask_stride_elems)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    const int r0 = max(row - pad_size, 0);
    const int r1 = min(row + pad_size, height - 1);
    const int c0 = max(col - pad_size, 0);
    const int c1 = min(col + pad_size, width - 1);

    unsigned int sum = 0u;
    for (int r = r0; r <= r1; r++) {
        const uint16_t *d = deriv + (ptrdiff_t)r * deriv_stride_elems;
        for (int c = c0; c <= c1; c++)
            sum += d[c];
    }

    mask[(ptrdiff_t)row * mask_stride_elems + col] =
        (sum > mask_index) ? 1 : 0;
}

} // extern "C"
