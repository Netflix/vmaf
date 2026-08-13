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
 * Device side of cambi_preprocessing(): the decimate-and-convert-to-10-bit
 * step and the anti-dithering filter. Doing these on the device removes the
 * device -> host -> device round trip the extractor otherwise needs before
 * the pyramid can start.
 *
 * WHY THE RESAMPLE TAKES INDEX TABLES
 * -----------------------------------
 * The CPU walks the source with an ACCUMULATED float:
 *
 *     float x = start_x;
 *     for (j ...) { ori_x = (int)(x + 0.5); ...; x += ratio_x; }
 *
 * Computing start_x + j * ratio_x instead rounds differently, so a kernel
 * that derives the index arithmetically is not bit-exact. The host therefore
 * runs the same accumulation once (out_w + out_h steps, trivial) and uploads
 * ori_x[] / ori_y[]; the kernel only gathers. When input and output sizes
 * match, the host fills the identity mapping, which reproduces the CPU's
 * same-size fast path exactly.
 *
 * WHY ANTI-DITHERING CAN BE OUT OF PLACE
 * --------------------------------------
 * The CPU filter is in-place, but it writes (i,j) and only ever reads (i,j+1),
 * (i+1,j) and (i+1,j+1) -- all strictly later in raster order. So no write is
 * ever read back, and an out-of-place kernel is bit-identical.
 *
 * Note the CPU's loops leave the bottom-right pixel (height-1, width-1)
 * untouched: the last-column case sits inside the `i < height-1` loop and the
 * last-row loop stops at `j < width-1`. The kernel copies it through.
 *
 * Strides: source stride is in BYTES for the 8-bit variant and in uint16_t
 * ELEMENTS for the 16-bit one, matching the CPU's own handling. Destination
 * strides are always in elements.
 */

extern "C" {

/* bpc <= 8. shift_left is 10 - bpc. */
__global__ void cambi_preprocess_u8_kernel(const unsigned char *src, uint16_t *dst,
                                           int out_w, int out_h,
                                           ptrdiff_t src_stride_bytes,
                                           ptrdiff_t dst_stride_elems,
                                           int shift_left,
                                           const int *ori_x, const int *ori_y)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= out_w || i >= out_h)
        return;

    const unsigned char v = src[(ptrdiff_t) ori_y[i] * src_stride_bytes + ori_x[j]];
    dst[(ptrdiff_t) i * dst_stride_elems + j] = (uint16_t)((unsigned) v << shift_left);
}

/*
 * bpc >= 9. A single signed shift covers both directions:
 *   shift > 0  -> (v + rounding) >> shift    (bpc > 10)
 *   shift <= 0 -> v << (-shift)              (bpc <= 10; bpc == 10 is a copy)
 */
__global__ void cambi_preprocess_u16_kernel(const uint16_t *src, uint16_t *dst,
                                            int out_w, int out_h,
                                            ptrdiff_t src_stride_elems,
                                            ptrdiff_t dst_stride_elems,
                                            int shift, int rounding,
                                            const int *ori_x, const int *ori_y)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= out_w || i >= out_h)
        return;

    const unsigned v = src[(ptrdiff_t) ori_y[i] * src_stride_elems + ori_x[j]];
    const unsigned out = (shift > 0) ? ((v + (unsigned) rounding) >> shift)
                                     : (v << (-shift));
    dst[(ptrdiff_t) i * dst_stride_elems + j] = (uint16_t) out;
}

/* Out-of-place equivalent of anti_dithering_filter(). */
__global__ void cambi_antidither_kernel(const uint16_t *src, uint16_t *dst,
                                        int width, int height,
                                        ptrdiff_t src_stride_elems,
                                        ptrdiff_t dst_stride_elems)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= width || i >= height)
        return;

    const uint16_t *s = src + (ptrdiff_t) i * src_stride_elems;
    const uint16_t *sd = src + (ptrdiff_t)(i + 1) * src_stride_elems;
    unsigned out;

    if (i < height - 1 && j < width - 1)
        out = ((unsigned) s[j] + s[j + 1] + sd[j] + sd[j + 1]) >> 2;
    else if (i < height - 1)              /* last column */
        out = ((unsigned) s[j] + sd[j]) >> 1;
    else if (j < width - 1)               /* last row */
        out = ((unsigned) s[j] + s[j + 1]) >> 1;
    else                                  /* (height-1, width-1): untouched */
        out = s[j];

    dst[(ptrdiff_t) i * dst_stride_elems + j] = (uint16_t) out;
}

} // extern "C"
