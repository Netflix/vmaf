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
 * Equivalent of calculate_c_values() in cambi.c.
 *
 * WHAT THE CPU ACTUALLY COMPUTES
 * ------------------------------
 * The CPU maintains `histograms` incrementally: a first pass seeds the top
 * pad_size rows, then three unrolled loops add an arriving row, slide
 * (subtract departing + add arriving), and finally drain. Each pixel's
 * contribution enters via increment_range() over columns [j-pad, j+pad].
 *
 * Tracing all four phases, the invariant when calculate_c_values_row() runs
 * for output row i is simply:
 *
 *   histograms[v][j] = # masked pixels whose (value - v_band_base) == v
 *                      within rows [max(i-pad,0), min(i+pad,height-1)]
 *                       and cols [max(j-pad,0), min(j+pad,width-1)]
 *
 * i.e. a clipped 2-D box count over value bins. The ring buffer and the
 * column scatter are an incremental *encoding* of that, not extra semantics.
 *
 * WHY THIS IS ONE KERNEL AND NOT THREE
 * ------------------------------------
 * A literal port would need the histogram materialised: v_band_size * width
 * uint16 per output row, which is megabytes per row and cannot be held for
 * many rows at once. But c_value_pixel() reads only the bins at
 * compact_v + all_diffs[...], i.e. at most 2*num_diffs+1 bins out of
 * v_band_size -- typically 9 out of several hundred.
 *
 * So each thread accumulates only those few bins, as deltas relative to its
 * own compact value. Nothing global is materialised, there is no scatter, no
 * atomics, and no row-to-row serial dependency.
 *
 * SHARED-MEMORY TILING
 * --------------------
 * Neighbouring output pixels overlap almost entirely, so the block
 * cooperatively stages its window footprint -- blockDim plus a pad_size halo
 * on every side -- into shared memory first. At 65x65 that turns ~4225 global
 * loads per output pixel into ~27.
 *
 * The staged value is the pixel's COMPACT value, or SENTINEL when the pixel
 * is masked out, out of band, or outside the image. Folding all three
 * exclusions into one sentinel means the inner loop is a single compare, and
 * one uint16 array covers what would otherwise be separate image and mask
 * tiles. A real compact value is always < v_band_size (order 1e3), so 0xFFFF
 * can never collide with one.
 *
 * Tiling cannot change the result: the accumulation is integer counting, so
 * it is independent of traversal order.
 *
 * BIT-EXACTNESS
 * -------------
 * Counts are integers, so the box sum is exact regardless of order. The only
 * float work is
 *     (float)(diff_weights[d] * p_0 * p_x) * reciprocal_lut[p_x + p_0]
 * which is one integer product, one conversion, one multiply -- no add, so
 * no FMA contraction is possible and the result is bit-identical to the CPU.
 *
 * Strides for image/mask are in uint16_t ELEMENTS. c_values is width-packed
 * (c_values[row * width + col]), matching the CPU.
 *
 * The caller must supply
 *     (blockDim.x + 2*pad_size) * (blockDim.y + 2*pad_size) * sizeof(uint16_t)
 * bytes of dynamic shared memory.
 */

/* num_diffs is 1 << max_log_contrast. The default max_log_contrast is 2
 * (num_diffs == 4). The host must not dispatch to this kernel above
 * MAX_NUM_DIFFS and should fall back to the CPU path instead. */
#define CAMBI_CUDA_MAX_NUM_DIFFS 16
#define CAMBI_CUDA_MAX_BINS (2 * CAMBI_CUDA_MAX_NUM_DIFFS + 1)

#define CAMBI_CUDA_SENTINEL ((uint16_t) 0xFFFF)

extern "C" {

__global__ void cambi_c_values_kernel(const uint16_t *image, const uint16_t *mask,
                                      float *c_values,
                                      int width, int height,
                                      ptrdiff_t stride_elems,
                                      int pad_size, int num_diffs,
                                      unsigned int v_band_base,
                                      unsigned int v_band_size,
                                      int vlt_luma,
                                      const uint16_t *tvi_for_diff,
                                      const int *diff_weights,
                                      const int *all_diffs,
                                      const float *recip_lut)
{
    extern __shared__ uint16_t s_cv[];

    const int tile_w = blockDim.x + 2 * pad_size;
    const int tile_h = blockDim.y + 2 * pad_size;
    const int origin_x = blockIdx.x * blockDim.x - pad_size;
    const int origin_y = blockIdx.y * blockDim.y - pad_size;

    /* Stage the block's footprint. Masked-out, out-of-band and out-of-image
     * pixels all become SENTINEL so the inner loop tests once. */
    const int nthreads = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < tile_w * tile_h; idx += nthreads) {
        const int ty = idx / tile_w;
        const int tx = idx - ty * tile_w;
        const int gy = origin_y + ty;
        const int gx = origin_x + tx;

        uint16_t v = CAMBI_CUDA_SENTINEL;
        if (gy >= 0 && gy < height && gx >= 0 && gx < width) {
            const ptrdiff_t off = (ptrdiff_t) gy * stride_elems + gx;
            if (mask[off]) {
                const unsigned int cv =
                    (unsigned int)(uint16_t)(image[off] - (uint16_t) v_band_base);
                if (cv < v_band_size)
                    v = (uint16_t) cv;
            }
        }
        s_cv[idx] = v;
    }
    __syncthreads();

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height)
        return;

    float *out = &c_values[(ptrdiff_t) row * width + col];

    /* CPU memsets c_values to 0 and only assigns where the mask is set. */
    if (!mask[(ptrdiff_t) row * stride_elems + col]) {
        *out = 0.0f;
        return;
    }

    /* c_value_pixel() is called with the ADJUSTED value (raw + num_diffs)
     * and v_band_offset_val = v_band_base + num_diffs, so compact_v is
     * raw - v_band_base either way. Kept in adjusted space to match the
     * tvi / vlt_luma comparisons exactly. */
    const int value_adj = (int) image[(ptrdiff_t) row * stride_elems + col] + num_diffs;
    const int v_band_offset_val = (int) v_band_base + num_diffs;
    const int compact_v_signed = value_adj - v_band_offset_val;

    if ((unsigned int) compact_v_signed >= v_band_size) {
        *out = 0.0f;
        return;
    }

    /* Bin counts relative to this pixel's own compact value. Index
     * num_diffs is delta 0 (that is p_0). */
    unsigned short cnt[CAMBI_CUDA_MAX_BINS];
    for (int k = 0; k < 2 * num_diffs + 1; k++)
        cnt[k] = 0;

    /* This thread's window occupies tile rows [threadIdx.y, threadIdx.y+2*pad]
     * and tile cols [threadIdx.x, threadIdx.x+2*pad] by construction of the
     * halo, so no clamping is needed here -- the sentinel already covers
     * everything outside the image. */
    const int win = 2 * pad_size + 1;
    for (int dy = 0; dy < win; dy++) {
        const uint16_t *trow = &s_cv[(threadIdx.y + dy) * tile_w + threadIdx.x];
        for (int dx = 0; dx < win; dx++) {
            const uint16_t cv = trow[dx];
            if (cv == CAMBI_CUDA_SENTINEL)
                continue;
            const int delta = (int) cv - compact_v_signed;
            if (delta >= -num_diffs && delta <= num_diffs)
                cnt[delta + num_diffs]++;
        }
    }

    const unsigned short p_0 = cnt[num_diffs];
    float c_value = 0.0f;

    for (int d = 0; d < num_diffs; d++) {
        const int off1 = all_diffs[num_diffs + d + 1];
        const int off2 = all_diffs[num_diffs - d - 1];

        if ((value_adj <= (int) tvi_for_diff[d]) && ((value_adj + off1) > vlt_luma)) {
            const int idx2 = compact_v_signed + off2;

            /* A bin outside [-num_diffs, num_diffs] cannot have been
             * accumulated; a bin whose absolute index is negative cannot
             * exist, which is the CPU's `idx2 >= 0` guard. Both are zero. */
            unsigned short p_1 = 0;
            if (off1 >= -num_diffs && off1 <= num_diffs)
                p_1 = cnt[off1 + num_diffs];

            unsigned short p_2 = 0;
            if (idx2 >= 0 && off2 >= -num_diffs && off2 <= num_diffs)
                p_2 = cnt[off2 + num_diffs];

            float val;
            if (p_1 > p_2)
                val = (float)(diff_weights[d] * (int) p_0 * (int) p_1) * recip_lut[p_1 + p_0];
            else
                val = (float)(diff_weights[d] * (int) p_0 * (int) p_2) * recip_lut[p_2 + p_0];

            if (val > c_value)
                c_value = val;
        }
    }

    *out = c_value;
}

} // extern "C"
