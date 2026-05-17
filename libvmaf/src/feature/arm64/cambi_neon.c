/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include <arm_neon.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

void cambi_increment_range_neon(uint16_t *arr, int left, int right)
{
    uint16x8_t one = vdupq_n_u16(1);
    int col = left;
    for (; col + 8 <= right; col += 8) {
        uint16x8_t data = vld1q_u16(&arr[col]);
        data = vaddq_u16(data, one);
        vst1q_u16(&arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]++;
    }
}

void cambi_decrement_range_neon(uint16_t *arr, int left, int right)
{
    uint16x8_t one = vdupq_n_u16(1);
    int col = left;
    for (; col + 8 <= right; col += 8) {
        uint16x8_t data = vld1q_u16(&arr[col]);
        data = vsubq_u16(data, one);
        vst1q_u16(&arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]--;
    }
}

void get_derivative_data_for_row_neon(const uint16_t *image_data, uint16_t *derivative_buffer,
                                      int width, int height, int row, int stride)
{
    uint16x8_t ones = vdupq_n_u16(1);

    if (row == height - 1) {
        /* Last row: only horizontal derivatives */
        int col = 0;
        for (; col + 8 <= width - 1; col += 8) {
            uint16x8_t vals1 = vld1q_u16(&image_data[row * stride + col]);
            uint16x8_t vals2 = vld1q_u16(&image_data[row * stride + col + 1]);
            /* cmpeq returns 0xFFFF for equal, 0 for not */
            uint16x8_t eq = vceqq_u16(vals1, vals2);
            /* AND with 1 to get 1/0 instead of 0xFFFF/0 */
            vst1q_u16(&derivative_buffer[col], vandq_u16(ones, eq));
        }
        for (; col < width - 1; col++) {
            derivative_buffer[col] =
                (image_data[row * stride + col] == image_data[row * stride + col + 1]);
        }
        derivative_buffer[width - 1] = 1;
    } else {
        /* Interior rows: horizontal AND vertical derivatives */
        int col = 0;
        for (; col + 8 <= width - 1; col += 8) {
            uint16x8_t h1 = vld1q_u16(&image_data[row * stride + col]);
            uint16x8_t h2 = vld1q_u16(&image_data[row * stride + col + 1]);
            uint16x8_t horiz_eq = vandq_u16(ones, vceqq_u16(h1, h2));

            uint16x8_t v1 = vld1q_u16(&image_data[row * stride + col]);
            uint16x8_t v2 = vld1q_u16(&image_data[(row + 1) * stride + col]);
            uint16x8_t vert_eq = vandq_u16(ones, vceqq_u16(v1, v2));

            vst1q_u16(&derivative_buffer[col], vandq_u16(horiz_eq, vert_eq));
        }
        for (; col < width; col++) {
            bool horizontal_derivative =
                (col == width - 1 ||
                 image_data[row * stride + col] == image_data[row * stride + col + 1]);
            bool vertical_derivative =
                image_data[row * stride + col] == image_data[(row + 1) * stride + col];
            derivative_buffer[col] = horizontal_derivative && vertical_derivative;
        }
    }
}

/*
 * calculate_c_values_row_neon — NEON-assisted port of calculate_c_values_row.
 *
 * CAMBI uses scatter/gather on the histogram array (non-contiguous uint16
 * reads indexed by per-pixel luma values).  NEON provides no gather
 * instruction, so the inner loop body is scalar.  The NEON contribution is:
 *
 *   1. A vectorised zero-mask scan: load 8 mask values at a time with
 *      vld1q_u16 and test with vmaxvq_u16.  If the max is zero the entire
 *      8-pixel block is skipped without entering the per-pixel c_value loop.
 *      On typical content where large flat regions have mask == 0 this
 *      eliminates the majority of inner-loop iterations.
 *
 *   2. The per-active-pixel inner loop is the scalar reference verbatim,
 *      which guarantees bit-identical output.  (No float reduction tree,
 *      no lane-widening accumulation — ADR-0452 / ADR-0138/0139 contract.)
 *
 * A pure-scalar loop over the same logic exists in cambi.c as
 * calculate_c_values_row; this function must remain numerically identical
 * to that scalar path for every active pixel.
 */
void calculate_c_values_row_neon(float *c_values, const uint16_t *histograms, const uint16_t *image,
                                 const uint16_t *mask, int row, int width, ptrdiff_t stride,
                                 const uint16_t num_diffs, const uint16_t *tvi_thresholds,
                                 uint16_t vlt_luma, const int *diff_weights, const int *all_diffs,
                                 const float *reciprocal_lut)
{
    int v_lo_signed_sc = (int)vlt_luma - 3 * (int)num_diffs + 1;
    uint16_t v_band_base = v_lo_signed_sc > 0 ? (uint16_t)v_lo_signed_sc : 0;
    uint16_t v_band_size = tvi_thresholds[num_diffs - 1] + 1 - v_band_base;

    const uint16_t *image_row = &image[row * stride];
    const uint16_t *mask_row = &mask[row * stride];
    float *c_row = &c_values[row * width];

    int col = 0;
    /* Fast-skip 8 columns at a time when all masks are zero. */
    for (; col + 8 <= width; col += 8) {
        uint16x8_t mv = vld1q_u16(&mask_row[col]);
        if (vmaxvq_u16(mv) == 0) {
            /* All 8 lanes masked out — write zeros and move on. */
            float32x4_t z = vdupq_n_f32(0.0f);
            vst1q_f32(&c_row[col], z);
            vst1q_f32(&c_row[col + 4], z);
            continue;
        }
        /* At least one active lane: process each pixel individually via the
         * scalar reference to guarantee bit-identical output. */
        for (int k = col; k < col + 8; k++) {
            if (!mask_row[k]) {
                c_row[k] = 0.0f;
                continue;
            }
            uint16_t value = (uint16_t)(image_row[k] + num_diffs);
            int compact_v_signed = (int)image_row[k] - (int)v_band_base;
            if ((unsigned)compact_v_signed >= v_band_size) {
                c_row[k] = 0.0f;
                continue;
            }
            uint16_t compact_v_sc = (uint16_t)compact_v_signed;
            uint16_t p_0 = histograms[compact_v_sc * width + k];
            float c_v = 0.0f;
            for (int d = 0; d < num_diffs; d++) {
                if ((value <= tvi_thresholds[d]) &&
                    ((value + all_diffs[num_diffs + d + 1]) > vlt_luma)) {
                    int idx1 = compact_v_signed + all_diffs[num_diffs + d + 1];
                    int idx2 = compact_v_signed + all_diffs[num_diffs - d - 1];
                    uint16_t p_1 = histograms[idx1 * width + k];
                    uint16_t p_2 = (idx2 >= 0) ? histograms[idx2 * width + k] : 0;
                    uint16_t p_max = (p_1 > p_2) ? p_1 : p_2;
                    float val =
                        (float)(diff_weights[d] * p_0 * p_max) * reciprocal_lut[p_max + p_0];
                    if (val > c_v)
                        c_v = val;
                }
            }
            c_row[k] = c_v;
        }
    }

    /* Scalar tail for remaining columns. */
    for (; col < width; col++) {
        if (mask_row[col]) {
            uint16_t value = (uint16_t)(image_row[col] + num_diffs);
            int compact_v_signed = (int)image_row[col] - (int)v_band_base;
            if ((unsigned)compact_v_signed >= v_band_size) {
                c_row[col] = 0.0f;
                continue;
            }
            uint16_t compact_v_sc = (uint16_t)compact_v_signed;
            uint16_t p_0 = histograms[compact_v_sc * width + col];
            float c_v = 0.0f;
            for (int d = 0; d < num_diffs; d++) {
                if ((value <= tvi_thresholds[d]) &&
                    ((value + all_diffs[num_diffs + d + 1]) > vlt_luma)) {
                    int idx1 = compact_v_signed + all_diffs[num_diffs + d + 1];
                    int idx2 = compact_v_signed + all_diffs[num_diffs - d - 1];
                    uint16_t p_1 = histograms[idx1 * width + col];
                    uint16_t p_2 = (idx2 >= 0) ? histograms[idx2 * width + col] : 0;
                    uint16_t p_max = (p_1 > p_2) ? p_1 : p_2;
                    float val =
                        (float)(diff_weights[d] * p_0 * p_max) * reciprocal_lut[p_max + p_0];
                    if (val > c_v)
                        c_v = val;
                }
            }
            c_row[col] = c_v;
        } else {
            c_row[col] = 0.0f;
        }
    }
}
