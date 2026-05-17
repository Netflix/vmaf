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

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

void cambi_increment_range_avx512(uint16_t *arr, int left, int right)
{
    __m512i val_vector = _mm512_set1_epi16(1);
    int col = left;
    for (; col + 31 < right; col += 32) {
        __m512i data = _mm512_loadu_si512((__m512i *)&arr[col]);
        data = _mm512_add_epi16(data, val_vector);
        _mm512_storeu_si512((__m512i *)&arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]++;
    }
}

void cambi_decrement_range_avx512(uint16_t *arr, int left, int right)
{
    __m512i val_vector = _mm512_set1_epi16(1);
    int col = left;
    for (; col + 31 < right; col += 32) {
        __m512i data = _mm512_loadu_si512((__m512i *)&arr[col]);
        data = _mm512_sub_epi16(data, val_vector);
        _mm512_storeu_si512((__m512i *)&arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]--;
    }
}

void get_derivative_data_for_row_avx512(const uint16_t *image_data, uint16_t *derivative_buffer,
                                        int width, int height, int row, int stride)
{
    if (row == height - 1) {
        __m512i ones = _mm512_set1_epi16(1);
        int col = 0;
        for (; col + 31 < width - 1; col += 32) {
            __m512i vals1 = _mm512_loadu_si512((__m512i *)&image_data[row * stride + col]);
            __m512i vals2 = _mm512_loadu_si512((__m512i *)&image_data[row * stride + col + 1]);
            __mmask32 eq_mask = _mm512_cmpeq_epi16_mask(vals1, vals2);
            _mm512_storeu_si512((__m512i *)&derivative_buffer[col],
                                _mm512_maskz_mov_epi16(eq_mask, ones));
        }
        for (; col < width - 1; col++) {
            derivative_buffer[col] =
                (image_data[row * stride + col] == image_data[row * stride + col + 1]);
        }
        derivative_buffer[width - 1] = 1;
    } else {
        __m512i ones = _mm512_set1_epi16(1);
        int col = 0;
        for (; col + 31 < width - 1; col += 32) {
            __m512i horiz_vals1 = _mm512_loadu_si512((__m512i *)&image_data[row * stride + col]);
            __m512i horiz_vals2 =
                _mm512_loadu_si512((__m512i *)&image_data[row * stride + col + 1]);
            __mmask32 horiz_mask = _mm512_cmpeq_epi16_mask(horiz_vals1, horiz_vals2);
            __m512i vert_vals1 = _mm512_loadu_si512((__m512i *)&image_data[row * stride + col]);
            __m512i vert_vals2 =
                _mm512_loadu_si512((__m512i *)&image_data[(row + 1) * stride + col]);
            __mmask32 vert_mask = _mm512_cmpeq_epi16_mask(vert_vals1, vert_vals2);
            __mmask32 combined = horiz_mask & vert_mask;
            _mm512_storeu_si512((__m512i *)&derivative_buffer[col],
                                _mm512_maskz_mov_epi16(combined, ones));
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
 * calculate_c_values_row_avx512 — 16-lane wide port of calculate_c_values_row_avx2.
 *
 * CAMBI is an integer pipeline: all histogram counts are uint16 and the
 * per-lane arithmetic is integer until the final float multiply by
 * reciprocal_lut.  No float reduction trees exist, so AVX-512 output is
 * bit-identical to the scalar reference when run on the same histogram state.
 * (ADR-0452 / SIMD-bitexact contract per ADR-0138/0139.)
 *
 * Design notes vs the AVX2 sibling:
 *  - 16-lane i32 gather (scale=2 → vpgatherdps with 4-byte gather on 2-byte
 *    elements is not directly supported; use _mm512_i32gather_epi32 scale=2
 *    + mask with lo16, matching the AVX2 pattern).
 *  - AVX-512 mask registers replace the AVX2 testz+continue shortcut: we use
 *    _mm512_mask_storeu_ps to conditionally write lanes; an all-zero __mmask16
 *    causes no writes without the testz branch.
 *  - The inner predicate loop uses __mmask16 throughout to avoid the
 *    _mm512_testz fallback.
 *  - Gather: _mm512_i32gather_epi32 with scale=2 (uint16 elements at byte
 *    offsets index*2). The AVX-512 version carries no imm8 scale restriction;
 *    scale=2 is legal and saves the shift-then-add offset arithmetic.
 */
void calculate_c_values_row_avx512(float *c_values, const uint16_t *histograms,
                                   const uint16_t *image, const uint16_t *mask, int row, int width,
                                   ptrdiff_t stride, const uint16_t num_diffs,
                                   const uint16_t *tvi_thresholds, uint16_t vlt_luma,
                                   const int *diff_weights, const int *all_diffs,
                                   const float *reciprocal_lut)
{
    int v_lo_signed_sc = (int)vlt_luma - 3 * (int)num_diffs + 1;
    uint16_t v_band_base = v_lo_signed_sc > 0 ? (uint16_t)v_lo_signed_sc : 0;
    uint16_t v_band_size = tvi_thresholds[num_diffs - 1] + 1 - v_band_base;

    /* 16-lane lane-index vector: lane i holds value i. */
    const __m512i col_base = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i width_v = _mm512_set1_epi32(width);
    const __m512i num_diffs_v = _mm512_set1_epi32(num_diffs);
    const __m512i vlt_luma_v = _mm512_set1_epi32(vlt_luma);
    const __m512i lo16_mask = _mm512_set1_epi32(0xFFFF);
    const __m512i band_offset_v = _mm512_set1_epi32((int)num_diffs + (int)v_band_base);
    const __m512i band_max_v = _mm512_set1_epi32((int)v_band_size - 1);
    const __m512i zero = _mm512_setzero_si512();

    const uint16_t *image_row = &image[row * stride];
    const uint16_t *mask_row = &mask[row * stride];
    float *c_row = &c_values[row * width];

    int col = 0;
    for (; col + 16 < width; col += 16) {
        /* Load 16 mask values, promote to i32. */
        __m256i mask16 = _mm256_loadu_si256((const __m256i *)&mask_row[col]);
        __m512i mask32 = _mm512_cvtepu16_epi32(mask16);
        /* Active lane mask: mask32[i] != 0. */
        __mmask16 mask_active = _mm512_cmpneq_epi32_mask(mask32, zero);

        /* Skip chunk if no lane is active: mask store of zeros is a no-op. */
        if (mask_active == 0) {
            _mm512_mask_storeu_ps(&c_row[col], (__mmask16)0xFFFF, _mm512_setzero_ps());
            continue;
        }

        /* value = image[col + lane] + num_diffs (adjusted space). */
        __m256i img16 = _mm256_loadu_si256((const __m256i *)&image_row[col]);
        __m512i value_v = _mm512_add_epi32(_mm512_cvtepu16_epi32(img16), num_diffs_v);

        /* compact_v = value_v - band_offset, clamped to [0, band_max]. */
        __m512i compact_v = _mm512_sub_epi32(value_v, band_offset_v);
        compact_v = _mm512_max_epi32(compact_v, zero);
        compact_v = _mm512_min_epi32(compact_v, band_max_v);

        __m512i col_v = _mm512_add_epi32(_mm512_set1_epi32(col), col_base);

        /* p_0 gather: histograms[compact_v * width + col + lane]. */
        __m512i p0_idx = _mm512_add_epi32(_mm512_mullo_epi32(compact_v, width_v), col_v);
        __m512i p0 =
            _mm512_and_si512(_mm512_i32gather_epi32(p0_idx, (const int *)histograms, 2), lo16_mask);

        __m512 c_value = _mm512_setzero_ps();

        for (int d = 0; d < num_diffs; d++) {
            int delta_plus = all_diffs[num_diffs + d + 1];
            int delta_minus = all_diffs[num_diffs - d - 1];
            int weight = diff_weights[d];
            int tvi_thresh = tvi_thresholds[d];

            /* pred_a: value <= tvi_thresh — compare gives mask of lanes satisfying. */
            __mmask16 pred_a = _mm512_cmple_epi32_mask(value_v, _mm512_set1_epi32(tvi_thresh));

            /* pred_b: (value + delta_plus) > vlt_luma. */
            __m512i value_plus = _mm512_add_epi32(value_v, _mm512_set1_epi32(delta_plus));
            __mmask16 pred_b = _mm512_cmpgt_epi32_mask(value_plus, vlt_luma_v);

            __mmask16 predicate = pred_a & pred_b & mask_active;
            if (predicate == 0)
                continue;

            /* compact_plus clamped for safe gather; compact_minus with OOB tracking. */
            __m512i compact_plus_raw = _mm512_add_epi32(compact_v, _mm512_set1_epi32(delta_plus));
            __m512i compact_plus = _mm512_min_epi32(compact_plus_raw, band_max_v);

            __m512i compact_minus_raw = _mm512_add_epi32(compact_v, _mm512_set1_epi32(delta_minus));
            __mmask16 p2_inbounds =
                _mm512_cmpgt_epi32_mask(compact_minus_raw, _mm512_set1_epi32(-1));
            __m512i compact_minus = _mm512_max_epi32(compact_minus_raw, zero);

            /* p1 / p2 gathers. */
            __m512i p1_idx = _mm512_add_epi32(_mm512_mullo_epi32(compact_plus, width_v), col_v);
            __m512i p1 = _mm512_and_si512(
                _mm512_i32gather_epi32(p1_idx, (const int *)histograms, 2), lo16_mask);

            __m512i p2_idx = _mm512_add_epi32(_mm512_mullo_epi32(compact_minus, width_v), col_v);
            __m512i p2 = _mm512_and_si512(
                _mm512_i32gather_epi32(p2_idx, (const int *)histograms, 2), lo16_mask);
            /* Zero OOB lanes in p2. */
            p2 = _mm512_maskz_mov_epi32(p2_inbounds, p2);

            __m512i p_max = _mm512_max_epu32(p1, p2);
            __m512i denom = _mm512_add_epi32(p_max, p0);

            /* num = weight * p0 * p_max; all values bounded by uint16 so i32 mul is safe. */
            __m512i num_int =
                _mm512_mullo_epi32(_mm512_set1_epi32(weight), _mm512_mullo_epi32(p0, p_max));
            __m512 num_f = _mm512_cvtepi32_ps(num_int);

            /* rcp = reciprocal_lut[denom]; LUT is hot in L1. */
            __m512 rcp = _mm512_i32gather_ps(denom, reciprocal_lut, 4);

            __m512 val = _mm512_mul_ps(num_f, rcp);
            /* Mask off lanes where predicate is false. */
            val = _mm512_maskz_mov_ps(predicate, val);
            c_value = _mm512_max_ps(c_value, val);
        }

        /* Apply active-lane mask and store 16 floats. */
        c_value = _mm512_maskz_mov_ps(mask_active, c_value);
        _mm512_storeu_ps(&c_row[col], c_value);
    }

    /* Scalar tail (fewer than 16 columns remaining, same as AVX2 tail). */
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
