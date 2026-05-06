/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

void cambi_increment_range_avx2(uint16_t *arr, int left, int right) {
    __m256i val_vector = _mm256_set1_epi16(1);
    int col = left;
    for (; col + 15 < right; col += 16) {
        __m256i data = _mm256_loadu_si256((__m256i*) &arr[col]);
        data = _mm256_add_epi16(data, val_vector);
        _mm256_storeu_si256((__m256i*) &arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]++;
    } 
}

void cambi_decrement_range_avx2(uint16_t *arr, int left, int right) {
    __m256i val_vector = _mm256_set1_epi16(1);
    int col = left;
    for (; col + 15 < right; col += 16) {
        __m256i data = _mm256_loadu_si256((__m256i*) &arr[col]);
        data = _mm256_sub_epi16(data, val_vector);
        _mm256_storeu_si256((__m256i*) &arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]--;
    } 
}

void calculate_c_values_row_avx2(
    float *c_values, const uint16_t *histograms, const uint16_t *image, const uint16_t *mask,
    int row, int width, ptrdiff_t stride,
    const uint16_t num_diffs, const uint16_t *tvi_thresholds, uint16_t vlt_luma,
    const int *diff_weights, const int *all_diffs,
    const float *reciprocal_lut)
{
    const __m256i col_base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i width_v = _mm256_set1_epi32(width);
    const __m256i num_diffs_v = _mm256_set1_epi32(num_diffs);
    const __m256i vlt_luma_v = _mm256_set1_epi32(vlt_luma);
    const __m256i lo16_mask = _mm256_set1_epi32(0xFFFF);
    const __m256i all_ones = _mm256_set1_epi32(-1);

    const uint16_t *image_row = &image[row * stride];
    const uint16_t *mask_row  = &mask[row * stride];
    float *c_row = &c_values[row * width];

    int col = 0;
    // Vector loop: require at least one extra column past the chunk so the
    // 4-byte vpgatherdd at the worst-case lane never overruns the histogram buffer.
    for (; col + 8 < width; col += 8) {
        // Load 8 mask values, promote to int32, build active mask.
        __m128i mask16 = _mm_loadu_si128((const __m128i*)&mask_row[col]);
        __m256i mask32 = _mm256_cvtepu16_epi32(mask16);
        __m256i mask_active = _mm256_cmpgt_epi32(mask32, _mm256_setzero_si256());

        // Skip the entire chunk if no lane is active.
        if (_mm256_testz_si256(mask_active, mask_active)) {
            _mm256_storeu_ps(&c_row[col], _mm256_setzero_ps());
            continue;
        }

        // value = image[col + lane] + num_diffs
        __m128i img16 = _mm_loadu_si128((const __m128i*)&image_row[col]);
        __m256i value_v = _mm256_add_epi32(_mm256_cvtepu16_epi32(img16), num_diffs_v);

        __m256i col_v = _mm256_add_epi32(_mm256_set1_epi32(col), col_base);

        // p_0 = histograms[value * width + col + lane]
        __m256i p0_idx = _mm256_add_epi32(_mm256_mullo_epi32(value_v, width_v), col_v);
        __m256i p0 = _mm256_and_si256(
            _mm256_i32gather_epi32((const int*)histograms, p0_idx, 2), lo16_mask);

        __m256 c_value = _mm256_setzero_ps();

        for (int d = 0; d < num_diffs; d++) {
            int delta_plus = all_diffs[num_diffs + d + 1];
            int delta_minus = all_diffs[num_diffs - d - 1];
            int weight = diff_weights[d];
            int tvi_thresh = tvi_thresholds[d];

            // pred_a = (value <= tvi_thresh) = NOT (value > tvi_thresh)
            __m256i pred_a_neg = _mm256_cmpgt_epi32(value_v, _mm256_set1_epi32(tvi_thresh));
            __m256i pred_a = _mm256_xor_si256(pred_a_neg, all_ones);

            __m256i value_plus = _mm256_add_epi32(value_v, _mm256_set1_epi32(delta_plus));
            __m256i pred_b = _mm256_cmpgt_epi32(value_plus, vlt_luma_v);
            __m256i predicate = _mm256_and_si256(pred_a, pred_b);

            if (_mm256_testz_si256(predicate, predicate)) continue;

            __m256i value_minus = _mm256_add_epi32(value_v, _mm256_set1_epi32(delta_minus));

            // p_1 / p_2 gathers
            __m256i p1_idx = _mm256_add_epi32(_mm256_mullo_epi32(value_plus, width_v), col_v);
            __m256i p1 = _mm256_and_si256(
                _mm256_i32gather_epi32((const int*)histograms, p1_idx, 2), lo16_mask);

            __m256i p2_idx = _mm256_add_epi32(_mm256_mullo_epi32(value_minus, width_v), col_v);
            __m256i p2 = _mm256_and_si256(
                _mm256_i32gather_epi32((const int*)histograms, p2_idx, 2), lo16_mask);

            __m256i p_max = _mm256_max_epu32(p1, p2);
            __m256i denom = _mm256_add_epi32(p_max, p0);

            // num = (float)(weight * p_0 * p_max), all uint16-bounded so int32 mul fits
            __m256i num_int = _mm256_mullo_epi32(_mm256_set1_epi32(weight),
                                                  _mm256_mullo_epi32(p0, p_max));
            __m256 num_f = _mm256_cvtepi32_ps(num_int);

            // rcp = reciprocal_lut[denom]; LUT is small, hot in L1
            __m256 rcp = _mm256_i32gather_ps(reciprocal_lut, denom, 4);

            __m256 val = _mm256_mul_ps(num_f, rcp);
            // mask off lanes where predicate is false
            val = _mm256_and_ps(val, _mm256_castsi256_ps(predicate));
            c_value = _mm256_max_ps(c_value, val);
        }

        // Apply mask: lanes with mask == 0 keep 0
        c_value = _mm256_and_ps(c_value, _mm256_castsi256_ps(mask_active));
        _mm256_storeu_ps(&c_row[col], c_value);
    }

    // Scalar tail
    for (; col < width; col++) {
        if (mask_row[col]) {
            uint16_t value = (uint16_t)(image_row[col] + num_diffs);
            uint16_t p_0 = histograms[value * width + col];
            float c_v = 0.0f;
            for (int d = 0; d < num_diffs; d++) {
                if ((value <= tvi_thresholds[d]) && ((value + all_diffs[num_diffs + d + 1]) > vlt_luma)) {
                    uint16_t p_1 = histograms[(value + all_diffs[num_diffs + d + 1]) * width + col];
                    uint16_t p_2 = histograms[(value + all_diffs[num_diffs - d - 1]) * width + col];
                    uint16_t p_max = (p_1 > p_2) ? p_1 : p_2;
                    float val = (float)(diff_weights[d] * p_0 * p_max) * reciprocal_lut[p_max + p_0];
                    if (val > c_v) c_v = val;
                }
            }
            c_row[col] = c_v;
        } else {
            c_row[col] = 0.0f;
        }
    }
}

void get_derivative_data_for_row_avx2(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride) {
    // For the last row, we only compute horizontal derivatives
    if (row == height - 1) {
        __m256i ones = _mm256_set1_epi16(1);
        int col = 0;
        for (; col + 15 < width - 1; col += 16) {
            __m256i vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i vals2 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col + 1]);
            __m256i result = _mm256_cmpeq_epi16(vals1, vals2);
            _mm256_storeu_si256((__m256i*) &derivative_buffer[col], _mm256_and_si256(ones, result));
        }
        for (; col < width - 1; col++) {
            derivative_buffer[col] = (image_data[row * stride + col] == image_data[row * stride + col + 1]);
        }
        derivative_buffer[width - 1] = 1;
    }
    else {
        __m256i ones = _mm256_set1_epi16(1);
        int col = 0;
        for (; col + 15 < width - 1; col += 16) {
            __m256i horiz_vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i horiz_vals2 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col + 1]);
            __m256i horiz_result = _mm256_and_si256(ones, _mm256_cmpeq_epi16(horiz_vals1, horiz_vals2));
            __m256i vert_vals1 = _mm256_loadu_si256((__m256i*) &image_data[row * stride + col]);
            __m256i vert_vals2 = _mm256_loadu_si256((__m256i*) &image_data[(row + 1) * stride + col]);
            __m256i vert_result = _mm256_and_si256(ones, _mm256_cmpeq_epi16(vert_vals1, vert_vals2));
            _mm256_storeu_si256((__m256i*) &derivative_buffer[col], _mm256_and_si256(horiz_result, vert_result));
        }
        for (; col < width; col++) {
            bool horizontal_derivative = (col == width - 1 || image_data[row * stride + col] == image_data[row * stride + col + 1]);
            bool vertical_derivative = image_data[row * stride + col] == image_data[(row + 1) * stride + col];
            derivative_buffer[col] =  horizontal_derivative && vertical_derivative;
        }
    }
}