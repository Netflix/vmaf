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
#include <stdint.h>
#include <stdbool.h>

void cambi_increment_range_avx512(uint16_t *arr, int left, int right) {
    __m512i val_vector = _mm512_set1_epi16(1);
    int col = left;
    for (; col + 31 < right; col += 32) {
        __m512i data = _mm512_loadu_si512((__m512i*) &arr[col]);
        data = _mm512_add_epi16(data, val_vector);
        _mm512_storeu_si512((__m512i*) &arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]++;
    }
}

void cambi_decrement_range_avx512(uint16_t *arr, int left, int right) {
    __m512i val_vector = _mm512_set1_epi16(1);
    int col = left;
    for (; col + 31 < right; col += 32) {
        __m512i data = _mm512_loadu_si512((__m512i*) &arr[col]);
        data = _mm512_sub_epi16(data, val_vector);
        _mm512_storeu_si512((__m512i*) &arr[col], data);
    }
    for (; col < right; col++) {
        arr[col]--;
    }
}

void get_derivative_data_for_row_avx512(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride) {
    if (row == height - 1) {
        __m512i ones = _mm512_set1_epi16(1);
        int col = 0;
        for (; col + 31 < width - 1; col += 32) {
            __m512i vals1 = _mm512_loadu_si512((__m512i*) &image_data[row * stride + col]);
            __m512i vals2 = _mm512_loadu_si512((__m512i*) &image_data[row * stride + col + 1]);
            __mmask32 eq_mask = _mm512_cmpeq_epi16_mask(vals1, vals2);
            _mm512_storeu_si512((__m512i*) &derivative_buffer[col], _mm512_maskz_mov_epi16(eq_mask, ones));
        }
        for (; col < width - 1; col++) {
            derivative_buffer[col] = (image_data[row * stride + col] == image_data[row * stride + col + 1]);
        }
        derivative_buffer[width - 1] = 1;
    }
    else {
        __m512i ones = _mm512_set1_epi16(1);
        int col = 0;
        for (; col + 31 < width - 1; col += 32) {
            __m512i horiz_vals1 = _mm512_loadu_si512((__m512i*) &image_data[row * stride + col]);
            __m512i horiz_vals2 = _mm512_loadu_si512((__m512i*) &image_data[row * stride + col + 1]);
            __mmask32 horiz_mask = _mm512_cmpeq_epi16_mask(horiz_vals1, horiz_vals2);
            __m512i vert_vals1 = _mm512_loadu_si512((__m512i*) &image_data[row * stride + col]);
            __m512i vert_vals2 = _mm512_loadu_si512((__m512i*) &image_data[(row + 1) * stride + col]);
            __mmask32 vert_mask = _mm512_cmpeq_epi16_mask(vert_vals1, vert_vals2);
            __mmask32 combined = horiz_mask & vert_mask;
            _mm512_storeu_si512((__m512i*) &derivative_buffer[col], _mm512_maskz_mov_epi16(combined, ones));
        }
        for (; col < width; col++) {
            bool horizontal_derivative = (col == width - 1 || image_data[row * stride + col] == image_data[row * stride + col + 1]);
            bool vertical_derivative = image_data[row * stride + col] == image_data[(row + 1) * stride + col];
            derivative_buffer[col] = horizontal_derivative && vertical_derivative;
        }
    }
}
