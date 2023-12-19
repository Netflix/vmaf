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