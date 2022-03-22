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
