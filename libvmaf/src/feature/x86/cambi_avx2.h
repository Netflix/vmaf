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

#ifndef X86_AVX2_CAMBI_H_
#define X86_AVX2_CAMBI_H_

#include <stddef.h>
#include <stdint.h>

void cambi_increment_range_avx2(uint16_t *arr, int left, int right);

void cambi_decrement_range_avx2(uint16_t *arr, int left, int right);

void get_derivative_data_for_row_avx2(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride);

#endif /* X86_AVX2_CAMBI_H_ */
