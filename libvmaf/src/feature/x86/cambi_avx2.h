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

struct VmafPicture;
void decimate_avx2(struct VmafPicture *image, unsigned width, unsigned height);
void filter_mode_avx2(const struct VmafPicture *image, int width, int height, uint16_t *buffer);

void cambi_increment_range_avx2(uint16_t *arr, int left, int right);

void cambi_decrement_range_avx2(uint16_t *arr, int left, int right);

void get_derivative_data_for_row_avx2(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride);

void calculate_c_values_row_avx2(
    float *c_values, const uint16_t *histograms, const uint16_t *image, const uint16_t *mask,
    int row, int width, ptrdiff_t stride,
    const uint16_t num_diffs, const uint16_t *tvi_thresholds, uint16_t vlt_luma,
    const int *diff_weights, const int *all_diffs,
    const float *reciprocal_lut);

void calculate_c_values_avx2(struct VmafPicture *pic, const struct VmafPicture *mask_pic,
                             float *c_values, uint16_t *histograms, uint16_t window_size,
                             const uint16_t num_diffs, const uint16_t *tvi_for_diff, uint16_t vlt_luma,
                             const int *diff_weights, const int *all_diffs, int width, int height);

#endif /* X86_AVX2_CAMBI_H_ */
