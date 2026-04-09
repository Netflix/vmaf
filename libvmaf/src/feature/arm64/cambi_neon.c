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

void get_derivative_data_for_row_neon(const uint16_t *image_data,
                                      uint16_t *derivative_buffer,
                                      int width, int height, int row,
                                      int stride)
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
                (image_data[row * stride + col] ==
                 image_data[row * stride + col + 1]);
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

            vst1q_u16(&derivative_buffer[col],
                       vandq_u16(horiz_eq, vert_eq));
        }
        for (; col < width; col++) {
            bool horizontal_derivative =
                (col == width - 1 ||
                 image_data[row * stride + col] ==
                     image_data[row * stride + col + 1]);
            bool vertical_derivative =
                image_data[row * stride + col] ==
                image_data[(row + 1) * stride + col];
            derivative_buffer[col] =
                horizontal_derivative && vertical_derivative;
        }
    }
}
