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

/**
 * INTEGER_FILTER_5_s[i] = round(FILTER_5_s[i] * 2^16)
 */
static const uint16_t INTEGER_FILTER_5_s[5] = {

       3571, 16004, 26386, 16004, 3571
};

void integer_convolution_f32_c_s(const uint16_t *filter, int filter_width, const int16_t *src, int16_t *dst, int16_t *tmp, int width, int height, int src_stride, int dst_stride, int inp_size_bits);

int integer_compute_motion(const int16_t *ref, const int16_t *dis, int w, int h, int ref_stride, int dis_stride, double *score);