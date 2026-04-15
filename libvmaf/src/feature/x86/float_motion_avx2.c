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
#include "float_motion_avx2.h"

float float_sad_line_avx2(const float *img1, const float *img2, int w)
{
    const __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
    float accum = 0.0f;
    int j = 0;

    for (; j + 8 <= w; j += 8) {
        __m256 a = _mm256_loadu_ps(img1 + j);
        __m256 b = _mm256_loadu_ps(img2 + j);
        __m256 diff = _mm256_sub_ps(a, b);
        __m256 abs_diff = _mm256_and_ps(diff, _mm256_castsi256_ps(abs_mask));

        /*
         * Sequential accumulation in float to match scalar ordering.
         * SIMD is used for vectorised load/subtract/abs; the reduction
         * mirrors the scalar loop so that rounding is bit-identical.
         */
        float tmp[8];
        _mm256_storeu_ps(tmp, abs_diff);
        accum += tmp[0]; accum += tmp[1];
        accum += tmp[2]; accum += tmp[3];
        accum += tmp[4]; accum += tmp[5];
        accum += tmp[6]; accum += tmp[7];
    }

    for (; j < w; j++) {
        float diff = img1[j] - img2[j];
        accum += diff < 0 ? -diff : diff;
    }

    return accum;
}
