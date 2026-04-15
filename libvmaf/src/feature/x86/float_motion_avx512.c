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
#include "float_motion_avx512.h"

static inline __m512 avx512_abs_ps(__m512 x)
{
    const __m512i mask = _mm512_set1_epi32(0x7FFFFFFF);
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), mask));
}

float float_sad_line_avx512(const float *img1, const float *img2, int w)
{
    float accum = 0.0f;
    int j = 0;

    for (; j + 16 <= w; j += 16) {
        __m512 a = _mm512_loadu_ps(img1 + j);
        __m512 b = _mm512_loadu_ps(img2 + j);
        __m512 diff = _mm512_sub_ps(a, b);
        __m512 abs_diff = avx512_abs_ps(diff);

        /*
         * Sequential accumulation in float to match scalar ordering.
         * SIMD is used for vectorised load/subtract/abs; the reduction
         * mirrors the scalar loop so that rounding is bit-identical.
         */
        float tmp[16];
        _mm512_storeu_ps(tmp, abs_diff);
        accum += tmp[0];  accum += tmp[1];
        accum += tmp[2];  accum += tmp[3];
        accum += tmp[4];  accum += tmp[5];
        accum += tmp[6];  accum += tmp[7];
        accum += tmp[8];  accum += tmp[9];
        accum += tmp[10]; accum += tmp[11];
        accum += tmp[12]; accum += tmp[13];
        accum += tmp[14]; accum += tmp[15];
    }

    for (; j < w; j++) {
        float diff = img1[j] - img2[j];
        accum += diff < 0 ? -diff : diff;
    }

    return accum;
}
