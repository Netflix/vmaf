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
#include "float_psnr_avx512.h"

double float_psnr_noise_line_avx512(const float *ref, const float *dis, int w)
{
    /* Accumulate in double to eliminate SIMD lane-reorder precision loss */
    __m512d dsum0 = _mm512_setzero_pd();
    __m512d dsum1 = _mm512_setzero_pd();
    int j = 0;

    for (; j + 16 <= w; j += 16) {
        __m512 r = _mm512_loadu_ps(ref + j);
        __m512 d = _mm512_loadu_ps(dis + j);
        __m512 diff = _mm512_sub_ps(r, d);
        __m512 sq = _mm512_mul_ps(diff, diff);

        __m256 lo = _mm512_castps512_ps256(sq);
        __m256 hi = _mm512_extractf32x8_ps(sq, 1);
        dsum0 = _mm512_add_pd(dsum0, _mm512_cvtps_pd(lo));
        dsum1 = _mm512_add_pd(dsum1, _mm512_cvtps_pd(hi));
    }

    double result = _mm512_reduce_add_pd(_mm512_add_pd(dsum0, dsum1));

    for (; j < w; j++) {
        float diff = ref[j] - dis[j];
        result += (double)(diff * diff);
    }

    return result;
}
