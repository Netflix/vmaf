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
#include "float_psnr_avx2.h"

double float_psnr_noise_line_avx2(const float *ref, const float *dis, int w)
{
    /* Accumulate in double to eliminate SIMD lane-reorder precision loss */
    __m256d dsum0 = _mm256_setzero_pd();
    __m256d dsum1 = _mm256_setzero_pd();
    int j = 0;

    for (; j + 8 <= w; j += 8) {
        __m256 r = _mm256_loadu_ps(ref + j);
        __m256 d = _mm256_loadu_ps(dis + j);
        __m256 diff = _mm256_sub_ps(r, d);
        __m256 sq = _mm256_mul_ps(diff, diff);

        __m128 lo = _mm256_castps256_ps128(sq);
        __m128 hi = _mm256_extractf128_ps(sq, 1);
        dsum0 = _mm256_add_pd(dsum0, _mm256_cvtps_pd(lo));
        dsum1 = _mm256_add_pd(dsum1, _mm256_cvtps_pd(hi));
    }

    __m256d total = _mm256_add_pd(dsum0, dsum1);
    __m128d tlo = _mm256_castpd256_pd128(total);
    __m128d thi = _mm256_extractf128_pd(total, 1);
    __m128d s = _mm_add_pd(tlo, thi);
    s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    double result = _mm_cvtsd_f64(s);

    for (; j < w; j++) {
        float diff = ref[j] - dis[j];
        result += (double)(diff * diff);
    }

    return result;
}
