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
#include "ansnr_avx2.h"

void ansnr_mse_line_avx2(const float *ref, const float *dis,
                          float *sig_accum, float *noise_accum, int w)
{
    /* Accumulate in double to eliminate SIMD lane-reorder precision loss */
    __m256d sig_dsum0 = _mm256_setzero_pd();
    __m256d sig_dsum1 = _mm256_setzero_pd();
    __m256d noise_dsum0 = _mm256_setzero_pd();
    __m256d noise_dsum1 = _mm256_setzero_pd();
    int j = 0;

    for (; j + 8 <= w; j += 8) {
        __m256 r = _mm256_loadu_ps(ref + j);
        __m256 d = _mm256_loadu_ps(dis + j);
        __m256 diff = _mm256_sub_ps(r, d);
        __m256 sig_val = _mm256_mul_ps(r, r);
        __m256 noise_val = _mm256_mul_ps(diff, diff);

        __m128 sig_lo = _mm256_castps256_ps128(sig_val);
        __m128 sig_hi = _mm256_extractf128_ps(sig_val, 1);
        sig_dsum0 = _mm256_add_pd(sig_dsum0, _mm256_cvtps_pd(sig_lo));
        sig_dsum1 = _mm256_add_pd(sig_dsum1, _mm256_cvtps_pd(sig_hi));

        __m128 noise_lo = _mm256_castps256_ps128(noise_val);
        __m128 noise_hi = _mm256_extractf128_ps(noise_val, 1);
        noise_dsum0 = _mm256_add_pd(noise_dsum0, _mm256_cvtps_pd(noise_lo));
        noise_dsum1 = _mm256_add_pd(noise_dsum1, _mm256_cvtps_pd(noise_hi));
    }

    /* Horizontal reduce to double, then truncate to float for per-row sum */
    __m256d sig_total = _mm256_add_pd(sig_dsum0, sig_dsum1);
    __m128d sig_tlo = _mm256_castpd256_pd128(sig_total);
    __m128d sig_thi = _mm256_extractf128_pd(sig_total, 1);
    __m128d sig_s = _mm_add_pd(sig_tlo, sig_thi);
    sig_s = _mm_add_sd(sig_s, _mm_unpackhi_pd(sig_s, sig_s));
    float sig_result = (float)_mm_cvtsd_f64(sig_s);

    __m256d noise_total = _mm256_add_pd(noise_dsum0, noise_dsum1);
    __m128d noise_tlo = _mm256_castpd256_pd128(noise_total);
    __m128d noise_thi = _mm256_extractf128_pd(noise_total, 1);
    __m128d noise_s = _mm_add_pd(noise_tlo, noise_thi);
    noise_s = _mm_add_sd(noise_s, _mm_unpackhi_pd(noise_s, noise_s));
    float noise_result = (float)_mm_cvtsd_f64(noise_s);

    /* Scalar tail */
    for (; j < w; j++) {
        float r = ref[j];
        float d = dis[j];
        float diff = r - d;
        sig_result += r * r;
        noise_result += diff * diff;
    }

    *sig_accum += sig_result;
    *noise_accum += noise_result;
}
