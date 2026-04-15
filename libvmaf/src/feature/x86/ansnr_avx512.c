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
#include "ansnr_avx512.h"

void ansnr_mse_line_avx512(const float *ref, const float *dis,
                            float *sig_accum, float *noise_accum, int w)
{
    /* Accumulate in double to eliminate SIMD lane-reorder precision loss */
    __m512d sig_dsum0 = _mm512_setzero_pd();
    __m512d sig_dsum1 = _mm512_setzero_pd();
    __m512d noise_dsum0 = _mm512_setzero_pd();
    __m512d noise_dsum1 = _mm512_setzero_pd();
    int j = 0;

    for (; j + 16 <= w; j += 16) {
        __m512 r = _mm512_loadu_ps(ref + j);
        __m512 d = _mm512_loadu_ps(dis + j);
        __m512 diff = _mm512_sub_ps(r, d);
        __m512 sig_val = _mm512_mul_ps(r, r);
        __m512 noise_val = _mm512_mul_ps(diff, diff);

        __m256 sig_lo = _mm512_castps512_ps256(sig_val);
        __m256 sig_hi = _mm512_extractf32x8_ps(sig_val, 1);
        sig_dsum0 = _mm512_add_pd(sig_dsum0, _mm512_cvtps_pd(sig_lo));
        sig_dsum1 = _mm512_add_pd(sig_dsum1, _mm512_cvtps_pd(sig_hi));

        __m256 noise_lo = _mm512_castps512_ps256(noise_val);
        __m256 noise_hi = _mm512_extractf32x8_ps(noise_val, 1);
        noise_dsum0 = _mm512_add_pd(noise_dsum0, _mm512_cvtps_pd(noise_lo));
        noise_dsum1 = _mm512_add_pd(noise_dsum1, _mm512_cvtps_pd(noise_hi));
    }

    float sig_result = (float)_mm512_reduce_add_pd(
        _mm512_add_pd(sig_dsum0, sig_dsum1));
    float noise_result = (float)_mm512_reduce_add_pd(
        _mm512_add_pd(noise_dsum0, noise_dsum1));

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
