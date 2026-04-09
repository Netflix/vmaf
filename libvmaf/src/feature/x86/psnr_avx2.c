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

#include <immintrin.h>
#include <stdint.h>

#include "psnr_avx2.h"

uint32_t psnr_sse_line_8_avx2(const uint8_t *ref, const uint8_t *dis,
                               unsigned w)
{
    __m256i sum = _mm256_setzero_si256();
    unsigned j = 0;

    for (; j + 32 <= w; j += 32) {
        __m256i r = _mm256_loadu_si256((const __m256i *)(ref + j));
        __m256i d = _mm256_loadu_si256((const __m256i *)(dis + j));

        /* Process low 16 bytes: widen uint8 → int16, subtract, square+sum */
        __m256i r_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r));
        __m256i d_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(d));
        __m256i diff_lo = _mm256_sub_epi16(r_lo, d_lo);
        /* madd_epi16: pairs of i16*i16 summed → i32 (8 results from 16 diffs) */
        __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        sum = _mm256_add_epi32(sum, sq_lo);

        /* Process high 16 bytes */
        __m256i r_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r, 1));
        __m256i d_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(d, 1));
        __m256i diff_hi = _mm256_sub_epi16(r_hi, d_hi);
        __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);
        sum = _mm256_add_epi32(sum, sq_hi);
    }

    /* Horizontal sum of 8x uint32 → scalar */
    __m128i lo128 = _mm256_castsi256_si128(sum);
    __m128i hi128 = _mm256_extracti128_si256(sum, 1);
    __m128i s128 = _mm_add_epi32(lo128, hi128);
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0xB1));
    uint32_t result = (uint32_t)_mm_cvtsi128_si32(s128);

    /* Scalar tail */
    for (; j < w; j++) {
        const int16_t e = ref[j] - dis[j];
        result += (uint32_t)(e * e);
    }

    return result;
}

uint64_t psnr_sse_line_16_avx2(const uint16_t *ref, const uint16_t *dis,
                                unsigned w)
{
    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    unsigned j = 0;

    for (; j + 16 <= w; j += 16) {
        __m256i r = _mm256_loadu_si256((const __m256i *)(ref + j));
        __m256i d = _mm256_loadu_si256((const __m256i *)(dis + j));

        /* Low 8 uint16 → int32, subtract, square */
        __m256i rl = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(r));
        __m256i dl = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(d));
        __m256i diff0 = _mm256_sub_epi32(rl, dl);
        __m256i sq0 = _mm256_mullo_epi32(diff0, diff0);

        /* High 8 uint16 → int32 */
        __m256i rh = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(r, 1));
        __m256i dh = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(d, 1));
        __m256i diff1 = _mm256_sub_epi32(rh, dh);
        __m256i sq1 = _mm256_mullo_epi32(diff1, diff1);

        /* Widen 8x uint32 → 4x uint64 each, accumulate */
        sum0 = _mm256_add_epi64(sum0,
            _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sq0)));
        sum1 = _mm256_add_epi64(sum1,
            _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sq0, 1)));
        sum0 = _mm256_add_epi64(sum0,
            _mm256_cvtepu32_epi64(_mm256_castsi256_si128(sq1)));
        sum1 = _mm256_add_epi64(sum1,
            _mm256_cvtepu32_epi64(_mm256_extracti128_si256(sq1, 1)));
    }

    /* Horizontal sum of 4+4 uint64 → scalar */
    __m256i total = _mm256_add_epi64(sum0, sum1);
    __m128i tlo = _mm256_castsi256_si128(total);
    __m128i thi = _mm256_extracti128_si256(total, 1);
    __m128i t128 = _mm_add_epi64(tlo, thi);
    t128 = _mm_add_epi64(t128, _mm_shuffle_epi32(t128, 0x4E));
    uint64_t result = (uint64_t)_mm_cvtsi128_si64(t128);

    /* Scalar tail */
    for (; j < w; j++) {
        const int32_t e = (int32_t)ref[j] - (int32_t)dis[j];
        result += (uint64_t)((uint32_t)(e * e));
    }

    return result;
}
