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

#include "psnr_avx512.h"

uint32_t psnr_sse_line_8_avx512(const uint8_t *ref, const uint8_t *dis,
                                 unsigned w)
{
    __m512i sum = _mm512_setzero_si512();
    unsigned j = 0;

    for (; j + 64 <= w; j += 64) {
        __m512i r = _mm512_loadu_si512((const __m512i *)(ref + j));
        __m512i d = _mm512_loadu_si512((const __m512i *)(dis + j));

        /* Process 4 chunks of 16 bytes each: widen u8→i16, diff, madd */
        __m256i r256_0 = _mm512_castsi512_si256(r);
        __m256i d256_0 = _mm512_castsi512_si256(d);
        __m256i r256_1 = _mm512_extracti64x4_epi64(r, 1);
        __m256i d256_1 = _mm512_extracti64x4_epi64(d, 1);

        /* Chunk 0: bytes 0-15 */
        __m512i r16_0 = _mm512_cvtepu8_epi16(r256_0);
        __m512i d16_0 = _mm512_cvtepu8_epi16(d256_0);
        __m512i diff_0 = _mm512_sub_epi16(r16_0, d16_0);
        __m512i sq_0 = _mm512_madd_epi16(diff_0, diff_0);
        sum = _mm512_add_epi32(sum, sq_0);

        /* Chunk 1: bytes 32-63 */
        __m512i r16_1 = _mm512_cvtepu8_epi16(r256_1);
        __m512i d16_1 = _mm512_cvtepu8_epi16(d256_1);
        __m512i diff_1 = _mm512_sub_epi16(r16_1, d16_1);
        __m512i sq_1 = _mm512_madd_epi16(diff_1, diff_1);
        sum = _mm512_add_epi32(sum, sq_1);
    }

    /* Reduce 16x uint32 → scalar */
    uint32_t result = _mm512_reduce_add_epi32(sum);

    /* Scalar tail */
    for (; j < w; j++) {
        const int16_t e = ref[j] - dis[j];
        result += (uint32_t)(e * e);
    }

    return result;
}

uint64_t psnr_sse_line_16_avx512(const uint16_t *ref, const uint16_t *dis,
                                  unsigned w)
{
    __m512i sum0 = _mm512_setzero_si512();
    __m512i sum1 = _mm512_setzero_si512();
    unsigned j = 0;

    for (; j + 32 <= w; j += 32) {
        __m512i r = _mm512_loadu_si512((const __m512i *)(ref + j));
        __m512i d = _mm512_loadu_si512((const __m512i *)(dis + j));

        /* Widen 32 uint16 → 2×16 int32, subtract, square */
        __m256i r_lo = _mm512_castsi512_si256(r);
        __m256i r_hi = _mm512_extracti64x4_epi64(r, 1);
        __m256i d_lo = _mm512_castsi512_si256(d);
        __m256i d_hi = _mm512_extracti64x4_epi64(d, 1);

        __m512i rl = _mm512_cvtepu16_epi32(r_lo);
        __m512i dl = _mm512_cvtepu16_epi32(d_lo);
        __m512i diff0 = _mm512_sub_epi32(rl, dl);
        __m512i sq0 = _mm512_mullo_epi32(diff0, diff0);

        __m512i rh = _mm512_cvtepu16_epi32(r_hi);
        __m512i dh = _mm512_cvtepu16_epi32(d_hi);
        __m512i diff1 = _mm512_sub_epi32(rh, dh);
        __m512i sq1 = _mm512_mullo_epi32(diff1, diff1);

        /* Accumulate to uint64: widen 16x u32 → 8x u64 each */
        __m256i sq0_lo = _mm512_castsi512_si256(sq0);
        __m256i sq0_hi = _mm512_extracti64x4_epi64(sq0, 1);
        sum0 = _mm512_add_epi64(sum0, _mm512_cvtepu32_epi64(sq0_lo));
        sum1 = _mm512_add_epi64(sum1, _mm512_cvtepu32_epi64(sq0_hi));

        __m256i sq1_lo = _mm512_castsi512_si256(sq1);
        __m256i sq1_hi = _mm512_extracti64x4_epi64(sq1, 1);
        sum0 = _mm512_add_epi64(sum0, _mm512_cvtepu32_epi64(sq1_lo));
        sum1 = _mm512_add_epi64(sum1, _mm512_cvtepu32_epi64(sq1_hi));
    }

    /* Reduce 8+8 uint64 → scalar */
    uint64_t result = _mm512_reduce_add_epi64(_mm512_add_epi64(sum0, sum1));

    /* Scalar tail */
    for (; j < w; j++) {
        const int32_t e = (int32_t)ref[j] - (int32_t)dis[j];
        result += (uint64_t)((uint32_t)(e * e));
    }

    return result;
}
