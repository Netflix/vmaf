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
#include <stddef.h>
#include <stdint.h>

void psnr_sse_8_avx2(const uint8_t *ref, const uint8_t *dis,
                     unsigned w, unsigned h,
                     ptrdiff_t stride_ref, ptrdiff_t stride_dis,
                     uint64_t *sse)
{
    __m256i acc = _mm256_setzero_si256();
    uint64_t total_sse = 0;

    for (unsigned i = 0; i < h; i++) {
        unsigned j = 0;
        __m256i row_acc = _mm256_setzero_si256();

        /* Process 32 uint8_t elements at a time */
        for (; j + 32 <= w; j += 32) {
            __m256i r = _mm256_loadu_si256((const __m256i *)(ref + j));
            __m256i d = _mm256_loadu_si256((const __m256i *)(dis + j));

            /* Zero-extend low 16 bytes to 16-bit */
            __m256i r_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r));
            __m256i d_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(d));

            /* Zero-extend high 16 bytes to 16-bit */
            __m256i r_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r, 1));
            __m256i d_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(d, 1));

            /* Compute signed differences */
            __m256i diff_lo = _mm256_sub_epi16(r_lo, d_lo);
            __m256i diff_hi = _mm256_sub_epi16(r_hi, d_hi);

            /* Square and sum adjacent pairs: int16*int16 -> int32 */
            __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
            __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

            /* Accumulate into 32-bit row accumulator */
            row_acc = _mm256_add_epi32(row_acc, sq_lo);
            row_acc = _mm256_add_epi32(row_acc, sq_hi);
        }

        /* Widen row_acc from 32-bit to 64-bit and add to main accumulator */
        __m256i row_lo = _mm256_unpacklo_epi32(row_acc, _mm256_setzero_si256());
        __m256i row_hi = _mm256_unpackhi_epi32(row_acc, _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, row_lo);
        acc = _mm256_add_epi64(acc, row_hi);

        /* Scalar tail */
        uint32_t sse_inner = 0;
        for (; j < w; j++) {
            const int16_t e = ref[j] - dis[j];
            sse_inner += e * e;
        }
        total_sse += sse_inner;

        ref += stride_ref;
        dis += stride_dis;
    }

    /* Horizontal sum of acc (4x64-bit) */
    __m128i lo128 = _mm256_castsi256_si128(acc);
    __m128i hi128 = _mm256_extracti128_si256(acc, 1);
    __m128i sum128 = _mm_add_epi64(lo128, hi128);
    __m128i sum64 = _mm_add_epi64(sum128, _mm_srli_si128(sum128, 8));
    total_sse += (uint64_t)_mm_extract_epi64(sum64, 0);

    *sse = total_sse;
}
