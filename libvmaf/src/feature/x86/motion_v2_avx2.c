/**
 *
 *  Copyright 2016-2025 Netflix, Inc.
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
#include <stdlib.h>

#include "feature/integer_motion.h"

static inline int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 1;
    return idx;
}

// Emulate arithmetic right shift of int64 by 16 in AVX2.
// AVX2 lacks srai_epi64; this uses the blend trick:
//   low dwords come from logical shift, high dwords from arithmetic shift.
static inline __m256i srai_epi64_16(__m256i v)
{
    __m256i lo = _mm256_srli_epi64(v, 16);
    __m256i hi = _mm256_srai_epi32(v, 16);
    return _mm256_blend_epi32(lo, hi, 0xAA);
}

// SIMD phase 2: x_conv + abs + SAD for one row of int32 y_row.
// Processes 8 int32 columns at a time via mullo_epi32 + int64 accumulation.
static inline uint32_t
x_conv_row_sad_avx2(const int32_t *y_row, unsigned w)
{
    const __m256i g0 = _mm256_set1_epi32(3571);
    const __m256i g1 = _mm256_set1_epi32(16004);
    const __m256i g2 = _mm256_set1_epi32(26386);
    const __m256i round64 = _mm256_set1_epi64x(1 << 15);
    const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);

    uint32_t row_sad = 0;

    // Scalar left edge (columns 0, 1) — mirror boundary
    unsigned j;
    for (j = 0; j < 2 && j < w; j++) {
        int64_t accum = 0;
        for (int k = 0; k < 5; k++) {
            int col = mirror((int)j - 2 + k, (int)w);
            accum += (int64_t)filter[k] * y_row[col];
        }
        int32_t val = (int32_t)((accum + (1 << 15)) >> 16);
        row_sad += abs(val);
    }

    // SIMD middle: need y_row[j-2]..y_row[j+9], so j+10 <= w
    __m256i sad_acc = _mm256_setzero_si256();
    for (; j + 10 <= w; j += 8) {
        __m256i y0 = _mm256_loadu_si256((__m256i*)(y_row + j - 2));
        __m256i y1 = _mm256_loadu_si256((__m256i*)(y_row + j - 1));
        __m256i y2 = _mm256_loadu_si256((__m256i*)(y_row + j));
        __m256i y3 = _mm256_loadu_si256((__m256i*)(y_row + j + 1));
        __m256i y4 = _mm256_loadu_si256((__m256i*)(y_row + j + 2));

        // Each product fits in int32
        __m256i p0 = _mm256_mullo_epi32(y0, g0);
        __m256i p1 = _mm256_mullo_epi32(y1, g1);
        __m256i p2 = _mm256_mullo_epi32(y2, g2);
        __m256i p3 = _mm256_mullo_epi32(y3, g1);
        __m256i p4 = _mm256_mullo_epi32(y4, g0);

        // Safe pairs that fit in int32
        __m256i s04 = _mm256_add_epi32(p0, p4);
        __m256i s13 = _mm256_add_epi32(p1, p3);

        // Widen to int64 and accumulate (lo 4 elements)
        __m256i acc_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(s04));
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(s13)));
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p2)));

        // hi 4 elements
        __m256i acc_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(s04, 1));
        acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(s13, 1)));
        acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p2, 1)));

        // Round and arithmetic right shift >>16
        acc_lo = srai_epi64_16(_mm256_add_epi64(acc_lo, round64));
        acc_hi = srai_epi64_16(_mm256_add_epi64(acc_hi, round64));

        // Pack int64 -> int32 (gather low 32 bits of each 64-bit lane)
        __m128i res_lo = _mm256_castsi256_si128(
            _mm256_permutevar8x32_epi32(acc_lo, perm_idx));
        __m128i res_hi = _mm256_castsi256_si128(
            _mm256_permutevar8x32_epi32(acc_hi, perm_idx));

        // Combine into 8 x int32, abs, accumulate
        __m256i result = _mm256_inserti128_si256(
            _mm256_castsi128_si256(res_lo), res_hi, 1);
        __m256i abs_result = _mm256_abs_epi32(result);

        sad_acc = _mm256_add_epi32(sad_acc, abs_result);
    }

    // Horizontal reduction of sad_acc (8 x int32 -> scalar)
    __m128i lo128 = _mm256_castsi256_si128(sad_acc);
    __m128i hi128 = _mm256_extracti128_si256(sad_acc, 1);
    __m128i sum128 = _mm_add_epi32(lo128, hi128);
    sum128 = _mm_add_epi32(sum128,
                _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128,
                _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1)));
    row_sad += (uint32_t)_mm_cvtsi128_si32(sum128);

    // Scalar right edge + tail
    for (; j < w; j++) {
        int64_t accum = 0;
        for (int k = 0; k < 5; k++) {
            int col = mirror((int)j - 2 + k, (int)w);
            accum += (int64_t)filter[k] * y_row[col];
        }
        int32_t val = (int32_t)((accum + (1 << 15)) >> 16);
        row_sad += abs(val);
    }

    return row_sad;
}

uint64_t motion_score_pipeline_16_avx2(const uint8_t *prev_u8, ptrdiff_t prev_stride,
                                       const uint8_t *cur_u8, ptrdiff_t cur_stride,
                                       int32_t *y_row, unsigned w, unsigned h,
                                       unsigned bpc)
{
    const uint16_t *prev = (const uint16_t *)prev_u8;
    const uint16_t *cur = (const uint16_t *)cur_u8;
    const ptrdiff_t p_stride = prev_stride / 2;
    const ptrdiff_t c_stride = cur_stride / 2;

    const __m256i g0 = _mm256_set1_epi32(3571);
    const __m256i g1 = _mm256_set1_epi32(16004);
    const __m256i g2 = _mm256_set1_epi32(26386);
    const __m256i round64 = _mm256_set1_epi64x(1 << (bpc - 1));
    const __m256i bpc_vec = _mm256_set1_epi64x(bpc);
    const __m256i perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);

    uint64_t sad = 0;

    for (unsigned i = 0; i < h; i++) {
        const uint16_t *pp[5], *cp[5];
        for (int k = 0; k < 5; k++) {
            int r = mirror((int)i - 2 + k, (int)h);
            pp[k] = prev + r * p_stride;
            cp[k] = cur + r * c_stride;
        }

        // Phase 1: diff + y_conv -> y_row (8 pixels at a time, int64 accum)
        unsigned j;
        __m256i nz_acc = _mm256_setzero_si256();
        for (j = 0; j + 8 <= w; j += 8) {
            __m256i d0 = _mm256_sub_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pp[0] + j))),
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(cp[0] + j))));
            __m256i d1 = _mm256_sub_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pp[1] + j))),
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(cp[1] + j))));
            __m256i d2 = _mm256_sub_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pp[2] + j))),
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(cp[2] + j))));
            __m256i d3 = _mm256_sub_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pp[3] + j))),
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(cp[3] + j))));
            __m256i d4 = _mm256_sub_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pp[4] + j))),
                _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(cp[4] + j))));

            __m256i prod0 = _mm256_mullo_epi32(d0, g0);
            __m256i prod1 = _mm256_mullo_epi32(d1, g1);
            __m256i prod2 = _mm256_mullo_epi32(d2, g2);
            __m256i prod3 = _mm256_mullo_epi32(d3, g1);
            __m256i prod4 = _mm256_mullo_epi32(d4, g0);

            __m256i acc_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod0));
            acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod1)));
            acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod2)));
            acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod3)));
            acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod4)));

            __m256i acc_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod0, 1));
            acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod1, 1)));
            acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod2, 1)));
            acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod3, 1)));
            acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod4, 1)));

            acc_lo = _mm256_srlv_epi64(_mm256_add_epi64(acc_lo, round64), bpc_vec);
            acc_hi = _mm256_srlv_epi64(_mm256_add_epi64(acc_hi, round64), bpc_vec);

            __m128i res_lo = _mm256_castsi256_si128(
                _mm256_permutevar8x32_epi32(acc_lo, perm_idx));
            __m128i res_hi = _mm256_castsi256_si128(
                _mm256_permutevar8x32_epi32(acc_hi, perm_idx));

            __m256i result = _mm256_inserti128_si256(
                _mm256_castsi128_si256(res_lo), res_hi, 1);
            _mm256_storeu_si256((__m256i*)(y_row + j), result);
            nz_acc = _mm256_or_si256(nz_acc, result);
        }

        // Scalar tail for phase 1
        int32_t nz_tail = 0;
        for (; j < w; j++) {
            int64_t accum = 0;
            for (int k = 0; k < 5; k++) {
                int32_t diff = pp[k][j] - cp[k][j];
                accum += (int64_t)filter[k] * diff;
            }
            y_row[j] = (int32_t)((accum + (1 << (bpc - 1))) >> bpc);
            nz_tail |= y_row[j];
        }

        if (_mm256_testz_si256(nz_acc, nz_acc) && !nz_tail) continue;

        // Phase 2: SIMD x_conv + abs + accumulate
        sad += x_conv_row_sad_avx2(y_row, w);
    }

    return sad;
}

uint64_t motion_score_pipeline_8_avx2(const uint8_t *prev, ptrdiff_t prev_stride,
                                      const uint8_t *cur, ptrdiff_t cur_stride,
                                      int32_t *y_row, unsigned w, unsigned h,
                                      unsigned bpc)
{
    (void)bpc;
    const __m256i f0 = _mm256_set1_epi16(3571);
    const __m256i f1 = _mm256_set1_epi16(16004);
    const __m256i f2 = _mm256_set1_epi16(26386);
    const __m256i round8 = _mm256_set1_epi32(1 << 7);

    uint64_t sad = 0;

    for (unsigned i = 0; i < h; i++) {
        const uint8_t *p[5], *c[5];
        for (int k = 0; k < 5; k++) {
            int r = mirror((int)i - 2 + k, (int)h);
            p[k] = prev + r * prev_stride;
            c[k] = cur + r * cur_stride;
        }

        // Phase 1: diff + y_conv -> y_row (16 columns at a time, shift >>8)
        unsigned j;
        __m256i nz_acc = _mm256_setzero_si256();
        for (j = 0; j + 16 <= w; j += 16) {
            __m256i d0 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(p[0] + j))),
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(c[0] + j))));
            __m256i d1 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(p[1] + j))),
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(c[1] + j))));
            __m256i d2 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(p[2] + j))),
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(c[2] + j))));
            __m256i d3 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(p[3] + j))),
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(c[3] + j))));
            __m256i d4 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(p[4] + j))),
                _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(c[4] + j))));

            __m256i lo = _mm256_mullo_epi16(d0, f0);
            __m256i hi = _mm256_mulhi_epi16(d0, f0);
            __m256i acc_lo = _mm256_unpacklo_epi16(lo, hi);
            __m256i acc_hi = _mm256_unpackhi_epi16(lo, hi);

            lo = _mm256_mullo_epi16(d1, f1);
            hi = _mm256_mulhi_epi16(d1, f1);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(lo, hi));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(lo, hi));

            lo = _mm256_mullo_epi16(d2, f2);
            hi = _mm256_mulhi_epi16(d2, f2);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(lo, hi));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(lo, hi));

            lo = _mm256_mullo_epi16(d3, f1);
            hi = _mm256_mulhi_epi16(d3, f1);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(lo, hi));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(lo, hi));

            lo = _mm256_mullo_epi16(d4, f0);
            hi = _mm256_mulhi_epi16(d4, f0);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(lo, hi));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(lo, hi));

            acc_lo = _mm256_srai_epi32(_mm256_add_epi32(acc_lo, round8), 8);
            acc_hi = _mm256_srai_epi32(_mm256_add_epi32(acc_hi, round8), 8);

            __m256i cols_0_7  = _mm256_permute2x128_si256(acc_lo, acc_hi, 0x20);
            __m256i cols_8_15 = _mm256_permute2x128_si256(acc_lo, acc_hi, 0x31);
            _mm256_storeu_si256((__m256i*)(y_row + j), cols_0_7);
            _mm256_storeu_si256((__m256i*)(y_row + j + 8), cols_8_15);
            nz_acc = _mm256_or_si256(nz_acc, _mm256_or_si256(cols_0_7, cols_8_15));
        }

        // Scalar tail for phase 1
        int32_t nz_tail = 0;
        for (; j < w; j++) {
            int32_t accum = 0;
            for (int k = 0; k < 5; k++) {
                int32_t diff = p[k][j] - c[k][j];
                accum += (int32_t)filter[k] * diff;
            }
            y_row[j] = (accum + (1 << 7)) >> 8;
            nz_tail |= y_row[j];
        }

        if (_mm256_testz_si256(nz_acc, nz_acc) && !nz_tail) continue;

        // Phase 2: SIMD x_conv + abs + accumulate
        sad += x_conv_row_sad_avx2(y_row, w);
    }

    return sad;
}
