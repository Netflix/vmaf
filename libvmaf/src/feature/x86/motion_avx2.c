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
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "feature/integer_motion.h"
#include "feature/common/alignment.h"

void x_convolution_16_avx2(const uint16_t *src, uint16_t *dst, unsigned width,
                           unsigned height, ptrdiff_t src_stride,
                           ptrdiff_t dst_stride)
{
    const unsigned radius = filter_width / 2;
    const unsigned left_edge = vmaf_ceiln(radius, 1);
    const unsigned right_edge = vmaf_floorn(width - (filter_width - radius), 1);
    const unsigned shift_add_round = 32768;
    const unsigned vector_loop = width < 16 ? 0 : (width >> 4) - 1;

    uint16_t *src_p = (uint16_t*) src + (left_edge - radius);
    unsigned nr = left_edge + 16 * vector_loop;
    uint16_t *src_pt = (uint16_t*) src + nr -radius;
    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = 0; j < left_edge; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }
    }

    for (unsigned i = 0; i < height; ++i) {
        uint16_t *src_p1 = src_p;
        for (unsigned j = 0; j < vector_loop; j = j + 1) {
            __m256i src1 = _mm256_loadu_si256((__m256i*) src_p1);
            __m256i kernel1 = _mm256_set1_epi16(3571);
            __m256i kernel2 = _mm256_set1_epi16(16004);
            __m256i kernel3 = _mm256_set1_epi16(26386);
            __m256i result = _mm256_mulhi_epu16(src1, kernel1);
            __m256i resultlo = _mm256_mullo_epi16(src1, kernel1);

            //src1 = src1 >> 16; //shift by a  pixel
            __m256i src2 = _mm256_loadu_si256((__m256i*) (src_p1 + 1));
            __m256i result2 = _mm256_mulhi_epu16(src2, kernel2);
            __m256i result2lo = _mm256_mullo_epi16(src2, kernel2);
            __m256i accum1_lo = _mm256_unpacklo_epi16(resultlo, result);
            __m256i accum1_hi = _mm256_unpackhi_epi16(resultlo, result);
            __m256i accum2_lo = _mm256_unpacklo_epi16(result2lo, result2);
            __m256i accum2_hi = _mm256_unpackhi_epi16(result2lo, result2);

            //Filter[3] value
            // src1= src1>>32;
            __m256i src3 = _mm256_loadu_si256((__m256i*) (src_p1 + 2));
            __m256i result3 = _mm256_mulhi_epu16(src3, kernel3);
            __m256i result3lo = _mm256_mullo_epi16(src3, kernel3);
            __m256i accum3_lo = _mm256_unpacklo_epi16(result3lo, result3);
            __m256i accum3_hi = _mm256_unpackhi_epi16(result3lo, result3);

            //filter 4
            src1 = _mm256_loadu_si256((__m256i*) (src_p1 + 3));
            result = _mm256_mulhi_epu16(src1, kernel2);
            resultlo = _mm256_mullo_epi16(src1, kernel2);

            //Filter 5
            src2 = _mm256_loadu_si256((__m256i*) (src_p1 + 4));
            result2 = _mm256_mulhi_epu16(src2, kernel1);
            result2lo = _mm256_mullo_epi16(src2, kernel1);

            __m256i accum4_lo =_mm256_unpacklo_epi16(resultlo, result);
            __m256i accum4_hi =_mm256_unpackhi_epi16(resultlo, result);
            __m256i accum5_lo =_mm256_unpacklo_epi16(result2lo, result2);
            __m256i accum5_hi =_mm256_unpackhi_epi16(result2lo, result2);

            __m256i addnum = _mm256_set1_epi32(32768);
            __m256i accum_lo = _mm256_add_epi32(accum1_lo, accum2_lo);
            __m256i accumi_lo = _mm256_add_epi32(accum3_lo, accum4_lo);
            accum5_lo = _mm256_add_epi32(accum5_lo, addnum);
            accum_lo = _mm256_add_epi32(accum5_lo, accum_lo);
            accum_lo = _mm256_add_epi32(accumi_lo, accum_lo);
            __m256i accum_hi = _mm256_add_epi32(accum1_hi, accum2_hi);
            __m256i accumi_hi = _mm256_add_epi32(accum3_hi, accum4_hi);
            accum_hi = _mm256_add_epi32(accum5_hi, accum_hi);
            accumi_hi = _mm256_add_epi32(accumi_hi, addnum);
            accum_hi = _mm256_add_epi32(accumi_hi, accum_hi);
            accum_lo = _mm256_srli_epi32(accum_lo, 0x10);
            accum_hi = _mm256_srli_epi32(accum_hi, 0x10);

            result = _mm256_packus_epi32(accum_lo, accum_hi);
            _mm256_storeu_si256(
                (__m256i*) (dst + i * dst_stride + j * 16 + left_edge), result);

            src_p1 += 16;
        }
        src_p += src_stride;
    }

    for (unsigned i = 0; i < height; ++i) {
        uint16_t *src_p1 = src_pt;
        for (unsigned j = nr; j < (right_edge); j++) {
            uint32_t accum = 0;
            uint16_t *src_p2 = src_p1;
            for (int k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2++;
            }
            src_p1++;
            dst[i * dst_stride + j] = (accum + shift_add_round) >> 16;
        }
        src_pt += src_stride;
    }

    for (unsigned i = 0; i < height; ++i) {
        for (unsigned j = right_edge; j < width; j++) {
            dst[i * dst_stride + j] =
                (edge_16(true, src, width, height, src_stride, i, j) +
                 shift_add_round) >> 16;
        }
    }
}

void y_convolution_16_avx2(void *src_v, uint16_t *dst, unsigned width,
                           unsigned height, ptrdiff_t src_stride,
                           ptrdiff_t dst_stride, unsigned inp_size_bits)
{
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned add_before_shift = 1u << (inp_size_bits - 1);
    const unsigned shift_var = inp_size_bits;
    const uint16_t *src = (const uint16_t *)src_v;

    /* Top edge rows: use scalar mirror-boundary code */
    for (unsigned i = 0; i < top_edge; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_16(false, src, width, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }

    /* Pre-broadcast filter coefficients */
    const __m256i k0 = _mm256_set1_epi16(filter[0]); /* 3571 */
    const __m256i k1 = _mm256_set1_epi16(filter[1]); /* 16004 */
    const __m256i k2 = _mm256_set1_epi16(filter[2]); /* 26386 */
    const __m256i addnum = _mm256_set1_epi32(add_before_shift);

    /* Interior rows: AVX2 vertical convolution */
    const uint16_t *src_base = src + (top_edge - radius) * src_stride;
    for (unsigned i = top_edge; i < bottom_edge; i++) {
        const uint16_t *row0 = src_base;
        const uint16_t *row1 = row0 + src_stride;
        const uint16_t *row2 = row1 + src_stride;
        const uint16_t *row3 = row2 + src_stride;
        const uint16_t *row4 = row3 + src_stride;
        unsigned j = 0;

        /* Process 16 pixels per iteration */
        for (; j + 16 <= width; j += 16) {
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(row0 + j));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(row1 + j));
            __m256i r2 = _mm256_loadu_si256((const __m256i *)(row2 + j));
            __m256i r3 = _mm256_loadu_si256((const __m256i *)(row3 + j));
            __m256i r4 = _mm256_loadu_si256((const __m256i *)(row4 + j));

            /* Exploit symmetry: filter[0]==filter[4], filter[1]==filter[3] */
            __m256i s04 = _mm256_add_epi16(r0, r4);
            __m256i s13 = _mm256_add_epi16(r1, r3);

            /* Compute 32-bit products for tap 0+4 (k0 * s04) */
            __m256i p04_lo = _mm256_mullo_epi16(s04, k0);
            __m256i p04_hi = _mm256_mulhi_epu16(s04, k0);
            __m256i acc_lo = _mm256_unpacklo_epi16(p04_lo, p04_hi);
            __m256i acc_hi = _mm256_unpackhi_epi16(p04_lo, p04_hi);

            /* Add tap 1+3 (k1 * s13) */
            __m256i p13_lo = _mm256_mullo_epi16(s13, k1);
            __m256i p13_hi = _mm256_mulhi_epu16(s13, k1);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(p13_lo, p13_hi));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(p13_lo, p13_hi));

            /* Add tap 2 (k2 * r2) */
            __m256i p2_lo = _mm256_mullo_epi16(r2, k2);
            __m256i p2_hi = _mm256_mulhi_epu16(r2, k2);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(p2_lo, p2_hi));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(p2_lo, p2_hi));

            /* Add rounding and shift */
            acc_lo = _mm256_add_epi32(acc_lo, addnum);
            acc_hi = _mm256_add_epi32(acc_hi, addnum);
            acc_lo = _mm256_srli_epi32(acc_lo, shift_var);
            acc_hi = _mm256_srli_epi32(acc_hi, shift_var);

            /* Pack 32-bit back to 16-bit */
            __m256i result = _mm256_packus_epi32(acc_lo, acc_hi);
            _mm256_storeu_si256((__m256i *)(dst + i * dst_stride + j), result);
        }

        /* Scalar remainder */
        for (; j < width; ++j) {
            uint32_t accum = 0;
            accum += filter[0] * row0[j];
            accum += filter[1] * row1[j];
            accum += filter[2] * row2[j];
            accum += filter[3] * row3[j];
            accum += filter[4] * row4[j];
            dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
        }

        src_base += src_stride;
    }

    /* Bottom edge rows: use scalar mirror-boundary code */
    for (unsigned i = bottom_edge; i < height; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_16(false, src, width, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }
}

void y_convolution_8_avx2(void *src_v, uint16_t *dst, unsigned width,
                           unsigned height, ptrdiff_t src_stride,
                           ptrdiff_t dst_stride, unsigned inp_size_bits)
{
    (void) inp_size_bits;
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned shift_var = 8;
    const unsigned add_before_shift = 1u << (shift_var - 1);
    const uint8_t *src = (const uint8_t *)src_v;

    /* Top edge rows: use scalar mirror-boundary code */
    for (unsigned i = 0; i < top_edge; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_8(src, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }

    /* Pre-broadcast filter coefficients */
    const __m256i k0 = _mm256_set1_epi16(filter[0]); /* 3571 */
    const __m256i k1 = _mm256_set1_epi16(filter[1]); /* 16004 */
    const __m256i k2 = _mm256_set1_epi16(filter[2]); /* 26386 */
    const __m256i addnum = _mm256_set1_epi32(add_before_shift);

    /* Interior rows: AVX2 vertical convolution */
    const uint8_t *src_base = src + (top_edge - radius) * src_stride;
    for (unsigned i = top_edge; i < bottom_edge; i++) {
        const uint8_t *row0 = src_base;
        const uint8_t *row1 = row0 + src_stride;
        const uint8_t *row2 = row1 + src_stride;
        const uint8_t *row3 = row2 + src_stride;
        const uint8_t *row4 = row3 + src_stride;
        unsigned j = 0;

        /* Process 16 pixels per iteration: load 8-bit, zero-extend to 16-bit */
        for (; j + 16 <= width; j += 16) {
            __m256i r0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(row0 + j)));
            __m256i r1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(row1 + j)));
            __m256i r2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(row2 + j)));
            __m256i r3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(row3 + j)));
            __m256i r4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(row4 + j)));

            /* Exploit symmetry: filter[0]==filter[4], filter[1]==filter[3] */
            __m256i s04 = _mm256_add_epi16(r0, r4);
            __m256i s13 = _mm256_add_epi16(r1, r3);

            /* Compute 32-bit products for tap 0+4 (k0 * s04) */
            __m256i p04_lo = _mm256_mullo_epi16(s04, k0);
            __m256i p04_hi = _mm256_mulhi_epu16(s04, k0);
            __m256i acc_lo = _mm256_unpacklo_epi16(p04_lo, p04_hi);
            __m256i acc_hi = _mm256_unpackhi_epi16(p04_lo, p04_hi);

            /* Add tap 1+3 (k1 * s13) */
            __m256i p13_lo = _mm256_mullo_epi16(s13, k1);
            __m256i p13_hi = _mm256_mulhi_epu16(s13, k1);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(p13_lo, p13_hi));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(p13_lo, p13_hi));

            /* Add tap 2 (k2 * r2) */
            __m256i p2_lo = _mm256_mullo_epi16(r2, k2);
            __m256i p2_hi = _mm256_mulhi_epu16(r2, k2);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(p2_lo, p2_hi));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(p2_lo, p2_hi));

            /* Add rounding and shift */
            acc_lo = _mm256_add_epi32(acc_lo, addnum);
            acc_hi = _mm256_add_epi32(acc_hi, addnum);
            acc_lo = _mm256_srli_epi32(acc_lo, shift_var);
            acc_hi = _mm256_srli_epi32(acc_hi, shift_var);

            /* Pack 32-bit back to 16-bit */
            __m256i result = _mm256_packus_epi32(acc_lo, acc_hi);
            _mm256_storeu_si256((__m256i *)(dst + i * dst_stride + j), result);
        }

        /* Scalar remainder */
        for (; j < width; ++j) {
            uint32_t accum = 0;
            accum += filter[0] * row0[j];
            accum += filter[1] * row1[j];
            accum += filter[2] * row2[j];
            accum += filter[3] * row3[j];
            accum += filter[4] * row4[j];
            dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
        }

        src_base += src_stride;
    }

    /* Bottom edge rows: use scalar mirror-boundary code */
    for (unsigned i = bottom_edge; i < height; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_8(src, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }
}

void sad_avx2(const uint16_t *a, const uint16_t *b, unsigned w, unsigned h,
              ptrdiff_t stride_a, ptrdiff_t stride_b, uint64_t *sad)
{
    uint64_t total_sad = 0;
    const __m256i zero = _mm256_setzero_si256();

    for (unsigned i = 0; i < h; i++) {
        __m256i row_sad = _mm256_setzero_si256();
        unsigned j = 0;

        for (; j + 16 <= w; j += 16) {
            __m256i va = _mm256_loadu_si256((const __m256i *)(a + j));
            __m256i vb = _mm256_loadu_si256((const __m256i *)(b + j));
            __m256i mx = _mm256_max_epu16(va, vb);
            __m256i mn = _mm256_min_epu16(va, vb);
            __m256i diff = _mm256_sub_epi16(mx, mn);
            /* Zero-extend 16-bit diffs to 32-bit and accumulate */
            __m256i diff_lo = _mm256_unpacklo_epi16(diff, zero);
            __m256i diff_hi = _mm256_unpackhi_epi16(diff, zero);
            row_sad = _mm256_add_epi32(row_sad, diff_lo);
            row_sad = _mm256_add_epi32(row_sad, diff_hi);
        }

        __m128i lo = _mm256_castsi256_si128(row_sad);
        __m128i hi = _mm256_extracti128_si256(row_sad, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_add_epi32(sum128,
                               _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1,0,3,2)));
        sum128 = _mm_add_epi32(sum128,
                               _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0,1,0,1)));
        uint32_t row_total = (uint32_t)_mm_cvtsi128_si32(sum128);

        for (; j < w; j++) {
            row_total += abs(a[j] - b[j]);
        }

        total_sad += row_total;
        a += stride_a;
        b += stride_b;
    }

    *sad = total_sad;
}
