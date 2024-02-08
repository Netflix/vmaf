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
