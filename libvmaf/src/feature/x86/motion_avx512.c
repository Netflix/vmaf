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

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>

#include "feature/integer_motion.h"
#include "feature/common/alignment.h"
#include "picture.h"

void x_convolution_16_avx512(const uint16_t *src, uint16_t *dst, unsigned width,
                             unsigned height, ptrdiff_t src_stride,
                             ptrdiff_t dst_stride)
{
    const unsigned radius = filter_width / 2;
    const unsigned left_edge = vmaf_ceiln(radius, 1);
    const unsigned right_edge = vmaf_floorn(width - (filter_width - radius), 1);
    const unsigned shift_add_round = 32768;
    const unsigned vector_loop = width < 32 ? 0 : (width>>5) -1;
    uint16_t *src_p = (uint16_t*) src + (left_edge - radius);
    unsigned nr = left_edge + 32 *vector_loop;
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
        for (unsigned j = 0; j <vector_loop; j=j+1) {
            __m512i src1            = _mm512_loadu_si512 ((__m512i *)src_p1);
            __m512i kernel1         = _mm512_set1_epi16( 3571);
            __m512i kernel2         = _mm512_set1_epi16( 16004);
            __m512i kernel3         = _mm512_set1_epi16( 26386) ;
            __m512i result          = _mm512_mulhi_epu16(src1,kernel1);
            __m512i resultlo        = _mm512_mullo_epi16(src1,kernel1);

            // src1= src1>>16; //shift by a  pixel
            __m512i src2            = _mm512_loadu_si512 ((__m512i *)(src_p1+1));
            __m512i result2         = _mm512_mulhi_epu16(src2,kernel2);
            __m512i result2lo       = _mm512_mullo_epi16(src2,kernel2);
            __m512i accum1_lo       = _mm512_unpacklo_epi16(resultlo, result);
            __m512i accum1_hi       = _mm512_unpackhi_epi16(resultlo, result);
            __m512i accum2_lo       = _mm512_unpacklo_epi16(result2lo, result2);
            __m512i accum2_hi       = _mm512_unpackhi_epi16(result2lo, result2);

            // Filter[3] value
            // src1= src1>>32;
            __m512i src3            = _mm512_loadu_si512 ((__m512i *)(src_p1+2));
            __m512i result3         = _mm512_mulhi_epu16(src3,kernel3);
            __m512i result3lo       = _mm512_mullo_epi16(src3,kernel3);
            __m512i accum3_lo       = _mm512_unpacklo_epi16 (result3lo, result3);
            __m512i accum3_hi       = _mm512_unpackhi_epi16 (result3lo, result3);
            //filter 4
            src1      = _mm512_loadu_si512 ((__m512i *)(src_p1+3));
            result    = _mm512_mulhi_epu16(src1,kernel2);
            resultlo  = _mm512_mullo_epi16(src1,kernel2);

            //Filter 5
            src2      = _mm512_loadu_si512((__m512i *)(src_p1+4));
            result2   = _mm512_mulhi_epu16(src2,kernel1);
            result2lo = _mm512_mullo_epi16(src2,kernel1);

            __m512i accum4_lo =_mm512_unpacklo_epi16(resultlo, result);
            __m512i accum4_hi =_mm512_unpackhi_epi16(resultlo, result);
            __m512i accum5_lo =_mm512_unpacklo_epi16(result2lo, result2);
            __m512i accum5_hi =_mm512_unpackhi_epi16(result2lo, result2);

            __m512i addnum    = _mm512_set1_epi32(32768);
            __m512i accum_lo  = _mm512_add_epi32(accum1_lo,accum2_lo);
            __m512i accumi_lo = _mm512_add_epi32(accum3_lo,accum4_lo);
                    accum5_lo = _mm512_add_epi32(accum5_lo,addnum);
                    accum_lo  = _mm512_add_epi32(accum5_lo,accum_lo);
                    accum_lo  = _mm512_add_epi32(accumi_lo,accum_lo);
            __m512i accum_hi  = _mm512_add_epi32(accum1_hi,accum2_hi);
            __m512i accumi_hi = _mm512_add_epi32(accum3_hi,accum4_hi);
                    accum_hi  = _mm512_add_epi32(accum5_hi,accum_hi);
                    accumi_hi = _mm512_add_epi32(accumi_hi,addnum);
                    accum_hi  = _mm512_add_epi32(accumi_hi,accum_hi);
                    accum_lo  = _mm512_srli_epi32(accum_lo, 0x10);
                    accum_hi  = _mm512_srli_epi32(accum_hi, 0x10);

            result = _mm512_packus_epi32(accum_lo,accum_hi);
            _mm512_storeu_si512((__m512i *) (dst+ i * dst_stride + j*32+ left_edge),result);
            src_p1+=32;
        }

        src_p += src_stride;
    }

   for (unsigned i = 0; i < height; ++i) {
        uint16_t *src_p1 = src_pt;
        for (unsigned j = nr; j < right_edge; j++) {
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

static inline uint32_t
edge_8(const uint8_t *src, int height, int stride, int i, int j)
{
    int radius = filter_width / 2;
    uint32_t accum = 0;

    // MIRROR | ЯOЯЯIM
    for (unsigned k = 0; k < filter_width; ++k) {
        int i_tap = i - radius + k;
        int j_tap = j;

        if (i_tap < 0)
            i_tap = -i_tap;
        else if (i_tap >= height)
            i_tap = height - (i_tap - height + 1);

        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}



void y_convolution_8_avx512(void *src, uint16_t *dst, unsigned width,
                unsigned height, ptrdiff_t src_stride, ptrdiff_t dst_stride,
                unsigned inp_size_bits)
{
    (void) inp_size_bits;
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned shift_var = 8;
    const unsigned add_before_shift = (int) pow(2, (shift_var - 1));

    for (unsigned i = 0; i < top_edge; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_8(src, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }

    __m512i f0 = _mm512_set1_epi32(filter[0]);
    __m512i f12 = _mm512_set1_epi32(filter[1] + (filter[2] << 16));
    __m512i f34 = _mm512_set1_epi32(filter[3] + (filter[4] << 16));

    __m512i add_before_shift_512 = _mm512_set1_epi32(add_before_shift);

    unsigned width_mod_64 = width - (width % 64);

    uint8_t *src_p = (uint8_t*) src + (top_edge - radius) * src_stride;

    for (unsigned i = top_edge; i < bottom_edge; i++) {
        uint8_t *src_p1 = src_p;
        for (unsigned j = 0; j < width_mod_64; j+=64) {
            uint8_t *src_p2 = src_p1;

            __m512i d0 = _mm512_loadu_si512((__m512i*)(src_p2));
            __m512i d1 = _mm512_loadu_si512((__m512i*)(src_p2 + src_stride));
            __m512i d2 = _mm512_loadu_si512((__m512i*)(src_p2 + (src_stride * 2)));
            __m512i d3 = _mm512_loadu_si512((__m512i*)(src_p2 + (src_stride * 3)));
            __m512i d4 = _mm512_loadu_si512((__m512i*)(src_p2 + (src_stride * 4)));

            __m512i d0_lo = _mm512_unpacklo_epi8(d0, _mm512_setzero_si512());
            __m512i d0_hi = _mm512_unpackhi_epi8(d0, _mm512_setzero_si512());
            __m512i d1_lo = _mm512_unpacklo_epi8(d1, _mm512_setzero_si512());
            __m512i d1_hi = _mm512_unpackhi_epi8(d1, _mm512_setzero_si512());
            __m512i d2_lo = _mm512_unpacklo_epi8(d2, _mm512_setzero_si512());
            __m512i d2_hi = _mm512_unpackhi_epi8(d2, _mm512_setzero_si512());
            __m512i d3_lo = _mm512_unpacklo_epi8(d3, _mm512_setzero_si512());
            __m512i d3_hi = _mm512_unpackhi_epi8(d3, _mm512_setzero_si512());
            __m512i d4_lo = _mm512_unpacklo_epi8(d4, _mm512_setzero_si512());
            __m512i d4_hi = _mm512_unpackhi_epi8(d4, _mm512_setzero_si512());

            __m512i d0_lolo = _mm512_unpacklo_epi16(d0_lo, _mm512_setzero_si512());
            __m512i d0_hilo = _mm512_unpackhi_epi16(d0_lo, _mm512_setzero_si512());
            __m512i d0_lohi = _mm512_unpacklo_epi16(d0_hi, _mm512_setzero_si512());
            __m512i d0_hihi = _mm512_unpackhi_epi16(d0_hi, _mm512_setzero_si512());

            __m512i d12_lolo = _mm512_unpacklo_epi16(d1_lo, d2_lo);
            __m512i d12_hilo = _mm512_unpackhi_epi16(d1_lo, d2_lo);
            __m512i d12_lohi = _mm512_unpacklo_epi16(d1_hi, d2_hi);
            __m512i d12_hihi = _mm512_unpackhi_epi16(d1_hi, d2_hi);

            __m512i d34_lolo = _mm512_unpacklo_epi16(d3_lo, d4_lo);
            __m512i d34_hilo = _mm512_unpackhi_epi16(d3_lo, d4_lo);
            __m512i d34_lohi = _mm512_unpacklo_epi16(d3_hi, d4_hi);
            __m512i d34_hihi = _mm512_unpackhi_epi16(d3_hi, d4_hi);

            __m512i accum0_lolo = _mm512_mullo_epi32(d0_lolo, f0);
            __m512i accum0_hilo = _mm512_mullo_epi32(d0_hilo, f0);
            __m512i accum0_lohi = _mm512_mullo_epi32(d0_lohi, f0);
            __m512i accum0_hihi = _mm512_mullo_epi32(d0_hihi, f0);

            accum0_lolo = _mm512_add_epi32(accum0_lolo, _mm512_madd_epi16(d12_lolo, f12));
            accum0_lolo = _mm512_add_epi32(accum0_lolo, _mm512_madd_epi16(d34_lolo, f34));
            accum0_hilo = _mm512_add_epi32(accum0_hilo, _mm512_madd_epi16(d12_hilo, f12));
            accum0_hilo = _mm512_add_epi32(accum0_hilo, _mm512_madd_epi16(d34_hilo, f34));

            accum0_lohi = _mm512_add_epi32(accum0_lohi, _mm512_madd_epi16(d12_lohi, f12));
            accum0_lohi = _mm512_add_epi32(accum0_lohi, _mm512_madd_epi16(d34_lohi, f34));
            accum0_hihi = _mm512_add_epi32(accum0_hihi, _mm512_madd_epi16(d12_hihi, f12));
            accum0_hihi = _mm512_add_epi32(accum0_hihi, _mm512_madd_epi16(d34_hihi, f34));
            
            accum0_lolo = _mm512_add_epi32(add_before_shift_512, accum0_lolo);
            accum0_hilo = _mm512_add_epi32(add_before_shift_512, accum0_hilo);
            accum0_lohi = _mm512_add_epi32(add_before_shift_512, accum0_lohi);
            accum0_hihi = _mm512_add_epi32(add_before_shift_512, accum0_hihi);

            accum0_lolo = _mm512_srli_epi32(accum0_lolo, shift_var);
            accum0_hilo = _mm512_srli_epi32(accum0_hilo, shift_var);
            accum0_lohi = _mm512_srli_epi32(accum0_lohi, shift_var);
            accum0_hihi = _mm512_srli_epi32(accum0_hihi, shift_var);

            __m512i res0 = _mm512_packus_epi32(accum0_lolo, accum0_hilo);
            __m512i res8 = _mm512_packus_epi32(accum0_lohi, accum0_hihi);
            __m512i tmp = res0;

            res0 = _mm512_permutex2var_epi64(res0, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), res8);
            res8 = _mm512_permutex2var_epi64(tmp, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), res8);

            _mm512_storeu_si512((__m512i*)(dst + i * dst_stride + j), res0);
            _mm512_storeu_si512((__m512i*)(dst + i * dst_stride + j + 32), res8);

            src_p1 += 64;
        }

        for (unsigned j = width_mod_64; j < width; ++j) {
            uint8_t *src_p2 = src_p1;
            uint32_t accum = 0;
            for (unsigned k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2 += src_stride;
            }
            dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
            src_p1++;
        }
        src_p += src_stride;
    }

    for (unsigned i = bottom_edge; i < height; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_8(src, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }
}


void y_convolution_16_avx512(void *src, uint16_t *dst, unsigned width,
                 unsigned height, ptrdiff_t src_stride,
                 ptrdiff_t dst_stride, unsigned inp_size_bits)
{
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned add_before_shift = (int) pow(2, (inp_size_bits - 1));
    const unsigned shift_var = inp_size_bits;

    unsigned width_mod_32 = width - (width % 32);

    __m512i f0 = _mm512_set1_epi16(filter[0]);
    __m512i f1 = _mm512_set1_epi16(filter[1]);
    __m512i f2 = _mm512_set1_epi16(filter[2]);
    __m512i f3 = _mm512_set1_epi16(filter[3]);
    __m512i f4 = _mm512_set1_epi16(filter[4]);

    __m512i add_before_shift_512 = _mm512_set1_epi32(add_before_shift);

    uint16_t *src_p = (uint16_t*) src + (top_edge - radius) * src_stride;
    for (unsigned i = 0; i < top_edge; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_16(false, src, width, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }

    for (unsigned i = top_edge; i < bottom_edge; i++) {
        uint16_t *src_p1 = src_p;
        for (unsigned j = 0; j < width_mod_32; j+=32) {
            uint16_t *src_p2 = src_p1;

            __m512i d0 = _mm512_loadu_si512((__m512i*)(src_p2));
            __m512i d1 = _mm512_loadu_si512((__m512i*)(src_p2 + src_stride));
            __m512i d2 = _mm512_loadu_si512((__m512i*)(src_p2 + (src_stride * 2)));
            __m512i d3 = _mm512_loadu_si512((__m512i*)(src_p2 + (src_stride * 3)));
            __m512i d4 = _mm512_loadu_si512((__m512i*)(src_p2 + (src_stride * 4)));

            __m512i result0         = _mm512_mulhi_epu16(d0,f0);
            __m512i result0lo       = _mm512_mullo_epi16(d0,f0);
            __m512i accum0_lo       = _mm512_unpacklo_epi16(result0lo, result0);
            __m512i accum0_hi       = _mm512_unpackhi_epi16(result0lo, result0);

            __m512i result1         = _mm512_mulhi_epu16(d1,f1);
            __m512i result1lo       = _mm512_mullo_epi16(d1,f1);
            __m512i accum1_lo       = _mm512_unpacklo_epi16(result1lo, result1);
            __m512i accum1_hi       = _mm512_unpackhi_epi16(result1lo, result1);

            __m512i result2         = _mm512_mulhi_epu16(d2,f2);
            __m512i result2lo       = _mm512_mullo_epi16(d2,f2);
            __m512i accum2_lo       = _mm512_unpacklo_epi16(result2lo, result2);
            __m512i accum2_hi       = _mm512_unpackhi_epi16(result2lo, result2);

            __m512i result3         = _mm512_mulhi_epu16(d3,f3);
            __m512i result3lo       = _mm512_mullo_epi16(d3,f3);
            __m512i accum3_lo       = _mm512_unpacklo_epi16(result3lo, result3);
            __m512i accum3_hi       = _mm512_unpackhi_epi16(result3lo, result3);

            __m512i result4         = _mm512_mulhi_epu16(d4,f4);
            __m512i result4lo       = _mm512_mullo_epi16(d4,f4);
            __m512i accum4_lo       = _mm512_unpacklo_epi16(result4lo, result4);
            __m512i accum4_hi       = _mm512_unpackhi_epi16(result4lo, result4);

            accum0_lo = _mm512_add_epi32(accum0_lo,accum1_lo);
            accum2_lo = _mm512_add_epi32(accum2_lo,accum3_lo);
            accum0_lo = _mm512_add_epi32(accum0_lo,accum4_lo);
            accum0_lo = _mm512_add_epi32(accum0_lo,accum2_lo);

            accum0_hi = _mm512_add_epi32(accum0_hi,accum1_hi);
            accum2_hi = _mm512_add_epi32(accum2_hi,accum3_hi);
            accum0_hi = _mm512_add_epi32(accum0_hi,accum4_hi);
            accum0_hi = _mm512_add_epi32(accum0_hi,accum2_hi);

            accum0_lo = _mm512_add_epi32(add_before_shift_512, accum0_lo);
            accum0_hi = _mm512_add_epi32(add_before_shift_512, accum0_hi);
            accum0_lo = _mm512_srli_epi32(accum0_lo, shift_var);
            accum0_hi = _mm512_srli_epi32(accum0_hi, shift_var);
            __m512i accum0 = _mm512_packus_epi32(accum0_lo, accum0_hi);
            _mm512_storeu_si512((__m512i*)(dst + i * dst_stride + j), accum0);
            src_p1 += 32;
        }

        for (unsigned j = width_mod_32; j < width; ++j) {
            uint16_t *src_p2 = src_p1;
            uint32_t accum = 0;
            for (unsigned k = 0; k < filter_width; ++k) {
                accum += filter[k] * (*src_p2);
                src_p2 += src_stride;
            }
            dst[i * dst_stride + j] = (accum + add_before_shift) >> shift_var;
            src_p1++;
        }

        src_p += src_stride;
    }

    for (unsigned i = bottom_edge; i < height; i++) {
        for (unsigned j = 0; j < width; ++j) {
            dst[i * dst_stride + j] =
                (edge_16(false, src, width, height, src_stride, i, j) +
                 add_before_shift) >> shift_var;
        }
    }
}

void sad_avx512(VmafPicture *pic_a, VmafPicture *pic_b, uint64_t *sad)
{
    *sad = 0;

    uint16_t *a = pic_a->data[0];
    uint16_t *b = pic_b->data[0];
    
    __uint32_t height = pic_a->h[0];
    __uint32_t width = pic_a->w[0];

    unsigned width_mod_64 = width - (width % 64);

    __m512i final_accum = _mm512_setzero_si512();

    for (unsigned i = 0; i < height; i++) {
        uint32_t inner_sad = 0;
        __m512i inter_accum_lo = _mm512_setzero_si512();
        __m512i inter_accum_hi = _mm512_setzero_si512();

        for (unsigned j = 0; j < width_mod_64; j+=64) {
            __m512i da = _mm512_loadu_si512((__m512i*)(a + j));
            __m512i db = _mm512_loadu_si512((__m512i*)(b + j));
            __m512i da_32 = _mm512_loadu_si512((__m512i*)(a + j + 32));
            __m512i db_32 = _mm512_loadu_si512((__m512i*)(b + j + 32));
            __m512i da_lo = _mm512_unpacklo_epi16(da, _mm512_setzero_si512());
            __m512i da_hi = _mm512_unpackhi_epi16(da, _mm512_setzero_si512());
            __m512i db_lo = _mm512_unpacklo_epi16(db, _mm512_setzero_si512());
            __m512i db_hi = _mm512_unpackhi_epi16(db, _mm512_setzero_si512());
            __m512i da_32_lo = _mm512_unpacklo_epi16(da_32, _mm512_setzero_si512());
            __m512i da_32_hi = _mm512_unpackhi_epi16(da_32, _mm512_setzero_si512());
            __m512i db_32_lo = _mm512_unpacklo_epi16(db_32, _mm512_setzero_si512());
            __m512i db_32_hi = _mm512_unpackhi_epi16(db_32, _mm512_setzero_si512());

            __m512i abs_da_m_db_lo = _mm512_abs_epi32(_mm512_sub_epi32(da_lo, db_lo));
            __m512i abs_da_m_db_hi = _mm512_abs_epi32(_mm512_sub_epi32(da_hi, db_hi));
            __m512i abs_da_m_db_32_lo = _mm512_abs_epi32(_mm512_sub_epi32(da_32_lo, db_32_lo));
            __m512i abs_da_m_db_32_hi = _mm512_abs_epi32(_mm512_sub_epi32(da_32_hi, db_32_hi));
            inter_accum_lo = _mm512_add_epi32(inter_accum_lo, abs_da_m_db_lo);
            inter_accum_hi = _mm512_add_epi32(inter_accum_hi, abs_da_m_db_hi);
            inter_accum_lo = _mm512_add_epi32(inter_accum_lo, abs_da_m_db_32_lo);
            inter_accum_hi = _mm512_add_epi32(inter_accum_hi, abs_da_m_db_32_hi);
        }

        for (unsigned j = width_mod_64; j < pic_a->w[0]; j++) {
            inner_sad += abs(a[j] - b[j]);
        }
        *sad += inner_sad;

        final_accum = _mm512_add_epi64(final_accum, _mm512_cvtepi32_epi64(_mm256_add_epi32(
            _mm512_castsi512_si256(inter_accum_lo), _mm512_extracti64x4_epi64(inter_accum_lo, 1))));
        final_accum = _mm512_add_epi64(final_accum, _mm512_cvtepi32_epi64(_mm256_add_epi32(
            _mm512_castsi512_si256(inter_accum_hi), _mm512_extracti64x4_epi64(inter_accum_hi, 1))));

        a += (pic_a->stride[0] / 2);
        b += (pic_b->stride[0] / 2);
    }
    __m256i r4 = _mm256_add_epi64(_mm512_castsi512_si256(final_accum), _mm512_extracti64x4_epi64(final_accum, 1));
    __m128i r2 = _mm_add_epi64(_mm256_castsi256_si128(r4), _mm256_extracti64x2_epi64(r4, 1));
    __uint64_t r1 = r2[0] + r2[1];
    
    *sad += r1;

}
