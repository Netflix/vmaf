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
#include <math.h>

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


#define print_128_16(a) \
{ \
    printf("%hd ", _mm_extract_epi16(a, 0)); \
    printf("%hd ", _mm_extract_epi16(a, 1)); \
    printf("%hd ", _mm_extract_epi16(a, 2)); \
    printf("%hd ", _mm_extract_epi16(a, 3)); \
    printf("%hd ", _mm_extract_epi16(a, 4)); \
    printf("%hd ", _mm_extract_epi16(a, 5)); \
    printf("%hd ", _mm_extract_epi16(a, 6)); \
    printf("%hd ", _mm_extract_epi16(a, 7)); \
}

#define print_128_x(a) \
{ \
    printf("%x ", _mm_extract_epi32(a, 0)); \
    printf("%x ", _mm_extract_epi32(a, 1)); \
    printf("%x ", _mm_extract_epi32(a, 2)); \
    printf("%x ", _mm_extract_epi32(a, 3)); \
}
#define print_128_32(a) \
{ \
    printf("%d ", _mm_extract_epi32(a, 0)); \
    printf("%d ", _mm_extract_epi32(a, 1)); \
    printf("%d ", _mm_extract_epi32(a, 2)); \
    printf("%d ", _mm_extract_epi32(a, 3)); \
}
#define print_128_ps(a) \
{ \
    printf("%f ", _mm256_cvtss_f32(_mm256_castps128_ps256(a))); \
    printf("%f ", _mm256_cvtss_f32(_mm256_castps128_ps256(_mm_permute_ps(a, 0x1)))); \
    printf("%f ", _mm256_cvtss_f32(_mm256_castps128_ps256(_mm_permute_ps(a, 0x2)))); \
    printf("%f ", _mm256_cvtss_f32(_mm256_castps128_ps256(_mm_permute_ps(a, 0x3)))); \
}
#define print_128_pd(a) \
{ \
    printf("%lf ", _mm256_cvtsd_f64(_mm256_castpd128_pd256(a))); \
    printf("%lf ", _mm256_cvtsd_f64(_mm256_castpd128_pd256(_mm_permute_pd(a, 0x1)))); \
}
#define print_128_32u(a) \
{ \
    printf("%u ", _mm_extract_epi32(a, 0)); \
    printf("%u ", _mm_extract_epi32(a, 1)); \
    printf("%u ", _mm_extract_epi32(a, 2)); \
    printf("%u ", _mm_extract_epi32(a, 3)); \
}
#define print_128_64(a) \
{ \
    printf("%lld ", _mm_extract_epi64(a, 0)); \
    printf("%lld ", _mm_extract_epi64(a, 1)); \
}

#define print_256_16(a) \
{ \
    print_128_16(_mm256_extracti128_si256(a,0)); \
    print_128_16(_mm256_extracti128_si256(a,1)); \
}
#define print_256_32(a) \
{ \
    print_128_32(_mm256_extracti128_si256(a,0)); \
    print_128_32(_mm256_extracti128_si256(a,1)); \
}
#define print_256_x(a) \
{ \
    print_128_x(_mm256_extracti128_si256(a,0)); \
    print_128_x(_mm256_extracti128_si256(a,1)); \
}
#define print_256_ps(a) \
{ \
    print_128_ps(_mm256_extractf128_ps(a,0)); \
    print_128_ps(_mm256_extractf128_ps(a,1)); \
}
#define print_256_pd(a) \
{ \
    print_128_pd(_mm256_extractf128_pd(a,0)); \
    print_128_pd(_mm256_extractf128_pd(a,1)); \
}
#define print_256_32u(a) \
{ \
    print_128_32u(_mm256_extracti128_si256(a,0)); \
    print_128_32u(_mm256_extracti128_si256(a,1)); \
}
#define print_256_64(a) \
{ \
    print_128_64(_mm256_extracti128_si256(a,0)); \
    print_128_64(_mm256_extracti128_si256(a,1)); \
}

void y_convolution_8_avx2(void *src, uint16_t *dst, unsigned width,
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

    __m256i f0 = _mm256_set1_epi32(filter[0]);
    __m256i f12 = _mm256_set1_epi32(filter[1] + (filter[2] << 16));
    __m256i f34 = _mm256_set1_epi32(filter[3] + (filter[4] << 16));

    __m256i add_before_shift_256 = _mm256_set1_epi32(add_before_shift);

    unsigned width_mod_32 = width - (width % 32);

    uint8_t *src_p = (uint8_t*) src + (top_edge - radius) * src_stride;

    for (unsigned i = top_edge; i < bottom_edge; i++) {
        uint8_t *src_p1 = src_p;
        for (unsigned j = 0; j < width_mod_32; j+=32) {
            uint8_t *src_p2 = src_p1;

            __m256i d0 = _mm256_loadu_si256((__m256i*)(src_p2));
            __m256i d1 = _mm256_loadu_si256((__m256i*)(src_p2 + src_stride));
            __m256i d2 = _mm256_loadu_si256((__m256i*)(src_p2 + (src_stride * 2)));
            __m256i d3 = _mm256_loadu_si256((__m256i*)(src_p2 + (src_stride * 3)));
            __m256i d4 = _mm256_loadu_si256((__m256i*)(src_p2 + (src_stride * 4)));

            __m256i d0_lo = _mm256_unpacklo_epi8(d0, _mm256_setzero_si256());
            __m256i d0_hi = _mm256_unpackhi_epi8(d0, _mm256_setzero_si256());
            __m256i d1_lo = _mm256_unpacklo_epi8(d1, _mm256_setzero_si256());
            __m256i d1_hi = _mm256_unpackhi_epi8(d1, _mm256_setzero_si256());
            __m256i d2_lo = _mm256_unpacklo_epi8(d2, _mm256_setzero_si256());
            __m256i d2_hi = _mm256_unpackhi_epi8(d2, _mm256_setzero_si256());
            __m256i d3_lo = _mm256_unpacklo_epi8(d3, _mm256_setzero_si256());
            __m256i d3_hi = _mm256_unpackhi_epi8(d3, _mm256_setzero_si256());
            __m256i d4_lo = _mm256_unpacklo_epi8(d4, _mm256_setzero_si256());
            __m256i d4_hi = _mm256_unpackhi_epi8(d4, _mm256_setzero_si256());

            __m256i d0_lolo = _mm256_unpacklo_epi16(d0_lo, _mm256_setzero_si256());
            __m256i d0_hilo = _mm256_unpackhi_epi16(d0_lo, _mm256_setzero_si256());
            __m256i d0_lohi = _mm256_unpacklo_epi16(d0_hi, _mm256_setzero_si256());
            __m256i d0_hihi = _mm256_unpackhi_epi16(d0_hi, _mm256_setzero_si256());

            __m256i d12_lolo = _mm256_unpacklo_epi16(d1_lo, d2_lo);
            __m256i d12_hilo = _mm256_unpackhi_epi16(d1_lo, d2_lo);
            __m256i d12_lohi = _mm256_unpacklo_epi16(d1_hi, d2_hi);
            __m256i d12_hihi = _mm256_unpackhi_epi16(d1_hi, d2_hi);

            __m256i d34_lolo = _mm256_unpacklo_epi16(d3_lo, d4_lo);
            __m256i d34_hilo = _mm256_unpackhi_epi16(d3_lo, d4_lo);
            __m256i d34_lohi = _mm256_unpacklo_epi16(d3_hi, d4_hi);
            __m256i d34_hihi = _mm256_unpackhi_epi16(d3_hi, d4_hi);

            __m256i accum0_lolo = _mm256_mullo_epi32(d0_lolo, f0);
            __m256i accum0_hilo = _mm256_mullo_epi32(d0_hilo, f0);
            __m256i accum0_lohi = _mm256_mullo_epi32(d0_lohi, f0);
            __m256i accum0_hihi = _mm256_mullo_epi32(d0_hihi, f0);

            accum0_lolo = _mm256_add_epi32(accum0_lolo, _mm256_madd_epi16(d12_lolo, f12));
            accum0_lolo = _mm256_add_epi32(accum0_lolo, _mm256_madd_epi16(d34_lolo, f34));
            accum0_hilo = _mm256_add_epi32(accum0_hilo, _mm256_madd_epi16(d12_hilo, f12));
            accum0_hilo = _mm256_add_epi32(accum0_hilo, _mm256_madd_epi16(d34_hilo, f34));

            accum0_lohi = _mm256_add_epi32(accum0_lohi, _mm256_madd_epi16(d12_lohi, f12));
            accum0_lohi = _mm256_add_epi32(accum0_lohi, _mm256_madd_epi16(d34_lohi, f34));
            accum0_hihi = _mm256_add_epi32(accum0_hihi, _mm256_madd_epi16(d12_hihi, f12));
            accum0_hihi = _mm256_add_epi32(accum0_hihi, _mm256_madd_epi16(d34_hihi, f34));
            
            accum0_lolo = _mm256_add_epi32(add_before_shift_256, accum0_lolo);
            accum0_hilo = _mm256_add_epi32(add_before_shift_256, accum0_hilo);
            accum0_lohi = _mm256_add_epi32(add_before_shift_256, accum0_lohi);
            accum0_hihi = _mm256_add_epi32(add_before_shift_256, accum0_hihi);

            accum0_lolo = _mm256_srli_epi32(accum0_lolo, shift_var);
            accum0_hilo = _mm256_srli_epi32(accum0_hilo, shift_var);
            accum0_lohi = _mm256_srli_epi32(accum0_lohi, shift_var);
            accum0_hihi = _mm256_srli_epi32(accum0_hihi, shift_var);

            __m256i res0 = _mm256_packus_epi32(accum0_lolo, accum0_hilo);
            __m256i res8 = _mm256_packus_epi32(accum0_lohi, accum0_hihi);
            __m256i tmp = res0;

            res0 = _mm256_permute2x128_si256(res0, res8, 0x20);
            res8 = _mm256_permute2x128_si256(tmp, res8, 0x31);

            _mm256_storeu_si256((__m256i*)(dst + i * dst_stride + j), res0);
            _mm256_storeu_si256((__m256i*)(dst + i * dst_stride + j + 16), res8);

            src_p1 += 32;
        }

        for (unsigned j = width_mod_32; j < width; ++j) {
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


void y_convolution_16_avx2(void *src, uint16_t *dst, unsigned width,
                 unsigned height, ptrdiff_t src_stride,
                 ptrdiff_t dst_stride, unsigned inp_size_bits)
{
    const unsigned radius = filter_width / 2;
    const unsigned top_edge = vmaf_ceiln(radius, 1);
    const unsigned bottom_edge = vmaf_floorn(height - (filter_width - radius), 1);
    const unsigned add_before_shift = (int) pow(2, (inp_size_bits - 1));
    const unsigned shift_var = inp_size_bits;

    unsigned width_mod_16 = width - (width % 16);

    __m256i f0 = _mm256_set1_epi16(filter[0]);
    __m256i f1 = _mm256_set1_epi16(filter[1]);
    __m256i f2 = _mm256_set1_epi16(filter[2]);
    __m256i f3 = _mm256_set1_epi16(filter[3]);
    __m256i f4 = _mm256_set1_epi16(filter[4]);
    __m256i add_before_shift_256 = _mm256_set1_epi32(add_before_shift);

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
        for (unsigned j = 0; j < width_mod_16; j+=16) {
            uint16_t *src_p2 = src_p1;

            __m256i d0 = _mm256_loadu_si256((__m256i*)(src_p2));
            __m256i d1 = _mm256_loadu_si256((__m256i*)(src_p2 + src_stride));
            __m256i d2 = _mm256_loadu_si256((__m256i*)(src_p2 + (src_stride * 2)));
            __m256i d3 = _mm256_loadu_si256((__m256i*)(src_p2 + (src_stride * 3)));
            __m256i d4 = _mm256_loadu_si256((__m256i*)(src_p2 + (src_stride * 4)));

	    __m256i result0         = _mm256_mulhi_epu16(d0,f0);
            __m256i result0lo       = _mm256_mullo_epi16(d0,f0);
            __m256i accum0_lo       = _mm256_unpacklo_epi16(result0lo, result0);
            __m256i accum0_hi       = _mm256_unpackhi_epi16(result0lo, result0);

            __m256i result1         = _mm256_mulhi_epu16(d1,f1);
            __m256i result1lo       = _mm256_mullo_epi16(d1,f1);
            __m256i accum1_lo       = _mm256_unpacklo_epi16(result1lo, result1);
            __m256i accum1_hi       = _mm256_unpackhi_epi16(result1lo, result1);

            __m256i result2         = _mm256_mulhi_epu16(d2,f2);
            __m256i result2lo       = _mm256_mullo_epi16(d2,f2);
            __m256i accum2_lo       = _mm256_unpacklo_epi16(result2lo, result2);
            __m256i accum2_hi       = _mm256_unpackhi_epi16(result2lo, result2);

            __m256i result3         = _mm256_mulhi_epu16(d3,f3);
            __m256i result3lo       = _mm256_mullo_epi16(d3,f3);
            __m256i accum3_lo       = _mm256_unpacklo_epi16(result3lo, result3);
            __m256i accum3_hi       = _mm256_unpackhi_epi16(result3lo, result3);

            __m256i result4         = _mm256_mulhi_epu16(d4,f4);
            __m256i result4lo       = _mm256_mullo_epi16(d4,f4);
            __m256i accum4_lo       = _mm256_unpacklo_epi16(result4lo, result4);
            __m256i accum4_hi       = _mm256_unpackhi_epi16(result4lo, result4);

            accum0_lo = _mm256_add_epi32(accum0_lo,accum1_lo);
            accum2_lo = _mm256_add_epi32(accum2_lo,accum3_lo);
            accum0_lo = _mm256_add_epi32(accum0_lo,accum4_lo);
            accum0_lo = _mm256_add_epi32(accum0_lo,accum2_lo);

            accum0_hi = _mm256_add_epi32(accum0_hi,accum1_hi);
            accum2_hi = _mm256_add_epi32(accum2_hi,accum3_hi);
            accum0_hi = _mm256_add_epi32(accum0_hi,accum4_hi);
            accum0_hi = _mm256_add_epi32(accum0_hi,accum2_hi);

            accum0_lo = _mm256_add_epi32(add_before_shift_256, accum0_lo);
            accum0_hi = _mm256_add_epi32(add_before_shift_256, accum0_hi);
            accum0_lo = _mm256_srli_epi32(accum0_lo, shift_var);
            accum0_hi = _mm256_srli_epi32(accum0_hi, shift_var);
            __m256i accum0 = _mm256_packus_epi32(accum0_lo, accum0_hi);
            _mm256_storeu_si256((__m256i*)(dst + i * dst_stride + j), accum0);
            src_p1 += 16;

        }

        for (unsigned j = width_mod_16; j < width; ++j) {
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

void sad_avx2(VmafPicture *pic_a, VmafPicture *pic_b, uint64_t *sad)
{
    *sad = 0;

    uint16_t *a = pic_a->data[0];
    uint16_t *b = pic_b->data[0];
    
    __uint32_t height = pic_a->h[0];
    __uint32_t width = pic_a->w[0];

    __m256i final_accum = _mm256_setzero_si256();

    unsigned width_mod_32 = width - (width % 32);
    
    for (unsigned i = 0; i < height; i++) {
        uint32_t inner_sad = 0;
        __m256i inter_accum_lo = _mm256_setzero_si256();
        __m256i inter_accum_hi = _mm256_setzero_si256();

        for (unsigned j = 0; j < width_mod_32; j+=32) {
            __m256i da = _mm256_loadu_si256((__m256i*)(a + j));
            __m256i db = _mm256_loadu_si256((__m256i*)(b + j));
            __m256i da_32 = _mm256_loadu_si256((__m256i*)(a + j + 16));
            __m256i db_32 = _mm256_loadu_si256((__m256i*)(b + j + 16));
            __m256i da_lo = _mm256_unpacklo_epi16(da, _mm256_setzero_si256());
            __m256i da_hi = _mm256_unpackhi_epi16(da, _mm256_setzero_si256());
            __m256i db_lo = _mm256_unpacklo_epi16(db, _mm256_setzero_si256());
            __m256i db_hi = _mm256_unpackhi_epi16(db, _mm256_setzero_si256());
            __m256i da_32_lo = _mm256_unpacklo_epi16(da_32, _mm256_setzero_si256());
            __m256i da_32_hi = _mm256_unpackhi_epi16(da_32, _mm256_setzero_si256());
            __m256i db_32_lo = _mm256_unpacklo_epi16(db_32, _mm256_setzero_si256());
            __m256i db_32_hi = _mm256_unpackhi_epi16(db_32, _mm256_setzero_si256());

            __m256i abs_da_m_db_lo = _mm256_abs_epi32(_mm256_sub_epi32(da_lo, db_lo));
            __m256i abs_da_m_db_hi = _mm256_abs_epi32(_mm256_sub_epi32(da_hi, db_hi));
            __m256i abs_da_m_db_32_lo = _mm256_abs_epi32(_mm256_sub_epi32(da_32_lo, db_32_lo));
            __m256i abs_da_m_db_32_hi = _mm256_abs_epi32(_mm256_sub_epi32(da_32_hi, db_32_hi));
            inter_accum_lo = _mm256_add_epi32(inter_accum_lo, abs_da_m_db_lo);
            inter_accum_hi = _mm256_add_epi32(inter_accum_hi, abs_da_m_db_hi);
            inter_accum_lo = _mm256_add_epi32(inter_accum_lo, abs_da_m_db_32_lo);
            inter_accum_hi = _mm256_add_epi32(inter_accum_hi, abs_da_m_db_32_hi);
        }

        for (unsigned j = width_mod_32; j < pic_a->w[0]; j++) {
            inner_sad += abs(a[j] - b[j]);
        }
        *sad += inner_sad;

        final_accum = _mm256_add_epi64(final_accum, _mm256_cvtepi32_epi64(_mm_add_epi32(
            _mm256_castsi256_si128(inter_accum_lo), _mm256_extracti128_si256(inter_accum_lo, 1))));
        final_accum = _mm256_add_epi64(final_accum, _mm256_cvtepi32_epi64(_mm_add_epi32(
            _mm256_castsi256_si128(inter_accum_hi), _mm256_extracti128_si256(inter_accum_hi, 1))));

        a += (pic_a->stride[0] / 2);
        b += (pic_b->stride[0] / 2);
    }
    __uint64_t r1 = final_accum[0] + final_accum[1] + final_accum[2] + final_accum[3];
    
    *sad += r1;
}
