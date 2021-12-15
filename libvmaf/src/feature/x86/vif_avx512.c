/**
     *
     *  Copyright 2016-2020 Netflix, Inc.
     *
     *     Licensed under the BSD+Patent License (the "License"); you may not
     *     use this file except in compliance with the License. You may obtain a
     *     copy of the License at
     *
     *         https://opensource.org/licenses/BSDplusPatent
     *
     *     Unless required by applicable law or agreed to in writing, software
     *     distributed under the License is distributed on an "AS IS" BASIS,
     *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
     *     implied. See the License for the specific language governing
     *     permissions and limitations under the License.
     *
     */

#include <immintrin.h>
#include <stdint.h>
#include <stdint.h>
#include <stddef.h>
#include "stdio.h"

#include "feature/integer_vif.h"

void vif_filter1d_8_avx512(VifBuffer buf, unsigned w, unsigned h)
{
    const unsigned fwidth = vif_filter1d_width[0];
    const uint16_t *vif_filt_s0 = vif_filter1d_table[0];
    __m256i x = _mm256_set1_epi32(128);
    const uint8_t *ref = (uint8_t *)buf.ref;
    const uint8_t *dis = (uint8_t *)buf.dis;
    const unsigned fwidth_half = fwidth >> 1;
    const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
    __m512i mask2 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    __m512i mask3 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);

    __m512i f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, fc0, fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8;

    fc0 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)vif_filt_s0));
    fc1 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 2)));
    fc2 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 4)));
    fc3 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 6)));
    fc4 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 8)));
    fc5 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 10)));
    fc6 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 12)));
    fc7 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 14)));
    fc8 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 16)));

    f1 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)vif_filt_s0));
    f2 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 1)));
    f3 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 2)));
    f4 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 3)));
    f5 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 4)));
    f6 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 5)));
    f7 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 6)));
    f8 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 7)));
    f9 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s0 + 8)));

    __m512i fq1 = _mm512_set1_epi64(vif_filt_s0[0]);
    __m512i fq2 = _mm512_set1_epi64(vif_filt_s0[1]);
    __m512i fq3 = _mm512_set1_epi64(vif_filt_s0[2]);
    __m512i fq4 = _mm512_set1_epi64(vif_filt_s0[3]);
    __m512i fq5 = _mm512_set1_epi64(vif_filt_s0[4]);
    __m512i fq6 = _mm512_set1_epi64(vif_filt_s0[5]);
    __m512i fq7 = _mm512_set1_epi64(vif_filt_s0[6]);
    __m512i fq8 = _mm512_set1_epi64(vif_filt_s0[7]);
    __m512i fq9 = _mm512_set1_epi64(vif_filt_s0[8]);

    for (unsigned i = 0; i < h; ++i)
    {
        //VERTICAL
        int ii = i - fwidth_half;

        for (unsigned j = 0; j < w; j = j + 32)
        {
            __m512i accum_ref_lo, accum_ref_hi, accum_dis_lo, accum_dis_hi,
                accum_ref_dis_lo, accum_ref_dis_hi, accum_mu2_lo,
                accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
            accum_ref_lo = accum_ref_hi = accum_dis_lo = accum_dis_hi =
                accum_ref_dis_lo = accum_ref_dis_hi = accum_mu2_lo =
                    accum_mu2_hi = accum_mu1_lo = accum_mu1_hi = _mm512_setzero_si512();

            __m512i dislo, dishi, refdislo, refdishi, final_resultlo,
                final_resulthi;
            dislo = dishi = refdislo = refdishi = final_resultlo =
                final_resulthi = _mm512_setzero_si512();

            int ii_check = ii;
            __m512i g0, g1, g2, g3, g4, g5, g6, g7, g8, g20, g21, g22,
                g23, g24, g25, g26, g27, g28;
            __m512i s0, s1, s2, s3, s4, s5, s6, s7, s8, s20, s21, s22,
                s23, s24, s25, s26, s27, s28, sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7, sg8;

            g0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + (buf.stride * ii_check) + j)));
            g1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 1) + j)));
            g2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 2) + j)));
            g3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 3) + j)));
            g4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 4) + j)));
            g5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 5) + j)));
            g6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 6) + j)));
            g7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 7) + j)));

            s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + (buf.stride * ii_check) + j)));
            s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 1) + j)));
            s2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 2) + j)));
            s3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 3) + j)));
            s4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 4) + j)));
            s5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 5) + j)));
            s6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 6) + j)));
            s7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 7) + j)));

            __m512i s0lo = _mm512_unpacklo_epi16(s0, s1);
            __m512i s0hi = _mm512_unpackhi_epi16(s0, s1);
            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s0lo, fc0));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s0hi, fc0));
            __m512i s1lo = _mm512_unpacklo_epi16(s2, s3);
            __m512i s1hi = _mm512_unpackhi_epi16(s2, s3);
            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s1lo, fc1));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s1hi, fc1));
            __m512i s2lo = _mm512_unpacklo_epi16(s4, s5);
            __m512i s2hi = _mm512_unpackhi_epi16(s4, s5);
            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s2lo, fc2));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s2hi, fc2));
            __m512i s3lo = _mm512_unpacklo_epi16(s6, s7);
            __m512i s3hi = _mm512_unpackhi_epi16(s6, s7);
            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s3lo, fc3));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s3hi, fc3));

            __m512i g0lo = _mm512_unpacklo_epi16(g0, g1);
            __m512i g0hi = _mm512_unpackhi_epi16(g0, g1);
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(g0lo, fc0));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(g0hi, fc0));
            __m512i g1lo = _mm512_unpacklo_epi16(g2, g3);
            __m512i g1hi = _mm512_unpackhi_epi16(g2, g3);
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(g1lo, fc1));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(g1hi, fc1));
            __m512i g2lo = _mm512_unpacklo_epi16(g4, g5);
            __m512i g2hi = _mm512_unpackhi_epi16(g4, g5);
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(g2lo, fc2));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(g2hi, fc2));
            __m512i g3lo = _mm512_unpacklo_epi16(g6, g7);
            __m512i g3hi = _mm512_unpackhi_epi16(g6, g7);
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(g3lo, fc3));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(g3hi, fc3));

            g20 = _mm512_mullo_epi16(g0, g0);
            g21 = _mm512_mullo_epi16(g1, g1);
            g22 = _mm512_mullo_epi16(g2, g2);
            g23 = _mm512_mullo_epi16(g3, g3);
            g24 = _mm512_mullo_epi16(g4, g4);
            g25 = _mm512_mullo_epi16(g5, g5);
            g26 = _mm512_mullo_epi16(g6, g6);
            g27 = _mm512_mullo_epi16(g7, g7);

            s20 = _mm512_mullo_epi16(s0, s0);
            s21 = _mm512_mullo_epi16(s1, s1);
            s22 = _mm512_mullo_epi16(s2, s2);
            s23 = _mm512_mullo_epi16(s3, s3);
            s24 = _mm512_mullo_epi16(s4, s4);
            s25 = _mm512_mullo_epi16(s5, s5);
            s26 = _mm512_mullo_epi16(s6, s6);
            s27 = _mm512_mullo_epi16(s7, s7);

            sg0 = _mm512_mullo_epi16(s0, g0);
            sg1 = _mm512_mullo_epi16(s1, g1);
            sg2 = _mm512_mullo_epi16(s2, g2);
            sg3 = _mm512_mullo_epi16(s3, g3);
            sg4 = _mm512_mullo_epi16(s4, g4);
            sg5 = _mm512_mullo_epi16(s5, g5);
            sg6 = _mm512_mullo_epi16(s6, g6);
            sg7 = _mm512_mullo_epi16(s7, g7);

            __m512i result2 = _mm512_mulhi_epu16(g20, f1);
            __m512i result2lo = _mm512_mullo_epi16(g20, f1);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result2lo, result2));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result2lo, result2));
            __m512i result3 = _mm512_mulhi_epu16(g21, f2);
            __m512i result3lo = _mm512_mullo_epi16(g21, f2);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result3lo, result3));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result3lo, result3));
            __m512i result4 = _mm512_mulhi_epu16(g22, f3);
            __m512i result4lo = _mm512_mullo_epi16(g22, f3);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result4lo, result4));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result4lo, result4));
            __m512i result5 = _mm512_mulhi_epu16(g23, f4);
            __m512i result5lo = _mm512_mullo_epi16(g23, f4);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result5lo, result5));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result5lo, result5));
            __m512i result6 = _mm512_mulhi_epu16(g24, f5);
            __m512i result6lo = _mm512_mullo_epi16(g24, f5);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result6lo, result6));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result6lo, result6));
            __m512i result7 = _mm512_mulhi_epu16(g25, f6);
            __m512i result7lo = _mm512_mullo_epi16(g25, f6);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result7lo, result7));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result7lo, result7));
            __m512i result8 = _mm512_mulhi_epu16(g26, f7);
            __m512i result8lo = _mm512_mullo_epi16(g26, f7);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result8lo, result8));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result8lo, result8));
            __m512i result9 = _mm512_mulhi_epu16(g27, f8);
            __m512i result9lo = _mm512_mullo_epi16(g27, f8);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result9lo, result9));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result9lo, result9));

            result2 = _mm512_mulhi_epu16(s20, f1);
            result2lo = _mm512_mullo_epi16(s20, f1);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result2lo, result2));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result2lo, result2));
            result3 = _mm512_mulhi_epu16(s21, f2);
            result3lo = _mm512_mullo_epi16(s21, f2);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result3lo, result3));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result3lo, result3));
            result4 = _mm512_mulhi_epu16(s22, f3);
            result4lo = _mm512_mullo_epi16(s22, f3);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result4lo, result4));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result4lo, result4));
            result5 = _mm512_mulhi_epu16(s23, f4);
            result5lo = _mm512_mullo_epi16(s23, f4);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result5lo, result5));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result5lo, result5));
            result6 = _mm512_mulhi_epu16(s24, f5);
            result6lo = _mm512_mullo_epi16(s24, f5);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result6lo, result6));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result6lo, result6));
            result7 = _mm512_mulhi_epu16(s25, f6);
            result7lo = _mm512_mullo_epi16(s25, f6);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result7lo, result7));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result7lo, result7));
            result8 = _mm512_mulhi_epu16(s26, f7);
            result8lo = _mm512_mullo_epi16(s26, f7);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result8lo, result8));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result8lo, result8));
            result9 = _mm512_mulhi_epu16(s27, f8);
            result9lo = _mm512_mullo_epi16(s27, f8);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result9lo, result9));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result9lo, result9));

            result2 = _mm512_mulhi_epu16(sg0, f1);
            result2lo = _mm512_mullo_epi16(sg0, f1);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result2lo, result2));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result2lo, result2));

            result3 = _mm512_mulhi_epu16(sg1, f2);
            result3lo = _mm512_mullo_epi16(sg1, f2);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result3lo, result3));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result3lo, result3));

            result4 = _mm512_mulhi_epu16(sg2, f3);
            result4lo = _mm512_mullo_epi16(sg2, f3);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result4lo, result4));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result4lo, result4));
            result5 = _mm512_mulhi_epu16(sg3, f4);
            result5lo = _mm512_mullo_epi16(sg3, f4);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result5lo, result5));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result5lo, result5));
            result6 = _mm512_mulhi_epu16(sg4, f5);
            result6lo = _mm512_mullo_epi16(sg4, f5);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result6lo, result6));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result6lo, result6));
            result7 = _mm512_mulhi_epu16(sg5, f6);
            result7lo = _mm512_mullo_epi16(sg5, f6);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result7lo, result7));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result7lo, result7));
            result8 = _mm512_mulhi_epu16(sg6, f7);
            result8lo = _mm512_mullo_epi16(sg6, f7);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result8lo, result8));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result8lo, result8));
            result9 = _mm512_mulhi_epu16(sg7, f8);
            result9lo = _mm512_mullo_epi16(sg7, f8);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result9lo, result9));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result9lo, result9));

            g0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 8) + j)));
            g1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 9) + j)));
            g2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 10) + j)));
            g3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 11) + j)));
            g4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 12) + j)));
            g5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 13) + j)));
            g6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 14) + j)));
            g7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 15) + j)));

            s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 8) + j)));
            s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 9) + j)));
            s2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 10) + j)));
            s3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 11) + j)));
            s4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 12) + j)));
            s5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 13) + j)));
            s6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 14) + j)));
            s7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 15) + j)));

            s0lo = _mm512_unpacklo_epi16(s0, s1);
            s0hi = _mm512_unpackhi_epi16(s0, s1);
            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s0lo, fc4));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s0hi, fc4));
            s1lo = _mm512_unpacklo_epi16(s2, s3);
            s1hi = _mm512_unpackhi_epi16(s2, s3);
            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s1lo, fc5));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s1hi, fc5));
            s2lo = _mm512_unpacklo_epi16(s4, s5);
            s2hi = _mm512_unpackhi_epi16(s4, s5);
            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s2lo, fc6));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s2hi, fc6));
            s3lo = _mm512_unpacklo_epi16(s6, s7);
            s3hi = _mm512_unpackhi_epi16(s6, s7);
            accum_mu2_lo =
                _mm512_add_epi32(accum_mu2_lo, _mm512_madd_epi16(s3lo, fc7));
            accum_mu2_hi =
                _mm512_add_epi32(accum_mu2_hi, _mm512_madd_epi16(s3hi, fc7));

            g0lo = _mm512_unpacklo_epi16(g0, g1);
            g0hi = _mm512_unpackhi_epi16(g0, g1);
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(g0lo, fc4));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(g0hi, fc4));
            g1lo = _mm512_unpacklo_epi16(g2, g3);
            g1hi = _mm512_unpackhi_epi16(g2, g3);
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(g1lo, fc5));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(g1hi, fc5));
            g2lo = _mm512_unpacklo_epi16(g4, g5);
            g2hi = _mm512_unpackhi_epi16(g4, g5);
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(g2lo, fc6));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(g2hi, fc6));
            g3lo = _mm512_unpacklo_epi16(g6, g7);
            g3hi = _mm512_unpackhi_epi16(g6, g7);
            accum_mu1_lo =
                _mm512_add_epi32(accum_mu1_lo, _mm512_madd_epi16(g3lo, fc7));
            accum_mu1_hi =
                _mm512_add_epi32(accum_mu1_hi, _mm512_madd_epi16(g3hi, fc7));

            g20 = _mm512_mullo_epi16(g0, g0);
            g21 = _mm512_mullo_epi16(g1, g1);
            g22 = _mm512_mullo_epi16(g2, g2);
            g23 = _mm512_mullo_epi16(g3, g3);
            g24 = _mm512_mullo_epi16(g4, g4);
            g25 = _mm512_mullo_epi16(g5, g5);
            g26 = _mm512_mullo_epi16(g6, g6);
            g27 = _mm512_mullo_epi16(g7, g7);

            s20 = _mm512_mullo_epi16(s0, s0);
            s21 = _mm512_mullo_epi16(s1, s1);
            s22 = _mm512_mullo_epi16(s2, s2);
            s23 = _mm512_mullo_epi16(s3, s3);
            s24 = _mm512_mullo_epi16(s4, s4);
            s25 = _mm512_mullo_epi16(s5, s5);
            s26 = _mm512_mullo_epi16(s6, s6);
            s27 = _mm512_mullo_epi16(s7, s7);

            sg0 = _mm512_mullo_epi16(s0, g0);
            sg1 = _mm512_mullo_epi16(s1, g1);
            sg2 = _mm512_mullo_epi16(s2, g2);
            sg3 = _mm512_mullo_epi16(s3, g3);
            sg4 = _mm512_mullo_epi16(s4, g4);
            sg5 = _mm512_mullo_epi16(s5, g5);
            sg6 = _mm512_mullo_epi16(s6, g6);
            sg7 = _mm512_mullo_epi16(s7, g7);

            result2 = _mm512_mulhi_epu16(g20, f9);
            result2lo = _mm512_mullo_epi16(g20, f9);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result2lo, result2));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result2lo, result2));
            result3 = _mm512_mulhi_epu16(g21, f8);
            result3lo = _mm512_mullo_epi16(g21, f8);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result3lo, result3));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result3lo, result3));
            result4 = _mm512_mulhi_epu16(g22, f7);
            result4lo = _mm512_mullo_epi16(g22, f7);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result4lo, result4));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result4lo, result4));
            result5 = _mm512_mulhi_epu16(g23, f6);
            result5lo = _mm512_mullo_epi16(g23, f6);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result5lo, result5));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result5lo, result5));
            result6 = _mm512_mulhi_epu16(g24, f5);
            result6lo = _mm512_mullo_epi16(g24, f5);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result6lo, result6));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result6lo, result6));
            result7 = _mm512_mulhi_epu16(g25, f4);
            result7lo = _mm512_mullo_epi16(g25, f4);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result7lo, result7));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result7lo, result7));
            result8 = _mm512_mulhi_epu16(g26, f3);
            result8lo = _mm512_mullo_epi16(g26, f3);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result8lo, result8));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result8lo, result8));
            result9 = _mm512_mulhi_epu16(g27, f2);
            result9lo = _mm512_mullo_epi16(g27, f2);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result9lo, result9));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result9lo, result9));

            result2 = _mm512_mulhi_epu16(s20, f9);
            result2lo = _mm512_mullo_epi16(s20, f9);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result2lo, result2));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result2lo, result2));
            result3 = _mm512_mulhi_epu16(s21, f8);
            result3lo = _mm512_mullo_epi16(s21, f8);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result3lo, result3));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result3lo, result3));
            result4 = _mm512_mulhi_epu16(s22, f7);
            result4lo = _mm512_mullo_epi16(s22, f7);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result4lo, result4));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result4lo, result4));
            result5 = _mm512_mulhi_epu16(s23, f6);
            result5lo = _mm512_mullo_epi16(s23, f6);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result5lo, result5));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result5lo, result5));
            result6 = _mm512_mulhi_epu16(s24, f5);
            result6lo = _mm512_mullo_epi16(s24, f5);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result6lo, result6));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result6lo, result6));
            result7 = _mm512_mulhi_epu16(s25, f4);
            result7lo = _mm512_mullo_epi16(s25, f4);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result7lo, result7));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result7lo, result7));
            result8 = _mm512_mulhi_epu16(s26, f3);
            result8lo = _mm512_mullo_epi16(s26, f3);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result8lo, result8));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result8lo, result8));
            result9 = _mm512_mulhi_epu16(s27, f2);
            result9lo = _mm512_mullo_epi16(s27, f2);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result9lo, result9));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result9lo, result9));

            result2 = _mm512_mulhi_epu16(sg0, f9);
            result2lo = _mm512_mullo_epi16(sg0, f9);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result2lo, result2));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result2lo, result2));
            result3 = _mm512_mulhi_epu16(sg1, f8);
            result3lo = _mm512_mullo_epi16(sg1, f8);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result3lo, result3));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result3lo, result3));
            result4 = _mm512_mulhi_epu16(sg2, f7);
            result4lo = _mm512_mullo_epi16(sg2, f7);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result4lo, result4));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result4lo, result4));
            result5 = _mm512_mulhi_epu16(sg3, f6);
            result5lo = _mm512_mullo_epi16(sg3, f6);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result5lo, result5));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result5lo, result5));
            result6 = _mm512_mulhi_epu16(sg4, f5);
            result6lo = _mm512_mullo_epi16(sg4, f5);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result6lo, result6));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result6lo, result6));
            result7 = _mm512_mulhi_epu16(sg5, f4);
            result7lo = _mm512_mullo_epi16(sg5, f4);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result7lo, result7));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result7lo, result7));
            result8 = _mm512_mulhi_epu16(sg6, f3);
            result8lo = _mm512_mullo_epi16(sg6, f3);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result8lo, result8));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result8lo, result8));
            result9 = _mm512_mulhi_epu16(sg7, f2);
            result9lo = _mm512_mullo_epi16(sg7, f2);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result9lo, result9));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result9lo, result9));

            g0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 16) + j)));
            g1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(ref + buf.stride * (ii_check + 17) + j)));

            s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 16) + j)));
            s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i *)(dis + buf.stride * (ii_check + 17) + j)));

            s0lo = _mm512_unpacklo_epi16(s0, s1);
            s0hi = _mm512_unpackhi_epi16(s0, s1);
            accum_mu2_lo = _mm512_add_epi32(
                accum_mu2_lo, _mm512_madd_epi16(s0lo, fc8));
            accum_mu2_hi = _mm512_add_epi32(
                accum_mu2_hi, _mm512_madd_epi16(s0hi, fc8));

            g0lo = _mm512_unpacklo_epi16(g0, g1);
            g0hi = _mm512_unpackhi_epi16(g0, g1);
            accum_mu1_lo = _mm512_add_epi32(
                accum_mu1_lo, _mm512_madd_epi16(g0lo, fc8));
            accum_mu1_hi = _mm512_add_epi32(
                accum_mu1_hi, _mm512_madd_epi16(g0hi, fc8));

            g20 = _mm512_mullo_epi16(g0, g0);
            s20 = _mm512_mullo_epi16(s0, s0);
            sg0 = _mm512_mullo_epi16(s0, g0);

            result2 = _mm512_mulhi_epu16(g20, f1);
            result2lo = _mm512_mullo_epi16(g20, f1);
            final_resultlo = _mm512_add_epi32(
                final_resultlo, _mm512_unpacklo_epi16(result2lo, result2));
            final_resulthi = _mm512_add_epi32(
                final_resulthi, _mm512_unpackhi_epi16(result2lo, result2));

            result2 = _mm512_mulhi_epu16(s20, f1);
            result2lo = _mm512_mullo_epi16(s20, f1);
            dislo = _mm512_add_epi32(dislo,
                                     _mm512_unpacklo_epi16(result2lo, result2));
            dishi = _mm512_add_epi32(dishi,
                                     _mm512_unpackhi_epi16(result2lo, result2));

            result2 = _mm512_mulhi_epu16(sg0, f1);
            result2lo = _mm512_mullo_epi16(sg0, f1);
            refdislo = _mm512_add_epi32(
                refdislo, _mm512_unpacklo_epi16(result2lo, result2));
            refdishi = _mm512_add_epi32(
                refdishi, _mm512_unpackhi_epi16(result2lo, result2));

            __m512i accumref_lo =
                _mm512_permutex2var_epi64(final_resultlo, mask2, final_resulthi);
            __m512i accumref_hi =
                _mm512_permutex2var_epi64(final_resultlo, mask3, final_resulthi);
            __m512i accumdis_lo =
                _mm512_permutex2var_epi64(dislo, mask2, dishi);
            __m512i accumdis_hi =
                _mm512_permutex2var_epi64(dislo, mask3, dishi);
            __m512i accumrefdis_lo =
                _mm512_permutex2var_epi64(refdislo, mask2, refdishi);
            __m512i accumrefdis_hi =
                _mm512_permutex2var_epi64(refdislo, mask3, refdishi);
            __m512i x = _mm512_set1_epi32(128);
            __m512i accumu1_lo = _mm512_add_epi32(
                x, _mm512_permutex2var_epi64(accum_mu1_lo, mask2, accum_mu1_hi));
            __m512i accumu1_hi = _mm512_add_epi32(
                x, _mm512_permutex2var_epi64(accum_mu1_lo, mask3, accum_mu1_hi));
            __m512i accumu2_lo = _mm512_add_epi32(
                x, _mm512_permutex2var_epi64(accum_mu2_lo, mask2, accum_mu2_hi));
            __m512i accumu2_hi = _mm512_add_epi32(
                x, _mm512_permutex2var_epi64(accum_mu2_lo, mask3, accum_mu2_hi));
            accumu1_lo = _mm512_srli_epi32(accumu1_lo, 0x08);
            accumu1_hi = _mm512_srli_epi32(accumu1_hi, 0x08);
            accumu2_lo = _mm512_srli_epi32(accumu2_lo, 0x08);
            accumu2_hi = _mm512_srli_epi32(accumu2_hi, 0x08);
            _mm512_storeu_si512((__m512i *)(buf.tmp.ref + j), accumref_lo);
            _mm512_storeu_si512((__m512i *)(buf.tmp.ref + j + 16), accumref_hi);
            _mm512_storeu_si512((__m512i *)(buf.tmp.dis + j), accumdis_lo);
            _mm512_storeu_si512((__m512i *)(buf.tmp.dis + j + 16), accumdis_hi);
            _mm512_storeu_si512((__m512i *)(buf.tmp.ref_dis + j), accumrefdis_lo);
            _mm512_storeu_si512((__m512i *)(buf.tmp.ref_dis + j + 16), accumrefdis_hi);
            _mm512_storeu_si512((__m512i *)(buf.tmp.mu1 + j), accumu1_lo);
            _mm512_storeu_si512((__m512i *)(buf.tmp.mu1 + j + 16), accumu1_hi);
            _mm512_storeu_si512((__m512i *)(buf.tmp.mu2 + j), accumu2_lo);
            _mm512_storeu_si512((__m512i *)(buf.tmp.mu2 + j + 16), accumu2_hi);
        }

        PADDING_SQ_DATA(buf, w, fwidth_half);
        //horizontal

#pragma unroll(4)
        for (unsigned j = 0; j < w; j = j + 32)
        {
            __m512i refdislo, refdishi, mu2lo, mu2hi, mu1lo, mu1hi, s2;
            refdislo = refdishi = mu2lo = mu2hi = mu1lo = s2 = mu1hi = _mm512_setzero_si512();
            int jj = j - fwidth_half;
            int jj_check = jj;

            {

                __m512i s0 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check));
                __m512i s1 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 1));
                __m512i s2 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 2));
                __m512i s3 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 3));
                __m512i s4 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 4));
                __m512i s5 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 5));
                __m512i s6 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 6));
                __m512i s7 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 7));
                __m512i s8 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 8));

                __m512i s00 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 16));
                __m512i s11 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 17));
                __m512i s22 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 18));
                __m512i s33 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 19));
                __m512i s44 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 20));
                __m512i s55 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 21));
                __m512i s66 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 22));
                __m512i s77 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 23));
                __m512i s88 =
                    _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 24));

                s0 = _mm512_packus_epi32(s0, s00);
                s1 = _mm512_packus_epi32(s1, s11);
                s2 = _mm512_packus_epi32(s2, s22);
                s3 = _mm512_packus_epi32(s3, s33);
                s4 = _mm512_packus_epi32(s4, s44);
                s5 = _mm512_packus_epi32(s5, s55);
                s6 = _mm512_packus_epi32(s6, s66);
                s7 = _mm512_packus_epi32(s7, s77);
                s8 = _mm512_packus_epi32(s8, s88);
                __m512i result2 = _mm512_mulhi_epu16(s0, f1);
                __m512i result2lo = _mm512_mullo_epi16(s0, f1);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s1, f2);
                result2lo = _mm512_mullo_epi16(s1, f2);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s2, f3);
                result2lo = _mm512_mullo_epi16(s2, f3);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s3, f4);
                result2lo = _mm512_mullo_epi16(s3, f4);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s4, f5);
                result2lo = _mm512_mullo_epi16(s4, f5);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s5, f6);
                result2lo = _mm512_mullo_epi16(s5, f6);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s6, f7);
                result2lo = _mm512_mullo_epi16(s6, f7);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s7, f8);
                result2lo = _mm512_mullo_epi16(s7, f8);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s8, f9);
                result2lo = _mm512_mullo_epi16(s8, f9);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));

                __m512i g0 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check));
                __m512i g1 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 1));
                __m512i g2 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 2));
                __m512i g3 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 3));
                __m512i g4 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 4));
                __m512i g5 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 5));
                __m512i g6 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 6));
                __m512i g7 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 7));
                __m512i g8 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 8));
                __m512i g00 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 16));
                __m512i g11 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 17));
                __m512i g22 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 18));
                __m512i g33 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 19));
                __m512i g44 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 20));
                __m512i g55 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 21));
                __m512i g66 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 22));
                __m512i g77 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 23));
                __m512i g88 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 24));
                //***********experiment unroll loop in width doer mu2
                g0 = _mm512_packus_epi32(g0, g00);
                g1 = _mm512_packus_epi32(g1, g11);
                g2 = _mm512_packus_epi32(g2, g22);
                g3 = _mm512_packus_epi32(g3, g33);
                g4 = _mm512_packus_epi32(g4, g44);
                g5 = _mm512_packus_epi32(g5, g55);
                g6 = _mm512_packus_epi32(g6, g66);
                g7 = _mm512_packus_epi32(g7, g77);
                g8 = _mm512_packus_epi32(g8, g88);
                result2 = _mm512_mulhi_epu16(g0, f1);
                result2lo = _mm512_mullo_epi16(g0, f1);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g1, f2);
                result2lo = _mm512_mullo_epi16(g1, f2);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g2, f3);
                result2lo = _mm512_mullo_epi16(g2, f3);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g3, f4);
                result2lo = _mm512_mullo_epi16(g3, f4);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g4, f5);
                result2lo = _mm512_mullo_epi16(g4, f5);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g5, f6);
                result2lo = _mm512_mullo_epi16(g5, f6);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g6, f7);
                result2lo = _mm512_mullo_epi16(g6, f7);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g7, f8);
                result2lo = _mm512_mullo_epi16(g7, f8);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g8, f9);
                result2lo = _mm512_mullo_epi16(g8, f9);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));

                s0 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 9));
                s1 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 10));
                s2 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 11));
                s3 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 12));
                s4 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 13));
                s5 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 14));
                s6 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 15));
                s7 = s00;

                s00 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 25));
                s11 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 26));
                s22 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 27));
                s33 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 28));
                s44 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 29));
                s55 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 30));
                s66 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 31));
                s77 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check + 32));

                s0 = _mm512_packus_epi32(s0, s00);
                s1 = _mm512_packus_epi32(s1, s11);
                s2 = _mm512_packus_epi32(s2, s22);
                s3 = _mm512_packus_epi32(s3, s33);
                s4 = _mm512_packus_epi32(s4, s44);
                s5 = _mm512_packus_epi32(s5, s55);
                s6 = _mm512_packus_epi32(s6, s66);
                s7 = _mm512_packus_epi32(s7, s77);

                result2 = _mm512_mulhi_epu16(s0, f8);
                result2lo = _mm512_mullo_epi16(s0, f8);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s1, f7);
                result2lo = _mm512_mullo_epi16(s1, f7);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s2, f6);
                result2lo = _mm512_mullo_epi16(s2, f6);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s3, f5);
                result2lo = _mm512_mullo_epi16(s3, f5);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s4, f4);
                result2lo = _mm512_mullo_epi16(s4, f4);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s5, f3);
                result2lo = _mm512_mullo_epi16(s5, f3);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s6, f2);
                result2lo = _mm512_mullo_epi16(s6, f2);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(s7, f1);
                result2lo = _mm512_mullo_epi16(s7, f1);
                mu1lo = _mm512_add_epi32(
                    mu1lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu1hi = _mm512_add_epi32(
                    mu1hi, _mm512_unpackhi_epi16(result2lo, result2));

                g0 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 9));
                g1 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 10));
                g2 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 11));
                g3 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 12));
                g4 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 13));
                g5 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 14));
                g6 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 15));
                g7 = g00;

                g00 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 25));
                g11 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 26));
                g22 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 27));
                g33 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 28));
                g44 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 29));
                g55 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 30));
                g66 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 31));
                g77 = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check + 32));

                //***********experiment unroll loop in width doer mu2
                g0 = _mm512_packus_epi32(g0, g00);
                g1 = _mm512_packus_epi32(g1, g11);
                g2 = _mm512_packus_epi32(g2, g22);
                g3 = _mm512_packus_epi32(g3, g33);
                g4 = _mm512_packus_epi32(g4, g44);
                g5 = _mm512_packus_epi32(g5, g55);
                g6 = _mm512_packus_epi32(g6, g66);
                g7 = _mm512_packus_epi32(g7, g77);
                g8 = _mm512_packus_epi32(g8, g88);
                result2 = _mm512_mulhi_epu16(g0, f8);
                result2lo = _mm512_mullo_epi16(g0, f8);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g1, f7);
                result2lo = _mm512_mullo_epi16(g1, f7);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g2, f6);
                result2lo = _mm512_mullo_epi16(g2, f6);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g3, f5);
                result2lo = _mm512_mullo_epi16(g3, f5);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g4, f4);
                result2lo = _mm512_mullo_epi16(g4, f4);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g5, f3);
                result2lo = _mm512_mullo_epi16(g5, f3);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g6, f2);
                result2lo = _mm512_mullo_epi16(g6, f2);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
                result2 = _mm512_mulhi_epu16(g7, f1);
                result2lo = _mm512_mullo_epi16(g7, f1);
                mu2lo = _mm512_add_epi32(
                    mu2lo, _mm512_unpacklo_epi16(result2lo, result2));
                mu2hi = _mm512_add_epi32(
                    mu2hi, _mm512_unpackhi_epi16(result2lo, result2));
            }

            _mm512_storeu_si512((__m512i *)(buf.mu1_32 + (dst_stride * i) + j), mu1lo);
            _mm512_storeu_si512((__m512i *)(buf.mu1_32 + (dst_stride * i) + j + 16), mu1hi);
            _mm512_storeu_si512((__m512i *)(buf.mu2_32 + (dst_stride * i) + j), mu2lo);
            _mm512_storeu_si512((__m512i *)(buf.mu2_32 + (dst_stride * i) + j + 16), mu2hi);
        }

#pragma unroll(4)
        for (unsigned j = 0; j < w; j = j + 16)
        {
            __m512i refdislo, refdishi, reflo, refhi, dislo, dishi;
            refdislo = refdishi = reflo = refhi = dislo = dishi = _mm512_setzero_si512();
            int jj = j - fwidth_half;
            int jj_check = jj;
            __m512i addnum = _mm512_set1_epi64(32768);
            {

                __m512i s0 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check)));
                __m512i s1 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 1)));
                __m512i s2 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 2)));
                __m512i s3 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 3)));
                __m512i s4 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 4)));
                __m512i s5 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 5)));
                __m512i s6 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 6)));
                __m512i s7 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 7)));
                __m512i s8 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 8)));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s0, fq1));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s1, fq2));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s2, fq3));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s3, fq4));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s4, fq5));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s5, fq6));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s6, fq7));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s7, fq8));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s8, fq9));

                //__m512i s9 =
                //_mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256*)(buf.tmp.ref+jj_check+8)));
                __m512i s10 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 9)));
                __m512i s11 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 10)));
                __m512i s12 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 11)));
                __m512i s13 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 12)));
                __m512i s14 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 13)));
                __m512i s15 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 14)));
                __m512i s16 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 15)));
                __m512i s17 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 16)));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s8, fq1));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s10, fq2));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s11, fq3));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s12, fq4));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s13, fq5));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s14, fq6));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s15, fq7));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s16, fq8));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s17, fq9));

                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s10, fq8));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s11, fq7));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s12, fq6));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s13, fq5));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s14, fq4));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s15, fq3));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s16, fq2));
                reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s17, fq1));

                s10 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 17)));
                s11 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 18)));
                s12 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 19)));
                s13 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 20)));
                s14 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 21)));
                s15 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 22)));
                s16 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 23)));
                s17 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref + jj_check + 24)));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s10, fq8));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s11, fq7));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s12, fq6));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s13, fq5));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s14, fq4));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s15, fq3));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s16, fq2));
                refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s17, fq1));

                __m512i g0 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check)));
                __m512i g1 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 1)));
                __m512i g2 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 2)));
                __m512i g3 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 3)));
                __m512i g4 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 4)));
                __m512i g5 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 5)));
                __m512i g6 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 6)));
                __m512i g7 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 7)));
                __m512i g8 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 8)));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g0, fq1));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g1, fq2));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g2, fq3));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g3, fq4));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g4, fq5));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g5, fq6));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g6, fq7));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g7, fq8));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g8, fq9));

                //__m512i g9 =
                //_mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256*)(buf.tmp.dis+jj_check+8)));
                __m512i g10 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 9)));
                __m512i g11 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 10)));
                __m512i g12 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 11)));
                __m512i g13 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 12)));
                __m512i g14 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 13)));
                __m512i g15 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 14)));
                __m512i g16 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 15)));
                __m512i g17 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 16)));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g8, fq1));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g10, fq2));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g11, fq3));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g12, fq4));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g13, fq5));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g14, fq6));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g15, fq7));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g16, fq8));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g17, fq9));

                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g10, fq8));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g11, fq7));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g12, fq6));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g13, fq5));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g14, fq4));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g15, fq3));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g16, fq2));
                dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g17, fq1));

                g10 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 17)));
                g11 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 18)));
                g12 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 19)));
                g13 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 20)));
                g14 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 21)));
                g15 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 22)));
                g16 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 23)));
                g17 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.dis + jj_check + 24)));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g10, fq8));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g11, fq7));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g12, fq6));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g13, fq5));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g14, fq4));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g15, fq3));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g16, fq2));
                dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g17, fq1));

                __m512i sg0 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check)));
                __m512i sg1 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 1)));
                __m512i sg2 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 2)));
                __m512i sg3 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 3)));
                __m512i sg4 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 4)));
                __m512i sg5 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 5)));
                __m512i sg6 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 6)));
                __m512i sg7 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 7)));
                __m512i sg8 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 8)));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg0, fq1));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg1, fq2));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg2, fq3));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg3, fq4));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg4, fq5));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg5, fq6));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg6, fq7));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg7, fq8));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg8, fq9));

                //__m512i  sg9 =
                //_mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256*)(buf.tmp.ref_dis+jj_check+8)));
                __m512i sg10 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 9)));
                __m512i sg11 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 10)));
                __m512i sg12 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 11)));
                __m512i sg13 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 12)));
                __m512i sg14 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 13)));
                __m512i sg15 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 14)));
                __m512i sg16 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 15)));
                __m512i sg17 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 16)));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg8, fq1));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg10, fq2));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg11, fq3));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg12, fq4));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg13, fq5));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg14, fq6));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg15, fq7));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg16, fq8));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg17, fq9));

                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg10, fq8));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg11, fq7));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg12, fq6));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg13, fq5));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg14, fq4));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg15, fq3));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg16, fq2));
                refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg17, fq1));

                sg10 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 17)));
                sg11 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 18)));
                sg12 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 19)));
                sg13 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 20)));
                sg14 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 21)));
                sg15 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 22)));
                sg16 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 23)));
                sg17 = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256 *)(buf.tmp.ref_dis + jj_check + 24)));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg10, fq8));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg11, fq7));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg12, fq6));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg13, fq5));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg14, fq4));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg15, fq3));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg16, fq2));
                refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg17, fq1));
            }
            reflo = _mm512_add_epi64(reflo, addnum);
            reflo = _mm512_srli_epi64(reflo, 0x10);
            refhi = _mm512_add_epi64(refhi, addnum);
            refhi = _mm512_srli_epi64(refhi, 0x10);

            __m512i mask5 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18,
                                             16, 14, 12, 10, 8, 6, 4, 2, 0);
            _mm512_storeu_si512((__m512i *)(buf.ref_sq + (dst_stride * i) + j),
                                _mm512_permutex2var_epi32(reflo, mask5, refhi));

            dislo = _mm512_add_epi64(dislo, addnum);
            dislo = _mm512_srli_epi64(dislo, 0x10);
            dishi = _mm512_add_epi64(dishi, addnum);
            dishi = _mm512_srli_epi64(dishi, 0x10);

            _mm512_storeu_si512((__m512i *)(buf.dis_sq + (dst_stride * i) + j),
                                _mm512_permutex2var_epi32(dislo, mask5, dishi));

            refdislo = _mm512_add_epi64(refdislo, addnum);
            refdislo = _mm512_srli_epi64(refdislo, 0x10);
            refdishi = _mm512_add_epi64(refdishi, addnum);
            refdishi = _mm512_srli_epi64(refdishi, 0x10);

            _mm512_storeu_si512((__m512i *)(buf.ref_dis + (dst_stride * i) + j),
                                _mm512_permutex2var_epi32(refdislo, mask5, refdishi));
        }
    }
}

void vif_filter1d_16_avx512(VifBuffer buf, unsigned w, unsigned h, int scale,
                            int bpc)
{
    const unsigned fwidth = vif_filter1d_width[scale];
    const uint16_t *vif_filt = vif_filter1d_table[scale];
    const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    int fwidth_half = fwidth >> 1;

    int32_t add_shift_round_HP, shift_HP;
    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;

    if (scale == 0)
    {
        shift_HP = 16;
        add_shift_round_HP = 32768;
        shift_VP = bpc;
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP_sq = (bpc - 8) * 2;
        add_shift_round_VP_sq = (bpc == 8) ? 0 : 1 << (shift_VP_sq - 1);
    }
    else
    {
        shift_HP = 16;
        add_shift_round_HP = 32768;
        shift_VP = 16;
        add_shift_round_VP = 32768;
        shift_VP_sq = 16;
        add_shift_round_VP_sq = 32768;
    }
    __m512i addnum64 = _mm512_set1_epi64(add_shift_round_VP_sq);
    __m512i addnum = _mm512_set1_epi32(add_shift_round_VP);
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    __m512i mask2 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);

    {
        for (unsigned i = 0; i < h; ++i)
        {
            //VERTICAL
            int ii = i - fwidth_half;
            int n = w >> 5;
            for (unsigned j = 0; j < n << 5; j = j + 32)
            {

                __m512i mask3 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);   //first half of 512
                __m512i mask4 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4); //second half of 512
                int ii_check = ii;
                __m512i accumr_lo, accumr_hi, accumd_lo, accumd_hi, rmul1, rmul2,
                    dmul1, dmul2, accumref1, accumref2, accumref3, accumref4,
                    accumrefdis1, accumrefdis2, accumrefdis3, accumrefdis4,
                    accumdis1, accumdis2, accumdis3, accumdis4;
                accumr_lo = accumr_hi = accumd_lo = accumd_hi = rmul1 = rmul2 = dmul1 = dmul2 = accumref1 = accumref2 = accumref3 = accumref4 = accumrefdis1 = accumrefdis2 = accumrefdis3 =
                    accumrefdis4 = accumdis1 = accumdis2 = accumdis3 = accumdis4 = _mm512_setzero_si512();

                for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi)
                {

                    const uint16_t fcoeff = vif_filt[fi];
                    __m512i f1 = _mm512_set1_epi16(vif_filt[fi]);
                    __m512i ref1 = _mm512_loadu_si512(
                        (__m512i *)(ref + (ii_check * stride) + j));
                    __m512i dis1 = _mm512_loadu_si512(
                        (__m512i *)(dis + (ii_check * stride) + j));
                    __m512i result2 = _mm512_mulhi_epu16(ref1, f1);
                    __m512i result2lo = _mm512_mullo_epi16(ref1, f1);
                    __m512i rmult1 = _mm512_unpacklo_epi16(result2lo, result2);
                    __m512i rmult2 = _mm512_unpackhi_epi16(result2lo, result2);
                    rmul1 = _mm512_permutex2var_epi64(rmult1, mask3, rmult2);
                    rmul2 = _mm512_permutex2var_epi64(rmult1, mask4, rmult2);
                    accumr_lo = _mm512_add_epi32(accumr_lo, rmul1);
                    accumr_hi = _mm512_add_epi32(accumr_hi, rmul2);
                    __m512i d0 = _mm512_mulhi_epu16(dis1, f1);
                    __m512i d0lo = _mm512_mullo_epi16(dis1, f1);
                    __m512i dmult1 = _mm512_unpacklo_epi16(d0lo, d0);
                    __m512i dmult2 = _mm512_unpackhi_epi16(d0lo, d0);
                    dmul1 = _mm512_permutex2var_epi64(dmult1, mask3, dmult2);
                    dmul2 = _mm512_permutex2var_epi64(dmult1, mask4, dmult2);
                    accumd_lo = _mm512_add_epi32(accumd_lo, dmul1);
                    accumd_hi = _mm512_add_epi32(accumd_hi, dmul2);

                    __m512i sg0 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(rmul1));
                    __m512i sg1 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(rmul1, 1));
                    __m512i sg2 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(rmul2));
                    __m512i sg3 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(rmul2, 1));
                    __m128i l0 = _mm512_castsi512_si128(ref1);
                    __m128i l1 = _mm512_extracti32x4_epi32(ref1, 1);
                    __m128i l2 = _mm512_extracti32x4_epi32(ref1, 2);
                    __m128i l3 = _mm512_extracti32x4_epi32(ref1, 3);
                    accumref1 = _mm512_add_epi64(accumref1,
                                                 _mm512_mul_epu32(sg0, _mm512_cvtepu16_epi64(l0)));
                    accumref2 = _mm512_add_epi64(accumref2,
                                                 _mm512_mul_epu32(sg2, _mm512_cvtepu16_epi64(l2)));
                    accumref3 = _mm512_add_epi64(accumref3,
                                                 _mm512_mul_epu32(sg1, _mm512_cvtepu16_epi64(l1)));
                    accumref4 = _mm512_add_epi64(accumref4,
                                                 _mm512_mul_epu32(sg3, _mm512_cvtepu16_epi64(l3)));
                    l0 = _mm512_castsi512_si128(dis1);
                    l1 = _mm512_extracti32x4_epi32(dis1, 1);
                    l2 = _mm512_extracti32x4_epi32(dis1, 2);
                    l3 = _mm512_extracti32x4_epi32(dis1, 3);

                    accumrefdis1 = _mm512_add_epi64(accumrefdis1,
                                                    _mm512_mul_epu32(sg0, _mm512_cvtepu16_epi64(l0)));
                    accumrefdis2 = _mm512_add_epi64(accumrefdis2,
                                                    _mm512_mul_epu32(sg2, _mm512_cvtepu16_epi64(l2)));
                    accumrefdis3 = _mm512_add_epi64(accumrefdis3,
                                                    _mm512_mul_epu32(sg1, _mm512_cvtepu16_epi64(l1)));
                    accumrefdis4 = _mm512_add_epi64(accumrefdis4,
                                                    _mm512_mul_epu32(sg3, _mm512_cvtepu16_epi64(l3)));
                    __m512i sd0 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(dmul1));
                    __m512i sd1 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(dmul1, 1));
                    __m512i sd2 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(dmul2));
                    __m512i sd3 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(dmul2, 1));
                    accumdis1 = _mm512_add_epi64(accumdis1,
                                                 _mm512_mul_epu32(sd0, _mm512_cvtepu16_epi64(l0)));
                    accumdis2 = _mm512_add_epi64(accumdis2,
                                                 _mm512_mul_epu32(sd2, _mm512_cvtepu16_epi64(l2)));
                    accumdis3 = _mm512_add_epi64(accumdis3,
                                                 _mm512_mul_epu32(sd1, _mm512_cvtepu16_epi64(l1)));
                    accumdis4 = _mm512_add_epi64(accumdis4,
                                                 _mm512_mul_epu32(sd3, _mm512_cvtepu16_epi64(l3)));
                }
                accumr_lo = _mm512_add_epi32(accumr_lo, addnum);
                accumr_hi = _mm512_add_epi32(accumr_hi, addnum);
                accumr_lo = _mm512_srli_epi32(accumr_lo, shift_VP);
                accumr_hi = _mm512_srli_epi32(accumr_hi, shift_VP);
                _mm512_storeu_si512((__m512i *)(buf.tmp.mu1 + j), accumr_lo);
                _mm512_storeu_si512((__m512i *)(buf.tmp.mu1 + j + 16), accumr_hi);

                accumd_lo = _mm512_add_epi32(accumd_lo, addnum);
                accumd_hi = _mm512_add_epi32(accumd_hi, addnum);
                accumd_lo = _mm512_srli_epi32(accumd_lo, shift_VP);
                accumd_hi = _mm512_srli_epi32(accumd_hi, shift_VP);
                _mm512_storeu_si512((__m512i *)(buf.tmp.mu2 + j), accumd_lo);
                _mm512_storeu_si512((__m512i *)(buf.tmp.mu2 + j + 16), accumd_hi);

                accumref1 = _mm512_add_epi64(accumref1, addnum64);
                accumref2 = _mm512_add_epi64(accumref2, addnum64);
                accumref3 = _mm512_add_epi64(accumref3, addnum64);
                accumref4 = _mm512_add_epi64(accumref4, addnum64);
                accumref1 = _mm512_srli_epi64(accumref1, shift_VP_sq);
                accumref2 = _mm512_srli_epi64(accumref2, shift_VP_sq);
                accumref3 = _mm512_srli_epi64(accumref3, shift_VP_sq);
                accumref4 = _mm512_srli_epi64(accumref4, shift_VP_sq);

                _mm512_storeu_si512((__m512i *)(buf.tmp.ref + j),
                                    _mm512_permutex2var_epi32(accumref1, mask2, accumref3));
                _mm512_storeu_si512((__m512i *)(buf.tmp.ref + 16 + j),
                                    _mm512_permutex2var_epi32(accumref2, mask2, accumref4));

                accumrefdis1 = _mm512_add_epi64(accumrefdis1, addnum64);
                accumrefdis2 = _mm512_add_epi64(accumrefdis2, addnum64);
                accumrefdis3 = _mm512_add_epi64(accumrefdis3, addnum64);
                accumrefdis4 = _mm512_add_epi64(accumrefdis4, addnum64);
                accumrefdis1 = _mm512_srli_epi64(accumrefdis1, shift_VP_sq);
                accumrefdis2 = _mm512_srli_epi64(accumrefdis2, shift_VP_sq);
                accumrefdis3 = _mm512_srli_epi64(accumrefdis3, shift_VP_sq);
                accumrefdis4 = _mm512_srli_epi64(accumrefdis4, shift_VP_sq);

                _mm512_storeu_si512((__m512i *)(buf.tmp.ref_dis + j),
                                    _mm512_permutex2var_epi32(accumrefdis1, mask2, accumrefdis3));
                _mm512_storeu_si512((__m512i *)(buf.tmp.ref_dis + 16 + j),
                                    _mm512_permutex2var_epi32(accumrefdis2, mask2, accumrefdis4));

                accumdis1 = _mm512_add_epi64(accumdis1, addnum64);
                accumdis2 = _mm512_add_epi64(accumdis2, addnum64);
                accumdis3 = _mm512_add_epi64(accumdis3, addnum64);
                accumdis4 = _mm512_add_epi64(accumdis4, addnum64);
                accumdis1 = _mm512_srli_epi64(accumdis1, shift_VP_sq);
                accumdis2 = _mm512_srli_epi64(accumdis2, shift_VP_sq);
                accumdis3 = _mm512_srli_epi64(accumdis3, shift_VP_sq);
                accumdis4 = _mm512_srli_epi64(accumdis4, shift_VP_sq);

                _mm512_storeu_si512((__m512i *)(buf.tmp.dis + j),
                                    _mm512_permutex2var_epi32(accumdis1, mask2, accumdis3));
                _mm512_storeu_si512((__m512i *)(buf.tmp.dis + 16 + j),
                                    _mm512_permutex2var_epi32(accumdis2, mask2, accumdis4));
            }

            for (unsigned j = n << 5; j < w; ++j)
            {
                uint32_t accum_mu1 = 0;
                uint32_t accum_mu2 = 0;
                uint64_t accum_ref = 0;
                uint64_t accum_dis = 0;
                uint64_t accum_ref_dis = 0;
                for (unsigned fi = 0; fi < fwidth; ++fi)
                {
                    int ii = i - fwidth / 2;
                    int ii_check = ii + fi;
                    const uint16_t fcoeff = vif_filt[fi];
                    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
                    uint16_t *ref = buf.ref;
                    uint16_t *dis = buf.dis;
                    uint16_t imgcoeff_ref = ref[ii_check * stride + j];
                    uint16_t imgcoeff_dis = dis[ii_check * stride + j];
                    uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                    uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                    accum_mu1 += img_coeff_ref;
                    accum_mu2 += img_coeff_dis;
                    accum_ref += img_coeff_ref * (uint64_t)imgcoeff_ref;
                    accum_dis += img_coeff_dis * (uint64_t)imgcoeff_dis;
                    accum_ref_dis += img_coeff_ref * (uint64_t)imgcoeff_dis;
                }
                buf.tmp.mu1[j] = (uint16_t)((accum_mu1 + add_shift_round_VP) >> shift_VP);
                buf.tmp.mu2[j] = (uint16_t)((accum_mu2 + add_shift_round_VP) >> shift_VP);
                buf.tmp.ref[j] = (uint32_t)((accum_ref + add_shift_round_VP_sq) >> shift_VP_sq);
                buf.tmp.dis[j] = (uint32_t)((accum_dis + add_shift_round_VP_sq) >> shift_VP_sq);
                buf.tmp.ref_dis[j] = (uint32_t)((accum_ref_dis + add_shift_round_VP_sq) >> shift_VP_sq);
            }

            PADDING_SQ_DATA(buf, w, fwidth_half);

            //HORIZONTAL
            n = w >> 4;
            for (unsigned j = 0; j < n << 4; j = j + 16)
            {
                int jj = j - fwidth_half;
                int jj_check = jj;
                __m512i accumdl, accumrlo, accumdlo, accumrhi, accumdhi;
                accumrlo = accumdlo = accumrhi = accumdhi = _mm512_setzero_si512();
                __m512i mask2 = _mm512_set_epi32(30, 28, 14, 12, 26, 24, 10, 8, 22, 20, 6, 4, 18, 16, 2, 0);
#pragma unroll(4)
                for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj)
                {

                    __m512i refconvol = _mm512_loadu_si512((__m512i *)(buf.tmp.mu1 + jj_check));
                    __m512i fcoeff = _mm512_set1_epi16(vif_filt[fj]);
                    __m512i result2 = _mm512_mulhi_epu16(refconvol, fcoeff);
                    __m512i result2lo = _mm512_mullo_epi16(refconvol, fcoeff);
                    accumrlo = _mm512_add_epi32(accumrlo, _mm512_unpacklo_epi16(result2lo, result2));
                    accumrhi = _mm512_add_epi32(accumrhi, _mm512_unpackhi_epi16(result2lo, result2));

                    __m512i disconvol = _mm512_loadu_si512((__m512i *)(buf.tmp.mu2 + jj_check));
                    result2 = _mm512_mulhi_epu16(disconvol, fcoeff);
                    result2lo = _mm512_mullo_epi16(disconvol, fcoeff);
                    accumdlo = _mm512_add_epi32(accumdlo,
                                                _mm512_unpacklo_epi16(result2lo, result2));
                    accumdhi = _mm512_add_epi32(accumdhi,
                                                _mm512_unpackhi_epi16(result2lo, result2));
                }

                _mm512_storeu_si512((__m512i *)(buf.mu1_32 + (dst_stride * i) + j),
                                    _mm512_permutex2var_epi32(accumrlo, mask2, accumrhi));
                _mm512_storeu_si512((__m512i *)(buf.mu2_32 + (dst_stride * i) + j),
                                    _mm512_permutex2var_epi32(accumdlo, mask2, accumdhi));
            }

            for (unsigned j = 0; j < n << 4; j = j + 16)
            {

                __m512i refdislo, refdishi, reflo, refhi, dislo, dishi;
                refdislo = refdishi = reflo = refhi = dislo = dishi = _mm512_setzero_si512();
                int jj = j - fwidth_half;
                int jj_check = jj;
                __m512i addnum = _mm512_set1_epi64(add_shift_round_HP);
#pragma unroll(2)
                for (unsigned fj = 0; fj < fwidth; fj = ++fj, jj_check = jj + fj)
                {

                    __m512i f1 = _mm512_set1_epi64(vif_filt[fj]);
                    __m512i s0 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i *)(buf.tmp.ref + jj_check)));
                    reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s0, f1));
                    __m512i s3 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i *)(buf.tmp.ref + jj_check + 8)));
                    refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s3, f1));

                    __m512i g0 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i *)(buf.tmp.dis + jj_check)));
                    dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g0, f1));
                    __m512i g3 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i *)(buf.tmp.dis + jj_check + 8)));
                    dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g3, f1));

                    __m512i sg0 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i *)(buf.tmp.ref_dis + jj_check)));
                    refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg0, f1));
                    __m512i sg3 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i *)(buf.tmp.ref_dis + jj_check + 8)));
                    refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg3, f1));
                }
                reflo = _mm512_add_epi64(reflo, addnum);
                reflo = _mm512_srli_epi64(reflo, 0x10);
                refhi = _mm512_add_epi64(refhi, addnum);
                refhi = _mm512_srli_epi64(refhi, 0x10);

                _mm512_storeu_si512((__m512i *)(buf.ref_sq + (dst_stride * i) + j),
                                    _mm512_permutex2var_epi32(reflo, mask2, refhi));

                dislo = _mm512_add_epi64(dislo, addnum);
                dislo = _mm512_srli_epi64(dislo, 0x10);
                dishi = _mm512_add_epi64(dishi, addnum);
                dishi = _mm512_srli_epi64(dishi, 0x10);
                _mm512_storeu_si512((__m512i *)(buf.dis_sq + (dst_stride * i) + j),
                                    _mm512_permutex2var_epi32(dislo, mask2, dishi));

                refdislo = _mm512_add_epi64(refdislo, addnum);
                refdislo = _mm512_srli_epi64(refdislo, 0x10);
                refdishi = _mm512_add_epi64(refdishi, addnum);
                refdishi = _mm512_srli_epi64(refdishi, 0x10);
                _mm512_storeu_si512((__m512i *)(buf.ref_dis + (dst_stride * i) + j),
                                    _mm512_permutex2var_epi32(refdislo, mask2, refdishi));
            }

            for (unsigned j = n << 4; j < w; ++j)
            {
                uint32_t accum_mu1 = 0;
                uint32_t accum_mu2 = 0;
                uint64_t accum_ref = 0;
                uint64_t accum_dis = 0;
                uint64_t accum_ref_dis = 0;
                int jj = j - fwidth_half;
                int jj_check = jj;
                for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj)
                {
                    const uint16_t fcoeff = vif_filt[fj];
                    accum_mu1 += fcoeff * ((uint32_t)buf.tmp.mu1[jj_check]);
                    accum_mu2 += fcoeff * ((uint32_t)buf.tmp.mu2[jj_check]);
                    accum_ref += fcoeff * ((uint64_t)buf.tmp.ref[jj_check]);
                    accum_dis += fcoeff * ((uint64_t)buf.tmp.dis[jj_check]);
                    accum_ref_dis += fcoeff * ((uint64_t)buf.tmp.ref_dis[jj_check]);
                }
                const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
                buf.mu1_32[i * dst_stride + j] = accum_mu1;
                buf.mu2_32[i * dst_stride + j] = accum_mu2;
                buf.ref_sq[i * dst_stride + j] = (uint32_t)((accum_ref + add_shift_round_HP) >> shift_HP);
                buf.dis_sq[i * dst_stride + j] = (uint32_t)((accum_dis + add_shift_round_HP) >> shift_HP);
                buf.ref_dis[i * dst_stride + j] = (uint32_t)((accum_ref_dis + add_shift_round_HP) >> shift_HP);
            }
        }
    }
}

void vif_filter1d_rd_8_avx512(VifBuffer buf, unsigned w, unsigned h)
{
    const unsigned fwidth = vif_filter1d_width[1];
    const uint16_t *vif_filt_s1 = vif_filter1d_table[1];
    int fwidth_x = (fwidth % 2 == 0) ? fwidth : fwidth + 1;
    const uint8_t *ref = (uint8_t *)(buf.ref);
    const uint8_t *dis = (uint8_t *)buf.dis;
    const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
    __m512i addnum = _mm512_set1_epi32(32768);
    __m512i mask1 = _mm512_set_epi16(60, 56, 28, 24, 52, 48, 20, 16, 44,
                                     40, 12, 8, 36, 32, 4, 0, 60, 56, 28, 24,
                                     52, 48, 20, 16, 44, 40, 12, 8, 36, 32, 4, 0);
    __m512i x = _mm512_set1_epi32(128);
    __m512i mask2 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    __m512i mask3 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    int fwidth_half = fwidth >> 1;
    __m512i f0, f1, f2, f3, f4, f5, f6, f7, f8;

    f0 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)vif_filt_s1));
    f1 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s1 + 2)));
    f2 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s1 + 4)));
    f3 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s1 + 6)));
    f4 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i *)(vif_filt_s1 + 8)));

    __m512i fcoeff = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)vif_filt_s1));
    __m512i fcoeff1 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s1 + 1)));
    __m512i fcoeff2 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s1 + 2)));
    __m512i fcoeff3 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s1 + 3)));
    __m512i fcoeff4 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i *)(vif_filt_s1 + 4)));

    for (unsigned i = 0; i < h; ++i)
    {
        //VERTICAL
        int n = w >> 5;
        int ii = i - fwidth_half;
        for (unsigned j = 0; j < n << 5; j = j + 32)
        {

            int ii_check = ii;
            __m512i accum_mu2_lo, accum_mu1_lo, accum_mu2_hi, accum_mu1_hi;
            accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi = _mm512_setzero_si512();

            {
                __m512i g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g20, g21;
                __m512i s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s20, s21, sg0, sg1;

                g0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + (buf.stride * ii_check) + j)));
                g1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check) + buf.stride + j)));
                g2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check + 2) + j)));
                g3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check + 3) + j)));
                g4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check + 4) + j)));
                g5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check + 5) + j)));
                g6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check + 6) + j)));
                g7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check + 7) + j)));
                g8 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check + 8) + j)));
                g9 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(ref + buf.stride * (ii_check + 9) + j)));

                s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + (buf.stride * ii_check) + j)));
                s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 1) + j)));
                s2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 2) + j)));
                s3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 3) + j)));
                s4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 4) + j)));
                s5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 5) + j)));
                s6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 6) + j)));
                s7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 7) + j)));
                s8 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 8) + j)));
                s9 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(dis + buf.stride * (ii_check + 9) + j)));

                __m512i s0lo = _mm512_unpacklo_epi16(s0, s1);
                __m512i s0hi = _mm512_unpackhi_epi16(s0, s1);
                accum_mu2_lo = _mm512_add_epi32(accum_mu2_lo,
                                                _mm512_madd_epi16(s0lo, f0));
                accum_mu2_hi = _mm512_add_epi32(accum_mu2_hi,
                                                _mm512_madd_epi16(s0hi, f0));
                __m512i s1lo = _mm512_unpacklo_epi16(s2, s3);
                __m512i s1hi = _mm512_unpackhi_epi16(s2, s3);
                accum_mu2_lo = _mm512_add_epi32(accum_mu2_lo,
                                                _mm512_madd_epi16(s1lo, f1));
                accum_mu2_hi = _mm512_add_epi32(accum_mu2_hi,
                                                _mm512_madd_epi16(s1hi, f1));
                __m512i s2lo = _mm512_unpacklo_epi16(s4, s5);
                __m512i s2hi = _mm512_unpackhi_epi16(s4, s5);
                accum_mu2_lo = _mm512_add_epi32(accum_mu2_lo,
                                                _mm512_madd_epi16(s2lo, f2));
                accum_mu2_hi = _mm512_add_epi32(accum_mu2_hi,
                                                _mm512_madd_epi16(s2hi, f2));
                __m512i s3lo = _mm512_unpacklo_epi16(s6, s7);
                __m512i s3hi = _mm512_unpackhi_epi16(s6, s7);
                accum_mu2_lo = _mm512_add_epi32(accum_mu2_lo,
                                                _mm512_madd_epi16(s3lo, f3));
                accum_mu2_hi = _mm512_add_epi32(accum_mu2_hi,
                                                _mm512_madd_epi16(s3hi, f3));
                __m512i s4lo = _mm512_unpacklo_epi16(s8, s9);
                __m512i s4hi = _mm512_unpackhi_epi16(s8, s9);
                accum_mu2_lo = _mm512_add_epi32(accum_mu2_lo,
                                                _mm512_madd_epi16(s4lo, f4));
                accum_mu2_hi = _mm512_add_epi32(accum_mu2_hi,
                                                _mm512_madd_epi16(s4hi, f4));

                __m512i g0lo = _mm512_unpacklo_epi16(g0, g1);
                __m512i g0hi = _mm512_unpackhi_epi16(g0, g1);
                accum_mu1_lo = _mm512_add_epi32(accum_mu1_lo,
                                                _mm512_madd_epi16(g0lo, f0));
                accum_mu1_hi = _mm512_add_epi32(accum_mu1_hi,
                                                _mm512_madd_epi16(g0hi, f0));
                __m512i g1lo = _mm512_unpacklo_epi16(g2, g3);
                __m512i g1hi = _mm512_unpackhi_epi16(g2, g3);
                accum_mu1_lo = _mm512_add_epi32(accum_mu1_lo,
                                                _mm512_madd_epi16(g1lo, f1));
                accum_mu1_hi = _mm512_add_epi32(accum_mu1_hi,
                                                _mm512_madd_epi16(g1hi, f1));
                __m512i g2lo = _mm512_unpacklo_epi16(g4, g5);
                __m512i g2hi = _mm512_unpackhi_epi16(g4, g5);
                accum_mu1_lo = _mm512_add_epi32(accum_mu1_lo,
                                                _mm512_madd_epi16(g2lo, f2));
                accum_mu1_hi = _mm512_add_epi32(accum_mu1_hi,
                                                _mm512_madd_epi16(g2hi, f2));
                __m512i g3lo = _mm512_unpacklo_epi16(g6, g7);
                __m512i g3hi = _mm512_unpackhi_epi16(g6, g7);
                accum_mu1_lo = _mm512_add_epi32(accum_mu1_lo,
                                                _mm512_madd_epi16(g3lo, f3));
                accum_mu1_hi = _mm512_add_epi32(accum_mu1_hi,
                                                _mm512_madd_epi16(g3hi, f3));
                __m512i g4lo = _mm512_unpacklo_epi16(g8, g9);
                __m512i g4hi = _mm512_unpackhi_epi16(g8, g9);
                accum_mu1_lo = _mm512_add_epi32(accum_mu1_lo,
                                                _mm512_madd_epi16(g4lo, f4));
                accum_mu1_hi = _mm512_add_epi32(accum_mu1_hi,
                                                _mm512_madd_epi16(g4hi, f4));
            }
            __m512i accumu1_lo = _mm512_add_epi32(x,
                                                  _mm512_permutex2var_epi64(accum_mu1_lo, mask2, accum_mu1_hi));
            __m512i accumu1_hi = _mm512_add_epi32(x,
                                                  _mm512_permutex2var_epi64(accum_mu1_lo, mask3, accum_mu1_hi));
            __m512i accumu2_lo = _mm512_add_epi32(x,
                                                  _mm512_permutex2var_epi64(accum_mu2_lo, mask2, accum_mu2_hi));
            __m512i accumu2_hi = _mm512_add_epi32(x,
                                                  _mm512_permutex2var_epi64(accum_mu2_lo, mask3, accum_mu2_hi));
            accumu1_lo = _mm512_srli_epi32(accumu1_lo, 0x08);
            accumu1_hi = _mm512_srli_epi32(accumu1_hi, 0x08);
            accumu2_lo = _mm512_srli_epi32(accumu2_lo, 0x08);
            accumu2_hi = _mm512_srli_epi32(accumu2_hi, 0x08);
            _mm512_storeu_si512((__m512i *)(buf.tmp.ref_convol + j), accumu1_lo);
            _mm512_storeu_si512((__m512i *)(buf.tmp.ref_convol + j + 16), accumu1_hi);
            _mm512_storeu_si512((__m512i *)(buf.tmp.dis_convol + j), accumu2_lo);
            _mm512_storeu_si512((__m512i *)(buf.tmp.dis_convol + j + 16), accumu2_hi);
        }
        for (unsigned j = n << 5; j < w; ++j)
        {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            for (unsigned fi = 0; fi < fwidth; ++fi)
            {
                int ii = i - fwidth_half;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt_s1[fi];
                const uint8_t *ref = (uint8_t *)buf.ref;
                const uint8_t *dis = (uint8_t *)buf.dis;
                accum_ref += fcoeff * (uint32_t)ref[ii_check * buf.stride + j];
                accum_dis += fcoeff * (uint32_t)dis[ii_check * buf.stride + j];
            }
            buf.tmp.ref_convol[j] = (accum_ref + 128) >> 8;
            buf.tmp.dis_convol[j] = (accum_dis + 128) >> 8;
        }

        PADDING_SQ_DATA_2(buf, w, fwidth_half);

        //HORIZONTAL
        n = w >> 4;
        for (unsigned j = 0; j < n << 4; j = j + 16)
        {
            int jj = j - fwidth_half;
            int jj_check = jj;
            __m512i accumdl, accumrlo, accumdlo, accumrhi, accumdhi, padzero;
            accumrlo = accumdlo = accumrhi = accumdhi = padzero = _mm512_setzero_si512();

            {

                __m512i refconvol = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check));
                __m512i refconvol1 = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check + 1));
                __m512i refconvol2 = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check + 2));
                __m512i refconvol3 = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check + 3));
                __m512i refconvol4 = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check + 4));
                __m512i refconvol5 = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check + 5));
                __m512i refconvol6 = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check + 6));
                __m512i refconvol7 = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check + 7));
                __m512i refconvol8 = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check + 8));

                __m512i result2 = _mm512_mulhi_epu16(refconvol, fcoeff);
                __m512i result2lo = _mm512_mullo_epi16(refconvol, fcoeff);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result2lo, result2));
                accumrhi = _mm512_add_epi32(accumrhi, _mm512_unpackhi_epi16(result2lo, result2));
                __m512i result3 = _mm512_mulhi_epu16(refconvol1, fcoeff1);
                __m512i result3lo = _mm512_mullo_epi16(refconvol1, fcoeff1);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result3lo, result3));
                accumrhi = _mm512_add_epi32(accumrhi, _mm512_unpackhi_epi16(result3lo, result3));
                __m512i result4 = _mm512_mulhi_epu16(refconvol2, fcoeff2);
                __m512i result4lo = _mm512_mullo_epi16(refconvol2, fcoeff2);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result4lo, result4));
                accumrhi = _mm512_add_epi32(
                    accumrhi, _mm512_unpackhi_epi16(result4lo, result4));
                __m512i result5 = _mm512_mulhi_epu16(refconvol3, fcoeff3);
                __m512i result5lo = _mm512_mullo_epi16(refconvol3, fcoeff3);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result5lo, result5));
                accumrhi = _mm512_add_epi32(
                    accumrhi, _mm512_unpackhi_epi16(result5lo, result5));
                __m512i result6 = _mm512_mulhi_epu16(refconvol4, fcoeff4);
                __m512i result6lo = _mm512_mullo_epi16(refconvol4, fcoeff4);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result6lo, result6));
                accumrhi = _mm512_add_epi32(
                    accumrhi, _mm512_unpackhi_epi16(result6lo, result6));
                __m512i result7 = _mm512_mulhi_epu16(refconvol5, fcoeff3);
                __m512i result7lo = _mm512_mullo_epi16(refconvol5, fcoeff3);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result7lo, result7));
                accumrhi = _mm512_add_epi32(
                    accumrhi, _mm512_unpackhi_epi16(result7lo, result7));
                __m512i result8 = _mm512_mulhi_epu16(refconvol6, fcoeff2);
                __m512i result8lo = _mm512_mullo_epi16(refconvol6, fcoeff2);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result8lo, result8));
                accumrhi = _mm512_add_epi32(
                    accumrhi, _mm512_unpackhi_epi16(result8lo, result8));
                __m512i result9 = _mm512_mulhi_epu16(refconvol7, fcoeff1);
                __m512i result9lo = _mm512_mullo_epi16(refconvol7, fcoeff1);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result9lo, result9));
                accumrhi = _mm512_add_epi32(
                    accumrhi, _mm512_unpackhi_epi16(result9lo, result9));
                __m512i result10 = _mm512_mulhi_epu16(refconvol8, fcoeff);
                __m512i result10lo = _mm512_mullo_epi16(refconvol8, fcoeff);
                accumrlo = _mm512_add_epi32(
                    accumrlo, _mm512_unpacklo_epi16(result10lo, result10));
                accumrhi = _mm512_add_epi32(
                    accumrhi, _mm512_unpackhi_epi16(result10lo, result10));

                __m512i disconvol = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check));
                __m512i disconvol1 = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check + 1));
                __m512i disconvol2 = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check + 2));
                __m512i disconvol3 = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check + 3));
                __m512i disconvol4 = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check + 4));
                __m512i disconvol5 = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check + 5));
                __m512i disconvol6 = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check + 6));
                __m512i disconvol7 = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check + 7));
                __m512i disconvol8 = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check + 8));
                result2 = _mm512_mulhi_epu16(disconvol, fcoeff);
                result2lo = _mm512_mullo_epi16(disconvol, fcoeff);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result2lo, result2));
                accumdhi = _mm512_add_epi32(
                    accumdhi, _mm512_unpackhi_epi16(result2lo, result2));
                result3 = _mm512_mulhi_epu16(disconvol1, fcoeff1);
                result3lo = _mm512_mullo_epi16(disconvol1, fcoeff1);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result3lo, result3));
                accumdhi = _mm512_add_epi32(
                    accumdhi, _mm512_unpackhi_epi16(result3lo, result3));
                result4 = _mm512_mulhi_epu16(disconvol2, fcoeff2);
                result4lo = _mm512_mullo_epi16(disconvol2, fcoeff2);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result4lo, result4));
                accumdhi = _mm512_add_epi32(
                    accumdhi, _mm512_unpackhi_epi16(result4lo, result4));
                result5 = _mm512_mulhi_epu16(disconvol3, fcoeff3);
                result5lo = _mm512_mullo_epi16(disconvol3, fcoeff3);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result5lo, result5));
                accumdhi = _mm512_add_epi32(
                    accumdhi, _mm512_unpackhi_epi16(result5lo, result5));
                result6 = _mm512_mulhi_epu16(disconvol4, fcoeff4);
                result6lo = _mm512_mullo_epi16(disconvol4, fcoeff4);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result6lo, result6));
                accumdhi = _mm512_add_epi32(
                    accumdhi, _mm512_unpackhi_epi16(result6lo, result6));
                result7 = _mm512_mulhi_epu16(disconvol5, fcoeff3);
                result7lo = _mm512_mullo_epi16(disconvol5, fcoeff3);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result7lo, result7));
                accumdhi = _mm512_add_epi32(
                    accumdhi, _mm512_unpackhi_epi16(result7lo, result7));
                result8 = _mm512_mulhi_epu16(disconvol6, fcoeff2);
                result8lo = _mm512_mullo_epi16(disconvol6, fcoeff2);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result8lo, result8));
                accumdhi = _mm512_add_epi32(accumdhi, _mm512_unpackhi_epi16(result8lo, result8));
                result9 = _mm512_mulhi_epu16(
                    disconvol7, fcoeff1);
                result9lo = _mm512_mullo_epi16(disconvol7, fcoeff1);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result9lo, result9));
                accumdhi = _mm512_add_epi32(
                    accumdhi, _mm512_unpackhi_epi16(result9lo, result9));
                result10 = _mm512_mulhi_epu16(disconvol8, fcoeff);
                result10lo = _mm512_mullo_epi16(disconvol8, fcoeff);
                accumdlo = _mm512_add_epi32(
                    accumdlo, _mm512_unpacklo_epi16(result10lo, result10));
                accumdhi = _mm512_add_epi32(
                    accumdhi, _mm512_unpackhi_epi16(result10lo, result10));
            }

            accumdlo = _mm512_add_epi32(accumdlo, addnum);
            accumdhi = _mm512_add_epi32(accumdhi, addnum);
            accumrlo = _mm512_add_epi32(accumrlo, addnum);
            accumrhi = _mm512_add_epi32(accumrhi, addnum);
            accumdlo = _mm512_srli_epi32(accumdlo, 0x10);
            accumdhi = _mm512_srli_epi32(accumdhi, 0x10);
            accumrlo = _mm512_srli_epi32(accumrlo, 0x10);
            accumrhi = _mm512_srli_epi32(accumrhi, 0x10);

            __m512i result = _mm512_permutex2var_epi16(accumdlo, mask1, accumdhi);
            __m512i resultd = _mm512_permutex2var_epi16(accumrlo, mask1, accumrhi);

            _mm256_storeu_si256((__m256i *)(buf.mu1 + i * stride + j), _mm512_castsi512_si256(resultd));
            _mm256_storeu_si256((__m256i *)(buf.mu2 + i * stride + j), _mm512_castsi512_si256(result));
        }

        for (unsigned j = n << 4; j < w; ++j)
        {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int jj = j - fwidth_half;
            int jj_check = jj;
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj)
            {
                const uint16_t fcoeff = vif_filt_s1[fj];
                accum_ref += fcoeff * buf.tmp.ref_convol[jj_check];
                accum_dis += fcoeff * buf.tmp.dis_convol[jj_check];
            }
            buf.mu1[i * stride + j] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride + j] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }
}

void vif_filter1d_rd_16_avx512(VifBuffer buf, unsigned w, unsigned h, int scale,
                               int bpc)
{
    const unsigned fwidth = vif_filter1d_width[scale + 1];
    const uint16_t *vif_filt = vif_filter1d_table[scale + 1];
    int32_t add_shift_round_VP, shift_VP;
    int fwidth_half = fwidth >> 1;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t stride16 = buf.stride_16 / sizeof(uint16_t);
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
    __m256i mask2 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    __m256i mask1 = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);

    if (scale == 0)
    {
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP = bpc;
    }
    else
    {
        add_shift_round_VP = 32768;
        shift_VP = 16;
    }

    for (unsigned i = 0; i < h; ++i)
    {
        //VERTICAL

        int n = w >> 4;
        int ii = i - fwidth_half;
        for (unsigned j = 0; j < n << 4; j = j + 32)
        {
            int ii_check = ii;
            __m512i accumr_lo, accumr_hi, accumd_lo, accumd_hi, rmul1, rmul2, dmul1, dmul2;
            accumr_lo = accumr_hi = accumd_lo = accumd_hi = rmul1 = rmul2 = dmul1 = dmul2 = _mm512_setzero_si512();
            __m512i mask3 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);   //first half of 512
            __m512i mask4 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4); //second half of 512
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi)
            {

                const uint16_t fcoeff = vif_filt[fi];
                __m512i f1 = _mm512_set1_epi16(vif_filt[fi]);
                __m512i ref1 = _mm512_loadu_si512((__m512i *)(ref + (ii_check * stride) + j));
                __m512i dis1 = _mm512_loadu_si512((__m512i *)(dis + (ii_check * stride) + j));
                __m512i result2 = _mm512_mulhi_epu16(ref1, f1);
                __m512i result2lo = _mm512_mullo_epi16(ref1, f1);
                rmul1 = _mm512_unpacklo_epi16(result2lo, result2);
                rmul2 = _mm512_unpackhi_epi16(result2lo, result2);
                accumr_lo = _mm512_add_epi32(accumr_lo, rmul1);
                accumr_hi = _mm512_add_epi32(accumr_hi, rmul2);

                __m512i d0 = _mm512_mulhi_epu16(dis1, f1);
                __m512i d0lo = _mm512_mullo_epi16(dis1, f1);
                dmul1 = _mm512_unpacklo_epi16(d0lo, d0);
                dmul2 = _mm512_unpackhi_epi16(d0lo, d0);
                accumd_lo = _mm512_add_epi32(accumd_lo, dmul1);
                accumd_hi = _mm512_add_epi32(accumd_hi, dmul2);
            }
            __m512i addnum = _mm512_set1_epi32(add_shift_round_VP);
            accumr_lo = _mm512_add_epi32(accumr_lo, addnum);
            accumr_hi = _mm512_add_epi32(accumr_hi, addnum);
            accumr_lo = _mm512_srli_epi32(accumr_lo, shift_VP);
            accumr_hi = _mm512_srli_epi32(accumr_hi, shift_VP);

            _mm512_storeu_si512((__m512i *)(buf.tmp.ref_convol + j),
                                _mm512_permutex2var_epi64(accumr_lo, mask3, accumr_hi));
            _mm512_storeu_si512((__m512i *)(buf.tmp.ref_convol + j + 16),
                                _mm512_permutex2var_epi64(accumr_lo, mask4, accumr_hi));

            accumd_lo = _mm512_add_epi32(accumd_lo, addnum);
            accumd_hi = _mm512_add_epi32(accumd_hi, addnum);
            accumd_lo = _mm512_srli_epi32(accumd_lo, shift_VP);
            accumd_hi = _mm512_srli_epi32(accumd_hi, shift_VP);
            _mm512_storeu_si512((__m512i *)(buf.tmp.dis_convol + j),
                                _mm512_permutex2var_epi64(accumd_lo, mask3, accumd_hi));
            _mm512_storeu_si512((__m512i *)(buf.tmp.dis_convol + j + 16),
                                _mm512_permutex2var_epi64(accumd_lo, mask4, accumd_hi));
        }

        // //VERTICAL
        for (unsigned j = n << 4; j < w; ++j)
        {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int ii_check = ii;
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi)
            {
                const uint16_t fcoeff = vif_filt[fi];
                accum_ref += fcoeff * ((uint32_t)ref[ii_check * stride + j]);
                accum_dis += fcoeff * ((uint32_t)dis[ii_check * stride + j]);
            }
            buf.tmp.ref_convol[j] = (uint16_t)((accum_ref + add_shift_round_VP) >> shift_VP);
            buf.tmp.dis_convol[j] = (uint16_t)((accum_dis + add_shift_round_VP) >> shift_VP);
        }

        PADDING_SQ_DATA_2(buf, w, fwidth_half);

        //HORIZONTAL
        n = w >> 4;
        for (unsigned j = 0; j < n << 4; j = j + 16)
        {
            int jj = j - fwidth_half;
            int jj_check = jj;
            __m512i accumdl, accumrlo, accumdlo, accumrhi, accumdhi;
            accumrlo = accumdlo = accumrhi = accumdhi = _mm512_setzero_si512();
            const uint16_t *ref = (uint16_t *)buf.tmp.ref_convol;
            const uint16_t *dis = (uint16_t *)buf.dis;
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj)
            {

                __m512i refconvol = _mm512_loadu_si512((__m512i *)(buf.tmp.ref_convol + jj_check));
                __m512i fcoeff = _mm512_set1_epi16(vif_filt[fj]);
                __m512i result2 = _mm512_mulhi_epu16(refconvol, fcoeff);
                __m512i result2lo = _mm512_mullo_epi16(refconvol, fcoeff);
                accumrlo = _mm512_add_epi32(accumrlo, _mm512_unpacklo_epi16(result2lo, result2));
                accumrhi = _mm512_add_epi32(accumrhi, _mm512_unpackhi_epi16(result2lo, result2));
                __m512i disconvol = _mm512_loadu_si512((__m512i *)(buf.tmp.dis_convol + jj_check));
                result2 = _mm512_mulhi_epu16(disconvol, fcoeff);
                result2lo = _mm512_mullo_epi16(disconvol, fcoeff);
                accumdlo = _mm512_add_epi32(accumdlo, _mm512_unpacklo_epi16(result2lo, result2));
                accumdhi = _mm512_add_epi32(accumdhi, _mm512_unpackhi_epi16(result2lo, result2));
            }

            __m512i addnum = _mm512_set1_epi32(32768);
            accumdlo = _mm512_add_epi32(accumdlo, addnum);
            accumdhi = _mm512_add_epi32(accumdhi, addnum);
            accumrlo = _mm512_add_epi32(accumrlo, addnum);
            accumrhi = _mm512_add_epi32(accumrhi, addnum);
            accumdlo = _mm512_srli_epi32(accumdlo, 0x10);
            accumdhi = _mm512_srli_epi32(accumdhi, 0x10);
            accumrlo = _mm512_srli_epi32(accumrlo, 0x10);
            accumrhi = _mm512_srli_epi32(accumrhi, 0x10);

            __m512i mask2 = _mm512_set_epi16(60, 56, 28, 24, 52, 48, 20, 16,
                                             44, 40, 12, 8, 36, 32, 4, 0, 60,
                                             56, 28, 24, 52, 48, 20, 16, 44,
                                             40, 12, 8, 36, 32, 4, 0);
            _mm256_storeu_si256((__m256i *)(buf.mu1 + (stride16 * i) + j),
                                _mm512_castsi512_si256(_mm512_permutex2var_epi16(accumrlo, mask2, accumrhi)));
            _mm256_storeu_si256((__m256i *)(buf.mu2 + (stride16 * i) + j),
                                _mm512_castsi512_si256(_mm512_permutex2var_epi16(accumdlo, mask2, accumdhi)));
        }

        for (unsigned j = n << 4; j < w; ++j)
        {
            uint32_t accum_ref = 0;
            uint32_t accum_dis = 0;
            int jj = j - fwidth_half;
            int jj_check = jj;
            for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj)
            {
                const uint16_t fcoeff = vif_filt[fj];
                accum_ref += fcoeff * ((uint32_t)buf.tmp.ref_convol[jj_check]);
                accum_dis += fcoeff * ((uint32_t)buf.tmp.dis_convol[jj_check]);
            }
            buf.mu1[i * stride16 + j] = (uint16_t)((accum_ref + 32768) >> 16);
            buf.mu2[i * stride16 + j] = (uint16_t)((accum_dis + 32768) >> 16);
        }
    }
}
