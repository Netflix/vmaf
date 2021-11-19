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
#include <string.h>
#include <assert.h>
#include "stdio.h"
#include "feature/common/macros.h"

#ifdef __GNUC__
#pragma GCC target "avx512f"
#pragma GCC target "avx512bw"
#endif

#include "feature/integer_vif.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#ifdef _MSC_VER
#include <intrin.h>

static inline int __builtin_clz(unsigned x) {
    return (int)__lzcnt(x);
}

static inline int __builtin_clzll(unsigned long long x) {
    return (int)__lzcnt64(x);
}

#endif

static FORCE_INLINE inline uint16_t
get_best16_from32(uint32_t temp, int* x)
{
    int k = __builtin_clz(temp);
    k = 16 - k;
    temp = temp >> k;
    *x = -k;
    return temp;
}

static FORCE_INLINE inline uint16_t get_best16_from64(uint64_t temp, int* x)
{
    assert(temp >= 0x20000);
    int k = __builtin_clzll(temp);
    k = 48 - k;
    temp = temp >> k;
    *x = -k;
    return (uint16_t)temp;
}

static inline void
pad_top_and_bottom(VifBuffer buf, unsigned h, int fwidth)
{
    const unsigned fwidth_half = fwidth / 2;
    unsigned char* ref = buf.ref;
    unsigned char* dis = buf.dis;
    for (unsigned i = 1; i <= fwidth_half; ++i) {
        size_t offset = buf.stride * i;
        memcpy(ref - offset, ref + offset, buf.stride);
        memcpy(dis - offset, dis + offset, buf.stride);
        memcpy(ref + buf.stride * (h - 1) + buf.stride * i,
            ref + buf.stride * (h - 1) - buf.stride * i,
            buf.stride);
        memcpy(dis + buf.stride * (h - 1) + buf.stride * i,
            dis + buf.stride * (h - 1) - buf.stride * i,
            buf.stride);
    }
}

static inline void
decimate_and_pad(VifBuffer buf, unsigned w, unsigned h, int scale)
{
    uint16_t* ref = buf.ref;
    uint16_t* dis = buf.dis;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    const ptrdiff_t mu_stride = buf.stride_16 / sizeof(uint16_t);

    for (unsigned i = 0; i < h / 2; ++i) {
        for (unsigned j = 0; j < w / 2; ++j) {
            ref[i * stride + j] = buf.mu1[(i * 2) * mu_stride + (j * 2)];
            dis[i * stride + j] = buf.mu2[(i * 2) * mu_stride + (j * 2)];
        }
    }
    pad_top_and_bottom(buf, h / 2, vif_filter1d_width[scale]);
}

void vif_statistic_8_avx512(struct VifState* s, float* num, float* den, unsigned w, unsigned h) {
    const unsigned fwidth = vif_filter1d_width[0];
    const uint16_t* vif_filt_s0 = vif_filter1d_table[0];
    VifBuffer buf = s->buf;
    const uint8_t* ref = (uint8_t*)buf.ref;
    const uint8_t* dis = (uint8_t*)buf.dis;
    const unsigned fwidth_half = fwidth >> 1;
    const uint16_t* log2_table = s->log2_table;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;
    __m512i mask2 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    __m512i mask3 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    __m512i f1, f2, f3, f4, f5, f6, f7, f8, f9, fc0, fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8;
    uint32_t* xx_filt = buf.ref_sq;
    uint32_t* yy_filt = buf.dis_sq;
    uint32_t* xy_filt = buf.ref_dis;

#if defined __GNUC__
#define ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined (_MSC_VER)  && (!defined UNDER_CE)
#define ALIGNED(x) __declspec (align(x))
#else 
#define ALIGNED(x)
#endif

    // variables used for 16 sample block vif computation
    ALIGNED(32) uint32_t xx[16];
    ALIGNED(32) uint32_t yy[16];
    ALIGNED(32) uint32_t xy[16];

    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    int64_t num_val, den_val;
    int64_t accum_x = 0, accum_x2 = 0;
    int64_t accum_num_log = 0.0;
    int64_t accum_den_log = 0.0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;

    fc0 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)vif_filt_s0));
    fc1 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 2)));
    fc2 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 4)));
    fc3 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 6)));
    fc4 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 8)));
    fc5 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 10)));
    fc6 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 12)));
    fc7 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 14)));
    fc8 = _mm512_broadcastd_epi32(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 16)));

    f1 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)vif_filt_s0));
    f2 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 1)));
    f3 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 2)));
    f4 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 3)));
    f5 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 4)));
    f6 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 5)));
    f7 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 6)));
    f8 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 7)));
    f9 = _mm512_broadcastw_epi16(_mm_loadu_si128((__m128i*)(vif_filt_s0 + 8)));

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
            __m512i g0, g1, g2, g3, g4, g5, g6, g7, g20, g21, g22,
                g23, g24, g25, g26, g27;
            __m512i s0, s1, s2, s3, s4, s5, s6, s7, s20, s21, s22,
                s23, s24, s25, s26, s27, sg0, sg1, sg2, sg3, sg4, sg5, sg6, sg7;

            g0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + (buf.stride * ii_check) + j)));
            g1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 1) + j)));
            g2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 2) + j)));
            g3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 3) + j)));
            g4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 4) + j)));
            g5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 5) + j)));
            g6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 6) + j)));
            g7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 7) + j)));

            s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + (buf.stride * ii_check) + j)));
            s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 1) + j)));
            s2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 2) + j)));
            s3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 3) + j)));
            s4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 4) + j)));
            s5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 5) + j)));
            s6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 6) + j)));
            s7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 7) + j)));

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
                (__m256i*)(ref + buf.stride * (ii_check + 8) + j)));
            g1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 9) + j)));
            g2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 10) + j)));
            g3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 11) + j)));
            g4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 12) + j)));
            g5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 13) + j)));
            g6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 14) + j)));
            g7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 15) + j)));

            s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 8) + j)));
            s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 9) + j)));
            s2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 10) + j)));
            s3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 11) + j)));
            s4 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 12) + j)));
            s5 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 13) + j)));
            s6 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 14) + j)));
            s7 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 15) + j)));

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
                (__m256i*)(ref + buf.stride * (ii_check + 16) + j)));
            g1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(ref + buf.stride * (ii_check + 17) + j)));

            s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 16) + j)));
            s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                (__m256i*)(dis + buf.stride * (ii_check + 17) + j)));

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
            _mm512_storeu_si512((__m512i*)(buf.tmp.ref + j), accumref_lo);
            _mm512_storeu_si512((__m512i*)(buf.tmp.ref + j + 16), accumref_hi);
            _mm512_storeu_si512((__m512i*)(buf.tmp.dis + j), accumdis_lo);
            _mm512_storeu_si512((__m512i*)(buf.tmp.dis + j + 16), accumdis_hi);
            _mm512_storeu_si512((__m512i*)(buf.tmp.ref_dis + j), accumrefdis_lo);
            _mm512_storeu_si512((__m512i*)(buf.tmp.ref_dis + j + 16), accumrefdis_hi);
            _mm512_storeu_si512((__m512i*)(buf.tmp.mu1 + j), accumu1_lo);
            _mm512_storeu_si512((__m512i*)(buf.tmp.mu1 + j + 16), accumu1_hi);
            _mm512_storeu_si512((__m512i*)(buf.tmp.mu2 + j), accumu2_lo);
            _mm512_storeu_si512((__m512i*)(buf.tmp.mu2 + j + 16), accumu2_hi);
        }

        PADDING_SQ_DATA(buf, w, fwidth_half);
        //horizontal

        //HORIZONTAL
        const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
        for (unsigned jj = 0; jj < w; jj += 16) {
            __m512i mu1;
            __m512i mu1sq;
            __m512i mu2sq;
            __m512i mu1mu2;
            __m512i mask5 = _mm512_set_epi32(30, 28, 14, 12, 26, 24, 10, 8, 22, 20, 6, 4, 18, 16, 2, 0);
            {
                __m512i fq = _mm512_set1_epi32(vif_filt_s0[fwidth / 2]);
                __m512i acc0 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + jj + 0)), fq);
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi32(vif_filt_s0[fj]);
                    acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + jj - fwidth / 2 + fj + 0)), fq));
                    acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + jj + fwidth / 2 - fj + 0)), fq));
                }
                mu1 = acc0;
                __m512i acc0_lo_512 = _mm512_unpacklo_epi32(acc0, _mm512_setzero_si512());
                __m512i acc0_hi_512 = _mm512_unpackhi_epi32(acc0, _mm512_setzero_si512());
                acc0_lo_512 = _mm512_mul_epu32(acc0_lo_512, acc0_lo_512);
                acc0_hi_512 = _mm512_mul_epu32(acc0_hi_512, acc0_hi_512);
                acc0_lo_512 = _mm512_srli_epi64(_mm512_add_epi64(acc0_lo_512, _mm512_set1_epi64(0x80000000)), 32);
                acc0_hi_512 = _mm512_srli_epi64(_mm512_add_epi64(acc0_hi_512, _mm512_set1_epi64(0x80000000)), 32);
                mu1sq = _mm512_permutex2var_epi32(acc0_lo_512, mask5, acc0_hi_512);
            }

            {
                __m512i fq = _mm512_set1_epi32(vif_filt_s0[fwidth / 2]);
                __m512i acc0 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + jj + 0)), fq);
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi32(vif_filt_s0[fj]);
                    acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + jj - fwidth / 2 + fj + 0)), fq));
                    acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + jj + fwidth / 2 - fj + 0)), fq));
                }
                __m512i acc0lo_512 = _mm512_unpacklo_epi32(acc0, _mm512_setzero_si512());
                __m512i acc0hi_512 = _mm512_unpackhi_epi32(acc0, _mm512_setzero_si512());
                __m512i mu1lo_512 = _mm512_unpacklo_epi32(mu1, _mm512_setzero_si512());
                __m512i mu1hi_512 = _mm512_unpackhi_epi32(mu1, _mm512_setzero_si512());

                mu1lo_512 = _mm512_mul_epu32(mu1lo_512, acc0lo_512);
                mu1hi_512 = _mm512_mul_epu32(mu1hi_512, acc0hi_512);
                mu1lo_512 = _mm512_srli_epi64(_mm512_add_epi64(mu1lo_512, _mm512_set1_epi64(0x80000000)), 32);
                mu1hi_512 = _mm512_srli_epi64(_mm512_add_epi64(mu1hi_512, _mm512_set1_epi64(0x80000000)), 32);

                mu1mu2 = _mm512_permutex2var_epi32(mu1lo_512, mask5, mu1hi_512);
                acc0lo_512 = _mm512_mul_epu32(acc0lo_512, acc0lo_512);
                acc0hi_512 = _mm512_mul_epu32(acc0hi_512, acc0hi_512);
                acc0lo_512 = _mm512_srli_epi64(_mm512_add_epi64(acc0lo_512, _mm512_set1_epi64(0x80000000)), 32);
                acc0hi_512 = _mm512_srli_epi64(_mm512_add_epi64(acc0hi_512, _mm512_set1_epi64(0x80000000)), 32);
                mu2sq = _mm512_permutex2var_epi32(acc0lo_512, mask5, acc0hi_512);
            }

            // filter horizontally ref
            {
                __m512i rounder_512 = _mm512_set1_epi64(0x8000);
                __m512i fq_512 = _mm512_set1_epi64(vif_filt_s0[fwidth / 2]);
                __m512i s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + jj + 0))); // 4
                __m512i s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + jj + 8))); // 4
                __m512i acc0_512 = _mm512_add_epi64(rounder_512, _mm512_mul_epu32(s0_512, fq_512));
                __m512i acc2_512 = _mm512_add_epi64(rounder_512, _mm512_mul_epu32(s2_512, fq_512));
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi64(vif_filt_s0[fj]);
                    s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 0))); // 4
                    s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + jj - fwidth / 2 + fj + 8))); // 4
                    acc0_512 = _mm512_add_epi64(acc0_512, _mm512_mul_epu32(s0_512, fq));
                    acc2_512 = _mm512_add_epi64(acc2_512, _mm512_mul_epu32(s2_512, fq));
                    s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 0))); // 4
                    s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + jj + fwidth / 2 - fj + 8))); // 4
                    acc0_512 = _mm512_add_epi64(acc0_512, _mm512_mul_epu32(s0_512, fq));
                    acc2_512 = _mm512_add_epi64(acc2_512, _mm512_mul_epu32(s2_512, fq));
                }
                acc0_512 = _mm512_srli_epi64(acc0_512, 16);
                acc2_512 = _mm512_srli_epi64(acc2_512, 16);
                __m512i mask = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
                __m512i refsq = _mm512_permutex2var_epi32(acc0_512, mask, acc2_512);
                _mm512_storeu_si512((__m512i*) & xx[0], _mm512_sub_epi32(refsq, mu1sq));
            }

            // filter horizontally dis
            {
                __m512i rounder_512 = _mm512_set1_epi64(0x8000);
                __m512i fq_512 = _mm512_set1_epi64(vif_filt_s0[fwidth / 2]);
                __m512i s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis+ jj + 0))); // 4
                __m512i s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + jj + 8))); // 4
                __m512i acc0_512 = _mm512_add_epi64(rounder_512, _mm512_mul_epu32(s0_512, fq_512));
                __m512i acc2_512 = _mm512_add_epi64(rounder_512, _mm512_mul_epu32(s2_512, fq_512));
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi64(vif_filt_s0[fj]);
                    s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 0))); // 4
                    s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + jj - fwidth / 2 + fj + 8))); // 4
                    acc0_512 = _mm512_add_epi64(acc0_512, _mm512_mul_epu32(s0_512, fq));
                    acc2_512 = _mm512_add_epi64(acc2_512, _mm512_mul_epu32(s2_512, fq));
                    s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 0))); // 4
                    s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + jj + fwidth / 2 - fj + 8))); // 4
                    acc0_512 = _mm512_add_epi64(acc0_512, _mm512_mul_epu32(s0_512, fq));
                    acc2_512 = _mm512_add_epi64(acc2_512, _mm512_mul_epu32(s2_512, fq));
                }
                acc0_512 = _mm512_srli_epi64(acc0_512, 16);
                acc2_512 = _mm512_srli_epi64(acc2_512, 16);
                __m512i mask = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
                __m512i dissq = _mm512_permutex2var_epi32(acc0_512, mask, acc2_512);
                _mm512_storeu_si512((__m512i*)&yy[0], _mm512_max_epi32(_mm512_sub_epi32(dissq, mu2sq), _mm512_setzero_si512()));
            }

            // filter horizontally ref_dis, producing 16 samples
            {
                __m512i rounder_512 = _mm512_set1_epi64(0x8000);
                __m512i fq_512 = _mm512_set1_epi64(vif_filt_s0[fwidth / 2]);
                __m512i s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + jj + 0))); // 4
                __m512i s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + jj + 8))); // 4
                __m512i acc0_512 = _mm512_add_epi64(rounder_512, _mm512_mul_epu32(s0_512, fq_512));
                __m512i acc2_512 = _mm512_add_epi64(rounder_512, _mm512_mul_epu32(s2_512, fq_512));
                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi64(vif_filt_s0[fj]);
                    s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 0))); // 4
                    s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + jj - fwidth / 2 + fj + 8))); // 4
                    acc0_512 = _mm512_add_epi64(acc0_512, _mm512_mul_epu32(s0_512, fq));
                    acc2_512 = _mm512_add_epi64(acc2_512, _mm512_mul_epu32(s2_512, fq));
                    s0_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 0))); // 4
                    s2_512 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + jj + fwidth / 2 - fj + 8))); // 4
                    acc0_512 = _mm512_add_epi64(acc0_512, _mm512_mul_epu32(s0_512, fq));
                    acc2_512 = _mm512_add_epi64(acc2_512, _mm512_mul_epu32(s2_512, fq));
                }
                acc0_512 = _mm512_srli_epi64(acc0_512, 16);
                acc2_512 = _mm512_srli_epi64(acc2_512, 16);
                __m512i mask = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
                __m512i refdis = _mm512_permutex2var_epi32(acc0_512, mask, acc2_512);
                _mm512_storeu_si512((__m512i*)&xy[0], _mm512_sub_epi32(refdis, mu1mu2));


            }

            for (unsigned int j = 0; j < 16; j++) {
                int32_t sigma1_sq = xx[j];
                int32_t sigma2_sq = yy[j];
                int32_t sigma12 = xy[j];

                if (sigma1_sq >= sigma_nsq) {
                    uint32_t log_den_stage1 = (uint32_t)(sigma_nsq + sigma1_sq);
                    int x;
                    uint16_t log_den1 = get_best16_from32(log_den_stage1, &x);

                    /**
                    * log values are taken from the look-up table generated by
                    * log_generate() function which is called in integer_combo_threadfunc
                    * den_val in float is log2(1 + sigma1_sq/2)
                    * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                    * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                    * x because best 16 bits are taken
                    */
                    accum_x += x + 17;
                    den_val = log2_table[log_den1];

                    // this check can go away, things will work anyway
                    if (sigma12 > 0) {
                        // num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
                        /**
                        * In floating-point numerator = log2((1.0f + (g * g * sigma1_sq)/(sv_sq + sigma_nsq))
                        *
                        * In Fixed-point the above is converted to
                        * numerator = log2((sv_sq + sigma_nsq)+(g * g * sigma1_sq))- log2(sv_sq + sigma_nsq)
                        */

                        const double eps = 65536 * 1.0e-10;
                        double g = sigma12 / (sigma1_sq + eps); // this epsilon can go away
                        int32_t sv_sq = sigma2_sq - g * sigma12;

                        if (sigma2_sq == 0) { // this was... sigma2_sq < eps
                            g = 0.0;
                        }

                        sv_sq = (uint32_t)(MAX(sv_sq, 0));

                        g = MIN(g, vif_enhn_gain_limit);

                        int x1, x2;
                        uint32_t numer1 = (sv_sq + sigma_nsq);
                        int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
                        uint16_t numlog = get_best16_from64((uint64_t)numer1_tmp, &x1);
                        uint16_t denlog = get_best16_from64((uint64_t)numer1, &x2);
                        accum_x2 += (x2 - x1);
                        num_val = log2_table[numlog] - log2_table[denlog];
                        accum_num_log += num_val;
                        accum_den_log += den_val;

                    }
                    else {
                        //accum_num_log += 0;
                        accum_den_log += den_val;
                    }
                }
                else {
                    accum_num_non_log += sigma2_sq;
                    accum_den_non_log++;
                }
            }
        }
    }
    //changed calculation to increase performance
    num[0] = accum_num_log / 2048.0 + accum_x2 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 - accum_x + accum_den_non_log;
}

void vif_statistic_16_avx512(struct VifState* s, float* num, float* den, unsigned w, unsigned h, int bpc, int scale) {
    const unsigned fwidth = vif_filter1d_width[scale];
    const uint16_t* vif_filt = vif_filter1d_table[scale];
    VifBuffer buf = s->buf;
    const ptrdiff_t dst_stride = buf.stride_32 / sizeof(uint32_t);
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    int fwidth_half = fwidth >> 1;

    int32_t add_shift_round_HP, shift_HP;
    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;
    const uint16_t* log2_table = s->log2_table;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;
    __m512i mask2 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    uint32_t* xx_filt = buf.ref_sq;
    uint32_t* yy_filt = buf.dis_sq;
    uint32_t* xy_filt = buf.ref_dis;

    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    int64_t num_val, den_val;
    int64_t accum_x = 0, accum_x2 = 0;
    int64_t accum_num_log = 0.0;
    int64_t accum_den_log = 0.0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;

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
    uint16_t* ref = buf.ref;
    uint16_t* dis = buf.dis;

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
                        (__m512i*)(ref + (ii_check * stride) + j));
                    __m512i dis1 = _mm512_loadu_si512(
                        (__m512i*)(dis + (ii_check * stride) + j));
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
                _mm512_storeu_si512((__m512i*)(buf.tmp.mu1 + j), accumr_lo);
                _mm512_storeu_si512((__m512i*)(buf.tmp.mu1 + j + 16), accumr_hi);

                accumd_lo = _mm512_add_epi32(accumd_lo, addnum);
                accumd_hi = _mm512_add_epi32(accumd_hi, addnum);
                accumd_lo = _mm512_srli_epi32(accumd_lo, shift_VP);
                accumd_hi = _mm512_srli_epi32(accumd_hi, shift_VP);
                _mm512_storeu_si512((__m512i*)(buf.tmp.mu2 + j), accumd_lo);
                _mm512_storeu_si512((__m512i*)(buf.tmp.mu2 + j + 16), accumd_hi);

                accumref1 = _mm512_add_epi64(accumref1, addnum64);
                accumref2 = _mm512_add_epi64(accumref2, addnum64);
                accumref3 = _mm512_add_epi64(accumref3, addnum64);
                accumref4 = _mm512_add_epi64(accumref4, addnum64);
                accumref1 = _mm512_srli_epi64(accumref1, shift_VP_sq);
                accumref2 = _mm512_srli_epi64(accumref2, shift_VP_sq);
                accumref3 = _mm512_srli_epi64(accumref3, shift_VP_sq);
                accumref4 = _mm512_srli_epi64(accumref4, shift_VP_sq);

                _mm512_storeu_si512((__m512i*)(buf.tmp.ref + j),
                    _mm512_permutex2var_epi32(accumref1, mask2, accumref3));
                _mm512_storeu_si512((__m512i*)(buf.tmp.ref + 16 + j),
                    _mm512_permutex2var_epi32(accumref2, mask2, accumref4));

                accumrefdis1 = _mm512_add_epi64(accumrefdis1, addnum64);
                accumrefdis2 = _mm512_add_epi64(accumrefdis2, addnum64);
                accumrefdis3 = _mm512_add_epi64(accumrefdis3, addnum64);
                accumrefdis4 = _mm512_add_epi64(accumrefdis4, addnum64);
                accumrefdis1 = _mm512_srli_epi64(accumrefdis1, shift_VP_sq);
                accumrefdis2 = _mm512_srli_epi64(accumrefdis2, shift_VP_sq);
                accumrefdis3 = _mm512_srli_epi64(accumrefdis3, shift_VP_sq);
                accumrefdis4 = _mm512_srli_epi64(accumrefdis4, shift_VP_sq);

                _mm512_storeu_si512((__m512i*)(buf.tmp.ref_dis + j),
                    _mm512_permutex2var_epi32(accumrefdis1, mask2, accumrefdis3));
                _mm512_storeu_si512((__m512i*)(buf.tmp.ref_dis + 16 + j),
                    _mm512_permutex2var_epi32(accumrefdis2, mask2, accumrefdis4));

                accumdis1 = _mm512_add_epi64(accumdis1, addnum64);
                accumdis2 = _mm512_add_epi64(accumdis2, addnum64);
                accumdis3 = _mm512_add_epi64(accumdis3, addnum64);
                accumdis4 = _mm512_add_epi64(accumdis4, addnum64);
                accumdis1 = _mm512_srli_epi64(accumdis1, shift_VP_sq);
                accumdis2 = _mm512_srli_epi64(accumdis2, shift_VP_sq);
                accumdis3 = _mm512_srli_epi64(accumdis3, shift_VP_sq);
                accumdis4 = _mm512_srli_epi64(accumdis4, shift_VP_sq);

                _mm512_storeu_si512((__m512i*)(buf.tmp.dis + j),
                    _mm512_permutex2var_epi32(accumdis1, mask2, accumdis3));
                _mm512_storeu_si512((__m512i*)(buf.tmp.dis + 16 + j),
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
                    uint16_t* ref = buf.ref;
                    uint16_t* dis = buf.dis;
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
                __m512i accumrlo, accumdlo, accumrhi, accumdhi;
                accumrlo = accumdlo = accumrhi = accumdhi = _mm512_setzero_si512();
                __m512i mask2 = _mm512_set_epi32(30, 28, 14, 12, 26, 24, 10, 8, 22, 20, 6, 4, 18, 16, 2, 0);
#pragma unroll(4)
                for (unsigned fj = 0; fj < fwidth; ++fj, jj_check = jj + fj)
                {

                    __m512i refconvol = _mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + jj_check));
                    __m512i fcoeff = _mm512_set1_epi16(vif_filt[fj]);
                    __m512i result2 = _mm512_mulhi_epu16(refconvol, fcoeff);
                    __m512i result2lo = _mm512_mullo_epi16(refconvol, fcoeff);
                    accumrlo = _mm512_add_epi32(accumrlo, _mm512_unpacklo_epi16(result2lo, result2));
                    accumrhi = _mm512_add_epi32(accumrhi, _mm512_unpackhi_epi16(result2lo, result2));

                    __m512i disconvol = _mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + jj_check));
                    result2 = _mm512_mulhi_epu16(disconvol, fcoeff);
                    result2lo = _mm512_mullo_epi16(disconvol, fcoeff);
                    accumdlo = _mm512_add_epi32(accumdlo,
                        _mm512_unpacklo_epi16(result2lo, result2));
                    accumdhi = _mm512_add_epi32(accumdhi,
                        _mm512_unpackhi_epi16(result2lo, result2));
                }

                _mm512_storeu_si512((__m512i*)(buf.mu1_32 + j),
                    _mm512_permutex2var_epi32(accumrlo, mask2, accumrhi));
                _mm512_storeu_si512((__m512i*)(buf.mu2_32 + j),
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
                        _mm256_loadu_si256((__m256i*)(buf.tmp.ref + jj_check)));
                    reflo = _mm512_add_epi64(reflo, _mm512_mul_epu32(s0, f1));
                    __m512i s3 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.ref + jj_check + 8)));
                    refhi = _mm512_add_epi64(refhi, _mm512_mul_epu32(s3, f1));

                    __m512i g0 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.dis + jj_check)));
                    dislo = _mm512_add_epi64(dislo, _mm512_mul_epu32(g0, f1));
                    __m512i g3 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.dis + jj_check + 8)));
                    dishi = _mm512_add_epi64(dishi, _mm512_mul_epu32(g3, f1));

                    __m512i sg0 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + jj_check)));
                    refdislo = _mm512_add_epi64(refdislo, _mm512_mul_epu32(sg0, f1));
                    __m512i sg3 = _mm512_cvtepu32_epi64(
                        _mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + jj_check + 8)));
                    refdishi = _mm512_add_epi64(refdishi, _mm512_mul_epu32(sg3, f1));
                }
                reflo = _mm512_add_epi64(reflo, addnum);
                reflo = _mm512_srli_epi64(reflo, 0x10);
                refhi = _mm512_add_epi64(refhi, addnum);
                refhi = _mm512_srli_epi64(refhi, 0x10);

                _mm512_storeu_si512((__m512i*)(buf.ref_sq + j),
                    _mm512_permutex2var_epi32(reflo, mask2, refhi));

                dislo = _mm512_add_epi64(dislo, addnum);
                dislo = _mm512_srli_epi64(dislo, 0x10);
                dishi = _mm512_add_epi64(dishi, addnum);
                dishi = _mm512_srli_epi64(dishi, 0x10);
                _mm512_storeu_si512((__m512i*)(buf.dis_sq + j),
                    _mm512_permutex2var_epi32(dislo, mask2, dishi));

                refdislo = _mm512_add_epi64(refdislo, addnum);
                refdislo = _mm512_srli_epi64(refdislo, 0x10);
                refdishi = _mm512_add_epi64(refdishi, addnum);
                refdishi = _mm512_srli_epi64(refdishi, 0x10);
                _mm512_storeu_si512((__m512i*)(buf.ref_dis + j),
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
                buf.mu1_32[j] = accum_mu1;
                buf.mu2_32[j] = accum_mu2;
                buf.ref_sq[j] = (uint32_t)((accum_ref + add_shift_round_HP) >> shift_HP);
                buf.dis_sq[j] = (uint32_t)((accum_dis + add_shift_round_HP) >> shift_HP);
                buf.ref_dis[j] = (uint32_t)((accum_ref_dis + add_shift_round_HP) >> shift_HP);
            }

            for (unsigned j = 0; j < w; ++j) {
                uint32_t mu1_val = buf.mu1_32[j];
                uint32_t mu2_val = buf.mu2_32[j];
                uint32_t mu1_sq_val = (uint32_t)((((uint64_t)mu1_val * mu1_val)
                    + 2147483648) >> 32);
                uint32_t mu2_sq_val = (uint32_t)((((uint64_t)mu2_val * mu2_val)
                    + 2147483648) >> 32);
                uint32_t mu1_mu2_val = (uint32_t)((((uint64_t)mu1_val * mu2_val)
                    + 2147483648) >> 32);

                uint32_t xx_filt_val = xx_filt[j];
                uint32_t yy_filt_val = yy_filt[j];
                uint32_t xy_filt_val = xy_filt[j];

                int32_t sigma1_sq = (int32_t)(xx_filt_val - mu1_sq_val);
                int32_t sigma2_sq = (int32_t)(yy_filt_val - mu2_sq_val);
                int32_t sigma12 = (int32_t)(xy_filt_val - mu1_mu2_val);

                sigma2_sq = MAX(sigma2_sq, 0);
                if (sigma1_sq >= sigma_nsq) {
                    uint32_t log_den_stage1 = (uint32_t)(sigma_nsq + sigma1_sq);
                    int x;
                    uint16_t log_den1 = get_best16_from32(log_den_stage1, &x);

                    /**
                    * log values are taken from the look-up table generated by
                    * log_generate() function which is called in integer_combo_threadfunc
                    * den_val in float is log2(1 + sigma1_sq/2)
                    * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                    * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                    * x because best 16 bits are taken
                    */
                    accum_x += x + 17;
                    den_val = log2_table[log_den1];

                    // this check can go away, things will work anyway
                    if (sigma12 > 0) {
                        // num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
                        /**
                        * In floating-point numerator = log2((1.0f + (g * g * sigma1_sq)/(sv_sq + sigma_nsq))
                        *
                        * In Fixed-point the above is converted to
                        * numerator = log2((sv_sq + sigma_nsq)+(g * g * sigma1_sq))- log2(sv_sq + sigma_nsq)
                        */

                        const double eps = 65536 * 1.0e-10;
                        double g = sigma12 / (sigma1_sq + eps); // this epsilon can go away
                        int32_t sv_sq = sigma2_sq - g * sigma12;

                        if (sigma2_sq == 0) { // this was... sigma2_sq < eps
                            g = 0.0;
                        }

                        sv_sq = (uint32_t)(MAX(sv_sq, 0));

                        g = MIN(g, vif_enhn_gain_limit);

                        int x1, x2;
                        uint32_t numer1 = (sv_sq + sigma_nsq);
                        int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
                        uint16_t numlog = get_best16_from64((uint64_t)numer1_tmp, &x1);
                        uint16_t denlog = get_best16_from64((uint64_t)numer1, &x2);
                        accum_x2 += (x2 - x1);
                        num_val = log2_table[numlog] - log2_table[denlog];
                        accum_num_log += num_val;
                        accum_den_log += den_val;

                    }
                    else {
                        //accum_num_log += 0;
                        accum_den_log += den_val;
                    }
                }
                else {
                    accum_num_non_log += sigma2_sq;
                    accum_den_non_log++;
                }
            }
       }
    }



    /**
        * In floating-point there are two types of numerator scores and denominator scores
        * 1. num = 1 - sigma1_sq * constant den =1  when sigma1_sq<2  here constant=4/(255*255)
        * 2. num = log2(((sigma2_sq+2)*sigma1_sq)/((sigma2_sq+2)*sigma1_sq-sigma12*sigma12) den=log2(1+(sigma1_sq/2)) else
        *
        * In fixed-point separate accumulator is used for non-log score accumulations and log-based score accumulation
        * For non-log accumulator of numerator, only sigma1_sq * constant in fixed-point is accumulated
        * log based values are separately accumulated.
        * While adding both accumulator values the non-log accumulator is converted such that it is equivalent to 1 - sigma1_sq * constant(1's are accumulated with non-log denominator accumulator)
    */
    //log has to be divided by 2048 as log_value = log2(i*2048)  i=16384 to 65535
    //num[0] = accum_num_log / 2048.0 + (accum_den_non_log - (accum_num_non_log / 65536.0) / (255.0*255.0));
    //den[0] = accum_den_log / 2048.0 + accum_den_non_log;

    //changed calculation to increase performance
    num[0] = accum_num_log / 2048.0 + accum_x2 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 - accum_x + accum_den_non_log;
}

void vif_subsample_rd_8_avx512(VifBuffer buf, unsigned w, unsigned h)
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
    __m512i f0, f1, f2, f3, f4;

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
                __m512i g0, g1, g2, g3, g4, g5, g6, g7, g8, g9;
                __m512i s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;

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
            __m512i accumrlo, accumdlo, accumrhi, accumdhi, padzero;
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

        for (unsigned j = n << 4; j < w; j = j++)
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
    decimate_and_pad(buf, w, h, 0);
}

void vif_subsample_rd_16_avx512(VifBuffer buf, unsigned w, unsigned h, int scale,
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
            __m512i accumrlo, accumdlo, accumrhi, accumdhi;
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
    decimate_and_pad(buf, w, h, scale);
}


// it is avx512 because there's a couple of instructions (lzcnt, 64 bit int <-> double conversion) which require avx512
// what a pity... 
static void vif_statistic_avx512(VifBuffer buf, float* num, float* den,
    unsigned w, unsigned h, uint16_t* log2_table,
    double vif_enhn_gain_limit)
{
    uint32_t* xx_filt = buf.ref_sq;
    uint32_t* yy_filt = buf.dis_sq;
    uint32_t* xy_filt = buf.ref_dis;

    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    int64_t num_val, den_val;
    int64_t accum_x[4] = { 0 };
    int64_t accum_x2[4] = { 0 };
    int64_t accum_num_log[4] = { 0 };
    int64_t accum_den_log[4] = { 0 };
    int64_t accum_num_non_log[4] = { 0 };
    int64_t accum_den_non_log[4] = { 0 };

    __m256i maccum_x = _mm256_setzero_si256();
    __m256i maccum_x2 = _mm256_setzero_si256();
    __m256i maccum_num_log = _mm256_setzero_si256();
    __m256i maccum_den_log = _mm256_setzero_si256();
    __m256i maccum_num_non_log = _mm256_setzero_si256();
    __m256i maccum_den_non_log = _mm256_setzero_si256();

    const ptrdiff_t stride = buf.stride_32 / sizeof(uint32_t);
    const double eps = 65536 * 1.0e-10;

    /**
        * In floating-point there are two types of numerator scores and denominator scores
        * 1. num = 1 - sigma1_sq * constant den =1  when sigma1_sq<2  here constant=4/(255*255)
        * 2. num = log2(((sigma2_sq+2)*sigma1_sq)/((sigma2_sq+2)*sigma1_sq-sigma12*sigma12) den=log2(1+(sigma1_sq/2)) else
        *
        * In fixed-point separate accumulator is used for non-log score accumulations and log-based score accumulation
        * For non-log accumulator of numerator, only sigma1_sq * constant in fixed-point is accumulated
        * log based values are separately accumulated.
        * While adding both accumulator values the non-log accumulator is converted such that it is equivalent to 1 - sigma1_sq * constant(1's are accumulated with non-log denominator accumulator)
    */
    for (unsigned i = 0; i < h; ++i) {
        for (unsigned jj = 0; jj < w; jj += 4) {

            __m256i mu1 = _mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.mu1_32 + i * stride + jj)));
            __m256i mu2 = _mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.mu2_32 + i * stride + jj)));
            __m256i mu1mu2 = _mm256_srli_epi64(_mm256_add_epi64(_mm256_mullo_epi64(mu1, mu2), _mm256_set1_epi64x(0x80000000)), 32);
            __m256i mu1sq = _mm256_srli_epi64(_mm256_add_epi64(_mm256_mullo_epi64(mu1, mu1), _mm256_set1_epi64x(0x80000000)), 32);
            __m256i mu2sq = _mm256_srli_epi64(_mm256_add_epi64(_mm256_mullo_epi64(mu2, mu2), _mm256_set1_epi64x(0x80000000)), 32);

            __m256i xx = _mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.ref_sq + i * stride + jj)));
            __m256i yy = _mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.dis_sq + i * stride + jj)));
            __m256i xy = _mm256_cvtepu32_epi64(_mm_loadu_si128((__m128i*)(buf.ref_dis + i * stride + jj)));

            __m256i msigma1 = _mm256_sub_epi64(xx, mu1sq);
            __m256i msigma2 = _mm256_sub_epi64(yy, mu2sq);
            __m256i msigma12 = _mm256_sub_epi64(xy, mu1mu2);
            msigma2 = _mm256_max_epi64(msigma2, _mm256_setzero_si256());
            msigma12 = _mm256_max_epi64(msigma12, _mm256_setzero_si256());

            // log stage
            __m256i mlog_den_stage1 = _mm256_add_epi64(msigma1, _mm256_set1_epi64x(sigma_nsq));
            __m256i mnorm = _mm256_sub_epi64(_mm256_set1_epi64x(48), _mm256_lzcnt_epi64(mlog_den_stage1));
            __m256i mlog_den1 = _mm256_srlv_epi64(mlog_den_stage1, mnorm);
            __m256i mx = _mm256_sub_epi64(_mm256_setzero_si256(), mnorm);
            // note: I'm getting 32 bit here, but I need just 16!
            __m256i mden_val = _mm256_i32gather_epi64(log2_table, _mm256_cvtusepi64_epi32(mlog_den1), sizeof(*log2_table));
            mden_val = _mm256_and_si256(mden_val, _mm256_set1_epi64x(0xffff)); // we took 64 bits, we need 16
            __m256i msigma1_mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(sigma_nsq), msigma1);
            __m256i msigma2_mask = _mm256_cmpgt_epi64(msigma2, _mm256_setzero_si256());
            msigma12 = _mm256_and_si256(msigma2_mask, msigma12);
            maccum_x = _mm256_add_epi64(maccum_x, _mm256_andnot_si256(msigma1_mask, _mm256_add_epi64(mx, _mm256_set1_epi64x(17))));
            __m256d msigma1_d = _mm256_cvtepu64_pd(msigma1);
            __m256d mg = _mm256_div_pd(_mm256_cvtepu64_pd(msigma12), _mm256_add_pd(msigma1_d, _mm256_set1_pd(eps)));
            __m256i msv_sq = _mm256_cvttpd_epi64(_mm256_sub_pd(_mm256_cvtepi64_pd(msigma2), _mm256_mul_pd(mg, _mm256_cvtepi64_pd(msigma12))));
            msv_sq = _mm256_max_epi64(msv_sq, _mm256_setzero_si256());
            mg = _mm256_min_pd(mg, _mm256_set1_pd(vif_enhn_gain_limit));

            __m256i mnumer1 = _mm256_add_epi64(msv_sq, _mm256_set1_epi64x(sigma_nsq));
            __m256i mnumer1_tmp = _mm256_add_epi64(mnumer1, _mm256_cvttpd_epi64(_mm256_mul_pd(_mm256_mul_pd(mg, mg), msigma1_d)));
            // TODO: macro
            __m256i mnumer1_tmp_lz = _mm256_lzcnt_epi64(mnumer1_tmp);
            mnorm = _mm256_sub_epi64(_mm256_set1_epi64x(48), mnumer1_tmp_lz);
            __m256i mnumlog = _mm256_srlv_epi64(mnumer1_tmp, mnorm);
            // TODO: macro
            __m256i mnumer1_lz = _mm256_lzcnt_epi64(mnumer1);
            mnorm = _mm256_sub_epi64(_mm256_set1_epi64x(48), mnumer1_lz);
            __m256i mdenlog = _mm256_srlv_epi64(mnumer1, mnorm);
            maccum_x2 = _mm256_add_epi64(maccum_x2, _mm256_andnot_si256(msigma1_mask, _mm256_sub_epi64(mnumer1_lz, mnumer1_tmp_lz)));

            __m256i m0 = _mm256_and_si256(_mm256_set1_epi64x(0xffff), _mm256_i32gather_epi64(log2_table, _mm256_cvtusepi64_epi32(mnumlog), sizeof(*log2_table))); // we took 64 bits, we need 16
            __m256i m1 = _mm256_and_si256(_mm256_set1_epi64x(0xffff), _mm256_i32gather_epi64(log2_table, _mm256_cvtusepi64_epi32(mdenlog), sizeof(*log2_table))); // we took 64 bits, we need 16
            __m256i mnum_val = _mm256_sub_epi64(m0, m1);

            maccum_num_log = _mm256_add_epi64(maccum_num_log, _mm256_andnot_si256(msigma1_mask, mnum_val));
            maccum_den_log = _mm256_add_epi64(maccum_den_log, _mm256_andnot_si256(msigma1_mask, mden_val));

            // non log stage
            maccum_num_non_log = _mm256_add_epi64(maccum_num_non_log, _mm256_and_si256(msigma1_mask, msigma2));
            maccum_den_non_log = _mm256_add_epi64(maccum_den_non_log, _mm256_and_si256(msigma1_mask, _mm256_set1_epi64x(1)));

            for (unsigned xx = 0; xx < 4; xx++) {
                unsigned j = jj + xx;
                uint32_t mu1_val = buf.mu1_32[i * stride + j];
                uint32_t mu2_val = buf.mu2_32[i * stride + j];
                uint32_t mu1_sq_val = (uint32_t)((((uint64_t)mu1_val * mu1_val)
                    + 2147483648) >> 32);
                uint32_t mu2_sq_val = (uint32_t)((((uint64_t)mu2_val * mu2_val)
                    + 2147483648) >> 32);
                uint32_t mu1_mu2_val = (uint32_t)((((uint64_t)mu1_val * mu2_val)
                    + 2147483648) >> 32);

                uint32_t xx_filt_val = xx_filt[i * stride + j];
                uint32_t yy_filt_val = yy_filt[i * stride + j];
                uint32_t xy_filt_val = xy_filt[i * stride + j];

                int32_t sigma1_sq = (int32_t)(xx_filt_val - mu1_sq_val);
                int32_t sigma2_sq = (int32_t)(yy_filt_val - mu2_sq_val);
                int32_t sigma12 = (int32_t)(xy_filt_val - mu1_mu2_val);

                sigma2_sq = MAX(sigma2_sq, 0);
                sigma12 = MAX(sigma12, 0);
                if (sigma2_sq == 0) { // this will zero g, but we're still on integers!
                    sigma12 = 0;
                }

                assert(msigma1.m256i_u64[xx] == sigma1_sq);
                assert(msigma12.m256i_u64[xx] == sigma12);
                assert(msigma2.m256i_u64[xx] == sigma2_sq);

                if (sigma1_sq >= sigma_nsq) {
                    uint32_t log_den_stage1 = (uint32_t)(sigma_nsq + sigma1_sq);
                    int x;
                    uint16_t log_den1 = get_best16_from32(log_den_stage1, &x);
                    assert(mx.m256i_u64[xx] == x);
                    assert(mlog_den1.m256i_u64[xx] == log_den1);

                    /**
                    * log values are taken from the look-up table generated by
                    * log_generate() function which is called in integer_combo_threadfunc
                    * den_val in float is log2(1 + sigma1_sq/2)
                    * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                    * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                    * x because best 16 bits are taken
                    */
                    accum_x[xx] += x + 17;
                    den_val = log2_table[log_den1];
                    int64_t den_val = log2_table[log_den1];

                    assert(maccum_x.m256i_u64[xx] == accum_x[xx]);
                    assert(mden_val.m256i_u64[xx] == den_val);

                    // num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
                    /**
                    * In floating-point numerator = log2((1.0f + (g * g * sigma1_sq)/(sv_sq + sigma_nsq))
                    *
                    * In Fixed-point the above is converted to
                    * numerator = log2((sv_sq + sigma_nsq)+(g * g * sigma1_sq))- log2(sv_sq + sigma_nsq)
                    */
                    double g = sigma12 / (sigma1_sq + eps); // this epsilon can go away
                    int32_t sv_sq = sigma2_sq - g * sigma12;
                    sv_sq = (uint32_t)(MAX(sv_sq, 0));
                    g = MIN(g, vif_enhn_gain_limit);
                    assert(msv_sq.m256i_i64[xx] == sv_sq);
                    assert(mg.m256d_f64[xx] == g);

                    int x1, x2;
                    uint32_t numer1 = (sv_sq + sigma_nsq);
                    int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
                    uint16_t numlog = get_best16_from64((uint64_t)numer1_tmp, &x1);
                    uint16_t denlog = get_best16_from64((uint64_t)numer1, &x2);
                    accum_x2[xx] += (x2 - x1);
                    assert(maccum_x2.m256i_u64[xx] == accum_x2[xx]);
                    num_val = log2_table[numlog] - log2_table[denlog];
                    assert(mnum_val.m256i_i64[xx] == num_val);

                    accum_num_log[xx] += num_val;
                    accum_den_log[xx] += den_val;
                }
                else {
                    accum_num_non_log[xx] += sigma2_sq;
                    accum_den_non_log[xx]++;
                }
                assert(maccum_num_log.m256i_i64[xx] == accum_num_log[xx]);
                assert(maccum_den_log.m256i_i64[xx] == accum_den_log[xx]);
                assert(maccum_num_non_log.m256i_i64[xx] == accum_num_non_log[xx]);
                assert(maccum_den_non_log.m256i_i64[xx] == accum_den_non_log[xx]);
                assert(maccum_x.m256i_u64[xx] == accum_x[xx]);
                assert(maccum_x2.m256i_u64[xx] == accum_x2[xx]);
            }
        }
    }
    int64_t accum_num_log_all = _mm256_extract_epi64(maccum_num_log, 0) + _mm256_extract_epi64(maccum_num_log, 1) + _mm256_extract_epi64(maccum_num_log, 2) + _mm256_extract_epi64(maccum_num_log, 3);
    int64_t accum_den_log_all = _mm256_extract_epi64(maccum_den_log, 0) + _mm256_extract_epi64(maccum_den_log, 1) + _mm256_extract_epi64(maccum_den_log, 2) + _mm256_extract_epi64(maccum_den_log, 3);
    int64_t accum_num_non_log_all = _mm256_extract_epi64(maccum_num_non_log, 0) + _mm256_extract_epi64(maccum_num_non_log, 1) + _mm256_extract_epi64(maccum_num_non_log, 2) + _mm256_extract_epi64(maccum_num_non_log, 3);
    int64_t accum_den_non_log_all = _mm256_extract_epi64(maccum_den_non_log, 0) + _mm256_extract_epi64(maccum_den_non_log, 1) + _mm256_extract_epi64(maccum_den_non_log, 2) + _mm256_extract_epi64(maccum_den_non_log, 3);
    int64_t accum_x_all = _mm256_extract_epi64(maccum_x, 0) + _mm256_extract_epi64(maccum_x, 1) + _mm256_extract_epi64(maccum_x, 2) + _mm256_extract_epi64(maccum_x, 3);
    int64_t accum_x2_all = _mm256_extract_epi64(maccum_x2, 0) + _mm256_extract_epi64(maccum_x2, 1) + _mm256_extract_epi64(maccum_x2, 2) + _mm256_extract_epi64(maccum_x2, 3);

    //changed calculation to increase performance
    num[0] = accum_num_log_all / 2048.0 + accum_x2_all + (accum_den_non_log_all - ((accum_num_non_log_all) / 16384.0) / (65025.0));
    den[0] = accum_den_log_all / 2048.0 - (accum_x_all)+accum_den_non_log_all;
}

// reference implementation for avx512
static void vif_statistic_avx512_ref(VifBuffer buf, float* num, float* den,
    unsigned w, unsigned h, uint16_t* log2_table,
    double vif_enhn_gain_limit)
{
    uint32_t* xx_filt = buf.ref_sq;
    uint32_t* yy_filt = buf.dis_sq;
    uint32_t* xy_filt = buf.ref_dis;

    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    int64_t num_val, den_val;
    int64_t accum_x[4] = { 0 };
    int64_t accum_x2[4] = { 0 };
    int64_t accum_num_log[4] = { 0 };
    int64_t accum_den_log[4] = { 0 };
    int64_t accum_num_non_log[4] = { 0 };
    int64_t accum_den_non_log[4] = { 0 };
    /**
        * In floating-point there are two types of numerator scores and denominator scores
        * 1. num = 1 - sigma1_sq * constant den =1  when sigma1_sq<2  here constant=4/(255*255)
        * 2. num = log2(((sigma2_sq+2)*sigma1_sq)/((sigma2_sq+2)*sigma1_sq-sigma12*sigma12) den=log2(1+(sigma1_sq/2)) else
        *
        * In fixed-point separate accumulator is used for non-log score accumulations and log-based score accumulation
        * For non-log accumulator of numerator, only sigma1_sq * constant in fixed-point is accumulated
        * log based values are separately accumulated.
        * While adding both accumulator values the non-log accumulator is converted such that it is equivalent to 1 - sigma1_sq * constant(1's are accumulated with non-log denominator accumulator)
    */
    for (unsigned i = 0; i < h; ++i) {
        for (unsigned jj = 0; jj < w; jj += 4) {
            for (unsigned xx = 0; xx < 4; xx++) {
                unsigned j = jj + xx;
                const ptrdiff_t stride = buf.stride_32 / sizeof(uint32_t);
                uint32_t mu1_val = buf.mu1_32[i * stride + j];
                uint32_t mu2_val = buf.mu2_32[i * stride + j];
                uint32_t mu1_sq_val = (uint32_t)((((uint64_t)mu1_val * mu1_val)
                    + 2147483648) >> 32);
                uint32_t mu2_sq_val = (uint32_t)((((uint64_t)mu2_val * mu2_val)
                    + 2147483648) >> 32);
                uint32_t mu1_mu2_val = (uint32_t)((((uint64_t)mu1_val * mu2_val)
                    + 2147483648) >> 32);

                uint32_t xx_filt_val = xx_filt[i * stride + j];
                uint32_t yy_filt_val = yy_filt[i * stride + j];
                uint32_t xy_filt_val = xy_filt[i * stride + j];

                int32_t sigma1_sq = (int32_t)(xx_filt_val - mu1_sq_val);
                int32_t sigma2_sq = (int32_t)(yy_filt_val - mu2_sq_val);
                int32_t sigma12 = (int32_t)(xy_filt_val - mu1_mu2_val);

                sigma2_sq = MAX(sigma2_sq, 0);
                sigma12 = MAX(sigma12, 0);
                if (sigma1_sq >= sigma_nsq) {
                    uint32_t log_den_stage1 = (uint32_t)(sigma_nsq + sigma1_sq);
                    int x;
                    uint16_t log_den1 = get_best16_from32(log_den_stage1, &x);

                    /**
                    * log values are taken from the look-up table generated by
                    * log_generate() function which is called in integer_combo_threadfunc
                    * den_val in float is log2(1 + sigma1_sq/2)
                    * here it is converted to equivalent of log2(2+sigma1_sq) - log2(2) i.e log2(2*65536+sigma1_sq) - 17
                    * multiplied by 2048 as log_value = log2(i)*2048 i=16384 to 65535 generated using log_value
                    * x because best 16 bits are taken
                    */
                    accum_x[xx] += x + 17;
                    den_val = log2_table[log_den1];

                    if (sigma2_sq == 0) { // this will zero g, but we're still on integers!
                        sigma12 = 0;
                    }

                    const double eps = 65536 * 1.0e-10;
                    double g = sigma12 / (sigma1_sq + eps); // this epsilon can go away
                    int32_t sv_sq = sigma2_sq - g * sigma12;
                    sv_sq = (uint32_t)(MAX(sv_sq, 0));
                    g = MIN(g, vif_enhn_gain_limit);

                    int x1, x2;
                    uint32_t numer1 = (sv_sq + sigma_nsq);
                    int64_t numer1_tmp = (int64_t)((g * g * sigma1_sq)) + numer1; //numerator
                    uint16_t numlog = get_best16_from64((uint64_t)numer1_tmp, &x1);
                    uint16_t denlog = get_best16_from64((uint64_t)numer1, &x2);
                    accum_x2[xx] += (x2 - x1);
                    num_val = log2_table[numlog] - log2_table[denlog];
                    accum_num_log[xx] += num_val;
                    accum_den_log[xx] += den_val;
                }
                else {
                    accum_num_non_log[xx] += sigma2_sq;
                    accum_den_non_log[xx]++;
                }
            }
        }
    }
    int64_t accum_num_log_all = accum_num_log[0] + accum_num_log[1] + accum_num_log[2] + accum_num_log[3];
    int64_t accum_den_log_all = accum_den_log[0] + accum_den_log[1] + accum_den_log[2] + accum_den_log[3];
    int64_t accum_num_non_log_all = accum_num_non_log[0] + accum_num_non_log[1] + accum_num_non_log[2] + accum_num_non_log[3];
    int64_t accum_den_non_log_all = accum_den_non_log[0] + accum_den_non_log[1] + accum_den_non_log[2] + accum_den_non_log[3];
    int64_t accum_x_all = accum_x[0] + accum_x[1] + accum_x[2] + accum_x[3];
    int64_t accum_x2_all = accum_x2[0] + accum_x2[1] + accum_x2[2] + accum_x2[3];

    //changed calculation to increase performance
    num[0] = accum_num_log_all / 2048.0 + accum_x2_all + (accum_den_non_log_all - ((accum_num_non_log_all) / 16384.0) / (65025.0));
    den[0] = accum_den_log_all / 2048.0 - (accum_x_all)+accum_den_non_log_all;
}
