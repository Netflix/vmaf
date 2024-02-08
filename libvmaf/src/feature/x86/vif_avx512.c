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
#include "feature/integer_vif.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static inline void
pad_top_and_bottom(VifBuffer buf, unsigned h, int fwidth)
{
    const unsigned fwidth_half = fwidth / 2;
    unsigned char *ref = buf.ref;
    unsigned char *dis = buf.dis;
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
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;
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

typedef struct Residuals512 {
    __m512i maccum_num_log;
    __m512i maccum_den_log;
    __m512i maccum_num_non_log;
    __m512i maccum_den_non_log;
} Residuals512;

// compute VIF on a 16 pixel block from xx (ref variance), yy (clamped dis variance), xy (ref dis covariance)
static inline void vif_statistic_avx512(Residuals512 *out, __m512i xx, __m512i xy, __m512i yy, const uint16_t *log2_table, double vif_enhn_gain_limit)
{
    //float equivalent of 2. (2 * 65536)
    static const int32_t sigma_nsq = 65536 << 1;

    __m512i maccum_num_log = out->maccum_num_log;
    __m512i maccum_den_log = out->maccum_den_log;
    __m512i maccum_num_non_log = out->maccum_num_non_log;
    __m512i maccum_den_non_log = out->maccum_den_non_log;

    const double eps = 65536 * 1.0e-10;

    for (int b = 0; b < 16; b += 8) {
        __m512i msigma1 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(xx));
        __m512i msigma2 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(yy));
        __m512i msigma12 = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(xy));
        xx = _mm512_castsi256_si512(_mm512_extracti64x4_epi64(xx, 1));
        yy = _mm512_castsi256_si512(_mm512_extracti64x4_epi64(yy, 1));
        xy = _mm512_castsi256_si512(_mm512_extracti64x4_epi64(xy, 1));
        msigma2 = _mm512_max_epi64(msigma2, _mm512_setzero_si512());
        msigma12 = _mm512_max_epi64(msigma12, _mm512_setzero_si512());

        // log stage
        __m512i mlog_den_stage1 = _mm512_add_epi64(msigma1, _mm512_set1_epi64(sigma_nsq));
        __m512i mnorm = _mm512_sub_epi64(_mm512_set1_epi64(48), _mm512_lzcnt_epi64(mlog_den_stage1));
        __m512i mlog_den1 = _mm512_srlv_epi64(mlog_den_stage1, mnorm);
        // note: I'm getting 32 bit here, but I need just 16!
        __m512i mden_val = _mm512_i32gather_epi64(_mm512_cvtusepi64_epi32(mlog_den1), log2_table, sizeof(*log2_table));
        mden_val = _mm512_and_si512(mden_val, _mm512_set1_epi64(0xffff)); // we took 64 bits, we need 16
        mden_val = _mm512_add_epi64(mden_val, _mm512_slli_epi64(mnorm, 11));
        mden_val = _mm512_sub_epi64(mden_val, _mm512_set1_epi64(2048 * 17));
        __mmask8 msigma1_mask = _mm512_cmpgt_epi64_mask(_mm512_set1_epi64(sigma_nsq), msigma1);
        __mmask8 msigma2_mask = _mm512_cmpgt_epi64_mask(msigma2, _mm512_setzero_si512());
        __mmask8 msigma12_mask = _mm512_cmpgt_epi64_mask(msigma12, _mm512_setzero_si512());
        __m512d msigma1_d = _mm512_cvtepu64_pd(msigma1);
        __m512d mg = _mm512_div_pd(_mm512_cvtepu64_pd(msigma12), _mm512_add_pd(msigma1_d, _mm512_set1_pd(eps)));
        __m512i msv_sq = _mm512_cvttpd_epi64(_mm512_sub_pd(_mm512_cvtepi64_pd(msigma2), _mm512_mul_pd(mg, _mm512_cvtepi64_pd(msigma12))));
        msv_sq = _mm512_max_epi64(msv_sq, _mm512_setzero_si512());
        mg = _mm512_min_pd(mg, _mm512_set1_pd(vif_enhn_gain_limit));

        __m512i mnumer1 = _mm512_add_epi64(msv_sq, _mm512_set1_epi64(sigma_nsq));
        __m512i mnumer1_lz = _mm512_sub_epi64(_mm512_set1_epi64(48), _mm512_lzcnt_epi64(mnumer1));
        __m512i mnumer1_mantissa = _mm512_srlv_epi64(mnumer1, mnumer1_lz);
        __m512i mnumer1_mantissa_log = _mm512_and_si512(_mm512_set1_epi64(0xffff), _mm512_i32gather_epi64(_mm512_cvtusepi64_epi32(mnumer1_mantissa), log2_table, sizeof(*log2_table))); // we took 64 bits, we need 16
        __m512i mnumer1_log = _mm512_add_epi64(mnumer1_mantissa_log, _mm512_slli_epi64(mnumer1_lz, 11));

        __m512i mnumer1_tmp = _mm512_add_epi64(mnumer1, _mm512_cvttpd_epi64(_mm512_mul_pd(_mm512_mul_pd(mg, mg), msigma1_d)));
        __m512i mnumer1_tmp_lz = _mm512_sub_epi64(_mm512_set1_epi64(48), _mm512_lzcnt_epi64(mnumer1_tmp));
        __m512i mnumer1_tmp_mantissa = _mm512_srlv_epi64(mnumer1_tmp, mnumer1_tmp_lz);
        __m512i mnumer1_tmp_mantissa_log = _mm512_and_si512(_mm512_set1_epi64(0xffff), _mm512_i32gather_epi64(_mm512_cvtusepi64_epi32(mnumer1_tmp_mantissa), log2_table, sizeof(*log2_table))); // we took 64 bits, we need 16
        __m512i mnumer1_tmp_log = _mm512_add_epi64(mnumer1_tmp_mantissa_log, _mm512_slli_epi64(mnumer1_tmp_lz, 11));

        __m512i mnum_val = _mm512_sub_epi64(mnumer1_tmp_log, mnumer1_log);

        maccum_num_log = _mm512_mask_add_epi64(maccum_num_log, (~msigma1_mask) & msigma12_mask & msigma2_mask, maccum_num_log, mnum_val);
        maccum_den_log = _mm512_mask_add_epi64(maccum_den_log, ~msigma1_mask, maccum_den_log, mden_val);

        // non log stage
        maccum_num_non_log = _mm512_mask_add_epi64(maccum_num_non_log, msigma1_mask, maccum_num_non_log, msigma2);
        maccum_den_non_log = _mm512_mask_add_epi64(maccum_den_non_log, msigma1_mask, maccum_den_non_log, _mm512_set1_epi64(1));
    }

    out->maccum_num_log = maccum_num_log;
    out->maccum_den_log = maccum_den_log;
    out->maccum_num_non_log = maccum_num_non_log;
    out->maccum_den_non_log = maccum_den_non_log;
}

void vif_statistic_8_avx512(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h) {
    const unsigned fwidth = vif_filter1d_width[0];
    const uint16_t *vif_filt = vif_filter1d_table[0];
    VifBuffer buf = s->buf;
    const uint8_t *ref = (uint8_t*)buf.ref;
    const uint8_t *dis = (uint8_t*)buf.dis;
    const unsigned fwidth_half = fwidth >> 1;
    const uint16_t *log2_table = s->log2_table;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;

#if defined __GNUC__
#define ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined (_MSC_VER)  && (!defined UNDER_CE)
#define ALIGNED(x) __declspec (align(x))
#else
#define ALIGNED(x)
#endif

    int64_t accum_num_log = 0;
    int64_t accum_den_log = 0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;

    __m512i round_128 = _mm512_set1_epi32(128);
    __m512i mask2 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);

    Residuals512 residuals;
    residuals.maccum_den_log = _mm512_setzero_si512();
    residuals.maccum_num_log = _mm512_setzero_si512();
    residuals.maccum_den_non_log = _mm512_setzero_si512();
    residuals.maccum_num_non_log = _mm512_setzero_si512();
    for (unsigned i = 0; i < h; ++i)
    {
        // VERTICAL
        int i_back = i - fwidth_half;
        int i_forward = i + fwidth_half;

        // First consider all blocks of 16 elements until it's not possible anymore
        unsigned n = w >> 4;
        for (unsigned jj = 0; jj < n << 4; jj += 16) {

            __m512i f0 = _mm512_set1_epi32(vif_filt[fwidth / 2]);
            __m512i r0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ref + (buf.stride * i) + jj)));
            __m512i d0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(dis + (buf.stride * i) + jj)));

            // filtered r,d
            __m512i accum_mu1 = _mm512_mullo_epi32(r0, f0);
            __m512i accum_mu2 = _mm512_mullo_epi32(d0, f0);
            __m512i accum_ref = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(r0, r0));
            __m512i accum_dis = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(d0, d0));
            __m512i accum_ref_dis = _mm512_mullo_epi32(f0, _mm512_mullo_epi32(r0, d0));

            for (unsigned int tap = 0; tap < fwidth / 2; tap++) {
                int ii_back = i_back + tap;
                int ii_forward = i_forward - tap;

                __m512i f0 = _mm512_set1_epi32(vif_filt[tap]);
                __m512i r0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ref + (buf.stride * ii_back) + jj)));
                __m512i d0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(dis + (buf.stride * ii_back) + jj)));
                __m512i r1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ref + (buf.stride * ii_forward) + jj)));
                __m512i d1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(dis + (buf.stride * ii_forward) + jj)));

                accum_mu1 = _mm512_add_epi32(accum_mu1, _mm512_mullo_epi32(_mm512_add_epi32(r0, r1), f0));
                accum_mu2 = _mm512_add_epi32(accum_mu2, _mm512_mullo_epi32(_mm512_add_epi32(d0, d1), f0));
                accum_ref = _mm512_add_epi32(accum_ref, _mm512_mullo_epi32(f0, _mm512_add_epi32(_mm512_mullo_epi32(r0, r0), _mm512_mullo_epi32(r1, r1))));
                accum_dis = _mm512_add_epi32(accum_dis, _mm512_mullo_epi32(f0, _mm512_add_epi32(_mm512_mullo_epi32(d0, d0), _mm512_mullo_epi32(d1, d1))));
                accum_ref_dis = _mm512_add_epi32(accum_ref_dis, _mm512_mullo_epi32(f0, _mm512_add_epi32(_mm512_mullo_epi32(d0, r0), _mm512_mullo_epi32(d1, r1))));
            }
            accum_mu1 = _mm512_add_epi32(accum_mu1, round_128);
            accum_mu2 = _mm512_add_epi32(accum_mu2, round_128);
            accum_mu1 = _mm512_srli_epi32(accum_mu1, 0x08);
            accum_mu2 = _mm512_srli_epi32(accum_mu2, 0x08);

            _mm512_storeu_si512((__m512i*)(buf.tmp.mu1 + jj), accum_mu1);
            _mm512_storeu_si512((__m512i*)(buf.tmp.mu2 + jj), accum_mu2);
            _mm512_storeu_si512((__m512i*)(buf.tmp.ref + jj), accum_ref);
            _mm512_storeu_si512((__m512i*)(buf.tmp.dis + jj), accum_dis);
            _mm512_storeu_si512((__m512i*)(buf.tmp.ref_dis + jj), accum_ref_dis);
        }
        // Then consider the remaining elements individually
        for (unsigned j = n << 4; j < w; ++j) {
            uint32_t accum_mu1 = 0;
            uint32_t accum_mu2 = 0;
            uint64_t accum_ref = 0;
            uint64_t accum_dis = 0;
            uint64_t accum_ref_dis = 0;

            for (unsigned fi = 0; fi < fwidth; ++fi) {
                int ii = i - fwidth_half;
                int ii_check = ii + fi;
                const uint16_t fcoeff = vif_filt[fi];
                uint16_t imgcoeff_ref = ref[ii_check * buf.stride + j];
                uint16_t imgcoeff_dis = dis[ii_check * buf.stride + j];
                uint32_t img_coeff_ref = fcoeff * (uint32_t)imgcoeff_ref;
                uint32_t img_coeff_dis = fcoeff * (uint32_t)imgcoeff_dis;
                accum_mu1 += img_coeff_ref;
                accum_mu2 += img_coeff_dis;
                accum_ref += img_coeff_ref * (uint64_t)imgcoeff_ref;
                accum_dis += img_coeff_dis * (uint64_t)imgcoeff_dis;
                accum_ref_dis += img_coeff_ref * (uint64_t)imgcoeff_dis;
            }

            buf.tmp.mu1[j] = (accum_mu1 + 128) >> 8;
            buf.tmp.mu2[j] = (accum_mu2 + 128) >> 8;
            buf.tmp.ref[j] = accum_ref;
            buf.tmp.dis[j] = accum_dis;
            buf.tmp.ref_dis[j] = accum_ref_dis;
        }

        PADDING_SQ_DATA(buf, w, fwidth_half);

        //HORIZONTAL
        for (unsigned j = 0; j < n << 4; j += 16) {
            __m512i mu1sq;
            __m512i mu2sq;
            __m512i mu1mu2;
            __m512i xx;
            __m512i yy;
            __m512i xy;
            __m512i mask5 = _mm512_set_epi32(30, 28, 14, 12, 26, 24, 10, 8, 22, 20, 6, 4, 18, 16, 2, 0);
            // compute mu1sq, mu2sq, mu1mu2
            {
                __m512i fq = _mm512_set1_epi32(vif_filt[fwidth / 2]);
                __m512i acc0 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + j + 0)), fq);
                __m512i acc1 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + j + 0)), fq);

                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi32(vif_filt[fj]);
                    acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + j - fwidth / 2 + fj + 0)), fq));
                    acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + j + fwidth / 2 - fj + 0)), fq));
                    acc1 = _mm512_add_epi64(acc1, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + j - fwidth / 2 + fj + 0)), fq));
                    acc1 = _mm512_add_epi64(acc1, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + j + fwidth / 2 - fj + 0)), fq));
                }
                __m512i mu1 = acc0;
                __m512i acc0_lo_512 = _mm512_unpacklo_epi32(acc0, _mm512_setzero_si512());
                __m512i acc0_hi_512 = _mm512_unpackhi_epi32(acc0, _mm512_setzero_si512());
                acc0_lo_512 = _mm512_mul_epu32(acc0_lo_512, acc0_lo_512);
                acc0_hi_512 = _mm512_mul_epu32(acc0_hi_512, acc0_hi_512);
                acc0_lo_512 = _mm512_srli_epi64(_mm512_add_epi64(acc0_lo_512, _mm512_set1_epi64(0x80000000)), 32);
                acc0_hi_512 = _mm512_srli_epi64(_mm512_add_epi64(acc0_hi_512, _mm512_set1_epi64(0x80000000)), 32);
                mu1sq = _mm512_permutex2var_epi32(acc0_lo_512, mask5, acc0_hi_512);

                __m512i acc0lo_512 = _mm512_unpacklo_epi32(acc1, _mm512_setzero_si512());
                __m512i acc0hi_512 = _mm512_unpackhi_epi32(acc1, _mm512_setzero_si512());
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

            // compute xx, yy, xy
            {
                __m512i rounder = _mm512_set1_epi64(0x8000);
                __m512i fq = _mm512_set1_epi64(vif_filt[fwidth / 2]);
                __m512i s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + 0))); // 4
                __m512i s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + 8))); // 4
                __m512i refsq_lo = _mm512_add_epi64(rounder, _mm512_mul_epu32(s0, fq));
                __m512i refsq_hi = _mm512_add_epi64(rounder, _mm512_mul_epu32(s2, fq));

                s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + 0))); // 4
                s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + 8))); // 4
                __m512i dissq_lo = _mm512_add_epi64(rounder, _mm512_mul_epu32(s0, fq));
                __m512i dissq_hi = _mm512_add_epi64(rounder, _mm512_mul_epu32(s2, fq));

                s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + 0))); // 4
                s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + 8))); // 4
                __m512i refdis_lo = _mm512_add_epi64(rounder, _mm512_mul_epu32(s0, fq));
                __m512i refdis_hi = _mm512_add_epi64(rounder, _mm512_mul_epu32(s2, fq));

                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi64(vif_filt[fj]);
                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j - fwidth / 2 + fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j - fwidth / 2 + fj + 8))); // 4
                    refsq_lo = _mm512_add_epi64(refsq_lo, _mm512_mul_epu32(s0, fq));
                    refsq_hi = _mm512_add_epi64(refsq_hi, _mm512_mul_epu32(s2, fq));
                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + fwidth / 2 - fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + fwidth / 2 - fj + 8))); // 4
                    refsq_lo = _mm512_add_epi64(refsq_lo, _mm512_mul_epu32(s0, fq));
                    refsq_hi = _mm512_add_epi64(refsq_hi, _mm512_mul_epu32(s2, fq));

                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j - fwidth / 2 + fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j - fwidth / 2 + fj + 8))); // 4
                    dissq_lo = _mm512_add_epi64(dissq_lo, _mm512_mul_epu32(s0, fq));
                    dissq_hi = _mm512_add_epi64(dissq_hi, _mm512_mul_epu32(s2, fq));
                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + fwidth / 2 - fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + fwidth / 2 - fj + 8))); // 4
                    dissq_lo = _mm512_add_epi64(dissq_lo, _mm512_mul_epu32(s0, fq));
                    dissq_hi = _mm512_add_epi64(dissq_hi, _mm512_mul_epu32(s2, fq));

                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j - fwidth / 2 + fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j - fwidth / 2 + fj + 8))); // 4
                    refdis_lo = _mm512_add_epi64(refdis_lo, _mm512_mul_epu32(s0, fq));
                    refdis_hi = _mm512_add_epi64(refdis_hi, _mm512_mul_epu32(s2, fq));
                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + fwidth / 2 - fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + fwidth / 2 - fj + 8))); // 4
                    refdis_lo = _mm512_add_epi64(refdis_lo, _mm512_mul_epu32(s0, fq));
                    refdis_hi = _mm512_add_epi64(refdis_hi, _mm512_mul_epu32(s2, fq));
                }
                refsq_lo = _mm512_srli_epi64(refsq_lo, 16);
                refsq_hi = _mm512_srli_epi64(refsq_hi, 16);
                __m512i refsq = _mm512_permutex2var_epi32(refsq_lo, mask2, refsq_hi);
                xx = _mm512_sub_epi32(refsq, mu1sq);

                dissq_lo = _mm512_srli_epi64(dissq_lo, 16);
                dissq_hi = _mm512_srli_epi64(dissq_hi, 16);
                __m512i dissq = _mm512_permutex2var_epi32(dissq_lo, mask2, dissq_hi);
                yy = _mm512_max_epi32(_mm512_sub_epi32(dissq, mu2sq), _mm512_setzero_si512());

                refdis_lo = _mm512_srli_epi64(refdis_lo, 16);
                refdis_hi = _mm512_srli_epi64(refdis_hi, 16);
                __m512i refdis = _mm512_permutex2var_epi32(refdis_lo, mask2, refdis_hi);
                xy = _mm512_sub_epi32(refdis, mu1mu2);
            }
            vif_statistic_avx512(&residuals, xx, xy, yy, log2_table, vif_enhn_gain_limit);
        }

        if ((n << 4) != w) {
            VifResiduals residuals = vif_compute_line_residuals(s, n << 4, w, 0);
            accum_num_log += residuals.accum_num_log;
            accum_den_log += residuals.accum_den_log;
            accum_num_non_log += residuals.accum_num_non_log;
            accum_den_non_log += residuals.accum_den_non_log;
        }
    }

    accum_num_log += _mm512_reduce_add_epi64(residuals.maccum_num_log);
    accum_den_log += _mm512_reduce_add_epi64(residuals.maccum_den_log);
    accum_num_non_log += _mm512_reduce_add_epi64(residuals.maccum_num_non_log);
    accum_den_non_log += _mm512_reduce_add_epi64(residuals.maccum_den_non_log);
    num[0] = accum_num_log / 2048.0 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 + accum_den_non_log;
}

void vif_statistic_16_avx512(struct VifPublicState *s, float *num, float *den, unsigned w, unsigned h, int bpc, int scale) {
    const unsigned fwidth = vif_filter1d_width[scale];
    const uint16_t *vif_filt = vif_filter1d_table[scale];
    VifBuffer buf = s->buf;
    const ptrdiff_t stride = buf.stride / sizeof(uint16_t);
    int fwidth_half = fwidth >> 1;

    int32_t add_shift_round_VP, shift_VP;
    int32_t add_shift_round_VP_sq, shift_VP_sq;
    const uint16_t *log2_table = s->log2_table;
    double vif_enhn_gain_limit = s->vif_enhn_gain_limit;
    __m512i mask2 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);

    Residuals512 residuals;
    residuals.maccum_den_log = _mm512_setzero_si512();
    residuals.maccum_num_log = _mm512_setzero_si512();
    residuals.maccum_den_non_log = _mm512_setzero_si512();
    residuals.maccum_num_non_log = _mm512_setzero_si512();

    int64_t accum_num_log = 0;
    int64_t accum_den_log = 0;
    int64_t accum_num_non_log = 0;
    int64_t accum_den_non_log = 0;

    if (scale == 0)
    {
        shift_VP = bpc;
        add_shift_round_VP = 1 << (bpc - 1);
        shift_VP_sq = (bpc - 8) * 2;
        add_shift_round_VP_sq = (bpc == 8) ? 0 : 1 << (shift_VP_sq - 1);
    }
    else
    {
        shift_VP = 16;
        add_shift_round_VP = 32768;
        shift_VP_sq = 16;
        add_shift_round_VP_sq = 32768;
    }
    __m512i addnum64 = _mm512_set1_epi64(add_shift_round_VP_sq);
    __m512i addnum = _mm512_set1_epi32(add_shift_round_VP);
    uint16_t *ref = buf.ref;
    uint16_t *dis = buf.dis;

    for (unsigned i = 0; i < h; ++i)
    {
        //VERTICAL
        int ii = i - fwidth_half;
        int n = w >> 5;
        for (int j = 0; j < n << 5; j = j + 32)
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
                __m512i f1 = _mm512_set1_epi16(fcoeff);
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
        for (int j = 0; j < n << 4; j = j + 16)
        {
            __m512i mu1sq;
            __m512i mu2sq;
            __m512i mu1mu2;
            __m512i xx;
            __m512i yy;
            __m512i xy;
            __m512i mask5 = _mm512_set_epi32(30, 28, 14, 12, 26, 24, 10, 8, 22, 20, 6, 4, 18, 16, 2, 0);
            // compute mu1sq, mu2sq, mu1mu2
            {
                __m512i fq = _mm512_set1_epi32(vif_filt[fwidth / 2]);
                __m512i acc0 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + j + 0)), fq);
                __m512i acc1 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + j + 0)), fq);

                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi32(vif_filt[fj]);
                    acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + j - fwidth / 2 + fj + 0)), fq));
                    acc0 = _mm512_add_epi64(acc0, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu1 + j + fwidth / 2 - fj + 0)), fq));
                    acc1 = _mm512_add_epi64(acc1, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + j - fwidth / 2 + fj + 0)), fq));
                    acc1 = _mm512_add_epi64(acc1, _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(buf.tmp.mu2 + j + fwidth / 2 - fj + 0)), fq));
                }
                __m512i mu1 = acc0;
                __m512i acc0_lo_512 = _mm512_unpacklo_epi32(acc0, _mm512_setzero_si512());
                __m512i acc0_hi_512 = _mm512_unpackhi_epi32(acc0, _mm512_setzero_si512());
                acc0_lo_512 = _mm512_mul_epu32(acc0_lo_512, acc0_lo_512);
                acc0_hi_512 = _mm512_mul_epu32(acc0_hi_512, acc0_hi_512);
                acc0_lo_512 = _mm512_srli_epi64(_mm512_add_epi64(acc0_lo_512, _mm512_set1_epi64(0x80000000)), 32);
                acc0_hi_512 = _mm512_srli_epi64(_mm512_add_epi64(acc0_hi_512, _mm512_set1_epi64(0x80000000)), 32);
                mu1sq = _mm512_permutex2var_epi32(acc0_lo_512, mask5, acc0_hi_512);

                __m512i acc0lo_512 = _mm512_unpacklo_epi32(acc1, _mm512_setzero_si512());
                __m512i acc0hi_512 = _mm512_unpackhi_epi32(acc1, _mm512_setzero_si512());
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

            // compute xx, yy, xy
            {
                __m512i rounder = _mm512_set1_epi64(0x8000);
                __m512i fq = _mm512_set1_epi64(vif_filt[fwidth / 2]);
                __m512i s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + 0))); // 4
                __m512i s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + 8))); // 4
                __m512i refsq_lo = _mm512_add_epi64(rounder, _mm512_mul_epu32(s0, fq));
                __m512i refsq_hi = _mm512_add_epi64(rounder, _mm512_mul_epu32(s2, fq));

                s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + 0))); // 4
                s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + 8))); // 4
                __m512i dissq_lo = _mm512_add_epi64(rounder, _mm512_mul_epu32(s0, fq));
                __m512i dissq_hi = _mm512_add_epi64(rounder, _mm512_mul_epu32(s2, fq));

                s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + 0))); // 4
                s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + 8))); // 4
                __m512i refdis_lo = _mm512_add_epi64(rounder, _mm512_mul_epu32(s0, fq));
                __m512i refdis_hi = _mm512_add_epi64(rounder, _mm512_mul_epu32(s2, fq));

                for (unsigned fj = 0; fj < fwidth / 2; ++fj) {
                    __m512i fq = _mm512_set1_epi64(vif_filt[fj]);
                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j - fwidth / 2 + fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j - fwidth / 2 + fj + 8))); // 4
                    refsq_lo = _mm512_add_epi64(refsq_lo, _mm512_mul_epu32(s0, fq));
                    refsq_hi = _mm512_add_epi64(refsq_hi, _mm512_mul_epu32(s2, fq));
                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + fwidth / 2 - fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref + j + fwidth / 2 - fj + 8))); // 4
                    refsq_lo = _mm512_add_epi64(refsq_lo, _mm512_mul_epu32(s0, fq));
                    refsq_hi = _mm512_add_epi64(refsq_hi, _mm512_mul_epu32(s2, fq));

                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j - fwidth / 2 + fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j - fwidth / 2 + fj + 8))); // 4
                    dissq_lo = _mm512_add_epi64(dissq_lo, _mm512_mul_epu32(s0, fq));
                    dissq_hi = _mm512_add_epi64(dissq_hi, _mm512_mul_epu32(s2, fq));
                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + fwidth / 2 - fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.dis + j + fwidth / 2 - fj + 8))); // 4
                    dissq_lo = _mm512_add_epi64(dissq_lo, _mm512_mul_epu32(s0, fq));
                    dissq_hi = _mm512_add_epi64(dissq_hi, _mm512_mul_epu32(s2, fq));

                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j - fwidth / 2 + fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j - fwidth / 2 + fj + 8))); // 4
                    refdis_lo = _mm512_add_epi64(refdis_lo, _mm512_mul_epu32(s0, fq));
                    refdis_hi = _mm512_add_epi64(refdis_hi, _mm512_mul_epu32(s2, fq));
                    s0 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + fwidth / 2 - fj + 0))); // 4
                    s2 = _mm512_cvtepu32_epi64(_mm256_loadu_si256((__m256i*)(buf.tmp.ref_dis + j + fwidth / 2 - fj + 8))); // 4
                    refdis_lo = _mm512_add_epi64(refdis_lo, _mm512_mul_epu32(s0, fq));
                    refdis_hi = _mm512_add_epi64(refdis_hi, _mm512_mul_epu32(s2, fq));
                }
                refsq_lo = _mm512_srli_epi64(refsq_lo, 16);
                refsq_hi = _mm512_srli_epi64(refsq_hi, 16);
                __m512i refsq = _mm512_permutex2var_epi32(refsq_lo, mask2, refsq_hi);
                xx = _mm512_sub_epi32(refsq, mu1sq);

                dissq_lo = _mm512_srli_epi64(dissq_lo, 16);
                dissq_hi = _mm512_srli_epi64(dissq_hi, 16);
                __m512i dissq = _mm512_permutex2var_epi32(dissq_lo, mask2, dissq_hi);
                yy = _mm512_max_epi32(_mm512_sub_epi32(dissq, mu2sq), _mm512_setzero_si512());

                refdis_lo = _mm512_srli_epi64(refdis_lo, 16);
                refdis_hi = _mm512_srli_epi64(refdis_hi, 16);
                __m512i refdis = _mm512_permutex2var_epi32(refdis_lo, mask2, refdis_hi);
                xy = _mm512_sub_epi32(refdis, mu1mu2);
            }
            vif_statistic_avx512(&residuals, xx, xy, yy, log2_table, vif_enhn_gain_limit);
        }

        if ((n << 4) != (int)w) {
            VifResiduals residuals =
                vif_compute_line_residuals(s, n << 4, w, scale);
            accum_num_log += residuals.accum_num_log;
            accum_den_log += residuals.accum_den_log;
            accum_num_non_log += residuals.accum_num_non_log;
            accum_den_non_log += residuals.accum_den_non_log;
        }
    }

    accum_num_log += _mm512_reduce_add_epi64(residuals.maccum_num_log);
    accum_den_log += _mm512_reduce_add_epi64(residuals.maccum_den_log);
    accum_num_non_log += _mm512_reduce_add_epi64(residuals.maccum_num_non_log);
    accum_den_non_log += _mm512_reduce_add_epi64(residuals.maccum_den_non_log);


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
    num[0] = accum_num_log / 2048.0 + (accum_den_non_log - ((accum_num_non_log) / 16384.0) / (65025.0));
    den[0] = accum_den_log / 2048.0 + accum_den_non_log;
}

void vif_subsample_rd_8_avx512(VifBuffer buf, unsigned w, unsigned h)
{
    const unsigned fwidth = vif_filter1d_width[1];
    const uint16_t *vif_filt_s1 = vif_filter1d_table[1];
    const uint8_t *ref = (uint8_t *)buf.ref;
    const uint8_t *dis = (uint8_t *)buf.dis;
    const ptrdiff_t stride = buf.stride_16 / sizeof(uint16_t);
    __m512i addnum = _mm512_set1_epi32(32768);

    // __m512i mask1 = _mm512_set_epi16(60, 56, 28, 24, 52, 48, 20, 16, 44,
    //                                  40, 12, 8, 36, 32, 4, 0, 60, 56, 28, 24,
    //                                  52, 48, 20, 16, 44, 40, 12, 8, 36, 32, 4, 0);
    const int M = 1 << 16;
    __m512i mask1 = _mm512_set_epi32(60 * M + 56, 28 * M + 24, 52 * M + 48, 20 * M + 16,
                                     44 * M + 40, 12 * M +  8, 36 * M + 32,  4 * M +  0,
                                     60 * M + 56, 28 * M + 24, 52 * M + 48, 20 * M + 16,
                                     44 * M + 40, 12 * M +  8, 36 * M + 32,  4 * M +  0);

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
        for (int j = 0; j < n << 5; j = j + 32)
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
        for (int j = 0; j < n << 4; j = j + 16)
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
        for (int j = 0; j < n << 4; j = j + 32)
        {
            int ii_check = ii;
            __m512i accumr_lo, accumr_hi, accumd_lo, accumd_hi, rmul1, rmul2, dmul1, dmul2;
            accumr_lo = accumr_hi = accumd_lo = accumd_hi = rmul1 = rmul2 = dmul1 = dmul2 = _mm512_setzero_si512();
            __m512i mask3 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);   //first half of 512
            __m512i mask4 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4); //second half of 512
            for (unsigned fi = 0; fi < fwidth; ++fi, ii_check = ii + fi)
            {

                const uint16_t fcoeff = vif_filt[fi];
                __m512i f1 = _mm512_set1_epi16(fcoeff);
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
        for (int j = 0; j < n << 4; j = j + 16)
        {
            int jj = j - fwidth_half;
            int jj_check = jj;
            __m512i accumrlo, accumdlo, accumrhi, accumdhi;
            accumrlo = accumdlo = accumrhi = accumdhi = _mm512_setzero_si512();
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

            // __m512i mask2 = _mm512_set_epi16(60, 56, 28, 24, 52, 48, 20, 16, 44,
            //                                  40, 12, 8, 36, 32, 4, 0, 60, 56, 28, 24,
            //                                  52, 48, 20, 16, 44, 40, 12, 8, 36, 32, 4, 0);
            const int M = 1 << 16;
            __m512i mask2 = _mm512_set_epi32(60 * M + 56, 28 * M + 24, 52 * M + 48, 20 * M + 16,
                                             44 * M + 40, 12 * M +  8, 36 * M + 32,  4 * M +  0,
                                             60 * M + 56, 28 * M + 24, 52 * M + 48, 20 * M + 16,
                                             44 * M + 40, 12 * M +  8, 36 * M + 32,  4 * M +  0);

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
