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

#include "feature/integer_adm.h"

#include <immintrin.h>

void adm_dwt2_8_avx2(const uint8_t *src, const adm_dwt_band_t *dst,
                     AdmBuffer *buf, int w, int h, int src_stride,
                     int dst_stride)
{
    const int16_t *filter_lo = dwt2_db2_coeffs_lo;
    const int16_t *filter_hi = dwt2_db2_coeffs_hi;

    const int16_t shift_HP = 16;
    const int32_t add_shift_HP = 32768;
    int **ind_y = buf->ind_y;
    int **ind_x = buf->ind_x;

    int16_t *tmplo = (int16_t *)buf->tmp_ref;
    int16_t *tmphi = tmplo + w;
    int32_t accum;

    __m256i dwt2_db2_coeffs_lo_sum_const = _mm256_set1_epi32(5931776);
    __m256i fl0 =
        _mm256_broadcastd_epi32(_mm_loadu_si128((__m128i *)filter_lo));
    __m256i fl1 =
        _mm256_broadcastd_epi32(_mm_loadu_si128((__m128i *)(filter_lo + 2)));
    __m256i fh0 =
        _mm256_broadcastd_epi32(_mm_loadu_si128((__m128i *)filter_hi));
    __m256i fh1 =
        _mm256_broadcastd_epi32(_mm_loadu_si128((__m128i *)(filter_hi + 2)));
    __m256i add_shift_VP_vex = _mm256_set1_epi32(128);
    __m256i pad_register = _mm256_setzero_si256();
    __m256i add_shift_HP_vex = _mm256_set1_epi32(32768);

    for (int i = 0; i < (h + 1) / 2; ++i) {
        /* Vertical pass. */

        for (int j = 0; j < w; j = j + 16) {

            __m256i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
            accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                _mm256_setzero_si256();
            __m256i s0, s1, s2, s3;

            s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src + (ind_y[0][i] * src_stride) + j)));
            s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src + (ind_y[1][i] * src_stride) + j)));
            s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src + (ind_y[2][i] * src_stride) + j)));
            s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                (__m128i *)(src + (ind_y[3][i] * src_stride) + j)));

            __m256i s0lo = _mm256_unpacklo_epi16(s0, s1);
            __m256i s0hi = _mm256_unpackhi_epi16(s0, s1);
            accum_mu2_lo =
                _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s0lo, fl0));
            accum_mu2_hi =
                _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s0hi, fl0));

            __m256i s1lo = _mm256_unpacklo_epi16(s2, s3);
            __m256i s1hi = _mm256_unpackhi_epi16(s2, s3);
            accum_mu2_lo =
                _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s1lo, fl1));
            accum_mu2_hi =
                _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s1hi, fl1));

            accum_mu2_lo =
                _mm256_sub_epi32(accum_mu2_lo, dwt2_db2_coeffs_lo_sum_const);
            accum_mu2_hi =
                _mm256_sub_epi32(accum_mu2_hi, dwt2_db2_coeffs_lo_sum_const);

            accum_mu2_lo = _mm256_add_epi32(accum_mu2_lo, add_shift_VP_vex);
            accum_mu2_lo = _mm256_srli_epi32(accum_mu2_lo, 0x08);
            accum_mu2_hi = _mm256_add_epi32(accum_mu2_hi, add_shift_VP_vex);
            accum_mu2_hi = _mm256_srli_epi32(accum_mu2_hi, 0x08);
            accum_mu2_lo = _mm256_blend_epi16(accum_mu2_lo, pad_register, 0xAA);
            accum_mu2_hi = _mm256_blend_epi16(accum_mu2_hi, pad_register, 0xAA);

            accum_mu2_hi = _mm256_packus_epi32(accum_mu2_lo, accum_mu2_hi);
            _mm256_storeu_si256((__m256i *)(tmplo + j), accum_mu2_hi);

            accum_mu1_lo =
                _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s0lo, fh0));
            accum_mu1_hi =
                _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s0hi, fh0));
            accum_mu1_lo =
                _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s1lo, fh1));
            accum_mu1_hi =
                _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s1hi, fh1));

            accum_mu1_lo = _mm256_add_epi32(accum_mu1_lo, add_shift_VP_vex);
            accum_mu1_lo = _mm256_srli_epi32(accum_mu1_lo, 0x08);
            accum_mu1_hi = _mm256_add_epi32(accum_mu1_hi, add_shift_VP_vex);
            accum_mu1_hi = _mm256_srli_epi32(accum_mu1_hi, 0x08);
            accum_mu1_lo = _mm256_blend_epi16(accum_mu1_lo, pad_register, 0xAA);
            accum_mu1_hi = _mm256_blend_epi16(accum_mu1_hi, pad_register, 0xAA);
            accum_mu1_hi = _mm256_packus_epi32(accum_mu1_lo, accum_mu1_hi);
            _mm256_storeu_si256((__m256i *)(tmphi + j), accum_mu1_hi);
            // for( int k =0; k<16;k++){
            //     fprintf(stderr, "actual value hi tmp is %d \n",tmphi[j +k]);
            // }
        }

        int j0 = ind_x[0][0];
        int j1 = ind_x[1][0];
        int j2 = ind_x[2][0];
        int j3 = ind_x[3][0];

        int16_t s0 = tmplo[j0];
        int16_t s1 = tmplo[j1];
        int16_t s2 = tmplo[j2];
        int16_t s3 = tmplo[j3];

        accum = 0;
        accum += (int32_t)filter_lo[0] * s0;
        accum += (int32_t)filter_lo[1] * s1;
        accum += (int32_t)filter_lo[2] * s2;
        accum += (int32_t)filter_lo[3] * s3;
        dst->band_a[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        accum = 0;
        accum += (int32_t)filter_hi[0] * s0;
        accum += (int32_t)filter_hi[1] * s1;
        accum += (int32_t)filter_hi[2] * s2;
        accum += (int32_t)filter_hi[3] * s3;
        dst->band_v[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        s0 = tmphi[j0];
        s1 = tmphi[j1];
        s2 = tmphi[j2];
        s3 = tmphi[j3];

        accum = 0;
        accum += (int32_t)filter_lo[0] * s0;
        accum += (int32_t)filter_lo[1] * s1;
        accum += (int32_t)filter_lo[2] * s2;
        accum += (int32_t)filter_lo[3] * s3;
        dst->band_h[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        accum = 0;
        accum += (int32_t)filter_hi[0] * s0;
        accum += (int32_t)filter_hi[1] * s1;
        accum += (int32_t)filter_hi[2] * s2;
        accum += (int32_t)filter_hi[3] * s3;
        dst->band_d[i * dst_stride] = (accum + add_shift_HP) >> shift_HP;

        for (int j = 1; j < (w + 1) / 2; j = j + 16) {
            {
                __m256i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
                accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                    _mm256_setzero_si256();

                __m256i s00, s22, s33, s44;

                s00 = _mm256_loadu_si256((__m256i *)(tmplo + ind_x[0][j]));
                s22 = _mm256_loadu_si256((__m256i *)(tmplo + ind_x[2][j]));
                s33 = _mm256_loadu_si256((__m256i *)(tmplo + 16 + ind_x[0][j]));
                s44 = _mm256_loadu_si256((__m256i *)(tmplo + 16 + ind_x[2][j]));

                accum_mu2_lo =
                    _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s00, fl0));
                accum_mu2_hi =
                    _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s33, fl0));
                accum_mu2_lo =
                    _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s22, fl1));
                accum_mu2_hi =
                    _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s44, fl1));

                accum_mu2_lo = _mm256_add_epi32(accum_mu2_lo, add_shift_HP_vex);
                accum_mu2_lo = _mm256_srli_epi32(accum_mu2_lo, 0x10);
                accum_mu2_hi = _mm256_add_epi32(accum_mu2_hi, add_shift_HP_vex);
                accum_mu2_hi = _mm256_srli_epi32(accum_mu2_hi, 0x10);

                accum_mu2_hi = _mm256_packus_epi32(accum_mu2_lo, accum_mu2_hi);
                accum_mu2_hi = _mm256_permute4x64_epi64(accum_mu2_hi, 0xD8);
                _mm256_storeu_si256(
                    (__m256i *)(dst->band_a + i * dst_stride + j),
                    accum_mu2_hi);

                accum_mu1_lo =
                    _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s00, fh0));
                accum_mu1_hi =
                    _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s33, fh0));
                accum_mu1_lo =
                    _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s22, fh1));
                accum_mu1_hi =
                    _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s44, fh1));

                accum_mu1_lo = _mm256_add_epi32(accum_mu1_lo, add_shift_HP_vex);
                accum_mu1_lo = _mm256_srli_epi32(accum_mu1_lo, 0x10);
                accum_mu1_hi = _mm256_add_epi32(accum_mu1_hi, add_shift_HP_vex);
                accum_mu1_hi = _mm256_srli_epi32(accum_mu1_hi, 0x10);

                accum_mu1_hi = _mm256_packus_epi32(accum_mu1_lo, accum_mu1_hi);
                accum_mu1_hi = _mm256_permute4x64_epi64(accum_mu1_hi, 0xD8);
                _mm256_storeu_si256(
                    (__m256i *)(dst->band_v + i * dst_stride + j),
                    accum_mu1_hi);
            }

            {
                __m256i accum_mu2_lo, accum_mu2_hi, accum_mu1_lo, accum_mu1_hi;
                accum_mu2_lo = accum_mu2_hi = accum_mu1_lo = accum_mu1_hi =
                    _mm256_setzero_si256();

                __m256i s00, s22, s33, s44;

                __m256i add_shift_HP_vex = _mm256_set1_epi32(32768);

                s00 = _mm256_loadu_si256((__m256i *)(tmphi + ind_x[0][j]));
                s22 = _mm256_loadu_si256((__m256i *)(tmphi + ind_x[2][j]));
                s33 = _mm256_loadu_si256((__m256i *)(tmphi + 16 + ind_x[0][j]));
                s44 = _mm256_loadu_si256((__m256i *)(tmphi + 16 + ind_x[2][j]));

                accum_mu2_lo =
                    _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s00, fl0));
                accum_mu2_hi =
                    _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s33, fl0));
                accum_mu2_lo =
                    _mm256_add_epi32(accum_mu2_lo, _mm256_madd_epi16(s22, fl1));
                accum_mu2_hi =
                    _mm256_add_epi32(accum_mu2_hi, _mm256_madd_epi16(s44, fl1));

                accum_mu2_lo = _mm256_add_epi32(accum_mu2_lo, add_shift_HP_vex);
                accum_mu2_lo = _mm256_srli_epi32(accum_mu2_lo, 0x10);
                accum_mu2_hi = _mm256_add_epi32(accum_mu2_hi, add_shift_HP_vex);
                accum_mu2_hi = _mm256_srli_epi32(accum_mu2_hi, 0x10);

                accum_mu2_hi = _mm256_packus_epi32(accum_mu2_lo, accum_mu2_hi);
                accum_mu2_hi = _mm256_permute4x64_epi64(accum_mu2_hi, 0xD8);
                _mm256_storeu_si256(
                    (__m256i *)(dst->band_h + i * dst_stride + j),
                    accum_mu2_hi);

                accum_mu1_lo =
                    _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s00, fh0));
                accum_mu1_hi =
                    _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s33, fh0));
                accum_mu1_lo =
                    _mm256_add_epi32(accum_mu1_lo, _mm256_madd_epi16(s22, fh1));
                accum_mu1_hi =
                    _mm256_add_epi32(accum_mu1_hi, _mm256_madd_epi16(s44, fh1));

                accum_mu1_lo = _mm256_add_epi32(accum_mu1_lo, add_shift_HP_vex);
                accum_mu1_lo = _mm256_srli_epi32(accum_mu1_lo, 0x10);
                accum_mu1_hi = _mm256_add_epi32(accum_mu1_hi, add_shift_HP_vex);
                accum_mu1_hi = _mm256_srli_epi32(accum_mu1_hi, 0x10);

                accum_mu1_hi = _mm256_packus_epi32(accum_mu1_lo, accum_mu1_hi);
                accum_mu1_hi = _mm256_permute4x64_epi64(accum_mu1_hi, 0xD8);
                _mm256_storeu_si256(
                    (__m256i *)(dst->band_d + i * dst_stride + j),
                    accum_mu1_hi);
            }
        }
    }
}
