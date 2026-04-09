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
#include <math.h>
#include <string.h>

#include "feature/adm_tools.h"
#include "mem.h"
#include "float_adm_avx512.h"

static inline __m512 avx512_abs_ps(__m512 v)
{
    const __m512i mask = _mm512_set1_epi32(0x7FFFFFFF);
    return _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(v), mask));
}

void float_adm_dwt2_avx512(const float *src, const adm_dwt_band_t_s *dst,
                            int **ind_y, int **ind_x, int w, int h,
                            int src_stride, int dst_stride)
{
    const float filter_lo[4] = {
        0.482962913144690f, 0.836516303737469f,
        0.224143868041857f, -0.129409522550921f
    };
    const float filter_hi[4] = {
        -0.129409522550921f, -0.224143868041857f,
        0.836516303737469f, -0.482962913144690f
    };

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float *tmplo = aligned_malloc(ALIGN_CEIL(sizeof(float) * (w + 32)), MAX_ALIGN);
    float *tmphi = aligned_malloc(ALIGN_CEIL(sizeof(float) * (w + 32)), MAX_ALIGN);
    if (!tmplo || !tmphi) {
        aligned_free(tmplo); aligned_free(tmphi);
        return;
    }
    // Zero guard zone to avoid garbage reads at right boundary
    memset(tmplo + w, 0, 32 * sizeof(float));
    memset(tmphi + w, 0, 32 * sizeof(float));

    __m512 fl0 = _mm512_set1_ps(filter_lo[0]);
    __m512 fl1 = _mm512_set1_ps(filter_lo[1]);
    __m512 fl2 = _mm512_set1_ps(filter_lo[2]);
    __m512 fl3 = _mm512_set1_ps(filter_lo[3]);

    __m512 fh0 = _mm512_set1_ps(filter_hi[0]);
    __m512 fh1 = _mm512_set1_ps(filter_hi[1]);
    __m512 fh2 = _mm512_set1_ps(filter_hi[2]);
    __m512 fh3 = _mm512_set1_ps(filter_hi[3]);

    int i, j;

    for (i = 0; i < (h + 1) / 2; ++i) {
        const float *row0 = src + ind_y[0][i] * src_px_stride;
        const float *row1 = src + ind_y[1][i] * src_px_stride;
        const float *row2 = src + ind_y[2][i] * src_px_stride;
        const float *row3 = src + ind_y[3][i] * src_px_stride;

        /* Vertical pass: process 16 floats per iteration with AVX-512 */
        for (j = 0; j + 16 <= w; j += 16) {
            __m512 s0 = _mm512_loadu_ps(row0 + j);
            __m512 s1 = _mm512_loadu_ps(row1 + j);
            __m512 s2 = _mm512_loadu_ps(row2 + j);
            __m512 s3 = _mm512_loadu_ps(row3 + j);

            /* Low-pass filter: lo[0]*s0 + lo[1]*s1 + lo[2]*s2 + lo[3]*s3 */
            __m512 lo_acc = _mm512_mul_ps(fl0, s0);
            lo_acc = _mm512_fmadd_ps(fl1, s1, lo_acc);
            lo_acc = _mm512_fmadd_ps(fl2, s2, lo_acc);
            lo_acc = _mm512_fmadd_ps(fl3, s3, lo_acc);
            _mm512_storeu_ps(tmplo + j, lo_acc);

            /* High-pass filter: hi[0]*s0 + hi[1]*s1 + hi[2]*s2 + hi[3]*s3 */
            __m512 hi_acc = _mm512_mul_ps(fh0, s0);
            hi_acc = _mm512_fmadd_ps(fh1, s1, hi_acc);
            hi_acc = _mm512_fmadd_ps(fh2, s2, hi_acc);
            hi_acc = _mm512_fmadd_ps(fh3, s3, hi_acc);
            _mm512_storeu_ps(tmphi + j, hi_acc);
        }

        /* Scalar tail for vertical pass */
        for (; j < w; ++j) {
            float s0 = row0[j];
            float s1 = row1[j];
            float s2 = row2[j];
            float s3 = row3[j];

            tmplo[j] = filter_lo[0] * s0 + filter_lo[1] * s1 +
                        filter_lo[2] * s2 + filter_lo[3] * s3;
            tmphi[j] = filter_hi[0] * s0 + filter_hi[1] * s1 +
                        filter_hi[2] * s2 + filter_hi[3] * s3;
        }

        /* Horizontal pass: vectorized interior with deinterleave,
         * scalar for left boundary (j=0 reflection) and right tail. */
        {
            const int half_w = (w + 1) / 2;

            /* Permutation indices for even/odd deinterleaving across 2 zmm regs */
            const __m512i idx_even = _mm512_set_epi32(
                30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0);
            const __m512i idx_odd  = _mm512_set_epi32(
                31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1);

            /* Scalar: j=0 (left boundary has reflection) */
            {
                int j0 = ind_x[0][0], j1 = ind_x[1][0];
                int j2 = ind_x[2][0], j3 = ind_x[3][0];
                float sl0 = tmplo[j0], sl1 = tmplo[j1];
                float sl2 = tmplo[j2], sl3 = tmplo[j3];

                dst->band_a[i * dst_px_stride] =
                    filter_lo[0]*sl0 + filter_lo[1]*sl1 +
                    filter_lo[2]*sl2 + filter_lo[3]*sl3;
                dst->band_v[i * dst_px_stride] =
                    filter_hi[0]*sl0 + filter_hi[1]*sl1 +
                    filter_hi[2]*sl2 + filter_hi[3]*sl3;

                float sh0 = tmphi[j0], sh1 = tmphi[j1];
                float sh2 = tmphi[j2], sh3 = tmphi[j3];

                dst->band_h[i * dst_px_stride] =
                    filter_lo[0]*sh0 + filter_lo[1]*sh1 +
                    filter_lo[2]*sh2 + filter_lo[3]*sh3;
                dst->band_d[i * dst_px_stride] =
                    filter_hi[0]*sh0 + filter_hi[1]*sh1 +
                    filter_hi[2]*sh2 + filter_hi[3]*sh3;
            }

            /* AVX-512 vectorized interior: 16 outputs per iteration.
             * For j>=1, tap indices are regular stride-2:
             *   tap0 = tmplo[2j-1], tap1 = tmplo[2j], tap2 = tmplo[2j+1], tap3 = tmplo[2j+2]
             * Deinterleave even/odd from 32-element load pairs. */
            j = 1;
            for (; j + 16 <= half_w && 2 * j + 32 < w; j += 16) {
                /* 4 loads from tmplo covering indices [2j-1 .. 2j+32] */
                __m512 A = _mm512_loadu_ps(tmplo + 2*j - 1);
                __m512 B = _mm512_loadu_ps(tmplo + 2*j - 1 + 16);
                __m512 tap0 = _mm512_permutex2var_ps(A, idx_even, B);
                __m512 tap1 = _mm512_permutex2var_ps(A, idx_odd,  B);

                __m512 C = _mm512_loadu_ps(tmplo + 2*j + 1);
                __m512 D = _mm512_loadu_ps(tmplo + 2*j + 1 + 16);
                __m512 tap2 = _mm512_permutex2var_ps(C, idx_even, D);
                __m512 tap3 = _mm512_permutex2var_ps(C, idx_odd,  D);

                __m512 ba = _mm512_mul_ps(fl0, tap0);
                ba = _mm512_fmadd_ps(fl1, tap1, ba);
                ba = _mm512_fmadd_ps(fl2, tap2, ba);
                ba = _mm512_fmadd_ps(fl3, tap3, ba);
                _mm512_storeu_ps(dst->band_a + i * dst_px_stride + j, ba);

                __m512 bv = _mm512_mul_ps(fh0, tap0);
                bv = _mm512_fmadd_ps(fh1, tap1, bv);
                bv = _mm512_fmadd_ps(fh2, tap2, bv);
                bv = _mm512_fmadd_ps(fh3, tap3, bv);
                _mm512_storeu_ps(dst->band_v + i * dst_px_stride + j, bv);

                /* Same deinterleave from tmphi → band_h, band_d */
                A = _mm512_loadu_ps(tmphi + 2*j - 1);
                B = _mm512_loadu_ps(tmphi + 2*j - 1 + 16);
                tap0 = _mm512_permutex2var_ps(A, idx_even, B);
                tap1 = _mm512_permutex2var_ps(A, idx_odd,  B);

                C = _mm512_loadu_ps(tmphi + 2*j + 1);
                D = _mm512_loadu_ps(tmphi + 2*j + 1 + 16);
                tap2 = _mm512_permutex2var_ps(C, idx_even, D);
                tap3 = _mm512_permutex2var_ps(C, idx_odd,  D);

                __m512 bh = _mm512_mul_ps(fl0, tap0);
                bh = _mm512_fmadd_ps(fl1, tap1, bh);
                bh = _mm512_fmadd_ps(fl2, tap2, bh);
                bh = _mm512_fmadd_ps(fl3, tap3, bh);
                _mm512_storeu_ps(dst->band_h + i * dst_px_stride + j, bh);

                __m512 bd = _mm512_mul_ps(fh0, tap0);
                bd = _mm512_fmadd_ps(fh1, tap1, bd);
                bd = _mm512_fmadd_ps(fh2, tap2, bd);
                bd = _mm512_fmadd_ps(fh3, tap3, bd);
                _mm512_storeu_ps(dst->band_d + i * dst_px_stride + j, bd);
            }

            /* Scalar tail: right boundary + remaining positions */
            for (; j < half_w; ++j) {
                int j0 = ind_x[0][j], j1 = ind_x[1][j];
                int j2 = ind_x[2][j], j3 = ind_x[3][j];
                float sl0 = tmplo[j0], sl1 = tmplo[j1];
                float sl2 = tmplo[j2], sl3 = tmplo[j3];

                dst->band_a[i * dst_px_stride + j] =
                    filter_lo[0]*sl0 + filter_lo[1]*sl1 +
                    filter_lo[2]*sl2 + filter_lo[3]*sl3;
                dst->band_v[i * dst_px_stride + j] =
                    filter_hi[0]*sl0 + filter_hi[1]*sl1 +
                    filter_hi[2]*sl2 + filter_hi[3]*sl3;

                float sh0 = tmphi[j0], sh1 = tmphi[j1];
                float sh2 = tmphi[j2], sh3 = tmphi[j3];

                dst->band_h[i * dst_px_stride + j] =
                    filter_lo[0]*sh0 + filter_lo[1]*sh1 +
                    filter_lo[2]*sh2 + filter_lo[3]*sh3;
                dst->band_d[i * dst_px_stride + j] =
                    filter_hi[0]*sh0 + filter_hi[1]*sh1 +
                    filter_hi[2]*sh2 + filter_hi[3]*sh3;
            }
        }
    }

    aligned_free(tmplo);
    aligned_free(tmphi);
}

void float_adm_csf_avx512(const float *src, float *dst, float *flt,
                           int w, int h, int src_stride, int dst_stride,
                           float factor, float one_by_30)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    __m512 vfactor = _mm512_set1_ps(factor);
    __m512 vone_by_30 = _mm512_set1_ps(one_by_30);

    int i, j;

    for (i = 0; i < h; ++i) {
        const float *src_row = src + i * src_px_stride;
        float *dst_row = dst + i * dst_px_stride;
        float *flt_row = flt + i * dst_px_stride;

        for (j = 0; j + 16 <= w; j += 16) {
            __m512 sv = _mm512_loadu_ps(src_row + j);
            __m512 dst_val = _mm512_mul_ps(vfactor, sv);
            _mm512_storeu_ps(dst_row + j, dst_val);
            __m512 abs_dst = avx512_abs_ps(dst_val);
            __m512 flt_val = _mm512_mul_ps(vone_by_30, abs_dst);
            _mm512_storeu_ps(flt_row + j, flt_val);
        }

        /* Scalar tail */
        for (; j < w; ++j) {
            float dst_val = factor * src_row[j];
            dst_row[j] = dst_val;
            flt_row[j] = one_by_30 * fabsf(dst_val);
        }
    }
}

float float_adm_csf_den_scale_avx512(const float *src, int w, int h,
                                      int src_stride, int left, int top,
                                      int right, int bottom, float factor)
{
    (void)w; (void)h;
    int src_px_stride = src_stride / sizeof(float);

    __m512 vfactor = _mm512_set1_ps(factor);
    __m512 accum_vec = _mm512_setzero_ps();

    float accum = 0.0f;
    int i, j;

    for (i = top; i < bottom; ++i) {
        const float *row = src + i * src_px_stride;
        __m512 accum_inner = _mm512_setzero_ps();

        for (j = left; j + 16 <= right; j += 16) {
            __m512 sv = _mm512_loadu_ps(row + j);
            __m512 val = avx512_abs_ps(_mm512_mul_ps(vfactor, sv));
            __m512 val2 = _mm512_mul_ps(val, val);
            __m512 val3 = _mm512_mul_ps(val2, val);
            accum_inner = _mm512_add_ps(accum_inner, val3);
        }

        accum_vec = _mm512_add_ps(accum_vec, accum_inner);

        /* Scalar tail */
        for (; j < right; ++j) {
            float val = fabsf(factor * row[j]);
            accum += val * val * val;
        }
    }

    accum += _mm512_reduce_add_ps(accum_vec);

    return accum;
}

float float_adm_sum_cube_avx512(const float *x, int w, int h, int stride,
                                int left, int top, int right, int bottom)
{
    (void)w; (void)h;
    int px_stride = stride / sizeof(float);

    __m512 accum_vec = _mm512_setzero_ps();
    float accum = 0.0f;
    int i, j;

    for (i = top; i < bottom; ++i) {
        const float *row = x + i * px_stride;
        __m512 accum_inner = _mm512_setzero_ps();

        for (j = left; j + 16 <= right; j += 16) {
            __m512 val = avx512_abs_ps(_mm512_loadu_ps(row + j));
            __m512 val2 = _mm512_mul_ps(val, val);
            __m512 val3 = _mm512_mul_ps(val2, val);
            accum_inner = _mm512_add_ps(accum_inner, val3);
        }

        accum_vec = _mm512_add_ps(accum_vec, accum_inner);

        /* Scalar tail */
        for (; j < right; ++j) {
            float val = fabsf(row[j]);
            accum += val * val * val;
        }
    }

    accum += _mm512_reduce_add_ps(accum_vec);

    return accum;
}
