/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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
#include "float_adm_avx2.h"
#include "mem.h"

static const int FLOAT_ABS_MASK_I = 0x7FFFFFFF;

static const float dwt2_db2_coeffs_lo[4] = {
     0.482962913144690f,
     0.836516303737469f,
     0.224143868041857f,
    -0.129409522550921f
};

static const float dwt2_db2_coeffs_hi[4] = {
    -0.129409522550921f,
    -0.224143868041857f,
     0.836516303737469f,
    -0.482962913144690f
};

void float_adm_dwt2_avx2(const float *src, const adm_dwt_band_t_s *dst,
                          int **ind_y, int **ind_x,
                          int w, int h, int src_stride, int dst_stride)
{
    const float *filter_lo = dwt2_db2_coeffs_lo;
    const float *filter_hi = dwt2_db2_coeffs_hi;

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float *tmplo = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);
    float *tmphi = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);

    /* Broadcast filter coefficients for vertical pass. */
    __m256 vlo0 = _mm256_set1_ps(filter_lo[0]);
    __m256 vlo1 = _mm256_set1_ps(filter_lo[1]);
    __m256 vlo2 = _mm256_set1_ps(filter_lo[2]);
    __m256 vlo3 = _mm256_set1_ps(filter_lo[3]);

    __m256 vhi0 = _mm256_set1_ps(filter_hi[0]);
    __m256 vhi1 = _mm256_set1_ps(filter_hi[1]);
    __m256 vhi2 = _mm256_set1_ps(filter_hi[2]);
    __m256 vhi3 = _mm256_set1_ps(filter_hi[3]);

    for (int i = 0; i < (h + 1) / 2; ++i) {

        const float *row0 = src + ind_y[0][i] * src_px_stride;
        const float *row1 = src + ind_y[1][i] * src_px_stride;
        const float *row2 = src + ind_y[2][i] * src_px_stride;
        const float *row3 = src + ind_y[3][i] * src_px_stride;

        /* Vertical pass: process 8 columns at a time with AVX2. */
        int j = 0;
        for (; j + 8 <= w; j += 8) {
            __m256 s0 = _mm256_loadu_ps(row0 + j);
            __m256 s1 = _mm256_loadu_ps(row1 + j);
            __m256 s2 = _mm256_loadu_ps(row2 + j);
            __m256 s3 = _mm256_loadu_ps(row3 + j);

            /* Low-pass vertical. */
            __m256 lo_acc = _mm256_mul_ps(s0, vlo0);
            lo_acc = _mm256_add_ps(lo_acc, _mm256_mul_ps(s1, vlo1));
            lo_acc = _mm256_add_ps(lo_acc, _mm256_mul_ps(s2, vlo2));
            lo_acc = _mm256_add_ps(lo_acc, _mm256_mul_ps(s3, vlo3));
            _mm256_storeu_ps(tmplo + j, lo_acc);

            /* High-pass vertical. */
            __m256 hi_acc = _mm256_mul_ps(s0, vhi0);
            hi_acc = _mm256_add_ps(hi_acc, _mm256_mul_ps(s1, vhi1));
            hi_acc = _mm256_add_ps(hi_acc, _mm256_mul_ps(s2, vhi2));
            hi_acc = _mm256_add_ps(hi_acc, _mm256_mul_ps(s3, vhi3));
            _mm256_storeu_ps(tmphi + j, hi_acc);
        }

        /* Scalar tail for vertical pass. */
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

        /* Horizontal pass: scalar, since it uses indirect indexing via ind_x. */
        for (j = 0; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            float sl0 = tmplo[j0];
            float sl1 = tmplo[j1];
            float sl2 = tmplo[j2];
            float sl3 = tmplo[j3];

            /* band_a: lo vertical, lo horizontal. */
            dst->band_a[i * dst_px_stride + j] =
                filter_lo[0] * sl0 + filter_lo[1] * sl1 +
                filter_lo[2] * sl2 + filter_lo[3] * sl3;

            /* band_v: lo vertical, hi horizontal. */
            dst->band_v[i * dst_px_stride + j] =
                filter_hi[0] * sl0 + filter_hi[1] * sl1 +
                filter_hi[2] * sl2 + filter_hi[3] * sl3;

            float sh0 = tmphi[j0];
            float sh1 = tmphi[j1];
            float sh2 = tmphi[j2];
            float sh3 = tmphi[j3];

            /* band_h: hi vertical, lo horizontal. */
            dst->band_h[i * dst_px_stride + j] =
                filter_lo[0] * sh0 + filter_lo[1] * sh1 +
                filter_lo[2] * sh2 + filter_lo[3] * sh3;

            /* band_d: hi vertical, hi horizontal. */
            dst->band_d[i * dst_px_stride + j] =
                filter_hi[0] * sh0 + filter_hi[1] * sh1 +
                filter_hi[2] * sh2 + filter_hi[3] * sh3;
        }
    }

    aligned_free(tmplo);
    aligned_free(tmphi);
}

void float_adm_csf_avx2(const float *src, float *dst, float *flt,
                         int w, int h, int src_stride, int dst_stride,
                         float factor, float one_by_30)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    __m256 abs_mask = _mm256_broadcast_ss((const float *)&FLOAT_ABS_MASK_I);
    __m256 vfactor = _mm256_set1_ps(factor);
    __m256 vone_by_30 = _mm256_set1_ps(one_by_30);

    for (int i = 0; i < h; ++i) {
        int src_offset = i * src_px_stride;
        int dst_offset = i * dst_px_stride;
        int j = 0;

        /* Process 16 floats per iteration (dual accumulators for throughput). */
        for (; j + 16 <= w; j += 16) {
            __m256 s0 = _mm256_loadu_ps(src + src_offset + j);
            __m256 s1 = _mm256_loadu_ps(src + src_offset + j + 8);

            __m256 d0 = _mm256_mul_ps(vfactor, s0);
            __m256 d1 = _mm256_mul_ps(vfactor, s1);

            _mm256_storeu_ps(dst + dst_offset + j, d0);
            _mm256_storeu_ps(dst + dst_offset + j + 8, d1);

            __m256 a0 = _mm256_and_ps(d0, abs_mask);
            __m256 a1 = _mm256_and_ps(d1, abs_mask);

            __m256 f0 = _mm256_mul_ps(vone_by_30, a0);
            __m256 f1 = _mm256_mul_ps(vone_by_30, a1);

            _mm256_storeu_ps(flt + dst_offset + j, f0);
            _mm256_storeu_ps(flt + dst_offset + j + 8, f1);
        }

        /* Process 8 floats. */
        for (; j + 8 <= w; j += 8) {
            __m256 s = _mm256_loadu_ps(src + src_offset + j);
            __m256 d = _mm256_mul_ps(vfactor, s);
            _mm256_storeu_ps(dst + dst_offset + j, d);

            __m256 a = _mm256_and_ps(d, abs_mask);
            __m256 f = _mm256_mul_ps(vone_by_30, a);
            _mm256_storeu_ps(flt + dst_offset + j, f);
        }

        /* Scalar tail. */
        for (; j < w; ++j) {
            float dst_val = factor * src[src_offset + j];
            dst[dst_offset + j] = dst_val;
            flt[dst_offset + j] = one_by_30 * fabsf(dst_val);
        }
    }
}

float float_adm_csf_den_scale_avx2(const float *src, int w, int h,
                                    int src_stride, int left, int top,
                                    int right, int bottom, float factor)
{
    (void)w; (void)h;
    int src_px_stride = src_stride / sizeof(float);

    __m256 abs_mask = _mm256_broadcast_ss((const float *)&FLOAT_ABS_MASK_I);
    __m256 vfactor = _mm256_set1_ps(factor);

    double accum = 0;

    for (int i = top; i < bottom; ++i) {
        const float *row = src + i * src_px_stride;
        __m256d dsum0 = _mm256_setzero_pd();
        __m256d dsum1 = _mm256_setzero_pd();
        int j = left;

        /* Process 8 floats per iteration: cube in float, accumulate in double. */
        for (; j + 8 <= right; j += 8) {
            __m256 s = _mm256_loadu_ps(row + j);
            __m256 v = _mm256_and_ps(_mm256_mul_ps(vfactor, s), abs_mask);
            __m256 vsq = _mm256_mul_ps(v, v);
            __m256 vcube = _mm256_mul_ps(vsq, v);

            dsum0 = _mm256_add_pd(dsum0, _mm256_cvtps_pd(_mm256_castps256_ps128(vcube)));
            dsum1 = _mm256_add_pd(dsum1, _mm256_cvtps_pd(_mm256_extractf128_ps(vcube, 1)));
        }

        /* Horizontal reduction in double. */
        __m256d dtotal = _mm256_add_pd(dsum0, dsum1);
        __m128d dlo = _mm256_castpd256_pd128(dtotal);
        __m128d dhi = _mm256_extractf128_pd(dtotal, 1);
        __m128d ds = _mm_add_pd(dlo, dhi);
        ds = _mm_add_sd(ds, _mm_unpackhi_pd(ds, ds));
        double row_accum = _mm_cvtsd_f64(ds);

        /* Scalar tail. */
        for (; j < right; ++j) {
            float val = fabsf(factor * row[j]);
            row_accum += (double)(val * val * val);
        }

        accum += row_accum;
    }

    return (float)accum;
}

float float_adm_sum_cube_avx2(const float *x, int w, int h, int stride,
                               int left, int top, int right, int bottom)
{
    (void)w; (void)h;
    int px_stride = stride / sizeof(float);

    __m256 abs_mask = _mm256_broadcast_ss((const float *)&FLOAT_ABS_MASK_I);

    double accum = 0;

    for (int i = top; i < bottom; ++i) {
        const float *row = x + i * px_stride;
        __m256d dsum0 = _mm256_setzero_pd();
        __m256d dsum1 = _mm256_setzero_pd();
        int j = left;

        /* Process 8 floats per iteration: cube in float, accumulate in double. */
        for (; j + 8 <= right; j += 8) {
            __m256 v = _mm256_and_ps(_mm256_loadu_ps(row + j), abs_mask);
            __m256 vsq = _mm256_mul_ps(v, v);
            __m256 vcube = _mm256_mul_ps(vsq, v);

            dsum0 = _mm256_add_pd(dsum0, _mm256_cvtps_pd(_mm256_castps256_ps128(vcube)));
            dsum1 = _mm256_add_pd(dsum1, _mm256_cvtps_pd(_mm256_extractf128_ps(vcube, 1)));
        }

        /* Horizontal reduction in double. */
        __m256d dtotal = _mm256_add_pd(dsum0, dsum1);
        __m128d dlo = _mm256_castpd256_pd128(dtotal);
        __m128d dhi = _mm256_extractf128_pd(dtotal, 1);
        __m128d ds = _mm_add_pd(dlo, dhi);
        ds = _mm_add_sd(ds, _mm_unpackhi_pd(ds, ds));
        double row_accum = _mm_cvtsd_f64(ds);

        /* Scalar tail. */
        for (; j < right; ++j) {
            float val = fabsf(row[j]);
            row_accum += (double)(val * val * val);
        }

        accum += row_accum;
    }

    return (float)accum;
}
