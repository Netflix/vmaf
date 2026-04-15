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

#include <math.h>
#include <arm_neon.h>
#include "float_adm_neon.h"
#include "mem.h"

static const float dwt2_db2_coeffs_lo_s[4] = {
    0.482962913144690f, 0.836516303737469f,
    0.224143868041857f, -0.129409522550921f
};

static const float dwt2_db2_coeffs_hi_s[4] = {
    -0.129409522550921f, -0.224143868041857f,
    0.836516303737469f, -0.482962913144690f
};

void float_adm_dwt2_neon(const float *src, const adm_dwt_band_t_s *dst,
                          int **ind_y, int **ind_x,
                          int w, int h, int src_stride, int dst_stride)
{
    const float *filter_lo = dwt2_db2_coeffs_lo_s;
    const float *filter_hi = dwt2_db2_coeffs_hi_s;

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float *tmplo = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);
    float *tmphi = aligned_malloc(ALIGN_CEIL(sizeof(float) * w), MAX_ALIGN);

    /* Load filter coefficients into NEON registers. */
    const float32x4_t flo = vld1q_f32(filter_lo);
    const float32x4_t fhi = vld1q_f32(filter_hi);

    int i, j;

    for (i = 0; i < (h + 1) / 2; ++i) {
        const float *row0 = src + ind_y[0][i] * src_px_stride;
        const float *row1 = src + ind_y[1][i] * src_px_stride;
        const float *row2 = src + ind_y[2][i] * src_px_stride;
        const float *row3 = src + ind_y[3][i] * src_px_stride;

        /* Vertical pass: process 4 columns at a time with NEON. */
        j = 0;
        for (; j + 3 < w; j += 4) {
            float32x4_t s0 = vld1q_f32(row0 + j);
            float32x4_t s1 = vld1q_f32(row1 + j);
            float32x4_t s2 = vld1q_f32(row2 + j);
            float32x4_t s3 = vld1q_f32(row3 + j);

            /* lo = filter_lo[0]*s0 + filter_lo[1]*s1
             *    + filter_lo[2]*s2 + filter_lo[3]*s3 */
            float32x4_t lo = vmulq_laneq_f32(s0, flo, 0);
            lo = vmlaq_laneq_f32(lo, s1, flo, 1);
            lo = vmlaq_laneq_f32(lo, s2, flo, 2);
            lo = vmlaq_laneq_f32(lo, s3, flo, 3);
            vst1q_f32(tmplo + j, lo);

            /* hi = filter_hi[0]*s0 + filter_hi[1]*s1
             *    + filter_hi[2]*s2 + filter_hi[3]*s3 */
            float32x4_t hi = vmulq_laneq_f32(s0, fhi, 0);
            hi = vmlaq_laneq_f32(hi, s1, fhi, 1);
            hi = vmlaq_laneq_f32(hi, s2, fhi, 2);
            hi = vmlaq_laneq_f32(hi, s3, fhi, 3);
            vst1q_f32(tmphi + j, hi);
        }

        /* Scalar tail for remaining columns. */
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

        /* Horizontal pass (lo and hi): scalar due to indirect indexing. */
        for (j = 0; j < (w + 1) / 2; ++j) {
            int j0 = ind_x[0][j];
            int j1 = ind_x[1][j];
            int j2 = ind_x[2][j];
            int j3 = ind_x[3][j];

            float sl0 = tmplo[j0];
            float sl1 = tmplo[j1];
            float sl2 = tmplo[j2];
            float sl3 = tmplo[j3];

            float accum;
            int off = i * dst_px_stride + j;

            accum  = filter_lo[0] * sl0;
            accum += filter_lo[1] * sl1;
            accum += filter_lo[2] * sl2;
            accum += filter_lo[3] * sl3;
            dst->band_a[off] = accum;

            accum  = filter_hi[0] * sl0;
            accum += filter_hi[1] * sl1;
            accum += filter_hi[2] * sl2;
            accum += filter_hi[3] * sl3;
            dst->band_v[off] = accum;

            float sh0 = tmphi[j0];
            float sh1 = tmphi[j1];
            float sh2 = tmphi[j2];
            float sh3 = tmphi[j3];

            accum  = filter_lo[0] * sh0;
            accum += filter_lo[1] * sh1;
            accum += filter_lo[2] * sh2;
            accum += filter_lo[3] * sh3;
            dst->band_h[off] = accum;

            accum  = filter_hi[0] * sh0;
            accum += filter_hi[1] * sh1;
            accum += filter_hi[2] * sh2;
            accum += filter_hi[3] * sh3;
            dst->band_d[off] = accum;
        }
    }

    aligned_free(tmplo);
    aligned_free(tmphi);
}

void float_adm_csf_neon(const float *src, float *dst, float *flt,
                         int w, int h, int src_stride, int dst_stride,
                         float factor, float one_by_30)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float32x4_t v_factor = vdupq_n_f32(factor);
    float32x4_t v_one_by_30 = vdupq_n_f32(one_by_30);

    int i, j;

    for (i = 0; i < h; ++i) {
        int src_off = i * src_px_stride;
        int dst_off = i * dst_px_stride;

        j = 0;
        for (; j + 3 < w; j += 4) {
            float32x4_t s = vld1q_f32(src + src_off + j);
            float32x4_t dst_val = vmulq_f32(v_factor, s);
            vst1q_f32(dst + dst_off + j, dst_val);
            float32x4_t abs_dst = vabsq_f32(dst_val);
            float32x4_t flt_val = vmulq_f32(v_one_by_30, abs_dst);
            vst1q_f32(flt + dst_off + j, flt_val);
        }

        /* Scalar tail. */
        for (; j < w; ++j) {
            float dst_val = factor * src[src_off + j];
            dst[dst_off + j] = dst_val;
            flt[dst_off + j] = one_by_30 * fabsf(dst_val);
        }
    }
}

float float_adm_csf_den_scale_neon(const float *src, int w, int h,
                                    int src_stride, int left, int top,
                                    int right, int bottom, float factor)
{
    (void)w; (void)h;
    int src_px_stride = src_stride / sizeof(float);

    float32x4_t v_factor = vdupq_n_f32(factor);
    float32x4_t v_accum = vdupq_n_f32(0.0f);

    int i, j;
    float accum = 0.0f;

    for (i = top; i < bottom; ++i) {
        float32x4_t v_inner = vdupq_n_f32(0.0f);
        const float *row = src + i * src_px_stride;

        j = left;
        for (; j + 3 < right; j += 4) {
            float32x4_t s = vld1q_f32(row + j);
            float32x4_t val = vabsq_f32(vmulq_f32(v_factor, s));
            /* val^3 = val * val * val */
            float32x4_t val2 = vmulq_f32(val, val);
            float32x4_t val3 = vmulq_f32(val2, val);
            v_inner = vaddq_f32(v_inner, val3);
        }

        v_accum = vaddq_f32(v_accum, v_inner);

        /* Scalar tail. */
        for (; j < right; ++j) {
            float val = fabsf(factor * row[j]);
            accum += val * val * val;
        }
    }

    accum += vaddvq_f32(v_accum);

    return accum;
}

float float_adm_sum_cube_neon(const float *x, int w, int h, int stride,
                               int left, int top, int right, int bottom)
{
    (void)w; (void)h;
    int px_stride = stride / sizeof(float);

    float32x4_t v_accum = vdupq_n_f32(0.0f);
    float accum = 0.0f;

    int i, j;

    for (i = top; i < bottom; ++i) {
        float32x4_t v_inner = vdupq_n_f32(0.0f);
        const float *row = x + i * px_stride;

        j = left;
        for (; j + 3 < right; j += 4) {
            float32x4_t s = vld1q_f32(row + j);
            float32x4_t val = vabsq_f32(s);
            /* val^3 = val * val * val */
            float32x4_t val2 = vmulq_f32(val, val);
            float32x4_t val3 = vmulq_f32(val2, val);
            v_inner = vaddq_f32(v_inner, val3);
        }

        v_accum = vaddq_f32(v_accum, v_inner);

        /* Scalar tail. */
        for (; j < right; ++j) {
            float val = fabsf(row[j]);
            accum += val * val * val;
        }
    }

    accum += vaddvq_f32(v_accum);

    return accum;
}
