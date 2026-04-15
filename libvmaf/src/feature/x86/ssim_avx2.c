/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include "ssim_avx2.h"

void ssim_precompute_avx2(const float *ref, const float *cmp,
                           float *ref_sq, float *cmp_sq,
                           float *ref_cmp, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 r = _mm256_loadu_ps(ref + i);
        __m256 c = _mm256_loadu_ps(cmp + i);
        _mm256_storeu_ps(ref_sq + i, _mm256_mul_ps(r, r));
        _mm256_storeu_ps(cmp_sq + i, _mm256_mul_ps(c, c));
        _mm256_storeu_ps(ref_cmp + i, _mm256_mul_ps(r, c));
    }
    for (; i < n; i++) {
        ref_sq[i] = ref[i] * ref[i];
        cmp_sq[i] = cmp[i] * cmp[i];
        ref_cmp[i] = ref[i] * cmp[i];
    }
}

void ssim_variance_avx2(float *ref_sigma_sqd, float *cmp_sigma_sqd,
                         float *sigma_both, const float *ref_mu,
                         const float *cmp_mu, int n)
{
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 rm = _mm256_loadu_ps(ref_mu + i);
        __m256 cm = _mm256_loadu_ps(cmp_mu + i);

        __m256 rs = _mm256_loadu_ps(ref_sigma_sqd + i);
        rs = _mm256_sub_ps(rs, _mm256_mul_ps(rm, rm));
        rs = _mm256_max_ps(rs, zero);
        _mm256_storeu_ps(ref_sigma_sqd + i, rs);

        __m256 cs = _mm256_loadu_ps(cmp_sigma_sqd + i);
        cs = _mm256_sub_ps(cs, _mm256_mul_ps(cm, cm));
        cs = _mm256_max_ps(cs, zero);
        _mm256_storeu_ps(cmp_sigma_sqd + i, cs);

        __m256 sb = _mm256_loadu_ps(sigma_both + i);
        sb = _mm256_sub_ps(sb, _mm256_mul_ps(rm, cm));
        _mm256_storeu_ps(sigma_both + i, sb);
    }
    for (; i < n; i++) {
        ref_sigma_sqd[i] -= ref_mu[i] * ref_mu[i];
        if (ref_sigma_sqd[i] < 0.0f) ref_sigma_sqd[i] = 0.0f;
        cmp_sigma_sqd[i] -= cmp_mu[i] * cmp_mu[i];
        if (cmp_sigma_sqd[i] < 0.0f) cmp_sigma_sqd[i] = 0.0f;
        sigma_both[i] -= ref_mu[i] * cmp_mu[i];
    }
}

void ssim_accumulate_avx2(const float *ref_mu, const float *cmp_mu,
                           const float *ref_sigma_sqd,
                           const float *cmp_sigma_sqd,
                           const float *sigma_both, int n,
                           float C1, float C2, float C3,
                           double *ssim_sum, double *l_sum,
                           double *c_sum, double *s_sum)
{
    __m256 vC1 = _mm256_set1_ps(C1);
    __m256 vC2 = _mm256_set1_ps(C2);
    __m256 vC3 = _mm256_set1_ps(C3);
    __m256 v2  = _mm256_set1_ps(2.0f);
    __m256 vzero = _mm256_setzero_ps();

    __m256d dssim0 = _mm256_setzero_pd();
    __m256d dssim1 = _mm256_setzero_pd();
    __m256d dl0 = _mm256_setzero_pd();
    __m256d dl1 = _mm256_setzero_pd();
    __m256d dc0 = _mm256_setzero_pd();
    __m256d dc1 = _mm256_setzero_pd();
    __m256d ds0 = _mm256_setzero_pd();
    __m256d ds1 = _mm256_setzero_pd();

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 rm = _mm256_loadu_ps(ref_mu + i);
        __m256 cm = _mm256_loadu_ps(cmp_mu + i);
        __m256 rs = _mm256_loadu_ps(ref_sigma_sqd + i);
        __m256 cs = _mm256_loadu_ps(cmp_sigma_sqd + i);
        __m256 sb = _mm256_loadu_ps(sigma_both + i);

        /* sigma_ref_sigma_cmp = sqrt(rs * cs) */
        __m256 srsc = _mm256_sqrt_ps(_mm256_mul_ps(rs, cs));

        /* l = (2*rm*cm + C1) / (rm^2 + cm^2 + C1) */
        __m256 l_num = _mm256_add_ps(_mm256_mul_ps(v2, _mm256_mul_ps(rm, cm)), vC1);
        __m256 l_den = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rm, rm),
                                                     _mm256_mul_ps(cm, cm)), vC1);
        __m256 l = _mm256_div_ps(l_num, l_den);

        /* c = (2*srsc + C2) / (rs + cs + C2) */
        __m256 c_num = _mm256_add_ps(_mm256_mul_ps(v2, srsc), vC2);
        __m256 c_den = _mm256_add_ps(_mm256_add_ps(rs, cs), vC2);
        __m256 c = _mm256_div_ps(c_num, c_den);

        /* Clamp sigma_both: if (sb < 0 && srsc <= 0) then 0 else sb */
        __m256 sb_neg = _mm256_cmp_ps(sb, vzero, _CMP_LT_OQ);
        __m256 srsc_le0 = _mm256_cmp_ps(srsc, vzero, _CMP_LE_OQ);
        __m256 clamp_mask = _mm256_and_ps(sb_neg, srsc_le0);
        __m256 clamped_sb = _mm256_blendv_ps(sb, vzero, clamp_mask);

        /* s = (clamped_sb + C3) / (srsc + C3) */
        __m256 s_num = _mm256_add_ps(clamped_sb, vC3);
        __m256 s_den = _mm256_add_ps(srsc, vC3);
        __m256 s = _mm256_div_ps(s_num, s_den);

        /* ssim = l * c * s */
        __m256 ssim_val = _mm256_mul_ps(_mm256_mul_ps(l, c), s);

        /* Accumulate to double: convert low 4 and high 4 floats to double */
        __m128 lo4_ssim = _mm256_castps256_ps128(ssim_val);
        __m128 hi4_ssim = _mm256_extractf128_ps(ssim_val, 1);
        dssim0 = _mm256_add_pd(dssim0, _mm256_cvtps_pd(lo4_ssim));
        dssim1 = _mm256_add_pd(dssim1, _mm256_cvtps_pd(hi4_ssim));

        __m128 lo4_l = _mm256_castps256_ps128(l);
        __m128 hi4_l = _mm256_extractf128_ps(l, 1);
        dl0 = _mm256_add_pd(dl0, _mm256_cvtps_pd(lo4_l));
        dl1 = _mm256_add_pd(dl1, _mm256_cvtps_pd(hi4_l));

        __m128 lo4_c = _mm256_castps256_ps128(c);
        __m128 hi4_c = _mm256_extractf128_ps(c, 1);
        dc0 = _mm256_add_pd(dc0, _mm256_cvtps_pd(lo4_c));
        dc1 = _mm256_add_pd(dc1, _mm256_cvtps_pd(hi4_c));

        __m128 lo4_s = _mm256_castps256_ps128(s);
        __m128 hi4_s = _mm256_extractf128_ps(s, 1);
        ds0 = _mm256_add_pd(ds0, _mm256_cvtps_pd(lo4_s));
        ds1 = _mm256_add_pd(ds1, _mm256_cvtps_pd(hi4_s));
    }

    /* Horizontal sum of double accumulators */
    __m256d tot_ssim = _mm256_add_pd(dssim0, dssim1);
    __m256d tot_l = _mm256_add_pd(dl0, dl1);
    __m256d tot_c = _mm256_add_pd(dc0, dc1);
    __m256d tot_s = _mm256_add_pd(ds0, ds1);

    /* 4-wide → scalar: hadd then extract */
    __m128d ss_lo = _mm256_castpd256_pd128(tot_ssim);
    __m128d ss_hi = _mm256_extractf128_pd(tot_ssim, 1);
    __m128d ss2 = _mm_add_pd(ss_lo, ss_hi);
    ss2 = _mm_add_pd(ss2, _mm_shuffle_pd(ss2, ss2, 1));
    double local_ssim;
    _mm_store_sd(&local_ssim, ss2);

    __m128d l_lo = _mm256_castpd256_pd128(tot_l);
    __m128d l_hi = _mm256_extractf128_pd(tot_l, 1);
    __m128d l2 = _mm_add_pd(l_lo, l_hi);
    l2 = _mm_add_pd(l2, _mm_shuffle_pd(l2, l2, 1));
    double local_l;
    _mm_store_sd(&local_l, l2);

    __m128d c_lo = _mm256_castpd256_pd128(tot_c);
    __m128d c_hi = _mm256_extractf128_pd(tot_c, 1);
    __m128d c2 = _mm_add_pd(c_lo, c_hi);
    c2 = _mm_add_pd(c2, _mm_shuffle_pd(c2, c2, 1));
    double local_c;
    _mm_store_sd(&local_c, c2);

    __m128d s_lo = _mm256_castpd256_pd128(tot_s);
    __m128d s_hi = _mm256_extractf128_pd(tot_s, 1);
    __m128d s2 = _mm_add_pd(s_lo, s_hi);
    s2 = _mm_add_pd(s2, _mm_shuffle_pd(s2, s2, 1));
    double local_s;
    _mm_store_sd(&local_s, s2);

    /* Scalar tail */
    for (; i < n; i++) {
        float srsc = sqrtf(ref_sigma_sqd[i] * cmp_sigma_sqd[i]);
        double lv = (2.0 * ref_mu[i] * cmp_mu[i] + C1) /
                    (ref_mu[i]*ref_mu[i] + cmp_mu[i]*cmp_mu[i] + C1);
        double cv = (2.0 * srsc + C2) /
                    (ref_sigma_sqd[i] + cmp_sigma_sqd[i] + C2);
        float csb = (sigma_both[i] < 0.0f && srsc <= 0.0f) ?
                    0.0f : sigma_both[i];
        double sv = (csb + C3) / (srsc + C3);
        local_ssim += lv * cv * sv;
        local_l += lv;
        local_c += cv;
        local_s += sv;
    }

    *ssim_sum += local_ssim;
    *l_sum += local_l;
    *c_sum += local_c;
    *s_sum += local_s;
}
