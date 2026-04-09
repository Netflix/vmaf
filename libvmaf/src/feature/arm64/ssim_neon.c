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

#include <arm_neon.h>
#include <math.h>

#include "ssim_neon.h"

void ssim_precompute_neon(const float *ref, const float *cmp,
                           float *ref_sq, float *cmp_sq,
                           float *ref_cmp, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t r = vld1q_f32(ref + i);
        float32x4_t c = vld1q_f32(cmp + i);
        vst1q_f32(ref_sq + i, vmulq_f32(r, r));
        vst1q_f32(cmp_sq + i, vmulq_f32(c, c));
        vst1q_f32(ref_cmp + i, vmulq_f32(r, c));
    }
    for (; i < n; i++) {
        ref_sq[i] = ref[i] * ref[i];
        cmp_sq[i] = cmp[i] * cmp[i];
        ref_cmp[i] = ref[i] * cmp[i];
    }
}

void ssim_variance_neon(float *ref_sigma_sqd, float *cmp_sigma_sqd,
                         float *sigma_both, const float *ref_mu,
                         const float *cmp_mu, int n)
{
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t rm = vld1q_f32(ref_mu + i);
        float32x4_t cm = vld1q_f32(cmp_mu + i);

        float32x4_t rs = vld1q_f32(ref_sigma_sqd + i);
        rs = vsubq_f32(rs, vmulq_f32(rm, rm));
        rs = vmaxq_f32(rs, zero);
        vst1q_f32(ref_sigma_sqd + i, rs);

        float32x4_t cs = vld1q_f32(cmp_sigma_sqd + i);
        cs = vsubq_f32(cs, vmulq_f32(cm, cm));
        cs = vmaxq_f32(cs, zero);
        vst1q_f32(cmp_sigma_sqd + i, cs);

        float32x4_t sb = vld1q_f32(sigma_both + i);
        sb = vsubq_f32(sb, vmulq_f32(rm, cm));
        vst1q_f32(sigma_both + i, sb);
    }
    for (; i < n; i++) {
        ref_sigma_sqd[i] -= ref_mu[i] * ref_mu[i];
        if (ref_sigma_sqd[i] < 0.0f) ref_sigma_sqd[i] = 0.0f;
        cmp_sigma_sqd[i] -= cmp_mu[i] * cmp_mu[i];
        if (cmp_sigma_sqd[i] < 0.0f) cmp_sigma_sqd[i] = 0.0f;
        sigma_both[i] -= ref_mu[i] * cmp_mu[i];
    }
}

void ssim_accumulate_neon(const float *ref_mu, const float *cmp_mu,
                           const float *ref_sigma_sqd,
                           const float *cmp_sigma_sqd,
                           const float *sigma_both, int n,
                           float C1, float C2, float C3,
                           double *ssim_sum, double *l_sum,
                           double *c_sum, double *s_sum)
{
    float32x4_t vC1 = vdupq_n_f32(C1);
    float32x4_t vC2 = vdupq_n_f32(C2);
    float32x4_t vC3 = vdupq_n_f32(C3);
    float32x4_t v2  = vdupq_n_f32(2.0f);
    float32x4_t vzero = vdupq_n_f32(0.0f);

    float64x2_t dssim0 = vdupq_n_f64(0.0);
    float64x2_t dssim1 = vdupq_n_f64(0.0);
    float64x2_t dl0 = vdupq_n_f64(0.0);
    float64x2_t dl1 = vdupq_n_f64(0.0);
    float64x2_t dc0 = vdupq_n_f64(0.0);
    float64x2_t dc1 = vdupq_n_f64(0.0);
    float64x2_t ds0 = vdupq_n_f64(0.0);
    float64x2_t ds1 = vdupq_n_f64(0.0);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t rm = vld1q_f32(ref_mu + i);
        float32x4_t cm = vld1q_f32(cmp_mu + i);
        float32x4_t rs = vld1q_f32(ref_sigma_sqd + i);
        float32x4_t cs = vld1q_f32(cmp_sigma_sqd + i);
        float32x4_t sb = vld1q_f32(sigma_both + i);

        /* sqrt(rs * cs) */
        float32x4_t srsc = vsqrtq_f32(vmulq_f32(rs, cs));

        /* l = (2*rm*cm + C1) / (rm^2 + cm^2 + C1) */
        float32x4_t l_num = vaddq_f32(vmulq_f32(v2, vmulq_f32(rm, cm)), vC1);
        float32x4_t l_den = vaddq_f32(vaddq_f32(vmulq_f32(rm, rm),
                                                   vmulq_f32(cm, cm)), vC1);
        float32x4_t l = vdivq_f32(l_num, l_den);

        /* c = (2*srsc + C2) / (rs + cs + C2) */
        float32x4_t c_num = vaddq_f32(vmulq_f32(v2, srsc), vC2);
        float32x4_t c_den = vaddq_f32(vaddq_f32(rs, cs), vC2);
        float32x4_t c = vdivq_f32(c_num, c_den);

        /* Clamp sigma_both: if (sb < 0 && srsc <= 0) then 0 else sb */
        uint32x4_t sb_neg = vcltq_f32(sb, vzero);
        uint32x4_t srsc_le0 = vcleq_f32(srsc, vzero);
        uint32x4_t clamp_mask = vandq_u32(sb_neg, srsc_le0);
        float32x4_t clamped_sb = vbslq_f32(clamp_mask, vzero, sb);

        /* s = (clamped_sb + C3) / (srsc + C3) */
        float32x4_t s = vdivq_f32(vaddq_f32(clamped_sb, vC3),
                                    vaddq_f32(srsc, vC3));

        float32x4_t ssim_val = vmulq_f32(vmulq_f32(l, c), s);

        /* Accumulate to double: widen low 2 and high 2 floats */
        dssim0 = vaddq_f64(dssim0, vcvt_f64_f32(vget_low_f32(ssim_val)));
        dssim1 = vaddq_f64(dssim1, vcvt_f64_f32(vget_high_f32(ssim_val)));

        dl0 = vaddq_f64(dl0, vcvt_f64_f32(vget_low_f32(l)));
        dl1 = vaddq_f64(dl1, vcvt_f64_f32(vget_high_f32(l)));

        dc0 = vaddq_f64(dc0, vcvt_f64_f32(vget_low_f32(c)));
        dc1 = vaddq_f64(dc1, vcvt_f64_f32(vget_high_f32(c)));

        ds0 = vaddq_f64(ds0, vcvt_f64_f32(vget_low_f32(s)));
        ds1 = vaddq_f64(ds1, vcvt_f64_f32(vget_high_f32(s)));
    }

    /* Horizontal sum */
    float64x2_t tot_ssim = vaddq_f64(dssim0, dssim1);
    float64x2_t tot_l = vaddq_f64(dl0, dl1);
    float64x2_t tot_c = vaddq_f64(dc0, dc1);
    float64x2_t tot_s = vaddq_f64(ds0, ds1);

    double local_ssim = vaddvq_f64(tot_ssim);
    double local_l = vaddvq_f64(tot_l);
    double local_c = vaddvq_f64(tot_c);
    double local_s = vaddvq_f64(tot_s);

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
