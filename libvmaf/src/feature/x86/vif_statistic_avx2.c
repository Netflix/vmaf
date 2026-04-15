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

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/**
 * Vectorized log2f approximation for 8 floats.
 * Uses IEEE 754 exponent extraction + degree-8 minimax polynomial.
 * Coefficients match vif_tools.c log2_poly_s[] / log2f_approx() for consistency.
 */
static inline __m256 log2_ps_avx2(__m256 x)
{
    __m256i xi = _mm256_castps_si256(x);

    // Extract exponent: ((bits >> 23) & 0xFF) - 127
    __m256i exp_i = _mm256_sub_epi32(
        _mm256_and_si256(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(0xFF)),
        _mm256_set1_epi32(127));
    __m256 e = _mm256_cvtepi32_ps(exp_i);

    // Normalize mantissa to [1, 2): clear exponent, set to 127
    __m256i mantissa_i = _mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)),
        _mm256_set1_epi32(0x3F800000));
    __m256 m = _mm256_castsi256_ps(mantissa_i);

    // t = m - 1, t in [0, 1)
    __m256 t = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));

    // Degree-8 minimax polynomial for log2(1+t) on [0, 1)
    // Coefficients from vif_tools.c log2_poly_s[], Horner evaluation
    __m256 p = _mm256_set1_ps(-0.012671635276421f);
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps( 0.064841182402670f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(-0.157048836463065f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps( 0.257167726303123f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(-0.353800560300520f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps( 0.480131410397451f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(-0.721314327952201f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps( 1.442694803896991f));
    p = _mm256_mul_ps(p, t);  // last coefficient is 0, equivalent to * t

    return _mm256_add_ps(e, p);
}

void vif_statistic_s_avx2(
    const float *mu1, const float *mu2,
    const float *xx_filt, const float *yy_filt, const float *xy_filt,
    float *num, float *den,
    int w, int h,
    int mu1_stride, int mu2_stride,
    int xx_filt_stride, int yy_filt_stride, int xy_filt_stride,
    double vif_enhn_gain_limit)
{
    const float sigma_nsq = 2.0f;
    const float sigma_max_inv = 4.0f / (255.0f * 255.0f);
    const float eps = 1.0e-10f;
    const float vif_egl_f = (float)vif_enhn_gain_limit;

    int mu1_px_stride = mu1_stride / sizeof(float);
    int mu2_px_stride = mu2_stride / sizeof(float);
    int xx_filt_px_stride = xx_filt_stride / sizeof(float);
    int yy_filt_px_stride = yy_filt_stride / sizeof(float);
    int xy_filt_px_stride = xy_filt_stride / sizeof(float);

    __m256 v_sigma_nsq = _mm256_set1_ps(sigma_nsq);
    __m256 v_sigma_max_inv = _mm256_set1_ps(sigma_max_inv);
    __m256 v_eps = _mm256_set1_ps(eps);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_egl = _mm256_set1_ps(vif_egl_f);

    float accum_num = 0.0f;
    float accum_den = 0.0f;

    for (int i = 0; i < h; ++i) {
        const float *mu1_row = mu1 + i * mu1_px_stride;
        const float *mu2_row = mu2 + i * mu2_px_stride;
        const float *xx_row = xx_filt + i * xx_filt_px_stride;
        const float *yy_row = yy_filt + i * yy_filt_px_stride;
        const float *xy_row = xy_filt + i * xy_filt_px_stride;

        float accum_inner_num = 0.0f;
        float accum_inner_den = 0.0f;

        int j = 0;
        int w8 = w & ~7;

        for (; j < w8; j += 8) {
            // Load 8 floats from each array
            __m256 v_mu1 = _mm256_loadu_ps(mu1_row + j);
            __m256 v_mu2 = _mm256_loadu_ps(mu2_row + j);
            __m256 v_xx = _mm256_loadu_ps(xx_row + j);
            __m256 v_yy = _mm256_loadu_ps(yy_row + j);
            __m256 v_xy = _mm256_loadu_ps(xy_row + j);

            // mu1_sq, mu2_sq, mu1_mu2
            __m256 v_mu1_sq = _mm256_mul_ps(v_mu1, v_mu1);
            __m256 v_mu2_sq = _mm256_mul_ps(v_mu2, v_mu2);
            __m256 v_mu1_mu2 = _mm256_mul_ps(v_mu1, v_mu2);

            // sigma1_sq, sigma2_sq, sigma12
            __m256 v_sigma1_sq = _mm256_sub_ps(v_xx, v_mu1_sq);
            __m256 v_sigma2_sq = _mm256_sub_ps(v_yy, v_mu2_sq);
            __m256 v_sigma12 = _mm256_sub_ps(v_xy, v_mu1_mu2);

            // sigma1_sq = max(sigma1_sq, 0), sigma2_sq = max(sigma2_sq, 0)
            v_sigma1_sq = _mm256_max_ps(v_sigma1_sq, v_zero);
            v_sigma2_sq = _mm256_max_ps(v_sigma2_sq, v_zero);

            // g = sigma12 / (sigma1_sq + eps)
            __m256 v_g = _mm256_div_ps(v_sigma12,
                _mm256_add_ps(v_sigma1_sq, v_eps));

            // sv_sq = sigma2_sq - g * sigma12
            __m256 v_sv_sq = _mm256_sub_ps(v_sigma2_sq,
                _mm256_mul_ps(v_g, v_sigma12));

            // if (sigma1_sq < eps): g = 0, sv_sq = sigma2_sq, sigma1_sq = 0
            __m256 mask_s1_lt_eps = _mm256_cmp_ps(v_sigma1_sq, v_eps, _CMP_LT_OS);
            v_g = _mm256_andnot_ps(mask_s1_lt_eps, v_g);
            v_sv_sq = _mm256_blendv_ps(v_sv_sq, v_sigma2_sq, mask_s1_lt_eps);
            v_sigma1_sq = _mm256_andnot_ps(mask_s1_lt_eps, v_sigma1_sq);

            // if (sigma2_sq < eps): g = 0, sv_sq = 0
            __m256 mask_s2_lt_eps = _mm256_cmp_ps(v_sigma2_sq, v_eps, _CMP_LT_OS);
            v_g = _mm256_andnot_ps(mask_s2_lt_eps, v_g);
            v_sv_sq = _mm256_andnot_ps(mask_s2_lt_eps, v_sv_sq);

            // if (g < 0): sv_sq = sigma2_sq, g = 0
            __m256 mask_g_lt_0 = _mm256_cmp_ps(v_g, v_zero, _CMP_LT_OS);
            v_sv_sq = _mm256_blendv_ps(v_sv_sq, v_sigma2_sq, mask_g_lt_0);
            v_g = _mm256_max_ps(v_g, v_zero);

            // sv_sq = max(sv_sq, eps)
            v_sv_sq = _mm256_max_ps(v_sv_sq, v_eps);

            // g = min(g, vif_enhn_gain_limit)
            v_g = _mm256_min_ps(v_g, v_egl);

            // num_val = log2f(1 + (g*g*sigma1_sq) / (sv_sq + sigma_nsq))
            __m256 v_g_sq = _mm256_mul_ps(v_g, v_g);
            __m256 v_num_arg = _mm256_add_ps(v_one,
                _mm256_div_ps(
                    _mm256_mul_ps(v_g_sq, v_sigma1_sq),
                    _mm256_add_ps(v_sv_sq, v_sigma_nsq)));
            __m256 v_num_val = log2_ps_avx2(v_num_arg);

            // den_val = log2f(1 + sigma1_sq / sigma_nsq)
            __m256 v_den_arg = _mm256_add_ps(v_one,
                _mm256_div_ps(v_sigma1_sq, v_sigma_nsq));
            __m256 v_den_val = log2_ps_avx2(v_den_arg);

            // if (sigma12 < 0): num_val = 0
            __m256 mask_s12_lt_0 = _mm256_cmp_ps(v_sigma12, v_zero, _CMP_LT_OS);
            v_num_val = _mm256_andnot_ps(mask_s12_lt_0, v_num_val);

            // if (sigma1_sq < sigma_nsq): num_val = 1 - sigma2_sq * sigma_max_inv
            //                             den_val = 1
            __m256 mask_s1_lt_nsq = _mm256_cmp_ps(
                // We need original sigma1_sq before zeroing.
                // Re-compute from xx and mu1_sq for the comparison.
                _mm256_max_ps(_mm256_sub_ps(v_xx, v_mu1_sq), v_zero),
                v_sigma_nsq, _CMP_LT_OS);
            __m256 v_alt_num = _mm256_sub_ps(v_one,
                _mm256_mul_ps(
                    _mm256_max_ps(_mm256_sub_ps(v_yy, v_mu2_sq), v_zero),
                    v_sigma_max_inv));
            v_num_val = _mm256_blendv_ps(v_num_val, v_alt_num, mask_s1_lt_nsq);
            v_den_val = _mm256_blendv_ps(v_den_val, v_one, mask_s1_lt_nsq);

            /* Extract and accumulate sequentially in float to match scalar precision */
            {
                float tmp_n[8] __attribute__((aligned(32)));
                float tmp_d[8] __attribute__((aligned(32)));
                _mm256_store_ps(tmp_n, v_num_val);
                _mm256_store_ps(tmp_d, v_den_val);
                for (int k = 0; k < 8; k++) {
                    accum_inner_num += tmp_n[k];
                    accum_inner_den += tmp_d[k];
                }
            }
        }

        // Scalar tail for remaining pixels
        for (; j < w; ++j) {
            float mu1_val = mu1_row[j];
            float mu2_val = mu2_row[j];
            float mu1_sq_val = mu1_val * mu1_val;
            float mu2_sq_val = mu2_val * mu2_val;
            float mu1_mu2_val = mu1_val * mu2_val;
            float xx_filt_val = xx_row[j];
            float yy_filt_val = yy_row[j];
            float xy_filt_val = xy_row[j];

            float sigma1_sq = MAX(xx_filt_val - mu1_sq_val, 0.0f);
            float sigma2_sq = MAX(yy_filt_val - mu2_sq_val, 0.0f);
            float sigma12 = xy_filt_val - mu1_mu2_val;

            float g = sigma12 / (sigma1_sq + eps);
            float sv_sq = sigma2_sq - g * sigma12;

            if (sigma1_sq < eps) { g = 0.0f; sv_sq = sigma2_sq; sigma1_sq = 0.0f; }
            if (sigma2_sq < eps) { g = 0.0f; sv_sq = 0.0f; }
            if (g < 0.0f) { sv_sq = sigma2_sq; g = 0.0f; }
            sv_sq = MAX(sv_sq, eps);
            g = MIN(g, vif_egl_f);

            float num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + sigma_nsq));
            float den_val = log2f(1.0f + sigma1_sq / sigma_nsq);

            if (sigma12 < 0.0f) num_val = 0.0f;
            if (sigma1_sq < sigma_nsq) {
                num_val = 1.0f - sigma2_sq * sigma_max_inv;
                den_val = 1.0f;
            }

            accum_inner_num += num_val;
            accum_inner_den += den_val;
        }

        accum_num += accum_inner_num;
        accum_den += accum_inner_den;
    }

    *num = accum_num;
    *den = accum_den;
}
