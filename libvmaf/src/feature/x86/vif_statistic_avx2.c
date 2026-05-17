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
#include <stddef.h>

#include "vif_statistic_avx2.h"

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
    __m256i exp_i =
        _mm256_sub_epi32(_mm256_and_si256(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(0xFF)),
                         _mm256_set1_epi32(127));
    __m256 e = _mm256_cvtepi32_ps(exp_i);

    // Normalize mantissa to [1, 2): clear exponent, set to 127
    __m256i mantissa_i = _mm256_or_si256(_mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)),
                                         _mm256_set1_epi32(0x3F800000));
    __m256 m = _mm256_castsi256_ps(mantissa_i);

    // t = m - 1, t in [0, 1)
    __m256 t = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));

    // Degree-8 minimax polynomial for log2(1+t) on [0, 1)
    // Coefficients from vif_tools.c log2_poly_s[], Horner evaluation
    __m256 p = _mm256_set1_ps(-0.012671635276421f);
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(0.064841182402670f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(-0.157048836463065f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(0.257167726303123f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(-0.353800560300520f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(0.480131410397451f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(-0.721314327952201f));
    p = _mm256_add_ps(_mm256_mul_ps(p, t), _mm256_set1_ps(1.442694803896991f));
    p = _mm256_mul_ps(p, t); // last coefficient is 0, equivalent to * t

    return _mm256_add_ps(e, p);
}

/* Per-invocation constants used by both SIMD and scalar kernels. Keeping
 * the scalar-float constants and the __m256 broadcasts together makes the
 * helpers below bit-identical to the inline form — ADR-0138 / ADR-0139. */
struct vif_stat_consts {
    float sigma_nsq;
    float sigma_max_inv;
    float eps;
    float vif_egl_f;
    __m256 v_sigma_nsq;
    __m256 v_sigma_max_inv;
    __m256 v_eps;
    __m256 v_zero;
    __m256 v_one;
    __m256 v_egl;
};

/* Per-lane state carried between the clamp+ratio stage and the log+reduce
 * stage. Kept as an explicit struct so both halves of the 8-lane block can
 * live in their own functions without losing the ADR-0139 per-lane
 * scalar-float reduction contract. */
struct vif_simd8_lane {
    __m256 v_mu1_sq;
    __m256 v_mu2_sq;
    __m256 v_xx;
    __m256 v_yy;
    __m256 v_sigma1_sq;
    __m256 v_sigma12;
    __m256 v_sv_sq;
    __m256 v_g;
};

static inline struct vif_simd8_lane vif_stat_simd8_compute(const float *mu1_row,
                                                           const float *mu2_row,
                                                           const float *xx_row, const float *yy_row,
                                                           const float *xy_row, int j,
                                                           const struct vif_stat_consts *c)
{
    __m256 v_mu1 = _mm256_loadu_ps(mu1_row + j);
    __m256 v_mu2 = _mm256_loadu_ps(mu2_row + j);
    __m256 v_xx = _mm256_loadu_ps(xx_row + j);
    __m256 v_yy = _mm256_loadu_ps(yy_row + j);
    __m256 v_xy = _mm256_loadu_ps(xy_row + j);

    __m256 v_mu1_sq = _mm256_mul_ps(v_mu1, v_mu1);
    __m256 v_mu2_sq = _mm256_mul_ps(v_mu2, v_mu2);
    __m256 v_mu1_mu2 = _mm256_mul_ps(v_mu1, v_mu2);

    __m256 v_sigma1_sq = _mm256_max_ps(_mm256_sub_ps(v_xx, v_mu1_sq), c->v_zero);
    __m256 v_sigma2_sq = _mm256_max_ps(_mm256_sub_ps(v_yy, v_mu2_sq), c->v_zero);
    __m256 v_sigma12 = _mm256_sub_ps(v_xy, v_mu1_mu2);

    __m256 v_g = _mm256_div_ps(v_sigma12, _mm256_add_ps(v_sigma1_sq, c->v_eps));
    __m256 v_sv_sq = _mm256_sub_ps(v_sigma2_sq, _mm256_mul_ps(v_g, v_sigma12));

    __m256 mask_s1_lt_eps = _mm256_cmp_ps(v_sigma1_sq, c->v_eps, _CMP_LT_OS);
    v_g = _mm256_andnot_ps(mask_s1_lt_eps, v_g);
    v_sv_sq = _mm256_blendv_ps(v_sv_sq, v_sigma2_sq, mask_s1_lt_eps);
    v_sigma1_sq = _mm256_andnot_ps(mask_s1_lt_eps, v_sigma1_sq);

    __m256 mask_s2_lt_eps = _mm256_cmp_ps(v_sigma2_sq, c->v_eps, _CMP_LT_OS);
    v_g = _mm256_andnot_ps(mask_s2_lt_eps, v_g);
    v_sv_sq = _mm256_andnot_ps(mask_s2_lt_eps, v_sv_sq);

    __m256 mask_g_lt_0 = _mm256_cmp_ps(v_g, c->v_zero, _CMP_LT_OS);
    v_sv_sq = _mm256_blendv_ps(v_sv_sq, v_sigma2_sq, mask_g_lt_0);
    v_g = _mm256_max_ps(v_g, c->v_zero);

    v_sv_sq = _mm256_max_ps(v_sv_sq, c->v_eps);
    v_g = _mm256_min_ps(v_g, c->v_egl);

    struct vif_simd8_lane L = {
        .v_mu1_sq = v_mu1_sq,
        .v_mu2_sq = v_mu2_sq,
        .v_xx = v_xx,
        .v_yy = v_yy,
        .v_sigma1_sq = v_sigma1_sq,
        .v_sigma12 = v_sigma12,
        .v_sv_sq = v_sv_sq,
        .v_g = v_g,
    };
    return L;
}

/* Finalize one 8-lane block: compute log2f num/den, apply the two late
 * branches (sigma12 < 0 and sigma1_sq < sigma_nsq), and per-lane reduce
 * into the scalar-float accumulators (ADR-0139). */
static inline void vif_stat_simd8_reduce(const struct vif_simd8_lane *L,
                                         const struct vif_stat_consts *c, float *accum_num,
                                         float *accum_den)
{
    __m256 v_g_sq = _mm256_mul_ps(L->v_g, L->v_g);
    __m256 v_num_arg =
        _mm256_add_ps(c->v_one, _mm256_div_ps(_mm256_mul_ps(v_g_sq, L->v_sigma1_sq),
                                              _mm256_add_ps(L->v_sv_sq, c->v_sigma_nsq)));
    __m256 v_num_val = log2_ps_avx2(v_num_arg);
    __m256 v_den_val =
        log2_ps_avx2(_mm256_add_ps(c->v_one, _mm256_div_ps(L->v_sigma1_sq, c->v_sigma_nsq)));

    __m256 mask_s12_lt_0 = _mm256_cmp_ps(L->v_sigma12, c->v_zero, _CMP_LT_OS);
    v_num_val = _mm256_andnot_ps(mask_s12_lt_0, v_num_val);

    /* Recompute original sigma1_sq (before the <eps zero step) for the
     * sigma1_sq < sigma_nsq branch. */
    __m256 mask_s1_lt_nsq = _mm256_cmp_ps(
        _mm256_max_ps(_mm256_sub_ps(L->v_xx, L->v_mu1_sq), c->v_zero), c->v_sigma_nsq, _CMP_LT_OS);
    __m256 v_alt_num = _mm256_sub_ps(
        c->v_one, _mm256_mul_ps(_mm256_max_ps(_mm256_sub_ps(L->v_yy, L->v_mu2_sq), c->v_zero),
                                c->v_sigma_max_inv));
    v_num_val = _mm256_blendv_ps(v_num_val, v_alt_num, mask_s1_lt_nsq);
    v_den_val = _mm256_blendv_ps(v_den_val, c->v_one, mask_s1_lt_nsq);

    _Alignas(32) float tmp_n[8];
    _Alignas(32) float tmp_d[8];
    _mm256_store_ps(tmp_n, v_num_val);
    _mm256_store_ps(tmp_d, v_den_val);
    for (int k = 0; k < 8; k++) {
        *accum_num += tmp_n[k];
        *accum_den += tmp_d[k];
    }
}

static inline void vif_stat_simd8_block(const float *mu1_row, const float *mu2_row,
                                        const float *xx_row, const float *yy_row,
                                        const float *xy_row, int j, const struct vif_stat_consts *c,
                                        float *accum_num, float *accum_den)
{
    struct vif_simd8_lane L =
        vif_stat_simd8_compute(mu1_row, mu2_row, xx_row, yy_row, xy_row, j, c);
    vif_stat_simd8_reduce(&L, c, accum_num, accum_den);
}

static inline void vif_stat_scalar_pixel(float mu1_val, float mu2_val, float xx_filt_val,
                                         float yy_filt_val, float xy_filt_val,
                                         const struct vif_stat_consts *c, float *accum_num,
                                         float *accum_den)
{
    float mu1_sq_val = mu1_val * mu1_val;
    float mu2_sq_val = mu2_val * mu2_val;
    float mu1_mu2_val = mu1_val * mu2_val;

    float sigma1_sq = MAX(xx_filt_val - mu1_sq_val, 0.0f);
    float sigma2_sq = MAX(yy_filt_val - mu2_sq_val, 0.0f);
    float sigma12 = xy_filt_val - mu1_mu2_val;

    float g = sigma12 / (sigma1_sq + c->eps);
    float sv_sq = sigma2_sq - g * sigma12;

    if (sigma1_sq < c->eps) {
        g = 0.0f;
        sv_sq = sigma2_sq;
        sigma1_sq = 0.0f;
    }
    if (sigma2_sq < c->eps) {
        g = 0.0f;
        sv_sq = 0.0f;
    }
    if (g < 0.0f) {
        sv_sq = sigma2_sq;
        g = 0.0f;
    }
    sv_sq = MAX(sv_sq, c->eps);
    g = MIN(g, c->vif_egl_f);

    float num_val = log2f(1.0f + (g * g * sigma1_sq) / (sv_sq + c->sigma_nsq));
    float den_val = log2f(1.0f + sigma1_sq / c->sigma_nsq);

    if (sigma12 < 0.0f)
        num_val = 0.0f;
    if (sigma1_sq < c->sigma_nsq) {
        num_val = 1.0f - sigma2_sq * c->sigma_max_inv;
        den_val = 1.0f;
    }

    *accum_num += num_val;
    *accum_den += den_val;
}

static inline void vif_stat_consts_init(struct vif_stat_consts *c, double vif_enhn_gain_limit,
                                        double vif_sigma_nsq)
{
    c->sigma_nsq = (float)vif_sigma_nsq;
    /* Bit-exactness fix (ADR-0138): scalar computes
     *   sigma_max_inv = powf(vif_sigma_nsq, 2.0f) / (255.0 * 255.0)
     * where "255.0 * 255.0" is a double-precision multiplication (255.0
     * is a double literal).  Using 255.0f here switches to a float-only
     * division that produces a different rounding, causing 1-5e-6
     * divergence starting at frame 3 of src01_hrc00/hrc01.
     * Mirror the scalar expression exactly: powf result (float), then
     * divide by the double-precision constant 65025.0. */
    c->sigma_max_inv = (float)(powf((float)vif_sigma_nsq, 2.0f) / (255.0 * 255.0));
    c->eps = 1.0e-10f;
    c->vif_egl_f = (float)vif_enhn_gain_limit;
    c->v_sigma_nsq = _mm256_set1_ps(c->sigma_nsq);
    c->v_sigma_max_inv = _mm256_set1_ps(c->sigma_max_inv);
    c->v_eps = _mm256_set1_ps(c->eps);
    c->v_zero = _mm256_setzero_ps();
    c->v_one = _mm256_set1_ps(1.0f);
    c->v_egl = _mm256_set1_ps(c->vif_egl_f);
}

void vif_statistic_s_avx2(const float *mu1, const float *mu2, const float *xx_filt,
                          const float *yy_filt, const float *xy_filt, float *num, float *den, int w,
                          int h, int mu1_stride, int mu2_stride, int xx_filt_stride,
                          int yy_filt_stride, int xy_filt_stride, double vif_enhn_gain_limit,
                          double vif_sigma_nsq)
{
    struct vif_stat_consts c;
    vif_stat_consts_init(&c, vif_enhn_gain_limit, vif_sigma_nsq);

    const ptrdiff_t mu1_px_stride = (ptrdiff_t)mu1_stride / (ptrdiff_t)sizeof(float);
    const ptrdiff_t mu2_px_stride = (ptrdiff_t)mu2_stride / (ptrdiff_t)sizeof(float);
    const ptrdiff_t xx_filt_px_stride = (ptrdiff_t)xx_filt_stride / (ptrdiff_t)sizeof(float);
    const ptrdiff_t yy_filt_px_stride = (ptrdiff_t)yy_filt_stride / (ptrdiff_t)sizeof(float);
    const ptrdiff_t xy_filt_px_stride = (ptrdiff_t)xy_filt_stride / (ptrdiff_t)sizeof(float);

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
        const int w8 = w & ~7;
        int j = 0;
        for (; j < w8; j += 8) {
            vif_stat_simd8_block(mu1_row, mu2_row, xx_row, yy_row, xy_row, j, &c, &accum_inner_num,
                                 &accum_inner_den);
        }
        for (; j < w; ++j) {
            vif_stat_scalar_pixel(mu1_row[j], mu2_row[j], xx_row[j], yy_row[j], xy_row[j], &c,
                                  &accum_inner_num, &accum_inner_den);
        }

        accum_num += accum_inner_num;
        accum_den += accum_inner_den;
    }

    *num = accum_num;
    *den = accum_den;
}
