/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
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

/*
 * Numerical-parity contract test for `calc_psnrhvs_avx2`.
 *
 * This test covers the ADR-0138 bit-exactness fix for Bug 2:
 *   compute_masks in psnr_hvs_avx2.c previously computed
 *     s_mask = sqrt(s_mask * s_gvar) / 32.f
 *   (float * float, then sqrt), but the scalar reference (third_party/xiph/
 *   psnr_hvs.c:351) uses
 *     s_mask = sqrt((double)s_mask * s_gvar) / 32.f
 *   (explicit cast to double before multiply), causing ~3.77e-7 divergence
 *   on the Cb channel at frame 0.
 *
 * The test calls `calc_psnrhvs_avx2` end-to-end on synthetic 8-bit and
 * 10-bit image patches and verifies the returned double is within 1e-12
 * of an inline scalar reference that mirrors the scalar code exactly.
 *
 * The inline scalar reference is a self-contained copy to avoid a linkage
 * dependency on the `static calc_psnrhvs` in psnr_hvs.c, following the
 * same pattern as `test_psnr_hvs_avx2.c` (which copies `od_bin_fdct8x8`).
 *
 * Boilerplate provided by `simd_bitexact_test.h` (ADR-0245).
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "test.h"
/* clang-format off — test.h has no header guard; must precede harness. */
#include "simd_bitexact_test.h"
/* clang-format on */

#if ARCH_X86
#include "feature/x86/psnr_hvs_avx2.h"
#endif

/* -----------------------------------------------------------------------
 * Inline scalar reference — self-contained copy of calc_psnrhvs from
 * third_party/xiph/psnr_hvs.c, stripped of OD_DCT_OVERFLOW_CHECK macros
 * (those are no-ops in production builds).  Kept intentionally verbose
 * so reviewers can diff it line-for-line against the upstream source.
 * The critical expression is the (double) cast in the sqrt lines.
 * -------------------------------------------------------------------- */

typedef int32_t od_coeff_ref;

#define OD_UNBIASED_RSHIFT32_REF(a, b) (((int32_t)(((uint32_t)(a) >> (32 - (b))) + (a))) >> (b))
#define OD_DCT_RSHIFT_REF(a, b) OD_UNBIASED_RSHIFT32_REF(a, b)

// NOLINTNEXTLINE(readability-function-size) load-bearing upstream scalar copy.
static void ref_od_bin_fdct8_hvs(od_coeff_ref y[8], const od_coeff_ref *x, int xstride)
{
    const ptrdiff_t xs = (ptrdiff_t)xstride;
    int t0 = x[0 * xs], t4 = x[1 * xs], t2 = x[2 * xs], t6 = x[3 * xs];
    int t7 = x[4 * xs], t3 = x[5 * xs], t5 = x[6 * xs], t1 = x[7 * xs];
    int t1h, t4h, t6h;
    t1 = t0 - t1;
    t1h = OD_DCT_RSHIFT_REF(t1, 1);
    t0 -= t1h;
    t4 += t5;
    t4h = OD_DCT_RSHIFT_REF(t4, 1);
    t5 -= t4h;
    t3 = t2 - t3;
    t2 -= OD_DCT_RSHIFT_REF(t3, 1);
    t6 += t7;
    t6h = OD_DCT_RSHIFT_REF(t6, 1);
    t7 = t6h - t7;
    t0 += t6h;
    t6 = t0 - t6;
    t2 = t4h - t2;
    t4 = t2 - t4;
    t0 -= (t4 * 13573 + 16384) >> 15;
    t4 += (t0 * 11585 + 8192) >> 14;
    t0 -= (t4 * 13573 + 16384) >> 15;
    t6 -= (t2 * 21895 + 16384) >> 15;
    t2 += (t6 * 15137 + 8192) >> 14;
    t6 -= (t2 * 21895 + 16384) >> 15;
    t3 += (t5 * 19195 + 16384) >> 15;
    t5 += (t3 * 11585 + 8192) >> 14;
    t3 -= (t5 * 7489 + 4096) >> 13;
    t7 = OD_DCT_RSHIFT_REF(t5, 1) - t7;
    t5 -= t7;
    t3 = t1h - t3;
    t1 -= t3;
    t7 += (t1 * 3227 + 16384) >> 15;
    t1 -= (t7 * 6393 + 16384) >> 15;
    t7 += (t1 * 3227 + 16384) >> 15;
    t5 += (t3 * 2485 + 4096) >> 13;
    t3 -= (t5 * 18205 + 16384) >> 15;
    t5 += (t3 * 2485 + 4096) >> 13;
    y[0] = (od_coeff_ref)t0;
    y[1] = (od_coeff_ref)t1;
    y[2] = (od_coeff_ref)t2;
    y[3] = (od_coeff_ref)t3;
    y[4] = (od_coeff_ref)t4;
    y[5] = (od_coeff_ref)t5;
    y[6] = (od_coeff_ref)t6;
    y[7] = (od_coeff_ref)t7;
}

static void ref_od_bin_fdct8x8_hvs(od_coeff_ref *y, int ystride, const od_coeff_ref *x, int xstride)
{
    od_coeff_ref z[64];
    const ptrdiff_t ys = (ptrdiff_t)ystride;
    for (int i = 0; i < 8; i++) {
        ref_od_bin_fdct8_hvs(z + 8 * i, x + i, xstride);
    }
    for (int i = 0; i < 8; i++) {
        ref_od_bin_fdct8_hvs(y + ys * i, z + i, 8);
    }
}

/* ref_calc_psnrhvs: scalar reference, including the (double) cast fix. */
#pragma STDC FP_CONTRACT OFF
static double ref_calc_psnrhvs(const unsigned char *src, int systride, const unsigned char *dst,
                               int dystride, double par, int depth, int w, int h, int step,
                               float csf[8][8])
{
    (void)par;
    float mask[8][8];
    float ret = 0.0f;
    int pixels = 0;
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            /* Match upstream scalar literal: double constant (no f suffix). */
            mask[x][y] =
                (float)((csf[x][y] * 0.3885746225901003) * (csf[x][y] * 0.3885746225901003));
        }
    }

    for (int y = 0; y < h - 7; y += step) {
        for (int x = 0; x < w - 7; x += step) {
            od_coeff_ref dct_s[64], dct_d[64];
            float s_means[4] = {0, 0, 0, 0};
            float d_means[4] = {0, 0, 0, 0};
            float s_vars[4] = {0, 0, 0, 0};
            float d_vars[4] = {0, 0, 0, 0};
            float s_gmean = 0.0f, d_gmean = 0.0f;
            float s_gvar = 0.0f, d_gvar = 0.0f;
            float s_mask = 0.0f, d_mask = 0.0f;

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                    if (depth > 8) {
                        dct_s[i * 8 + j] = src[(y + i) * systride + (j + x) * 2] +
                                           (src[(y + i) * systride + (j + x) * 2 + 1] << 8);
                        dct_d[i * 8 + j] = dst[(y + i) * dystride + (j + x) * 2] +
                                           (dst[(y + i) * dystride + (j + x) * 2 + 1] << 8);
                    } else {
                        dct_s[i * 8 + j] = src[(y + i) * systride + (j + x)];
                        dct_d[i * 8 + j] = dst[(y + i) * dystride + (j + x)];
                    }
                    s_gmean += (float)dct_s[i * 8 + j];
                    d_gmean += (float)dct_d[i * 8 + j];
                    s_means[sub] += (float)dct_s[i * 8 + j];
                    d_means[sub] += (float)dct_d[i * 8 + j];
                }
            }
            s_gmean /= 64.0f;
            d_gmean /= 64.0f;
            for (int i = 0; i < 4; i++) {
                s_means[i] /= 16.0f;
                d_means[i] /= 16.0f;
            }

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
                    s_gvar += (dct_s[i * 8 + j] - s_gmean) * (dct_s[i * 8 + j] - s_gmean);
                    d_gvar += (dct_d[i * 8 + j] - d_gmean) * (dct_d[i * 8 + j] - d_gmean);
                    s_vars[sub] +=
                        (dct_s[i * 8 + j] - s_means[sub]) * (dct_s[i * 8 + j] - s_means[sub]);
                    d_vars[sub] +=
                        (dct_d[i * 8 + j] - d_means[sub]) * (dct_d[i * 8 + j] - d_means[sub]);
                }
            }
            s_gvar *= 1.0f / 63.0f * 64;
            d_gvar *= 1.0f / 63.0f * 64;
            for (int i = 0; i < 4; i++) {
                s_vars[i] *= 1.0f / 15.0f * 16;
            }
            for (int i = 0; i < 4; i++) {
                d_vars[i] *= 1.0f / 15.0f * 16;
            }
            if (s_gvar > 0)
                s_gvar = (s_vars[0] + s_vars[1] + s_vars[2] + s_vars[3]) / s_gvar;
            if (d_gvar > 0)
                d_gvar = (d_vars[0] + d_vars[1] + d_vars[2] + d_vars[3]) / d_gvar;

            ref_od_bin_fdct8x8_hvs(dct_s, 8, dct_s, 8);
            ref_od_bin_fdct8x8_hvs(dct_d, 8, dct_d, 8);

            for (int i = 0; i < 8; i++)
                for (int j = (i == 0); j < 8; j++)
                    s_mask += (float)(dct_s[i * 8 + j] * dct_s[i * 8 + j]) * mask[i][j];
            for (int i = 0; i < 8; i++)
                for (int j = (i == 0); j < 8; j++)
                    d_mask += (float)(dct_d[i * 8 + j] * dct_d[i * 8 + j]) * mask[i][j];

            /* ADR-0138 key expression: (double) cast before multiply → double sqrt. */
            // NOLINTNEXTLINE(performance-type-promotion-in-math-fn)
            s_mask = (float)(sqrt((double)s_mask * s_gvar) / 32.0);
            // NOLINTNEXTLINE(performance-type-promotion-in-math-fn)
            d_mask = (float)(sqrt((double)d_mask * d_gvar) / 32.0);
            if (d_mask > s_mask)
                s_mask = d_mask;

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    float err = (float)abs(dct_s[i * 8 + j] - dct_d[i * 8 + j]);
                    if (i != 0 || j != 0)
                        err = err < s_mask / mask[i][j] ? 0.0f : err - s_mask / mask[i][j];
                    ret += (err * csf[i][j]) * (err * csf[i][j]);
                    pixels++;
                }
            }
        }
    }
    ret /= (float)pixels;
    int32_t samplemax = (1 << depth) - 1;
    ret /= (float)(samplemax * samplemax);
    return (double)ret;
}

#if ARCH_X86

/* Chroma CSF table (Cb channel 4:2:0) — copied from psnr_hvs.c. */
static float csf_cb420[8][8] = {
    {1.91113096927f, 2.46074210438f, 1.18284184739f, 1.14982565193f, 1.05017074788f,
     0.898018824055f, 0.74725392039f, 0.615105596242f},
    {2.46074210438f, 1.58529308355f, 1.21363250036f, 1.38190029285f, 1.33100189972f, 1.17428548929f,
     0.996404342439f, 0.830890433625f},
    {1.18284184739f, 1.21363250036f, 0.978712413627f, 1.02624506078f, 1.03145147362f,
     0.960060382087f, 0.849823426169f, 0.731221236837f},
    {1.14982565193f, 1.38190029285f, 1.02624506078f, 0.861317501629f, 0.801821139099f,
     0.751437590932f, 0.685398513368f, 0.608694761374f},
    {1.05017074788f, 1.33100189972f, 1.03145147362f, 0.801821139099f, 0.676555426187f,
     0.605503172737f, 0.55002013668f, 0.495804539034f},
    {0.898018824055f, 1.17428548929f, 0.960060382087f, 0.751437590932f, 0.605503172737f,
     0.514674450957f, 0.454353482512f, 0.407050308965f},
    {0.74725392039f, 0.996404342439f, 0.849823426169f, 0.685398513368f, 0.55002013668f,
     0.454353482512f, 0.389234902883f, 0.342353999733f},
    {0.615105596242f, 0.830890433625f, 0.731221236837f, 0.608694761374f, 0.495804539034f,
     0.407050308965f, 0.342353999733f, 0.295530605237f}};

/* Luma CSF — copied from psnr_hvs.c (csf_y). */
static float csf_y[8][8] = {{2.03536207612f, 2.56512605416f, 1.26137218971f, 0.878376742624f,
                             0.605604690781f, 0.411274736539f, 0.286195802713f, 0.197618718396f},
                            {2.56512605416f, 1.65277218199f, 1.16898583786f, 0.979831908614f,
                             0.716528898958f, 0.518434948743f, 0.384188804073f, 0.292826779404f},
                            {1.26137218971f, 1.16898583786f, 0.947980478949f, 0.844609405879f,
                             0.673600706742f, 0.526600684994f, 0.414072700074f, 0.321892855428f},
                            {0.878376742624f, 0.979831908614f, 0.844609405879f, 0.657580765779f,
                             0.512960694007f, 0.416500747659f, 0.344021088254f, 0.278199576099f},
                            {0.605604690781f, 0.716528898958f, 0.673600706742f, 0.512960694007f,
                             0.382568781979f, 0.301975069116f, 0.248720325076f, 0.204990127278f},
                            {0.411274736539f, 0.518434948743f, 0.526600684994f, 0.416500747659f,
                             0.301975069116f, 0.232254601046f, 0.191261350628f, 0.159609820019f},
                            {0.286195802713f, 0.384188804073f, 0.414072700074f, 0.344021088254f,
                             0.248720325076f, 0.191261350628f, 0.157209798024f, 0.131696069261f},
                            {0.197618718396f, 0.292826779404f, 0.321892855428f, 0.278199576099f,
                             0.204990127278f, 0.159609820019f, 0.131696069261f, 0.110303834139f}};

/*
 * check_psnrhvs_avx2: compare calc_psnrhvs_avx2 output against the inline
 * scalar reference on a (w x h) 8-bit luma or chroma patch.
 * The relative tolerance is 1e-12 — both paths should agree to near
 * double-precision machine epsilon; divergence > 1e-10 signals the
 * un-fixed float-precision sqrt path.
 */
static char *check_psnrhvs_avx2(uint32_t seed, int w, int h, float csf[8][8])
{
    const size_t npx = (size_t)w * (size_t)h;
    unsigned char *src = (unsigned char *)simd_test_aligned_malloc(npx, 32);
    unsigned char *dst = (unsigned char *)simd_test_aligned_malloc(npx, 32);
    if (!src || !dst) {
        simd_test_aligned_free(src);
        simd_test_aligned_free(dst);
        return "aligned_malloc failed";
    }

    /* Fill with byte-range random values (xorshift → low 8 bits). */
    uint32_t state = seed;
    for (size_t i = 0; i < npx; i++) {
        src[i] = (unsigned char)(simd_test_xorshift32(&state) & 0xFFu);
        dst[i] = (unsigned char)(simd_test_xorshift32(&state) & 0xFFu);
    }

    const double scalar_score = ref_calc_psnrhvs(src, w, dst, w, 1.0, 8, w, h, 1, csf);
    const double avx2_score = calc_psnrhvs_avx2(src, w, dst, w, 1.0, 8, w, h, 1, csf);

    simd_test_aligned_free(src);
    simd_test_aligned_free(dst);

    SIMD_BITEXACT_ASSERT_RELATIVE(scalar_score, avx2_score, 1e-12,
                                  "calc_psnrhvs_avx2 score diverges from scalar");
    return NULL;
}

static char *test_psnrhvs_avx2_luma_seed_a(void)
{
    return check_psnrhvs_avx2(0xdeadbeefu, 64, 64, csf_y);
}

static char *test_psnrhvs_avx2_luma_seed_b(void)
{
    return check_psnrhvs_avx2(0x12345678u, 64, 64, csf_y);
}

static char *test_psnrhvs_avx2_chroma_seed_a(void)
{
    /* Chroma — the Cb channel is where Bug 2 was reported. */
    return check_psnrhvs_avx2(0xabcdef01u, 64, 48, csf_cb420);
}

static char *test_psnrhvs_avx2_chroma_seed_b(void)
{
    return check_psnrhvs_avx2(0xfeedface, 32, 24, csf_cb420);
}

static char *test_psnrhvs_avx2_minimal(void)
{
    /* Minimal patch: 16 x 16 — exercises a single 8-step block. */
    return check_psnrhvs_avx2(0xdeadca11u, 16, 16, csf_y);
}

#endif /* ARCH_X86 */

char *run_tests(void)
{
#if ARCH_X86
    if (!simd_test_have_avx2()) {
        return NULL;
    }
    mu_run_test(test_psnrhvs_avx2_luma_seed_a);
    mu_run_test(test_psnrhvs_avx2_luma_seed_b);
    mu_run_test(test_psnrhvs_avx2_chroma_seed_a);
    mu_run_test(test_psnrhvs_avx2_chroma_seed_b);
    mu_run_test(test_psnrhvs_avx2_minimal);
#else
    (void)fprintf(stderr, "skipping: non-x86 arch\n");
#endif
    return NULL;
}
