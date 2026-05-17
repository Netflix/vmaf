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
 * Numerical-parity contract test for the float-VIF AVX2 statistic kernel
 * (`vif_statistic_s_avx2`).
 *
 * Verifies the ADR-0138 bit-exactness fix: the sigma_max_inv constant must
 * be computed with the same precision as the scalar reference in vif_tools.c.
 * The scalar uses `powf(sigma_nsq, 2.0f) / (255.0 * 255.0)` where the
 * denominator is a double-precision product; the AVX2 path previously used
 * `255.0f * 255.0f` (float), causing 1-5e-6 divergence starting at frame 3
 * of src01_hrc00/hrc01.
 *
 * The test calls both paths on synthetic random inputs and requires the
 * num/den outputs to agree within a tight relative tolerance (1e-7), covering
 * pixel values that exercise the sigma1_sq < sigma_nsq branch (the branch
 * that uses sigma_max_inv).
 *
 * Boilerplate provided by `simd_bitexact_test.h` (ADR-0245).
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "config.h"
#include "test.h"
/* clang-format off — test.h has no header guard; must precede harness. */
#include "simd_bitexact_test.h"
/* clang-format on */

#include "feature/vif_tools.h"

#if ARCH_X86
#include "feature/x86/vif_statistic_avx2.h"
#endif

/* Test dimensions: width chosen non-multiple-of-8 to exercise the tail path;
 * height 7 to accumulate across rows. */
#define VIF_TEST_W 37
#define VIF_TEST_H 7
/* Stride in bytes = width * sizeof(float), rounded up to 32-byte alignment. */
#define VIF_STRIDE_PX ((VIF_TEST_W + 7) & ~7)
#define VIF_STRIDE_B ((size_t)VIF_STRIDE_PX * sizeof(float))

/*
 * VIF statistic inputs use plausible image-statistics ranges.
 * mu values in [0, 255], xx/yy/xy_filt (variance-like) in [0, 5000],
 * with some negative xy values to exercise the sigma12 < 0 branch.
 */
#define VIF_MU_LO 0.0f
#define VIF_MU_HI 255.0f
#define VIF_VAR_LO (-500.0f)
#define VIF_VAR_HI 5000.0f

/* Relative tolerance: the fix changes sigma_max_inv by at most one ULP in
 * float32 (~1.2e-7).  Allow 1e-6 to accommodate accumulated row errors. */
#define VIF_REL_TOL 1e-6

#if ARCH_X86

static char *check_vif_stat_avx2(uint32_t seed, int w, int h)
{
    const int stride_px = (w + 7) & ~7;
    const size_t stride_b = (size_t)stride_px * sizeof(float);
    const size_t total = (size_t)stride_px * (size_t)h;

    float *mu1 = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);
    float *mu2 = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);
    float *xx_flt = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);
    float *yy_flt = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);
    float *xy_flt = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);

    if (!mu1 || !mu2 || !xx_flt || !yy_flt || !xy_flt) {
        simd_test_aligned_free(mu1);
        simd_test_aligned_free(mu2);
        simd_test_aligned_free(xx_flt);
        simd_test_aligned_free(yy_flt);
        simd_test_aligned_free(xy_flt);
        return "aligned_malloc failed";
    }

    simd_test_fill_random_f32(mu1, total, VIF_MU_LO, VIF_MU_HI, seed);
    simd_test_fill_random_f32(mu2, total, VIF_MU_LO, VIF_MU_HI, seed ^ 0x11111111u);
    simd_test_fill_random_f32(xx_flt, total, VIF_VAR_LO, VIF_VAR_HI, seed ^ 0x22222222u);
    simd_test_fill_random_f32(yy_flt, total, VIF_VAR_LO, VIF_VAR_HI, seed ^ 0x33333333u);
    simd_test_fill_random_f32(xy_flt, total, VIF_VAR_LO, VIF_VAR_HI, seed ^ 0x44444444u);

    const double sigma_nsq = 2.0; /* default VMAF VIF sigma_nsq */
    const double egl = 100.0;     /* default enhancement gain limit */
    const int stride_b_int = (int)stride_b;

    float num_scalar = 0.0f, den_scalar = 0.0f;
    vif_statistic_s(mu1, mu2, xx_flt, yy_flt, xy_flt, &num_scalar, &den_scalar, w, h, stride_b_int,
                    stride_b_int, stride_b_int, stride_b_int, stride_b_int, egl, sigma_nsq);

    float num_avx2 = 0.0f, den_avx2 = 0.0f;
    vif_statistic_s_avx2(mu1, mu2, xx_flt, yy_flt, xy_flt, &num_avx2, &den_avx2, w, h, stride_b_int,
                         stride_b_int, stride_b_int, stride_b_int, stride_b_int, egl, sigma_nsq);

    simd_test_aligned_free(mu1);
    simd_test_aligned_free(mu2);
    simd_test_aligned_free(xx_flt);
    simd_test_aligned_free(yy_flt);
    simd_test_aligned_free(xy_flt);

    SIMD_BITEXACT_ASSERT_RELATIVE((double)num_scalar, (double)num_avx2, VIF_REL_TOL,
                                  "vif_statistic_s_avx2 num outside tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE((double)den_scalar, (double)den_avx2, VIF_REL_TOL,
                                  "vif_statistic_s_avx2 den outside tolerance");
    return NULL;
}

static char *test_vif_avx2_seed_a(void)
{
    return check_vif_stat_avx2(0xdeadbeefu, VIF_TEST_W, VIF_TEST_H);
}

static char *test_vif_avx2_seed_b(void)
{
    return check_vif_stat_avx2(0x12345678u, VIF_TEST_W, VIF_TEST_H);
}

static char *test_vif_avx2_aligned_w(void)
{
    /* Width is a multiple of 8 — exercises pure SIMD path with no tail. */
    return check_vif_stat_avx2(0xabcdef01u, 32, VIF_TEST_H);
}

static char *test_vif_avx2_tiny(void)
{
    /* Single row, very small width — exercises tail-only path. */
    return check_vif_stat_avx2(0xfeedface, 3, 1);
}

/* Fixture that maximises coverage of the sigma1_sq < sigma_nsq branch
 * (the branch that directly uses sigma_max_inv, i.e. the fix target).
 * We set xx_flt to small values so sigma1_sq = xx - mu1^2 < sigma_nsq. */
static char *test_vif_avx2_sigma_max_inv_branch(void)
{
    const int w = VIF_TEST_W;
    const int h = VIF_TEST_H;
    const int stride_px = (w + 7) & ~7;
    const size_t total = (size_t)stride_px * (size_t)h;
    const int stride_b_int = stride_px * (int)sizeof(float);

    float *mu1 = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);
    float *mu2 = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);
    float *xx_flt = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);
    float *yy_flt = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);
    float *xy_flt = (float *)simd_test_aligned_malloc(total * sizeof(float), 32);

    if (!mu1 || !mu2 || !xx_flt || !yy_flt || !xy_flt) {
        simd_test_aligned_free(mu1);
        simd_test_aligned_free(mu2);
        simd_test_aligned_free(xx_flt);
        simd_test_aligned_free(yy_flt);
        simd_test_aligned_free(xy_flt);
        return "aligned_malloc failed";
    }

    /* Fill mu with large values so mu^2 ≫ xx_flt → sigma1_sq < 0 → clamped
     * to 0 < sigma_nsq (2.0), exercising the sigma_max_inv branch. */
    simd_test_fill_random_f32(mu1, total, 100.0f, 200.0f, 0xaaaabbbb);
    simd_test_fill_random_f32(mu2, total, 100.0f, 200.0f, 0xbbbbcccc);
    simd_test_fill_random_f32(xx_flt, total, 0.0f, 1.0f, 0xccccdddd);
    simd_test_fill_random_f32(yy_flt, total, 0.0f, 5000.0f, 0xddddeeee);
    simd_test_fill_random_f32(xy_flt, total, -500.0f, 500.0f, 0xeeeeffff);

    const double sigma_nsq = 2.0;
    const double egl = 100.0;

    float num_scalar = 0.0f, den_scalar = 0.0f;
    vif_statistic_s(mu1, mu2, xx_flt, yy_flt, xy_flt, &num_scalar, &den_scalar, w, h, stride_b_int,
                    stride_b_int, stride_b_int, stride_b_int, stride_b_int, egl, sigma_nsq);

    float num_avx2 = 0.0f, den_avx2 = 0.0f;
    vif_statistic_s_avx2(mu1, mu2, xx_flt, yy_flt, xy_flt, &num_avx2, &den_avx2, w, h, stride_b_int,
                         stride_b_int, stride_b_int, stride_b_int, stride_b_int, egl, sigma_nsq);

    simd_test_aligned_free(mu1);
    simd_test_aligned_free(mu2);
    simd_test_aligned_free(xx_flt);
    simd_test_aligned_free(yy_flt);
    simd_test_aligned_free(xy_flt);

    SIMD_BITEXACT_ASSERT_RELATIVE((double)num_scalar, (double)num_avx2, VIF_REL_TOL,
                                  "vif_statistic_s_avx2 sigma_max_inv branch num");
    SIMD_BITEXACT_ASSERT_RELATIVE((double)den_scalar, (double)den_avx2, VIF_REL_TOL,
                                  "vif_statistic_s_avx2 sigma_max_inv branch den");
    return NULL;
}

#endif /* ARCH_X86 */

char *run_tests(void)
{
#if ARCH_X86
    if (!simd_test_have_avx2()) {
        return NULL;
    }
    mu_run_test(test_vif_avx2_seed_a);
    mu_run_test(test_vif_avx2_seed_b);
    mu_run_test(test_vif_avx2_aligned_w);
    mu_run_test(test_vif_avx2_tiny);
    mu_run_test(test_vif_avx2_sigma_max_inv_branch);
#else
    (void)fprintf(stderr, "skipping: non-x86 arch\n");
#endif
    return NULL;
}
