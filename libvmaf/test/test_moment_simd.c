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
 * Numerical-parity contract test for the float_moment SIMD kernels
 * (T7-19, ADR-0179).
 *
 * The contract per `moment_avx2.c` / `moment_neon.c` headers is
 * tolerance-bounded, not bit-exact: the lane-widening + per-row
 * tail divergence yields residuals well inside the snapshot gate's
 * tolerance but not byte-for-byte equal to the scalar reference.
 * The scalar TU's auto-vectorisation (and any compiler-driven
 * precision behaviour) further removes the bit-exact guarantee.
 *
 * Tolerance: 1e-9 absolute on the post-normalisation score (range
 * 0..2^16 for 8-bit pixel squares), which is ~5 orders of magnitude
 * tighter than the snapshot gate's `places=4`.
 *
 * Boilerplate (xorshift PRNG, portable aligned alloc, AVX2 gate,
 * relative-tolerance assertion) is provided by
 * `simd_bitexact_test.h` (ADR-0245).
 */

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "test.h"
/* clang-format off — `test.h` has no header guard, must precede the
 * harness include to avoid a `mu_report` redefinition. */
#include "simd_bitexact_test.h"
/* clang-format on */

#include "feature/moment.h"

#if ARCH_X86
#include "feature/x86/moment_avx2.h"
#endif
#if ARCH_AARCH64
#include "feature/arm64/moment_neon.h"
#endif

#define ALIGN_BYTES 32
#define TEST_W 73 /* not a multiple of 4 or 8 — exercises tail */
#define TEST_H 17

/* Relative tolerance: 1e-7 of the scalar score. Residual sources:
 * per-row tail order, lane-pair cross-add precision, scalar-TU
 * auto-vectorisation. Still ~500× tighter than the production
 * snapshot gate (`places=4` ⇒ |abs| < 5e-5 on a normalised score). */
#define MOMENT_REL_TOL 1e-7

/* Pixel input range matches the post-`picture_copy` 8-bit float
 * layout of the float_moment extractor: values in [0, 256). */
#define MOMENT_FILL_LO 0.0f
#define MOMENT_FILL_HI 256.0f

#if ARCH_X86

static char *check_avx2(uint32_t seed, int w, int h)
{
    const int stride_floats = (w + 7) & ~7;
    const size_t bytes = (size_t)stride_floats * (size_t)h * sizeof(float);
    const int stride_bytes = stride_floats * (int)sizeof(float);

    float *buf = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    if (!buf) {
        return "aligned_malloc failed";
    }
    simd_test_fill_random_f32(buf, (size_t)stride_floats * (size_t)h, MOMENT_FILL_LO,
                              MOMENT_FILL_HI, seed);

    double s_scalar = 0.0;
    double s_avx2 = 0.0;
    (void)compute_1st_moment(buf, w, h, stride_bytes, &s_scalar);
    (void)compute_1st_moment_avx2(buf, w, h, stride_bytes, &s_avx2);

    double t_scalar = 0.0;
    double t_avx2 = 0.0;
    (void)compute_2nd_moment(buf, w, h, stride_bytes, &t_scalar);
    (void)compute_2nd_moment_avx2(buf, w, h, stride_bytes, &t_avx2);

    simd_test_aligned_free(buf);

    SIMD_BITEXACT_ASSERT_RELATIVE(s_scalar, s_avx2, MOMENT_REL_TOL,
                                  "compute_1st_moment_avx2 outside relative tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE(t_scalar, t_avx2, MOMENT_REL_TOL,
                                  "compute_2nd_moment_avx2 outside relative tolerance");
    return NULL;
}

static char *test_avx2_seed_a(void)
{
    return check_avx2(0xdeadbeefu, TEST_W, TEST_H);
}
static char *test_avx2_seed_b(void)
{
    return check_avx2(0x12345678u, TEST_W, TEST_H);
}
static char *test_avx2_aligned_w(void)
{
    return check_avx2(0xabcdef01u, 64, 16);
}
static char *test_avx2_tiny(void)
{
    return check_avx2(0xfeedface, 9, 1);
}

#endif /* ARCH_X86 */

#if ARCH_AARCH64

static char *check_neon(uint32_t seed, int w, int h)
{
    const int stride_floats = (w + 3) & ~3;
    const size_t bytes = (size_t)stride_floats * (size_t)h * sizeof(float);
    const int stride_bytes = stride_floats * (int)sizeof(float);

    float *buf = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    if (!buf) {
        return "aligned_malloc failed";
    }
    simd_test_fill_random_f32(buf, (size_t)stride_floats * (size_t)h, MOMENT_FILL_LO,
                              MOMENT_FILL_HI, seed);

    double s_scalar = 0.0;
    double s_neon = 0.0;
    (void)compute_1st_moment(buf, w, h, stride_bytes, &s_scalar);
    (void)compute_1st_moment_neon(buf, w, h, stride_bytes, &s_neon);

    double t_scalar = 0.0;
    double t_neon = 0.0;
    (void)compute_2nd_moment(buf, w, h, stride_bytes, &t_scalar);
    (void)compute_2nd_moment_neon(buf, w, h, stride_bytes, &t_neon);

    simd_test_aligned_free(buf);

    SIMD_BITEXACT_ASSERT_RELATIVE(s_scalar, s_neon, MOMENT_REL_TOL,
                                  "compute_1st_moment_neon outside relative tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE(t_scalar, t_neon, MOMENT_REL_TOL,
                                  "compute_2nd_moment_neon outside relative tolerance");
    return NULL;
}

static char *test_neon_seed_a(void)
{
    return check_neon(0xdeadbeefu, TEST_W, TEST_H);
}
static char *test_neon_seed_b(void)
{
    return check_neon(0x12345678u, TEST_W, TEST_H);
}
static char *test_neon_aligned_w(void)
{
    return check_neon(0xabcdef01u, 64, 16);
}
static char *test_neon_tiny(void)
{
    return check_neon(0xfeedface, 5, 1);
}

#endif /* ARCH_AARCH64 */

char *run_tests(void)
{
#if ARCH_X86
    if (!simd_test_have_avx2()) {
        return NULL;
    }
    mu_run_test(test_avx2_seed_a);
    mu_run_test(test_avx2_seed_b);
    mu_run_test(test_avx2_aligned_w);
    mu_run_test(test_avx2_tiny);
#elif ARCH_AARCH64
    mu_run_test(test_neon_seed_a);
    mu_run_test(test_neon_seed_b);
    mu_run_test(test_neon_aligned_w);
    mu_run_test(test_neon_tiny);
#else
    (void)fprintf(stderr, "skipping: arch lacks moment SIMD\n");
#endif
    return NULL;
}
