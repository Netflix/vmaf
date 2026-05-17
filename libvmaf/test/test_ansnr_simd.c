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
 * Numerical-parity contract test for the ansnr MSE SIMD line kernels
 * (T3-7, ADR-0245 coverage audit 2026-05-15).
 *
 * The per-line kernels (`ansnr_mse_line_avx2`, `ansnr_mse_line_avx512`,
 * `ansnr_mse_line_neon`) are tolerance-bounded against the scalar inner
 * loop, not bit-exact.  Sources of divergence:
 *   - AVX2/AVX-512: per-line double accumulator is narrowed back to float
 *     before adding to the running inter-row total in `ansnr_mse_s`;
 *     residuals are sub-ULP at the final log10 call.
 *   - NEON: uses float64x2_t accumulators throughout but the tail scalar
 *     fallback narrows back to float; same regime.
 *
 * The test exercises each kernel in isolation (single row, multiple rows,
 * width not a multiple of the vector width) and compares the accumulated
 * sig/noise values with the scalar inner loop using a relative tolerance
 * of 1e-6 — ~4 orders of magnitude tighter than the snapshot gate's
 * `places=4` threshold.
 *
 * Boilerplate (xorshift PRNG, aligned alloc, AVX2 CPU gate, relative
 * tolerance assertion) is from `simd_bitexact_test.h` (ADR-0245).
 */

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "test.h"
/* clang-format off — test.h has no header guard; must precede harness
 * include to avoid a mu_report redefinition conflict. */
#include "simd_bitexact_test.h"
/* clang-format on */

#if ARCH_X86
#include "feature/x86/ansnr_avx2.h"
#if HAVE_AVX512
#include "feature/x86/ansnr_avx512.h"
#endif
#endif
#if ARCH_AARCH64
#include "feature/arm64/ansnr_neon.h"
#endif

#define ALIGN_BYTES 64
/* Width chosen so that it exercises the tail path on both AVX2 (8-wide)
 * and NEON (4-wide): not a multiple of 8 or 4. */
#define TEST_W 73
/* Multiple rows to verify per-row accumulation. */
#define TEST_H 17

/* Pixel input range: post-picture_copy 8-bit float in [0, 256). */
#define ANSNR_FILL_LO 0.0f
#define ANSNR_FILL_HI 256.0f

/*
 * Relative tolerance: 1e-6 of the larger of the two values.
 * Tighter than the snapshot gate's `places=4` (~5e-5 abs at typical score
 * magnitudes) but looser than the float32 machine epsilon (1.19e-7) to
 * account for double-to-float narrowing in the inter-row accumulation path.
 */
#define ANSNR_REL_TOL 1e-6

/*
 * run_scalar_line_loop: accumulate ref^2 and (ref-dis)^2 over one row
 * using the pure C inner loop from ansnr_tools.c lines 107-119.  This
 * is the reference that the SIMD kernels must match within ANSNR_REL_TOL.
 */
static void run_scalar_line_loop(const float *ref, const float *dis, float *sig_out,
                                 float *noise_out, int w)
{
    float sig = 0.0f;
    float noise = 0.0f;
    for (int j = 0; j < w; j++) {
        float ref_val = ref[j];
        float dis_val = dis[j];
        sig += ref_val * ref_val;
        noise += (ref_val - dis_val) * (ref_val - dis_val);
    }
    *sig_out += sig;
    *noise_out += noise;
}

#if ARCH_X86

static char *check_avx2(uint32_t seed, int w)
{
    const int stride = (w + 7) & ~7;
    const size_t bytes = (size_t)stride * sizeof(float);

    float *ref = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    float *dis = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    if (!ref || !dis) {
        simd_test_aligned_free(ref);
        simd_test_aligned_free(dis);
        return "aligned_malloc failed";
    }

    simd_test_fill_random_f32(ref, (size_t)stride, ANSNR_FILL_LO, ANSNR_FILL_HI, seed);
    simd_test_fill_random_f32(dis, (size_t)stride, ANSNR_FILL_LO, ANSNR_FILL_HI,
                              seed ^ 0xffff0000u);

    float sig_scalar = 0.0f;
    float noise_scalar = 0.0f;
    run_scalar_line_loop(ref, dis, &sig_scalar, &noise_scalar, w);

    float sig_avx2 = 0.0f;
    float noise_avx2 = 0.0f;
    ansnr_mse_line_avx2(ref, dis, &sig_avx2, &noise_avx2, w);

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    SIMD_BITEXACT_ASSERT_RELATIVE((double)sig_scalar, (double)sig_avx2, ANSNR_REL_TOL,
                                  "ansnr_mse_line_avx2 sig outside relative tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE((double)noise_scalar, (double)noise_avx2, ANSNR_REL_TOL,
                                  "ansnr_mse_line_avx2 noise outside relative tolerance");
    return NULL;
}

static char *test_avx2_seed_a(void)
{
    return check_avx2(0xdeadbeefu, TEST_W);
}
static char *test_avx2_seed_b(void)
{
    return check_avx2(0x12345678u, TEST_W);
}
static char *test_avx2_aligned_w(void)
{
    return check_avx2(0xabcdef01u, 64);
}
static char *test_avx2_tiny(void)
{
    return check_avx2(0xfeedface, 9);
}

#if HAVE_AVX512

static char *check_avx512(uint32_t seed, int w)
{
    const int stride = (w + 15) & ~15;
    const size_t bytes = (size_t)stride * sizeof(float);

    float *ref = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    float *dis = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    if (!ref || !dis) {
        simd_test_aligned_free(ref);
        simd_test_aligned_free(dis);
        return "aligned_malloc failed";
    }

    simd_test_fill_random_f32(ref, (size_t)stride, ANSNR_FILL_LO, ANSNR_FILL_HI, seed);
    simd_test_fill_random_f32(dis, (size_t)stride, ANSNR_FILL_LO, ANSNR_FILL_HI,
                              seed ^ 0xffff0000u);

    float sig_scalar = 0.0f;
    float noise_scalar = 0.0f;
    run_scalar_line_loop(ref, dis, &sig_scalar, &noise_scalar, w);

    float sig_avx512 = 0.0f;
    float noise_avx512 = 0.0f;
    ansnr_mse_line_avx512(ref, dis, &sig_avx512, &noise_avx512, w);

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    SIMD_BITEXACT_ASSERT_RELATIVE((double)sig_scalar, (double)sig_avx512, ANSNR_REL_TOL,
                                  "ansnr_mse_line_avx512 sig outside relative tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE((double)noise_scalar, (double)noise_avx512, ANSNR_REL_TOL,
                                  "ansnr_mse_line_avx512 noise outside relative tolerance");
    return NULL;
}

static char *test_avx512_seed_a(void)
{
    return check_avx512(0xdeadbeefu, TEST_W);
}
static char *test_avx512_seed_b(void)
{
    return check_avx512(0x12345678u, TEST_W);
}
static char *test_avx512_aligned_w(void)
{
    return check_avx512(0xabcdef01u, 64);
}
static char *test_avx512_tiny(void)
{
    return check_avx512(0xfeedface, 17);
}

#endif /* HAVE_AVX512 */

#endif /* ARCH_X86 */

#if ARCH_AARCH64

static char *check_neon(uint32_t seed, int w)
{
    const int stride = (w + 3) & ~3;
    const size_t bytes = (size_t)stride * sizeof(float);

    float *ref = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    float *dis = (float *)simd_test_aligned_malloc(bytes, ALIGN_BYTES);
    if (!ref || !dis) {
        simd_test_aligned_free(ref);
        simd_test_aligned_free(dis);
        return "aligned_malloc failed";
    }

    simd_test_fill_random_f32(ref, (size_t)stride, ANSNR_FILL_LO, ANSNR_FILL_HI, seed);
    simd_test_fill_random_f32(dis, (size_t)stride, ANSNR_FILL_LO, ANSNR_FILL_HI,
                              seed ^ 0xffff0000u);

    float sig_scalar = 0.0f;
    float noise_scalar = 0.0f;
    run_scalar_line_loop(ref, dis, &sig_scalar, &noise_scalar, w);

    float sig_neon = 0.0f;
    float noise_neon = 0.0f;
    ansnr_mse_line_neon(ref, dis, &sig_neon, &noise_neon, w);

    simd_test_aligned_free(ref);
    simd_test_aligned_free(dis);

    SIMD_BITEXACT_ASSERT_RELATIVE((double)sig_scalar, (double)sig_neon, ANSNR_REL_TOL,
                                  "ansnr_mse_line_neon sig outside relative tolerance");
    SIMD_BITEXACT_ASSERT_RELATIVE((double)noise_scalar, (double)noise_neon, ANSNR_REL_TOL,
                                  "ansnr_mse_line_neon noise outside relative tolerance");
    return NULL;
}

static char *test_neon_seed_a(void)
{
    return check_neon(0xdeadbeefu, TEST_W);
}
static char *test_neon_seed_b(void)
{
    return check_neon(0x12345678u, TEST_W);
}
static char *test_neon_aligned_w(void)
{
    return check_neon(0xabcdef01u, 64);
}
static char *test_neon_tiny(void)
{
    return check_neon(0xfeedface, 5);
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
#if HAVE_AVX512
    /* AVX-512 path tested only when both AVX2 and AVX-512 are present;
     * simd_test_have_avx2 already confirmed AVX2, and the
     * ansnr_mse_s dispatcher only selects AVX-512 when AVX-512 flags
     * are also set, so we probe directly via vmaf_get_cpu_flags. */
    {
        unsigned flags = vmaf_get_cpu_flags_x86();
        if (flags & VMAF_X86_CPU_FLAG_AVX512) {
            mu_run_test(test_avx512_seed_a);
            mu_run_test(test_avx512_seed_b);
            mu_run_test(test_avx512_aligned_w);
            mu_run_test(test_avx512_tiny);
        }
    }
#endif /* HAVE_AVX512 */
#elif ARCH_AARCH64
    mu_run_test(test_neon_seed_a);
    mu_run_test(test_neon_seed_b);
    mu_run_test(test_neon_aligned_w);
    mu_run_test(test_neon_tiny);
#else
    (void)fprintf(stderr, "skipping: arch lacks ansnr MSE SIMD\n");
#endif
    return NULL;
}
