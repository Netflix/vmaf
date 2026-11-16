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
#include <stdint.h>
#include <stdlib.h>

#include "config.h"
#include "cpu.h"
#include "test.h"
#include "libvmaf/picture.h"
#include "feature/feature_collector.h"
#include "feature/feature_extractor.h"
#include "feature/integer_motion.h"
#include "feature/motion_tools.h"
#include "feature/common/convolution.h"
#include "feature/common/convolution_internal.h"

/* The historical single-bounce formula, embedded verbatim for regression
 * comparison. This is deliberately duplicated: it is the pre-fix behavior,
 * and comparing against it over the full in-contract range is the only way
 * to mechanically prove the fix changed nothing for size >= 3. */
static int mirror_reference_pre_fix(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 2;
    return idx;
}

/* T1: for every size in the normal (in-contract) range, and every idx within
 * the 5-tap/radius-2 call contract, motion_mirror() must be bit-identical to
 * the old single-bounce formula. This proves the fix changes no output for
 * width/height >= 3. */
static char *test_motion_mirror_matches_pre_fix_for_normal_sizes()
{
    for (int size = 3; size <= 8192; size++) {
        for (int idx = -2; idx <= size + 1; idx++) {
            int got = motion_mirror(idx, size);
            int want = mirror_reference_pre_fix(idx, size);
            mu_assert("motion_mirror must match pre-fix formula for size >= 3",
                      got == want);
        }
    }
    return NULL;
}

/* T2: exact reflect-101 semantics for the sub-3 sizes the fix targets. */
static char *test_motion_mirror_sub3_sizes()
{
    for (int idx = -2; idx <= 3; idx++)
        mu_assert("size==1 must map every idx to 0", motion_mirror(idx, 1) == 0);

    static const int expected[][2] = {
        {-2, 0}, {-1, 1}, {0, 0}, {1, 1}, {2, 0}, {3, 1},
    };
    const int n = sizeof(expected) / sizeof(expected[0]);
    for (int i = 0; i < n; i++) {
        int idx = expected[i][0];
        int want = expected[i][1];
        mu_assert("size==2 reflect-101 mismatch", motion_mirror(idx, 2) == want);
    }

    return NULL;
}

// Host-side transcription of the CUDA mirror() formula (motion_score.cu), both
// pre- and post-fix, for comparison purposes only -- this is not compiled CUDA code
// (no local CUDA toolchain), just the identical arithmetic expression in C.
static int cuda_mirror_pre_fix(int idx, int sup)
{
    int out = abs(idx);
    return (out < sup) ? out : (sup - (out - sup + 1));
}

static int cuda_mirror_post_fix(int idx, int sup)
{
    if (sup == 1) return 0;
    int out = abs(idx);
    return (out < sup) ? out : (sup - (out - sup + 1));
}

// T3: host-side comparison of the CUDA mirror() formula, pre- and post-fix.
// For sup >= 2 the fix must be a no-op; for sup == 1 the post-fix version
// must degenerate safely to 0 (the pre-fix version is not asserted at
// sup == 1 since it is genuinely out of range there and is simply no longer
// called with sup == 1).
static char *test_cuda_mirror_formula_matches_pre_fix_for_sup_ge_2()
{
    for (int sup = 2; sup <= 8192; sup++) {
        for (int idx = -2; idx <= sup + 1; idx++) {
            int got = cuda_mirror_post_fix(idx, sup);
            int want = cuda_mirror_pre_fix(idx, sup);
            mu_assert("cuda mirror post-fix must match pre-fix formula for sup >= 2",
                      got == want);
        }
    }

    for (int idx = -2; idx <= 3; idx++)
        mu_assert("cuda mirror post-fix must map every idx to 0 for sup == 1",
                  cuda_mirror_post_fix(idx, 1) == 0);

    return NULL;
}

/* The historical single-bounce reflection from convolution_internal.h's
 * convolution_edge_*_s() functions, transcribed verbatim (the original code no
 * longer exists after the fix). Both branches -- horizontal on `width` and
 * vertical on `height` -- used this exact same arithmetic, so a single
 * reference is enough. */
static int convolution_mirror_reference_pre_fix(int tap, int size)
{
    if (tap < 0) return -tap;
    if (tap >= size) return size - (tap - size + 2);
    return tap;
}

/* T4: convolution_mirror() must be bit-identical to the pre-fix inline
 * reflection over the whole in-contract range, proving the float convolution
 * path's output is unchanged for width/height >= 3.
 *
 * The range covered here is the 5-tap / radius-2 contract this fix is
 * responsible for. convolution_internal.h is also instantiated with other
 * filter widths by other float features; those get the same guarantee
 * structurally, without a separate exhaustive loop: for any radius r and any
 * size >= r + 1, an in-contract tap lies in [-r, size - 1 + r], for which the
 * new loop takes at most one reflection and therefore evaluates the exact same
 * expression as the old single-bounce code. */
static char *test_convolution_mirror_matches_pre_fix_for_normal_sizes()
{
    for (int size = 3; size <= 8192; size++) {
        for (int tap = -2; tap <= size + 1; tap++) {
            int got = convolution_mirror(tap, size);
            int want = convolution_mirror_reference_pre_fix(tap, size);
            mu_assert("convolution_mirror must match pre-fix formula for size >= 3",
                      got == want);
        }
    }
    return NULL;
}

/* The tiny frame sizes the fix makes reachable. Every one of them has a
 * dimension below the 5-tap filter's radius of 2, which is exactly where the
 * old single-bounce reflection escaped the buffer. */
static const struct { unsigned w, h; } tiny_sizes[] = {
    {1, 1}, {1, 2}, {2, 1}, {2, 2}, {1, 8}, {8, 1}, {2, 8}, {8, 2},
};
static const unsigned n_tiny_sizes = sizeof(tiny_sizes) / sizeof(tiny_sizes[0]);

/* Both dispatch choices have to be exercised explicitly. A mask of 0 forces
 * the scalar kernels on every host; ~0u picks the best available, which is a
 * different kernel on an x86 CI runner than on arm64. Testing only the latter
 * would silently leave the scalar path uncovered wherever SIMD is present --
 * convolution_f32_c_s() in particular returns early into the AVX kernel as
 * soon as AVX2 is available, so its C fallback would never run there. */
static const unsigned cpu_masks[] = { 0, ~0u };
static const unsigned n_cpu_masks = sizeof(cpu_masks) / sizeof(cpu_masks[0]);

static void fill_random_picture(VmafPicture *pic, unsigned seed)
{
    srand(seed);
    const int max_val = 1 << pic->bpc;
    for (unsigned p = 0; p < 3; p++) {
        uint8_t *base = pic->data[p];
        for (unsigned i = 0; i < pic->h[p]; i++) {
            uint8_t *row = base + i * pic->stride[p];
            for (unsigned j = 0; j < pic->w[p]; j++) {
                if (pic->bpc == 8)
                    row[j] = (uint8_t)(rand() % max_val);
                else
                    ((uint16_t *)row)[j] = (uint16_t)(rand() % max_val);
            }
        }
    }
}

/* Drives a temporal feature extractor over two frames and reads back the score
 * `feature_name` produced for frame index 1. Modelled on
 * test_motion_neon.c's compute_motion_sad(). */
static int run_motion_fex(VmafFeatureExtractor *fex, const char *feature_name,
                          unsigned w, unsigned h, unsigned bpc,
                          unsigned cpu_mask, double *score)
{
    int err;
    vmaf_set_cpu_flags_mask(cpu_mask);

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, NULL);
    if (err) return err;

    VmafPicture prev_pic, cur_pic;
    err = vmaf_picture_alloc(&prev_pic, VMAF_PIX_FMT_YUV420P, bpc, w, h);
    if (err) return err;
    err = vmaf_picture_alloc(&cur_pic, VMAF_PIX_FMT_YUV420P, bpc, w, h);
    if (err) return err;

    fill_random_picture(&prev_pic, 100);
    fill_random_picture(&cur_pic, 200);

    VmafFeatureCollector *vfc;
    err = vmaf_feature_collector_init(&vfc);
    if (err) return err;

    err = vmaf_feature_extractor_context_extract(fex_ctx, &prev_pic, NULL,
                                                 &prev_pic, NULL, 0, vfc);
    if (err) return err;

    if (fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_PREV_REF)
        fex_ctx->fex->prev_ref = prev_pic;

    err = vmaf_feature_extractor_context_extract(fex_ctx, &cur_pic, NULL,
                                                 &cur_pic, NULL, 1, vfc);
    if (err) return err;

    err = vmaf_feature_collector_get_score(vfc, feature_name, score, 1);
    if (err) return err;

    err = vmaf_feature_extractor_context_close(fex_ctx);
    if (err) return err;
    err = vmaf_feature_extractor_context_destroy(fex_ctx);
    if (err) return err;

    vmaf_feature_collector_destroy(vfc);
    vmaf_picture_unref(&prev_pic);
    vmaf_picture_unref(&cur_pic);

    return 0;
}

/* The integer `motion` extractor must run to completion at frame sizes smaller
 * than the filter radius, touching only valid memory (the real teeth of this
 * test come from running it under ASan) and producing a well-defined,
 * repeatable score. There is no meaningful golden value at 1x1, so what is
 * asserted is: the call succeeds, the score is finite, and two identical runs
 * agree bit-for-bit. Both the 8 bpc and the 16 bpc pipeline are covered, and
 * both the scalar and the best-available SIMD dispatch.
 *
 * On top of the per-mask determinism check, every mask's score is compared
 * against the cpu_masks[0] (== 0, forced scalar) score for the same
 * (size, bpc). That cross-mask equality is what gives the x86 AVX2/AVX-512
 * motion kernels their only runtime coverage for this fix: upstream CI runs
 * `meson test` on an x86_64 runner, where the ~0u pass dispatches to those
 * kernels and must land on exactly the same SAD as the C reference. */
static char *test_integer_motion_pipeline_tiny_sizes()
{
    static const unsigned bpcs[] = { 8, 10, 12, 16 };
    const unsigned n_bpcs = sizeof(bpcs) / sizeof(bpcs[0]);

    vmaf_init_cpu();

    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("motion");
    mu_assert("the integer motion feature extractor must exist", fex);

    for (unsigned s = 0; s < n_tiny_sizes; s++) {
        for (unsigned b = 0; b < n_bpcs; b++) {
            double scalar_score = -1.;

            for (unsigned c = 0; c < n_cpu_masks; c++) {
                double score_a = -1., score_b = -1.;
                int err;

                err = run_motion_fex(fex,
                        "VMAF_integer_feature_motion_sad_score",
                        tiny_sizes[s].w, tiny_sizes[s].h, bpcs[b],
                        cpu_masks[c], &score_a);
                mu_assert("integer motion extraction must succeed at tiny sizes",
                          !err);
                mu_assert("integer motion score must be finite at tiny sizes",
                          isfinite(score_a));

                err = run_motion_fex(fex,
                        "VMAF_integer_feature_motion_sad_score",
                        tiny_sizes[s].w, tiny_sizes[s].h, bpcs[b],
                        cpu_masks[c], &score_b);
                mu_assert("integer motion re-run must succeed at tiny sizes",
                          !err);
                mu_assert("integer motion score must be deterministic",
                          score_a == score_b);

                /* cpu_masks[0] is 0, i.e. the forced-scalar reference. */
                if (c == 0)
                    scalar_score = score_a;
                else
                    mu_assert("integer motion score must not depend on CPU dispatch",
                              score_a == scalar_score);
            }
        }
    }

    vmaf_set_cpu_flags_mask(~0u);
    return NULL;
}

/* Relative equality with a small floor, for the float path's cross-dispatch
 * comparison. See the comment on test_float_motion_pipeline_tiny_sizes() for
 * why that one comparison cannot be exact. */
static int motion_scores_close(double a, double b)
{
    const double diff = fabs(a - b);
    const double scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return diff <= 1e-6 * scale + 1e-9;
}

/* The same pipeline-safety and determinism check for the float `motion` path,
 * which reflects through convolution_mirror() instead. Skipped when the
 * library was configured with -Denable_float=false, in which case the
 * extractor simply is not registered.
 *
 * Both CPU masks matter here: convolution_f32_c_s() hands off to
 * convolution_f32_avx_s() whenever AVX2 is available and returns, so on an x86
 * runner the ~0u pass covers only the AVX kernel and the scalar
 * convolution_y_c_s()/convolution_x_c_s() pair -- the other half of the
 * boundary fix -- would go untouched without the cpu_mask == 0 pass.
 *
 * Every mask's score is also cross-checked against the cpu_masks[0] (forced
 * scalar) score for the same (size, bpc), so that a dispatch-dependent
 * boundary regression cannot hide on an x86 runner. Unlike the integer path,
 * this comparison is a tolerance and not `==`: convolution_avx.c is built with
 * -mfma while convolution.c is not, so the compiler contracts the shared
 * convolution_edge_s() accumulation into fused multiply-adds in the AVX
 * translation unit only. The two float results are therefore legitimately
 * allowed to differ in the last bits. A reflection bug changes the score by
 * O(1), which motion_scores_close() still catches by a wide margin. */
static char *test_float_motion_pipeline_tiny_sizes()
{
    vmaf_init_cpu();

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name("float_motion");
    if (!fex) return NULL; /* float features disabled at build time */

    static const unsigned bpcs[] = { 8, 10, 16 };
    const unsigned n_bpcs = sizeof(bpcs) / sizeof(bpcs[0]);

    for (unsigned s = 0; s < n_tiny_sizes; s++) {
        for (unsigned b = 0; b < n_bpcs; b++) {
            double scalar_score = -1.;

            for (unsigned c = 0; c < n_cpu_masks; c++) {
                double score_a = -1., score_b = -1.;
                int err;

                err = run_motion_fex(fex, "VMAF_feature_motion_score",
                                     tiny_sizes[s].w, tiny_sizes[s].h, bpcs[b],
                                     cpu_masks[c], &score_a);
                mu_assert("float motion extraction must succeed at tiny sizes",
                          !err);
                mu_assert("float motion score must be finite at tiny sizes",
                          isfinite(score_a));

                err = run_motion_fex(fex, "VMAF_feature_motion_score",
                                     tiny_sizes[s].w, tiny_sizes[s].h, bpcs[b],
                                     cpu_masks[c], &score_b);
                mu_assert("float motion re-run must succeed at tiny sizes",
                          !err);
                mu_assert("float motion score must be deterministic",
                          score_a == score_b);

                /* cpu_masks[0] is 0, i.e. the forced-scalar reference. */
                if (c == 0)
                    scalar_score = score_a;
                else
                    mu_assert("float motion score must not depend on CPU dispatch",
                              motion_scores_close(score_a, scalar_score));
            }
        }
    }

    vmaf_set_cpu_flags_mask(~0u);
    return NULL;
}

/* ------------------------------------------------------------------------ *
 * Canary-buffer write-bounds tests.
 *
 * The boundary fix in convolution.c / convolution_avx.c is a fix for an
 * out-of-bounds WRITE, not merely an out-of-bounds read: at width or height
 * below the filter width the old borders_left/borders_right (and the AVX
 * i_vec_end/j_vec_end) computations went negative, so the trailing edge loop
 * started at a negative index and stored outside dst. A sanitizer catches
 * that, but upstream CI runs no sanitizer job. These tests catch it on any
 * host, with no tooling, by surrounding every destination buffer with guard
 * elements holding a sentinel and asserting the sentinel survives.
 * ------------------------------------------------------------------------ */

/* A value the convolutions below cannot produce from the test inputs. Written
 * element by element rather than memset so the exact bit pattern is known and
 * can be compared with ==. */
#define CANARY_VALUE (-123456.789f)

/* Guard elements placed before and after every real destination region. The
 * pre-fix code overran by at most a couple of elements in either direction, so
 * this is comfortably wide. */
#define CANARY_GUARD (8)

static void canary_fill(float *buf, int n)
{
    for (int i = 0; i < n; i++)
        buf[i] = CANARY_VALUE;
}

static int canary_intact(const float *buf, int n)
{
    for (int i = 0; i < n; i++) {
        if (buf[i] != CANARY_VALUE)
            return 0;
    }
    return 1;
}

/* Source pixels, big enough for every case below. Values are arbitrary; only
 * the destination bounds are under test. */
static const float canary_src[8] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

extern void convolution_x_c_s(const float *filter, int filter_width,
                              const float *src, float *dst, int width,
                              int height, int src_stride, int dst_stride,
                              int step);
extern void convolution_y_c_s(const float *filter, int filter_width,
                              const float *src, float *dst, int width,
                              int height, int src_stride, int dst_stride,
                              int step);

/* T7: the scalar convolution kernels must not write outside dst at the tiny
 * dimensions the fix targets. Runs everywhere, on every host and every CI job.
 *
 * The swept dimension goes 1..4, i.e. everything strictly below the 5-tap
 * filter width, which is exactly the range where the unclamped
 * borders_left/borders_right went negative. The other dimension is held small
 * and fixed so that the real dst region is exactly packed and any overrun
 * necessarily lands in a guard rather than in unused stride padding. */
static char *test_convolution_scalar_write_bounds()
{
    /* Horizontal pass: height == 1, so dst is exactly `width` floats. */
    for (int width = 1; width <= 4; width++) {
        float buf[CANARY_GUARD + 4 + CANARY_GUARD];
        const int n = (int)(sizeof(buf) / sizeof(buf[0]));

        canary_fill(buf, n);
        convolution_x_c_s(FILTER_5_s, 5, canary_src, buf + CANARY_GUARD,
                          width, 1, width, width, 1);

        mu_assert("convolution_x_c_s wrote before dst",
                  canary_intact(buf, CANARY_GUARD));
        mu_assert("convolution_x_c_s wrote past dst",
                  canary_intact(buf + CANARY_GUARD + width,
                                n - CANARY_GUARD - width));
    }

    /* Vertical pass: width == 2, so dst is exactly height*2 floats. */
    for (int height = 1; height <= 4; height++) {
        const int width = 2;
        float buf[CANARY_GUARD + 4 * 2 + CANARY_GUARD];
        const int n = (int)(sizeof(buf) / sizeof(buf[0]));
        const int real = width * height;

        canary_fill(buf, n);
        convolution_y_c_s(FILTER_5_s, 5, canary_src, buf + CANARY_GUARD,
                          width, height, width, width, 1);

        mu_assert("convolution_y_c_s wrote before dst",
                  canary_intact(buf, CANARY_GUARD));
        mu_assert("convolution_y_c_s wrote past dst",
                  canary_intact(buf + CANARY_GUARD + real,
                                n - CANARY_GUARD - real));
    }

    return NULL;
}

#if ARCH_X86
/* AVX_STEP is file-local to convolution_avx.c and the header exposes no
 * accessor, so its value is duplicated here. The AVX kernels size their
 * caller-supplied tmp scratch as vmaf_ceiln(width, AVX_STEP) floats per row. */
#define TEST_AVX_STEP (8)

static int canary_ceiln(int n, int m)
{
    return n % m ? n + (m - n % m) : n;
}

/* convolution.h requires every array argument to be 32-byte aligned. */
static float *canary_align32(float *p)
{
    uintptr_t a = (uintptr_t)p;
    a = (a + 31u) & ~(uintptr_t)31u;
    return (float *)a;
}

/* Runs all three AVX convolution kernels for one (width, height) with both the
 * tmp scratch and dst guarded, and verifies every guard survives.
 *
 * width and height stay <= 4 here, which keeps every vectorised loop empty:
 * vmaf_floorn(width - 2, 8) and vmaf_floorn(width, 8) are both 0, and the
 * vertical vector loop runs over [radius, max(height - radius, ...)) which is
 * empty for height <= 4. So only the edge loops -- the ones the clamps fix --
 * execute, and the aligned loads/stores in the scanline helpers are never
 * reached (which is why the packed, stride == width source layout below is
 * safe even though it does not 32-byte align individual rows). */
static char *avx_write_bounds_case(int width, int height)
{
    static float raw_src1[128], raw_src2[128], raw_tmp[128], raw_dst[128];

    const int tmp_stride = canary_ceiln(width, TEST_AVX_STEP);
    const int tmp_real = height * tmp_stride;
    const int tmp_len = CANARY_GUARD + tmp_real + CANARY_GUARD;
    const int dst_stride = width;
    const int dst_real = height * dst_stride;
    const int dst_len = CANARY_GUARD + dst_real + CANARY_GUARD;

    float *src1 = canary_align32(raw_src1);
    float *src2 = canary_align32(raw_src2);
    float *tmp_base = canary_align32(raw_tmp);
    float *dst_base = canary_align32(raw_dst);
    float *tmp = tmp_base + CANARY_GUARD;
    float *dst = dst_base + CANARY_GUARD;

    for (int i = 0; i < width * height; i++) {
        src1[i] = (float)(i + 1);
        src2[i] = (float)(width * height - i);
    }

#define AVX_CANARY_CHECK(name)                                                \
    do {                                                                      \
        mu_assert(name " wrote before tmp",                                   \
                  canary_intact(tmp_base, CANARY_GUARD));                     \
        mu_assert(name " wrote past tmp",                                     \
                  canary_intact(tmp_base + CANARY_GUARD + tmp_real,           \
                                CANARY_GUARD));                               \
        mu_assert(name " wrote before dst",                                   \
                  canary_intact(dst_base, CANARY_GUARD));                     \
        mu_assert(name " wrote past dst",                                     \
                  canary_intact(dst_base + CANARY_GUARD + dst_real,           \
                                CANARY_GUARD));                               \
    } while (0)

    canary_fill(tmp_base, tmp_len);
    canary_fill(dst_base, dst_len);
    convolution_f32_avx_s(FILTER_5_s, 5, src1, dst, tmp, width, height,
                          width, dst_stride);
    AVX_CANARY_CHECK("convolution_f32_avx_s");

    canary_fill(tmp_base, tmp_len);
    canary_fill(dst_base, dst_len);
    convolution_f32_avx_sq_s(FILTER_5_s, 5, src1, dst, tmp, width, height,
                             width, dst_stride);
    AVX_CANARY_CHECK("convolution_f32_avx_sq_s");

    canary_fill(tmp_base, tmp_len);
    canary_fill(dst_base, dst_len);
    convolution_f32_avx_xy_s(FILTER_5_s, 5, src1, src2, dst, tmp, width,
                             height, width, width, dst_stride);
    AVX_CANARY_CHECK("convolution_f32_avx_xy_s");

#undef AVX_CANARY_CHECK

    return NULL;
}
#endif /* ARCH_X86 */

/* T8: the AVX convolution kernels must not write outside tmp or dst at tiny
 * dimensions. x86-only: convolution_avx.c is compiled (and ARCH_X86 defined)
 * only for x86 with asm enabled, so this is inert on every other target. */
static char *test_convolution_avx_write_bounds()
{
#if ARCH_X86
    for (int width = 1; width <= 4; width++) {
        char *msg = avx_write_bounds_case(width, 1);
        if (msg) return msg;
    }
    for (int height = 1; height <= 4; height++) {
        char *msg = avx_write_bounds_case(2, height);
        if (msg) return msg;
    }
#endif
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_motion_mirror_matches_pre_fix_for_normal_sizes);
    mu_run_test(test_motion_mirror_sub3_sizes);
    mu_run_test(test_cuda_mirror_formula_matches_pre_fix_for_sup_ge_2);
    mu_run_test(test_convolution_mirror_matches_pre_fix_for_normal_sizes);
    mu_run_test(test_integer_motion_pipeline_tiny_sizes);
    mu_run_test(test_float_motion_pipeline_tiny_sizes);
    mu_run_test(test_convolution_scalar_write_bounds);
    mu_run_test(test_convolution_avx_write_bounds);
    return NULL;
}
