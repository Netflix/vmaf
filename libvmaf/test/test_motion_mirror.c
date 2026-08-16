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

#include "cpu.h"
#include "test.h"
#include "libvmaf/picture.h"
#include "feature/feature_collector.h"
#include "feature/feature_extractor.h"
#include "feature/integer_motion.h"
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

/* T5: the integer `motion` extractor must run to completion at frame sizes
 * smaller than the filter radius, touching only valid memory (the real teeth
 * of this test come from running it under ASan) and producing a well-defined,
 * repeatable score. There is no meaningful golden value at 1x1, so what is
 * asserted is: the call succeeds, the score is finite, and two identical runs
 * agree bit-for-bit. Both the 8 bpc and the 16 bpc pipeline are covered, and
 * both the scalar and the best-available SIMD dispatch. */
static char *test_integer_motion_pipeline_tiny_sizes()
{
    static const unsigned bpcs[] = { 8, 10, 12, 16 };
    const unsigned n_bpcs = sizeof(bpcs) / sizeof(bpcs[0]);
    static const unsigned cpu_masks[] = { 0, ~0u };
    const unsigned n_cpu_masks = sizeof(cpu_masks) / sizeof(cpu_masks[0]);

    vmaf_init_cpu();

    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("motion");
    mu_assert("the integer motion feature extractor must exist", fex);

    for (unsigned s = 0; s < n_tiny_sizes; s++) {
        for (unsigned b = 0; b < n_bpcs; b++) {
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
            }
        }
    }

    vmaf_set_cpu_flags_mask(~0u);
    return NULL;
}

/* T6: the same pipeline-safety and determinism check for the float `motion`
 * path, which reflects through convolution_mirror() instead. Skipped when the
 * library was configured with -Denable_float=false, in which case the
 * extractor simply is not registered. */
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
            double score_a = -1., score_b = -1.;
            int err;

            err = run_motion_fex(fex, "VMAF_feature_motion_score",
                                 tiny_sizes[s].w, tiny_sizes[s].h, bpcs[b],
                                 ~0u, &score_a);
            mu_assert("float motion extraction must succeed at tiny sizes",
                      !err);
            mu_assert("float motion score must be finite at tiny sizes",
                      isfinite(score_a));

            err = run_motion_fex(fex, "VMAF_feature_motion_score",
                                 tiny_sizes[s].w, tiny_sizes[s].h, bpcs[b],
                                 ~0u, &score_b);
            mu_assert("float motion re-run must succeed at tiny sizes", !err);
            mu_assert("float motion score must be deterministic",
                      score_a == score_b);
        }
    }

    vmaf_set_cpu_flags_mask(~0u);
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
    return NULL;
}
