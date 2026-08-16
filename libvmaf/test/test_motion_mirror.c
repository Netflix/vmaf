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

#include <stdlib.h>

#include "test.h"
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

char *run_tests()
{
    mu_run_test(test_motion_mirror_matches_pre_fix_for_normal_sizes);
    mu_run_test(test_motion_mirror_sub3_sizes);
    mu_run_test(test_cuda_mirror_formula_matches_pre_fix_for_sup_ge_2);
    mu_run_test(test_convolution_mirror_matches_pre_fix_for_normal_sizes);
    return NULL;
}
