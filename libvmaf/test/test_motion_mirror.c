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

#include "test.h"
#include "feature/integer_motion.h"

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

char *run_tests()
{
    mu_run_test(test_motion_mirror_matches_pre_fix_for_normal_sizes);
    mu_run_test(test_motion_mirror_sub3_sizes);
    return NULL;
}
