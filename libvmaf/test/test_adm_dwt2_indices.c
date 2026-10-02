/**
 *
 *  Copyright 2026 Lusoris
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
#include "feature/integer_adm.h"

#define MAX_N 66
#define MAX_N_HALF ((MAX_N + 1) / 2)

/* Symmetric extension of tap position `idx` into an input of `n` samples. */
static int mirror(int idx, int n)
{
    if (idx < 0) return -idx;
    if (idx >= n) return 2 * n - idx - 1;
    return idx;
}

static int indices_match(int **ind, int n)
{
    for (int i = 0; i < (n + 1) / 2; i++) {
        for (int k = 0; k < 4; k++) {
            if (ind[k][i] != mirror(2 * i - 1 + k, n)) {
                fprintf(stderr, "n=%d output=%d tap=%d: index %d, expected %d\n",
                        n, i, k, ind[k][i], mirror(2 * i - 1 + k, n));
                return 0;
            }
        }
    }
    return 1;
}

/* A 3- or 4-sample input is what scale 3 sees for frame dimensions 17 to 32. */
static char *test_dwt2_src_indices_stay_inside_input()
{
    static int y[4][MAX_N_HALF], x[4][MAX_N_HALF];
    int *ind_y[4] = { y[0], y[1], y[2], y[3] };
    int *ind_x[4] = { x[0], x[1], x[2], x[3] };

    for (int n = 3; n <= MAX_N; n++) {
        /* Pair every width with a height on the other side of the range. */
        const int w = n, h = MAX_N + 3 - n;
        dwt2_src_indices_filt(ind_y, ind_x, w, h);
        mu_assert("horizontal dwt2 source index outside the input",
                  indices_match(ind_x, w));
        mu_assert("vertical dwt2 source index outside the input",
                  indices_match(ind_y, h));
    }

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_dwt2_src_indices_stay_inside_input);
    return NULL;
}
