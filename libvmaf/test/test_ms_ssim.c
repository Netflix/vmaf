/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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
#include <stdlib.h>

#include "test.h"
#include "feature/ms_ssim.h"

#define W 176
#define H 176

/* A 1-pixel checkerboard against its inverse: the two frames are
 * anti-correlated, so the structure term of the first scale is negative. */
static char *test_ms_ssim_anticorrelated_is_finite()
{
    const int stride = W * sizeof(float);
    float *ref = malloc((size_t)stride * H);
    float *cmp = malloc((size_t)stride * H);
    mu_assert("allocation failed", ref && cmp);

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            const float v = ((i + j) & 1) ? 255.f : 0.f;
            ref[i * W + j] = v;
            cmp[i * W + j] = 255.f - v;
        }
    }

    double score = 0., l[5], c[5], s[5];
    const int err = compute_ms_ssim(ref, cmp, W, H, stride, stride, &score, l, c, s);
    free(ref);
    free(cmp);

    mu_assert("compute_ms_ssim failed", err == 0);
    mu_assert("the first scale should be anti-correlated", s[0] < 0.);
    mu_assert("ms_ssim of anti-correlated frames must not be NaN", !isnan(score));
    mu_assert("ms_ssim must lie in [0, 1]", score >= 0. && score <= 1.);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_ms_ssim_anticorrelated_is_finite);
    return NULL;
}
