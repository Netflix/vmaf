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

#include <errno.h>
#include <limits.h>
#include <math.h>

#include "test.h"
#include "model.h"
#include "libvmaf/libvmaf.h"

static const enum VmafPoolingMethod methods[] = {
    VMAF_POOL_METHOD_PERC5, VMAF_POOL_METHOD_PERC10,
    VMAF_POOL_METHOD_PERC20, VMAF_POOL_METHOD_MEDIAN,
};

static char *test_interpolated_percentiles(void)
{
    VmafContext *vmaf = NULL;
    VmafConfiguration cfg = { 0 };
    mu_assert("init failed", !vmaf_init(&vmaf, cfg));
    const double values[] = { 4., 1., 3., 2. };
    const double expected[] = { 1.15, 1.3, 1.6, 2.5 };
    for (unsigned i = 0; i < 4; i++)
        mu_assert("import failed", !vmaf_import_feature_score(vmaf, "f", values[i], i));
    for (unsigned i = 0; i < 4; i++) {
        double score = 0.;
        mu_assert("percentile pooling failed",
                  !vmaf_feature_score_pooled(vmaf, "f", methods[i], &score, 0, 3));
        mu_assert("wrong linear interpolation", fabs(score - expected[i]) < 1e-12);
    }
    for (unsigned i = 0; i < 4; i++) {
        double score;
        mu_assert("lookup failed", !vmaf_feature_score_at_index(vmaf, "f", &score, i));
        mu_assert("pooling reordered stored scores", score == values[i]);
    }
    double score;
    mu_assert("legacy mean failed",
              !vmaf_feature_score_pooled(vmaf, "f", VMAF_POOL_METHOD_MEAN, &score, 0, 3));
    mu_assert("legacy mean changed", score == 2.5);
    mu_assert("close failed", !vmaf_close(vmaf));
    return NULL;
}

static char *test_singleton_ties_and_negative_scores(void)
{
    VmafContext *vmaf = NULL;
    VmafConfiguration cfg = { 0 };
    mu_assert("init failed", !vmaf_init(&vmaf, cfg));
    for (unsigned i = 0; i < 3; i++)
        mu_assert("import failed", !vmaf_import_feature_score(vmaf, "f", -7., i));
    for (unsigned i = 0; i < 4; i++) {
        double score;
        mu_assert("singleton failed",
                  !vmaf_feature_score_pooled(vmaf, "f", methods[i], &score, 1, 1));
        mu_assert("singleton changed", score == -7.);
        mu_assert("ties failed",
                  !vmaf_feature_score_pooled(vmaf, "f", methods[i], &score, 0, 2));
        mu_assert("ties changed", fabs(score + 7.) < 1e-12);
    }
    mu_assert("close failed", !vmaf_close(vmaf));
    return NULL;
}

static char *test_subsampled_interval(void)
{
    VmafContext *vmaf = NULL;
    VmafConfiguration cfg = { .n_subsample = 2 };
    mu_assert("init failed", !vmaf_init(&vmaf, cfg));
    const double values[] = { 100., -100., 2., -100., 4., -100., 200. };
    for (unsigned i = 0; i < 7; i++)
        mu_assert("import failed", !vmaf_import_feature_score(vmaf, "f", values[i], i));
    double score;
    mu_assert("subsample pooling failed",
              !vmaf_feature_score_pooled(vmaf, "f", VMAF_POOL_METHOD_MEDIAN, &score, 1, 5));
    mu_assert("wrong selected frames", score == 3.);
    score = 123.;
    mu_assert("empty selected interval accepted",
              vmaf_feature_score_pooled(vmaf, "f", VMAF_POOL_METHOD_MEDIAN, &score, 1, 1)
              == -EINVAL);
    mu_assert("failed pooling overwrote output", score == 123.);
    mu_assert("close failed", !vmaf_close(vmaf));
    return NULL;
}

static char *test_missing_and_invalid_samples(void)
{
    VmafContext *vmaf = NULL;
    VmafConfiguration cfg = { 0 };
    mu_assert("init failed", !vmaf_init(&vmaf, cfg));
    mu_assert("import failed", !vmaf_import_feature_score(vmaf, "f", 1., 0));
    mu_assert("import failed", !vmaf_import_feature_score(vmaf, "f", 3., 2));
    double score = 123.;
    mu_assert("missing frame accepted",
              vmaf_feature_score_pooled(vmaf, "f", VMAF_POOL_METHOD_PERC10, &score, 0, 2));
    mu_assert("failed pooling overwrote output", score == 123.);
    mu_assert("NULL output accepted",
              vmaf_feature_score_pooled(vmaf, "f", VMAF_POOL_METHOD_PERC10, NULL, 0, 0)
              == -EINVAL);
    mu_assert("reversed interval accepted",
              vmaf_feature_score_pooled(vmaf, "f", VMAF_POOL_METHOD_PERC10, &score, 2, 0)
              == -EINVAL);
    const double bad[] = { NAN, INFINITY, -INFINITY };
    for (unsigned i = 0; i < 3; i++) {
        mu_assert("import failed", !vmaf_import_feature_score(vmaf, "bad", bad[i], i));
        mu_assert("non-finite score accepted",
                  vmaf_feature_score_pooled(vmaf, "bad", VMAF_POOL_METHOD_PERC10, &score, i, i)
                  == -EINVAL);
        mu_assert("failed pooling overwrote output", score == 123.);
    }
    mu_assert("large endpoint accepted without a score",
              vmaf_feature_score_pooled(vmaf, "f", VMAF_POOL_METHOD_PERC10,
                                        &score, UINT_MAX, UINT_MAX));
    mu_assert("close failed", !vmaf_close(vmaf));
    return NULL;
}

static char *test_model_pooling_uses_percentile(void)
{
    VmafContext *vmaf = NULL;
    VmafConfiguration cfg = { 0 };
    mu_assert("init failed", !vmaf_init(&vmaf, cfg));
    /* Cached model scores exercise the public model-pooling delegation
     * without introducing model/feature numerics into a pooling test. */
    VmafModel model = { .name = "cached_model" };
    for (unsigned i = 0; i < 4; i++)
        mu_assert("import failed",
                  !vmaf_import_feature_score(vmaf, model.name, i + 1., i));
    double score;
    mu_assert("model percentile pooling failed",
              !vmaf_score_pooled(vmaf, &model, VMAF_POOL_METHOD_PERC10, &score, 0, 3));
    mu_assert("model percentile differs from feature pooling", fabs(score - 1.3) < 1e-12);
    mu_assert("close failed", !vmaf_close(vmaf));
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_interpolated_percentiles);
    mu_run_test(test_singleton_ties_and_negative_scores);
    mu_run_test(test_subsampled_interval);
    mu_run_test(test_missing_and_invalid_samples);
    mu_run_test(test_model_pooling_uses_percentile);
    return NULL;
}
