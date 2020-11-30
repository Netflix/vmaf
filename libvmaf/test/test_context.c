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

#include "test.h"
#include "libvmaf/libvmaf.h"

static char *test_context_init_and_close()
{
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg = { 0 };

    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

static char *test_get_feature_score()
{
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg = { 0 };

    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);

    err = vmaf_import_feature_score(vmaf, "feature_a", 100., 0);
    err |= vmaf_import_feature_score(vmaf, "feature_a", 200., 1);
    err |= vmaf_import_feature_score(vmaf, "feature_a", 300., 2);
    mu_assert("problem during vmaf_import_feature_score", !err);

    double score;
    err = vmaf_feature_score_at_index(vmaf, "feature_a", &score, 0);
    mu_assert("problem during vmaf_feature_score_at_index", !err);
    mu_assert("retrieved feature score does not match", score == 100.);
    err = vmaf_feature_score_at_index(vmaf, "feature_a", &score, 1);
    mu_assert("problem during vmaf_feature_score_at_index", !err);
    mu_assert("retrieved feature score does not match", score == 200.);
    err = vmaf_feature_score_at_index(vmaf, "feature_a", &score, 2);
    mu_assert("problem during vmaf_feature_score_at_index", !err);
    mu_assert("retrieved feature score does not match", score == 300.);

    err = vmaf_feature_score_pooled(vmaf, "feature_a", VMAF_POOL_METHOD_MEAN,
                                    &score, 0, 2);
    mu_assert("problem during vmaf_feature_score_pooled", !err);
    mu_assert("pooled feature score does not match expected value",
              score == 200.);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_context_init_and_close);
    mu_run_test(test_get_feature_score);
    return NULL;
}
