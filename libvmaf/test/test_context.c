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

#include <errno.h>

#include "config.h"
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

static char *test_use_feature_invalid_options()
{
    VmafContext *vmaf;
    VmafConfiguration cfg = { 0 };
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("context initialization failed", !err);
    VmafFeatureDictionary *options = NULL;
    err = vmaf_feature_dictionary_set(&options, "motion_force_zero", "invalid");
    mu_assert("option dictionary creation failed", !err);
    err = vmaf_use_feature(vmaf, "motion", options);
    mu_assert("invalid option should be rejected", err == -EINVAL);
    /* The original dictionary was consumed after a successful copy. */
    options = NULL;
    err = vmaf_use_feature(vmaf, "motion", NULL);
    mu_assert("registration retry failed", !err);
    vmaf_close(vmaf);
    return NULL;
}

#if VMAF_BUILT_IN_MODELS
static char *test_use_model_invalid_options()
{
    VmafContext *vmaf;
    VmafConfiguration cfg = { 0 };
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("context initialization failed", !err);
    VmafModel *model;
    VmafModelConfig model_cfg = { 0 };
    err = vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("model load failed", !err);
    VmafFeatureDictionary *options = NULL;
    err = vmaf_feature_dictionary_set(&options, "motion_force_zero", "invalid");
    mu_assert("option dictionary creation failed", !err);
    err = vmaf_model_feature_overload(model, "motion", options);
    mu_assert("model override failed", !err);
    for (unsigned i = 0; i < 3; i++) {
        err = vmaf_use_features_from_model(vmaf, model);
        mu_assert("invalid model option should be rejected", err == -EINVAL);
    }
    options = NULL;
    err = vmaf_feature_dictionary_set(&options, "motion_force_zero", "true");
    mu_assert("retry options creation failed", !err);
    err = vmaf_model_feature_overload(model, "motion", options);
    mu_assert("model correction failed", !err);
    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("corrected model registration failed", !err);
    vmaf_close(vmaf);
    vmaf_model_destroy(model);
    return NULL;
}
#endif

char *run_tests()
{
    mu_run_test(test_context_init_and_close);
    mu_run_test(test_get_feature_score);
    mu_run_test(test_use_feature_invalid_options);
#if VMAF_BUILT_IN_MODELS
    mu_run_test(test_use_model_invalid_options);
#endif
    return NULL;
}
