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

#include <stdint.h>
#include "test.h"
#include "model.c"

static char *test_model_load_and_destroy()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = {
        .path = "../../model/vmaf_v0.6.1.pkl",
    };
    err = vmaf_model_load_from_path(&model, &cfg);
    mu_assert("problem during vmaf_model_load_from_path", !err);

    /*
    for (unsigned i = 0; i < model->n_features; i++)
        fprintf(stderr, "feature name: %s slope: %f intercept: %f\n",
                model->feature[i].name,
                model->feature[i].slope,
                model->feature[i].intercept
        );
    */

    vmaf_model_destroy(model);

    return NULL;
}

static char *test_model_check_default_behavior_unset_flags()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .name = "some_vmaf",
    };
    err = vmaf_model_load_from_path(&model, &cfg);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Model name is inconsistent.\n", !strcmp(model->name, "some_vmaf"));
    mu_assert("Clipping must be enabled by default.\n", model->score_clip.enabled);
    mu_assert("Score transform must be disabled by default.\n", !model->score_transform.enabled);
    /* TODO: add check for confidence interval */
    mu_assert("Feature 0 name must be VMAF_feature_adm2_score.\n", !strcmp(model->feature[0].name, "'VMAF_feature_adm2_score'"));

    vmaf_model_destroy(model);

    return NULL;
}

static char *test_model_check_default_behavior_set_flags()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .name = "some_vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    err = vmaf_model_load_from_path(&model, &cfg);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Model name is inconsistent.\n", !strcmp(model->name, "some_vmaf"));
    mu_assert("Clipping must be enabled by default.\n", model->score_clip.enabled);
    mu_assert("Score transform must be disabled by default.\n", !model->score_transform.enabled);
    /* TODO: add check for confidence interval */
    mu_assert("Feature 0 name must be VMAF_feature_adm2_score.\n", !strcmp(model->feature[0].name, "'VMAF_feature_adm2_score'"));

    vmaf_model_destroy(model);

    return NULL;
}

static char *test_model_set_flags()
{
    int err;

    VmafModel *model1;
    VmafModelConfig cfg1 = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .flags = VMAF_MODEL_FLAG_ENABLE_TRANSFORM,
    };
    err = vmaf_model_load_from_path(&model1, &cfg1);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Score transform must be enabled.\n",
              model1->score_transform.enabled);
    mu_assert("Clipping must be enabled.\n",
              model1->score_clip.enabled);
    vmaf_model_destroy(model1);

    VmafModel *model2;
    VmafModelConfig cfg2 = {
        .path = "../../model/vmaf_v0.6.1.pkl",
        .flags = VMAF_MODEL_FLAG_DISABLE_CLIP,
    };
    err = vmaf_model_load_from_path(&model2, &cfg2);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Score transform must be disabled.\n",
              !model2->score_transform.enabled);
    mu_assert("Clipping must be disabled.\n",
              !model2->score_clip.enabled);
    vmaf_model_destroy(model2);

    VmafModel  *model3;
    VmafModelConfig  cfg3 = {
            .path = "../../model/vmaf_v0.6.1.pkl",
    };
    err = vmaf_model_load_from_path(&model3, &cfg3);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("feature[0].opts_dict must be NULL.\n",
              !model3->feature[0].opts_dict);
    mu_assert("feature[1].opts_dict must be NULL.\n",
              !model3->feature[1].opts_dict);
    mu_assert("feature[2].opts_dict must be NULL.\n",
              !model3->feature[2].opts_dict);
    mu_assert("feature[3].opts_dict must be NULL.\n",
              !model3->feature[3].opts_dict);
    mu_assert("feature[4].opts_dict must be NULL.\n",
              !model3->feature[4].opts_dict);
    mu_assert("feature[5].opts_dict must be NULL.\n",
              !model3->feature[5].opts_dict);

    VmafModel  *model4;
    VmafModelConfig  cfg4 = {
            .path = "../../model/vmaf_v0.6.1neg.pkl",
    };
    err = vmaf_model_load_from_path(&model4, &cfg4);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("feature[0].opts_dict must not be NULL.\n",
              model4->feature[0].opts_dict);
    mu_assert("feature[1].opts_dict must be NULL.\n",
              !model4->feature[1].opts_dict);
    mu_assert("feature[2].opts_dict must not be NULL.\n",
              model4->feature[2].opts_dict);
    mu_assert("feature[3].opts_dict must not be NULL.\n",
              model4->feature[3].opts_dict);
    mu_assert("feature[4].opts_dict must not be NULL.\n",
              model4->feature[4].opts_dict);
    mu_assert("feature[5].opts_dict must not be NULL.\n",
              model4->feature[5].opts_dict);

    VmafDictionaryEntry *entry = NULL;
    entry = vmaf_dictionary_get(&model4->feature[0].opts_dict, "adm_enhn_gain_limit", 0);
    mu_assert("feature[0].opts_dict must have key adm_enhn_gain_limit.\n",
              strcmp(entry->key, "adm_enhn_gain_limit")==0);
    mu_assert("feature[0].opts_dict[\"adm_enhn_gain_limit\"] must have value 1.0.\n",
              strcmp(entry->val, "1.0")==0);
    entry = vmaf_dictionary_get(&model4->feature[2].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[2].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit")==0);
    mu_assert("feature[2].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.0.\n",
              strcmp(entry->val, "1.0")==0);
    entry = vmaf_dictionary_get(&model4->feature[3].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[3].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit")==0);
    mu_assert("feature[3].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.0.\n",
              strcmp(entry->val, "1.0")==0);
    entry = vmaf_dictionary_get(&model4->feature[4].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[4].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit")==0);
    mu_assert("feature[4].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.0.\n",
              strcmp(entry->val, "1.0")==0);
    entry = vmaf_dictionary_get(&model4->feature[5].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[5].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit")==0);
    mu_assert("feature[5].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.0.\n",
              strcmp(entry->val, "1.0")==0);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_model_load_and_destroy);
    mu_run_test(test_model_check_default_behavior_unset_flags);
    mu_run_test(test_model_check_default_behavior_set_flags);
    mu_run_test(test_model_set_flags);
    return NULL;
}
