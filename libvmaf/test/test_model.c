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
#include "read_json_model.h"

static int model_compare(VmafModel *model_a, VmafModel *model_b)
{
    int err = 0;

   //err += model_a->type != model_b->type;

    err += model_a->slope != model_b->slope;
    err += model_a->intercept != model_b->intercept;

    err += model_a->n_features != model_a->n_features;
    for (unsigned i = 0; i < model_a->n_features; i++) {
       //err += strcmp(model_a->feature[i].name, model_b->feature[i].name) != 0;
       err += model_a->feature[i].slope != model_b->feature[i].slope;
       err += model_a->feature[i].intercept != model_b->feature[i].intercept;
       err += !model_a->feature[i].opts_dict != !model_b->feature[i].opts_dict;
    }

    err += model_a->score_clip.enabled != model_b->score_clip.enabled;
    err += model_a->score_clip.min != model_b->score_clip.min;
    err += model_a->score_clip.max != model_b->score_clip.max;

    err += model_a->norm_type != model_b->norm_type;

    err += model_a->score_transform.enabled != model_b->score_transform.enabled;
    err += model_a->score_transform.p0.enabled != model_b->score_transform.p0.enabled;
    err += model_a->score_transform.p0.value != model_b->score_transform.p0.value;
    err += model_a->score_transform.p1.enabled != model_b->score_transform.p1.enabled;
    err += model_a->score_transform.p1.value != model_b->score_transform.p1.value;
    err += model_a->score_transform.p2.enabled != model_b->score_transform.p2.enabled;
    err += model_a->score_transform.p2.value != model_b->score_transform.p2.value;
    err += model_a->score_transform.out_lte_in != model_b->score_transform.out_lte_in;
    err += model_a->score_transform.out_gte_in != model_b->score_transform.out_gte_in;

    return err;
}


static char *test_json_model()
{
    int err = 0;

    VmafModel *model_json;
    VmafModelConfig cfg_json = { 0 };
    const char *path_json = "../../model/vmaf_v0.6.1neg.json";

    err = vmaf_read_json_model_from_path(&model_json, &cfg_json, path_json);
    mu_assert("problem during vmaf_read_json_model", !err);

    VmafModel *model_pkl;
    VmafModelConfig cfg_pkl = { 0 };
    const char *path_pkl = "../../model/vmaf_v0.6.1neg.pkl";

    err = vmaf_model_load_from_path(&model_pkl, &cfg_pkl, path_pkl);
    mu_assert("problem during vmaf_model_load_from_path", !err);

    err = model_compare(model_json, model_pkl);
    mu_assert("parsed json/pkl models do not match", !err);

    vmaf_model_destroy(model_json);
    vmaf_model_destroy(model_pkl);
    return NULL;
}

static char *test_model_collection()
{
    int err = 0;

    // pkl
    const char *pkl_path = "../../model/vmaf_rb_v0.6.3/vmaf_rb_v0.6.3.pkl";
    VmafModel *pkl_model;
    VmafModelCollection *pkl_model_collection = NULL;
    const VmafModelConfig pkl_cfg = { 0 };

    err = vmaf_model_collection_load_from_path(&pkl_model,
                                               &pkl_model_collection, &pkl_cfg,
                                               pkl_path);
    mu_assert("problem during load_model_collection", !err);

    // json
    const char *json_path = "../../model/vmaf_b_v0.6.3.json";
    VmafModel *json_model;
    VmafModelCollection *json_model_collection = NULL;
    const VmafModelConfig json_cfg = { 0 };
    err = vmaf_model_collection_load_from_path(&json_model,
                                               &json_model_collection,
                                               &json_cfg, json_path);
    mu_assert("problem during load_model_collection", !err);

    err = model_compare(json_model, pkl_model);
    mu_assert("parsed json/pkl models do not match", !err);

    vmaf_model_collection_destroy(pkl_model_collection);
    vmaf_model_collection_destroy(json_model_collection);
    return NULL;
}

static char *test_model_load_and_destroy()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = { 0 };
    const char *path = "../../model/vmaf_v0.6.1.pkl";
    err = vmaf_model_load_from_path(&model, &cfg, path);
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
        .name = "some_vmaf",
    };
    const char *path = "../../model/vmaf_v0.6.1.pkl";
    err = vmaf_model_load_from_path(&model, &cfg, path);
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
        .name = "some_vmaf",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };
    const char *path = "../../model/vmaf_v0.6.1.pkl";
    err = vmaf_model_load_from_path(&model, &cfg, path);
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
        .flags = VMAF_MODEL_FLAG_ENABLE_TRANSFORM,
    };
    const char *path1 = "../../model/vmaf_v0.6.1.pkl";
    err = vmaf_model_load_from_path(&model1, &cfg1, path1);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Score transform must be enabled.\n",
              model1->score_transform.enabled);
    mu_assert("Clipping must be enabled.\n",
              model1->score_clip.enabled);
    vmaf_model_destroy(model1);

    VmafModel *model2;
    VmafModelConfig cfg2 = {
        .flags = VMAF_MODEL_FLAG_DISABLE_CLIP,
    };
    const char *path2 = "../../model/vmaf_v0.6.1.pkl";
    err = vmaf_model_load_from_path(&model2, &cfg2, path2);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Score transform must be disabled.\n",
              !model2->score_transform.enabled);
    mu_assert("Clipping must be disabled.\n",
              !model2->score_clip.enabled);
    vmaf_model_destroy(model2);

    VmafModel  *model3;
    VmafModelConfig  cfg3 = { 0 };
    const char *path3 = "../../model/vmaf_v0.6.1.pkl";
    err = vmaf_model_load_from_path(&model3, &cfg3, path3);
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
    VmafModelConfig  cfg4 = { 0 };
    const char *path4 = "../../model/vmaf_v0.6.1neg.pkl";
    err = vmaf_model_load_from_path(&model4, &cfg4, path4);
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
    mu_run_test(test_json_model);
    //mu_run_test(test_model_collection);
    mu_run_test(test_model_load_and_destroy);
    mu_run_test(test_model_check_default_behavior_unset_flags);
    mu_run_test(test_model_check_default_behavior_set_flags);
    mu_run_test(test_model_set_flags);
    return NULL;
}
