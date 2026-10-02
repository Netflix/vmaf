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

#include "config.h"
#include "test.h"
#include "dict.h"

/* model.c is included by this test, so its dictionary calls can be exercised
 * with deterministic failures without changing the library's allocator/API. */
static VmafDictionary *test_dictionary_merge(VmafDictionary **a,
                                             VmafDictionary **b, uint64_t flags);
static int test_dictionary_copy(VmafDictionary **src, VmafDictionary **dst);
static int test_dictionary_free(VmafDictionary **dict);
#define vmaf_dictionary_merge test_dictionary_merge
#define vmaf_dictionary_copy test_dictionary_copy
#define vmaf_dictionary_free test_dictionary_free
#include "model.c"
#undef vmaf_dictionary_merge
#undef vmaf_dictionary_copy
#undef vmaf_dictionary_free

static int fail_merge, fail_copy;
static VmafDictionary *tracked_input, *tracked_partial;
static unsigned input_frees, partial_frees;

static VmafDictionary *test_dictionary_merge(VmafDictionary **a,
                                             VmafDictionary **b, uint64_t flags)
{
    if (fail_merge) return NULL;
    return vmaf_dictionary_merge(a, b, flags);
}

static int test_dictionary_copy(VmafDictionary **src, VmafDictionary **dst)
{
    if (!fail_copy) return vmaf_dictionary_copy(src, dst);
    /* A real dictionary copy can fail after allocating some entries. */
    int err = vmaf_dictionary_set(dst, "partial", "copy", 0);
    tracked_partial = *dst;
    return err ? err : -ENOMEM;
}

static int test_dictionary_free(VmafDictionary **dict)
{
    if (dict && *dict) {
        if (*dict == tracked_input) {
            input_frees++;
            tracked_input = NULL;
        }
        if (*dict == tracked_partial) {
            partial_frees++;
            tracked_partial = NULL;
        }
    }
    return vmaf_dictionary_free(dict);
}
#include "read_json_model.h"

static int model_compare(VmafModel *model_a, VmafModel *model_b)
{
    int err = 0;

   //err += model_a->type != model_b->type;

    err += model_a->slope != model_b->slope;
    err += model_a->intercept != model_b->intercept;

    err += model_a->n_features != model_b->n_features;
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
    err += model_a->score_transform.knots.enabled != model_b->score_transform.knots.enabled;
    for (unsigned i = 0; i < model_a->score_transform.knots.n_knots; i++) {
        err += model_a->score_transform.knots.list[i].x != model_b->score_transform.knots.list[i].x;
        err += model_a->score_transform.knots.list[i].y != model_b->score_transform.knots.list[i].y;
    }
    err += model_a->score_transform.out_lte_in != model_b->score_transform.out_lte_in;
    err += model_a->score_transform.out_gte_in != model_b->score_transform.out_gte_in;

    return err;
}

static char *test_json_model()
{
    int err = 0;

    VmafModel *model_json;
    VmafModelConfig cfg_json = { 0 };
    const char *path_json = JSON_MODEL_PATH"vmaf_v0.6.1neg.json";

    err = vmaf_read_json_model_from_path(&model_json, &cfg_json, path_json);
    mu_assert("problem during vmaf_read_json_model", !err);

    VmafModel *model;
    VmafModelConfig cfg = { 0 };
    const char *version = "vmaf_v0.6.1neg";

    err = vmaf_model_load(&model, &cfg, version);
    mu_assert("problem during vmaf_model_load_from_path", !err);

    err = model_compare(model_json, model);
    mu_assert("parsed json/built-in models do not match", !err);

    vmaf_model_destroy(model_json);
    vmaf_model_destroy(model);
    return NULL;
}

#if VMAF_BUILT_IN_MODELS
static char *test_built_in_model()
{
    int err = 0;

    VmafModel *model;
    VmafModelConfig cfg = { 0 };
    const char *version = "vmaf_v0.6.1neg";
    err = vmaf_model_load(&model, &cfg, version);
    mu_assert("problem during vmaf_model_load", !err);

    VmafModel *model_file;
    VmafModelConfig cfg_file = { 0 };
    const char *path = JSON_MODEL_PATH"vmaf_v0.6.1neg.json";
    err = vmaf_model_load_from_path(&model_file, &cfg_file, path);
    mu_assert("problem during vmaf_model_load_from_path", !err);

    err = model_compare(model, model_file);
    mu_assert("parsed buffer/file models do not match", !err);

    vmaf_model_destroy(model);
    vmaf_model_destroy(model_file);
    return NULL;
}
#endif

static char *test_model_load_and_destroy()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = { 0 };
    const char *path = JSON_MODEL_PATH"vmaf_float_v0.6.1.json";
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

static char *test_model_feature()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = { 0 };
    const char *version = "vmaf_v0.6.1";
    err = vmaf_model_load(&model, &cfg, version);
    mu_assert("problem during vmaf_model_load", !err);

    VmafFeatureDictionary *dict = NULL;
    err = vmaf_feature_dictionary_set(&dict, "adm_enhancement_gain_limit",
                                      "1.1");
    mu_assert("problem during vmaf_feature_dictionary_set", !err);

    mu_assert("feature 0 should be \"VMAF_integer_feature_adm2_score\"",
              !strcmp("VMAF_integer_feature_adm2_score",
                      model->feature[0].name));
    mu_assert("feature 0 \"VMAF_integer_feature_adm2_score\" "
              "should have a NULL opts_dict",
              !model->feature[0].opts_dict);

    err = vmaf_model_feature_overload(model, "adm", dict);

    mu_assert("feature 0 should be \"VMAF_integer_feature_adm2_score\"",
              !strcmp("VMAF_integer_feature_adm2_score",
                      model->feature[0].name));
    mu_assert("feature 0 \"VMAF_integer_feature_adm2_score\" "
              "should have a non-NULL opts_dict",
              model->feature[0].opts_dict);

    const VmafDictionaryEntry *e =
        vmaf_dictionary_get(&model->feature[0].opts_dict,
                           "adm_enhancement_gain_limit", 0);
    mu_assert("dict should have a new key/val pair",
              !strcmp(e->key, "adm_enhancement_gain_limit") &&
              !strcmp(e->val, "1.1"));

    VmafModel *model_neg;
    VmafModelConfig cfg_neg = { 0 };
    const char *version_neg = "vmaf_v0.6.1neg";
    err = vmaf_model_load(&model_neg, &cfg_neg, version_neg);
    mu_assert("problem during vmaf_model_load", !err);

    err = model_compare(model, model_neg);
    mu_assert("overloaded model should match model_neg", err);

    VmafFeatureDictionary *dict_neg = NULL;
    err = vmaf_feature_dictionary_set(&dict_neg, "adm_enhancement_gain_limit",
                                      "1.2");
    mu_assert("problem during vmaf_feature_dictionary_set", !err);

    mu_assert("feature 0 should be \"VMAF_integer_feature_adm2_score\"",
              !strcmp("VMAF_integer_feature_adm2_score",
                      model->feature[0].name));
    mu_assert("feature 0 \"VMAF_integer_feature_adm2_score\" "
              "should have a non-NULL opts_dict",
              model->feature[0].opts_dict);
    const VmafDictionaryEntry *e2 =
        vmaf_dictionary_get(&model->feature[0].opts_dict,
                           "adm_enhancement_gain_limit", 0);
    mu_assert("dict should have an existing key/val pair",
              !strcmp(e2->key, "adm_enhancement_gain_limit") &&
              !strcmp(e2->val, "1.1"));

    err = vmaf_model_feature_overload(model, "adm", dict_neg);

    mu_assert("feature 0 should be \"VMAF_integer_feature_adm2_score\"",
              !strcmp("VMAF_integer_feature_adm2_score",
                      model->feature[0].name));
    mu_assert("feature 0 \"VMAF_integer_feature_adm2_score\" "
              "should have a non-NULL opts_dict",
              model->feature[0].opts_dict);
    const VmafDictionaryEntry *e3 =
        vmaf_dictionary_get(&model->feature[0].opts_dict,
                           "adm_enhancement_gain_limit", 0);
    mu_assert("dict should have an updated key/val pair",
              !strcmp(e3->key, "adm_enhancement_gain_limit") &&
              !strcmp(e3->val, "1.2"));

    vmaf_model_destroy(model);
    vmaf_model_destroy(model_neg);

    return NULL;
}

static char *test_model_check_default_behavior_unset_flags()
{
    int err;

    VmafModel *model;
    VmafModelConfig cfg = {
        .name = "some_vmaf",
    };
    const char *path = JSON_MODEL_PATH"vmaf_float_v0.6.1.json";
    err = vmaf_model_load_from_path(&model, &cfg, path);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Model name is inconsistent.\n", !strcmp(model->name, "some_vmaf"));
    mu_assert("Clipping must be enabled by default.\n", model->score_clip.enabled);
    mu_assert("Score transform must be disabled by default.\n", !model->score_transform.enabled);
    /* TODO: add check for confidence interval */
    mu_assert("Feature 0 name must be VMAF_feature_adm2_score.\n",
              !strcmp(model->feature[0].name, "VMAF_feature_adm2_score"));

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
    const char *path = JSON_MODEL_PATH"vmaf_float_v0.6.1.json";
    err = vmaf_model_load_from_path(&model, &cfg, path);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Model name is inconsistent.\n", !strcmp(model->name, "some_vmaf"));
    mu_assert("Clipping must be enabled by default.\n", model->score_clip.enabled);
    mu_assert("Score transform must be disabled by default.\n", !model->score_transform.enabled);
    /* TODO: add check for confidence interval */
    mu_assert("Feature 0 name must be VMAF_feature_adm2_score.\n",
              !strcmp(model->feature[0].name, "VMAF_feature_adm2_score"));

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
    const char *path1 = JSON_MODEL_PATH"vmaf_float_v0.6.1.json";
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
    const char *path2 = JSON_MODEL_PATH"vmaf_float_v0.6.1.json";
    err = vmaf_model_load_from_path(&model2, &cfg2, path2);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Score transform must be disabled.\n",
              !model2->score_transform.enabled);
    mu_assert("Clipping must be disabled.\n",
              !model2->score_clip.enabled);
    vmaf_model_destroy(model2);

    VmafModel  *model3;
    VmafModelConfig  cfg3 = { 0 };
    const char *path3 = JSON_MODEL_PATH"vmaf_float_v0.6.1.json";
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
    vmaf_model_destroy(model3);

    VmafModel  *model4;
    VmafModelConfig  cfg4 = { 0 };
    const char *path4 = JSON_MODEL_PATH"vmaf_float_v0.6.1neg.json";
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

    const VmafDictionaryEntry *entry = NULL;
    entry = vmaf_dictionary_get(&model4->feature[0].opts_dict, "adm_enhn_gain_limit", 0);
    mu_assert("feature[0].opts_dict must have key adm_enhn_gain_limit.\n",
              strcmp(entry->key, "adm_enhn_gain_limit")==0);
    mu_assert("feature[0].opts_dict[\"adm_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1")==0);
    entry = vmaf_dictionary_get(&model4->feature[2].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[2].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit")==0);
    mu_assert("feature[2].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1")==0);
    entry = vmaf_dictionary_get(&model4->feature[3].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[3].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit")==0);
    mu_assert("feature[3].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1")==0);
    entry = vmaf_dictionary_get(&model4->feature[4].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[4].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit")==0);
    mu_assert("feature[4].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1")==0);
    entry = vmaf_dictionary_get(&model4->feature[5].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[5].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit")==0);
    mu_assert("feature[5].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1")==0);

    vmaf_model_destroy(model4);
    return NULL;
}

static char *test_model_overload_merge_failure_consumes_dictionary(void)
{
    VmafModel *model = NULL;
    VmafModelConfig cfg = { 0 };
    int err = vmaf_model_load_from_path(&model, &cfg,
                                       JSON_MODEL_PATH"vmaf_v0.6.1.json");
    mu_assert("could not load model", !err);
    VmafFeatureDictionary *dict = NULL;
    err = vmaf_feature_dictionary_set(&dict, "adm_enhn_gain_limit", "1");
    mu_assert("could not create options", !err);
    tracked_input = (VmafDictionary*)dict;
    input_frees = 0;
    fail_merge = 1;
    err = vmaf_model_feature_overload(model, "adm", dict);
    fail_merge = 0;
    mu_assert("merge failure not reported", err == -ENOMEM);
    mu_assert("input dictionary not consumed on merge failure", input_frees == 1);
    vmaf_model_destroy(model);
    return NULL;
}

static char *test_collection_overload_copy_failure(void)
{
    /* No matching features are needed to exercise the collection copy path. */
    VmafModel lead = { 0 }, member = { 0 };
    VmafModel *members[] = { &member };
    VmafModelCollection collection = { .model = members, .cnt = 1 };
    VmafModelCollection *mc = &collection;
    VmafFeatureDictionary *dict = NULL;
    int err = vmaf_feature_dictionary_set(&dict, "adm_enhn_gain_limit", "1");
    mu_assert("could not create options", !err);
    tracked_input = (VmafDictionary*)dict;
    input_frees = partial_frees = 0;
    fail_copy = 1;
    err = vmaf_model_collection_feature_overload(&lead, &mc, "adm", dict);
    fail_copy = 0;
    mu_assert("copy failure was swallowed", err == -ENOMEM);
    mu_assert("partial copy not freed", partial_frees == 1);
    mu_assert("input dictionary not consumed", input_frees == 1);
    return NULL;
}

static char *test_model_overload_argument_ownership(void)
{
    VmafModel model = { 0 };
    VmafModelCollection collection = { 0 }, *mc = &collection, *empty = NULL;
    VmafFeatureDictionary *dict = NULL;
    int err = vmaf_feature_dictionary_set(&dict, "adm_enhn_gain_limit", "1");
    mu_assert("could not create options", !err);
    tracked_input = (VmafDictionary*)dict;
    input_frees = 0;
    mu_assert("NULL model accepted",
              vmaf_model_feature_overload(NULL, "adm", dict) == -EINVAL);
    mu_assert("NULL feature accepted",
              vmaf_model_feature_overload(&model, NULL, dict) == -EINVAL);
    mu_assert("NULL options accepted",
              vmaf_model_feature_overload(&model, "adm", NULL) == -EINVAL);
    mu_assert("NULL collection pointer accepted",
              vmaf_model_collection_feature_overload(&model, NULL, "adm", dict) == -EINVAL);
    mu_assert("NULL collection accepted",
              vmaf_model_collection_feature_overload(&model, &empty, "adm", dict) == -EINVAL);
    mu_assert("NULL lead accepted",
              vmaf_model_collection_feature_overload(NULL, &mc, "adm", dict) == -EINVAL);
    mu_assert("NULL collection feature accepted",
              vmaf_model_collection_feature_overload(&model, &mc, NULL, dict) == -EINVAL);
    mu_assert("NULL collection options accepted",
              vmaf_model_collection_feature_overload(&model, &mc, "adm", NULL) == -EINVAL);
    mu_assert("invalid arguments consumed caller's dictionary", !input_frees);
    mu_assert("caller dictionary no longer usable",
              vmaf_dictionary_get((VmafDictionary**)&dict, "adm_enhn_gain_limit", 0));
    mu_assert("no-match overload failed",
              !vmaf_model_feature_overload(&model, "unknown", dict));
    mu_assert("no-match overload did not consume dictionary", input_frees == 1);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_model_overload_merge_failure_consumes_dictionary);
    mu_run_test(test_collection_overload_copy_failure);
    mu_run_test(test_model_overload_argument_ownership);
    mu_run_test(test_json_model);
#if VMAF_BUILT_IN_MODELS
    mu_run_test(test_built_in_model);
#endif
    mu_run_test(test_model_load_and_destroy);
    mu_run_test(test_model_check_default_behavior_unset_flags);
    mu_run_test(test_model_check_default_behavior_set_flags);
    mu_run_test(test_model_set_flags);
    mu_run_test(test_model_feature);
    return NULL;
}
