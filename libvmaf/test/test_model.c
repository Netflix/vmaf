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

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "test.h"
#include "model.c"
#include "read_json_model.h"

static int model_compare(VmafModel *model_a, VmafModel *model_b)
{
    int err = 0;

    err += model_a->slope != model_b->slope;
    err += model_a->intercept != model_b->intercept;

    err += model_a->n_features != model_b->n_features;
    for (unsigned i = 0; i < model_a->n_features; i++) {
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

/* Read the whole file into a malloc'd buffer; caller frees.
 * Returns NULL on error; sets *len on success. */
static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        (void)fclose(f);
        return NULL;
    }
    const long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) {
        (void)fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        (void)fclose(f);
        return NULL;
    }
    const size_t got = fread(buf, 1, (size_t)sz, f);
    (void)fclose(f);
    if (got != (size_t)sz) {
        free(buf);
        return NULL;
    }
    buf[sz] = '\0';
    *len = (size_t)sz;
    return buf;
}

static int append_fmt(char *dst, size_t dst_sz, size_t *off, const char *fmt, ...)
{
    if (*off >= dst_sz)
        return -ENOSPC;

    va_list args;
    va_start(args, fmt);
    const int wrote = vsnprintf(&dst[*off], dst_sz - *off, fmt, args);
    va_end(args);
    if (wrote < 0)
        return -EINVAL;
    if ((size_t)wrote >= dst_sz - *off)
        return -ENOSPC;
    *off += (size_t)wrote;
    return 0;
}

static char *test_json_model()
{
    int err = 0;

    VmafModel *model_json;
    VmafModelConfig cfg_json = {0};
    const char *path_json = JSON_MODEL_PATH "vmaf_v0.6.1neg.json";

    err = vmaf_read_json_model_from_path(&model_json, &cfg_json, path_json);
    mu_assert("problem during vmaf_read_json_model", !err);

    VmafModel *model;
    VmafModelConfig cfg = {0};
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
    VmafModelConfig cfg = {0};
    const char *version = "vmaf_v0.6.1neg";
    err = vmaf_model_load(&model, &cfg, version);
    mu_assert("problem during vmaf_model_load", !err);

    VmafModel *model_file;
    VmafModelConfig cfg_file = {0};
    const char *path = JSON_MODEL_PATH "vmaf_v0.6.1neg.json";
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
    VmafModelConfig cfg = {0};
    const char *path = JSON_MODEL_PATH "vmaf_float_v0.6.1.json";
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
    VmafModelConfig cfg = {0};
    const char *version = "vmaf_v0.6.1";
    err = vmaf_model_load(&model, &cfg, version);
    mu_assert("problem during vmaf_model_load", !err);

    VmafFeatureDictionary *dict = NULL;
    err = vmaf_feature_dictionary_set(&dict, "adm_enhancement_gain_limit", "1.1");
    mu_assert("problem during vmaf_feature_dictionary_set", !err);

    mu_assert("feature 0 should be \"VMAF_integer_feature_adm2_score\"",
              !strcmp("VMAF_integer_feature_adm2_score", model->feature[0].name));
    mu_assert("feature 0 \"VMAF_integer_feature_adm2_score\" "
              "should have a NULL opts_dict",
              !model->feature[0].opts_dict);

    err = vmaf_model_feature_overload(model, "adm", dict);

    mu_assert("feature 0 should be \"VMAF_integer_feature_adm2_score\"",
              !strcmp("VMAF_integer_feature_adm2_score", model->feature[0].name));
    mu_assert("feature 0 \"VMAF_integer_feature_adm2_score\" "
              "should have a non-NULL opts_dict",
              model->feature[0].opts_dict);

    const VmafDictionaryEntry *e =
        vmaf_dictionary_get(&model->feature[0].opts_dict, "adm_enhancement_gain_limit", 0);
    mu_assert("dict lookup must return entry", e != NULL);
    mu_assert("dict should have a new key/val pair",
              !strcmp(e->key, "adm_enhancement_gain_limit") && !strcmp(e->val, "1.1"));

    VmafModel *model_neg;
    VmafModelConfig cfg_neg = {0};
    const char *version_neg = "vmaf_v0.6.1neg";
    err = vmaf_model_load(&model_neg, &cfg_neg, version_neg);
    mu_assert("problem during vmaf_model_load", !err);

    err = model_compare(model, model_neg);
    mu_assert("overloaded model should match model_neg", err);

    VmafFeatureDictionary *dict_neg = NULL;
    err = vmaf_feature_dictionary_set(&dict_neg, "adm_enhancement_gain_limit", "1.2");
    mu_assert("problem during vmaf_feature_dictionary_set", !err);

    mu_assert("feature 0 should be \"VMAF_integer_feature_adm2_score\"",
              !strcmp("VMAF_integer_feature_adm2_score", model->feature[0].name));
    mu_assert("feature 0 \"VMAF_integer_feature_adm2_score\" "
              "should have a non-NULL opts_dict",
              model->feature[0].opts_dict);
    const VmafDictionaryEntry *e2 =
        vmaf_dictionary_get(&model->feature[0].opts_dict, "adm_enhancement_gain_limit", 0);
    mu_assert("dict lookup must return entry", e2 != NULL);
    mu_assert("dict should have an existing key/val pair",
              !strcmp(e2->key, "adm_enhancement_gain_limit") && !strcmp(e2->val, "1.1"));

    err = vmaf_model_feature_overload(model, "adm", dict_neg);

    mu_assert("feature 0 should be \"VMAF_integer_feature_adm2_score\"",
              !strcmp("VMAF_integer_feature_adm2_score", model->feature[0].name));
    mu_assert("feature 0 \"VMAF_integer_feature_adm2_score\" "
              "should have a non-NULL opts_dict",
              model->feature[0].opts_dict);
    const VmafDictionaryEntry *e3 =
        vmaf_dictionary_get(&model->feature[0].opts_dict, "adm_enhancement_gain_limit", 0);
    mu_assert("dict lookup must return entry", e3 != NULL);
    mu_assert("dict should have an updated key/val pair",
              !strcmp(e3->key, "adm_enhancement_gain_limit") && !strcmp(e3->val, "1.2"));

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
    const char *path = JSON_MODEL_PATH "vmaf_float_v0.6.1.json";
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
    const char *path = JSON_MODEL_PATH "vmaf_float_v0.6.1.json";
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
    const char *path1 = JSON_MODEL_PATH "vmaf_float_v0.6.1.json";
    err = vmaf_model_load_from_path(&model1, &cfg1, path1);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Score transform must be enabled.\n", model1->score_transform.enabled);
    mu_assert("Clipping must be enabled.\n", model1->score_clip.enabled);
    vmaf_model_destroy(model1);

    VmafModel *model2;
    VmafModelConfig cfg2 = {
        .flags = VMAF_MODEL_FLAG_DISABLE_CLIP,
    };
    const char *path2 = JSON_MODEL_PATH "vmaf_float_v0.6.1.json";
    err = vmaf_model_load_from_path(&model2, &cfg2, path2);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("Score transform must be disabled.\n", !model2->score_transform.enabled);
    mu_assert("Clipping must be disabled.\n", !model2->score_clip.enabled);
    vmaf_model_destroy(model2);

    VmafModel *model3;
    VmafModelConfig cfg3 = {0};
    const char *path3 = JSON_MODEL_PATH "vmaf_float_v0.6.1.json";
    err = vmaf_model_load_from_path(&model3, &cfg3, path3);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("feature[0].opts_dict must be NULL.\n", !model3->feature[0].opts_dict);
    mu_assert("feature[1].opts_dict must be NULL.\n", !model3->feature[1].opts_dict);
    mu_assert("feature[2].opts_dict must be NULL.\n", !model3->feature[2].opts_dict);
    mu_assert("feature[3].opts_dict must be NULL.\n", !model3->feature[3].opts_dict);
    mu_assert("feature[4].opts_dict must be NULL.\n", !model3->feature[4].opts_dict);
    mu_assert("feature[5].opts_dict must be NULL.\n", !model3->feature[5].opts_dict);
    vmaf_model_destroy(model3);

    VmafModel *model4;
    VmafModelConfig cfg4 = {0};
    const char *path4 = JSON_MODEL_PATH "vmaf_float_v0.6.1neg.json";
    err = vmaf_model_load_from_path(&model4, &cfg4, path4);
    mu_assert("problem during vmaf_model_load_from_path", !err);
    mu_assert("feature[0].opts_dict must not be NULL.\n", model4->feature[0].opts_dict);
    mu_assert("feature[1].opts_dict must be NULL.\n", !model4->feature[1].opts_dict);
    mu_assert("feature[2].opts_dict must not be NULL.\n", model4->feature[2].opts_dict);
    mu_assert("feature[3].opts_dict must not be NULL.\n", model4->feature[3].opts_dict);
    mu_assert("feature[4].opts_dict must not be NULL.\n", model4->feature[4].opts_dict);
    mu_assert("feature[5].opts_dict must not be NULL.\n", model4->feature[5].opts_dict);

    const VmafDictionaryEntry *entry = NULL;
    entry = vmaf_dictionary_get(&model4->feature[0].opts_dict, "adm_enhn_gain_limit", 0);
    mu_assert("feature[0].opts_dict lookup must return entry.\n", entry != NULL);
    mu_assert("feature[0].opts_dict must have key adm_enhn_gain_limit.\n",
              strcmp(entry->key, "adm_enhn_gain_limit") == 0);
    mu_assert("feature[0].opts_dict[\"adm_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1") == 0);
    entry = vmaf_dictionary_get(&model4->feature[2].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[2].opts_dict lookup must return entry.\n", entry != NULL);
    mu_assert("feature[2].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit") == 0);
    mu_assert("feature[2].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1") == 0);
    entry = vmaf_dictionary_get(&model4->feature[3].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[3].opts_dict lookup must return entry.\n", entry != NULL);
    mu_assert("feature[3].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit") == 0);
    mu_assert("feature[3].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1") == 0);
    entry = vmaf_dictionary_get(&model4->feature[4].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[4].opts_dict lookup must return entry.\n", entry != NULL);
    mu_assert("feature[4].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit") == 0);
    mu_assert("feature[4].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1") == 0);
    entry = vmaf_dictionary_get(&model4->feature[5].opts_dict, "vif_enhn_gain_limit", 0);
    mu_assert("feature[5].opts_dict lookup must return entry.\n", entry != NULL);
    mu_assert("feature[5].opts_dict must have key vif_enhn_gain_limit.\n",
              strcmp(entry->key, "vif_enhn_gain_limit") == 0);
    mu_assert("feature[5].opts_dict[\"vif_enhn_gain_limit\"] must have value 1.\n",
              strcmp(entry->val, "1") == 0);

    vmaf_model_destroy(model4);
    return NULL;
}

/* Exercises vmaf_read_json_model_from_buffer — never hit by the existing
 * tests, which only use vmaf_read_json_model_from_path. Round-trip: file →
 * buffer → parse, and compare against the path-parsed model. */
static char *test_json_model_from_buffer(void)
{
    const char *path = JSON_MODEL_PATH "vmaf_float_v0.6.1.json";
    size_t len = 0;
    char *buf = slurp(path, &len);
    mu_assert("slurp failed", buf != NULL);

    VmafModel *m_buf;
    VmafModelConfig cfg_buf = {0};
    int err = vmaf_read_json_model_from_buffer(&m_buf, &cfg_buf, buf, (int)len);
    mu_assert("from_buffer failed", !err);

    VmafModel *m_path;
    VmafModelConfig cfg_path = {0};
    err = vmaf_read_json_model_from_path(&m_path, &cfg_path, path);
    mu_assert("from_path failed", !err);

    mu_assert("buffer/path models diverge", !model_compare(m_buf, m_path));

    vmaf_model_destroy(m_buf);
    vmaf_model_destroy(m_path);
    free(buf);
    return NULL;
}

/* Missing path → -EINVAL from the fopen guard in vmaf_read_json_model_from_path. */
static char *test_json_model_missing_path(void)
{
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err =
        vmaf_read_json_model_from_path(&m, &cfg, "/nonexistent/path/vmaf_does_not_exist.json");
    mu_assert("missing path should return -EINVAL", err == -EINVAL);
    return NULL;
}

/* Malformed JSON buffer → non-zero error from the parser. */
static char *test_json_model_malformed_buffer(void)
{
    const char garbage[] = "{this is definitely not valid json}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, garbage, (int)sizeof(garbage) - 1);
    mu_assert("malformed JSON should return non-zero", err != 0);
    /* On the error path the parser may still have allocated *m; free if so. */
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* Empty buffer → non-zero error. */
static char *test_json_model_empty_buffer(void)
{
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, "", 0);
    mu_assert("empty buffer should return non-zero", err != 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* Exercises vmaf_read_json_model_collection_from_path on the bootstrap
 * ensemble model. Verifies model_collection_parse iterates its keyed
 * sub-models ("0", "1", …) and fills both *model (first) and
 * *model_collection (rest). */
static char *test_json_model_collection_from_path(void)
{
    const char *path = JSON_MODEL_PATH "vmaf_b_v0.6.3.json";
    VmafModel *m = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = {.name = "vmaf_b"};
    int err = vmaf_read_json_model_collection_from_path(&m, &mc, &cfg, path);
    mu_assert("collection_from_path failed", !err);
    mu_assert("first model not populated", m != NULL);
    mu_assert("collection not populated", mc != NULL);

    vmaf_model_destroy(m);
    vmaf_model_collection_destroy(mc);
    return NULL;
}

/* Same ensemble model via the buffer entry point. */
static char *test_json_model_collection_from_buffer(void)
{
    const char *path = JSON_MODEL_PATH "vmaf_b_v0.6.3.json";
    size_t len = 0;
    char *buf = slurp(path, &len);
    mu_assert("slurp failed", buf != NULL);

    VmafModel *m = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = {.name = "vmaf_b_buf"};
    int err = vmaf_read_json_model_collection_from_buffer(&m, &mc, &cfg, buf, (int)len);
    mu_assert("collection_from_buffer failed", !err);
    mu_assert("first model not populated", m != NULL);
    mu_assert("collection not populated", mc != NULL);

    vmaf_model_destroy(m);
    vmaf_model_collection_destroy(mc);
    free(buf);
    return NULL;
}

/* Missing path for the collection API → -EINVAL. */
static char *test_json_model_collection_missing_path(void)
{
    VmafModel *m = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = {0};
    int err =
        vmaf_read_json_model_collection_from_path(&m, &mc, &cfg, "/nonexistent/path/vmaf_b.json");
    mu_assert("missing collection path should return -EINVAL", err == -EINVAL);
    return NULL;
}

/* Collection buffer that is not an object → model_collection_parse early
 * -EINVAL branch. */
static char *test_json_model_collection_malformed_buffer(void)
{
    const char garbage[] = "[1, 2, 3]";
    VmafModel *m = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_collection_from_buffer(&m, &mc, &cfg, garbage,
                                                          (int)sizeof(garbage) - 1);
    mu_assert("non-object collection should return non-zero", err != 0);
    return NULL;
}

/* Hits parser branches that upstream model JSONs don't exercise:
 *   - model_type = RESIDUEBOOTSTRAP_LIBSVMNUSVR
 *   - norm_type = none
 *   - score_transform with knots array + out_lte_in
 *   - feature_opts_dicts with string and bool values (not just numbers)
 * We don't assert a specific return code — libsvm's parser tolerates a
 * lot of garbage, so the call may succeed or fail depending on where in
 * the token stream it gives up. All we care about here is that the
 * pre-libsvm branches get exercised without crashing. */
static char *test_json_model_synthetic_branches(void)
{
    const char json[] =
        "{"
        "\"model_dict\": {"
        "\"model_type\": \"RESIDUEBOOTSTRAP_LIBSVMNUSVR\","
        "\"norm_type\": \"none\","
        "\"score_transform\": {"
        "\"enabled\": true,"
        "\"p0\": 1.0,"
        "\"p1\": 2.0,"
        "\"p2\": 3.0,"
        "\"knots\": [[0.0, 0.0], [1.0, 1.0]],"
        "\"out_lte_in\": \"true\","
        "\"out_gte_in\": \"false\""
        "},"
        "\"feature_names\": [\"f1\", \"f2\"],"
        "\"slopes\": [1.0, 0.1, 0.2],"
        "\"intercepts\": [0.0, 0.0, 0.0],"
        "\"feature_opts_dicts\": ["
        "{\"k_num\": 1.5, \"k_str\": \"hello\", \"k_true\": true, \"k_false\": false}"
        "],"
        "\"score_clip\": [0, 100],"
        "\"model\": \"not-a-real-libsvm-payload\""
        "}"
        "}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    /* libsvm's string parser is permissive and may accept arbitrary bytes,
     * so don't assert on err — the point is that parse_model_dict /
     * parse_score_transform / parse_feature_opts_dicts executed without
     * crashing and any allocation got cleaned up. */
    (void)err;
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

static char *test_json_model_allows_more_than_64_features(void)
{
    char json[8192];
    size_t off = 0;
    int err = append_fmt(json, sizeof(json), &off, "{\"model_dict\":{\"feature_names\":[");
    for (unsigned i = 0; i < 65u && !err; i++) {
        err = append_fmt(json, sizeof(json), &off, "%s\"feature_%u\"", i ? "," : "", i);
    }
    if (!err)
        err = append_fmt(json, sizeof(json), &off, "],\"slopes\":[1.0");
    for (unsigned i = 0; i < 65u && !err; i++)
        err = append_fmt(json, sizeof(json), &off, ",%u.0", i + 1u);
    if (!err)
        err = append_fmt(json, sizeof(json), &off, "],\"intercepts\":[0.0");
    for (unsigned i = 0; i < 65u && !err; i++)
        err = append_fmt(json, sizeof(json), &off, ",%u.0", i);
    if (!err)
        err = append_fmt(json, sizeof(json), &off, "]}}");
    mu_assert("synthetic model JSON builder overflowed", !err);

    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)off);
    mu_assert("65-feature JSON model must parse", !err);
    mu_assert("feature count must be preserved", m->n_features == 65u);
    mu_assert("feature capacity must grow past the old fixed limit", m->feature_cap >= 65u);
    mu_assert("last feature name must parse", strcmp(m->feature[64].name, "feature_64") == 0);
    mu_assert("last feature slope must parse", m->feature[64].slope == 65.0);
    mu_assert("last feature intercept must parse", m->feature[64].intercept == 64.0);
    vmaf_model_destroy(m);
    return NULL;
}

static char *test_json_model_allows_more_than_10_knots(void)
{
    char json[2048];
    size_t off = 0;
    int err = append_fmt(json, sizeof(json), &off,
                         "{\"model_dict\":{\"score_transform\":{\"enabled\":true,\"knots\":[");
    for (unsigned i = 0; i < 11u && !err; i++)
        err = append_fmt(json, sizeof(json), &off, "%s[%u.0,%u.0]", i ? "," : "", i, i + 1u);
    if (!err)
        err = append_fmt(json, sizeof(json), &off, "]}}}");
    mu_assert("synthetic knot JSON builder overflowed", !err);

    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)off);
    mu_assert("11-knot JSON model must parse", !err);
    mu_assert("knot count must be preserved", m->score_transform.knots.n_knots == 11u);
    mu_assert("knot capacity must grow past the old fixed limit",
              m->score_transform.knots.cap >= 11u);
    mu_assert("last knot x must parse", m->score_transform.knots.list[10].x == 10.0);
    mu_assert("last knot y must parse", m->score_transform.knots.list[10].y == 11.0);
    vmaf_model_destroy(m);
    return NULL;
}

/* parse_model_dict: unknown model_type value → -EINVAL (line 333). */
static char *test_json_model_unknown_model_type(void)
{
    const char json[] = "{\"model_dict\": {\"model_type\": \"NOT_A_REAL_TYPE\"}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("unknown model_type must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_model_dict: unknown norm_type value → -EINVAL (line 347). */
static char *test_json_model_unknown_norm_type(void)
{
    const char json[] = "{\"model_dict\": {\"norm_type\": \"weird-norm\"}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("unknown norm_type must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_model_dict: model_type not a string → -EINVAL (line 324). */
static char *test_json_model_model_type_not_string(void)
{
    const char json[] = "{\"model_dict\": {\"model_type\": 42}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-string model_type must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_model_dict: norm_type not a string → -EINVAL (line 340). */
static char *test_json_model_norm_type_not_string(void)
{
    const char json[] = "{\"model_dict\": {\"norm_type\": 7}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-string norm_type must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_model_dict: score_transform not an object → -EINVAL (line 308). */
static char *test_json_model_score_transform_not_object(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": [1,2,3]}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-object score_transform must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_score_transform: p0 neither null nor number → -EINVAL (line 216). */
static char *test_json_model_score_transform_p0_bad_type(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"p0\": \"oops\"}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("string p0 must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_score_transform: p1 bad type → -EINVAL (line 228). */
static char *test_json_model_score_transform_p1_bad_type(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"p1\": true}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("bool p1 must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_score_transform: p2 bad type → -EINVAL (line 240). */
static char *test_json_model_score_transform_p2_bad_type(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"p2\": false}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("bool p2 must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_score_transform: knots neither null nor array → -EINVAL (line 255). */
static char *test_json_model_score_transform_knots_bad_type(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"knots\": 99}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("number knots must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_score_transform: out_lte_in not a string → -EINVAL (line 262). */
static char *test_json_model_score_transform_out_lte_in_not_string(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"out_lte_in\": 1}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-string out_lte_in must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_score_transform: out_gte_in not a string → -EINVAL (line 271). */
static char *test_json_model_score_transform_out_gte_in_not_string(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"out_gte_in\": 1}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-string out_gte_in must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_score_transform: enabled neither true nor false → -EINVAL (line 204). */
static char *test_json_model_score_transform_enabled_bad_type(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"enabled\": 7}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-bool enabled must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_feature_names: non-string element → -EINVAL (line 180). */
static char *test_json_model_feature_names_non_string(void)
{
    const char json[] = "{\"model_dict\": {\"feature_names\": [\"ok\", 42]}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-string feature name must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_slopes: non-number element → -EINVAL (line 116). */
static char *test_json_model_slopes_non_number(void)
{
    const char json[] = "{\"model_dict\": {\"slopes\": [1.0, \"x\"]}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-number slope must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_intercepts: first element not a number → -EINVAL (line 92). */
static char *test_json_model_intercepts_first_not_number(void)
{
    const char json[] = "{\"model_dict\": {\"intercepts\": [\"nope\"]}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-number first intercept must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_knots: outer element not an array → -EINVAL (line 149). */
static char *test_json_model_knots_outer_not_array(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"knots\": [42]}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-array knot must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_knots_list: knot pair holds >2 numbers → -EINVAL (line 132). */
static char *test_json_model_knots_too_many_values(void)
{
    const char json[] = "{\"model_dict\": {\"score_transform\": {\"knots\": [[0.0, 1.0, 2.0]]}}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("knot triple must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_feature_opts_dicts: value type other than number/bool/string
 * (here: null) → -EINVAL (line 77). */
static char *test_json_model_feature_opts_dict_bad_value_type(void)
{
    const char json[] = "{\"model_dict\": {\"feature_opts_dicts\": [{\"k\": null}]}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("null opts value must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_model_dict: score_clip not an array → -EINVAL (line 354). */
static char *test_json_model_score_clip_not_array(void)
{
    const char json[] = "{\"model_dict\": {\"score_clip\": 0}}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-array score_clip must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* parse_model_dict: top-level model_dict value not an object → -EINVAL (line 299). */
static char *test_json_model_model_dict_not_object(void)
{
    const char json[] = "{\"model_dict\": [1,2]}";
    VmafModel *m = NULL;
    VmafModelConfig cfg = {0};
    int err = vmaf_read_json_model_from_buffer(&m, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("non-object model_dict must reject", err < 0);
    if (m)
        vmaf_model_destroy(m);
    return NULL;
}

/* Collection parser hits json_skip() for keys that don't match the
 * generated "%d" index sequence (line 556). Also hits the -EINVAL early
 * return when the inner model payload is malformed (line 538). */
static char *test_json_model_collection_skips_unknown_keys(void)
{
    const char json[] = "{"
                        "\"extra_meta\": \"ignored\","
                        "\"0\": \"not an object — will fail inner parse\""
                        "}";
    VmafModel *m = NULL;
    VmafModelCollection *mc = NULL;
    VmafModelConfig cfg = {.name = "synth"};
    int err =
        vmaf_read_json_model_collection_from_buffer(&m, &mc, &cfg, json, (int)sizeof(json) - 1);
    mu_assert("bad inner model should fail the collection parse", err != 0);
    if (m)
        vmaf_model_destroy(m);
    if (mc)
        vmaf_model_collection_destroy(mc);
    return NULL;
}

/* Exercises the score_transform parser branches (p0, p1, p2, out_gte_in)
 * via VMAF_MODEL_FLAG_ENABLE_TRANSFORM on a model that actually carries a
 * score_transform block. vmaf_v0.6.1.json has one. */
static char *test_json_model_score_transform(void)
{
    VmafModel *m;
    VmafModelConfig cfg = {
        .flags = VMAF_MODEL_FLAG_ENABLE_TRANSFORM,
    };
    const char *path = JSON_MODEL_PATH "vmaf_v0.6.1.json";
    int err = vmaf_read_json_model_from_path(&m, &cfg, path);
    mu_assert("load failed", !err);
    mu_assert("score_transform should be enabled", m->score_transform.enabled);
    mu_assert("p0 should be enabled", m->score_transform.p0.enabled);
    mu_assert("p1 should be enabled", m->score_transform.p1.enabled);
    mu_assert("p2 should be enabled", m->score_transform.p2.enabled);
    mu_assert("out_gte_in should be set", m->score_transform.out_gte_in);
    vmaf_model_destroy(m);
    return NULL;
}

static char *test_version_next(void)
{
    const void *next = NULL;
    const char *version = NULL;
    unsigned count = 0;
    while ((next = vmaf_model_version_next(next, &version)) != NULL) {
        const VmafBuiltInModel *m = next;
        mu_assert("vmaf_model_version_next must hand out the stored version pointer",
                  m->version == version);
        count++;
    }
    mu_assert("vmaf_model_version_next must iterate every built-in model exactly once",
              count == BUILT_IN_MODEL_CNT);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_json_model);
#if VMAF_BUILT_IN_MODELS
    mu_run_test(test_built_in_model);
#endif
    mu_run_test(test_model_load_and_destroy);
    mu_run_test(test_model_check_default_behavior_unset_flags);
    mu_run_test(test_model_check_default_behavior_set_flags);
    mu_run_test(test_model_set_flags);
    mu_run_test(test_model_feature);
    mu_run_test(test_json_model_from_buffer);
    mu_run_test(test_json_model_missing_path);
    mu_run_test(test_json_model_malformed_buffer);
    mu_run_test(test_json_model_empty_buffer);
    mu_run_test(test_json_model_collection_from_path);
    mu_run_test(test_json_model_collection_from_buffer);
    mu_run_test(test_json_model_collection_missing_path);
    mu_run_test(test_json_model_collection_malformed_buffer);
    mu_run_test(test_json_model_score_transform);
    mu_run_test(test_json_model_synthetic_branches);
    mu_run_test(test_json_model_allows_more_than_64_features);
    mu_run_test(test_json_model_allows_more_than_10_knots);
    mu_run_test(test_json_model_collection_skips_unknown_keys);
    mu_run_test(test_json_model_unknown_model_type);
    mu_run_test(test_json_model_unknown_norm_type);
    mu_run_test(test_json_model_model_type_not_string);
    mu_run_test(test_json_model_norm_type_not_string);
    mu_run_test(test_json_model_score_transform_not_object);
    mu_run_test(test_json_model_score_transform_p0_bad_type);
    mu_run_test(test_json_model_score_transform_p1_bad_type);
    mu_run_test(test_json_model_score_transform_p2_bad_type);
    mu_run_test(test_json_model_score_transform_knots_bad_type);
    mu_run_test(test_json_model_score_transform_out_lte_in_not_string);
    mu_run_test(test_json_model_score_transform_out_gte_in_not_string);
    mu_run_test(test_json_model_score_transform_enabled_bad_type);
    mu_run_test(test_json_model_feature_names_non_string);
    mu_run_test(test_json_model_slopes_non_number);
    mu_run_test(test_json_model_intercepts_first_not_number);
    mu_run_test(test_json_model_knots_outer_not_array);
    mu_run_test(test_json_model_knots_too_many_values);
    mu_run_test(test_json_model_feature_opts_dict_bad_value_type);
    mu_run_test(test_json_model_score_clip_not_array);
    mu_run_test(test_json_model_model_dict_not_object);
    mu_run_test(test_version_next);
    return NULL;
}
